#!/usr/bin/env python3
"""
Within- vs between-region EMD of per-pixel top-N intensities, plotted vs N.

For each annotated polygon interior, sample spectra and take the top-N
intensities (sorted descending). Compare distributions with 1D Wasserstein
distance averaged over the N ranks:

  within  = EMD between random halves of the same region
  between = EMD between pixels from different regions

A rising between/within ratio for small N (e.g. TOP3) supports using top
intensities as region-stable fingerprints.

Run (repo root):
  python scripts/topn_region_emd.py
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy.stats import wasserstein_distance

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from annotation_edge_metrics import (  # noqa: E402
    _dedupe_polygons,
    _load_polygons_for_stem,
)
from debug_edge_maps import _resolve_normalization_mode  # noqa: E402
from solace_annotation_gallery import (  # noqa: E402
    _opt_path,
    _resolve_slide_rows,
)
from view_annotation_boundaries import (  # noqa: E402
    apply_annotation_spatial_transform,
    resolve_transpose_spatial_mode,
    spatial_hw_after_transpose_mode,
    spatial_hw_from_npy,
)

logger = logging.getLogger(__name__)


def _polygon_interior_mask(
    poly: np.ndarray,
    shape_hw: tuple[int, int],
    *,
    erode: int,
) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    filled = np.zeros((h, w), dtype=np.uint8)
    p = np.int32(np.round(np.asarray(poly, dtype=np.float32).reshape(-1, 2)))
    if p.shape[0] < 3:
        return filled.astype(bool)
    cv2.fillPoly(filled, [p], 255)
    e = max(0, int(erode))
    if e > 0:
        k = 2 * e + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        filled = cv2.erode(filled, kernel, iterations=1)
    return filled.astype(bool)


def _sample_coords(mask: np.ndarray, max_pixels: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    n = int(ys.size)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if n > max_pixels:
        sel = rng.choice(n, size=int(max_pixels), replace=False)
        ys, xs = ys[sel], xs[sel]
    return np.stack([ys, xs], axis=1).astype(np.int32)


def _topn_values(spectra: np.ndarray, n: int) -> np.ndarray:
    """Return (P, n) top intensities per row, sorted descending."""
    if spectra.size == 0:
        return np.zeros((0, n), dtype=np.float64)
    c = int(spectra.shape[1])
    k = min(int(n), c)
    if k <= 0:
        return np.zeros((spectra.shape[0], 0), dtype=np.float64)
    part = np.partition(np.asarray(spectra, dtype=np.float64), -k, axis=1)[:, -k:]
    part.sort(axis=1)
    return part[:, ::-1]


def _mean_rank_emd(a: np.ndarray, b: np.ndarray) -> float:
    """Mean 1D Wasserstein over columns (rank channels)."""
    if a.size == 0 or b.size == 0 or a.shape[1] == 0:
        return float("nan")
    dists = []
    for j in range(a.shape[1]):
        aa = a[:, j]
        bb = b[:, j]
        if aa.size == 0 or bb.size == 0:
            continue
        dists.append(float(wasserstein_distance(aa, bb)))
    if not dists:
        return float("nan")
    return float(np.mean(dists))


def _within_emd(topn: np.ndarray, rng: np.random.Generator) -> float:
    p = int(topn.shape[0])
    if p < 4:
        return float("nan")
    perm = rng.permutation(p)
    mid = p // 2
    return _mean_rank_emd(topn[perm[:mid]], topn[perm[mid:]])


def _load_msi_aligned(npy: Path, *, spatial_mode: str, normalization: str) -> np.ndarray:
    """Load HxWxC MSI; apply annotation spatial transform (not debug_edge transpose)."""
    img = np.load(npy, mmap_mode="r")
    if img.ndim == 2:
        img = np.asarray(img[..., np.newaxis], dtype=np.float32)
    elif img.ndim != 3:
        raise ValueError(f"Expected HxWxC, got {img.shape} for {npy}")
    else:
        img = np.asarray(img, dtype=np.float32)
    img = apply_annotation_spatial_transform(img, spatial_mode)
    mode = _resolve_normalization_mode(normalization)
    if mode == "none":
        return img
    # Reuse debug_edge_maps helpers via a tiny local TIC (avoid double-transpose).
    from debug_edge_maps import _spatial_tic_normalize, _tic_normalize

    if mode == "tic":
        return _tic_normalize(img)
    if mode == "spatial_tic":
        return _spatial_tic_normalize(img)
    raise ValueError(f"Unknown normalization: {normalization!r}")


def _collect_region_spectra(
    msi: np.ndarray,
    polys: list[np.ndarray],
    main_cats: list[str],
    *,
    erode: int,
    min_pixels: int,
    max_pixels: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    h, w = int(msi.shape[0]), int(msi.shape[1])
    regions: list[dict[str, Any]] = []
    for i, (poly, main) in enumerate(zip(polys, main_cats)):
        mask = _polygon_interior_mask(poly, (h, w), erode=erode)
        coords = _sample_coords(mask, max_pixels, rng)
        if coords.shape[0] < min_pixels:
            continue
        ys, xs = coords[:, 0], coords[:, 1]
        spectra = np.asarray(msi[ys, xs, :], dtype=np.float32)
        regions.append(
            {
                "id": i,
                "main_cat1": str(main),
                "n_pixels": int(spectra.shape[0]),
                "spectra": spectra,
            }
        )
    return regions


def _pair_indices(n_regions: int, max_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if n_regions < 2:
        return []
    all_pairs = [(i, j) for i in range(n_regions) for j in range(i + 1, n_regions)]
    if len(all_pairs) <= max_pairs:
        return all_pairs
    sel = rng.choice(len(all_pairs), size=int(max_pairs), replace=False)
    return [all_pairs[int(k)] for k in sel]


def analyze_slide_regions(
    regions: list[dict[str, Any]],
    ns: list[int],
    *,
    max_between_pairs: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_max = max(int(n) for n in ns)
    # Precompute top-n_max once per region.
    tops: list[np.ndarray] = []
    for r in regions:
        tops.append(_topn_values(r["spectra"], n_max))

    pairs = _pair_indices(len(regions), max_between_pairs, rng)
    rows: list[dict[str, Any]] = []
    for n in ns:
        n = int(n)
        within_vals: list[float] = []
        between_vals: list[float] = []
        between_same: list[float] = []
        between_diff: list[float] = []
        for t in tops:
            w = _within_emd(t[:, :n], rng)
            if np.isfinite(w):
                within_vals.append(w)
        for i, j in pairs:
            d = _mean_rank_emd(tops[i][:, :n], tops[j][:, :n])
            if not np.isfinite(d):
                continue
            between_vals.append(d)
            mi = str(regions[i]["main_cat1"])
            mj = str(regions[j]["main_cat1"])
            if mi == mj:
                between_same.append(d)
            else:
                between_diff.append(d)

        mean_w = float(np.mean(within_vals)) if within_vals else float("nan")
        mean_b = float(np.mean(between_vals)) if between_vals else float("nan")
        ratio = mean_b / mean_w if mean_w > 0 and np.isfinite(mean_w) and np.isfinite(mean_b) else float("nan")
        rows.append(
            {
                "N": n,
                "mean_within_emd": mean_w,
                "mean_between_emd": mean_b,
                "mean_between_same_main_emd": float(np.mean(between_same)) if between_same else float("nan"),
                "mean_between_diff_main_emd": float(np.mean(between_diff)) if between_diff else float("nan"),
                "between_within_ratio": ratio,
                "n_within": len(within_vals),
                "n_between": len(between_vals),
                "n_between_same": len(between_same),
                "n_between_diff": len(between_diff),
                "std_within_emd": float(np.std(within_vals)) if within_vals else float("nan"),
                "std_between_emd": float(np.std(between_vals)) if between_vals else float("nan"),
            }
        )
    return {
        "n_regions": len(regions),
        "n_pairs": len(pairs),
        "rows": rows,
    }


def _aggregate_rows(per_slide: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool by N: weighted mean of within/between by n_within / n_between."""
    by_n: dict[int, list[dict[str, Any]]] = {}
    for slide in per_slide:
        for row in slide["rows"]:
            by_n.setdefault(int(row["N"]), []).append(row)

    out: list[dict[str, Any]] = []
    for n in sorted(by_n):
        items = by_n[n]
        tw = sum(int(r["n_within"]) for r in items)
        tb = sum(int(r["n_between"]) for r in items)
        ts = sum(int(r["n_between_same"]) for r in items)
        td = sum(int(r["n_between_diff"]) for r in items)

        def _wmean(key: str, weights: list[int]) -> float:
            num = 0.0
            den = 0.0
            for r, w in zip(items, weights):
                v = r.get(key)
                if v is None or not np.isfinite(float(v)) or w <= 0:
                    continue
                num += float(v) * w
                den += w
            return float(num / den) if den > 0 else float("nan")

        mean_w = _wmean("mean_within_emd", [int(r["n_within"]) for r in items])
        mean_b = _wmean("mean_between_emd", [int(r["n_between"]) for r in items])
        mean_same = _wmean("mean_between_same_main_emd", [int(r["n_between_same"]) for r in items])
        mean_diff = _wmean("mean_between_diff_main_emd", [int(r["n_between_diff"]) for r in items])
        ratio = mean_b / mean_w if mean_w > 0 and np.isfinite(mean_w) and np.isfinite(mean_b) else float("nan")
        out.append(
            {
                "N": n,
                "mean_within_emd": mean_w,
                "mean_between_emd": mean_b,
                "mean_between_same_main_emd": mean_same,
                "mean_between_diff_main_emd": mean_diff,
                "between_within_ratio": ratio,
                "n_within": tw,
                "n_between": tb,
                "n_between_same": ts,
                "n_between_diff": td,
                "n_slides": len(items),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_emd_vs_n(
    rows: list[dict[str, Any]],
    out_png: Path,
    *,
    title: str,
    dpi: int,
    save_pdf: bool,
    style: str = "paper",
    annotate_n: int | None = 3,
    xlabel: str = "N (top intensities per pixel)",
    ylabel: str = "Mean Wasserstein distance",
) -> None:
    if not rows:
        logger.warning("No rows to plot")
        return
    style = str(style).strip().lower()
    ns = [int(r["N"]) for r in rows]
    within = [float(r["mean_within_emd"]) for r in rows]
    between = [float(r["mean_between_emd"]) for r in rows]
    ratio = [float(r["between_within_ratio"]) for r in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.6) if style == "paper" else (7.2, 4.6))
    ax.plot(ns, within, "o-", color="#2c7bb6", label="Within region", linewidth=2, markersize=6)
    ax.plot(ns, between, "s-", color="#d7191c", label="Between regions", linewidth=2, markersize=6)

    if style == "full":
        same = [float(r.get("mean_between_same_main_emd", float("nan"))) for r in rows]
        diff = [float(r.get("mean_between_diff_main_emd", float("nan"))) for r in rows]
        if any(np.isfinite(same)):
            ax.plot(
                ns,
                same,
                "^--",
                color="#fdae61",
                label="Between (same main_cat1)",
                linewidth=1.5,
                markersize=5,
            )
        if any(np.isfinite(diff)):
            ax.plot(
                ns,
                diff,
                "v--",
                color="#abd9e9",
                label="Between (diff main_cat1)",
                linewidth=1.5,
                markersize=5,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)

    if style == "paper" and annotate_n is not None:
        n_ann = int(annotate_n)
        if n_ann in ns:
            i = ns.index(n_ann)
            r = ratio[i]
            if np.isfinite(r):
                y = between[i]
                ax.annotate(
                    f"N={n_ann}: between/within ≈ {r:.0f}×",
                    xy=(n_ann, y),
                    xytext=(n_ann + 0.6, y * 1.08 if y > 0 else y),
                    fontsize=9,
                    color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
                )

    if style == "full":
        ax2 = ax.twinx()
        ax2.plot(ns, ratio, "D:", color="#4d4d4d", label="Between / within", linewidth=1.5, markersize=5)
        ax2.set_ylabel("Between / within ratio")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=True, fontsize=8)
    else:
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    if save_pdf:
        fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("Wrote %s", out_png)


@hydra.main(version_base=None, config_path="configs", config_name="topn_region_emd")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = cfg.annotation
    analysis = cfg.analysis
    plot_cfg = cfg.plot

    ann_json = _opt_path(ann.annotation_json)
    if ann_json is None or not ann_json.is_file():
        raise FileNotFoundError(f"annotation_json not found: {ann.annotation_json}")

    spatial_mode = resolve_transpose_spatial_mode(ann)
    slides = _resolve_slide_rows(ann)
    ignore = list(ann.ignore) if ann.ignore is not None else []
    keep = set(ann.keep) if ann.keep is not None else None
    category_keys = tuple(str(k) for k in ann.category_keys)

    ns = [int(x) for x in analysis.ns]
    min_pixels = int(analysis.min_pixels)
    max_pixels = int(analysis.max_pixels)
    erode = int(analysis.interior_erode)
    max_pairs = int(analysis.max_between_pairs)
    seed = int(analysis.seed)
    normalization = str(analysis.normalization)

    plot_style = str(getattr(plot_cfg, "style", "paper"))
    annotate_n_raw = getattr(plot_cfg, "annotate_n", 3)
    annotate_n = int(annotate_n_raw) if annotate_n_raw is not None else None
    xlabel = str(getattr(plot_cfg, "xlabel", "N (top intensities per pixel)"))
    ylabel = str(getattr(plot_cfg, "ylabel", "Mean Wasserstein distance"))

    def _plot_kwargs(title: str) -> dict[str, Any]:
        return {
            "title": title,
            "dpi": int(plot_cfg.dpi),
            "save_pdf": bool(plot_cfg.save_pdf),
            "style": plot_style,
            "annotate_n": annotate_n,
            "xlabel": xlabel,
            "ylabel": ylabel,
        }

    per_slide: list[dict[str, Any]] = []
    for slide_i, (npy, index_str, _lab) in enumerate(slides):
        npy = Path(npy)
        if not npy.is_file():
            logger.warning("Missing npy %s — skip", npy)
            continue
        polys, cats, main_cats, n_keys = _load_polygons_for_stem(
            ann_json,
            index_str,
            ignore=ignore,
            keep=keep,
            category_keys=category_keys,
        )
        polys, cats, main_cats = _dedupe_polygons(polys, cats, main_cats)
        if not polys:
            logger.warning("No regions for stem %s — skip", index_str)
            continue

        file_hw = spatial_hw_from_npy(npy)
        if file_hw is None:
            raise ValueError(f"Cannot read shape: {npy}")
        shape_hw = spatial_hw_after_transpose_mode(file_hw, spatial_mode)
        logger.info(
            "Slide %s stem=%s shape=%s mode=%s regions=%d (via_keys=%d)",
            npy.name,
            index_str,
            shape_hw,
            spatial_mode,
            len(polys),
            n_keys,
        )

        msi = _load_msi_aligned(npy, spatial_mode=spatial_mode, normalization=normalization)
        if msi.shape[0] != shape_hw[0] or msi.shape[1] != shape_hw[1]:
            raise ValueError(f"MSI shape {msi.shape[:2]} != expected {shape_hw}")

        rng = np.random.default_rng(seed + slide_i)
        regions = _collect_region_spectra(
            msi,
            polys,
            main_cats,
            erode=erode,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            rng=rng,
        )
        logger.info(
            "  kept %d regions with >=%d interior pixels (max sample %d)",
            len(regions),
            min_pixels,
            max_pixels,
        )
        if len(regions) < 2:
            logger.warning("  need >=2 regions — skip")
            continue

        result = analyze_slide_regions(
            regions,
            ns,
            max_between_pairs=max_pairs,
            seed=seed + 1000 + slide_i,
        )
        result["slide"] = npy.name
        result["stem"] = index_str
        per_slide.append(result)
        _write_csv(out_dir / f"emd_vs_n__{npy.stem}.csv", result["rows"])
        row3 = next((r for r in result["rows"] if int(r["N"]) == 3), None)
        if row3 is not None:
            logger.info(
                "  n_regions=%d n_pairs=%d | N=3 within=%.4g between=%.4g ratio=%.3f",
                result["n_regions"],
                result["n_pairs"],
                row3["mean_within_emd"],
                row3["mean_between_emd"],
                row3["between_within_ratio"],
            )
        else:
            logger.info("  n_regions=%d n_pairs=%d", result["n_regions"], result["n_pairs"])
        del msi

    if not per_slide:
        raise RuntimeError("No slides produced results")

    pooled = _aggregate_rows(per_slide)
    _write_csv(out_dir / "emd_vs_n_pooled.csv", pooled)
    _plot_emd_vs_n(pooled, out_dir / "emd_vs_n.png", **_plot_kwargs(str(plot_cfg.title)))
    # Also per-slide plots
    for slide in per_slide:
        slide_title = str(plot_cfg.title)
        if slide_title:
            slide_title = f"{slide_title}\n({slide['slide']})"
        else:
            slide_title = str(slide["slide"])
        _plot_emd_vs_n(
            slide["rows"],
            out_dir / f"emd_vs_n__{Path(slide['slide']).stem}.png",
            **_plot_kwargs(slide_title),
        )

    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
