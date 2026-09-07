#!/usr/bin/env python3
"""
Across SoLaCE edge vs same-side top-N EMD (simple TOP3 stress test).

At strong SoLaCE edge pixels:
  across = EMD(top-N on +normal side, top-N on -normal side)
  along  = EMD(two patches on the same side, separated along the tangent)

R = mean(across) / mean(along). R >> 1 means top-N jumps across SoLaCE edges.

Run:
  python scripts/topn_solace_edge_emd.py
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
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import gaussian_filter

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from topn_region_emd import (  # noqa: E402
    _load_msi_aligned,
    _mean_rank_emd,
    _topn_values,
)
from view_annotation_boundaries import (  # noqa: E402
    resolve_transpose_spatial_mode,
    spatial_hw_after_transpose_mode,
    spatial_hw_from_npy,
)

logger = logging.getLogger(__name__)


def _tissue_mask(msi: np.ndarray) -> np.ndarray:
    return np.asarray(msi, dtype=np.float32).sum(axis=-1) > 0


def _compute_solace(msi: np.ndarray, mask: np.ndarray, *, rank_config: str, hd_method: str) -> np.ndarray:
    from solace_edge_aware_smoothing import _compute_solace_guide

    return _compute_solace_guide(
        msi,
        mask,
        rank_config_path=str(rank_config),
        hd_method=str(hd_method),
    )


def _edge_sites(
    solace: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float,
    max_edges: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (K, 2) array of (y, x) strong-edge sites on tissue."""
    m = np.asarray(mask, dtype=bool)
    vals = np.asarray(solace, dtype=np.float64)
    tissue_vals = vals[m]
    if tissue_vals.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    thr = float(np.percentile(tissue_vals, float(percentile)))
    ys, xs = np.nonzero(m & (vals >= thr))
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if ys.size > max_edges:
        sel = rng.choice(ys.size, size=int(max_edges), replace=False)
        ys, xs = ys[sel], xs[sel]
    return np.stack([ys, xs], axis=1).astype(np.int32)


def _unit_normals(solace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ny, nx unit gradient of smoothed SoLaCE (points toward increasing edge strength)."""
    g = gaussian_filter(np.asarray(solace, dtype=np.float64), sigma=1.0)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    n = np.sqrt(gy * gy + gx * gx) + 1e-8
    return gy / n, gx / n


def _disk_spectra(
    msi: np.ndarray,
    mask: np.ndarray,
    cy: float,
    cx: float,
    radius: int,
) -> np.ndarray | None:
    h, w = mask.shape
    r = int(max(0, radius))
    y0 = int(np.floor(cy - r))
    y1 = int(np.ceil(cy + r))
    x0 = int(np.floor(cx - r))
    x1 = int(np.ceil(cx + r))
    if y1 < 0 or x1 < 0 or y0 >= h or x0 >= w:
        return None
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(h - 1, y1)
    x1 = min(w - 1, x1)
    ys = []
    xs = []
    r2 = float(r) + 0.5
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r2 * r2 and mask[y, x]:
                ys.append(y)
                xs.append(x)
    if len(ys) < 3:
        return None
    return np.asarray(msi[np.asarray(ys), np.asarray(xs), :], dtype=np.float32)


def _pair_emd(a: np.ndarray, b: np.ndarray, n: int) -> float:
    return _mean_rank_emd(_topn_values(a, n), _topn_values(b, n))


def analyze_slide(
    msi: np.ndarray,
    solace: np.ndarray,
    *,
    ns: list[int],
    edge_percentile: float,
    max_edges: int,
    offset_px: float,
    patch_radius: int,
    seed: int,
) -> dict[str, Any]:
    mask = _tissue_mask(msi)
    rng = np.random.default_rng(seed)
    sites = _edge_sites(
        solace,
        mask,
        percentile=edge_percentile,
        max_edges=max_edges,
        rng=rng,
    )
    ny, nx = _unit_normals(solace)
    d = float(offset_px)
    r = int(patch_radius)

    # Collect valid patch triples once; reuse spectra for all N.
    pairs_across: list[tuple[np.ndarray, np.ndarray]] = []
    pairs_along: list[tuple[np.ndarray, np.ndarray]] = []
    for y, x in sites:
        y = int(y)
        x = int(x)
        vy, vx = float(ny[y, x]), float(nx[y, x])
        # Tangent = rotate normal 90°.
        ty, tx = -vx, vy
        a = _disk_spectra(msi, mask, y + d * vy, x + d * vx, r)
        b = _disk_spectra(msi, mask, y - d * vy, x - d * vx, r)
        c = _disk_spectra(msi, mask, y + d * vy + 2 * d * ty, x + d * vx + 2 * d * tx, r)
        if a is None or b is None or c is None:
            continue
        pairs_across.append((a, b))
        pairs_along.append((a, c))

    rows: list[dict[str, Any]] = []
    for n in ns:
        n = int(n)
        across_vals = [_pair_emd(a, b, n) for a, b in pairs_across]
        along_vals = [_pair_emd(a, c, n) for a, c in pairs_along]
        across_vals = [v for v in across_vals if np.isfinite(v)]
        along_vals = [v for v in along_vals if np.isfinite(v)]
        mean_across = float(np.mean(across_vals)) if across_vals else float("nan")
        mean_along = float(np.mean(along_vals)) if along_vals else float("nan")
        ratio = (
            mean_across / mean_along
            if mean_along > 0 and np.isfinite(mean_across) and np.isfinite(mean_along)
            else float("nan")
        )
        rows.append(
            {
                "N": n,
                "mean_across_emd": mean_across,
                "mean_along_emd": mean_along,
                "across_along_ratio": ratio,
                "n_pairs": min(len(across_vals), len(along_vals)),
            }
        )
    return {"n_sites": int(sites.shape[0]), "n_pairs": len(pairs_across), "rows": rows}


def _aggregate(per_slide: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_n: dict[int, list[dict[str, Any]]] = {}
    for slide in per_slide:
        for row in slide["rows"]:
            by_n.setdefault(int(row["N"]), []).append(row)
    out: list[dict[str, Any]] = []
    for n in sorted(by_n):
        items = by_n[n]
        weights = [max(1, int(r["n_pairs"])) for r in items]

        def _wmean(key: str) -> float:
            num = den = 0.0
            for r, w in zip(items, weights):
                v = float(r[key])
                if not np.isfinite(v):
                    continue
                num += v * w
                den += w
            return float(num / den) if den > 0 else float("nan")

        mean_a = _wmean("mean_across_emd")
        mean_l = _wmean("mean_along_emd")
        ratio = mean_a / mean_l if mean_l > 0 and np.isfinite(mean_a) and np.isfinite(mean_l) else float("nan")
        out.append(
            {
                "N": n,
                "mean_across_emd": mean_a,
                "mean_along_emd": mean_l,
                "across_along_ratio": ratio,
                "n_pairs": sum(int(r["n_pairs"]) for r in items),
                "n_slides": len(items),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_paper(
    rows: list[dict[str, Any]],
    out_png: Path,
    *,
    title: str,
    dpi: int,
    save_pdf: bool,
    annotate_n: int | None,
    xlabel: str,
    ylabel: str,
) -> None:
    if not rows:
        return
    ns = [int(r["N"]) for r in rows]
    across = [float(r["mean_across_emd"]) for r in rows]
    along = [float(r["mean_along_emd"]) for r in rows]
    ratio = [float(r["across_along_ratio"]) for r in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ns, along, "o-", color="#2c7bb6", label="Same side (along edge)", linewidth=2, markersize=6)
    ax.plot(ns, across, "s-", color="#d7191c", label="Across edge", linewidth=2, markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)

    if annotate_n is not None and int(annotate_n) in ns:
        i = ns.index(int(annotate_n))
        r = ratio[i]
        if np.isfinite(r):
            ax.annotate(
                f"N={annotate_n}: across/along ≈ {r:.1f}×",
                xy=(ns[i], across[i]),
                xytext=(ns[i] + 0.6, across[i] * 1.05 if across[i] > 0 else across[i]),
                fontsize=9,
                color="#333333",
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
            )
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    if save_pdf:
        fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("Wrote %s", out_png)


def _resolve_paths(cfg: DictConfig) -> list[Path]:
    raw = OmegaConf.to_container(cfg.data.paths, resolve=True)
    paths = [Path(to_absolute_path(str(p))).expanduser().resolve() for p in raw]
    slide = cfg.data.slide
    if isinstance(slide, str) and slide.strip().lower() in ("all", "*"):
        return paths
    if isinstance(slide, (list, tuple)):
        return [paths[int(i)] for i in slide]
    return [paths[int(slide)]]


@hydra.main(version_base=None, config_path="configs", config_name="topn_solace_edge_emd")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fake annotation-like node so resolve_transpose_spatial_mode works.
    spatial_mode = resolve_transpose_spatial_mode(cfg.data)
    paths = _resolve_paths(cfg)
    ns = [int(x) for x in cfg.analysis.ns]
    rank_config = str(to_absolute_path(str(cfg.solace.rank_config)))
    hd_method = str(cfg.solace.hd_method)

    per_slide: list[dict[str, Any]] = []
    for i, npy in enumerate(paths):
        if not npy.is_file():
            logger.warning("Missing %s — skip", npy)
            continue
        file_hw = spatial_hw_from_npy(npy)
        if file_hw is None:
            raise ValueError(f"Cannot read shape: {npy}")
        shape_hw = spatial_hw_after_transpose_mode(file_hw, spatial_mode)
        logger.info("Slide %s shape=%s mode=%s", npy.name, shape_hw, spatial_mode)

        msi = _load_msi_aligned(
            npy,
            spatial_mode=spatial_mode,
            normalization=str(cfg.data.normalization),
        )
        mask = _tissue_mask(msi)
        logger.info("Computing SoLaCE ...")
        solace = _compute_solace(msi, mask, rank_config=rank_config, hd_method=hd_method)

        result = analyze_slide(
            msi,
            solace,
            ns=ns,
            edge_percentile=float(cfg.analysis.edge_percentile),
            max_edges=int(cfg.analysis.max_edges),
            offset_px=float(cfg.analysis.offset_px),
            patch_radius=int(cfg.analysis.patch_radius),
            seed=int(cfg.analysis.seed) + i,
        )
        result["slide"] = npy.name
        per_slide.append(result)
        _write_csv(out_dir / f"edge_emd__{npy.stem}.csv", result["rows"])
        row3 = next((r for r in result["rows"] if int(r["N"]) == 3), None)
        logger.info(
            "  sites=%d pairs=%d | N=3 across=%.4g along=%.4g ratio=%.2f",
            result["n_sites"],
            result["n_pairs"],
            row3["mean_across_emd"] if row3 else float("nan"),
            row3["mean_along_emd"] if row3 else float("nan"),
            row3["across_along_ratio"] if row3 else float("nan"),
        )
        del msi

    if not per_slide:
        raise RuntimeError("No slides produced results")

    pooled = _aggregate(per_slide)
    _write_csv(out_dir / "edge_emd_pooled.csv", pooled)
    plot = cfg.plot
    _plot_paper(
        pooled,
        out_dir / "edge_emd_vs_n.png",
        title=str(plot.title),
        dpi=int(plot.dpi),
        save_pdf=bool(plot.save_pdf),
        annotate_n=int(plot.annotate_n) if plot.annotate_n is not None else None,
        xlabel=str(plot.xlabel),
        ylabel=str(plot.ylabel),
    )
    paper_dir = _REPO / "outputs" / "topn_solace_edge_emd" / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(paper_dir / "edge_emd_pooled.csv", pooled)
    _plot_paper(
        pooled,
        paper_dir / "edge_emd_vs_n_paper.png",
        title=str(plot.title),
        dpi=int(plot.dpi),
        save_pdf=bool(plot.save_pdf),
        annotate_n=int(plot.annotate_n) if plot.annotate_n is not None else None,
        xlabel=str(plot.xlabel),
        ylabel=str(plot.ylabel),
    )
    caption = (
        "Mean Wasserstein distance of per-pixel top-N intensities in patches on opposite "
        "sides of strong SoLaCE edges (across) versus two patches on the same side "
        "(along). Large across/along ratios indicate that top-N fingerprints jump at "
        "SoLaCE boundaries."
    )
    (paper_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    logger.info("Done. Outputs in %s and %s", out_dir, paper_dir)


if __name__ == "__main__":
    main()
