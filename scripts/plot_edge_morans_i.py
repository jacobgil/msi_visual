#!/usr/bin/env python3
"""
Moran's I spatial-coherence comparison across soft edge methods.

Reads saved raw grayscale edge maps from hd_methods_gallery runs
(``.../solace_baselines/raw_grayscale/<slide>/{method}.png``) and writes:

Global mode (default):
  - morans_i_by_slide.csv
  - morans_i_comparison.png

Local mode (``--local``): Anselin local Moran's I (LISA) with rook neighbors
  - local_morans_i_by_slide.csv
  - local_morans_i_comparison.png  (mean local I on top-p% edge pixels)
  - lisa_maps/<slide>/<method>.png  (optional, ``--save-lisa-maps``)

Example:
  python scripts/plot_edge_morans_i.py \\
    --raw-root outputs/hd_methods_gallery_solace_baselines/2026-07-19/06-22-17/solace_baselines/raw_grayscale \\
    --raw-root outputs/hd_methods_gallery_solace_baselines/2026-07-16/17-01-37/solace_baselines/raw_grayscale \\
    --local --edge-pct 10 --save-lisa-maps \\
    --out-dir outputs/hd_methods_gallery_solace_baselines/local_morans_i_comparison
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

METHODS = (
    "soft_landmark_contrast",
    "distance_multi",
    "tic_sobel",
)
METHOD_LABELS = {
    "soft_landmark_contrast": "SoLaCE",
    "distance_multi": "Difference magnitude",
    "tic_sobel": "TIC Sobel",
}
METHOD_COLORS = {
    "soft_landmark_contrast": "#1f77b4",
    "distance_multi": "#ff7f0e",
    "tic_sobel": "#2ca02c",
}


def _load_gray01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return arr / 255.0


def _rank_normalize(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map valid values to ranks in (0, 1] so scale cannot dominate Moran's I."""
    out = np.zeros_like(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    vals = np.asarray(x, dtype=np.float64)[m]
    n = int(vals.size)
    if n == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = (np.arange(1, n + 1, dtype=np.float64) / float(n))
    out[m] = ranks
    return out


def morans_i_rook(x: np.ndarray, mask: np.ndarray) -> float:
    """
    Global Moran's I with rook (4-neighbor) contiguity on a regular grid.

    Only tissue pixels (mask) participate; neighbor links must be tissue–tissue.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if int(m.sum()) < 4:
        return float("nan")

    mu = float(x[m].mean())
    z = np.zeros_like(x, dtype=np.float64)
    z[m] = x[m] - mu

    # Count each undirected edge once (right + down), then double for symmetric W.
    a_r = m[:, :-1] & m[:, 1:]
    a_d = m[:-1, :] & m[1:, :]
    num = float(np.sum(z[:, :-1][a_r] * z[:, 1:][a_r]) + np.sum(z[:-1, :][a_d] * z[1:, :][a_d]))
    w_sum = float(a_r.sum() + a_d.sum())
    num *= 2.0
    w_sum *= 2.0

    denom = float(np.sum(z[m] ** 2))
    n = float(m.sum())
    if w_sum < 1.0 or denom < 1e-12:
        return float("nan")
    return float((n / w_sum) * (num / denom))


def local_morans_i_rook(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Anselin local Moran's I (LISA) with binary rook weights.

    I_i = (z_i / m2) * sum_j w_ij z_j,  m2 = sum_k z_k^2 / n
    Outside tissue → NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    n = int(m.sum())
    if n < 4:
        return out

    mu = float(x[m].mean())
    z = np.zeros_like(x, dtype=np.float64)
    z[m] = x[m] - mu
    m2 = float(np.sum(z[m] ** 2) / float(n))
    if m2 < 1e-12:
        return out

    # Neighbor lag: sum of z over tissue rook neighbors (binary W).
    lag = np.zeros_like(z, dtype=np.float64)
    # right / left
    a_r = m[:, :-1] & m[:, 1:]
    lag[:, :-1][a_r] += z[:, 1:][a_r]
    lag[:, 1:][a_r] += z[:, :-1][a_r]
    # down / up
    a_d = m[:-1, :] & m[1:, :]
    lag[:-1, :][a_d] += z[1:, :][a_d]
    lag[1:, :][a_d] += z[:-1, :][a_d]

    out[m] = (z[m] / m2) * lag[m]
    return out


def _short_slide_label(slug: str) -> str:
    m = re.search(r"ID(\d+[a-z]?)", slug, flags=re.I)
    if m:
        return f"ID{m.group(1)}"
    m = re.search(r"(NRL\d+|Dataset\d+[^_]*)", slug, flags=re.I)
    if m:
        return m.group(1)
    return slug if len(slug) <= 22 else slug[:20] + "…"


def _slide_sort_key(slug: str) -> tuple:
    m = re.search(r"ID(\d+)([a-z]?)", slug, flags=re.I)
    if m:
        return (0, int(m.group(1)), m.group(2) or "", slug)
    m = re.search(r"NRL(\d+)", slug, flags=re.I)
    if m:
        return (1, int(m.group(1)), "", slug)
    m = re.search(r"Dataset(\d+)", slug, flags=re.I)
    if m:
        return (2, int(m.group(1)), "", slug)
    return (3, 0, "", slug)


def collect_slide_dirs(raw_roots: list[Path]) -> dict[str, Path]:
    """
    Map slide slug -> directory. Later roots override earlier ones for duplicates.
    """
    out: dict[str, Path] = {}
    for root in raw_roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"raw-root not found: {root}")
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if all((d / f"{m}.png").is_file() for m in METHODS):
                out[d.name] = d
    return out


def _prepare_x(arr: np.ndarray, mask: np.ndarray, *, rank_normalize: bool) -> np.ndarray:
    if rank_normalize:
        return _rank_normalize(arr, mask)
    x = np.asarray(arr, dtype=np.float64).copy()
    x[~mask] = 0.0
    return x


def _edge_roi_mask(x: np.ndarray, mask: np.ndarray, edge_pct: float) -> np.ndarray:
    """Top ``edge_pct`` percent of tissue pixels by edge response."""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(m, dtype=bool)
    vals = np.asarray(x, dtype=np.float64)[m]
    n = int(vals.size)
    if n == 0:
        return out
    pct = float(np.clip(edge_pct, 0.1, 100.0))
    k = max(1, int(np.ceil(n * pct / 100.0)))
    # Threshold at the k-th largest value.
    if k >= n:
        out[m] = True
        return out
    thr = np.partition(vals, n - k)[n - k]
    out[m] = np.asarray(x, dtype=np.float64)[m] >= thr
    # If ties inflate the set, keep exactly the top-k by stable rank.
    if int(out.sum()) > k:
        idx = np.flatnonzero(m)
        order = np.argsort(-np.asarray(x, dtype=np.float64)[m], kind="mergesort")
        keep = np.zeros(n, dtype=bool)
        keep[order[:k]] = True
        out = np.zeros_like(m, dtype=bool)
        out.flat[idx[keep]] = True
    return out


def compute_rows_global(slide_dirs: dict[str, Path], *, rank_normalize: bool) -> list[dict]:
    rows: list[dict] = []
    for slug in sorted(slide_dirs.keys(), key=_slide_sort_key):
        d = slide_dirs[slug]
        maps = {m: _load_gray01(d / f"{m}.png") for m in METHODS}
        mask = np.zeros(next(iter(maps.values())).shape, dtype=bool)
        for arr in maps.values():
            mask |= arr > 0
        if int(mask.sum()) < 4:
            continue
        for method, arr in maps.items():
            x = _prepare_x(arr, mask, rank_normalize=rank_normalize)
            mi = morans_i_rook(x, mask)
            rows.append(
                {
                    "slide": slug,
                    "slide_label": _short_slide_label(slug),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "morans_i": mi,
                    "n_tissue": int(mask.sum()),
                    "rank_normalize": bool(rank_normalize),
                }
            )
    return rows


def compute_rows_local(
    slide_dirs: dict[str, Path],
    *,
    rank_normalize: bool,
    edge_pct: float,
    lisa_dir: Path | None,
) -> list[dict]:
    rows: list[dict] = []
    for slug in sorted(slide_dirs.keys(), key=_slide_sort_key):
        d = slide_dirs[slug]
        maps = {m: _load_gray01(d / f"{m}.png") for m in METHODS}
        mask = np.zeros(next(iter(maps.values())).shape, dtype=bool)
        for arr in maps.values():
            mask |= arr > 0
        if int(mask.sum()) < 4:
            continue
        for method, arr in maps.items():
            x = _prepare_x(arr, mask, rank_normalize=rank_normalize)
            lisa = local_morans_i_rook(x, mask)
            edge_m = _edge_roi_mask(x, mask, edge_pct)
            tissue_vals = lisa[mask]
            edge_vals = lisa[edge_m]
            pos = tissue_vals[np.isfinite(tissue_vals) & (tissue_vals > 0)]
            rows.append(
                {
                    "slide": slug,
                    "slide_label": _short_slide_label(slug),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "local_morans_i_mean": float(np.nanmean(tissue_vals)),
                    "local_morans_i_mean_pos": float(np.nanmean(pos)) if pos.size else float("nan"),
                    "local_morans_i_edge_mean": float(np.nanmean(edge_vals)),
                    "n_tissue": int(mask.sum()),
                    "n_edge": int(edge_m.sum()),
                    "edge_pct": float(edge_pct),
                    "rank_normalize": bool(rank_normalize),
                }
            )
            if lisa_dir is not None:
                _save_lisa_map(lisa, mask, lisa_dir / slug / f"{method}.png")
    return rows


def _save_lisa_map(lisa: np.ndarray, mask: np.ndarray, path: Path) -> None:
    """Diverging RdBu map of local Moran's I; non-tissue transparent-ish dark."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    vals = lisa[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    vmax = float(np.nanpercentile(np.abs(vals), 98))
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(lisa, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    # Dim non-tissue
    overlay = np.zeros((*lisa.shape, 4), dtype=np.float32)
    overlay[~mask] = (0.15, 0.15, 0.15, 1.0)
    ax.imshow(overlay, interpolation="nearest")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Local Moran's I", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_comparison_plot(
    rows: list[dict],
    path: Path,
    *,
    value_key: str,
    ylabel: str,
    title: str,
    mean_ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    slides = sorted({r["slide"] for r in rows}, key=_slide_sort_key)
    by_slide = {s: {} for s in slides}
    for r in rows:
        by_slide[r["slide"]][r["method"]] = float(r[value_key])
    labels = [next(r["slide_label"] for r in rows if r["slide"] == s) for s in slides]

    x = np.arange(len(slides), dtype=np.float64)
    width = 0.25
    fig, axes = plt.subplots(
        1, 2, figsize=(max(10.0, 0.55 * len(slides) + 4.0), 4.8), gridspec_kw={"width_ratios": [3.2, 1.0]}
    )
    ax, ax_mean = axes

    for i, method in enumerate(METHODS):
        vals = [by_slide[s].get(method, float("nan")) for s in slides]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlim(-0.6, len(slides) - 0.4)

    means = []
    for method in METHODS:
        vals = np.asarray([by_slide[s].get(method, float("nan")) for s in slides], dtype=np.float64)
        means.append(float(np.nanmean(vals)))
    ax_mean.bar(
        np.arange(len(METHODS)),
        means,
        color=[METHOD_COLORS[m] for m in METHODS],
        edgecolor="none",
    )
    ax_mean.set_xticks(np.arange(len(METHODS)))
    ax_mean.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right", fontsize=8)
    ax_mean.set_ylabel(mean_ylabel)
    ax_mean.set_title("Mean over slides")
    ax_mean.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
    ax_mean.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(means):
        ax_mean.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        action="append",
        dest="raw_roots",
        type=Path,
        required=True,
        help="Path to raw_grayscale/ (repeatable; later roots override duplicate slides)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: first raw-root's parent / metrics_morans_i)",
    )
    p.add_argument(
        "--no-rank-normalize",
        action="store_true",
        help="Skip rank normalization (not recommended for method comparison)",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Use Anselin local Moran's I (LISA) instead of global Moran's I",
    )
    p.add_argument(
        "--edge-pct",
        type=float,
        default=10.0,
        help="For --local: summarize mean local I on top this %% of edge pixels (default: 10)",
    )
    p.add_argument(
        "--save-lisa-maps",
        action="store_true",
        help="For --local: also write per-slide LISA heatmaps under lisa_maps/",
    )
    args = p.parse_args()

    slide_dirs = collect_slide_dirs(list(args.raw_roots))
    if not slide_dirs:
        raise SystemExit("No slides found with all three method PNGs.")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(args.raw_roots[0]).resolve().parent / (
            "metrics_local_morans_i" if args.local else "metrics_morans_i"
        )

    rank_normalize = not args.no_rank_normalize

    if args.local:
        lisa_dir = out_dir / "lisa_maps" if args.save_lisa_maps else None
        rows = compute_rows_local(
            slide_dirs,
            rank_normalize=rank_normalize,
            edge_pct=float(args.edge_pct),
            lisa_dir=lisa_dir,
        )
        csv_path = out_dir / "local_morans_i_by_slide.csv"
        plot_path = out_dir / "local_morans_i_comparison.png"
        write_csv(
            rows,
            csv_path,
            [
                "slide",
                "slide_label",
                "method",
                "method_label",
                "local_morans_i_mean",
                "local_morans_i_mean_pos",
                "local_morans_i_edge_mean",
                "n_tissue",
                "n_edge",
                "edge_pct",
                "rank_normalize",
            ],
        )
        save_comparison_plot(
            rows,
            plot_path,
            value_key="local_morans_i_edge_mean",
            ylabel=f"Mean local Moran's I (top {args.edge_pct:g}% edge pixels)",
            title=f"Local Moran's I on edge ridges (rank-normalized, rook, top {args.edge_pct:g}%)",
            mean_ylabel="Mean local Moran's I",
        )
        value_key = "local_morans_i_edge_mean"
    else:
        rows = compute_rows_global(slide_dirs, rank_normalize=rank_normalize)
        csv_path = out_dir / "morans_i_by_slide.csv"
        plot_path = out_dir / "morans_i_comparison.png"
        write_csv(
            rows,
            csv_path,
            ["slide", "slide_label", "method", "method_label", "morans_i", "n_tissue", "rank_normalize"],
        )
        save_comparison_plot(
            rows,
            plot_path,
            value_key="morans_i",
            ylabel="Moran's I (higher = more spatial coherence)",
            title="Spatial coherence of edge maps (rank-normalized, rook neighbors)",
            mean_ylabel="Mean Moran's I",
        )
        value_key = "morans_i"

    by_m: dict[str, list[float]] = {m: [] for m in METHODS}
    for r in rows:
        by_m[r["method"]].append(float(r[value_key]))
    print(f"Slides: {len(slide_dirs)}  mode={'local' if args.local else 'global'}")
    for m in METHODS:
        vals = np.asarray(by_m[m], dtype=np.float64)
        print(f"  {METHOD_LABELS[m]:22s}  mean={np.nanmean(vals):.4f}  median={np.nanmedian(vals):.4f}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    if args.local and args.save_lisa_maps:
        print(f"Wrote LISA maps under {out_dir / 'lisa_maps'}")


if __name__ == "__main__":
    main()
