#!/usr/bin/env python3
"""
For a benchmark-style visualization PNG + MSI, sweep Gaussian blur and plot:
  (top) blurred RGB | HD edge | viz edge | h·v overlap (Dice numerator, same prep as metric)
  | (bottom) metric bars: trustworthiness, Spearman cos., continuous Dice, soft P/R,
    weighted gradient alignment (edge_agreement_gradient_alignment).

Skips ZADU in MSIVisualizationMetrics for speed (same pairwise sampling + trustworthiness
+ Spearman cosine as the rest of get_metrics).

Usage (repo root):
  python scripts/viz_blur_metrics_sweep.py \\
    --png "C:/Users/.../visualizations/Extractions__NRL4485-s2__5_bins__0.npy__8c3b690b__r0__s42__random__pca.png"

Optional:
  --npy path/to/msi.npy          # if not in config data.npy_paths
  --config scripts/configs/benchmark_parametric_methods.yaml
  --n-blur 10                    # original + N blur levels (sigmas 1..N)
  --no-show-edges                # RGB only (default: show HD + viz edge panels per blur)
  --no-square                    # linear edges for Dice/soft P/R (default: square before metrics)
  --out path/to/out.png
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from omegaconf import OmegaConf
from PIL import Image
from sklearn.manifold import trustworthiness

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark_parametric_methods import (  # noqa: E402
    _load_msi,
    _maybe_lab_to_rgb,
    _path_identifier,
    _to_uint8_rgb,
)
from debug_edge_maps import _edges_for_continuous_metrics, _normalize_map_percentile  # noqa: E402
from debug_visualization_metrics import DebugNMFEvaluator, _viz_edge_maps_multiscale  # noqa: E402
from msi_visual.metrics import MSIVisualizationMetrics, smoothness_saliency_metrics  # noqa: E402


def _parse_file_id_from_viz_stem(stem: str) -> str:
    m = re.match(r"^(.+?)__r\d+__", stem)
    if not m:
        raise ValueError(f"Could not parse file_id from viz stem (expected ...__r{{k}}__...): {stem!r}")
    return m.group(1)


def _parse_method_tag(stem: str) -> str:
    m = re.search(r"__(?:random|superpixel)__(.+)$", stem)
    return m.group(1) if m else "pca"


def _resolve_npy_path(file_id: str, cfg, npy_override: Path | None) -> Path:
    if npy_override is not None:
        p = npy_override.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--npy not found: {p}")
        if _path_identifier(p) != file_id:
            print(
                f"Warning: --npy identifier {_path_identifier(p)!r} != viz file_id {file_id!r} "
                "(continuing anyway).",
                file=sys.stderr,
            )
        return p
    paths = list(getattr(cfg.data, "npy_paths", []) or [])
    for s in paths:
        p = Path(str(s))
        if not p.is_file():
            continue
        if _path_identifier(p) == file_id:
            return p
    raise FileNotFoundError(
        f"No MSI in config data.npy_paths matches file_id={file_id!r}. "
        f"Pass --npy explicitly."
    )


def _apply_min_viz_edge_percentile(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    min_viz_edge_percentile: float,
) -> np.ndarray:
    """Match debug_visualization_metrics edge_evaluation min_viz step."""
    x = np.array(arr, dtype=np.float32, copy=True)
    if min_viz_edge_percentile > 0 and np.any(valid_mask):
        thresh = float(np.percentile(x[valid_mask], min_viz_edge_percentile))
        x = np.where((valid_mask) & (x >= thresh), x, 0.0).astype(np.float32)
    x[~valid_mask] = 0.0
    return x


def _overlap_map_for_display(
    hd_edge_norm: np.ndarray,
    viz_edge_norm: np.ndarray,
    valid_mask: np.ndarray,
    min_viz_edge_percentile: float,
    *,
    square: bool,
) -> np.ndarray:
    """Spatial h·v (after same prep as continuous Dice), scaled to [0,1] for imshow."""
    ve = _apply_min_viz_edge_percentile(viz_edge_norm, valid_mask, min_viz_edge_percentile)
    hd_d, viz_d = _edges_for_continuous_metrics(hd_edge_norm, ve, square=square)
    prod = (hd_d * viz_d).astype(np.float64)
    if not np.any(valid_mask):
        return np.zeros_like(prod, dtype=np.float32)
    v = prod[valid_mask]
    hi = float(np.percentile(v, 99.0)) if v.size else 0.0
    if hi < 1e-15:
        out = np.zeros_like(prod, dtype=np.float32)
    else:
        out = np.clip(prod / hi, 0.0, 1.0).astype(np.float32)
    out[~valid_mask] = 0.0
    return out


def _gaussian_blur_rgb_u8(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 1e-9:
        return np.asarray(img, dtype=np.uint8, order="C").copy()
    k = max(3, int(2 * round(3 * float(sigma)) + 1))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(np.asarray(img, dtype=np.uint8), (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def _trustworthiness_spearman_only(msi_norm: np.ndarray, viz_rgb_u8: np.ndarray, num_samples: int) -> tuple[float, float]:
    """Trustworthiness + Spearman cosine; no ZADU (matches MSIVisualizationMetrics sampling)."""
    random.seed(42)
    np.random.seed(42)
    m = MSIVisualizationMetrics(msi_norm, viz_rgb_u8, mask=None, num_samples=int(num_samples))
    cosine = m.data["Cosine"]
    maxabs = m.data["L-∞"]
    outputs = m.data["Output"]
    sm = smoothness_saliency_metrics(cosine, maxabs, outputs)
    tw = float(
        trustworthiness(
            m.data_subset,
            m.visualization_subset,
            metric="euclidean",
            n_neighbors=100,
        )
    )
    sp = float(sm["Spearman Cosine"])
    return tw, sp


def main() -> None:
    ap = argparse.ArgumentParser(description="Blur sweep: Trustworthiness, Spearman cosine, continuous Dice.")
    ap.add_argument(
        "--png",
        type=Path,
        default=Path(
            r"C:\Users\PahnkeLab\Desktop\maldi_msi\visualizations\Extractions__NRL4485-s2__5_bins__0.npy__8c3b690b__r0__s42__random__pca.png"
        ),
        help="Benchmark-style visualization PNG.",
    )
    ap.add_argument("--npy", type=Path, default=None, help="Explicit path to MSI .npy (optional).")
    ap.add_argument(
        "--config",
        type=Path,
        default=_REPO / "scripts" / "configs" / "benchmark_parametric_methods.yaml",
        help="OmegaConf yaml (same edge/hd settings as benchmark).",
    )
    ap.add_argument("--n-blur", type=int, default=10, help="Number of blurred variants (sigmas 1..N); sigma=0 is original.")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    ap.add_argument(
        "--no-show-edges",
        dest="show_edges",
        action="store_false",
        default=True,
        help="Omit HD/viz edge panels; only plot blurred RGB and metric bars (edges are shown by default).",
    )
    ap.add_argument(
        "--no-square",
        action="store_true",
        help="Use linear edge maps for Dice/soft P/R (default: square before continuous metrics).",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    png_path = args.png.expanduser().resolve()
    if not png_path.is_file():
        raise SystemExit(f"PNG not found: {png_path}")

    cfg = OmegaConf.load(args.config)
    cfg.edge_evaluation.metric = "continuous_dice"
    cfg.edge_evaluation.square_before_continuous_metrics = not bool(args.no_square)

    stem = png_path.stem
    file_id = _parse_file_id_from_viz_stem(stem)
    method_tag = _parse_method_tag(stem)
    npy_path = _resolve_npy_path(file_id, cfg, args.npy)

    msi_img = _load_msi(npy_path, bool(cfg.data.transpose_msi), bool(cfg.data.tic_normalize))
    msi_mask = msi_img.sum(axis=-1) > 0

    base_rgb = _to_uint8_rgb(np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8))
    if base_rgb.shape[:2] != msi_img.shape[:2]:
        raise SystemExit(
            f"Shape mismatch: viz {base_rgb.shape[:2]} vs MSI {msi_img.shape[:2]}. Resize viz or pick matching pair."
        )
    base_rgb = _maybe_lab_to_rgb(base_rgb, method_tag, cfg)
    base_rgb = base_rgb.copy()
    base_rgb[~msi_mask] = 0

    print(f"file_id={file_id}", flush=True)
    print(f"MSI: {npy_path}", flush=True)
    print(f"PNG: {png_path}", flush=True)

    edge_eval = DebugNMFEvaluator(msi_img, cfg, edge_only=True)
    num_pairwise = int(getattr(cfg.correlation, "pairwise_num_samples", 8000))

    low_pct = float(cfg.spatial_normalization.low_percentile)
    high_pct = float(cfg.spatial_normalization.high_percentile)
    hd_edge_n = _normalize_map_percentile(
        edge_eval.highd_edge,
        edge_eval.valid_mask,
        low_pct=low_pct,
        high_pct=high_pct,
    )

    n_blur = max(1, int(args.n_blur))
    sigmas = [0.0] + [float(s) for s in range(1, n_blur + 1)]
    labels = ["σ=0 (orig)"] + [f"σ={s}" for s in range(1, n_blur + 1)]

    rows: list[dict[str, float | str]] = []
    for sigma, lab in zip(sigmas, labels):
        viz = _gaussian_blur_rgb_u8(base_rgb, sigma)
        tw, spearman_cos = _trustworthiness_spearman_only(msi_img, viz, num_pairwise)
        edge_out = edge_eval.evaluate_edge_evaluation_only(viz)
        dice = float(edge_out.get("edge_agreement_continuous_dice", float("nan")))
        soft_p = float(edge_out.get("edge_agreement_soft_precision", float("nan")))
        soft_r = float(edge_out.get("edge_agreement_soft_recall", float("nan")))
        grad_al = float(edge_out.get("edge_agreement_gradient_alignment", float("nan")))
        rows.append(
            {
                "blur_label": lab,
                "sigma": float(sigma),
                "trustworthiness": tw,
                "spearman_cosine": spearman_cos,
                "continuous_dice": dice,
                "soft_precision": soft_p,
                "soft_recall": soft_r,
                "gradient_alignment": grad_al,
            }
        )
        print(
            f"{lab}: tw={tw:.4f} spearman={spearman_cos:.4f} dice={dice:.4f} "
            f"soft_P={soft_p:.4f} soft_R={soft_r:.4f} grad_align={grad_al:.4f}",
            flush=True,
        )

    min_viz_pct = float(getattr(cfg.edge_evaluation, "min_viz_edge_percentile", 0.0))
    sq_cont = bool(getattr(cfg.edge_evaluation, "square_before_continuous_metrics", False))

    # Figure: each row = (optional 4-panel edges) RGB on top, bars below
    n_levels = len(sigmas)
    fig_h = min(3.2 * n_levels, 120)
    fig_w = 20.0 if args.show_edges else 14.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    outer = GridSpec(n_levels, 1, figure=fig, hspace=0.38)

    metric_names = [
        "Trustworthiness",
        "Spearman cos.",
        "Cont. Dice",
        "Soft prec.",
        "Soft recall",
        "Grad align.",
    ]
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#44AA99"]

    for i, r in enumerate(rows):
        gss = outer[i].subgridspec(2, 1, height_ratios=[2.2, 1.0] if args.show_edges else [2.1, 1.0])
        sigma_i = float(r["sigma"])
        viz_show = _gaussian_blur_rgb_u8(base_rgb, sigma_i)

        if args.show_edges:
            g_top = gss[0].subgridspec(1, 4, wspace=0.1)
            ax_rgb = fig.add_subplot(g_top[0, 0])
            ax_hd = fig.add_subplot(g_top[0, 1])
            ax_ve = fig.add_subplot(g_top[0, 2])
            ax_ov = fig.add_subplot(g_top[0, 3])
            ax_rgb.imshow(viz_show)
            ax_rgb.set_title(f"{r['blur_label']}\nViz RGB", fontsize=9, fontweight="600")
            ax_rgb.axis("off")
            ax_hd.imshow(hd_edge_n, cmap="magma", vmin=0.0, vmax=1.0)
            ax_hd.set_title("HD edge (ref)", fontsize=9)
            ax_hd.axis("off")
            ve_raw, _ = _viz_edge_maps_multiscale(
                viz_show,
                edge_eval.valid_mask,
                edge_eval.edge_multiscale_sigmas,
                edge_eval.edge_multiscale_aggregation,
                edge_eval.edge_method_cfg,
            )
            ve_n = _normalize_map_percentile(ve_raw, edge_eval.valid_mask, low_pct=low_pct, high_pct=high_pct)
            ax_ve.imshow(ve_n, cmap="magma", vmin=0.0, vmax=1.0)
            ax_ve.set_title("Viz edge", fontsize=9)
            ax_ve.axis("off")
            ov = _overlap_map_for_display(
                hd_edge_n,
                ve_n,
                edge_eval.valid_mask,
                min_viz_pct,
                square=sq_cont,
            )
            ax_ov.imshow(ov, cmap="viridis", vmin=0.0, vmax=1.0)
            d = float(r["continuous_dice"])
            sq_note = "sq" if sq_cont else "lin"
            ax_ov.set_title(f"h·v ({sq_note})\nDice={d:.4f}", fontsize=8)
            ax_ov.axis("off")
        else:
            ax_img = fig.add_subplot(gss[0])
            ax_img.imshow(viz_show)
            ax_img.set_title(r["blur_label"], fontsize=10, fontweight="600")
            ax_img.axis("off")

        ax_bar = fig.add_subplot(gss[1])

        vals = [
            float(r["trustworthiness"]),
            float(r["spearman_cosine"]),
            float(r["continuous_dice"]),
            float(r["soft_precision"]),
            float(r["soft_recall"]),
            float(r["gradient_alignment"]),
        ]
        x = np.arange(6)
        bars = ax_bar.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_names, fontsize=6, rotation=22, ha="right")
        ax_bar.set_ylabel("value")
        ax_bar.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)
        finite_vals = [v for v in vals if np.isfinite(v)]
        ymin = (min(finite_vals + [0.0]) - 0.05) if finite_vals else -1.0
        ymax = (max(finite_vals + [0.0]) + 0.05) if finite_vals else 1.0
        if ymax - ymin < 0.15:
            ymin, ymax = ymin - 0.05, ymax + 0.05
        ax_bar.set_ylim(ymin, ymax)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.grid(axis="y", alpha=0.25, linestyle="--")
        for b, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            h = b.get_height()
            ax_bar.text(
                b.get_x() + b.get_width() / 2,
                h,
                f"{v:.3f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=6,
            )

    fig.suptitle(
        f"Blur sweep — {png_path.name}\nMSI: {npy_path.name}",
        fontsize=11,
        fontweight="600",
        y=1.002,
    )
    out_path = args.out
    if out_path is None:
        out_path = png_path.parent / f"{stem}__blur_metrics_sweep.png"
    else:
        out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out_path}", flush=True)

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "blur_label",
                "sigma",
                "trustworthiness",
                "spearman_cosine",
                "continuous_dice",
                "soft_precision",
                "soft_recall",
                "gradient_alignment",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
