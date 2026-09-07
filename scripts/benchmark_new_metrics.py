#!/usr/bin/env python3
"""
Augment an existing comprehensive-benchmark CSV with two new metric families and
build paper-ready tables + figures.

New metrics per (method, slide):
  SOLACE edge agreement (HD reference = soft_landmark_contrast, "SOLACE"):
    - SOLACE-Dice              (edge_agreement_continuous_dice)
    - SOLACE Soft Precision    (edge_agreement_soft_precision)
    - SOLACE Soft Recall       (edge_agreement_soft_recall)
    - SOLACE Grad. Alignment   (edge_agreement_gradient_alignment)
  Chroma / contrast (per-visualization, no reference):
    - LAB Chroma Entropy       (lab_chroma_entropy)
    - LAB Mean Chroma          (lab_mean_chroma)
    - Luminance RMS Contrast   (luminance_rms_contrast)

Inputs are taken from the benchmark CSV (existing metrics + .npy file locations) and
the pre-rendered visualization PNGs (``<npy-stem>_<method>.png``) sitting next to it.

Usage (repo root), maldi env:
  python scripts/benchmark_new_metrics.py \
    --csv "E:/MSImaging-data/_msi_visual/Benchmarking/mouse_brain_comprehensive_benchmark_large/benchmark_new_large.csv"
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Import torch first on Windows so it loads its own DLLs (OpenMP/MKL) before
# cv2/others pull in incompatible copies (avoids WinError 1114 on c10.dll).
try:
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    pass

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import rankdata

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmark_parametric_methods import _load_msi, _to_uint8_rgb  # noqa: E402
from debug_edge_maps import _compute_viz_contrast_color_metrics  # noqa: E402
from debug_visualization_metrics import DebugNMFEvaluator  # noqa: E402

# New metric columns: friendly name -> source key produced by the pipeline.
SOLACE_METRICS = {
    "SOLACE-Dice": "edge_agreement_continuous_dice",
    "SOLACE Soft Precision": "edge_agreement_soft_precision",
    "SOLACE Soft Recall": "edge_agreement_soft_recall",
    "SOLACE Grad. Alignment": "edge_agreement_gradient_alignment",
}
# L-infinity (max-channel) HD-reference variants of the SOLACE edge metrics.
# Matched reference for methods that optimize L-inf / saliency structure (e.g. SALO).
SOLACE_LINF_METRICS = {
    "SOLACE-Dice L-inf": "edge_agreement_linf_continuous_dice",
    "SOLACE Soft Precision L-inf": "edge_agreement_linf_soft_precision",
    "SOLACE Soft Recall L-inf": "edge_agreement_linf_soft_recall",
    "SOLACE Grad. Alignment L-inf": "edge_agreement_linf_gradient_alignment",
}
CHROMA_METRICS = {
    "LAB Chroma Entropy": "lab_chroma_entropy",
    "LAB Mean Chroma": "lab_mean_chroma",
    "Luminance RMS Contrast": "luminance_rms_contrast",
}
NEW_METRIC_COLS = list(SOLACE_METRICS) + list(CHROMA_METRICS)
# Stored in the augmented CSV but kept out of the headline figures/ranks.
EXTRA_METRIC_COLS = list(SOLACE_LINF_METRICS)

# Existing metrics carried into the summary table for context.
CONTEXT_METRICS = ["Trustworthiness", "Spearman Cosine", "Avg. Saliency gamma=2.0"]

PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#95C623", "#6C5B7B", "#E07A5F", "#1B998B", "#4C5B5C",
    "#8338EC", "#FB5607", "#3A86FF", "#FF006E",
]


def _resolve_png(viz_dir: Path, npy_path: str, method: str, use_eq: bool) -> Path | None:
    stem = Path(npy_path).stem
    suffix = "_eq" if use_eq else ""
    cand = viz_dir / f"{stem}_{method}{suffix}.png"
    if cand.is_file():
        return cand
    # Fall back to the non-eq variant if the eq is missing (and vice-versa).
    alt = viz_dir / f"{stem}_{method}.png"
    return alt if alt.is_file() else None


def _prep_viz(png_path: Path, msi_shape_hw: tuple[int, int], valid_mask: np.ndarray) -> np.ndarray:
    rgb = _to_uint8_rgb(np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8))
    if rgb.shape[:2] != msi_shape_hw:
        rgb = cv2.resize(rgb, (msi_shape_hw[1], msi_shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    rgb = rgb.copy()
    rgb[~valid_mask] = 0
    return rgb


def compute_new_metrics(df: pd.DataFrame, viz_dir: Path, cfg, use_eq: bool) -> pd.DataFrame:
    out = df.copy()
    for col in NEW_METRIC_COLS + EXTRA_METRIC_COLS:
        out[col] = np.nan

    evaluator_cache: dict[str, DebugNMFEvaluator] = {}
    n_ok = 0
    n_missing = 0
    for path, sub in out.groupby("path"):
        npy_path = Path(str(path))
        if not npy_path.is_file():
            print(f"  [skip] MSI not found: {npy_path}")
            n_missing += len(sub)
            continue
        random.seed(0)
        np.random.seed(0)
        if str(path) not in evaluator_cache:
            msi = _load_msi(npy_path, bool(cfg.data.transpose_msi), bool(cfg.data.tic_normalize))
            evaluator_cache[str(path)] = DebugNMFEvaluator(msi, cfg, edge_only=True)
        ev = evaluator_cache[str(path)]
        valid_mask = ev.valid_mask
        shape_hw = valid_mask.shape
        print(f"[MSI] {npy_path.name}  ({shape_hw[0]}x{shape_hw[1]}, {len(sub)} methods)")
        for idx, row in sub.iterrows():
            method = str(row["method"])
            png = _resolve_png(viz_dir, str(path), method, use_eq)
            if png is None:
                print(f"    [miss] no PNG for {method}")
                n_missing += 1
                continue
            try:
                viz = _prep_viz(png, shape_hw, valid_mask)
                edge = ev.evaluate_edge_evaluation_only(viz)
                color = _compute_viz_contrast_color_metrics(viz, valid_mask)
                for name, key in {**SOLACE_METRICS, **SOLACE_LINF_METRICS}.items():
                    out.at[idx, name] = float(edge.get(key, np.nan))
                for name, key in CHROMA_METRICS.items():
                    out.at[idx, name] = float(color.get(key, np.nan))
                n_ok += 1
                print(
                    f"    {method:22s} Dice={out.at[idx, 'SOLACE-Dice']:.4f} "
                    f"ChromaEnt={out.at[idx, 'LAB Chroma Entropy']:.3f} "
                    f"LumRMS={out.at[idx, 'Luminance RMS Contrast']:.4f}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"    [error] {method}: {e}")
                n_missing += 1
    print(f"\nComputed new metrics for {n_ok} rows ({n_missing} missing/failed).")
    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = NEW_METRIC_COLS + [m for m in CONTEXT_METRICS if m in df.columns]
    rows = []
    for method, sub in df.groupby("method"):
        row: dict[str, object] = {"method": method, "n_slides": int(sub["path"].nunique())}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            row[f"{m} mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{m} std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values("SOLACE-Dice mean", ascending=False).reset_index(drop=True)


# All ranked metrics are higher-is-better. NaN is treated as the worst outcome.
RANK_METRICS = NEW_METRIC_COLS
RANK_CONTEXT = ["Trustworthiness", "Spearman Cosine", "Avg. Saliency gamma=2.0"]


def build_avg_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Rank methods within each slide (1 = best), then average ranks over slides.

    Returns a per-method table with ``<metric> avg_rank`` columns plus overall
    averages over the SOLACE metrics, the chroma metrics, and all new metrics.
    """
    metrics = [m for m in (RANK_METRICS + RANK_CONTEXT) if m in df.columns]
    methods = sorted(df["method"].unique())
    # Accumulate per-slide ranks: {metric: {method: [ranks...]}}
    acc: dict[str, dict[str, list[float]]] = {m: {mth: [] for mth in methods} for m in metrics}

    for _path, sub in df.groupby("path"):
        sub = sub.drop_duplicates(subset="method")
        present = sub["method"].tolist()
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").to_numpy(dtype=float)
            # Higher is better -> best gets rank 1. NaN -> -inf so it ranks last.
            vals = np.where(np.isfinite(vals), vals, -np.inf)
            ranks = rankdata(-vals, method="average")
            for mth, rk in zip(present, ranks):
                acc[m][mth].append(float(rk))

    rows = []
    for mth in methods:
        row: dict[str, object] = {"method": mth}
        for m in metrics:
            rks = acc[m][mth]
            row[f"{m} avg_rank"] = float(np.mean(rks)) if rks else float("nan")
        solace_ranks = [row[f"{m} avg_rank"] for m in SOLACE_METRICS if f"{m} avg_rank" in row]
        chroma_ranks = [row[f"{m} avg_rank"] for m in CHROMA_METRICS if f"{m} avg_rank" in row]
        new_ranks = [row[f"{m} avg_rank"] for m in NEW_METRIC_COLS if f"{m} avg_rank" in row]
        row["SOLACE avg_rank"] = float(np.mean(solace_ranks)) if solace_ranks else float("nan")
        row["Chroma avg_rank"] = float(np.mean(chroma_ranks)) if chroma_ranks else float("nan")
        row["Overall new-metric avg_rank"] = float(np.mean(new_ranks)) if new_ranks else float("nan")
        rows.append(row)
    ranks_df = pd.DataFrame(rows)
    return ranks_df.sort_values("Overall new-metric avg_rank").reset_index(drop=True)


def write_rank_tables(ranks: pd.DataFrame, out_dir: Path) -> None:
    ranks.to_csv(out_dir / "new_metrics_avg_ranks.csv", index=False)
    cols = (["Overall new-metric avg_rank", "SOLACE avg_rank", "Chroma avg_rank"]
            + [f"{m} avg_rank" for m in NEW_METRIC_COLS])
    header = ["Method"] + [c.replace(" avg_rank", "") for c in cols]
    md = ["# Average ranks (lower = better)", "",
          "Methods are ranked 1..N within each slide (1 = best) for every metric, then ranks "
          "are averaged over slides. All metrics are treated as higher-is-better; NaN outputs "
          "(e.g. Trimap3D blanks) rank last.", "",
          "| " + " | ".join(header) + " |",
          "| " + " | ".join("---" for _ in header) + " |"]
    for _, r in ranks.iterrows():
        cells = [str(r["method"])] + [f"{r[c]:.2f}" for c in cols]
        md.append("| " + " | ".join(cells) + " |")
    (out_dir / "new_metrics_avg_ranks.md").write_text("\n".join(md), encoding="utf-8")


def write_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    summary.to_csv(out_dir / "new_metrics_summary.csv", index=False)

    # Markdown
    cols = NEW_METRIC_COLS + [m for m in CONTEXT_METRICS if f"{m} mean" in summary.columns]
    header = ["Method"] + cols
    md = ["# New benchmark metrics (SOLACE edge agreement + chroma)", "",
          "Mean ± SD over slides. **SOLACE** uses the soft-landmark-contrast HD edge map as "
          "reference; chroma / contrast are per-visualization color statistics.", "",
          "| " + " | ".join(header) + " |",
          "| " + " | ".join("---" for _ in header) + " |"]
    for _, r in summary.iterrows():
        cells = [str(r["method"])]
        for m in cols:
            cells.append(f"{r[f'{m} mean']:.3f} ± {r[f'{m} std']:.3f}")
        md.append("| " + " | ".join(cells) + " |")
    (out_dir / "new_metrics_summary.md").write_text("\n".join(md), encoding="utf-8")

    # LaTeX (SOLACE + chroma headline metrics)
    tex_cols = ["SOLACE-Dice", "SOLACE Soft Precision", "SOLACE Soft Recall",
                "LAB Chroma Entropy", "Luminance RMS Contrast"]
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{SOLACE edge-agreement and chroma metrics on the mouse-brain benchmark "
             r"(mean$\pm$SD over sections).}",
             r"\label{tab:new_metrics}",
             r"\begin{tabular}{l" + "c" * len(tex_cols) + "}", r"\toprule",
             "Method & " + " & ".join(c.replace("SOLACE-Dice", "SOLACE Dice") for c in tex_cols) + r" \\",
             r"\midrule"]
    for _, r in summary.iterrows():
        cells = [str(r["method"])]
        for m in tex_cols:
            cells.append(f"{r[f'{m} mean']:.3f}$\\pm${r[f'{m} std']:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "new_metrics_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _bar_panel(ax, summary: pd.DataFrame, metric: str, color_by_method: dict[str, str]) -> None:
    s = summary.sort_values(f"{metric} mean", ascending=False)
    methods = s["method"].tolist()
    means = s[f"{metric} mean"].to_numpy()
    stds = s[f"{metric} std"].to_numpy()
    colors = [color_by_method[m] for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.7,
           capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=7)
    ax.set_title(metric, fontsize=10, fontweight="600")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _rank_series(values: pd.Series) -> pd.Series:
    """Rank higher-is-better values: 1 = best. NaN ranks last."""
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    v = np.where(np.isfinite(v), v, -np.inf)
    return pd.Series(rankdata(-v, method="average"), index=values.index)


def write_linf_comparison(df: pd.DataFrame, out_dir: Path, color_by_method: dict[str, str],
                          dpi: int = 300) -> list[Path]:
    """Compare the cosine/L2 SOLACE reference vs the L-inf reference per method.

    Motivation: methods that optimize L-inf / max-channel (saliency) structure (SALO)
    should be judged against the L-inf HD edge, not only the general one.
    """
    g = df.groupby("method")
    cos = g["SOLACE-Dice"].mean()
    linf = g["SOLACE-Dice L-inf"].mean()
    tbl = pd.DataFrame({"SOLACE-Dice (cos ref)": cos, "SOLACE-Dice (L-inf ref)": linf})
    tbl["Delta (L-inf - cos)"] = tbl["SOLACE-Dice (L-inf ref)"] - tbl["SOLACE-Dice (cos ref)"]
    tbl["rank cos"] = _rank_series(tbl["SOLACE-Dice (cos ref)"])
    tbl["rank L-inf"] = _rank_series(tbl["SOLACE-Dice (L-inf ref)"])
    tbl["rank change (cos->L-inf)"] = tbl["rank cos"] - tbl["rank L-inf"]  # +ve = improves
    tbl = tbl.sort_values("rank change (cos->L-inf)", ascending=False)
    tbl.to_csv(out_dir / "solace_dice_cos_vs_linf.csv")

    md = ["# SOLACE-Dice: cosine/L2 reference vs L-inf reference", "",
          "Method means over slides. Positive **rank change** means the method climbs the "
          "ranking when judged against the L-inf (max-channel / saliency) HD edge instead of the "
          "general one. Methods that optimize L-inf structure (e.g. SALO) are expected to gain.", "",
          "| Method | Dice (cos) | Dice (L-inf) | Delta | rank cos | rank L-inf | rank change |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for m, r in tbl.iterrows():
        md.append(
            f"| {m} | {r['SOLACE-Dice (cos ref)']:.3f} | {r['SOLACE-Dice (L-inf ref)']:.3f} | "
            f"{r['Delta (L-inf - cos)']:+.3f} | {r['rank cos']:.0f} | {r['rank L-inf']:.0f} | "
            f"{r['rank change (cos->L-inf)']:+.0f} |"
        )
    (out_dir / "solace_dice_cos_vs_linf.md").write_text("\n".join(md), encoding="utf-8")

    # Grouped bars: cos ref vs L-inf ref, sorted by L-inf reference.
    order = tbl.sort_values("SOLACE-Dice (L-inf ref)", ascending=False).index.tolist()
    x = np.arange(len(order))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(x - w / 2, [tbl.loc[m, "SOLACE-Dice (cos ref)"] for m in order], w,
           label="cosine/L2 reference", color="#2E86AB", edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, [tbl.loc[m, "SOLACE-Dice (L-inf ref)"] for m in order], w,
           label="L-inf reference", color="#F18F01", edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("SOLACE-Dice (mean over slides)")
    ax.set_title("SOLACE-Dice vs reference edge: cosine/L2 vs L-inf", fontsize=13, fontweight="600")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p = out_dir / "solace_dice_cos_vs_linf_bars.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [p]


def make_rank_figures(ranks: pd.DataFrame, out_dir: Path, color_by_method: dict[str, str],
                      dpi: int = 300) -> list[Path]:
    saved: list[Path] = []
    n_methods = len(ranks)

    # A. Overall new-metric average rank (lower = better).
    s = ranks.sort_values("Overall new-metric avg_rank")
    methods = s["method"].tolist()
    vals = s["Overall new-metric avg_rank"].to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, color=[color_by_method[m] for m in methods],
                  edgecolor="white", linewidth=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Average rank over new metrics (lower = better)")
    ax.set_title("Overall average rank across SOLACE + chroma metrics", fontsize=13, fontweight="600")
    ax.axhline((n_methods + 1) / 2.0, color="#888888", linestyle="--", linewidth=1.0,
               alpha=0.8, label="chance (mid-rank)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    p = out_dir / "new_metrics_avg_rank_bar.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p)

    # B. Rank heatmap (methods x metrics), ordered by overall rank.
    metric_cols = [f"{m} avg_rank" for m in NEW_METRIC_COLS]
    mat = s[metric_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.15 * len(NEW_METRIC_COLS) + 3.5, 0.5 * n_methods + 2.0))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=n_methods)
    ax.set_xticks(np.arange(len(NEW_METRIC_COLS)))
    ax.set_xticklabels(NEW_METRIC_COLS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="black")
    ax.set_title("Average rank per metric (1 = best)", fontsize=13, fontweight="600")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("avg rank")
    fig.tight_layout()
    p = out_dir / "new_metrics_rank_heatmap.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p)
    return saved


def make_figures(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, dpi: int = 300) -> list[Path]:
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": "white",
                         "axes.facecolor": "#FAFBFC"})
    all_methods = sorted(df["method"].unique())
    color_by_method = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(all_methods)}
    saved: list[Path] = []

    # 1. Grid of per-metric bar charts (mean±SD over slides).
    ncols = 3
    nrows = int(np.ceil(len(NEW_METRIC_COLS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, metric in zip(axes, NEW_METRIC_COLS):
        _bar_panel(ax, summary, metric, color_by_method)
    for ax in axes[len(NEW_METRIC_COLS):]:
        ax.axis("off")
    fig.suptitle("New benchmark metrics per method (mean ± SD over slides)",
                 fontsize=13, fontweight="600")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out_dir / "new_metrics_bars.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p)

    # 2. Trade-off scatter: SOLACE-Dice vs LAB Chroma Entropy (per-slide points + method means).
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for m in all_methods:
        sub = df[df["method"] == m]
        xs = pd.to_numeric(sub["SOLACE-Dice"], errors="coerce")
        ys = pd.to_numeric(sub["LAB Chroma Entropy"], errors="coerce")
        ax.scatter(xs, ys, s=28, color=color_by_method[m], alpha=0.45, edgecolors="none")
        mx, my = float(np.nanmean(xs)), float(np.nanmean(ys))
        ax.scatter([mx], [my], s=170, color=color_by_method[m], edgecolors="black",
                   linewidths=1.1, label=m, zorder=5)
    ax.set_xlabel("SOLACE-Dice (edge agreement)", fontsize=11)
    ax.set_ylabel("LAB Chroma Entropy (color variety)", fontsize=11)
    ax.set_title("Edge fidelity vs color variety", fontsize=13, fontweight="600")
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.95)
    fig.tight_layout()
    p = out_dir / "solace_dice_vs_chroma_scatter.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p)

    # 3. SOLACE-Dice vs Trustworthiness (if present): does edge fidelity track neighbor preservation?
    if "Trustworthiness" in df.columns:
        fig, ax = plt.subplots(figsize=(8.4, 6.2))
        for m in all_methods:
            sub = df[df["method"] == m]
            xs = pd.to_numeric(sub["SOLACE-Dice"], errors="coerce")
            ys = pd.to_numeric(sub["Trustworthiness"], errors="coerce")
            mx, my = float(np.nanmean(xs)), float(np.nanmean(ys))
            ax.scatter(xs, ys, s=28, color=color_by_method[m], alpha=0.45, edgecolors="none")
            ax.scatter([mx], [my], s=170, color=color_by_method[m], edgecolors="black",
                       linewidths=1.1, label=m, zorder=5)
        ax.set_xlabel("SOLACE-Dice (edge agreement)", fontsize=11)
        ax.set_ylabel("Trustworthiness (existing metric)", fontsize=11)
        ax.set_title("SOLACE edge agreement vs Trustworthiness", fontsize=13, fontweight="600")
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.95)
        fig.tight_layout()
        p = out_dir / "solace_dice_vs_trustworthiness_scatter.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(p)

    return saved


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Add SOLACE + chroma metrics to a benchmark CSV.")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(
            r"E:/MSImaging-data/_msi_visual/Benchmarking/mouse_brain_comprehensive_benchmark_large/benchmark_new_large.csv"
        ),
    )
    ap.add_argument("--viz-dir", type=Path, default=None, help="PNG folder (default: CSV parent).")
    ap.add_argument(
        "--config",
        type=Path,
        default=_REPO / "scripts" / "configs" / "benchmark_parametric_methods.yaml",
    )
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output folder (default: <viz-dir>/new_metrics_solace_chroma).")
    ap.add_argument("--use-eq", action="store_true",
                    help="Use the histogram-equalized (_eq) PNGs instead of the raw ones.")
    ap.add_argument("--reuse-augmented", action="store_true",
                    help="Reuse an existing *_with_solace_chroma.csv and only rebuild tables/figures.")
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def main() -> None:
    args = get_args()
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    viz_dir = (args.viz_dir or csv_path.parent).expanduser().resolve()
    out_dir = (args.out_dir or (viz_dir / "new_metrics_solace_chroma")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    print(f"CSV      : {csv_path}")
    print(f"Viz dir  : {viz_dir}  (use_eq={args.use_eq})")
    print(f"Out dir  : {out_dir}")
    print(f"HD method: {cfg.edge_detection.hd_method} | edge metric: {cfg.edge_evaluation.metric}\n")

    augmented = out_dir / (csv_path.stem + "_with_solace_chroma.csv")
    if args.reuse_augmented and augmented.is_file():
        print(f"Reusing existing augmented CSV (skip metric recompute): {augmented}\n")
        df = pd.read_csv(augmented)
    else:
        df = pd.read_csv(csv_path)
        df = compute_new_metrics(df, viz_dir, cfg, args.use_eq)
        df.to_csv(augmented, index=False)
        print(f"\nWrote augmented CSV: {augmented}")

    summary = build_summary(df)
    write_tables(summary, out_dir)
    print(f"Wrote tables: {out_dir / 'new_metrics_summary.csv'} (+ .md, .tex)")

    ranks = build_avg_ranks(df)
    write_rank_tables(ranks, out_dir)
    print(f"Wrote rank tables: {out_dir / 'new_metrics_avg_ranks.csv'} (+ .md)")

    all_methods = sorted(df["method"].unique())
    color_by_method = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(all_methods)}
    figs = make_figures(df, summary, out_dir, dpi=args.dpi)
    figs += make_rank_figures(ranks, out_dir, color_by_method, dpi=args.dpi)
    if "SOLACE-Dice L-inf" in df.columns:
        figs += write_linf_comparison(df, out_dir, color_by_method, dpi=args.dpi)
    for p in figs:
        print(f"Wrote figure: {p}")

    print("\n=== Method ranking by SOLACE-Dice (mean over slides) ===")
    for _, r in summary.iterrows():
        print(
            f"  {str(r['method']):22s} Dice={r['SOLACE-Dice mean']:.3f}  "
            f"SoftP={r['SOLACE Soft Precision mean']:.3f}  "
            f"SoftR={r['SOLACE Soft Recall mean']:.3f}  "
            f"ChromaEnt={r['LAB Chroma Entropy mean']:.3f}  "
            f"LumRMS={r['Luminance RMS Contrast mean']:.4f}"
        )

    print("\n=== Average ranks (lower = better) ===")
    for _, r in ranks.iterrows():
        print(
            f"  {str(r['method']):22s} overall={r['Overall new-metric avg_rank']:.2f}  "
            f"SOLACE={r['SOLACE avg_rank']:.2f}  Chroma={r['Chroma avg_rank']:.2f}"
        )


if __name__ == "__main__":
    main()
