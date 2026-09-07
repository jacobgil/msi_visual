#!/usr/bin/env python3
"""
Plots for the cosine/L2 vs L-inf SOLACE story (why SALO / FastICA are penalised by the
general edge reference but recover under the max-channel / saliency reference).

Writes per-panel figures + a single multi-panel paper figure.

Examples:
  python scripts/benchmark_solace_linf_plots.py
  python scripts/benchmark_solace_linf_plots.py \\
    --csv .../new_metrics_solace_chroma_eq/benchmark_new_large_with_solace_chroma.csv \\
    --out-dir .../new_metrics_solace_linf_eq --label equalized
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

_BASE = Path(
    r"E:/MSImaging-data/_msi_visual/Benchmarking/mouse_brain_comprehensive_benchmark_large"
)
HIGHLIGHT = {"SaliencyOptimization", "FastICA3D"}
PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#95C623", "#6C5B7B",
    "#E07A5F", "#1B998B", "#4C5B5C", "#8338EC", "#FB5607", "#3A86FF", "#FF006E",
]


def _rank_best_first(vals: np.ndarray) -> np.ndarray:
    v = np.where(np.isfinite(vals), vals, -np.inf)
    return rankdata(-v, method="average")


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _prepare(df: pd.DataFrame) -> dict:
    methods = sorted(df["method"].unique())
    g = df.groupby("method")
    cos = g["SOLACE-Dice"].mean()
    linf = g["SOLACE-Dice L-inf"].mean()
    rank_cos = pd.Series(_rank_best_first(cos.to_numpy()), index=cos.index)
    rank_linf = pd.Series(_rank_best_first(linf.to_numpy()), index=linf.index)
    return {
        "df": df,
        "methods": methods,
        "g": g,
        "cos": cos,
        "linf": linf,
        "rank_cos": rank_cos,
        "rank_linf": rank_linf,
        "rank_change": rank_cos - rank_linf,
        "color_by": {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)},
    }


def _draw_slope(ax, d: dict) -> None:
    methods, cos, linf = d["methods"], d["cos"], d["linf"]
    rank_cos, rank_linf = d["rank_cos"], d["rank_linf"]
    anchor = HIGHLIGHT | {"TOP3", "PercentileRatio", "NMF3D", "TSNE3D", "Trimap3D"}
    for m in methods:
        r, e = float(cos[m]), float(linf[m])
        if m in HIGHLIGHT:
            color, lw, alpha, ms, z = "#2A9D8F", 3.2, 1.0, 8, 6
        else:
            color, lw, alpha, ms, z = "#B7BEC6", 1.3, 0.7, 3.5, 2
        ax.plot([0, 1], [r, e], "-o", color=color, lw=lw, alpha=alpha, markersize=ms, zorder=z)
    for m in methods:
        if m not in anchor:
            continue
        r = float(cos[m])
        ax.text(
            -0.03, r, m, va="center", ha="right", fontsize=7.5,
            fontweight="bold" if m in HIGHLIGHT else "normal",
            color="#1B7F72" if m in HIGHLIGHT else "#444444",
        )
    for m in HIGHLIGHT | {"TOP3", "PercentileRatio"}:
        e = float(linf[m])
        txt = f"{m}  ({int(rank_cos[m])}\u2192{int(rank_linf[m])})"
        ax.text(
            1.03, e, txt, va="center", ha="left", fontsize=7.5,
            fontweight="bold" if m in HIGHLIGHT else "normal",
            color="#1B7F72" if m in HIGHLIGHT else "#444444",
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cosine / L2", "L-\u221e"], fontsize=10)
    ax.set_ylabel("SOLACE-Dice")
    ax.set_xlim(-0.45, 1.7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_rank_change(ax, d: dict) -> None:
    rc = d["rank_change"].sort_values(ascending=True)
    colors = ["#2A9D8F" if v > 0 else ("#C73E1D" if v < 0 else "#9AA0A6") for v in rc.to_numpy()]
    ax.barh(np.arange(len(rc)), rc.to_numpy(), color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(np.arange(len(rc)))
    ax.set_yticklabels([f"{m}" + ("  \u2605" if m in HIGHLIGHT else "") for m in rc.index], fontsize=8)
    ax.axvline(0, color="#333333", lw=1.0)
    ax.set_xlabel("Rank change (cos \u2192 L-\u221e); + = improves")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(rc.to_numpy()):
        ax.text(
            v + (0.08 if v >= 0 else -0.08), i, f"{v:+.0f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=7.5,
        )


def _draw_family(axes, d: dict) -> None:
    families = [
        ("SOLACE-Dice", "SOLACE-Dice L-inf", "Continuous Dice"),
        ("SOLACE Soft Precision", "SOLACE Soft Precision L-inf", "Soft precision"),
        ("SOLACE Soft Recall", "SOLACE Soft Recall L-inf", "Soft recall"),
    ]
    g = d["g"]
    for ax, (ccol, lcol, title) in zip(axes, families):
        cvals = g[ccol].mean()
        lvals = g[lcol].mean()
        order = lvals.sort_values(ascending=False).index.tolist()
        x = np.arange(len(order))
        w = 0.4
        ax.bar(x - w / 2, [cvals[m] for m in order], w, label="cos/L2",
               color="#2E86AB", edgecolor="white", linewidth=0.5)
        ax.bar(x + w / 2, [lvals[m] for m in order], w, label="L-\u221e",
               color="#F18F01", edgecolor="white", linewidth=0.5)
        for i, m in enumerate(order):
            if m in HIGHLIGHT:
                ax.text(i, max(cvals[m], lvals[m]) + 0.008, "\u2605",
                        ha="center", va="bottom", fontsize=9, color="#C73E1D")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6)
        ax.set_title(title, fontsize=10, fontweight="600")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")


def _draw_fingerprint(ax, d: dict) -> None:
    df, methods, color_by = d["df"], d["methods"], d["color_by"]
    ccol = _find_col(df, "Correlation Cosine")
    lcol = _find_col(df, "Correlation L-\u221e", "Correlation L-inf", "Correlation L-∞")
    if not ccol or not lcol:
        ax.text(0.5, 0.5, "Correlation columns missing", ha="center", va="center")
        ax.axis("off")
        return
    lim_lo, lim_hi = 0.3, 1.0
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="#888888", lw=1.1, label="y = x")
    for m in methods:
        sub = df[df["method"] == m]
        xs = pd.to_numeric(sub[ccol], errors="coerce")
        ys = pd.to_numeric(sub[lcol], errors="coerce")
        ax.scatter(xs, ys, s=18, color=color_by[m], alpha=0.35, edgecolors="none")
        mx, my = float(np.nanmean(xs)), float(np.nanmean(ys))
        big = m in HIGHLIGHT
        ax.scatter(
            [mx], [my], s=200 if big else 110, color=color_by[m],
            edgecolors="black", linewidths=1.4 if big else 0.9,
            marker="*" if big else "o", label=m, zorder=5,
        )
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Correlation — Cosine")
    ax.set_ylabel("Correlation — L-\u221e")
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, loc="lower right", ncol=2, framealpha=0.95)


def save_individual(d: dict, out_dir: Path, dpi: int, label: str) -> list[Path]:
    saved: list[Path] = []
    suffix = f" ({label})" if label else ""

    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    _draw_slope(ax, d)
    ax.set_title(
        f"SOLACE-Dice under matched edge reference{suffix}\n"
        "L-\u221e / saliency optimisers (green, bold) leap up; context in grey",
        fontsize=12, fontweight="600",
    )
    fig.tight_layout()
    p = out_dir / "solace_dice_cos_to_linf_slope.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _draw_rank_change(ax, d)
    ax.set_title(f"Who benefits from the L-\u221e edge reference{suffix}", fontsize=13, fontweight="600")
    fig.tight_layout()
    p = out_dir / "solace_linf_rank_change_bars.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.6))
    _draw_family(axes, d)
    fig.suptitle(
        f"SOLACE edge family: cosine/L2 vs L-\u221e{suffix}  (\u2605 = L-\u221e optimisers)",
        fontsize=13, fontweight="600",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "solace_family_cos_vs_linf.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p)

    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    _draw_fingerprint(ax, d)
    ax.set_title(
        f"Spectral-faithfulness fingerprint{suffix}\n"
        "above the diagonal = preserves L-\u221e (saliency) structure",
        fontsize=12, fontweight="600",
    )
    fig.tight_layout()
    p = out_dir / "correlation_cosine_vs_linf.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p)
    return saved


def save_combined(d: dict, out_dir: Path, dpi: int, label: str) -> Path:
    """Paper multi-panel: A slope | B fingerprint / C family / D rank-change."""
    suffix = f" — {label}" if label else ""
    fig = plt.figure(figsize=(14.5, 12.5))
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.15, 0.95, 0.95],
        hspace=0.32, wspace=0.22,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_slope(ax_a, d)
    ax_a.set_title("A  SOLACE-Dice: cos/L2 \u2192 L-\u221e", fontsize=11, fontweight="700", loc="left")

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_fingerprint(ax_b, d)
    ax_b.set_title("B  Spectral fingerprint (existing metrics)", fontsize=11, fontweight="700", loc="left")

    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])
    # Family needs 3 axes — use a nested gridspec for row 1 spanning both cols
    # Rebuild: put family on row 1 spanning full width
    ax_c1.remove()
    ax_c2.remove()
    gs_c = gs[1, :].subgridspec(1, 3, wspace=0.28)
    axes_c = [fig.add_subplot(gs_c[0, i]) for i in range(3)]
    _draw_family(axes_c, d)
    axes_c[0].set_title("C  Continuous Dice", fontsize=10, fontweight="700", loc="left")
    axes_c[1].set_title("Soft precision", fontsize=10, fontweight="700", loc="left")
    axes_c[2].set_title("Soft recall", fontsize=10, fontweight="700", loc="left")

    ax_d = fig.add_subplot(gs[2, :])
    _draw_rank_change(ax_d, d)
    ax_d.set_title("D  Rank change under L-\u221e reference (\u2605 = saliency / L-\u221e optimisers)",
                   fontsize=11, fontweight="700", loc="left")

    fig.suptitle(
        f"Matched edge reference for saliency methods{suffix}",
        fontsize=14, fontweight="700", y=0.995,
    )
    out = out_dir / "solace_linf_combined.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    # Also PDF for papers
    out_pdf = out_dir / "solace_linf_combined.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def save_raw_vs_eq_panel(raw_csv: Path, eq_csv: Path, out_dir: Path, dpi: int) -> Path | None:
    """Side-by-side rank-change (raw vs equalized) for the L-inf story."""
    if not raw_csv.is_file() or not eq_csv.is_file():
        return None
    d_raw = _prepare(pd.read_csv(raw_csv))
    d_eq = _prepare(pd.read_csv(eq_csv))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharex=True)
    for ax, d, title in zip(axes, [d_raw, d_eq], ["Raw visualizations", "Equalized visualizations"]):
        _draw_rank_change(ax, d)
        ax.set_title(title, fontsize=12, fontweight="600")
    fig.suptitle(
        "L-\u221e reference benefit: raw vs equalized  (\u2605 = SALO / FastICA)",
        fontsize=13, fontweight="700",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "solace_linf_rank_change_raw_vs_eq.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Also slope comparison: SALO/FastICA raw vs eq under both refs
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    x = np.array([0, 1, 2, 3], dtype=float)
    xticks = ["raw\ncos", "raw\nL-\u221e", "eq\ncos", "eq\nL-\u221e"]
    for m, color in [("SaliencyOptimization", "#2A9D8F"), ("FastICA3D", "#E07A5F"),
                     ("TOP3", "#4C5B5C"), ("SpearmanOptimization", "#8338EC")]:
        ys = [
            float(d_raw["cos"][m]), float(d_raw["linf"][m]),
            float(d_eq["cos"][m]), float(d_eq["linf"][m]),
        ]
        lw = 2.8 if m in HIGHLIGHT else 1.6
        ax.plot(x, ys, "-o", color=color, lw=lw, markersize=7, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_ylabel("SOLACE-Dice")
    ax.set_title("SALO / FastICA across display normalization and edge reference",
                 fontsize=12, fontweight="600")
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out2 = out_dir / "salo_across_eq_and_linf.png"
    fig.savefig(out2, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out2.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=Path,
        default=_BASE / "new_metrics_solace_chroma" / "benchmark_new_large_with_solace_chroma.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=_BASE / "new_metrics_solace_linf")
    ap.add_argument("--label", type=str, default="raw", help="Figure subtitle label (raw / equalized).")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--also-eq", action="store_true",
        help="Also generate equalized plots + raw-vs-eq comparison panels.",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "SOLACE-Dice L-inf" not in df.columns:
        raise SystemExit("CSV missing 'SOLACE-Dice L-inf' — re-run benchmark_new_metrics.py first.")
    d = _prepare(df)

    saved = save_individual(d, out_dir, args.dpi, args.label)
    combined = save_combined(d, out_dir, args.dpi, args.label)
    saved.append(combined)
    saved.append(combined.with_suffix(".pdf"))

    if args.also_eq:
        eq_csv = _BASE / "new_metrics_solace_chroma_eq" / "benchmark_new_large_with_solace_chroma.csv"
        eq_out = _BASE / "new_metrics_solace_linf_eq"
        if eq_csv.is_file():
            d_eq = _prepare(pd.read_csv(eq_csv))
            eq_out.mkdir(parents=True, exist_ok=True)
            saved += save_individual(d_eq, eq_out, args.dpi, "equalized")
            c2 = save_combined(d_eq, eq_out, args.dpi, "equalized")
            saved.append(c2)
            saved.append(c2.with_suffix(".pdf"))
            cmp_out = _BASE / "new_metrics_solace_linf"
            p = save_raw_vs_eq_panel(args.csv, eq_csv, cmp_out, args.dpi)
            if p is not None:
                saved.append(p)
                saved.append(p.parent / "salo_across_eq_and_linf.png")
        else:
            print(f"Equalized CSV not found: {eq_csv}")

    for p in saved:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
