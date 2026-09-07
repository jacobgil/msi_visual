#!/usr/bin/env python3
"""
Compare the raw vs histogram-equalized visualizations on the new metrics
(SOLACE-Dice, LAB Chroma Entropy, Luminance RMS Contrast).

Reads the two augmented CSVs written by benchmark_new_metrics.py (raw + --use-eq)
and produces a delta table + grouped-bar figures. Answers: how much of each
method's low display-metric score is a contrast/display artifact that equalization
removes, vs a genuine structural deficiency that survives equalization?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = ["SOLACE-Dice", "LAB Chroma Entropy", "Luminance RMS Contrast"]
PALETTE = ["#2E86AB", "#C73E1D"]  # raw, equalized

_BASE = Path(
    r"E:/MSImaging-data/_msi_visual/Benchmarking/mouse_brain_comprehensive_benchmark_large"
)


def _method_means(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    agg = {m: "mean" for m in METRICS}
    return df.groupby("method").agg(agg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=_BASE / "new_metrics_solace_chroma" / "benchmark_new_large_with_solace_chroma.csv")
    ap.add_argument("--eq", type=Path,
                    default=_BASE / "new_metrics_solace_chroma_eq" / "benchmark_new_large_with_solace_chroma.csv")
    ap.add_argument("--out-dir", type=Path, default=_BASE / "new_metrics_raw_vs_eq")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _method_means(args.raw)
    eq = _method_means(args.eq)
    methods = list(raw.index)

    # Delta table (sorted by SOLACE-Dice gain from equalization).
    rows = []
    for m in methods:
        row = {"method": m}
        for met in METRICS:
            r, e = float(raw.loc[m, met]), float(eq.loc[m, met])
            row[f"{met} (raw)"] = r
            row[f"{met} (eq)"] = e
            row[f"{met} Δ"] = e - r
        rows.append(row)
    tbl = pd.DataFrame(rows).sort_values("SOLACE-Dice Δ", ascending=False).reset_index(drop=True)
    tbl.to_csv(out_dir / "raw_vs_eq_comparison.csv", index=False)

    md = ["# Raw vs equalized visualizations — new metrics", "",
          "Method means over slides. **Δ = equalized − raw.** A large positive Δ means the "
          "raw score was suppressed by low display contrast (equalization recovers it); a Δ near "
          "zero (or negative) means the raw score reflects genuine structure (or that equalization "
          "injects spurious texture).", "",
          "| Method | Dice raw | Dice eq | Dice Δ | ChromaEnt raw | ChromaEnt eq | ChromaEnt Δ | LumRMS raw | LumRMS eq | LumRMS Δ |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in tbl.iterrows():
        md.append(
            f"| {r['method']} | {r['SOLACE-Dice (raw)']:.3f} | {r['SOLACE-Dice (eq)']:.3f} | "
            f"{r['SOLACE-Dice Δ']:+.3f} | {r['LAB Chroma Entropy (raw)']:.2f} | "
            f"{r['LAB Chroma Entropy (eq)']:.2f} | {r['LAB Chroma Entropy Δ']:+.2f} | "
            f"{r['Luminance RMS Contrast (raw)']:.3f} | {r['Luminance RMS Contrast (eq)']:.3f} | "
            f"{r['Luminance RMS Contrast Δ']:+.3f} |"
        )
    (out_dir / "raw_vs_eq_comparison.md").write_text("\n".join(md), encoding="utf-8")

    # Grouped-bar figure: raw vs eq for each metric.
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.0 * len(METRICS), 5.4))
    for ax, met in zip(np.atleast_1d(axes), METRICS):
        order = (eq[met] - raw[met]).sort_values(ascending=False).index.tolist()
        x = np.arange(len(order))
        w = 0.4
        ax.bar(x - w / 2, [raw.loc[m, met] for m in order], w, label="raw",
               color=PALETTE[0], edgecolor="white", linewidth=0.6)
        ax.bar(x + w / 2, [eq.loc[m, met] for m in order], w, label="equalized",
               color=PALETTE[1], edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_title(met, fontsize=11, fontweight="600")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)
    fig.suptitle("Raw vs histogram-equalized visualizations (method means over slides)",
                 fontsize=13, fontweight="600")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out_dir / "raw_vs_eq_bars.png"
    fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Slope figure: SOLACE-Dice raw -> eq per method (highlights who gains).
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for m in methods:
        r, e = float(raw.loc[m, "SOLACE-Dice"]), float(eq.loc[m, "SOLACE-Dice"])
        color = "#2A9D8F" if e >= r else "#C73E1D"
        ax.plot([0, 1], [r, e], "-o", color=color, alpha=0.8, markersize=6)
        ax.text(1.02, e, m, va="center", fontsize=8)
        ax.text(-0.02, r, m, va="center", ha="right", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "equalized"], fontsize=11)
    ax.set_ylabel("SOLACE-Dice (mean over slides)")
    ax.set_title("Effect of equalization on SOLACE-Dice\n(green = gains, red = loses)",
                 fontsize=12, fontweight="600")
    ax.set_xlim(-0.35, 1.35)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p2 = out_dir / "solace_dice_raw_vs_eq_slope.png"
    fig.savefig(p2, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote {out_dir / 'raw_vs_eq_comparison.csv'} (+ .md)")
    print(f"Wrote {p}")
    print(f"Wrote {p2}")
    print("\nSOLACE-Dice gain from equalization (Δ, sorted):")
    for _, r in tbl.iterrows():
        print(f"  {r['method']:22s} raw={r['SOLACE-Dice (raw)']:.3f} -> eq={r['SOLACE-Dice (eq)']:.3f}  "
              f"Δ={r['SOLACE-Dice Δ']:+.3f}")


if __name__ == "__main__":
    main()
