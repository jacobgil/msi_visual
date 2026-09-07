#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "metric"


def _metric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_mean")]


def _build_combo_label(df: pd.DataFrame) -> pd.Series:
    return df["method"].astype(str) + " + " + df["sampling_mode"].astype(str)


def _aggregate_over_path(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=["path", "method", "sampling_mode"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    long_df["combo"] = _build_combo_label(long_df)
    grouped = (
        long_df.groupby(["combo", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            n_paths=("path", "nunique"),
        )
        .fillna({"std": 0.0})
    )
    return grouped


def _plot_metric(grouped: pd.DataFrame, metric: str, out_dir: Path) -> None:
    metric_df = grouped[grouped["metric"] == metric].copy()
    metric_df = metric_df.sort_values("mean", ascending=False)

    sns.set_theme(style="whitegrid", context="talk")
    fig_w = max(10, 1.2 * len(metric_df))
    fig, ax = plt.subplots(figsize=(fig_w, 7), constrained_layout=True)

    palette = sns.color_palette("Set2", n_colors=len(metric_df))
    bars = ax.bar(
        metric_df["combo"],
        metric_df["mean"],
        yerr=metric_df["std"],
        capsize=8,
        linewidth=1.0,
        edgecolor="#222222",
        color=palette,
        alpha=0.95,
    )

    ax.set_title(f"{metric} across method + sampling_mode", pad=14, weight="bold")
    ax.set_xlabel("Method + Sampling mode")
    ax.set_ylabel("Mean over paths")
    ax.tick_params(axis="x", rotation=35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min = float((metric_df["mean"] - metric_df["std"]).min())
    y_max = float((metric_df["mean"] + metric_df["std"]).max())
    if y_min >= 0 and y_max <= 1.05:
        ax.set_ylim(0.0, 1.05)

    label_offset = max(abs(y_max) * 0.02, 0.01)
    for bar, value in zip(bars, metric_df["mean"]):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        ax.text(x, y + label_offset, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    metric_file = _sanitize_filename(metric)
    out_path = out_dir / f"{metric_file}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate benchmark CSV over path and compare method+sampling_mode "
            "with bar charts and std error bars."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("benchmark_parametric_methods_aggregate.csv"),
        help="Input aggregate CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "benchmark_parametric_method_charts",
        help="Directory where plots and summary CSV are written.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of metric columns (must end with _mean). Default: all *_mean columns.",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    required = {"path", "method", "sampling_mode"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metric_cols = _metric_columns(df)
    if args.metrics:
        requested = [m for m in args.metrics if m.endswith("_mean")]
        unknown = sorted(set(requested).difference(metric_cols))
        if unknown:
            raise ValueError(f"Requested metrics not in CSV: {unknown}")
        metric_cols = requested

    if not metric_cols:
        raise ValueError("No *_mean metrics found to aggregate.")

    grouped = _aggregate_over_path(df, metric_cols)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.output_dir / "summary_by_method_sampling_mode.csv"
    grouped.sort_values(["metric", "mean"], ascending=[True, False]).to_csv(summary_csv, index=False)

    for metric in sorted(grouped["metric"].unique()):
        _plot_metric(grouped, metric, args.output_dir)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved {grouped['metric'].nunique()} metric chart(s) in: {args.output_dir}")


if __name__ == "__main__":
    main()
