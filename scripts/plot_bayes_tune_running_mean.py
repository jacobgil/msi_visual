#!/usr/bin/env python3
"""
Running-mean and rolling-median convergence plots from a ``bayes_tune_mics_lmc.py`` trials CSV.

For each objective metric (composite score + ``objective_norm__*`` columns), plots
per-trial values, the cumulative running mean, and a trailing rolling median vs
iteration, with shaded phase regions (OFAT, hill, bayes, …) inferred from the
``phase`` column.

The rolling median reflects recent typical trial quality and can recover after early
OFAT failures, unlike the cumulative running mean.

Example::

    python scripts/plot_bayes_tune_running_mean.py \\
        --csv outputs/bayes_tune_mics_lmc/2026-05-18/20-22-35/bayes_mics_trials.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TRIALS_NAME = "bayes_mics_trials.csv"

_OBJECTIVE_METRIC_ORDER: tuple[str, ...] = (
    "score",
    "objective_norm__continuous_dice",
    "objective_norm__luminance_rms_contrast",
    "objective_norm__lab_chroma_entropy",
)

_METRIC_LABELS: dict[str, str] = {
    "score": "Composite objective",
    "objective_norm__continuous_dice": "Normalized continuous Dice",
    "objective_norm__luminance_rms_contrast": "Normalized luminance RMS contrast",
    "objective_norm__lab_chroma_entropy": "Normalized LAB chroma entropy",
}

_PHASE_COLORS: dict[str, str] = {
    "ofat": "#dbeafe",
    "hill": "#fef3c7",
    "bayes": "#dcfce7",
    "bayesian": "#dcfce7",
}

_PHASE_LABELS: dict[str, str] = {
    "ofat": "OFAT",
    "hill": "Hill climbing",
    "bayes": "Bayesian",
    "bayesian": "Bayesian",
}


def _metric_display_name(col: str) -> str:
    if col in _METRIC_LABELS:
        return _METRIC_LABELS[col]
    if col.startswith("objective_norm__"):
        return col.removeprefix("objective_norm__").replace("_", " ").title()
    return col.replace("_", " ").title()


def _objective_metric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in _OBJECTIVE_METRIC_ORDER:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            cols.append(c)
    for c in sorted(df.columns):
        if c.startswith("objective_norm__") and c not in cols:
            if pd.to_numeric(df[c], errors="coerce").notna().any():
                cols.append(c)
    return cols


def _running_mean(values: pd.Series) -> pd.Series:
    """Cumulative mean; NaNs excluded from the running count."""
    v = pd.to_numeric(values, errors="coerce")
    return v.expanding(min_periods=1).mean()


def _rolling_median(values: pd.Series, window: int) -> pd.Series:
    """Trailing median over the last ``window`` trials (min 1)."""
    v = pd.to_numeric(values, errors="coerce")
    w = max(1, int(window))
    return v.rolling(window=w, min_periods=1).median()


def _phase_spans(df: pd.DataFrame) -> list[tuple[str, float, float, int]]:
    """
    Return contiguous ``(phase, iter_start, iter_end, n_trials)`` blocks in iteration order.
    """
    if df.empty or "phase" not in df.columns:
        return []

    work = df.sort_values("iteration").reset_index(drop=True)
    spans: list[tuple[str, float, float, int]] = []
    phase = str(work.loc[0, "phase"]).strip().lower()
    i0 = 0
    for i in range(1, len(work)):
        p = str(work.loc[i, "phase"]).strip().lower()
        if p != phase:
            spans.append(
                (
                    phase,
                    float(work.loc[i0, "iteration"]),
                    float(work.loc[i - 1, "iteration"]),
                    i - i0,
                )
            )
            phase = p
            i0 = i
    spans.append(
        (
            phase,
            float(work.loc[i0, "iteration"]),
            float(work.loc[len(work) - 1, "iteration"]),
            len(work) - i0,
        )
    )
    return spans


def _print_phase_summary(spans: list[tuple[str, float, float, int]]) -> None:
    print("Optimization phases (by iteration):")
    for phase, start, end, n in spans:
        label = _PHASE_LABELS.get(phase, phase.upper())
        print(f"  {label:<16} iterations {int(start):>4}-{int(end):<4}  ({n} trials)")


def _annotate_phases(ax: plt.Axes, spans: list[tuple[str, float, float, int]]) -> None:
    if not spans:
        return

    y_top = 1.0
    for phase, start, end, _n in spans:
        color = _PHASE_COLORS.get(phase, "#f3f4f6")
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.55, zorder=0)
        label = _PHASE_LABELS.get(phase, phase.upper())
        mid = 0.5 * (start + end)
        ax.text(
            mid,
            y_top,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="600",
            color="#374151",
            clip_on=False,
        )


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return slug or "metric"


def _plot_metric_on_ax(
    ax: plt.Axes,
    it: pd.Series,
    y: pd.Series,
    *,
    spans: list[tuple[str, float, float, int]],
    show_trials: bool,
    rolling_window: int,
    show_rolling_median: bool,
    show_legend: bool,
    title: str | None = None,
) -> None:
    run_mean = _running_mean(y)
    roll_med = _rolling_median(y, rolling_window) if show_rolling_median else None

    _annotate_phases(ax, spans)

    if show_trials:
        ax.plot(
            it,
            y,
            linestyle="none",
            marker="o",
            markersize=4 if title else 3,
            color="#94a3b8",
            alpha=0.75 if title else 0.65,
            label="Trial value",
            zorder=2,
        )

    ax.plot(
        it,
        run_mean,
        linewidth=2.2 if title else 2.0,
        color="#1d4ed8",
        label="Running mean",
        zorder=3,
    )

    if roll_med is not None:
        ax.plot(
            it,
            roll_med,
            linewidth=2.0 if title else 1.8,
            color="#c2410c",
            label=f"Rolling median (w={rolling_window})",
            zorder=4,
        )

    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, zorder=1)

    fin = y[np.isfinite(y)]
    if fin.size:
        lo, hi = float(fin.min()), float(fin.max())
        pad = max((hi - lo) * 0.12, 0.02)
        ax.set_ylim(lo - pad, hi + pad)

    if len(it):
        ax.set_xlim(float(it.min()) - 0.5, float(it.max()) + 0.5)

    if show_legend:
        ax.legend(loc="lower right", frameon=False, fontsize=8 if not title else 9)


def plot_running_mean_metric(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    spans: list[tuple[str, float, float, int]],
    show_trials: bool = True,
    rolling_window: int = 15,
    show_rolling_median: bool = True,
) -> None:
    work = df.sort_values("iteration").reset_index(drop=True)
    it = work["iteration"].astype(float)
    y = pd.to_numeric(work[metric], errors="coerce")

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    _plot_metric_on_ax(
        ax,
        it,
        y,
        spans=spans,
        show_trials=show_trials,
        rolling_window=rolling_window,
        show_rolling_median=show_rolling_median,
        show_legend=True,
        title=f"{_metric_display_name(metric)} — running mean & rolling median",
    )
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel(_metric_display_name(metric))

    fig.subplots_adjust(top=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_running_mean_grid(
    df: pd.DataFrame,
    metrics: list[str],
    out_path: Path,
    *,
    spans: list[tuple[str, float, float, int]],
    rolling_window: int = 15,
    show_rolling_median: bool = True,
) -> None:
    n = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), dpi=160, squeeze=False)
    flat = axes.ravel()

    work = df.sort_values("iteration").reset_index(drop=True)
    it = work["iteration"].astype(float)

    for i, (ax, metric) in enumerate(zip(flat, metrics)):
        y = pd.to_numeric(work[metric], errors="coerce")
        _plot_metric_on_ax(
            ax,
            it,
            y,
            spans=spans,
            show_trials=True,
            rolling_window=rolling_window,
            show_rolling_median=show_rolling_median,
            show_legend=(i == 0),
            title=_metric_display_name(metric),
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")

    for ax in flat[len(metrics) :]:
        ax.set_visible(False)

    med_note = f", rolling median w={rolling_window}" if show_rolling_median else ""
    fig.suptitle(f"Objective metrics — running mean{med_note}", fontsize=12, fontweight="600", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def resolve_trials_csv(csv: Path | None, run_dir: Path | None) -> Path:
    if csv is not None:
        path = csv.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Trials CSV not found: {path}")
        return path
    if run_dir is None:
        raise ValueError("Pass --csv or --run-dir")
    path = run_dir.expanduser().resolve() / DEFAULT_TRIALS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Trials CSV not found: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Running-mean plots for bayes_tune_mics_lmc trials.")
    ap.add_argument("--csv", type=Path, default=None, help="Path to bayes_mics_trials.csv")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Hydra run directory (uses bayes_mics_trials.csv inside)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <csv_dir>/running_mean_plots)",
    )
    ap.add_argument(
        "--no-trial-markers",
        action="store_true",
        help="Omit faint per-trial markers; show running mean only",
    )
    ap.add_argument(
        "--rolling-window",
        type=int,
        default=15,
        help="Trailing window size for rolling median (default: 15)",
    )
    ap.add_argument(
        "--no-rolling-median",
        action="store_true",
        help="Omit rolling median overlay; show running mean only",
    )
    args = ap.parse_args(argv)

    csv_path = resolve_trials_csv(args.csv, args.run_dir)
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else csv_path.parent / "running_mean_plots"
    )

    df = pd.read_csv(csv_path)
    if "iteration" not in df.columns:
        print("CSV missing 'iteration' column.", file=sys.stderr)
        return 1

    metrics = _objective_metric_columns(df)
    if not metrics:
        print("No objective metric columns found in CSV.", file=sys.stderr)
        return 1

    spans = _phase_spans(df)
    _print_phase_summary(spans)
    print(f"\nPlotting {len(metrics)} objective metric(s) -> {out_dir}")

    show_trials = not args.no_trial_markers
    show_rolling_median = not args.no_rolling_median
    rolling_window = max(1, int(args.rolling_window))
    for metric in metrics:
        out_path = out_dir / f"running_mean_{_safe_slug(metric)}.png"
        plot_running_mean_metric(
            df,
            metric,
            out_path,
            spans=spans,
            show_trials=show_trials,
            rolling_window=rolling_window,
            show_rolling_median=show_rolling_median,
        )
        print(f"  Wrote {out_path.name}")

    grid_path = out_dir / "running_mean_all_objectives.png"
    plot_running_mean_grid(
        df,
        metrics,
        grid_path,
        spans=spans,
        rolling_window=rolling_window,
        show_rolling_median=show_rolling_median,
    )
    print(f"  Wrote {grid_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
