#!/usr/bin/env python3
"""
Sensitivity-style exploratory plots from a Bayesian / OFAT tuning run produced by
``bayes_tune_mics_lmc.py``.

Reads ``bayes_mics_trials.csv`` inside the Hydra output folder (or a path you pass explicitly),
detects hyperparameter columns vs bookkeeping metrics, then writes PNG summaries under
``<run_dir>/sensitivity_plots/`` by default. **Boxplot panels include every varied
hyperparameter** (numerics too, as discrete levels or quantile bins when many unique values).

Example::

    python scripts/plot_bayes_tune_sensitivity.py \\
        --run-dir outputs/bayes_tune_mics_lmc/2026-05-17/21-28-11

    python scripts/plot_bayes_tune_sensitivity.py \\
        --csv path/to/bayes_mics_trials.csv \\
        --out-dir ./my_plots

Plots for every saved outcome (composite score + per-metric raw & normalized columns)::

    python scripts/plot_bayes_tune_sensitivity.py \\
        --run-dir outputs/bayes_tune_mics_lmc/2026-05-17/21-28-11 \\
        --all-metrics

Optional: restrict rows with ``--phase ofat`` or ``--phase bayesian`` when exploring phases separately.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


DEFAULT_TRIALS_NAME = "bayes_mics_trials.csv"

_EXCLUDE_NAMES = frozenset(
    {
        "iteration",
        "phase",
        "objective_score_mode",
        "score",
        "best_score",
        "elapsed_sec",
        "image_paths",
        "viz_edge_paths",
        "tag_leader",
        "metrics_by_image_json",
    }
)

"""Outcome columns emitted by bayes_tune_mics_lmc (used with --all-metrics)."""
_STANDARD_METRICS_ORDER: tuple[str, ...] = (
    "score",
    "continuous_dice_mean_over_images",
    "luminance_rms_contrast_mean_over_images",
    "lab_chroma_entropy_mean_over_images",
    "objective_norm__continuous_dice",
    "objective_norm__luminance_rms_contrast",
    "objective_norm__lab_chroma_entropy",
)

_EXCLUDE_PREFIXES = (
    "rank_mean__",
    "objective_norm__",
)


def _default_trials_csv(run_dir: Path) -> Path:
    p = run_dir / DEFAULT_TRIALS_NAME
    if not p.is_file():
        raise FileNotFoundError(f"No {DEFAULT_TRIALS_NAME} under {run_dir.resolve()}")
    return p


def _infer_hyperparameter_columns(df: pd.DataFrame, y_col: str) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c == y_col or c in _EXCLUDE_NAMES:
            continue
        if any(c.startswith(pref) for pref in _EXCLUDE_PREFIXES):
            continue
        if c.endswith("_mean_over_images"):
            continue
        cols.append(c)
    return cols


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        lowered = s.astype(str).str.strip().str.lower()
        mapped = lowered.map(
            {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
        )
        if mapped.notna().mean() > 0.9:
            return mapped.astype("boolean")
    return s


def _is_numeric_hp(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
        return False
    if pd.api.types.is_numeric_dtype(s):
        return True
    coerced = pd.to_numeric(s, errors="coerce")
    valid = coerced.notna().sum()
    return valid >= max(3, int(0.85 * len(s)))


def _boxplot_group_labels(s: pd.Series, *, max_levels: int = 40) -> tuple[pd.Series, str | None]:
    """
    Build a string label per row for boxplot grouping.
    Many distinct numeric values → quantile bins; many string levels → top-(max-1) + __other__.
    Returns (labels, title_suffix or None).
    """
    s = s.copy()
    nu = int(s.nunique(dropna=True))
    if nu <= 0:
        return pd.Series([""] * len(s), index=s.index), None
    if nu <= max_levels:
        return s.astype(str), None

    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().sum() >= max(3, int(0.85 * len(s))):
        n_bins = min(max_levels, max(3, nu))
        try:
            cat = pd.qcut(sn, q=n_bins, duplicates="drop")
            return cat.astype(str), "(binned by quantiles)"
        except (ValueError, TypeError):
            pass

    vc = s.astype(str).value_counts()
    top = list(vc.nlargest(max(1, max_levels - 1)).index)
    lab = s.astype(str)
    return lab.where(lab.isin(top), "__other__"), None


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    xs = x[mask]
    ys = y[mask]
    if scipy_stats is not None:
        r, p = scipy_stats.spearmanr(xs, ys)
        return float(r), float(p)
    rx = pd.Series(xs).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(ys).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return float("nan"), float("nan")
    r = float(np.corrcoef(rx, ry)[0, 1])
    # rough two-sided p-value not computed without scipy
    return r, float("nan")


def _maybe_log_x(ax: plt.Axes, s: pd.Series) -> None:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) < 2:
        return
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin > 0 and vmax / max(vmin, 1e-30) >= 50.0:
        ax.set_xscale("log")


def plot_correlation_summary(
    df: pd.DataFrame,
    hp_cols: list[str],
    y_col: str,
    out_path: Path,
) -> None:
    y = df[y_col].to_numpy(dtype=np.float64)
    records: list[tuple[str, float]] = []
    for col in hp_cols:
        s = df[col]
        if not _is_numeric_hp(s):
            continue
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        r, _ = _spearman(x, y)
        if math.isfinite(r):
            records.append((col, abs(r)))

    records.sort(key=lambda t: t[1], reverse=True)
    if not records:
        print("[warn] No numeric hyperparameters for correlation summary.", file=sys.stderr)
        return

    names = [t[0] for t in records]
    vals = [t[1] for t in records]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(names)), 4.5))
    ax.barh(names[::-1], vals[::-1], color="steelblue")
    ax.set_xlabel("|Spearman ρ| vs " + y_col)
    ax.set_title("Hyperparameter sensitivity (rank correlation magnitude)")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_numeric_grid(
    df: pd.DataFrame,
    numeric_cols: list[str],
    y_col: str,
    out_path: Path,
    ncol: int = 3,
) -> None:
    y = df[y_col].to_numpy(dtype=np.float64)
    n = len(numeric_cols)
    if n == 0:
        return
    nrow = int(math.ceil(n / ncol))
    fig_w = 3.8 * ncol
    fig_h = 2.9 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), squeeze=False)
    for i, col in enumerate(numeric_cols):
        r, cdiv = divmod(i, ncol)
        ax = axes[r][cdiv]
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], alpha=0.55, s=18, edgecolors="none", c="tab:blue")
        rho, _ = _spearman(x, y)
        if math.isfinite(rho):
            ax.set_title(f"{col}\nρ≈{rho:+.2f}", fontsize=9)
        else:
            ax.set_title(col, fontsize=9)
        ax.set_xlabel(col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)
        _maybe_log_x(ax, df[col])
    for j in range(n, nrow * ncol):
        r, cdiv = divmod(j, ncol)
        axes[r][cdiv].set_visible(False)
    fig.suptitle(f"Sensitivity: {y_col} vs numeric hyperparameters", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_categorical_grid(
    df: pd.DataFrame,
    box_cols: list[str],
    y_col: str,
    out_path: Path,
    ncol: int = 2,
    max_levels: int = 40,
) -> None:
    y = df[y_col]
    n = len(box_cols)
    if n == 0:
        return
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.2 * nrow), squeeze=False)
    for i, col in enumerate(box_cols):
        r, cdiv = divmod(i, ncol)
        ax = axes[r][cdiv]
        grp_labels, bin_note = _boxplot_group_labels(df[col], max_levels=max_levels)
        grp = pd.DataFrame({col: grp_labels, y_col: y}).dropna(subset=[y_col])
        vc = grp[col].value_counts()
        levels = list(vc.index)

        parts = grp.groupby(col, observed=False)[y_col].apply(list)
        data = [np.asarray(parts[lvl], dtype=np.float64) for lvl in levels if lvl in parts.index]
        labels = [lvl for lvl in levels if lvl in parts.index]
        if not data:
            ax.set_visible(False)
            continue
        positions = np.arange(1, len(data) + 1)
        ax.boxplot(data, positions=positions, vert=True, showfliers=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(y_col)
        title = col if not bin_note else f"{col}\n{bin_note}"
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
    for j in range(n, nrow * ncol):
        r, cdiv = divmod(j, ncol)
        axes[r][cdiv].set_visible(False)
    fig.suptitle(f"Sensitivity: {y_col} by all hyperparameters (categorical / binned)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _metric_subdir_name(col: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in col.strip())
    return f"y_{safe}" if safe else "y_metric"


def _resolve_metric_columns(df: pd.DataFrame, *, all_metrics: bool, single_y: str) -> list[str]:
    if all_metrics:
        found = [c for c in _STANDARD_METRICS_ORDER if c in df.columns]
        if not found:
            raise SystemExit(
                "No standard metric columns in CSV (--all-metrics). "
                "Expected at least one of: " + ", ".join(_STANDARD_METRICS_ORDER)
            )
        return found
    if single_y not in df.columns:
        raise SystemExit(f"Column {single_y!r} not in CSV. Available: {list(df.columns)}")
    return [single_y]


def run_sensitivity_bundle(
    *,
    csv_path: Path,
    df: pd.DataFrame,
    y_col: str,
    out_dir: Path,
    max_panels: int,
    max_boxplot_panels: int,
    max_box_levels: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ys = pd.to_numeric(df[y_col], errors="coerce")
    df = df.loc[np.isfinite(ys)].copy()

    hp_cols = _infer_hyperparameter_columns(df, y_col)
    if not hp_cols:
        print(f"[warn] No hyperparameter columns for y={y_col!r}; skip.", file=sys.stderr)
        return

    numeric_cols: list[str] = []
    boxplot_cols: list[str] = []
    for col in hp_cols:
        s = _coerce_bool_series(df[col])
        df[col] = s
        nuniq = df[col].nunique(dropna=True)
        if nuniq <= 1:
            continue
        boxplot_cols.append(col)
        if _is_numeric_hp(df[col]):
            numeric_cols.append(col)

    meta_lines = [
        f"CSV: {csv_path}",
        f"Rows used: {len(df)}",
        f"Outcome: {y_col}",
        f"Numeric HPs — scatter/correlation ({len(numeric_cols)}): {', '.join(numeric_cols)}",
        f"All varying HPs — boxplots ({len(boxplot_cols)}): {', '.join(boxplot_cols)}",
    ]
    if scipy_stats is None:
        meta_lines.append("Note: scipy not installed; Spearman uses pandas ranks + Pearson.")

    (out_dir / "sensitivity_run_summary.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    plot_correlation_summary(df, hp_cols, y_col, out_dir / "sensitivity_correlation_summary.png")

    mpp = max(4, max_panels)
    for k, chunk_start in enumerate(range(0, len(numeric_cols), mpp)):
        chunk = numeric_cols[chunk_start : chunk_start + mpp]
        suffix = "" if k == 0 else f"_{k + 1}"
        plot_numeric_grid(df, chunk, y_col, out_dir / f"sensitivity_numeric_scatters{suffix}.png")

    mpp_box = max(4, max_boxplot_panels)
    for k, chunk_start in enumerate(range(0, len(boxplot_cols), mpp_box)):
        chunk = boxplot_cols[chunk_start : chunk_start + mpp_box]
        suffix = "" if k == 0 else f"_{k + 1}"
        plot_categorical_grid(
            df,
            chunk,
            y_col,
            out_dir / f"sensitivity_categorical_boxplots{suffix}.png",
            max_levels=max_box_levels,
        )

    print(f"Wrote {y_col} plots under {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot hyperparameter sensitivity from bayes_tune trials CSV.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path, help="Hydra output folder containing bayes_mics_trials.csv")
    g.add_argument("--csv", type=Path, help="Path to bayes_mics_trials.csv")

    ap.add_argument("--out-dir", type=Path, default=None, help="Root output directory for PNGs (default: <run>/sensitivity_plots)")
    ap.add_argument("--y-col", type=str, default="score", help="Outcome column when not using --all-metrics (default: score)")
    ap.add_argument(
        "--all-metrics",
        action="store_true",
        help="Emit separate plots for score, dice, RMS contrast, entropy, and normalized objective terms (when present)",
    )
    ap.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=("all", "ofat", "bayesian", "hill", "random"),
        help="Filter trials by phase column",
    )
    ap.add_argument("--max-panels-per-figure", type=int, default=24, help="Split numeric scatters after this many panels")
    ap.add_argument(
        "--max-boxplot-panels-per-figure",
        type=int,
        default=48,
        help="Split categorical/boxplot grid after this many panels",
    )
    ap.add_argument(
        "--max-boxplot-levels",
        type=int,
        default=40,
        help="Max distinct x-groups per panel",
    )

    args = ap.parse_args()

    if args.csv is not None:
        csv_path = args.csv.expanduser().resolve()
        run_dir = csv_path.parent
    else:
        run_dir = args.run_dir.expanduser().resolve()
        csv_path = _default_trials_csv(run_dir)

    base_out = args.out_dir.expanduser().resolve() if args.out_dir else (run_dir / "sensitivity_plots")
    base_out.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_csv(csv_path)
    metric_cols = _resolve_metric_columns(df_full, all_metrics=args.all_metrics, single_y=args.y_col)

    if args.phase != "all":
        if "phase" not in df_full.columns:
            raise SystemExit("CSV has no 'phase' column; cannot filter.")
        df_full = df_full[df_full["phase"].astype(str).str.lower() == args.phase.lower()].copy()
        if df_full.empty:
            raise SystemExit(f"No rows left after phase filter {args.phase!r}")

    use_subdirs = len(metric_cols) > 1 or bool(args.all_metrics)

    for yc in metric_cols:
        tgt = base_out / _metric_subdir_name(yc) if use_subdirs else base_out
        run_sensitivity_bundle(
            csv_path=csv_path,
            df=df_full,
            y_col=yc,
            out_dir=tgt,
            max_panels=args.max_panels_per_figure,
            max_boxplot_panels=args.max_boxplot_panels_per_figure,
            max_box_levels=args.max_boxplot_levels,
        )

    print(f"Done ({len(metric_cols)} metric outcome(s)); root={base_out}")


if __name__ == "__main__":
    main()
