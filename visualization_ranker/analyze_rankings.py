"""
Analyze visualization_rankings.json: average rank and std per method, per sampling mode.
Generates colorful plots and a wins heatmap.
Correlates benchmark metrics with pathologist rankings.
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, rankdata

RANKINGS_FILE = Path(r"C:\Users\PahnkeLab\Desktop\maldi_msi\visualization_ranker\output-8slides-chapter1\visualization_rankings.json")
#Path(__file__).resolve().parent / "visualization_rankings.json"
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_CSV = REPO_ROOT / "++benchmarking++" / "benchmark_parametric_methods.csv"
# Batch edge/quality scores; merged by viz_path/viz_filename for analysis.
EDGE_AGREEMENT_FOLDER_CSV = REPO_ROOT / "edge_agreement_folder_new.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

# High-resolution export + large, paper-ready typography (figures are meant to be
# combined into a single multi-panel figure, so text must stay legible when scaled down).
SAVE_DPI = 300

# Benchmark CSV viz_path often points at deleted outputs/ runs; PNGs live under ++benchmarking++.
VIZ_PNG_FALLBACK_DIRS = (
    REPO_ROOT / "++benchmarking++" / "visualizations4_only-8_chapter1" / "visualizations",
    REPO_ROOT / "visualizations",
    REPO_ROOT / "A_collect-all-images-for-benchmarking",
)
_VIZ_PNG_INDEX: dict[str, Path] | None = None
_BENCHMARK_DF_CACHE: dict[str, pd.DataFrame] = {}

# User-facing label: column stays edge_agreement_f1 but values come from continuous Dice when merged
EDGE_METRIC_LABEL = "SpecEdge-Dice"

# Primary figures use random pixel sampling; superpixel-only figures go in OUTPUT_DIR / SUPERPIXEL_SUBDIR
DEFAULT_SAMPLING_MODE = "random"
SUPERPIXEL_SUBDIR = "superpixel"


def superpixel_output_dir(base: Path) -> Path:
    p = base / SUPERPIXEL_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_path_for_sampling(base_dir: Path, filename: str, sampling: str) -> Path:
    """
    Random (default) outputs live in base_dir; superpixel outputs in base_dir/superpixel/.
    filename should include extension, e.g. 'metric_pathologist_correlations.png'.
    """
    if sampling == DEFAULT_SAMPLING_MODE:
        return base_dir / filename
    return superpixel_output_dir(base_dir) / filename


def metric_display_name(metric: str) -> str:
    """Human-readable labels for plots (internal / CSV column names stay snake_case)."""
    _map = {
        "edge_agreement_f1": EDGE_METRIC_LABEL,
        "edge_agreement_f_beta": "Edge agreement Fβ",
        "edge_agreement_linf_f1": "Edge agreement L∞ F1",
        "edge_agreement_linf_f_beta": "Edge agreement L∞ Fβ",
        "lab_chroma_entropy": "LAB chroma entropy",
        "luminance_rms_contrast": "Luminance RMS contrast",
        "spearman_cosine": "Spearman cosine",
        "trustworthiness": "Trustworthiness",
    }
    if metric in _map:
        return _map[metric]
    return metric.replace("_", " ").title()


# Print merge / missing-file notes at most once per process (read_benchmark_csv is called many times)
_EDGE_FOLDER_LOG = {"missing_csv": False, "merged_once": False}

# Metrics where lower value = better (for ranking: rank 1 = lowest)
LOWER_IS_BETTER_METRICS = {"mrre_false", "mrre_missing", "lcmc"}

# Key metrics to always show in dedicated scatter plot
SELECTED_METRICS_PLOT = [
    "trustworthiness",
    "edge_agreement_f1",
    "luminance_rms_contrast",
    "lab_chroma_entropy",
    "spearman_cosine",
]

# Edge F1 metrics only (exclude precision/recall)
EDGE_F1_METRICS = {
    "edge_agreement_f1",
    "edge_agreement_f_beta",
    "edge_agreement_linf_f1",
    "edge_agreement_linf_f_beta",
}

# Metrics to show in correlation bar chart
CORRELATION_PLOT_METRICS = {
    "edge_agreement_f1",
    "luminance_rms_contrast",
    "lab_chroma_entropy",
    "spearman_cosine",
    "trustworthiness",
}

# Y-axis order for metric_pathologist_correlations.png (same left/right panels).
CORRELATION_PLOT_METRICS_ORDER = tuple(
    m for m in SELECTED_METRICS_PLOT if m in CORRELATION_PLOT_METRICS
)

# Distinct, vibrant color palette (avoiding pure red/green for accessibility)
PALETTE = [
    "#2E86AB",  # steel blue
    "#A23B72",  # magenta
    "#F18F01",  # amber
    "#C73E1D",  # terracotta
    "#3B1F2B",  # dark plum
    "#95C623",  # lime
    "#6C5B7B",  # mauve
    "#E07A5F",  # coral
]


def _merge_edge_agreement_continuous_dice_from_folder(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows whose viz_path matches EDGE_AGREEMENT_FOLDER_CSV, set edge_agreement_f1 to
    edge_agreement_continuous_dice and copy visualization quality metrics from the batch run.
    """
    if df.empty or not EDGE_AGREEMENT_FOLDER_CSV.exists():
        if not EDGE_AGREEMENT_FOLDER_CSV.exists():
            print(f"Note: edge agreement folder CSV not found (skip merge): {EDGE_AGREEMENT_FOLDER_CSV}")
        return df
    if "viz_path" not in df.columns:
        return df

    edge_df = pd.read_csv(EDGE_AGREEMENT_FOLDER_CSV)
    source_columns = {
        "edge_agreement_f1": "edge_agreement_continuous_dice",
        "luminance_rms_contrast": "luminance_rms_contrast",
        "lab_chroma_entropy": "lab_chroma_entropy",
    }
    source_columns = {dst: src for dst, src in source_columns.items() if src in edge_df.columns}
    if edge_df.empty or not source_columns:
        return df

    lookups: dict[str, dict[str, float]] = {dst: {} for dst in source_columns}

    def _add_keys(path_str: str, metric_values: dict[str, float]) -> None:
        p = str(path_str).strip()
        if not p:
            return
        keys = [p.replace("\\", "/").lower(), Path(p).name.lower()]
        try:
            keys.append(str(Path(p).resolve()))
        except Exception:
            pass
        for dst, val in metric_values.items():
            for key in keys:
                lookups[dst][key] = val

    for _, row in edge_df.iterrows():
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        metric_values: dict[str, float] = {}
        for dst, src in source_columns.items():
            try:
                v = float(row.get(src))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                metric_values[dst] = v
        if not metric_values:
            continue
        vp = row.get("viz_path")
        if vp is not None and str(vp).strip():
            _add_keys(str(vp), metric_values)
        vfn = row.get("viz_filename")
        if vfn is not None and str(vfn).strip():
            _add_keys(str(vfn), metric_values)

    out = df.copy()
    n_ok_by_metric: dict[str, int] = {dst: 0 for dst in source_columns}
    new_vals_by_metric: dict[str, list[float]] = {dst: [] for dst in source_columns}

    for _, row in out.iterrows():
        vp = row.get("viz_path")
        if vp is None or (isinstance(vp, float) and np.isnan(vp)):
            for dst in source_columns:
                new_vals_by_metric[dst].append(row.get(dst, np.nan))
            continue
        vp = str(vp).strip()
        keys = [vp.replace("\\", "/").lower(), Path(vp).name.lower()]
        try:
            keys.insert(0, str(Path(vp).resolve()))
        except Exception:
            pass
        for dst in source_columns:
            value = None
            for key in keys:
                value = lookups[dst].get(key)
                if value is not None:
                    break
            if value is not None:
                n_ok_by_metric[dst] += 1
                new_vals_by_metric[dst].append(value)
            else:
                new_vals_by_metric[dst].append(row.get(dst, np.nan))

    for dst, vals in new_vals_by_metric.items():
        out[dst] = vals

    if not _EDGE_FOLDER_LOG["merged_once"]:
        merged_parts = ", ".join(f"{metric}: {n_ok}/{len(out)}" for metric, n_ok in n_ok_by_metric.items())
        print(f"Merged batch metrics into benchmark rows ({merged_parts}) from {EDGE_AGREEMENT_FOLDER_CSV.name}")
        _EDGE_FOLDER_LOG["merged_once"] = True
    return out


_COLOR_METRICS = ("luminance_rms_contrast", "lab_chroma_entropy")


def _build_viz_png_index() -> dict[str, Path]:
    """Basename (lower) -> PNG under nested collect folders (chapter-1 folder uses flat names)."""
    global _VIZ_PNG_INDEX
    if _VIZ_PNG_INDEX is not None:
        return _VIZ_PNG_INDEX

    index: dict[str, Path] = {}
    collect_root = REPO_ROOT / "A_collect-all-images-for-benchmarking"
    if collect_root.is_dir():
        for png in collect_root.rglob("*.png"):
            key = png.name.lower()
            if key not in index:
                index[key] = png
    _VIZ_PNG_INDEX = index
    return index


def _resolve_viz_png(viz_path_str: str, index: dict[str, Path]) -> Path | None:
    """Resolve a benchmark viz_path to an on-disk PNG (original path or known fallback folders)."""
    p = Path(viz_path_str)
    if p.is_file():
        return p
    base = p.name
    for root in VIZ_PNG_FALLBACK_DIRS:
        candidate = root / base
        if candidate.is_file():
            return candidate
    return index.get(base.lower())


def _fill_color_metrics_from_viz_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LAB/chroma metrics from viz PNGs when missing from benchmark CSV."""
    if df.empty:
        return df

    out = df.copy()
    for col in _COLOR_METRICS:
        if col not in out.columns:
            out[col] = np.nan

    if "viz_path" not in out.columns:
        return out

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from PIL import Image

    from scripts.debug_edge_maps import _compute_viz_contrast_color_metrics

    png_index = _build_viz_png_index()
    cache: dict[str, dict[str, float]] = {}
    filled = 0
    for idx, row in out.iterrows():
        if all(np.isfinite(pd.to_numeric(row.get(c), errors="coerce")) for c in _COLOR_METRICS):
            continue
        vp = row.get("viz_path")
        if vp is None or (isinstance(vp, float) and np.isnan(vp)):
            continue
        vp_str = str(vp).strip()
        if not vp_str:
            continue
        cache_key = Path(vp_str).name.lower()

        if cache_key not in cache:
            png_path = _resolve_viz_png(vp_str, png_index)
            if png_path is None:
                cache[cache_key] = {}
            else:
                try:
                    rgb = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8)
                    valid_mask = np.any(rgb > 0, axis=-1)
                    cache[cache_key] = _compute_viz_contrast_color_metrics(rgb, valid_mask)
                except Exception:
                    cache[cache_key] = {}

        for col in _COLOR_METRICS:
            val = cache[cache_key].get(col)
            if val is None or not np.isfinite(val):
                continue
            cur = pd.to_numeric(out.at[idx, col], errors="coerce")
            if not np.isfinite(cur):
                out.at[idx, col] = float(val)
                filled += 1

    if filled:
        print(f"Computed color metrics from viz PNGs ({filled} cell values filled)")
    elif not any(
        np.isfinite(pd.to_numeric(out[c], errors="coerce")).any() for c in _COLOR_METRICS
    ):
        print("Warning: luminance_rms_contrast / lab_chroma_entropy missing (no viz PNG match)")
    return out


def read_benchmark_csv(benchmark_path: Path | None = None) -> pd.DataFrame:
    """Load benchmark CSV and overlay continuous Dice from EDGE_AGREEMENT_FOLDER_CSV when viz_path matches."""
    path = benchmark_path or BENCHMARK_CSV
    cache_key = str(path.resolve())
    if cache_key in _BENCHMARK_DF_CACHE:
        return _BENCHMARK_DF_CACHE[cache_key].copy()

    df = pd.read_csv(path)
    df = _merge_edge_agreement_continuous_dice_from_folder(df)
    df = _fill_color_metrics_from_viz_paths(df)
    _BENCHMARK_DF_CACHE[cache_key] = df.copy()
    return df.copy()


def get_method_from_filename(filename: str) -> str | None:
    """Extract method name from filename. E.g. __superpixel__parametric_mics_lmc__mics2.png -> parametric_mics_lmc__mics2"""
    for sampling in ("superpixel", "random"):
        pattern = f"__{sampling}__(.+)\\.png"
        m = re.search(pattern, filename)
        if m:
            return m.group(1)
    return None


def load_rankings(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze(path: Path | None = None) -> dict[str, dict[str, dict]]:
    """
    Analyze rankings by sampling mode and method.
    Returns: { sampling: { method: {"mean": float, "std": float, "n": int, "ranks": list} } }
    """
    path = path or RANKINGS_FILE
    data = load_rankings(path)

    # sampling -> method -> list of ranks
    by_sampling_method: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for key, entries in data.items():
        if not isinstance(entries, list):
            continue
        # Key format: "superpixel__..." or "random__..."
        parts = key.split("__", 1)
        if len(parts) < 2:
            continue
        sampling = parts[0]

        for item in entries:
            if not isinstance(item, dict) or "filename" not in item or "rank" not in item:
                continue
            method = get_method_from_filename(item["filename"])
            if method is None:
                continue
            by_sampling_method[sampling][method].append(item["rank"])

    # Compute stats per sampling, per method
    result: dict[str, dict[str, dict]] = {}
    for sampling in sorted(by_sampling_method.keys()):
        result[sampling] = {}
        for method, ranks in by_sampling_method[sampling].items():
            n = len(ranks)
            mean_rank = statistics.mean(ranks)
            std_rank = statistics.stdev(ranks) if n > 1 else 0.0
            result[sampling][method] = {
                "mean": mean_rank,
                "std": std_rank,
                "n": n,
                "ranks": ranks,
            }
    return result


def compute_wins_matrix(path: Path | None = None) -> dict[str, np.ndarray]:
    """
    For each sampling mode, compute wins[i][j] = number of times method i beat method j.
    Returns: { sampling: (methods, matrix) } where matrix[i,j] = wins of method i over method j.
    """
    path = path or RANKINGS_FILE
    data = load_rankings(path)

    # sampling -> method -> method -> count of wins
    wins_raw: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for key, entries in data.items():
        if not isinstance(entries, list):
            continue
        parts = key.split("__", 1)
        if len(parts) < 2:
            continue
        sampling = parts[0]

        # Build method -> rank for this session
        method_ranks: dict[str, int] = {}
        for item in entries:
            if not isinstance(item, dict) or "filename" not in item or "rank" not in item:
                continue
            method = get_method_from_filename(item["filename"])
            if method is None:
                continue
            method_ranks[method] = item["rank"]

        # For each pair: if rank_a < rank_b, a beats b
        methods = list(method_ranks.keys())
        for i, a in enumerate(methods):
            for b in methods[i + 1 :]:
                ra, rb = method_ranks[a], method_ranks[b]
                if ra < rb:
                    wins_raw[sampling][a][b] += 1
                elif rb < ra:
                    wins_raw[sampling][b][a] += 1

    # Collect all methods per sampling (from raw data) and build matrices
    all_methods_per_sampling: dict[str, set[str]] = defaultdict(set)
    for key, entries in data.items():
        if not isinstance(entries, list):
            continue
        parts = key.split("__", 1)
        if len(parts) < 2:
            continue
        sampling = parts[0]
        for item in entries:
            if isinstance(item, dict) and "filename" in item:
                m = get_method_from_filename(item["filename"])
                if m:
                    all_methods_per_sampling[sampling].add(m)

    result: dict[str, tuple[list[str], np.ndarray]] = {}
    for sampling in sorted(wins_raw.keys()):
        all_methods = sorted(all_methods_per_sampling.get(sampling, set()))
        n = len(all_methods)
        method_to_idx = {m: i for i, m in enumerate(all_methods)}
        matrix = np.zeros((n, n))
        for winner, losers in wins_raw[sampling].items():
            i = method_to_idx[winner]
            for loser, count in losers.items():
                j = method_to_idx[loser]
                matrix[i, j] = count
        result[sampling] = (all_methods, matrix)
    return result


def print_report(stats: dict[str, dict[str, dict]]) -> None:
    """Print formatted report per sampling mode."""
    for sampling in sorted(stats.keys()):
        methods = stats[sampling]
        print(f"\n{'='*60}")
        print(f"  SAMPLING: {sampling.upper()}")
        print(f"{'='*60}")

        # Sort by mean rank (best first)
        sorted_methods = sorted(methods.items(), key=lambda x: x[1]["mean"])

        print(f"{'Method':<40} {'Mean':>8} {'Std':>8} {'N':>6}")
        print("-" * 64)
        for method, s in sorted_methods:
            print(f"{method:<40} {s['mean']:>8.2f} {s['std']:>8.2f} {s['n']:>6}")

        # Rank comparison summary
        print(f"\n  Rank order (best to worst):")
        for i, (method, s) in enumerate(sorted_methods, 1):
            print(f"    {i}. {method} (mean={s['mean']:.2f} ± {s['std']:.2f})")


def _shorten_method(name: str) -> str:
    """Shorten method names for axis labels."""
    short = {
        "parametric_mics_lmc__mics2": "pMiCS",
        "parametric_mics_lmc__mics1": "pMiCS-mics1",
        "parametric_mics_lmc__mics3": "pMiCS-mics3",
        "parametric_umap": "pUMAP",
        "nmf_3d": "NMF",
        "pca": "PCA",
        "top3": "TOP3",
    }
    return short.get(name, name.replace("parametric_mics_lmc__", "pMiCS-"))


def _metrics_figure_axes(
    n_panels: int,
    *,
    nrows: int = 2,
    ncols: int = 3,
) -> tuple[plt.Figure, list]:
    """Grid of metric subplots (default 2×3) instead of one wide row."""
    n_panels = max(1, int(n_panels))
    ncols = max(1, int(ncols))
    nrows = max(1, int(nrows))
    while nrows * ncols < n_panels:
        ncols += 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 5.6 * nrows),
        squeeze=False,
    )
    flat = axes.ravel().tolist()
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat[:n_panels]


def _set_method_xaxis(ax, x: np.ndarray, methods: list[str], *, xlabel: bool = False) -> None:
    """Method tick labels on every panel (sharex would hide labels on upper rows)."""
    ax.set_xticks(x)
    ax.set_xticklabels([_shorten_method(m) for m in methods], rotation=45, ha="right")
    ax.tick_params(axis="x", labelbottom=True)
    if xlabel:
        ax.set_xlabel("Method")


def plot_mean_ranks(stats: dict[str, dict[str, dict]], out_path: Path) -> None:
    """Bar chart of mean rank ± std per method: random and superpixel side by side."""
    s_keys = sorted(stats.keys())
    n = len(s_keys)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6.5), squeeze=False)
    axes = axes[0]

    for idx, sampling in enumerate(s_keys):
        ax = axes[idx]
        methods_data = stats[sampling]
        sorted_methods = sorted(methods_data.items(), key=lambda x: x[1]["mean"])
        methods = [m[0] for m in sorted_methods]
        means = [m[1]["mean"] for m in sorted_methods]
        stds = [m[1]["std"] for m in sorted_methods]
        labels = [_shorten_method(m) for m in methods]

        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, capsize=4, color=PALETTE[: len(methods)], edgecolor="white", linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean rank (lower = better)", fontsize=20)
        ax.set_title(f"{sampling.capitalize()} sampling", fontsize=23, fontweight="600")
        ax.set_ylim(0, max(means) + max(stds) + 0.5 if stds else 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved mean ranks plot to {out_path}")


def plot_wins_heatmap(wins_data: dict[str, tuple[list[str], np.ndarray]], out_path: Path) -> None:
    """Heatmap: cell[i,j] = times method i beat method j. Random and superpixel side by side."""
    s_keys = sorted(wins_data.keys())
    n = len(s_keys)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7), squeeze=False)
    axes = axes[0]

    for idx, sampling in enumerate(s_keys):
        ax = axes[idx]
        methods, matrix = wins_data[sampling]
        labels = [_shorten_method(m) for m in methods]

        vmax = matrix.max() if matrix.size > 0 else 1
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(vmax, 1))

        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Method beaten (column)")
        ax.set_ylabel("Method winning (row)")
        ax.set_title(f"{sampling.capitalize()} sampling: head-to-head wins", fontsize=23, fontweight="600")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(methods)):
            for j in range(len(methods)):
                val = int(matrix[i, j])
                color = "white" if val > vmax * 0.5 else "black"
                ax.text(j, i, str(val) if val > 0 else "—", ha="center", va="center", color=color, fontsize=16)

        cbar = plt.colorbar(im, ax=ax, label="Number of wins")
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Number of wins", fontsize=18)

    plt.tight_layout()
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved wins heatmap to {out_path}")


def plot_combined_summary(
    stats: dict[str, dict[str, dict]],
    wins_data: dict[str, tuple[list[str], np.ndarray]],
    out_path: Path,
) -> None:
    """Bar + heatmap per sampling: random and superpixel side by side (2 columns × 2 rows)."""
    s_modes = sorted(stats.keys())
    n = max(1, len(s_modes))
    fig = plt.figure(figsize=(9 * n, 13))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.1], width_ratios=[1] * n, hspace=0.45, wspace=0.28)

    for col, sampling in enumerate(s_modes):
        if sampling not in stats or sampling not in wins_data:
            continue
        methods_data = stats[sampling]
        sorted_methods = sorted(methods_data.items(), key=lambda x: x[1]["mean"])
        methods = [m[0] for m in sorted_methods]
        means = [m[1]["mean"] for m in sorted_methods]
        stds = [m[1]["std"] for m in sorted_methods]
        labels = [_shorten_method(m) for m in methods]

        ax1 = fig.add_subplot(gs[0, col])
        x = np.arange(len(methods))
        ax1.bar(x, means, yerr=stds, capsize=4, color=PALETTE[: len(methods)], edgecolor="white", linewidth=1.2)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("Mean rank (lower = better)")
        ax1.set_title(f"{sampling.capitalize()}: mean rank per method")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        ax2 = fig.add_subplot(gs[1, col])
        win_methods, matrix = wins_data[sampling]
        method_to_idx = {m: i for i, m in enumerate(win_methods)}
        order_ix = [method_to_idx[m] for m in methods if m in method_to_idx]
        matrix_ordered = matrix[np.ix_(order_ix, order_ix)] if order_ix else matrix
        vmax = matrix_ordered.max() if matrix_ordered.size > 0 else 1
        im = ax2.imshow(matrix_ordered, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(vmax, 1))
        n = len(methods)
        ax2.set_xticks(np.arange(n))
        ax2.set_yticks(np.arange(n))
        ax2.set_xticklabels(labels)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel("Beaten")
        ax2.set_ylabel("Winner")
        ax2.set_title(f"{sampling.capitalize()}: head-to-head wins")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(n):
            for j in range(n):
                val = int(matrix_ordered[i, j]) if i < matrix_ordered.shape[0] and j < matrix_ordered.shape[1] else 0
                color = "white" if val > vmax * 0.5 else "black"
                ax2.text(j, i, str(val) if val > 0 else "—", ha="center", va="center", color=color, fontsize=16)
        cbar = plt.colorbar(im, ax=ax2, label="Wins")
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Wins", fontsize=18)

    plt.suptitle("Visualization ranking analysis (random vs superpixel)", fontsize=26, fontweight="700", y=0.995)
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved combined summary to {out_path}")


def build_rankings_lookup(rankings_data: dict) -> dict[str, dict[str, int]]:
    """
    Build key -> {method: pathologist_rank} for matching with CSV rows.
    Key format: {sampling}__{file_id}__r{repeat_idx}_s{seed}
    """
    lookup: dict[str, dict[str, int]] = {}
    for key, entries in rankings_data.items():
        if not isinstance(entries, list):
            continue
        method_ranks: dict[str, int] = {}
        for item in entries:
            if not isinstance(item, dict) or "filename" not in item or "rank" not in item:
                continue
            method = get_method_from_filename(item["filename"])
            if method is None:
                continue
            method_ranks[method] = item["rank"]
        if method_ranks:
            lookup[key] = method_ranks
    return lookup


def load_merged_benchmark_data(
    rankings_path: Path | None = None,
    benchmark_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str]] | None:
    """
    Load benchmark CSV and merge with pathologist rankings.
    Returns (df with pathologist_rank, list of metric column names) or None.
    """
    rankings_path = rankings_path or RANKINGS_FILE
    benchmark_path = benchmark_path or BENCHMARK_CSV

    if not rankings_path.exists() or not benchmark_path.exists():
        return None

    with open(rankings_path, encoding="utf-8") as f:
        rankings_data = json.load(f)
    lookup = build_rankings_lookup(rankings_data)

    df = read_benchmark_csv(benchmark_path)
    if df.empty:
        return None

    def get_pathologist_rank(row) -> float | None:
        key = f"{row['sampling_mode']}__{row['file_id']}__r{int(row['repeat_idx'])}_s{int(row['seed'])}"
        if key not in lookup:
            return None
        method = row["method"]
        if method not in lookup[key]:
            return None
        return float(lookup[key][method])

    df["pathologist_rank"] = df.apply(get_pathologist_rank, axis=1)

    exclude = {
        "path", "file_id", "status", "viz_path", "edge_maps_dir",
        "sampling_mode", "method", "mics_group", "cv_mode",
        "pathologist_rank",
    }
    metric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Ensure edge F1 metrics are included if present
    for m in EDGE_F1_METRICS:
        if m in df.columns and m not in metric_cols and pd.api.types.is_numeric_dtype(df[m]):
            metric_cols.append(m)
    return df, metric_cols


def compute_metric_correlations(
    rankings_path: Path | None = None,
    benchmark_path: Path | None = None,
) -> pd.DataFrame | None:
    """
    Match CSV rows with pathologist rankings by (file_id, sampling_mode, repeat_idx, seed).
    Compute Pearson correlation of each numeric metric with pathologist rank.
    Returns DataFrame with columns: metric, spearman_r, spearman_p, n_matched, sampling_mode.
    """
    merged = load_merged_benchmark_data(rankings_path, benchmark_path)
    if merged is None:
        if not (rankings_path or RANKINGS_FILE).exists():
            print(f"Rankings file not found: {rankings_path or RANKINGS_FILE}")
        elif not (benchmark_path or BENCHMARK_CSV).exists():
            print(f"Benchmark CSV not found: {benchmark_path or BENCHMARK_CSV}")
        return None

    df, metric_cols = merged
    matched = df["pathologist_rank"].notna()
    n_matched = matched.sum()
    if n_matched < 10:
        print("Too few matches for correlation analysis")
        return None

    # Add metric ranks within each comparison group for within-group correlation
    group_cols = ["file_id", "sampling_mode", "repeat_idx", "seed"]
    df_ranked = _metric_rank_in_groups(df.loc[matched], metric_cols, group_cols)

    results = []
    for sampling in df.loc[matched, "sampling_mode"].unique():
        mask = matched & (df["sampling_mode"] == sampling)
        sub = df.loc[mask]
        sub_ranked = df_ranked[df_ranked["sampling_mode"] == sampling]
        path_rank = sub["pathologist_rank"].values
        for col in metric_cols:
            vals = sub[col].values
            valid = np.isfinite(vals) & np.isfinite(path_rank)
            min_valid = 3 if col in EDGE_F1_METRICS else 5
            if valid.sum() < min_valid:
                continue
            # Pooled correlation: metric value vs pathologist rank
            # Raw: positive r = when metric↑, pathologist_rank↑ (both move together)
            # For "higher is better": agreement = metric↑ & pathologist_rank↓ → we want negative raw r
            # For "lower is better": agreement = metric↑ (bad) & pathologist_rank↑ (bad) → we want positive raw r
            # Normalize: store r such that positive = agreement with pathologist preference
            r_pooled, p_pooled = pearsonr(vals[valid], path_rank[valid])
            higher_better = col not in LOWER_IS_BETTER_METRICS
            if higher_better:
                r_pooled = -r_pooled  # flip so positive = agreement

            # For edge F1: use within-group (metric_rank vs pathologist_rank)
            # Both use 1=best, so positive r = agreement (metric rank 1 ↔ pathologist rank 1)
            if col in EDGE_F1_METRICS:
                rank_col = f"{col}_rank"
                if rank_col in sub_ranked.columns:
                    mrank = sub_ranked[rank_col].values
                    valid_wg = np.isfinite(mrank) & np.isfinite(path_rank)
                    if valid_wg.sum() >= min_valid:
                        r_wg, p_wg = pearsonr(mrank[valid_wg], path_rank[valid_wg])
                        # Within-group: positive r = metric rank agrees with pathologist (both 1=best)
                        results.append({
                            "sampling_mode": sampling,
                            "metric": col,
                            "spearman_r": r_wg,
                            "spearman_p": p_wg,
                            "n": int(valid_wg.sum()),
                        })
                        continue  # Use within-group for edge F1

            results.append({
                "sampling_mode": sampling,
                "metric": col,
                "spearman_r": r_pooled,
                "spearman_p": p_pooled,
                "n": int(valid.sum()),
            })

    return pd.DataFrame(results)


def print_correlation_report(corr_df: pd.DataFrame) -> None:
    """Print top metrics by |correlation| with pathologist rank."""
    pass


def _metric_rank_in_groups(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add metric_rank columns: within each group, rank 1=best. Higher-is-better by default."""
    if group_cols is None:
        group_cols = ["file_id", "sampling_mode", "repeat_idx", "seed"]
    df = df.copy()
    for col in metric_cols:
        higher_better = col not in LOWER_IS_BETTER_METRICS

        def rank_transform(x: pd.Series) -> pd.Series:
            valid = np.isfinite(x.values)
            out = np.full(len(x), np.nan, dtype=float)
            if valid.sum() < 2:
                return pd.Series(out, index=x.index)
            vals = x.values[valid]
            if higher_better:
                out[valid] = rankdata(-vals)
            else:
                out[valid] = rankdata(vals)
            return pd.Series(out, index=x.index)

        df[f"{col}_rank"] = df.groupby(list(group_cols))[col].transform(rank_transform)
    return df


def print_edge_f1_rank_pairs(
    rankings_path: Path | None = None,
    benchmark_path: Path | None = None,
    metric: str = "edge_agreement_f1",
) -> None:
    """Save (metric_rank, pathologist_rank) pairs for edge F1 metrics to CSV."""
    merged = load_merged_benchmark_data(rankings_path, benchmark_path)
    if merged is None:
        return
    df, metric_cols = merged
    matched = df["pathologist_rank"].notna()
    df_matched = df.loc[matched]
    if metric not in metric_cols or len(df_matched) < 3:
        return
    df_ranked = _metric_rank_in_groups(df_matched, [metric])
    rank_col = f"{metric}_rank"
    if rank_col not in df_ranked.columns:
        return

    out_dir = OUTPUT_DIR if OUTPUT_DIR else Path(__file__).resolve().parent
    out_csv = out_dir / f"edge_f1_rank_pairs_{metric}.csv"
    df_out = df_ranked[["file_id", "method", "repeat_idx", "seed", "sampling_mode", rank_col, "pathologist_rank"]].dropna(subset=[rank_col, "pathologist_rank"])
    df_out.to_csv(out_csv, index=False)


def _scatter_metric_rank_vs_pathologist(
    ax,
    sub: pd.DataFrame,
    metric: str,
    color: str,
    title_suffix: str = "",
) -> None:
    rank_col = f"{metric}_rank"
    if rank_col not in sub.columns:
        ax.set_visible(False)
        return
    x = sub[rank_col].values
    y = sub["pathologist_rank"].values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.scatter(x, y, alpha=0.6, s=40, c=color, edgecolors="white", linewidth=0.5)
    r, p = pearsonr(x, y)
    mlab = metric_display_name(metric)
    ax.set_xlabel(f"{mlab} rank (1=best)")
    ax.set_ylabel("Pathologist rank (1=best)")
    ax.set_title(f"{mlab}{title_suffix}\nPearson r={r:.3f}, p={p:.2e}")
    max_rank = int(sub[rank_col].max()) if np.isfinite(sub[rank_col]).any() else 5
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(0.5, 5.5)
    ax.plot([0.5, min(5.5, max_rank + 0.5)], [0.5, min(5.5, max_rank + 0.5)], "k--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.arange(1, max_rank + 1))


def plot_metric_pathologist_scatter(
    corr_df: pd.DataFrame,
    rankings_path: Path | None = None,
    benchmark_path: Path | None = None,
    out_dir: Path | None = None,
    top_n: int = 8,
) -> None:
    """Scatter: metric rank vs pathologist rank. Random (left) and superpixel (right) side by side per metric."""
    if corr_df is None or corr_df.empty:
        return

    merged = load_merged_benchmark_data(rankings_path, benchmark_path)
    if merged is None:
        return

    df, metric_cols = merged
    matched = df["pathologist_rank"].notna()
    df_matched = df.loc[matched].copy()
    if len(df_matched) < 5:
        return

    df_ranked = _metric_rank_in_groups(df_matched, metric_cols)
    out_dir = out_dir or OUTPUT_DIR

    sub_corr_r = corr_df[corr_df["sampling_mode"] == DEFAULT_SAMPLING_MODE].copy()
    if sub_corr_r.empty:
        sub_corr_r = corr_df[corr_df["sampling_mode"] == corr_df["sampling_mode"].iloc[0]].copy()
    sub_corr_r["abs_r"] = sub_corr_r["spearman_r"].abs()
    top_metrics = sub_corr_r.nlargest(top_n, "abs_r")
    n_plots = len(top_metrics)
    if n_plots < 1:
        return

    sub_random = df_ranked[df_ranked["sampling_mode"] == DEFAULT_SAMPLING_MODE]
    sub_sp = df_ranked[df_ranked["sampling_mode"] == "superpixel"]

    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 5.0 * n_plots), squeeze=False)
    for row, (_, row_m) in enumerate(top_metrics.iterrows()):
        metric = row_m["metric"]
        _scatter_metric_rank_vs_pathologist(axes[row, 0], sub_random, metric, PALETTE[0], " (random)")
        _scatter_metric_rank_vs_pathologist(axes[row, 1], sub_sp, metric, PALETTE[2], " (superpixel)")
    plt.suptitle("Metric rank vs pathologist rank (random | superpixel)", fontsize=24, fontweight="700", y=1.005)
    plt.tight_layout()
    p = out_dir / "metric_pathologist_scatter.png"
    plt.savefig(p, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved scatter plots to {p}")

    edge_metrics = (
        corr_df.loc[
            (corr_df["sampling_mode"] == DEFAULT_SAMPLING_MODE) & (corr_df["metric"].isin(EDGE_F1_METRICS)),
            "metric",
        ]
        .drop_duplicates()
        .tolist()
    )
    if not edge_metrics:
        edge_metrics = corr_df.loc[corr_df["metric"].isin(EDGE_F1_METRICS), "metric"].drop_duplicates().tolist()
    n_edge = len(edge_metrics)
    if n_edge > 0:
        fig, axes = plt.subplots(n_edge, 2, figsize=(12, 5.0 * n_edge), squeeze=False)
        for row, metric in enumerate(edge_metrics):
            _scatter_metric_rank_vs_pathologist(axes[row, 0], sub_random, metric, PALETTE[1], " (random)")
            _scatter_metric_rank_vs_pathologist(axes[row, 1], sub_sp, metric, PALETTE[1], " (superpixel)")
        plt.suptitle(f"{EDGE_METRIC_LABEL} vs pathologist rank (random | superpixel)", fontsize=24, fontweight="700", y=1.005)
        plt.tight_layout()
        p = out_dir / "metric_pathologist_scatter_specedge.png"
        plt.savefig(p, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved edge agreement scatter plots to {p}")

    plot_selected_metrics_scatter(df_ranked, out_dir, SELECTED_METRICS_PLOT)


def plot_selected_metrics_scatter(
    df_ranked: pd.DataFrame,
    out_dir: Path,
    metrics: list[str],
) -> None:
    """Selected metrics: two rows (random | superpixel) × columns = one metric each."""
    sub_random = df_ranked[df_ranked["sampling_mode"] == DEFAULT_SAMPLING_MODE]
    sub_sp = df_ranked[df_ranked["sampling_mode"] == "superpixel"]
    available = [m for m in metrics if f"{m}_rank" in sub_random.columns or f"{m}_rank" in sub_sp.columns]
    available = [m for m in available if f"{m}_rank" in sub_random.columns and f"{m}_rank" in sub_sp.columns]
    if not available:
        return
    n = len(available)
    fig, axes = plt.subplots(2, n, figsize=(6.5 * n, 11), squeeze=False)
    for col, metric in enumerate(available):
        for row, (sub, lab) in enumerate([(sub_random, "Random"), (sub_sp, "Superpixel")]):
            ax = axes[row, col]
            rank_col = f"{metric}_rank"
            if rank_col not in sub.columns:
                ax.set_visible(False)
                continue
            x = sub[rank_col].values
            y = sub["pathologist_rank"].values
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            ax.scatter(x, y, alpha=0.6, s=40, c=PALETTE[col % len(PALETTE)], edgecolors="white", linewidth=0.5)
            r, p = pearsonr(x, y) if len(x) >= 2 else (float("nan"), float("nan"))
            mlab = metric_display_name(metric)
            ax.set_xlabel(f"{mlab} rank (1=best)")
            ax.set_ylabel("Pathologist rank (1=best)")
            title = "Spearman correlation" if metric == "spearman_cosine" else mlab
            ax.set_title(f"{title} ({lab})\nPearson r={r:.3f}, p={p:.2e}")
            max_rank = int(sub[rank_col].max()) if np.isfinite(sub[rank_col]).any() else 5
            ax.set_xlim(0.5, max_rank + 0.5)
            ax.set_ylim(0.5, 5.5)
            ax.plot([0.5, min(5.5, max_rank + 0.5)], [0.5, min(5.5, max_rank + 0.5)], "k--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks(np.arange(1, max_rank + 1))
    plt.suptitle(
        ", ".join(metric_display_name(m) for m in available) + " (random | superpixel)",
        fontsize=24,
        fontweight="700",
        y=1.005,
    )
    plt.tight_layout()
    p = out_dir / "metric_pathologist_scatter_selected.png"
    plt.savefig(p, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved selected metrics scatter to {p}")


def plot_metric_correlations(corr_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of metric correlations: random vs superpixel side by side."""
    if corr_df is None or corr_df.empty:
        return
    order = list(CORRELATION_PLOT_METRICS_ORDER)
    if not order:
        order = sorted(
            {m for m in CORRELATION_PLOT_METRICS if m in set(corr_df["metric"].unique())},
            key=lambda x: str(x),
        )
    s_modes = sorted(corr_df["sampling_mode"].unique())
    n = len(s_modes)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 6.5))
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes)

    for ax, sampling in zip(axes, s_modes):
        sub = corr_df[
            (corr_df["sampling_mode"] == sampling)
            & (corr_df["metric"].isin(CORRELATION_PLOT_METRICS))
        ].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        r_list: list[float] = []
        for m in order:
            row = sub.loc[sub["metric"] == m, "spearman_r"]
            if row.empty or not np.isfinite(float(row.iloc[0])):
                r_list.append(float("nan"))
            else:
                r_list.append(float(row.iloc[0]))

        y = np.arange(len(order))
        widths = [0.0 if not np.isfinite(r) else r for r in r_list]
        colors: list[str] = []
        for r in r_list:
            if not np.isfinite(r):
                colors.append("#cccccc")
            elif r < 0:
                colors.append(PALETTE[0])
            else:
                colors.append(PALETTE[1])

        ax.barh(y, widths, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([metric_display_name(m) for m in order], fontsize=17)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel("Pearson r (positive = agrees with pathologist)")
        ax.set_title(f"{sampling.capitalize()} sampling")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

    plt.suptitle("Metric correlation with pathologist ranking", fontsize=24, fontweight="700", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved metric correlation plot to {out_path}")


def save_csv(stats: dict[str, dict[str, dict]], out_path: Path) -> None:
    """Save summary to CSV."""
    rows = []
    for sampling in sorted(stats.keys()):
        for method, s in stats[sampling].items():
            rows.append({
                "sampling": sampling,
                "method": method,
                "mean_rank": round(s["mean"], 4),
                "std_rank": round(s["std"], 4),
                "n": s["n"],
            })
    # Sort by sampling, then mean_rank
    rows.sort(key=lambda r: (r["sampling"], r["mean_rank"]))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("sampling,method,mean_rank,std_rank,n\n")
        for r in rows:
            f.write(f"{r['sampling']},{r['method']},{r['mean_rank']},{r['std_rank']},{r['n']}\n")
    print(f"\nSaved CSV to {out_path}")


def plot_trustworthiness_vs_f1(
    benchmark_path: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """Scatter: trustworthiness (x) vs SpecEdge-Dice (y). Random (○) and superpixel (□) in one figure."""
    benchmark_path = benchmark_path or BENCHMARK_CSV
    out_dir = out_dir or OUTPUT_DIR
    if not benchmark_path.exists():
        return

    df = read_benchmark_csv(benchmark_path)
    if df.empty or "trustworthiness" not in df.columns or "edge_agreement_f1" not in df.columns:
        return

    sub = df[["trustworthiness", "edge_agreement_f1", "method", "sampling_mode"]].dropna()
    sub = sub[~sub["method"].isin(["parametric_mics_lmc__mics1", "parametric_mics_lmc__mics3"])]
    if sub.empty:
        return

    from matplotlib.lines import Line2D

    methods = sorted(sub["method"].unique(), key=lambda m: (_shorten_method(m), m))
    method_to_idx = {m: i for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for method in methods:
        for sampling, marker in [("random", "o"), ("superpixel", "s")]:
            mask = (sub["method"] == method) & (sub["sampling_mode"] == sampling)
            if not mask.any():
                continue
            x = sub.loc[mask, "trustworthiness"].values
            y = sub.loc[mask, "edge_agreement_f1"].values
            color = PALETTE[method_to_idx[method] % len(PALETTE)]
            ax.scatter(x, y, c=color, marker=marker, s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Trustworthiness")
    ax.set_ylabel(EDGE_METRIC_LABEL)
    ax.set_title(f"Trustworthiness vs {EDGE_METRIC_LABEL} by method (○ random, □ superpixel)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=13, label="random"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=13, label="superpixel"),
    ]
    for i, m in enumerate(methods):
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[i % len(PALETTE)], markersize=13, label=_shorten_method(m))
        )
    ax.legend(handles=legend_elements, ncol=2, loc="upper left", fontsize=15)
    plt.tight_layout()
    p = out_dir / "trustworthiness_vs_specedge_dice.png"
    plt.savefig(p, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved trustworthiness vs {EDGE_METRIC_LABEL} plot to {p}")


def plot_trustworthiness_vs_f1_averaged(
    benchmark_path: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """Scatter: mean trustworthiness vs mean SpecEdge-Dice per method (○ random, □ superpixel)."""
    benchmark_path = benchmark_path or BENCHMARK_CSV
    out_dir = out_dir or OUTPUT_DIR
    if not benchmark_path.exists():
        return

    df = read_benchmark_csv(benchmark_path)
    if df.empty or "trustworthiness" not in df.columns or "edge_agreement_f1" not in df.columns:
        return

    sub = df[["trustworthiness", "edge_agreement_f1", "method", "sampling_mode"]].dropna()
    sub = sub[~sub["method"].isin(["parametric_mics_lmc__mics1", "parametric_mics_lmc__mics3"])]
    if sub.empty:
        return

    agg = sub.groupby(["method", "sampling_mode"]).agg(
        trustworthiness=("trustworthiness", "mean"),
        edge_agreement_f1=("edge_agreement_f1", "mean"),
    ).reset_index()

    from matplotlib.lines import Line2D

    methods = sorted(agg["method"].unique(), key=lambda m: (_shorten_method(m), m))
    method_to_idx = {m: i for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for method in methods:
        for sampling, marker in [("random", "o"), ("superpixel", "s")]:
            row = agg[(agg["method"] == method) & (agg["sampling_mode"] == sampling)]
            if row.empty:
                continue
            x = row["trustworthiness"].values[0]
            y = row["edge_agreement_f1"].values[0]
            color = PALETTE[method_to_idx[method] % len(PALETTE)]
            ax.scatter(x, y, c=color, marker=marker, s=100, alpha=0.8, edgecolors="white", linewidth=1)

    ax.set_xlabel("Trustworthiness (mean)")
    ax.set_ylabel(f"{EDGE_METRIC_LABEL} (mean)")
    ax.set_title(f"Trustworthiness vs {EDGE_METRIC_LABEL} by method, averaged (○ random, □ superpixel)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=13, label="random"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=13, label="superpixel"),
    ]
    for i, m in enumerate(methods):
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[i % len(PALETTE)], markersize=13, label=_shorten_method(m))
        )
    ax.legend(handles=legend_elements, ncol=2, loc="upper left", fontsize=15)
    plt.tight_layout()
    p = out_dir / "trustworthiness_vs_specedge_dice_averaged.png"
    plt.savefig(p, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved trustworthiness vs {EDGE_METRIC_LABEL} (averaged) plot to {p}")


def plot_f1_sampling_comparison(
    benchmark_path: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """Bar chart: mean SpecEdge-Dice per method, random vs superpixel side by side."""
    benchmark_path = benchmark_path or BENCHMARK_CSV
    out_dir = out_dir or OUTPUT_DIR
    if not benchmark_path.exists():
        return

    df = read_benchmark_csv(benchmark_path)
    if df.empty or "edge_agreement_f1" not in df.columns:
        return

    agg = df.groupby(["method", "sampling_mode"])["edge_agreement_f1"].agg(["mean", "std", "count"]).reset_index()
    agg = agg[~agg["method"].isin(["parametric_mics_lmc__mics1", "parametric_mics_lmc__mics3"])]
    if agg.empty:
        return
    methods = sorted(agg["method"].unique(), key=lambda m: (_shorten_method(m), m))

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(methods))
    width = 0.35

    for i, sampling in enumerate(["random", "superpixel"]):
        vals = []
        errs = []
        for m in methods:
            row = agg[(agg["method"] == m) & (agg["sampling_mode"] == sampling)]
            if row.empty:
                vals.append(np.nan)
                errs.append(0)
            else:
                vals.append(row["mean"].values[0])
                n = row["count"].values[0]
                se = row["std"].values[0] / (n ** 0.5) if n > 1 else 0
                errs.append(se)
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=sampling, color=PALETTE[i], edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offset, vals, yerr=errs, fmt="none", color="black", capsize=2)

    ax.set_ylabel(f"{EDGE_METRIC_LABEL} (mean ± SE)")
    ax.set_xlabel("Method")
    ax.set_xticks(x)
    ax.set_xticklabels([_shorten_method(m) for m in methods], rotation=45, ha="right")
    ax.set_title(f"{EDGE_METRIC_LABEL}: random vs superpixel sampling")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = out_dir / "specedge_dice_random_vs_superpixel.png"
    plt.savefig(p, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved random vs superpixel comparison to {p}")


def plot_metrics_per_method(
    benchmark_path: Path | None = None,
    out_dir: Path | None = None,
    metrics: tuple[str, ...] = (
        "trustworthiness",
        "edge_agreement_f1",
        "luminance_rms_contrast",
        "lab_chroma_entropy",
        "spearman_cosine",
    ),
) -> None:
    """Bar charts for selected metrics — random vs superpixel paired per method."""
    benchmark_path = benchmark_path or BENCHMARK_CSV
    out_dir = out_dir or OUTPUT_DIR
    if not benchmark_path.exists():
        return

    df = read_benchmark_csv(benchmark_path)
    df = df[~df["method"].isin(["parametric_mics_lmc__mics1", "parametric_mics_lmc__mics3"])]
    if df.empty:
        return

    metric_labels = {
        "trustworthiness": "Trustworthiness",
        "edge_agreement_f1": EDGE_METRIC_LABEL,
        "luminance_rms_contrast": "Luminance RMS contrast",
        "lab_chroma_entropy": "LAB chroma entropy",
        "spearman_cosine": "Spearman cosine",
    }

    methods = sorted(df["method"].unique(), key=lambda m: (_shorten_method(m), m))
    x = np.arange(len(methods))
    width = 0.35

    plot_metrics = [
        m
        for m in metrics
        if m in df.columns and pd.to_numeric(df[m], errors="coerce").notna().any()
    ]
    if not plot_metrics:
        return
    fig, axes = _metrics_figure_axes(len(plot_metrics))
    for ax, metric in zip(axes, plot_metrics):
        for i, sampling in enumerate(["random", "superpixel"]):
            vals = []
            errs = []
            for m in methods:
                row = df.loc[(df["method"] == m) & (df["sampling_mode"] == sampling), metric]
                if row.empty:
                    vals.append(np.nan)
                    errs.append(0)
                else:
                    vals.append(float(row.mean()))
                    n = int(row.count())
                    se = float(row.std()) / (n ** 0.5) if n > 1 else 0.0
                    errs.append(se)
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=sampling, color=PALETTE[i], edgecolor="white", linewidth=0.5)
            ax.errorbar(x + offset, vals, yerr=errs, fmt="none", color="black", capsize=2)

        ax.set_ylabel(f"{metric_labels.get(metric, metric)} (mean ± SE)")
        ax.set_title(metric_labels.get(metric, metric))
        _set_method_xaxis(ax, x, methods)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Method")
    plt.tight_layout()
    p = out_dir / "metrics_per_method.png"
    plt.savefig(p, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved metrics per method to {p}")


def plot_metrics_per_method_avg_rank(
    benchmark_path: Path | None = None,
    out_dir: Path | None = None,
    metrics: tuple[str, ...] = (
        "trustworthiness",
        "edge_agreement_f1",
        "luminance_rms_contrast",
        "lab_chroma_entropy",
        "spearman_cosine",
    ),
) -> None:
    """Same layout as ``plot_metrics_per_method``, but y-axis = mean within-group rank (1=best per MSI/seed)."""
    benchmark_path = benchmark_path or BENCHMARK_CSV
    out_dir = out_dir or OUTPUT_DIR
    if not benchmark_path.exists():
        return

    df = read_benchmark_csv(benchmark_path)
    df = df[~df["method"].isin(["parametric_mics_lmc__mics1", "parametric_mics_lmc__mics3"])]
    if df.empty:
        return

    available = [
        m
        for m in metrics
        if m in df.columns and pd.to_numeric(df[m], errors="coerce").notna().any()
    ]
    if not available:
        return

    df_ranked = _metric_rank_in_groups(df, available)

    metric_labels = {
        "trustworthiness": "Trustworthiness",
        "edge_agreement_f1": EDGE_METRIC_LABEL,
        "luminance_rms_contrast": "Luminance RMS contrast",
        "lab_chroma_entropy": "LAB chroma entropy",
        "spearman_cosine": "Spearman cosine",
    }

    methods = sorted(df_ranked["method"].unique(), key=lambda m: (_shorten_method(m), m))
    x = np.arange(len(methods))
    width = 0.35

    fig, axes = _metrics_figure_axes(len(available))
    n_methods = len(methods)
    for ax, metric in zip(axes, available):
        rank_col = f"{metric}_rank"
        for i, sampling in enumerate(["random", "superpixel"]):
            vals = []
            errs = []
            sub = df_ranked[df_ranked["sampling_mode"] == sampling]
            for m in methods:
                r = sub.loc[sub["method"] == m, rank_col]
                r = r[np.isfinite(r.values)]
                if len(r) == 0:
                    vals.append(np.nan)
                    errs.append(0.0)
                else:
                    vals.append(float(np.mean(r.values)))
                    n = int(len(r))
                    errs.append(float(np.std(r.values, ddof=1)) / (n ** 0.5) if n > 1 else 0.0)
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=sampling, color=PALETTE[i], edgecolor="white", linewidth=0.5)
            ax.errorbar(x + offset, vals, yerr=errs, fmt="none", color="black", capsize=2)

        ax.set_ylabel(f"{metric_labels.get(metric, metric)} — avg rank (± SE)")
        ax.set_title(metric_labels.get(metric, metric))
        _set_method_xaxis(ax, x, methods)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0.5, n_methods + 0.5)
        ax.axhline(y=(1 + n_methods) / 2, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    axes[-1].set_xlabel("Method")
    plt.suptitle("Average rank per method", fontsize=24, fontweight="700", y=1.02)
    plt.tight_layout()
    p = out_dir / "metrics_per_method_avg_rank.png"
    plt.savefig(p, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved average-rank metrics per method to {p}")


def _setup_plot_style() -> None:
    """Apply a clean, modern, paper-ready plot style with large, legible typography."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": SAVE_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.titleweight": "600",
        "axes.labelsize": 20,
        "axes.labelweight": "500",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 17,
        "figure.titlesize": 24,
        "figure.titleweight": "700",
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def main():
    if not RANKINGS_FILE.exists():
        print(f"File not found: {RANKINGS_FILE}")
        return

    _setup_plot_style()

    stats = analyze(RANKINGS_FILE)
    print_report(stats)

    csv_path = OUTPUT_DIR / "ranking_analysis.csv"
    save_csv(stats, csv_path)

    wins_data = compute_wins_matrix(RANKINGS_FILE)
    plot_mean_ranks(stats, OUTPUT_DIR / "ranking_mean_ranks.png")
    plot_wins_heatmap(wins_data, OUTPUT_DIR / "ranking_wins_heatmap.png")
    plot_combined_summary(stats, wins_data, OUTPUT_DIR / "ranking_summary.png")

    # Correlate benchmark metrics with pathologist rankings
    corr_df = compute_metric_correlations(RANKINGS_FILE, BENCHMARK_CSV)
    if corr_df is not None:
        print_correlation_report(corr_df)
        corr_df.to_csv(OUTPUT_DIR / "metric_pathologist_correlations.csv", index=False)
        print_edge_f1_rank_pairs(RANKINGS_FILE, BENCHMARK_CSV, metric="edge_agreement_f1")
        plot_metric_correlations(corr_df, OUTPUT_DIR / "metric_pathologist_correlations.png")
        plot_metric_pathologist_scatter(
            corr_df, RANKINGS_FILE, BENCHMARK_CSV, OUTPUT_DIR, top_n=8
        )

    # SpecEdge-Dice vs trustworthiness (random = main dir; superpixel in superpixel/)
    plot_trustworthiness_vs_f1(BENCHMARK_CSV, OUTPUT_DIR)
    plot_trustworthiness_vs_f1_averaged(BENCHMARK_CSV, OUTPUT_DIR)
    plot_f1_sampling_comparison(BENCHMARK_CSV, OUTPUT_DIR)
    plot_metrics_per_method(BENCHMARK_CSV, OUTPUT_DIR)
    plot_metrics_per_method_avg_rank(BENCHMARK_CSV, OUTPUT_DIR)


if __name__ == "__main__":
    main()
