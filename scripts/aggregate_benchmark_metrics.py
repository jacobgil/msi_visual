#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Set, Tuple


GROUP_KEYS = ("method", "sampling_mode")
NON_METRIC_COLUMNS = {
    "path",
    "method",
    "sampling_mode",
    "seed",
    "repeat_idx",
    "status",
    "file_id",
    "error",
    "viz_path",
    "n_repeats",
}
INCLUDED_OUTPUT_SUBSTRINGS = (
    "sampling",
    "method",
    "n_images",
    "n_seeds",
    "spearman",
    "edge_agreement",
    "zadu",
    "faithfulness",
    "trustworthiness",
    "continuity",
    "contunuity",
    "nmf_kl",
)
EXCLUDED_INPUT_PATHS = {
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset11_kidney-nedc",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset3_ovary\0.npy",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset12_lung\5bins\0.npy",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset6_lung\0.npy",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset9_kidney2\0.npy",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset11_kidney-nedc\0.npy"
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset1_bactvesicles\0.npy",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset6_lung\0.npy",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset2_braintumor\0.npy",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset13_TMA-pancreas\0.npy",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset8_plantleaf\0.npy",
    # r"E:\MSImaging-data\_msi_visual\Extractions\Metaspace_datasets\Dataset9_kidney2\0.npy"

}
MICS_METHOD_PREFIX = "parametric_mics_lmc"


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _normalize_path_for_match(path: str) -> str:
    # Use slash/case-insensitive comparison to be robust to path formatting.
    return str(path).strip().replace("\\", "/").lower()


def _exclude_input_paths(
    rows: Sequence[Dict[str, str]],
    excluded_paths: Set[str],
) -> Tuple[List[Dict[str, str]], int]:
    excluded_norm = {_normalize_path_for_match(p) for p in excluded_paths}
    kept: List[Dict[str, str]] = []
    dropped = 0
    for row in rows:
        row_path = _normalize_path_for_match(str(row.get("path", "")))
        if row_path in excluded_norm:
            dropped += 1
            continue
        kept.append(dict(row))
    return kept, dropped


def _discover_metric_columns(rows: Sequence[Dict[str, str]]) -> List[str]:
    metric_columns: List[str] = []
    if not rows:
        return metric_columns

    for column in rows[0].keys():
        if column in NON_METRIC_COLUMNS:
            continue
        if any(_parse_float(row.get(column)) is not None for row in rows):
            metric_columns.append(column)
    return metric_columns


def _group_rows(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        method = str(row.get("method", "")).strip()
        sampling_mode = str(row.get("sampling_mode", "")).strip()
        if not method or not sampling_mode:
            continue
        grouped.setdefault((method, sampling_mode), []).append(row)
    return grouped


def _print_missing_paths_by_method(rows: Sequence[Dict[str, str]]) -> None:
    by_sampling: Dict[str, Dict[str, Set[str]]] = {}
    for row in rows:
        sampling_mode = str(row.get("sampling_mode", "")).strip()
        method = str(row.get("method", "")).strip()
        path = str(row.get("path", "")).strip()
        if not sampling_mode or not method or not path:
            continue
        by_sampling.setdefault(sampling_mode, {}).setdefault(method, set()).add(path)

    if not by_sampling:
        print("No valid method/sampling/path rows found for path coverage report.")
        return

    print("=== Missing path coverage by method ===")
    any_missing = False
    for sampling_mode in sorted(by_sampling.keys()):
        methods_to_paths = by_sampling[sampling_mode]
        all_paths: Set[str] = set()
        for paths in methods_to_paths.values():
            all_paths.update(paths)

        print(f"[sampling_mode={sampling_mode}]")
        for method in sorted(methods_to_paths.keys()):
            missing = sorted(all_paths - methods_to_paths[method])
            if not missing:
                continue
            any_missing = True
            print(f"  method={method} missing_paths={len(missing)}")
            for path in missing:
                print(f"    - {path}")

    if not any_missing:
        print("All methods cover the same set of paths (within each sampling_mode).")


def _aggregate_over_seeds(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Aggregate over seeds when multiple rows exist per (path, sampling_mode, method).
    If input is already aggregated (one row per path/method/sampling_mode), returns as-is.
    """
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        if status and status != "ok":
            continue
        path = str(row.get("path", "")).strip()
        sampling_mode = str(row.get("sampling_mode", "")).strip()
        method = str(row.get("method", "")).strip()
        if not path or not sampling_mode or not method:
            continue
        key = (path, sampling_mode, method)
        grouped.setdefault(key, []).append(dict(row))

    # If no status column or all groups have exactly 1 row, return as-is (already aggregated)
    has_status = any("status" in r for r in rows)
    if not has_status or all(len(g) == 1 for g in grouped.values()):
        return [list(g)[0] for _, g in sorted(grouped.items())]

    # Aggregate over seeds
    all_columns = set()
    for g in grouped.values():
        for r in g:
            all_columns.update(r.keys())
    metric_cols = [
        c for c in all_columns
        if c not in NON_METRIC_COLUMNS
        and not c.endswith("_mean")
        and not c.endswith("_std")
        and any(_parse_float(r.get(c)) is not None for g in grouped.values() for r in g)
    ]
    if not metric_cols:
        metric_cols = [c for c in all_columns if c.endswith("_mean") and c not in NON_METRIC_COLUMNS]

    out_rows: List[Dict[str, str]] = []
    for (path, sampling_mode, method), grows in grouped.items():
        out: Dict[str, str] = {
            "path": path,
            "sampling_mode": sampling_mode,
            "method": method,
            "n_repeats": str(len(grows)),
        }
        for col in metric_cols:
            vals = [_parse_float(r.get(col)) for r in grows]
            finite = [v for v in vals if v is not None]
            if finite:
                base = col[:-5] if col.endswith("_mean") else col
                out[f"{base}_mean"] = str(mean(finite))
                out[f"{base}_std"] = str(pstdev(finite)) if len(finite) > 1 else "0.0"
        out_rows.append(out)
    return sorted(out_rows, key=lambda r: (r["path"], r["sampling_mode"], r["method"]))


def _filter_to_paths_present_in_all_methods(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    by_sampling: Dict[str, Dict[str, Set[str]]] = {}
    for row in rows:
        sampling_mode = str(row.get("sampling_mode", "")).strip()
        method = str(row.get("method", "")).strip()
        path = str(row.get("path", "")).strip()
        if not sampling_mode or not method or not path:
            continue
        by_sampling.setdefault(sampling_mode, {}).setdefault(method, set()).add(path)

    common_paths_by_sampling: Dict[str, Set[str]] = {}
    for sampling_mode, methods_to_paths in by_sampling.items():
        method_path_sets = list(methods_to_paths.values())
        if not method_path_sets:
            common_paths_by_sampling[sampling_mode] = set()
            continue
        common = set(method_path_sets[0])
        for path_set in method_path_sets[1:]:
            common &= path_set
        common_paths_by_sampling[sampling_mode] = common

    filtered_rows: List[Dict[str, str]] = []
    dropped_rows = 0
    for row in rows:
        sampling_mode = str(row.get("sampling_mode", "")).strip()
        path = str(row.get("path", "")).strip()
        if not sampling_mode or not path:
            dropped_rows += 1
            continue
        if path in common_paths_by_sampling.get(sampling_mode, set()):
            filtered_rows.append(row)
        else:
            dropped_rows += 1
    return filtered_rows, dropped_rows


def _aggregate_over_images(
    grouped_rows: Dict[Tuple[str, str], List[Dict[str, str]]],
    metric_columns: Sequence[str],
) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []
    for (method, sampling_mode), rows in sorted(grouped_rows.items()):
        n_repeats_vals = [_parse_float(r.get("n_repeats")) for r in rows]
        n_repeats_vals = [v for v in n_repeats_vals if v is not None and v >= 1]
        n_seeds_total = int(sum(n_repeats_vals)) if n_repeats_vals else 0
        n_seeds_per_image = float(mean(n_repeats_vals)) if n_repeats_vals else 0.0

        out: Dict[str, object] = {
            "method": method,
            "sampling_mode": sampling_mode,
            "n_images": len(rows),
            "n_seeds_total": n_seeds_total,
            "n_seeds_per_image": round(n_seeds_per_image, 2),
        }
        for column in metric_columns:
            values = [_parse_float(r.get(column)) for r in rows]
            finite_values = [v for v in values if v is not None]
            if not finite_values:
                continue
            out[f"{column}_img_mean"] = float(mean(finite_values))
            out[f"{column}_img_std"] = float(pstdev(finite_values)) if len(finite_values) > 1 else 0.0
        out_rows.append(out)
    return out_rows


def _numeric_variability(values: Iterable[object], tol: float = 1e-12) -> bool:
    finite: List[float] = []
    for v in values:
        parsed = _parse_float(v)
        if parsed is not None:
            finite.append(parsed)
    if len(finite) <= 1:
        return False
    return (max(finite) - min(finite)) > tol


def _drop_non_variable_columns(
    rows: Sequence[Dict[str, object]],
    keep_columns: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[str]]:
    if not rows:
        return [], []

    columns = list(rows[0].keys())
    keep = set(keep_columns)
    kept_columns: List[str] = []
    removed_columns: List[str] = []
    for col in columns:
        if col in keep or _numeric_variability((r.get(col) for r in rows)):
            kept_columns.append(col)
        else:
            removed_columns.append(col)

    filtered_rows: List[Dict[str, object]] = []
    for row in rows:
        filtered_rows.append({k: row.get(k) for k in kept_columns})
    return filtered_rows, removed_columns


def _is_included_output_column(column: str) -> bool:
    lowered = column.lower()
    if "edge_agreement" in lowered:
        return True
    if "std" in lowered:
        return False
    return any(token in lowered for token in INCLUDED_OUTPUT_SUBSTRINGS)


def _keep_included_columns(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[str]]:
    if not rows:
        return [], []
    columns = list(rows[0].keys())
    kept_columns = [c for c in columns if _is_included_output_column(c)]
    removed_columns = [c for c in columns if c not in kept_columns]
    filtered_rows = [{k: row.get(k) for k in kept_columns} for row in rows]
    return filtered_rows, removed_columns


def _print_paths_where_umap_beats_mics(
    rows: Sequence[Dict[str, str]],
    metric_column: str,
    umap_method: str = "parametric_umap",
    mics_methods: Set[str] | None = None,
    sampling_mode_filter: str | None = "superpixel",
) -> None:
    if not rows:
        print(
            "=== Paths where parametric_umap beats parametric_mics_lmc* "
            f"({metric_column}, sampling={sampling_mode_filter or 'all'}) ==="
        )
        print("None (no rows available).")
        return

    if metric_column not in rows[0]:
        print(
            "=== Paths where parametric_umap beats parametric_mics_lmc* "
            f"({metric_column}, sampling={sampling_mode_filter or 'all'}) ==="
        )
        print(f"None (metric column not found: {metric_column})")
        return

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        path = str(row.get("path", "")).strip()
        sampling_mode = str(row.get("sampling_mode", "")).strip()
        method = str(row.get("method", "")).strip()
        if not path or not sampling_mode or not method:
            continue
        if sampling_mode_filter and sampling_mode != sampling_mode_filter:
            continue
        grouped.setdefault((path, sampling_mode), []).append(row)

    winners: List[Tuple[str, str, float, float]] = []

    for (path, sampling_mode), grows in grouped.items():
        per_method_values: Dict[str, List[float]] = {}
        for row in grows:
            method = str(row.get("method", "")).strip()
            if not method:
                continue
            val = _parse_float(row.get(metric_column))
            if val is None:
                continue
            per_method_values.setdefault(method, []).append(val)

        per_method_mean: Dict[str, float] = {}
        for method, vals in per_method_values.items():
            if vals:
                per_method_mean[method] = float(mean(vals))

        umap_vals = [v for m, v in per_method_mean.items() if m == umap_method]
        if mics_methods:
            mics_vals = [v for m, v in per_method_mean.items() if m in mics_methods]
        else:
            mics_vals = [v for m, v in per_method_mean.items() if m.startswith("parametric_mics_lmc")]
        if not umap_vals or not mics_vals:
            continue
        best_umap = max(umap_vals)
        best_mics = max(mics_vals)
        if best_umap > best_mics:
            winners.append((path, sampling_mode, best_umap, best_mics))

    print(
        "=== Paths where parametric_umap beats parametric_mics_lmc* "
        f"({metric_column}, sampling={sampling_mode_filter or 'all'}) ==="
    )
    if not winners:
        print("None.")
        return

    for path, sampling_mode, best_umap, best_mics in sorted(winners):
        delta = best_umap - best_mics
        print(
            f"[sampling_mode={sampling_mode}] {path} | "
            f"umap={best_umap:.6f} best_mics={best_mics:.6f} delta={delta:.6f}"
        )

    unique_paths = sorted({path for path, _, _, _ in winners})
    print("--- Unique paths (any sampling_mode) ---")
    for path in unique_paths:
        print(path)


def _build_method_wins_rows(
    rows: Sequence[Dict[str, str]],
    metric_columns: Sequence[str],
    allowed_methods: Set[str] | None = None,
) -> List[Dict[str, object]]:
    if not rows:
        return []
    valid_metric_columns = [m for m in metric_columns if m in rows[0]]
    if not valid_metric_columns:
        return []

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        path = str(row.get("path", "")).strip()
        sampling_mode = str(row.get("sampling_mode", "")).strip()
        method = str(row.get("method", "")).strip()
        if not path or not sampling_mode or not method:
            continue
        if allowed_methods and method not in allowed_methods:
            continue
        grouped.setdefault((path, sampling_mode), []).append(row)

    def _average_tie_ranks_desc(score_by_method: Dict[str, float]) -> Dict[str, float]:
        # Higher score = better rank (1 is best). Ties get average rank.
        ordered = sorted(score_by_method.items(), key=lambda kv: kv[1], reverse=True)
        out: Dict[str, float] = {}
        i = 0
        n = len(ordered)
        while i < n:
            j = i + 1
            while j < n and ordered[j][1] == ordered[i][1]:
                j += 1
            avg_rank = 0.5 * ((i + 1) + j)
            for k in range(i, j):
                out[ordered[k][0]] = float(avg_rank)
            i = j
        return out

    out_rows: List[Dict[str, object]] = []
    for metric_column in valid_metric_columns:
        rank_lists_by_method_sampling: Dict[Tuple[str, str], List[float]] = {}

        for (_, sampling_mode), grows in grouped.items():
            per_method_values: Dict[str, List[float]] = {}
            for row in grows:
                method = str(row.get("method", "")).strip()
                val = _parse_float(row.get(metric_column))
                if not method or val is None:
                    continue
                per_method_values.setdefault(method, []).append(val)
            if not per_method_values:
                continue

            per_method_mean = {m: float(mean(vs)) for m, vs in per_method_values.items() if vs}
            if len(per_method_mean) < 2:
                continue
            ranks = _average_tie_ranks_desc(per_method_mean)
            for method, rank_val in ranks.items():
                key = (method, sampling_mode)
                rank_lists_by_method_sampling.setdefault(key, []).append(float(rank_val))

        for (method, sampling_mode), rank_vals in sorted(rank_lists_by_method_sampling.items()):
            n_cases = len(rank_vals)
            out_rows.append(
                {
                    "method": method,
                    "sampling_mode": sampling_mode,
                    "metric": metric_column,
                    "n_cases": int(n_cases),
                    "avg_rank": float(mean(rank_vals)) if n_cases > 0 else float("nan"),
                    "best_possible_rank": 1.0,
                }
            )
    return out_rows


def _method_to_filename_tag(method: str) -> str:
    safe = []
    for ch in str(method):
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _write_rows_csv(rows: Sequence[Dict[str, object]], csv_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate benchmark metrics over images for each method + sampling_mode, "
            "then create a comparison CSV with non-variable columns removed."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("benchmark_parametric_methods_aggregate.csv"),
        help="Input aggregate CSV with one row per image/method/sampling_mode.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_parametric_methods_method_sampling_aggregate.csv"),
        help="Output CSV aggregated over images (one row per method + sampling_mode).",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("benchmark_parametric_methods_method_sampling_comparison.csv"),
        help="Output comparison CSV with non-variable columns removed.",
    )
    parser.add_argument(
        "--wins-csv",
        type=Path,
        default=Path("benchmark_parametric_methods_method_sampling_wins.csv"),
        help="Output CSV with path-level wins per method and sampling_mode.",
    )
    parser.add_argument(
        "--winner-metric",
        type=str,
        default="nmf_lr_mean_spearman_mean",
        help=(
            "Metric column from input CSV used to report paths where "
            "parametric_umap beats parametric_mics_lmc*."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_csv: Path = args.input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows_all = _load_rows(input_csv)
    if not rows_all:
        raise ValueError(f"Input CSV has no rows: {input_csv}")
    input_row_count = len(rows_all)
    rows_after_exclusion, excluded_rows = _exclude_input_paths(rows_all, EXCLUDED_INPUT_PATHS)
    rows_before_seed_agg = len(rows_after_exclusion)
    rows_after_exclusion = _aggregate_over_seeds(rows_after_exclusion)
    if rows_before_seed_agg != len(rows_after_exclusion):
        print(f"Aggregated over seeds: {rows_before_seed_agg} -> {len(rows_after_exclusion)} rows")
    if not rows_after_exclusion:
        raise ValueError("No rows remain after excluding configured input paths.")
    _print_missing_paths_by_method(rows_after_exclusion)
    rows, dropped_rows = _filter_to_paths_present_in_all_methods(rows_after_exclusion)
    if not rows:
        raise ValueError("No rows remain after discarding paths missing in some methods.")

    metric_columns = _discover_metric_columns(rows)
    grouped = _group_rows(rows)
    aggregated = _aggregate_over_images(grouped, metric_columns)
    aggregated, removed_by_rule = _keep_included_columns(aggregated)
    _write_rows_csv(aggregated, args.output_csv)

    comparison_rows, removed_cols = _drop_non_variable_columns(
        aggregated,
        keep_columns=("method", "sampling_mode", "n_images", "n_seeds_total", "n_seeds_per_image"),
    )
    _write_rows_csv(comparison_rows, args.comparison_csv)

    wins_metric_columns = [m for m in _discover_metric_columns(rows_after_exclusion) if not str(m).endswith("_std")]
    mics_methods = sorted(
        {
            str(r.get("method", "")).strip()
            for r in rows_after_exclusion
            if str(r.get("method", "")).strip().startswith(MICS_METHOD_PREFIX)
        }
    )
    if not mics_methods:
        raise ValueError(f"No methods found with prefix: {MICS_METHOD_PREFIX}")

    written_rank_files: List[Path] = []
    for mics_method in mics_methods:
        methods_for_ranking = {"parametric_umap", "pca", mics_method}
        rank_rows = _build_method_wins_rows(
            rows_after_exclusion,
            metric_columns=wins_metric_columns,
            allowed_methods=methods_for_ranking,
        )
        if not rank_rows:
            continue
        rank_path = args.wins_csv.with_name(
            f"{args.wins_csv.stem}__rank_vs_{_method_to_filename_tag(mics_method)}{args.wins_csv.suffix}"
        )
        _write_rows_csv(rank_rows, rank_path)
        written_rank_files.append(rank_path)

    print(f"Loaded rows: {input_row_count}")
    print(f"Dropped rows by excluded input paths: {excluded_rows}")
    print(f"Rows used after path-coverage filter: {len(rows)}")
    print(f"Dropped rows missing all-method path coverage: {dropped_rows}")
    print(f"Groups (method + sampling_mode): {len(aggregated)}")
    if aggregated and "n_seeds_total" in aggregated[0]:
        print("Seeds per group (method + sampling_mode):")
        for r in aggregated:
            print(f"  {r['method']} + {r['sampling_mode']}: {r['n_seeds_total']} total, {r['n_seeds_per_image']} per image")
    print(f"Discovered metric columns: {len(metric_columns)}")
    print(f"Wrote aggregate-over-images CSV: {args.output_csv}")
    print(f"Wrote comparison CSV: {args.comparison_csv}")
    if written_rank_files:
        print("Wrote rank comparison CSV files:")
        for p in written_rank_files:
            print(f"  - {p}")
    else:
        print("No rank comparison CSV files were written (no comparable rows).")
    print(f"Removed columns by requested filters: {len(removed_by_rule)}")
    print(f"Removed non-variable columns: {len(removed_cols)}")


if __name__ == "__main__":
    main()
