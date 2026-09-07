#!/usr/bin/env python3
"""
Train NMF components from MSI data and regress them from RGB visualization.

Pipeline:
1) Load RGB image + MSI .npy
2) Fit NMF with k components on MSI spectra
3) Normalize component intensities spatially (per component)
4) Run 5-fold XGBoost regression for each component
5) Report/save accuracy metrics and optional component maps
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from sklearn.decomposition import NMF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from msi_visual.parametric_mics_lmc import random_pixel_sampling, uniform_superpixel_sampling


def _load_rgb(path: Path) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb


def _load_msi(path: Path, transpose_msi: bool) -> np.ndarray:
    msi = np.load(path, mmap_mode=None)
    if msi.ndim != 3:
        raise ValueError(f"Expected 3D MSI array, got shape {msi.shape}")

    if transpose_msi:
        msi = np.transpose(msi, (1, 2, 0))

    if msi.dtype != np.float32:
        msi = msi.astype(np.float32, copy=False)

    if np.nanmin(msi) < 0:
        raise ValueError("MSI array contains negative values. NMF requires non-negative inputs.")

    return msi


def _resize_rgb_to_shape(rgb: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = hw
    pil = Image.fromarray(rgb, mode="RGB")
    pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    sums = msi.sum(axis=-1, keepdims=True)
    sums = sums + 1e-8
    return msi / sums


def _build_mask(
    msi: np.ndarray,
    rgb: np.ndarray,
    use_msi_nonzero_mask: bool,
    use_rgb_nonzero_mask: bool,
) -> np.ndarray:
    mask = np.ones(msi.shape[:2], dtype=bool)
    if use_msi_nonzero_mask:
        mask &= (msi.sum(axis=-1) > 0)
    if use_rgb_nonzero_mask:
        mask &= (rgb.sum(axis=-1) > 0)
    return mask


def _resolve_rgb_paths(cfg: DictConfig) -> List[Path]:
    paths: List[Path] = []

    if "rgb_paths" in cfg.data and cfg.data.rgb_paths:
        for p in cfg.data.rgb_paths:
            if p:
                paths.append(Path(p))

    rgb_glob = cfg.data.rgb_glob if "rgb_glob" in cfg.data else None
    if rgb_glob:
        for p in sorted(glob.glob(str(rgb_glob))):
            paths.append(Path(p))

    if cfg.data.rgb_path:
        paths.append(Path(cfg.data.rgb_path))

    # Stable de-duplication.
    deduped: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            deduped.append(p)
            seen.add(key)

    if not deduped:
        raise ValueError("No RGB paths resolved. Set data.rgb_path, data.rgb_paths, or data.rgb_glob.")

    missing = [str(p) for p in deduped if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RGB path(s): {missing}")

    return deduped


def _normalize_targets(w: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return w
    if mode == "minmax_per_component":
        mins = w.min(axis=0, keepdims=True)
        maxs = w.max(axis=0, keepdims=True)
        return (w - mins) / (maxs - mins + 1e-8)
    raise ValueError(f"Unsupported target normalization mode: {mode}")


def _xgb_from_cfg(cfg: DictConfig) -> XGBRegressor:
    return XGBRegressor(
        objective=cfg.objective,
        n_estimators=int(cfg.n_estimators),
        learning_rate=float(cfg.learning_rate),
        max_depth=int(cfg.max_depth),
        min_child_weight=float(cfg.min_child_weight),
        subsample=float(cfg.subsample),
        colsample_bytree=float(cfg.colsample_bytree),
        reg_lambda=float(cfg.reg_lambda),
        reg_alpha=float(cfg.reg_alpha),
        tree_method=str(cfg.tree_method),
        n_jobs=int(cfg.n_jobs),
        random_state=42,
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a_std = a.std()
        b_std = b.std()
        if a_std < 1e-12 or b_std < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _rankdata_average(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.shape[0], dtype=np.float64)
        sorted_vals = values[order]
        n = len(values)
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_vals[j] == sorted_vals[i]:
                j += 1
            avg_rank = 0.5 * (i + j - 1) + 1.0
            ranks[order[i:j]] = avg_rank
            i = j
        return ranks

    def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
        return _safe_pearson(_rankdata_average(a), _rankdata_average(b))

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson": _safe_pearson(y_true, y_pred),
        "spearman": _safe_spearman(y_true, y_pred),
    }


def _to_uint8_map(
    x: np.ndarray,
    clip01: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if vmin is not None and vmax is not None:
        denom = float(vmax) - float(vmin)
        if denom <= 1e-12:
            x = np.zeros_like(x, dtype=np.float32)
        else:
            x = (x - float(vmin)) / denom
            x = np.clip(x, 0.0, 1.0)
    elif clip01:
        x = np.clip(x, 0.0, 1.0)
    else:
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.zeros_like(x, dtype=np.float32)
    return np.uint8(np.round(x * 255.0))


def _save_debug_component_images(
    output_dir: Path,
    debug_dirname: str,
    rgb_stem: str,
    target_map: np.ndarray,
    pred_map: np.ndarray,
    first_n_components: int,
    original_rgb: np.ndarray | None = None,
    save_sum_abs_error_image: bool = True,
    sum_abs_error_vmax: float = 1.0,
    error_display_gain: float = 5.0,
) -> None:
    n_comp = int(target_map.shape[-1])
    n_show = n_comp if int(first_n_components) <= 0 else max(1, min(int(first_n_components), n_comp))
    debug_root = output_dir / str(debug_dirname)
    rgb_dir = debug_root / rgb_stem
    rgb_dir.mkdir(parents=True, exist_ok=True)

    if original_rgb is not None:
        Image.fromarray(np.asarray(original_rgb, dtype=np.uint8), mode="RGB").save(
            rgb_dir / "rgb_input.png"
        )

    for comp_idx in range(n_show):
        target = target_map[:, :, comp_idx]
        pred = pred_map[:, :, comp_idx]
        abs_err = np.abs(pred - target)

        Image.fromarray(_to_uint8_map(target, clip01=True), mode="L").save(
            rgb_dir / f"component_{comp_idx:02d}_target.png"
        )
        Image.fromarray(_to_uint8_map(pred, clip01=True), mode="L").save(
            rgb_dir / f"component_{comp_idx:02d}_pred.png"
        )
        Image.fromarray(_to_uint8_map(abs_err, clip01=False), mode="L").save(
            rgb_dir / f"component_{comp_idx:02d}_abs_error.png"
        )

    if save_sum_abs_error_image:
        # Summary error image: per-pixel mean absolute error across all components.
        mean_abs_err = np.mean(np.abs(pred_map - target_map), axis=-1)
        avg_mean_abs_err = float(np.mean(mean_abs_err))
        print(
            f"[debug] {rgb_stem} mean_abs_error_all_components avg_pixel={avg_mean_abs_err:.6f} "
            f"(display_gain={float(error_display_gain):.3f})"
        )
        display_map = mean_abs_err * float(error_display_gain)
        Image.fromarray(
            _to_uint8_map(display_map, clip01=False, vmin=0.0, vmax=float(sum_abs_error_vmax)),
            mode="L",
        ).save(
            rgb_dir / "mean_abs_error_all_components.png"
        )


def _sorted_ranking_rows(
    ranking_rows: List[Dict[str, float]], rank_by: str, higher_is_better: set
) -> List[Dict[str, float]]:
    if not ranking_rows:
        return []
    if rank_by not in ranking_rows[0]:
        raise ValueError(f"comparison.rank_by='{rank_by}' is not a valid metric key.")

    def _sort_key(r: Dict[str, float]) -> Tuple[float, float, float]:
        primary = -r[rank_by] if rank_by in higher_is_better else r[rank_by]
        return (primary, r["mean_rmse"], r["mean_mae"])

    return sorted(ranking_rows, key=_sort_key)


def _write_ranking_csv(csv_path: Path, ranking: List[Dict[str, float]]) -> None:
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("rank,rgb_path,mean_r2,mean_rmse,mean_mae,mean_pearson,mean_spearman\n")
        for item in ranking:
            f.write(
                f"{item['rank']},\"{item['rgb_path']}\",{item['mean_r2']:.8f},"
                f"{item['mean_rmse']:.8f},{item['mean_mae']:.8f},"
                f"{item['mean_pearson']:.8f},{item['mean_spearman']:.8f}\n"
            )


def _make_rgb_collage(
    items: List[Dict[str, float]],
    rgb_cache: Dict[str, np.ndarray],
    output_path: Path,
    thumb_size: int = 256,
    header: str | None = None,
    rows: int = 1,
    cols: int = 5,
) -> None:
    if not items:
        return

    pad = 8
    text_h = 28
    header_h = 30 if header else 0
    canvas_w = cols * thumb_size + (cols + 1) * pad
    canvas_h = header_h + rows * (thumb_size + text_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    if header:
        draw.text((pad, 8), header, fill=(255, 255, 255))

    for i, item in enumerate(items[: rows * cols]):
        r = i // cols
        c = i % cols
        x0 = pad + c * (thumb_size + pad)
        y0 = header_h + pad + r * (thumb_size + text_h)
        rgb = rgb_cache.get(str(item["rgb_path"]))
        if rgb is None:
            continue
        thumb = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").resize(
            (thumb_size, thumb_size), resample=Image.BILINEAR
        )
        canvas.paste(thumb, (x0, y0))
        label = f"#{item['rank']} {Path(item['rgb_path']).name}"
        draw.text((x0, y0 + thumb_size + 4), label[:48], fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _aggregate_error_map(
    maps: List[np.ndarray], selected_indices: List[int], objective: str
) -> np.ndarray:
    stack = np.stack([maps[i] for i in selected_indices], axis=0)
    if objective == "min":
        return np.min(stack, axis=0)
    if objective == "mean":
        return np.mean(stack, axis=0)
    raise ValueError(f"Unsupported panel objective: {objective}")


def _panel_score(agg_map: np.ndarray, valid_mask: np.ndarray) -> float:
    valid = agg_map[valid_mask]
    if valid.size == 0:
        return float(np.mean(agg_map))
    return float(np.mean(valid))


def _select_best_panel_greedy(
    error_maps: List[np.ndarray], valid_mask: np.ndarray, panel_size: int, objective: str
) -> Tuple[List[int], float, np.ndarray]:
    n = len(error_maps)
    k = max(1, min(int(panel_size), n))
    selected: List[int] = []
    remaining = list(range(n))
    best_agg = None
    best_score = float("inf")

    for _ in range(k):
        step_best_idx = None
        step_best_score = float("inf")
        step_best_agg = None
        for idx in remaining:
            candidate = selected + [idx]
            agg = _aggregate_error_map(error_maps, candidate, objective)
            score = _panel_score(agg, valid_mask)
            if score < step_best_score:
                step_best_score = score
                step_best_idx = idx
                step_best_agg = agg
        if step_best_idx is None:
            break
        selected.append(step_best_idx)
        remaining.remove(step_best_idx)
        best_score = step_best_score
        best_agg = step_best_agg

    if best_agg is None:
        best_agg = np.zeros_like(error_maps[0], dtype=np.float32)
        best_score = _panel_score(best_agg, valid_mask)
    return selected, best_score, best_agg


def _sample_nmf_train_indices(msi: np.ndarray, valid_mask: np.ndarray, cfg: DictConfig) -> np.ndarray:
    h, w = msi.shape[:2]
    n_valid = int(valid_mask.sum())
    method = str(cfg.nmf.sampling.method).strip().lower()
    budget = int(cfg.nmf.sampling.num_samples)
    rng_seed = int(cfg.nmf.sampling.random_state)

    if method == "all" or budget <= 0 or budget >= n_valid:
        return np.arange(n_valid, dtype=np.int64)

    x_raw = msi.reshape(-1, msi.shape[-1])
    if method == "random":
        _, sampled_flat_indices, _ = random_pixel_sampling(
            x_raw,
            approx_budget=budget,
            height=h,
            width=w,
            roi_mask=valid_mask,
            rng_seed=rng_seed,
        )
    elif method == "superpixel":
        _, sampled_flat_indices, _ = uniform_superpixel_sampling(
            X=msi,
            X_raw=x_raw,
            approx_budget=budget,
            init_superpixels=int(cfg.nmf.sampling.superpixel_init_segments),
            init_points_per_superpixel=int(cfg.nmf.sampling.superpixel_points_per_segment),
            compactness=float(cfg.nmf.sampling.superpixel_compactness),
            max_iters=int(cfg.nmf.sampling.superpixel_max_iters),
            verbose=bool(cfg.nmf.sampling.superpixel_verbose),
            visualize=False,
            rng_seed=rng_seed,
            roi_mask=valid_mask,
            pca_fit_step=int(cfg.nmf.sampling.superpixel_pca_fit_step),
        )
    else:
        raise ValueError(f"Unknown nmf.sampling.method: {method}")

    if sampled_flat_indices is None or len(sampled_flat_indices) == 0:
        # Fallback to random valid sampling to avoid failures.
        rng = np.random.default_rng(rng_seed)
        valid_positions = np.flatnonzero(valid_mask.ravel())
        k = min(len(valid_positions), max(1, budget))
        sampled_flat_indices = rng.choice(valid_positions, size=k, replace=False)

    # Map global flattened indices -> local index in valid pixel ordering.
    valid_flat = np.flatnonzero(valid_mask.ravel())
    flat_to_valid = np.full(h * w, -1, dtype=np.int64)
    flat_to_valid[valid_flat] = np.arange(len(valid_flat), dtype=np.int64)

    sampled_valid_indices = flat_to_valid[np.asarray(sampled_flat_indices, dtype=np.int64)]
    sampled_valid_indices = sampled_valid_indices[sampled_valid_indices >= 0]
    sampled_valid_indices = np.unique(sampled_valid_indices)

    if len(sampled_valid_indices) == 0:
        raise ValueError("NMF sampling returned no valid points.")

    if len(sampled_valid_indices) > budget:
        rng = np.random.default_rng(rng_seed)
        sampled_valid_indices = np.sort(rng.choice(sampled_valid_indices, size=budget, replace=False))

    return sampled_valid_indices.astype(np.int64, copy=False)


def _run_cv_per_component(x: np.ndarray, y: np.ndarray, cv_cfg: DictConfig, xgb_cfg: DictConfig) -> Dict[str, object]:
    n_splits = int(cv_cfg.n_splits)
    if x.shape[0] < n_splits:
        raise ValueError(f"Not enough samples for CV: {x.shape[0]} < n_splits={n_splits}")

    kf = KFold(
        n_splits=n_splits,
        shuffle=bool(cv_cfg.shuffle),
        random_state=int(cv_cfg.random_state),
    )

    n_comp = y.shape[1]
    component_metrics: Dict[str, object] = {}
    for comp_idx in range(n_comp):
        y_comp = y[:, comp_idx]
        fold_metrics = []
        for train_idx, test_idx in kf.split(x):
            model = _xgb_from_cfg(xgb_cfg)
            model.fit(x[train_idx], y_comp[train_idx])
            pred = model.predict(x[test_idx])
            fold_metrics.append(_compute_metrics(y_comp[test_idx], pred))

        r2_vals = np.array([m["r2"] for m in fold_metrics], dtype=np.float64)
        mae_vals = np.array([m["mae"] for m in fold_metrics], dtype=np.float64)
        rmse_vals = np.array([m["rmse"] for m in fold_metrics], dtype=np.float64)
        pearson_vals = np.array([m["pearson"] for m in fold_metrics], dtype=np.float64)
        spearman_vals = np.array([m["spearman"] for m in fold_metrics], dtype=np.float64)
        component_metrics[f"component_{comp_idx}"] = {
            "folds": fold_metrics,
            "mean": {
                "r2": float(r2_vals.mean()),
                "mae": float(mae_vals.mean()),
                "rmse": float(rmse_vals.mean()),
                "pearson": float(pearson_vals.mean()),
                "spearman": float(spearman_vals.mean()),
            },
            "std": {
                "r2": float(r2_vals.std()),
                "mae": float(mae_vals.std()),
                "rmse": float(rmse_vals.std()),
                "pearson": float(pearson_vals.std()),
                "spearman": float(spearman_vals.std()),
            },
        }

    all_r2 = np.array([component_metrics[k]["mean"]["r2"] for k in component_metrics], dtype=np.float64)
    all_mae = np.array([component_metrics[k]["mean"]["mae"] for k in component_metrics], dtype=np.float64)
    all_rmse = np.array([component_metrics[k]["mean"]["rmse"] for k in component_metrics], dtype=np.float64)
    all_pearson = np.array([component_metrics[k]["mean"]["pearson"] for k in component_metrics], dtype=np.float64)
    all_spearman = np.array([component_metrics[k]["mean"]["spearman"] for k in component_metrics], dtype=np.float64)
    return {
        "components": component_metrics,
        "overall": {
            "mean_r2": float(all_r2.mean()),
            "mean_mae": float(all_mae.mean()),
            "mean_rmse": float(all_rmse.mean()),
            "mean_pearson": float(all_pearson.mean()),
            "mean_spearman": float(all_spearman.mean()),
        },
    }


@hydra.main(version_base=None, config_path="configs", config_name="train_nmf_xgb_from_rgb")
def main(cfg: DictConfig) -> None:
    rgb_paths = _resolve_rgb_paths(cfg)
    msi_path = Path(cfg.data.msi_npy_path)

    if not msi_path.exists():
        raise FileNotFoundError(f"MSI .npy path not found: {msi_path}")

    output_dir = Path(cfg.output.out_dir) if cfg.output.out_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    msi = _load_msi(msi_path, bool(cfg.data.transpose_msi))

    if bool(cfg.nmf.tic_normalize):
        msi = _tic_normalize(msi)

    # Use first RGB for validating shape alignment and base mask creation.
    ref_rgb = _load_rgb(rgb_paths[0])
    if cfg.data.resize_rgb_to_msi and ref_rgb.shape[:2] != msi.shape[:2]:
        ref_rgb = _resize_rgb_to_shape(ref_rgb, msi.shape[:2])
    elif ref_rgb.shape[:2] != msi.shape[:2]:
        raise ValueError(
            f"RGB/MSI size mismatch: RGB {ref_rgb.shape[:2]} vs MSI {msi.shape[:2]}. "
            "Enable data.resize_rgb_to_msi to auto-align."
        )

    mask = _build_mask(
        msi=msi,
        rgb=ref_rgb,
        use_msi_nonzero_mask=bool(cfg.data.use_msi_nonzero_mask),
        use_rgb_nonzero_mask=False,
    )
    if not np.any(mask):
        raise ValueError("Mask removed all pixels. Adjust mask settings.")

    x_msi = msi[mask].astype(np.float32)
    n_valid = x_msi.shape[0]

    nmf_train_idx = _sample_nmf_train_indices(msi, mask, cfg)
    x_msi_nmf = x_msi[nmf_train_idx]
    print(
        f"Valid pixels: {n_valid:,}, MSI bins: {x_msi.shape[1]} | "
        f"NMF fit pixels: {x_msi_nmf.shape[0]:,} ({100.0 * x_msi_nmf.shape[0] / max(1, n_valid):.1f}%)"
    )

    print("Fitting NMF...")
    nmf = NMF(
        n_components=int(cfg.nmf.k_components),
        init=str(cfg.nmf.init),
        max_iter=int(cfg.nmf.max_iter),
        tol=float(cfg.nmf.tol),
        random_state=int(cfg.nmf.random_state),
    )
    w_nmf = nmf.fit_transform(x_msi_nmf)

    target_scope = str(cfg.nmf.target_scope).strip().lower()
    w_all = None
    if target_scope == "sampled":
        target_positions = nmf_train_idx
        y_targets_matrix = _normalize_targets(w_nmf.astype(np.float32, copy=False), str(cfg.nmf.target_normalization))
    elif target_scope == "all_valid":
        w_all = nmf.transform(x_msi)
        target_positions = np.arange(n_valid, dtype=np.int64)
        y_targets_matrix = _normalize_targets(w_all.astype(np.float32, copy=False), str(cfg.nmf.target_normalization))
    else:
        raise ValueError(f"Unsupported nmf.target_scope: {target_scope}")

    y_row_by_valid = np.full(n_valid, -1, dtype=np.int64)
    y_row_by_valid[target_positions] = np.arange(len(target_positions), dtype=np.int64)

    # Shared eval subset for fair ranking across visualizations.
    eval_positions = target_positions.copy()
    max_samples = int(cfg.cv.max_samples)
    if 0 < max_samples < len(eval_positions):
        rng = np.random.default_rng(int(cfg.cv.random_state))
        eval_positions = np.sort(rng.choice(eval_positions, size=max_samples, replace=False))
        print(f"Subsampled to {len(eval_positions):,} target pixels for faster CV.")

    visualization_target_scope = str(getattr(cfg.nmf, "visualization_target_scope", "target_scope")).strip().lower()

    results: Dict[str, object] = {
        "data": {
            "rgb_paths": [str(p) for p in rgb_paths],
            "msi_npy_path": str(msi_path),
            "valid_pixels": int(mask.sum()),
        },
        "nmf": {
            "k_components": int(cfg.nmf.k_components),
            "reconstruction_err": float(nmf.reconstruction_err_),
            "fit_pixels": int(len(nmf_train_idx)),
            "target_scope": target_scope,
            "visualization_target_scope": visualization_target_scope,
            "target_pixels": int(len(target_positions)),
        },
        "comparison": {
            "visualizations": [],
            "ranking": [],
            "failed": [],
        },
    }

    n_comp = y_targets_matrix.shape[1]
    h, w_spatial = msi.shape[:2]
    if visualization_target_scope == "target_scope":
        target_full_for_viz = np.zeros((n_valid, n_comp), dtype=np.float32)
        target_full_for_viz[target_positions] = y_targets_matrix
    elif visualization_target_scope == "all_valid":
        if w_all is None:
            w_all = nmf.transform(x_msi)
        target_full_for_viz = _normalize_targets(
            w_all.astype(np.float32, copy=False),
            str(cfg.nmf.target_normalization),
        )
    else:
        raise ValueError(
            f"Unsupported nmf.visualization_target_scope: {visualization_target_scope}"
        )

    target_map = np.zeros((h, w_spatial, n_comp), dtype=np.float32)
    target_map[mask] = target_full_for_viz
    if bool(cfg.output.save_component_maps):
        np.save(output_dir / "nmf_component_targets.npy", target_map)

    print(f"Comparing {len(rgb_paths)} visualization(s) with 5-fold XGBoost regression...")
    ranking_rows = []
    rgb_cache: Dict[str, np.ndarray] = {}
    panel_error_maps: List[np.ndarray] = []
    panel_rgb_paths: List[str] = []
    rank_by = str(getattr(cfg.comparison, "rank_by", "mean_spearman"))
    higher_is_better = set(
        getattr(
            cfg.comparison,
            "higher_is_better_metrics",
            ["mean_r2", "mean_pearson", "mean_spearman"],
        )
    )
    ranking_csv_path = output_dir / "ranking.csv"
    for rgb_idx, rgb_path in enumerate(rgb_paths):
        try:
            rgb = _load_rgb(rgb_path)
            if cfg.data.resize_rgb_to_msi and rgb.shape[:2] != msi.shape[:2]:
                rgb = _resize_rgb_to_shape(rgb, msi.shape[:2])
            elif rgb.shape[:2] != msi.shape[:2]:
                raise ValueError(
                    f"RGB/MSI size mismatch for {rgb_path}: RGB {rgb.shape[:2]} vs MSI {msi.shape[:2]}. "
                    "Enable data.resize_rgb_to_msi to auto-align."
                )

            x_rgb_valid = (rgb[mask].astype(np.float32) / 255.0)
            rgb_cache[str(rgb_path)] = rgb
            if bool(cfg.data.use_rgb_nonzero_mask):
                rgb_valid = (rgb.sum(axis=-1) > 0)[mask]
                rgb_eval_positions = eval_positions[rgb_valid[eval_positions]]
            else:
                rgb_eval_positions = eval_positions

            y_eval_rows = y_row_by_valid[rgb_eval_positions]
            y_eval_rows = y_eval_rows[y_eval_rows >= 0]
            if len(y_eval_rows) < int(cfg.cv.n_splits):
                raise ValueError(
                    f"Not enough evaluation pixels for {rgb_path}: {len(y_eval_rows)} "
                    f"(need at least {cfg.cv.n_splits})."
                )

            x_eval = x_rgb_valid[rgb_eval_positions[: len(y_eval_rows)]]
            y_eval = y_targets_matrix[y_eval_rows]

            cv_result = _run_cv_per_component(x_eval, y_eval, cfg.cv, cfg.xgb)
            overall = cv_result["overall"]
            print(
                f"[{rgb_idx + 1}/{len(rgb_paths)}] {rgb_path.name} | "
                f"R2={overall['mean_r2']:.4f}, MAE={overall['mean_mae']:.4f}, RMSE={overall['mean_rmse']:.4f}, "
                f"Pearson={overall['mean_pearson']:.4f}, Spearman={overall['mean_spearman']:.4f}"
            )

            per_rgb = {
                "rgb_path": str(rgb_path),
                "cv_pixels": int(len(y_eval_rows)),
                "components": cv_result["components"],
                "overall": overall,
            }
            results["comparison"]["visualizations"].append(per_rgb)
            ranking_rows.append(
                {
                    "rgb_path": str(rgb_path),
                    "mean_r2": float(overall["mean_r2"]),
                    "mean_mae": float(overall["mean_mae"]),
                    "mean_rmse": float(overall["mean_rmse"]),
                    "mean_pearson": float(overall["mean_pearson"]),
                    "mean_spearman": float(overall["mean_spearman"]),
                }
            )

            need_prediction_maps = bool(cfg.output.train_full_models) or bool(
                getattr(cfg.output, "save_debug_component_images", False)
            ) or bool(getattr(cfg.output, "track_best_panel", True))
            if need_prediction_maps:
                pred_map = np.zeros((msi.shape[0], msi.shape[1], n_comp), dtype=np.float32)
                train_positions = np.asarray(rgb_eval_positions[: len(y_eval_rows)], dtype=np.int64)
                train_rows = y_row_by_valid[train_positions]
                train_rows = train_rows[train_rows >= 0]
                x_train_full = x_rgb_valid[train_positions[: len(train_rows)]]
                y_train_full = y_targets_matrix[train_rows]

                x_pred_all = x_rgb_valid
                pred_valid = np.zeros((x_pred_all.shape[0], n_comp), dtype=np.float32)
                for comp_idx in range(n_comp):
                    model = _xgb_from_cfg(cfg.xgb)
                    model.fit(x_train_full, y_train_full[:, comp_idx])
                    pred_valid[:, comp_idx] = model.predict(x_pred_all).astype(np.float32)

                pred_map[mask] = pred_valid
                if bool(cfg.output.train_full_models) and bool(cfg.output.save_component_maps):
                    np.save(output_dir / f"xgb_component_predictions__{rgb_path.stem}.npy", pred_map)
                if bool(getattr(cfg.output, "save_debug_component_images", False)):
                    _save_debug_component_images(
                        output_dir=output_dir,
                        debug_dirname=str(getattr(cfg.output, "debug_dirname", "debug")),
                        rgb_stem=rgb_path.stem,
                        target_map=target_map,
                        pred_map=pred_map,
                        first_n_components=int(getattr(cfg.output, "debug_first_n_components", 5)),
                        original_rgb=rgb,
                        save_sum_abs_error_image=bool(
                            getattr(cfg.output, "save_debug_sum_abs_error_image", True)
                        ),
                        sum_abs_error_vmax=float(getattr(cfg.output, "debug_error_vmax", 1.0)),
                        error_display_gain=float(getattr(cfg.output, "debug_error_display_gain", 5.0)),
                    )

                if bool(getattr(cfg.output, "track_best_panel", True)):
                    err_map = np.mean(np.abs(pred_map - target_map), axis=-1).astype(np.float32)
                    panel_error_maps.append(err_map)
                    panel_rgb_paths.append(str(rgb_path))
                    panel_size = int(getattr(cfg.output, "best_panel_size", 8))
                    panel_objective = str(getattr(cfg.output, "best_panel_objective", "min")).strip().lower()
                    selected_idx, panel_score, agg_map = _select_best_panel_greedy(
                        panel_error_maps, mask, panel_size, panel_objective
                    )
                    tracking_dir = (
                        output_dir
                        / str(getattr(cfg.output, "debug_dirname", "debug"))
                        / str(getattr(cfg.output, "best_panel_tracking_dirname", "panel_tracking"))
                    )
                    tracking_dir.mkdir(parents=True, exist_ok=True)

                    selected_items = []
                    for local_rank, idx in enumerate(selected_idx, start=1):
                        p = panel_rgb_paths[idx]
                        selected_items.append(
                            {
                                "rank": local_rank,
                                "rgb_path": p,
                                "mean_r2": 0.0,
                                "mean_rmse": 0.0,
                                "mean_mae": 0.0,
                                "mean_pearson": 0.0,
                                "mean_spearman": 0.0,
                            }
                        )
                    _make_rgb_collage(
                        items=selected_items,
                        rgb_cache=rgb_cache,
                        output_path=tracking_dir / "best_panel.png",
                        thumb_size=int(getattr(cfg.output, "best_worst_thumb_size", 256)),
                        header=(
                            f"Best panel up to step {rgb_idx + 1} | "
                            f"objective={panel_objective} | score={panel_score:.6f}"
                        ),
                        rows=max(1, int(np.ceil(len(selected_items) / max(1, int(getattr(cfg.output, "best_panel_columns", 4)))))),
                        cols=max(1, int(getattr(cfg.output, "best_panel_columns", 4))),
                    )

                    agg_uint8 = _to_uint8_map(
                        agg_map * float(getattr(cfg.output, "debug_error_display_gain", 5.0)),
                        clip01=False,
                        vmin=0.0,
                        vmax=float(getattr(cfg.output, "debug_error_vmax", 1.0)),
                    )
                    Image.fromarray(agg_uint8, mode="L").save(tracking_dir / "panel_aggregate_error.png")
                    with (tracking_dir / "panel_info.json").open("w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "step": int(rgb_idx + 1),
                                "objective": panel_objective,
                                "panel_size": int(panel_size),
                                "score_mean_error_on_mask": float(panel_score),
                                "selected_rgb_paths": [panel_rgb_paths[i] for i in selected_idx],
                            },
                            f,
                            indent=2,
                        )
        except Exception as e:
            print(f"[{rgb_idx + 1}/{len(rgb_paths)}] {rgb_path.name} | FAILED: {e}")
            results["comparison"]["failed"].append(
                {
                    "rgb_path": str(rgb_path),
                    "error": str(e),
                }
            )
            if bool(cfg.output.save_ranking_csv) and bool(getattr(cfg.output, "save_ranking_csv_incremental", True)):
                incremental_sorted = _sorted_ranking_rows(ranking_rows, rank_by, higher_is_better)
                incremental_ranking = [{"rank": idx + 1, **row} for idx, row in enumerate(incremental_sorted)]
                _write_ranking_csv(ranking_csv_path, incremental_ranking)
            continue

        if bool(cfg.output.save_ranking_csv) and bool(getattr(cfg.output, "save_ranking_csv_incremental", True)):
            incremental_sorted = _sorted_ranking_rows(ranking_rows, rank_by, higher_is_better)
            incremental_ranking = [{"rank": idx + 1, **row} for idx, row in enumerate(incremental_sorted)]
            _write_ranking_csv(ranking_csv_path, incremental_ranking)
            if bool(getattr(cfg.output, "save_best_worst_collage", True)) and incremental_ranking:
                k = max(1, int(getattr(cfg.output, "best_worst_count", 5)))
                best_items = incremental_ranking[:k]
                worst_items = list(reversed(incremental_ranking[-k:]))
                collage_items = best_items + worst_items
                _make_rgb_collage(
                    items=collage_items,
                    rgb_cache=rgb_cache,
                    output_path=output_dir / "best_vs_worst_collage.png",
                    thumb_size=int(getattr(cfg.output, "best_worst_thumb_size", 256)),
                    header=f"Top {k} best (row 1) and top {k} worst (row 2) by {rank_by} [incremental]",
                    rows=2,
                    cols=k,
                )

    if ranking_rows:
        ranking_rows = _sorted_ranking_rows(ranking_rows, rank_by, higher_is_better)
        results["comparison"]["ranking"] = [
            {"rank": idx + 1, **row} for idx, row in enumerate(ranking_rows)
        ]
        print(f"Ranking (best -> worst, by {rank_by}):")
        for item in results["comparison"]["ranking"]:
            print(
                f"  {item['rank']:02d}. {Path(item['rgb_path']).name} | "
                f"R2={item['mean_r2']:.4f}, RMSE={item['mean_rmse']:.4f}, MAE={item['mean_mae']:.4f}, "
                f"Pearson={item['mean_pearson']:.4f}, Spearman={item['mean_spearman']:.4f}"
            )

        if bool(getattr(cfg.output, "save_best_worst_collage", True)):
            k = max(1, int(getattr(cfg.output, "best_worst_count", 5)))
            best_items = results["comparison"]["ranking"][:k]
            worst_items = list(reversed(results["comparison"]["ranking"][-k:]))
            collage_items = best_items + worst_items
            _make_rgb_collage(
                items=collage_items,
                rgb_cache=rgb_cache,
                output_path=output_dir / "best_vs_worst_collage.png",
                thumb_size=int(getattr(cfg.output, "best_worst_thumb_size", 256)),
                header=f"Top {k} best (row 1) and top {k} worst (row 2) by {rank_by}",
                rows=2,
                cols=k,
            )
    else:
        print("No successful visualizations to rank (all failed).")

    if bool(cfg.output.save_ranking_csv) and results["comparison"]["ranking"]:
        _write_ranking_csv(ranking_csv_path, results["comparison"]["ranking"])

    if bool(getattr(cfg.output, "save_summary_txt", True)):
        summary_path = output_dir / "summary.txt"
        rank_by = str(getattr(cfg.comparison, "rank_by", "mean_spearman"))
        top_n = int(getattr(cfg.output, "top_n_in_summary", 10))
        top_n = max(1, top_n)
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("NMF + XGBoost Comparison Summary\n")
            f.write("=" * 80 + "\n")
            f.write(f"MSI file: {msi_path}\n")
            f.write(f"Visualizations provided: {len(rgb_paths)}\n")
            f.write(f"Successful: {len(results['comparison']['visualizations'])}\n")
            f.write(f"Failed: {len(results['comparison']['failed'])}\n")
            f.write(f"NMF components: {int(cfg.nmf.k_components)}\n")
            f.write(f"NMF fit pixels: {int(len(nmf_train_idx))}\n")
            f.write(f"Target scope: {target_scope}\n")
            f.write(f"Ranking metric: {rank_by}\n")
            f.write("\nTop ranked visualizations:\n")

            if results["comparison"]["ranking"]:
                for item in results["comparison"]["ranking"][:top_n]:
                    f.write(
                        f"  {item['rank']:02d}. {item['rgb_path']} | "
                        f"R2={item['mean_r2']:.4f} "
                        f"RMSE={item['mean_rmse']:.4f} "
                        f"MAE={item['mean_mae']:.4f} "
                        f"Pearson={item['mean_pearson']:.4f} "
                        f"Spearman={item['mean_spearman']:.4f}\n"
                    )
            else:
                f.write("  (no successful visualizations)\n")

            if results["comparison"]["failed"]:
                f.write("\nFailed visualizations:\n")
                for item in results["comparison"]["failed"]:
                    f.write(f"  - {item['rgb_path']} | error: {item['error']}\n")

    if bool(cfg.output.save_metrics_json):
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
