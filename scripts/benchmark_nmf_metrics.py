#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from msi_visual.parametric_mics_lmc import random_pixel_sampling, uniform_superpixel_sampling


@dataclass
class NMFPredictionConfig:
    k_components: int = 6
    max_iter: int = 300
    tol: float = 1e-4
    init: str = "nndsvda"
    random_state: int = 42
    target_normalization: str = "minmax_per_component"
    target_scope: str = "all_valid"  # sampled | all_valid
    sampling_method: str = "random"  # all | random | superpixel
    sampling_num_samples: int = 50000
    superpixel_init_segments: int = 256
    superpixel_points_per_segment: int = 8
    superpixel_compactness: float = 10.0
    superpixel_max_iters: int = 6
    superpixel_pca_fit_step: int = 4
    superpixel_verbose: bool = False
    cv_n_splits: int = 5
    cv_shuffle: bool = True
    cv_random_state: int = 42
    cv_max_samples: int = 300000
    xgb_n_estimators: int = 300
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_min_child_weight: float = 2.0
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 1.0
    xgb_reg_alpha: float = 0.0
    xgb_tree_method: str = "hist"
    xgb_n_jobs: int = -1


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


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_std = a.std()
    b_std = b.std()
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_pearson(_rankdata_average(a), _rankdata_average(b))


def _normalize_targets(w: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return w
    if mode == "minmax_per_component":
        mins = w.min(axis=0, keepdims=True)
        maxs = w.max(axis=0, keepdims=True)
        return (w - mins) / (maxs - mins + 1e-8)
    raise ValueError(f"Unsupported target normalization mode: {mode}")


def _xgb_from_cfg(cfg: NMFPredictionConfig) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(cfg.xgb_n_estimators),
        learning_rate=float(cfg.xgb_learning_rate),
        max_depth=int(cfg.xgb_max_depth),
        min_child_weight=float(cfg.xgb_min_child_weight),
        subsample=float(cfg.xgb_subsample),
        colsample_bytree=float(cfg.xgb_colsample_bytree),
        reg_lambda=float(cfg.xgb_reg_lambda),
        reg_alpha=float(cfg.xgb_reg_alpha),
        tree_method=str(cfg.xgb_tree_method),
        n_jobs=int(cfg.xgb_n_jobs),
        random_state=int(cfg.cv_random_state),
    )


def _sample_nmf_train_indices(msi: np.ndarray, valid_mask: np.ndarray, cfg: NMFPredictionConfig) -> np.ndarray:
    h, w = msi.shape[:2]
    n_valid = int(valid_mask.sum())
    method = str(cfg.sampling_method).strip().lower()
    budget = int(cfg.sampling_num_samples)
    rng_seed = int(cfg.random_state)

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
            init_superpixels=int(cfg.superpixel_init_segments),
            init_points_per_superpixel=int(cfg.superpixel_points_per_segment),
            compactness=float(cfg.superpixel_compactness),
            max_iters=int(cfg.superpixel_max_iters),
            verbose=bool(cfg.superpixel_verbose),
            visualize=False,
            rng_seed=rng_seed,
            roi_mask=valid_mask,
            pca_fit_step=int(cfg.superpixel_pca_fit_step),
        )
    else:
        raise ValueError(f"Unknown NMF sampling method: {method}")

    if sampled_flat_indices is None or len(sampled_flat_indices) == 0:
        rng = np.random.default_rng(rng_seed)
        valid_positions = np.flatnonzero(valid_mask.ravel())
        k = min(len(valid_positions), max(1, budget))
        sampled_flat_indices = rng.choice(valid_positions, size=k, replace=False)

    valid_flat = np.flatnonzero(valid_mask.ravel())
    flat_to_valid = np.full(h * w, -1, dtype=np.int64)
    flat_to_valid[valid_flat] = np.arange(len(valid_flat), dtype=np.int64)
    sampled_valid_indices = flat_to_valid[np.asarray(sampled_flat_indices, dtype=np.int64)]
    sampled_valid_indices = sampled_valid_indices[sampled_valid_indices >= 0]
    sampled_valid_indices = np.unique(sampled_valid_indices)
    if len(sampled_valid_indices) == 0:
        raise ValueError("NMF sampling returned no valid points.")
    return sampled_valid_indices.astype(np.int64, copy=False)


class NMFPredictionEvaluator:
    def __init__(self, msi_img: np.ndarray, cfg: NMFPredictionConfig, metric_mask: np.ndarray | None = None):
        self.cfg = cfg
        self.msi = np.asarray(msi_img, dtype=np.float32)
        self.mask = self.msi.sum(axis=-1) > 0
        if not np.any(self.mask):
            raise ValueError("MSI mask is empty.")
        self.x_msi = self.msi[self.mask].astype(np.float32, copy=False)
        if metric_mask is None:
            self.metric_mask_valid = np.ones(self.x_msi.shape[0], dtype=bool)
        else:
            metric_mask = np.asarray(metric_mask, dtype=bool)
            if metric_mask.shape != self.mask.shape:
                raise ValueError(
                    f"metric_mask shape mismatch: expected {self.mask.shape}, got {metric_mask.shape}"
                )
            # Keep only pixels valid in MSI and selected by metric mask.
            self.metric_mask_valid = metric_mask[self.mask]
            if not np.any(self.metric_mask_valid):
                raise ValueError("metric_mask removed all valid pixels for metric evaluation.")
        self._prepare_targets()

    def _prepare_targets(self) -> None:
        nmf_train_idx = _sample_nmf_train_indices(self.msi, self.mask, self.cfg)
        x_train = self.x_msi[nmf_train_idx]
        nmf = NMF(
            n_components=int(self.cfg.k_components),
            init=str(self.cfg.init),
            max_iter=int(self.cfg.max_iter),
            tol=float(self.cfg.tol),
            random_state=int(self.cfg.random_state),
        )
        w_nmf = nmf.fit_transform(x_train)
        target_scope = str(self.cfg.target_scope).strip().lower()
        if target_scope == "sampled":
            target_positions = nmf_train_idx
            y_targets = _normalize_targets(w_nmf.astype(np.float32, copy=False), self.cfg.target_normalization)
        elif target_scope == "all_valid":
            w_all = nmf.transform(self.x_msi)
            target_positions = np.arange(self.x_msi.shape[0], dtype=np.int64)
            y_targets = _normalize_targets(w_all.astype(np.float32, copy=False), self.cfg.target_normalization)
        else:
            raise ValueError(f"Unsupported target_scope: {target_scope}")

        y_row_by_valid = np.full(self.x_msi.shape[0], -1, dtype=np.int64)
        y_row_by_valid[target_positions] = np.arange(len(target_positions), dtype=np.int64)

        eval_positions = target_positions.copy()
        if 0 < int(self.cfg.cv_max_samples) < len(eval_positions):
            rng = np.random.default_rng(int(self.cfg.cv_random_state))
            eval_positions = np.sort(rng.choice(eval_positions, size=int(self.cfg.cv_max_samples), replace=False))

        self.y_targets = y_targets
        self.y_row_by_valid = y_row_by_valid
        self.eval_positions = eval_positions
        self.n_components = int(y_targets.shape[1])
        self.nmf_reconstruction_err = float(nmf.reconstruction_err_)

    def _build_eval_arrays(self, rgb_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_rgb_valid = (np.asarray(rgb_uint8, dtype=np.float32)[self.mask] / 255.0).astype(np.float32, copy=False)
        eval_positions = self.eval_positions[self.metric_mask_valid[self.eval_positions]]
        y_rows = self.y_row_by_valid[eval_positions]
        y_rows = y_rows[y_rows >= 0]
        x_eval = x_rgb_valid[eval_positions[: len(y_rows)]]
        y_eval = self.y_targets[y_rows]
        eval_positions = eval_positions[: len(y_rows)]
        return x_eval, y_eval, eval_positions

    def evaluate_rgb(self, rgb_uint8: np.ndarray) -> Dict[str, float]:
        metrics, _ = self.evaluate_rgb_with_details(rgb_uint8)
        return metrics

    def evaluate_rgb_with_details(self, rgb_uint8: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
        x_eval, y_eval, eval_positions = self._build_eval_arrays(rgb_uint8)

        kf = KFold(
            n_splits=int(self.cfg.cv_n_splits),
            shuffle=bool(self.cfg.cv_shuffle),
            random_state=int(self.cfg.cv_random_state),
        )

        comp_means = []
        y_pred_oof = np.full_like(y_eval, np.nan, dtype=np.float32)
        for comp_idx in range(self.n_components):
            y = y_eval[:, comp_idx]
            fold_r2, fold_mae, fold_rmse, fold_p, fold_s = [], [], [], [], []
            for train_idx, test_idx in kf.split(x_eval):
                model = _xgb_from_cfg(self.cfg)
                model.fit(x_eval[train_idx], y[train_idx])
                pred = model.predict(x_eval[test_idx])
                y_pred_oof[test_idx, comp_idx] = pred.astype(np.float32, copy=False)
                fold_r2.append(float(r2_score(y[test_idx], pred)))
                fold_mae.append(float(mean_absolute_error(y[test_idx], pred)))
                fold_rmse.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
                fold_p.append(_safe_pearson(y[test_idx], pred))
                fold_s.append(_safe_spearman(y[test_idx], pred))
            comp_means.append(
                {
                    "r2": float(np.mean(fold_r2)),
                    "mae": float(np.mean(fold_mae)),
                    "rmse": float(np.mean(fold_rmse)),
                    "pearson": float(np.mean(fold_p)),
                    "spearman": float(np.mean(fold_s)),
                }
            )

        mean_r2 = float(np.mean([m["r2"] for m in comp_means]))
        mean_mae = float(np.mean([m["mae"] for m in comp_means]))
        mean_rmse = float(np.mean([m["rmse"] for m in comp_means]))
        mean_pearson = float(np.mean([m["pearson"] for m in comp_means]))
        mean_spearman = float(np.mean([m["spearman"] for m in comp_means]))
        metrics = {
            "nmf_mean_r2": mean_r2,
            "nmf_mean_mae": mean_mae,
            "nmf_mean_rmse": mean_rmse,
            "nmf_mean_pearson": mean_pearson,
            "nmf_mean_spearman": mean_spearman,
            "nmf_components": self.n_components,
            "nmf_reconstruction_err": self.nmf_reconstruction_err,
        }
        details: Dict[str, Any] = {
            "eval_positions": eval_positions.astype(np.int64, copy=False),
            "y_true": y_eval.astype(np.float32, copy=False),
            "y_pred_oof": y_pred_oof,
        }
        return metrics, details

