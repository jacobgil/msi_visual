#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from msi_visual.metrics import MSIVisualizationMetrics
from sklearn.decomposition import NMF, PCA
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from skimage.metrics import structural_similarity as ssim

from scripts.benchmark_nmf_metrics import _sample_nmf_train_indices, _xgb_from_cfg

try:
    from scripts.debug_edge_maps import _landmark_knn_edge, _normalize_landmark_k_list
except ImportError:
    try:
        from debug_edge_maps import _landmark_knn_edge, _normalize_landmark_k_list
    except ImportError:
        _landmark_knn_edge = None

        def _normalize_landmark_k_list(raw: object) -> List[int]:
            default: List[int] = [15]
            if raw is None:
                return default
            if OmegaConf.is_config(raw):
                if OmegaConf.is_list(raw):
                    raw = list(raw)
                else:
                    try:
                        v = int(raw)
                        return [max(1, v)]
                    except (TypeError, ValueError):
                        return default
            elif not isinstance(raw, (list, tuple)):
                try:
                    v = int(raw)
                    return [max(1, v)]
                except (TypeError, ValueError):
                    return default
            out: List[int] = []
            for x in raw:
                try:
                    v = int(x)
                    if v > 0:
                        out.append(v)
                except (TypeError, ValueError):
                    continue
            out = sorted(set(out))
            return out if out else default

try:
    from scripts.debug_edge_maps import _highd_edge_maps_ensemble_cluster
except ImportError:
    try:
        from debug_edge_maps import _highd_edge_maps_ensemble_cluster
    except ImportError:
        _highd_edge_maps_ensemble_cluster = None

try:
    from scripts.debug_edge_maps import _highd_edge_maps_nmf_component_edges
except ImportError:
    try:
        from debug_edge_maps import _highd_edge_maps_nmf_component_edges
    except ImportError:
        _highd_edge_maps_nmf_component_edges = None

try:
    from scripts.debug_edge_maps import _gmm_kl_apply_optional_pca
except ImportError:
    try:
        from debug_edge_maps import _gmm_kl_apply_optional_pca
    except ImportError:
        _gmm_kl_apply_optional_pca = None

try:
    from scripts.debug_edge_maps import _apply_hd_edge_power
except ImportError:
    try:
        from debug_edge_maps import _apply_hd_edge_power
    except ImportError:
        _apply_hd_edge_power = None

try:
    from scripts.debug_edge_maps import (
        _apply_equalize_rgb_uint8,
        _compute_continuous_dice,
        _compute_soft_precision_recall,
        _compute_weighted_gradient_alignment,
        _edges_for_continuous_metrics,
        _build_npy_png_shape_matching_report,
        _filter_viz_paths_to_msi_shape,
        _pair_npy_png_paths_by_shape,
        _pick_msi_npy_for_shape_pairing,
        _pairing_unset_for_inference,
    )
except ImportError:
    from debug_edge_maps import (
        _apply_equalize_rgb_uint8,
        _compute_continuous_dice,
        _compute_soft_precision_recall,
        _compute_weighted_gradient_alignment,
        _edges_for_continuous_metrics,
        _build_npy_png_shape_matching_report,
        _filter_viz_paths_to_msi_shape,
        _pair_npy_png_paths_by_shape,
        _pick_msi_npy_for_shape_pairing,
        _pairing_unset_for_inference,
    )

logger = logging.getLogger(__name__)


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


def _percentile_normalize_vector(values: np.ndarray, low_pct: float, high_pct: float, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    lo = float(np.percentile(x, low_pct))
    hi = float(np.percentile(x, high_pct))
    if hi <= lo + eps:
        return np.zeros_like(x, dtype=np.float32)
    out = (x - lo) / (hi - lo + eps)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_targets_percentile(w: np.ndarray, mode: str, low_pct: float, high_pct: float) -> np.ndarray:
    if mode == "none":
        return np.asarray(w, dtype=np.float32)
    if mode == "minmax_per_component":
        out = np.zeros_like(w, dtype=np.float32)
        for j in range(w.shape[1]):
            out[:, j] = _percentile_normalize_vector(w[:, j], low_pct, high_pct)
        return out
    raise ValueError(f"Unsupported target normalization mode: {mode}")


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    sums = msi.sum(axis=-1, keepdims=True)
    sums = sums + 1e-8
    return msi / sums


def _spatial_tic_normalize(msi: np.ndarray) -> np.ndarray:
    """TIC per pixel, then channel-wise spatial scaling and clipping."""
    processed = _tic_normalize(msi)
    channel_max = np.percentile(processed, 100, axis=(0, 1), keepdims=True)
    processed = processed / (channel_max + 1e-8)
    return np.clip(processed, 0.0, 1.0).astype(np.float32, copy=False)


def _resolve_normalization_mode(value: object) -> str:
    """Normalize config values to one of: none, tic, spatial_tic."""
    if isinstance(value, bool):
        return "tic" if value else "none"
    s = str(value).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    if s in {"none", "off", "false", "0"}:
        return "none"
    if s in {"tic", "total_ion_count", "true", "1", "on"}:
        return "tic"
    if s in {"spatial_tic", "spatialtic", "sp_tic"}:
        return "spatial_tic"
    raise ValueError(
        f"Unknown normalization mode: {value!r}. Use one of: none, tic, spatial_tic."
    )


def _load_msi(path: Path, transpose_msi: bool, normalization: str = "tic") -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI shape HxWxC (or CxHxW with transpose), got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    norm_mode = _resolve_normalization_mode(normalization)
    if norm_mode == "spatial_tic":
        img = _spatial_tic_normalize(img)
    elif norm_mode == "tic":
        img = _tic_normalize(img)
    elif norm_mode == "none":
        pass
    else:
        raise ValueError(
            f"Unknown normalization mode after parsing: {normalization!r}. "
            "Use one of: none, tic, spatial_tic."
        )
    return img


def _to_uint8_rgb(viz: np.ndarray) -> np.ndarray:
    arr = np.asarray(viz)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr
    out = arr.astype(np.float32)
    if np.nanmax(out) <= 1.0:
        out = out * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_visualization(path: Path, target_hw: Tuple[int, int], allow_resize: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        viz = np.load(path, mmap_mode=None)
        viz_rgb = _to_uint8_rgb(viz)
    else:
        viz_rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = target_hw
    if viz_rgb.shape[:2] != (h, w):
        if not allow_resize:
            raise ValueError(
                f"Visualization shape mismatch for {path}: got {viz_rgb.shape[:2]}, expected {(h, w)}"
            )
        viz_rgb = cv2.resize(viz_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return viz_rgb


def _sobel_edge_energy(img2d: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    x = np.asarray(img2d, dtype=np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    vals = mag[mask]
    score = float(np.mean(vals)) if vals.size > 0 else 0.0
    return mag, score


def _normalize_edge_energy_by_component_power(
    edge_magnitude: np.ndarray,
    component_map: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    grad_vals = np.asarray(edge_magnitude, dtype=np.float32)[mask]
    comp_vals = np.asarray(component_map, dtype=np.float32)[mask]
    if grad_vals.size == 0 or comp_vals.size == 0:
        return 0.0
    grad_mean = float(np.mean(grad_vals))
    power_mean = float(np.mean(comp_vals * comp_vals))
    return grad_mean / (power_mean + eps)


def _normalize_channel_percentile(
    img2d: np.ndarray,
    valid_mask: np.ndarray,
    low_pct: float,
    high_pct: float,
    eps: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(img2d, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[valid_mask]
    if vals.size == 0:
        return out
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    if hi <= lo + eps:
        return out
    out = (x - lo) / (hi - lo + eps)
    out = np.clip(out, 0.0, 1.0)
    out[~valid_mask] = 0.0
    return out.astype(np.float32, copy=False)


def _highd_edge_maps_base(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    hd_cfg: object,
) -> Tuple[np.ndarray, np.ndarray]:
    normalize_channels = bool(getattr(hd_cfg, "normalize_channels", False)) if hd_cfg is not None else False
    channel_low = float(getattr(hd_cfg, "channel_low_percentile", 1.0)) if hd_cfg is not None else 1.0
    channel_high = float(getattr(hd_cfg, "channel_high_percentile", 99.0)) if hd_cfg is not None else 99.0
    use_log1p = bool(getattr(hd_cfg, "log1p", False)) if hd_cfg is not None else False
    agg_mode = str(getattr(hd_cfg, "aggregation", "mean")).strip().lower() if hd_cfg is not None else "mean"
    topk = int(getattr(hd_cfg, "topk", 5)) if hd_cfg is not None else 5

    c = int(msi.shape[2])
    edges = np.zeros((*msi.shape[:2], c), dtype=np.float32)
    for i in range(c):
        ch = np.asarray(msi[:, :, i], dtype=np.float32)
        if use_log1p:
            ch = np.log1p(np.clip(ch, 0.0, None)).astype(np.float32, copy=False)
        if normalize_channels:
            ch = _normalize_channel_percentile(ch, valid_mask, channel_low, channel_high)
        edges[:, :, i] = _sobel_edge_energy(ch, valid_mask)[0]

    if agg_mode == "max":
        out = np.max(edges, axis=2)
    elif agg_mode == "topk_mean":
        k = max(1, min(int(topk), c))
        top = np.partition(edges, kth=c - k, axis=2)[:, :, c - k :]
        out = np.mean(top, axis=2)
    else:
        out = np.mean(edges, axis=2)
    out_linf = np.max(edges, axis=2)
    out = np.asarray(out, dtype=np.float32)
    out_linf = np.asarray(out_linf, dtype=np.float32)
    out[~valid_mask] = 0.0
    out_linf[~valid_mask] = 0.0
    return out, out_linf


def _viz_edge_map(rgb_uint8: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edge, _ = _sobel_edge_energy(gray, valid_mask)
    edge[~valid_mask] = 0.0
    return edge


def _viz_edge_map_linf(rgb_uint8: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_uint8, dtype=np.float32) / 255.0
    h, w, _ = rgb.shape
    edge = np.zeros((h, w), dtype=np.float32)
    dx = np.max(np.abs(rgb[1:, :, :] - rgb[:-1, :, :]), axis=-1)
    dy = np.max(np.abs(rgb[:, 1:, :] - rgb[:, :-1, :]), axis=-1)
    edge[1:, :] += dx
    edge[:-1, :] += dx
    edge[:, 1:] += dy
    edge[:, :-1] += dy
    edge[~valid_mask] = 0.0
    return edge


def _normalize_on_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32).copy()
    vals = x[mask]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi > lo:
        x = (x - lo) / (hi - lo + 1e-8)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    x[~mask] = 0.0
    return x


def _parse_multiscale_edge_cfg(edge_cfg: object) -> Tuple[List[float], str]:
    if edge_cfg is None:
        return [0.0], "mean"
    ms_cfg = getattr(edge_cfg, "multi_scale", None)
    if ms_cfg is None:
        return [0.0], "mean"

    enabled = bool(getattr(ms_cfg, "enabled", False))
    aggregation = str(getattr(ms_cfg, "aggregation", "mean")).strip().lower()
    if aggregation not in {"mean", "max"}:
        aggregation = "mean"
    if not enabled:
        return [0.0], aggregation

    raw_sigmas = list(getattr(ms_cfg, "sigmas", [0.0, 1.0, 2.0]))
    cleaned: List[float] = []
    for s in raw_sigmas:
        try:
            sv = float(s)
        except Exception:
            continue
        if sv < 0:
            continue
        cleaned.append(sv)
    if not cleaned:
        cleaned = [0.0]
    return sorted(set(cleaned)), aggregation


_HD_METHODS_VALID = frozenset(
    {
        "gradient",
        "canny",
        "gmm_kl",
        "distance_multi",
        "landmark_knn",
        "soft_landmark_contrast",
        "ensemble_cluster",
        "nmf_component_edges",
    }
)


def normalize_hd_methods_list(edge_cfg: Any) -> List[str]:
    """
    Parse ``edge_detection.hd_method`` as a single string or a list of strings.
    Unknown entries are dropped; if nothing valid remains, ``["gradient"]`` is used.
    """
    if edge_cfg is None:
        return ["gradient"]
    raw = getattr(edge_cfg, "hd_method", None)
    if raw is None:
        return ["gradient"]
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip().lower() for x in raw]
    elif isinstance(raw, ListConfig):
        items = [str(x).strip().lower() for x in list(raw)]
    else:
        items = [str(raw).strip().lower()]
    valid = [m for m in items if m in _HD_METHODS_VALID]
    valid = list(dict.fromkeys(valid))
    return valid if valid else ["gradient"]


def _parse_edge_detection_cfg(cfg: DictConfig) -> Dict[str, object]:
    edge_cfg = getattr(cfg, "edge_detection", None)
    _hd_methods = normalize_hd_methods_list(edge_cfg)
    canny_cfg = getattr(edge_cfg, "canny", None) if edge_cfg is not None else None
    gmm_cfg = getattr(edge_cfg, "gmm", None) if edge_cfg is not None else None
    out = {
        "hd_method": _hd_methods[0],
        "rgb_method": str(getattr(edge_cfg, "rgb_method", "gradient")).strip().lower() if edge_cfg is not None else "gradient",
        "canny_low_threshold": float(getattr(canny_cfg, "low_threshold", 40.0)) if canny_cfg is not None else 40.0,
        "canny_high_threshold": float(getattr(canny_cfg, "high_threshold", 120.0)) if canny_cfg is not None else 120.0,
        "canny_aperture_size": int(getattr(canny_cfg, "aperture_size", 3)) if canny_cfg is not None else 3,
        "canny_l2gradient": bool(getattr(canny_cfg, "l2gradient", True)) if canny_cfg is not None else True,
        "canny_n_jobs": int(getattr(canny_cfg, "n_jobs", 0)) if canny_cfg is not None else 0,
        "canny_rgb_mode": str(getattr(canny_cfg, "rgb_mode", "max_channel")).strip().lower() if canny_cfg is not None else "max_channel",
        "gmm_n_components": int(getattr(gmm_cfg, "n_components", 12)) if gmm_cfg is not None else 12,
        "gmm_covariance_type": str(getattr(gmm_cfg, "covariance_type", "diag")).strip().lower() if gmm_cfg is not None else "diag",
        "gmm_reg_covar": float(getattr(gmm_cfg, "reg_covar", 1e-6)) if gmm_cfg is not None else 1e-6,
        "gmm_max_iter": int(getattr(gmm_cfg, "max_iter", 200)) if gmm_cfg is not None else 200,
        "gmm_random_state": int(getattr(gmm_cfg, "random_state", 42)) if gmm_cfg is not None else 42,
        "gmm_max_samples": int(getattr(gmm_cfg, "max_samples", 200000)) if gmm_cfg is not None else 200000,
        "gmm_pca_enabled": bool(getattr(gmm_cfg, "pca_enabled", False)) if gmm_cfg is not None else False,
        "gmm_pca_n_components": int(getattr(gmm_cfg, "pca_n_components", 64)) if gmm_cfg is not None else 64,
        "gmm_pca_random_state": int(getattr(gmm_cfg, "pca_random_state", 42)) if gmm_cfg is not None else 42,
    }
    lk_cfg = None
    if edge_cfg is not None:
        lk_cfg = getattr(edge_cfg, "soft_landmark_contrast", None)
        if lk_cfg is None:
            lk_cfg = getattr(edge_cfg, "landmark_knn", None)
    raw_lk_k = getattr(lk_cfg, "k", 15) if lk_cfg is not None else 15
    lk_ks = _normalize_landmark_k_list(raw_lk_k)
    out["lk_k_list"] = lk_ks
    out["lk_k"] = int(lk_ks[0])
    out["lk_n_landmarks"] = int(getattr(lk_cfg, "n_landmarks", 1000)) if lk_cfg is not None else 1000
    out["lk_spatial_radii"] = list(getattr(lk_cfg, "spatial_radii", [1])) if lk_cfg is not None else [1]
    out["lk_spatial_connectivity"] = str(getattr(lk_cfg, "spatial_connectivity", "4")).strip().lower() if lk_cfg is not None else "4"
    out["lk_random_state"] = int(getattr(lk_cfg, "random_state", 42)) if lk_cfg is not None else 42
    out["lk_aggregation"] = str(getattr(lk_cfg, "aggregation", "max")).strip().lower() if lk_cfg is not None else "max"
    out["lk_pca_enabled"] = bool(getattr(lk_cfg, "pca_enabled", True)) if lk_cfg is not None else True
    out["lk_pca_n_components"] = int(getattr(lk_cfg, "pca_n_components", 64)) if lk_cfg is not None else 64
    out["lk_score_mode"] = str(getattr(lk_cfg, "score_mode", "neighbor")).strip().lower() if lk_cfg is not None else "neighbor"
    out["lk_contrast_texture_weight"] = float(getattr(lk_cfg, "contrast_texture_weight", 0.75)) if lk_cfg is not None else 0.75
    out["lk_contrast_aggregation"] = str(getattr(lk_cfg, "contrast_aggregation", "max")).strip().lower() if lk_cfg is not None else "max"
    out["lk_contrast_batch_size"] = int(getattr(lk_cfg, "contrast_batch_size", 8192)) if lk_cfg is not None else 8192
    out["lk_soft_topk"] = int(getattr(lk_cfg, "soft_topk", 128)) if lk_cfg is not None else 128
    out["lk_soft_temperature"] = float(getattr(lk_cfg, "soft_temperature", 0.08)) if lk_cfg is not None else 0.08
    out["lk_soft_patch_radius"] = int(getattr(lk_cfg, "soft_patch_radius", 1)) if lk_cfg is not None else 1
    out["lk_soft_scale_aggregation"] = str(getattr(lk_cfg, "soft_scale_aggregation", "mean")).strip().lower() if lk_cfg is not None else "mean"
    out["lk_soft_chunk_size"] = int(getattr(lk_cfg, "soft_chunk_size", 64)) if lk_cfg is not None else 64
    if out["hd_method"] not in {
        "gradient",
        "canny",
        "gmm_kl",
        "distance_multi",
        "landmark_knn",
        "soft_landmark_contrast",
        "ensemble_cluster",
        "nmf_component_edges",
    }:
        out["hd_method"] = "gradient"
    if out["rgb_method"] not in {"gradient", "canny"}:
        out["rgb_method"] = "gradient"
    if out["canny_aperture_size"] not in {3, 5, 7}:
        out["canny_aperture_size"] = 3
    if out["canny_rgb_mode"] not in {"gray", "max_channel", "mean_channel", "any_channel"}:
        out["canny_rgb_mode"] = "max_channel"
    if out["gmm_covariance_type"] not in {"full", "tied", "diag", "spherical"}:
        out["gmm_covariance_type"] = "diag"
    out["gmm_n_components"] = max(2, int(out["gmm_n_components"]))
    out["gmm_max_iter"] = max(10, int(out["gmm_max_iter"]))
    out["gmm_max_samples"] = max(0, int(out["gmm_max_samples"]))
    out["gmm_pca_n_components"] = max(1, int(out["gmm_pca_n_components"]))
    return out


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(img, dtype=np.float32)
    x = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
    return cv2.GaussianBlur(x, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def _to_u8_from01(x01: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x01, dtype=np.float32), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _aggregate_channel_stack(stack: np.ndarray, mode: str, topk: int) -> np.ndarray:
    if stack.ndim != 3 or stack.shape[2] == 0:
        return np.zeros(stack.shape[:2], dtype=np.float32)
    m = str(mode).strip().lower()
    if m == "max":
        return np.max(stack, axis=2).astype(np.float32, copy=False)
    if m == "topk_mean":
        c = int(stack.shape[2])
        k = max(1, min(int(topk), c))
        top = np.partition(stack, kth=c - k, axis=2)[:, :, c - k :]
        return np.mean(top, axis=2).astype(np.float32, copy=False)
    return np.mean(stack, axis=2).astype(np.float32, copy=False)


def _canny_edge_from_u8(
    img_u8: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    aperture_size: int,
    l2gradient: bool,
) -> np.ndarray:
    e = cv2.Canny(
        img_u8,
        threshold1=float(low_threshold),
        threshold2=float(high_threshold),
        apertureSize=int(aperture_size),
        L2gradient=bool(l2gradient),
    )
    return (e.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def _canny_stack_from_channels(
    channels01: np.ndarray,
    sigma: float,
    low_threshold: float,
    high_threshold: float,
    aperture_size: int,
    l2gradient: bool,
    n_jobs: int,
) -> np.ndarray:
    h, w, c = channels01.shape
    stack = np.zeros((h, w, c), dtype=np.float32)
    sigma_f = float(sigma)

    def _one_channel(i: int) -> Tuple[int, np.ndarray]:
        ch = np.asarray(channels01[:, :, i], dtype=np.float32)
        if sigma_f > 0:
            ch = _gaussian_blur(ch, sigma_f)
        ch_u8 = _to_u8_from01(ch)
        return i, _canny_edge_from_u8(ch_u8, low_threshold, high_threshold, aperture_size, l2gradient)

    nj = int(n_jobs)
    if nj <= 0:
        nj = max(1, (os.cpu_count() or 1))
    nj = min(nj, c)
    if nj <= 1 or c <= 2:
        for i in range(c):
            _, e = _one_channel(i)
            stack[:, :, i] = e
        return stack

    with ThreadPoolExecutor(max_workers=nj) as ex:
        for i, e in ex.map(_one_channel, range(c)):
            stack[:, :, i] = e
    return stack


def _neighbor_dist(a: np.ndarray, b: np.ndarray, metric: str, eps: float = 1e-8) -> np.ndarray:
    """Compute distance between pixel spectral vectors a and b (HxWxC)."""
    metric = str(metric).strip().lower()
    if metric == "l2":
        return np.linalg.norm(a - b, axis=-1).astype(np.float32, copy=False)
    if metric == "sam":
        dot = np.sum(a * b, axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cos = np.clip(dot / (na * nb + eps), -1.0, 1.0)
        return (np.arccos(cos) / np.pi).astype(np.float32, copy=False)
    if metric == "cosine":
        dot = np.sum(a * b, axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cos = np.clip(dot / (na * nb + eps), -1.0, 1.0)
        return (1.0 - cos).astype(np.float32, copy=False)
    if metric == "chebyshev":
        return np.max(np.abs(a - b), axis=-1).astype(np.float32, copy=False)
    if metric == "manhattan":
        return np.sum(np.abs(a - b), axis=-1).astype(np.float32, copy=False)
    if metric == "braycurtis":
        diff = np.abs(a - b)
        total = np.abs(a) + np.abs(b) + eps
        return np.sum(diff, axis=-1) / np.sum(total, axis=-1)
    return np.linalg.norm(a - b, axis=-1).astype(np.float32, copy=False)


def _compute_neighborhood_edge_map(
    channels: np.ndarray,
    valid_mask: np.ndarray,
    metric: str,
    radius: int = 1,
    sigma: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Edge map from max spatial-neighbor distance per pixel."""
    h, w, _ = channels.shape
    r = max(1, int(radius))
    edge_max = np.zeros((h, w), dtype=np.float32)
    for dy in range(0, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx <= 0:
                continue
            if dy == 0 and dx == 0:
                continue
            if dy > 0 or (dy == 0 and dx > 0):
                y0a, y1a = dy, h
                y0b, y1b = 0, h - dy
                if dx >= 0:
                    x0a, x1a = dx, w
                    x0b, x1b = 0, w - dx
                else:
                    x0a, x1a = 0, w + dx
                    x0b, x1b = -dx, w
                a = channels[y0a:y1a, x0a:x1a, :]
                b = channels[y0b:y1b, x0b:x1b, :]
                m = valid_mask[y0a:y1a, x0a:x1a] & valid_mask[y0b:y1b, x0b:x1b]
                if not np.any(m):
                    continue
                dist = _neighbor_dist(a, b, metric, eps)
                if sigma > 0:
                    w_sp = float(np.exp(-((dx * dx + dy * dy) / (2.0 * sigma * sigma))))
                else:
                    w_sp = 1.0
                d = np.where(m, dist * w_sp, 0.0).astype(np.float32, copy=False)
                edge_max[y0a:y1a, x0a:x1a] = np.maximum(edge_max[y0a:y1a, x0a:x1a], d)
                edge_max[y0b:y1b, x0b:x1b] = np.maximum(edge_max[y0b:y1b, x0b:x1b], d)
    edge_max[~valid_mask] = 0.0
    return edge_max


def _zscore_map(arr: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score on valid pixels; invalid pixels set to 0."""
    out = np.zeros_like(arr, dtype=np.float32)
    vals = arr[mask]
    if vals.size == 0:
        return out
    mu = float(np.mean(vals))
    std = float(np.std(vals))
    if std < eps:
        return out
    out = (arr - mu) / (std + eps)
    out[~mask] = 0.0
    return out.astype(np.float32, copy=False)


def _highd_edge_maps_distance_multi(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    hd_cfg: object,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute edge maps with multiple distance functions, z-score each, mean-aggregate."""
    distances = list(getattr(hd_cfg, "distance_functions", ["l2", "sam", "cosine", "chebyshev", "manhattan"]) or ["l2"])
    nb_cfg = getattr(hd_cfg, "neighborhood", None) or {}
    radius = int(getattr(nb_cfg, "radius", 1))
    sigma = float(getattr(nb_cfg, "sigma", 1.0))
    normalize_channels = bool(getattr(hd_cfg, "normalize_channels", False))
    channel_low = float(getattr(hd_cfg, "channel_low_percentile", 1.0))
    channel_high = float(getattr(hd_cfg, "channel_high_percentile", 99.0))
    use_log1p = bool(getattr(hd_cfg, "log1p", False))

    h, w, c = msi.shape
    channels = np.zeros((h, w, c), dtype=np.float32)
    for i in range(c):
        ch = np.asarray(msi[:, :, i], dtype=np.float32)
        if use_log1p:
            ch = np.log1p(np.clip(ch, 0.0, None)).astype(np.float32, copy=False)
        if normalize_channels:
            ch = _normalize_channel_percentile(ch, valid_mask, channel_low, channel_high)
        channels[:, :, i] = ch

    edge_maps: List[np.ndarray] = []
    for metric in distances:
        m = str(metric).strip().lower()
        em = _compute_neighborhood_edge_map(channels, valid_mask, m, radius=radius, sigma=sigma)
        z = _zscore_map(em, valid_mask)
        if m == "chebyshev" and np.any(valid_mask):
            em_v = em[valid_mask]
            z_v = z[valid_mask]
            print(
                f"[chebyshev] raw neighbor-distance (channel-norm spectra) | mean={float(np.mean(em_v)):.6g} "
                f"max={float(np.max(em_v)):.6g}",
                flush=True,
            )
            print(
                f"[chebyshev] after z-score | mean={float(np.mean(z_v)):.6g} max={float(np.max(z_v)):.6g}",
                flush=True,
            )
        edge_maps.append(z)

    if not edge_maps:
        out = np.zeros(valid_mask.shape, dtype=np.float32)
        out[~valid_mask] = 0.0
        return out, out.copy()

    out = np.mean(np.stack(edge_maps, axis=0), axis=0).astype(np.float32, copy=False)
    out[~valid_mask] = 0.0
    return out, out.copy()


def _aggregate_multiscale_maps(maps: List[np.ndarray], valid_mask: np.ndarray, mode: str) -> np.ndarray:
    if not maps:
        out = np.zeros(valid_mask.shape, dtype=np.float32)
        out[~valid_mask] = 0.0
        return out
    if len(maps) == 1:
        out = np.asarray(maps[0], dtype=np.float32)
        out[~valid_mask] = 0.0
        return out

    # Normalize each scale on the valid region so aggregation is not dominated by one scale's magnitude.
    normalized = [_normalize_on_mask(m, valid_mask) for m in maps]
    stack = np.stack(normalized, axis=0)
    if str(mode).strip().lower() == "max":
        out = np.max(stack, axis=0)
    else:
        out = np.mean(stack, axis=0)
    out = np.asarray(out, dtype=np.float32)
    out[~valid_mask] = 0.0
    return out


def _highd_edge_maps_multiscale(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    sigmas: List[float],
    aggregation: str,
    hd_cfg: object = None,
    edge_method_cfg: Dict[str, object] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    edge_method_cfg = edge_method_cfg or {}
    method = str(edge_method_cfg.get("hd_method", "gradient")).strip().lower()
    edge_maps: List[np.ndarray] = []
    edge_linf_maps: List[np.ndarray] = []
    if method == "gmm_kl":
        x = np.asarray(msi[valid_mask], dtype=np.float32)
        if x.size == 0:
            base_edge = np.zeros(valid_mask.shape, dtype=np.float32)
            base_edge_linf = np.zeros(valid_mask.shape, dtype=np.float32)
        else:
            max_samples = int(edge_method_cfg.get("gmm_max_samples", 0))
            fit_x = x
            if 0 < max_samples < x.shape[0]:
                rng = np.random.default_rng(int(edge_method_cfg.get("gmm_random_state", 42)))
                idx = rng.choice(x.shape[0], size=max_samples, replace=False)
                fit_x = x[idx]
            if _gmm_kl_apply_optional_pca is not None:
                x, fit_x = _gmm_kl_apply_optional_pca(x, fit_x, edge_method_cfg)
            n_components = min(int(edge_method_cfg.get("gmm_n_components", 12)), fit_x.shape[0])
            gmm = GaussianMixture(
                n_components=max(2, n_components),
                covariance_type=str(edge_method_cfg.get("gmm_covariance_type", "diag")),
                reg_covar=float(edge_method_cfg.get("gmm_reg_covar", 1e-6)),
                max_iter=int(edge_method_cfg.get("gmm_max_iter", 200)),
                random_state=int(edge_method_cfg.get("gmm_random_state", 42)),
            )
            gmm.fit(fit_x)
            probs = gmm.predict_proba(x).astype(np.float32, copy=False)
            eps = 1e-8
            p_map = np.zeros((valid_mask.shape[0], valid_mask.shape[1], probs.shape[1]), dtype=np.float32)
            p_map[valid_mask] = probs

            def _sym_kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                aa = np.clip(a, eps, 1.0)
                bb = np.clip(b, eps, 1.0)
                return 0.5 * (
                    np.sum(aa * (np.log(aa) - np.log(bb)), axis=-1)
                    + np.sum(bb * (np.log(bb) - np.log(aa)), axis=-1)
                )

            h, w, _ = p_map.shape
            sum_edge = np.zeros((h, w), dtype=np.float32)
            max_edge = np.zeros((h, w), dtype=np.float32)

            m_v = valid_mask[:-1, :] & valid_mask[1:, :]
            if np.any(m_v):
                d_v = _sym_kl(p_map[:-1, :, :], p_map[1:, :, :]).astype(np.float32, copy=False)
                d_v = np.where(m_v, d_v, 0.0)
                sum_edge[:-1, :] += d_v
                sum_edge[1:, :] += d_v
                max_edge[:-1, :] = np.maximum(max_edge[:-1, :], d_v)
                max_edge[1:, :] = np.maximum(max_edge[1:, :], d_v)

            m_h = valid_mask[:, :-1] & valid_mask[:, 1:]
            if np.any(m_h):
                d_h = _sym_kl(p_map[:, :-1, :], p_map[:, 1:, :]).astype(np.float32, copy=False)
                d_h = np.where(m_h, d_h, 0.0)
                sum_edge[:, :-1] += d_h
                sum_edge[:, 1:] += d_h
                max_edge[:, :-1] = np.maximum(max_edge[:, :-1], d_h)
                max_edge[:, 1:] = np.maximum(max_edge[:, 1:], d_h)

            base_edge = sum_edge.astype(np.float32, copy=False)
            base_edge_linf = max_edge.astype(np.float32, copy=False)
            base_edge[~valid_mask] = 0.0
            base_edge_linf[~valid_mask] = 0.0
        for sigma in sigmas:
            edge_maps.append(_gaussian_blur(base_edge, float(sigma)))
            edge_linf_maps.append(_gaussian_blur(base_edge_linf, float(sigma)))
    elif method == "distance_multi":
        base_edge, base_edge_linf = _highd_edge_maps_distance_multi(msi, valid_mask, hd_cfg)
        for sigma in sigmas:
            edge_maps.append(_gaussian_blur(base_edge, float(sigma)))
            edge_linf_maps.append(_gaussian_blur(base_edge_linf, float(sigma)))
    elif method in {"landmark_knn", "soft_landmark_contrast"} and _landmark_knn_edge is not None:
        lk_radii = list(edge_method_cfg.get("lk_spatial_radii", [1]))
        lk_ks = edge_method_cfg.get("lk_k_list")
        if lk_ks is None:
            lk_ks = _normalize_landmark_k_list(edge_method_cfg.get("lk_k", 15))
        base_edge, base_edge_linf = _landmark_knn_edge(
            msi,
            valid_mask,
            k=lk_ks,
            n_landmarks=int(edge_method_cfg.get("lk_n_landmarks", 1000)),
            spatial_radii=[int(r) for r in lk_radii] if lk_radii else None,
            spatial_connectivity=str(edge_method_cfg.get("lk_spatial_connectivity", "4")).strip().lower(),
            random_state=int(edge_method_cfg.get("lk_random_state", 42)),
            aggregation=str(edge_method_cfg.get("lk_aggregation", "max")).strip().lower(),
            pca_enabled=bool(edge_method_cfg.get("lk_pca_enabled", True)),
            pca_n_components=int(edge_method_cfg.get("lk_pca_n_components", 64)),
            score_mode=str(edge_method_cfg.get("lk_score_mode", "neighbor")).strip().lower(),
            contrast_texture_weight=float(edge_method_cfg.get("lk_contrast_texture_weight", 0.75)),
            contrast_aggregation=str(edge_method_cfg.get("lk_contrast_aggregation", "max")).strip().lower(),
            contrast_batch_size=int(edge_method_cfg.get("lk_contrast_batch_size", 8192)),
            soft_topk=int(edge_method_cfg.get("lk_soft_topk", 128)),
            soft_temperature=float(edge_method_cfg.get("lk_soft_temperature", 0.08)),
            soft_patch_radius=int(edge_method_cfg.get("lk_soft_patch_radius", 1)),
            soft_scale_aggregation=str(edge_method_cfg.get("lk_soft_scale_aggregation", "mean")).strip().lower(),
            soft_chunk_size=int(edge_method_cfg.get("lk_soft_chunk_size", 64)),
        )
        for sigma in sigmas:
            edge_maps.append(_gaussian_blur(base_edge, float(sigma)))
            edge_linf_maps.append(_gaussian_blur(base_edge_linf, float(sigma)))
    elif method == "ensemble_cluster" and _highd_edge_maps_ensemble_cluster is not None:
        base_edge, base_edge_linf = _highd_edge_maps_ensemble_cluster(msi, valid_mask, hd_cfg)
        for sigma in sigmas:
            edge_maps.append(_gaussian_blur(base_edge, float(sigma)))
            edge_linf_maps.append(_gaussian_blur(base_edge_linf, float(sigma)))
    elif method == "nmf_component_edges" and _highd_edge_maps_nmf_component_edges is not None:
        base_edge, base_edge_linf = _highd_edge_maps_nmf_component_edges(msi, valid_mask, hd_cfg)
        for sigma in sigmas:
            edge_maps.append(_gaussian_blur(base_edge, float(sigma)))
            edge_linf_maps.append(_gaussian_blur(base_edge_linf, float(sigma)))
    elif method == "canny":
        normalize_channels = bool(getattr(hd_cfg, "normalize_channels", False)) if hd_cfg is not None else False
        channel_low = float(getattr(hd_cfg, "channel_low_percentile", 1.0)) if hd_cfg is not None else 1.0
        channel_high = float(getattr(hd_cfg, "channel_high_percentile", 99.0)) if hd_cfg is not None else 99.0
        use_log1p = bool(getattr(hd_cfg, "log1p", False)) if hd_cfg is not None else False
        channel_agg = str(getattr(hd_cfg, "aggregation", "mean")).strip().lower() if hd_cfg is not None else "mean"
        topk = int(getattr(hd_cfg, "topk", 5)) if hd_cfg is not None else 5
        msi_proc = np.asarray(msi, dtype=np.float32)
        channels = np.zeros_like(msi_proc, dtype=np.float32)
        for i in range(msi_proc.shape[2]):
            ch = np.asarray(msi_proc[:, :, i], dtype=np.float32)
            if use_log1p:
                ch = np.log1p(np.clip(ch, 0.0, None)).astype(np.float32, copy=False)
            if normalize_channels:
                ch = _normalize_channel_percentile(ch, valid_mask, channel_low, channel_high)
            else:
                # Keep Canny robust when upstream normalization (e.g. TIC) yields tiny ranges.
                ch = _normalize_on_mask(ch, valid_mask)
            channels[:, :, i] = ch
        for sigma in sigmas:
            canny_stack = _canny_stack_from_channels(
                channels01=channels,
                sigma=float(sigma),
                low_threshold=float(edge_method_cfg.get("canny_low_threshold", 40.0)),
                high_threshold=float(edge_method_cfg.get("canny_high_threshold", 120.0)),
                aperture_size=int(edge_method_cfg.get("canny_aperture_size", 3)),
                l2gradient=bool(edge_method_cfg.get("canny_l2gradient", True)),
                n_jobs=int(edge_method_cfg.get("canny_n_jobs", 0)),
            )
            edge_maps.append(_aggregate_channel_stack(canny_stack, mode=channel_agg, topk=topk))
            edge_linf_maps.append(np.max(canny_stack, axis=2).astype(np.float32, copy=False))
    else:
        # Compute high-D gradient edges once, then smooth the 2D edge maps per scale.
        base_edge, base_edge_linf = _highd_edge_maps_base(msi, valid_mask, hd_cfg=hd_cfg)
        for sigma in sigmas:
            edge_maps.append(_gaussian_blur(base_edge, float(sigma)))
            edge_linf_maps.append(_gaussian_blur(base_edge_linf, float(sigma)))
    merged = _aggregate_multiscale_maps(edge_maps, valid_mask, aggregation)
    merged_linf = _aggregate_multiscale_maps(edge_linf_maps, valid_mask, aggregation)
    if _apply_hd_edge_power is not None:
        merged = _apply_hd_edge_power(merged, valid_mask, hd_cfg)
        merged_linf = _apply_hd_edge_power(merged_linf, valid_mask, hd_cfg)
    return merged, merged_linf


def _viz_edge_maps_multiscale(
    rgb_uint8: np.ndarray,
    valid_mask: np.ndarray,
    sigmas: List[float],
    aggregation: str,
    edge_method_cfg: Dict[str, object] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    edge_method_cfg = edge_method_cfg or {}
    method = str(edge_method_cfg.get("rgb_method", "gradient")).strip().lower()
    rgb = np.asarray(rgb_uint8, dtype=np.float32)
    edge_maps: List[np.ndarray] = []
    edge_linf_maps: List[np.ndarray] = []
    if method == "canny":
        rgb_mode = str(edge_method_cfg.get("canny_rgb_mode", "max_channel")).strip().lower()
        for sigma in sigmas:
            rgb_s = _gaussian_blur(rgb, float(sigma))
            rgb_s01 = np.clip(rgb_s / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
            if rgb_mode == "gray":
                gray = cv2.cvtColor(np.clip(rgb_s, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                e = _canny_edge_from_u8(
                    _to_u8_from01(gray),
                    low_threshold=float(edge_method_cfg.get("canny_low_threshold", 40.0)),
                    high_threshold=float(edge_method_cfg.get("canny_high_threshold", 120.0)),
                    aperture_size=int(edge_method_cfg.get("canny_aperture_size", 3)),
                    l2gradient=bool(edge_method_cfg.get("canny_l2gradient", True)),
                )
                edge_maps.append(e)
                edge_linf_maps.append(e)
            else:
                stack = _canny_stack_from_channels(
                    channels01=rgb_s01,
                    sigma=0.0,
                    low_threshold=float(edge_method_cfg.get("canny_low_threshold", 40.0)),
                    high_threshold=float(edge_method_cfg.get("canny_high_threshold", 120.0)),
                    aperture_size=int(edge_method_cfg.get("canny_aperture_size", 3)),
                    l2gradient=bool(edge_method_cfg.get("canny_l2gradient", True)),
                    n_jobs=int(edge_method_cfg.get("canny_n_jobs", 0)),
                )
                if rgb_mode == "mean_channel":
                    e = np.mean(stack, axis=2).astype(np.float32, copy=False)
                elif rgb_mode == "any_channel":
                    e = (np.max(stack, axis=2) > 0).astype(np.float32)
                else:
                    e = np.max(stack, axis=2).astype(np.float32, copy=False)
                edge_maps.append(e)
                edge_linf_maps.append(np.max(stack, axis=2).astype(np.float32, copy=False))
    else:
        for sigma in sigmas:
            rgb_s = _gaussian_blur(rgb, float(sigma))
            rgb_s_u8 = np.clip(rgb_s, 0, 255).astype(np.uint8)
            edge_maps.append(_viz_edge_map(rgb_s_u8, valid_mask))
            edge_linf_maps.append(_viz_edge_map_linf(rgb_s_u8, valid_mask))
    return (
        _aggregate_multiscale_maps(edge_maps, valid_mask, aggregation),
        _aggregate_multiscale_maps(edge_linf_maps, valid_mask, aggregation),
    )


def _normalize_map_percentile(arr: np.ndarray, mask: np.ndarray, low_pct: float, high_pct: float, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32).copy()
    vals = x[mask]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    if hi <= lo + eps:
        x = np.zeros_like(x, dtype=np.float32)
    else:
        x = (x - lo) / (hi - lo + eps)
        x = np.clip(x, 0.0, 1.0)
    x[~mask] = 0.0
    return x


def _apply_equalize_gray01(
    arr01: np.ndarray,
    mask: np.ndarray,
    method: str,
    clip_limit: float,
    tile_grid_size: int,
) -> np.ndarray:
    x = np.asarray(arr01, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    u8 = (x * 255.0).astype(np.uint8)
    m = str(method).strip().lower()
    if m == "clahe":
        tgs = max(2, int(tile_grid_size))
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tgs, tgs))
        out_u8 = clahe.apply(u8)
    else:
        out_u8 = cv2.equalizeHist(u8)
    out = (out_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False)
    out[~mask] = 0.0
    return out


def _clahe_params(cfg: DictConfig) -> Tuple[bool, str, float, int, Dict[str, bool]]:
    clahe_cfg = getattr(cfg, "clahe", None)
    enabled = bool(getattr(clahe_cfg, "enabled", False)) if clahe_cfg is not None else False
    method = str(getattr(clahe_cfg, "method", "clahe")).strip().lower() if clahe_cfg is not None else "clahe"
    if method not in {"clahe", "hist"}:
        method = "clahe"
    clip_limit = float(getattr(clahe_cfg, "clip_limit", 2.0)) if clahe_cfg is not None else 2.0
    tile_grid_size = int(getattr(clahe_cfg, "tile_grid_size", 8)) if clahe_cfg is not None else 8
    apply_to_cfg = getattr(clahe_cfg, "apply_to", None) if clahe_cfg is not None else None
    apply_to = {
        "nmf_rgb_features": bool(getattr(apply_to_cfg, "nmf_rgb_features", False)) if apply_to_cfg is not None else False,
        "nmf_component_maps": bool(getattr(apply_to_cfg, "nmf_component_maps", False)) if apply_to_cfg is not None else False,
        "edge_viz": bool(getattr(apply_to_cfg, "edge_viz", False)) if apply_to_cfg is not None else False,
        "edge_hd": bool(getattr(apply_to_cfg, "edge_hd", False)) if apply_to_cfg is not None else False,
    }
    return enabled, method, clip_limit, tile_grid_size, apply_to


def _edge_auc_metrics(
    highd_edge: np.ndarray,
    viz_edge: np.ndarray,
    valid_mask: np.ndarray,
    positive_percentile: float,
    min_positive_pixels: int,
) -> Dict[str, float]:
    y_true = np.zeros(valid_mask.shape, dtype=np.uint8)
    hd_vals = highd_edge[valid_mask]
    if hd_vals.size == 0:
        return {"edge_agreement_auc_roc": float("nan"), "edge_agreement_auc_pr": float("nan")}
    thr = float(np.percentile(hd_vals, positive_percentile))
    y_true[valid_mask] = (highd_edge[valid_mask] >= thr).astype(np.uint8)
    y = y_true[valid_mask].astype(np.int32)
    scores = viz_edge[valid_mask].astype(np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos < max(1, int(min_positive_pixels)) or n_neg < 1:
        return {"edge_agreement_auc_roc": float("nan"), "edge_agreement_auc_pr": float("nan")}
    try:
        auc_roc = float(roc_auc_score(y, scores))
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(y, scores))
    except Exception:
        auc_pr = float("nan")
    return {"edge_agreement_auc_roc": auc_roc, "edge_agreement_auc_pr": auc_pr}


def _weighted_nanmean(values: List[float], weights: List[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        if not np.isfinite(v) or not np.isfinite(w) or w <= 0:
            continue
        num += float(v) * float(w)
        den += float(w)
    if den <= 0:
        return float("nan")
    return float(num / den)


def _edge_auc_metrics_multi_percentile(
    highd_edge: np.ndarray,
    viz_edge: np.ndarray,
    valid_mask: np.ndarray,
    percentiles: List[float],
    min_positive_pixels: int,
    weight_mode: str,
    tail_power: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cleaned: List[float] = []
    for p in percentiles:
        try:
            pv = float(p)
        except Exception:
            continue
        pv = min(max(pv, 0.0), 100.0)
        cleaned.append(pv)
    if not cleaned:
        return {
            "edge_agreement_auc_roc_weighted": float("nan"),
            "edge_agreement_auc_pr_weighted": float("nan"),
            "edge_agreement_auc_multi_valid_percentiles": 0.0,
        }

    unique_percentiles = sorted(set(cleaned))
    roc_vals: List[float] = []
    pr_vals: List[float] = []
    weights: List[float] = []
    valid_count = 0

    mode = str(weight_mode).strip().lower()
    for p in unique_percentiles:
        metrics_p = _edge_auc_metrics(
            highd_edge=highd_edge,
            viz_edge=viz_edge,
            valid_mask=valid_mask,
            positive_percentile=float(p),
            min_positive_pixels=int(min_positive_pixels),
        )
        p_tag = str(p).replace(".", "p")
        roc_p = float(metrics_p["edge_agreement_auc_roc"])
        pr_p = float(metrics_p["edge_agreement_auc_pr"])
        out[f"edge_agreement_auc_roc_p{p_tag}"] = roc_p
        out[f"edge_agreement_auc_pr_p{p_tag}"] = pr_p
        roc_vals.append(roc_p)
        pr_vals.append(pr_p)

        if mode == "uniform":
            w = 1.0
        elif mode == "tail_linear":
            w = max(float(p) / 100.0, 1e-6)
        else:
            # Default: emphasize sharper, high-percentile edges.
            w = max(float(p) / 100.0, 1e-6) ** float(tail_power)
        weights.append(float(w))
        if np.isfinite(roc_p) or np.isfinite(pr_p):
            valid_count += 1

    out["edge_agreement_auc_roc_weighted"] = _weighted_nanmean(roc_vals, weights)
    out["edge_agreement_auc_pr_weighted"] = _weighted_nanmean(pr_vals, weights)
    out["edge_agreement_auc_multi_valid_percentiles"] = float(valid_count)
    return out


def _compute_prf1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    f_beta: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Return (precision, recall, f1, f_beta). f_beta=1 gives F1; >1 emphasizes recall; <1 emphasizes precision."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    fn = int(np.sum(yt & (~yp)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    if (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0
    if abs(f_beta - 1.0) < 1e-9:
        f_b = f1
    elif (precision + recall) > 0:
        b2 = float(f_beta) ** 2
        f_b = float((1.0 + b2) * precision * recall / (b2 * precision + recall))
    else:
        f_b = 0.0
    return precision, recall, f1, f_b


def _binary_from_threshold(
    arr01: np.ndarray,
    valid_mask: np.ndarray,
    mode: str,
    percentile: float,
    value: float,
    otsu_pre_equalize: bool = True,
    otsu_equalize_method: str = "hist",
    otsu_clahe_clip_limit: float = 2.0,
    otsu_clahe_tile_grid_size: int = 8,
) -> np.ndarray:
    x = np.asarray(arr01, dtype=np.float32)
    vals = x[valid_mask]
    if vals.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)
    m = str(mode).strip().lower()
    if m == "value":
        thr = float(value)
    elif m == "otsu":
        x_u8 = _to_u8_from01(x)
        if bool(otsu_pre_equalize):
            x_eq = _apply_equalize_gray01(
                x_u8.astype(np.float32) / 255.0,
                valid_mask,
                method=otsu_equalize_method,
                clip_limit=float(otsu_clahe_clip_limit),
                tile_grid_size=int(otsu_clahe_tile_grid_size),
            )
            x_u8 = _to_u8_from01(x_eq)
        vals_u8 = x_u8[valid_mask]
        thr_u8, _ = cv2.threshold(vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(thr_u8 / 255.0)
    else:
        p = min(max(float(percentile), 0.0), 100.0)
        thr = float(np.percentile(vals, p))
    out = np.zeros_like(valid_mask, dtype=bool)
    out[valid_mask] = x[valid_mask] >= thr
    return out


def _select_corr_trust_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        key = str(k).replace(" ", "_").replace("∞", "inf").replace("%", "pct").lower()
        if ("correlation" in key) or ("spearman" in key) or ("trustworthiness" in key):
            try:
                out[key] = float(v)
            except Exception:
                continue
    return out


def _normalize_metric_key(key: object) -> str:
    return str(key).replace(" ", "_").replace("∞", "inf").replace("%", "pct").lower()


def _metrics_requested_set(cfg: DictConfig) -> set[str] | None:
    metrics_cfg = getattr(cfg, "metrics", None)
    if metrics_cfg is None:
        return None
    raw = getattr(metrics_cfg, "compute", None)
    if raw is None:
        return None
    try:
        items = list(raw)
    except Exception:
        items = [raw]
    out = set()
    for v in items:
        s = str(v).strip().lower()
        if s:
            out.add(s)
    return out if out else None


def _metric_enabled(requested: set[str] | None, name: str) -> bool:
    if requested is None:
        return True
    n = str(name).strip().lower()
    return ("all" in requested) or (n in requested)


def _is_nmf_free_mode(requested: set[str] | None) -> bool:
    if requested is None:
        return False
    cleaned = {str(v).strip().lower() for v in requested if str(v).strip()}
    if not cleaned:
        return False
    return cleaned.issubset({"correlation"})


def _needs_spectral_nmf_prediction_targets(cfg: object) -> bool:
    """MiCS-style spectral NMF on MSI + supervised heads (nmf_lr, mi_knn, stratification, …).

    When false, benchmark-style runs only need HD edge maps / edge_agreement / global correlation
    without fitting NMF on the hyperspectral cube.
    """
    requested = _metrics_requested_set(cfg)
    if requested is None:
        return True
    cleaned = {str(v).strip().lower() for v in requested if str(v).strip()}
    if not cleaned:
        return True
    if "all" in cleaned:
        return True
    spectral = {"nmf_lr", "nmf_kl_linear", "mi_knn", "edge_stratification", "component_roi_metrics"}
    return bool(cleaned & spectral)


def _correlation_feature_spaces(corr_cfg: object) -> List[str]:
    if corr_cfg is None:
        return ["raw"]
    raw = getattr(corr_cfg, "feature_spaces", None)
    if raw is None:
        raw = getattr(corr_cfg, "feature_space", None)
    if raw is None:
        return ["raw"]
    if isinstance(raw, str):
        items = [raw]
    else:
        try:
            items = list(raw)
        except Exception:
            items = [raw]
    spaces: List[str] = []
    for v in items:
        s = str(v).strip().lower()
        if s:
            spaces.append(s)
    if not spaces:
        spaces = ["raw"]
    if "all" in spaces:
        spaces = ["raw", "pca", "nmf"]
    # deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for s in spaces:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _embed_valid_pixels_to_image(
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    h, w = valid_mask.shape
    k = int(values.shape[1])
    out = np.zeros((h, w, k), dtype=np.float32)
    out[valid_mask] = np.asarray(values, dtype=np.float32)
    return out


def _pca_reduce_for_correlation(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    x = np.asarray(msi[valid_mask], dtype=np.float32)
    if x.size == 0:
        return np.zeros((*msi.shape[:2], 1), dtype=np.float32)
    k = max(1, min(int(n_components), int(x.shape[0]), int(x.shape[1])))
    pca = PCA(n_components=k, random_state=int(random_state), svd_solver="randomized")
    z = pca.fit_transform(x).astype(np.float32, copy=False)
    return _embed_valid_pixels_to_image(z, valid_mask)


def _nmf_reduce_for_correlation(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    n_components: int,
    random_state: int,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    x = np.asarray(msi[valid_mask], dtype=np.float32)
    if x.size == 0:
        return np.zeros((*msi.shape[:2], 1), dtype=np.float32)
    x = np.clip(x, 0.0, None)
    k = max(1, min(int(n_components), int(x.shape[0]), int(x.shape[1])))
    nmf = NMF(
        n_components=k,
        init="nndsvda",
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(random_state),
    )
    w = nmf.fit_transform(x).astype(np.float32, copy=False)
    return _embed_valid_pixels_to_image(w, valid_mask)


def _compute_correlation_metrics_multi_space(
    msi: np.ndarray,
    rgb_uint8: np.ndarray,
    valid_mask: np.ndarray,
    corr_cfg: object,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    spaces = _correlation_feature_spaces(corr_cfg)
    num_samples = int(getattr(corr_cfg, "pairwise_num_samples", 8000)) if corr_cfg is not None else 8000
    n_components = int(getattr(corr_cfg, "n_components", 64)) if corr_cfg is not None else 64
    pca_random_state = int(getattr(corr_cfg, "pca_random_state", 42)) if corr_cfg is not None else 42
    nmf_random_state = int(getattr(corr_cfg, "nmf_random_state", 42)) if corr_cfg is not None else 42
    nmf_max_iter = int(getattr(corr_cfg, "nmf_max_iter", 500)) if corr_cfg is not None else 500
    nmf_tol = float(getattr(corr_cfg, "nmf_tol", 1e-4)) if corr_cfg is not None else 1e-4

    logger.info("correlation metrics spaces=%s", spaces)
    first_space = spaces[0] if spaces else "raw"
    for space in spaces:
        t0 = time.perf_counter()
        if space == "raw":
            msi_for_metrics = np.asarray(msi, dtype=np.float32)
            suffix = "raw"
        elif space == "pca":
            msi_for_metrics = _pca_reduce_for_correlation(
                msi=msi,
                valid_mask=valid_mask,
                n_components=n_components,
                random_state=pca_random_state,
            )
            suffix = f"pca{msi_for_metrics.shape[-1]}"
        elif space == "nmf":
            msi_for_metrics = _nmf_reduce_for_correlation(
                msi=msi,
                valid_mask=valid_mask,
                n_components=n_components,
                random_state=nmf_random_state,
                max_iter=nmf_max_iter,
                tol=nmf_tol,
            )
            suffix = f"nmf{msi_for_metrics.shape[-1]}"
        else:
            logger.warning("Unknown correlation feature space '%s'; skipping", space)
            continue

        pair = MSIVisualizationMetrics(
            msi_for_metrics,
            rgb_uint8,
            mask=valid_mask,
            num_samples=num_samples,
        ).get_metrics()
        for key, value in pair.items():
            try:
                base_key = _normalize_metric_key(key)
                out[f"{suffix}__{base_key}"] = float(value)
                if space == first_space:
                    # Keep backward-compatible unprefixed keys from the first selected space.
                    out[base_key] = float(value)
            except Exception:
                continue
        logger.info(
            "correlation space complete | space=%s | msi_shape=%s | n_metrics=%d | time=%.2fs",
            suffix,
            tuple(msi_for_metrics.shape),
            int(len(pair)),
            float(time.perf_counter() - t0),
        )
    return out


def _entropy_u8(values_u8: np.ndarray) -> float:
    if values_u8.size == 0:
        return float("nan")
    hist = np.bincount(values_u8.astype(np.uint8), minlength=256).astype(np.float64)
    p = hist / max(1.0, float(np.sum(hist)))
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _color_diversity_metrics(
    rgb_uint8: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    rgb = np.asarray(rgb_uint8, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return out
    pix = rgb[valid_mask]
    if pix.size == 0:
        return out
    r = pix[:, 0]
    g = pix[:, 1]
    b = pix[:, 2]
    out["color_entropy_r"] = _entropy_u8(r)
    out["color_entropy_g"] = _entropy_u8(g)
    out["color_entropy_b"] = _entropy_u8(b)
    out["color_entropy_mean"] = float(np.nanmean([out["color_entropy_r"], out["color_entropy_g"], out["color_entropy_b"]]))
    out["color_std_r"] = float(np.std(r.astype(np.float32)))
    out["color_std_g"] = float(np.std(g.astype(np.float32)))
    out["color_std_b"] = float(np.std(b.astype(np.float32)))
    out["color_std_mean"] = float(np.mean([out["color_std_r"], out["color_std_g"], out["color_std_b"]]))
    out["color_used_bins_r"] = float(np.sum(np.bincount(r, minlength=256) > 0))
    out["color_used_bins_g"] = float(np.sum(np.bincount(g, minlength=256) > 0))
    out["color_used_bins_b"] = float(np.sum(np.bincount(b, minlength=256) > 0))
    out["color_used_bins_mean"] = float(
        np.mean([out["color_used_bins_r"], out["color_used_bins_g"], out["color_used_bins_b"]])
    )
    # Quantized unique colors to keep memory/runtime bounded.
    rq = (r >> 3).astype(np.int32)
    gq = (g >> 3).astype(np.int32)
    bq = (b >> 3).astype(np.int32)
    packed_q = rq + (gq << 5) + (bq << 10)  # 32x32x32 bins
    out["color_unique_q32"] = float(np.unique(packed_q).size)
    out["color_unique_q32_ratio"] = float(out["color_unique_q32"] / max(1.0, float(pix.shape[0])))
    return out


def _row_normalize_nonnegative(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
    s = np.sum(x, axis=1, keepdims=True).astype(np.float32, copy=False)
    s = np.maximum(s, float(eps))
    return (x / s).astype(np.float32, copy=False)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    den = np.sum(ez, axis=1, keepdims=True)
    den = np.maximum(den, 1e-12)
    return (ez / den).astype(np.float32, copy=False)


def _kl_divergence_mean(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    yt = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    yp = np.clip(np.asarray(y_pred, dtype=np.float64), float(eps), 1.0)
    yt = yt / np.maximum(np.sum(yt, axis=1, keepdims=True), float(eps))
    kl_rows = np.sum(yt * (np.log(yt + float(eps)) - np.log(yp + float(eps))), axis=1)
    return float(np.mean(kl_rows))


def _train_linear_softmax_kl(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: object,
) -> Tuple[np.ndarray, np.ndarray]:
    epochs = int(getattr(cfg, "epochs", 300)) if cfg is not None else 300
    lr = float(getattr(cfg, "learning_rate", 0.05)) if cfg is not None else 0.05
    weight_decay = float(getattr(cfg, "weight_decay", 1e-4)) if cfg is not None else 1e-4
    batch_size = int(getattr(cfg, "batch_size", 2048)) if cfg is not None else 2048
    random_state = int(getattr(cfg, "random_state", 42)) if cfg is not None else 42

    x = np.asarray(x_train, dtype=np.float32)
    y = _row_normalize_nonnegative(y_train)
    n, d = x.shape
    k = y.shape[1]
    rng = np.random.default_rng(random_state)
    w = (0.01 * rng.standard_normal((d, k))).astype(np.float32)
    b = np.zeros((k,), dtype=np.float32)
    if n <= 0:
        return w, b
    batch_size = max(1, min(batch_size, n))

    for _ in range(max(1, epochs)):
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = x[idx]
            yb = y[idx]
            probs = _softmax_rows(xb @ w + b)
            grad_logits = (probs - yb).astype(np.float32) / float(max(1, len(idx)))
            grad_w = xb.T @ grad_logits + float(weight_decay) * w
            grad_b = np.sum(grad_logits, axis=0)
            w = w - float(lr) * grad_w
            b = b - float(lr) * grad_b
    return w.astype(np.float32, copy=False), b.astype(np.float32, copy=False)


def _compute_nmf_kl_linear_scores(
    x_eval_rgb01: np.ndarray,
    y_eval_nmf_prob: np.ndarray,
    sample_folds: np.ndarray,
    cfg: object,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y_true = _row_normalize_nonnegative(y_eval_nmf_prob)
    x = np.asarray(x_eval_rgb01, dtype=np.float32)
    n = x.shape[0]
    if n <= 0 or y_true.shape[0] != n:
        out["nmf_kl_info_score"] = float("nan")
        out["nmf_kl_mean"] = float("nan")
        out["nmf_kl_weight_entropy_exp"] = float("nan")
        return out

    max_train_samples = int(getattr(cfg, "max_train_samples", 120000)) if cfg is not None else 120000
    random_state = int(getattr(cfg, "random_state", 42)) if cfg is not None else 42
    n_folds = int(np.max(sample_folds)) + 1 if sample_folds.size > 0 else 0
    y_pred_oof = np.full_like(y_true, np.nan, dtype=np.float32)

    for fold in range(max(0, n_folds)):
        test_idx = np.where(sample_folds == fold)[0]
        train_idx = np.where(sample_folds != fold)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        if 0 < max_train_samples < len(train_idx):
            rng = np.random.default_rng(random_state + 991 * (fold + 1))
            train_idx = rng.choice(train_idx, size=max_train_samples, replace=False)
        w, b = _train_linear_softmax_kl(x[train_idx], y_true[train_idx], cfg)
        y_pred_oof[test_idx] = _softmax_rows(x[test_idx] @ w + b)

    valid = np.all(np.isfinite(y_pred_oof), axis=1)
    if not np.any(valid):
        out["nmf_kl_info_score"] = float("nan")
        out["nmf_kl_mean"] = float("nan")
    else:
        kl_mean = _kl_divergence_mean(y_true[valid], y_pred_oof[valid], eps=float(getattr(cfg, "eps", 1e-8)))
        out["nmf_kl_mean"] = float(kl_mean)
        out["nmf_kl_info_score"] = float(np.exp(-kl_mean))

    # Fit on all eval points for singular-spectrum entropy diagnostic.
    w_all, _ = _train_linear_softmax_kl(x, y_true, cfg)
    svals = np.linalg.svd(np.asarray(w_all, dtype=np.float64), full_matrices=False, compute_uv=False)
    s3 = np.zeros((3,), dtype=np.float64)
    take = min(3, int(svals.size))
    if take > 0:
        s3[:take] = svals[:take]
    ssum = float(np.sum(s3))
    if ssum <= 1e-12:
        p = np.zeros_like(s3)
    else:
        p = s3 / ssum
    ent = float(-np.sum([pi * np.log(pi) for pi in p if pi > 0]))
    out["nmf_kl_w_s1_norm"] = float(p[0])
    out["nmf_kl_w_s2_norm"] = float(p[1])
    out["nmf_kl_w_s3_norm"] = float(p[2])
    out["nmf_kl_weight_entropy"] = ent
    out["nmf_kl_weight_entropy_exp"] = float(np.exp(ent))
    return out


def _pca_project_msi_for_hd_edges(msi: np.ndarray, valid_mask: np.ndarray, hd_cfg: object) -> np.ndarray:
    pca_cfg = getattr(hd_cfg, "pca", None) if hd_cfg is not None else None
    enabled = bool(getattr(pca_cfg, "enabled", False)) if pca_cfg is not None else False
    if not enabled:
        return np.asarray(msi, dtype=np.float32)

    x = np.asarray(msi[valid_mask], dtype=np.float32)
    if x.size == 0:
        return np.asarray(msi, dtype=np.float32)
    n_components = int(getattr(pca_cfg, "n_components", 16))
    random_state = int(getattr(pca_cfg, "random_state", 42))
    k = max(1, min(int(n_components), int(x.shape[0]), int(x.shape[1])))
    pca = PCA(n_components=k, random_state=random_state, svd_solver="randomized")
    z = pca.fit_transform(x).astype(np.float32, copy=False)
    out = np.zeros((msi.shape[0], msi.shape[1], k), dtype=np.float32)
    out[valid_mask] = z
    return out




def _block_ids_for_eval_positions(
    eval_positions: np.ndarray,
    valid_flat: np.ndarray,
    image_hw: Tuple[int, int],
    block_size: int,
) -> np.ndarray:
    h, w = image_hw
    flat_idx = valid_flat[eval_positions]
    rows = flat_idx // w
    cols = flat_idx % w
    br = rows // block_size
    bc = cols // block_size
    nbc = (w + block_size - 1) // block_size
    return (br * nbc + bc).astype(np.int64, copy=False)


def _assign_blocks_to_folds(
    block_ids: np.ndarray,
    n_folds: int,
    random_state: int,
) -> Dict[int, int]:
    unique_blocks = np.unique(block_ids)
    rng = np.random.default_rng(int(random_state))
    shuffled = unique_blocks.copy()
    rng.shuffle(shuffled)
    mapping: Dict[int, int] = {}
    for i, b in enumerate(shuffled):
        mapping[int(b)] = int(i % n_folds)
    return mapping


def _sample_uniform_across_blocks(
    sample_indices: np.ndarray,
    block_ids_for_samples: np.ndarray,
    budget: int,
    random_state: int,
) -> np.ndarray:
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    if budget <= 0 or len(sample_indices) <= budget:
        return sample_indices
    rng = np.random.default_rng(int(random_state))
    unique_blocks = np.unique(block_ids_for_samples)
    n_blocks = len(unique_blocks)
    if n_blocks == 0:
        return rng.choice(sample_indices, size=budget, replace=False).astype(np.int64)

    per_block = max(1, budget // n_blocks)
    chosen: List[int] = []
    block_to_candidates: Dict[int, np.ndarray] = {}
    for b in unique_blocks:
        cand = sample_indices[block_ids_for_samples == b]
        block_to_candidates[int(b)] = cand
        take = min(per_block, len(cand))
        if take > 0:
            pick = rng.choice(cand, size=take, replace=False)
            chosen.extend([int(v) for v in pick])

    chosen_set = set(chosen)
    if len(chosen) < budget:
        remaining = [int(v) for v in sample_indices if int(v) not in chosen_set]
        if remaining:
            extra = rng.choice(np.asarray(remaining, dtype=np.int64), size=min(budget - len(chosen), len(remaining)), replace=False)
            chosen.extend([int(v) for v in extra])

    if len(chosen) > budget:
        chosen = [int(v) for v in rng.choice(np.asarray(chosen, dtype=np.int64), size=budget, replace=False)]
    return np.asarray(chosen, dtype=np.int64)


class DebugNMFEvaluator:
    def __init__(self, msi_img: np.ndarray, cfg: DictConfig, edge_only: bool = False):
        self.cfg = cfg
        self.edge_only = bool(edge_only)
        self.msi = np.asarray(msi_img, dtype=np.float32)
        self.valid_mask = self.msi.sum(axis=-1) > 0
        if not np.any(self.valid_mask):
            raise ValueError("MSI valid mask is empty.")
        self.x_msi_valid = self.msi[self.valid_mask].astype(np.float32, copy=False)
        self.valid_flat = np.flatnonzero(self.valid_mask.ravel())
        self.edge_multiscale_sigmas, self.edge_multiscale_aggregation = _parse_multiscale_edge_cfg(
            getattr(self.cfg, "edge_agreement", None)
        )
        self.edge_method_cfg = _parse_edge_detection_cfg(self.cfg)
        self.hd_edge_cfg = getattr(self.cfg, "hd_edges", None)
        self.msi_for_hd_edges = _pca_project_msi_for_hd_edges(self.msi, self.valid_mask, self.hd_edge_cfg)
        msi_for_hd_edges = self.msi_for_hd_edges
        if int(msi_for_hd_edges.shape[2]) != int(self.msi.shape[2]):
            logger.info(
                "hd edge PCA enabled | channels %d -> %d",
                int(self.msi.shape[2]),
                int(msi_for_hd_edges.shape[2]),
            )
        self.highd_edge, self.highd_edge_linf = _highd_edge_maps_multiscale(
            msi_for_hd_edges,
            self.valid_mask,
            sigmas=self.edge_multiscale_sigmas,
            aggregation=self.edge_multiscale_aggregation,
            hd_cfg=self.hd_edge_cfg,
            edge_method_cfg=self.edge_method_cfg,
        )
        if self.edge_only:
            self.n_components = 0
            self.eval_positions = np.zeros(0, dtype=np.int64)
            self.target_positions = np.zeros(0, dtype=np.int64)
            self.y_targets = np.zeros((0, 1), dtype=np.float32)
            self.y_prob_targets = np.zeros((0, 1), dtype=np.float32)
            self.y_row_by_valid = np.full(self.x_msi_valid.shape[0], -1, dtype=np.int64)
            return
        if not _needs_spectral_nmf_prediction_targets(self.cfg):
            logger.info(
                "DebugNMFEvaluator: skipping MSI NMF prediction targets "
                "(enable nmf_lr, nmf_kl_linear, mi_knn, edge_stratification, or component_roi_metrics in metrics.compute)"
            )
            self._stub_empty_nmf_targets()
            return
        self._prepare_targets()

    def _stub_empty_nmf_targets(self) -> None:
        self.n_components = 0
        self.eval_positions = np.zeros(0, dtype=np.int64)
        self.target_positions = np.zeros(0, dtype=np.int64)
        self.y_targets = np.zeros((0, 1), dtype=np.float32)
        self.y_prob_targets = np.zeros((0, 1), dtype=np.float32)
        self.y_row_by_valid = np.full(self.x_msi_valid.shape[0], -1, dtype=np.int64)

    def _prepare_targets(self) -> None:
        nmf_train_idx = _sample_nmf_train_indices(self.msi, self.valid_mask, self.cfg.nmf_prediction)
        x_train = self.x_msi_valid[nmf_train_idx]
        nmf = NMF(
            n_components=int(self.cfg.nmf_prediction.k_components),
            init=str(self.cfg.nmf_prediction.init),
            max_iter=int(self.cfg.nmf_prediction.max_iter),
            tol=float(self.cfg.nmf_prediction.tol),
            random_state=int(self.cfg.nmf_prediction.random_state),
        )
        w_nmf = nmf.fit_transform(x_train)
        target_scope = str(self.cfg.nmf_prediction.target_scope).strip().lower()
        if target_scope == "sampled":
            target_positions = nmf_train_idx
            y_prob_targets = _row_normalize_nonnegative(w_nmf.astype(np.float32, copy=False))
            y_targets = _normalize_targets_percentile(
                w_nmf.astype(np.float32, copy=False),
                self.cfg.nmf_prediction.target_normalization,
                low_pct=float(self.cfg.nmf_prediction.target_norm_low_percentile),
                high_pct=float(self.cfg.nmf_prediction.target_norm_high_percentile),
            )
        elif target_scope == "all_valid":
            w_all = nmf.transform(self.x_msi_valid)
            target_positions = np.arange(self.x_msi_valid.shape[0], dtype=np.int64)
            y_prob_targets = _row_normalize_nonnegative(w_all.astype(np.float32, copy=False))
            y_targets = _normalize_targets_percentile(
                w_all.astype(np.float32, copy=False),
                self.cfg.nmf_prediction.target_normalization,
                low_pct=float(self.cfg.nmf_prediction.target_norm_low_percentile),
                high_pct=float(self.cfg.nmf_prediction.target_norm_high_percentile),
            )
        else:
            raise ValueError(f"Unsupported target_scope: {target_scope}")

        y_row_by_valid = np.full(self.x_msi_valid.shape[0], -1, dtype=np.int64)
        y_row_by_valid[target_positions] = np.arange(len(target_positions), dtype=np.int64)

        eval_positions = target_positions.copy()
        if 0 < int(self.cfg.nmf_prediction.cv_max_samples) < len(eval_positions):
            rng = np.random.default_rng(int(self.cfg.nmf_prediction.cv_random_state))
            eval_positions = np.sort(rng.choice(eval_positions, size=int(self.cfg.nmf_prediction.cv_max_samples), replace=False))

        self.y_targets = y_targets
        self.y_prob_targets = y_prob_targets
        self.y_row_by_valid = y_row_by_valid
        self.eval_positions = eval_positions
        self.n_components = int(y_targets.shape[1])
        self.target_positions = target_positions

    def compute_highd_edge_maps_for_hd_method(self, hd_method: str) -> tuple[np.ndarray, np.ndarray]:
        """Recompute HD / L∞ HD edge maps using the same MSI projection as ``__init__``."""
        cfg = dict(self.edge_method_cfg)
        cfg["hd_method"] = str(hd_method).strip().lower()
        return _highd_edge_maps_multiscale(
            self.msi_for_hd_edges,
            self.valid_mask,
            sigmas=self.edge_multiscale_sigmas,
            aggregation=self.edge_multiscale_aggregation,
            hd_cfg=self.hd_edge_cfg,
            edge_method_cfg=cfg,
        )

    def _vector_to_map(
        self,
        values: np.ndarray,
        eval_positions: np.ndarray,
        fill_value: float = float("nan"),
    ) -> np.ndarray:
        h, w = self.valid_mask.shape
        out = np.full((h * w,), fill_value, dtype=np.float32)
        pixel_idx = self.valid_flat[eval_positions]
        out[pixel_idx] = values.astype(np.float32, copy=False)
        return out.reshape(h, w)

    def _target_component_map(self, comp_idx: int) -> np.ndarray:
        h, w = self.valid_mask.shape
        out = np.zeros((h * w,), dtype=np.float32)
        valid_idx = self.valid_flat[self.target_positions]
        out[valid_idx] = self.y_targets[:, comp_idx].astype(np.float32, copy=False)
        return out.reshape(h, w)

    def evaluate_edge_evaluation_only(
        self,
        rgb_uint8: np.ndarray,
        *,
        highd_edge_override: np.ndarray | None = None,
        highd_edge_linf_override: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Recompute viz edges + edge_evaluation (F1 or continuous_dice). Uses self.highd_edge / highd_edge_linf
        from ``__init__`` unless overrides are passed. Skips AUC/Spearman/SSIM — use full evaluate_visualization for those.
        """
        hd_src = self.highd_edge if highd_edge_override is None else highd_edge_override
        hd_linf_src = self.highd_edge_linf if highd_edge_linf_override is None else highd_edge_linf_override
        clahe_enabled, clahe_method, clahe_clip_limit, clahe_tile_grid_size, clahe_apply_to = _clahe_params(self.cfg)
        rgb_for_edge = np.asarray(rgb_uint8, dtype=np.uint8)
        if clahe_enabled and bool(clahe_apply_to.get("edge_viz", False)):
            rgb_for_edge = _apply_equalize_rgb_uint8(
                rgb_for_edge,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        viz_edge, viz_edge_linf = _viz_edge_maps_multiscale(
            rgb_for_edge,
            self.valid_mask,
            sigmas=self.edge_multiscale_sigmas,
            aggregation=self.edge_multiscale_aggregation,
            edge_method_cfg=self.edge_method_cfg,
        )
        hd_edge_norm = _normalize_map_percentile(
            hd_src,
            self.valid_mask,
            low_pct=float(self.cfg.spatial_normalization.low_percentile),
            high_pct=float(self.cfg.spatial_normalization.high_percentile),
        )
        hd_edge_linf_norm = _normalize_map_percentile(
            hd_linf_src,
            self.valid_mask,
            low_pct=float(self.cfg.spatial_normalization.low_percentile),
            high_pct=float(self.cfg.spatial_normalization.high_percentile),
        )
        viz_edge_norm = _normalize_map_percentile(
            viz_edge,
            self.valid_mask,
            low_pct=float(self.cfg.spatial_normalization.low_percentile),
            high_pct=float(self.cfg.spatial_normalization.high_percentile),
        )
        viz_edge_linf_norm = _normalize_map_percentile(
            viz_edge_linf,
            self.valid_mask,
            low_pct=float(self.cfg.spatial_normalization.low_percentile),
            high_pct=float(self.cfg.spatial_normalization.high_percentile),
        )
        if clahe_enabled and bool(clahe_apply_to.get("edge_hd", False)):
            hd_edge_norm = _apply_equalize_gray01(
                hd_edge_norm,
                self.valid_mask,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
            hd_edge_linf_norm = _apply_equalize_gray01(
                hd_edge_linf_norm,
                self.valid_mask,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        if clahe_enabled and bool(clahe_apply_to.get("edge_viz", False)):
            viz_edge_norm = _apply_equalize_gray01(
                viz_edge_norm,
                self.valid_mask,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
            viz_edge_linf_norm = _apply_equalize_gray01(
                viz_edge_linf_norm,
                self.valid_mask,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        out: dict[str, Any] = {}
        eval_cfg = getattr(self.cfg, "edge_evaluation", None)
        eval_enabled = bool(getattr(eval_cfg, "enabled", True)) if eval_cfg is not None else True
        if not eval_enabled:
            return out
        eval_mode = str(getattr(eval_cfg, "threshold_mode", "otsu")) if eval_cfg is not None else "otsu"
        eval_percentile = float(getattr(eval_cfg, "threshold_percentile", 90.0)) if eval_cfg is not None else 90.0
        eval_value = float(getattr(eval_cfg, "threshold_value", 0.5)) if eval_cfg is not None else 0.5
        eval_include_linf = bool(getattr(eval_cfg, "include_linf", True)) if eval_cfg is not None else True
        eval_f_beta = float(getattr(eval_cfg, "f_beta", 1.0)) if eval_cfg is not None else 1.0
        eval_metric = str(getattr(eval_cfg, "metric", "f1")).strip().lower() if eval_cfg is not None else "f1"
        min_viz_edge_percentile = (
            float(getattr(eval_cfg, "min_viz_edge_percentile", 0.0)) if eval_cfg is not None else 0.0
        )
        otsu_pre_eq = bool(getattr(eval_cfg, "otsu_pre_equalize", True)) if eval_cfg is not None else True
        otsu_eq_method = str(getattr(eval_cfg, "otsu_equalize_method", "hist")) if eval_cfg is not None else "hist"
        otsu_clahe_clip = float(getattr(eval_cfg, "otsu_clahe_clip_limit", 2.0)) if eval_cfg is not None else 2.0
        otsu_clahe_grid = int(getattr(eval_cfg, "otsu_clahe_tile_grid_size", 8)) if eval_cfg is not None else 8

        def _apply_min_viz_edge(arr: np.ndarray) -> np.ndarray:
            x = np.array(arr, dtype=np.float32, copy=True)
            if min_viz_edge_percentile > 0 and np.any(self.valid_mask):
                thresh = float(np.percentile(x[self.valid_mask], min_viz_edge_percentile))
                x = np.where((self.valid_mask) & (x >= thresh), x, 0.0).astype(np.float32)
            x[~self.valid_mask] = 0.0
            return x

        viz_eval = _apply_min_viz_edge(viz_edge_norm)
        viz_linf_eval = _apply_min_viz_edge(viz_edge_linf_norm)

        sq_cont = bool(getattr(eval_cfg, "square_before_continuous_metrics", False)) if eval_cfg is not None else False
        hd_d, viz_d = _edges_for_continuous_metrics(hd_edge_norm, viz_eval, square=sq_cont)
        sp, sr = _compute_soft_precision_recall(hd_d, viz_d, self.valid_mask)
        out["edge_agreement_soft_precision"] = float(sp)
        out["edge_agreement_soft_recall"] = float(sr)
        if eval_include_linf:
            hd_ld, viz_ld = _edges_for_continuous_metrics(hd_edge_linf_norm, viz_linf_eval, square=sq_cont)
            sp2, sr2 = _compute_soft_precision_recall(hd_ld, viz_ld, self.valid_mask)
            out["edge_agreement_linf_soft_precision"] = float(sp2)
            out["edge_agreement_linf_soft_recall"] = float(sr2)
        else:
            out["edge_agreement_linf_soft_precision"] = float("nan")
            out["edge_agreement_linf_soft_recall"] = float("nan")

        if bool(getattr(eval_cfg, "compute_gradient_alignment", True)) if eval_cfg is not None else True:
            out["edge_agreement_gradient_alignment"] = float(
                _compute_weighted_gradient_alignment(hd_edge_norm, viz_eval, self.valid_mask)
            )
            if eval_include_linf:
                out["edge_agreement_linf_gradient_alignment"] = float(
                    _compute_weighted_gradient_alignment(
                        hd_edge_linf_norm, viz_linf_eval, self.valid_mask
                    )
                )
            else:
                out["edge_agreement_linf_gradient_alignment"] = float("nan")
        else:
            out["edge_agreement_gradient_alignment"] = float("nan")
            out["edge_agreement_linf_gradient_alignment"] = float("nan")

        if eval_metric == "continuous_dice":
            out["edge_agreement_eval_metric"] = "continuous_dice"
            out["edge_agreement_continuous_dice"] = float(
                _compute_continuous_dice(hd_d, viz_d, self.valid_mask)
            )
            out["edge_agreement_precision"] = float("nan")
            out["edge_agreement_recall"] = float("nan")
            out["edge_agreement_f1"] = float("nan")
            out["edge_agreement_f_beta"] = float("nan")
            if eval_include_linf:
                out["edge_agreement_linf_continuous_dice"] = float(
                    _compute_continuous_dice(hd_ld, viz_ld, self.valid_mask)
                )
                out["edge_agreement_linf_precision"] = float("nan")
                out["edge_agreement_linf_recall"] = float("nan")
                out["edge_agreement_linf_f1"] = float("nan")
                out["edge_agreement_linf_f_beta"] = float("nan")
            else:
                out["edge_agreement_linf_continuous_dice"] = float("nan")
                out["edge_agreement_linf_precision"] = float("nan")
                out["edge_agreement_linf_recall"] = float("nan")
                out["edge_agreement_linf_f1"] = float("nan")
                out["edge_agreement_linf_f_beta"] = float("nan")
        else:
            out["edge_agreement_eval_metric"] = "f1"
            out["edge_agreement_continuous_dice"] = float("nan")
            hd_bin = _binary_from_threshold(
                hd_edge_norm,
                self.valid_mask,
                eval_mode,
                eval_percentile,
                eval_value,
                otsu_pre_equalize=otsu_pre_eq,
                otsu_equalize_method=otsu_eq_method,
                otsu_clahe_clip_limit=otsu_clahe_clip,
                otsu_clahe_tile_grid_size=otsu_clahe_grid,
            )
            viz_bin = _binary_from_threshold(
                viz_eval,
                self.valid_mask,
                eval_mode,
                eval_percentile,
                eval_value,
                otsu_pre_equalize=otsu_pre_eq,
                otsu_equalize_method=otsu_eq_method,
                otsu_clahe_clip_limit=otsu_clahe_clip,
                otsu_clahe_tile_grid_size=otsu_clahe_grid,
            )
            p, r, f1, f_b = _compute_prf1(hd_bin[self.valid_mask], viz_bin[self.valid_mask], f_beta=eval_f_beta)
            out["edge_agreement_precision"] = float(p)
            out["edge_agreement_recall"] = float(r)
            out["edge_agreement_f1"] = float(f1)
            out["edge_agreement_f_beta"] = float(f_b)
            if eval_include_linf:
                hd_linf_bin = _binary_from_threshold(
                    hd_edge_linf_norm,
                    self.valid_mask,
                    eval_mode,
                    eval_percentile,
                    eval_value,
                    otsu_pre_equalize=otsu_pre_eq,
                    otsu_equalize_method=otsu_eq_method,
                    otsu_clahe_clip_limit=otsu_clahe_clip,
                    otsu_clahe_tile_grid_size=otsu_clahe_grid,
                )
                viz_linf_bin = _binary_from_threshold(
                    viz_linf_eval,
                    self.valid_mask,
                    eval_mode,
                    eval_percentile,
                    eval_value,
                    otsu_pre_equalize=otsu_pre_eq,
                    otsu_equalize_method=otsu_eq_method,
                    otsu_clahe_clip_limit=otsu_clahe_clip,
                    otsu_clahe_tile_grid_size=otsu_clahe_grid,
                )
                p2, r2, f12, f_b2 = _compute_prf1(
                    hd_linf_bin[self.valid_mask], viz_linf_bin[self.valid_mask], f_beta=eval_f_beta
                )
                out["edge_agreement_linf_precision"] = float(p2)
                out["edge_agreement_linf_recall"] = float(r2)
                out["edge_agreement_linf_f1"] = float(f12)
                out["edge_agreement_linf_f_beta"] = float(f_b2)
                out["edge_agreement_linf_continuous_dice"] = float("nan")
        return out

    def evaluate_edge_agreement_and_maps(
        self,
        rgb_uint8: np.ndarray,
        *,
        highd_edge_override: np.ndarray | None = None,
        highd_edge_linf_override: np.ndarray | None = None,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Full edge-agreement metrics (AUC, Spearman, SSIM, multi-percentile AUC) plus normalized
        HD / RGB edge maps, then merges threshold-style metrics from ``evaluate_edge_evaluation_only``.
        Works with ``edge_only=True`` (no NMF). RGB viz edges are computed twice if threshold
        metrics are enabled (acceptable for tooling).
        Pass ``highd_edge_override`` / ``highd_edge_linf_override`` to benchmark multiple ``hd_method`` values
        without re-instantiating the evaluator.
        """
        hd_src = self.highd_edge if highd_edge_override is None else highd_edge_override
        hd_linf_src = self.highd_edge_linf if highd_edge_linf_override is None else highd_edge_linf_override
        clahe_enabled, clahe_method, clahe_clip_limit, clahe_tile_grid_size, clahe_apply_to = _clahe_params(
            self.cfg
        )
        t_edge0 = time.perf_counter()
        rgb_for_edge = np.asarray(rgb_uint8, dtype=np.uint8)
        if clahe_enabled and bool(clahe_apply_to.get("edge_viz", False)):
            rgb_for_edge = _apply_equalize_rgb_uint8(
                rgb_for_edge,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        viz_edge, viz_edge_linf = _viz_edge_maps_multiscale(
            rgb_for_edge,
            self.valid_mask,
            sigmas=self.edge_multiscale_sigmas,
            aggregation=self.edge_multiscale_aggregation,
            edge_method_cfg=self.edge_method_cfg,
        )
        sp = float(self.cfg.spatial_normalization.low_percentile)
        hp = float(self.cfg.spatial_normalization.high_percentile)
        hd_edge_norm = _normalize_map_percentile(hd_src, self.valid_mask, low_pct=sp, high_pct=hp)
        hd_edge_linf_norm = _normalize_map_percentile(hd_linf_src, self.valid_mask, low_pct=sp, high_pct=hp)
        viz_edge_norm = _normalize_map_percentile(viz_edge, self.valid_mask, low_pct=sp, high_pct=hp)
        viz_edge_linf_norm = _normalize_map_percentile(viz_edge_linf, self.valid_mask, low_pct=sp, high_pct=hp)
        if clahe_enabled and bool(clahe_apply_to.get("edge_hd", False)):
            hd_edge_norm = _apply_equalize_gray01(
                hd_edge_norm,
                self.valid_mask,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
            hd_edge_linf_norm = _apply_equalize_gray01(
                hd_edge_linf_norm,
                self.valid_mask,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        if clahe_enabled and bool(clahe_apply_to.get("edge_viz", False)):
            viz_edge_norm = _apply_equalize_gray01(
                viz_edge_norm,
                self.valid_mask,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
            viz_edge_linf_norm = _apply_equalize_gray01(
                viz_edge_linf_norm,
                self.valid_mask,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        edge_agreement_spearman = _safe_spearman(
            hd_edge_norm[self.valid_mask], viz_edge_norm[self.valid_mask]
        )
        edge_agreement_pearson = _safe_pearson(
            hd_edge_norm[self.valid_mask], viz_edge_norm[self.valid_mask]
        )
        edge_agreement_linf_spearman = _safe_spearman(
            hd_edge_linf_norm[self.valid_mask], viz_edge_linf_norm[self.valid_mask]
        )
        try:
            edge_agreement_ssim = float(
                ssim(
                    hd_edge_norm,
                    viz_edge_norm,
                    data_range=1.0,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                )
            )
        except Exception:
            edge_agreement_ssim = float("nan")
        auc_metrics = _edge_auc_metrics(
            hd_edge_norm,
            viz_edge_norm,
            self.valid_mask,
            positive_percentile=float(self.cfg.edge_agreement.auc_positive_percentile),
            min_positive_pixels=int(self.cfg.edge_agreement.auc_min_positive_pixels),
        )
        mp_cfg = getattr(self.cfg.edge_agreement, "multi_percentile", None)
        if mp_cfg is None:
            mp_metrics = {
                "edge_agreement_auc_roc_weighted": float("nan"),
                "edge_agreement_auc_pr_weighted": float("nan"),
                "edge_agreement_auc_multi_valid_percentiles": 0.0,
            }
        else:
            mp_metrics = _edge_auc_metrics_multi_percentile(
                hd_edge_norm,
                viz_edge_norm,
                self.valid_mask,
                percentiles=[float(v) for v in list(getattr(mp_cfg, "percentiles", [70, 80, 85, 90, 95]))],
                min_positive_pixels=int(self.cfg.edge_agreement.auc_min_positive_pixels),
                weight_mode=str(getattr(mp_cfg, "weight_mode", "tail_power")),
                tail_power=float(getattr(mp_cfg, "tail_power", 2.0)),
            )
        edge_summary: Dict[str, float] = {}
        edge_summary.update(
            {
                "edge_agreement_hd_vs_viz_spearman": edge_agreement_spearman,
                "edge_agreement_hd_vs_viz_pearson": edge_agreement_pearson,
                "edge_agreement_hd_vs_viz_l_infinity_spearman": edge_agreement_linf_spearman,
                "edge_agreement_hd_vs_viz_ssim": edge_agreement_ssim,
                "edge_agreement_auc_roc": float(auc_metrics["edge_agreement_auc_roc"]),
                "edge_agreement_auc_pr": float(auc_metrics["edge_agreement_auc_pr"]),
                "edge_agreement_auc_roc_weighted": float(mp_metrics["edge_agreement_auc_roc_weighted"]),
                "edge_agreement_auc_pr_weighted": float(mp_metrics["edge_agreement_auc_pr_weighted"]),
                "edge_agreement_auc_multi_valid_percentiles": float(
                    mp_metrics["edge_agreement_auc_multi_valid_percentiles"]
                ),
            }
        )
        for k, v in mp_metrics.items():
            if k in edge_summary:
                continue
            try:
                edge_summary[str(k)] = float(v)
            except Exception:
                continue
        edge_maps = {
            "hd_edge_norm": np.asarray(hd_edge_norm, dtype=np.float32),
            "hd_edge_linf_norm": np.asarray(hd_edge_linf_norm, dtype=np.float32),
            "viz_edge_norm": np.asarray(viz_edge_norm, dtype=np.float32),
            "viz_edge_linf_norm": np.asarray(viz_edge_linf_norm, dtype=np.float32),
        }
        logger.info(
            "evaluate_edge_agreement_and_maps | auc_roc=%.4f | auc_pr=%.4f | time=%.2fs",
            float(edge_summary.get("edge_agreement_auc_roc", float("nan"))),
            float(edge_summary.get("edge_agreement_auc_pr", float("nan"))),
            float(time.perf_counter() - t_edge0),
        )
        thr = self.evaluate_edge_evaluation_only(
            rgb_uint8,
            highd_edge_override=hd_src,
            highd_edge_linf_override=hd_linf_src,
        )
        for k, v in thr.items():
            try:
                edge_summary[str(k)] = float(v)
            except (TypeError, ValueError):
                edge_summary[str(k)] = float("nan")
        return edge_summary, edge_maps

    def evaluate_visualization(
        self,
        rgb_uint8: np.ndarray,
        return_component_maps: bool = True,
        return_edge_maps: bool = False,
    ) -> Tuple[Dict[str, object], List[Dict[str, np.ndarray]]]:
        if self.edge_only:
            raise RuntimeError(
                "DebugNMFEvaluator(edge_only=True) cannot run full evaluate_visualization; "
                "use evaluate_edge_evaluation_only(...) instead."
            )
        t_eval0 = time.perf_counter()
        requested = _metrics_requested_set(self.cfg)
        enable_nmf_lr = _metric_enabled(requested, "nmf_lr")
        enable_nmf_kl_linear = _metric_enabled(requested, "nmf_kl_linear")
        enable_mi_knn = _metric_enabled(requested, "mi_knn")
        enable_edge_agreement = _metric_enabled(requested, "edge_agreement")
        enable_correlation = _metric_enabled(requested, "correlation")
        enable_edge_stratification = _metric_enabled(requested, "edge_stratification")
        enable_component_roi = _metric_enabled(requested, "component_roi_metrics")
        enable_zadu_bins = _metric_enabled(requested, "zadu_bins")
        clahe_enabled, clahe_method, clahe_clip_limit, clahe_tile_grid_size, clahe_apply_to = _clahe_params(self.cfg)
        log_every = int(getattr(getattr(self.cfg, "logging", None), "component_log_every", 8))
        logger.info(
            "evaluate_visualization start | components=%d | enabled=%s",
            int(self.n_components),
            {
                "nmf_lr": enable_nmf_lr,
                "nmf_kl_linear": enable_nmf_kl_linear,
                "mi_knn": enable_mi_knn,
                "edge_agreement": enable_edge_agreement,
                "correlation": enable_correlation,
                "edge_stratification": enable_edge_stratification,
                "component_roi_metrics": enable_component_roi,
                "zadu_bins": enable_zadu_bins,
                "clahe": clahe_enabled,
                "equalize_method": clahe_method,
                "edge_methods": {
                    "hd": str(self.edge_method_cfg.get("hd_method", "gradient")),
                    "rgb": str(self.edge_method_cfg.get("rgb_method", "gradient")),
                },
            },
        )

        rgb_for_nmf = np.asarray(rgb_uint8, dtype=np.uint8)
        if clahe_enabled and bool(clahe_apply_to.get("nmf_rgb_features", False)):
            rgb_for_nmf = _apply_equalize_rgb_uint8(
                rgb_for_nmf,
                method=clahe_method,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        x_rgb_valid = (np.asarray(rgb_for_nmf, dtype=np.float32)[self.valid_mask] / 255.0).astype(np.float32, copy=False)
        prediction_ready = int(self.n_components) > 0 and int(self.eval_positions.size) > 0
        quantile_summary: List[Dict[str, float]] = []
        frequency_bins: List[Dict[str, float]] = []

        y_rows = np.zeros(0, dtype=np.int64)
        block_ids = None
        use_block_cv = False
        n_folds = 2
        if not prediction_ready:
            eval_positions = np.zeros(0, dtype=np.int64)
            nc = max(1, x_rgb_valid.shape[1])
            x_eval = np.zeros((0, nc), dtype=np.float32)
            y_eval = np.zeros((0, 0), dtype=np.float32)
            sample_folds = np.zeros(0, dtype=np.int64)
            y_pred_oof = np.zeros((0, 0), dtype=np.float32)
        else:
            eval_positions = self.eval_positions.copy()
            y_rows = self.y_row_by_valid[eval_positions]
            y_rows = y_rows[y_rows >= 0]
            eval_positions = eval_positions[: len(y_rows)]
            x_eval = x_rgb_valid[eval_positions]
            y_eval = self.y_targets[y_rows]

            use_block_cv = bool(getattr(self.cfg.block_cv, "enabled", True))
            n_folds = int(getattr(self.cfg.block_cv, "n_folds", int(self.cfg.nmf_prediction.cv_n_splits)))
            n_folds = max(2, n_folds)
            block_ids = None
            sample_folds = None
            if use_block_cv:
                block_size = int(getattr(self.cfg.block_cv, "block_size", 64))
                block_size = max(1, block_size)
                block_ids = _block_ids_for_eval_positions(
                    eval_positions=eval_positions,
                    valid_flat=self.valid_flat,
                    image_hw=self.valid_mask.shape,
                    block_size=block_size,
                )
                block_to_fold = _assign_blocks_to_folds(
                    block_ids=block_ids,
                    n_folds=n_folds,
                    random_state=int(getattr(self.cfg.block_cv, "random_state", self.cfg.nmf_prediction.cv_random_state)),
                )
                sample_folds = np.array([block_to_fold[int(b)] for b in block_ids], dtype=np.int64)
            else:
                kf = KFold(
                    n_splits=n_folds,
                    shuffle=bool(self.cfg.nmf_prediction.cv_shuffle),
                    random_state=int(self.cfg.nmf_prediction.cv_random_state),
                )
                sample_folds = np.full(x_eval.shape[0], -1, dtype=np.int64)
                for fold, (_, test_idx) in enumerate(kf.split(x_eval)):
                    sample_folds[test_idx] = fold

            y_pred_oof = np.full_like(y_eval, np.nan, dtype=np.float32)

        per_component: List[Dict[str, float]] = []
        component_maps: List[Dict[str, np.ndarray]] = []
        mi_sums: List[float] = []
        mi_maxs: List[float] = []
        nmf_kl_summary: Dict[str, float] = {}

        if enable_nmf_kl_linear:
            logger.info("metric: nmf_kl_linear start")
            if not prediction_ready:
                logger.warning("nmf_kl_linear skipped: no spectral NMF targets (see metrics.compute)")
                nmf_kl_summary = {}
            else:
                t_kl0 = time.perf_counter()
                y_prob_eval = self.y_prob_targets[y_rows]
                nmf_kl_summary = _compute_nmf_kl_linear_scores(
                    x_eval_rgb01=x_eval,
                    y_eval_nmf_prob=y_prob_eval,
                    sample_folds=sample_folds,
                    cfg=getattr(self.cfg, "nmf_kl_linear", None),
                )
                logger.info(
                    "nmf_kl_linear complete | infoscore=%.4f | kl=%.4f | entropy_exp=%.4f | time=%.2fs",
                    float(nmf_kl_summary.get("nmf_kl_info_score", float("nan"))),
                    float(nmf_kl_summary.get("nmf_kl_mean", float("nan"))),
                    float(nmf_kl_summary.get("nmf_kl_weight_entropy_exp", float("nan"))),
                    float(time.perf_counter() - t_kl0),
                )

        t_comp_loop0 = time.perf_counter()
        if enable_nmf_lr or enable_mi_knn or enable_edge_stratification:
            logger.info("metric: nmf_lr / mi_knn / per_component start")
        for comp_idx in range(self.n_components):
            t_comp0 = time.perf_counter()
            y = y_eval[:, comp_idx]
            fold_r2, fold_mae, fold_rmse, fold_p, fold_s = [], [], [], [], []
            for fold in range(n_folds):
                test_idx = np.where(sample_folds == fold)[0]
                train_idx = np.where(sample_folds != fold)[0]
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                if use_block_cv and block_ids is not None:
                    train_idx = _sample_uniform_across_blocks(
                        sample_indices=train_idx,
                        block_ids_for_samples=block_ids[train_idx],
                        budget=int(self.cfg.nmf_prediction.sampling_num_samples),
                        random_state=int(self.cfg.nmf_prediction.cv_random_state) + comp_idx * 1000 + fold,
                    )
                elif 0 < int(self.cfg.nmf_prediction.sampling_num_samples) < len(train_idx):
                    rng = np.random.default_rng(int(self.cfg.nmf_prediction.cv_random_state) + comp_idx * 1000 + fold)
                    train_idx = rng.choice(train_idx, size=int(self.cfg.nmf_prediction.sampling_num_samples), replace=False)
                if len(train_idx) == 0:
                    continue
                model = _xgb_from_cfg(self.cfg.nmf_prediction)
                model.fit(x_eval[train_idx], y[train_idx])
                pred = model.predict(x_eval[test_idx]).astype(np.float32, copy=False)
                y_pred_oof[test_idx, comp_idx] = pred
                fold_r2.append(float(r2_score(y[test_idx], pred)))
                fold_mae.append(float(mean_absolute_error(y[test_idx], pred)))
                fold_rmse.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
                # Compute correlation metrics on percentile-normalized and clipped vectors.
                y_corr = _percentile_normalize_vector(
                    y[test_idx],
                    low_pct=float(self.cfg.nmf_prediction.target_norm_low_percentile),
                    high_pct=float(self.cfg.nmf_prediction.target_norm_high_percentile),
                )
                pred_corr = _percentile_normalize_vector(
                    pred,
                    low_pct=float(self.cfg.nmf_prediction.target_norm_low_percentile),
                    high_pct=float(self.cfg.nmf_prediction.target_norm_high_percentile),
                )
                fold_p.append(_safe_pearson(y_corr, pred_corr))
                fold_s.append(_safe_spearman(y_corr, pred_corr))

            if enable_mi_knn:
                try:
                    mi_train_idx = np.arange(x_eval.shape[0], dtype=np.int64)
                    if use_block_cv and block_ids is not None:
                        mi_train_idx = _sample_uniform_across_blocks(
                            sample_indices=mi_train_idx,
                            block_ids_for_samples=block_ids,
                            budget=int(self.cfg.nmf_prediction.sampling_num_samples),
                            random_state=int(self.cfg.nmf_prediction.cv_random_state) + comp_idx * 2000 + 17,
                        )
                    elif 0 < int(self.cfg.nmf_prediction.sampling_num_samples) < len(mi_train_idx):
                        rng = np.random.default_rng(int(self.cfg.nmf_prediction.cv_random_state) + comp_idx * 2000 + 17)
                        mi_train_idx = rng.choice(mi_train_idx, size=int(self.cfg.nmf_prediction.sampling_num_samples), replace=False)
                    x_mi = x_eval[mi_train_idx]
                    y_mi = y[mi_train_idx]
                    mi_vec = mutual_info_regression(
                        x_mi,
                        y_mi,
                        discrete_features=False,
                        n_neighbors=int(self.cfg.mi.n_neighbors),
                        random_state=int(self.cfg.mi.random_state),
                    )
                    mi_sum = float(np.sum(mi_vec))
                    mi_max = float(np.max(mi_vec)) if mi_vec.size > 0 else 0.0
                except Exception:
                    mi_sum = 0.0
                    mi_max = 0.0
            else:
                mi_sum = float("nan")
                mi_max = float("nan")
            mi_sums.append(mi_sum)
            mi_maxs.append(mi_max)

            y_true_map_raw = self._vector_to_map(y, eval_positions, fill_value=float("nan"))
            y_pred_map_raw = self._vector_to_map(y_pred_oof[:, comp_idx], eval_positions, fill_value=float("nan"))
            comp_valid_mask = self.valid_mask & np.isfinite(y_true_map_raw) & np.isfinite(y_pred_map_raw)
            if np.any(comp_valid_mask):
                spatial_mean_true = float(np.mean(y_true_map_raw[comp_valid_mask]))
            else:
                spatial_mean_true = float("nan")

            y_true_map = _normalize_map_percentile(
                y_true_map_raw,
                comp_valid_mask,
                low_pct=float(self.cfg.spatial_normalization.low_percentile),
                high_pct=float(self.cfg.spatial_normalization.high_percentile),
            )
            y_pred_map = _normalize_map_percentile(
                y_pred_map_raw,
                comp_valid_mask,
                low_pct=float(self.cfg.spatial_normalization.low_percentile),
                high_pct=float(self.cfg.spatial_normalization.high_percentile),
            )
            if clahe_enabled and bool(clahe_apply_to.get("nmf_component_maps", False)):
                y_true_map = _apply_equalize_gray01(
                    y_true_map,
                    comp_valid_mask,
                    method=clahe_method,
                    clip_limit=clahe_clip_limit,
                    tile_grid_size=clahe_tile_grid_size,
                )
                y_pred_map = _apply_equalize_gray01(
                    y_pred_map,
                    comp_valid_mask,
                    method=clahe_method,
                    clip_limit=clahe_clip_limit,
                    tile_grid_size=clahe_tile_grid_size,
                )
            true_edge, edge_energy_true = _sobel_edge_energy(y_true_map, comp_valid_mask)
            pred_edge, edge_energy_pred = _sobel_edge_energy(y_pred_map, comp_valid_mask)
            edge_energy_true = _normalize_edge_energy_by_component_power(true_edge, y_true_map, comp_valid_mask)
            edge_energy_pred = _normalize_edge_energy_by_component_power(pred_edge, y_pred_map, comp_valid_mask)
            edge_corr_spearman = _safe_spearman(true_edge[comp_valid_mask], pred_edge[comp_valid_mask])
            edge_corr_pearson = _safe_pearson(true_edge[comp_valid_mask], pred_edge[comp_valid_mask])
            if return_component_maps:
                component_maps.append(
                    {
                        "component": np.array([comp_idx], dtype=np.int64),
                        "y_true_map": y_true_map.astype(np.float32, copy=False),
                        "y_pred_map": y_pred_map.astype(np.float32, copy=False),
                        "true_edge_map": true_edge.astype(np.float32, copy=False),
                        "pred_edge_map": pred_edge.astype(np.float32, copy=False),
                        "nmf_spearman": np.array([float(np.mean(fold_s))], dtype=np.float32),
                        "edge_spearman": np.array([float(edge_corr_spearman)], dtype=np.float32),
                    }
                )

            per_component.append(
                {
                    "component": int(comp_idx),
                    "r2": float(np.mean(fold_r2)),
                    "mae": float(np.mean(fold_mae)),
                    "rmse": float(np.mean(fold_rmse)),
                    "pearson": float(np.mean(fold_p)),
                    "spearman": float(np.mean(fold_s)),
                    "edge_energy_true": edge_energy_true,
                    "edge_energy_pred": edge_energy_pred,
                    "edge_corr_spearman": edge_corr_spearman,
                    "edge_corr_pearson": edge_corr_pearson,
                    "mi_sum": mi_sum,
                    "mi_max": mi_max,
                    "spatial_mean_true": spatial_mean_true,
                }
            )
            if log_every > 0 and (((comp_idx + 1) % log_every) == 0 or (comp_idx + 1) == self.n_components):
                logger.info(
                    "component progress %d/%d | comp_time=%.2fs | elapsed=%.2fs",
                    int(comp_idx + 1),
                    int(self.n_components),
                    float(time.perf_counter() - t_comp0),
                    float(time.perf_counter() - t_eval0),
                )

        if enable_nmf_lr or enable_mi_knn or enable_edge_stratification:
            logger.info(
                "metric: nmf_lr / mi_knn / per_component complete | components=%d | time=%.2fs",
                int(self.n_components),
                float(time.perf_counter() - t_comp_loop0),
            )

        edge_for_bins = np.array([], dtype=np.float64)
        if enable_edge_stratification and len(per_component) == 0:
            logger.warning("edge_stratification skipped: no per-component metrics (spectral NMF prediction disabled)")
        elif enable_edge_stratification:
            logger.info("metric: edge_stratification start")
            t_bins0 = time.perf_counter()
            # Binning signal: gradient energy of each GT NMF component.
            edge_values = np.array([d["edge_energy_true"] for d in per_component], dtype=np.float64)
            finite_edge = np.isfinite(edge_values)
            if np.any(finite_edge):
                median_edge = float(np.median(edge_values[finite_edge]))
                edge_for_bins = np.where(finite_edge, edge_values, median_edge)
            else:
                edge_for_bins = np.zeros_like(edge_values, dtype=np.float64)
            n_quantiles = int(self.cfg.edge_stratification.n_quantiles)
            n_quantiles = max(2, min(10, n_quantiles))
            if np.allclose(edge_for_bins, edge_for_bins[0]):
                groups = np.zeros_like(edge_for_bins, dtype=int)
                q_count = 1
            else:
                bins = np.quantile(edge_for_bins, np.linspace(0.0, 1.0, n_quantiles + 1))
                bins[0] -= 1e-12
                bins[-1] += 1e-12
                groups = np.digitize(edge_for_bins, bins[1:-1], right=True)
                q_count = n_quantiles

            for q in range(q_count):
                idx = np.where(groups == q)[0]
                if idx.size == 0:
                    continue
                sel = [per_component[i] for i in idx]
                quantile_summary.append(
                    {
                        "quantile": int(q),
                        "n_components": int(len(sel)),
                        "edge_energy_min": float(np.min([s["edge_energy_true"] for s in sel])),
                        "edge_energy_max": float(np.max([s["edge_energy_true"] for s in sel])),
                        "mean_r2": float(np.mean([s["r2"] for s in sel])),
                        "mean_mae": float(np.mean([s["mae"] for s in sel])),
                        "mean_rmse": float(np.mean([s["rmse"] for s in sel])),
                        "mean_edge_corr_spearman": float(np.mean([s["edge_corr_spearman"] for s in sel])),
                        "mean_edge_energy_true": float(np.mean([s["edge_energy_true"] for s in sel])),
                    }
                )

            # Fixed N-bin partition based on edge energy.
            if len(per_component) > 0:
                n_bins = int(getattr(self.cfg.edge_stratification, "n_bins", 10))
                n_bins = max(2, min(50, n_bins))
                if np.allclose(edge_for_bins, edge_for_bins[0]):
                    groups_n = np.zeros_like(edge_for_bins, dtype=np.int64)
                    thresholds = np.array([float(edge_for_bins[0])] * (n_bins + 1), dtype=np.float64)
                else:
                    thresholds = np.quantile(edge_for_bins, np.linspace(0.0, 1.0, n_bins + 1))
                    thresholds[0] -= 1e-12
                    thresholds[-1] += 1e-12
                    groups_n = np.digitize(edge_for_bins, thresholds[1:-1], right=True)
                metric_keys = [
                    "r2",
                    "mae",
                    "rmse",
                    "pearson",
                    "spearman",
                    "edge_energy_true",
                    "edge_energy_pred",
                    "edge_corr_spearman",
                    "edge_corr_pearson",
                    "mi_sum",
                    "mi_max",
                    "spatial_mean_true",
                ]
                for gid in range(n_bins):
                    idx = np.where(groups_n == gid)[0]
                    if idx.size == 0:
                        continue
                    sel = [per_component[i] for i in idx]
                    out = {
                        "bin": f"bin_{gid:02d}",
                        "bin_index": int(gid),
                        "n_components": int(len(sel)),
                        "edge_energy_min": float(np.min([s["edge_energy_true"] for s in sel])),
                        "edge_energy_max": float(np.max([s["edge_energy_true"] for s in sel])),
                        "edge_energy_bin_lower_threshold": float(thresholds[gid]),
                        "edge_energy_bin_upper_threshold": float(thresholds[gid + 1]),
                    }
                    for mk in metric_keys:
                        out[f"mean_{mk}"] = float(np.mean([float(s[mk]) for s in sel]))

                    # Optional ZADU metrics per bin:
                    # compute per-component ROI metrics, then average over components in the bin.
                    if enable_zadu_bins and bool(getattr(self.cfg.zadu_bins, "enabled", True)):
                        top_frac = float(getattr(self.cfg.zadu_bins, "top_fraction", 0.20))
                        top_frac = min(max(top_frac, 0.001), 0.999)
                        zadu_samples = int(getattr(self.cfg.zadu_bins, "pairwise_num_samples", 4000))
                        min_pixels = int(getattr(self.cfg.zadu_bins, "min_pixels", 64))
                        per_component_zadu: Dict[str, List[float]] = {}
                        roi_pixels_list: List[float] = []
                        n_component_rois = 0
                        for comp_id in idx:
                            comp_map = self._target_component_map(int(comp_id))
                            vals = comp_map[self.valid_mask]
                            if vals.size == 0:
                                continue
                            thr = float(np.quantile(vals, 1.0 - top_frac))
                            roi_mask = (comp_map >= thr) & self.valid_mask
                            n_roi = int(np.sum(roi_mask))
                            if n_roi < min_pixels:
                                continue
                            try:
                                zadu_all = MSIVisualizationMetrics(
                                    self.msi,
                                    rgb_uint8,
                                    mask=roi_mask,
                                    num_samples=zadu_samples,
                                ).get_metrics()
                                zadu_sel = _select_corr_trust_metrics(zadu_all)
                                if not zadu_sel:
                                    continue
                                n_component_rois += 1
                                roi_pixels_list.append(float(n_roi))
                                for zk, zv in zadu_sel.items():
                                    per_component_zadu.setdefault(zk, []).append(float(zv))
                            except Exception:
                                continue

                        out["zadu_component_rois_used"] = float(n_component_rois)
                        if roi_pixels_list:
                            out["zadu_roi_pixels_mean"] = float(np.mean(roi_pixels_list))
                            out["zadu_roi_pixels_min"] = float(np.min(roi_pixels_list))
                            out["zadu_roi_pixels_max"] = float(np.max(roi_pixels_list))
                        for zk, vals in per_component_zadu.items():
                            if vals:
                                out[f"zadu_{zk}"] = float(np.mean(vals))
                    frequency_bins.append(out)
            logger.info(
                "edge stratification complete | quantiles=%d | bins=%d | time=%.2fs",
                int(len(quantile_summary)),
                int(len(frequency_bins)),
                float(time.perf_counter() - t_bins0),
            )

        component_roi_rows: List[Dict[str, float]] = []
        component_roi_summary: Dict[str, float] = {}
        if enable_component_roi and bool(getattr(self.cfg.component_roi_metrics, "enabled", False)):
            logger.info("metric: component_roi_metrics start")
            t_roi0 = time.perf_counter()
            top_frac = float(getattr(self.cfg.component_roi_metrics, "top_fraction", 0.20))
            top_frac = min(max(top_frac, 0.001), 0.999)
            num_samples = int(getattr(self.cfg.component_roi_metrics, "num_samples", 4000))
            min_pixels = int(getattr(self.cfg.component_roi_metrics, "min_pixels", 64))
            agg: Dict[str, List[float]] = {}
            for comp_idx in range(self.n_components):
                comp_map = self._target_component_map(comp_idx)
                vals = comp_map[self.valid_mask]
                if vals.size == 0:
                    continue
                thr = float(np.quantile(vals, 1.0 - top_frac))
                roi_mask = (comp_map >= thr) & self.valid_mask
                n_roi = int(np.sum(roi_mask))
                row: Dict[str, float] = {"component": float(comp_idx), "roi_pixels": float(n_roi)}
                if n_roi < min_pixels:
                    component_roi_rows.append(row)
                    continue
                try:
                    m = MSIVisualizationMetrics(
                        self.msi,
                        rgb_uint8,
                        mask=roi_mask,
                        num_samples=num_samples,
                    ).get_metrics()
                    for k, v in m.items():
                        try:
                            kk = "roi_" + str(k).replace(" ", "_").replace("∞", "inf").replace("%", "pct").lower()
                            fv = float(v)
                            row[kk] = fv
                            agg.setdefault(kk, []).append(fv)
                        except Exception:
                            continue
                except Exception:
                    pass
                component_roi_rows.append(row)
            for k, vals in agg.items():
                if vals:
                    component_roi_summary[f"{k}_mean"] = float(np.mean(vals))
            logger.info(
                "component_roi_metrics complete | rows=%d | summary_keys=%d | time=%.2fs",
                int(len(component_roi_rows)),
                int(len(component_roi_summary)),
                float(time.perf_counter() - t_roi0),
            )

        summary = {
            "cv_mode": float(1 if use_block_cv else 0),
            "n_eval_pixels": int(x_eval.shape[0]),
            "n_components": int(self.n_components),
        }
        if enable_nmf_lr:
            summary.update(
                {
                    "nmf_lr_mean_r2": float(np.mean([d["r2"] for d in per_component])),
                    "nmf_lr_mean_mae": float(np.mean([d["mae"] for d in per_component])),
                    "nmf_lr_mean_rmse": float(np.mean([d["rmse"] for d in per_component])),
                    "nmf_lr_mean_pearson": float(np.mean([d["pearson"] for d in per_component])),
                    "nmf_lr_mean_spearman": float(np.mean([d["spearman"] for d in per_component])),
                    "nmf_edge_corr_mean_spearman": float(np.mean([d["edge_corr_spearman"] for d in per_component])),
                    "nmf_edge_corr_mean_pearson": float(np.mean([d["edge_corr_pearson"] for d in per_component])),
                }
            )
        if enable_mi_knn:
            summary.update(
                {
                    "mi_knn_mean_sum": float(np.nanmean(np.asarray(mi_sums, dtype=np.float64))),
                    "mi_knn_mean_max": float(np.nanmean(np.asarray(mi_maxs, dtype=np.float64))),
                }
            )
        if enable_edge_agreement:
            logger.info("metric: edge_agreement start")
            t_edge0 = time.perf_counter()
            rgb_for_edge = np.asarray(rgb_uint8, dtype=np.uint8)
            if clahe_enabled and bool(clahe_apply_to.get("edge_viz", False)):
                rgb_for_edge = _apply_equalize_rgb_uint8(
                    rgb_for_edge,
                    method=clahe_method,
                    clip_limit=clahe_clip_limit,
                    tile_grid_size=clahe_tile_grid_size,
                )
            viz_edge, viz_edge_linf = _viz_edge_maps_multiscale(
                rgb_for_edge,
                self.valid_mask,
                sigmas=self.edge_multiscale_sigmas,
                aggregation=self.edge_multiscale_aggregation,
                edge_method_cfg=self.edge_method_cfg,
            )
            hd_edge_norm = _normalize_map_percentile(
                self.highd_edge,
                self.valid_mask,
                low_pct=float(self.cfg.spatial_normalization.low_percentile),
                high_pct=float(self.cfg.spatial_normalization.high_percentile),
            )
            hd_edge_linf_norm = _normalize_map_percentile(
                self.highd_edge_linf,
                self.valid_mask,
                low_pct=float(self.cfg.spatial_normalization.low_percentile),
                high_pct=float(self.cfg.spatial_normalization.high_percentile),
            )
            viz_edge_norm = _normalize_map_percentile(
                viz_edge,
                self.valid_mask,
                low_pct=float(self.cfg.spatial_normalization.low_percentile),
                high_pct=float(self.cfg.spatial_normalization.high_percentile),
            )
            viz_edge_linf_norm = _normalize_map_percentile(
                viz_edge_linf,
                self.valid_mask,
                low_pct=float(self.cfg.spatial_normalization.low_percentile),
                high_pct=float(self.cfg.spatial_normalization.high_percentile),
            )
            if clahe_enabled and bool(clahe_apply_to.get("edge_hd", False)):
                hd_edge_norm = _apply_equalize_gray01(
                    hd_edge_norm,
                    self.valid_mask,
                    method=clahe_method,
                    clip_limit=clahe_clip_limit,
                    tile_grid_size=clahe_tile_grid_size,
                )
                hd_edge_linf_norm = _apply_equalize_gray01(
                    hd_edge_linf_norm,
                    self.valid_mask,
                    method=clahe_method,
                    clip_limit=clahe_clip_limit,
                    tile_grid_size=clahe_tile_grid_size,
                )
            if clahe_enabled and bool(clahe_apply_to.get("edge_viz", False)):
                viz_edge_norm = _apply_equalize_gray01(
                    viz_edge_norm,
                    self.valid_mask,
                    method=clahe_method,
                    clip_limit=clahe_clip_limit,
                    tile_grid_size=clahe_tile_grid_size,
                )
                viz_edge_linf_norm = _apply_equalize_gray01(
                    viz_edge_linf_norm,
                    self.valid_mask,
                    method=clahe_method,
                    clip_limit=clahe_clip_limit,
                    tile_grid_size=clahe_tile_grid_size,
                )
            edge_agreement_spearman = _safe_spearman(hd_edge_norm[self.valid_mask], viz_edge_norm[self.valid_mask])
            edge_agreement_pearson = _safe_pearson(hd_edge_norm[self.valid_mask], viz_edge_norm[self.valid_mask])
            edge_agreement_linf_spearman = _safe_spearman(
                hd_edge_linf_norm[self.valid_mask], viz_edge_linf_norm[self.valid_mask]
            )
            try:
                edge_agreement_ssim = float(
                    ssim(
                        hd_edge_norm,
                        viz_edge_norm,
                        data_range=1.0,
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                    )
                )
            except Exception:
                edge_agreement_ssim = float("nan")
            auc_metrics = _edge_auc_metrics(
                hd_edge_norm,
                viz_edge_norm,
                self.valid_mask,
                positive_percentile=float(self.cfg.edge_agreement.auc_positive_percentile),
                min_positive_pixels=int(self.cfg.edge_agreement.auc_min_positive_pixels),
            )
            mp_cfg = getattr(self.cfg.edge_agreement, "multi_percentile", None)
            if mp_cfg is None:
                mp_metrics = {
                    "edge_agreement_auc_roc_weighted": float("nan"),
                    "edge_agreement_auc_pr_weighted": float("nan"),
                    "edge_agreement_auc_multi_valid_percentiles": 0.0,
                }
            else:
                mp_metrics = _edge_auc_metrics_multi_percentile(
                    hd_edge_norm,
                    viz_edge_norm,
                    self.valid_mask,
                    percentiles=[float(v) for v in list(getattr(mp_cfg, "percentiles", [70, 80, 85, 90, 95]))],
                    min_positive_pixels=int(self.cfg.edge_agreement.auc_min_positive_pixels),
                    weight_mode=str(getattr(mp_cfg, "weight_mode", "tail_power")),
                    tail_power=float(getattr(mp_cfg, "tail_power", 2.0)),
                )
            summary.update(
                {
                    "edge_agreement_hd_vs_viz_spearman": edge_agreement_spearman,
                    "edge_agreement_hd_vs_viz_pearson": edge_agreement_pearson,
                    "edge_agreement_hd_vs_viz_l_infinity_spearman": edge_agreement_linf_spearman,
                    "edge_agreement_hd_vs_viz_ssim": edge_agreement_ssim,
                    "edge_agreement_auc_roc": float(auc_metrics["edge_agreement_auc_roc"]),
                    "edge_agreement_auc_pr": float(auc_metrics["edge_agreement_auc_pr"]),
                    "edge_agreement_auc_roc_weighted": float(mp_metrics["edge_agreement_auc_roc_weighted"]),
                    "edge_agreement_auc_pr_weighted": float(mp_metrics["edge_agreement_auc_pr_weighted"]),
                    "edge_agreement_auc_multi_valid_percentiles": float(
                        mp_metrics["edge_agreement_auc_multi_valid_percentiles"]
                    ),
                }
            )
            # Optional thresholded edge agreement (precision/recall/F1), matching debug_edge_maps design.
            eval_cfg = getattr(self.cfg, "edge_evaluation", None)
            eval_enabled = bool(getattr(eval_cfg, "enabled", True)) if eval_cfg is not None else True
            if eval_enabled:
                eval_mode = str(getattr(eval_cfg, "threshold_mode", "otsu")) if eval_cfg is not None else "otsu"
                eval_percentile = float(getattr(eval_cfg, "threshold_percentile", 90.0)) if eval_cfg is not None else 90.0
                eval_value = float(getattr(eval_cfg, "threshold_value", 0.5)) if eval_cfg is not None else 0.5
                eval_include_linf = bool(getattr(eval_cfg, "include_linf", True)) if eval_cfg is not None else True
                eval_f_beta = float(getattr(eval_cfg, "f_beta", 1.0)) if eval_cfg is not None else 1.0
                eval_metric = str(getattr(eval_cfg, "metric", "f1")).strip().lower() if eval_cfg is not None else "f1"
                min_viz_edge_percentile = (
                    float(getattr(eval_cfg, "min_viz_edge_percentile", 0.0)) if eval_cfg is not None else 0.0
                )
                otsu_pre_eq = bool(getattr(eval_cfg, "otsu_pre_equalize", True)) if eval_cfg is not None else True
                otsu_eq_method = str(getattr(eval_cfg, "otsu_equalize_method", "hist")) if eval_cfg is not None else "hist"
                otsu_clahe_clip = (
                    float(getattr(eval_cfg, "otsu_clahe_clip_limit", 2.0)) if eval_cfg is not None else 2.0
                )
                otsu_clahe_grid = (
                    int(getattr(eval_cfg, "otsu_clahe_tile_grid_size", 8)) if eval_cfg is not None else 8
                )

                def _apply_min_viz_edge(arr: np.ndarray) -> np.ndarray:
                    x = np.array(arr, dtype=np.float32, copy=True)
                    if min_viz_edge_percentile > 0 and np.any(self.valid_mask):
                        thresh = float(np.percentile(x[self.valid_mask], min_viz_edge_percentile))
                        x = np.where((self.valid_mask) & (x >= thresh), x, 0.0).astype(np.float32)
                    x[~self.valid_mask] = 0.0
                    return x

                viz_eval = _apply_min_viz_edge(viz_edge_norm)
                viz_linf_eval = _apply_min_viz_edge(viz_edge_linf_norm)

                sq_cont = bool(getattr(eval_cfg, "square_before_continuous_metrics", False)) if eval_cfg is not None else False
                hd_d, viz_d = _edges_for_continuous_metrics(hd_edge_norm, viz_eval, square=sq_cont)
                sp, sr = _compute_soft_precision_recall(hd_d, viz_d, self.valid_mask)
                summary["edge_agreement_soft_precision"] = float(sp)
                summary["edge_agreement_soft_recall"] = float(sr)
                if eval_include_linf:
                    hd_ld, viz_ld = _edges_for_continuous_metrics(hd_edge_linf_norm, viz_linf_eval, square=sq_cont)
                    sp2, sr2 = _compute_soft_precision_recall(hd_ld, viz_ld, self.valid_mask)
                    summary["edge_agreement_linf_soft_precision"] = float(sp2)
                    summary["edge_agreement_linf_soft_recall"] = float(sr2)
                else:
                    summary["edge_agreement_linf_soft_precision"] = float("nan")
                    summary["edge_agreement_linf_soft_recall"] = float("nan")

                if bool(getattr(eval_cfg, "compute_gradient_alignment", True)) if eval_cfg is not None else True:
                    summary["edge_agreement_gradient_alignment"] = float(
                        _compute_weighted_gradient_alignment(hd_edge_norm, viz_eval, self.valid_mask)
                    )
                    if eval_include_linf:
                        summary["edge_agreement_linf_gradient_alignment"] = float(
                            _compute_weighted_gradient_alignment(
                                hd_edge_linf_norm, viz_linf_eval, self.valid_mask
                            )
                        )
                    else:
                        summary["edge_agreement_linf_gradient_alignment"] = float("nan")
                else:
                    summary["edge_agreement_gradient_alignment"] = float("nan")
                    summary["edge_agreement_linf_gradient_alignment"] = float("nan")

                if eval_metric == "continuous_dice":
                    summary["edge_agreement_eval_metric"] = "continuous_dice"
                    summary["edge_agreement_continuous_dice"] = float(
                        _compute_continuous_dice(hd_d, viz_d, self.valid_mask)
                    )
                    summary["edge_agreement_precision"] = float("nan")
                    summary["edge_agreement_recall"] = float("nan")
                    summary["edge_agreement_f1"] = float("nan")
                    summary["edge_agreement_f_beta"] = float("nan")
                    if eval_include_linf:
                        summary["edge_agreement_linf_continuous_dice"] = float(
                            _compute_continuous_dice(hd_ld, viz_ld, self.valid_mask)
                        )
                        summary["edge_agreement_linf_precision"] = float("nan")
                        summary["edge_agreement_linf_recall"] = float("nan")
                        summary["edge_agreement_linf_f1"] = float("nan")
                        summary["edge_agreement_linf_f_beta"] = float("nan")
                    else:
                        summary["edge_agreement_linf_continuous_dice"] = float("nan")
                        summary["edge_agreement_linf_precision"] = float("nan")
                        summary["edge_agreement_linf_recall"] = float("nan")
                        summary["edge_agreement_linf_f1"] = float("nan")
                        summary["edge_agreement_linf_f_beta"] = float("nan")
                else:
                    summary["edge_agreement_eval_metric"] = "f1"
                    summary["edge_agreement_continuous_dice"] = float("nan")
                    hd_bin = _binary_from_threshold(
                        hd_edge_norm,
                        self.valid_mask,
                        eval_mode,
                        eval_percentile,
                        eval_value,
                        otsu_pre_equalize=otsu_pre_eq,
                        otsu_equalize_method=otsu_eq_method,
                        otsu_clahe_clip_limit=otsu_clahe_clip,
                        otsu_clahe_tile_grid_size=otsu_clahe_grid,
                    )
                    viz_bin = _binary_from_threshold(
                        viz_eval,
                        self.valid_mask,
                        eval_mode,
                        eval_percentile,
                        eval_value,
                        otsu_pre_equalize=otsu_pre_eq,
                        otsu_equalize_method=otsu_eq_method,
                        otsu_clahe_clip_limit=otsu_clahe_clip,
                        otsu_clahe_tile_grid_size=otsu_clahe_grid,
                    )
                    p, r, f1, f_b = _compute_prf1(hd_bin[self.valid_mask], viz_bin[self.valid_mask], f_beta=eval_f_beta)
                    summary["edge_agreement_precision"] = float(p)
                    summary["edge_agreement_recall"] = float(r)
                    summary["edge_agreement_f1"] = float(f1)
                    summary["edge_agreement_f_beta"] = float(f_b)
                    if eval_include_linf:
                        hd_linf_bin = _binary_from_threshold(
                            hd_edge_linf_norm,
                            self.valid_mask,
                            eval_mode,
                            eval_percentile,
                            eval_value,
                            otsu_pre_equalize=otsu_pre_eq,
                            otsu_equalize_method=otsu_eq_method,
                            otsu_clahe_clip_limit=otsu_clahe_clip,
                            otsu_clahe_tile_grid_size=otsu_clahe_grid,
                        )
                        viz_linf_bin = _binary_from_threshold(
                            viz_linf_eval,
                            self.valid_mask,
                            eval_mode,
                            eval_percentile,
                            eval_value,
                            otsu_pre_equalize=otsu_pre_eq,
                            otsu_equalize_method=otsu_eq_method,
                            otsu_clahe_clip_limit=otsu_clahe_clip,
                            otsu_clahe_tile_grid_size=otsu_clahe_grid,
                        )
                        p2, r2, f12, f_b2 = _compute_prf1(
                            hd_linf_bin[self.valid_mask], viz_linf_bin[self.valid_mask], f_beta=eval_f_beta
                        )
                        summary["edge_agreement_linf_precision"] = float(p2)
                        summary["edge_agreement_linf_recall"] = float(r2)
                        summary["edge_agreement_linf_f1"] = float(f12)
                        summary["edge_agreement_linf_f_beta"] = float(f_b2)
                        summary["edge_agreement_linf_continuous_dice"] = float("nan")
            for k, v in mp_metrics.items():
                if k in summary:
                    continue
                try:
                    summary[str(k)] = float(v)
                except Exception:
                    continue
            edge_maps_for_output = {
                "hd_edge_norm": np.asarray(hd_edge_norm, dtype=np.float32),
                "hd_edge_linf_norm": np.asarray(hd_edge_linf_norm, dtype=np.float32),
                "viz_edge_norm": np.asarray(viz_edge_norm, dtype=np.float32),
                "viz_edge_linf_norm": np.asarray(viz_edge_linf_norm, dtype=np.float32),
            }
            logger.info(
                "edge_agreement complete | auc_roc=%.4f | auc_pr=%.4f | time=%.2fs",
                float(summary.get("edge_agreement_auc_roc", float("nan"))),
                float(summary.get("edge_agreement_auc_pr", float("nan"))),
                float(time.perf_counter() - t_edge0),
            )
        else:
            edge_maps_for_output = {}
        if enable_correlation:
            logger.info("metric: correlation start")
            t_corr0 = time.perf_counter()
            corr_metrics = _compute_correlation_metrics_multi_space(
                msi=self.msi,
                rgb_uint8=rgb_uint8,
                valid_mask=self.valid_mask,
                corr_cfg=getattr(self.cfg, "correlation", None),
            )
            summary.update(corr_metrics)
            logger.info(
                "correlation metrics complete | n_metrics=%d | time=%.2fs",
                int(len(corr_metrics)),
                float(time.perf_counter() - t_corr0),
            )
        summary.update(nmf_kl_summary)
        summary.update(component_roi_summary)
        result = {
            "summary": summary,
            "per_component": per_component,
            "quantiles": quantile_summary,
            "frequency_bins": frequency_bins,
            "component_roi_metrics": component_roi_rows,
        }
        if return_edge_maps and edge_maps_for_output:
            result["edge_maps"] = edge_maps_for_output
        logger.info("evaluate_visualization done | total_time=%.2fs", float(time.perf_counter() - t_eval0))
        return result, component_maps


def _save_component_debug_panels(
    output_dir: Path,
    viz_stem: str,
    component_maps: List[Dict[str, np.ndarray]],
    valid_mask: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in component_maps:
        comp_idx = int(np.asarray(entry["component"]).reshape(-1)[0])
        y_true_map = np.asarray(entry["y_true_map"], dtype=np.float32)
        y_pred_map = np.asarray(entry["y_pred_map"], dtype=np.float32)
        true_edge_map = np.asarray(entry["true_edge_map"], dtype=np.float32)
        pred_edge_map = np.asarray(entry["pred_edge_map"], dtype=np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        panels = [
            ("NMF Ground Truth", y_true_map),
            ("NMF Predicted", y_pred_map),
            ("GT Edge Magnitude", true_edge_map),
            ("Pred Edge Magnitude", pred_edge_map),
        ]
        for ax, (title, arr) in zip(axes.ravel(), panels):
            masked = np.ma.array(arr, mask=~valid_mask)
            im = ax.imshow(masked, cmap="viridis")
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f"{viz_stem} | component {comp_idx}", fontsize=12)
        panel_path = output_dir / f"{viz_stem}__component_{comp_idx:02d}.png"
        fig.savefig(panel_path, dpi=180)
        plt.close(fig)


def _save_component_comparison_panels(
    output_dir: Path,
    components_by_viz: Dict[str, List[Dict[str, np.ndarray]]],
    viz_images_by_viz: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> None:
    if not components_by_viz:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    def _viz_sort_key(name: str) -> Tuple[int, str]:
        n = name.lower()
        if "mics" in n:
            return (0, n)
        if "umap" in n:
            return (1, n)
        if "pca" in n:
            return (2, n)
        return (3, n)

    viz_names = sorted(list(components_by_viz.keys()), key=_viz_sort_key)
    min_components = min(
        int(len(entries)) for entries in components_by_viz.values() if entries is not None
    )
    if min_components <= 0:
        return

    for comp_idx in range(min_components):
        n_rows = len(viz_names)
        n_cols = 5
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.0 * n_cols, 3.2 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )
        for row_idx, viz_name in enumerate(viz_names):
            entry = components_by_viz[viz_name][comp_idx]
            y_true_map = np.asarray(entry["y_true_map"], dtype=np.float32)
            y_pred_map = np.asarray(entry["y_pred_map"], dtype=np.float32)
            true_edge_map = np.asarray(entry["true_edge_map"], dtype=np.float32)
            pred_edge_map = np.asarray(entry["pred_edge_map"], dtype=np.float32)
            viz_rgb = np.asarray(viz_images_by_viz[viz_name], dtype=np.uint8)
            nmf_s = float(np.asarray(entry.get("nmf_spearman", np.array([np.nan], dtype=np.float32))).reshape(-1)[0])
            edge_s = float(np.asarray(entry.get("edge_spearman", np.array([np.nan], dtype=np.float32))).reshape(-1)[0])
            metric_text = f"Spearman NMF={nmf_s:.3f} | Edge={edge_s:.3f}"
            panels = [
                ("Visualization RGB", viz_rgb),
                ("NMF Ground Truth", y_true_map),
                ("NMF Predicted", y_pred_map),
                ("GT Edge Magnitude", true_edge_map),
                ("Pred Edge Magnitude", pred_edge_map),
            ]
            for col_idx, (title, arr) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                if arr.ndim == 3:
                    disp = arr.copy()
                    disp[~valid_mask] = 0
                    im = ax.imshow(disp)
                else:
                    masked = np.ma.array(arr, mask=~valid_mask)
                    im = ax.imshow(masked, cmap="viridis")
                row_title = f"{viz_name}\n{metric_text}\n{title}" if col_idx == 0 else title
                ax.set_title(row_title, fontsize=10)
                ax.axis("off")
                if arr.ndim != 3:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        fig.suptitle(f"Component comparison | component {comp_idx}", fontsize=13)
        out_path = output_dir / f"component_{comp_idx:02d}__all_visualizations.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def _write_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    import csv

    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_main_metrics(result: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    summary = result.get("summary", {})
    for k, v in summary.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue

    for bin_row in result.get("frequency_bins", []):
        bin_name = str(bin_row.get("bin", "unknown"))
        for k, v in bin_row.items():
            if k == "bin":
                continue
            key = f"bin_{bin_name}__{k}"
            try:
                out[key] = float(v)
            except Exception:
                continue
    return out


def _write_transposed_metrics_csv(path: Path, metrics_by_viz: Dict[str, Dict[str, float]]) -> None:
    if not metrics_by_viz:
        return
    viz_names = list(metrics_by_viz.keys())
    metric_keys: List[str] = []
    for m in metrics_by_viz.values():
        for k in m.keys():
            if k not in metric_keys:
                metric_keys.append(k)

    rows: List[Dict[str, object]] = []
    for mk in metric_keys:
        row: Dict[str, object] = {"metric": mk}
        for viz_name in viz_names:
            row[viz_name] = metrics_by_viz[viz_name].get(mk, float("nan"))
        rows.append(row)
    _write_csv(path, rows)


@hydra.main(version_base=None, config_path="configs", config_name="debug_visualization_metrics")
def main(cfg: DictConfig) -> None:
    log_cfg = getattr(cfg, "logging", None)
    level_name = str(getattr(log_cfg, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    npy_dir_r = OmegaConf.select(cfg, "input.visualization_npy_dir")
    png_dir_r = OmegaConf.select(cfg, "input.visualization_png_dir")
    vpaths = OmegaConf.select(cfg, "input.visualization_paths")
    pairing = str(OmegaConf.select(cfg, "input.visualization_pairing") or "").strip().lower()
    try:
        n_vpaths = len(vpaths) if vpaths is not None else 0
    except TypeError:
        n_vpaths = 1 if vpaths else 0
    use_shape_pairing = pairing in ("by_shape", "shape", "size", "by_size")
    if not use_shape_pairing and _pairing_unset_for_inference(pairing) and npy_dir_r and png_dir_r and n_vpaths == 0:
        use_shape_pairing = True
        logger.info(
            "visualization_pairing inferred as by_shape (both npy/png dirs set, visualization_paths empty)",
        )
    npy_dir_pair: Path | None = None
    png_dir_pair: Path | None = None
    viz_paths: List[Path] | None = None
    if use_shape_pairing:
        if not npy_dir_r or not png_dir_r:
            raise ValueError(
                "Shape pairing requires input.visualization_npy_dir and input.visualization_png_dir "
                "(both non-empty), or set input.visualization_paths instead.",
            )
        npy_dir_pair = Path(str(npy_dir_r))
        png_dir_pair = Path(str(png_dir_r))
        if not npy_dir_pair.is_dir():
            raise FileNotFoundError(f"visualization_npy_dir is not a directory: {npy_dir_pair}")
        if not png_dir_pair.is_dir():
            raise FileNotFoundError(f"visualization_png_dir is not a directory: {png_dir_pair}")
    else:
        if not vpaths or n_vpaths == 0:
            raise ValueError(
                "input.visualization_paths must contain at least one file path, "
                "or set both input.visualization_npy_dir and input.visualization_png_dir for shape pairing "
                "(optionally set visualization_pairing=by_shape explicitly).",
            )
        viz_paths = [Path(str(p)) for p in vpaths]
        for p in viz_paths:
            if not p.exists():
                raise FileNotFoundError(f"Visualization file not found: {p}")

    npy_raw = OmegaConf.select(cfg, "input.npy_path")
    npy_path: Path
    if npy_raw is not None and str(npy_raw).strip() != "":
        npy_path = Path(str(npy_raw))
    elif use_shape_pairing and npy_dir_pair is not None and png_dir_pair is not None:
        npy_path = _pick_msi_npy_for_shape_pairing(
            npy_dir_pair,
            png_dir_pair,
            transpose_msi=bool(cfg.data.transpose_msi),
        )
    else:
        raise ValueError(
            "input.npy_path is required unless you use shape pairing with visualization_npy_dir + "
            "visualization_png_dir (MSI auto-picked from a .npy that matches a .png size).",
        )
    if not npy_path.exists():
        raise FileNotFoundError(f"MSI file not found: {npy_path}")

    output_dir = Path(str(cfg.output.out_dir)) if cfg.output.out_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output_dir=%s", output_dir)
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    normalization_raw = getattr(cfg.data, "normalization", None)
    if normalization_raw is None:
        # Backward compatibility with older configs.
        normalization_raw = bool(getattr(cfg.data, "tic_normalize", False))
    normalization = _resolve_normalization_mode(normalization_raw)

    msi = _load_msi(
        npy_path,
        transpose_msi=bool(cfg.data.transpose_msi),
        normalization=normalization,
    )
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty.")
    requested = _metrics_requested_set(cfg)
    nmf_free_mode = _is_nmf_free_mode(requested)
    evaluator: DebugNMFEvaluator | None = None
    if not nmf_free_mode:
        evaluator = DebugNMFEvaluator(msi, cfg)
        valid_mask = evaluator.valid_mask
    logger.info(
        "loaded MSI | shape=%s | valid_pixels=%d | nmf_free_mode=%s",
        tuple(msi.shape),
        int(valid_mask.sum()),
        bool(nmf_free_mode),
    )

    if use_shape_pairing:
        assert npy_dir_pair is not None and png_dir_pair is not None
        report = _build_npy_png_shape_matching_report(
            npy_dir_pair,
            png_dir_pair,
            msi_npy_path=npy_path,
            target_hw=(int(msi.shape[0]), int(msi.shape[1])),
            transpose_msi=bool(cfg.data.transpose_msi),
        )
        print(report, flush=True)
        logger.info("%s", report)
        try:
            (output_dir / "shape_matching_report.txt").write_text(report, encoding="utf-8")
            logger.info("Wrote shape_matching_report.txt under %s", output_dir)
        except OSError as exc:
            logger.warning("Could not write shape_matching_report.txt: %s", exc)

        viz_paths = _pair_npy_png_paths_by_shape(
            npy_dir_pair,
            png_dir_pair,
            target_hw=(int(msi.shape[0]), int(msi.shape[1])),
            transpose_msi=bool(cfg.data.transpose_msi),
        )
        if not viz_paths:
            raise ValueError(
                "No .npy/.png pairs at MSI spatial size. "
                f"MSI is {msi.shape[0]}x{msi.shape[1]}; the viz directories had no matching pair at that "
                f"(H,W). Align input.npy_path with the slide/grid used in the viz folders, or set "
                f"input.allow_resize=true with explicit input.visualization_paths.",
            )
        n_pairs = len(viz_paths) // 2
        logger.info(
            "visualization_pairing=by_shape | %d pairs (%d files) at MSI size %dx%d from %s and %s",
            n_pairs,
            len(viz_paths),
            int(msi.shape[0]),
            int(msi.shape[1]),
            npy_dir_pair,
            png_dir_pair,
        )
        for p in viz_paths:
            if not p.exists():
                raise FileNotFoundError(f"Visualization file not found: {p}")

        n_pair_files = len(viz_paths)
        viz_paths = [p for p in viz_paths if p.suffix.lower() != ".npy"]
        if len(viz_paths) < n_pair_files:
            logger.info(
                "by_shape: removed %d paired .npy path(s) from viz list; metrics use image exports only",
                n_pair_files - len(viz_paths),
            )

    assert viz_paths is not None

    allow_resize = bool(getattr(cfg.input, "allow_resize", False))
    if not allow_resize:
        tgt_hw = (int(msi.shape[0]), int(msi.shape[1]))
        n_before = len(viz_paths)
        viz_paths = _filter_viz_paths_to_msi_shape(viz_paths, tgt_hw)
        if len(viz_paths) < n_before:
            logger.info(
                "dropped %d visualization(s) not matching MSI size %dx%d",
                n_before - len(viz_paths),
                tgt_hw[0],
                tgt_hw[1],
            )
        if not viz_paths:
            raise ValueError(
                "No visualizations left at MSI spatial size with allow_resize=false. "
                f"MSI is {tgt_hw[0]}x{tgt_hw[1]}. Point input.npy_path at the same slide/grid as the viz files, "
                "or set input.allow_resize=true.",
            )

    summary_rows: List[Dict[str, object]] = []
    metrics_by_viz: Dict[str, Dict[str, float]] = {}
    component_maps_by_viz: Dict[str, List[Dict[str, np.ndarray]]] = {}
    viz_images_by_viz: Dict[str, np.ndarray] = {}
    for viz_path in viz_paths:
        t_viz0 = time.perf_counter()
        logger.info("processing visualization: %s", viz_path)
        viz_rgb = _load_visualization(
            viz_path,
            target_hw=(msi.shape[0], msi.shape[1]),
            allow_resize=allow_resize,
        )
        if nmf_free_mode:
            t_corr0 = time.perf_counter()
            summary = {
                "n_eval_pixels": int(valid_mask.sum()),
                "n_components": 0,
            }
            corr_metrics: Dict[str, float] = {}
            if _metric_enabled(requested, "correlation"):
                logger.info("metric: correlation start (nmf-free)")
                corr_metrics = _compute_correlation_metrics_multi_space(
                    msi,
                    viz_rgb,
                    valid_mask,
                    getattr(cfg, "correlation", None),
                )
                summary.update(corr_metrics)
                logger.info("metric: correlation complete (nmf-free)")
            logger.info("metric: color_diversity start")
            summary.update(_color_diversity_metrics(viz_rgb, valid_mask))
            logger.info("metric: color_diversity complete")
            result = {
                "summary": summary,
                "per_component": [],
                "quantiles": [],
                "frequency_bins": [],
                "component_roi_metrics": [],
            }
            component_maps = []
            logger.info(
                "nmf-free metrics complete | n_metrics=%d | time=%.2fs",
                int(len(summary)),
                float(time.perf_counter() - t_corr0),
            )
        else:
            assert evaluator is not None
            result, component_maps = evaluator.evaluate_visualization(viz_rgb)
            # Add visual color-diversity diagnostics to make structure-vs-color mismatches explicit.
            try:
                summary_obj = result.get("summary", {})
                if isinstance(summary_obj, dict):
                    summary_obj.update(_color_diversity_metrics(viz_rgb, valid_mask))
            except Exception:
                pass
        stem = viz_path.stem
        _write_json(output_dir / f"{stem}__metrics.json", result)
        component_maps_by_viz[stem] = component_maps
        viz_images_by_viz[stem] = viz_rgb
        if bool(getattr(cfg.output, "save_per_method_component_panels", False)):
            panel_dir = output_dir / str(getattr(cfg.output, "component_panels_subdir", "component_panels"))
            _save_component_debug_panels(panel_dir, stem, component_maps, valid_mask)

        per_component_rows = []
        for d in result["per_component"]:
            row = {"visualization": str(viz_path)}
            row.update(d)
            per_component_rows.append(row)
        _write_csv(output_dir / f"{stem}__per_component.csv", per_component_rows)

        quantile_rows = []
        for d in result["quantiles"]:
            row = {"visualization": str(viz_path)}
            row.update(d)
            quantile_rows.append(row)
        _write_csv(output_dir / f"{stem}__quantiles.csv", quantile_rows)

        freq_rows = []
        for d in result.get("frequency_bins", []):
            row = {"visualization": str(viz_path)}
            row.update(d)
            freq_rows.append(row)
        _write_csv(output_dir / f"{stem}__frequency_bins.csv", freq_rows)

        comp_roi_rows = []
        for d in result.get("component_roi_metrics", []):
            row = {"visualization": str(viz_path)}
            row.update(d)
            comp_roi_rows.append(row)
        _write_csv(output_dir / f"{stem}__component_roi_metrics.csv", comp_roi_rows)

        main_metrics = _flatten_main_metrics(result)
        metrics_by_viz[stem] = main_metrics
        summary_row = {"visualization": str(viz_path), "visualization_id": stem}
        summary_row.update(main_metrics)
        summary_rows.append(summary_row)
        print(f"[ok] metrics computed for {viz_path}")
        logger.info(
            "finished visualization: %s | total_time=%.2fs | n_main_metrics=%d",
            viz_path,
            float(time.perf_counter() - t_viz0),
            int(len(main_metrics)),
        )

    _write_csv(output_dir / "summary_metrics.csv", summary_rows)
    _write_transposed_metrics_csv(output_dir / "summary_metrics_transposed.csv", metrics_by_viz)
    if bool(getattr(cfg.output, "save_component_panels", True)):
        panel_dir = output_dir / str(getattr(cfg.output, "component_panels_subdir", "component_panels"))
        t_panel0 = time.perf_counter()
        _save_component_comparison_panels(
            panel_dir,
            component_maps_by_viz,
            viz_images_by_viz,
            valid_mask,
        )
        logger.info("saved comparison component panels to %s | time=%.2fs", panel_dir, float(time.perf_counter() - t_panel0))
    print(f"Saved summary: {output_dir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()

