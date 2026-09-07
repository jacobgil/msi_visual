#!/usr/bin/env python3
"""
Rank visualizations by edge agreement vs HD edges: F1 (binarized), continuous Dice, or
experimental gradient-SSIM (SSIM of gx, gy, |g| from MSI scalar vs viz luminance, averaged).
Optional weighted composite (``evaluation.composite``): continuous Dice plus luminance contrast /
LAB color-variety metrics; ``composite.normalize: average_rank`` maps tie-aware ranks to [0,1] per metric,
or use ``minmax`` / ``raw`` — weight × scaled score summed into ``composite_score``.
HD reference edges: ``edge_detection.hd_method`` including ``top3_nmf_max`` / ``top3_nmf_mean``
(TOP3+NMF stack on the full MSI cube; configure ``top3``, ``nmf``, ``equalize``, ``edge_pre_smooth``,
``edge_maps``, ``spatial_normalization`` like ``gallery_top3_nmf_edges``).

Debug PNGs under viz_edges_debug/ when debug.save_viz_edges is true: ``{stem}__triple.png`` with three
panels (viz | viz edges | HD edges + score); if ``debug.save_edge_equalized`` is true, also
``{stem}__triple_eq.png`` with the same layout and histogram-equalized viz/HD edge panels (same as
``hd_edge_norm_eq``). If ``debug.local_dice_window > 0``, a fourth panel shows a local soft-Dice heatmap.
Top/bottom galleries are visualization-only thumbnails.

Usage:
  python scripts/rank_visualizations_by_f1.py
  python scripts/rank_visualizations_by_f1.py input.npy_path=path/to/data.npy input.visualization_folder=path/to/viz/
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

# Ensure script dir (containing debug_edge_maps) is on path
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from debug_edge_maps import (
    _add_agreement_legend,
    _aggregate_multiscale_maps,
    _agreement_overlay_rgb,
    _apply_equalize_rgb_uint8,
    _apply_hd_edge_power,
    _binary_from_threshold,
    _compute_continuous_dice,
    _compute_gradient_ssim_mean,
    _compute_soft_precision_recall,
    _compute_prf1,
    _compute_viz_contrast_color_metrics,
    _edges_for_continuous_metrics,
    _edge_maps_multiscale,
    _equalize_gray_u8,
    _gaussian_blur,
    _load_msi,
    _load_visualization,
    _luminance_from_rgb01,
    _normalize_map_percentile,
    _normalize_on_mask,
    _parse_edge_detection_cfg,
    _pca_project_msi_for_edges,
    _resolve_normalization_mode,
    _gray_u8_to_rgb_colormap,
    _save_map_png_and_npy,
    _save_side_by_side_rgb,
    _to_uint8_gray01,
)

logger = logging.getLogger(__name__)

SUPPORTED_VIZ_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".npy"}


def _local_dice_heatmap_f32(
    viz_bin: np.ndarray,
    hd_bin: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
) -> np.ndarray:
    """Per-pixel local Dice from k×k box sums: 2·Σ(v·h)/(Σv+Σh). Three box sums, O(HW).

    **Not** the same quantity as the headline F1/Dice: that is one global ratio
    ``2·TP/(2·TP+FP+FN)`` (equivalently ``2|A∩B|/(|A|+|B|)`` on binarized edges over ``valid_mask``).
    Here each pixel is a *different* neighborhood ratio; ``mean(heatmap)`` over the image is generally
    **not** equal to global Dice (overlapping windows, mean of ratios ≠ ratio of totals, and
    off-edge pixels are 0).

    When ``evaluation.metric`` is ``continuous_dice`` or ``gradient_ssim``, the score bar still shows
    that metric; this panel is always from **binarized** edges for visualization.
    """
    import cv2

    k = max(1, int(window))
    a = np.where(valid_mask, np.asarray(viz_bin, dtype=np.float32), 0.0)
    b = np.where(valid_mask, np.asarray(hd_bin, dtype=np.float32), 0.0)
    ab = a * b
    ksize = (k, k)
    sa = cv2.boxFilter(a, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    sb = cv2.boxFilter(b, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    sab = cv2.boxFilter(ab, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    eps = 1e-8
    denom = sa + sb + eps
    dice = (2.0 * sab / denom).astype(np.float32)
    # Windows with no edge mass in either map: treat as undefined for display — not 1.0.
    # Using 1.0 here made almost the whole image bright (sparse edges → most windows empty)
    # while global Dice stayed low.
    empty = (sa + sb) <= eps
    dice = np.where(empty, 0.0, dice)
    dice = np.where(valid_mask, dice, 0.0)
    dice = np.clip(dice, 0.0, 1.0)
    return dice


def _local_dice_summary_stats(
    local_dice: np.ndarray, valid_mask: np.ndarray
) -> tuple[float, float]:
    """Mean of local-Dice heatmap on ``valid_mask``, and mean only where heatmap > 0 (non-degenerate windows)."""
    vm = np.asarray(valid_mask, dtype=bool)
    h = np.asarray(local_dice, dtype=np.float32)
    vals = h[vm]
    if vals.size == 0:
        return float("nan"), float("nan")
    mean_all = float(np.mean(vals))
    pos = vals > 1e-6
    mean_pos = float(np.mean(vals[pos])) if np.any(pos) else float("nan")
    return mean_all, mean_pos


def _local_continuous_dice_map(
    hd_edge: np.ndarray,
    viz_edge: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
) -> np.ndarray:
    """Patch-centered continuous Dice at each pixel via box sums."""
    import cv2

    k = max(1, int(window))
    m = np.asarray(valid_mask, dtype=bool)
    h = np.where(m, np.asarray(hd_edge, dtype=np.float32), 0.0)
    v = np.where(m, np.asarray(viz_edge, dtype=np.float32), 0.0)
    ksize = (k, k)
    sh = cv2.boxFilter(h, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    sv = cv2.boxFilter(v, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    shv = cv2.boxFilter(h * v, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    denom = sh + sv + 1e-8
    out = (2.0 * shv / denom).astype(np.float32)
    out = np.where((sh + sv) > 1e-8, out, 0.0)
    out = np.where(m, out, 0.0)
    return np.clip(out, 0.0, 1.0)


def _local_luminance_rms_map(
    viz_rgb: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
) -> np.ndarray:
    """Patch-centered luminance RMS contrast (local std) at each pixel."""
    import cv2

    k = max(1, int(window))
    m = np.asarray(valid_mask, dtype=bool)
    y = _luminance_from_rgb01(viz_rgb).astype(np.float32)
    y = np.where(m, y, 0.0)
    mf = m.astype(np.float32)
    ksize = (k, k)
    cnt = cv2.boxFilter(mf, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    sy = cv2.boxFilter(y, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    sy2 = cv2.boxFilter(y * y, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    mean = sy / np.maximum(cnt, 1.0)
    var = np.maximum(sy2 / np.maximum(cnt, 1.0) - mean * mean, 0.0)
    out = np.sqrt(var).astype(np.float32)
    out = np.where((cnt > 1.0) & m, out, 0.0)
    return out


def _patch_starts(length: int, patch: int, stride: int) -> list[int]:
    n = max(1, int(length))
    p = max(1, int(patch))
    s = max(1, int(stride))
    starts = list(range(0, n, s))
    if not starts or starts[-1] + p < n:
        starts.append(max(0, n - p))
    return sorted(set(starts))


def _local_lab_chroma_entropy_map(
    viz_rgb: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
    *,
    bins_a: int = 16,
    bins_b: int = 16,
    stride: int | None = None,
    min_valid_fraction: float = 0.25,
) -> np.ndarray:
    """Patch-wise LAB a,b joint entropy (same definition as global ``lab_chroma_entropy``)."""
    import cv2

    H, W = valid_mask.shape
    patch = max(1, int(window))
    st = max(1, int(stride)) if stride is not None else max(1, patch // 4)
    ba, bb = max(2, int(bins_a)), max(2, int(bins_b))
    ent_max = float(np.log(ba * bb))

    viz = np.asarray(viz_rgb)
    if np.issubdtype(viz.dtype, np.integer):
        rgb_u8 = np.clip(viz[..., :3], 0, 255).astype(np.uint8)
    else:
        from debug_edge_maps import _to_uint8_rgb

        rgb_u8 = _to_uint8_rgb(viz)[..., :3]
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    a_idx = np.clip((lab[..., 1].astype(np.int32) * ba) // 256, 0, ba - 1)
    b_idx = np.clip((lab[..., 2].astype(np.int32) * bb) // 256, 0, bb - 1)
    joint = (a_idx * bb + b_idx).astype(np.int32)

    acc = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    min_valid = max(1, int(round(patch * patch * float(min_valid_fraction))))

    for y0 in _patch_starts(H, patch, st):
        y1 = min(H, y0 + patch)
        for x0 in _patch_starts(W, patch, st):
            x1 = min(W, x0 + patch)
            m = np.asarray(valid_mask[y0:y1, x0:x1], dtype=bool)
            if int(m.sum()) < min_valid:
                continue
            vals = joint[y0:y1, x0:x1][m]
            hist = np.bincount(vals.ravel(), minlength=ba * bb).astype(np.float64)
            total = float(hist.sum())
            if total <= 0:
                continue
            p = hist[hist > 0] / total
            ent = float(-np.sum(p * np.log(p + 1e-15)))
            ent_norm = float(np.clip(ent / max(ent_max, 1e-12), 0.0, 1.0))
            acc[y0:y1, x0:x1] += ent_norm
            count[y0:y1, x0:x1] += 1.0

    covered = (count > 0) & np.asarray(valid_mask, dtype=bool)
    out = np.zeros((H, W), dtype=np.float32)
    out[covered] = (acc[covered] / count[covered]).astype(np.float32)
    return out


def _minmax_maps_on_mask(maps: list[np.ndarray], valid_mask: np.ndarray) -> list[np.ndarray]:
    vals = []
    m = np.asarray(valid_mask, dtype=bool)
    for arr in maps:
        x = np.asarray(arr, dtype=np.float32)
        v = x[m]
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return [np.zeros_like(arr, dtype=np.float32) for arr in maps]
    all_vals = np.concatenate(vals)
    lo, hi = float(all_vals.min()), float(all_vals.max())
    if hi <= lo + 1e-12:
        return [np.where(m, 0.5, 0.0).astype(np.float32) for arr in maps]
    return [np.where(m, np.clip((np.asarray(arr, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0), 0.0).astype(np.float32) for arr in maps]


def _per_location_rank_score_maps(maps: list[np.ndarray], valid_mask: np.ndarray) -> list[np.ndarray]:
    """Rank images independently at each pixel/location. Higher raw value gets higher score in [0, 1]."""
    if not maps:
        return []
    m = np.asarray(valid_mask, dtype=bool)
    stack = np.stack([np.asarray(arr, dtype=np.float32) for arr in maps], axis=0)
    n = int(stack.shape[0])
    if n == 1:
        out = np.where(m[None, :, :], 1.0, 0.0).astype(np.float32)
        return [out[0]]
    stack = np.where(m[None, :, :], stack, -np.inf)
    order = np.argsort(-stack, axis=0, kind="mergesort")
    scores = np.zeros_like(stack, dtype=np.float32)
    pos_scores = (1.0 - (np.arange(n, dtype=np.float32) / float(n - 1))).reshape(n, 1, 1)
    np.put_along_axis(scores, order, pos_scores, axis=0)
    scores[:, ~m] = 0.0
    return [scores[i].astype(np.float32, copy=False) for i in range(n)]


_RANK_STAGES = 10

_COMPOSITE_KEYS = (
    "continuous_dice",
    "luminance_rms_contrast",
    "luminance_gradient_mean",
    "lab_chroma_entropy",
    "lab_mean_chroma",
)


def _composite_normalize_mode(cfg: dict[str, Any]) -> str:
    """average_rank | minmax | raw — resolves legacy ``normalize_batch`` when ``normalize`` unset."""
    raw = str(cfg.get("normalize", "") or "").strip().lower()
    aliases_rank = {"average_rank", "avg_rank", "rank"}
    aliases_minmax = {"minmax", "min_max"}
    aliases_raw = {"raw", "none", "off"}
    if raw in aliases_rank:
        return "average_rank"
    if raw in aliases_minmax:
        return "minmax"
    if raw in aliases_raw:
        return "raw"
    if raw:
        return "minmax"
    return "raw" if not bool(cfg.get("normalize_batch", True)) else "minmax"


def _scores_from_average_ranks_higher_better(col: np.ndarray) -> np.ndarray:
    """Per-column scores in [0, 1]: tie-aware average ranks among finite entries (higher raw → higher score)."""
    n = int(col.shape[0])
    out = np.full(n, np.nan, dtype=np.float64)
    fin = np.isfinite(col)
    if not np.any(fin):
        return out
    idx = np.nonzero(fin)[0]
    sub = col[fin].astype(np.float64, copy=False)
    order = np.argsort(-sub, kind="mergesort")
    sorted_vals = sub[order]
    ranks_ordered = np.empty_like(sorted_vals, dtype=np.float64)
    ns = int(sorted_vals.shape[0])
    i = 0
    while i < ns:
        j = i
        while j + 1 < ns and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks_ordered[i : j + 1] = avg_rank
        i = j + 1
    inv = np.empty_like(order)
    inv[order] = np.arange(ns, dtype=np.int64)
    full_ranks = ranks_ordered[inv]
    denom = float(max(ns - 1, 1))
    scores_sub = 1.0 - (full_ranks - 1.0) / denom
    out[idx] = scores_sub
    return out


def _apply_composite_scores(rows: list[dict[str, Any]], composite_cfg: dict[str, Any] | None) -> None:
    """Set ``composite_score`` on each row from weighted components (min-max, tie-average ranks, or raw)."""
    if not composite_cfg or not bool(composite_cfg.get("enabled")):
        for r in rows:
            r["composite_score"] = float("nan")
        return
    weights_raw = composite_cfg.get("weights") or {}
    wvec = np.array([float(weights_raw.get(k, 0) or 0) for k in _COMPOSITE_KEYS], dtype=np.float64)
    if float(np.sum(np.abs(wvec))) < 1e-15:
        for r in rows:
            r["composite_score"] = float("nan")
        return
    norm_mode = _composite_normalize_mode(composite_cfg)
    mat = np.array([[float(r.get(k, np.nan)) for k in _COMPOSITE_KEYS] for r in rows], dtype=np.float64)
    normed = np.empty_like(mat)
    for j, _k in enumerate(_COMPOSITE_KEYS):
        col = mat[:, j]
        if abs(wvec[j]) < 1e-15:
            normed[:, j] = 0.0
            continue
        if not np.any(np.isfinite(col)):
            normed[:, j] = 0.5
            continue
        if norm_mode == "average_rank":
            normed[:, j] = _scores_from_average_ranks_higher_better(col)
        elif norm_mode == "minmax":
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
            if hi <= lo + 1e-12:
                normed[:, j] = 0.5
            else:
                normed[:, j] = np.clip((col - lo) / (hi - lo), 0.0, 1.0)
        else:
            normed[:, j] = col
    for i, r in enumerate(rows):
        total = 0.0
        ok = True
        for j, _k in enumerate(_COMPOSITE_KEYS):
            if abs(wvec[j]) < 1e-15:
                continue
            v = float(normed[i, j])
            raw = float(mat[i, j])
            if not np.isfinite(v) or not np.isfinite(raw):
                ok = False
                break
            total += float(wvec[j]) * v
        r["composite_score"] = float(total) if ok else float("nan")


def _metric_label_for_eval(eval_metric: str) -> str:
    m = str(eval_metric).strip().lower()
    if m == "continuous_dice":
        return "Dice"
    if m in ("gradient_ssim", "grad_ssim"):
        return "GradSSIM"
    return "F1"


def _stage(n: int, msg: str) -> None:
    """Print progress to stdout (always visible)."""
    print(f"[rank_visualizations_by_f1 {n}/{_RANK_STAGES}] {msg}", flush=True)


def _find_visualizations(folder: Path, skip_eq: bool = False) -> list[Path]:
    paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIZ_EXT
    ]
    if skip_eq:
        to_skip = [p for p in paths if "eq" in p.stem.lower()]
        paths = [p for p in paths if "eq" not in p.stem.lower()]
        if to_skip:
            logger.info("Skipping %d images with 'eq' in filename", len(to_skip))
    return sorted(paths, key=lambda p: p.name.lower())


def _compute_viz_f1_worker(
    viz_path: Path,
    target_hw: tuple[int, int],
    valid_mask: np.ndarray,
    hd_scalar: np.ndarray,
    hd_edge_n: np.ndarray,
    sigmas: list[float],
    aggregation: str,
    edge_method_cfg: dict[str, Any],
    eval_mode: str,
    eval_percentile: float,
    eval_value: float,
    eval_otsu_pre_equalize: bool,
    eval_otsu_equalize_method: str,
    eval_otsu_clahe_clip_limit: float,
    eval_otsu_clahe_tile_grid_size: int,
    allow_resize: bool,
    equalize_visualization: bool = False,
    equalize_visualization_method: str = "hist",
    equalize_visualization_clahe_clip_limit: float = 2.0,
    equalize_visualization_clahe_tile_grid_size: int = 8,
    save_viz_edges: bool = False,
    save_viz_edges_binary: bool = False,
    save_agreement_overlay: bool = False,
    save_npy: bool = False,
    save_edge_equalized: bool = False,
    viz_edges_dir: Path | None = None,
    save_edge_equalize_method: str = "hist",
    save_edge_clahe_clip_limit: float = 2.0,
    save_edge_clahe_tile_grid_size: int = 8,
    spatial_low_percentile: float = 1.0,
    spatial_high_percentile: float = 95.0,
    spatial_normalization_enabled: bool = True,
    divide_by_255: bool = False,
    edge_method_cfg_override: dict[str, Any] | None = None,
    display_gamma: float | None = None,
    min_viz_edge_percentile: float = 0.0,
    eval_metric: str = "f1",
    square_before_continuous_metrics: bool = False,
    equalize_edges_before_metric: bool = False,
    edge_colormap: str | None = None,
    local_dice_window: int = 0,
    extra_hd_edge_n: np.ndarray | None = None,
    extra_hd_label: str | None = None,
    first_panel: str = "viz",
    composite_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Worker: load viz, compute edges, rank by F1, continuous Dice, or gradient SSIM. Top-level for multiprocessing."""
    local_dice_f32: np.ndarray | None = None
    local_dice_summary: tuple[float, float] | None = None
    comp_weights: dict[str, Any] = {}
    composite_on = False
    lab_bins_a, lab_bins_b = 16, 16
    if composite_cfg:
        composite_on = bool(composite_cfg.get("enabled"))
        if composite_on:
            comp_weights = dict(composite_cfg.get("weights") or {})
            lab_bins_a = int(composite_cfg.get("lab_entropy_bins_a", 16) or 16)
            lab_bins_b = int(composite_cfg.get("lab_entropy_bins_b", 16) or 16)

    def _cw(name: str) -> float:
        return float(comp_weights.get(name, 0) or 0)

    need_cd = composite_on and abs(_cw("continuous_dice")) > 1e-15
    need_aux = composite_on and any(
        abs(_cw(k)) > 1e-15
        for k in (
            "luminance_rms_contrast",
            "luminance_gradient_mean",
            "lab_chroma_entropy",
            "lab_mean_chroma",
        )
    )
    continuous_dice = float("nan")
    luminance_rms_contrast = float("nan")
    luminance_gradient_mean = float("nan")
    lab_chroma_entropy = float("nan")
    lab_mean_chroma = float("nan")
    try:
        viz_rgb = _load_visualization(viz_path, target_hw=target_hw, allow_resize=allow_resize)
        if equalize_visualization:
            viz_rgb = _apply_equalize_rgb_uint8(
                viz_rgb,
                equalize_visualization_method,
                equalize_visualization_clahe_clip_limit,
                equalize_visualization_clahe_tile_grid_size,
            )
        edge_cfg = edge_method_cfg if edge_method_cfg_override is None else {**edge_method_cfg, **edge_method_cfg_override}
        if divide_by_255:
            viz_rgb = viz_rgb.astype(np.float32) / 255.0
        if need_aux:
            cq = _compute_viz_contrast_color_metrics(
                viz_rgb,
                valid_mask,
                lab_entropy_bins_a=lab_bins_a,
                lab_entropy_bins_b=lab_bins_b,
            )
            luminance_rms_contrast = float(cq["luminance_rms_contrast"])
            luminance_gradient_mean = float(cq["luminance_gradient_mean"])
            lab_chroma_entropy = float(cq["lab_chroma_entropy"])
            lab_mean_chroma = float(cq["lab_mean_chroma"])
        v_edge, v_edge_linf = _edge_maps_multiscale(
            "rgb",
            viz_rgb,
            valid_mask,
            sigmas=sigmas,
            aggregation=aggregation,
            hd_cfg=None,
            edge_method_cfg=edge_cfg,
        )
        if divide_by_255:
            v_edge_n = np.asarray(v_edge, dtype=np.float32)
            v_edge_n[~valid_mask] = 0.0
        elif spatial_normalization_enabled:
            v_edge_n = _normalize_map_percentile(
                v_edge,
                valid_mask,
                low_pct=spatial_low_percentile,
                high_pct=spatial_high_percentile,
            )
        else:
            v_edge_n = _normalize_on_mask(v_edge, valid_mask)
        if eval_otsu_pre_equalize:
            v_eval = _equalize_gray_u8(
                _to_uint8_gray01(v_edge_n),
                enabled=True,
                method=eval_otsu_equalize_method,
                clahe_clip_limit=eval_otsu_clahe_clip_limit,
                clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
            ).astype(np.float32) / 255.0
            v_eval[~valid_mask] = 0.0
        else:
            v_eval = v_edge_n
        if min_viz_edge_percentile > 0 and np.any(valid_mask):
            thresh = float(np.percentile(v_eval[valid_mask], min_viz_edge_percentile))
            v_eval = np.where((valid_mask) & (v_eval >= thresh), v_eval, 0.0).astype(np.float32)
        hd_metric = hd_edge_n
        v_metric = v_eval
        if equalize_edges_before_metric:
            hd_metric = _equalize_gray_u8(
                _to_uint8_gray01(hd_edge_n),
                enabled=True,
                method=eval_otsu_equalize_method,
                clahe_clip_limit=eval_otsu_clahe_clip_limit,
                clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
            ).astype(np.float32) / 255.0
            hd_metric[~valid_mask] = 0.0
            v_metric = _equalize_gray_u8(
                _to_uint8_gray01(v_eval),
                enabled=True,
                method=eval_otsu_equalize_method,
                clahe_clip_limit=eval_otsu_clahe_clip_limit,
                clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
            ).astype(np.float32) / 255.0
            v_metric[~valid_mask] = 0.0
        m = str(eval_metric).strip().lower()
        use_dice = m == "continuous_dice"
        use_grad_ssim = m in ("gradient_ssim", "grad_ssim")
        ssim_gx = ssim_gy = ssim_mag = float("nan")
        gradient_ssim_val = float("nan")
        if use_grad_ssim:
            viz_scalar = _luminance_from_rgb01(viz_rgb)
            mean_ssim, ssim_gx, ssim_gy, ssim_mag = _compute_gradient_ssim_mean(
                hd_scalar, viz_scalar, valid_mask
            )
            gradient_ssim_val = mean_ssim
            score = mean_ssim
            p, r, f1, tp, fp, fn = float("nan"), float("nan"), float("nan"), 0, 0, 0
        elif use_dice:
            hd_d, v_d = _edges_for_continuous_metrics(
                hd_metric, v_metric, square=square_before_continuous_metrics
            )
            score = _compute_continuous_dice(hd_d, v_d, valid_mask)
            p, r, f1, tp, fp, fn = float("nan"), float("nan"), score, 0, 0, 0
        else:
            hd_eval = _equalize_gray_u8(
                _to_uint8_gray01(hd_edge_n),
                enabled=eval_otsu_pre_equalize,
                method=eval_otsu_equalize_method,
                clahe_clip_limit=eval_otsu_clahe_clip_limit,
                clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
            ).astype(np.float32) / 255.0
            hd_eval[~valid_mask] = 0.0
            hd_bin = _binary_from_threshold(
                hd_eval,
                valid_mask,
                eval_mode,
                eval_percentile,
                eval_value,
                otsu_pre_equalize=eval_otsu_pre_equalize,
                otsu_equalize_method=eval_otsu_equalize_method,
                otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
                otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
            )
            viz_bin = _binary_from_threshold(
                v_eval,
                valid_mask,
                eval_mode,
                eval_percentile,
                eval_value,
                otsu_pre_equalize=eval_otsu_pre_equalize,
                otsu_equalize_method=eval_otsu_equalize_method,
                otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
                otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
            )
            p, r, f1, tp, fp, fn = _compute_prf1(hd_bin[valid_mask], viz_bin[valid_mask])
            score = f1
        if need_cd and not use_dice:
            hd_d_cd, v_d_cd = _edges_for_continuous_metrics(
                hd_metric, v_metric, square=square_before_continuous_metrics
            )
            continuous_dice = float(_compute_continuous_dice(hd_d_cd, v_d_cd, valid_mask))
        elif use_dice:
            continuous_dice = float(score)
        hd_bin_ld: np.ndarray | None = None
        viz_bin_ld: np.ndarray | None = None
        if local_dice_window > 0 and save_viz_edges and viz_edges_dir is not None:
            if not use_dice and not use_grad_ssim:
                hd_bin_ld = hd_bin
                viz_bin_ld = viz_bin
            else:
                hd_eval_ld = _equalize_gray_u8(
                    _to_uint8_gray01(hd_edge_n),
                    enabled=eval_otsu_pre_equalize,
                    method=eval_otsu_equalize_method,
                    clahe_clip_limit=eval_otsu_clahe_clip_limit,
                    clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
                ).astype(np.float32) / 255.0
                hd_eval_ld[~valid_mask] = 0.0
                hd_bin_ld = _binary_from_threshold(
                    hd_eval_ld,
                    valid_mask,
                    eval_mode,
                    eval_percentile,
                    eval_value,
                    otsu_pre_equalize=eval_otsu_pre_equalize,
                    otsu_equalize_method=eval_otsu_equalize_method,
                    otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
                    otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
                )
                viz_bin_ld = _binary_from_threshold(
                    v_eval,
                    valid_mask,
                    eval_mode,
                    eval_percentile,
                    eval_value,
                    otsu_pre_equalize=eval_otsu_pre_equalize,
                    otsu_equalize_method=eval_otsu_equalize_method,
                    otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
                    otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
                )
        hd_s, v_s = _edges_for_continuous_metrics(
            hd_metric, v_metric, square=square_before_continuous_metrics
        )
        soft_p, soft_r = _compute_soft_precision_recall(hd_s, v_s, valid_mask)
        if save_viz_edges_binary and viz_edges_dir is not None and not use_dice and not use_grad_ssim:
            out_dir = Path(viz_edges_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            viz_bin_u8 = (viz_bin.astype(np.uint8) * 255)
            orig_u8 = (np.clip(viz_rgb, 0.0, 1.0) * 255.0).astype(np.uint8) if viz_rgb.dtype != np.uint8 else viz_rgb
            if orig_u8.ndim == 2:
                orig_u8 = np.stack([orig_u8] * 3, axis=-1)
            _save_side_by_side_rgb(orig_u8, viz_bin_u8, out_dir / f"{viz_path.stem}__viz_edge_binary.png")
            if save_npy:
                np.save(out_dir / f"{viz_path.stem}__viz_edge_binary.npy", viz_bin.astype(np.uint8))
        if save_agreement_overlay and viz_edges_dir is not None and not use_dice and not use_grad_ssim:
            overlay_rgb = _agreement_overlay_rgb(hd_bin, viz_bin, valid_mask)
            overlay_img = Image.fromarray(overlay_rgb, mode="RGB")
            overlay_img = _add_agreement_legend(overlay_img)
            Path(viz_edges_dir).mkdir(parents=True, exist_ok=True)
            orig_u8 = (np.clip(viz_rgb, 0.0, 1.0) * 255.0).astype(np.uint8) if viz_rgb.dtype != np.uint8 else viz_rgb
            if orig_u8.ndim == 2:
                orig_u8 = np.stack([orig_u8] * 3, axis=-1)
            legend_height = 28
            orig_h, orig_w = orig_u8.shape[:2]
            pad = np.full((legend_height, orig_w, 3), 30, dtype=np.uint8)
            orig_padded = np.concatenate([orig_u8, pad], axis=0)
            _save_side_by_side_rgb(orig_padded, np.asarray(overlay_img), Path(viz_edges_dir) / f"{viz_path.stem}__agreement_overlay.png")
        if save_viz_edges and viz_edges_dir is not None:
            if use_dice:
                mlabel = "Dice"
            elif use_grad_ssim:
                mlabel = "GradSSIM"
            else:
                mlabel = "F1"
            viz_u8 = (np.clip(viz_rgb, 0.0, 1.0) * 255.0).astype(np.uint8) if viz_rgb.dtype != np.uint8 else viz_rgb
            v_triple = v_metric if equalize_edges_before_metric else v_edge_n
            hd_triple = hd_metric if equalize_edges_before_metric else hd_edge_n
            if (
                local_dice_window > 0
                and hd_bin_ld is not None
                and viz_bin_ld is not None
            ):
                local_dice_f32 = _local_dice_heatmap_f32(
                    viz_bin_ld, hd_bin_ld, valid_mask, local_dice_window
                )
                local_dice_summary = _local_dice_summary_stats(local_dice_f32, valid_mask)
            triple_base = Path(viz_edges_dir) / f"{viz_path.stem}__triple.png"
            _save_triple_debug_png(
                triple_base,
                viz_u8,
                v_triple,
                hd_triple,
                valid_mask,
                float(score),
                mlabel,
                raw_max_scale=1.414 if divide_by_255 else None,
                display_gamma=display_gamma if divide_by_255 else None,
                colormap=edge_colormap,
                local_dice_f32=local_dice_f32,
                local_dice_summary=local_dice_summary,
                extra_edge_n=extra_hd_edge_n,
                extra_edge_label=extra_hd_label,
                first_panel=first_panel,
            )
            if save_edge_equalized:
                _save_triple_debug_png(
                    triple_base.with_name(f"{viz_path.stem}__triple_eq.png"),
                    viz_u8,
                    v_triple,
                    hd_triple,
                    valid_mask,
                    float(score),
                    mlabel,
                    raw_max_scale=1.414 if divide_by_255 else None,
                    display_gamma=display_gamma if divide_by_255 else None,
                    colormap=edge_colormap,
                    local_dice_f32=local_dice_f32,
                    local_dice_summary=local_dice_summary,
                    equalize_png=True,
                    equalize_method=save_edge_equalize_method,
                    clahe_clip_limit=save_edge_clahe_clip_limit,
                    clahe_tile_grid_size=save_edge_clahe_tile_grid_size,
                    extra_edge_n=extra_hd_edge_n,
                    extra_edge_label=extra_hd_label,
                    first_panel=first_panel,
                )
            if save_npy:
                np.save(Path(viz_edges_dir) / f"{viz_path.stem}__viz_edge.npy", np.asarray(v_edge_n, dtype=np.float32))
        out_row: dict[str, Any] = {
            "path": str(viz_path),
            "name": viz_path.name,
            "f1": score,
            "precision": p,
            "recall": r,
            "soft_precision": soft_p,
            "soft_recall": soft_r,
            "gradient_ssim": gradient_ssim_val,
            "ssim_grad_x": ssim_gx,
            "ssim_grad_y": ssim_gy,
            "ssim_grad_mag": ssim_mag,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_local_dice": float("nan"),
            "mean_local_dice_pos": float("nan"),
            "continuous_dice": continuous_dice,
            "luminance_rms_contrast": luminance_rms_contrast,
            "luminance_gradient_mean": luminance_gradient_mean,
            "lab_chroma_entropy": lab_chroma_entropy,
            "lab_mean_chroma": lab_mean_chroma,
            "composite_score": float("nan"),
        }
        if local_dice_summary is not None:
            out_row["mean_local_dice"] = local_dice_summary[0]
            out_row["mean_local_dice_pos"] = local_dice_summary[1]
        return out_row
    except Exception as e:
        return {
            "path": str(viz_path),
            "name": viz_path.name,
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "soft_precision": float("nan"),
            "soft_recall": float("nan"),
            "gradient_ssim": float("nan"),
            "ssim_grad_x": float("nan"),
            "ssim_grad_y": float("nan"),
            "ssim_grad_mag": float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "mean_local_dice": float("nan"),
            "mean_local_dice_pos": float("nan"),
            "continuous_dice": float("nan"),
            "luminance_rms_contrast": float("nan"),
            "luminance_gradient_mean": float("nan"),
            "lab_chroma_entropy": float("nan"),
            "lab_mean_chroma": float("nan"),
            "composite_score": float("nan"),
            "error": str(e),
        }


def _edge_map_to_rgb(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    raw_max_scale: float | None = None,
    display_gamma: float | None = None,
    colormap: str | None = None,
    *,
    equalize_png: bool = False,
    equalize_method: str = "hist",
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
) -> np.ndarray:
    """Convert edge map (float) to uint8 RGB for display. Optional matplotlib colormap on luminance.

    When ``equalize_png`` is true, applies the same histogram/CLAHE step as ``_save_map_png_and_npy``
    (after linear uint8 conversion and optional gamma).
    """
    x = np.asarray(arr, dtype=np.float32)
    x[~valid_mask] = 0.0
    if raw_max_scale is not None and raw_max_scale > 1e-8:
        x = x / float(raw_max_scale)
    u8 = _to_uint8_gray01(x)
    if display_gamma is not None and 0 < display_gamma < 1:
        u8 = (np.power(np.clip(u8.astype(np.float32) / 255.0, 0.0, 1.0), display_gamma) * 255.0).astype(np.uint8)
    u8 = _equalize_gray_u8(
        u8,
        enabled=equalize_png,
        method=equalize_method,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )
    cmn = (str(colormap).strip() if colormap is not None else "") or ""
    if cmn and cmn.lower() not in ("none", "null", "~"):
        try:
            return _gray_u8_to_rgb_colormap(u8, cmn)
        except Exception as exc:
            logger.warning("edge colormap %r failed (%s); using grayscale RGB", cmn, exc)
    return np.stack([u8] * 3, axis=-1)


def _save_triple_debug_png(
    out_path: Path,
    viz_u8: np.ndarray,
    v_edge_n: np.ndarray,
    hd_edge_n: np.ndarray,
    valid_mask: np.ndarray,
    score: float,
    metric_label: str,
    raw_max_scale: float | None,
    display_gamma: float | None,
    colormap: str | None = None,
    local_dice_f32: np.ndarray | None = None,
    local_dice_summary: tuple[float, float] | None = None,
    extra_edge_n: np.ndarray | None = None,
    extra_edge_label: str | None = None,
    first_panel: str = "viz",
    *,
    equalize_png: bool = False,
    equalize_method: str = "hist",
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
) -> None:
    """Save [viz | viz edges | HD edges] and optional [local Dice] to viz_edges_debug."""
    from PIL import ImageDraw, ImageFont

    v = np.asarray(viz_u8, dtype=np.uint8)
    if v.ndim == 2:
        v = np.stack([v] * 3, axis=-1)
    viz_edge_rgb = _edge_map_to_rgb(
        v_edge_n,
        valid_mask,
        raw_max_scale=raw_max_scale,
        display_gamma=display_gamma,
        colormap=colormap,
        equalize_png=equalize_png,
        equalize_method=equalize_method,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )
    hd_edge_rgb = _edge_map_to_rgb(
        hd_edge_n,
        valid_mask,
        raw_max_scale=raw_max_scale,
        display_gamma=display_gamma,
        colormap=colormap,
        equalize_png=equalize_png,
        equalize_method=equalize_method,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )
    sep = np.full((v.shape[0], 4, 3), 255, dtype=np.uint8)
    eq_tag = " (eq)" if equalize_png else ""
    fp = str(first_panel).strip().lower()
    if fp in {"hd", "hd_edge", "edge", "edges", "hd_edges"}:
        panels: list[np.ndarray] = [hd_edge_rgb, v, viz_edge_rgb]
        labels = [f"HD edges{eq_tag}", "viz", f"viz edges{eq_tag}"]
    elif fp in {"viz_edge", "viz_edges", "rgb_edge", "rgb_edges"}:
        panels = [viz_edge_rgb, v, hd_edge_rgb]
        labels = [f"viz edges{eq_tag}", "viz", f"HD edges{eq_tag}"]
    else:
        panels = [v, viz_edge_rgb, hd_edge_rgb]
        labels = ["viz", f"viz edges{eq_tag}", f"HD edges{eq_tag}"]
    if extra_edge_n is not None:
        extra_rgb = _edge_map_to_rgb(
            extra_edge_n,
            valid_mask,
            raw_max_scale=raw_max_scale,
            display_gamma=display_gamma,
            colormap=colormap,
            equalize_png=equalize_png,
            equalize_method=equalize_method,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
        )
        panels.append(extra_rgb)
        labels.append(str(extra_edge_label or "extra HD edges")[:120])
    if local_dice_f32 is not None:
        ld_rgb = _edge_map_to_rgb(
            np.asarray(local_dice_f32, dtype=np.float32),
            valid_mask,
            raw_max_scale=None,
            display_gamma=None,
            colormap=colormap,
            equalize_png=False,
        )
        panels.append(ld_rgb)
        labels.append("local Dice")
    combined = panels[0]
    for p in panels[1:]:
        combined = np.concatenate([combined, sep, p], axis=1)
    h, w = combined.shape[:2]
    has_ld_sum = local_dice_summary is not None
    bar_h = 52 if has_ld_sum else 36
    bar = np.full((bar_h, w, 3), 40, dtype=np.uint8)
    img = Image.fromarray(np.concatenate([bar, combined], axis=0), mode="RGB")
    try:
        font = ImageFont.truetype("arial.ttf", 10)
        font_small = ImageFont.truetype("arial.ttf", 9)
    except OSError:
        font = font_small = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    orig_w = int(v.shape[1])
    sep_w = 4
    stride = orig_w + sep_w
    for i, lab in enumerate(labels):
        draw.text((stride * i + 8, 4), lab, fill=(200, 200, 200), font=font)
    score_txt = f"{metric_label}={score:.4f}"
    score_y = 18 if has_ld_sum else 20
    draw.text((8, score_y), score_txt, fill=(220, 220, 120), font=font_small)
    if has_ld_sum:
        ma, mp = local_dice_summary
        if np.isfinite(mp):
            local_txt = f"mean(local)={ma:.4f} | mean(local|>0)={mp:.4f}"
        else:
            local_txt = f"mean(local)={ma:.4f} | mean(local|>0)=nan"
        draw.text((8, 34), local_txt, fill=(180, 200, 255), font=font_small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _compute_viz_edge_for_loaded_rgb(
    viz_rgb: np.ndarray,
    valid_mask: np.ndarray,
    sigmas: list[float],
    aggregation: str,
    edge_method_cfg: dict[str, Any],
    *,
    spatial_low_percentile: float,
    spatial_high_percentile: float,
    spatial_normalization_enabled: bool,
    divide_by_255: bool,
    edge_method_cfg_override: dict[str, Any] | None,
    min_viz_edge_percentile: float,
) -> np.ndarray:
    edge_cfg = edge_method_cfg if edge_method_cfg_override is None else {**edge_method_cfg, **edge_method_cfg_override}
    v_edge, _v_edge_linf = _edge_maps_multiscale(
        "rgb",
        viz_rgb,
        valid_mask,
        sigmas=sigmas,
        aggregation=aggregation,
        hd_cfg=None,
        edge_method_cfg=edge_cfg,
    )
    if divide_by_255:
        v_edge_n = np.asarray(v_edge, dtype=np.float32)
        v_edge_n[~valid_mask] = 0.0
    elif spatial_normalization_enabled:
        v_edge_n = _normalize_map_percentile(
            v_edge,
            valid_mask,
            low_pct=spatial_low_percentile,
            high_pct=spatial_high_percentile,
        )
    else:
        v_edge_n = _normalize_on_mask(v_edge, valid_mask)
    if min_viz_edge_percentile > 0 and np.any(valid_mask):
        thresh = float(np.percentile(v_edge_n[valid_mask], min_viz_edge_percentile))
        v_edge_n = np.where((valid_mask) & (v_edge_n >= thresh), v_edge_n, 0.0).astype(np.float32)
    return v_edge_n


def _save_spatial_coverage_plot(steps: list[int], sums: list[float], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.plot(steps, sums, marker="o", linewidth=2)
    ax.set_xlabel("Selected visualizations")
    ax.set_ylabel("Sum of per-pixel max spatial quality")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _save_spatial_coverage_comparison_plot(
    series: list[tuple[str, list[int], list[float]]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    for label, steps, sums in series:
        if steps and sums:
            ax.plot(steps, sums, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Selected visualizations")
    ax.set_ylabel("Sum of per-pixel max spatial quality")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _write_spatial_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = ["selection_rank", "rank", "name", "path", "quality_sum", "quality_mean", "marginal_gain"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_hd_edge_n_for_method(
    cfg: DictConfig,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    hd_cfg: Any,
    method: str,
    sigmas: list[float],
    aggregation: str,
    spatial_enabled: bool,
) -> np.ndarray:
    method = str(method).strip().lower()
    if method in ("top3_nmf_max", "top3_nmf_mean"):
        from gallery_top3_nmf_edges import _compute_top3_nmf_aggregate_v_agg

        mode = "max" if method == "top3_nmf_max" else "mean"
        base_edge = _compute_top3_nmf_aggregate_v_agg(cfg, msi, valid_mask, mode=mode)
        base_edge_linf = np.asarray(base_edge, dtype=np.float32).copy()
        edge_maps: list[np.ndarray] = []
        edge_linf_maps: list[np.ndarray] = []
        for s in sigmas:
            sigma = float(s)
            if sigma <= 0:
                edge_maps.append(base_edge)
                edge_linf_maps.append(base_edge_linf)
            else:
                edge_maps.append(_gaussian_blur(base_edge, sigma))
                edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
        merged = _aggregate_multiscale_maps(edge_maps, valid_mask, aggregation, normalize=True)
        merged = _apply_hd_edge_power(merged, valid_mask, hd_cfg)
    else:
        edge_cfg = dict(_parse_edge_detection_cfg(cfg))
        edge_cfg["hd_method"] = method
        msi_for_edges = _pca_project_msi_for_edges(msi, valid_mask, hd_cfg)
        hd_result = _edge_maps_multiscale(
            "hd",
            msi_for_edges,
            valid_mask,
            sigmas=sigmas,
            aggregation=aggregation,
            hd_cfg=hd_cfg,
            edge_method_cfg=edge_cfg,
        )
        merged = hd_result[0]

    if spatial_enabled:
        return _normalize_map_percentile(
            merged,
            valid_mask,
            low_pct=float(cfg.spatial_normalization.low_percentile),
            high_pct=float(cfg.spatial_normalization.high_percentile),
        )
    return _normalize_on_mask(merged, valid_mask)


def _create_gallery_simple(
    paths_with_labels: list[tuple[Path, str]],
    out_path: Path,
    columns: int = 5,
    cell_padding: int = 8,
    max_cell_size: tuple[int, int] | None = (400, 400),
    show_labels: bool = True,
) -> None:
    """Create a grid image from visualization paths with labels (PIL-only, no ImageDraw font)."""
    if not paths_with_labels:
        return
    images: list[tuple[Image.Image, str]] = []
    for p, label in paths_with_labels:
        if not p.exists():
            continue
        with Image.open(p) as im:
            images.append((im.convert("RGB"), label))
    _arrange_gallery_grid(images, out_path, columns, cell_padding, max_cell_size, show_labels=show_labels)


def _create_gallery_from_images(
    images: list[tuple[Image.Image, str]],
    out_path: Path,
    columns: int = 5,
    cell_padding: int = 8,
    max_cell_size: tuple[int, int] | None = (400, 400),
    show_labels: bool = True,
) -> None:
    """Create a grid image from already-loaded PIL images."""
    _arrange_gallery_grid(images, out_path, columns, cell_padding, max_cell_size, show_labels=show_labels)


def _arrange_gallery_grid(
    images: list[tuple[Image.Image, str]],
    out_path: Path,
    columns: int = 5,
    cell_padding: int = 8,
    max_cell_size: tuple[int, int] | None = (400, 400),
    show_labels: bool = True,
) -> None:
    """Arrange (Image, label) pairs in a grid and save."""
    if not images:
        return
    from PIL import ImageDraw, ImageFont

    if max_cell_size:
        resized = []
        for im, label in images:
            w, h = im.size
            scale = min(max_cell_size[0] / w, max_cell_size[1] / h, 1.0)
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            resized.append((im, label))
        images = resized
    max_w = max(im.width for im, _ in images)
    max_h = max(im.height for im, _ in images)
    label_h = 24 if show_labels else 0
    cell_w = max_w + 2 * cell_padding
    cell_h = max_h + 2 * cell_padding + label_h
    n = len(images)
    rows = (n + columns - 1) // columns
    grid_w = columns * cell_w
    grid_h = rows * cell_h
    canvas = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(images):
        row, col = divmod(i, columns)
        x0 = col * cell_w
        y0 = row * cell_h
        x = x0 + (cell_w - im.width) // 2
        y = y0 + cell_padding
        canvas.paste(im, (x, y))
        if show_labels:
            text_y = y0 + cell_padding + im.height + 2
            draw.text((x0 + cell_padding, text_y), label[:80], fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


@hydra.main(version_base=None, config_path="configs", config_name="rank_visualizations_by_f1")
def main(cfg: DictConfig) -> None:
    log_level = str(getattr(getattr(cfg, "logging", None), "level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    npy_raw = OmegaConf.select(cfg, "input.npy_path")
    if npy_raw is None or str(npy_raw).strip() == "":
        raise ValueError(
            "input.npy_path is required (MSI .npy). Set it in scripts/configs/rank_visualizations_by_f1.yaml "
            "or pass e.g. input.npy_path=E:/path/to/data.npy",
        )
    npy_path = Path(str(npy_raw))
    viz_folder = Path(str(cfg.input.visualization_folder))
    if not npy_path.exists():
        raise FileNotFoundError(f"MSI file not found: {npy_path}")
    if not viz_folder.exists() or not viz_folder.is_dir():
        raise FileNotFoundError(f"Visualization folder not found: {viz_folder}")

    out_dir = Path(str(cfg.output.out_dir)) if cfg.output.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    _stage(1, f"Paths: MSI={npy_path} | viz_folder={viz_folder} | out_dir={out_dir}")
    logger.info("Starting | npy=%s | viz_folder=%s | out_dir=%s", npy_path, viz_folder, out_dir)

    skip_eq = bool(getattr(cfg.input, "skip_eq", False))
    viz_paths = _find_visualizations(viz_folder, skip_eq=skip_eq)
    if not viz_paths:
        raise ValueError(f"No visualization files found in {viz_folder}")
    _stage(1, f"Found {len(viz_paths)} visualization file(s) to score")
    logger.info("Found %d visualizations in %s", len(viz_paths), viz_folder)

    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    t0 = time.perf_counter()
    _stage(2, "Loading MSI and building valid mask...")
    logger.info("Loading MSI...")
    normalization = _resolve_normalization_mode(getattr(cfg.data, "normalization", "tic"))
    msi = _load_msi(
        npy_path,
        transpose_msi=bool(cfg.data.transpose_msi),
        normalization=normalization,
    )
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty.")
    logger.info("loaded msi shape=%s valid_pixels=%d", tuple(msi.shape), int(valid_mask.sum()))
    _stage(2, f"MSI shape={tuple(msi.shape)} | valid_pixels={int(valid_mask.sum())}")
    hd_scalar = np.linalg.norm(np.asarray(msi, dtype=np.float32), axis=-1)

    debug_cfg_early = getattr(cfg, "debug", None)
    save_individual_hd_edges = bool(getattr(debug_cfg_early, "save_individual_hd_edges", False)) if debug_cfg_early else False
    hd_cfg = getattr(cfg, "hd_edges", None)
    if save_individual_hd_edges and hd_cfg is not None:
        hd_cfg = OmegaConf.merge(hd_cfg, OmegaConf.create({"save_individual_distance_maps": True}))
    edge_method_cfg = _parse_edge_detection_cfg(cfg)
    ms_cfg = getattr(cfg, "edge_maps", None)
    sigmas = [float(v) for v in list(getattr(getattr(ms_cfg, "multi_scale", None), "sigmas", [0.0])) or [0.0]]
    aggregation = str(getattr(getattr(ms_cfg, "multi_scale", None), "aggregation", "mean"))

    _stage(3, "Computing high-dimensional (MSI) edge maps...")
    hd_method = str(edge_method_cfg.get("hd_method", "gradient")).strip().lower()

    if hd_method in ("top3_nmf_max", "top3_nmf_mean"):
        from gallery_top3_nmf_edges import _compute_top3_nmf_aggregate_v_agg

        mode = "max" if hd_method == "top3_nmf_max" else "mean"
        logger.info(
            "Computing HD edges (TOP3+NMF stack, %s) on full MSI %s — uses top3/nmf/equalize/edge_* from config; "
            "not hd_edges PCA projection",
            mode,
            tuple(msi.shape),
        )
        t_hd = time.perf_counter()
        base_edge = _compute_top3_nmf_aggregate_v_agg(cfg, msi, valid_mask, mode=mode)
        base_edge_linf = np.asarray(base_edge, dtype=np.float32).copy()
        edge_maps: list[np.ndarray] = []
        edge_linf_maps: list[np.ndarray] = []
        for s in sigmas:
            sigma = float(s)
            if sigma <= 0:
                edge_maps.append(base_edge)
                edge_linf_maps.append(base_edge_linf)
            else:
                edge_maps.append(_gaussian_blur(base_edge, sigma))
                edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
        merged = _aggregate_multiscale_maps(edge_maps, valid_mask, aggregation, normalize=True)
        merged_linf = _aggregate_multiscale_maps(edge_linf_maps, valid_mask, aggregation, normalize=True)
        merged = _apply_hd_edge_power(merged, valid_mask, hd_cfg)
        merged_linf = _apply_hd_edge_power(merged_linf, valid_mask, hd_cfg)
        hd_edge = merged
        hd_result = (merged, merged_linf)
        individual_hd_maps = None
    else:
        logger.info("Computing HD edges (%s)...", hd_method)
        msi_for_edges = _pca_project_msi_for_edges(msi, valid_mask, hd_cfg)
        t_hd = time.perf_counter()
        hd_result = _edge_maps_multiscale(
            "hd",
            msi_for_edges,
            valid_mask,
            sigmas=sigmas,
            aggregation=aggregation,
            hd_cfg=hd_cfg,
            edge_method_cfg=edge_method_cfg,
        )
        hd_edge = hd_result[0]
        individual_hd_maps = hd_result[2] if len(hd_result) > 2 else None

    spatial_enabled = bool(getattr(cfg.spatial_normalization, "enabled", True))
    if spatial_enabled:
        hd_edge_n = _normalize_map_percentile(
            hd_edge,
            valid_mask,
            low_pct=float(cfg.spatial_normalization.low_percentile),
            high_pct=float(cfg.spatial_normalization.high_percentile),
        )
    else:
        hd_edge_n = _normalize_on_mask(hd_edge, valid_mask)
    logger.info("HD edges computed | %.2fs", time.perf_counter() - t_hd)
    _stage(3, f"HD edges done in {time.perf_counter() - t_hd:.2f}s (kept in memory for all viz)")

    debug_cfg = getattr(cfg, "debug", None)
    save_viz_edges = bool(getattr(debug_cfg, "save_viz_edges", False)) if debug_cfg else False
    save_viz_edges_binary = bool(getattr(debug_cfg, "save_viz_edges_binary", False)) if debug_cfg else False
    save_agreement_overlay = bool(getattr(debug_cfg, "save_agreement_overlay", False)) if debug_cfg else False
    save_individual_hd_edges = bool(getattr(debug_cfg, "save_individual_hd_edges", False)) if debug_cfg else False
    save_npy = bool(getattr(debug_cfg, "save_npy", False)) if debug_cfg else False
    save_edge_equalized = bool(getattr(debug_cfg, "save_edge_equalized", False)) if debug_cfg else False
    edge_colormap = OmegaConf.select(cfg, "debug.edge_colormap", default=None)
    if edge_colormap is not None:
        edge_colormap = str(edge_colormap).strip()
        if edge_colormap.lower() in ("null", "none", "~", ""):
            edge_colormap = None
    local_dice_window = int(OmegaConf.select(cfg, "debug.local_dice_window", default=0) or 0)
    if local_dice_window < 0:
        local_dice_window = 0
    extra_hd_method = OmegaConf.select(cfg, "debug.extra_hd_method", default=None)
    extra_hd_edge_n: np.ndarray | None = None
    extra_hd_label: str | None = None
    if extra_hd_method is not None:
        extra_hd_method_s = str(extra_hd_method).strip().lower()
        if extra_hd_method_s not in {"", "none", "null", "~", "false", "off"}:
            logger.info("Computing extra debug HD edge panel (%s)...", extra_hd_method_s)
            extra_hd_edge_n = _compute_hd_edge_n_for_method(
                cfg,
                msi,
                valid_mask,
                hd_cfg,
                extra_hd_method_s,
                sigmas,
                aggregation,
                spatial_enabled,
            )
            extra_hd_label = str(OmegaConf.select(cfg, "debug.extra_hd_label", default=extra_hd_method_s))
    viz_edges_dir = out_dir / "viz_edges_debug" if (save_viz_edges or save_viz_edges_binary or save_agreement_overlay or save_individual_hd_edges) else None
    if viz_edges_dir is not None:
        viz_edges_dir.mkdir(parents=True, exist_ok=True)
    if save_viz_edges:
        eq_suffix = "_eq" if save_edge_equalized else ""
        _save_map_png_and_npy(
            viz_edges_dir / f"hd_edge_norm{eq_suffix}",
            hd_edge_n,
            save_npy=save_npy,
            equalize_png=save_edge_equalized,
            equalize_method="hist",
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
            colormap=edge_colormap,
        )
        logger.info("debug: saving viz edges to %s (HD edges saved)", viz_edges_dir)
    if save_agreement_overlay and viz_edges_dir is not None:
        logger.info("debug: saving agreement overlays to %s", viz_edges_dir)
    if save_individual_hd_edges and individual_hd_maps and viz_edges_dir is not None:
        eq_suffix = "_eq" if save_edge_equalized else ""
        for metric_name, arr in individual_hd_maps.items():
            if spatial_enabled:
                arr_n = _normalize_map_percentile(
                    arr,
                    valid_mask,
                    low_pct=float(cfg.spatial_normalization.low_percentile),
                    high_pct=float(cfg.spatial_normalization.high_percentile),
                )
            else:
                arr_n = _normalize_on_mask(arr, valid_mask)
            _save_map_png_and_npy(
                viz_edges_dir / f"hd_edge_{metric_name}_norm{eq_suffix}",
                arr_n,
                save_npy=save_npy,
                equalize_png=save_edge_equalized,
                equalize_method="hist",
                clahe_clip_limit=2.0,
                clahe_tile_grid_size=8,
                colormap=edge_colormap,
            )
        logger.info("debug: saved %d individual distance maps (l2, sam, cosine, etc.) to %s", len(individual_hd_maps), viz_edges_dir)

    eval_cfg = getattr(cfg, "evaluation", None)
    eval_mode = str(getattr(eval_cfg, "threshold_mode", "otsu")) if eval_cfg else "otsu"
    eval_percentile = float(getattr(eval_cfg, "threshold_percentile", 85.0)) if eval_cfg else 85.0
    eval_value = float(getattr(eval_cfg, "threshold_value", 0.5)) if eval_cfg else 0.5
    eval_otsu_pre_equalize = bool(getattr(eval_cfg, "otsu_pre_equalize", False)) if eval_cfg else False
    eval_otsu_equalize_method = str(getattr(eval_cfg, "otsu_equalize_method", "hist")) if eval_cfg else "hist"
    eval_otsu_clahe_clip_limit = float(getattr(eval_cfg, "otsu_clahe_clip_limit", 2.0)) if eval_cfg else 2.0
    eval_otsu_clahe_tile_grid_size = int(getattr(eval_cfg, "otsu_clahe_tile_grid_size", 8)) if eval_cfg else 8
    min_viz_edge_percentile = float(getattr(eval_cfg, "min_viz_edge_percentile", 0.0)) if eval_cfg else 0.0
    eval_metric = str(getattr(eval_cfg, "metric", "f1")).strip().lower() if eval_cfg else "f1"
    metric_label = _metric_label_for_eval(eval_metric)
    square_before_continuous_metrics = (
        bool(getattr(eval_cfg, "square_before_continuous_metrics", False)) if eval_cfg else False
    )
    equalize_edges_before_metric = (
        bool(getattr(eval_cfg, "equalize_edges_before_metric", False)) if eval_cfg else False
    )
    inp = getattr(cfg, "input", None)
    equalize_visualization = bool(getattr(inp, "equalize_visualization", False)) if inp else False
    equalize_visualization_method = str(getattr(inp, "equalize_visualization_method", "hist")) if inp else "hist"
    equalize_visualization_clahe_clip_limit = (
        float(getattr(inp, "equalize_visualization_clahe_clip_limit", 2.0)) if inp else 2.0
    )
    equalize_visualization_clahe_tile_grid_size = (
        int(getattr(inp, "equalize_visualization_clahe_tile_grid_size", 8)) if inp else 8
    )
    composite_cfg_resolved: dict[str, Any] | None = None
    composite_sort = False
    if eval_cfg is not None:
        comp_eval = getattr(eval_cfg, "composite", None)
        if comp_eval is not None and bool(OmegaConf.select(comp_eval, "enabled", default=False)):
            composite_cfg_resolved = OmegaConf.to_container(comp_eval, resolve=True)
            ww = composite_cfg_resolved.get("weights") or {}
            composite_sort = (
                sum(abs(float(ww.get(k, 0) or 0)) for k in _COMPOSITE_KEYS) > 1e-15
            )
    _stage(
        4,
        f"Evaluation: metric={eval_metric} ({metric_label}) | threshold={eval_mode} | min_viz_edge_percentile={min_viz_edge_percentile}"
        f" | square_before_continuous_metrics={square_before_continuous_metrics}"
        f" | equalize_edges_before_metric={equalize_edges_before_metric}"
        f" | equalize_visualization={equalize_visualization}"
        f" | composite_sort={composite_sort}",
    )

    if save_viz_edges_binary and viz_edges_dir is not None:
        hd_eval = _equalize_gray_u8(
            _to_uint8_gray01(hd_edge_n),
            enabled=eval_otsu_pre_equalize,
            method=eval_otsu_equalize_method,
            clahe_clip_limit=eval_otsu_clahe_clip_limit,
            clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
        ).astype(np.float32) / 255.0
        hd_eval[~valid_mask] = 0.0
        hd_bin = _binary_from_threshold(
            hd_eval,
            valid_mask,
            eval_mode,
            eval_percentile,
            eval_value,
            otsu_pre_equalize=eval_otsu_pre_equalize,
            otsu_equalize_method=eval_otsu_equalize_method,
            otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
            otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
        )
        hd_bin_u8 = (hd_bin.astype(np.uint8) * 255)
        Image.fromarray(hd_bin_u8, mode="L").save(viz_edges_dir / "hd_edge_binary.png")
        if save_npy:
            np.save(viz_edges_dir / "hd_edge_binary.npy", hd_bin.astype(np.uint8))
        logger.info("debug: saved HD binary edges to %s", viz_edges_dir)

    n_jobs = int(getattr(cfg, "n_jobs", 0))
    if n_jobs == 0:
        n_jobs = max(1, os.cpu_count() or 1)

    target_hw = (msi.shape[0], msi.shape[1])
    allow_resize = bool(cfg.input.allow_resize)
    divide_by_255 = bool(getattr(cfg.input, "divide_by_255", False))
    edge_cfg_override = {"rgb_scale_01": True, "rgb_no_normalize": True} if divide_by_255 else None
    # When divide_by_255: no normalization (raw gradients). Otherwise follow config.
    spatial_enabled = bool(getattr(cfg.spatial_normalization, "enabled", True)) and not divide_by_255
    display_gamma_val = None
    if divide_by_255 and debug_cfg is not None:
        g = getattr(debug_cfg, "display_gamma", None)
        if g is not None and 0 < float(g) < 1:
            display_gamma_val = float(g)
    first_panel = str(OmegaConf.select(cfg, "debug.first_panel", default="viz")).strip().lower()

    worker_args = [
        (
            p,
            target_hw,
            valid_mask,
            hd_scalar,
            hd_edge_n,
            sigmas,
            aggregation,
            edge_method_cfg,
            eval_mode,
            eval_percentile,
            eval_value,
            eval_otsu_pre_equalize,
            eval_otsu_equalize_method,
            eval_otsu_clahe_clip_limit,
            eval_otsu_clahe_tile_grid_size,
            allow_resize,
            equalize_visualization,
            equalize_visualization_method,
            equalize_visualization_clahe_clip_limit,
            equalize_visualization_clahe_tile_grid_size,
            save_viz_edges,
            save_viz_edges_binary,
            save_agreement_overlay,
            save_npy,
            save_edge_equalized,
            viz_edges_dir,
            "hist",  # save_edge_equalize_method
            2.0,  # save_edge_clahe_clip_limit
            8,  # save_edge_clahe_tile_grid_size
            float(cfg.spatial_normalization.low_percentile),
            float(cfg.spatial_normalization.high_percentile),
            spatial_enabled,
            divide_by_255,
            edge_cfg_override,
            display_gamma_val,
            min_viz_edge_percentile,
            eval_metric,
            square_before_continuous_metrics,
            equalize_edges_before_metric,
            edge_colormap,
            local_dice_window,
            extra_hd_edge_n,
            extra_hd_label,
            first_panel,
            composite_cfg_resolved,
        )
        for p in viz_paths
    ]

    t_viz = time.perf_counter()
    _stage(
        5,
        f"Scoring each visualization (viz edges vs HD; {n_jobs} job(s)) — HD edges reused, not recomputed per viz",
    )
    if n_jobs <= 1:
        logger.info("Processing %d visualizations (sequential)...", len(viz_paths))
        try:
            from tqdm import tqdm
            iterator = tqdm(worker_args, desc=f"viz {metric_label}", unit="viz")
        except ImportError:
            iterator = worker_args
        results = []
        for i, args in enumerate(iterator, 1):
            r = _compute_viz_f1_worker(*args)
            results.append(r)
            score_str = f"{metric_label}={r['f1']:.4f}" if not np.isnan(r["f1"]) else "FAILED"
            logger.info("  [%d/%d] %s | %s", i, len(viz_paths), r["name"], score_str)
    else:
        logger.info("Processing %d visualizations with %d workers...", len(viz_paths), n_jobs)
        from multiprocessing import Pool
        with Pool(processes=n_jobs) as pool:
            results = pool.starmap(_compute_viz_f1_worker, worker_args)
    logger.info("viz scores computed | %d items | %.2fs", len(results), time.perf_counter() - t_viz)
    _stage(6, f"Finished scoring {len(results)} viz in {time.perf_counter() - t_viz:.2f}s")

    valid_results = [r for r in results if not np.isnan(r["f1"])]
    invalid_results = [r for r in results if np.isnan(r["f1"])]
    for r in invalid_results:
        logger.warning("failed %s: %s", r["name"], r.get("error", "unknown"))

    _apply_composite_scores(valid_results, composite_cfg_resolved)
    if composite_sort:
        valid_results.sort(
            key=lambda r: float(r["composite_score"])
            if np.isfinite(r.get("composite_score", np.nan))
            else float("-inf"),
            reverse=True,
        )
        sort_desc = "composite_score"
    else:
        valid_results.sort(key=lambda r: r["f1"], reverse=True)
        sort_desc = metric_label
    _stage(7, f"Sorted by {sort_desc} | {len(valid_results)} valid | {len(invalid_results)} failed")
    if valid_results:
        top_r, bot_r = valid_results[0], valid_results[-1]
        if composite_sort and np.isfinite(top_r.get("composite_score", np.nan)):
            bot_cs = bot_r.get("composite_score", np.nan)
            bot_cs_txt = f"{float(bot_cs):.4f}" if np.isfinite(bot_cs) else "nan"
            logger.info(
                "Ranking complete | %d valid, %d failed | top comp=%.4f %s=%.4f (%s) | bottom comp=%s %s=%.4f (%s)",
                len(valid_results),
                len(invalid_results),
                float(top_r["composite_score"]),
                metric_label,
                top_r["f1"],
                top_r["name"],
                bot_cs_txt,
                metric_label,
                bot_r["f1"],
                bot_r["name"],
            )
        else:
            logger.info(
                "Ranking complete | %d valid, %d failed | top %s=%.4f (%s) | bottom %s=%.4f (%s)",
                len(valid_results),
                len(invalid_results),
                metric_label,
                top_r["f1"],
                top_r["name"],
                metric_label,
                bot_r["f1"],
                bot_r["name"],
            )
    all_sorted = valid_results + invalid_results

    _stage(8, f"Writing {cfg.output.csv_name}...")
    logger.info("Writing rankings CSV...")
    csv_path = out_dir / str(cfg.output.csv_name)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        import csv
        if all_sorted:
            csv_fields = [
                "rank",
                "name",
                "path",
                "metric",
                "f1",
                "composite_score",
                "continuous_dice",
                "luminance_rms_contrast",
                "luminance_gradient_mean",
                "lab_chroma_entropy",
                "lab_mean_chroma",
                "precision",
                "recall",
                "soft_precision",
                "soft_recall",
                "gradient_ssim",
                "ssim_grad_x",
                "ssim_grad_y",
                "ssim_grad_mag",
                "tp",
                "fp",
                "fn",
                "mean_local_dice",
                "mean_local_dice_pos",
            ]
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for i, r in enumerate(all_sorted, start=1):
                row = {"rank": i, "metric": eval_metric, **{k: v for k, v in r.items() if k != "error"}}
                writer.writerow(row)
    logger.info("  saved %s", csv_path)

    top_n = int(cfg.output.gallery_top_n)
    bottom_n = int(cfg.output.gallery_bottom_n)
    cols = int(cfg.output.gallery_columns)
    gallery_cell_padding = int(OmegaConf.select(cfg, "output.gallery_cell_padding", default=8) or 0)
    gallery_rows = int(OmegaConf.select(cfg, "output.gallery_rows", default=0) or 0)
    gallery_show_labels = bool(OmegaConf.select(cfg, "output.gallery_show_labels", default=True))
    gallery_prepend_hd_edge = bool(OmegaConf.select(cfg, "output.gallery_prepend_hd_edge", default=False))
    gallery_edge_colormap = str(OmegaConf.select(cfg, "output.gallery_edge_colormap", default=edge_colormap or "PuBu"))

    def _label(r: dict, rank: int) -> str:
        if composite_sort and np.isfinite(r.get("composite_score", np.nan)):
            return f"#{rank} comp={r['composite_score']:.4f} | {metric_label}={r['f1']:.4f} {r['name']}"
        return f"#{rank} {metric_label}={r['f1']:.4f} {r['name']}"

    top_tuples = [(Path(r["path"]), r, i) for i, r in enumerate(valid_results[:top_n], 1)]
    bottom_tuples = [(Path(r["path"]), r, len(valid_results) - bottom_n + i) for i, r in enumerate(valid_results[-bottom_n:] if bottom_n else [], 1)]

    gallery_panel = str(OmegaConf.select(cfg, "debug.gallery_panel", default="triple")).strip().lower()

    def _gallery_image_path(viz_path: Path) -> Path:
        if save_viz_edges and viz_edges_dir is not None and gallery_panel not in {"viz", "visualization", "original"}:
            suffix = "__triple_eq.png" if gallery_panel in {"triple_eq", "eq", "equalized"} else "__triple.png"
            panel_path = viz_edges_dir / f"{viz_path.stem}{suffix}"
            if panel_path.exists():
                return panel_path
        return viz_path

    def _load_gallery_images(tuples: list[tuple[Path, dict, int]]) -> list[tuple[Image.Image, str]]:
        images: list[tuple[Image.Image, str]] = []
        for p, r, rank in tuples:
            img_path = _gallery_image_path(p)
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                images.append((im.convert("RGB"), _label(r, rank)))
        if gallery_prepend_hd_edge:
            hd_edge_rgb = _edge_map_to_rgb(
                hd_edge_n,
                valid_mask,
                raw_max_scale=None,
                display_gamma=None,
                colormap=gallery_edge_colormap,
                equalize_png=False,
            )
            images.append((Image.fromarray(hd_edge_rgb, mode="RGB"), "HD edge"))
        return images

    def _gallery_cols(n_items: int) -> int:
        if gallery_rows > 0:
            return max(1, int(np.ceil(n_items / float(gallery_rows))))
        return cols

    _stage(9, "Building gallery_top.png / gallery_bottom.png...")
    logger.info("Creating galleries (%s panels when available)...", gallery_panel)
    if top_tuples:
        top_images = _load_gallery_images(top_tuples)
        _create_gallery_from_images(
            top_images,
            out_dir / "gallery_top.png",
            columns=_gallery_cols(len(top_images)),
            cell_padding=gallery_cell_padding,
            show_labels=gallery_show_labels,
        )
        logger.info("  saved gallery_top.png (%d images)", len(top_tuples))
    if bottom_tuples:
        bottom_images = _load_gallery_images(bottom_tuples)
        _create_gallery_from_images(
            bottom_images,
            out_dir / "gallery_bottom.png",
            columns=_gallery_cols(len(bottom_images)),
            cell_padding=gallery_cell_padding,
            show_labels=gallery_show_labels,
        )
        logger.info("  saved gallery_bottom.png (%d images)", len(bottom_tuples))

    spatial_cov_enabled = bool(OmegaConf.select(cfg, "spatial_coverage.enabled", default=False))
    if spatial_cov_enabled and valid_results:
        cov_t0 = time.perf_counter()
        cov_n = int(OmegaConf.select(cfg, "spatial_coverage.n_select", default=top_n) or top_n)
        cov_window = int(OmegaConf.select(cfg, "spatial_coverage.patch_size", default=32) or 32)
        cov_save_maps = bool(OmegaConf.select(cfg, "spatial_coverage.save_quality_maps", default=True))
        cov_first_global = bool(OmegaConf.select(cfg, "spatial_coverage.first_global_best", default=True))
        cov_colormap = str(OmegaConf.select(cfg, "spatial_coverage.colormap", default=edge_colormap or "viridis"))
        cov_rank_mode = str(OmegaConf.select(cfg, "spatial_coverage.normalize", default="per_location_rank")).strip().lower()

        def _cov_weight(name: str, default: float = 0.0) -> float:
            val = OmegaConf.select(cfg, f"spatial_coverage.weights.{name}", default=None)
            if val is None:
                val = OmegaConf.select(cfg, f"evaluation.composite.weights.{name}", default=default)
            return float(val or 0.0)

        cov_w_dice = _cov_weight("continuous_dice", 1.0)
        cov_w_contrast = _cov_weight("luminance_rms_contrast", 0.0)
        if abs(cov_w_dice) + abs(cov_w_contrast) <= 1e-12:
            cov_w_dice = 1.0

        _stage(9, f"Computing spatial coverage maps (window={cov_window}, n_select={cov_n})...")
        spatial_dir = out_dir / "spatial_coverage"
        spatial_dir.mkdir(parents=True, exist_ok=True)

        raw_quality_items: list[tuple[dict[str, Any], int, np.ndarray, np.ndarray]] = []
        for original_rank, r in enumerate(valid_results, start=1):
            viz_path = Path(r["path"])
            viz_rgb_cov = _load_visualization(viz_path, target_hw=target_hw, allow_resize=allow_resize)
            if equalize_visualization:
                viz_rgb_cov = _apply_equalize_rgb_uint8(
                    viz_rgb_cov,
                    equalize_visualization_method,
                    equalize_visualization_clahe_clip_limit,
                    equalize_visualization_clahe_tile_grid_size,
                )
            if divide_by_255:
                viz_rgb_cov = viz_rgb_cov.astype(np.float32) / 255.0

            v_edge_cov = _compute_viz_edge_for_loaded_rgb(
                viz_rgb_cov,
                valid_mask,
                sigmas,
                aggregation,
                edge_method_cfg,
                spatial_low_percentile=float(cfg.spatial_normalization.low_percentile),
                spatial_high_percentile=float(cfg.spatial_normalization.high_percentile),
                spatial_normalization_enabled=spatial_enabled,
                divide_by_255=divide_by_255,
                edge_method_cfg_override=edge_cfg_override,
                min_viz_edge_percentile=min_viz_edge_percentile,
            )
            hd_cov, v_cov = _edges_for_continuous_metrics(
                hd_edge_n,
                v_edge_cov,
                square=square_before_continuous_metrics,
            )
            local_dice_map = _local_continuous_dice_map(hd_cov, v_cov, valid_mask, cov_window)
            local_contrast_map = _local_luminance_rms_map(viz_rgb_cov, valid_mask, cov_window)
            raw_quality_items.append((r, original_rank, local_dice_map, local_contrast_map))

        if cov_rank_mode in {"per_location_rank", "rank", "local_rank"}:
            dice_component_maps = _per_location_rank_score_maps([x[2] for x in raw_quality_items], valid_mask)
            contrast_component_maps = _per_location_rank_score_maps([x[3] for x in raw_quality_items], valid_mask)
        elif cov_rank_mode in {"minmax", "min_max"}:
            dice_component_maps = _minmax_maps_on_mask([x[2] for x in raw_quality_items], valid_mask)
            contrast_component_maps = _minmax_maps_on_mask([x[3] for x in raw_quality_items], valid_mask)
        else:
            dice_component_maps = [np.where(valid_mask, x[2], 0.0).astype(np.float32) for x in raw_quality_items]
            contrast_component_maps = [np.where(valid_mask, x[3], 0.0).astype(np.float32) for x in raw_quality_items]
        quality_maps: list[np.ndarray] = []
        for idx, (r, original_rank, _local_dice_map, _local_contrast_map) in enumerate(raw_quality_items):
            denom = max(1e-12, abs(cov_w_dice) + abs(cov_w_contrast))
            q = (cov_w_dice * dice_component_maps[idx] + cov_w_contrast * contrast_component_maps[idx]) / denom
            q = np.where(valid_mask, np.clip(q, 0.0, 1.0), 0.0).astype(np.float32)
            quality_maps.append(q)
            r["spatial_quality_mean"] = float(np.mean(q[valid_mask])) if np.any(valid_mask) else float("nan")
            if cov_save_maps:
                stem = Path(r["path"]).stem
                prefix = spatial_dir / f"rank_{original_rank:03d}__{stem}__spatial_quality"
                _save_map_png_and_npy(
                    prefix,
                    q,
                    save_npy=save_npy,
                    equalize_png=False,
                    equalize_method="hist",
                    clahe_clip_limit=2.0,
                    clahe_tile_grid_size=8,
                    colormap=cov_colormap,
                )

        selected: list[int] = []
        current = np.zeros_like(quality_maps[0], dtype=np.float32)
        valid = valid_mask.astype(bool)
        if cov_first_global:
            selected.append(0)
            current = np.maximum(current, quality_maps[0])

        step_rows: list[dict[str, Any]] = []
        steps: list[int] = []
        sums: list[float] = []

        def _quality_sum(x: np.ndarray) -> float:
            return float(np.sum(x[valid])) if np.any(valid) else float("nan")

        if selected:
            qsum = _quality_sum(current)
            steps.append(1)
            sums.append(qsum)
            step_rows.append(
                {
                    "selection_rank": 1,
                    "rank": 1,
                    "name": valid_results[0]["name"],
                    "path": valid_results[0]["path"],
                    "quality_sum": qsum,
                    "quality_mean": float(np.mean(current[valid])) if np.any(valid) else float("nan"),
                    "marginal_gain": qsum,
                }
            )

        top_steps: list[int] = []
        top_sums: list[float] = []
        top_step_rows: list[dict[str, Any]] = []
        top_current = np.zeros_like(quality_maps[0], dtype=np.float32)
        for idx in range(min(top_n, len(quality_maps))):
            prev_sum = _quality_sum(top_current) if idx > 0 else 0.0
            top_current = np.maximum(top_current, quality_maps[idx])
            qsum = _quality_sum(top_current)
            top_steps.append(idx + 1)
            top_sums.append(qsum)
            r = valid_results[idx]
            top_step_rows.append(
                {
                    "selection_rank": idx + 1,
                    "rank": idx + 1,
                    "name": r["name"],
                    "path": r["path"],
                    "quality_sum": qsum,
                    "quality_mean": float(np.mean(top_current[valid])) if np.any(valid) else float("nan"),
                    "marginal_gain": qsum - prev_sum,
                }
            )

        while len(selected) < min(cov_n, len(quality_maps)):
            best_i = None
            best_sum = -np.inf
            best_merged = None
            prev_sum = _quality_sum(current) if selected else 0.0
            for i, q in enumerate(quality_maps):
                if i in selected:
                    continue
                merged = np.maximum(current, q)
                qsum = _quality_sum(merged)
                if qsum > best_sum:
                    best_sum = qsum
                    best_i = i
                    best_merged = merged
            if best_i is None or best_merged is None:
                break
            selected.append(best_i)
            current = best_merged
            steps.append(len(selected))
            sums.append(best_sum)
            r = valid_results[best_i]
            step_rows.append(
                {
                    "selection_rank": len(selected),
                    "rank": best_i + 1,
                    "name": r["name"],
                    "path": r["path"],
                    "quality_sum": best_sum,
                    "quality_mean": float(np.mean(current[valid])) if np.any(valid) else float("nan"),
                    "marginal_gain": best_sum - prev_sum,
                }
            )

        _write_spatial_coverage_csv(spatial_dir / "spatial_coverage_selection.csv", step_rows)
        _write_spatial_coverage_csv(spatial_dir / "gallery_top_spatial_coverage_selection.csv", top_step_rows)
        _save_spatial_coverage_plot(steps, sums, spatial_dir / "spatial_coverage_quality_sum.png")
        _save_spatial_coverage_plot(top_steps, top_sums, spatial_dir / "gallery_top_spatial_coverage_quality_sum.png")
        _save_spatial_coverage_comparison_plot(
            [
                ("greedy spatial coverage", steps, sums),
                ("gallery_top order", top_steps, top_sums),
            ],
            spatial_dir / "spatial_coverage_quality_sum_comparison.png",
        )
        _save_map_png_and_npy(
            spatial_dir / "spatial_coverage_max_quality",
            current,
            save_npy=save_npy,
            equalize_png=False,
            equalize_method="hist",
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
            colormap=cov_colormap,
        )
        selected_images: list[tuple[Image.Image, str]] = []
        for sel_rank, idx in enumerate(selected, start=1):
            r = valid_results[idx]
            img_path = _gallery_image_path(Path(r["path"]))
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                selected_images.append(
                    (
                        im.convert("RGB"),
                        f"#{sel_rank} rank={idx + 1} q={r.get('spatial_quality_mean', float('nan')):.4f} {r['name']}",
                    )
                )
        hd_edge_rgb = _edge_map_to_rgb(
            hd_edge_n,
            valid_mask,
            raw_max_scale=None,
            display_gamma=None,
            colormap=gallery_edge_colormap,
            equalize_png=False,
        )
        selected_images.append((Image.fromarray(hd_edge_rgb, mode="RGB"), "HD edge"))
        if selected_images:
            _create_gallery_from_images(
                selected_images,
                spatial_dir / "spatial_coverage_gallery.png",
                columns=_gallery_cols(len(selected_images)),
                cell_padding=gallery_cell_padding,
                show_labels=gallery_show_labels,
            )
        logger.info("spatial coverage saved to %s in %.2fs", spatial_dir, time.perf_counter() - cov_t0)

    _stage(10, f"Done | total {time.perf_counter() - t0:.2f}s | outputs in {out_dir}")
    logger.info("Done | total %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
