#!/usr/bin/env python3
"""
Training and inference script for MSIParametricMiCSLMC / MSIParametricMiCSUnet / MSIParametricMiCSMoE with Hydra.

Supports:
- Training: Single .npy file or folder (`model.model_type`: ``mics_lmc``, ``mics_unet``, or ``mixture_of_experts``).
- Inference: Load trained model and run visualization on a folder/glob of .npy files.
- Hyperparameter sweeps via Hydra; checkpoints record ``model_class`` for correct reload.
"""

import gc
import os
import glob
import traceback
from typing import Any, Optional
import numpy as np
from pathlib import Path
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from msi_visual.parametric_mics_lmc import (
    MSIParametricMiCSLMC,
    build_miss_stratified_weight_map,
    create_pcc_model,
)
from msi_visual.parametric_mics_moe import MSIParametricMiCSMoE
from msi_visual.parametric_mics_unet import MSIParametricMiCSUnet, SmallUNet
from msi_visual.quality_metrics import (
    compute_highd_summary,
    save_quality_score_image,
    save_edge_quality_image,
    save_structure_quality_image,
    save_contrast_quality_image,
)


def _omega_sel_to_plain_dict(sel: Any) -> dict[str, Any]:
    """Hydra OmegaConf subtree or plain dict → plain dict (``to_container`` rejects raw dict defaults)."""
    if sel is None:
        return {}
    if OmegaConf.is_config(sel):
        oc = OmegaConf.to_container(sel, resolve=True)
        return dict(oc) if isinstance(oc, dict) else {}
    if isinstance(sel, dict):
        return dict(sel)
    return {}


def _clusters_auto_tune_kw_from_cfg(cfg_model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "clusters_auto_tune": OmegaConf.select(cfg_model, "clusters_auto_tune", default=None),
        "clusters_auto_tune_k_min": int(
            OmegaConf.select(cfg_model, "clusters_auto_tune_k_min", default=2),
        ),
        "clusters_auto_tune_k_max": int(
            OmegaConf.select(cfg_model, "clusters_auto_tune_k_max", default=256),
        ),
        "clusters_auto_tune_n_candidates": int(
            OmegaConf.select(cfg_model, "clusters_auto_tune_n_candidates", default=20),
        ),
        "clusters_auto_tune_ladder": str(
            OmegaConf.select(cfg_model, "clusters_auto_tune_ladder", default="legacy"),
        ),
        "clusters_auto_tune_k_fine": int(
            OmegaConf.select(cfg_model, "clusters_auto_tune_k_fine", default=2048),
        ),
        "edge_stratified_grid_cells": int(
            OmegaConf.select(cfg_model, "edge_stratified_grid_cells", default=16),
        ),
        "edge_stratified_edge_floor": float(
            OmegaConf.select(cfg_model, "edge_stratified_edge_floor", default=0.05),
        ),
        "edge_stratified_uniform_mix": float(
            OmegaConf.select(cfg_model, "edge_stratified_uniform_mix", default=0.15),
        ),
    }
    raw_step = OmegaConf.select(cfg_model, "clusters_auto_tune_k_step", default=None)
    try:
        si = int(raw_step) if raw_step is not None else None
    except (TypeError, ValueError):
        si = None
    if si is not None and int(si) >= 1:
        out["clusters_auto_tune_k_step"] = int(si)
    return out


def _miss_stratified_kw_from_cfg(cfg_model: Any) -> dict[str, Any]:
    ms = _omega_sel_to_plain_dict(OmegaConf.select(cfg_model, "miss_stratified", default={}))
    if not ms:
        return {}
    return {
        "miss_stratified_quality_gamma": float(ms.get("quality_gamma", 2.0)),
        "miss_stratified_hd_edge_floor": float(ms.get("hd_edge_floor", 0.05)),
        "miss_stratified_uniformity_floor": float(ms.get("uniformity_floor", 0.02)),
        "miss_stratified_uniform_mix": float(ms.get("uniform_mix", 0.0)),
        "miss_stratified_quota_by_weight": bool(ms.get("quota_by_weight", True)),
        "miss_stratified_sampler": str(ms.get("sampler", "weighted")),
        "miss_stratified_weight_percentile": _parse_optional_float(
            ms.get("weight_percentile", 65.0)
        ),
        "miss_stratified_top_k": _parse_optional_int(ms.get("top_k")),
        "miss_stratified_additive": bool(ms.get("additive", True)),
        "finetune_miss_weighted_batches": bool(ms.get("finetune_weighted_batches", False)),
    }


def _parse_optional_int(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("", "~", "null", "none"):
        return None
    return int(val)


def _apply_miss_kwargs_to_model(model: MSIParametricMiCSLMC, model_kwargs: dict[str, Any]) -> None:
    """Copy miss-stratified sampling fields onto an existing model (finetune path)."""
    model.pixel_sampling = "miss_stratified"
    for key in (
        "spatial_quality_map_for_sampling",
        "hd_edge_map_for_sampling",
        "miss_stratified_quality_gamma",
        "miss_stratified_hd_edge_floor",
        "miss_stratified_uniformity_floor",
        "miss_stratified_uniform_mix",
        "miss_stratified_quota_by_weight",
        "miss_stratified_sampler",
        "miss_stratified_weight_percentile",
        "miss_stratified_top_k",
        "miss_stratified_additive",
        "prior_sampled_flat_indices",
        "finetune_miss_weighted_batches",
    ):
        if key in model_kwargs:
            setattr(model, key, model_kwargs[key])


def _parse_optional_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("", "~", "null", "none"):
        return None
    return float(val)


def _coerce_model_clusters_list(cfg_model: Any) -> Optional[list[int]]:
    cluster_on = bool(OmegaConf.select(cfg_model, "cluster", default=False))
    if not cluster_on:
        return None
    raw = OmegaConf.select(cfg_model, "clusters")
    if raw is None:
        return []
    if OmegaConf.is_config(raw):
        try:
            if OmegaConf.is_list(raw):
                cont = OmegaConf.to_container(raw, resolve=True)
                if isinstance(cont, list):
                    return [int(x) for x in cont]
        except Exception:
            pass
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    s = str(raw).strip()
    slo = s.lower()
    if not s or slo in ("null", "~", "none"):
        return []
    return [int(x) for x in str(raw).split("-") if str(x).strip()]


def _parse_mixture_of_experts_cfg(model_cfg: Any) -> Optional[dict[str, Any]]:
    mt = str(OmegaConf.select(model_cfg, "model_type", default="mics_lmc")).lower().strip()
    if mt != "mixture_of_experts":
        return None
    mo = _omega_sel_to_plain_dict(OmegaConf.select(model_cfg, "mixture_of_experts", default={}))
    experts_raw = mo.get("experts")
    if not experts_raw or not isinstance(experts_raw, (list, tuple)):
        raise ValueError(
            "model_type=mixture_of_experts requires model.mixture_of_experts.experts as a non-empty list"
        )
    base_nl = int(OmegaConf.select(model_cfg, "num_layers", default=2))
    base_fac = float(OmegaConf.select(model_cfg, "factor", default=1.0))
    specs: list[dict[str, Any]] = []
    for item in experts_raw:
        row = _omega_sel_to_plain_dict(item) if OmegaConf.is_config(item) else dict(item)
        spec_row: dict[str, Any] = {
            "num_layers": int(row.get("num_layers", base_nl)),
            "factor": float(row.get("factor", base_fac)),
        }
        for kopt in ("clusters", "cluster_on_pca", "cluster_pca_dims"):
            if kopt in row:
                spec_row[kopt] = row[kopt]
        specs.append(spec_row)
    fusion = str(mo.get("fusion", "mean")).strip().lower().replace("-", "_")
    if fusion not in {"mean", "softmax_gate"}:
        raise ValueError("mixture_of_experts.fusion must be 'mean' or 'softmax_gate'")
    gh = mo.get("gate_hidden", mo.get("gate_hidden_dim", 0))
    raw_sd = mo.get("shared_cluster_pca_dims")
    shared_dims_out: Optional[int]
    if raw_sd is None:
        shared_dims_out = None
    else:
        s = str(raw_sd).strip().lower()
        if s in ("~", "null", "none", ""):
            shared_dims_out = None
        else:
            shared_dims_out = int(raw_sd)
    return {
        "expert_specs": specs,
        "moe_fusion": fusion,
        "moe_layer_norm_before_fuse": bool(mo.get("layer_norm_before_fuse", True)),
        "moe_gate_temperature": float(mo.get("gate_temperature", 1.0)),
        "moe_gate_hidden_dim": int(gh or 0),
        "moe_share_cluster_pca": bool(mo.get("share_cluster_pca", False)),
        "moe_shared_cluster_pca_dims": shared_dims_out,
    }


def _build_hd_edge_map_for_train(cfg: DictConfig, img: np.ndarray) -> np.ndarray:
    """HD reference edge map (H,W) in [0,1] — same pipeline as bayes_tune / edge_stratified sampling."""
    hp = OmegaConf.select(cfg, "model.hd_edge_map_path", default=None)
    if hp is not None and str(hp).strip().lower() not in ("", "~", "null", "none"):
        pth = Path(str(hp))
        try:
            from hydra.utils import to_absolute_path

            if not pth.is_file():
                pth = Path(to_absolute_path(str(hp)))
        except Exception:
            pass
        if not pth.is_file():
            raise FileNotFoundError(f"model.hd_edge_map_path not found: {hp}")
        edge_map = np.squeeze(np.load(pth)).astype(np.float64)
        if edge_map.shape != img.shape[:2]:
            raise ValueError(f"hd_edge_map_path shape {edge_map.shape} != MSI spatial {img.shape[:2]}")
        return edge_map

    rank_sel = OmegaConf.select(cfg, "rank_config", default=None)
    if rank_sel is None or str(rank_sel).strip() == "":
        return compute_hd_edge_target_simple(img)

    from hydra.utils import to_absolute_path
    from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method

    rank_cfg = OmegaConf.load(to_absolute_path(str(rank_sel)))
    valid_mask = img.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty; cannot build HD edge map")
    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    if multi is None or not bool(getattr(multi, "enabled", False)):
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    spatial_enabled = bool(getattr(rank_cfg.spatial_normalization, "enabled", True))
    hd_method = str(OmegaConf.select(rank_cfg, "edge_detection.hd_method", default="soft_landmark_contrast"))
    method_for_map = "soft_landmark_contrast"
    if hd_method != method_for_map:
        print(
            f"[train] rank_config hd_method={hd_method}; using {method_for_map} for HD edge map"
        )
    return _compute_hd_edge_n_for_method(
        rank_cfg,
        img,
        valid_mask,
        getattr(rank_cfg, "hd_edges", None),
        method_for_map,
        sigmas,
        aggregation,
        spatial_enabled,
    )


def _attach_hd_edge_map_for_sampling_if_needed(
    cfg: DictConfig,
    img: np.ndarray,
    model_kwargs: dict[str, Any],
) -> None:
    """Set ``hd_edge_map_for_sampling`` when ``pixel_sampling`` is edge- or miss-stratified."""
    ps = str(model_kwargs.get("pixel_sampling", "superpixel")).lower()
    if ps not in ("edge_stratified", "edge_stratified_hd", "miss_stratified", "miss"):
        return
    model_kwargs["hd_edge_map_for_sampling"] = _build_hd_edge_map_for_train(cfg, img)
    if OmegaConf.select(cfg, "model.hd_edge_map_path", default=None) is not None:
        print("[train] edge_stratified: HD edge map from model.hd_edge_map_path")
    elif OmegaConf.select(cfg, "rank_config", default=None):
        print("[train] edge_stratified: soft_landmark_contrast HD edge map from rank_config")
    else:
        print("[train] edge_stratified: Sobel ion-sum HD edge proxy (no rank_config)")


def compute_spatial_edge_quality_arrays(
    cfg: DictConfig,
    img: np.ndarray,
    viz_rgb: np.ndarray,
    *,
    hd_edge_n: np.ndarray | None = None,
) -> dict[str, Any]:
    """Patch-wise HD↔viz edge agreement (rank_visualizations spatial_coverage logic)."""
    rank_sel = OmegaConf.select(cfg, "rank_config", default=None)
    if rank_sel is None or str(rank_sel).strip() == "":
        raise ValueError("rank_config required for spatial edge quality / miss_stratified weights")

    from hydra.utils import to_absolute_path
    from bayes_tune_mics_lmc import _compute_viz_edge_n, _to_uint8_rgb
    from debug_edge_maps import _compute_continuous_dice, _edges_for_continuous_metrics
    from rank_visualizations_by_f1 import (
        _local_continuous_dice_map,
        _local_lab_chroma_entropy_map,
        _local_luminance_rms_map,
    )

    rank_cfg = OmegaConf.load(to_absolute_path(str(rank_sel)))
    valid_mask = np.asarray(img.sum(axis=-1) > 0, dtype=bool)
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty; cannot compute spatial edge quality")

    if hd_edge_n is None:
        hd_edge_n = _build_hd_edge_map_for_train(cfg, img)
    hd_edge_n = np.asarray(hd_edge_n, dtype=np.float64)

    viz_u8 = _to_uint8_rgb(viz_rgb)
    square = bool(
        OmegaConf.select(rank_cfg, "evaluation.square_before_continuous_metrics", default=False)
    )
    viz_edge_n = _compute_viz_edge_n(viz_u8, valid_mask, rank_cfg)
    hd_d, v_d = _edges_for_continuous_metrics(hd_edge_n, viz_edge_n, square=square)

    window = int(OmegaConf.select(cfg, "spatial_quality.patch_size", default=16))
    local_dice = _local_continuous_dice_map(hd_d, v_d, valid_mask, window)

    w_dice = float(OmegaConf.select(cfg, "spatial_quality.weights.continuous_dice", default=1.0))
    w_contrast = float(
        OmegaConf.select(cfg, "spatial_quality.weights.luminance_rms_contrast", default=0.0)
    )
    w_lab = float(OmegaConf.select(cfg, "spatial_quality.weights.lab_chroma_entropy", default=1.0))
    bins_a = int(OmegaConf.select(cfg, "spatial_quality.lab_entropy_bins_a", default=16))
    bins_b = int(OmegaConf.select(cfg, "spatial_quality.lab_entropy_bins_b", default=16))
    lab_stride = OmegaConf.select(cfg, "spatial_quality.lab_entropy_stride", default=None)
    if lab_stride is None:
        lab_stride_i = max(1, window // 4)
    else:
        lab_stride_i = max(1, int(lab_stride))

    def _minmax_component(raw: np.ndarray) -> np.ndarray:
        fin = raw[valid_mask]
        fin = fin[np.isfinite(fin)]
        if fin.size:
            lo, hi = float(fin.min()), float(fin.max())
            return np.where(
                valid_mask,
                np.clip((raw - lo) / max(hi - lo, 1e-12), 0.0, 1.0),
                0.0,
            ).astype(np.float32)
        return np.zeros_like(raw, dtype=np.float32)

    contrast_norm = np.zeros_like(local_dice, dtype=np.float32)
    if abs(w_contrast) > 1e-12:
        local_contrast = _local_luminance_rms_map(viz_u8, valid_mask, window)
        contrast_norm = _minmax_component(local_contrast)
    else:
        local_contrast = np.zeros_like(local_dice, dtype=np.float32)

    lab_entropy_norm = np.zeros_like(local_dice, dtype=np.float32)
    if abs(w_lab) > 1e-12:
        local_lab = _local_lab_chroma_entropy_map(
            viz_u8,
            valid_mask,
            window,
            bins_a=bins_a,
            bins_b=bins_b,
            stride=lab_stride_i,
        )
        lab_entropy_norm = _minmax_component(local_lab)
    else:
        local_lab = np.zeros_like(local_dice, dtype=np.float32)

    terms: list[tuple[float, np.ndarray]] = []
    if abs(w_dice) > 1e-12:
        terms.append((w_dice, local_dice))
    if abs(w_contrast) > 1e-12:
        terms.append((w_contrast, contrast_norm))
    if abs(w_lab) > 1e-12:
        terms.append((w_lab, lab_entropy_norm))
    if not terms:
        terms.append((1.0, local_dice))

    denom = max(1e-12, sum(abs(w) for w, _ in terms))
    quality = sum(float(w) * arr for w, arr in terms) / denom
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)
    quality[~valid_mask] = 0.0

    global_dice = float(_compute_continuous_dice(hd_d, v_d, valid_mask))
    mean_q = float(np.mean(quality[valid_mask])) if np.any(valid_mask) else float("nan")
    return {
        "quality": quality,
        "local_dice": local_dice,
        "contrast_norm": contrast_norm,
        "lab_entropy_norm": lab_entropy_norm,
        "local_lab_entropy": local_lab,
        "local_luminance_rms": local_contrast,
        "hd_edge_n": hd_edge_n,
        "viz_edge_n": viz_edge_n,
        "valid_mask": valid_mask,
        "global_dice": global_dice,
        "mean_q": mean_q,
        "window": window,
        "viz_u8": viz_u8,
    }


def _edge_agreement_metrics(
    cfg: DictConfig,
    img: np.ndarray,
    viz_rgb: np.ndarray,
) -> dict[str, float]:
    """Global HD↔viz edge dice and mean spatial recall after a fit."""
    sq = compute_spatial_edge_quality_arrays(cfg, img, viz_rgb)
    vm = sq["valid_mask"]
    q = sq["quality"]
    return {
        "global_dice": float(sq["global_dice"]),
        "mean_recall": float(np.mean(q[vm])) if np.any(vm) else float("nan"),
        "mean_local_dice": float(np.mean(sq["local_dice"][vm])) if np.any(vm) else float("nan"),
    }


def _prepare_miss_stratified_sampling(
    cfg: DictConfig,
    img: np.ndarray,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Bootstrap fit → spatial quality → miss weight map for ``pixel_sampling=miss_stratified``."""
    ms = _omega_sel_to_plain_dict(OmegaConf.select(cfg, "model.miss_stratified", default={}))
    two_pass = bool(ms.get("two_pass", True))
    q_path = ms.get("quality_map_path") or OmegaConf.select(
        cfg, "model.spatial_quality_map_path", default=None
    )
    hd_edge = _build_hd_edge_map_for_train(cfg, img)
    valid_mask = np.asarray(img.sum(axis=-1) > 0, dtype=bool)

    gamma = float(
        ms.get("quality_gamma", model_kwargs.get("miss_stratified_quality_gamma", 2.0))
    )
    hd_floor = float(
        ms.get("hd_edge_floor", model_kwargs.get("miss_stratified_hd_edge_floor", 0.05))
    )
    uni_floor = float(
        ms.get("uniformity_floor", model_kwargs.get("miss_stratified_uniformity_floor", 0.02))
    )

    quality: np.ndarray | None = None
    viz_boot: np.ndarray | None = None
    sq: dict[str, Any] | None = None
    refit_mode = str(ms.get("refit_mode", "finetune")).strip().lower()
    boot_model: MSIParametricMiCSLMC | None = None
    boot_prior_flat: np.ndarray | None = None
    if q_path is not None and str(q_path).strip().lower() not in ("", "~", "null", "none"):
        from hydra.utils import to_absolute_path

        pth = Path(str(q_path))
        if not pth.is_file():
            pth = Path(to_absolute_path(str(q_path)))
        if not pth.is_file():
            raise FileNotFoundError(f"miss_stratified quality_map_path not found: {q_path}")
        quality = np.squeeze(np.load(pth)).astype(np.float32)
        if quality.shape != img.shape[:2]:
            raise ValueError(f"quality map shape {quality.shape} != MSI spatial {img.shape[:2]}")
        print(f"[train] miss_stratified: quality map from {pth}")
    elif two_pass:
        boot_ps = str(ms.get("bootstrap_sampling", "superpixel")).lower()
        boot_kw = dict(model_kwargs)
        boot_kw["pixel_sampling"] = boot_ps
        boot_ep = ms.get("bootstrap_epochs")
        if boot_ep is not None:
            boot_kw["num_epochs"] = int(boot_ep)
        if boot_ps in ("edge_stratified", "edge_stratified_hd"):
            _attach_hd_edge_map_for_sampling_if_needed(cfg, img, boot_kw)
        print(
            f"[train] miss_stratified: bootstrap fit with pixel_sampling={boot_ps} "
            f"epochs={boot_kw.get('num_epochs')}"
        )
        boot_model = MSIParametricMiCSLMC(**boot_kw)
        boot_model.fit(img)
        if getattr(boot_model, "sampled_flat_indices", None) is not None:
            boot_prior_flat = np.asarray(
                boot_model.sampled_flat_indices, dtype=np.int64
            ).copy()
        viz_boot = boot_model.predict(img)
        sq = compute_spatial_edge_quality_arrays(cfg, img, viz_boot, hd_edge_n=hd_edge)
        quality = sq["quality"]
        print(
            f"[train] bootstrap edge agreement: global_dice={sq['global_dice']:.4f} "
            f"mean_recall={float(np.mean(quality[valid_mask])):.4f}"
        )
    else:
        raise ValueError(
            "miss_stratified requires model.miss_stratified.two_pass=true (default) "
            "or model.miss_stratified.quality_map_path / model.spatial_quality_map_path"
        )

    miss_quality_mode = str(ms.get("miss_quality", "dice_only")).strip().lower()
    quality_for_miss = quality
    if (
        miss_quality_mode in ("dice", "dice_only")
        and sq is not None
        and "local_dice" in sq
    ):
        quality_for_miss = np.asarray(sq["local_dice"], dtype=np.float32)
        print("[train] miss map uses local edge dice only (not LAB entropy blend)")
    elif miss_quality_mode not in ("full", "blend", ""):
        print(f"[train] miss_quality={miss_quality_mode!r} unknown; using full recall blend")

    miss_weight = build_miss_stratified_weight_map(
        hd_edge,
        quality_for_miss,
        valid_mask,
        quality_gamma=gamma,
        hd_edge_floor=hd_floor,
        uniformity_floor=uni_floor,
    )
    model_kwargs["pixel_sampling"] = "miss_stratified"
    model_kwargs["hd_edge_map_for_sampling"] = hd_edge
    model_kwargs["spatial_quality_map_for_sampling"] = miss_weight
    if boot_prior_flat is not None:
        model_kwargs["prior_sampled_flat_indices"] = boot_prior_flat
    mean_miss = float(np.mean(miss_weight[valid_mask])) if np.any(valid_mask) else float("nan")
    print(
        f"[train] miss_stratified: weight mean={mean_miss:.4f} "
        f"(γ={gamma}, hd_floor={hd_floor}, uniform={uni_floor})"
    )
    out = {
        "miss_weight": miss_weight,
        "hd_edge": hd_edge,
        "quality": quality,
        "recall": quality,
        "valid_mask": valid_mask,
        "viz_bootstrap": viz_boot,
        "refit_mode": refit_mode,
        "finetune_epochs": int(ms.get("finetune_epochs", 10)),
        "finetune_lr_scale": float(ms.get("finetune_lr_scale", 0.1)),
        "finetune_warmup_epochs": int(ms.get("finetune_warmup_epochs", 0)),
        "bootstrap_sampled_flat_indices": boot_prior_flat,
    }
    if refit_mode in ("finetune", "continue", "warm_start") and boot_model is not None:
        _apply_miss_kwargs_to_model(boot_model, model_kwargs)
        out["boot_model"] = boot_model
    elif boot_model is not None:
        boot_model.release_resources()
        del boot_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def build_sampled_miss_pixels_map(
    height: int,
    width: int,
    sampled_flat_indices: np.ndarray,
    valid_mask: np.ndarray,
    miss_weight: np.ndarray | None = None,
    *,
    dot_radius: int = 2,
) -> np.ndarray:
    """Raster of main-fit training pixels; value = miss weight at each sampled location."""
    m = np.zeros((height, width), dtype=np.float32)
    flat = np.asarray(sampled_flat_indices, dtype=np.int64).ravel()
    n_pix = height * width
    flat = flat[(flat >= 0) & (flat < n_pix)]
    if flat.size == 0:
        return m

    if miss_weight is not None:
        w = np.asarray(miss_weight, dtype=np.float32).ravel()
        vals = w[flat]
    else:
        vals = np.ones(flat.shape[0], dtype=np.float32)
    np.maximum.at(m.ravel(), flat, vals)

    r = max(0, int(dot_radius))
    if r > 0:
        k = 2 * r + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel)

    m[~np.asarray(valid_mask, dtype=bool)] = 0.0
    return m


def build_sampling_category_map(
    height: int,
    width: int,
    sampled_flat_indices: np.ndarray,
    valid_mask: np.ndarray,
    bootstrap_sampled_flat_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Raster how pixels were chosen for training (uint8).

    0 = not sampled, 1 = bootstrap / prior only, 2 = phase-2 miss (new) only.
    """
    h, w = int(height), int(width)
    m = np.zeros((h, w), dtype=np.uint8)
    n_pix = h * w
    final = np.asarray(sampled_flat_indices, dtype=np.int64).ravel()
    final = final[(final >= 0) & (final < n_pix)]
    counts = {"total": int(final.size), "bootstrap_only": 0, "miss_new_only": 0}
    if final.size == 0:
        return m, counts

    if bootstrap_sampled_flat_indices is not None:
        boot = np.unique(
            np.asarray(bootstrap_sampled_flat_indices, dtype=np.int64).ravel()
        )
        boot = boot[(boot >= 0) & (boot < n_pix)]
        boot_set = set(boot.tolist())
        for f in final:
            if int(f) in boot_set:
                m.ravel()[int(f)] = 1
                counts["bootstrap_only"] += 1
            else:
                m.ravel()[int(f)] = 2
                counts["miss_new_only"] += 1
    else:
        m.ravel()[final] = 2
        counts["miss_new_only"] = int(final.size)
        counts["bootstrap_only"] = 0

    m[~np.asarray(valid_mask, dtype=bool)] = 0
    return m, counts


def render_sampling_viz_rgb(
    base_rgb: np.ndarray,
    sampled_flat_indices: np.ndarray,
    height: int,
    width: int,
    valid_mask: np.ndarray,
    *,
    bootstrap_sampled_flat_indices: np.ndarray | None = None,
    dot_radius: int = 1,
    ring_alpha: float = 0.85,
    ring_width: int = 1,
    base_dim: float = 0.45,
    color_bootstrap: tuple[int, int, int] = (80, 200, 255),
    color_miss_new: tuple[int, int, int] = (255, 220, 50),
) -> np.ndarray:
    """RGB map of training pixels over bootstrap viz (cyan=prior, yellow=new misses)."""
    from bayes_tune_mics_lmc import _to_uint8_rgb

    base = np.asarray(_to_uint8_rgb(base_rgb), dtype=np.uint8)[..., :3].copy()
    dim = float(np.clip(base_dim, 0.0, 1.0))
    out = (base.astype(np.float32) * dim).astype(np.uint8)
    vm = np.asarray(valid_mask, dtype=bool)
    out[~vm] = 0

    n_pix = int(height) * int(width)
    final = np.asarray(sampled_flat_indices, dtype=np.int64).ravel()
    final = final[(final >= 0) & (final < n_pix)]
    if final.size == 0:
        return out

    boot_set: set[int] = set()
    if bootstrap_sampled_flat_indices is not None:
        boot = np.asarray(bootstrap_sampled_flat_indices, dtype=np.int64).ravel()
        boot = boot[(boot >= 0) & (boot < n_pix)]
        boot_set = set(int(x) for x in boot.tolist())

    boot_flat = np.array([f for f in final if int(f) in boot_set], dtype=np.int64)
    new_flat = np.array([f for f in final if int(f) not in boot_set], dtype=np.int64)

    if boot_flat.size:
        out = draw_training_samples_on_rgb(
            out,
            boot_flat,
            height,
            width,
            dot_radius=dot_radius,
            dot_color=color_bootstrap,
            ring_alpha=ring_alpha,
            ring_width=ring_width,
            hollow=True,
        )
    if new_flat.size:
        out = draw_training_samples_on_rgb(
            out,
            new_flat,
            height,
            width,
            dot_radius=dot_radius,
            dot_color=color_miss_new,
            ring_alpha=ring_alpha,
            ring_width=ring_width,
            hollow=True,
        )
    return out


def _save_sampling_category_png(path: Path, category_map: np.ndarray) -> None:
    """Save discrete 0/1/2 sampling map as RGB (black / cyan / yellow)."""
    lut = np.array(
        [
            [0, 0, 0],
            [80, 200, 255],
            [255, 220, 50],
        ],
        dtype=np.uint8,
    )
    rgb = lut[np.clip(category_map, 0, 2)]
    Image.fromarray(rgb, mode="RGB").save(path)


def draw_training_samples_on_rgb(
    base_rgb: np.ndarray,
    sampled_flat_indices: np.ndarray,
    height: int,
    width: int,
    *,
    dot_radius: int = 1,
    dot_color: tuple[int, int, int] = (255, 230, 40),
    ring_alpha: float = 0.55,
    ring_width: int = 1,
    hollow: bool = True,
) -> np.ndarray:
    """Overlay main-fit training pixels as small hollow rings (see-through on recall map)."""
    out = np.asarray(base_rgb, dtype=np.uint8)[..., :3].copy()
    flat = np.asarray(sampled_flat_indices, dtype=np.int64).ravel()
    n_pix = height * width
    flat = flat[(flat >= 0) & (flat < n_pix)]
    if flat.size == 0:
        return out
    rows, cols = np.unravel_index(flat, (height, width))
    r = max(1, int(dot_radius))
    alpha = float(np.clip(ring_alpha, 0.0, 1.0))
    thickness = max(1, int(ring_width))
    rings = np.zeros_like(out)
    for row, col in zip(rows.tolist(), cols.tolist()):
        if hollow:
            cv2.circle(
                rings,
                (int(col), int(row)),
                r,
                dot_color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.circle(rings, (int(col), int(row)), r, dot_color, -1, lineType=cv2.LINE_AA)
    if alpha >= 1.0 - 1e-6:
        return np.where(rings > 0, rings, out)
    return cv2.addWeighted(out, 1.0 - alpha, rings, alpha, 0.0)


def save_miss_stratified_maps(
    cfg: DictConfig,
    output_dir: Path,
    stem: str,
    miss_weight: np.ndarray,
    viz_low_quality_fit: np.ndarray,
    valid_mask: np.ndarray,
    *,
    recall: np.ndarray | None = None,
    viz_original: np.ndarray | None = None,
    sampled_flat_indices: np.ndarray | None = None,
    bootstrap_sampled_flat_indices: np.ndarray | None = None,
) -> dict[str, str]:
    """Save recall / miss maps and 3-column panel: bootstrap viz | recall | low-recall refit viz."""
    ms = _omega_sel_to_plain_dict(OmegaConf.select(cfg, "model.miss_stratified", default={}))
    save_maps = bool(OmegaConf.select(cfg, "model.miss_stratified.save_maps", default=True))
    save_panel = bool(ms.get("save_panel", True))
    if not save_maps and not save_panel:
        return {}

    from bayes_tune_mics_lmc import _to_uint8_rgb
    from debug_edge_maps import _save_map_png_and_npy
    from debug_edge_maps import _gray_u8_to_rgb_colormap, _to_uint8_gray01
    miss_cmap = str(ms.get("colormap", "hot"))
    recall_cmap = str(
        ms.get("recall_colormap", OmegaConf.select(cfg, "spatial_quality.colormap", default="PuBu"))
    )
    save_npy = bool(ms.get("save_npy", True))
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    if save_maps and bool(ms.get("save_miss_weight_map", True)):
        base = output_dir / f"{stem}_miss_stratified_weight"
        _save_map_png_and_npy(
            base,
            miss_weight,
            save_npy=save_npy,
            equalize_png=False,
            equalize_method="hist",
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
            colormap=miss_cmap,
        )
        out["weight"] = str(base.with_suffix(".png"))

    # Sampling target for top-k / weighted miss_stratified: (1-recall)^γ × HD edge
    if save_maps and bool(ms.get("save_miss_map", True)):
        miss_base = output_dir / f"{stem}_miss"
        _save_map_png_and_npy(
            miss_base,
            miss_weight,
            save_npy=save_npy,
            equalize_png=bool(ms.get("miss_map_equalize", True)),
            equalize_method="hist",
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
            colormap=str(ms.get("miss_colormap", miss_cmap)),
        )
        out["miss"] = str(miss_base.with_suffix(".png"))
        print(f"[train] miss sampling map (top-k ranks this): {out['miss']}")

    if (
        save_maps
        and bool(ms.get("save_miss_badness_map", True))
        and recall is not None
    ):
        gamma = float(ms.get("quality_gamma", 2.0))
        q = np.clip(np.asarray(recall, dtype=np.float32), 0.0, 1.0)
        bad = np.power(np.clip(1.0 - q, 0.0, 1.0), max(0.1, gamma))
        bad = np.where(valid_mask, bad, 0.0).astype(np.float32)
        bad_base = output_dir / f"{stem}_miss_badness"
        _save_map_png_and_npy(
            bad_base,
            bad,
            save_npy=save_npy,
            equalize_png=bool(ms.get("miss_map_equalize", True)),
            equalize_method="hist",
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
            colormap=str(ms.get("miss_colormap", miss_cmap)),
        )
        out["miss_badness"] = str(bad_base.with_suffix(".png"))

    if save_maps and recall is not None:
        rbase = output_dir / f"{stem}_spatial_recall"
        _save_map_png_and_npy(
            rbase,
            recall,
            save_npy=save_npy,
            equalize_png=False,
            equalize_method="hist",
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
            colormap=recall_cmap,
        )
        out["recall"] = str(rbase.with_suffix(".png"))

    h_map, w_map = int(miss_weight.shape[0]), int(miss_weight.shape[1])
    dot_r = int(ms.get("sample_dot_radius", 1))
    dot_alpha = float(ms.get("sample_dot_alpha", 0.55))
    dot_hollow = bool(ms.get("sample_dot_hollow", True))
    dot_ring_w = int(ms.get("sample_dot_ring_width", 1))
    dot_color = tuple(int(x) for x in ms.get("sample_dot_color", (255, 230, 40))[:3])
    recall_with_samples_rgb: np.ndarray | None = None
    if recall is not None and sampled_flat_indices is not None:
        r_rgb_plain = _gray_u8_to_rgb_colormap(_to_uint8_gray01(recall), recall_cmap)
        r_rgb_plain = np.asarray(r_rgb_plain, dtype=np.uint8).copy()
        r_rgb_plain[~valid_mask] = 0
        recall_with_samples_rgb = draw_training_samples_on_rgb(
            r_rgb_plain,
            sampled_flat_indices,
            h_map,
            w_map,
            dot_radius=dot_r,
            dot_color=dot_color,
            ring_alpha=dot_alpha,
            ring_width=dot_ring_w,
            hollow=dot_hollow,
        )
        if save_maps and bool(ms.get("save_recall_sample_overlay", True)):
            overlay_path = output_dir / f"{stem}_spatial_recall_with_samples.png"
            Image.fromarray(recall_with_samples_rgb, mode="RGB").save(overlay_path)
            out["recall_with_samples"] = str(overlay_path)

    if save_maps and viz_original is not None and bool(ms.get("save_bootstrap_viz", True)):
        boot_path = output_dir / f"{stem}_viz_bootstrap.png"
        Image.fromarray(_to_uint8_rgb(viz_original), mode="RGB").save(boot_path)
        out["viz_bootstrap"] = str(boot_path)

    if save_maps and bool(ms.get("save_sampled_map", False)) and sampled_flat_indices is not None:
        h, w = miss_weight.shape[:2]
        sampled_map = build_sampled_miss_pixels_map(
            h,
            w,
            sampled_flat_indices,
            valid_mask,
            miss_weight,
            dot_radius=int(ms.get("sample_dot_radius", 2)),
        )
        sbase = output_dir / f"{stem}_miss_stratified_sampled_pixels"
        _save_map_png_and_npy(
            sbase,
            sampled_map,
            save_npy=save_npy,
            equalize_png=True,
            equalize_method="hist",
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
            colormap=str(ms.get("sample_colormap", miss_cmap)),
        )
        out["sampled"] = str(sbase.with_suffix(".png"))

    save_sampling_viz = bool(ms.get("save_sampling_viz", True))
    if (
        save_maps
        and save_sampling_viz
        and sampled_flat_indices is not None
    ):
        h, w = miss_weight.shape[:2]
        cat_map, cat_counts = build_sampling_category_map(
            h,
            w,
            sampled_flat_indices,
            valid_mask,
            bootstrap_sampled_flat_indices=bootstrap_sampled_flat_indices,
        )
        cat_path = output_dir / f"{stem}_sampling_categories.png"
        _save_sampling_category_png(cat_path, cat_map)
        out["sampling_categories"] = str(cat_path)
        if save_npy:
            np.save(output_dir / f"{stem}_sampling_categories.npy", cat_map)

        base_for_viz = viz_original if viz_original is not None else viz_low_quality_fit
        samp_rgb = render_sampling_viz_rgb(
            base_for_viz,
            sampled_flat_indices,
            h,
            w,
            valid_mask,
            bootstrap_sampled_flat_indices=bootstrap_sampled_flat_indices,
            dot_radius=int(ms.get("sample_dot_radius", 1)),
            ring_alpha=float(ms.get("sample_dot_alpha", 0.55)),
            ring_width=int(ms.get("sample_dot_ring_width", 1)),
            base_dim=float(ms.get("sampling_viz_base_dim", 0.45)),
        )
        viz_path = output_dir / f"{stem}_sampling_viz.png"
        Image.fromarray(samp_rgb, mode="RGB").save(viz_path)
        out["sampling_viz"] = str(viz_path)
        print(
            f"[train] sampling map: {viz_path} "
            f"(cyan=bootstrap n={cat_counts['bootstrap_only']}, "
            f"yellow=new miss n={cat_counts['miss_new_only']}, total={cat_counts['total']})"
        )

    if save_panel:
        if viz_original is None:
            raise ValueError(
                f"{stem}: recall panel needs bootstrap viz (miss_stratified.two_pass=true); "
                "quality_map_path-only runs cannot build the 3-column panel"
            )
        if recall is None:
            raise ValueError(f"{stem}: recall panel needs recall/quality map")

        orig_u8 = np.asarray(_to_uint8_rgb(viz_original), dtype=np.uint8)[..., :3]
        if recall_with_samples_rgb is not None:
            r_rgb = recall_with_samples_rgb
        else:
            r_rgb = _gray_u8_to_rgb_colormap(_to_uint8_gray01(recall), recall_cmap)
            r_rgb = np.asarray(r_rgb, dtype=np.uint8).copy()
            r_rgb[~valid_mask] = 0
        fit_u8 = np.asarray(_to_uint8_rgb(viz_low_quality_fit), dtype=np.uint8)[..., :3]
        panels = [orig_u8, r_rgb[..., :3], fit_u8]
        n_train = int(np.asarray(sampled_flat_indices).size) if sampled_flat_indices is not None else 0
        labels = (
            "1 bootstrap (superpixel)",
            f"2 recall + train pixels (n={n_train})",
            "3 refit (miss_stratified)",
        )

        sep = np.full((panels[0].shape[0], 2, 3), 255, dtype=np.uint8)
        combined = panels[0]
        for p in panels[1:]:
            combined = np.concatenate([combined, sep, p], axis=1)

        if bool(ms.get("panel_add_labels", True)):
            try:
                from PIL import ImageDraw, ImageFont

                bar_h = max(20, int(ms.get("panel_label_height", 22)))
                canvas = Image.new("RGB", (combined.shape[1], combined.shape[0] + bar_h), (32, 32, 32))
                canvas.paste(Image.fromarray(combined), (0, bar_h))
                draw = ImageDraw.Draw(canvas)
                font = ImageFont.load_default()
                col_w = combined.shape[1] // 3
                for i, lab in enumerate(labels):
                    draw.text((6 + i * col_w, 4), lab, fill=(230, 230, 230), font=font)
                combined = np.asarray(canvas, dtype=np.uint8)
            except Exception:
                pass

        panel_path = (output_dir / f"{stem}_recall_panel.png").resolve()
        Image.fromarray(combined, mode="RGB").save(panel_path)
        out["panel"] = str(panel_path)
        legacy = (output_dir / f"{stem}_viz_miss_stratified_panel.png").resolve()
        Image.fromarray(combined, mode="RGB").save(legacy)
        out["panel_legacy"] = str(legacy)
        print(f"[train] RECALL PANEL (3 columns) saved: {panel_path}")

    if "panel" not in out and save_panel:
        print(
            f"[train] Warning: recall panel not written for {stem} "
            f"(need bootstrap viz + recall map; check rank_config / two_pass)"
        )
    elif "recall" in out:
        print(f"[train] spatial recall map: {out['recall']}")
    return out


def save_spatial_edge_quality_maps(
    cfg: DictConfig,
    img: np.ndarray,
    viz_rgb: np.ndarray,
    output_dir: Path,
    stem: str,
    *,
    hd_edge_n: np.ndarray | None = None,
) -> dict[str, str]:
    """Patch-wise HD↔viz edge agreement map (same logic as rank_visualizations spatial_coverage)."""
    if not bool(OmegaConf.select(cfg, "spatial_quality.enabled", default=True)):
        return {}

    try:
        sq = compute_spatial_edge_quality_arrays(cfg, img, viz_rgb, hd_edge_n=hd_edge_n)
    except ValueError as e:
        print(f"[train] spatial_quality skipped: {e}")
        return {}

    from debug_edge_maps import _save_map_png_and_npy

    quality = sq["quality"]
    local_dice = sq["local_dice"]
    contrast_norm = sq["contrast_norm"]
    hd_edge_n = sq["hd_edge_n"]
    viz_edge_n = sq["viz_edge_n"]
    valid_mask = sq["valid_mask"]
    global_dice = sq["global_dice"]
    mean_q = sq["mean_q"]
    window = sq["window"]
    viz_u8 = sq["viz_u8"]

    cmap = str(OmegaConf.select(cfg, "spatial_quality.colormap", default="PuBu"))
    save_npy = bool(OmegaConf.select(cfg, "spatial_quality.save_npy", default=True))
    out: dict[str, str] = {}

    def _save(base: Path, arr: np.ndarray) -> None:
        _save_map_png_and_npy(
            base,
            arr,
            save_npy=save_npy,
            equalize_png=False,
            equalize_method="hist",
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
            colormap=cmap,
        )
        out[base.name] = str(base.with_suffix(".png"))

    output_dir.mkdir(parents=True, exist_ok=True)
    _save(output_dir / f"{stem}_spatial_quality", quality)
    _save(output_dir / f"{stem}_spatial_quality_local_dice", local_dice)
    _save(output_dir / f"{stem}_spatial_quality_local_contrast", contrast_norm)
    if float(OmegaConf.select(cfg, "spatial_quality.weights.lab_chroma_entropy", default=0.0)) > 1e-12:
        _save(output_dir / f"{stem}_spatial_quality_local_lab_entropy", sq["lab_entropy_norm"])
    _save(output_dir / f"{stem}_hd_edge_reference", hd_edge_n)
    _save(output_dir / f"{stem}_viz_edge", viz_edge_n)

    if bool(OmegaConf.select(cfg, "spatial_quality.save_panel", default=True)):
        from debug_edge_maps import _gray_u8_to_rgb_colormap, _to_uint8_gray01

        q_rgb = _gray_u8_to_rgb_colormap(_to_uint8_gray01(quality), cmap)
        q_rgb = np.asarray(q_rgb, dtype=np.uint8).copy()
        q_rgb[~valid_mask] = 0

        left = np.asarray(viz_u8, dtype=np.uint8)[..., :3]
        sep = np.full((left.shape[0], 2, 3), 255, dtype=np.uint8)
        combined = np.concatenate([left, sep, q_rgb[..., :3]], axis=1)

        title_tpl = OmegaConf.select(cfg, "spatial_quality.panel_title", default=None)
        if title_tpl is not None and str(title_tpl).strip().lower() not in ("", "~", "null", "none"):
            title = str(title_tpl).format(
                stem=stem,
                mean_q=mean_q,
                global_dice=global_dice,
                window=window,
            )
        else:
            title = (
                f"MiCS viz  |  spatial edge quality  "
                f"(mean={mean_q:.3f}, dice={global_dice:.3f}, w={window})"
            )

        if bool(OmegaConf.select(cfg, "spatial_quality.panel_add_title", default=True)):
            bar_h = max(24, int(OmegaConf.select(cfg, "spatial_quality.panel_title_height", default=32)))
            h, w = combined.shape[:2]
            canvas = Image.new("RGB", (w, h + bar_h), (24, 24, 24))
            canvas.paste(Image.fromarray(combined), (0, bar_h))
            try:
                from PIL import ImageDraw, ImageFont

                draw = ImageDraw.Draw(canvas)
                draw.text((6, 4), title, fill=(235, 235, 235), font=ImageFont.load_default())
            except Exception:
                pass
            combined = np.asarray(canvas, dtype=np.uint8)

        panel_path = output_dir / f"{stem}_viz_spatial_quality_panel.png"
        Image.fromarray(combined, mode="RGB").save(panel_path)
        out["panel"] = str(panel_path)

    print(
        f"[train] spatial edge quality (window={window}): "
        f"mean_map={mean_q:.4f} global_continuous_dice={global_dice:.4f} "
        f"→ {stem}_spatial_quality.png"
        + (
            f" panel={stem}_viz_spatial_quality_panel.png"
            if bool(OmegaConf.select(cfg, "spatial_quality.save_panel", default=True))
            else ""
        )
    )
    return out


def compute_hd_edge_target_simple(msi: np.ndarray) -> np.ndarray:
    """Ion-sum magnitude, 1–99% stretch, Sobel magnitude → float32 [0, 1] (proxy HD edge target)."""
    h = np.linalg.norm(msi.astype(np.float32), axis=-1)
    mask = h > 0
    if not np.any(mask):
        return np.zeros_like(h, dtype=np.float32)
    lo, hi = np.percentile(h[mask], [1.0, 99.0])
    z = np.clip((h - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
    z[~mask] = 0.0
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag[~mask] = 0.0
    mx = float(mag[mask].max())
    if mx > 1e-8:
        mag = (mag / mx).astype(np.float32)
    return mag


def load_and_preprocess_image(img_path, transpose=False):
    """Load and preprocess a single .npy image file."""
    img = np.load(img_path, mmap_mode=None)
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    # In-place sum normalization (no copy); works if img is float (safe here as we just cast to float32 above)
    img_sum = img.sum(axis=-1, keepdims=True)
    np.add(img_sum, 1e-6, out=img_sum)
    np.divide(img, img_sum, out=img)
    if transpose:
        img = img.transpose().transpose(1, 2, 0)[::-1, :, :].copy()
    return img


def compute_uncertainty_map(model, img):
    """Compute a normalized uncertainty (entropy) map from cluster classifiers."""
    if not getattr(model, "visualiation_to_cluster", None):
        raise ValueError("No classifiers available - enable clustering and load classifiers.")

    if isinstance(model, MSIParametricMiCSUnet):
        return model.uncertainty2(img)

    img_flat = img.reshape(img.shape[0] * img.shape[1], -1)
    mask = img.sum(axis=-1) > 0

    model.model.eval()
    for classifier in model.visualiation_to_cluster:
        classifier.eval()

    batch_size = 1024 * 32
    entropies = []

    with torch.no_grad():
        device = next(model.model.parameters()).device
        x_splits = torch.split(torch.from_numpy(img_flat).float(), batch_size)
        for x_batch_cpu in x_splits:
            x_batch = x_batch_cpu.to(device)

            batch_entropy = None
            for ei, classifier in enumerate(model.visualiation_to_cluster):
                clf = classifier.to(device)
                emb = model._embed_for_cluster_head(x_batch, ei)
                logits = clf(emb)
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
                batch_entropy = entropy if batch_entropy is None else batch_entropy + entropy

            batch_entropy = (batch_entropy / len(model.visualiation_to_cluster)).cpu().numpy()
            entropies.append(batch_entropy)

    entropy_map = np.concatenate(entropies, axis=0).reshape(img.shape[0], img.shape[1])
    entropy_map[~mask] = 0.0

    if mask.any():
        min_val = entropy_map[mask].min()
        max_val = entropy_map[mask].max()
        if max_val > min_val:
            entropy_map = (entropy_map - min_val) / (max_val - min_val)
        else:
            entropy_map = np.zeros_like(entropy_map)
    else:
        entropy_map = np.zeros_like(entropy_map)

    return entropy_map


def save_uncertainty_image(model, img, output_path, filename):
    """Generate and save an uncertainty image (entropy map)."""
    uncertainty = compute_uncertainty_map(model, img)
    uncertainty_uint8 = (uncertainty * 255).clip(0, 255).astype(np.uint8)
    equalized = cv2.equalizeHist(uncertainty_uint8)
    equalized[uncertainty_uint8 == 0] = 0
    img_pil = Image.fromarray(equalized, mode="L")
    img_pil.save(output_path / f"{filename}_uncertainty.png")
    return equalized


def load_trained_model(model_path, cfg):
    """Load a trained model from a saved checkpoint."""
    print(f"\n{'='*80}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*80}\n")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get config from checkpoint or use provided config
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        # Use saved config for model parameters, but allow override from cfg
        model_kwargs = {
            'number_of_points': saved_config.get('model', {}).get('number_of_points', cfg.model.number_of_points),
            'sampling': saved_config.get('model', {}).get('sampling', cfg.model.sampling),
            'num_epochs': saved_config.get('model', {}).get('num_epochs', cfg.model.num_epochs),
            'number_of_components': saved_config.get('model', {}).get('number_of_components', cfg.model.number_of_components),
            'num_layers': saved_config.get('model', {}).get('num_layers', getattr(cfg.model, 'num_layers', 2)),
            'lab_to_rgb': saved_config.get('model', {}).get('lab_to_rgb', cfg.model.lab_to_rgb),
            'cluster': saved_config.get('model', {}).get('cluster', cfg.model.cluster),
            'clusters': saved_config.get('model', {}).get(
                'clusters',
                _coerce_model_clusters_list(cfg.model),
            ),
            'num_samples': saved_config.get('model', {}).get('num_samples', cfg.model.num_samples),
            'beta': saved_config.get('model', {}).get('beta', cfg.model.beta),
            'k_epoch': saved_config.get('model', {}).get('k_epoch', cfg.model.k_epoch),
            'temperature': saved_config.get('model', {}).get('temperature', cfg.model.temperature),
            'lr': saved_config.get('model', {}).get('lr', cfg.model.lr),
            'batch_size': saved_config.get('model', {}).get('batch_size', cfg.model.batch_size),
            'warmup_epochs': saved_config.get('model', {}).get('warmup_epochs', cfg.model.warmup_epochs),
            'factor': saved_config.get('model', {}).get('factor', cfg.model.factor),
            'random_state': saved_config.get('model', {}).get('random_state', cfg.model.random_state),
            'verbose': cfg.model.verbose,
            'cluster_loss_weight': saved_config.get('model', {}).get('cluster_loss_weight', getattr(cfg.model, 'cluster_loss_weight', 1.0)),
            'category_loss_weight': saved_config.get('model', {}).get('category_loss_weight', getattr(cfg.model, 'category_loss_weight', 1.0)),
            'pixel_sampling': saved_config.get('model', {}).get('pixel_sampling', getattr(cfg.model, 'pixel_sampling', 'superpixel')),
            'pca_fit_step': saved_config.get('model', {}).get('pca_fit_step', getattr(cfg.model, 'pca_fit_step', 4)),
            'cluster_on_pca': saved_config.get('model', {}).get('cluster_on_pca', getattr(cfg.model, 'cluster_on_pca', False)),
            'cluster_pca_dims': saved_config.get('model', {}).get('cluster_pca_dims', getattr(cfg.model, 'cluster_pca_dims', 50)),
            'predict_percentile_low': float(
                saved_config.get('model', {}).get(
                    'predict_percentile_low', getattr(cfg.model, 'predict_percentile_low', 0.001),
                ),
            ),
            'predict_percentile_high': float(
                saved_config.get('model', {}).get(
                    'predict_percentile_high', getattr(cfg.model, 'predict_percentile_high', 99.999),
                ),
            ),
            'predict_spatial_smooth_sigma': float(
                saved_config.get('model', {}).get(
                    'predict_spatial_smooth_sigma', getattr(cfg.model, 'predict_spatial_smooth_sigma', 0.0),
                ),
            ),
            'clusters_auto_tune': saved_config.get('model', {}).get(
                'clusters_auto_tune',
                OmegaConf.select(cfg.model, 'clusters_auto_tune', default=None),
            ),
            'clusters_auto_tune_k_min': int(
                saved_config.get('model', {}).get(
                    'clusters_auto_tune_k_min',
                    OmegaConf.select(cfg.model, 'clusters_auto_tune_k_min', default=2),
                ),
            ),
            'clusters_auto_tune_k_max': int(
                saved_config.get('model', {}).get(
                    'clusters_auto_tune_k_max',
                    OmegaConf.select(cfg.model, 'clusters_auto_tune_k_max', default=256),
                ),
            ),
            'clusters_auto_tune_n_candidates': int(
                saved_config.get('model', {}).get(
                    'clusters_auto_tune_n_candidates',
                    OmegaConf.select(cfg.model, 'clusters_auto_tune_n_candidates', default=20),
                ),
            ),
        }
        _step_resume = saved_config.get('model', {}).get(
            'clusters_auto_tune_k_step',
            OmegaConf.select(cfg.model, 'clusters_auto_tune_k_step', default=None),
        )
        try:
            _si_rs = int(_step_resume) if _step_resume is not None else None
        except (TypeError, ValueError):
            _si_rs = None
        if _si_rs is not None and int(_si_rs) >= 1:
            model_kwargs['clusters_auto_tune_k_step'] = int(_si_rs)
    else:
        # Fallback to provided config
        model_kwargs = {
            'number_of_points': cfg.model.number_of_points,
            'sampling': cfg.model.sampling,
            'num_epochs': cfg.model.num_epochs,
            'number_of_components': cfg.model.number_of_components,
            'num_layers': getattr(cfg.model, 'num_layers', 2),
            'lab_to_rgb': cfg.model.lab_to_rgb,
            'cluster': cfg.model.cluster,
            'clusters': _coerce_model_clusters_list(cfg.model),
            'num_samples': cfg.model.num_samples,
            'beta': cfg.model.beta,
            'k_epoch': cfg.model.k_epoch,
            'temperature': cfg.model.temperature,
            'lr': cfg.model.lr,
            'batch_size': cfg.model.batch_size,
            'warmup_epochs': cfg.model.warmup_epochs,
            'factor': cfg.model.factor,
            'random_state': cfg.model.random_state,
            'verbose': cfg.model.verbose,
            'cluster_loss_weight': getattr(cfg.model, 'cluster_loss_weight', 1.0),
            'category_loss_weight': getattr(cfg.model, 'category_loss_weight', 1.0),
            'pixel_sampling': getattr(cfg.model, 'pixel_sampling', 'superpixel'),
            'pca_fit_step': getattr(cfg.model, 'pca_fit_step', 4),
            'cluster_on_pca': getattr(cfg.model, 'cluster_on_pca', False),
            'cluster_pca_dims': getattr(cfg.model, 'cluster_pca_dims', 50),
            'predict_percentile_low': float(getattr(cfg.model, 'predict_percentile_low', 0.001)),
            'predict_percentile_high': float(getattr(cfg.model, 'predict_percentile_high', 99.999)),
            'predict_spatial_smooth_sigma': float(getattr(cfg.model, 'predict_spatial_smooth_sigma', 0.0)),
            **_clusters_auto_tune_kw_from_cfg(cfg.model),
        }

    if model_kwargs.get("cluster") and isinstance(model_kwargs.get("clusters"), str):
        cs = str(model_kwargs["clusters"])
        model_kwargs["clusters"] = [int(x) for x in cs.split("-") if str(x).strip()]

    print(f"Model kwargs: {model_kwargs}")

    mc = str(checkpoint.get("model_class", "mics_lmc")).lower().strip()
    input_shape = checkpoint.get('input_shape')
    if input_shape is None:
        raise ValueError("Checkpoint missing input_shape; cannot rebuild model for inference.")

    ub = int(checkpoint.get("unet_base_channels") or 32)

    if mc == "mics_unet":
        model_kwargs["unet_base_channels"] = ub
        model_kwargs["dice_loss_weight"] = 0.0
        model_kwargs["contrast_loss_weight"] = 0.0
        model_kwargs["hd_edges"] = None
        model = MSIParametricMiCSUnet(**model_kwargs)
    elif mc == "mixture_of_experts":
        if "config" not in checkpoint:
            raise ValueError("mixture_of_experts checkpoints require checkpoint['config'] for expert layout")
        sm = dict(saved_config.get("model") or {})
        mo = _omega_sel_to_plain_dict(sm.get("mixture_of_experts") or {})
        experts_raw = mo.get("experts")
        if not experts_raw or not isinstance(experts_raw, (list, tuple)):
            raise ValueError("checkpoint model.mixture_of_experts.experts must be a non-empty list")
        base_nl = int(sm.get("num_layers", model_kwargs["num_layers"]))
        base_fac = float(sm.get("factor", model_kwargs["factor"]))
        expert_specs: list[dict[str, Any]] = []
        for row in experts_raw:
            r = _omega_sel_to_plain_dict(row) if OmegaConf.is_config(row) else dict(row)
            expert_specs.append(
                {
                    "num_layers": int(r.get("num_layers", base_nl)),
                    "factor": float(r.get("factor", base_fac)),
                    **({k: r[k] for k in ("clusters", "cluster_on_pca", "cluster_pca_dims") if k in r}),
                }
            )
        fusion = str(mo.get("fusion", "mean")).strip().lower().replace("-", "_")
        if fusion not in {"mean", "softmax_gate"}:
            raise ValueError(f"invalid mixture fusion {fusion!r}")
        gh = mo.get("gate_hidden", mo.get("gate_hidden_dim", 0))
        raw_sd = mo.get("shared_cluster_pca_dims")
        sd_out: Optional[int]
        if raw_sd is None:
            sd_out = None
        else:
            zs = str(raw_sd).strip().lower()
            if zs in ("~", "null", "none", ""):
                sd_out = None
            else:
                sd_out = int(raw_sd)
        model = MSIParametricMiCSMoE(
            moe_expert_specs=expert_specs,
            moe_fusion=fusion,
            moe_layer_norm_before_fuse=bool(mo.get("layer_norm_before_fuse", True)),
            moe_gate_temperature=float(mo.get("gate_temperature", 1.0)),
            moe_gate_hidden_dim=int(gh or 0),
            moe_share_cluster_pca=bool(mo.get("share_cluster_pca", False)),
            moe_shared_cluster_pca_dims=sd_out,
            **model_kwargs,
        )
    else:
        model = MSIParametricMiCSLMC(**model_kwargs)

    # Ensure clusters are a list of ints when enabled
    if model.cluster and isinstance(model.clusters, str):
        model.clusters = [int(cluster) for cluster in model.clusters.split('-') if cluster]

    input_dim = int(input_shape[-1])
    if mc == "mics_unet":
        _, _, c_spec = input_shape
        model.model = SmallUNet(int(c_spec), model.number_of_components, base=ub)
        if torch.cuda.is_available():
            model.model = model.model.cuda()
    elif mc == "mixture_of_experts":
        model.model = model._create_default_model(input_dim)
    else:
        model.model = create_pcc_model(
            input_dim,
            model.number_of_components,
            model.factor,
            model.num_layers,
        )

    # Load model weights
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.eval()

    # Load clustering classifiers if available
    cluster_state_dicts = checkpoint.get('cluster_state_dicts')
    if model.cluster and cluster_state_dicts:
        model.visualiation_to_cluster = []
        for state_dict in cluster_state_dicts:
            out_dim = next(iter(state_dict.values())).shape[0]
            layer = torch.nn.Sequential(torch.nn.Linear(model.number_of_components, out_dim))
            layer.load_state_dict(state_dict)
            if torch.cuda.is_available():
                layer = layer.cuda()
            model.visualiation_to_cluster.append(layer)
    else:
        model.visualiation_to_cluster = []
    
    print(f"Model loaded successfully!")
    if 'input_shape' in checkpoint:
        print(f"Original training input shape: {checkpoint['input_shape']}")
    
    return model


def find_files_with_glob(pattern):
    """Find files matching a glob pattern."""
    if '*' in pattern or '?' in pattern or '[' in pattern:
        # It's a glob pattern
        files = sorted(glob.glob(pattern))
        return [Path(f) for f in files]
    else:
        # It's a directory or file path
        path = Path(pattern)
        if path.is_file():
            return [path]
        elif path.is_dir():
            # Find all .npy files in directory
            files = sorted(glob.glob(str(path / "*.npy")))
            return [Path(f) for f in files]
        else:
            raise ValueError(f"Path does not exist: {pattern}")


def run_inference(cfg, model, input_files, output_dir):
    """Run inference on a list of files using a trained model."""
    print(f"\n{'='*80}")
    print(f"Running inference on {len(input_files)} file(s)")
    print(f"{'='*80}\n")
    
    results = []
    for idx, file_path in enumerate(input_files):
        try:
            print(f"\nProcessing file {idx + 1}/{len(input_files)}: {file_path}")
            
            # Load and preprocess image
            img = load_and_preprocess_image(file_path, transpose=cfg.data.transpose)
            
            # Generate and save visualization
            filename = Path(file_path).stem
            base_viz = model.predict(img)
            save_visualization(
                model,
                img,
                output_dir,
                filename,
                viz=base_viz,
                histogram_equalize=bool(OmegaConf.select(cfg, "viz_histogram_equalize", default=False)),
            )
            viz_path = output_dir / f"{filename}_viz.png"

            quality_path = None
            edge_quality_path = None
            structure_quality_path = None
            contrast_quality_path = None
            try:
                zero_top_n = int(getattr(cfg.model, "quality_zero_top_n", 10))
                summary_method = getattr(cfg.model, "quality_highd_summary", "pca1")
                window_size = int(getattr(cfg.model, "quality_window_size", 9))
                structure_window = int(getattr(cfg.model, "quality_structure_window", 5))
                summary = compute_highd_summary(img, method=summary_method, random_state=cfg.model.random_state)

                save_quality_score_image(
                    model,
                    img,
                    output_dir,
                    filename,
                    zero_top_n,
                    base_viz=base_viz,
                    show_progress=True,
                )
                quality_path = output_dir / f"{filename}_quality.png"

                save_edge_quality_image(
                    model,
                    img,
                    output_dir,
                    filename,
                    window_size=window_size,
                    highd_summary=summary_method,
                    base_viz=base_viz,
                    summary=summary,
                )
                edge_quality_path = output_dir / f"{filename}_edge_quality.png"

                save_structure_quality_image(
                    model,
                    img,
                    output_dir,
                    filename,
                    window_size=structure_window,
                    highd_summary=summary_method,
                    base_viz=base_viz,
                    summary=summary,
                )
                structure_quality_path = output_dir / f"{filename}_structure_quality.png"

                save_contrast_quality_image(
                    model,
                    img,
                    output_dir,
                    filename,
                    window_size=window_size,
                    highd_summary=summary_method,
                    base_viz=base_viz,
                    summary=summary,
                )
                contrast_quality_path = output_dir / f"{filename}_contrast_quality.png"
            except Exception as e:
                print(f"  ⚠ Quality metrics skipped for {file_path}: {e}")

            results.append({
                "input": str(file_path),
                "viz": str(viz_path),
                "quality": str(quality_path) if quality_path else None,
                "edge_quality": str(edge_quality_path) if edge_quality_path else None,
                "structure_quality": str(structure_quality_path) if structure_quality_path else None,
                "contrast_quality": str(contrast_quality_path) if contrast_quality_path else None,
            })
            
            print(f"  ✓ Visualization saved: {output_dir / f'{filename}_viz.png'}")
            if quality_path:
                print(f"  ✓ Quality saved: {quality_path}")
            if edge_quality_path:
                print(f"  ✓ Edge quality saved: {edge_quality_path}")
            if structure_quality_path:
                print(f"  ✓ Structure quality saved: {structure_quality_path}")
            if contrast_quality_path:
                print(f"  ✓ Contrast quality saved: {contrast_quality_path}")

            try:
                save_spatial_edge_quality_maps(
                    cfg,
                    img,
                    base_viz,
                    output_dir,
                    filename,
                )
            except Exception as sq_exc:
                print(f"  ⚠ Spatial edge quality map skipped for {file_path}: {sq_exc}")
            
            del img
            
            import gc
            import torch

            # Clean up CUDA memory (if applicable)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Run Python garbage collection
            gc.collect()

        except Exception as e:
            import traceback
            print(f"\n  ✗ Error processing {file_path}: {e}")
            traceback.print_exc()
            if not cfg.continue_on_error:
                raise
            continue
    
    return results

def save_visualization(
    model,
    img,
    output_path,
    filename,
    viz=None,
    *,
    histogram_equalize: bool = False,
):
    """Save ``*_viz.png`` from ``model.predict`` output (uint8 LAB→RGB if enabled).

    ``histogram_equalize`` applies OpenCV ``equalizeHist`` per channel — it boosts micro-contrast but
    often looks **grainy** on MSI embeddings; ``predict()`` already uses global percentile scaling.
    """
    if viz is None:
        viz = model.predict(img)

    if histogram_equalize:
        viz_out = np.zeros_like(viz)
        for i in range(viz.shape[2]):
            viz_out[:, :, i] = cv2.equalizeHist(viz[:, :, i])
    else:
        viz_out = np.ascontiguousarray(viz)

    img_pil = Image.fromarray(viz_out)
    img_pil.save(output_path / f"{filename}_viz.png")

    return viz_out


def train_on_file(cfg, input_path, output_dir, file_idx=0, total_files=1) -> tuple[Any, Path, dict[str, str]]:
    """Train model on a single file."""
    print(f"\n{'='*80}")
    print(f"Training on file {file_idx + 1}/{total_files}: {input_path}")
    print(f"{'='*80}\n")
    
    # Load and preprocess image
    img = load_and_preprocess_image(input_path, transpose=cfg.data.transpose)
    
    # Create model with config parameters
    model_kwargs = {
        'number_of_points': cfg.model.number_of_points,
        'sampling': cfg.model.sampling,
        'num_epochs': cfg.model.num_epochs,
        'number_of_components': cfg.model.number_of_components,
        'num_layers': getattr(cfg.model, 'num_layers', 2),
        'lab_to_rgb': cfg.model.lab_to_rgb,
        'cluster': cfg.model.cluster,
        'clusters': _coerce_model_clusters_list(cfg.model),
        'num_samples': cfg.model.num_samples,
        'beta': cfg.model.beta,
        'k_epoch': cfg.model.k_epoch,
        'temperature': cfg.model.temperature,
        'lr': cfg.model.lr,
        'batch_size': cfg.model.batch_size,
        'warmup_epochs': cfg.model.warmup_epochs,
        'factor': cfg.model.factor,
        'random_state': cfg.model.random_state,
        'verbose': cfg.model.verbose,
        'cluster_loss_weight': getattr(cfg.model, 'cluster_loss_weight', 1.0),
        'category_loss_weight': getattr(cfg.model, 'category_loss_weight', 1.0),
        'pixel_sampling': getattr(cfg.model, 'pixel_sampling', 'superpixel'),
        'pca_fit_step': getattr(cfg.model, 'pca_fit_step', 4),
        'cluster_on_pca': getattr(cfg.model, 'cluster_on_pca', False),
        'cluster_pca_dims': getattr(cfg.model, 'cluster_pca_dims', 50),
        'predict_percentile_low': float(getattr(cfg.model, 'predict_percentile_low', 0.001)),
        'predict_percentile_high': float(getattr(cfg.model, 'predict_percentile_high', 99.999)),
        'predict_spatial_smooth_sigma': float(getattr(cfg.model, 'predict_spatial_smooth_sigma', 0.0)),
        **_clusters_auto_tune_kw_from_cfg(cfg.model),
        **_miss_stratified_kw_from_cfg(cfg.model),
    }

    miss_meta: dict[str, Any] = {}
    mt = str(getattr(cfg.model, "model_type", "mics_lmc")).lower().strip()
    hd_edges = None
    dice_w = float(getattr(cfg.model, "dice_loss_weight", 0.0) or 0.0)
    if mt == "mics_unet":
        hp = OmegaConf.select(cfg, "model.hd_edges_path", default=None)
        if hp is not None and str(hp).strip() not in ("~", "null", "None", ""):
            pth = Path(str(hp))
            if not pth.is_file():
                try:
                    from hydra.utils import to_absolute_path

                    pth = Path(to_absolute_path(str(hp)))
                except Exception:
                    pass
            if not pth.is_file():
                raise FileNotFoundError(f"model.hd_edges_path not found: {hp}")
            hd_edges = np.load(pth)
            hd_edges = np.squeeze(hd_edges).astype(np.float32)
            if hd_edges.shape != img.shape[:2]:
                raise ValueError(f"hd_edges shape {hd_edges.shape} != MSI spatial {img.shape[:2]}")
        elif dice_w > 1e-12:
            hd_edges = compute_hd_edge_target_simple(img)
            print(
                "[train] mics_unet: built hd_edges from MSI ion-sum Sobel proxy "
                "(set model.hd_edges_path for fixed-reference edges)."
            )
        model = MSIParametricMiCSUnet(
            **model_kwargs,
            unet_base_channels=int(getattr(cfg.model, "unet_base_channels", 32)),
            dice_loss_weight=dice_w,
            contrast_loss_weight=float(getattr(cfg.model, "contrast_loss_weight", 0.0) or 0.0),
            hd_edges=hd_edges,
        )
    elif mt == "mixture_of_experts":
        moe_bundle = _parse_mixture_of_experts_cfg(cfg.model)
        if moe_bundle is None:
            raise ValueError(
                "model_type=mixture_of_experts requires model.mixture_of_experts with a non-empty experts list"
            )
        model = MSIParametricMiCSMoE(
            **model_kwargs,
            moe_expert_specs=moe_bundle["expert_specs"],
            moe_fusion=moe_bundle["moe_fusion"],
            moe_layer_norm_before_fuse=moe_bundle["moe_layer_norm_before_fuse"],
            moe_gate_temperature=moe_bundle["moe_gate_temperature"],
            moe_gate_hidden_dim=moe_bundle["moe_gate_hidden_dim"],
            moe_share_cluster_pca=moe_bundle["moe_share_cluster_pca"],
            moe_shared_cluster_pca_dims=moe_bundle["moe_shared_cluster_pca_dims"],
        )
    else:
        ps = str(model_kwargs.get("pixel_sampling", "superpixel")).lower()
        if ps in ("miss_stratified", "miss"):
            miss_meta = _prepare_miss_stratified_sampling(cfg, img, model_kwargs)
            refit_mode = str(miss_meta.get("refit_mode", "finetune")).lower()
            boot = miss_meta.get("boot_model")
            if refit_mode in ("finetune", "continue", "warm_start") and boot is not None:
                print(
                    f"[train] miss_stratified refit_mode=finetune: "
                    f"{int(miss_meta.get('finetune_epochs', 10))} epochs at "
                    f"lr×{float(miss_meta.get('finetune_lr_scale', 0.1))}"
                )
                model = boot
                model.finetune_miss_stratified(
                    img,
                    finetune_epochs=int(miss_meta.get("finetune_epochs", 10)),
                    finetune_lr_scale=float(miss_meta.get("finetune_lr_scale", 0.1)),
                    finetune_warmup_epochs=int(miss_meta.get("finetune_warmup_epochs", 0)),
                )
            else:
                if boot is not None:
                    boot.release_resources()
                    del boot
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                model = MSIParametricMiCSLMC(**model_kwargs)
                model.fit(img)
        else:
            _attach_hd_edge_map_for_sampling_if_needed(cfg, img, model_kwargs)
            model = MSIParametricMiCSLMC(**model_kwargs)
            model.fit(img)
    
    # Save model
    model_path = output_dir / f"model_{Path(input_path).stem}.pth"
    cluster_state_dicts = None
    if model.cluster and getattr(model, "visualiation_to_cluster", None):
        cluster_state_dicts = [layer.state_dict() for layer in model.visualiation_to_cluster]

    if mt == "mics_unet":
        model_class_saved = "mics_unet"
    elif mt == "mixture_of_experts":
        model_class_saved = "mixture_of_experts"
    else:
        model_class_saved = "mics_lmc"

    save_payload: dict = {
        'model_state_dict': model.model.state_dict(),
        'cluster_state_dicts': cluster_state_dicts,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'input_shape': img.shape,
        'model_class': model_class_saved,
    }
    if mt == 'mics_unet':
        save_payload['unet_base_channels'] = int(getattr(cfg.model, 'unet_base_channels', 32))
    torch.save(save_payload, model_path)
    print(f"\nModel saved to: {model_path}")

    stem = Path(input_path).stem
    artifact_paths: dict[str, str] = {}
    refit_viz: np.ndarray | None = None
    if miss_meta:
        refit_viz = model.predict(img)
        try:
            boot_viz = miss_meta.get("viz_bootstrap")
            if boot_viz is not None:
                m_boot = _edge_agreement_metrics(cfg, img, boot_viz)
                m_ref = _edge_agreement_metrics(cfg, img, refit_viz)
                print(
                    "[train] edge agreement  bootstrap → refit: "
                    f"global_dice {m_boot['global_dice']:.4f} → {m_ref['global_dice']:.4f} "
                    f"(Δ {m_ref['global_dice'] - m_boot['global_dice']:+.4f}); "
                    f"mean_recall {m_boot['mean_recall']:.4f} → {m_ref['mean_recall']:.4f} "
                    f"(Δ {m_ref['mean_recall'] - m_boot['mean_recall']:+.4f})"
                )
            flat_idx = getattr(model, "sampled_flat_indices", None)
            recall_map = miss_meta.get("recall")
            if recall_map is None:
                recall_map = miss_meta.get("quality")
            if flat_idx is not None and recall_map is not None:
                vm = miss_meta["valid_mask"]
                r_flat = np.asarray(recall_map, dtype=np.float32).reshape(-1)[
                    np.asarray(flat_idx, dtype=np.int64)
                ]
                r_tissue = float(np.mean(recall_map[vm])) if np.any(vm) else float("nan")
                print(
                    f"[train] sampled recall mean={float(np.mean(r_flat)):.4f} "
                    f"(tissue mean={r_tissue:.4f}; lower sampled mean => more low-recall focus)"
                )
            artifact_paths = save_miss_stratified_maps(
                cfg,
                output_dir,
                stem,
                miss_meta["miss_weight"],
                refit_viz,
                miss_meta["valid_mask"],
                recall=recall_map,
                viz_original=miss_meta.get("viz_bootstrap"),
                sampled_flat_indices=None if flat_idx is None else np.asarray(flat_idx),
                bootstrap_sampled_flat_indices=miss_meta.get(
                    "bootstrap_sampled_flat_indices"
                ),
            )
            if artifact_paths.get("panel"):
                print(f"[train] >>> 3-column recall panel: {artifact_paths['panel']}")
        except Exception as e:
            print(f"ERROR: Miss-stratified recall panel failed for {input_path}: {e}")
            traceback.print_exc()
    
    # Generate and save visualization
    if cfg.save_visualizations:
        base_viz = refit_viz if refit_viz is not None else model.predict(img)
        save_visualization(
            model,
            img,
            output_dir,
            Path(input_path).stem,
            viz=base_viz,
            histogram_equalize=bool(OmegaConf.select(cfg, "viz_histogram_equalize", default=False)),
        )
        print(f"Visualization saved to: {output_dir / f'{Path(input_path).stem}_viz.png'}")
        try:
            zero_top_n = int(getattr(cfg.model, "quality_zero_top_n", 10))
            summary_method = getattr(cfg.model, "quality_highd_summary", "pca1")
            window_size = int(getattr(cfg.model, "quality_window_size", 9))
            structure_window = int(getattr(cfg.model, "quality_structure_window", 5))
            summary = compute_highd_summary(img, method=summary_method, random_state=cfg.model.random_state)

            save_quality_score_image(
                model,
                img,
                output_dir,
                Path(input_path).stem,
                zero_top_n,
                base_viz=base_viz,
                show_progress=True,
            )
            print(f"Quality image saved to: {output_dir / f'{Path(input_path).stem}_quality.png'}")

            save_edge_quality_image(
                model,
                img,
                output_dir,
                Path(input_path).stem,
                window_size=window_size,
                highd_summary=summary_method,
                base_viz=base_viz,
                summary=summary,
            )
            print(f"Edge quality image saved to: {output_dir / f'{Path(input_path).stem}_edge_quality.png'}")

            save_structure_quality_image(
                model,
                img,
                output_dir,
                Path(input_path).stem,
                window_size=structure_window,
                highd_summary=summary_method,
                base_viz=base_viz,
                summary=summary,
            )
            print(f"Structure quality image saved to: {output_dir / f'{Path(input_path).stem}_structure_quality.png'}")

            save_contrast_quality_image(
                model,
                img,
                output_dir,
                Path(input_path).stem,
                window_size=window_size,
                highd_summary=summary_method,
                base_viz=base_viz,
                summary=summary,
            )
            print(f"Contrast quality image saved to: {output_dir / f'{Path(input_path).stem}_contrast_quality.png'}")
        except Exception as e:
            print(f"Warning: Quality metrics skipped for {input_path}: {e}")

        try:
            hd_cached = getattr(model, "hd_edge_map_for_sampling", None)
            save_spatial_edge_quality_maps(
                cfg,
                img,
                base_viz,
                output_dir,
                Path(input_path).stem,
                hd_edge_n=None if hd_cached is None else np.asarray(hd_cached, dtype=np.float64),
            )
        except Exception as e:
            print(f"Warning: Spatial edge quality map skipped for {input_path}: {e}")
    
    return model, model_path, artifact_paths


@hydra.main(version_base=None, config_path="configs", config_name="train_parametric_mics_lmc")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    
    # Hydra creates an output directory automatically
    # We'll use it but also create a custom subdirectory with date
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_output_dir = Path(HydraConfig.get().run.dir)
    except:
        # Fallback if HydraConfig is not available
        hydra_output_dir = Path.cwd() / "outputs"
    
    # Create custom output directory with date
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.output_dir:
        base_output_dir = Path(cfg.output_dir)
    else:
        # Use Hydra's output directory as base
        base_output_dir = hydra_output_dir.parent
    
    output_dir = base_output_dir / f"parametric_mics_lmc_{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Hydra output directory: {hydra_output_dir}")
    print(f"{'='*80}\n")
    
    # Copy Hydra config to our output directory
    hydra_config_path = hydra_output_dir / ".hydra" / "config.yaml"
    if hydra_config_path.exists():
        import shutil
        hydra_dir = output_dir / ".hydra"
        hydra_dir.mkdir(exist_ok=True)
        shutil.copytree(hydra_output_dir / ".hydra", hydra_dir, dirs_exist_ok=True)
        print(f"Hydra configuration copied to: {hydra_dir}")
    
    # Also save config to output directory
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
        print(f"Configuration saved to: {config_path}")
    
    # Check if we're in inference mode
    is_inference = (hasattr(cfg, 'inference') and 
                   cfg.inference.model_path and 
                   (cfg.inference.inference_folder or cfg.mode == "inference"))
    
    if is_inference:
        # Inference mode: load model and run inference
        if not hasattr(cfg, 'inference') or not cfg.inference.model_path:
            raise ValueError("Inference mode requires inference.model_path to be specified")
        
        if not cfg.inference.inference_folder:
            raise ValueError("Inference mode requires inference.inference_folder to be specified")
        
        model_path = Path(cfg.inference.model_path)
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load trained model
        model = load_trained_model(model_path, cfg)
        
        # Find files to process using glob pattern support
        inference_pattern = cfg.inference.inference_folder
        input_files = find_files_with_glob(inference_pattern)
        
        if not input_files:
            raise ValueError(f"No files found matching pattern: {inference_pattern}")
        
        print(f"Found {len(input_files)} file(s) to process for inference")
        
        # Run inference
        visualizations = run_inference(cfg, model, input_files, output_dir)
        
        print(f"\n{'='*80}")
        print(f"Inference completed! Visualizations saved to: {output_dir}")
        print(f"{'='*80}\n")
        
        # Save summary
        summary_path = output_dir / "inference_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Inference Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"Date: {date_str}\n")
            f.write(f"Model path: {cfg.inference.model_path}\n")
            f.write(f"Inference folder/pattern: {cfg.inference.inference_folder}\n")
            f.write(f"Number of files processed: {len(visualizations)}\n")
            f.write(f"\nConfiguration:\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write(f"\n\nVisualizations saved:\n")
            for item in visualizations:
                f.write(f"  {item['input']} -> {item['viz']}\n")
                if item.get('quality'):
                    f.write(f"    quality -> {item['quality']}\n")
                if item.get('edge_quality'):
                    f.write(f"    edge_quality -> {item['edge_quality']}\n")
                if item.get('structure_quality'):
                    f.write(f"    structure_quality -> {item['structure_quality']}\n")
                if item.get('contrast_quality'):
                    f.write(f"    contrast_quality -> {item['contrast_quality']}\n")
        
        print(f"Summary saved to: {summary_path}")
        
    else:
        # Training mode: determine input files
        if not cfg.data.input_path:
            raise ValueError("Training mode requires data.input_path to be specified")
        
        input_path = Path(cfg.data.input_path)
        
        if input_path.is_file() and input_path.suffix == '.npy':
            # Single file
            input_files = [input_path]
        elif input_path.is_dir():
            # Folder of .npy files
            input_files = sorted([Path(f) for f in glob.glob(str(input_path / "*.npy"))])
            if not input_files:
                raise ValueError(f"No .npy files found in directory: {input_path}")
        else:
            raise ValueError(f"Input path must be a .npy file or directory: {input_path}")
        
        print(f"Found {len(input_files)} file(s) to process for training")
        # Training mode: train models
        # Store only model paths, not the actual models to save memory
        model_paths = []
        recall_panels: list[str] = []
        for idx, file_path in enumerate(input_files):
            try:
                model, model_path, artifacts = train_on_file(
                    cfg, file_path, output_dir, idx, len(input_files)
                )
                model_paths.append(model_path)
                if artifacts.get("panel"):
                    recall_panels.append(artifacts["panel"])
                # Free the model from memory after saving
                del model
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"\nError processing {file_path}: {e}")
                if not cfg.continue_on_error:
                    raise
                continue
        
        print(f"\n{'='*80}")
        print(f"Training completed! Results saved to: {output_dir}")
        print(f"{'='*80}\n")
        
        # Save summary
        summary_path = output_dir / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"Date: {date_str}\n")
            f.write(f"Input path: {cfg.data.input_path}\n")
            f.write(f"Number of files processed: {len(model_paths)}\n")
            f.write(f"\nConfiguration:\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write(f"\n\nModels saved:\n")
            for model_path in model_paths:
                f.write(f"  - {model_path}\n")
            if recall_panels:
                f.write(f"\nRecall panels (3 columns: bootstrap | recall | low-recall refit):\n")
                for p in recall_panels:
                    f.write(f"  - {p}\n")
        
        print(f"Summary saved to: {summary_path}")
        if recall_panels:
            print(f"Recall panel(s): {recall_panels[0]}" + (f" (+{len(recall_panels)-1} more)" if len(recall_panels) > 1 else ""))
        
        # Run inference after training if inference_folder is specified
        if hasattr(cfg, 'inference') and cfg.inference.inference_folder and model_paths:
            print(f"\n{'='*80}")
            print(f"Running inference after training...")
            print(f"{'='*80}\n")
            
            # Use the last trained model - load it from disk
            trained_model_path = model_paths[-1]
            print(f"Loading model from: {trained_model_path}")
            trained_model = load_trained_model(trained_model_path, cfg)
            
            # Find files to process using glob pattern support
            inference_pattern = cfg.inference.inference_folder
            inference_files = find_files_with_glob(inference_pattern)
            
            if not inference_files:
                print(f"Warning: No files found matching pattern: {inference_pattern}")
            else:
                print(f"Found {len(inference_files)} file(s) to process for inference")
                
                # Run inference using the trained model
                visualizations = run_inference(cfg, trained_model, inference_files, output_dir)
                
                # Free the model after inference
                del trained_model
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"\n{'='*80}")
                print(f"Inference after training completed! Visualizations saved to: {output_dir}")
                print(f"{'='*80}\n")
                
                # Update summary with inference results
                inference_summary_path = output_dir / "inference_after_training_summary.txt"
                with open(inference_summary_path, 'w') as f:
                    f.write(f"Inference After Training Summary\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Date: {date_str}\n")
                    f.write(f"Model used: {trained_model_path}\n")
                    f.write(f"Inference folder/pattern: {cfg.inference.inference_folder}\n")
                    f.write(f"Number of files processed: {len(visualizations)}\n")
                    f.write(f"\nConfiguration:\n")
                    f.write(OmegaConf.to_yaml(cfg))
                    f.write(f"\n\nVisualizations saved:\n")
                    for item in visualizations:
                        f.write(f"  {item['input']} -> {item['viz']}\n")
                        if item.get('quality'):
                            f.write(f"    quality -> {item['quality']}\n")
                        if item.get('edge_quality'):
                            f.write(f"    edge_quality -> {item['edge_quality']}\n")
                        if item.get('structure_quality'):
                            f.write(f"    structure_quality -> {item['structure_quality']}\n")
                        if item.get('contrast_quality'):
                            f.write(f"    contrast_quality -> {item['contrast_quality']}\n")
                
                print(f"Inference summary saved to: {inference_summary_path}")


if __name__ == "__main__":
    main()

