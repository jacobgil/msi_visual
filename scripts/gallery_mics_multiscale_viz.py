#!/usr/bin/env python3
"""
Continuous MiCS multi-scale display gallery (not palette segmentation).

Stage 1 modes (``stage1.mode``):
  - ``per_k`` (default): separate MiCS model per k; each ``predict()`` is its own continuous view
  - ``shared``: one multi-cluster MiCS + stage-2 display refinement (legacy)

Gallery (hub): MiCS k=8 | k=32 | k=128 on top → coarsest-k MiCS bottom center.

Run:
    python scripts/gallery_mics_multiscale_viz.py
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gallery_mics_cluster_soft import _save_hub_mosaic, _save_mosaic  # noqa: E402
from gallery_mics_edge import _resolve_edge_target  # noqa: E402
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from msi_visual.parametric_mics_multiscale_display import (  # noqa: E402
    MSIMiCSMultiDisplayRefine,
)
from msi_visual.parametric_mics_parallel_display import MSIMiCSParallelDisplay  # noqa: E402
from msi_visual.parametric_mics_perk_blend import MSIMiCSPerKBlend  # noqa: E402

logger = logging.getLogger(__name__)

_MICS_KEYS: tuple[str, ...] = (
    "number_of_points",
    "sampling",
    "number_of_components",
    "num_layers",
    "lab_to_rgb",
    "cluster",
    "clusters",
    "num_samples",
    "beta",
    "k_epoch",
    "temperature",
    "lr",
    "batch_size",
    "warmup_epochs",
    "factor",
    "verbose",
    "cluster_loss_weight",
    "category_loss_weight",
    "pixel_sampling",
    "pca_fit_step",
    "cluster_on_pca",
    "cluster_pca_dims",
    "predict_percentile_low",
    "predict_percentile_high",
    "predict_spatial_smooth_sigma",
)


def _abs_path_str(path: Path) -> str:
    return str(path.expanduser().resolve())


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    s = msi.sum(axis=-1, keepdims=True)
    return msi / (s + 1e-8)


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(str(path), mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI HxWxC, got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI must be nonnegative.")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def _parse_clusters(val: Any) -> list[int]:
    if val is None:
        return [8, 32, 128]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip()]
    return [int(s)]


def _resolve_num_layers(cfg: DictConfig, ga: Any) -> int:
    raw = OmegaConf.select(ga, "num_layers", default=None)
    if raw is None or str(raw).strip().lower() in ("", "~", "null", "none"):
        return max(1, int(cfg.model.num_layers))
    return max(1, int(raw))


def _mics_kwargs_from_cfg(cfg: DictConfig, num_layers: int, seed: int, num_epochs: int) -> dict[str, Any]:
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    kw: dict[str, Any] = {"random_state": int(seed)}
    for name in _MICS_KEYS:
        if name in merged:
            kw[name] = merged[name]
    kw["num_epochs"] = int(num_epochs)
    kw["num_layers"] = max(1, int(num_layers))
    kw["clusters"] = _parse_clusters(merged.get("clusters", [8, 32, 128]))
    kw.setdefault("number_of_points", int(merged.get("number_of_points", 1000)))
    kw.setdefault("sampling", str(merged.get("sampling", "random")))
    kw.setdefault("number_of_components", int(merged.get("number_of_components", 3)))
    kw.setdefault("lab_to_rgb", bool(merged.get("lab_to_rgb", True)))
    kw.setdefault("cluster", bool(merged.get("cluster", True)))
    kw.setdefault("num_samples", int(merged.get("num_samples", 20000)))
    kw.setdefault("cluster_loss_weight", float(merged.get("cluster_loss_weight", 1.0)))
    kw.setdefault("pixel_sampling", str(merged.get("pixel_sampling", "superpixel")))
    kw.setdefault("predict_spatial_smooth_sigma", 0.0)
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw["learn_input_mask"] = False
    kw["input_mask_l1_weight"] = 0.0
    if not kw.get("cluster"):
        raise ValueError("gallery_mics_multiscale_viz requires model.cluster=true")
    return kw


def _parse_float_list(val: Any) -> list[float] | None:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    s = str(val).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None
    return [float(x) for x in s.replace(",", "-").split("-") if x.strip()]


def _stage2_config(cfg: DictConfig, mics_kw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    s2 = OmegaConf.to_container(getattr(cfg, "stage2", {}), resolve=True)
    if not isinstance(s2, dict):
        s2 = {}
    mode = str(s2.get("mode", "parallel_display")).lower().strip()
    enabled = bool(s2.get("enabled", True))
    num_epochs = int(s2.get("num_epochs", 80 if mode == "parallel_display" else 0)) if enabled else 0
    common = {
        "num_epochs": num_epochs,
        "verbose": bool(mics_kw.get("verbose", False)),
        "predict_percentile_low": float(mics_kw.get("predict_percentile_low", 1.0)),
        "predict_percentile_high": float(mics_kw.get("predict_percentile_high", 99.0)),
        "predict_spatial_smooth_sigma": float(s2.get("predict_spatial_smooth_sigma", 0.0)),
        "lab_to_rgb": bool(mics_kw.get("lab_to_rgb", True)),
    }
    if mode in ("parallel", "parallel_display", "parallel_heads"):
        return mode, {
            **common,
            "lr": float(s2.get("lr", 0.02)),
            "batch_size": int(s2.get("batch_size", mics_kw.get("batch_size", 1024))),
            "anchor_level_idx": int(s2.get("anchor_level_idx", 0)),
            "anchor_weight": float(s2.get("anchor_weight", 0.55)),
            "level_cluster_weights": _parse_float_list(s2.get("level_cluster_weights")),
            "level_residual_scales": _parse_float_list(s2.get("level_residual_scales")),
            "level_blend_alphas": _parse_float_list(s2.get("level_blend_alphas")),
            "field_coarse_scales": _parse_float_list(s2.get("field_coarse_scales")),
            "display_composition": str(s2.get("display_composition", "field_residual")),
            "embed_clamp": float(s2.get("embed_clamp", 80.0)),
            "diversity_weight": float(s2.get("diversity_weight", 0.06)),
            "diversity_margin": float(s2.get("diversity_margin", 0.08)),
            "separation_weight": float(s2.get("separation_weight", 0.25)),
            "temperature": float(s2.get("temperature", mics_kw.get("temperature", 100.0))),
        }
    return "centroid_blend", {
        **common,
        "lr": float(s2.get("lr", 0.015)),
        "cluster_scale": float(s2.get("cluster_scale", 0.35)),
        "display_mode": str(s2.get("display_mode", "centroid_blend")),
        "level_blend": _parse_float_list(s2.get("level_blend")),
        "init_display": str(s2.get("init_display", "pca")),
        "mix_temperature": float(s2.get("mix_temperature", 1.0)),
        "mix_coarse_bias": float(s2.get("mix_coarse_bias", 0.0)),
        "dice_loss_weight": float(s2.get("dice_loss_weight", 0.0)),
        "edge_corr_loss_weight": float(s2.get("edge_corr_loss_weight", 0.0)),
        "edge_mag_loss_weight": float(s2.get("edge_mag_loss_weight", 0.0)),
        "edge_mag_hd_power": float(s2.get("edge_mag_hd_power", 2.0)),
        "mics_preserve_weight": float(s2.get("mics_preserve_weight", 0.85)),
        "level_delta_l2_weight": float(s2.get("level_delta_l2_weight", 0.02)),
        "non_edge_l1_weight": float(s2.get("non_edge_l1_weight", 0.0)),
        "edge_loss_ramp_epochs": int(s2.get("edge_loss_ramp_epochs", 8)),
        "dice_edge_color_space": str(s2.get("dice_edge_color_space", "rgb_max")),
        "dice_edge_percentile_norm": bool(s2.get("dice_edge_percentile_norm", True)),
        "edge_delta_gate_power": float(s2.get("edge_delta_gate_power", 2.0)),
    }


def _mics_kwargs_for_clusters(
    cfg: DictConfig,
    clusters: list[int],
    num_layers: int,
    seed: int,
    num_epochs: int,
) -> dict[str, Any]:
    kw = _mics_kwargs_from_cfg(cfg, num_layers, seed, num_epochs)
    kw["clusters"] = [int(k) for k in clusters]
    return kw


def _parse_k_ladder(cfg: DictConfig, ga: Any, default: list[int]) -> list[int]:
    raw = OmegaConf.select(ga, "k_ladder", default=None)
    if raw is None:
        raw = OmegaConf.select(cfg, "stage1.k_ladder", default=None)
    if raw is None:
        return list(default)
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if isinstance(raw, str):
        return _parse_clusters(raw)
    if isinstance(raw, (list, tuple)):
        out = [int(x) for x in raw]
        if not out:
            raise ValueError("k_ladder must be non-empty")
        return out
    raise ValueError(f"Invalid k_ladder: {raw!r}")


def _mics_rgb(model: MSIParametricMiCSLMC, msi: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgb = np.asarray(model.predict(msi), dtype=np.uint8)[..., :3].copy()
    rgb[~valid_mask] = 0
    return rgb


def _blend_config(cfg: DictConfig, mics_kw: dict[str, Any]) -> dict[str, Any]:
    bl = OmegaConf.to_container(getattr(cfg, "blend", {}), resolve=True)
    if not isinstance(bl, dict):
        bl = {}
    enabled = bool(bl.get("enabled", True))
    return {
        "enabled": enabled,
        "num_epochs": int(bl.get("num_epochs", 40)) if enabled else 0,
        "lr": float(bl.get("lr", 0.02)),
        "mix_temperature": float(bl.get("mix_temperature", 1.0)),
        "mix_spatial_kernel": int(bl.get("mix_spatial_kernel", 1)),
        "blend_mode": str(bl.get("mode", bl.get("blend_mode", "conv"))),
        "local_edge_window": int(bl.get("local_edge_window", 15)),
        "local_edge_temperature": float(bl.get("local_edge_temperature", 0.08)),
        "hd_edge_power": float(bl.get("hd_edge_power", 1.0)),
        "dice_loss_weight": float(bl.get("dice_loss_weight", 1.0)),
        "dice_global_weight": float(bl.get("dice_global_weight", 1.0)),
        "dice_patch_weight": float(bl.get("dice_patch_weight", 0.5)),
        "dice_patches_per_step": int(bl.get("dice_patches_per_step", 8)),
        "dice_patch_size": int(bl.get("dice_patch_size", 48)),
        "mix_input": str(bl.get("mix_input", "emb_edge")),
        "mix_init_level_idx": int(bl.get("mix_init_level_idx", -1)),
        "preserve_weight": float(bl.get("preserve_weight", 0.0)),
        "preserve_level_idx": int(bl.get("preserve_level_idx", 0)),
        "edge_loss_ramp_epochs": int(bl.get("edge_loss_ramp_epochs", 8)),
        "center_mode": str(bl.get("center_mode", "mixed")).strip().lower(),
        "verbose": bool(mics_kw.get("verbose", False)),
        "predict_percentile_low": float(mics_kw.get("predict_percentile_low", 1.0)),
        "predict_percentile_high": float(mics_kw.get("predict_percentile_high", 99.0)),
        "predict_spatial_smooth_sigma": float(bl.get("predict_spatial_smooth_sigma", 0.0)),
        "lab_to_rgb": bool(mics_kw.get("lab_to_rgb", True)),
        "dice_edge_color_space": str(bl.get("dice_edge_color_space", "rgb_max")),
        "dice_edge_percentile_norm": bool(bl.get("dice_edge_percentile_norm", True)),
        "eval_refine": bool(bl.get("eval_refine", True)),
        "eval_grid_steps": int(bl.get("eval_grid_steps", 41)),
        "eval_logit_steps": int(bl.get("eval_logit_steps", 120)),
    }


def _run_per_k_gallery(
    *,
    cfg: DictConfig,
    ga: Any,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    num_layers: int,
    seed: int,
    stage1_epochs: int,
    k_ladder: list[int],
    layout: str,
    arrow_color: str,
    include_center: bool,
    center_k: int | None,
    ncols: int,
    fig_w: float,
    dpi: int,
    title_fs: int,
    out_dir: Path,
    pan_dir: Path,
    pixel_cache: dict[str, Any] | None,
    edge_target: np.ndarray,
    blend_kw: dict[str, Any],
    display_ref: MSIParametricMiCSLMC | None = None,
) -> None:
    level_rgbs: list[np.ndarray] = []
    level_titles: list[str] = []
    level_embs: list[np.ndarray] = []
    dice_by_k: dict[int, float] = {}
    center_rgb: np.ndarray | None = None
    center_title = "MiCS"
    display_ref_model = display_ref

    for idx, k in enumerate(k_ladder):
        kw = _mics_kwargs_for_clusters(cfg, [k], num_layers, seed + idx, stage1_epochs)
        logger.info(
            "Per-k MiCS %d/%d | k=%d | epochs=%d",
            idx + 1,
            len(k_ladder),
            k,
            stage1_epochs,
        )
        model: MSIParametricMiCSLMC | None = None
        try:
            model = MSIParametricMiCSLMC(**kw)
            model.fit(msi, pixel_sampling_cache=pixel_cache)
            emb = model.predict_embedding(msi).astype(np.float32, copy=False)
            level_embs.append(emb)
            if display_ref_model is None:
                display_ref_model = model
            rgb = _mics_rgb(model, msi, valid_mask)
            dice_by_k[k] = _global_continuous_dice(cfg, rgb, edge_target, valid_mask)
            logger.info("MiCS k=%d continuous Dice vs HD: %.4f", k, dice_by_k[k])
            path = pan_dir / f"panel__mics_k{k}.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            logger.info("MiCS panel k=%d: %s", k, _abs_path_str(path))
            np.save(out_dir / f"mics_k{k}_embedding.npy", emb)
            level_rgbs.append(rgb)
            level_titles.append(f"MiCS k={k}")
            if not blend_kw.get("enabled") and center_k is not None and k == center_k:
                center_rgb = rgb
            elif not blend_kw.get("enabled") and center_k is None and idx == 0:
                center_rgb = rgb
        finally:
            if model is not None and model is not display_ref_model:
                try:
                    model.release_resources()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    dice_mixed = float("nan")
    mix_weights: np.ndarray | None = None
    if blend_kw.get("enabled") and len(level_embs) >= 2:
        assert display_ref_model is not None
        blend = MSIMiCSPerKBlend(
            display_ref_model,
            k_ladder,
            num_epochs=int(blend_kw.get("num_epochs", 0)),
            lr=float(blend_kw.get("lr", 0.02)),
            mix_temperature=float(blend_kw.get("mix_temperature", 1.0)),
            mix_spatial_kernel=int(blend_kw.get("mix_spatial_kernel", 1)),
            blend_mode=str(blend_kw.get("blend_mode", "conv")),
            local_edge_window=int(blend_kw.get("local_edge_window", 15)),
            local_edge_temperature=float(blend_kw.get("local_edge_temperature", 0.08)),
            hd_edge_power=float(blend_kw.get("hd_edge_power", 1.0)),
            dice_loss_weight=float(blend_kw.get("dice_loss_weight", 1.0)),
            dice_global_weight=float(blend_kw.get("dice_global_weight", 1.0)),
            dice_patch_weight=float(blend_kw.get("dice_patch_weight", 0.5)),
            dice_patches_per_step=int(blend_kw.get("dice_patches_per_step", 8)),
            dice_patch_size=int(blend_kw.get("dice_patch_size", 48)),
            mix_input=str(blend_kw.get("mix_input", "emb_edge")),
            mix_init_level_idx=int(blend_kw.get("mix_init_level_idx", -1)),
            preserve_weight=float(blend_kw.get("preserve_weight", 0.0)),
            preserve_level_idx=int(blend_kw.get("preserve_level_idx", 0)),
            edge_loss_ramp_epochs=int(blend_kw.get("edge_loss_ramp_epochs", 8)),
            verbose=bool(blend_kw.get("verbose", False)),
            predict_percentile_low=float(blend_kw.get("predict_percentile_low", 1.0)),
            predict_percentile_high=float(blend_kw.get("predict_percentile_high", 99.0)),
            predict_spatial_smooth_sigma=float(blend_kw.get("predict_spatial_smooth_sigma", 0.0)),
            lab_to_rgb=bool(blend_kw.get("lab_to_rgb", True)),
            dice_edge_color_space=str(blend_kw.get("dice_edge_color_space", "rgb_max")),
            dice_edge_percentile_norm=bool(blend_kw.get("dice_edge_percentile_norm", True)),
        )
        try:
            blend.fit(
                msi,
                level_embs,
                hd_edges=edge_target,
                eval_dice_fn=lambda rgb: _global_continuous_dice(cfg, rgb, edge_target, valid_mask),
                eval_refine=bool(blend_kw.get("eval_refine", True)),
                eval_grid_steps=int(blend_kw.get("eval_grid_steps", 41)),
                eval_logit_steps=int(blend_kw.get("eval_logit_steps", 120)),
            )
            mixed_rgb = blend.predict_mixed_rgb()
            mix_weights = blend.predict_mix_weights()
            dice_mixed = _global_continuous_dice(cfg, mixed_rgb, edge_target, valid_mask)
            logger.info("Blended MiCS continuous Dice vs HD: %.4f", dice_mixed)
            if dice_by_k:
                best_k = max(dice_by_k, key=dice_by_k.get)
                delta = dice_mixed - dice_by_k[best_k]
                logger.info(
                    "Blend vs best single (k=%d, dice=%.4f): %+.4f",
                    best_k,
                    dice_by_k[best_k],
                    delta,
                )
            mixed_path = pan_dir / "panel__mixed_mics.png"
            Image.fromarray(mixed_rgb, mode="RGB").save(mixed_path)
            logger.info("Blended MiCS panel: %s", _abs_path_str(mixed_path))
            np.save(out_dir / "mix_weights.npy", mix_weights.astype(np.float32))
            center_mode = str(blend_kw.get("center_mode", "mixed"))
            if center_mode in ("mixed", "blend", "dice"):
                center_rgb = mixed_rgb
                center_title = "MiCS (blend)"
            elif center_mode.startswith("k"):
                ck = int(center_mode.replace("k", ""))
                if ck in dice_by_k:
                    center_rgb = level_rgbs[k_ladder.index(ck)]
                    center_title = f"MiCS k={ck}"
            elif center_mode.isdigit():
                ck = int(center_mode)
                if ck in k_ladder:
                    center_rgb = level_rgbs[k_ladder.index(ck)]
                    center_title = f"MiCS k={ck}"
        finally:
            try:
                blend.release_resources()
            except Exception:
                pass
            if display_ref_model is not None:
                try:
                    display_ref_model.release_resources()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif display_ref_model is not None:
        try:
            display_ref_model.release_resources()
        except Exception:
            pass

    if center_rgb is None and level_rgbs:
        center_rgb = level_rgbs[0]

    mosaic = out_dir / "gallery_mics_multiscale_viz.png"
    if layout in ("hub", "mics_center", "center") and include_center and center_rgb is not None:
        _save_hub_mosaic(
            level_rgbs,
            level_titles,
            center_rgb,
            center_title,
            out_path=mosaic,
            ncols=ncols,
            fig_w=fig_w,
            dpi=dpi,
            title_fs=title_fs,
            arrow_color=arrow_color,
        )
    else:
        grid_rgbs = list(level_rgbs)
        grid_titles = list(level_titles)
        if include_center and center_rgb is not None:
            grid_rgbs.append(center_rgb)
            grid_titles.append(center_title)
        _save_mosaic(
            grid_rgbs,
            grid_titles,
            out_path=mosaic,
            ncols=ncols,
            fig_w=fig_w,
            dpi=dpi,
            title_fs=title_fs,
        )
    logger.info("Gallery mosaic: %s", _abs_path_str(mosaic))

    summary = out_dir / "multiscale_viz_summary.txt"
    with summary.open("w", encoding="utf-8") as fh:
        fh.write(f"npy_path: {OmegaConf.select(ga, 'npy_path')}\n")
        fh.write(f"stage1_mode: per_k\n")
        fh.write(f"stage1_epochs: {stage1_epochs}\n")
        fh.write(f"num_layers: {num_layers}\n")
        fh.write(f"k_ladder: {k_ladder}\n")
        fh.write(f"layout: {layout}\n")
        fh.write(f"display_type: continuous_mics (not segmentation palette)\n")
        for k in k_ladder:
            if k in dice_by_k:
                fh.write(f"dice_k{k}: {dice_by_k[k]:.4f}\n")
        if np.isfinite(dice_mixed):
            fh.write(f"dice_mixed: {dice_mixed:.4f}\n")
            best_k = max(dice_by_k, key=dice_by_k.get)
            fh.write(f"dice_mixed_delta_vs_best_k: {dice_mixed - dice_by_k[best_k]:+.4f}\n")
        if blend_kw.get("enabled"):
            fh.write(f"blend_epochs: {blend_kw.get('num_epochs')}\n")
            fh.write(f"blend_mode: {blend_kw.get('blend_mode')}\n")
            fh.write(f"blend_center_mode: {blend_kw.get('center_mode')}\n")
        fh.write(f"\nmosaic: {_abs_path_str(mosaic)}\n")
    logger.info("Summary: %s", _abs_path_str(summary))


def _global_continuous_dice(
    cfg: DictConfig,
    viz_rgb: np.ndarray,
    hd_edges: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    from experiment_mics_gnn import _global_continuous_dice as _dice

    return _dice(cfg, viz_rgb, hd_edges, valid_mask)


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_multiscale_viz")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ga = getattr(cfg, "gallery", None)
    st1 = getattr(cfg, "stage1", None)
    if ga is None:
        raise ValueError("Config must define gallery:")

    npy_path = Path(to_absolute_path(str(ga.npy_path))).expanduser().resolve()
    if not npy_path.is_file():
        raise FileNotFoundError(f"gallery.npy_path not found: {npy_path}")

    msi = _load_msi(npy_path, bool(ga.transpose_msi), bool(ga.tic_normalize))
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    num_layers = _resolve_num_layers(cfg, ga)
    seed = int(OmegaConf.select(ga, "seed", default=42))
    layout = str(OmegaConf.select(ga, "layout", default="hub")).strip().lower()
    arrow_color = str(OmegaConf.select(ga, "arrow_color", default="white"))
    include_stage1 = bool(OmegaConf.select(ga, "include_stage1_mics_panel", default=True))
    include_mixed = bool(OmegaConf.select(ga, "include_mixed_panel", default=False))
    save_mix_weights = bool(OmegaConf.select(ga, "save_mix_weights", default=True))
    ncols = max(1, int(OmegaConf.select(ga, "ncols", default=3)))
    fig_w = float(OmegaConf.select(ga, "figure_size_inches", default=14))
    dpi = int(OmegaConf.select(ga, "dpi", default=300))
    title_fs = int(OmegaConf.select(ga, "panel_title_fontsize", default=10))

    stage1_epochs = int(OmegaConf.select(st1, "num_epochs", default=100))
    stage1_mode = str(OmegaConf.select(st1, "mode", default="per_k")).strip().lower()
    mics_kw = _mics_kwargs_from_cfg(cfg, num_layers, seed, stage1_epochs)
    k_ladder = _parse_k_ladder(cfg, ga, mics_kw["clusters"])
    stage2_mode, stage2_kw = _stage2_config(cfg, mics_kw)
    blend_kw = _blend_config(cfg, mics_kw)

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    pan_dir = out_dir / "panels"
    pan_dir.mkdir(parents=True, exist_ok=True)

    edge_target, edge_label = _resolve_edge_target(msi, cfg, ga)
    np.save(out_dir / "edge_target.npy", edge_target.astype(np.float32))

    share_sampling = bool(OmegaConf.select(ga, "share_pixel_sampling", default=True))
    cache_path_raw = OmegaConf.select(ga, "pixel_sampling_cache_path", default=None)
    cache_path: Path | None = None
    if cache_path_raw is not None and str(cache_path_raw).strip().lower() not in (
        "",
        "~",
        "null",
        "none",
    ):
        cache_path = Path(to_absolute_path(str(cache_path_raw))).expanduser().resolve()

    cache_kw = _mics_kwargs_for_clusters(cfg, [k_ladder[0]], num_layers, seed, stage1_epochs)
    per_k_mode = stage1_mode in ("per_k", "per-k", "separate", "independent")
    pixel_cache = resolve_pixel_sampling_cache(
        msi,
        cache_kw,
        out_dir=out_dir,
        share=share_sampling,
        cache_path=cache_path,
        strip_cluster_labels=per_k_mode,
    )

    if per_k_mode:
        center_k_raw = OmegaConf.select(ga, "center_k", default=None)
        center_k = int(center_k_raw) if center_k_raw is not None else None
        _run_per_k_gallery(
            cfg=cfg,
            ga=ga,
            msi=msi,
            valid_mask=valid_mask,
            num_layers=num_layers,
            seed=seed,
            stage1_epochs=stage1_epochs,
            k_ladder=k_ladder,
            layout=layout,
            arrow_color=arrow_color,
            include_center=include_stage1,
            center_k=center_k,
            ncols=ncols,
            fig_w=fig_w,
            dpi=dpi,
            title_fs=title_fs,
            out_dir=out_dir,
            pan_dir=pan_dir,
            pixel_cache=pixel_cache,
            edge_target=edge_target,
            blend_kw=blend_kw,
        )
        return

    if stage1_mode not in ("shared", "multi", "multi_cluster"):
        raise ValueError(f"Unknown stage1.mode={stage1_mode!r}; use per_k | shared")

    mics: MSIParametricMiCSLMC | None = None
    refiner: MSIMiCSMultiDisplayRefine | MSIMiCSParallelDisplay | None = None
    parallel_mode = stage2_mode in ("parallel", "parallel_display", "parallel_heads")
    try:
        logger.info(
            "Stage 1: MiCS (layers=%d, clusters=%s, %d epochs)",
            num_layers,
            mics_kw.get("clusters"),
            stage1_epochs,
        )
        mics = MSIParametricMiCSLMC(**mics_kw)
        mics.fit(msi, pixel_sampling_cache=pixel_cache)
        base_emb = mics.predict_embedding(msi)
        np.save(out_dir / "stage1_mics_embedding.npy", base_emb.astype(np.float32))

        mics_rgb = np.asarray(mics.predict(msi), dtype=np.uint8)[..., :3].copy()
        mics_rgb[~valid_mask] = 0
        dice_stage1 = _global_continuous_dice(cfg, mics_rgb, edge_target, valid_mask)
        logger.info("Stage 1 MiCS continuous Dice vs HD: %.4f", dice_stage1)

        if include_stage1:
            mics_path = pan_dir / "panel__stage1_mics.png"
            Image.fromarray(mics_rgb, mode="RGB").save(mics_path)
            logger.info("Stage-1 MiCS panel: %s", _abs_path_str(mics_path))

        if parallel_mode:
            logger.info(
                "Stage 2: parallel display heads (epochs=%s, anchor_w=%s, diversity_w=%s)",
                stage2_kw.get("num_epochs"),
                stage2_kw.get("anchor_weight"),
                stage2_kw.get("diversity_weight"),
            )
            refiner = MSIMiCSParallelDisplay(mics, **stage2_kw)
            refiner.fit(
                msi,
                pixel_sampling_cache=pixel_cache,
                base_embedding=base_emb,
            )
        else:
            logger.info(
                "Stage 2: centroid blend (epochs=%s, mode=%s)",
                stage2_kw.get("num_epochs"),
                stage2_kw.get("display_mode"),
            )
            use_edge_loss = (
                float(stage2_kw.get("dice_loss_weight", 0.0)) > 0.0
                or float(stage2_kw.get("edge_corr_loss_weight", 0.0)) > 0.0
                or float(stage2_kw.get("edge_mag_loss_weight", 0.0)) > 0.0
            )
            refiner = MSIMiCSMultiDisplayRefine(mics, **stage2_kw)
            refiner.fit(
                msi,
                hd_edges=edge_target if use_edge_loss else None,
                pixel_sampling_cache=pixel_cache,
                base_embedding=base_emb,
            )

        level_rgbs: list[np.ndarray] = []
        level_titles: list[str] = []
        for idx, k in enumerate(refiner.level_ks):
            rgb = refiner.predict_level_display(idx)
            level_rgbs.append(rgb)
            title_prefix = "MiCS head k=" if parallel_mode else "MiCS view k="
            level_titles.append(f"{title_prefix}{k}")
            path = pan_dir / f"panel__mics_view_k{k}.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            logger.info("Continuous MiCS panel k=%s: %s", k, _abs_path_str(path))

        mixed_rgb: np.ndarray | None = None
        dice_mixed = float("nan")
        if not parallel_mode and (
            include_mixed or int(stage2_kw.get("num_epochs", 0)) > 0
        ):
            assert isinstance(refiner, MSIMiCSMultiDisplayRefine)
            mixed_rgb = refiner.predict_mixed()
            mixed_path = pan_dir / "panel__mixed_mics_view.png"
            Image.fromarray(mixed_rgb, mode="RGB").save(mixed_path)
            logger.info("Mixed continuous MiCS view: %s", _abs_path_str(mixed_path))
            dice_mixed = _global_continuous_dice(cfg, mixed_rgb, edge_target, valid_mask)
            if not (np.isfinite(dice_mixed) and dice_mixed >= dice_stage1 - 0.02):
                logger.warning(
                    "Mixed view diverged from stage-1 MiCS edge dice (%.4f vs %.4f)",
                    dice_mixed,
                    dice_stage1,
                )
                mixed_rgb = mics_rgb.copy()
                dice_mixed = dice_stage1

        if save_mix_weights and isinstance(refiner, MSIMiCSMultiDisplayRefine):
            weights = refiner.predict_mix_weights()
            np.save(out_dir / "mix_weights.npy", weights.astype(np.float32))
            mean_w = weights[valid_mask].mean(axis=0) if weights.ndim == 3 else weights.mean(axis=0)
            logger.info(
                "Mean mix weights (coarse→fine): %s",
                ", ".join(f"{w:.3f}" for w in mean_w),
            )

        mosaic = out_dir / "gallery_mics_multiscale_viz.png"
        hub_center_rgb = mics_rgb
        hub_center_title = "MiCS"
        if layout in ("hub", "mics_center", "center"):
            _save_hub_mosaic(
                level_rgbs,
                level_titles,
                hub_center_rgb,
                hub_center_title,
                out_path=mosaic,
                ncols=ncols,
                fig_w=fig_w,
                dpi=dpi,
                title_fs=title_fs,
                arrow_color=arrow_color,
            )
        else:
            grid_rgbs = list(level_rgbs)
            grid_titles = list(level_titles)
            grid_rgbs.append(mics_rgb)
            grid_titles.append("MiCS")
            if include_mixed and mixed_rgb is not None:
                grid_rgbs.append(mixed_rgb)
                grid_titles.append("Mixed MiCS view")
            _save_mosaic(
                grid_rgbs,
                grid_titles,
                out_path=mosaic,
                ncols=ncols,
                fig_w=fig_w,
                dpi=dpi,
                title_fs=title_fs,
            )
        logger.info("Gallery mosaic: %s", _abs_path_str(mosaic))

        summary = out_dir / "multiscale_viz_summary.txt"
        with summary.open("w", encoding="utf-8") as fh:
            fh.write(f"npy_path: {npy_path}\n")
            fh.write(f"edge_target: {edge_label}\n")
            fh.write(f"stage1_mode: shared\n")
            fh.write(f"num_layers: {num_layers}\n")
            fh.write(f"clusters: {refiner.level_ks}\n")
            fh.write(f"layout: {layout}\n")
            fh.write(f"stage1_epochs: {stage1_epochs}\n")
            fh.write(f"stage1_dice: {dice_stage1:.4f}\n")
            fh.write(f"display_type: continuous_mics (not segmentation palette)\n")
            fh.write(f"stage2_mode: {stage2_mode}\n")
            fh.write(f"stage2_epochs: {stage2_kw.get('num_epochs')}\n")
            if mixed_rgb is not None and np.isfinite(dice_mixed):
                fh.write(f"mixed_dice: {dice_mixed:.4f}\n")
                fh.write(f"mixed_dice_delta: {dice_mixed - dice_stage1:+.4f}\n")
            fh.write("\n")
            for k in sorted(stage2_kw):
                fh.write(f"{k}: {stage2_kw.get(k)}\n")
            fh.write(f"\nmosaic: {_abs_path_str(mosaic)}\n")
        logger.info("Summary: %s", _abs_path_str(summary))

    finally:
        if refiner is not None:
            try:
                refiner.release_resources()
            except Exception:
                pass
        if mics is not None:
            try:
                mics.release_resources()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
