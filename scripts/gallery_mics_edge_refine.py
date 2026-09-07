#!/usr/bin/env python3
"""
Two-stage MiCS → U-Net edge refinement gallery.

Stage 1: standard MiCS (cluster + correlation) — visualization appeal preserved.
Stage 2: frozen MiCS + U-Net residual + edge Dice/corr + patch adversary.

Gallery: MiCS | refined | |Δ| heatmap | HD edge target | refined viz edges

Run:
    python scripts/gallery_mics_edge_refine.py
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
import matplotlib.pyplot as plt
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

from gallery_mics_edge import _edge_to_heatmap, _resolve_edge_target  # noqa: E402
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from msi_visual.parametric_mics_unet_refine import MSIMiCSUnetEdgeRefine  # noqa: E402
from train_parametric_mics_lmc import _build_hd_edge_map_for_train  # noqa: E402

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


def _mics_kwargs_from_cfg(cfg: DictConfig, seed: int, num_epochs: int) -> dict[str, Any]:
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    kw: dict[str, Any] = {"random_state": int(seed)}
    for name in _MICS_KEYS:
        if name in merged:
            kw[name] = merged[name]
    kw["num_epochs"] = int(num_epochs)
    kw["clusters"] = _parse_clusters(merged.get("clusters", [8, 32, 128]))
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw.setdefault("number_of_points", int(merged.get("number_of_points", 1000)))
    kw.setdefault("sampling", str(merged.get("sampling", "random")))
    kw.setdefault("number_of_components", int(merged.get("number_of_components", 3)))
    kw.setdefault("num_layers", int(merged.get("num_layers", 2)))
    kw.setdefault("lab_to_rgb", bool(merged.get("lab_to_rgb", True)))
    kw.setdefault("cluster", bool(merged.get("cluster", True)))
    kw.setdefault("num_samples", int(merged.get("num_samples", 20000)))
    kw.setdefault("cluster_loss_weight", float(merged.get("cluster_loss_weight", 1.0)))
    kw.setdefault("pixel_sampling", str(merged.get("pixel_sampling", "superpixel")))
    kw.setdefault("predict_spatial_smooth_sigma", 0.0)
    kw["learn_input_mask"] = False
    kw["input_mask_l1_weight"] = 0.0
    return kw


def _sampling_cache_kwargs(mics_kw: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "number_of_points",
        "sampling",
        "number_of_components",
        "num_layers",
        "num_samples",
        "random_state",
        "verbose",
        "pixel_sampling",
        "pca_fit_step",
        "cluster",
        "clusters",
        "cluster_on_pca",
        "cluster_pca_dims",
    )
    out = {k: mics_kw[k] for k in keys if k in mics_kw}
    out.setdefault("num_epochs", 1)
    out["clusters_auto_tune"] = None
    return out


def _stage2_kwargs(cfg: DictConfig, mics_kw: dict[str, Any]) -> dict[str, Any]:
    s2 = OmegaConf.to_container(getattr(cfg, "stage2", {}), resolve=True)
    if not isinstance(s2, dict):
        s2 = {}
    return {
        "num_epochs": int(s2.get("num_epochs", 80)),
        "lr": float(s2.get("lr", 0.02)),
        "unet_base_channels": int(s2.get("unet_base_channels", 32)),
        "gnn_input": str(s2.get("unet_input", s2.get("gnn_input", "embedding_plus_pca"))),
        "gnn_pca_dims": int(s2.get("unet_pca_dims", s2.get("gnn_pca_dims", 48))),
        "gnn_pca_fit_step": int(s2.get("unet_pca_fit_step", s2.get("gnn_pca_fit_step", 16))),
        "dice_loss_weight": float(s2.get("dice_loss_weight", 3.0)),
        "dice_patch_size": int(s2.get("dice_patch_size", 24)),
        "dice_patches_per_step": int(s2.get("dice_patches_per_step", 48)),
        "dice_patch_sampling": str(s2.get("dice_patch_sampling", "edge_stratified")),
        "dice_edge_uniform_mix": float(s2.get("dice_edge_uniform_mix", 0.08)),
        "dice_edge_power": float(s2.get("dice_edge_power", 3.0)),
        "dice_global_weight": float(s2.get("dice_global_weight", 0.35)),
        "dice_patch_weight": float(s2.get("dice_patch_weight", 0.65)),
        "dice_min_valid_fraction": float(s2.get("dice_min_valid_fraction", 0.25)),
        "mics_loss_weight": float(s2.get("mics_loss_weight", 0.0)),
        "delta_scale": float(s2.get("delta_scale", 0.12)),
        "delta_l2_weight": float(s2.get("delta_l2_weight", 0.05)),
        "delta_highpass": bool(s2.get("delta_highpass", False)),
        "delta_max_rel": float(s2.get("delta_max_rel", 0.12)),
        "delta_space": str(s2.get("delta_space", "display")),
        "contrast_loss_weight": float(s2.get("contrast_loss_weight", 0.01)),
        "edge_mag_loss_weight": float(s2.get("edge_mag_loss_weight", 0.5)),
        "edge_mag_hd_power": float(s2.get("edge_mag_hd_power", 2.5)),
        "edge_corr_loss_weight": float(s2.get("edge_corr_loss_weight", 0.8)),
        "gnn_hd_edge_channel": bool(s2.get("unet_hd_edge_channel", True)),
        "edge_delta_gate_strength": float(s2.get("edge_delta_gate_strength", 0.8)),
        "edge_delta_gate_power": float(s2.get("edge_delta_gate_power", 1.5)),
        "edge_delta_mask_only": bool(s2.get("edge_delta_mask_only", True)),
        "edge_loss_ramp_epochs": int(s2.get("edge_loss_ramp_epochs", 12)),
        "dice_edge_color_space": str(s2.get("dice_edge_color_space", "rgb_max")),
        "dice_edge_percentile_norm": bool(s2.get("dice_edge_percentile_norm", True)),
        "adv_loss_weight": float(s2.get("adv_loss_weight", 0.08)),
        "adv_patch_size": int(s2.get("adv_patch_size", 32)),
        "adv_patches_per_step": int(s2.get("adv_patches_per_step", 12)),
        "adv_d_lr": float(s2.get("adv_d_lr", 0.01)),
        "non_edge_l1_weight": float(s2.get("non_edge_l1_weight", 0.35)),
        "batch_size": int(s2.get("batch_size", mics_kw.get("batch_size", 1024))),
        "verbose": bool(mics_kw.get("verbose", False)),
        "predict_percentile_low": float(mics_kw.get("predict_percentile_low", 1.0)),
        "predict_percentile_high": float(mics_kw.get("predict_percentile_high", 99.0)),
        "predict_spatial_smooth_sigma": float(s2.get("predict_spatial_smooth_sigma", 0.0)),
        "predict_anchor_base_percentiles": bool(
            s2.get("predict_anchor_base_percentiles", True)
        ),
        "lab_to_rgb": bool(mics_kw.get("lab_to_rgb", True)),
    }


def _rgb_diff_heatmap(
    mics_rgb: np.ndarray,
    refined_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gain: float = 8.0,
) -> np.ndarray:
    m = np.asarray(valid_mask, dtype=bool)
    a = np.asarray(mics_rgb, dtype=np.float32)
    b = np.asarray(refined_rgb, dtype=np.float32)
    diff = np.sqrt(np.sum((a - b) ** 2, axis=-1)) / np.sqrt(3.0 * 255.0 ** 2)
    diff = np.clip(diff * float(gain), 0.0, 1.0)
    diff[~m] = 0.0
    import cv2

    u8 = np.uint8(255.0 * diff)
    u8_rgb = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    u8_rgb = cv2.cvtColor(u8_rgb, cv2.COLOR_BGR2RGB)
    u8_rgb[~m] = 0
    return u8_rgb


def _global_continuous_dice(
    cfg: DictConfig,
    viz_rgb: np.ndarray,
    hd_edges: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    from experiment_mics_gnn import _global_continuous_dice as _dice

    return _dice(cfg, viz_rgb, hd_edges, valid_mask)


def _viz_edge_map(
    cfg: DictConfig,
    viz_rgb: np.ndarray,
    hd_edges: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    from experiment_mics_gnn import _viz_edge_and_dice

    edge_map, _ = _viz_edge_and_dice(cfg, viz_rgb, hd_edges, valid_mask)
    return edge_map


def _save_mosaic(
    panel_rgbs: list[np.ndarray],
    titles: list[str],
    *,
    out_path: Path,
    ncols: int,
    fig_w: float,
    dpi: int,
    title_fs: int,
) -> None:
    n_p = len(panel_rgbs)
    nrows = int(np.ceil(n_p / ncols))
    mh = float(max(fig_w * 0.35, fig_w * 0.28 * max(1, nrows)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, mh))
    fig.patch.set_facecolor("black")
    if nrows == 1 and ncols == 1:
        ax_grid = np.array([[axes]])
    elif nrows == 1:
        ax_grid = np.asarray(axes).reshape(1, -1)
    elif ncols == 1:
        ax_grid = np.asarray(axes).reshape(-1, 1)
    else:
        ax_grid = np.asarray(axes)

    for idx, rgb in enumerate(panel_rgbs):
        r, c = divmod(idx, ncols)
        ax = ax_grid[r, c]
        ax.set_facecolor("black")
        ax.imshow(rgb)
        ax.set_title(titles[idx], fontsize=title_fs, color="white", pad=6)
        ax.axis("off")
    for idx in range(n_p, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax_grid[r, c].set_facecolor("black")
        ax_grid[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_edge_refine")
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

    seed = int(OmegaConf.select(ga, "seed", default=42))
    ncols = max(1, int(OmegaConf.select(ga, "ncols", default=3)))
    fig_w = float(OmegaConf.select(ga, "figure_size_inches", default=16))
    dpi = int(OmegaConf.select(ga, "dpi", default=300))
    title_fs = int(OmegaConf.select(ga, "panel_title_fontsize", default=10))
    diff_gain = float(OmegaConf.select(ga, "diff_gain", default=8.0))

    stage1_epochs = int(OmegaConf.select(st1, "num_epochs", default=100))
    mics_kw = _mics_kwargs_from_cfg(cfg, seed, stage1_epochs)
    stage2_kw = _stage2_kwargs(cfg, mics_kw)

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

    pixel_cache = resolve_pixel_sampling_cache(
        msi,
        _sampling_cache_kwargs(mics_kw),
        out_dir=out_dir,
        share=share_sampling,
        cache_path=cache_path,
    )

    mics: MSIParametricMiCSLMC | None = None
    refiner: MSIMiCSUnetEdgeRefine | None = None
    try:
        logger.info("Stage 1: MiCS (%d epochs, no edge loss)", stage1_epochs)
        mics = MSIParametricMiCSLMC(**mics_kw)
        mics.fit(msi, pixel_sampling_cache=pixel_cache)
        base_emb = mics.predict_embedding(msi)
        np.save(out_dir / "stage1_mics_embedding.npy", base_emb.astype(np.float32))

        mics_rgb = np.asarray(mics.predict(msi), dtype=np.uint8)[..., :3]
        mics_rgb[~valid_mask] = 0
        dice_stage1 = _global_continuous_dice(cfg, mics_rgb, edge_target, valid_mask)
        logger.info("Stage 1 continuous Dice vs HD: %.4f", dice_stage1)

        logger.info(
            "Stage 2: U-Net refine (epochs=%s, adv_w=%s, dice_w=%s)",
            stage2_kw.get("num_epochs"),
            stage2_kw.get("adv_loss_weight"),
            stage2_kw.get("dice_loss_weight"),
        )
        refiner = MSIMiCSUnetEdgeRefine(mics, **stage2_kw)
        refiner.fit(
            msi,
            hd_edges=edge_target,
            pixel_sampling_cache=pixel_cache,
            base_embedding=base_emb,
        )

        refined_rgb = np.asarray(refiner.predict(msi), dtype=np.uint8)[..., :3]
        refined_rgb[~valid_mask] = 0
        delta_emb = refiner.predict_delta()
        delta_mag = np.sqrt(np.sum(delta_emb.astype(np.float64) ** 2, axis=-1))
        delta_mag[~valid_mask] = 0.0
        delta_rms = float(np.sqrt((delta_emb[valid_mask].astype(np.float64) ** 2).mean()))
        delta_disp_rms = float("nan")
        if getattr(refiner, "delta_space", "") == "display":
            with torch.no_grad():
                disp = refiner._forward_delta_tensor().detach().cpu().numpy()
            dm = np.sqrt(np.sum(disp.astype(np.float64) ** 2, axis=-1))
            delta_disp_rms = float(np.sqrt((dm[valid_mask] ** 2).mean()))
        dice_stage2 = _global_continuous_dice(cfg, refined_rgb, edge_target, valid_mask)
        stage2_ok = (
            np.isfinite(dice_stage2)
            and dice_stage2 >= dice_stage1 - 0.02
            and np.isfinite(delta_rms)
        )
        if not stage2_ok:
            logger.warning(
                "Stage 2 degraded viz (dice %.4f vs stage1 %.4f); using stage-1 MiCS for display",
                dice_stage2,
                dice_stage1,
            )
            refined_rgb = mics_rgb.copy()
            dice_stage2 = dice_stage1
            diff_rgb = _rgb_diff_heatmap(mics_rgb, refined_rgb, valid_mask, gain=diff_gain)
            viz_edge = _viz_edge_map(cfg, refined_rgb, edge_target, valid_mask)
            viz_edge_rgb = _edge_to_heatmap(viz_edge, valid_mask)
        else:
            logger.info(
                "Stage 2 continuous Dice vs HD: %.4f (delta %+.4f)",
                dice_stage2,
                dice_stage2 - dice_stage1,
            )
            diff_rgb = _rgb_diff_heatmap(mics_rgb, refined_rgb, valid_mask, gain=diff_gain)
            edge_rgb = _edge_to_heatmap(edge_target, valid_mask)
            viz_edge = _viz_edge_map(cfg, refined_rgb, edge_target, valid_mask)
            viz_edge_rgb = _edge_to_heatmap(viz_edge, valid_mask)

        edge_rgb = _edge_to_heatmap(edge_target, valid_mask)

        titles = [
            f"MiCS (stage 1)\nDice {dice_stage1:.3f}",
            f"+ U-Net refine\nDice {dice_stage2:.3f}"
            + ("" if stage2_ok else " (fallback MiCS)"),
            f"|ΔRGB| ×{diff_gain:g}",
            f"HD edge ({edge_label})",
            "Refined viz edges",
        ]
        rgbs = [mics_rgb, refined_rgb, diff_rgb, edge_rgb, viz_edge_rgb]

        mosaic = out_dir / "gallery_mics_edge_refine.png"
        _save_mosaic(
            rgbs,
            titles,
            out_path=mosaic,
            ncols=ncols,
            fig_w=fig_w,
            dpi=dpi,
            title_fs=title_fs,
        )

        for name, rgb in zip(
            ("mics", "refined", "delta", "edge_target", "viz_edges"),
            rgbs,
            strict=True,
        ):
            Image.fromarray(rgb, mode="RGB").save(pan_dir / f"panel__{name}.png")

        summary = out_dir / "edge_refine_summary.txt"
        with summary.open("w", encoding="utf-8") as fh:
            fh.write(f"npy_path: {npy_path}\n")
            fh.write(f"edge_target: {edge_label}\n")
            fh.write(f"stage1_epochs: {stage1_epochs}\n")
            fh.write(f"stage1_dice: {dice_stage1:.4f}\n")
            fh.write(f"stage2_epochs: {stage2_kw.get('num_epochs')}\n")
            fh.write(f"stage2_used: {stage2_ok}\n")
            fh.write(f"stage2_dice: {dice_stage2:.4f}\n")
            fh.write(f"dice_delta: {dice_stage2 - dice_stage1:+.4f}\n")
            fh.write(f"delta_rms_display: {delta_disp_rms:.6f}\n")
            fh.write(f"delta_rms_embedding: {delta_rms:.6f}\n")
            fh.write(f"delta_max: {float(delta_mag.max()):.6f}\n\n")
            for k in (
                "adv_loss_weight",
                "non_edge_l1_weight",
                "dice_loss_weight",
                "edge_corr_loss_weight",
                "delta_scale",
                "delta_space",
            ):
                fh.write(f"{k}: {stage2_kw.get(k)}\n")
            fh.write(f"\nmosaic: {_abs_path_str(mosaic)}\n")

        logger.info("Gallery: %s", _abs_path_str(mosaic))
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
