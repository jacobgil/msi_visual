#!/usr/bin/env python3
"""
Two-stage MiCS → GNN spatial refinement.

Stage 1: train ``MSIParametricMiCSLMC`` (per-pixel PCC, MiCS losses).
Stage 2: freeze MiCS, train pixel-neighbor GNN residual
          ``e = e_mics + Δ_gnn`` with patch continuous Dice (+ correlation).

Run (repo root):
    python scripts/experiment_mics_gnn.py
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

from msi_visual.parametric_mics_gnn import MSIMiCSGNNRefine  # noqa: E402
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from msi_visual.parametric_mics_unet_refine import MSIMiCSUnetEdgeRefine  # noqa: E402
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
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
    "random_state",
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


def _parse_clusters(val: Any) -> list[int]:
    if val is None:
        return [1024, 2048, 4096]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip()]
    return [int(s)]


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    s = msi.sum(axis=-1, keepdims=True)
    return msi / (s + 1e-8)


def _load_rank_cfg(cfg: DictConfig) -> DictConfig:
    rank_sel = OmegaConf.select(cfg, "rank_config", default=None)
    if rank_sel is None or str(rank_sel).strip() == "":
        raise ValueError("rank_config required for continuous Dice evaluation")
    return OmegaConf.load(to_absolute_path(str(rank_sel)))


def _embedding_delta_stats(
    base_emb: np.ndarray,
    refined_emb: np.ndarray,
    delta: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    """RMS norms for GNN residual relative to stage-1 embedding."""
    m = np.asarray(valid_mask, dtype=bool)
    b = base_emb[m].astype(np.float64, copy=False)
    d = delta[m].astype(np.float64, copy=False)
    diff = (refined_emb - base_emb)[m].astype(np.float64, copy=False)
    base_rms = float(np.sqrt((b * b).mean()))
    delta_rms = float(np.sqrt((d * d).mean()))
    diff_rms = float(np.sqrt((diff * diff).mean()))
    return {
        "delta_rms": delta_rms,
        "diff_rms": diff_rms,
        "base_rms": base_rms,
        "delta_rel_rms": delta_rms / (base_rms + 1e-8),
        "max_abs_delta": float(np.abs(d).max()),
    }


def _rgb_diff_heatmap(
    mics_rgb: np.ndarray,
    refined_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gain: float = 8.0,
) -> np.ndarray:
    """Amplified per-pixel RGB L2 diff (makes subtle GNN changes visible)."""
    m = np.asarray(valid_mask, dtype=bool)
    a = np.asarray(mics_rgb, dtype=np.float32)
    b = np.asarray(refined_rgb, dtype=np.float32)
    diff = np.sqrt(np.sum((a - b) ** 2, axis=-1)) / np.sqrt(3.0 * 255.0 ** 2)
    diff = np.clip(diff * float(gain), 0.0, 1.0)
    diff[~m] = 0.0
    return np.uint8(255.0 * diff)


def _delta_magnitude_heatmap(
    delta: np.ndarray,
    valid_mask: np.ndarray,
    *,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Grayscale uint8 heatmap of L2 norm of embedding residual per pixel."""
    mag = np.sqrt(np.sum(np.asarray(delta, dtype=np.float32) ** 2, axis=-1))
    mask = np.asarray(valid_mask, dtype=bool)
    mag[~mask] = 0.0
    vals = mag[mask]
    if vals.size == 0:
        return np.zeros(mag.shape, dtype=np.uint8)
    p_lo = float(np.percentile(vals, low))
    p_hi = float(np.percentile(vals, high))
    out = (mag - p_lo) / (p_hi - p_lo + 1e-8)
    out = np.clip(out, 0.0, 1.0)
    gray = np.uint8(255.0 * out)
    gray[~mask] = 0
    return gray


def _rgb_from_embedding_anchor(
    emb: np.ndarray,
    anchor_emb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    low: float,
    high: float,
    lab_to_rgb: bool,
) -> np.ndarray:
    """Percentile stretch using anchor embedding stats (fair A/B comparison)."""
    import cv2

    result = np.asarray(emb, dtype=np.float32).copy()
    anchor = np.asarray(anchor_emb, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    for i in range(result.shape[-1]):
        ch = result[:, :, i]
        aval = anchor[:, :, i][mask]
        if aval.size == 0:
            p_low = float(np.percentile(anchor[:, :, i], low))
            p_high = float(np.percentile(anchor[:, :, i], high))
        else:
            p_low = float(np.percentile(aval, low))
            p_high = float(np.percentile(aval, high))
        ch = ch - p_low
        ch[ch < 0] = 0.0
        ch = ch / (p_high - p_low + 1e-8)
        ch[ch > 1] = 1.0
        result[:, :, i] = ch
    out = np.uint8(255.0 * result)
    out[~mask] = 0
    if lab_to_rgb and result.shape[-1] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    out[~mask] = 0
    return out


def _viz_edge_and_dice(
    cfg: DictConfig,
    viz_rgb: np.ndarray,
    hd_edges: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Eval-aligned viz edge map (H,W) in [0,1] and continuous Dice vs HD."""
    from bayes_tune_mics_lmc import _compute_viz_edge_n, _to_uint8_rgb
    from debug_edge_maps import _compute_continuous_dice, _edges_for_continuous_metrics

    rank_cfg = _load_rank_cfg(cfg)
    square = bool(
        OmegaConf.select(rank_cfg, "evaluation.square_before_continuous_metrics", default=False)
    )
    viz_u8 = _to_uint8_rgb(viz_rgb)
    hd_n = np.asarray(hd_edges, dtype=np.float64)
    m = np.asarray(valid_mask, dtype=bool)
    viz_edge_n = _compute_viz_edge_n(viz_u8, m, rank_cfg)
    hd_d, v_d = _edges_for_continuous_metrics(hd_n, viz_edge_n, square=square)
    dice = float(_compute_continuous_dice(hd_d, v_d, m))
    return np.asarray(viz_edge_n, dtype=np.float32), dice


def _global_continuous_dice(
    cfg: DictConfig,
    viz_rgb: np.ndarray,
    hd_edges: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """Global continuous Dice: 2·Σ(hd·viz_edge) / (Σhd + Σviz_edge) on valid pixels."""
    _, dice = _viz_edge_and_dice(cfg, viz_rgb, hd_edges, valid_mask)
    return dice


def _edge_map_heatmap(edge_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Uint8 grayscale heatmap for edge magnitude panels."""
    m = np.asarray(valid_mask, dtype=bool)
    em = np.asarray(edge_map, dtype=np.float32)
    em = np.clip(em, 0.0, 1.0)
    em[~m] = 0.0
    return np.uint8(255.0 * em)


def _save_edge_comparison_panel(
    out_dir: Path,
    *,
    mics_viz: np.ndarray,
    refined_viz: np.ndarray,
    hd_edges: np.ndarray,
    valid_mask: np.ndarray,
    cfg: DictConfig,
    dice_stage1: float,
    dice_stage2: float,
) -> Path:
    """MiCS vs GNN vs HD edge maps — shows edge alignment even when RGB looks similar."""
    m = np.asarray(valid_mask, dtype=bool)
    hd = np.clip(np.asarray(hd_edges, dtype=np.float32), 0.0, 1.0)
    hd[~m] = 0.0

    edge1, dice_e1 = _viz_edge_and_dice(cfg, mics_viz, hd_edges, valid_mask)
    edge2, dice_e2 = _viz_edge_and_dice(cfg, refined_viz, hd_edges, valid_mask)
    edge_delta = edge2 - edge1
    edge_delta[~m] = 0.0

    np.save(out_dir / "stage1_viz_edge.npy", edge1)
    np.save(out_dir / "stage2_viz_edge.npy", edge2)
    np.save(out_dir / "stage2_viz_edge_delta.npy", edge_delta.astype(np.float32))

    vmax = float(max(hd[m].max() if m.any() else 1.0, edge1[m].max(), edge2[m].max(), 1e-6))
    dmax = float(max(np.abs(edge_delta[m]).max() if m.any() else 0.0, 1e-6))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    panels = (
        (axes[0, 0], edge1, "magma", f"MiCS viz edges\nDice vs HD {dice_e1:.4f}"),
        (axes[0, 1], edge2, "magma", f"+ GNN viz edges\nDice vs HD {dice_e2:.4f}"),
        (axes[1, 0], hd, "magma", "HD reference edges"),
        (axes[1, 1], edge_delta, "RdBu_r", f"GNN - MiCS edges\nDice RGB {dice_stage1:.4f} -> {dice_stage2:.4f}"),
    )
    for ax, data, cmap, title in panels:
        if cmap == "RdBu_r":
            im = ax.imshow(data, cmap=cmap, vmin=-dmax, vmax=dmax)
        else:
            im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Edge maps (eval metric) | viz-edge Dice delta {dice_e2 - dice_e1:+.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "comparison_edge_maps.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(str(path), mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI HxWxC, got shape {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI must be nonnegative.")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def _mics_kwargs_from_cfg(cfg: DictConfig, seed: int, num_epochs: int) -> dict[str, Any]:
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    kw: dict[str, Any] = {"random_state": int(seed)}
    for name in _MICS_KEYS:
        if name in merged:
            kw[name] = merged[name]
    kw["num_epochs"] = int(num_epochs)
    kw["clusters"] = _parse_clusters(merged.get("clusters", [1024, 2048, 4096]))
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw.setdefault("num_layers", 7)
    kw.setdefault("predict_spatial_smooth_sigma", 0.0)
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
    kw = {
        "num_epochs": int(s2.get("num_epochs", 40)),
        "lr": float(s2.get("lr", 0.05)),
        "gnn_hidden_dim": int(s2.get("gnn_hidden_dim", 128)),
        "gnn_layers": int(s2.get("gnn_layers", 1)),
        "gnn_neighbor_radius": int(s2.get("gnn_neighbor_radius", 1)),
        "gnn_input": str(s2.get("gnn_input", "embedding")),
        "gnn_pca_dims": int(s2.get("gnn_pca_dims", 32)),
        "gnn_pca_fit_step": int(s2.get("gnn_pca_fit_step", 16)),
        "dice_loss_weight": float(s2.get("dice_loss_weight", 0.01)),
        "dice_patch_size": int(s2.get("dice_patch_size", 16)),
        "dice_patches_per_step": int(s2.get("dice_patches_per_step", 32)),
        "dice_patch_sampling": str(s2.get("dice_patch_sampling", "uniform")),
        "dice_edge_uniform_mix": float(s2.get("dice_edge_uniform_mix", 0.1)),
        "dice_edge_power": float(s2.get("dice_edge_power", 2.0)),
        "dice_global_weight": float(s2.get("dice_global_weight", 0.5)),
        "dice_patch_weight": float(s2.get("dice_patch_weight", 0.5)),
        "dice_min_valid_fraction": float(s2.get("dice_min_valid_fraction", 0.25)),
        "mics_loss_weight": float(s2.get("mics_loss_weight", 1.0)),
        "delta_scale": float(s2.get("delta_scale", 1.0)),
        "delta_l2_weight": float(s2.get("delta_l2_weight", 0.0)),
        "delta_highpass": bool(s2.get("delta_highpass", True)),
        "delta_max_rel": float(s2.get("delta_max_rel", 0.0)),
        "delta_space": str(s2.get("delta_space", "embedding")),
        "contrast_loss_weight": float(s2.get("contrast_loss_weight", 0.0)),
        "edge_mag_loss_weight": float(s2.get("edge_mag_loss_weight", 0.0)),
        "edge_mag_hd_power": float(s2.get("edge_mag_hd_power", 2.0)),
        "edge_corr_loss_weight": float(s2.get("edge_corr_loss_weight", 0.0)),
        "gnn_hd_edge_channel": bool(s2.get("gnn_hd_edge_channel", False)),
        "edge_delta_gate_strength": float(s2.get("edge_delta_gate_strength", 0.0)),
        "edge_delta_gate_power": float(s2.get("edge_delta_gate_power", 1.5)),
        "edge_loss_ramp_epochs": int(s2.get("edge_loss_ramp_epochs", 0)),
        "dice_edge_color_space": str(s2.get("dice_edge_color_space", "rgb_max")),
        "dice_edge_percentile_norm": bool(s2.get("dice_edge_percentile_norm", True)),
        "non_edge_l1_weight": float(s2.get("non_edge_l1_weight", 0.0)),
        "batch_size": int(s2.get("batch_size", mics_kw.get("batch_size", 1024))),
        "verbose": bool(mics_kw.get("verbose", False)),
        "predict_percentile_low": float(mics_kw.get("predict_percentile_low", 1.0)),
        "predict_percentile_high": float(mics_kw.get("predict_percentile_high", 99.0)),
        "predict_spatial_smooth_sigma": float(s2.get("predict_spatial_smooth_sigma", 0.0)),
        "predict_anchor_base_percentiles": bool(
            s2.get("predict_anchor_base_percentiles", True)
        ),
        "lab_to_rgb": bool(mics_kw.get("lab_to_rgb", True)),
        # U-Net-only knobs (ignored by GNN constructor via filtering)
        "unet_base_channels": int(s2.get("unet_base_channels", 32)),
        "adv_loss_weight": float(s2.get("adv_loss_weight", 0.0)),
        "edge_delta_mask_only": bool(s2.get("edge_delta_mask_only", False)),
    }
    return kw


def _build_refiner(mics: MSIParametricMiCSLMC, stage2_kw: dict[str, Any], refiner: str):
    kind = str(refiner).lower().strip()
    if kind == "unet":
        return MSIMiCSUnetEdgeRefine(mics, **stage2_kw)
    gnn_kw = {
        k: v
        for k, v in stage2_kw.items()
        if k not in ("unet_base_channels", "adv_loss_weight", "edge_delta_mask_only")
    }
    return MSIMiCSGNNRefine(mics, **gnn_kw)


@hydra.main(version_base=None, config_path="configs", config_name="experiment_mics_gnn")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    data = getattr(cfg, "data", None)
    pipe = getattr(cfg, "pipeline", None)
    st1 = getattr(cfg, "stage1", None)
    if data is None:
        raise ValueError("Config must define data:")

    npy_path = Path(to_absolute_path(str(data.npy_path))).expanduser().resolve()
    if not npy_path.is_file():
        raise FileNotFoundError(f"data.npy_path not found: {npy_path}")

    msi = _load_msi(npy_path, bool(data.transpose_msi), bool(data.tic_normalize))
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    seed = int(OmegaConf.select(pipe, "seed", default=42))
    stage1_epochs = int(OmegaConf.select(st1, "num_epochs", default=30))
    mics_kw = _mics_kwargs_from_cfg(cfg, seed, stage1_epochs)
    stage2_kw = _stage2_kwargs(cfg, mics_kw)

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )

    share = bool(OmegaConf.select(pipe, "share_pixel_sampling", default=True))
    cache_raw = OmegaConf.select(pipe, "pixel_sampling_cache_path", default=None)
    cache_path: Path | None = None
    if cache_raw is not None and str(cache_raw).strip().lower() not in ("", "~", "null", "none"):
        cache_path = Path(to_absolute_path(str(cache_raw))).expanduser().resolve()

    pixel_cache = resolve_pixel_sampling_cache(
        msi,
        _sampling_cache_kwargs(mics_kw),
        out_dir=out_dir,
        share=share,
        cache_path=cache_path,
    )

    ckpt_raw = OmegaConf.select(pipe, "stage1_checkpoint", default=None)
    ckpt_path: Path | None = None
    if ckpt_raw is not None and str(ckpt_raw).strip().lower() not in ("", "~", "null", "none"):
        ckpt_path = Path(to_absolute_path(str(ckpt_raw))).expanduser().resolve()

    mics: MSIParametricMiCSLMC | None = None
    refiner: MSIMiCSGNNRefine | MSIMiCSUnetEdgeRefine | None = None
    refiner_kind = str(OmegaConf.select(pipe, "refiner", default="gnn")).lower().strip()

    try:
        mics = MSIParametricMiCSLMC(**mics_kw)

        if ckpt_path is not None and ckpt_path.is_file():
            logger.info("Loading stage-1 checkpoint (skip training): %s", ckpt_path)
            blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            mics.set_image(msi, pixel_sampling_cache=pixel_cache)
            if mics.model is None:
                raise ValueError("Checkpoint load failed: MiCS model not initialized.")
            if "mics_state" in blob:
                mics.model.load_state_dict(blob["mics_state"])
            base_emb = (
                np.asarray(blob["mics_base_emb"], dtype=np.float32)
                if "mics_base_emb" in blob
                else mics.predict_embedding(msi)
            )
            np.save(out_dir / "stage1_mics_embedding.npy", base_emb)
        else:
            logger.info("Stage 1: training MiCS PCC (%d epochs)", stage1_epochs)
            mics.fit(msi, pixel_sampling_cache=pixel_cache)
            base_emb = mics.predict_embedding(msi)
            np.save(out_dir / "stage1_mics_embedding.npy", base_emb)
            if mics.model is not None:
                torch.save(
                    {
                        "mics_state": mics.model.state_dict(),
                        "mics_base_emb": base_emb,
                    },
                    out_dir / "stage1_mics.pt",
                )

        mics_viz = mics.predict(msi)
        Image.fromarray(mics_viz, mode="RGB").save(out_dir / "stage1_mics_viz.png")
        logger.info("Stage 1 done: %s", out_dir / "stage1_mics_viz.png")

        logger.info("Building HD edge map for stage-2 Dice loss...")
        hd_edges = _build_hd_edge_map_for_train(cfg, msi).astype(np.float32)
        np.save(out_dir / "hd_edges.npy", hd_edges)

        dice_stage1 = _global_continuous_dice(cfg, mics_viz, hd_edges, valid_mask)
        logger.info("Stage 1 continuous Dice (global): %.4f", dice_stage1)

        logger.info(
            "Stage 2: %s refine (epochs=%s, dice_w=%s, patch_sampling=%s, "
            "delta_scale=%s, delta_space=%s, delta_max_rel=%s, gate=%s, "
            "global_w=%s, non_edge=%s, mics_w=%s)",
            refiner_kind.upper(),
            stage2_kw["num_epochs"],
            stage2_kw["dice_loss_weight"],
            stage2_kw.get("dice_patch_sampling", "uniform"),
            stage2_kw["delta_scale"],
            stage2_kw.get("delta_space", "embedding"),
            stage2_kw.get("delta_max_rel", 0.0),
            stage2_kw.get("edge_delta_gate_strength", 0.0),
            stage2_kw.get("dice_global_weight", 0.5),
            stage2_kw.get("non_edge_l1_weight", 0.0),
            stage2_kw.get("mics_loss_weight", 0.0),
        )
        refiner = _build_refiner(mics, stage2_kw, refiner_kind)
        refiner.fit(
            msi,
            hd_edges=hd_edges,
            pixel_sampling_cache=pixel_cache,
            base_embedding=base_emb,
        )

        refined_emb = refiner.predict_embedding()
        refined_viz = refiner.predict()
        preview_gain = float(OmegaConf.select(pipe, "preview_delta_gain", default=0.0))
        refined_preview = (
            refiner.predict(viz_delta_gain=preview_gain)
            if preview_gain > 1.0
            else None
        )
        delta = refiner.predict_delta()
        dstats = _embedding_delta_stats(base_emb, refined_emb, delta, valid_mask)
        logger.info(
            "GNN delta stats: rms=%.4f rel_rms=%.4f max_abs=%.4f (vs stage-1 base)",
            dstats["delta_rms"],
            dstats["delta_rel_rms"],
            dstats["max_abs_delta"],
        )

        np.save(out_dir / "stage2_refined_embedding.npy", refined_emb)
        np.save(out_dir / "stage2_gnn_delta.npy", delta)
        Image.fromarray(refined_viz, mode="RGB").save(out_dir / "stage2_refined_viz.png")

        # Fair comparison: same percentile anchors as stage-1 embedding
        anchor_low = float(mics_kw.get("predict_percentile_low", 1.0))
        anchor_high = float(mics_kw.get("predict_percentile_high", 99.0))
        lab2rgb = bool(mics_kw.get("lab_to_rgb", True))
        mics_viz_fair = _rgb_from_embedding_anchor(
            base_emb, base_emb, valid_mask, low=anchor_low, high=anchor_high, lab_to_rgb=lab2rgb
        )
        refined_viz_fair = _rgb_from_embedding_anchor(
            refined_emb, base_emb, valid_mask, low=anchor_low, high=anchor_high, lab_to_rgb=lab2rgb
        )
        Image.fromarray(mics_viz_fair, mode="RGB").save(out_dir / "stage1_mics_viz_fair.png")
        Image.fromarray(refined_viz_fair, mode="RGB").save(out_dir / "stage2_refined_viz_fair.png")

        dice_stage2 = _global_continuous_dice(cfg, refined_viz, hd_edges, valid_mask)
        dice_stage1_fair = _global_continuous_dice(cfg, mics_viz_fair, hd_edges, valid_mask)
        dice_stage2_fair = _global_continuous_dice(cfg, refined_viz_fair, hd_edges, valid_mask)
        logger.info("Stage 2 continuous Dice (global): %.4f", dice_stage2)
        logger.info(
            "Continuous Dice delta (stage2 - stage1): %+.4f",
            dice_stage2 - dice_stage1,
        )

        if bool(OmegaConf.select(pipe, "save_comparison", default=True)):
            delta_space = str(stage2_kw.get("delta_space", "embedding"))
            rgb_diff = _rgb_diff_heatmap(mics_viz, refined_viz, valid_mask)
            ncols = 4 if refined_preview is not None else 3
            fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 5), dpi=150)
            if ncols == 3:
                axes = list(axes)
            axes[0].imshow(mics_viz)
            axes[0].set_title(f"MiCS\nDice {dice_stage1:.4f}")
            axes[0].axis("off")
            axes[1].imshow(refined_viz)
            axes[1].set_title(
                f"+ GNN ({delta_space})\nDice {dice_stage2:.4f}"
            )
            axes[1].axis("off")
            col = 2
            if refined_preview is not None:
                axes[col].imshow(refined_preview)
                axes[col].set_title(f"Preview {preview_gain:.0f}x Δ\n(not in Dice)")
                axes[col].axis("off")
                Image.fromarray(refined_preview, mode="RGB").save(
                    out_dir / "stage2_refined_viz_preview.png"
                )
                col += 1
            im = axes[col].imshow(rgb_diff, cmap="inferno", vmin=0, vmax=255)
            axes[col].set_title("|ΔRGB| x8")
            axes[col].axis("off")
            fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
            fig.tight_layout()
            cmp_path = out_dir / "comparison_mics_vs_refined.png"
            fig.savefig(cmp_path, bbox_inches="tight")
            plt.close(fig)
            Image.fromarray(rgb_diff, mode="L").save(out_dir / "comparison_rgb_diff.png")
            logger.info("Comparison: %s", cmp_path)

        if bool(OmegaConf.select(pipe, "save_delta_panel", default=True)):
            delta_heat = _delta_magnitude_heatmap(delta, valid_mask)
            Image.fromarray(delta_heat, mode="L").save(out_dir / "stage2_delta_magnitude.png")
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
            axes[0].imshow(mics_viz)
            axes[0].set_title(f"MiCS\nDice {dice_stage1:.4f}")
            axes[0].axis("off")
            axes[1].imshow(refined_viz)
            axes[1].set_title(f"+ GNN\nDice {dice_stage2:.4f}")
            axes[1].axis("off")
            im = axes[2].imshow(delta_heat, cmap="inferno")
            axes[2].set_title(
                f"|Δemb|\nrel_rms {dstats['delta_rel_rms']:.3f}"
            )
            axes[2].axis("off")
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            fig.tight_layout()
            delta_path = out_dir / "comparison_with_delta_heatmap.png"
            fig.savefig(delta_path, bbox_inches="tight")
            plt.close(fig)
            logger.info("Delta panel: %s", delta_path)

        if bool(OmegaConf.select(pipe, "save_edge_panel", default=True)):
            edge_path = _save_edge_comparison_panel(
                out_dir,
                mics_viz=mics_viz,
                refined_viz=refined_viz,
                hd_edges=hd_edges,
                valid_mask=valid_mask,
                cfg=cfg,
                dice_stage1=dice_stage1,
                dice_stage2=dice_stage2,
            )
            logger.info("Edge comparison: %s", edge_path)

        summary = out_dir / "run_summary.txt"
        with summary.open("w", encoding="utf-8") as fh:
            fh.write(f"npy_path: {npy_path}\n")
            fh.write(f"output_dir: {out_dir}\n")
            fh.write(f"stage1_epochs: {stage1_epochs}\n")
            fh.write(f"stage1_viz: {out_dir / 'stage1_mics_viz.png'}\n")
            fh.write(f"stage1_checkpoint: {out_dir / 'stage1_mics.pt'}\n")
            fh.write(f"stage2_epochs: {stage2_kw['num_epochs']}\n")
            fh.write(f"stage2_refined_viz: {out_dir / 'stage2_refined_viz.png'}\n")
            fh.write(f"hd_edges: {out_dir / 'hd_edges.npy'}\n")
            fh.write(f"continuous_dice_stage1: {dice_stage1:.6f}\n")
            fh.write(f"continuous_dice_stage2: {dice_stage2:.6f}\n")
            fh.write(f"continuous_dice_stage1_fair_norm: {dice_stage1_fair:.6f}\n")
            fh.write(f"continuous_dice_stage2_fair_norm: {dice_stage2_fair:.6f}\n")
            fh.write(f"continuous_dice_delta: {dice_stage2 - dice_stage1:+.6f}\n")
            fh.write(
                f"continuous_dice_delta_fair_norm: {dice_stage2_fair - dice_stage1_fair:+.6f}\n"
            )
            fh.write(f"gnn_delta_rms: {dstats['delta_rms']:.6f}\n")
            fh.write(f"gnn_delta_rel_rms: {dstats['delta_rel_rms']:.6f}\n")
            fh.write(f"gnn_delta_max_abs: {dstats['max_abs_delta']:.6f}\n")

        logger.info("Done. Outputs in %s", out_dir)
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
