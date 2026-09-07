#!/usr/bin/env python3
"""
Train SpatialMiCS: MiCS on 3× spatial features.

    X_spatial = concat( X ,  blur(X, σ_regional) ,  X − blur(X, σ_fine) )

Optionally trains a baseline per-pixel MiCS on the raw spectrum for comparison.

Run (repo root):
    python scripts/train_spatial_mics.py
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

from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from msi_visual.spatial_mics_features import build_spatial_mics_input  # noqa: E402
from train_parametric_mics_lmc import (  # noqa: E402
    _build_hd_edge_map_for_train,
    load_and_preprocess_image,
)

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
        return [128, 256, 512]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip()]
    return [int(s)]


def _load_msi(path: Path, transpose_msi: bool) -> np.ndarray:
    """Same TIC normalization as train_parametric_mics_lmc."""
    return load_and_preprocess_image(str(path), transpose=transpose_msi)


def _resolve_npy_path(cfg: DictConfig) -> Path:
    raw = OmegaConf.select(cfg, "data.npy_path", default=None)
    if raw is not None and str(raw).strip().lower() not in ("", "~", "null", "none"):
        return Path(to_absolute_path(str(raw))).expanduser().resolve()
    raw = OmegaConf.select(cfg, "data.input_path", default=None)
    if raw is None:
        raise ValueError("Config must define data.npy_path or data.input_path")
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _mics_kwargs_from_cfg(cfg: DictConfig, seed: int, num_epochs: int) -> dict[str, Any]:
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    kw: dict[str, Any] = {"random_state": int(seed)}
    for name in _MICS_KEYS:
        if name in merged:
            kw[name] = merged[name]
    kw["num_epochs"] = int(num_epochs)
    kw["clusters"] = _parse_clusters(merged.get("clusters", [128, 256, 512]))
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw.setdefault("num_layers", 5)
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


def _load_rank_cfg(cfg: DictConfig) -> DictConfig:
    rank_sel = OmegaConf.select(cfg, "rank_config", default=None)
    if rank_sel is None or str(rank_sel).strip() == "":
        raise ValueError("rank_config required for continuous Dice evaluation")
    return OmegaConf.load(to_absolute_path(str(rank_sel)))


def _global_continuous_dice(
    cfg: DictConfig,
    viz_rgb: np.ndarray,
    hd_edges: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
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
    return float(_compute_continuous_dice(hd_d, v_d, m))


def _train_and_predict(
    img: np.ndarray,
    mics_kw: dict[str, Any],
    pixel_cache: dict[str, Any],
    *,
    label: str,
) -> tuple[MSIParametricMiCSLMC, np.ndarray, np.ndarray]:
    logger.info("Training %s (%d epochs, input C=%d)...", label, mics_kw["num_epochs"], img.shape[-1])
    model = MSIParametricMiCSLMC(**mics_kw)
    model.fit(img, pixel_sampling_cache=pixel_cache)
    emb = model.predict_embedding(img)
    viz = model.predict(img)
    logger.info("%s done.", label)
    return model, emb, viz


def _save_comparison_panel(
    out_dir: Path,
    *,
    baseline_viz: np.ndarray | None,
    spatial_viz: np.ndarray,
    valid_mask: np.ndarray,
    dice_baseline: float | None,
    dice_spatial: float,
) -> None:
    m = np.asarray(valid_mask, dtype=bool)
    if baseline_viz is not None and dice_baseline is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        panels = (
            (axes[0], baseline_viz, f"MiCS baseline (1x)\nDice={dice_baseline:.4f}"),
            (axes[1], spatial_viz, f"SpatialMiCS (3x)\nDice={dice_spatial:.4f}"),
        )
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
        panels = ((axes, spatial_viz, f"SpatialMiCS (3x)\nDice={dice_spatial:.4f}"),)

    for ax, rgb, title in panels:
        show = np.asarray(rgb, dtype=np.uint8).copy()
        show[~m] = 0
        ax.imshow(show)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    if baseline_viz is not None and dice_baseline is not None:
        fig.suptitle(
            f"SpatialMiCS vs baseline | Dice delta {dice_spatial - dice_baseline:+.4f}",
            fontsize=12,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_spatial_mics.png", bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="train_spatial_mics")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    npy_path = _resolve_npy_path(cfg)
    if not npy_path.is_file():
        raise FileNotFoundError(f"MSI not found: {npy_path}")

    transpose = bool(OmegaConf.select(cfg, "data.transpose", default=False))
    msi = _load_msi(npy_path, transpose)
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    sm = OmegaConf.to_container(getattr(cfg, "spatial_mics", {}), resolve=True)
    if not isinstance(sm, dict):
        sm = {}
    sigma_regional = float(sm.get("sigma_regional", 3.0))
    sigma_fine = float(sm.get("sigma_fine", 0.7))
    block_mode = str(sm.get("block_mode", "bandpass"))
    detail_mode = str(sm.get("detail_mode", "abs"))
    block_normalize = bool(sm.get("block_normalize", True))
    block_low_pct = float(sm.get("block_low_pct", 1.0))
    block_high_pct = float(sm.get("block_high_pct", 99.0))
    block_gains_raw = sm.get("block_gains", [1.0, 1.0, 1.25])
    block_gains = tuple(float(v) for v in list(block_gains_raw))
    save_cube = bool(sm.get("save_feature_cube", False))

    pipe = getattr(cfg, "pipeline", None)
    seed = int(OmegaConf.select(pipe, "seed", default=42))
    share = bool(OmegaConf.select(pipe, "share_pixel_sampling", default=True))
    cache_raw = OmegaConf.select(pipe, "pixel_sampling_cache_path", default=None)
    cache_path: Path | None = None
    if cache_raw is not None and str(cache_raw).strip().lower() not in ("", "~", "null", "none"):
        cache_path = Path(to_absolute_path(str(cache_raw))).expanduser().resolve()

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )

    spatial_cube, feat_meta = build_spatial_mics_input(
        msi,
        valid_mask,
        sigma_regional=sigma_regional,
        sigma_fine=sigma_fine,
        block_mode=block_mode,
        detail_mode=detail_mode,
        block_normalize=block_normalize,
        block_low_pct=block_low_pct,
        block_high_pct=block_high_pct,
        block_gains=block_gains,
    )
    logger.info(
        "SpatialMiCS features: mode=%s C=%d -> %d (3x) | sigma_reg=%.2f sigma_fine=%.2f gains=%s",
        feat_meta.get("block_mode"),
        feat_meta["input_channels"],
        feat_meta["output_channels"],
        sigma_regional,
        sigma_fine,
        list(block_gains),
    )
    (out_dir / "spatial_mics_feature_meta.json").write_text(
        __import__("json").dumps(feat_meta, indent=2),
        encoding="utf-8",
    )
    if save_cube:
        np.save(out_dir / "spatial_mics_input.npy", spatial_cube.astype(np.float32))

    # Baseline: standard MiCS (1× raw spectrum), same hyperparams as spatial branch.
    baseline_input = msi.astype(np.float32, copy=False)

    spatial_epochs = int(OmegaConf.select(cfg, "model.num_epochs", default=30))
    mics_kw = _mics_kwargs_from_cfg(cfg, seed, spatial_epochs)
    pixel_cache = resolve_pixel_sampling_cache(
        spatial_cube,
        _sampling_cache_kwargs(mics_kw),
        out_dir=out_dir,
        share=share,
        cache_path=cache_path,
    )

    compare = OmegaConf.to_container(getattr(cfg, "compare_baseline", {}), resolve=True)
    if not isinstance(compare, dict):
        compare = {}
    compare_enabled = bool(compare.get("enabled", True))

    baseline_model: MSIParametricMiCSLMC | None = None
    baseline_viz: np.ndarray | None = None
    dice_baseline: float | None = None

    if compare_enabled:
        baseline_cache = resolve_pixel_sampling_cache(
            baseline_input,
            _sampling_cache_kwargs(mics_kw),
            out_dir=out_dir / "baseline_cache",
            share=share,
            cache_path=None,
        )
        baseline_model, _base_emb, baseline_viz = _train_and_predict(
            baseline_input,
            mics_kw,
            baseline_cache,
            label="baseline MiCS (mics_multiscale, 1x input)",
        )
        Image.fromarray(baseline_viz, mode="RGB").save(out_dir / "baseline_mics_viz.png")
        if baseline_model.model is not None:
            torch.save(
                {"mics_state": baseline_model.model.state_dict()},
                out_dir / "baseline_mics.pt",
            )
        del baseline_model
        gc.collect()

    spatial_model, spatial_emb, spatial_viz = _train_and_predict(
        spatial_cube,
        mics_kw,
        pixel_cache,
        label="SpatialMiCS (mics_multiscale, 3x input)",
    )
    Image.fromarray(spatial_viz, mode="RGB").save(out_dir / "spatial_mics_viz.png")
    np.save(out_dir / "spatial_mics_embedding.npy", spatial_emb.astype(np.float32))
    if spatial_model.model is not None:
        torch.save(
            {"mics_state": spatial_model.model.state_dict(), "feature_meta": feat_meta},
            out_dir / "spatial_mics.pt",
        )

    logger.info("Building HD edge map for Dice evaluation...")
    hd_edges = _build_hd_edge_map_for_train(cfg, msi).astype(np.float32)
    np.save(out_dir / "hd_edges.npy", hd_edges)

    dice_spatial = _global_continuous_dice(cfg, spatial_viz, hd_edges, valid_mask)
    logger.info("SpatialMiCS continuous Dice: %.4f", dice_spatial)

    if compare_enabled and baseline_viz is not None:
        dice_baseline = _global_continuous_dice(cfg, baseline_viz, hd_edges, valid_mask)
        logger.info("Baseline MiCS continuous Dice: %.4f", dice_baseline)
        logger.info("Dice delta (spatial - baseline): %+.4f", dice_spatial - dice_baseline)

    _save_comparison_panel(
        out_dir,
        baseline_viz=baseline_viz if compare_enabled else None,
        spatial_viz=spatial_viz,
        valid_mask=valid_mask,
        dice_baseline=dice_baseline,
        dice_spatial=dice_spatial,
    )

    summary = {
        "npy_path": str(npy_path),
        "feature_meta": feat_meta,
        "dice_spatial": dice_spatial,
        "dice_baseline": dice_baseline,
        "dice_delta": (
            None if dice_baseline is None else float(dice_spatial - dice_baseline)
        ),
    }
    (out_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info("Outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
