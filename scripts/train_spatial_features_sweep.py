#!/usr/bin/env python3
"""
Train MiCS on four spatial input recipes and build a comparison gallery.

Variants (no U-Net):
  1. Raw MiCS (1x)
  2. Blur-only: blur(X, sigma_regional)
  3. Log-ratio: log X - log blur(X, sigma_regional)
  4. Context SpatialMiCS (3x): [X, blur_reg, fine_detail]

Run (repo root):
    python scripts/train_spatial_features_sweep.py
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from msi_visual.spatial_mics_features import (  # noqa: E402
    build_log_ratio_mics_input,
    build_spatial_mics_input,
    build_spatial_mics_single_block,
)
from train_parametric_mics_lmc import (  # noqa: E402
    _build_hd_edge_map_for_train,
    load_and_preprocess_image,
)
from train_spatial_mics import (  # noqa: E402
    _global_continuous_dice,
    _mics_kwargs_from_cfg,
    _resolve_npy_path,
    _sampling_cache_kwargs,
    _train_and_predict,
)

logger = logging.getLogger(__name__)

_BG = (14, 14, 18)
_GAP = 6
_HEADER_H = 18
_TITLE_H = 20
_DICE_TAG_H = 14


@dataclass
class VariantResult:
    key: str
    label: str
    meta: dict[str, Any]
    viz: np.ndarray
    dice: float


def _equalize_rgb_u8_lab_l(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.equalizeHist(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def _draw_bar(width: int, height: int, text: str, *, font_size: int = 10) -> np.ndarray:
    img = Image.new("RGB", (width, height), (28, 28, 36))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, max(0, (height - th) // 2 - 1)), text, fill=(230, 230, 235), font=font)
    return np.asarray(img, dtype=np.uint8)


def _build_gallery(
    results: list[VariantResult],
    *,
    subtitle: str,
    equalize: bool,
    target_h: int,
) -> np.ndarray:
    resized: list[np.ndarray] = []
    cell_w = 0
    for r in results:
        show = _equalize_rgb_u8_lab_l(r.viz) if equalize else np.asarray(r.viz, dtype=np.uint8)
        h, w = show.shape[:2]
        tw = max(1, int(round(w * target_h / float(h))))
        resized.append(cv2.resize(show, (tw, target_h), interpolation=cv2.INTER_AREA))
        cell_w = max(cell_w, tw)

    n = len(results)
    canvas_w = n * cell_w + (n - 1) * _GAP
    title_row = _draw_bar(canvas_w, _TITLE_H, subtitle, font_size=11)
    header_row = np.full((_HEADER_H, canvas_w, 3), _BG, dtype=np.uint8)
    for i, r in enumerate(results):
        x0 = i * (cell_w + _GAP)
        header_row[:, x0 : x0 + cell_w] = _draw_bar(cell_w, _HEADER_H, r.label, font_size=9)

    tile_row = np.full((target_h, canvas_w, 3), _BG, dtype=np.uint8)
    for i, rgb in enumerate(resized):
        h, w = rgb.shape[:2]
        x0 = i * (cell_w + _GAP) + (cell_w - w) // 2
        tile_row[:, x0 : x0 + w] = rgb
        tag = f"dice={results[i].dice:.3f}"
        tx0 = i * (cell_w + _GAP)
        tile_row[target_h - _DICE_TAG_H : target_h, tx0 : tx0 + cell_w] = _draw_bar(
            cell_w, _DICE_TAG_H, tag, font_size=9
        )
    return np.vstack([title_row, header_row, tile_row])


def _sm_dict(cfg: DictConfig) -> dict[str, Any]:
    sm = OmegaConf.to_container(getattr(cfg, "spatial_mics", {}), resolve=True)
    return sm if isinstance(sm, dict) else {}


def _build_variants(
    msi: np.ndarray,
    mask: np.ndarray,
    sm: dict[str, Any],
) -> list[tuple[str, str, np.ndarray, dict[str, Any]]]:
    sigma_reg = float(sm.get("sigma_regional", 3.0))
    sigma_fine = float(sm.get("sigma_fine", 0.7))
    block_gains = tuple(float(v) for v in list(sm.get("block_gains", [1.0, 1.0, 1.25])))
    norm_kw = dict(
        block_normalize=bool(sm.get("block_normalize", True)),
        block_low_pct=float(sm.get("block_low_pct", 1.0)),
        block_high_pct=float(sm.get("block_high_pct", 99.0)),
    )

    raw = msi.astype(np.float32, copy=False)
    blur, blur_meta = build_spatial_mics_single_block(
        msi,
        mask,
        block="regional_blur",
        sigma_regional=sigma_reg,
        sigma_fine=sigma_fine,
        detail_mode=str(sm.get("detail_mode", "abs")),
        block_gain=float(sm.get("blur_only_gain", 1.0)),
        **norm_kw,
    )
    log_ratio, lr_meta = build_log_ratio_mics_input(
        msi,
        mask,
        sigma=sigma_reg,
        clip_log_ratio=sm.get("log_ratio_clip", 3.0),
        block_gain=float(sm.get("log_ratio_gain", 1.0)),
        **norm_kw,
    )
    context, ctx_meta = build_spatial_mics_input(
        msi,
        mask,
        sigma_regional=sigma_reg,
        sigma_fine=sigma_fine,
        block_mode="context",
        detail_mode=str(sm.get("detail_mode", "abs")),
        block_gains=block_gains,
        **norm_kw,
    )

    return [
        ("raw", "Raw MiCS (1x)", raw, {"feature_type": "raw", "output_channels": int(raw.shape[-1])}),
        ("blur_only", f"Blur only (sigma={sigma_reg:g})", blur, blur_meta),
        ("log_ratio", f"Log-ratio (sigma={sigma_reg:g})", log_ratio, lr_meta),
        ("context_3x", "Context SpatialMiCS (3x)", context, ctx_meta),
    ]


@hydra.main(version_base=None, config_path="configs", config_name="train_spatial_features_sweep")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    npy_path = _resolve_npy_path(cfg)
    if not npy_path.is_file():
        raise FileNotFoundError(f"MSI not found: {npy_path}")

    msi = load_and_preprocess_image(str(npy_path), transpose=bool(cfg.data.transpose))
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty valid mask.")

    sm = _sm_dict(cfg)
    equalize = bool(sm.get("equalize_viz", True))
    gallery_h = int(sm.get("gallery_target_h", 440))

    pipe = getattr(cfg, "pipeline", None)
    seed = int(OmegaConf.select(pipe, "seed", default=42))
    share = bool(OmegaConf.select(pipe, "share_pixel_sampling", default=True))
    epochs = int(OmegaConf.select(cfg, "model.num_epochs", default=30))
    base_mics_kw = _mics_kwargs_from_cfg(cfg, seed, epochs)

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    variants = _build_variants(msi, valid_mask, sm)
    logger.info("Building HD edge map for Dice evaluation...")
    hd_edges = _build_hd_edge_map_for_train(cfg, msi).astype(np.float32)

    results: list[VariantResult] = []
    for idx, (key, label, cube, meta) in enumerate(variants):
        logger.info("Variant %s: %s | C=%d", key, label, cube.shape[-1])
        mics_kw = dict(base_mics_kw)
        mics_kw["random_state"] = int(seed) + idx
        cache = resolve_pixel_sampling_cache(
            cube,
            _sampling_cache_kwargs(mics_kw),
            out_dir=out_dir / f"cache_{key}",
            share=share,
            cache_path=None,
        )
        model, _emb, viz = _train_and_predict(cube, mics_kw, cache, label=label)
        dice = _global_continuous_dice(
            cfg,
            _equalize_rgb_u8_lab_l(viz) if equalize else viz,
            hd_edges,
            valid_mask,
        )
        Image.fromarray(viz).save(out_dir / f"viz_{key}.png")
        np.save(out_dir / f"embedding_{key}.npy", _emb.astype(np.float32))
        (out_dir / f"meta_{key}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        results.append(VariantResult(key=key, label=label, meta=meta, viz=viz, dice=dice))
        logger.info("%s Dice=%.4f", key, dice)
        try:
            model.release_resources()
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    slide = npy_path.stem
    best = max(results, key=lambda r: r.dice)
    subtitle = (
        f"Spatial feature sweep | {slide} | best={best.key} dice={best.dice:.3f}"
    )
    gallery = _build_gallery(results, subtitle=subtitle, equalize=equalize, target_h=gallery_h)
    gallery_path = out_dir / "gallery_spatial_features_sweep.png"
    Image.fromarray(gallery).save(gallery_path, dpi=(150, 150))

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 5.2), dpi=150)
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        show = _equalize_rgb_u8_lab_l(r.viz.copy()) if equalize else r.viz.copy()
        show[~valid_mask] = 0
        ax.imshow(show)
        ax.set_title(f"{r.label}\ndice={r.dice:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(subtitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "gallery_spatial_features_sweep_panel.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "npy_path": str(npy_path),
        "variants": [
            {"key": r.key, "label": r.label, "dice": r.dice, "meta": r.meta} for r in results
        ],
        "best_key": best.key,
        "best_dice": best.dice,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Gallery -> %s", gallery_path)


if __name__ == "__main__":
    main()
