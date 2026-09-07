#!/usr/bin/env python3
"""
Spatial MiCS base embedding + fine-block residual stage.

  E_final = E_base + ΔE_fine

Gallery: base Spatial MiCS | |ΔE| heatmap | final combined (shared norm, equalized).

Run (repo root):
    python scripts/train_spatial_mics_residual.py
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
from msi_visual.parametric_mics_lmc import (  # noqa: E402
    MSIParametricMiCSLMC,
    _spatial_gaussian_smooth_channels,
)
from msi_visual.spatial_mics_features import (  # noqa: E402
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
)

logger = logging.getLogger(__name__)

_BG = (14, 14, 18)
_GAP = 8
_HEADER_H = 20
_TITLE_H = 22
_DICE_TAG_H = 16


def _equalize_rgb_u8_lab_l(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.equalizeHist(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def _maybe_equalize(rgb: np.ndarray, *, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.asarray(rgb, dtype=np.uint8)
    return _equalize_rgb_u8_lab_l(np.asarray(rgb, dtype=np.uint8))


def _compute_norm_anchor(
    emb: np.ndarray,
    mask: np.ndarray,
    low: float,
    high: float,
) -> list[tuple[float, float]]:
    m = np.asarray(mask, dtype=bool)
    anchor: list[tuple[float, float]] = []
    for c in range(int(emb.shape[-1])):
        vals = np.asarray(emb[..., c], dtype=np.float64)[m]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            anchor.append((0.0, 1.0))
        else:
            anchor.append((float(np.percentile(vals, low)), float(np.percentile(vals, high))))
    return anchor


def _embedding_to_rgb_with_anchor(
    emb: np.ndarray,
    mask: np.ndarray,
    anchor: list[tuple[float, float]],
    *,
    lab_to_rgb: bool,
    smooth_sigma: float,
) -> np.ndarray:
    result = np.asarray(emb, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=bool)
    for c, (p_lo, p_hi) in enumerate(anchor):
        ch = (result[..., c] - p_lo) / max(p_hi - p_lo, 1e-8)
        result[..., c] = np.clip(ch, 0.0, 1.0)
    result[~m] = 0.0
    result = _spatial_gaussian_smooth_channels(result, smooth_sigma)
    out = np.uint8(255.0 * result)
    out[~m] = 0
    if lab_to_rgb and int(emb.shape[-1]) == 3:
        out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    out[~m] = 0
    return out


def _delta_magnitude_heatmap(
    delta_emb: np.ndarray,
    mask: np.ndarray,
    *,
    gain: float = 1.0,
) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    mag = np.linalg.norm(np.asarray(delta_emb, dtype=np.float32), axis=-1)
    mag[~m] = 0.0
    v = mag[m]
    v = v[np.isfinite(v)]
    if v.size == 0:
        u8 = np.zeros(mag.shape, dtype=np.uint8)
    else:
        p99 = float(np.percentile(v, 99.0))
        scaled = np.clip(mag * float(gain) / max(p99, 1e-8), 0.0, 1.0)
        u8 = (255.0 * scaled).astype(np.uint8)
    u8[~m] = 0
    rgb = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
    rgb[~m] = 0
    return rgb


def _draw_text_bar(
    width: int,
    height: int,
    text: str,
    *,
    bg: tuple[int, int, int] = (28, 28, 36),
    fg: tuple[int, int, int] = (230, 230, 235),
    font_size: int = 11,
) -> np.ndarray:
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2 - 1), text, fill=fg, font=font)
    return np.asarray(img, dtype=np.uint8)


def _build_gallery(
    panels: list[np.ndarray],
    *,
    headers: tuple[str, ...],
    subtitle: str,
    dice_labels: list[str | None] | None = None,
    equalize_mics: bool = True,
    equalize_mask: list[bool] | None = None,
    target_h: int = 520,
) -> np.ndarray:
    if len(panels) != len(headers):
        raise ValueError("panels and headers length mismatch")

    n = len(panels)
    cell_w = 0
    resized: list[np.ndarray] = []
    for i, rgb in enumerate(panels):
        eq = equalize_mics and (equalize_mask[i] if equalize_mask else True)
        show = _maybe_equalize(rgb, enabled=eq)
        h, w = show.shape[:2]
        scale = target_h / float(h)
        tw = max(1, int(round(w * scale)))
        r = cv2.resize(show, (tw, target_h), interpolation=cv2.INTER_AREA)
        resized.append(r)
        cell_w = max(cell_w, tw)

    canvas_w = n * cell_w + (n - 1) * _GAP
    title_row = _draw_text_bar(canvas_w, _TITLE_H, subtitle, bg=(22, 22, 30), font_size=12)
    header_row = np.full((_HEADER_H, canvas_w, 3), _BG, dtype=np.uint8)
    for i, lab in enumerate(headers):
        x0 = i * (cell_w + _GAP)
        header_row[:, x0 : x0 + cell_w] = _draw_text_bar(cell_w, _HEADER_H, lab, font_size=11)

    tile_row = np.full((target_h, canvas_w, 3), _BG, dtype=np.uint8)
    for i, rgb in enumerate(resized):
        h, w = rgb.shape[:2]
        x0 = i * (cell_w + _GAP) + (cell_w - w) // 2
        tile_row[:, x0 : x0 + w] = rgb
        if dice_labels and i < len(dice_labels) and dice_labels[i]:
            tag = _draw_text_bar(
                cell_w,
                _DICE_TAG_H,
                dice_labels[i],
                bg=(20, 20, 28),
                font_size=10,
            )
            tx0 = i * (cell_w + _GAP)
            tile_row[target_h - _DICE_TAG_H : target_h, tx0 : tx0 + cell_w] = tag

    return np.vstack([title_row, header_row, tile_row])


def _build_feature_cube(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    sm: dict[str, Any],
    *,
    block_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    block_gains = tuple(float(v) for v in list(sm.get("block_gains", [1.0, 1.0, 1.25])))
    return build_spatial_mics_input(
        msi,
        valid_mask,
        sigma_regional=float(sm.get("sigma_regional", 3.0)),
        sigma_fine=float(sm.get("sigma_fine", 0.7)),
        block_mode=str(block_mode),
        detail_mode=str(sm.get("detail_mode", "abs")),
        block_normalize=bool(sm.get("block_normalize", True)),
        block_low_pct=float(sm.get("block_low_pct", 1.0)),
        block_high_pct=float(sm.get("block_high_pct", 99.0)),
        block_gains=block_gains,
    )


def _train_stage(
    img: np.ndarray,
    mics_kw: dict[str, Any],
    pixel_cache: dict[str, Any],
    *,
    label: str,
    residual_base: np.ndarray | None = None,
) -> tuple[MSIParametricMiCSLMC, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Training %s (%d epochs, C=%d)...", label, mics_kw["num_epochs"], img.shape[-1])
    model = MSIParametricMiCSLMC(**mics_kw)
    if residual_base is not None:
        model.set_residual_base(residual_base)
    model.fit(img, pixel_sampling_cache=pixel_cache)
    emb = model.predict_embedding(img)
    delta = model.predict_delta(img)
    viz = model.predict(img)
    logger.info("%s done.", label)
    return model, emb, delta, viz


@hydra.main(version_base=None, config_path="configs", config_name="train_spatial_mics_residual")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    npy_path = _resolve_npy_path(cfg)
    if not npy_path.is_file():
        raise FileNotFoundError(f"MSI not found: {npy_path}")

    transpose = bool(OmegaConf.select(cfg, "data.transpose", default=False))
    msi = load_and_preprocess_image(str(npy_path), transpose=transpose)
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    sm = OmegaConf.to_container(getattr(cfg, "spatial_mics", {}), resolve=True)
    if not isinstance(sm, dict):
        sm = {}
    rs = OmegaConf.to_container(getattr(cfg, "residual", {}), resolve=True)
    if not isinstance(rs, dict):
        rs = {}

    base_mode = str(rs.get("base_block_mode", "bandpass"))
    residual_block = str(rs.get("residual_block", "fine_contrast"))
    residual_gain = float(rs.get("residual_block_gain", 1.35))
    residual_epochs = int(rs.get("residual_epochs", 24))
    residual_beta = float(rs.get("residual_beta", 16.0))
    heatmap_gain = float(rs.get("delta_heatmap_gain", 1.15))
    equalize_viz = bool(rs.get("equalize_viz", True))
    gallery_h = int(rs.get("gallery_target_h", 520))

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

    base_cube, base_meta = _build_feature_cube(msi, valid_mask, sm, block_mode=base_mode)
    residual_cube, residual_meta = build_spatial_mics_single_block(
        msi,
        valid_mask,
        block=residual_block,
        sigma_regional=float(sm.get("sigma_regional", 3.0)),
        sigma_fine=float(sm.get("sigma_fine", 0.7)),
        detail_mode=str(sm.get("detail_mode", "abs")),
        block_normalize=bool(sm.get("block_normalize", True)),
        block_low_pct=float(sm.get("block_low_pct", 1.0)),
        block_high_pct=float(sm.get("block_high_pct", 99.0)),
        block_gain=residual_gain,
    )
    logger.info("Base cube (%s): %d ch", base_mode, base_cube.shape[-1])
    logger.info("Residual cube (%s): %d ch", residual_block, residual_cube.shape[-1])

    base_epochs = int(OmegaConf.select(cfg, "model.num_epochs", default=30))
    base_kw = _mics_kwargs_from_cfg(cfg, seed, base_epochs)
    residual_kw = dict(base_kw)
    residual_kw["num_epochs"] = residual_epochs
    residual_kw["random_state"] = int(seed) + 1
    residual_kw["beta"] = residual_beta

    base_cache = resolve_pixel_sampling_cache(
        base_cube,
        _sampling_cache_kwargs(base_kw),
        out_dir=out_dir / "base_cache",
        share=share,
        cache_path=cache_path,
    )
    residual_cache = resolve_pixel_sampling_cache(
        residual_cube,
        _sampling_cache_kwargs(residual_kw),
        out_dir=out_dir / "residual_cache",
        share=share,
        cache_path=None,
    )

    base_model, base_emb, _base_delta, _base_viz = _train_stage(
        base_cube,
        base_kw,
        base_cache,
        label=f"base SpatialMiCS ({base_mode})",
    )
    np.save(out_dir / "base_embedding.npy", base_emb.astype(np.float32))

    residual_model, final_emb, delta_emb, _final_viz = _train_stage(
        residual_cube,
        residual_kw,
        residual_cache,
        label=f"residual on {residual_block}",
        residual_base=base_emb,
    )
    np.save(out_dir / "delta_embedding.npy", delta_emb.astype(np.float32))
    np.save(out_dir / "final_embedding.npy", final_emb.astype(np.float32))

    pred_lo = float(base_kw.get("predict_percentile_low", 1.0))
    pred_hi = float(base_kw.get("predict_percentile_high", 99.0))
    lab_to_rgb = bool(base_kw.get("lab_to_rgb", True))
    smooth_sigma = float(base_kw.get("predict_spatial_smooth_sigma", 0.0))
    anchor = _compute_norm_anchor(final_emb, valid_mask, pred_lo, pred_hi)

    base_rgb_anchor = _embedding_to_rgb_with_anchor(
        base_emb, valid_mask, anchor, lab_to_rgb=lab_to_rgb, smooth_sigma=smooth_sigma
    )
    final_rgb_anchor = _embedding_to_rgb_with_anchor(
        final_emb, valid_mask, anchor, lab_to_rgb=lab_to_rgb, smooth_sigma=smooth_sigma
    )
    delta_rgb = _delta_magnitude_heatmap(delta_emb, valid_mask, gain=heatmap_gain)
    Image.fromarray(base_rgb_anchor).save(out_dir / "viz_base.png")
    Image.fromarray(final_rgb_anchor).save(out_dir / "viz_final.png")

    logger.info("Building HD edge map for Dice evaluation...")
    hd_edges = _build_hd_edge_map_for_train(cfg, msi).astype(np.float32)
    dice_base = _global_continuous_dice(
        cfg, _maybe_equalize(base_rgb_anchor, enabled=equalize_viz), hd_edges, valid_mask
    )
    dice_final = _global_continuous_dice(
        cfg, _maybe_equalize(final_rgb_anchor, enabled=equalize_viz), hd_edges, valid_mask
    )
    logger.info("Dice (shared anchor, equalized): base=%.4f final=%.4f delta=%+.4f", dice_base, dice_final, dice_final - dice_base)

    slide = npy_path.stem
    subtitle = (
        f"Spatial MiCS + embedding residual | {slide} | "
        f"base={base_mode}  residual={residual_block}  epochs={base_epochs}+{residual_epochs}"
    )
    headers = (
        f"Base Spatial MiCS ({base_mode})",
        f"|ΔE| ({residual_block})",
        "Base + residual",
    )
    dice_tags = [
        f"dice={dice_base:.3f}",
        None,
        f"dice={dice_final:.3f}",
    ]
    gallery = _build_gallery(
        [base_rgb_anchor, delta_rgb, final_rgb_anchor],
        headers=headers,
        subtitle=subtitle,
        dice_labels=dice_tags,
        equalize_mics=equalize_viz,
        equalize_mask=[True, False, True],
        target_h=gallery_h,
    )
    gallery_path = out_dir / "gallery_spatial_mics_residual.png"
    Image.fromarray(gallery).save(gallery_path, dpi=(150, 150))

    # Matplotlib version for slides.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=150)
    panel_data = [
        (base_rgb_anchor, headers[0], dice_tags[0], True),
        (delta_rgb, headers[1], None, False),
        (final_rgb_anchor, headers[2], dice_tags[2], True),
    ]
    for ax, (rgb, title, dice, do_eq) in zip(axes, panel_data):
        show = _maybe_equalize(rgb.copy(), enabled=equalize_viz and do_eq)
        show[~valid_mask] = 0
        ax.imshow(show)
        t = title.replace("\n", " ")
        if dice:
            t = f"{t}\n{dice}"
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "gallery_spatial_mics_residual_panel.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "npy_path": str(npy_path),
        "base_meta": base_meta,
        "residual_meta": residual_meta,
        "residual_epochs": residual_epochs,
        "residual_beta": residual_beta,
        "dice_base": dice_base,
        "dice_final": dice_final,
        "dice_delta": float(dice_final - dice_base),
        "delta_embedding_rms": float(
            np.sqrt(np.mean(np.sum(delta_emb[valid_mask] ** 2, axis=-1)))
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Gallery -> %s", gallery_path)

    del base_model, residual_model
    gc.collect()


if __name__ == "__main__":
    main()
