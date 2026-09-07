#!/usr/bin/env python3
"""
Train MiCS on mask-aware local z-score spectra and compare to raw baseline.

Run (repo root):
    python scripts/train_local_zscore_mics.py
    python scripts/train_local_zscore_mics.py local_zscore.sigma=3.0
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

import cv2
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
from msi_visual.spatial_mics_features import build_local_zscore_mics_input  # noqa: E402
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


def _draw_bar(width: int, height: int, text: str, *, font_size: int = 11) -> np.ndarray:
    img = Image.new("RGB", (width, height), (28, 28, 36))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2 - 1), text, fill=(230, 230, 235), font=font)
    return np.asarray(img, dtype=np.uint8)


def _build_gallery(
    panels: list[np.ndarray],
    *,
    headers: tuple[str, ...],
    subtitle: str,
    dice_labels: list[str | None],
    equalize: bool,
    target_h: int,
) -> np.ndarray:
    resized: list[np.ndarray] = []
    cell_w = 0
    for rgb in panels:
        show = _equalize_rgb_u8_lab_l(rgb) if equalize else np.asarray(rgb, dtype=np.uint8)
        h, w = show.shape[:2]
        tw = max(1, int(round(w * target_h / float(h))))
        r = cv2.resize(show, (tw, target_h), interpolation=cv2.INTER_AREA)
        resized.append(r)
        cell_w = max(cell_w, tw)

    n = len(panels)
    canvas_w = n * cell_w + (n - 1) * _GAP
    title_row = _draw_bar(canvas_w, _TITLE_H, subtitle, font_size=12)
    header_row = np.full((_HEADER_H, canvas_w, 3), _BG, dtype=np.uint8)
    for i, lab in enumerate(headers):
        x0 = i * (cell_w + _GAP)
        header_row[:, x0 : x0 + cell_w] = _draw_bar(cell_w, _HEADER_H, lab, font_size=11)

    tile_row = np.full((target_h, canvas_w, 3), _BG, dtype=np.uint8)
    for i, rgb in enumerate(resized):
        h, w = rgb.shape[:2]
        x0 = i * (cell_w + _GAP) + (cell_w - w) // 2
        tile_row[:, x0 : x0 + w] = rgb
        if dice_labels[i]:
            tx0 = i * (cell_w + _GAP)
            tile_row[target_h - _DICE_TAG_H : target_h, tx0 : tx0 + cell_w] = _draw_bar(
                cell_w, _DICE_TAG_H, dice_labels[i], font_size=10
            )
    return np.vstack([title_row, header_row, tile_row])


@hydra.main(version_base=None, config_path="configs", config_name="train_local_zscore_mics")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    npy_path = _resolve_npy_path(cfg)
    if not npy_path.is_file():
        raise FileNotFoundError(f"MSI not found: {npy_path}")

    msi = load_and_preprocess_image(str(npy_path), transpose=bool(cfg.data.transpose))
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty valid mask.")

    lz = OmegaConf.to_container(getattr(cfg, "local_zscore", {}), resolve=True)
    if not isinstance(lz, dict):
        lz = {}
    sigma = float(lz.get("sigma", 0.7))
    clip_z_raw = lz.get("clip_z", 5.0)
    clip_z = None if clip_z_raw is None else float(clip_z_raw)
    equalize = bool(lz.get("equalize_viz", True))
    gallery_h = int(lz.get("gallery_target_h", 520))

    z_cube, z_meta = build_local_zscore_mics_input(
        msi,
        valid_mask,
        sigma=sigma,
        eps=float(lz.get("eps", 1e-6)),
        clip_z=clip_z,
        block_normalize=bool(lz.get("block_normalize", True)),
        block_low_pct=float(lz.get("block_low_pct", 1.0)),
        block_high_pct=float(lz.get("block_high_pct", 99.0)),
        block_gain=float(lz.get("block_gain", 1.0)),
    )
    logger.info("Local z-score features: sigma=%.2f clip=%s C=%d", sigma, clip_z, z_cube.shape[-1])

    pipe = getattr(cfg, "pipeline", None)
    seed = int(OmegaConf.select(pipe, "seed", default=42))
    share = bool(OmegaConf.select(pipe, "share_pixel_sampling", default=True))
    epochs = int(OmegaConf.select(cfg, "model.num_epochs", default=30))
    mics_kw = _mics_kwargs_from_cfg(cfg, seed, epochs)

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (out_dir / "local_zscore_feature_meta.json").write_text(json.dumps(z_meta, indent=2), encoding="utf-8")

    z_cache = resolve_pixel_sampling_cache(
        z_cube,
        _sampling_cache_kwargs(mics_kw),
        out_dir=out_dir / "zscore_cache",
        share=share,
        cache_path=None,
    )
    _, _, z_viz = _train_and_predict(
        z_cube,
        mics_kw,
        z_cache,
        label=f"local z-score MiCS (sigma={sigma})",
    )
    Image.fromarray(z_viz).save(out_dir / "viz_local_zscore.png")

    compare = OmegaConf.to_container(getattr(cfg, "compare_baseline", {}), resolve=True)
    baseline_enabled = isinstance(compare, dict) and bool(compare.get("enabled", True))
    base_viz: np.ndarray | None = None
    dice_base: float | None = None

    if baseline_enabled:
        baseline_input = msi.astype(np.float32, copy=False)
        base_cache = resolve_pixel_sampling_cache(
            baseline_input,
            _sampling_cache_kwargs(mics_kw),
            out_dir=out_dir / "baseline_cache",
            share=share,
            cache_path=None,
        )
        _, _, base_viz = _train_and_predict(
            baseline_input,
            mics_kw,
            base_cache,
            label="raw MiCS baseline (1×)",
        )
        Image.fromarray(base_viz).save(out_dir / "viz_baseline.png")

    logger.info("Building HD edge map for Dice evaluation...")
    hd_edges = _build_hd_edge_map_for_train(cfg, msi).astype(np.float32)
    dice_z = _global_continuous_dice(cfg, z_viz, hd_edges, valid_mask)
    if baseline_enabled and base_viz is not None:
        dice_base = _global_continuous_dice(cfg, base_viz, hd_edges, valid_mask)
        logger.info("Dice baseline=%.4f local_zscore=%.4f delta=%+.4f", dice_base, dice_z, dice_z - dice_base)
    else:
        logger.info("Dice local_zscore=%.4f", dice_z)

    slide = npy_path.stem
    if baseline_enabled and base_viz is not None:
        headers = ("Raw MiCS (1×)", f"Local Z-Score MiCS (σ={sigma})")
        panels = [base_viz, z_viz]
        dice_labels = [
            f"dice={dice_base:.3f}" if dice_base is not None else None,
            f"dice={dice_z:.3f}",
        ]
    else:
        headers = (f"Local Z-Score MiCS (σ={sigma})",)
        panels = [z_viz]
        dice_labels = [f"dice={dice_z:.3f}"]

    subtitle = f"Local z-score MiCS | {slide} | z=(X−μ_local)/σ_local, σ={sigma}"
    gallery = _build_gallery(
        panels,
        headers=headers,
        subtitle=subtitle,
        dice_labels=dice_labels,
        equalize=equalize,
        target_h=gallery_h,
    )
    gallery_path = out_dir / "gallery_local_zscore_mics.png"
    Image.fromarray(gallery).save(gallery_path, dpi=(150, 150))

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.8), dpi=150, squeeze=False)
    for ax, rgb, title, dice in zip(axes[0], panels, headers, dice_labels):
        show = _equalize_rgb_u8_lab_l(rgb.copy()) if equalize else rgb.copy()
        show[~valid_mask] = 0
        ax.imshow(show)
        t = title
        if dice:
            t += f"\n{dice}"
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "gallery_local_zscore_mics_panel.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "npy_path": str(npy_path),
        "feature_meta": z_meta,
        "dice_local_zscore": dice_z,
        "dice_baseline": dice_base,
        "dice_delta": None if dice_base is None else float(dice_z - dice_base),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Gallery -> %s", gallery_path)
    gc.collect()


if __name__ == "__main__":
    main()
