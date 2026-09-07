#!/usr/bin/env python3
"""
Train coarse + fine Spatial MiCS experts, fuse cluster α with product-of-experts,
and write a comparison gallery.

Run (repo root):
    python scripts/train_spatial_mics_poe.py
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
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from msi_visual.spatial_mics_features import build_spatial_mics_input  # noqa: E402
from msi_visual.spatial_mics_poe import (  # noqa: E402
    align_cluster_alpha,
    alpha_to_soft_rgb_u8,
    product_of_experts_alpha,
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
_HEADER_H = 18
_GAP = 6


def _equalize_rgb_u8_lab_l(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.equalizeHist(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def _maybe_equalize(rgb: np.ndarray, *, enabled: bool) -> np.ndarray:
    if not enabled:
        return rgb
    return _equalize_rgb_u8_lab_l(rgb)


def _resolve_level_idx(model: MSIParametricMiCSLMC, level_idx: int) -> int:
    n = len(model.visualiation_to_cluster)
    idx = int(level_idx)
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        raise IndexError(f"cluster_level_idx {level_idx} out of range [0, {n})")
    return idx


def _draw_header_bar(width: int, labels: tuple[str, ...]) -> np.ndarray:
    n = len(labels)
    cell_w = (width - (n - 1) * _GAP) // n
    row = np.full((_HEADER_H, width, 3), _BG, dtype=np.uint8)
    for i, lab in enumerate(labels):
        x0 = i * (cell_w + _GAP)
        img = Image.new("RGB", (cell_w, _HEADER_H), (28, 28, 36))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), lab, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((cell_w - tw) // 2, (_HEADER_H - th) // 2 - 1), lab, fill=(230, 230, 235), font=font)
        row[:, x0 : x0 + cell_w] = np.asarray(img, dtype=np.uint8)
    return row


def _build_gallery_row(
    panels: list[np.ndarray],
    *,
    labels: tuple[str, ...],
    dice_labels: list[str | None] | None = None,
    equalize: bool = True,
    target_h: int = 480,
) -> np.ndarray:
    if len(panels) != len(labels):
        raise ValueError("panels and labels length mismatch")

    proc: list[np.ndarray] = []
    for rgb in panels:
        show = _maybe_equalize(np.asarray(rgb, dtype=np.uint8), enabled=equalize)
        h, w = show.shape[:2]
        scale = target_h / float(h)
        tw = max(1, int(round(w * scale)))
        proc.append(cv2.resize(show, (tw, target_h), interpolation=cv2.INTER_AREA))

    cell_w = max(p.shape[1] for p in proc)
    canvas_w = len(proc) * cell_w + (len(proc) - 1) * _GAP
    tile_row = np.full((target_h, canvas_w, 3), _BG, dtype=np.uint8)

    for i, rgb in enumerate(proc):
        h, w = rgb.shape[:2]
        x0 = i * (cell_w + _GAP) + (cell_w - w) // 2
        tile_row[:, x0 : x0 + w] = rgb
        if dice_labels and i < len(dice_labels) and dice_labels[i]:
            tag_h = 16
            tag = Image.new("RGB", (cell_w, tag_h), (20, 20, 28))
            draw = ImageDraw.Draw(tag)
            try:
                font = ImageFont.truetype("arial.ttf", 10)
            except OSError:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), dice_labels[i], font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((cell_w - tw) // 2, 2), dice_labels[i], fill=(220, 220, 225), font=font)
            tx0 = i * (cell_w + _GAP)
            tile_row[target_h - tag_h : target_h, tx0 : tx0 + cell_w] = np.asarray(tag, dtype=np.uint8)

    header = _draw_header_bar(canvas_w, labels)
    return np.vstack([header, tile_row])


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


@hydra.main(version_base=None, config_path="configs", config_name="train_spatial_mics_poe")
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
    poe = OmegaConf.to_container(getattr(cfg, "poe", {}), resolve=True)
    if not isinstance(poe, dict):
        poe = {}

    coarse_mode = str(poe.get("coarse_block_mode", "context"))
    fine_mode = str(poe.get("fine_block_mode", "bandpass"))
    w_coarse = float(poe.get("w_coarse", 0.45))
    w_fine = float(poe.get("w_fine", 0.55))
    level_idx_cfg = int(poe.get("cluster_level_idx", -1))
    equalize_viz = bool(poe.get("equalize_viz", True))

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

    coarse_cube, coarse_meta = _build_feature_cube(msi, valid_mask, sm, block_mode=coarse_mode)
    fine_cube, fine_meta = _build_feature_cube(msi, valid_mask, sm, block_mode=fine_mode)
    logger.info("Coarse expert (%s): C=%d", coarse_mode, coarse_cube.shape[-1])
    logger.info("Fine expert (%s): C=%d", fine_mode, fine_cube.shape[-1])

    spatial_epochs = int(OmegaConf.select(cfg, "model.num_epochs", default=30))
    mics_kw = _mics_kwargs_from_cfg(cfg, seed, spatial_epochs)

    coarse_cache = resolve_pixel_sampling_cache(
        coarse_cube,
        _sampling_cache_kwargs(mics_kw),
        out_dir=out_dir / "coarse_cache",
        share=share,
        cache_path=cache_path,
    )
    fine_cache = resolve_pixel_sampling_cache(
        fine_cube,
        _sampling_cache_kwargs(mics_kw),
        out_dir=out_dir / "fine_cache",
        share=share,
        cache_path=None,
    )

    coarse_model, _coarse_emb, coarse_embed_viz = _train_and_predict(
        coarse_cube,
        mics_kw,
        coarse_cache,
        label=f"coarse SpatialMiCS ({coarse_mode})",
    )
    fine_model, _fine_emb, fine_embed_viz = _train_and_predict(
        fine_cube,
        mics_kw,
        fine_cache,
        label=f"fine SpatialMiCS ({fine_mode})",
    )

    level_idx = _resolve_level_idx(coarse_model, level_idx_cfg)
    k_clusters = int(coarse_model.clusters[level_idx])
    logger.info(
        "Cluster level %d (K=%d) | PoE weights coarse=%.3f fine=%.3f",
        level_idx,
        k_clusters,
        w_coarse,
        w_fine,
    )

    alpha_coarse = coarse_model.predict_cluster_soft_probs(coarse_cube, level_idx=level_idx)
    alpha_fine = fine_model.predict_cluster_soft_probs(fine_cube, level_idx=level_idx)

    perm, alpha_fine_aligned, align_meta = align_cluster_alpha(alpha_coarse, alpha_fine, valid_mask)
    alpha_poe = product_of_experts_alpha(
        alpha_coarse,
        alpha_fine_aligned,
        valid_mask,
        w_a=w_coarse,
        w_b=w_fine,
    )

    rgb_coarse = alpha_to_soft_rgb_u8(alpha_coarse, valid_mask)
    rgb_fine = alpha_to_soft_rgb_u8(alpha_fine, valid_mask)
    rgb_poe = alpha_to_soft_rgb_u8(alpha_poe, valid_mask)

    np.save(out_dir / "alpha_coarse.npy", alpha_coarse.astype(np.float32))
    np.save(out_dir / "alpha_fine.npy", alpha_fine.astype(np.float32))
    np.save(out_dir / "alpha_fine_aligned.npy", alpha_fine_aligned.astype(np.float32))
    np.save(out_dir / "alpha_poe.npy", alpha_poe.astype(np.float32))
    Image.fromarray(rgb_coarse).save(out_dir / "viz_coarse_soft.png")
    Image.fromarray(rgb_fine).save(out_dir / "viz_fine_soft.png")
    Image.fromarray(rgb_poe).save(out_dir / "viz_poe_soft.png")
    Image.fromarray(coarse_embed_viz).save(out_dir / "viz_coarse_embedding.png")
    Image.fromarray(fine_embed_viz).save(out_dir / "viz_fine_embedding.png")

    logger.info("Building HD edge map for Dice evaluation...")
    hd_edges = _build_hd_edge_map_for_train(cfg, msi).astype(np.float32)
    dice_coarse = _global_continuous_dice(cfg, rgb_coarse, hd_edges, valid_mask)
    dice_fine = _global_continuous_dice(cfg, rgb_fine, hd_edges, valid_mask)
    dice_poe = _global_continuous_dice(cfg, rgb_poe, hd_edges, valid_mask)
    logger.info("Dice coarse=%.4f fine=%.4f PoE=%.4f", dice_coarse, dice_fine, dice_poe)

    panel_labels = (
        f"Coarse ({coarse_mode})\nK={k_clusters}",
        f"Fine ({fine_mode})\nK={k_clusters}",
        f"PoE α fusion\nw={w_coarse:.2f}/{w_fine:.2f}",
    )
    dice_tags = [
        f"dice={dice_coarse:.3f}",
        f"dice={dice_fine:.3f}",
        f"dice={dice_poe:.3f}",
    ]
    gallery_soft = _build_gallery_row(
        [rgb_coarse, rgb_fine, rgb_poe],
        labels=panel_labels,
        dice_labels=dice_tags,
        equalize=equalize_viz,
    )
    Image.fromarray(gallery_soft).save(out_dir / "gallery_poe_experts_soft.png", dpi=(150, 150))

    gallery_embed = _build_gallery_row(
        [coarse_embed_viz, fine_embed_viz, rgb_poe],
        labels=(
            f"Coarse embedding\n({coarse_mode})",
            f"Fine embedding\n({fine_mode})",
            f"PoE soft RGB\nw={w_coarse:.2f}/{w_fine:.2f}",
        ),
        dice_labels=[None, None, f"dice={dice_poe:.3f}"],
        equalize=equalize_viz,
    )
    Image.fromarray(gallery_embed).save(out_dir / "gallery_poe_experts_mixed.png", dpi=(150, 150))

    # Matplotlib version with titles for slides.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=150)
    for ax, rgb, title, dice in zip(
        axes,
        [rgb_coarse, rgb_fine, rgb_poe],
        panel_labels,
        dice_tags,
    ):
        show = _maybe_equalize(rgb.copy(), enabled=equalize_viz)
        show[~valid_mask] = 0
        ax.imshow(show)
        ax.set_title(f"{title.replace(chr(10), ' ')}\n{dice}", fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"Spatial MiCS PoE | align mean r={align_meta['mean_aligned_correlation']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "gallery_poe_experts_panel.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "npy_path": str(npy_path),
        "coarse_meta": coarse_meta,
        "fine_meta": fine_meta,
        "cluster_level_idx": level_idx,
        "k_clusters": k_clusters,
        "poe_weights": {"coarse": w_coarse, "fine": w_fine},
        "alignment": {
            "perm_other_to_ref": align_meta["perm_other_to_ref"],
            "mean_aligned_correlation": align_meta["mean_aligned_correlation"],
        },
        "dice_coarse_soft": dice_coarse,
        "dice_fine_soft": dice_fine,
        "dice_poe": dice_poe,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Outputs written to %s", out_dir)

    del coarse_model, fine_model
    gc.collect()


if __name__ == "__main__":
    main()
