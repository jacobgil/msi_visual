#!/usr/bin/env python3
"""
MiCS cross-inference gallery for a fixed set of MSI cubes (e.g. four serial sections).

Rows/columns follow ``gallery.slice_order`` (default **hemisphere** for NRL4485). Optional
``gallery.compare_methods`` adds **pUMAP** and **PCA** blocks stacked with **pMiCS** in one mosaic
(default ``gallery.method_layout: vertical`` = one row per method).
"""

from __future__ import annotations

import csv
import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from msi_visual.pca_3d import PCA3D  # noqa: E402
from msi_visual.parametric_umap import UMAPVirtualStain  # noqa: E402
from umap_kwargs import umap_kwargs as _umap_kwargs  # noqa: E402
from view_annotation_boundaries import apply_annotation_spatial_transform  # noqa: E402

logger = logging.getLogger(__name__)

# NRL4485-s2 hemisphere position by file stem (view_annotation_boundaries / umap_annotations).
_NRL4485_HEMISPHERE_BY_STEM: dict[str, int] = {"0": 1, "1": 2, "2": 0, "3": 3}

METHOD_LABELS: dict[str, str] = {
    "mics": "pMiCS",
    "parametric_umap": "pUMAP",
    "pca": "PCA",
}


@dataclass
class TrainRowTiming:
    method: str
    train_stem: str
    train_hemisphere: int
    train_seconds: float
    infer_stems: list[str] = field(default_factory=list)
    infer_seconds: list[float] = field(default_factory=list)

    @property
    def infer_total_seconds(self) -> float:
        return float(sum(self.infer_seconds))

    @property
    def total_seconds(self) -> float:
        return self.train_seconds + self.infer_total_seconds

_MICS_KEYS: tuple[str, ...] = (
    "number_of_points",
    "sampling",
    "num_epochs",
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
    "clusters_auto_tune",
)


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    s = msi.sum(axis=-1, keepdims=True)
    return msi / (s + 1e-8)


def _resolve_msi_layout_cfg(ga: Any) -> tuple[bool, str]:
    """
    Layout flags (same two-step pipeline as ``view_annotations_on_methods``):

    1. ``transpose_msi`` — C×H×W → H×W×C via ``(1, 2, 0)`` (``debug_edge_maps`` / benchmark).
    2. ``transpose_spatial_mode`` — spatial fix on H×W×C:
       ``chain_flip`` = ``transpose().transpose(1,2,0)[::-1,:,:]`` (``train_parametric_mics_lmc``,
       notebooks, ``view_annotations_on_methods``); ``swap_hw`` = ``(1,0,2)``.
    """
    raw_sp = getattr(ga, "transpose_spatial_mode", None)
    spatial = "none" if raw_sp is None else str(raw_sp).strip().lower()

    raw_tm = getattr(ga, "transpose_msi", False)
    cxhxw = False
    if isinstance(raw_tm, str):
        tm = raw_tm.strip().lower()
        if tm in ("chain_flip", "chain", "notebook", "umap", "transpose_chain"):
            spatial = "chain_flip"
        elif tm in ("swap_hw", "swap"):
            spatial = "swap_hw"
        elif tm in ("true", "1", "yes", "cxhxw"):
            cxhxw = True
        elif tm not in ("false", "0", "no", "none", "auto", ""):
            raise ValueError(
                f"gallery.transpose_msi must be bool or layout alias; got {raw_tm!r}. "
                "Use gallery.transpose_spatial_mode=chain_flip for the notebook transpose."
            )
    else:
        cxhxw = bool(raw_tm)

    return cxhxw, spatial


def _load_msi(
    path: Path,
    *,
    transpose_msi: bool,
    transpose_spatial_mode: str,
    tic_normalize: bool,
) -> np.ndarray:
    img = np.load(str(path), mmap_mode=None)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    elif img.ndim != 3:
        raise ValueError(f"Expected MSI H×W×C (or C×H×W with transpose_msi), got {img.shape} for {path}")

    on_disk = tuple(int(x) for x in img.shape)
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
        logger.info("%s transpose_msi (1,2,0): %s → %s", path.name, on_disk, tuple(img.shape))

    spatial = str(transpose_spatial_mode or "none").strip().lower()
    if spatial not in ("none", "off", "identity", ""):
        before = tuple(int(x) for x in img.shape)
        img = apply_annotation_spatial_transform(img, spatial)
        logger.info(
            "%s transpose_spatial_mode=%s: %s → %s",
            path.name,
            spatial,
            before,
            tuple(img.shape),
        )

    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError(f"MSI must be nonnegative: {path}")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def _align_msi_channels(
    vols: list[np.ndarray],
    paths: list[Path],
    *,
    mode: str,
    pad_side: str,
) -> list[np.ndarray]:
    """Harmonize channel count across cubes (HxWxC)."""
    mode = str(mode).strip().lower()
    pad_side = str(pad_side).strip().lower()
    if mode not in ("strict", "pad_max", "crop_min"):
        raise ValueError(f"gallery.channel_align must be strict, pad_max, or crop_min; got {mode!r}")
    if pad_side not in ("end", "start"):
        raise ValueError(f"gallery.channel_pad_side must be end or start; got {pad_side!r}")

    counts = [int(v.shape[-1]) for v in vols]
    if len(set(counts)) == 1:
        return vols

    if mode == "strict":
        raise ValueError(
            "Channel mismatch: "
            + ", ".join(f"{p.name} has C={c}" for p, c in zip(paths, counts))
            + ". Set gallery.channel_align=pad_max or crop_min."
        )

    target = max(counts) if mode == "pad_max" else min(counts)
    logger.warning(
        "Aligning MSI channels (%s -> C=%d): %s",
        mode,
        target,
        ", ".join(f"{p.name}=C{c}" for p, c in zip(paths, counts)),
    )

    out: list[np.ndarray] = []
    for vol, path, c in zip(vols, paths, counts):
        if c == target:
            out.append(vol)
            continue
        if mode == "crop_min":
            out.append(np.ascontiguousarray(vol[..., :target]))
            logger.info("  cropped %s C=%d -> C=%d (leading channels)", path.name, c, target)
            continue
        pad_c = target - c
        if pad_side == "end":
            pad_width = ((0, 0), (0, 0), (0, pad_c))
        else:
            pad_width = ((0, 0), (0, 0), (pad_c, 0))
        out.append(np.pad(vol, pad_width, mode="constant", constant_values=0.0))
        logger.info("  padded %s C=%d -> C=%d (%s)", path.name, c, target, pad_side)
    return out


def _to_uint8_rgb(viz: Any) -> np.ndarray:
    if isinstance(viz, list):
        viz = viz[0]
    arr = np.asarray(viz)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr
    out = arr.astype(np.float32)
    mx = float(np.nanmax(out)) if out.size else 0.0
    if mx <= 1.0 + 1e-6:
        if mx > 1e-12:
            out = (out / mx) * 255.0
        else:
            out = np.zeros_like(out)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _hist_equalize_rgb(viz_rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2

        out = np.asarray(viz_rgb, dtype=np.uint8).copy()
        for ch in range(min(3, out.shape[-1])):
            out[:, :, ch] = cv2.equalizeHist(out[:, :, ch])
        return out
    except Exception:
        return viz_rgb


def _parse_clusters(merged: dict[str, Any]) -> list[int]:
    cl = merged.get("clusters")
    if cl is None:
        return [8]
    if isinstance(cl, str):
        part = [x for x in str(cl).split("-") if str(x).strip()]
        return [int(x) for x in part]
    if isinstance(cl, (list, tuple)):
        return [int(x) for x in cl]
    return [int(cl)]


def _mics_kwargs_from_train_cfg(cfg: DictConfig, seed: int) -> dict[str, Any]:
    mt = str(OmegaConf.select(cfg.model, "model_type", default="mics_lmc")).strip().lower()
    if mt != "mics_lmc":
        raise ValueError(f"mics_cross_infer_gallery requires model.model_type=mics_lmc (got {mt!r})")

    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}

    kw: dict[str, Any] = {}
    for name in _MICS_KEYS:
        if name == "clusters":
            continue
        if name in merged:
            kw[name] = merged[name]

    kw.setdefault("number_of_points", int(merged.get("number_of_points", 1000)))
    kw.setdefault("sampling", str(merged.get("sampling", "coreset")))
    kw.setdefault("num_epochs", int(merged.get("num_epochs", 100)))
    kw.setdefault("number_of_components", int(merged.get("number_of_components", 3)))
    kw.setdefault("num_layers", int(merged.get("num_layers", 2)))
    kw.setdefault("lab_to_rgb", bool(merged.get("lab_to_rgb", True)))
    kw.setdefault("cluster", True)
    kw.setdefault("num_samples", int(merged.get("num_samples", 5000)))
    kw.setdefault("beta", float(merged.get("beta", 1.0)))
    kw.setdefault("k_epoch", int(merged.get("k_epoch", 20)))
    kw.setdefault("temperature", float(merged.get("temperature", 100.0)))
    kw.setdefault("lr", float(merged.get("lr", 1.0)))
    kw.setdefault("batch_size", int(merged.get("batch_size", 1024)))
    kw.setdefault("warmup_epochs", int(merged.get("warmup_epochs", 10)))
    kw.setdefault("factor", float(merged.get("factor", 1.0)))
    kw.setdefault("verbose", bool(merged.get("verbose", False)))
    kw.setdefault("cluster_loss_weight", float(merged.get("cluster_loss_weight", 1.0)))
    kw.setdefault("category_loss_weight", float(merged.get("category_loss_weight", 0.0)))
    kw.setdefault("pixel_sampling", str(merged.get("pixel_sampling", "superpixel")))
    kw.setdefault("pca_fit_step", int(merged.get("pca_fit_step", 4)))
    kw.setdefault("cluster_on_pca", bool(merged.get("cluster_on_pca", False)))
    kw.setdefault("cluster_pca_dims", int(merged.get("cluster_pca_dims", 50)))
    kw.setdefault("predict_percentile_low", float(merged.get("predict_percentile_low", 0.001)))
    kw.setdefault("predict_percentile_high", float(merged.get("predict_percentile_high", 99.999)))

    kw["clusters"] = _parse_clusters(merged)
    kw["clusters_auto_tune"] = None
    for sk in tuple(x for x in kw if str(x).startswith("clusters_auto_tune_")):
        kw.pop(sk, None)
    kw["random_state"] = int(seed)
    kw["cluster"] = True
    return kw


def _tissue_centroid_x(msi: np.ndarray) -> float:
    mask = msi.sum(axis=-1) > 0
    xs = np.flatnonzero(mask.any(axis=0))
    if xs.size == 0:
        return float(msi.shape[1]) / 2.0
    return float(np.mean(xs))


def _resolve_laterality_flip(stem: str, ga: Any, *, centroid_x: float, width: int) -> bool:
    """Return True if this cube should be flipped left-right for canonical display."""
    raw = getattr(ga, "laterality_flip", None)
    if raw is not None:
        mp = OmegaConf.to_container(raw, resolve=True)
        if isinstance(mp, dict):
            for key in (stem, int(stem) if stem.isdigit() else stem):
                if key in mp:
                    return bool(mp[key])
    target = str(getattr(ga, "canonical_laterality", "right")).strip().lower()
    if target in ("none", "off", "", "identity"):
        return False
    is_left_view = centroid_x < width * 0.5
    if target == "right":
        return is_left_view
    if target == "left":
        return not is_left_view
    raise ValueError(f"gallery.canonical_laterality must be none, right, or left; got {target!r}")


def _canonicalize_laterality(vols: list[np.ndarray], stems: list[str], ga: Any) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for vol, stem in zip(vols, stems):
        cx = _tissue_centroid_x(vol)
        w = int(vol.shape[1])
        if _resolve_laterality_flip(stem, ga, centroid_x=cx, width=w):
            vol = np.ascontiguousarray(vol[:, ::-1, :])
            logger.info("%s fliplr → canonical %s (centroid_x=%.1f / W=%d)", stem, getattr(ga, "canonical_laterality", "right"), cx, w)
        out.append(vol)
    return out


def _resolve_hemisphere_ids(stems: list[str], ga: Any) -> list[int]:
    raw = getattr(ga, "hemisphere_ids", None)
    if raw is not None:
        ids = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(ids, (list, tuple)) or len(ids) != len(stems):
            raise ValueError("gallery.hemisphere_ids must be a list parallel to gallery.npy_paths")
        return [int(x) for x in ids]
    out: list[int] = []
    for s in stems:
        if s in _NRL4485_HEMISPHERE_BY_STEM:
            out.append(_NRL4485_HEMISPHERE_BY_STEM[s])
        elif s.isdigit():
            out.append(int(s))
        else:
            raise ValueError(
                f"No hemisphere id for stem {s!r}; set gallery.hemisphere_ids parallel to npy_paths"
            )
    return out


def _apply_slice_order(
    paths: list[Path],
    msi_vols: list[np.ndarray],
    stems: list[str],
    ga: Any,
) -> tuple[list[Path], list[np.ndarray], list[str], list[int]]:
    hemi = _resolve_hemisphere_ids(stems, ga)
    mode = str(getattr(ga, "slice_order", "index")).strip().lower()
    if mode in ("index", "none", "", "file"):
        return paths, msi_vols, stems, hemi
    if mode != "hemisphere":
        raise ValueError(f"gallery.slice_order must be index or hemisphere; got {mode!r}")
    perm = sorted(range(len(stems)), key=lambda i: hemi[i])
    paths_o = [paths[i] for i in perm]
    vols_o = [msi_vols[i] for i in perm]
    stems_o = [stems[i] for i in perm]
    hemi_o = [hemi[i] for i in perm]
    logger.info(
        "slice_order=hemisphere: %s",
        ", ".join(f"{s}(hemi={h})" for s, h in zip(stems_o, hemi_o)),
    )
    return paths_o, vols_o, stems_o, hemi_o


def _cleanup_models() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass


def _pad_rgb(
    rgb: np.ndarray,
    target_h: int,
    target_w: int,
    *,
    fill: int = 0,
    align: str = "center",
) -> np.ndarray:
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    th, tw = int(target_h), int(target_w)
    if h > th or w > tw:
        raise ValueError(f"Panel {h}x{w} exceeds canvas {th}x{tw}")
    canvas = np.full((th, tw, 3), int(fill), dtype=np.uint8)
    if str(align).strip().lower() in {"top_left", "topleft", "upper_left"}:
        y0, x0 = 0, 0
    else:
        y0 = (th - h) // 2
        x0 = (tw - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = np.asarray(rgb, dtype=np.uint8)
    return canvas


def _downscale_rgb_max_width(rgb: np.ndarray, max_w: int) -> np.ndarray:
    """Shrink RGB for mosaic layout only (nearest if integer factor else area)."""
    mw = int(max_w)
    if mw <= 0:
        return rgb
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if w <= mw:
        return rgb
    try:
        import cv2

        nh = max(1, int(round(h * (mw / w))))
        return cv2.resize(np.asarray(rgb, dtype=np.uint8), (mw, nh), interpolation=cv2.INTER_AREA)
    except Exception:
        im = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        nh = max(1, int(round(h * (mw / w))))
        return np.asarray(im.resize((mw, nh), Image.Resampling.LANCZOS), dtype=np.uint8)


def _load_ui_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Prefer a clean system UI font; fall back to PIL default."""
    size = max(8, int(size))
    candidates: list[str] = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\calibrib.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    width: int = 3,
    dash: int = 10,
    gap: int = 6,
) -> None:
    """Dashed rectangle outline (inclusive pixel coords)."""
    x0, y0, x1, y1 = (int(v) for v in xy)
    w = max(1, int(width))
    d = max(2, int(dash))
    g = max(1, int(gap))
    # Top / bottom
    x = x0
    while x <= x1:
        x_end = min(x + d - 1, x1)
        draw.line([(x, y0), (x_end, y0)], fill=fill, width=w)
        draw.line([(x, y1), (x_end, y1)], fill=fill, width=w)
        x = x_end + 1 + g
    # Left / right
    y = y0
    while y <= y1:
        y_end = min(y + d - 1, y1)
        draw.line([(x0, y), (x0, y_end)], fill=fill, width=w)
        draw.line([(x1, y), (x1, y_end)], fill=fill, width=w)
        y = y_end + 1 + g


def _build_cross_infer_mosaic(
    grid: list[list[np.ndarray]],
    *,
    row_labels: list[str],
    col_labels: list[str] | None,
    cell_labels: list[list[str]] | None,
    cell_gap: int,
    margin: int,
    row_label_w: int,
    col_label_h: int = 16,
    bg: tuple[int, int, int] = (0, 0, 0),
    label_rgb: tuple[int, int, int] = (160, 160, 160),
    mark_train: bool = True,
    train_label: str = "Train",
    train_banner_h: int = 30,
    train_border_rgb: tuple[int, int, int] = (255, 220, 80),
    train_label_rgb: tuple[int, int, int] = (255, 230, 120),
) -> np.ndarray:
    """Tight black-background mosaic (no matplotlib whitespace).

    When ``mark_train`` is True, the diagonal cell (train section) in each row is
    outlined with a dashed border and labeled ``train_label`` above the panel.
    """
    n = len(grid)
    if n == 0 or len(grid[0]) != n:
        raise ValueError("grid must be square n×n")
    col_w = [max(int(grid[r][c].shape[1]) for r in range(n)) for c in range(n)]
    row_h = [max(int(grid[r][c].shape[0]) for c in range(n)) for r in range(n)]
    gap = max(0, int(cell_gap))
    m = max(0, int(margin))
    rlw = max(0, int(row_label_w))
    clh = max(0, int(col_label_h)) if col_labels else 0
    banner = max(0, int(train_banner_h)) if mark_train else 0
    total_w = rlw + m + sum(col_w) + gap * max(0, n - 1) + m
    total_h = m + clh + sum(h + banner for h in row_h) + gap * max(0, n - 1) + m
    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    font = _load_ui_font(16)
    train_font = _load_ui_font(22, bold=True)

    if col_labels and clh > 0:
        x = rlw + m
        for c in range(n):
            tw, th = _text_size(draw, col_labels[c], font)
            draw.text(
                (x + max(0, (col_w[c] - tw) // 2), m + max(0, (clh - th) // 2)),
                col_labels[c],
                fill=label_rgb,
                font=font,
            )
            x += col_w[c] + gap

    y = m + clh
    for r in range(n):
        content_y = y + banner
        if row_labels and rlw > 0:
            ty = content_y + max(0, row_h[r] // 2 - 7)
            draw.text((4, ty), row_labels[r], fill=label_rgb, font=font)
        x = rlw + m
        for c in range(n):
            im = Image.fromarray(np.asarray(grid[r][c], dtype=np.uint8), mode="RGB")
            dx = max(0, (col_w[c] - im.size[0]) // 2)
            dy = max(0, (row_h[r] - im.size[1]) // 2)
            px, py = x + dx, content_y + dy
            canvas.paste(im, (px, py))
            if cell_labels is not None:
                draw.text((px + 2, py + 2), cell_labels[r][c], fill=(220, 220, 220), font=font)
            if mark_train and r == c:
                pad = 3
                x0 = max(0, px - pad)
                y0 = max(0, py - pad)
                x1 = min(total_w - 1, px + im.size[0] - 1 + pad)
                y1 = min(total_h - 1, py + im.size[1] - 1 + pad)
                _draw_dashed_rect(
                    draw,
                    (x0, y0, x1, y1),
                    fill=train_border_rgb,
                    width=4,
                    dash=12,
                    gap=6,
                )
                tw, th = _text_size(draw, train_label, train_font)
                tx = x + max(0, (col_w[c] - tw) // 2)
                ty = y + max(0, (banner - th) // 2)
                draw.text((tx, ty), train_label, fill=train_label_rgb, font=train_font)
            x += col_w[c] + gap
        y += row_h[r] + banner + gap
    return np.asarray(canvas, dtype=np.uint8)


def _hstack_blocks(blocks: list[np.ndarray], gap: int) -> np.ndarray:
    if not blocks:
        raise ValueError("empty blocks")
    if len(blocks) == 1:
        return blocks[0]
    g = max(0, int(gap))
    max_h = max(int(b.shape[0]) for b in blocks)
    total_w = sum(int(b.shape[1]) for b in blocks) + g * (len(blocks) - 1)
    canvas = Image.new("RGB", (total_w, max_h), (0, 0, 0))
    x = 0
    for i, block in enumerate(blocks):
        im = Image.fromarray(np.asarray(block, dtype=np.uint8), mode="RGB")
        y = (max_h - im.size[1]) // 2
        canvas.paste(im, (x, y))
        x += im.size[0] + (g if i < len(blocks) - 1 else 0)
    return np.asarray(canvas, dtype=np.uint8)


def _vstack_blocks(blocks: list[np.ndarray], gap: int) -> np.ndarray:
    if not blocks:
        raise ValueError("empty blocks")
    if len(blocks) == 1:
        return blocks[0]
    g = max(0, int(gap))
    max_w = max(int(b.shape[1]) for b in blocks)
    total_h = sum(int(b.shape[0]) for b in blocks) + g * (len(blocks) - 1)
    canvas = Image.new("RGB", (max_w, total_h), (0, 0, 0))
    y = 0
    for i, block in enumerate(blocks):
        im = Image.fromarray(np.asarray(block, dtype=np.uint8), mode="RGB")
        x = (max_w - im.size[0]) // 2
        canvas.paste(im, (x, y))
        y += im.size[1] + (g if i < len(blocks) - 1 else 0)
    return np.asarray(canvas, dtype=np.uint8)


def _combine_method_blocks(blocks: list[np.ndarray], gap: int, layout: str) -> np.ndarray:
    mode = str(layout).strip().lower()
    if mode in {"vertical", "column", "rows", "vstack"}:
        return _vstack_blocks(blocks, gap)
    if mode in {"horizontal", "row", "columns", "hstack", "side_by_side"}:
        return _hstack_blocks(blocks, gap)
    raise ValueError(f"Unknown gallery.method_layout: {layout!r} (use vertical or horizontal)")


def _wrap_block_title(
    block: np.ndarray,
    title: str,
    *,
    title_h: int = 64,
    box_fill: tuple[int, int, int] = (36, 42, 54),
    box_outline: tuple[int, int, int] = (230, 232, 236),
    text_rgb: tuple[int, int, int] = (250, 250, 250),
) -> np.ndarray:
    """Prefix a method block with a large centered title inside a rounded box."""
    im = Image.fromarray(np.asarray(block, dtype=np.uint8), mode="RGB")
    # Scale title to block width so it stays readable on wide native mosaics.
    font_size = int(np.clip(round(im.size[0] * 0.055), 28, 52))
    font = _load_ui_font(font_size, bold=True)

    # Prefer font metrics over textbbox — bold UI fonts often ink past the bbox.
    probe = Image.new("RGB", (8, 8), (0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), title, font=font)
    tw = int(bbox[2] - bbox[0])
    try:
        ascent, descent = font.getmetrics()  # type: ignore[attr-defined]
        th = int(ascent + descent)
    except Exception:
        th = int(bbox[3] - bbox[1])
    th = max(th, int(bbox[3] - bbox[1]), int(round(font_size * 1.15)))

    outline_w = 3
    pad_x = max(40, font_size // 2 + 16) + outline_w
    pad_y = max(18, int(round(font_size * 0.42))) + outline_w
    box_h = th + 2 * pad_y
    radius = max(8, min(box_h // 5, 14))
    box_w = max(tw + 2 * pad_x, tw + 2 * radius + 24)
    band_h = max(int(title_h), box_h + 24)

    canvas = Image.new("RGB", (im.size[0], im.size[1] + band_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    bx0 = max(4, (im.size[0] - box_w) // 2)
    by0 = max(2, (band_h - box_h) // 2)
    bx1, by1 = bx0 + box_w, by0 + box_h
    draw.rounded_rectangle(
        (bx0, by0, bx1, by1),
        radius=radius,
        fill=box_fill,
        outline=box_outline,
        width=outline_w,
    )
    # Center with anchor so bold glyphs (e.g. "PCA") are not clipped by baseline offset.
    draw.text(
        ((bx0 + bx1) // 2, (by0 + by1) // 2),
        title,
        fill=text_rgb,
        font=font,
        anchor="mm",
    )

    canvas.paste(im, (0, band_h))
    return np.asarray(canvas, dtype=np.uint8)


def _prepare_display_grid(
    rgb_cells: list[list[np.ndarray]],
    *,
    pad_val: int,
    pad_align: str,
    mosaic_max_w: int | None,
) -> list[list[np.ndarray]]:
    n = len(rgb_cells)
    col_max_w = [
        max(int(rgb_cells[r][c].shape[1]) for r in range(n)) for c in range(n)
    ]
    row_max_h = [
        max(int(rgb_cells[r][c].shape[0]) for c in range(n)) for r in range(n)
    ]
    padded: list[list[np.ndarray]] = []
    for train_i in range(n):
        row_out: list[np.ndarray] = []
        th = row_max_h[train_i]
        for col_k in range(n):
            rgb = rgb_cells[train_i][col_k]
            row_out.append(_pad_rgb(rgb, th, col_max_w[col_k], fill=pad_val, align=pad_align))
        padded.append(row_out)
    if mosaic_max_w is not None and mosaic_max_w > 0:
        max_w = max(col_max_w)
        if max_w > mosaic_max_w:
            padded = [
                [_downscale_rgb_max_width(cell, mosaic_max_w) for cell in row] for row in padded
            ]
    return padded


def _sorted_timing_rows(timings: list[TrainRowTiming]) -> list[TrainRowTiming]:
    method_order = {key: i for i, key in enumerate(METHOD_LABELS)}
    return sorted(
        timings,
        key=lambda r: (r.train_hemisphere, r.train_stem, method_order.get(r.method, 99)),
    )


def _write_timing_csv(timings: list[TrainRowTiming], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "method_label",
        "train_stem",
        "train_hemisphere",
        "train_seconds",
        "infer_total_seconds",
        "total_seconds",
        "infer_stems",
        "infer_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in _sorted_timing_rows(timings):
            writer.writerow(
                {
                    "method": row.method,
                    "method_label": METHOD_LABELS.get(row.method, row.method),
                    "train_stem": row.train_stem,
                    "train_hemisphere": row.train_hemisphere,
                    "train_seconds": f"{row.train_seconds:.3f}",
                    "infer_total_seconds": f"{row.infer_total_seconds:.3f}",
                    "total_seconds": f"{row.total_seconds:.3f}",
                    "infer_stems": ";".join(row.infer_stems),
                    "infer_seconds": ";".join(f"{x:.3f}" for x in row.infer_seconds),
                }
            )


def _write_timing_paper_tables(timings: list[TrainRowTiming], out_dir: Path) -> list[Path]:
    """Tab-separated, Markdown, and LaTeX tables for copy-paste into a manuscript."""
    rows = _sorted_timing_rows(timings)
    if not rows:
        return []

    headers = [
        "Section",
        "Hemisphere",
        "Method",
        "Train (s)",
        "Inference (s)",
        "Total (s)",
    ]
    body: list[list[str]] = []
    for row in rows:
        body.append(
            [
                row.train_stem,
                str(row.train_hemisphere),
                METHOD_LABELS.get(row.method, row.method),
                f"{row.train_seconds:.1f}",
                f"{row.infer_total_seconds:.1f}",
                f"{row.total_seconds:.1f}",
            ]
        )

    for agg in _aggregate_timing_by_method(timings):
        body.append(
            [
                "mean",
                f"n={agg['n_sections']}",
                str(agg["label"]),
                f"{agg['train_mean']:.1f}",
                f"{agg['infer_mean']:.1f}",
                f"{agg['total_mean']:.1f}",
            ]
        )

    saved: list[Path] = []

    tsv_path = out_dir / "timing_table.tsv"
    tsv_lines = ["\t".join(headers)] + ["\t".join(r) for r in body]
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    saved.append(tsv_path)

    md_path = out_dir / "timing_table.md"
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    md_lines.extend("| " + " | ".join(r) + " |" for r in body)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    saved.append(md_path)

    tex_path = out_dir / "timing_table.tex"
    tex_cols = "lrlrrr"
    tex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Cross-inference runtime per section and method. "
        r"Training fits on one serial section; inference applies the model to all four sections.}",
        r"\label{tab:cross_infer_timing}",
        f"\\begin{{tabular}}{{{tex_cols}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    tex_lines.extend(" & ".join(r) + r" \\" for r in body)
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    saved.append(tex_path)

    return saved


_TIMING_PALETTE: dict[str, tuple[str, str]] = {
    "mics": ("#1B4965", "#62B6CB"),
    "parametric_umap": ("#5A189A", "#C77DFF"),
    "pca": ("#BC6C25", "#F4A261"),
}


def _apply_timing_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Segoe UI"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#CCCCCC",
            "legend.framealpha": 0.92,
        }
    )


def _section_xlabel(stem: str, hemi: int) -> str:
    return f"Section {stem}\n(hemisphere {hemi})"


def _save_timing_section_plot(
    rows: list[TrainRowTiming],
    *,
    stem: str,
    hemi: int,
    plot_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _apply_timing_plot_style()
    method_order = {key: i for i, key in enumerate(METHOD_LABELS)}
    rows = sorted(rows, key=lambda r: method_order.get(r.method, 99))
    labels = [METHOD_LABELS.get(r.method, r.method) for r in rows]

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.58
    train_vals = [r.train_seconds for r in rows]
    infer_vals = [r.infer_total_seconds for r in rows]
    train_colors = [_TIMING_PALETTE.get(r.method, ("#444444", "#888888"))[0] for r in rows]
    infer_colors = [_TIMING_PALETTE.get(r.method, ("#444444", "#888888"))[1] for r in rows]

    ax.bar(x, train_vals, width, label="Training", color=train_colors, edgecolor="white", linewidth=0.8)
    ax.bar(
        x,
        infer_vals,
        width,
        bottom=train_vals,
        label="Inference",
        color=infer_colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for i, (tr, inf) in enumerate(zip(train_vals, infer_vals)):
        total = tr + inf
        ax.text(i, total + max(total * 0.02, 0.8), f"{total:.0f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (s)")
    ax.set_title(_section_xlabel(stem, hemi).replace("\n", " "))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(tr + inf for tr, inf in zip(train_vals, infer_vals))
    ax.set_ylim(0, ymax * 1.18)
    legend_handles = [
        Patch(facecolor="#555555", edgecolor="white", label="Training (dark)"),
        Patch(facecolor="#AAAAAA", edgecolor="white", label="Inference (light)"),
    ]
    ax.legend(handles=legend_handles, frameon=True, loc="upper right", fontsize=8)
    fig.tight_layout()

    saved: list[Path] = []
    png_path = plot_dir / f"timing_section_{stem}.png"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    saved.append(png_path)
    if save_pdf:
        pdf_path = plot_dir / f"timing_section_{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        saved.append(pdf_path)
    plt.close(fig)
    return saved


def _aggregate_timing_by_method(
    timings: list[TrainRowTiming],
) -> list[dict[str, float | str | int]]:
    """Mean and std of train/infer/total per method across all training sections."""
    by_method: dict[str, list[TrainRowTiming]] = {}
    for row in timings:
        by_method.setdefault(row.method, []).append(row)

    method_order = {key: i for i, key in enumerate(METHOD_LABELS)}
    out: list[dict[str, float | str | int]] = []
    for method_key in sorted(by_method, key=lambda k: method_order.get(k, 99)):
        rows = by_method[method_key]
        train = np.array([r.train_seconds for r in rows], dtype=np.float64)
        infer = np.array([r.infer_total_seconds for r in rows], dtype=np.float64)
        total = train + infer
        out.append(
            {
                "method": method_key,
                "label": METHOD_LABELS.get(method_key, method_key),
                "n_sections": len(rows),
                "train_mean": float(np.mean(train)),
                "train_std": float(np.std(train, ddof=1)) if len(train) > 1 else 0.0,
                "infer_mean": float(np.mean(infer)),
                "infer_std": float(np.std(infer, ddof=1)) if len(infer) > 1 else 0.0,
                "total_mean": float(np.mean(total)),
                "total_std": float(np.std(total, ddof=1)) if len(total) > 1 else 0.0,
            }
        )
    return out


def _save_timing_summary_figure(
    timings: list[TrainRowTiming],
    out_dir: Path,
    *,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    """Publication-style figure: mean runtime per method, averaged over all sections."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not timings:
        return []

    _apply_timing_plot_style()
    plot_dir = out_dir / "timing_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    agg = _aggregate_timing_by_method(timings)
    n_meth = len(agg)
    fig, ax = plt.subplots(figsize=(max(5.2, 1.35 * n_meth + 2.8), 4.8))
    x = np.arange(n_meth, dtype=np.float64)
    width = 0.56

    train_means = [float(a["train_mean"]) for a in agg]
    infer_means = [float(a["infer_mean"]) for a in agg]
    total_stds = [float(a["total_std"]) for a in agg]
    train_colors = [_TIMING_PALETTE[str(a["method"])][0] for a in agg]
    infer_colors = [_TIMING_PALETTE[str(a["method"])][1] for a in agg]
    labels = [str(a["label"]) for a in agg]
    n_sec = int(agg[0]["n_sections"]) if agg else 0

    ax.bar(x, train_means, width, color=train_colors, edgecolor="white", linewidth=0.8, label="_nolegend_")
    ax.bar(
        x,
        infer_means,
        width,
        bottom=train_means,
        color=infer_colors,
        edgecolor="white",
        linewidth=0.8,
        label="_nolegend_",
    )
    totals = [tr + inf for tr, inf in zip(train_means, infer_means)]
    ax.errorbar(
        x,
        totals,
        yerr=total_stds,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.2,
        capsize=5,
        capthick=1.2,
        zorder=5,
    )
    for i, (total, std) in enumerate(zip(totals, total_stds)):
        ax.text(
            i,
            total + std + max(total * 0.04, 1.0),
            f"{total:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Mean cross-inference runtime ({n_sec} sections)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(t + s for t, s in zip(totals, total_stds))
    ax.set_ylim(0, ymax * 1.2)

    phase_patches = [
        Patch(facecolor="#555555", edgecolor="white", label="Training"),
        Patch(facecolor="#BBBBBB", edgecolor="white", label="Inference"),
    ]
    ax.legend(handles=phase_patches, loc="upper right", frameon=True, title="Phase")
    ax.text(
        0.02,
        0.98,
        "Error bars: SD across sections",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout()

    saved: list[Path] = []
    png_path = plot_dir / "timing_summary.png"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    saved.append(png_path)
    if save_pdf:
        pdf_path = plot_dir / "timing_summary.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        saved.append(pdf_path)
    plt.close(fig)
    return saved


def _save_timing_plots(
    timings: list[TrainRowTiming],
    out_dir: Path,
    *,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    if not timings:
        return []

    plot_dir = out_dir / "timing_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    by_stem: dict[str, list[TrainRowTiming]] = {}
    for row in timings:
        by_stem.setdefault(row.train_stem, []).append(row)

    saved: list[Path] = []
    saved.extend(_save_timing_summary_figure(timings, out_dir, dpi=dpi, save_pdf=save_pdf))

    for stem in sorted(by_stem, key=lambda s: (by_stem[s][0].train_hemisphere, s)):
        rows = by_stem[stem]
        hemi = rows[0].train_hemisphere
        saved.extend(
            _save_timing_section_plot(
                rows,
                stem=stem,
                hemi=hemi,
                plot_dir=plot_dir,
                dpi=dpi,
                save_pdf=save_pdf,
            )
        )

    return saved


def _cross_infer_mics(
    msi_vols: list[np.ndarray],
    stems: list[str],
    hemi_ids: list[int],
    *,
    cfg: DictConfig,
    seed: int,
    eq: bool,
    pan_dir: Path,
) -> tuple[list[list[np.ndarray]], list[TrainRowTiming]]:
    n = len(msi_vols)
    rgb_cells: list[list[np.ndarray]] = [[None] * n for _ in range(n)]  # type: ignore[list-item]
    timings: list[TrainRowTiming] = []
    for train_i in range(n):
        logger.info("[pMiCS] train on %s (%d/%d)", stems[train_i], train_i + 1, n)
        kw = _mics_kwargs_from_train_cfg(cfg, seed=seed + train_i)
        model: MSIParametricMiCSLMC | None = None
        infer_secs: list[float] = []
        t_train0 = time.perf_counter()
        try:
            model = MSIParametricMiCSLMC(**kw)
            model.fit(msi_vols[train_i])
            train_sec = time.perf_counter() - t_train0
            for infer_j in range(n):
                t_inf0 = time.perf_counter()
                rgb = _to_uint8_rgb(model.predict(msi_vols[infer_j]))
                if eq:
                    rgb = _hist_equalize_rgb(rgb)
                infer_secs.append(time.perf_counter() - t_inf0)
                rgb_cells[train_i][infer_j] = rgb
                lbl = f"pmics__train_{stems[train_i]}__infer_{stems[infer_j]}.png"
                Image.fromarray(rgb, mode="RGB").save(pan_dir / lbl)
        finally:
            if model is not None:
                try:
                    model.release_resources()
                except Exception:
                    pass
                del model
            _cleanup_models()
        if len(infer_secs) == n:
            timings.append(
                TrainRowTiming(
                    method="mics",
                    train_stem=stems[train_i],
                    train_hemisphere=int(hemi_ids[train_i]),
                    train_seconds=train_sec,
                    infer_stems=list(stems),
                    infer_seconds=infer_secs,
                )
            )
            logger.info(
                "[pMiCS] section %s: train=%.1fs infer=%.1fs total=%.1fs",
                stems[train_i],
                train_sec,
                sum(infer_secs),
                train_sec + sum(infer_secs),
            )
    for r in range(n):
        for c in range(n):
            if rgb_cells[r][c] is None:
                raise RuntimeError(f"Missing pMiCS panel train={r} col={c}")
    return rgb_cells, timings  # type: ignore[return-value]


def _cross_infer_pca(
    msi_vols: list[np.ndarray],
    stems: list[str],
    hemi_ids: list[int],
    *,
    cfg: DictConfig,
    eq: bool,
    pan_dir: Path,
) -> tuple[list[list[np.ndarray]], list[TrainRowTiming]]:
    n = len(msi_vols)
    pca_cfg = getattr(cfg, "pca", None)
    max_iter = int(getattr(pca_cfg, "max_iter", 2000)) if pca_cfg is not None else 2000
    rgb_cells: list[list[np.ndarray]] = [[None] * n for _ in range(n)]  # type: ignore[list-item]
    timings: list[TrainRowTiming] = []
    for train_i in range(n):
        logger.info("[PCA] train on %s (%d/%d)", stems[train_i], train_i + 1, n)
        t_train0 = time.perf_counter()
        model = PCA3D(max_iter=max_iter)
        model.fit([msi_vols[train_i]])
        train_sec = time.perf_counter() - t_train0
        infer_secs: list[float] = []
        for infer_j in range(n):
            t_inf0 = time.perf_counter()
            rgb = _to_uint8_rgb(model.predict(msi_vols[infer_j]))
            if eq:
                rgb = _hist_equalize_rgb(rgb)
            infer_secs.append(time.perf_counter() - t_inf0)
            rgb_cells[train_i][infer_j] = rgb
            lbl = f"pca__train_{stems[train_i]}__infer_{stems[infer_j]}.png"
            Image.fromarray(rgb, mode="RGB").save(pan_dir / lbl)
        timings.append(
            TrainRowTiming(
                method="pca",
                train_stem=stems[train_i],
                train_hemisphere=int(hemi_ids[train_i]),
                train_seconds=train_sec,
                infer_stems=list(stems),
                infer_seconds=infer_secs,
            )
        )
        logger.info(
            "[PCA] section %s: train=%.1fs infer=%.1fs total=%.1fs",
            stems[train_i],
            train_sec,
            sum(infer_secs),
            train_sec + sum(infer_secs),
        )
        del model
        _cleanup_models()
    return rgb_cells, timings  # type: ignore[return-value]


def _cross_infer_parametric_umap(
    msi_vols: list[np.ndarray],
    stems: list[str],
    hemi_ids: list[int],
    *,
    cfg: DictConfig,
    seed: int,
    eq: bool,
    pan_dir: Path,
) -> tuple[list[list[np.ndarray]], list[TrainRowTiming]]:
    n = len(msi_vols)
    sampling_mode = str(OmegaConf.select(cfg, "benchmark.sampling_mode", default="random"))
    rgb_cells: list[list[np.ndarray]] = [[None] * n for _ in range(n)]  # type: ignore[list-item]
    timings: list[TrainRowTiming] = []
    for train_i in range(n):
        logger.info("[pUMAP] train on %s (%d/%d)", stems[train_i], train_i + 1, n)
        mk = _umap_kwargs(cfg, sampling_mode, seed + train_i)
        model = UMAPVirtualStain(**mk, start_bin=0, end_bin=None)
        infer_secs: list[float] = []
        t_train0 = time.perf_counter()
        try:
            model.fit([msi_vols[train_i]], keras_fit_kwargs={}, roi_mask=None)
            train_sec = time.perf_counter() - t_train0
            for infer_j in range(n):
                t_inf0 = time.perf_counter()
                rgb = _to_uint8_rgb(model.predict(msi_vols[infer_j]))
                if eq:
                    rgb = _hist_equalize_rgb(rgb)
                infer_secs.append(time.perf_counter() - t_inf0)
                rgb_cells[train_i][infer_j] = rgb
                lbl = f"pumap__train_{stems[train_i]}__infer_{stems[infer_j]}.png"
                Image.fromarray(rgb, mode="RGB").save(pan_dir / lbl)
        finally:
            del model
            _cleanup_models()
        if len(infer_secs) == n:
            timings.append(
                TrainRowTiming(
                    method="parametric_umap",
                    train_stem=stems[train_i],
                    train_hemisphere=int(hemi_ids[train_i]),
                    train_seconds=train_sec,
                    infer_stems=list(stems),
                    infer_seconds=infer_secs,
                )
            )
            logger.info(
                "[pUMAP] section %s: train=%.1fs infer=%.1fs total=%.1fs",
                stems[train_i],
                train_sec,
                sum(infer_secs),
                train_sec + sum(infer_secs),
            )
    return rgb_cells, timings  # type: ignore[return-value]


def _enabled_compare_methods(cfg: DictConfig) -> list[str]:
    cm = getattr(cfg.gallery, "compare_methods", None)
    out: list[str] = ["mics"]
    if cm is None:
        return out
    if bool(getattr(cm, "parametric_umap", False)):
        out.append("parametric_umap")
    if bool(getattr(cm, "pca", False)):
        out.append("pca")
    return out


def _log_accelerators() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("PyTorch CUDA: %s", name)
        else:
            logger.info("PyTorch CUDA: not available")
    except Exception:
        pass
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info("TensorFlow GPU: %s", ", ".join(g.name for g in gpus))
        else:
            logger.info(
                "TensorFlow GPU: not available (Windows native GPU needs env "
                "maldi-pumap-gpu with tensorflow==2.10.1 + cudatoolkit 11.2)"
            )
    except Exception:
        pass


@hydra.main(version_base=None, config_path="configs", config_name="mics_cross_infer_gallery")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    _log_accelerators()

    ga = getattr(cfg, "gallery", None)
    if ga is None:
        raise ValueError("Config must define gallery:")

    raw_paths = OmegaConf.to_container(getattr(ga, "npy_paths", None), resolve=True)
    if not isinstance(raw_paths, (list, tuple)) or len(raw_paths) < 2:
        raise ValueError("gallery.npy_paths must be a list of at least two .npy paths")

    paths = [Path(to_absolute_path(str(p))).expanduser().resolve() for p in raw_paths]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing MSI: {p}")

    stems = [p.stem for p in paths]
    n = len(paths)
    transpose_msi, transpose_spatial = _resolve_msi_layout_cfg(ga)
    ticnorm = bool(getattr(ga, "tic_normalize", True))
    seed = int(getattr(ga, "seed", 42))
    eq = bool(getattr(ga, "equalize_visualization", False))
    pad_val = int(getattr(ga, "pad_value", 0))
    pad_align = str(getattr(ga, "pad_align", "top_left"))
    channel_align = str(getattr(ga, "channel_align", "pad_max"))
    channel_pad_side = str(getattr(ga, "channel_pad_side", "end"))
    dpi = int(getattr(ga, "dpi", 160))
    cell_gap = int(getattr(ga, "cell_gap", 2))
    mosaic_margin = int(getattr(ga, "mosaic_margin", 2))
    row_label_w = int(getattr(ga, "row_label_width", 22))
    show_cell = bool(getattr(ga, "show_cell_caption", False))
    show_row_labels = bool(getattr(ga, "show_row_labels", True))
    raw_mw = getattr(ga, "mosaic_max_panel_width", 640)
    mosaic_max_w = None if raw_mw is None else int(raw_mw)
    timing_cfg = getattr(ga, "timing", None)
    timing_enabled = True if timing_cfg is None else bool(getattr(timing_cfg, "enabled", True))
    timing_dpi = int(getattr(timing_cfg, "dpi", dpi)) if timing_cfg is not None else dpi
    timing_pdf = bool(getattr(timing_cfg, "save_pdf", True)) if timing_cfg is not None else True

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    pan_dir = out_dir / "panels"
    pan_dir.mkdir(parents=True, exist_ok=True)

    msi_vols = [
        _load_msi(
            p,
            transpose_msi=transpose_msi,
            transpose_spatial_mode=transpose_spatial,
            tic_normalize=ticnorm,
        )
        for p in paths
    ]
    msi_vols = _align_msi_channels(
        msi_vols,
        paths,
        mode=channel_align,
        pad_side=channel_pad_side,
    )
    paths, msi_vols, stems, hemi_ids = _apply_slice_order(paths, msi_vols, stems, ga)
    msi_vols = _canonicalize_laterality(msi_vols, stems, ga)
    n_chan = int(msi_vols[0].shape[-1])
    logger.info("Using C=%d for all %d cubes", n_chan, n)

    methods = _enabled_compare_methods(cfg)
    block_gap = int(getattr(ga, "method_block_gap", 12))
    method_layout = str(getattr(ga, "method_layout", "vertical"))
    row_labels = [str(hemi_ids[r]) if show_row_labels else "" for r in range(n)]
    col_labels = [str(hemi_ids[c]) for c in range(n)] if show_row_labels else None
    cell_labels: list[list[str]] | None = None
    if show_cell:
        cell_labels = [[f"{stems[r]}→{stems[c]}" for c in range(n)] for r in range(n)]

    method_grids: list[tuple[str, np.ndarray]] = []
    all_timings: list[TrainRowTiming] = []
    runners = {
        "mics": lambda: _cross_infer_mics(
            msi_vols, stems, hemi_ids, cfg=cfg, seed=seed, eq=eq, pan_dir=pan_dir
        ),
        "pca": lambda: _cross_infer_pca(
            msi_vols, stems, hemi_ids, cfg=cfg, eq=eq, pan_dir=pan_dir
        ),
        "parametric_umap": lambda: _cross_infer_parametric_umap(
            msi_vols, stems, hemi_ids, cfg=cfg, seed=seed, eq=eq, pan_dir=pan_dir
        ),
    }

    for method_key in methods:
        if method_key not in runners:
            raise ValueError(f"Unknown compare method: {method_key}")
        label = METHOD_LABELS.get(method_key, method_key)
        logger.info("=== Cross-infer method: %s ===", label)
        rgb_cells, method_timings = runners[method_key]()
        all_timings.extend(method_timings)
        disp_padded = _prepare_display_grid(
            rgb_cells,
            pad_val=pad_val,
            pad_align=pad_align,
            mosaic_max_w=mosaic_max_w,
        )
        block = _build_cross_infer_mosaic(
            disp_padded,
            row_labels=row_labels,
            col_labels=col_labels,
            cell_labels=cell_labels,
            cell_gap=cell_gap,
            margin=mosaic_margin,
            row_label_w=row_label_w if show_row_labels else 0,
        )
        block_titled = _wrap_block_title(block, label)
        method_grids.append((method_key, block_titled))
        single_path = out_dir / f"cross_infer_{method_key}.png"
        Image.fromarray(block_titled, mode="RGB").save(single_path, dpi=(dpi, dpi))
        logger.info("Saved %s", single_path.name)

    if len(method_grids) == 1:
        mosaic_rgb = method_grids[0][1]
    else:
        mosaic_rgb = _combine_method_blocks(
            [b for _, b in method_grids],
            block_gap,
            method_layout,
        )

    mosaic_path = out_dir / "mics_cross_infer_gallery.png"
    Image.fromarray(mosaic_rgb, mode="RGB").save(mosaic_path, dpi=(dpi, dpi))

    timing_paths: list[Path] = []
    if timing_enabled and all_timings:
        csv_path = out_dir / "timing_summary.csv"
        _write_timing_csv(all_timings, csv_path)
        timing_paths.append(csv_path)
        timing_paths.extend(_write_timing_paper_tables(all_timings, out_dir))
        timing_paths.extend(
            _save_timing_plots(all_timings, out_dir, dpi=timing_dpi, save_pdf=timing_pdf)
        )
        logger.info("Saved timing CSV, paper tables (tsv/md/tex), and plots")

    method_names = ", ".join(METHOD_LABELS.get(k, k) for k, _ in method_grids)
    readme_lines = [
        "Cross-inference gallery (train row → infer column; same hemisphere order on both axes).",
        f"Methods: {method_names}",
        f"Slice order: {getattr(ga, 'slice_order', 'index')} | hemisphere ids: {hemi_ids}",
        f"Paths: {[str(p) for p in paths]}",
        f"Combined mosaic: {mosaic_path.name} ({mosaic_rgb.shape[1]}x{mosaic_rgb.shape[0]} px)",
        f"Raw panels: {pan_dir.name}/",
    ]
    if timing_paths:
        readme_lines.append(
            "Timing: timing_summary.csv | timing_table.tsv | timing_table.md | timing_table.tex | timing_plots/timing_summary.png"
        )
    readme_lines.append("")
    (out_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"[mics_cross_infer_gallery] Saved {mosaic_path}", flush=True)
    logger.info("Done | mosaic=%dx%d | methods=%s", mosaic_rgb.shape[1], mosaic_rgb.shape[0], method_names)


if __name__ == "__main__":
    main()
