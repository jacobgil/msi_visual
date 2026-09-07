"""ATD-guided 256×256 H&E tile extraction at ~20× magnification from whole-slide images."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from msi_visual.extract.he_thumbnail import _bootstrap_openslide, _resize_longest_side

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MPP = 0.5  # ~20× on typical scanners
DEFAULT_TILE_PX = 256


def resolve_tile_stride_px(tile_px: int, tile_stride_px: int | None) -> int:
    """Stride in 20×-equivalent pixels; default non-overlapping (stride == tile size)."""
    if tile_stride_px is None:
        return int(tile_px)
    stride = int(tile_stride_px)
    if stride <= 0:
        return int(tile_px)
    if stride > tile_px:
        logger.warning("tile_stride_px=%d > tile_px=%d; capping stride to tile_px", stride, tile_px)
        return int(tile_px)
    return stride


@dataclass(frozen=True)
class TilePlan:
    """Level-0 grid geometry for tissue tiles (optionally overlapping)."""

    tile_size_l0: int
    step_l0: int
    tile_stride_px: int
    read_level: int
    read_size: int
    mpp: float
    level0_width: int
    level0_height: int
    row_indices: tuple[int, ...]
    col_indices: tuple[int, ...]
    row_coords_l0: tuple[int, ...]
    col_coords_l0: tuple[int, ...]


@dataclass
class ExtractedTile:
    row: int
    col: int
    x_l0: int
    y_l0: int
    rgb: np.ndarray  # uint8 (tile_px, tile_px, 3)


def tissue_mask_atd(
    rgb: np.ndarray,
    *,
    close_ksize: int = 7,
    open_ksize: int = 5,
    min_saturation: int = 8,
) -> np.ndarray:
    """
    Automatic tissue detection on a downsampled H&E RGB image.

    Combines HSV saturation and grayscale Otsu, then morphological cleanup.
    Returns a boolean mask (True = tissue).
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected H×W×3 RGB, got {rgb.shape}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    score = 0.35 * gray.astype(np.float32) + 0.65 * sat
    score_u8 = np.clip(score, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask > 0) & (sat >= float(min_saturation))
    mask_u8 = (mask.astype(np.uint8) * 255)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
    return mask_u8 > 0


def _resolve_mpp(slide: Any, target_mpp: float) -> float:
    import openslide

    raw_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    raw_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    vals: list[float] = []
    for raw in (raw_x, raw_y):
        if raw is None:
            continue
        try:
            v = float(raw)
            if v > 0:
                vals.append(v)
        except (TypeError, ValueError):
            continue
    if vals:
        return float(np.mean(vals))
    logger.warning("Slide MPP missing; assuming target_mpp=%.3f", target_mpp)
    return float(target_mpp)


def build_tile_plan(
    slide: Any,
    tissue_mask_l0: np.ndarray,
    *,
    target_mpp: float = DEFAULT_TARGET_MPP,
    tile_px: int = DEFAULT_TILE_PX,
    tile_stride_px: int | None = None,
    min_tissue_frac: float = 0.35,
) -> TilePlan:
    """Build a tile grid covering the tissue bounding box."""
    import openslide

    w0, h0 = slide.dimensions
    mpp = _resolve_mpp(slide, target_mpp)
    stride_px = resolve_tile_stride_px(tile_px, tile_stride_px)
    tile_size_l0 = max(1, int(round(tile_px * mpp / target_mpp)))
    step_l0 = max(1, int(round(stride_px * mpp / target_mpp)))

    read_level = 0
    read_size = tile_px
    for lv in range(slide.level_count):
        ds = float(slide.level_downsamples[lv])
        rs = max(1, int(round(tile_size_l0 / ds)))
        if rs >= tile_px:
            read_level = lv
            read_size = rs
        else:
            break

    mask = np.asarray(tissue_mask_l0, dtype=bool)
    if mask.shape != (h0, w0):
        mask = cv2.resize(mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("ATD tissue mask is empty; no tiles to extract.")

    r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    y_start = (r0 // step_l0) * step_l0
    x_start = (c0 // step_l0) * step_l0
    y_end = min(h0, ((r1 // step_l0) + 1) * step_l0)
    x_end = min(w0, ((c1 // step_l0) + 1) * step_l0)

    row_ids: list[int] = []
    col_ids: list[int] = []
    row_coords: list[int] = []
    col_coords: list[int] = []

    for y in range(y_start, y_end, step_l0):
        row_id = (y - y_start) // step_l0
        for x in range(x_start, x_end, step_l0):
            col_id = (x - x_start) // step_l0
            y1 = min(y + tile_size_l0, h0)
            x1 = min(x + tile_size_l0, w0)
            patch = mask[y:y1, x:x1]
            if patch.size == 0:
                continue
            if float(patch.mean()) < float(min_tissue_frac):
                continue
            row_ids.append(int(row_id))
            col_ids.append(int(col_id))
            row_coords.append(int(y))
            col_coords.append(int(x))

    if not row_coords:
        raise ValueError("No tiles passed min_tissue_frac after ATD.")

    return TilePlan(
        tile_size_l0=tile_size_l0,
        step_l0=step_l0,
        tile_stride_px=stride_px,
        read_level=read_level,
        read_size=read_size,
        mpp=mpp,
        level0_width=w0,
        level0_height=h0,
        row_indices=tuple(row_ids),
        col_indices=tuple(col_ids),
        row_coords_l0=tuple(row_coords),
        col_coords_l0=tuple(col_coords),
    )


def _read_tile_rgb(
    slide: Any,
    x_l0: int,
    y_l0: int,
    plan: TilePlan,
    *,
    tile_px: int,
) -> np.ndarray:
    region = slide.read_region((x_l0, y_l0), plan.read_level, (plan.read_size, plan.read_size)).convert("RGB")
    rgb = np.asarray(region, dtype=np.uint8)
    if rgb.shape[0] != tile_px or rgb.shape[1] != tile_px:
        rgb = np.asarray(
            Image.fromarray(rgb).resize((tile_px, tile_px), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    return rgb


def extract_tiles_from_slide(
    slide_path: Path,
    *,
    target_mpp: float = DEFAULT_TARGET_MPP,
    tile_px: int = DEFAULT_TILE_PX,
    tile_stride_px: int | None = None,
    atd_max_size: int = 2048,
    min_tissue_frac: float = 0.35,
) -> tuple[TilePlan, np.ndarray, np.ndarray, list[ExtractedTile]]:
    """
    Open a WSI, run ATD, and read all tissue tiles before closing the slide.
    """
    _bootstrap_openslide()
    import openslide

    slide_path = Path(slide_path).expanduser().resolve()
    slide = openslide.OpenSlide(str(slide_path))
    try:
        w0, h0 = slide.dimensions
        thumb = slide.get_thumbnail((int(atd_max_size), int(atd_max_size))).convert("RGB")
        thumb_rgb = np.asarray(thumb, dtype=np.uint8)
        mask_low = tissue_mask_atd(thumb_rgb)
        mask_l0 = cv2.resize(mask_low.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0
        plan = build_tile_plan(
            slide,
            mask_l0,
            target_mpp=target_mpp,
            tile_px=tile_px,
            tile_stride_px=tile_stride_px,
            min_tissue_frac=min_tissue_frac,
        )
        tiles: list[ExtractedTile] = []
        for r, c, y, x in zip(plan.row_indices, plan.col_indices, plan.row_coords_l0, plan.col_coords_l0):
            rgb = _read_tile_rgb(slide, x, y, plan, tile_px=tile_px)
            tiles.append(ExtractedTile(row=int(r), col=int(c), x_l0=int(x), y_l0=int(y), rgb=rgb))
        return plan, mask_l0, thumb_rgb, tiles
    finally:
        slide.close()


def grid_shape_from_plan(plan: TilePlan) -> tuple[int, int]:
    if not plan.row_indices:
        return 0, 0
    return int(max(plan.row_indices) + 1), int(max(plan.col_indices) + 1)


def _tile_display_rgb(rgb: np.ndarray, *, tile_px: int, display_px: int) -> np.ndarray:
    """Resize or center-crop a tile patch for mosaic / panel display."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if display_px == tile_px:
        return rgb
    if display_px < tile_px:
        cy, cx = tile_px // 2, tile_px // 2
        half = display_px // 2
        y0 = max(0, min(cy - half, tile_px - display_px))
        x0 = max(0, min(cx - half, tile_px - display_px))
        return rgb[y0 : y0 + display_px, x0 : x0 + display_px]
    return np.asarray(
        Image.fromarray(rgb).resize((display_px, display_px), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def build_he_mosaic(
    tiles: list[ExtractedTile],
    grid_h: int,
    grid_w: int,
    *,
    tile_px: int,
    display_px: int,
) -> np.ndarray:
    """Stitch extracted tiles into an H×W RGB mosaic (display_px per grid cell)."""
    out = np.zeros((grid_h * display_px, grid_w * display_px, 3), dtype=np.uint8)
    for tile in tiles:
        rgb = _tile_display_rgb(tile.rgb, tile_px=tile_px, display_px=display_px)
        y0 = tile.row * display_px
        x0 = tile.col * display_px
        out[y0 : y0 + display_px, x0 : x0 + display_px] = rgb
    return out


def save_thumbnail_overview(
    atd_thumb_rgb: np.ndarray,
    tissue_mask_l0: np.ndarray,
    out_path: Path,
    *,
    max_size: int = 4096,
) -> np.ndarray:
    """Save a tissue-masked H&E overview thumbnail for panels."""
    mask_low = cv2.resize(
        tissue_mask_l0.astype(np.uint8),
        (atd_thumb_rgb.shape[1], atd_thumb_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    rgb = atd_thumb_rgb.copy()
    rgb[mask_low == 0] = (240, 240, 240)
    rgb = _resize_longest_side(rgb, max_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)
    return rgb
