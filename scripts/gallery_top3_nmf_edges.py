#!/usr/bin/env python3
"""
Build a gallery of TOP3-on-NMF-coefficient panels plus optional edge / local-Dice rows,
diagnostics, and continuous Dice vs ``hd_edges.nmf_component_edges`` (same knobs as
``rank_visualizations_by_f1`` / ``debug_edge_maps``).

Exports ``_compute_top3_nmf_aggregate_v_agg`` for HD reference mode ``top3_nmf_max`` /
``top3_nmf_mean`` in ``rank_visualizations_by_f1.py``.

Run from repo root:
  python scripts/gallery_top3_nmf_edges.py
  python scripts/gallery_top3_nmf_edges.py input.npy_path=D:/data/0.npy

Config: ``scripts/configs/gallery_top3_nmf_edges.yaml``

When ``output.save_individual_assets`` is true, also writes ``panels/`` (one PNG per cell) and
``diagrams/`` (pixel coverage, Dice bars, optional combined copy, CSV copies).

With ``output.use_slide_name_from_args_txt``, reads ``args.txt`` beside the ``.npy`` (or
``input.args_txt_path``) for ``Namespace.id`` / ``slide_name`` / …, then writes under
``<run_dir>/<slide_slug>/`` and prefixes top-level gallery and metric filenames with ``<slide_slug>__``.
"""
from __future__ import annotations

import csv
import logging
import re
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import cv2
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import NMF

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from debug_edge_maps import (
    _apply_equalize_rgb_uint8,
    _binary_from_threshold,
    _compute_continuous_dice,
    _edge_maps_multiscale,
    _edges_for_continuous_metrics,
    _gray_u8_to_rgb_colormap,
    _highd_edge_maps_nmf_component_edges,
    _load_msi,
    _normalize_map_percentile,
    _normalize_on_mask,
    _parse_edge_detection_cfg,
    _resolve_normalization_mode,
    _to_uint8_gray01,
    _to_uint8_rgb,
    _viz_edge_map,
)
from msi_visual.percentile_ratio import TOP3

logger = logging.getLogger(__name__)


def _cfg_value_to_py(obj: Any) -> Any:
    """Convert OmegaConf nodes to plain Python; ``OmegaConf.to_container`` rejects ``None``."""
    if obj is None:
        return None
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _local_dice_heatmap_f32(
    viz_bin: np.ndarray,
    hd_bin: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
) -> np.ndarray:
    """Same local soft-Dice heatmap as ``rank_visualizations_by_f1._local_dice_heatmap_f32``."""
    k = max(1, int(window))
    a = np.where(valid_mask, np.asarray(viz_bin, dtype=np.float32), 0.0)
    b = np.where(valid_mask, np.asarray(hd_bin, dtype=np.float32), 0.0)
    ab = a * b
    ksize = (k, k)
    sa = cv2.boxFilter(a, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    sb = cv2.boxFilter(b, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    sab = cv2.boxFilter(ab, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    eps = 1e-8
    denom = sa + sb + eps
    dice = (2.0 * sab / denom).astype(np.float32)
    empty = (sa + sb) <= eps
    dice = np.where(empty, 0.0, dice)
    dice = np.where(valid_mask, dice, 0.0)
    return np.clip(dice, 0.0, 1.0)


def _one_k_top3_nmf_rgb_and_viz_edge(
    cfg: DictConfig,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """TOP3-on-NMF RGB uint8 and float viz edge map for one ``k`` (shared by gallery and aggregate export)."""
    coeffs = _nmf_coefficient_cube(msi, valid_mask, k, cfg.nmf)
    rgb_u8 = _top3_rgb_from_cube(cfg, coeffs)
    v_edge = _edge_map_from_top3_rgb(cfg, rgb_u8, valid_mask)
    return rgb_u8, np.asarray(v_edge, dtype=np.float32)


def _nmf_coefficient_cube(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    k: int,
    nmf_cfg: Any,
) -> np.ndarray:
    """Nonnegative NMF coefficients H×W×k (float32), with optional subsampled fit."""
    h, w, c = msi.shape
    x = np.maximum(np.asarray(msi[valid_mask], dtype=np.float64), 0.0)
    n_pix, n_chan = x.shape
    if n_pix < k or n_chan < 1:
        raise ValueError(f"NMF needs at least k={k} valid pixels and >=1 channel; got {n_pix} pixels, {n_chan} ch")
    fit_frac = float(getattr(nmf_cfg, "fit_fraction", 1.0))
    rs = int(getattr(nmf_cfg, "random_state", 42))
    rng = np.random.default_rng(rs)
    init = str(getattr(nmf_cfg, "init", "nndsvda"))
    max_iter = int(getattr(nmf_cfg, "max_iter", 500))
    tol = float(getattr(nmf_cfg, "tol", 1e-4))
    nmf = NMF(
        n_components=k,
        init=init,
        max_iter=max_iter,
        tol=tol,
        random_state=rs,
    )
    if fit_frac >= 1.0 - 1e-9:
        nmf.fit(x)
    else:
        n_fit = max(int(fit_frac * n_pix), k)
        n_fit = min(max(n_fit, k), n_pix)
        fit_idx = rng.choice(n_pix, size=n_fit, replace=False)
        nmf.fit(x[fit_idx])
    w_flat = nmf.transform(x).astype(np.float32, copy=False)
    cub = np.zeros((h, w, k), dtype=np.float32)
    cub[valid_mask] = w_flat
    return cub


def _prepare_rgb_for_viz_edges(cfg: DictConfig, rgb_u8: np.ndarray) -> np.ndarray:
    out = np.asarray(rgb_u8, dtype=np.uint8)
    eq = getattr(cfg, "equalize", None)
    if eq is not None and bool(getattr(eq, "enabled", False)) and bool(getattr(eq, "before_edges", False)):
        out = _apply_equalize_rgb_uint8(
            out,
            str(getattr(eq, "method", "hist")),
            float(getattr(eq, "clahe_clip_limit", 2.0)),
            int(getattr(eq, "clahe_tile_grid_size", 8)),
        )
    eps = getattr(cfg, "edge_pre_smooth", None)
    if eps is not None and bool(getattr(eps, "enabled", False)):
        sigma = float(getattr(eps, "sigma", 1.0))
        if sigma > 0:
            ksize = int(max(3, 2 * int(round(3 * sigma)) + 1))
            out = cv2.GaussianBlur(out, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return out


def _multiscale_sigmas_aggregation(cfg: DictConfig) -> tuple[list[float], str]:
    ms = getattr(cfg, "edge_maps", None)
    sigmas = [0.0]
    agg = "mean"
    if ms is not None and getattr(ms, "multi_scale", None) is not None:
        ms_sub = ms.multi_scale
        if bool(getattr(ms_sub, "enabled", False)):
            raw = getattr(ms_sub, "sigmas", [0.0]) or [0.0]
            sigmas = [float(v) for v in list(raw)]
            agg = str(getattr(ms_sub, "aggregation", "mean"))
    return sigmas, agg


def _edge_map_from_top3_rgb(cfg: DictConfig, rgb_u8: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Viz edge map matching ``_edge_maps_multiscale`` / rank (incl. multi_scale when enabled)."""
    rgb = _prepare_rgb_for_viz_edges(cfg, rgb_u8)
    edge_cfg = _parse_edge_detection_cfg(cfg)
    sigmas, agg = _multiscale_sigmas_aggregation(cfg)
    v_edge, _ = _edge_maps_multiscale(
        "rgb",
        rgb,
        valid_mask,
        sigmas=sigmas,
        aggregation=agg,
        hd_cfg=None,
        edge_method_cfg=edge_cfg,
    )
    return v_edge


def _spatial_norm_viz(cfg: DictConfig, arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    sp = getattr(cfg, "spatial_normalization", None)
    if sp is not None and bool(getattr(sp, "enabled", True)):
        return _normalize_map_percentile(
            arr,
            valid_mask,
            low_pct=float(getattr(sp, "low_percentile", 1.0)),
            high_pct=float(getattr(sp, "high_percentile", 99.0)),
        )
    return _normalize_on_mask(arr, valid_mask)


def _top3_rgb_from_cube(cfg: DictConfig, coeffs: np.ndarray) -> np.ndarray:
    t = cfg.top3
    top = TOP3(
        low=float(t.low),
        norm_percentile=float(t.norm_percentile),
    )(coeffs, to_lab=bool(t.to_lab))
    return _to_uint8_rgb(top)


def _top3_rgb_from_msi(cfg: DictConfig, msi: np.ndarray) -> np.ndarray:
    t = cfg.top3
    top = TOP3(
        low=float(t.low),
        norm_percentile=float(t.norm_percentile),
    )(msi, to_lab=bool(t.to_lab))
    return _to_uint8_rgb(top)


def _display_rgb(cfg: DictConfig, rgb_u8: np.ndarray) -> np.ndarray:
    out = np.asarray(rgb_u8, dtype=np.uint8)
    eq = getattr(cfg, "equalize", None)
    if eq is not None and bool(getattr(eq, "enabled", False)) and bool(getattr(eq, "display", False)):
        out = _apply_equalize_rgb_uint8(
            out,
            str(getattr(eq, "method", "hist")),
            float(getattr(eq, "clahe_clip_limit", 2.0)),
            int(getattr(eq, "clahe_tile_grid_size", 8)),
        )
    return out


def _maybe_downscale(img: np.ndarray, cell_max_size: Sequence[float] | None) -> np.ndarray:
    if not cell_max_size or len(cell_max_size) < 2:
        return img
    max_w, max_h = float(cell_max_size[0]), float(cell_max_size[1])
    h, w = img.shape[:2]
    if max_w <= 0 or max_h <= 0 or (w <= max_w and h <= max_h):
        return img
    s = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(round(w * s)), int(round(h * s))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    if img.ndim == 2:
        return cv2.resize(img, (nw, nh), interpolation=interp)
    return cv2.resize(img, (nw, nh), interpolation=interp)


def _paste_grid(
    cells: list[np.ndarray],
    *,
    ncols: int,
    column_padding: int,
    cell_gap: int,
    row_gap: int,
    row_label_height: int,
    titles: list[str] | None,
    bg: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Paste RGB uint8 cells in a grid (optional title strip per row above images)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise RuntimeError("Pillow is required for gallery layout") from e

    n = len(cells)
    if n == 0:
        raise ValueError("empty cells")
    nrows = int(np.ceil(n / ncols))
    col_w = [0] * ncols
    row_h = [0] * nrows
    for i, im in enumerate(cells):
        r, c = i // ncols, i % ncols
        h, w = im.shape[:2]
        col_w[c] = max(col_w[c], w)
        row_h[r] = max(row_h[r], h)
    gap_x, gap_y = int(cell_gap), int(row_gap)
    pad = int(column_padding)
    lab_h = int(row_label_height)
    total_w = sum(col_w) + pad * (ncols + 1) + gap_x * (ncols - 1)
    total_h = pad + nrows * lab_h + sum(row_h) + (nrows - 1) * row_gap + pad
    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    y_row = float(pad)
    for r in range(nrows):
        if titles and font:
            x_text = float(pad)
            for c in range(ncols):
                idx = r * ncols + c
                if idx < n:
                    draw.text((int(x_text), int(y_row)), str(titles[idx])[:120], fill=(255, 255, 255), font=font)
                x_text += float(col_w[c] + pad + gap_x)
        y_img = y_row + float(lab_h)
        x0 = float(pad)
        for c in range(ncols):
            idx = r * ncols + c
            if idx >= n:
                break
            im = cells[idx]
            pil = Image.fromarray(im) if im.ndim == 3 else Image.fromarray(im).convert("RGB")
            canvas.paste(
                pil,
                (
                    int(x0 + (col_w[c] - pil.size[0]) // 2),
                    int(y_img + (row_h[r] - pil.size[1]) // 2),
                ),
            )
            x0 += float(col_w[c] + pad + gap_x)
        y_row = y_img + float(row_h[r] + row_gap)
    return np.asarray(canvas, dtype=np.uint8)


def _mosaic_title_font(size: int, *, bold: bool = True):
    from PIL import ImageFont

    if bold:
        candidates = (
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/segoeuisb.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
    else:
        candidates = (
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    for path in candidates:
        try:
            return ImageFont.truetype(path, int(size))
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_title_font(
    text: str,
    *,
    max_width: int,
    max_size: int,
    min_size: int = 12,
    pad_x: int = 12,
    bold: bool = True,
):
    """Largest bold title font whose boxed text fits in ``max_width`` (including pad)."""
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = int(max_size)
    lo = int(min_size)
    best = _mosaic_title_font(lo, bold=bold)
    while size >= lo:
        font = _mosaic_title_font(size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = int(bbox[2] - bbox[0]) + 2 * int(pad_x)
        if tw <= int(max_width):
            return font, size
        size -= 1
        best = font
    return best, lo


def _draw_boxed_column_title(
    draw,
    *,
    area_left: int,
    col_w: int,
    y: int,
    text: str,
    font,
    text_color: tuple[int, int, int],
    box_fill: tuple[int, int, int],
    box_border: tuple[int, int, int],
    pad_x: int,
    pad_y: int,
    radius: int,
    border_width: int,
) -> int:
    """Draw a centered rounded title box within a column; return total box height.

    If the box would exceed ``col_w``, it is clamped to the column and text is left-padded
    inside the clamped box (caller should pass a font already fitted when possible).
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = int(bbox[2] - bbox[0])
    th = int(bbox[3] - bbox[1])
    box_w = min(int(col_w), tw + 2 * int(pad_x))
    box_h = th + 2 * int(pad_y)
    box_x = int(area_left + max(0, (col_w - box_w) // 2))
    box_y = int(y)
    draw.rounded_rectangle(
        (box_x, box_y, box_x + box_w, box_y + box_h),
        radius=int(radius),
        fill=box_fill,
        outline=box_border,
        width=int(border_width),
    )
    # Center text in the (possibly clamped) box.
    tx = box_x + max(0, (box_w - tw) // 2) - int(bbox[0])
    ty = box_y + int(pad_y) - int(bbox[1])
    draw.text((tx, ty), text, fill=text_color, font=font)
    return box_h


def _compose_single_image_method_gallery(
    cells: Sequence[np.ndarray],
    *,
    titles: Sequence[str],
    metric_lines: Sequence[Sequence[str]] | None = None,
    column_padding: int = 16,
    cell_gap: int = 16,
    title_font_size: int = 28,
    metric_font_size: int = 16,
    title_color: tuple[int, int, int] = (24, 45, 68),
    metric_color: tuple[int, int, int] = (40, 55, 70),
    title_box_fill: tuple[int, int, int] = (236, 244, 252),
    title_box_border: tuple[int, int, int] = (92, 146, 208),
    title_box_pad_x: int = 14,
    title_box_pad_y: int = 6,
    title_box_radius: int = 10,
    title_box_border_width: int = 2,
    metric_line_gap: int = 4,
    header_gap: int = 8,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    One row: method panels at native size (padded, not resized) with boxed titles and
    optional metric lines above each panel. Title font auto-shrinks per column so boxes
    do not spill into neighboring panels.
    """
    from PIL import Image, ImageDraw

    n = len(cells)
    if n == 0:
        raise ValueError("empty cells")
    if len(titles) != n:
        raise ValueError("titles length must match cells")
    metrics = list(metric_lines) if metric_lines is not None else [[] for _ in range(n)]
    if len(metrics) != n:
        raise ValueError("metric_lines length must match cells")

    metric_font = _mosaic_title_font(metric_font_size, bold=False)
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    col_w = [int(im.shape[1]) for im in cells]
    row_h = max(int(im.shape[0]) for im in cells)
    pad = int(column_padding)
    gap_x = int(cell_gap)

    # Per-column title fonts fitted to column width.
    title_fonts = []
    title_heights = []
    for i in range(n):
        text = str(titles[i])[:120]
        font, _sz = _fit_title_font(
            text,
            max_width=max(24, col_w[i] - 4),
            max_size=int(title_font_size),
            min_size=11,
            pad_x=title_box_pad_x,
            bold=True,
        )
        title_fonts.append(font)
        tb = draw_probe.textbbox((0, 0), text, font=font)
        title_heights.append(int(tb[3] - tb[1]) + 2 * int(title_box_pad_y))

    header_heights: list[int] = []
    for i in range(n):
        mh = 0
        for line in metrics[i]:
            mb = draw_probe.textbbox((0, 0), str(line), font=metric_font)
            mh += int(mb[3] - mb[1]) + int(metric_line_gap)
        if metrics[i]:
            mh += int(header_gap)
        header_heights.append(title_heights[i] + mh)
    header_h = max(header_heights) if header_heights else 0

    total_w = sum(col_w) + pad * (n + 1) + gap_x * (n - 1)
    total_h = pad + header_h + pad + row_h + pad
    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    x0 = float(pad)
    for i, im in enumerate(cells):
        text = str(titles[i])[:120]
        _draw_boxed_column_title(
            draw,
            area_left=int(x0),
            col_w=int(col_w[i]),
            y=int(pad),
            text=text,
            font=title_fonts[i],
            text_color=title_color,
            box_fill=title_box_fill,
            box_border=title_box_border,
            pad_x=title_box_pad_x,
            pad_y=title_box_pad_y,
            radius=title_box_radius,
            border_width=title_box_border_width,
        )
        y_m = float(pad + title_heights[i] + header_gap)
        for line in metrics[i]:
            mb = draw.textbbox((0, 0), str(line), font=metric_font)
            tw = int(mb[2] - mb[0])
            th = int(mb[3] - mb[1])
            tx = int(x0 + max(0, (col_w[i] - tw) // 2))
            draw.text((tx, int(y_m)), str(line), fill=metric_color, font=metric_font)
            y_m += float(th + metric_line_gap)

        pil = Image.fromarray(im) if im.ndim == 3 else Image.fromarray(im).convert("RGB")
        canvas.paste(
            pil,
            (
                int(x0 + (col_w[i] - pil.size[0]) // 2),
                int(pad + header_h + pad + (row_h - pil.size[1]) // 2),
            ),
        )
        x0 += float(col_w[i] + pad + gap_x)
    return np.asarray(canvas, dtype=np.uint8)


def _compose_all_images_mosaic(
    rows: Sequence[tuple[str, Sequence[np.ndarray]]],
    *,
    column_titles: Sequence[str],
    column_padding: int = 8,
    cell_gap: int = 8,
    row_gap: int = 12,
    header_height: int = 36,
    row_label_pad: int = 8,
    row_label_min_width: int = 0,
    show_row_labels: bool = False,
    title_font_size: int = 22,
    title_color: tuple[int, int, int] = (24, 45, 68),
    title_box_fill: tuple[int, int, int] = (236, 244, 252),
    title_box_border: tuple[int, int, int] = (92, 146, 208),
    title_box_pad_x: int = 22,
    title_box_pad_y: int = 10,
    title_box_radius: int = 12,
    title_box_border_width: int = 2,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Stack one row per input image and one column per method.

    Cells are never resized; each column is padded to the widest cell in that column and each
    row to the tallest cell in that row. Row heights may differ across the mosaic.
    """
    from PIL import Image, ImageDraw

    if not rows:
        raise ValueError("empty rows")
    ncols = len(column_titles)
    if ncols <= 0:
        raise ValueError("column_titles is empty")
    for row_label, cells in rows:
        if len(cells) != ncols:
            raise ValueError(
                f"row {row_label!r} has {len(cells)} cells, expected {ncols} column(s)"
            )

    title_font = _mosaic_title_font(title_font_size, bold=True)
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_box_heights: list[int] = []
    for title in column_titles:
        text = str(title)[:120]
        bbox = draw_probe.textbbox((0, 0), text, font=title_font)
        tw = int(bbox[2] - bbox[0])
        th = int(bbox[3] - bbox[1])
        title_box_heights.append(th + 2 * int(title_box_pad_y))
    header_h = max(int(header_height), max(title_box_heights, default=0) + 12)

    col_w = [0] * ncols
    row_h = [0] * len(rows)
    for r, (_label, cells) in enumerate(rows):
        for c, im in enumerate(cells):
            h, w = im.shape[:2]
            col_w[c] = max(col_w[c], int(w))
            row_h[r] = max(row_h[r], int(h))

    row_label_w = 0
    if show_row_labels:
        row_label_w = max(int(row_label_min_width), 0)
        for label, _cells in rows:
            bbox = draw_probe.textbbox((0, 0), str(label)[:120], font=title_font)
            row_label_w = max(row_label_w, int(bbox[2] - bbox[0]) + 2 * int(row_label_pad))

    pad = int(column_padding)
    gap_x = int(cell_gap)
    gap_y = int(row_gap)
    grid_left = int(row_label_w) + pad
    grid_w = sum(col_w) + pad * (ncols + 1) + gap_x * (ncols - 1)
    grid_h = sum(row_h) + (len(rows) - 1) * gap_y if row_h else 0
    total_w = grid_left + grid_w
    total_h = pad + header_h + pad + grid_h + pad
    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    x_title = float(grid_left + pad)
    for c, title in enumerate(column_titles):
        text = str(title)[:120]
        _draw_boxed_column_title(
            draw,
            area_left=int(x_title),
            col_w=int(col_w[c]),
            y=int(pad + max(0, (header_h - title_box_heights[c]) // 2)),
            text=text,
            font=title_font,
            text_color=title_color,
            box_fill=title_box_fill,
            box_border=title_box_border,
            pad_x=title_box_pad_x,
            pad_y=title_box_pad_y,
            radius=title_box_radius,
            border_width=title_box_border_width,
        )
        x_title += float(col_w[c] + pad + gap_x)

    y_row = float(pad + header_h + pad)
    for r, (label, cells) in enumerate(rows):
        if show_row_labels:
            text = str(label)[:120]
            bbox = draw.textbbox((0, 0), text, font=title_font)
            th = int(bbox[3] - bbox[1])
            draw.text(
                (int(row_label_pad), int(y_row + max(0, (row_h[r] - th) // 2))),
                text,
                fill=title_color,
                font=title_font,
            )
        x0 = float(grid_left + pad)
        for c, im in enumerate(cells):
            pil = Image.fromarray(im) if im.ndim == 3 else Image.fromarray(im).convert("RGB")
            canvas.paste(
                pil,
                (
                    int(x0 + (col_w[c] - pil.size[0]) // 2),
                    int(y_row + (row_h[r] - pil.size[1]) // 2),
                ),
            )
            x0 += float(col_w[c] + pad + gap_x)
        y_row += float(row_h[r] + gap_y)
    return np.asarray(canvas, dtype=np.uint8)


def _vstack_images(top: np.ndarray, bottom: np.ndarray, gap: int, bg: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    from PIL import Image

    w = max(top.shape[1], bottom.shape[1])
    h = top.shape[0] + gap + bottom.shape[0]
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(Image.fromarray(top), (0, 0))
    canvas.paste(Image.fromarray(bottom), (0, top.shape[0] + gap))
    return np.asarray(canvas, dtype=np.uint8)


def _compute_top3_nmf_aggregate_v_agg(
    cfg: DictConfig,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    mode: str = "max",
) -> np.ndarray:
    """
    Stack viz edge maps from TOP3-on-NMF (one per ``nmf.k_list`` entry), then ``max`` or ``mean``.
    Matches ``rank_visualizations_by_f1`` TOP3+NMF HD path (equalize / edge_pre_smooth / edge_maps / edge_detection).
    """
    nmf_cfg = cfg.nmf
    k_list = OmegaConf.to_container(getattr(nmf_cfg, "k_list", []), resolve=True)
    if not isinstance(k_list, (list, tuple)) or len(k_list) == 0:
        raise ValueError("nmf.k_list must be a non-empty list")
    ks = [int(k) for k in k_list]
    for k in ks:
        if k < 3:
            raise ValueError(f"TOP3 requires k>=3 components, got k={k}")

    edges: list[np.ndarray] = []
    for k in ks:
        _, v_edge = _one_k_top3_nmf_rgb_and_viz_edge(cfg, msi, valid_mask, k)
        edges.append(v_edge)
    stack = np.stack(edges, axis=0)
    m = str(mode).strip().lower()
    if m == "max":
        return np.max(stack, axis=0).astype(np.float32, copy=False)
    if m == "mean":
        return np.mean(stack, axis=0).astype(np.float32, copy=False)
    raise ValueError(f"mode must be max|mean, got {mode!r}")


def _raw_sobel_edge(cfg: DictConfig, rgb_u8: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Sobel edge energy (float) without ``_edge_maps_multiscale`` mask normalization (for aggregate_all_edges)."""
    rgb = _prepare_rgb_for_viz_edges(cfg, rgb_u8)
    edge_cfg = _parse_edge_detection_cfg(cfg)
    cs = str(edge_cfg.get("rgb_color_space", "gray")).strip().lower()
    method = str(edge_cfg.get("rgb_method", "gradient")).strip().lower()
    if method != "gradient":
        # Fall back to same path as multiscale for non-gradient
        return _edge_map_from_top3_rgb(cfg, rgb_u8, valid_mask)
    return _viz_edge_map(rgb, valid_mask, color_space=cs, already_01=False)


def _apply_light_plot_style(ax: Any) -> None:
    """White background + light gray grid (matches classic matplotlib gallery look)."""
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.55, color="#c8c8c8", zorder=0)
    ax.tick_params(colors="black", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#888888")


def _plot_step_line_axis(
    ax: Any,
    x_idx: np.ndarray,
    y: list[float] | list[int],
    step_labels: list[str],
    *,
    ylabel: str,
    title: str,
) -> None:
    """Shared cumulative-step line plot (coverage, L1, …)."""
    _apply_light_plot_style(ax)
    ax.plot(
        x_idx,
        y,
        color="#1f77b4",
        marker="s",
        markersize=5,
        linewidth=1.2,
        zorder=2,
    )
    ax.set_xticks(x_idx)
    ax.set_xticklabels(step_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, color="black")
    ax.set_xlabel("step (panel added)", color="black")
    ax.set_title(title, color="black", fontsize=10)


def _slug_filename(s: str) -> str:
    t = re.sub(r"[^\w\-.]+", "_", str(s).strip())
    t = re.sub(r"_+", "_", t).strip("_")
    return t[:128] if t else "item"


def _slide_name_from_args_txt(npy_path: Path, args_txt_path: str | None) -> str | None:
    """Read ``id`` / ``slide_name`` / … from ``args.txt`` next to the MSI cube (same as other repo tools)."""
    if args_txt_path:
        p = Path(to_absolute_path(str(args_txt_path)))
    else:
        p = npy_path.parent / "args.txt"
    if not p.is_file():
        logger.info("No args.txt at %s — outputs will not include a slide tag", p)
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace").replace("\\", "/")
        ns = eval(text, {"Namespace": Namespace})
    except Exception as e:
        logger.warning("Could not parse args.txt at %s: %s", p, e)
        return None
    for key in ("id", "slide_name", "slide", "name"):
        if hasattr(ns, key):
            v = getattr(ns, key)
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


def _continuous_dice_bar_colors(labels: list[str]) -> list[str]:
    """Highlight aggregate row (label contains ``agg``) in orange."""
    return ["#ff7f0e" if "agg" in str(lb).lower() else "#1f77b4" for lb in labels]


def _slide_prefixed_basename(slide_tag: str, filename: str) -> str:
    """Prefix a config filename with ``{slide_tag}__`` for top-level outputs (subdir already encodes slide)."""
    base = Path(filename).name
    if not slide_tag:
        return base
    return f"{slide_tag}__{base}"


def _save_panel_png(path: Path, img: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(img)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").convert("RGB").save(path)
    else:
        Image.fromarray(arr.astype(np.uint8, copy=False)).save(path)


def _save_diagram_pixel_coverage(
    dest: Path,
    *,
    coverages: list[float],
    step_labels: list[str],
    dpi: int,
) -> None:
    n_steps = len(step_labels)
    x_idx = np.arange(n_steps, dtype=np.float64)
    fig, ax_cov = plt.subplots(figsize=(10.0, 6.0), facecolor="white")
    fig.patch.set_facecolor("white")
    _plot_step_line_axis(
        ax_cov,
        x_idx,
        coverages,
        step_labels,
        ylabel="pixel coverage",
        title="Pixel coverage vs cumulative panels (max path)",
    )
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)


def _save_diagram_l1_energy(
    dest: Path,
    *,
    l1_sums: list[float],
    step_labels: list[str],
    dpi: int,
) -> None:
    n_steps = len(step_labels)
    x_idx = np.arange(n_steps, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.0, 6.0), facecolor="white")
    fig.patch.set_facecolor("white")
    _plot_step_line_axis(
        ax,
        x_idx,
        l1_sums,
        step_labels,
        ylabel="total edge energy (L1 on valid)",
        title="Total edge energy (L1 on valid)",
    )
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)


def _save_diagram_continuous_dice(
    dest: Path,
    *,
    labels: list[str],
    vals: list[float],
    title: str,
    dpi: int,
    figsize: tuple[float, float],
    bar_colors: list[str] | None = None,
) -> None:
    fig, ax_c = plt.subplots(figsize=figsize, facecolor="white")
    fig.patch.set_facecolor("white")
    _apply_light_plot_style(ax_c)
    cols = bar_colors if bar_colors is not None and len(bar_colors) == len(labels) else ["#1f77b4"] * len(labels)
    ax_c.bar(
        np.arange(len(labels)),
        vals,
        color=cols,
        edgecolor="#cccccc",
        linewidth=0.5,
        zorder=2,
    )
    ax_c.set_xticks(np.arange(len(labels)))
    ax_c.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax_c.set_title(title, color="black", fontsize=10)
    ax_c.set_ylabel("Continuous Dice", color="black")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)


def main(cfg: DictConfig) -> None:
    npy_path = Path(to_absolute_path(str(cfg.input.npy_path)))
    norm = _resolve_normalization_mode(getattr(cfg.data, "normalization", "tic"))
    msi = _load_msi(npy_path, transpose_msi=bool(cfg.data.transpose_msi), normalization=norm)
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty.")

    base_out = Path(cfg.output.out_dir) if cfg.output.out_dir else Path(HydraConfig.get().run.dir)
    use_slide = bool(getattr(cfg.output, "use_slide_name_from_args_txt", True))
    args_txt_override = OmegaConf.select(cfg, "input.args_txt_path")
    slide_raw = _slide_name_from_args_txt(npy_path, args_txt_override) if use_slide else None
    slide_tag = _slug_filename(slide_raw) if slide_raw else ""
    out_root = base_out / slide_tag if slide_tag else base_out
    out_root.mkdir(parents=True, exist_ok=True)
    if slide_raw:
        logger.info("Slide label from args.txt: %r → output subdir %s", slide_raw, slide_tag)

    k_list = OmegaConf.to_container(getattr(cfg.nmf, "k_list", []), resolve=True)
    if not isinstance(k_list, (list, tuple)) or len(k_list) == 0:
        raise ValueError("nmf.k_list must be a non-empty list")
    ks = [int(k) for k in k_list]
    for k in ks:
        if k < 3:
            raise ValueError(f"TOP3 requires k>=3, got {k}")

    o = cfg.output
    ncols = int(getattr(o, "gallery_columns", 4))
    cell_max = _cfg_value_to_py(getattr(o, "cell_max_size", None))
    cell_max_seq = cell_max if isinstance(cell_max, (list, tuple)) else None
    col_pad = int(getattr(o, "column_padding", 8))
    cell_gap = int(getattr(o, "cell_gap", 8))
    row_gap = int(getattr(o, "row_gap", 12))
    row_lab = int(getattr(o, "row_label_height", 34))
    split_gap = int(getattr(o, "split_gallery_gap", 24))
    hide_edges = bool(getattr(o, "hide_edge_panels", False)) or bool(getattr(o, "hide_edges", False))
    edge_panel = str(getattr(o, "edge_panel", "edges")).strip().lower()
    cmap_name = getattr(o, "edge_colormap", None)
    cmap_name = str(cmap_name).strip() if cmap_name else ""

    eval_cfg = cfg.evaluation
    thr_mode = str(getattr(eval_cfg, "threshold_mode", "otsu"))
    thr_pct = float(getattr(eval_cfg, "threshold_percentile", 90.0))
    thr_val = float(getattr(eval_cfg, "threshold_value", 0.5))
    otsu_pre = bool(getattr(eval_cfg, "otsu_pre_equalize", False))
    otsu_m = str(getattr(eval_cfg, "otsu_equalize_method", "hist"))
    otsu_clip = float(getattr(eval_cfg, "otsu_clahe_clip_limit", 2.0))
    otsu_tile = int(getattr(eval_cfg, "otsu_clahe_tile_grid_size", 8))
    sq_cont = bool(getattr(eval_cfg, "square_before_continuous_metrics", False))

    ld_win = int(getattr(getattr(o, "local_dice", None), "window", 3) or 3)
    ld_stretch = bool(getattr(getattr(o, "local_dice", None), "display_percentile_stretch", True))
    ld_lo = float(getattr(getattr(o, "local_dice", None), "stretch_low_percentile", 1.0))
    ld_hi = float(getattr(getattr(o, "local_dice", None), "stretch_high_percentile", 99.0))

    append_top3_msi_tile = bool(getattr(o, "append_top3_msi_tile", True))
    save_assets = bool(getattr(o, "save_individual_assets", True))
    panels_dir = out_root / str(getattr(o, "panels_dir", "panels"))
    diagrams_dir = out_root / str(getattr(o, "diagrams_dir", "diagrams"))

    logger.info("HD nmf_component_edges (reference)...")
    hd_cfg = getattr(cfg, "hd_edges", None)
    hd_primary, _hd_linf = _highd_edge_maps_nmf_component_edges(msi, valid_mask, hd_cfg)
    hd_n = _spatial_norm_viz(cfg, hd_primary, valid_mask)

    top3_cells: list[np.ndarray] = []
    edge_cells: list[np.ndarray] = []
    titles = [f"TOP3_NMF_k{k}" for k in ks]
    dice_rows: list[dict[str, Any]] = []
    raw_edges: list[np.ndarray] = []
    viz_edges_per_k: list[np.ndarray] = []

    for k in ks:
        rgb_u8, v_edge_f = _one_k_top3_nmf_rgb_and_viz_edge(cfg, msi, valid_mask, k)
        disp = _display_rgb(cfg, rgb_u8)
        disp = _maybe_downscale(disp, cell_max_seq)
        top3_cells.append(disp)

        viz_edges_per_k.append(v_edge_f)
        v_n = _spatial_norm_viz(cfg, v_edge_f, valid_mask)
        raw_edges.append(np.asarray(_raw_sobel_edge(cfg, rgb_u8, valid_mask), dtype=np.float32))

        h_met, v_met = _edges_for_continuous_metrics(hd_n, v_n, square=sq_cont)
        dice = _compute_continuous_dice(h_met, v_met, valid_mask)
        dice_rows.append({"k": k, "continuous_dice": dice})

        if not hide_edges:
            if edge_panel == "local_dice":
                viz_bin = _binary_from_threshold(
                    v_n,
                    valid_mask,
                    thr_mode,
                    thr_pct,
                    thr_val,
                    otsu_pre_equalize=otsu_pre,
                    otsu_equalize_method=otsu_m,
                    otsu_clahe_clip_limit=otsu_clip,
                    otsu_clahe_tile_grid_size=otsu_tile,
                )
                hd_bin = _binary_from_threshold(
                    hd_n,
                    valid_mask,
                    thr_mode,
                    thr_pct,
                    thr_val,
                    otsu_pre_equalize=otsu_pre,
                    otsu_equalize_method=otsu_m,
                    otsu_clahe_clip_limit=otsu_clip,
                    otsu_clahe_tile_grid_size=otsu_tile,
                )
                ld = _local_dice_heatmap_f32(viz_bin, hd_bin, valid_mask, ld_win)
                if ld_stretch and np.any(valid_mask):
                    lo = float(np.percentile(ld[valid_mask], ld_lo))
                    hi = float(np.percentile(ld[valid_mask], ld_hi))
                    if hi > lo + 1e-8:
                        ld = (np.clip(ld, lo, hi) - lo) / (hi - lo)
                    else:
                        ld = np.zeros_like(ld)
                ld_u8 = _to_uint8_gray01(ld)
                if cmap_name and cmap_name.lower() not in ("none", ""):
                    ld_rgb = _gray_u8_to_rgb_colormap(ld_u8, cmap_name)
                else:
                    ld_rgb = np.stack([ld_u8, ld_u8, ld_u8], axis=-1)
                edge_cells.append(_maybe_downscale(ld_rgb, cell_max_seq))
            else:
                ev = v_n
                ev_u8 = _to_uint8_gray01(ev)
                if cmap_name and cmap_name.lower() not in ("none", ""):
                    e_rgb = _gray_u8_to_rgb_colormap(ev_u8, cmap_name)
                else:
                    e_rgb = np.stack([ev_u8, ev_u8, ev_u8], axis=-1)
                edge_cells.append(_maybe_downscale(e_rgb, cell_max_seq))

    # TOP3 on raw MSI (last gallery tile) + Dice for that row
    rgb_msi_top = _top3_rgb_from_msi(cfg, msi)
    raw_msi_edge = np.asarray(_raw_sobel_edge(cfg, rgb_msi_top, valid_mask), dtype=np.float32)
    if append_top3_msi_tile:
        disp_msi = _display_rgb(cfg, rgb_msi_top)
        disp_msi = _maybe_downscale(disp_msi, cell_max_seq)
        top3_cells.append(disp_msi)
        titles.append(str(getattr(o, "top3_msi_tile_title", "TOP3 (MSI)")))
        v_edge_msi = _edge_map_from_top3_rgb(cfg, rgb_msi_top, valid_mask)
        v_n_msi = _spatial_norm_viz(cfg, v_edge_msi, valid_mask)
        h_m2, v_m2 = _edges_for_continuous_metrics(hd_n, v_n_msi, square=sq_cont)
        dice_msi = _compute_continuous_dice(h_m2, v_m2, valid_mask)
        dice_rows.append({"k": "full MSI", "continuous_dice": dice_msi})

    v_agg_raw = np.max(np.stack(viz_edges_per_k, axis=0), axis=0).astype(np.float32, copy=False)
    v_agg_n = _spatial_norm_viz(cfg, v_agg_raw, valid_mask)
    h_agg, v_agg = _edges_for_continuous_metrics(hd_n, v_agg_n, square=sq_cont)
    dice_agg_max = _compute_continuous_dice(h_agg, v_agg, valid_mask)
    dice_rows.append({"k": "agg (max)", "continuous_dice": dice_agg_max})

    # Optional: grayscale aggregate edge row (off by default; use TOP3 tile last instead)
    agg_cfg = getattr(o, "aggregate_all_edges", None)
    agg_cells: list[np.ndarray] = []
    agg_titles: list[str] = []
    if agg_cfg is not None and bool(getattr(agg_cfg, "enabled", False)):
        modes = _cfg_value_to_py(getattr(agg_cfg, "modes", None))
        if not isinstance(modes, (list, tuple)) or len(modes) == 0:
            modes = ["max"]
        stack = np.stack(raw_edges + [raw_msi_edge], axis=0)
        for m in modes:
            mm = str(m).strip().lower()
            if mm == "max":
                merged = np.max(stack, axis=0)
            elif mm == "mean":
                merged = np.mean(stack, axis=0)
            else:
                continue
            merged_n = _spatial_norm_viz(cfg, merged, valid_mask)
            u8 = _to_uint8_gray01(merged_n)
            if cmap_name and cmap_name.lower() not in ("none", ""):
                agg_rgb = _gray_u8_to_rgb_colormap(u8, cmap_name)
            else:
                agg_rgb = np.stack([u8, u8, u8], axis=-1)
            agg_cells.append(_maybe_downscale(agg_rgb, cell_max_seq))
            agg_titles.append(f"aggregate_all_{m}")

    # --- Diagnostics: cumulative pixel coverage (max path) + optional continuous Dice in one light figure ---
    diag_png: Path | None = None
    diag_csv: Path | None = None
    diag_cfg = getattr(o, "diagnostics", None)
    diag_enabled = diag_cfg is not None and bool(getattr(diag_cfg, "enabled", True))
    cd_cfg = getattr(o, "continuous_dice_chart", None)
    cd_enabled = cd_cfg is not None and bool(getattr(cd_cfg, "enabled", True))
    combine_dice_panel = bool(getattr(diag_cfg, "combine_continuous_dice_in_panel", True)) if diag_cfg else False
    combined_ok = (
        diag_enabled
        and diag_cfg is not None
        and cd_enabled
        and cd_cfg is not None
        and combine_dice_panel
    )

    cov_frac = float(getattr(diag_cfg, "coverage_frac_of_max", 0.1)) if diag_cfg is not None else 0.1
    step_edges = raw_edges + [raw_msi_edge]
    step_labels = [f"k={k}" for k in ks] + ["full MSI"]
    coverages: list[float] = []
    l1_on_valid: list[float] = []
    cum = None
    vm = valid_mask
    for r in step_edges:
        cum = r if cum is None else np.maximum(cum, r)
        vmax = float(np.max(cum[vm])) if np.any(vm) else 0.0
        tau = cov_frac * vmax
        coverages.append(float(np.mean((cum > tau) & vm)) if np.any(vm) else 0.0)
        pos = np.maximum(np.asarray(cum[vm], dtype=np.float64), 0.0)
        l1_on_valid.append(float(np.sum(pos)))
    n_steps = len(step_labels)

    if diag_enabled and diag_cfg is not None:
        diag_dpi = int(getattr(diag_cfg, "dpi", 150))
        diag_csv = out_root / _slide_prefixed_basename(
            slide_tag, str(getattr(diag_cfg, "csv", "gallery_edge_diagnostics.csv"))
        )
        with diag_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "step_label", "pixel_coverage", "total_edge_l1_on_valid"])
            for i in range(n_steps):
                w.writerow([i + 1, step_labels[i], coverages[i], l1_on_valid[i]])
        logger.info("Wrote diagnostics CSV: %s", diag_csv)

    cd_csv: Path | None = None
    cd_png: Path | None = None
    if cd_enabled and cd_cfg is not None:
        cd_csv = out_root / _slide_prefixed_basename(
            slide_tag, str(getattr(cd_cfg, "csv", "gallery_continuous_dice.csv"))
        )
        with cd_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["k", "continuous_dice"])
            w.writeheader()
            for row in dice_rows:
                w.writerow(row)
        logger.info("Wrote continuous Dice CSV: %s", cd_csv)

    # Figures: combined (coverage | L1 energy | Dice) in light style, or separate files if not combined_ok
    if combined_ok and diag_cfg is not None and cd_cfg is not None:
        diag_dpi = int(getattr(diag_cfg, "dpi", 150))
        w_fig = float(getattr(diag_cfg, "coverage_dice_fig_width", 14.0))
        h_fig = float(getattr(diag_cfg, "coverage_dice_fig_height", 4.0))
        ww = float(getattr(diag_cfg, "combined_three_panel_fig_width", w_fig * 1.5))
        fig_b, (ax_cov, ax_l1, ax_dice) = plt.subplots(1, 3, figsize=(ww, h_fig), facecolor="white")
        fig_b.patch.set_facecolor("white")
        x_idx = np.arange(n_steps, dtype=np.float64)
        _plot_step_line_axis(
            ax_cov,
            x_idx,
            coverages,
            step_labels,
            ylabel="pixel coverage",
            title="Pixel coverage vs cumulative panels (max path)",
        )
        _plot_step_line_axis(
            ax_l1,
            x_idx,
            l1_on_valid,
            step_labels,
            ylabel="total edge energy (L1 on valid)",
            title="Total edge energy (L1 on valid)",
        )

        labels = [str(r["k"]) for r in dice_rows]
        vals = [float(r["continuous_dice"]) for r in dice_rows]
        bcols = _continuous_dice_bar_colors(labels)
        _apply_light_plot_style(ax_dice)
        ax_dice.bar(
            np.arange(len(labels)),
            vals,
            color=bcols,
            edgecolor="#cccccc",
            linewidth=0.5,
            zorder=2,
        )
        ax_dice.set_xticks(np.arange(len(labels)))
        ax_dice.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax_dice.set_ylabel("Continuous Dice", color="black")
        ax_dice.set_title(
            str(getattr(cd_cfg, "title", "Continuous Dice (soft viz vs HD nmf_component_edges)")),
            color="black",
            fontsize=10,
        )

        fig_b.tight_layout()
        diag_png = out_root / _slide_prefixed_basename(
            slide_tag, str(getattr(diag_cfg, "png", "gallery_edge_diagnostics.png"))
        )
        fig_b.savefig(diag_png, dpi=diag_dpi, facecolor="white", edgecolor="none")
        plt.close(fig_b)
        cd_png = diag_png
        logger.info("Wrote combined coverage + L1 + Dice panel: %s", diag_png)
    elif diag_enabled and diag_cfg is not None:
        fig_d, (ax_cov, ax_l1) = plt.subplots(1, 2, figsize=(14.0, 6.0), facecolor="white")
        fig_d.patch.set_facecolor("white")
        x_idx = np.arange(n_steps, dtype=np.float64)
        _plot_step_line_axis(
            ax_cov,
            x_idx,
            coverages,
            step_labels,
            ylabel="pixel coverage",
            title="Pixel coverage vs cumulative panels (max path)",
        )
        _plot_step_line_axis(
            ax_l1,
            x_idx,
            l1_on_valid,
            step_labels,
            ylabel="total edge energy (L1 on valid)",
            title="Total edge energy (L1 on valid)",
        )
        fig_d.tight_layout()
        diag_png = out_root / _slide_prefixed_basename(
            slide_tag, str(getattr(diag_cfg, "png", "gallery_edge_diagnostics.png"))
        )
        fig_d.savefig(diag_png, dpi=int(getattr(diag_cfg, "dpi", 150)), facecolor="white", edgecolor="none")
        plt.close(fig_d)
        logger.info("Wrote diagnostics: %s", diag_png)

    if cd_enabled and cd_cfg is not None and not combined_ok:
        labels = [str(r["k"]) for r in dice_rows]
        vals = [float(r["continuous_dice"]) for r in dice_rows]
        bcols = _continuous_dice_bar_colors(labels)
        fig_c, ax_c = plt.subplots(figsize=tuple(getattr(cd_cfg, "figsize", [12.0, 4.0])), facecolor="white")
        fig_c.patch.set_facecolor("white")
        _apply_light_plot_style(ax_c)
        ax_c.bar(np.arange(len(labels)), vals, color=bcols, edgecolor="#cccccc", linewidth=0.5, zorder=2)
        ax_c.set_xticks(np.arange(len(labels)))
        ax_c.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax_c.set_title(str(getattr(cd_cfg, "title", "Continuous Dice")), color="black", fontsize=10)
        ax_c.set_ylabel("Continuous Dice", color="black")
        fig_c.tight_layout()
        cd_png = out_root / _slide_prefixed_basename(
            slide_tag, str(getattr(cd_cfg, "png", "gallery_continuous_dice.png"))
        )
        fig_c.savefig(cd_png, dpi=int(getattr(cd_cfg, "dpi", 150)), facecolor="white", edgecolor="none")
        plt.close(fig_c)
        logger.info("Wrote continuous Dice: %s", cd_png)

    if save_assets:
        diagrams_dir.mkdir(parents=True, exist_ok=True)
        if diag_enabled and diag_cfg is not None and coverages:
            _save_diagram_pixel_coverage(
                diagrams_dir / "pixel_coverage.png",
                coverages=coverages,
                step_labels=step_labels,
                dpi=int(getattr(diag_cfg, "dpi", 150)),
            )
        if diag_enabled and diag_cfg is not None and l1_on_valid:
            _save_diagram_l1_energy(
                diagrams_dir / "total_edge_l1_on_valid.png",
                l1_sums=l1_on_valid,
                step_labels=step_labels,
                dpi=int(getattr(diag_cfg, "dpi", 150)),
            )
        if cd_enabled and cd_cfg is not None and dice_rows:
            dlabels = [str(r["k"]) for r in dice_rows]
            dvals = [float(r["continuous_dice"]) for r in dice_rows]
            dbcols = _continuous_dice_bar_colors(dlabels)
            _save_diagram_continuous_dice(
                diagrams_dir / "continuous_dice_bars.png",
                labels=dlabels,
                vals=dvals,
                title=str(
                    getattr(
                        cd_cfg,
                        "title",
                        "Continuous Dice (soft viz vs HD nmf_component_edges)",
                    )
                ),
                dpi=int(getattr(cd_cfg, "dpi", 150)),
                figsize=tuple(getattr(cd_cfg, "figsize", [12.0, 4.0])),
                bar_colors=dbcols,
            )
        if combined_ok and diag_png is not None and diag_png.exists():
            shutil.copy2(diag_png, diagrams_dir / "coverage_l1_dice_combined.png")
        if diag_csv is not None and diag_csv.exists():
            shutil.copy2(diag_csv, diagrams_dir / diag_csv.name)
        if cd_csv is not None and cd_csv.exists():
            shutil.copy2(cd_csv, diagrams_dir / cd_csv.name)
        logger.info("Saved diagram assets under %s", diagrams_dir)

        panels_dir.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for img, title in zip(top3_cells, titles):
            _save_panel_png(panels_dir / f"panel_{_slug_filename(title)}.png", img)
            n_saved += 1
        if not hide_edges and edge_cells:
            for k, img in zip(ks, edge_cells):
                _save_panel_png(
                    panels_dir / f"panel_{edge_panel}_k{k}.png",
                    img,
                )
                n_saved += 1
        if agg_cells:
            for img, t in zip(agg_cells, agg_titles):
                _save_panel_png(panels_dir / f"panel_{_slug_filename(t)}.png", img)
                n_saved += 1
        logger.info("Saved %d panel image(s) under %s", n_saved, panels_dir)

    # --- Composite gallery PNG ---
    grid_top = _paste_grid(
        top3_cells,
        ncols=ncols,
        column_padding=col_pad,
        cell_gap=cell_gap,
        row_gap=row_gap,
        row_label_height=row_lab,
        titles=titles,
    )
    composite = grid_top
    if agg_cells:
        grid_agg = _paste_grid(
            agg_cells,
            ncols=min(len(agg_cells), ncols),
            column_padding=col_pad,
            cell_gap=cell_gap,
            row_gap=row_gap,
            row_label_height=row_lab,
            titles=agg_titles,
        )
        composite = _vstack_images(composite, grid_agg, split_gap)

    if not hide_edges and edge_cells:
        grid_bot = _paste_grid(
            edge_cells,
            ncols=ncols,
            column_padding=col_pad,
            cell_gap=cell_gap,
            row_gap=row_gap,
            row_label_height=row_lab,
            titles=[f"{edge_panel}_k={k}" for k in ks],
        )
        composite = _vstack_images(composite, grid_bot, split_gap)

    below = str(getattr(o, "below_top3", "all")).strip().lower()
    hide_diag_strip = bool(getattr(o, "hide_diagnostics_when_hiding_edges", False))
    if hide_edges and hide_diag_strip:
        below = "none"

    inc_diag = diag_png is not None and bool(getattr(diag_cfg, "include_in_gallery", True)) if diag_cfg else False
    inc_cd = cd_png is not None and bool(getattr(cd_cfg, "include_in_gallery", True)) if cd_cfg else False
    if below == "none":
        inc_diag = inc_cd = False
    elif below == "diagnostics":
        inc_cd = False
    elif below == "continuous_dice":
        inc_diag = False

    strip_paths: list[Path] = []
    if inc_diag and diag_png is not None:
        strip_paths.append(diag_png)
    if inc_cd and cd_png is not None and cd_png not in strip_paths:
        strip_paths.append(cd_png)

    strips: list[np.ndarray] = []
    from PIL import Image

    for p in strip_paths:
        im = np.asarray(Image.open(p).convert("RGB"))
        if im.shape[1] != composite.shape[1] and im.shape[1] > 0:
            scale = composite.shape[1] / im.shape[1]
            nh = max(1, int(round(im.shape[0] * scale)))
            im = cv2.resize(im, (composite.shape[1], nh), interpolation=cv2.INTER_AREA)
        strips.append(im)

    for s in strips:
        composite = _vstack_images(composite, s, split_gap // 2)

    gallery_path = out_root / _slide_prefixed_basename(
        slide_tag, str(getattr(o, "gallery_png", "gallery_top3_nmf_edges.png"))
    )
    from PIL import Image

    Image.fromarray(composite).save(gallery_path)
    logger.info("Wrote gallery: %s", gallery_path)


@hydra.main(version_base=None, config_path="configs", config_name="gallery_top3_nmf_edges")
def hydra_entry(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(cfg.logging.level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
