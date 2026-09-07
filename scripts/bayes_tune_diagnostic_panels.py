#!/usr/bin/env python3
"""
Build per-slide diagnostic panels from a completed bayes_tune_mics_lmc run.

Each panel is a tight 2-row mosaic:
  - H&E overview and SoLaCE reference
  - Top-ranked MiCS tiles (no captions; scores live in diagnostic_meta.json)

H&E display (``he.display_mode``):
  - ``letterbox`` — fit whole slide overview into the tile (no MSI registration)
  - ``register`` — warp H&E into MSI pixel space via ``registration.json`` / mask_only

Also writes ``diagnostic_panel_equalized.png`` with LAB-L equalized MiCS tiles.

Publication export (``output.export_scale``, default 4):
  - ``diagnostic_panel_hires.png`` / ``diagnostic_panel_equalized_hires.png``
  - Per-tile ``tiles_hires/`` (H&E re-letterboxed at export size; MiCS/SoLaCE Lanczos-upscaled)
  - PNG tagged at ``output.dpi`` (default 400)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

from msi_visual.extract.he_thumbnail import extract_he_from_wsi_at_dpi
from msi_visual.registration.he_msi import (
    apply_registration_to_he,
    letterbox_matrix,
    load_registration,
    register_he_to_msi,
    registration_matches_he_msi,
    resolve_registration_path,
    save_registration,
    warp_he_to_msi,
)
from msi_visual.registration.hydra_cfg import registration_kwargs_from_cfg, resolve_mics_from_registration_cfg
from msi_visual.registration.paths import msi_sample_id, resolve_he_thumbnail, resolve_he_wsi, slide_key_from_npy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

SOLACE_LABEL = "SoLaCE"


def _load_he_rgb_for_panel(
    slide_key: str,
    he_cfg: DictConfig,
    *,
    min_long_side_px: int | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load H&E RGB from OpenSlide (preferred) or cached thumbnail PNG."""
    meta: dict[str, Any] = {"he_source": "none"}
    if not bool(he_cfg.enabled):
        return None, meta

    source = str(OmegaConf.select(he_cfg, "source", default="auto")).strip().lower()
    thumb_path = resolve_he_thumbnail(slide_key, he_cfg, resolve_path=_resolve_path)
    wsi_path = resolve_he_wsi(slide_key, he_cfg, resolve_path=_resolve_path)

    target_dpi_raw = OmegaConf.select(he_cfg, "target_dpi")
    target_dpi = float(target_dpi_raw) if target_dpi_raw is not None else 400.0
    wsi_max_raw = OmegaConf.select(he_cfg, "wsi_max_size")
    wsi_max = int(wsi_max_raw) if wsi_max_raw is not None else None
    max_dpi_raw = OmegaConf.select(he_cfg, "max_dpi")
    max_dpi = float(max_dpi_raw) if max_dpi_raw is not None else None

    if source in ("wsi", "auto") and wsi_path is not None and wsi_path.is_file():
        try:
            rgb, wsi_meta = extract_he_from_wsi_at_dpi(
                wsi_path,
                target_dpi=target_dpi,
                max_size=wsi_max,
                min_long_side=min_long_side_px,
                max_dpi=max_dpi,
            )
            meta.update(wsi_meta)
            meta["he_source"] = "wsi"
            meta["wsi_path"] = str(wsi_path)
            meta["thumbnail_path"] = str(thumb_path) if thumb_path else None
            logger.info(
                "%s: H&E from WSI %s at %.0f dpi (min_long=%s) -> %dx%d",
                slide_key,
                wsi_path.name,
                float(wsi_meta.get("target_dpi", target_dpi)),
                min_long_side_px,
                rgb.shape[1],
                rgb.shape[0],
            )
            return rgb, meta
        except Exception as exc:
            logger.warning("%s: OpenSlide H&E failed (%s); falling back to thumbnail", slide_key, exc)

    if thumb_path is not None and thumb_path.is_file():
        rgb = _load_rgb(thumb_path)
        meta.update({"he_source": "thumbnail", "thumbnail_path": str(thumb_path)})
        return rgb, meta

    return None, meta


def _panel_capacity(cfg: DictConfig) -> tuple[int, int, int]:
    nrows = max(1, int(cfg.output.panel_rows))
    ncols = max(1, int(cfg.output.panel_columns))
    return ncols, nrows, ncols * nrows


def _mics_slots_for_panel(cfg: DictConfig, *, n_refs: int) -> int:
    _, _, capacity = _panel_capacity(cfg)
    explicit = OmegaConf.select(cfg, "top_k", default=None)
    if explicit is not None and str(explicit).strip().lower() not in ("", "null", "none", "~"):
        return max(0, int(explicit))
    return max(0, capacity - int(n_refs))


def _resolve_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _equalize_rgb_u8_lab_l(rgb: np.ndarray) -> np.ndarray:
    """Histogram-equalize LAB luminance (same as generate_visualization_panel)."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.equalizeHist(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def _load_msi_valid_mask(npy_path: Path, tune_cfg: DictConfig) -> np.ndarray:
    from debug_edge_maps import _load_msi

    normalization = str(OmegaConf.select(tune_cfg, "data.normalization", default="tic"))
    transpose_msi = bool(OmegaConf.select(tune_cfg, "data.transpose_msi", default=False))
    vol = _load_msi(npy_path, transpose_msi=transpose_msi, normalization=normalization)
    return vol.sum(axis=-1) > 0


def _mask_outside_msi_tissue(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    background_rgb: tuple[int, int, int],
) -> np.ndarray:
    out = np.asarray(rgb, dtype=np.uint8).copy()
    bg = np.asarray(background_rgb, dtype=np.uint8)
    out[~valid_mask] = bg
    return out


def _resolve_he_for_panel(
    *,
    slide_key: str,
    he_path: Path | None,
    he_raw: np.ndarray | None,
    npy_path: Path,
    img_idx: int,
    tune_cfg: DictConfig,
    cfg: DictConfig,
    out_hw: tuple[int, int],
    border_rgb: tuple[int, int, int],
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Prepare H&E for the panel tile (letterbox overview or MSI registration)."""
    meta: dict[str, Any] = {"he_path": str(he_path) if he_path else None, "alignment": "none"}
    if he_raw is None or he_path is None:
        return None, meta

    ref_h, ref_w = out_hw
    he_h, he_w = he_raw.shape[:2]
    display_raw = OmegaConf.select(cfg, "he.display_mode")
    if display_raw is None:
        reg_on = bool(OmegaConf.select(cfg, "registration.enabled", default=False))
        display_mode = "register" if reg_on else "letterbox"
    else:
        display_mode = str(display_raw).strip().lower()
    meta["display_mode"] = display_mode

    if display_mode in ("letterbox", "overview", "fit"):
        warp = letterbox_matrix(he_h, he_w, ref_h, ref_w)
        he_rgb = warp_he_to_msi(he_raw, warp, (ref_h, ref_w), border_value=border_rgb)
        meta["alignment"] = "letterbox"
        return he_rgb, meta

    if display_mode not in ("register", "registration", "reg"):
        raise ValueError(
            f"he.display_mode must be letterbox or register, got {display_mode!r}"
        )

    msi_hw = (
        (int(valid_mask.shape[0]), int(valid_mask.shape[1]))
        if valid_mask is not None
        else (ref_h, ref_w)
    )
    reg_enabled = bool(OmegaConf.select(cfg, "registration.enabled", default=True))
    reg_filename = str(OmegaConf.select(cfg, "registration.registration_filename", default="registration.json"))
    auto_register = bool(OmegaConf.select(cfg, "registration.auto_register", default=True))
    reregister_stale = bool(OmegaConf.select(cfg, "registration.reregister_if_stale", default=True))

    reg_path = resolve_registration_path(he_path, reg_filename, slide_key=slide_key)
    reg = load_registration(reg_path) if reg_enabled else None

    if reg is None and reg_enabled:
        legacy_path = resolve_registration_path(he_path, reg_filename)
        if legacy_path.is_file() and legacy_path != reg_path:
            legacy_reg = load_registration(legacy_path)
            if legacy_reg is not None and registration_matches_he_msi(
                legacy_reg,
                slide_key=slide_key,
                he_hw=(he_h, he_w),
                msi_hw=msi_hw,
            ):
                reg = legacy_reg

    needs_register = reg is None or (
        reregister_stale
        and not registration_matches_he_msi(
            reg,
            slide_key=slide_key,
            he_hw=(he_h, he_w),
            msi_hw=msi_hw,
        )
    )

    if needs_register and auto_register and reg_enabled:
        sample_id = msi_sample_id(slide_key) or slide_key
        rank_raw = OmegaConf.select(cfg, "registration.rank_config")
        rank_path = _resolve_path(str(rank_raw)) if rank_raw else None
        strategy = str(OmegaConf.select(cfg, "registration.strategy", default="mask_only")).strip().lower()
        mics_rgb, mics_source = None, None
        if strategy == "multistage":
            mics_rgb, mics_source = resolve_mics_from_registration_cfg(
                cfg,
                npy_path=npy_path,
                img_idx=img_idx,
                target_hw=out_hw,
                resolve_path=_resolve_path,
            )
        reg = register_he_to_msi(
            npy_path=npy_path,
            he_path=he_path,
            slide_key=slide_key,
            sample_id=sample_id,
            he_rgb=he_raw,
            **registration_kwargs_from_cfg(
                cfg,
                rank_path=rank_path,
                tune_cfg=tune_cfg,
                mics_rgb=mics_rgb,
                mics_source=mics_source,
            ),
        )
        save_registration(reg, reg_path)
        meta["registration_computed"] = True
        meta["mics_source"] = mics_source
        logger.info(
            "%s: registered H&E (%dx%d) -> MSI (%dx%d) method=%s iou=%.3f",
            slide_key,
            he_w,
            he_h,
            msi_hw[1],
            msi_hw[0],
            reg.method,
            float(reg.correlation or 0.0),
        )

    if reg is not None:
        he_rgb = apply_registration_to_he(he_raw, reg, (ref_h, ref_w), border_value=border_rgb)
        meta.update(
            {
                "alignment": reg.method,
                "registration_path": str(reg_path),
                "registration_success": reg.success,
                "registration_correlation": reg.correlation,
                "rotation_deg": reg.rotation_deg,
                "registration_stages": reg.stages,
            }
        )
    elif bool(OmegaConf.select(cfg, "registration.fallback_letterbox", default=True)):
        warp = letterbox_matrix(he_h, he_w, ref_h, ref_w)
        he_rgb = warp_he_to_msi(he_raw, warp, (ref_h, ref_w), border_value=border_rgb)
        meta["alignment"] = "letterbox_affine"
    else:
        return None, meta

    if valid_mask is not None and bool(OmegaConf.select(cfg, "registration.mask_to_msi_tissue", default=True)):
        he_rgb = _mask_outside_msi_tissue(he_rgb, valid_mask, background_rgb=border_rgb)
    return he_rgb, meta


def _parse_pipe_paths(raw: str) -> list[Path]:
    if not raw or str(raw).strip() in ("", "nan"):
        return []
    return [Path(p.strip()) for p in str(raw).split("|") if p.strip()]


def _pick_path_for_img(paths: list[Path], img_idx: int) -> Path | None:
    needle = f"__img{img_idx}__"
    for p in paths:
        if needle in p.name:
            return p
    if 0 <= img_idx < len(paths):
        return paths[img_idx]
    return None


def _load_tune_cfg(tune_run_dir: Path) -> DictConfig:
    cfg_path = tune_run_dir / "config_resolved.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    return OmegaConf.load(str(cfg_path))


def _load_trial_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _safe_float(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _metrics_for_img(row: dict[str, str], img_idx: int) -> dict[str, float]:
    raw = row.get("metrics_by_image_json", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    out: dict[str, float] = {}
    for key in ("continuous_dice", "luminance_rms_contrast", "lab_chroma_entropy"):
        vals = data.get(key)
        if isinstance(vals, list) and 0 <= img_idx < len(vals):
            v = _safe_float(vals[img_idx])
            if v is not None:
                out[key] = v
    return out


def _composite_score_for_image(metrics: dict[str, float], tune_cfg: DictConfig) -> float:
    from scripts.bayes_tune_mics_lmc import (
        _parse_metric_bounds,
        _parse_objective_weights,
        _weighted_linear_objective,
    )

    weights = _parse_objective_weights(tune_cfg)
    bounds = _parse_metric_bounds(tune_cfg)
    means = {
        "continuous_dice": float(metrics.get("continuous_dice", float("nan"))),
        "luminance_rms_contrast": float(metrics.get("luminance_rms_contrast", float("nan"))),
        "lab_chroma_entropy": float(metrics.get("lab_chroma_entropy", float("nan"))),
    }
    score, _ = _weighted_linear_objective(means, weights, bounds)
    return float(score)


def _top_trials_for_image(
    rows: list[dict[str, str]],
    img_idx: int,
    tune_cfg: DictConfig,
    *,
    top_k: int,
) -> list[tuple[float, dict[str, str], Path, dict[str, float]]]:
    scored: list[tuple[float, dict[str, str], Path, dict[str, float]]] = []
    for row in rows:
        metrics = _metrics_for_img(row, img_idx)
        if not metrics:
            continue
        mics_path = _pick_path_for_img(_parse_pipe_paths(row.get("image_paths", "")), img_idx)
        if mics_path is None or not mics_path.is_file():
            continue
        score = _composite_score_for_image(metrics, tune_cfg)
        if not np.isfinite(score):
            continue
        scored.append((score, row, mics_path, metrics))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: max(1, int(top_k))]


def _resolve_npy_paths_from_tune_cfg(tune_cfg: DictConfig) -> list[Path]:
    from scripts.bayes_tune_mics_lmc import _resolve_npy_paths

    return _resolve_npy_paths(tune_cfg)


def _compute_hd_edge_rgb(
    npy_path: Path,
    tune_cfg: DictConfig,
    tune_run_dir: Path,
    slide_key: str,
) -> np.ndarray:
    from debug_edge_maps import _gray_u8_to_rgb_colormap, _load_msi, _to_uint8_gray01
    from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method

    edge_cmap = str(OmegaConf.select(tune_cfg, "output.edge_colormap", default="PuBu"))
    cached = tune_run_dir / f"soft_landmark_contrast_reference__{slide_key}__{edge_cmap}.png"
    if cached.is_file():
        return _load_rgb(cached)

    rank_cfg = OmegaConf.load(to_absolute_path(str(tune_cfg.rank_config)))
    normalization = str(OmegaConf.select(tune_cfg, "data.normalization", default="tic"))
    transpose_msi = bool(OmegaConf.select(tune_cfg, "data.transpose_msi", default=False))
    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    if not bool(getattr(multi, "enabled", False)):
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    spatial_enabled = bool(getattr(rank_cfg.spatial_normalization, "enabled", True))

    msi_vol = _load_msi(npy_path, transpose_msi=transpose_msi, normalization=normalization)
    valid_mask = msi_vol.sum(axis=-1) > 0
    hd_edge_n = _compute_hd_edge_n_for_method(
        rank_cfg,
        msi_vol,
        valid_mask,
        getattr(rank_cfg, "hd_edges", None),
        "soft_landmark_contrast",
        sigmas,
        aggregation,
        spatial_enabled,
    )
    return _gray_u8_to_rgb_colormap(_to_uint8_gray01(hd_edge_n), edge_cmap)


def _panel_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _add_ref_caption(rgb: np.ndarray, text: str, *, bar_h: int, font_size: int) -> np.ndarray:
    """Small bottom caption bar for reference tiles (H&E, SoLaCE)."""
    img = np.asarray(rgb, dtype=np.uint8).copy()
    h, w = img.shape[:2]
    bar_h = min(max(16, int(bar_h)), h // 3)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (12, 12, 14), thickness=-1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0.0, img)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = _panel_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = max(0, (w - tw) // 2)
    ty = h - bar_h + max(0, (bar_h - th) // 2)
    draw.text((tx, ty), text, fill=(235, 235, 238), font=font)
    return np.asarray(pil, dtype=np.uint8)


def _compose_panel_grid(
    tiles: list[tuple[np.ndarray, str | None]],
    *,
    ncols: int,
    nrows: int,
    gap: int,
    bg: tuple[int, int, int],
    ref_caption_h: int,
    ref_font_size: int,
) -> np.ndarray:
    """Stitch equal-size tiles into a gap-separated grid; MiCS tiles have no caption."""
    if not tiles:
        raise ValueError("Panel has no tiles")
    gap = max(0, int(gap))
    cell_h, cell_w = tiles[0][0].shape[:2]
    canvas_w = ncols * cell_w + (ncols - 1) * gap
    canvas_h = nrows * cell_h + (nrows - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    expected = ncols * nrows
    if len(tiles) != expected:
        raise ValueError(f"Expected {expected} tiles for {ncols}x{nrows} grid, got {len(tiles)}")

    for idx, (rgb, label) in enumerate(tiles):
        r, c = divmod(idx, ncols)
        y0 = r * (cell_h + gap)
        x0 = c * (cell_w + gap)
        tile = rgb
        if label:
            tile = _add_ref_caption(tile, label, bar_h=ref_caption_h, font_size=ref_font_size)
        canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = tile
    return canvas


def _save_panel_png(panel_rgb: np.ndarray, out_path: Path, *, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(panel_rgb, dtype=np.uint8)).save(out_path, dpi=(int(dpi), int(dpi)))


def _resolve_export_scale(cfg: DictConfig, cell_h: int, cell_w: int) -> float:
    scale = float(OmegaConf.select(cfg, "output.export_scale", default=1.0))
    scale = max(1.0, scale)
    min_side_raw = OmegaConf.select(cfg, "output.cell_min_side")
    if min_side_raw is not None:
        want = max(1, int(min_side_raw))
        native_min = max(1, min(int(cell_h), int(cell_w)))
        scale = max(scale, float(want) / float(native_min))
    return scale


def _resolve_he_min_long_side(
    cfg: DictConfig,
    *,
    ref_h: int,
    ref_w: int,
    export_scale: float,
) -> int | None:
    """Pixels on the long WSI-read edge so letterbox uses downscale, not upscale."""
    if not bool(OmegaConf.select(cfg.he, "match_export_resolution", default=True)):
        return None
    margin = float(OmegaConf.select(cfg.he, "resolution_margin", default=1.05))
    native_long = max(int(ref_h), int(ref_w))
    target_hw = (
        int(round(ref_h * export_scale)),
        int(round(ref_w * export_scale)),
    )
    hires_long = max(target_hw)
    return max(native_long, hires_long, int(round(native_long * margin)))


def _upscale_rgb(rgb: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Upscale RGB with Lanczos (for publication exports)."""
    th, tw = int(out_hw[0]), int(out_hw[1])
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape[0] == th and rgb.shape[1] == tw:
        return rgb
    return cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LANCZOS4)


def _compose_kwargs_scaled(cfg: DictConfig, *, scale: float, bg: tuple[int, int, int]) -> dict[str, Any]:
    gap = int(cfg.output.grid_gap_px)
    ref_caption_h = int(cfg.output.ref_caption_height)
    ref_font_size = int(cfg.output.ref_font_size)
    if scale <= 1.0 + 1e-6:
        return {
            "gap": gap,
            "ref_caption_h": ref_caption_h,
            "ref_font_size": ref_font_size,
        }
    return {
        "gap": max(0, int(round(gap * scale))),
        "ref_caption_h": max(16, int(round(ref_caption_h * scale))),
        "ref_font_size": max(8, int(round(ref_font_size * scale))),
    }


def _build_hires_grid_tiles(
    grid_tiles: list[tuple[np.ndarray, str | None]],
    *,
    target_hw: tuple[int, int],
    he_hires: np.ndarray | None,
) -> list[tuple[np.ndarray, str | None]]:
    """Rebuild tiles at export resolution (H&E re-letterboxed; others Lanczos-upscaled)."""
    out: list[tuple[np.ndarray, str | None]] = []
    for i, (rgb, label) in enumerate(grid_tiles):
        if i == 0 and he_hires is not None:
            out.append((he_hires, label))
        else:
            out.append((_upscale_rgb(rgb, target_hw), label))
    return out


def _save_hires_tiles(
    tiles: list[tuple[np.ndarray, str | None]],
    out_dir: Path,
    *,
    ref_tile_count: int,
    dpi: int,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    ref_names = ("he", "solace")
    for i, (rgb, _) in enumerate(tiles):
        if i < ref_tile_count:
            stem = ref_names[i] if i < len(ref_names) else f"ref{i}"
            name = f"{stem}_hires.png"
        else:
            rank = i - ref_tile_count + 1
            name = f"mics_rank{rank:02d}_hires.png"
        path = out_dir / name
        Image.fromarray(rgb).save(path, dpi=(int(dpi), int(dpi)))
        paths.append(str(path))
    return paths


def _build_panel_for_slide(
    *,
    npy_path: Path,
    img_idx: int,
    tune_cfg: DictConfig,
    tune_run_dir: Path,
    trial_rows: list[dict[str, str]],
    cfg: DictConfig,
    out_dir: Path,
) -> dict[str, Any]:
    slide_key = slide_key_from_npy(npy_path)
    slide_out = out_dir / slide_key
    slide_out.mkdir(parents=True, exist_ok=True)

    ncols, nrows, capacity = _panel_capacity(cfg)

    ranked_all = _top_trials_for_image(trial_rows, img_idx, tune_cfg, top_k=capacity)
    if not ranked_all:
        raise ValueError(f"No trial MiCS PNGs found for image index {img_idx}")

    ref_rgb = _load_rgb(ranked_all[0][2])
    ref_h, ref_w = ref_rgb.shape[:2]
    gap = int(cfg.output.grid_gap_px)
    bg = tuple(int(v) for v in OmegaConf.to_container(cfg.output.background_rgb, resolve=True))

    hd_rgb = None
    if bool(cfg.compute_hd_edges):
        hd_rgb = _compute_hd_edge_rgb(npy_path, tune_cfg, tune_run_dir, slide_key)
        if hd_rgb.shape[:2] != (ref_h, ref_w):
            hd_rgb = cv2.resize(hd_rgb, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
        Image.fromarray(hd_rgb).save(slide_out / "solace_reference.png")

    valid_mask = _load_msi_valid_mask(npy_path, tune_cfg)

    export_scale = _resolve_export_scale(cfg, ref_h, ref_w)
    he_min_long = _resolve_he_min_long_side(cfg, ref_h=ref_h, ref_w=ref_w, export_scale=export_scale)

    he_path = resolve_he_thumbnail(slide_key, cfg.he, resolve_path=_resolve_path)
    he_raw, he_load_meta = _load_he_rgb_for_panel(slide_key, cfg.he, min_long_side_px=he_min_long)
    if he_raw is not None and bool(OmegaConf.select(cfg.he, "save_wsi_read", default=True)):
        Image.fromarray(he_raw).save(slide_out / "he_wsi_read.png")
    he_rgb, reg_meta = _resolve_he_for_panel(
        slide_key=slide_key,
        he_path=he_path,
        he_raw=he_raw,
        npy_path=npy_path,
        img_idx=img_idx,
        tune_cfg=tune_cfg,
        cfg=cfg,
        out_hw=(ref_h, ref_w),
        border_rgb=bg,
        valid_mask=valid_mask,
    )
    reg_meta = {**he_load_meta, **reg_meta}
    if he_rgb is not None:
        he_out_name = (
            "he_registered.png"
            if str(reg_meta.get("display_mode", "letterbox")) in ("register", "registration", "reg")
            or reg_meta.get("registration_success")
            else "he_overview.png"
        )
        Image.fromarray(he_rgb).save(slide_out / he_out_name)

    mics_slots = _mics_slots_for_panel(
        cfg,
        n_refs=int(he_rgb is not None) + int(hd_rgb is not None),
    )
    ranked = ranked_all[:mics_slots]
    if len(ranked) < mics_slots:
        logger.warning(
            "%s: only %d/%d ranked MiCS tiles; empty grid cells will be padded",
            slide_key,
            len(ranked),
            mics_slots,
        )

    if mics_slots <= 0:
        raise ValueError(f"Grid {ncols}x{nrows} has no room for MiCS after reference tiles")

    ref_label_he = str(OmegaConf.select(cfg, "output.ref_label_he", default="H&E"))
    ref_label_solace = str(OmegaConf.select(cfg, "output.ref_label_solace", default=SOLACE_LABEL))
    show_ref_labels = bool(OmegaConf.select(cfg, "output.show_ref_labels", default=True))
    if he_rgb is not None and reg_meta.get("registration_success"):
        ref_label_he = str(OmegaConf.select(cfg, "output.ref_label_he_registered", default="H&E (reg.)"))
    elif he_rgb is not None and str(reg_meta.get("display_mode", "")).lower() in (
        "letterbox",
        "overview",
        "fit",
    ):
        ref_label_he = str(OmegaConf.select(cfg, "output.ref_label_he", default="H&E"))

    ref_tiles: list[tuple[np.ndarray, str | None]] = []
    if he_rgb is not None:
        ref_tiles.append((he_rgb, ref_label_he if show_ref_labels else None))
    if hd_rgb is not None:
        ref_tiles.append((hd_rgb, ref_label_solace if show_ref_labels else None))

    mics_tiles: list[tuple[np.ndarray, str | None]] = []
    top_entries: list[dict[str, Any]] = []
    for rank, (score, row, mics_path, metrics) in enumerate(ranked, start=1):
        mics_rgb = _load_rgb(mics_path)
        if mics_rgb.shape[:2] != (ref_h, ref_w):
            mics_rgb = cv2.resize(mics_rgb, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
        iter_s = row.get("iteration", "?")
        clusters = row.get("clusters", "?")
        dice = metrics.get("continuous_dice")
        top_entries.append(
            {
                "rank": rank,
                "composite_score": score,
                "iteration": int(float(iter_s)) if str(iter_s).isdigit() else iter_s,
                "clusters": clusters,
                "continuous_dice": dice,
                "mics_path": str(mics_path),
            }
        )
        mics_tiles.append((mics_rgb, None))

    grid_tiles = ref_tiles + mics_tiles
    n_refs = len(ref_tiles)

    n_padded = 0
    if len(grid_tiles) < capacity:
        n_padded = capacity - len(grid_tiles)
        logger.warning(
            "%s: padding %d empty cell(s) to fill %dx%d grid (%d refs + %d MiCS)",
            slide_key,
            n_padded,
            ncols,
            nrows,
            n_refs,
            len(ranked),
        )
        blank = np.full((ref_h, ref_w, 3), bg, dtype=np.uint8)
        for _ in range(n_padded):
            grid_tiles.append((blank, None))
    elif len(grid_tiles) > capacity:
        grid_tiles = grid_tiles[:capacity]

    ref_caption_h = int(cfg.output.ref_caption_height)
    ref_font_size = int(cfg.output.ref_font_size)
    dpi = int(cfg.output.dpi)
    compose_kw = dict(
        ncols=ncols,
        nrows=nrows,
        gap=gap,
        bg=bg,
        ref_caption_h=ref_caption_h,
        ref_font_size=ref_font_size,
    )

    panel_path = slide_out / str(cfg.output.panel_filename)
    panel_rgb = _compose_panel_grid(grid_tiles, **compose_kw)
    save_native = bool(OmegaConf.select(cfg, "output.save_native_panel", default=True))
    if save_native:
        _save_panel_png(panel_rgb, panel_path, dpi=dpi)

    equalized_panel_path: Path | None = None
    if save_native and bool(OmegaConf.select(cfg, "output.save_equalized_panel", default=True)):
        eq_tiles = [
            (_equalize_rgb_u8_lab_l(rgb), label) if label is None else (rgb, label)
            for rgb, label in grid_tiles
        ]
        equalized_panel_path = slide_out / str(
            OmegaConf.select(cfg, "output.equalized_panel_filename", default="diagnostic_panel_equalized.png")
        )
        _save_panel_png(_compose_panel_grid(eq_tiles, **compose_kw), equalized_panel_path, dpi=dpi)

    target_hw = (int(round(ref_h * export_scale)), int(round(ref_w * export_scale)))
    hires_panel_path: Path | None = None
    hires_equalized_panel_path: Path | None = None
    hires_tile_paths: list[str] = []

    if export_scale > 1.0 + 1e-6 and bool(OmegaConf.select(cfg, "output.save_hires_panel", default=True)):
        he_hires = None
        if he_raw is not None and he_path is not None and he_rgb is not None:
            he_hires, _ = _resolve_he_for_panel(
                slide_key=slide_key,
                he_path=he_path,
                he_raw=he_raw,
                npy_path=npy_path,
                img_idx=img_idx,
                tune_cfg=tune_cfg,
                cfg=cfg,
                out_hw=target_hw,
                border_rgb=bg,
                valid_mask=valid_mask,
            )
            he_hires_name = (
                "he_registered_hires.png"
                if str(reg_meta.get("display_mode", "letterbox")) in ("register", "registration", "reg")
                or reg_meta.get("registration_success")
                else "he_overview_hires.png"
            )
            Image.fromarray(he_hires).save(slide_out / he_hires_name)

        hires_tiles = _build_hires_grid_tiles(grid_tiles, target_hw=target_hw, he_hires=he_hires)
        hires_compose = {**compose_kw, **_compose_kwargs_scaled(cfg, scale=export_scale, bg=bg)}
        hires_mosaic = _compose_panel_grid(hires_tiles, **hires_compose)
        hires_panel_path = slide_out / str(
            OmegaConf.select(cfg, "output.hires_panel_filename", default="diagnostic_panel_hires.png")
        )
        _save_panel_png(hires_mosaic, hires_panel_path, dpi=dpi)

        if bool(OmegaConf.select(cfg, "output.save_equalized_panel", default=True)):
            eq_hires = [
                (_equalize_rgb_u8_lab_l(rgb), label) if label is None else (rgb, label)
                for rgb, label in hires_tiles
            ]
            hires_equalized_panel_path = slide_out / str(
                OmegaConf.select(
                    cfg,
                    "output.hires_equalized_panel_filename",
                    default="diagnostic_panel_equalized_hires.png",
                )
            )
            _save_panel_png(
                _compose_panel_grid(eq_hires, **hires_compose),
                hires_equalized_panel_path,
                dpi=dpi,
            )

        if bool(OmegaConf.select(cfg, "output.save_hires_tiles", default=True)):
            tiles_dir = slide_out / str(
                OmegaConf.select(cfg, "output.hires_tiles_subdir", default="tiles_hires")
            )
            hires_tile_paths = _save_hires_tiles(
                hires_tiles,
                tiles_dir,
                ref_tile_count=n_refs,
                dpi=dpi,
            )

        logger.info(
            "%s: hires panel cells %dx%d (scale=%.2f), mosaic %dx%d px @ %d dpi",
            slide_key,
            target_hw[1],
            target_hw[0],
            export_scale,
            hires_mosaic.shape[1],
            hires_mosaic.shape[0],
            dpi,
        )

    if not save_native and hires_panel_path is not None:
        panel_path = hires_panel_path
        equalized_panel_path = hires_equalized_panel_path

    meta = {
        "slide_key": slide_key,
        "npy_path": str(npy_path),
        "panel_path": str(panel_path),
        "equalized_panel_path": str(equalized_panel_path) if equalized_panel_path else None,
        "hires_panel_path": str(hires_panel_path) if hires_panel_path else None,
        "hires_equalized_panel_path": str(hires_equalized_panel_path)
        if hires_equalized_panel_path
        else None,
        "hires_tile_paths": hires_tile_paths,
        "export": {
            "dpi": dpi,
            "export_scale": export_scale,
            "he_min_long_side": he_min_long,
            "he_wsi_read_shape": list(he_raw.shape[:2]) if he_raw is not None else None,
            "native_cell_hw": [ref_h, ref_w],
            "hires_cell_hw": list(target_hw) if export_scale > 1.0 else None,
        },
        "grid": {
            "columns": ncols,
            "rows": nrows,
            "capacity": capacity,
            "mics_slots": mics_slots,
            "padded_cells": n_padded,
        },
        "he_path": str(he_path) if he_path else None,
        "registration": reg_meta,
        "he_registered_path": str(slide_out / "he_registered.png") if he_rgb is not None else None,
        "top_trials": top_entries,
    }
    with (slide_out / "diagnostic_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return meta


@hydra.main(version_base=None, config_path="configs", config_name="bayes_tune_diagnostic_panels")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    tune_run_dir = _resolve_path(str(cfg.tune_run_dir))
    csv_path = tune_run_dir / "bayes_mics_trials.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")

    tune_cfg = _load_tune_cfg(tune_run_dir)
    npy_paths = _resolve_npy_paths_from_tune_cfg(tune_cfg)
    trial_rows = _load_trial_rows(csv_path)
    logger.info(
        "Loaded %d tuning trials; grid %dx%d",
        len(trial_rows),
        int(cfg.output.panel_columns),
        int(cfg.output.panel_rows),
    )

    out_raw = OmegaConf.select(cfg, "output.dir")
    out_dir = _resolve_path(str(out_raw)) if out_raw else tune_run_dir / "diagnostic_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for img_idx, npy_path in enumerate(npy_paths):
        logger.info("[%d/%d] %s", img_idx + 1, len(npy_paths), slide_key_from_npy(npy_path))
        try:
            meta = _build_panel_for_slide(
                npy_path=npy_path,
                img_idx=img_idx,
                tune_cfg=tune_cfg,
                tune_run_dir=tune_run_dir,
                trial_rows=trial_rows,
                cfg=cfg,
                out_dir=out_dir,
            )
            manifest.append(meta)
        except Exception:
            logger.exception("Failed panel for %s", npy_path)
            manifest.append({"npy_path": str(npy_path), "status": "error"})

    manifest_path = out_dir / "diagnostic_panels_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    logger.info("Wrote %d panel(s) under %s", len(manifest), out_dir)


if __name__ == "__main__":
    main()
