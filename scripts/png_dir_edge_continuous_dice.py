#!/usr/bin/env python3
"""
For each visualization under --png-dir, pair it to an MSI .npy using the same rules as
``list_dir_shapes.py`` (bucket by spatial HxW, then 1:N or stem-prefix assignment).

For each pair: compute HD edges from the MSI, RGB edges from the viz, and the continuous
Dice score; save a triple-panel PNG (viz | viz edges | HD edges + Dice).

Per MSI cube, also writes ``viz_edges_debug/<stem>/<stem>__dice_sorted_gallery.png``: a grid with
viz images on the first row, viz edges on the second, and HD edges in the last column, sorted by
Dice (best left). Toggle with ``gallery.enabled`` in the YAML.

Example:
  python scripts/png_dir_edge_continuous_dice.py \\
    --npy-dir "D:/data/cubes" --png-dir "D:/data/pngs" --out-dir "D:/out/edge_dice"

  # Or set ``paths`` in ``scripts/configs/png_dir_edge_continuous_dice.yaml`` and run:
  python scripts/png_dir_edge_continuous_dice.py --config scripts/configs/png_dir_edge_continuous_dice.yaml

Edge-related YAML defaults are merged from ``rank_visualizations_by_f1.yaml`` (see ``--base-config``).
The overlay lists ``data``, ``spatial_normalization``, ``edge_maps``, ``edge_detection``, ``hd_edges``,
``evaluation`` (``metric``: continuous_dice | f1 | gradient_ssim), ``debug``, and ``worker`` (edge
equalization for saved PNGs).

Paths from CLI override YAML.

At the end, prints mean continuous Dice per ``viz_type`` (parsed from each filename; benchmark
names use the segment after ``__r*__s*__sampling__``) and writes ``continuous_dice_by_viz_type.csv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from debug_edge_maps import (
    _apply_equalize_rgb_uint8,
    _edge_maps_multiscale,
    _equalize_gray_u8,
    _load_msi,
    _load_visualization,
    _normalize_map_percentile,
    _normalize_on_mask,
    _parse_edge_detection_cfg,
    _pca_project_msi_for_edges,
    _resolve_normalization_mode,
    _spatial_hw_msi_cube_file,
    _to_uint8_gray01,
)
from list_dir_shapes import _bucket_by_hw, _iter_files, _one_to_many_mapping
from rank_visualizations_by_f1 import _compute_viz_f1_worker, _edge_map_to_rgb

logger = logging.getLogger(__name__)

# Benchmark-style: ...__r{rep}__s{seed}__{sampling}__{method...}
_VIZ_TYPE_BENCHMARK_RE = re.compile(r"__r\d+__s\d+__[^_]+__(.+)$")
_VIZ_TYPE_INDEX_PREFIX_RE = re.compile(r"^\d+_(.+)$")


def _visualization_type_from_stem(stem: str) -> str:
    """
    Derive a visualization / method label from the filename stem.

    - Benchmark PNGs: take the segment after ``__r*__s*__sampling__`` (e.g. ``parametric_umap``).
    - Else if ``{digits}_rest``: drop the numeric prefix (e.g. ``0_Foo`` -> ``Foo``).
    - Else: use the full stem.
    """
    m = _VIZ_TYPE_BENCHMARK_RE.search(stem)
    if m:
        return m.group(1)
    m2 = _VIZ_TYPE_INDEX_PREFIX_RE.match(stem)
    if m2:
        return m2.group(1)
    return stem


def _print_viz_type_dice_summary(rows: list[dict[str, object]], *, out_dir: Path | None = None) -> None:
    """Mean continuous Dice per visualization type; print sorted by mean (desc). Optionally write CSV."""
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        d = row.get("continuous_dice")
        if d is None or d == "":
            continue
        try:
            v = float(d)
        except (TypeError, ValueError):
            continue
        vt = str(row.get("viz_type") or "").strip()
        if not vt:
            vt = _visualization_type_from_stem(Path(str(row.get("viz_name", ""))).stem)
        by_type[vt].append(v)

    if not by_type:
        print("\n=== Average continuous Dice by visualization type ===\n  (no valid scores)\n")
        return

    ranked = sorted(
        ((t, sum(vals) / len(vals), len(vals)) for t, vals in by_type.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    w_type = max(len(t) for t, _, _ in ranked)
    w_n = max(4, len(str(max(n for _, _, n in ranked))))

    print("\n=== Average continuous Dice by visualization type (sorted by mean, best first) ===")
    print(f"  {'n':>{w_n}}  {'mean Dice':>12}  type")
    for t, mean, n in ranked:
        print(f"  {n:{w_n}d}  {mean:12.6f}  {t}")

    if out_dir is not None:
        summary_path = out_dir / "continuous_dice_by_viz_type.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            wo = csv.writer(f)
            wo.writerow(["viz_type", "n_images", "mean_continuous_dice"])
            for t, mean, n in ranked:
                wo.writerow([t, n, f"{mean:.10g}"])
        print(f"\nWrote {summary_path}")


def _bucket_npy_msi(paths: list[Path], transpose_msi: bool) -> dict[tuple[int, int], list[Path]]:
    """Spatial (H, W) buckets for MSI cubes, consistent with _load_msi(..., transpose_msi=...)."""
    buckets: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for p in paths:
        try:
            hw = _spatial_hw_msi_cube_file(p, transpose_msi)
        except Exception as exc:
            logger.warning("skip %s when bucketing MSI | %s", p.name, exc)
            continue
        buckets[hw].append(p)
    for k in buckets:
        buckets[k].sort(key=lambda x: x.name.lower())
    return dict(buckets)


def _load_cfg(config_path: Path | None, base_config_path: Path | None) -> object:
    """
    Load config as ``OmegaConf.merge(base, overlay)``.

    Base defaults (HD / RGB edges, spatial norm, ``hd_edges``, ``edge_detection``, etc.) come from
    ``rank_visualizations_by_f1.yaml`` so they stay aligned with ``_compute_viz_f1_worker`` /
    ``debug_edge_maps.py``. The overlay file (default ``png_dir_edge_continuous_dice.yaml``) sets
    ``paths``, ``run``, ``gallery``, and any per-run overrides.
    """
    base_p = base_config_path or (_SCRIPTS / "configs" / "rank_visualizations_by_f1.yaml")
    if not base_p.is_file():
        raise FileNotFoundError(f"Base edge-pipeline config not found: {base_p}")
    base = OmegaConf.load(base_p)
    overlay_p = config_path if config_path is not None else (_SCRIPTS / "configs" / "png_dir_edge_continuous_dice.yaml")
    if not overlay_p.is_file():
        raise FileNotFoundError(f"Config overlay not found: {overlay_p}")
    overlay = OmegaConf.load(overlay_p)
    return OmegaConf.merge(base, overlay)


def _path_cli_or_yaml(cli: Path | None, cfg, yaml_key: str) -> Path | None:
    """CLI path wins; else ``yaml_key`` (e.g. paths.npy_dir). Null/empty in YAML means unset."""
    if cli is not None:
        return cli
    raw = OmegaConf.select(cfg, yaml_key)
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "~", "none"):
        return None
    return Path(s)


def _str_cli_or_yaml(cli: str | None, cfg, yaml_key: str, default: str) -> str:
    if cli is not None and str(cli).strip() != "":
        return str(cli)
    raw = OmegaConf.select(cfg, yaml_key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw)


def _int_cli_or_yaml(cli: int | None, cfg, yaml_key: str, default: int) -> int:
    if cli is not None:
        return int(cli)
    raw = OmegaConf.select(cfg, yaml_key)
    if raw is None:
        return default
    return int(raw)


def _skip_eq_from_cfg(cfg) -> bool:
    return bool(OmegaConf.select(cfg, "run.skip_eq")) or bool(
        OmegaConf.select(cfg, "input.skip_eq", default=False)
    )


def _hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = str(s).strip().lstrip("#")
    if len(s) == 6:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return 26, 29, 38


def _v_eval_and_viz_u8_for_gallery(
    vp: Path,
    target_hw: tuple[int, int],
    valid_mask: np.ndarray,
    sigmas: list[float],
    aggregation: str,
    edge_method_cfg: dict,
    eval_otsu_pre_equalize: bool,
    eval_otsu_equalize_method: str,
    eval_otsu_clahe_clip_limit: float,
    eval_otsu_clahe_tile_grid_size: int,
    allow_resize: bool,
    equalize_visualization: bool,
    equalize_visualization_method: str,
    equalize_visualization_clahe_clip_limit: float,
    equalize_visualization_clahe_tile_grid_size: int,
    spatial_low_percentile: float,
    spatial_high_percentile: float,
    spatial_normalization_enabled: bool,
    divide_by_255: bool,
    edge_cfg_override: dict | None,
    min_viz_edge_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Same viz + edge pipeline as ``_compute_viz_f1_worker`` (for gallery thumbnails)."""
    viz_rgb = _load_visualization(vp, target_hw=target_hw, allow_resize=allow_resize)
    if equalize_visualization:
        viz_rgb = _apply_equalize_rgb_uint8(
            viz_rgb,
            equalize_visualization_method,
            equalize_visualization_clahe_clip_limit,
            equalize_visualization_clahe_tile_grid_size,
        )
    edge_cfg = edge_method_cfg if edge_cfg_override is None else {**edge_method_cfg, **edge_cfg_override}
    if divide_by_255:
        viz_rgb = viz_rgb.astype(np.float32) / 255.0
    v_edge, _ = _edge_maps_multiscale(
        "rgb",
        viz_rgb,
        valid_mask,
        sigmas=sigmas,
        aggregation=aggregation,
        hd_cfg=None,
        edge_method_cfg=edge_cfg,
    )
    if divide_by_255:
        v_edge_n = np.asarray(v_edge, dtype=np.float32)
        v_edge_n[~valid_mask] = 0.0
    elif spatial_normalization_enabled:
        v_edge_n = _normalize_map_percentile(
            v_edge,
            valid_mask,
            low_pct=spatial_low_percentile,
            high_pct=spatial_high_percentile,
        )
    else:
        v_edge_n = _normalize_on_mask(v_edge, valid_mask)
    if eval_otsu_pre_equalize:
        v_eval = _equalize_gray_u8(
            _to_uint8_gray01(v_edge_n),
            enabled=True,
            method=eval_otsu_equalize_method,
            clahe_clip_limit=eval_otsu_clahe_clip_limit,
            clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
        ).astype(np.float32) / 255.0
        v_eval[~valid_mask] = 0.0
    else:
        v_eval = v_edge_n
    if min_viz_edge_percentile > 0 and np.any(valid_mask):
        thresh = float(np.percentile(v_eval[valid_mask], min_viz_edge_percentile))
        v_eval = np.where((valid_mask) & (v_eval >= thresh), v_eval, 0.0).astype(np.float32)
    if viz_rgb.dtype != np.uint8:
        viz_u8 = (np.clip(viz_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        viz_u8 = viz_rgb
    if viz_u8.ndim == 2:
        viz_u8 = np.stack([viz_u8] * 3, axis=-1)
    return viz_u8, v_eval


def _sort_viz_by_dice(
    viz_list: list[Path],
    results: list[dict],
) -> list[tuple[Path, dict, float]]:
    pairs: list[tuple[Path, dict, float]] = []
    for vp, r in zip(viz_list, results):
        d = r.get("f1")
        if d is None or (isinstance(d, float) and np.isnan(d)):
            key = float("-inf")
        else:
            key = float(d)
        pairs.append((vp, r, key))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _save_dice_sorted_gallery_png(
    out_path: Path,
    *,
    npy_stem: str,
    sorted_items: list[tuple[Path, dict, float]],
    valid_mask: np.ndarray,
    hd_edge_n: np.ndarray,
    target_hw: tuple[int, int],
    sigmas: list[float],
    aggregation: str,
    edge_method_cfg: dict,
    eval_otsu_pre_equalize: bool,
    eval_otsu_equalize_method: str,
    eval_otsu_clahe_clip_limit: float,
    eval_otsu_clahe_tile_grid_size: int,
    allow_resize: bool,
    equalize_visualization: bool,
    equalize_visualization_method: str,
    equalize_visualization_clahe_clip_limit: float,
    equalize_visualization_clahe_tile_grid_size: int,
    spatial_low_percentile: float,
    spatial_high_percentile: float,
    spatial_enabled: bool,
    divide_by_255: bool,
    edge_cfg_override: dict | None,
    display_gamma_val: float | None,
    min_viz_edge_percentile: float,
    gallery_cfg: object,
    equalize_edges_before_metric: bool = False,
    edge_colormap: str | None = None,
) -> None:
    """Row 1 = viz RGB, row 2 = viz edges; last column = HD edges (same height as both rows)."""
    from PIL import Image, ImageDraw, ImageFont

    if not sorted_items:
        return

    max_cell_w = int(OmegaConf.select(gallery_cfg, "max_cell_width", default=320))
    max_row_h = int(OmegaConf.select(gallery_cfg, "max_row_height", default=220))
    col_gap = int(OmegaConf.select(gallery_cfg, "column_gap", default=14))
    row_gap = int(OmegaConf.select(gallery_cfg, "row_gap", default=10))
    hd_max_w = int(OmegaConf.select(gallery_cfg, "hd_max_width", default=420))
    sep_w = int(OmegaConf.select(gallery_cfg, "hd_separator_width", default=4))
    header_h = int(OmegaConf.select(gallery_cfg, "header_bar_height", default=48))
    cap_h = int(OmegaConf.select(gallery_cfg, "caption_height", default=26))
    bg = _hex_to_rgb(str(OmegaConf.select(gallery_cfg, "background", default="#1a1d26")))
    sep_rgb = _hex_to_rgb(str(OmegaConf.select(gallery_cfg, "separator", default="#3d4450")))
    raw_max_scale = 1.414 if divide_by_255 else None

    hd_for_metric_col = np.asarray(hd_edge_n, dtype=np.float32)
    if equalize_edges_before_metric:
        hd_for_metric_col = _equalize_gray_u8(
            _to_uint8_gray01(hd_edge_n),
            enabled=True,
            method=eval_otsu_equalize_method,
            clahe_clip_limit=eval_otsu_clahe_clip_limit,
            clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
        ).astype(np.float32) / 255.0
        hd_for_metric_col[~valid_mask] = 0.0

    def _resize_to_box(arr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        im = Image.fromarray(np.asarray(arr, dtype=np.uint8).clip(0, 255), mode="RGB")
        w, h = im.size
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return np.asarray(im.resize((nw, nh), Image.Resampling.LANCZOS))

    packed: list[tuple[Path, float, np.ndarray, np.ndarray]] = []
    for vp, r, dice in sorted_items:
        if r.get("error"):
            continue
        try:
            viz_u8, v_eval = _v_eval_and_viz_u8_for_gallery(
                vp,
                target_hw,
                valid_mask,
                sigmas,
                aggregation,
                edge_method_cfg,
                eval_otsu_pre_equalize,
                eval_otsu_equalize_method,
                eval_otsu_clahe_clip_limit,
                eval_otsu_clahe_tile_grid_size,
                allow_resize,
                equalize_visualization,
                equalize_visualization_method,
                equalize_visualization_clahe_clip_limit,
                equalize_visualization_clahe_tile_grid_size,
                spatial_low_percentile,
                spatial_high_percentile,
                spatial_enabled,
                divide_by_255,
                edge_cfg_override,
                min_viz_edge_percentile,
            )
        except Exception as exc:
            logger.warning("gallery skip %s: %s", vp.name, exc)
            continue
        v_for_rgb = v_eval
        if equalize_edges_before_metric:
            v_for_rgb = _equalize_gray_u8(
                _to_uint8_gray01(v_eval),
                enabled=True,
                method=eval_otsu_equalize_method,
                clahe_clip_limit=eval_otsu_clahe_clip_limit,
                clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
            ).astype(np.float32) / 255.0
            v_for_rgb[~valid_mask] = 0.0
        edge_rgb = _edge_map_to_rgb(
            v_for_rgb,
            valid_mask,
            raw_max_scale=raw_max_scale,
            display_gamma=display_gamma_val,
            colormap=edge_colormap,
        )
        packed.append((vp, dice, _resize_to_box(viz_u8, max_cell_w, max_row_h), _resize_to_box(edge_rgb, max_cell_w, max_row_h)))

    if not packed:
        logger.warning("No columns for gallery %s", out_path.name)
        return

    n = len(packed)
    row1_h = max(p[2].shape[0] for p in packed)
    row2_h = max(p[3].shape[0] for p in packed)
    col_ws: list[int] = []
    cols_viz: list[np.ndarray] = []
    cols_edge: list[np.ndarray] = []
    for _vp, _dice, v1, e1 in packed:
        cw = max(v1.shape[1], e1.shape[1])
        col_ws.append(cw)
        im = Image.fromarray(v1, mode="RGB")
        cols_viz.append(np.asarray(im.resize((cw, row1_h), Image.Resampling.LANCZOS)))
        im = Image.fromarray(e1, mode="RGB")
        cols_edge.append(np.asarray(im.resize((cw, row2_h), Image.Resampling.LANCZOS)))

    mid_h = row1_h + row_gap + row2_h
    left_w = sum(col_ws) + col_gap * max(0, n - 1)
    left_h = cap_h + mid_h + cap_h

    hd_rgb = _edge_map_to_rgb(
        hd_for_metric_col,
        valid_mask,
        raw_max_scale=raw_max_scale,
        display_gamma=display_gamma_val,
        colormap=edge_colormap,
    )
    hd_im = Image.fromarray(hd_rgb, mode="RGB")
    hd_h_target = mid_h
    hd_w = min(hd_max_w, max(1, int(hd_im.width * hd_h_target / hd_im.height)))
    hd_im = hd_im.resize((hd_w, hd_h_target), Image.Resampling.LANCZOS)
    hd_arr = np.asarray(hd_im)

    total_w = left_w + sep_w + col_gap + hd_arr.shape[1]
    total_h = header_h + left_h

    canvas = Image.new("RGB", (total_w, total_h), color=bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_cap = ImageFont.truetype("arial.ttf", 11)
        font_hd = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font_title = font_cap = font_hd = ImageFont.load_default()

    title = f"{npy_stem}  |  continuous Dice (sorted, best left)"
    draw.text((16, 14), title, fill=(230, 232, 238), font=font_title)

    y0 = header_h
    x = 0
    for i in range(n):
        vp, dice, _, _ = packed[i]
        d_str = f"{dice:.4f}" if dice > float("-inf") else "nan"
        cap_im = Image.new("RGB", (col_ws[i], cap_h), color=bg)
        cdraw = ImageDraw.Draw(cap_im)
        cdraw.text((4, 4), f"Dice {d_str}", fill=(180, 200, 255), font=font_cap)
        canvas.paste(cap_im, (x, y0))

        canvas.paste(Image.fromarray(cols_viz[i], mode="RGB"), (x, y0 + cap_h))
        canvas.paste(Image.fromarray(cols_edge[i], mode="RGB"), (x, y0 + cap_h + row1_h + row_gap))

        fn = vp.name
        if len(fn) > 42:
            fn = fn[:20] + "..." + fn[-18:]
        fn_im = Image.new("RGB", (col_ws[i], cap_h), color=bg)
        fdraw = ImageDraw.Draw(fn_im)
        fdraw.text((4, 4), fn, fill=(140, 145, 155), font=font_cap)
        canvas.paste(fn_im, (x, y0 + cap_h + mid_h))

        x += col_ws[i] + col_gap

    sep = Image.new("RGB", (sep_w, left_h), color=sep_rgb)
    canvas.paste(sep, (left_w, y0))

    hx = left_w + sep_w + col_gap
    hd_cap = Image.new("RGB", (hd_arr.shape[1], cap_h), color=bg)
    hdraw = ImageDraw.Draw(hd_cap)
    hdraw.text((8, 4), "HD edges (reference)", fill=(255, 210, 140), font=font_hd)
    canvas.paste(hd_cap, (hx, y0))
    canvas.paste(Image.fromarray(hd_arr, mode="RGB"), (hx, y0 + cap_h))

    fn_bot = Image.new("RGB", (hd_arr.shape[1], cap_h), color=bg)
    bdraw = ImageDraw.Draw(fn_bot)
    bdraw.text((8, 4), "MSI distance map", fill=(140, 145, 155), font=font_cap)
    canvas.paste(fn_bot, (hx, y0 + cap_h + hd_h_target))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    logger.info("Saved gallery %s", out_path)


def _compute_hd_edges_for_npy(cfg, npy_path: Path):
    """Returns (msi, valid_mask, hd_scalar, hd_edge_n, sigmas, aggregation, edge_method_cfg)."""
    normalization = _resolve_normalization_mode(getattr(cfg.data, "normalization", "tic"))
    msi = _load_msi(
        npy_path,
        transpose_msi=bool(cfg.data.transpose_msi),
        normalization=normalization,
    )
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError(f"MSI valid mask is empty for {npy_path}")
    hd_scalar = np.linalg.norm(np.asarray(msi, dtype=np.float32), axis=-1)
    hd_cfg = getattr(cfg, "hd_edges", None)
    edge_method_cfg = _parse_edge_detection_cfg(cfg)
    ms_cfg = getattr(cfg, "edge_maps", None)
    sigmas = [float(v) for v in list(getattr(getattr(ms_cfg, "multi_scale", None), "sigmas", [0.0])) or [0.0]]
    aggregation = str(getattr(getattr(ms_cfg, "multi_scale", None), "aggregation", "mean"))
    msi_for_edges = _pca_project_msi_for_edges(msi, valid_mask, hd_cfg)
    hd_result = _edge_maps_multiscale(
        "hd",
        msi_for_edges,
        valid_mask,
        sigmas=sigmas,
        aggregation=aggregation,
        hd_cfg=hd_cfg,
        edge_method_cfg=edge_method_cfg,
    )
    hd_edge = hd_result[0]
    spatial_enabled = bool(getattr(cfg.spatial_normalization, "enabled", True))
    if spatial_enabled:
        hd_edge_n = _normalize_map_percentile(
            hd_edge,
            valid_mask,
            low_pct=float(cfg.spatial_normalization.low_percentile),
            high_pct=float(cfg.spatial_normalization.high_percentile),
        )
    else:
        hd_edge_n = _normalize_on_mask(hd_edge, valid_mask)
    return msi, valid_mask, hd_scalar, hd_edge_n, sigmas, aggregation, edge_method_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy-dir", type=Path, default=None, help="Directory of MSI .npy cubes (or set paths.npy_dir in YAML)")
    ap.add_argument("--png-dir", type=Path, default=None, help="Directory of visualization images (or set paths.png_dir in YAML)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (or set paths.out_dir in YAML)")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML overlay (default: png_dir_edge_continuous_dice.yaml), merged on top of --base-config",
    )
    ap.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Base YAML for edge/HD settings (default: rank_visualizations_by_f1.yaml)",
    )
    ap.add_argument(
        "--viz-ext",
        type=str,
        default=None,
        help="Comma-separated extensions under png-dir (default: run.viz_ext in YAML)",
    )
    ap.add_argument("--skip-eq", action="store_true", help="Skip files with 'eq' in the stem (overrides YAML)")
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel workers for viz scoring (default: run.n_jobs in YAML)",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.config, args.base_config)

    log_level = str(OmegaConf.select(cfg, "logging.level", default="INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    npy_dir = _path_cli_or_yaml(args.npy_dir, cfg, "paths.npy_dir")
    png_dir = _path_cli_or_yaml(args.png_dir, cfg, "paths.png_dir")
    out_dir = _path_cli_or_yaml(args.out_dir, cfg, "paths.out_dir")
    if npy_dir is None or png_dir is None or out_dir is None:
        print(
            "Set paths.npy_dir, paths.png_dir, and paths.out_dir in the YAML, or pass "
            "--npy-dir, --png-dir, and --out-dir.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not npy_dir.is_dir():
        print(f"npy-dir is not a directory: {npy_dir}", file=sys.stderr)
        sys.exit(1)
    if not png_dir.is_dir():
        print(f"png-dir is not a directory: {png_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    viz_ext_str = _str_cli_or_yaml(
        args.viz_ext,
        cfg,
        "run.viz_ext",
        "png,jpg,jpeg,tif,tiff,webp,bmp,npy",
    )
    n_jobs = _int_cli_or_yaml(args.n_jobs, cfg, "run.n_jobs", 1)
    skip_eq = args.skip_eq or _skip_eq_from_cfg(cfg)

    viz_exts = {"." + x.strip().lstrip(".").lower() for x in viz_ext_str.split(",") if x.strip()}
    npy_paths = _iter_files(npy_dir, {".npy"})
    viz_paths = _iter_files(png_dir, viz_exts)
    if skip_eq:
        viz_paths = [p for p in viz_paths if "eq" not in p.stem.lower()]

    transpose_msi = bool(cfg.data.transpose_msi)
    npy_buckets = _bucket_npy_msi(npy_paths, transpose_msi)
    viz_buckets = _bucket_by_hw(viz_paths)

    all_hw = sorted(set(npy_buckets) | set(viz_buckets))
    shared = set(npy_buckets) & set(viz_buckets)
    print(f"MSI buckets: {len(npy_buckets)} spatial sizes | viz buckets: {len(viz_buckets)} | shared: {len(shared)}")

    rows: list[dict[str, object]] = []
    t0 = time.perf_counter()

    edge_method_cfg = _parse_edge_detection_cfg(cfg)

    eval_cfg = getattr(cfg, "evaluation", None)
    eval_mode = str(getattr(eval_cfg, "threshold_mode", "otsu")) if eval_cfg else "otsu"
    eval_percentile = float(getattr(eval_cfg, "threshold_percentile", 85.0)) if eval_cfg else 85.0
    eval_value = float(getattr(eval_cfg, "threshold_value", 0.5)) if eval_cfg else 0.5
    eval_otsu_pre_equalize = bool(getattr(eval_cfg, "otsu_pre_equalize", False)) if eval_cfg else False
    eval_otsu_equalize_method = str(getattr(eval_cfg, "otsu_equalize_method", "hist")) if eval_cfg else "hist"
    eval_otsu_clahe_clip_limit = float(getattr(eval_cfg, "otsu_clahe_clip_limit", 2.0)) if eval_cfg else 2.0
    eval_otsu_clahe_tile_grid_size = int(getattr(eval_cfg, "otsu_clahe_tile_grid_size", 8)) if eval_cfg else 8
    min_viz_edge_percentile = float(getattr(eval_cfg, "min_viz_edge_percentile", 0.0)) if eval_cfg else 0.0
    square_before_continuous_metrics = (
        bool(getattr(eval_cfg, "square_before_continuous_metrics", False)) if eval_cfg else False
    )
    equalize_edges_before_metric = bool(
        OmegaConf.select(cfg, "evaluation.equalize_edges_before_metric", default=False)
    )
    eval_metric = str(OmegaConf.select(cfg, "evaluation.metric", default="continuous_dice")).strip().lower()

    allow_resize = bool(getattr(cfg.input, "allow_resize", True))
    inp = getattr(cfg, "input", None)
    equalize_visualization = bool(getattr(inp, "equalize_visualization", False)) if inp else False
    equalize_visualization_method = str(getattr(inp, "equalize_visualization_method", "hist")) if inp else "hist"
    equalize_visualization_clahe_clip_limit = (
        float(getattr(inp, "equalize_visualization_clahe_clip_limit", 2.0)) if inp else 2.0
    )
    equalize_visualization_clahe_tile_grid_size = (
        int(getattr(inp, "equalize_visualization_clahe_tile_grid_size", 8)) if inp else 8
    )
    divide_by_255 = bool(getattr(cfg.input, "divide_by_255", False))
    edge_cfg_override = {"rgb_scale_01": True, "rgb_no_normalize": True} if divide_by_255 else None
    spatial_enabled = bool(getattr(cfg.spatial_normalization, "enabled", True)) and not divide_by_255
    debug_cfg = getattr(cfg, "debug", None)
    display_gamma_val = None
    if divide_by_255 and debug_cfg is not None:
        g = getattr(debug_cfg, "display_gamma", None)
        if g is not None and 0 < float(g) < 1:
            display_gamma_val = float(g)

    save_viz_edges = bool(OmegaConf.select(cfg, "debug.save_viz_edges", default=True))
    save_viz_edges_binary = bool(OmegaConf.select(cfg, "debug.save_viz_edges_binary", default=False))
    save_agreement_overlay = bool(OmegaConf.select(cfg, "debug.save_agreement_overlay", default=False))
    save_npy = bool(OmegaConf.select(cfg, "debug.save_npy", default=False))
    save_edge_equalized = bool(OmegaConf.select(cfg, "debug.save_edge_equalized", default=False))
    edge_colormap = OmegaConf.select(cfg, "debug.edge_colormap", default=None)
    if edge_colormap is not None:
        edge_colormap = str(edge_colormap).strip()
        if edge_colormap.lower() in ("null", "none", "~", ""):
            edge_colormap = None
    local_dice_window = int(OmegaConf.select(cfg, "debug.local_dice_window", default=0) or 0)
    if local_dice_window < 0:
        local_dice_window = 0

    save_edge_equalize_method = str(OmegaConf.select(cfg, "worker.save_edge_equalize_method", default="hist"))
    save_edge_clahe_clip_limit = float(OmegaConf.select(cfg, "worker.save_edge_clahe_clip_limit", default=2.0))
    save_edge_clahe_tile_grid_size = int(OmegaConf.select(cfg, "worker.save_edge_clahe_tile_grid_size", default=8))

    for hw in all_hw:
        na = npy_buckets.get(hw, [])
        va = viz_buckets.get(hw, [])
        if not na or not va:
            continue
        by_npy, unassigned = _one_to_many_mapping(na, va)
        if unassigned:
            print(f"  (H,W)={hw}: unassigned viz ({len(unassigned)}): " + ", ".join(p.name for p in unassigned[:8]) + (" ..." if len(unassigned) > 8 else ""))

        for npy_path in sorted(by_npy, key=lambda p: p.name.lower()):
            viz_list = by_npy[npy_path]
            if not viz_list:
                continue
            viz_edges_dir = out_dir / "viz_edges_debug" / npy_path.stem
            viz_edges_dir.mkdir(parents=True, exist_ok=True)

            logger.info("HD edges for MSI %s (%d viz)", npy_path.name, len(viz_list))
            try:
                msi, valid_mask, hd_scalar, hd_edge_n, sigmas, aggregation, _em = _compute_hd_edges_for_npy(cfg, npy_path)
            except Exception as exc:
                logger.error("Failed HD edges for %s: %s", npy_path, exc)
                for vp in viz_list:
                    rows.append(
                        {
                            "npy_stem": npy_path.stem,
                            "npy_path": str(npy_path),
                            "viz_name": vp.name,
                            "viz_path": str(vp),
                            "viz_type": _visualization_type_from_stem(vp.stem),
                            "continuous_dice": "",
                            "error": str(exc),
                            "triple_png": "",
                        }
                    )
                continue

            target_hw = (msi.shape[0], msi.shape[1])
            worker_args = [
                (
                    vp,
                    target_hw,
                    valid_mask,
                    hd_scalar,
                    hd_edge_n,
                    sigmas,
                    aggregation,
                    edge_method_cfg,
                    eval_mode,
                    eval_percentile,
                    eval_value,
                    eval_otsu_pre_equalize,
                    eval_otsu_equalize_method,
                    eval_otsu_clahe_clip_limit,
                    eval_otsu_clahe_tile_grid_size,
                    allow_resize,
                    equalize_visualization,
                    equalize_visualization_method,
                    equalize_visualization_clahe_clip_limit,
                    equalize_visualization_clahe_tile_grid_size,
                    save_viz_edges,
                    save_viz_edges_binary,
                    save_agreement_overlay,
                    save_npy,
                    save_edge_equalized,
                    viz_edges_dir,
                    save_edge_equalize_method,
                    save_edge_clahe_clip_limit,
                    save_edge_clahe_tile_grid_size,
                    float(cfg.spatial_normalization.low_percentile),
                    float(cfg.spatial_normalization.high_percentile),
                    spatial_enabled,
                    divide_by_255,
                    edge_cfg_override,
                    display_gamma_val,
                    min_viz_edge_percentile,
                    eval_metric,
                    square_before_continuous_metrics,
                    equalize_edges_before_metric,
                    edge_colormap,
                    local_dice_window,
                )
                for vp in viz_list
            ]

            n_jobs = max(1, int(n_jobs))
            if n_jobs <= 1:
                results = [_compute_viz_f1_worker(*a) for a in worker_args]
            else:
                from multiprocessing import Pool

                with Pool(processes=n_jobs) as pool:
                    results = pool.starmap(_compute_viz_f1_worker, worker_args)

            for vp, r in zip(viz_list, results):
                triple = viz_edges_dir / f"{vp.stem}__triple.png"
                dice = r.get("f1")
                err = r.get("error", "")
                rows.append(
                    {
                        "npy_stem": npy_path.stem,
                        "npy_path": str(npy_path),
                        "viz_name": vp.name,
                        "viz_path": str(vp),
                        "viz_type": _visualization_type_from_stem(vp.stem),
                        "continuous_dice": dice if dice is not None and not (isinstance(dice, float) and np.isnan(dice)) else "",
                        "error": err,
                        "triple_png": str(triple) if triple.exists() else "",
                    }
                )
                if err:
                    logger.warning("%s | %s", vp.name, err)

            gallery_cfg = getattr(cfg, "gallery", None)
            if gallery_cfg is None:
                gallery_cfg = OmegaConf.create({})
            if bool(OmegaConf.select(gallery_cfg, "enabled", default=True)):
                gallery_path = viz_edges_dir / f"{npy_path.stem}__dice_sorted_gallery.png"
                _save_dice_sorted_gallery_png(
                    gallery_path,
                    npy_stem=npy_path.stem,
                    sorted_items=_sort_viz_by_dice(viz_list, results),
                    valid_mask=valid_mask,
                    hd_edge_n=hd_edge_n,
                    target_hw=target_hw,
                    sigmas=sigmas,
                    aggregation=aggregation,
                    edge_method_cfg=edge_method_cfg,
                    eval_otsu_pre_equalize=eval_otsu_pre_equalize,
                    eval_otsu_equalize_method=eval_otsu_equalize_method,
                    eval_otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
                    eval_otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
                    allow_resize=allow_resize,
                    equalize_visualization=equalize_visualization,
                    equalize_visualization_method=equalize_visualization_method,
                    equalize_visualization_clahe_clip_limit=equalize_visualization_clahe_clip_limit,
                    equalize_visualization_clahe_tile_grid_size=equalize_visualization_clahe_tile_grid_size,
                    spatial_low_percentile=float(cfg.spatial_normalization.low_percentile),
                    spatial_high_percentile=float(cfg.spatial_normalization.high_percentile),
                    spatial_enabled=spatial_enabled,
                    divide_by_255=divide_by_255,
                    edge_cfg_override=edge_cfg_override,
                    display_gamma_val=display_gamma_val,
                    min_viz_edge_percentile=min_viz_edge_percentile,
                    gallery_cfg=gallery_cfg,
                    equalize_edges_before_metric=equalize_edges_before_metric,
                    edge_colormap=edge_colormap,
                )

    csv_name = str(OmegaConf.select(cfg, "output.csv_name", default="continuous_dice_by_viz.csv"))
    csv_path = out_dir / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "npy_stem",
                "npy_path",
                "viz_name",
                "viz_path",
                "viz_type",
                "continuous_dice",
                "error",
                "triple_png",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # resolve=False: base YAML (e.g. rank_visualizations_by_f1) uses ${hydra:...} / ${now:...};
    # those require Hydra and fail with OmegaConf.to_yaml(..., resolve=True) outside hydra.main.
    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=False))

    print(f"Wrote {csv_path} ({len(rows)} row(s)) in {time.perf_counter() - t0:.2f}s")
    print(f"Triple panels under {out_dir / 'viz_edges_debug'}")
    _print_viz_type_dice_summary(rows, out_dir=out_dir)
    if not rows:
        print(
            "No rows: no matching spatial (H,W) between MSI .npy files and images in --png-dir, "
            "or no viz files assigned to an MSI.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
