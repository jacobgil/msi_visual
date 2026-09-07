#!/usr/bin/env python3
"""
Per-MSI gallery of high-dimensional edge methods (``gallery.hd_methods``).

Mirrors HD edge computation in ``rank_visualizations_by_f1.py`` (including ``top3_nmf_max`` /
``top3_nmf_mean`` via ``gallery_top3_nmf_edges``). Writes one multi-panel PNG per colormap under
``output.gallery_subdir/<cmap>/`` and optional raw grayscale under ``raw_grayscale/<slug>/``.
Output ``slug`` is the last up to three parent directory names plus the file stem (e.g. ``5_bins__0``);
if that collides across inputs, an 8-character path hash is appended.

Optional ``gallery.method_labels``: map internal method keys (e.g. ``soft_landmark_contrast``) to
short panel titles (e.g. SoLaCE). Filenames / raw PNG stems still use internal names.

Run from repo root:
  python scripts/hd_methods_gallery.py
  python scripts/hd_methods_gallery.py --config-name=hd_methods_gallery

Configs: ``scripts/configs/hd_methods.yaml`` (alias) or ``hd_methods_gallery.yaml``.
Preset for soft landmark vs L2 ``distance_multi``: ``python scripts/hd_methods_gallery.py --config-name=hd_methods_soft_vs_distance_l2``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for p in (_REPO, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from debug_edge_maps import (
    _aggregate_multiscale_maps,
    _apply_hd_edge_power,
    _compute_tic_sobel_edge_map,
    _edge_maps_multiscale,
    _gaussian_blur,
    _gray_u8_to_rgb_colormap,
    _load_msi,
    _normalize_map_percentile,
    _normalize_on_mask,
    _parse_edge_detection_cfg,
    _pca_project_msi_for_edges,
    _to_uint8_gray01,
)
from gallery_top3_nmf_edges import (
    _compose_all_images_mosaic,
    _compose_single_image_method_gallery,
    _maybe_downscale,
    _paste_grid,
    _slug_filename,
)
from edge_map_metrics import evaluate_edge_methods, format_panel_metric_lines

logger = logging.getLogger(__name__)


def _human_npy_slug(npy_path: Path) -> str:
    """Last few parent dirs + stem (e.g. ``slide__5_bins__0``), filesystem-safe, not globally unique."""
    p = npy_path.resolve()
    dirs = list(p.parts[:-1])[-3:]
    stem = p.stem
    base = "__".join(dirs + [stem]) if dirs else stem
    s = _slug_filename(base)
    return (s[:120] if s else "npy") or "npy"


def _unique_output_slugs(paths: list[Path]) -> dict[Path, str]:
    """Map each ``.npy`` path to a gallery/raw filename stem; disambiguate duplicate stems."""
    human = {p: _human_npy_slug(p) for p in paths}
    by_slug: dict[str, list[Path]] = {}
    for p, s in human.items():
        by_slug.setdefault(s, []).append(p)
    out: dict[Path, str] = {}
    for s, group in by_slug.items():
        if len(group) == 1:
            out[group[0]] = s
            continue
        for p in group:
            h = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
            out[p] = _slug_filename(f"{s}__{h}")[:128]
    return out


def _as_str_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v).strip() for v in x if v is not None and str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


def _method_display_labels(cfg: DictConfig) -> dict[str, str]:
    """gallery.method_labels: map internal hd_method name → panel title (SoLaCE, etc.)."""
    raw = OmegaConf.select(cfg, "gallery.method_labels", default=None)
    if raw is None:
        return {}
    cont = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(cont, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in cont.items():
        ks = str(k).strip()
        if not ks or v is None:
            continue
        vs = str(v).strip()
        if vs:
            out[ks] = vs
    return out


def _iter_npy_paths(cfg: DictConfig) -> list[Path]:
    paths: list[Path] = []
    raw = OmegaConf.select(cfg, "input.npy_paths")
    if raw is not None:
        cont = OmegaConf.to_container(raw, resolve=True)
        if isinstance(cont, (list, tuple)):
            for p in cont:
                if p is None:
                    continue
                s = str(p).strip()
                if s and s.lower() not in ("null", "none", "~"):
                    paths.append(Path(to_absolute_path(s)))
    d = OmegaConf.select(cfg, "input.npy_dir")
    if d is not None:
        s = str(d).strip()
        if s and s.lower() not in ("null", "none", "~"):
            dirp = Path(to_absolute_path(s))
            glob_pat = str(OmegaConf.select(cfg, "input.npy_glob", default="*.npy"))
            paths.extend(sorted(dirp.glob(glob_pat)))
    # single-file fallback (rank-style)
    if not paths:
        one = OmegaConf.select(cfg, "input.npy_path")
        if one is not None:
            s = str(one).strip()
            if s and s.lower() not in ("null", "none", "~"):
                paths.append(Path(to_absolute_path(s)))
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _multiscale_sigmas_aggregation(cfg: DictConfig) -> tuple[list[float], str]:
    ms = getattr(cfg, "edge_maps", None)
    ms_mult = getattr(ms, "multi_scale", None) if ms is not None else None
    sigmas_raw = list(getattr(ms_mult, "sigmas", [0.0])) if ms_mult is not None else [0.0]
    sigmas = [float(v) for v in sigmas_raw] if sigmas_raw else [0.0]
    aggregation = str(getattr(ms_mult, "aggregation", "mean")).strip().lower() if ms_mult is not None else "mean"
    return sigmas, aggregation


def _compute_merged_hd_edges(
    cfg: DictConfig,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    hd_method: str,
    *,
    msi_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (merged L2 edge map, merged L∞) matching ``rank_visualizations_by_f1``."""
    edge_method_cfg = dict(_parse_edge_detection_cfg(cfg))
    edge_method_cfg["hd_method"] = str(hd_method).strip().lower()
    hd_cfg = getattr(cfg, "hd_edges", None)
    sigmas, aggregation = _multiscale_sigmas_aggregation(cfg)
    method = edge_method_cfg["hd_method"]

    if method == "tic_sobel":
        logger.info("HD edges | tic_sobel (Sobel on total-ion-current map)")
        if msi_raw is None:
            raise ValueError("tic_sobel requires unnormalized msi (load with data.normalization=none)")
        sn = getattr(cfg, "spatial_normalization", None)
        lo = float(getattr(sn, "low_percentile", 1.0)) if sn is not None else 1.0
        hi = float(getattr(sn, "high_percentile", 99.0)) if sn is not None else 99.0
        rgb_cs = str(edge_method_cfg.get("rgb_color_space", "gray")).strip().lower()
        base_edge = _compute_tic_sobel_edge_map(
            msi_raw,
            valid_mask,
            rgb_color_space=rgb_cs,
            low_percentile=lo,
            high_percentile=hi,
        )
        base_edge_linf = np.asarray(base_edge, dtype=np.float32).copy()
        edge_maps: list[np.ndarray] = []
        edge_linf_maps: list[np.ndarray] = []
        for s in sigmas:
            sigma = float(s)
            if sigma <= 0:
                edge_maps.append(base_edge)
                edge_linf_maps.append(base_edge_linf)
            else:
                edge_maps.append(_gaussian_blur(base_edge, sigma))
                edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
        merged = _aggregate_multiscale_maps(edge_maps, valid_mask, aggregation, normalize=True)
        merged_linf = _aggregate_multiscale_maps(edge_linf_maps, valid_mask, aggregation, normalize=True)
        return merged.astype(np.float32, copy=False), merged_linf.astype(np.float32, copy=False)

    if method in ("top3_nmf_max", "top3_nmf_mean"):
        from gallery_top3_nmf_edges import _compute_top3_nmf_aggregate_v_agg

        mode = "max" if method == "top3_nmf_max" else "mean"
        logger.info("HD edges | %s (TOP3+NMF stack, mode=%s)", method, mode)
        base_edge = _compute_top3_nmf_aggregate_v_agg(cfg, msi, valid_mask, mode=mode)
        base_edge_linf = np.asarray(base_edge, dtype=np.float32).copy()
        edge_maps: list[np.ndarray] = []
        edge_linf_maps: list[np.ndarray] = []
        for s in sigmas:
            sigma = float(s)
            if sigma <= 0:
                edge_maps.append(base_edge)
                edge_linf_maps.append(base_edge_linf)
            else:
                edge_maps.append(_gaussian_blur(base_edge, sigma))
                edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
        merged = _aggregate_multiscale_maps(edge_maps, valid_mask, aggregation, normalize=True)
        merged_linf = _aggregate_multiscale_maps(edge_linf_maps, valid_mask, aggregation, normalize=True)
    else:
        logger.info("HD edges | %s", method)
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
        merged = hd_result[0]
        merged_linf = hd_result[1]

    merged = _apply_hd_edge_power(merged, valid_mask, hd_cfg)
    merged_linf = _apply_hd_edge_power(merged_linf, valid_mask, hd_cfg)
    return merged.astype(np.float32, copy=False), merged_linf.astype(np.float32, copy=False)


def _spatial_norm_edge(cfg: DictConfig, edge: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    sn = getattr(cfg, "spatial_normalization", None)
    if sn is not None and bool(getattr(sn, "enabled", True)):
        return _normalize_map_percentile(
            edge,
            valid_mask,
            low_pct=float(getattr(sn, "low_percentile", 1.0)),
            high_pct=float(getattr(sn, "high_percentile", 99.0)),
        )
    return _normalize_on_mask(edge, valid_mask)


def _colormap_names(cfg: DictConfig) -> list[str]:
    lst = OmegaConf.select(cfg, "output.edge_colormap_list")
    if lst is not None:
        names = _as_str_list(OmegaConf.to_container(lst, resolve=True))
        if names:
            return names
    one = OmegaConf.select(cfg, "output.edge_colormap")
    if one is not None:
        s = str(one).strip()
        if s and s.lower() not in ("null", "none", "~"):
            return [s]
    return ["gray"]


def _write_edge_colormaps_file(path: Path, names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def _resize_gray_u8_to_shape(gray_u8: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = gray_u8.shape[:2]
    if w == int(width) and h == int(height):
        return np.asarray(gray_u8, dtype=np.uint8)
    return np.asarray(
        Image.fromarray(np.asarray(gray_u8, dtype=np.uint8), mode="L").resize(
            (int(width), int(height)),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )


def _auto_compare_shape(items: list[tuple[str, np.ndarray]], mode: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(mode, tuple):
        return mode
    shapes = [(int(gray.shape[1]), int(gray.shape[0])) for _title, gray in items]
    if not shapes:
        return (1, 1)
    m = str(mode).strip().lower()
    if m in {"max", "maximum"}:
        return max(w for w, _h in shapes), max(h for _w, h in shapes)
    if m in {"median", "med"}:
        return int(round(float(np.median([w for w, _h in shapes])))), int(round(float(np.median([h for _w, h in shapes]))))

    # Auto chooses one of the observed shapes, avoiding arbitrary aspect-ratio changes.
    # If a shape is repeated, this naturally favors it; otherwise it picks the least
    # total relative resize in log-space across all inputs.
    candidates = sorted(set(shapes))
    best_shape = candidates[0]
    best_score = float("inf")
    for cw, ch in candidates:
        score = 0.0
        for w, h in shapes:
            score += abs(float(np.log(cw / max(1, w)))) + abs(float(np.log(ch / max(1, h))))
        if score < best_score - 1e-12:
            best_score = score
            best_shape = (cw, ch)
    return best_shape


def _save_metrics_summary_plot(
    metrics_by_slide: dict[str, dict[str, dict]],
    *,
    methods: Sequence[str],
    display_labels: dict[str, str],
    out_path: Path,
) -> None:
    """Summary: per-slide PESC bars + Spearman bars."""
    import matplotlib.pyplot as plt

    def lab(m: str) -> str:
        return display_labels.get(str(m), str(m))

    slides = list(metrics_by_slide.keys())
    if not slides:
        return

    colors = {
        methods[i]: c
        for i, c in enumerate(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][: len(methods)])
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor="white")
    ax_pesc, ax_sp = axes
    x = np.arange(len(slides), dtype=float)
    width = 0.8 / max(1, len(methods))

    for i, m in enumerate(methods):
        vals = [float(metrics_by_slide[s].get(m, {}).get("pesc", float("nan"))) for s in slides]
        ax_pesc.bar(x + (i - 0.5 * (len(methods) - 1)) * width, vals, width=width, color=colors.get(m), label=lab(m))
    ax_pesc.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--", zorder=0)
    ax_pesc.set_xticks(x)
    ax_pesc.set_xticklabels([s if len(s) <= 18 else s[:16] + "…" for s in slides], rotation=35, ha="right", fontsize=8)
    ax_pesc.set_ylabel("PESC (higher = better)")
    ax_pesc.set_title("Path-edge spectral consistency")
    ax_pesc.grid(True, axis="y", alpha=0.3)
    ax_pesc.legend(frameon=False, fontsize=8)

    for i, m in enumerate(methods):
        vals = [float(metrics_by_slide[s].get(m, {}).get("spearman", float("nan"))) for s in slides]
        ax_sp.bar(x + (i - 0.5 * (len(methods) - 1)) * width, vals, width=width, color=colors.get(m), label=lab(m))
    ax_sp.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--", zorder=0)
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels([s if len(s) <= 18 else s[:16] + "…" for s in slides], rotation=35, ha="right", fontsize=8)
    ax_sp.set_ylabel("Spearman(s, e)")
    ax_sp.set_title("Spectral distance vs path-max edge")
    ax_sp.grid(True, axis="y", alpha=0.3)
    ax_sp.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _write_metrics_csv(metrics_by_slide: dict[str, dict[str, dict]], out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for slide, by_m in metrics_by_slide.items():
        for method, met in by_m.items():
            rows.append(
                {
                    "slide": slide,
                    "method": method,
                    "pesc": met.get("pesc"),
                    "spearman": met.get("spearman"),
                    "e_high": met.get("e_high"),
                    "e_low": met.get("e_low"),
                    "n_pairs": met.get("n_pairs"),
                }
            )
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


@hydra.main(version_base=None, config_path="configs", config_name="hd_methods")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(cfg.logging.level).upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    npy_paths = _iter_npy_paths(cfg)
    if not npy_paths:
        raise ValueError("No input cubes: set input.npy_paths, input.npy_dir, or input.npy_path")

    methods = _as_str_list(OmegaConf.to_container(getattr(cfg.gallery, "hd_methods"), resolve=True))
    if not methods:
        raise ValueError("gallery.hd_methods is empty")

    disp_labels = _method_display_labels(cfg)

    def disp_title(m: str) -> str:
        return disp_labels.get(str(m), str(m))

    data_norm = OmegaConf.select(cfg, "data.normalization", default="tic")

    cmap_names = _colormap_names(cfg)
    out_gallery = str(getattr(cfg.output, "gallery_subdir", "hd_method_galleries"))
    png_suffix = str(getattr(cfg.output, "png_suffix", "_hd_methods.png"))
    save_raw = bool(getattr(cfg.output, "save_raw_grayscale", False))
    raw_sub = str(getattr(cfg.output, "raw_grayscale_subdir", "raw_grayscale"))

    gcols = int(getattr(cfg.gallery, "columns", 3))
    cell_gap = int(getattr(cfg.gallery, "cell_gap", 8))
    row_gap = int(getattr(cfg.gallery, "row_gap", 12))
    col_pad = int(getattr(cfg.gallery, "column_padding", 8))
    row_lab = int(getattr(cfg.gallery, "row_label_height", 30))
    raw_cm = OmegaConf.select(cfg, "gallery.cell_max_size")
    cell_max_seq = None
    if raw_cm is not None:
        cm = OmegaConf.to_container(raw_cm, resolve=True) if OmegaConf.is_config(raw_cm) else raw_cm
        if isinstance(cm, (list, tuple)) and len(cm) >= 2:
            cell_max_seq = cm
    compare_across = bool(OmegaConf.select(cfg, "gallery.compare_across_images", default=True))
    compare_cols = int(OmegaConf.select(cfg, "gallery.compare_across_images_columns", default=gcols))
    compare_cols = max(1, compare_cols)
    compare_shape_raw = OmegaConf.select(cfg, "gallery.compare_across_images_shape")
    compare_shape: str | tuple[int, int] = "auto"
    if compare_shape_raw is not None:
        cs = (
            OmegaConf.to_container(compare_shape_raw, resolve=True)
            if OmegaConf.is_config(compare_shape_raw)
            else compare_shape_raw
        )
        if isinstance(cs, (list, tuple)) and len(cs) >= 2:
            cw, ch = int(cs[0]), int(cs[1])
            if cw > 0 and ch > 0:
                compare_shape = (cw, ch)
        else:
            s = str(cs).strip().lower()
            compare_shape = "auto" if s in {"", "none", "null", "~"} else s

    all_images_mosaic = bool(OmegaConf.select(cfg, "gallery.all_images_mosaic", default=False))
    mosaic_filename = str(
        OmegaConf.select(cfg, "output.all_images_mosaic_filename", default="all_images_mosaic.png")
    )
    metrics_enabled = bool(OmegaConf.select(cfg, "gallery.metrics.enabled", default=True))
    metrics_pca = int(OmegaConf.select(cfg, "gallery.metrics.pca_components", default=64))
    metrics_n_pairs = int(OmegaConf.select(cfg, "gallery.metrics.n_pairs", default=12000))
    metrics_dist_min = float(OmegaConf.select(cfg, "gallery.metrics.dist_min", default=5.0))
    metrics_dist_max = float(OmegaConf.select(cfg, "gallery.metrics.dist_max", default=40.0))
    metrics_high_frac = float(OmegaConf.select(cfg, "gallery.metrics.high_frac", default=0.20))
    metrics_low_frac = float(OmegaConf.select(cfg, "gallery.metrics.low_frac", default=0.20))
    metrics_seed = int(OmegaConf.select(cfg, "gallery.metrics.random_state", default=42))
    styled_panels = bool(OmegaConf.select(cfg, "gallery.styled_panels", default=True))
    panel_title_fs = int(OmegaConf.select(cfg, "gallery.panel_title_font_size", default=28))
    panel_metric_fs = int(OmegaConf.select(cfg, "gallery.panel_metric_font_size", default=15))
    panel_bg_raw = OmegaConf.select(cfg, "gallery.panel_bg", default=[255, 255, 255])
    panel_bg = tuple(int(v) for v in OmegaConf.to_container(panel_bg_raw, resolve=True))

    _write_edge_colormaps_file(run_dir / out_gallery / "edge_colormaps.txt", cmap_names)

    transpose_msi = bool(OmegaConf.select(cfg, "data.transpose_msi", default=False))

    slug_by_path = _unique_output_slugs(npy_paths)
    across_by_method: dict[str, list[tuple[str, np.ndarray]]] = {}
    mosaic_gray_rows: list[tuple[str, list[np.ndarray], int, np.ndarray]] = []
    metrics_by_slide: dict[str, dict[str, dict]] = {}

    for npy_path in npy_paths:
        if not npy_path.is_file():
            logger.warning("Skip missing file: %s", npy_path)
            continue
        out_stem = slug_by_path[npy_path]
        logger.info("Processing %s (output stem: %s)", npy_path, out_stem)
        t0 = time.perf_counter()
        msi = _load_msi(npy_path, transpose_msi=transpose_msi, normalization=data_norm)
        msi_raw = (
            _load_msi(npy_path, transpose_msi=transpose_msi, normalization="none")
            if "tic_sobel" in methods
            else None
        )
        valid_mask = msi.sum(axis=-1) > 0
        if not np.any(valid_mask):
            logger.warning("Empty valid mask for %s", npy_path)
            continue

        titles: list[str] = [disp_title(m) for m in methods]
        raw_dir = run_dir / out_gallery / raw_sub / out_stem
        if save_raw:
            raw_dir.mkdir(parents=True, exist_ok=True)

        gray_by_method: list[np.ndarray] = []
        float_by_method: dict[str, np.ndarray] = {}
        for hd_method in methods:
            merged, _merged_linf = _compute_merged_hd_edges(
                cfg,
                msi,
                valid_mask,
                hd_method,
                msi_raw=msi_raw,
            )
            disp = _spatial_norm_edge(cfg, merged, valid_mask)
            float_by_method[str(hd_method)] = np.asarray(disp, dtype=np.float32)
            gray_u8 = _to_uint8_gray01(disp)
            gray_by_method.append(gray_u8)
            if compare_across:
                across_by_method.setdefault(str(hd_method), []).append((out_stem, gray_u8.copy()))
            if save_raw:
                Image.fromarray(gray_u8, mode="L").save(raw_dir / f"{_slug_filename(hd_method)}.png")

        panel_metric_lines: list[list[str]] = [[] for _ in methods]
        if metrics_enabled and float_by_method:
            logger.info(
                "Computing PESC | n_pairs=%d dist=[%.1f,%.1f] pca=%d",
                metrics_n_pairs,
                metrics_dist_min,
                metrics_dist_max,
                metrics_pca,
            )
            slide_metrics = evaluate_edge_methods(
                float_by_method,
                msi,
                valid_mask,
                pca_components=metrics_pca,
                n_pairs=metrics_n_pairs,
                dist_min=metrics_dist_min,
                dist_max=metrics_dist_max,
                high_frac=metrics_high_frac,
                low_frac=metrics_low_frac,
                random_state=metrics_seed,
            )
            metrics_by_slide[out_stem] = slide_metrics
            panel_metric_lines = [
                format_panel_metric_lines(slide_metrics[m]) for m in methods
            ]
            metrics_json = run_dir / out_gallery / "metrics" / f"{out_stem}.json"
            metrics_json.parent.mkdir(parents=True, exist_ok=True)

            def _jsonable(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {str(k): _jsonable(v) for k, v in obj.items()}
                if isinstance(obj, float):
                    return None if not np.isfinite(obj) else obj
                return obj

            metrics_json.write_text(json.dumps(_jsonable(slide_metrics), indent=2), encoding="utf-8")

        aggregate_edges = str(OmegaConf.select(cfg, "gallery.aggregate_edges", default="max")).strip().lower()
        if gray_by_method and aggregate_edges not in {"", "none", "false", "off", "disabled"}:
            if aggregate_edges in {"mean", "avg", "average"}:
                agg_edges = np.mean(np.stack(gray_by_method, axis=0).astype(np.float32), axis=0)
                agg_title = "mean_all_edges"
            elif aggregate_edges in {"max", "maximum"}:
                agg_edges = np.maximum.reduce(gray_by_method).astype(np.float32, copy=False)
                agg_title = "max_all_edges"
            else:
                raise ValueError("gallery.aggregate_edges must be one of: max, mean, none")
            agg_edges_u8 = np.clip(agg_edges, 0, 255).astype(np.uint8)
            titles.append(agg_title)
            gray_by_method.append(agg_edges_u8)
            panel_metric_lines.append([])
            if compare_across:
                across_by_method.setdefault(agg_title, []).append((out_stem, agg_edges_u8.copy()))
            if save_raw:
                Image.fromarray(agg_edges_u8, mode="L").save(raw_dir / f"{agg_title}.png")

        mosaic_gray_rows.append(
            (
                out_stem,
                [g.copy() for g in gray_by_method[: len(methods)]],
                int(gray_by_method[0].shape[0] * gray_by_method[0].shape[1]),
                np.asarray(valid_mask, dtype=bool).copy(),
            )
        )

        for cmap_name in cmap_names:
            cmap_dir = run_dir / out_gallery / _slug_filename(cmap_name)
            cmap_dir.mkdir(parents=True, exist_ok=True)
            cells_c: list[np.ndarray] = []
            for mi, gray_u8 in enumerate(gray_by_method):
                try:
                    rgb = _gray_u8_to_rgb_colormap(gray_u8, cmap_name, valid_mask=valid_mask)
                except Exception as e:
                    logger.warning("Colormap %r failed: %s — using gray", cmap_name, e)
                    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.uint8)
                    rgb[~valid_mask] = (255, 255, 255)
                cells_c.append(_maybe_downscale(rgb, cell_max_seq))
                # Save individual SoLaCE (soft_landmark_contrast) edge map in this colormap.
                if mi < len(methods) and str(methods[mi]) == "soft_landmark_contrast":
                    solace_dir = cmap_dir / "solace"
                    solace_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb, mode="RGB").save(solace_dir / f"{out_stem}_solace.png")
            if styled_panels:
                grid_c = _compose_single_image_method_gallery(
                    cells_c[: len(methods)],
                    titles=titles[: len(methods)],
                    metric_lines=panel_metric_lines[: len(methods)],
                    column_padding=max(col_pad, 16),
                    cell_gap=max(cell_gap, 16),
                    title_font_size=panel_title_fs,
                    metric_font_size=panel_metric_fs,
                    bg=panel_bg,
                )
                # append aggregate panel(s) if present using simple paste to the right
                if len(cells_c) > len(methods):
                    extra = _paste_grid(
                        cells_c[len(methods) :],
                        ncols=max(1, len(cells_c) - len(methods)),
                        column_padding=col_pad,
                        cell_gap=cell_gap,
                        row_gap=row_gap,
                        row_label_height=row_lab,
                        titles=titles[len(methods) :],
                        bg=panel_bg,
                    )
                    # stack horizontally with padding
                    from PIL import Image as _PILImage

                    left = _PILImage.fromarray(grid_c)
                    right = _PILImage.fromarray(extra)
                    h = max(left.height, right.height)
                    canvas = _PILImage.new("RGB", (left.width + 16 + right.width, h), panel_bg)
                    canvas.paste(left, (0, 0))
                    canvas.paste(right, (left.width + 16, 0))
                    grid_c = np.asarray(canvas, dtype=np.uint8)
            else:
                grid_c = _paste_grid(
                    cells_c,
                    ncols=gcols,
                    column_padding=col_pad,
                    cell_gap=cell_gap,
                    row_gap=row_gap,
                    row_label_height=row_lab,
                    titles=titles,
                    bg=panel_bg,
                )
            out_png = cmap_dir / f"{out_stem}{png_suffix}"
            Image.fromarray(grid_c).save(out_png)
            logger.info("Wrote %s", out_png)

        logger.info("Done %s in %.1fs", out_stem, time.perf_counter() - t0)

    if metrics_by_slide:
        metrics_dir = run_dir / out_gallery / "metrics"
        csv_path = metrics_dir / "edge_metrics_summary.csv"
        plot_path = metrics_dir / "edge_metrics_summary.png"
        _write_metrics_csv(metrics_by_slide, csv_path)
        _save_metrics_summary_plot(
            metrics_by_slide,
            methods=methods,
            display_labels=disp_labels,
            out_path=plot_path,
        )
        logger.info("Wrote metrics CSV %s", csv_path)
        logger.info("Wrote metrics summary plot %s", plot_path)

    if compare_across and across_by_method:
        compare_subdir = run_dir / out_gallery / "method_comparisons"
        for cmap_name in cmap_names:
            cmap_compare_dir = compare_subdir / _slug_filename(cmap_name)
            cmap_compare_dir.mkdir(parents=True, exist_ok=True)
            for method, items in across_by_method.items():
                cells: list[np.ndarray] = []
                titles_cmp: list[str] = []
                target_w, target_h = _auto_compare_shape(items, compare_shape)
                logger.info(
                    "Across-image comparison | method=%s common_shape=%dx%d mode=%s",
                    method,
                    target_w,
                    target_h,
                    compare_shape,
                )
                for title, gray_u8 in items:
                    gray_cmp = _resize_gray_u8_to_shape(gray_u8, target_w, target_h)
                    try:
                        rgb = _gray_u8_to_rgb_colormap(gray_cmp, cmap_name)
                    except Exception as e:
                        logger.warning("Colormap %r failed: %s — using gray", cmap_name, e)
                        rgb = np.stack([gray_cmp, gray_cmp, gray_cmp], axis=-1).astype(np.uint8)
                    cells.append(_maybe_downscale(rgb, cell_max_seq))
                    titles_cmp.append(title)
                grid = _paste_grid(
                    cells,
                    ncols=compare_cols,
                    column_padding=col_pad,
                    cell_gap=cell_gap,
                    row_gap=row_gap,
                    row_label_height=row_lab,
                    titles=titles_cmp,
                    bg=(0, 0, 0),
                )
                out_png = cmap_compare_dir / f"{_slug_filename(method)}_across_images.png"
                Image.fromarray(grid).save(out_png)
                logger.info("Wrote %s", out_png)

    if all_images_mosaic and mosaic_gray_rows:
        column_titles = [disp_title(m) for m in methods]
        mosaic_sort = str(
            OmegaConf.select(cfg, "gallery.all_images_mosaic_sort_by_area", default="descending")
        ).strip().lower()
        mosaic_bg_raw = OmegaConf.select(cfg, "gallery.all_images_mosaic_bg", default=[255, 255, 255])
        mosaic_bg = tuple(int(v) for v in OmegaConf.to_container(mosaic_bg_raw, resolve=True))
        mosaic_title_size = int(OmegaConf.select(cfg, "gallery.all_images_mosaic_title_font_size", default=22))
        show_row_labels = bool(
            OmegaConf.select(cfg, "gallery.all_images_mosaic_show_row_labels", default=False)
        )
        sorted_rows = list(mosaic_gray_rows)
        if mosaic_sort in {"descending", "desc", "large_to_small", "large-first"}:
            sorted_rows.sort(key=lambda item: item[2], reverse=True)
        elif mosaic_sort in {"ascending", "asc", "small_to_large", "small-first"}:
            sorted_rows.sort(key=lambda item: item[2])
        for cmap_name in cmap_names:
            cmap_dir = run_dir / out_gallery / _slug_filename(cmap_name)
            cmap_dir.mkdir(parents=True, exist_ok=True)
            rows_rgb: list[tuple[str, list[np.ndarray]]] = []
            for row_label, grays, _area, row_mask in sorted_rows:
                cells_rgb: list[np.ndarray] = []
                for gray_u8 in grays:
                    try:
                        rgb = _gray_u8_to_rgb_colormap(gray_u8, cmap_name, valid_mask=row_mask)
                    except Exception as e:
                        logger.warning("Colormap %r failed: %s — using gray", cmap_name, e)
                        rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.uint8)
                        rgb[~row_mask] = (255, 255, 255)
                    cells_rgb.append(rgb)
                rows_rgb.append((row_label, cells_rgb))
            mosaic = _compose_all_images_mosaic(
                rows_rgb,
                column_titles=column_titles,
                column_padding=col_pad,
                cell_gap=cell_gap,
                row_gap=row_gap,
                show_row_labels=show_row_labels,
                title_font_size=mosaic_title_size,
                bg=mosaic_bg,
            )
            out_png = cmap_dir / mosaic_filename
            Image.fromarray(mosaic).save(out_png)
            logger.info("Wrote all-images mosaic %s", out_png)


if __name__ == "__main__":
    main()
