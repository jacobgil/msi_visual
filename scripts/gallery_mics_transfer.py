#!/usr/bin/env python3
"""
Train MiCS on one slide, infer on many — grid gallery.

Layout (``gallery.layout``):

- ``by_setting`` (default): rows = MiCS parameter groups; columns = slides × modes.
- ``by_slide``: rows = slides; columns = settings × modes (xfer | xfer+ft | self).
- ``by_train_slide``: rows = train slides (one MiCS setting); columns = target slides × modes.
  Sweeps which slide is used for ``fit()``; self-trained row is shared across train rows.

Modes per slide/setting: **transfer**, optionally **transfer+finetune**, **self**,
**self+align** (train from scratch on the target; ``self_aligned_color_anchor: chroma``
keeps target LAB-L for edges, train-slide a/b for color; ``full`` anchors all channels),
and optionally **novelty** heatmaps: absolute (top-k mean distance to train refs) and relative
``log(d_self / d_train)`` using the xfer model embedding and each target's own ref bank.

Color EMD (``metrics.color_emd.enabled``): Earth mover's distance between LAB
histograms of each tile vs the train-slide reference (xfer prediction on the train
slide for that row).

Only slides matching ``gallery.spectral_filter`` (default: CarstenHopf 9510-bin /
start_mz=99) are used. Panels are letterboxed to a common cell size.

When ``metrics.enabled`` is true, each tile is scored (continuous Dice vs SoLACE,
contrast, entropy) and comparison figures are written for transfer vs finetuned
transfer vs self-trained.

Run:
  python scripts/gallery_mics_transfer.py --config-name=gallery_mics_transfer_carstenhopf
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bayes_tune_mics_lmc import (  # noqa: E402
    _compute_viz_edge_n,
    _resolve_npy_paths,
)
from benchmark_parametric_methods import (  # noqa: E402
    _mics_kwargs,
    _resolve_mics_parameter_groups,
    _seed_everything,
)
from debug_edge_maps import (  # noqa: E402
    _compute_continuous_dice,
    _compute_viz_contrast_color_metrics,
    _edges_for_continuous_metrics,
    _load_msi,
    _resolve_normalization_mode,
    _to_uint8_rgb,
)
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
from msi_visual.extract.he_thumbnail import extract_he_from_wsi_at_dpi  # noqa: E402
from msi_visual.registration.paths import resolve_he_thumbnail, resolve_he_wsi  # noqa: E402
from msi_visual.parametric_mics_lmc import (  # noqa: E402
    MSIParametricMiCSLMC,
    _spatial_gaussian_smooth_channels,
)
from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method  # noqa: E402

logger = logging.getLogger(__name__)


def _abs_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _slide_key(npy_path: Path) -> str:
    if npy_path.parent.name in ("5_bins", "5bins"):
        return npy_path.parent.parent.name
    return npy_path.parent.name


def _read_extraction_meta(npy_path: Path) -> dict[str, Any]:
    args_path = npy_path.parent / "args.txt"
    if not args_path.is_file():
        raise FileNotFoundError(f"Missing args.txt beside {npy_path}")
    ns: Namespace = eval(args_path.read_text(encoding="utf-8"))  # noqa: S307
    arr = np.load(str(npy_path), mmap_mode="r")
    if arr.ndim == 3:
        spectral = int(arr.shape[-1])
    elif arr.ndim == 2:
        spectral = 1
    else:
        raise ValueError(f"Unexpected MSI shape {arr.shape} for {npy_path}")
    return {
        "slide_key": _slide_key(npy_path),
        "start_mz": int(ns.start_mz),
        "end_mz": int(ns.end_mz),
        "bins": int(ns.bins),
        "spectral_bins": spectral,
    }


def _matches_spectral_filter(meta: dict[str, Any], filt: DictConfig) -> bool:
    if int(meta["spectral_bins"]) != int(filt.spectral_bins):
        return False
    if int(meta["start_mz"]) != int(filt.start_mz):
        return False
    if int(meta["end_mz"]) != int(filt.end_mz):
        return False
    if int(meta["bins"]) != int(filt.bins_per_mz):
        return False
    return True


def _filter_compatible_slides(npy_paths: list[Path], filt: DictConfig) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for p in npy_paths:
        meta = _read_extraction_meta(p)
        if _matches_spectral_filter(meta, filt):
            out.append((p, meta))
        else:
            logger.info(
                "Skip %s (start=%s end=%s bins=%s spectral=%s)",
                meta["slide_key"],
                meta["start_mz"],
                meta["end_mz"],
                meta["bins"],
                meta["spectral_bins"],
            )
    return out


def _slide_polarity(slide_key: str, npy_path: Path) -> str | None:
    key = str(slide_key).lower()
    if "_pos_" in key or key.endswith("_pos_rms") or "pos_rms" in key:
        return "pos"
    if "_neg_" in key or key.endswith("_neg_rms") or "neg_rms" in key:
        return "neg"
    parts = {p.lower() for p in npy_path.parts}
    if "positive" in parts:
        return "pos"
    if "negative" in parts:
        return "neg"
    return None


def _filter_by_polarity(
    slides: list[tuple[Path, dict[str, Any]]],
    polarity: str,
) -> list[tuple[Path, dict[str, Any]]]:
    raw = str(polarity).strip().lower()
    if raw in ("pos", "positive", "+", "p"):
        want = "pos"
    elif raw in ("neg", "negative", "-", "n"):
        want = "neg"
    else:
        raise ValueError(f"train_polarity must be pos or neg, got {polarity!r}")
    out = [
        (p, m)
        for p, m in slides
        if _slide_polarity(str(m["slide_key"]), p) == want
    ]
    if not out:
        keys = [m["slide_key"] for _, m in slides]
        raise ValueError(f"No compatible slides with polarity {want!r}. Slides: {keys}")
    return out


def _resolve_train_slide(
    slides: list[tuple[Path, dict[str, Any]]],
    ga: Any,
) -> tuple[int, Path, dict[str, Any]]:
    """Pick train slide; return its index in ``slides``."""
    key_raw = OmegaConf.select(ga, "train_slide_key")
    if key_raw is not None and str(key_raw).strip():
        wanted = str(key_raw).strip()
        for i, (path, meta) in enumerate(slides):
            if meta["slide_key"] == wanted:
                return i, path, meta
        raise ValueError(f"train_slide_key {wanted!r} not among compatible slides")

    pool = slides
    pol_raw = OmegaConf.select(ga, "train_polarity")
    if pol_raw is not None and str(pol_raw).strip().lower() not in (
        "",
        "any",
        "all",
        "null",
        "none",
        "~",
    ):
        pool = _filter_by_polarity(slides, str(pol_raw))
        logger.info(
            "Train polarity %s: %d candidate(s): %s",
            pol_raw,
            len(pool),
            [m["slide_key"] for _, m in pool],
        )

    idx = int(OmegaConf.select(ga, "train_index", default=0))
    if idx < 0 or idx >= len(pool):
        raise IndexError(
            f"train_index {idx} out of range for train pool of {len(pool)} slide(s)"
        )
    train_path, train_meta = pool[idx]
    for i, (path, meta) in enumerate(slides):
        if path.resolve() == train_path.resolve():
            return i, path, meta
    raise RuntimeError("Resolved train slide missing from compatible list")


def _load_parameter_groups(cfg: DictConfig) -> list[tuple[str, dict[str, Any]]]:
    groups_cfg_path = OmegaConf.select(cfg, "gallery.parameter_groups_config")
    if groups_cfg_path:
        gpath = _abs_path(str(groups_cfg_path))
        gcfg = OmegaConf.load(str(gpath))
        stub = OmegaConf.create({"mics_lmc": gcfg.mics_lmc})
    else:
        stub = OmegaConf.create(
            {
                "mics_lmc": {
                    "defaults_yaml": OmegaConf.select(
                        cfg, "gallery.defaults_yaml", default="scripts/configs/train_parametric_mics_lmc.yaml"
                    ),
                    "parameter_groups": OmegaConf.select(cfg, "gallery.parameter_groups"),
                }
            }
        )
    groups = _resolve_mics_parameter_groups(stub)
    wanted = OmegaConf.select(cfg, "gallery.mics_groups")
    if wanted is None:
        return groups
    names = {str(x).strip() for x in OmegaConf.to_container(wanted, resolve=True)}
    picked = [(n, ov) for n, ov in groups if n in names]
    if not picked:
        raise ValueError(f"No parameter groups matched gallery.mics_groups={sorted(names)}")
    return picked


def _merged_group_overrides(cfg: DictConfig, group_overrides: dict[str, Any]) -> dict[str, Any]:
    extra = OmegaConf.select(cfg, "gallery.mics_overrides")
    if extra is None:
        return dict(group_overrides)
    merged = dict(group_overrides)
    merged.update(OmegaConf.to_container(extra, resolve=True))
    return merged


def _target_sampling_cache_dir(cfg: DictConfig, run_dir: Path, slide_key: str) -> Path:
    raw = OmegaConf.select(cfg, "gallery.persistent_cache_dir")
    if raw:
        base = _abs_path(str(raw))
    else:
        base = run_dir / "caches" / "targets"
    return base / str(slide_key)


def _prebuild_target_sampling_caches(
    *,
    cfg: DictConfig,
    bench_cfg: DictConfig,
    group_overrides: dict[str, Any],
    seed: int,
    sampling_mode: str,
    run_dir: Path,
    msis: list[np.ndarray],
    slide_keys: list[str],
) -> list[dict[str, Any] | None]:
    ga = cfg.gallery
    if not bool(OmegaConf.select(ga, "share_pixel_sampling", default=True)):
        return [None] * len(msis)
    if not bool(OmegaConf.select(ga, "prebuild_target_caches", default=True)):
        return [None] * len(msis)
    model_kw = _mics_kwargs(bench_cfg, sampling_mode, seed, _merged_group_overrides(cfg, group_overrides))
    cache_path_raw = OmegaConf.select(ga, "pixel_sampling_cache_path")
    cache_path = _abs_path(str(cache_path_raw)) if cache_path_raw else None
    caches: list[dict[str, Any] | None] = []
    for i, (msi, slide_key) in enumerate(zip(msis, slide_keys, strict=True)):
        logger.info(
            "Target sampling cache %d/%d: %s",
            i + 1,
            len(msis),
            slide_key,
        )
        caches.append(
            resolve_pixel_sampling_cache(
                msi,
                model_kw,
                out_dir=_target_sampling_cache_dir(cfg, run_dir, slide_key),
                share=True,
                cache_path=cache_path,
                strip_cluster_labels=False,
            )
        )
    return caches


def _resolve_gallery_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _he_enabled(cfg: DictConfig) -> bool:
    raw = OmegaConf.select(cfg, "gallery.he")
    if raw is None:
        return bool(OmegaConf.select(cfg, "gallery.include_he_columns", default=False))
    return bool(OmegaConf.select(raw, "enabled", default=True))


def _load_he_rgb(slide_key: str, he_cfg: DictConfig) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load H&E overview from WSI (preferred) or cached thumbnail PNG."""
    meta: dict[str, Any] = {"he_source": "none"}
    if not bool(he_cfg.enabled):
        return None, meta

    source = str(OmegaConf.select(he_cfg, "source", default="auto")).strip().lower()
    thumb_path = resolve_he_thumbnail(slide_key, he_cfg, resolve_path=_resolve_gallery_path)
    wsi_path = resolve_he_wsi(slide_key, he_cfg, resolve_path=_resolve_gallery_path)

    target_dpi = float(OmegaConf.select(he_cfg, "target_dpi", default=400.0))
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
                max_dpi=max_dpi,
            )
            meta.update(wsi_meta)
            meta["he_source"] = "wsi"
            meta["wsi_path"] = str(wsi_path)
            meta["thumbnail_path"] = str(thumb_path) if thumb_path else None
            logger.info(
                "%s: H&E from WSI %s -> %dx%d",
                slide_key,
                wsi_path.name,
                rgb.shape[1],
                rgb.shape[0],
            )
            return rgb, meta
        except Exception as exc:
            logger.warning("%s: OpenSlide H&E failed (%s); trying thumbnail", slide_key, exc)

    if thumb_path is not None and thumb_path.is_file():
        rgb = np.asarray(Image.open(thumb_path).convert("RGB"), dtype=np.uint8)
        meta.update({"he_source": "thumbnail", "thumbnail_path": str(thumb_path)})
        return rgb, meta

    return None, meta


def _build_he_row(
    infer_items: list[tuple[Path, dict[str, Any]]],
    cfg: DictConfig,
    *,
    bg: tuple[int, int, int],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    he_cfg = OmegaConf.select(cfg, "gallery.he")
    if he_cfg is None or not _he_enabled(cfg):
        return [], []

    tiles: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for path, meta in infer_items:
        slide_key = str(meta["slide_key"])
        rgb, he_meta = _load_he_rgb(slide_key, he_cfg)
        he_meta["slide_key"] = slide_key
        he_meta["npy_path"] = str(path)
        if rgb is None:
            logger.warning("%s: no H&E found — placeholder tile", slide_key)
            rgb = np.full((64, 64, 3), bg, dtype=np.uint8)
            he_meta["he_source"] = "missing"
        tiles.append(rgb)
        metas.append(he_meta)
    return tiles, metas


def _mode_column_count(
    *,
    include_finetune: bool,
    include_finetune_color: bool,
    include_self: bool,
    include_self_aligned: bool,
    include_novelty: bool,
    include_relative_novelty: bool,
) -> int:
    n = 1
    if include_finetune:
        n += 1
    if include_finetune_color:
        n += 1
    if include_self:
        n += 1
    if include_self_aligned:
        n += 1
    if include_novelty:
        n += 1
    if include_relative_novelty:
        n += 1
    return n


def _bench_cfg(cfg: DictConfig) -> DictConfig:
    ga = cfg.gallery
    return OmegaConf.create(
        {
            "mics_lmc": {
                "defaults_yaml": OmegaConf.select(
                    cfg, "gallery.defaults_yaml", default="scripts/configs/train_parametric_mics_lmc.yaml"
                ),
            },
            "benchmark": {
                "num_samples": int(OmegaConf.select(ga, "num_samples", default=5000)),
                "shared_sampling": OmegaConf.select(ga, "shared_sampling", default={"enforce_same_pca_fit_step": True}),
            },
        }
    )


def _equalize_rgb_u8_lab_l(rgb: np.ndarray) -> np.ndarray:
    """Histogram-equalize LAB luminance; keep a/b chroma."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.equalizeHist(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def _equalize_per_channel_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.asarray(rgb, dtype=np.uint8).copy()
    for ch in range(min(3, out.shape[-1])):
        out[:, :, ch] = cv2.equalizeHist(out[:, :, ch])
    return out


def _maybe_equalize_rgb(rgb: np.ndarray, ga: Any) -> np.ndarray:
    if not bool(OmegaConf.select(ga, "equalize_visualization", default=True)):
        return rgb
    method = str(OmegaConf.select(ga, "equalize_method", default="lab_l")).strip().lower()
    if method in ("lab_l", "lab", "l"):
        return _equalize_rgb_u8_lab_l(rgb)
    if method in ("clahe", "hist"):
        from debug_edge_maps import _apply_equalize_rgb_uint8

        return _apply_equalize_rgb_uint8(
            np.asarray(rgb, dtype=np.uint8),
            "clahe" if method == "clahe" else "hist",
            float(OmegaConf.select(ga, "clahe_clip_limit", default=2.0)),
            int(OmegaConf.select(ga, "clahe_tile_grid_size", default=8)),
        )
    return _equalize_per_channel_rgb(rgb)


def _letterbox_rgb(
    img: np.ndarray,
    size: tuple[int, int],
    *,
    bg: tuple[int, int, int] = (14, 14, 16),
    allow_upscale: bool = False,
) -> np.ndarray:
    tw, th = int(size[0]), int(size[1])
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    h, w = arr.shape[:2]
    scale = min(tw / max(w, 1), th / max(h, 1))
    if not allow_upscale:
        scale = min(scale, 1.0)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(arr, (nw, nh), interpolation=interp)
    canvas = np.full((th, tw, 3), bg, dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _maybe_downscale_rgb(img: np.ndarray, cell_max_size: list[float] | None) -> np.ndarray:
    if not cell_max_size or len(cell_max_size) < 2:
        return img
    max_w, max_h = float(cell_max_size[0]), float(cell_max_size[1])
    h, w = img.shape[:2]
    if max_w <= 0 or max_h <= 0 or (w <= max_w and h <= max_h):
        return img
    scale = min(max_w / w, max_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _compute_cell_size(viz_list: list[np.ndarray], ga: Any) -> tuple[int, int]:
    raw_max = OmegaConf.to_container(OmegaConf.select(ga, "cell_max_size"), resolve=True)
    max_w, max_h = (480, 320) if not raw_max else (float(raw_max[0]), float(raw_max[1]))
    fixed = OmegaConf.select(ga, "cell_size")
    if fixed is not None:
        fs = OmegaConf.to_container(fixed, resolve=True)
        if isinstance(fs, (list, tuple)) and len(fs) >= 2:
            return int(fs[0]), int(fs[1])
    mw, mh = 1, 1
    for viz in viz_list:
        ds = _maybe_downscale_rgb(viz, [max_w, max_h] if raw_max else None)
        mh = max(mh, ds.shape[0])
        mw = max(mw, ds.shape[1])
    return mw, mh


def _build_labeled_grid(
    grid: list[list[np.ndarray]],
    row_labels: list[str],
    col_labels: list[str],
    *,
    cell_size: tuple[int, int],
    row_label_width: int,
    col_label_height: int,
    gap: int,
    bg: tuple[int, int, int],
) -> np.ndarray:
    nrows = len(grid)
    ncols = len(grid[0]) if grid else 0
    cw, ch = cell_size
    gap = int(gap)
    rlw = int(row_label_width)
    clh = int(col_label_height)
    total_w = rlw + gap + ncols * cw + (ncols + 1) * gap
    total_h = clh + gap + nrows * ch + (nrows + 1) * gap
    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x_cols = rlw + gap * 2
    for c, label in enumerate(col_labels):
        x = x_cols + c * (cw + gap) + cw // 2
        if font:
            lines = str(label).split("\n")
            for li, line in enumerate(lines[:2]):
                draw.text(
                    (x, gap + li * 11),
                    line[:56],
                    fill=(220, 220, 220),
                    font=font,
                    anchor="ma",
                )

    for r, (row_cells, row_label) in enumerate(zip(grid, row_labels, strict=True)):
        y = clh + gap * 2 + r * (ch + gap)
        if font:
            draw.text((gap, y + ch // 2), row_label[:40], fill=(220, 220, 220), font=font, anchor="lm")
        for c, cell in enumerate(row_cells):
            x = x_cols + c * (cw + gap)
            canvas.paste(Image.fromarray(cell), (x, y))
    return np.asarray(canvas, dtype=np.uint8)


def _compute_train_color_anchor(
    model: MSIParametricMiCSLMC,
    train_msi: np.ndarray,
) -> list[tuple[float, float]]:
    """Per-channel embedding percentiles on the train slide (shared color reference)."""
    train_mask = train_msi.sum(axis=-1) > 0
    train_emb = model.predict_embedding(train_msi)
    return _compute_norm_anchor(
        train_emb,
        train_mask,
        float(getattr(model, "predict_percentile_low", 0.001)),
        float(getattr(model, "predict_percentile_high", 99.999)),
    )


def _fit_predict_single(
    *,
    group_name: str,
    group_overrides: dict[str, Any],
    train_msi: np.ndarray,
    predict_msi: np.ndarray,
    bench_cfg: DictConfig,
    cfg: DictConfig,
    seed: int,
    sampling_mode: str,
    run_dir: Path,
    cache_subdir: str,
    pixel_sampling_cache: dict[str, Any] | None = None,
    color_anchor: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    run_seed = int(seed)
    overrides = _merged_group_overrides(cfg, group_overrides)
    model_kw = _mics_kwargs(bench_cfg, sampling_mode, run_seed, overrides)
    ga = cfg.gallery
    cache = pixel_sampling_cache
    if cache is None and bool(OmegaConf.select(ga, "share_pixel_sampling", default=True)):
        cache_path_raw = OmegaConf.select(ga, "pixel_sampling_cache_path")
        cache_path = _abs_path(str(cache_path_raw)) if cache_path_raw else None
        cache = resolve_pixel_sampling_cache(
            train_msi,
            model_kw,
            out_dir=run_dir / "caches" / cache_subdir,
            share=True,
            cache_path=cache_path,
            strip_cluster_labels=False,
        )
    model = MSIParametricMiCSLMC(**model_kw)
    model.fit(train_msi, pixel_sampling_cache=cache)
    if color_anchor is None:
        rgb = _to_uint8_rgb(model.predict(predict_msi))
    else:
        rgb = _predict_rgb_with_optional_anchor(model, predict_msi, color_anchor)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return rgb


def _compute_norm_anchor(
    emb: np.ndarray,
    mask: np.ndarray,
    low: float,
    high: float,
) -> list[tuple[float, float]]:
    anchor: list[tuple[float, float]] = []
    m = np.asarray(mask, dtype=bool)
    for c in range(int(emb.shape[-1])):
        vals = np.asarray(emb[..., c], dtype=np.float64)[m]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            anchor.append((0.0, 1.0))
        else:
            anchor.append((float(np.percentile(vals, low)), float(np.percentile(vals, high))))
    return anchor


def _blend_anchor_bounds(
    anchor_a: list[tuple[float, float]],
    anchor_b: list[tuple[float, float]],
    blend: float,
) -> list[tuple[float, float]]:
    """Blend percentile bounds; ``blend=1`` → ``anchor_a``, ``blend=0`` → ``anchor_b``."""
    b = float(np.clip(blend, 0.0, 1.0))
    out: list[tuple[float, float]] = []
    for (lo_a, hi_a), (lo_b, hi_b) in zip(anchor_a, anchor_b, strict=True):
        out.append(((1.0 - b) * lo_b + b * lo_a, (1.0 - b) * hi_b + b * hi_a))
    return out


def _resolve_render_anchor(
    train_anchor: list[tuple[float, float]],
    emb: np.ndarray,
    mask: np.ndarray,
    *,
    low: float,
    high: float,
    anchor_mode: str,
    anchor_blend: float,
    lab_to_rgb: bool,
    n_components: int,
) -> list[tuple[float, float]]:
    """Merge train/target anchors; ``chroma`` keeps target L; ``anchor_blend`` softens toward target."""
    mode = str(anchor_mode).strip().lower()
    if mode not in ("full", "chroma"):
        raise ValueError(f"anchor_mode must be 'full' or 'chroma', got {anchor_mode!r}")
    target_anchor = _compute_norm_anchor(emb, mask, low, high)
    if mode == "chroma" and lab_to_rgb and int(n_components) == 3:
        resolved = [target_anchor[0], train_anchor[1], train_anchor[2]]
    else:
        resolved = train_anchor
    if float(anchor_blend) >= 1.0 - 1e-9:
        return resolved
    return _blend_anchor_bounds(resolved, target_anchor, float(anchor_blend))


def _embedding_to_rgb_with_anchor(
    emb: np.ndarray,
    mask: np.ndarray,
    anchor: list[tuple[float, float]],
    *,
    lab_to_rgb: bool,
    n_components: int,
    smooth_sigma: float,
) -> np.ndarray:
    result = np.asarray(emb, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=bool)
    for c, (p_lo, p_hi) in enumerate(anchor):
        ch = result[..., c]
        ch = (ch - p_lo) / max(p_hi - p_lo, 1e-8)
        ch = np.clip(ch, 0.0, 1.0)
        result[..., c] = ch
    result[~m] = 0.0
    result = _spatial_gaussian_smooth_channels(result, smooth_sigma)
    out = np.uint8(255.0 * result)
    out[~m] = 0
    if lab_to_rgb and n_components == 3:
        out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    out[~m] = 0
    return out


def _parse_color_anchor_mode(raw: Any, *, label: str) -> str:
    mode = str(raw).strip().lower()
    if mode not in ("full", "chroma"):
        raise ValueError(f"{label} must be 'full' or 'chroma', got {mode!r}")
    return mode


def _predict_rgb_with_optional_anchor(
    model: MSIParametricMiCSLMC,
    msi: np.ndarray,
    anchor: list[tuple[float, float]] | None,
    *,
    anchor_mode: str = "full",
    anchor_blend: float = 1.0,
) -> np.ndarray:
    if anchor is None:
        return _to_uint8_rgb(model.predict(msi))
    emb = model.predict_embedding(msi)
    mask = msi.sum(axis=-1) > 0
    lab_to_rgb = bool(getattr(model, "lab_to_rgb", True))
    n_components = int(getattr(model, "number_of_components", 3))
    render_anchor = _resolve_render_anchor(
        train_anchor=anchor,
        emb=emb,
        mask=mask,
        low=float(getattr(model, "predict_percentile_low", 0.001)),
        high=float(getattr(model, "predict_percentile_high", 99.999)),
        anchor_mode=anchor_mode,
        anchor_blend=float(anchor_blend),
        lab_to_rgb=lab_to_rgb,
        n_components=n_components,
    )
    rgb = _embedding_to_rgb_with_anchor(
        emb,
        mask,
        render_anchor,
        lab_to_rgb=lab_to_rgb,
        n_components=n_components,
        smooth_sigma=float(getattr(model, "predict_spatial_smooth_sigma", 0.0)),
    )
    return rgb


def _clone_mics_for_finetune(model: MSIParametricMiCSLMC) -> MSIParametricMiCSLMC:
    """Deep-copy a fitted MiCS model so per-target finetune does not mutate the source."""
    return copy.deepcopy(model)


def _finetune_settings(ga: Any) -> dict[str, Any]:
    mode = str(OmegaConf.select(ga, "finetune_mode", default="transfer")).strip().lower()
    if mode not in ("transfer", "miss_stratified"):
        raise ValueError(f"gallery.finetune_mode must be 'transfer' or 'miss_stratified', got {mode!r}")
    return {
        "mode": mode,
        "epochs": int(OmegaConf.select(ga, "finetune_epochs", default=10)),
        "lr_scale": float(OmegaConf.select(ga, "finetune_lr_scale", default=0.1)),
        "warmup": int(OmegaConf.select(ga, "finetune_warmup_epochs", default=0)),
        "skip_train": bool(OmegaConf.select(ga, "finetune_skip_train_slide", default=True)),
        "anchor": bool(OmegaConf.select(ga, "finetune_anchor_percentiles", default=True)),
        "display_anchor_mode": _parse_color_anchor_mode(
            OmegaConf.select(ga, "finetune_display_anchor", default="full"),
            label="gallery.finetune_display_anchor",
        ),
        "display_anchor_blend": float(
            OmegaConf.select(ga, "finetune_display_anchor_blend", default=1.0)
        ),
        "color_anchor_mode": _parse_color_anchor_mode(
            OmegaConf.select(ga, "finetune_color_anchor", default="chroma"),
            label="gallery.finetune_color_anchor",
        ),
        "color_anchor_blend": float(
            OmegaConf.select(ga, "finetune_color_anchor_blend", default=0.6)
        ),
    }


def _finetune_model_on_target(
    model: MSIParametricMiCSLMC,
    msi: np.ndarray,
    ft_cfg: dict[str, Any],
    pixel_sampling_cache: dict[str, Any] | None = None,
) -> None:
    kwargs = {
        "finetune_epochs": int(ft_cfg["epochs"]),
        "finetune_lr_scale": float(ft_cfg["lr_scale"]),
        "finetune_warmup_epochs": int(ft_cfg["warmup"]),
    }
    if ft_cfg["mode"] == "miss_stratified":
        model.finetune_miss_stratified(msi, **kwargs)
    else:
        model.finetune_transfer(msi, pixel_sampling_cache=pixel_sampling_cache, **kwargs)


def _train_and_predict_transfer_row(
    *,
    group_name: str,
    group_overrides: dict[str, Any],
    train_msi: np.ndarray,
    target_msis: list[np.ndarray],
    bench_cfg: DictConfig,
    cfg: DictConfig,
    seed: int,
    sampling_mode: str,
    run_dir: Path,
    train_slide_index: int = 0,
    cache_subdir: str | None = None,
    target_sampling_caches: list[dict[str, Any] | None] | None = None,
    need_train_color_anchor: bool = False,
    include_finetune_color: bool = False,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[dict[str, float]] | None,
    list[tuple[float, float]] | None,
]:
    run_seed = int(seed)
    overrides = _merged_group_overrides(cfg, group_overrides)
    model_kw = _mics_kwargs(bench_cfg, sampling_mode, run_seed, overrides)
    ga = cfg.gallery
    cache = None
    cache_leaf = cache_subdir or str(OmegaConf.select(ga, "transfer_cache_subdir", default="transfer"))
    if bool(OmegaConf.select(ga, "share_pixel_sampling", default=True)):
        cache_path_raw = OmegaConf.select(ga, "pixel_sampling_cache_path")
        cache_path = _abs_path(str(cache_path_raw)) if cache_path_raw else None
        cache = resolve_pixel_sampling_cache(
            train_msi,
            model_kw,
            out_dir=run_dir / "caches" / group_name / cache_leaf,
            share=True,
            cache_path=cache_path,
            strip_cluster_labels=True,
        )
    model = MSIParametricMiCSLMC(**model_kw)
    logger.info("[%s] transfer: training on train slide …", group_name)
    model.fit(train_msi, pixel_sampling_cache=cache)

    include_finetune = bool(OmegaConf.select(ga, "include_finetune_columns", default=False))
    needs_finetune = include_finetune or include_finetune_color
    ft_cfg = _finetune_settings(ga) if needs_finetune else None
    novelty_cfg = _novelty_cfg(cfg)
    novelty_enabled = bool(novelty_cfg.get("enabled"))
    bg = tuple(int(x) for x in OmegaConf.to_container(ga.background_rgb, resolve=True))
    anchor: list[tuple[float, float]] | None = None
    use_anchor = bool(need_train_color_anchor) or (
        needs_finetune and ft_cfg is not None and bool(ft_cfg["anchor"])
    )
    if use_anchor:
        anchor = _compute_train_color_anchor(model, train_msi)

    ref_train = _build_train_reference_matrix(model, novelty_cfg) if novelty_enabled else None
    include_relative = novelty_enabled and bool(novelty_cfg.get("relative_enabled"))
    xfer_rgbs: list[np.ndarray] = []
    ft_rgbs: list[np.ndarray] | None = [] if include_finetune else None
    ft_color_rgbs: list[np.ndarray] | None = [] if include_finetune_color else None
    novelty_rgbs: list[np.ndarray] | None = [] if novelty_enabled else None
    rel_novelty_rgbs: list[np.ndarray] | None = [] if include_relative else None
    novelty_summaries: list[dict[str, float]] | None = [] if novelty_enabled else None
    for i, msi in enumerate(target_msis):
        logger.info("[%s] transfer predict %d/%d", group_name, i + 1, len(target_msis))
        xfer_rgb = _predict_rgb_with_optional_anchor(
            model,
            msi,
            anchor if use_anchor else None,
        )
        xfer_rgbs.append(xfer_rgb)

        if novelty_rgbs is not None and ref_train is not None:
            valid_mask = msi.sum(axis=-1) > 0
            target_cache = None
            if target_sampling_caches is not None and i < len(target_sampling_caches):
                target_cache = target_sampling_caches[i]
            ref_self = (
                _build_target_self_reference_matrix(model, msi, target_cache, novelty_cfg)
                if include_relative
                else None
            )
            d_train, d_self = _novelty_distance_maps(
                model, msi, ref_train, ref_self, novelty_cfg
            )
            rel_map = (
                _relative_novelty_map(d_train, d_self, novelty_cfg) if d_self is not None else None
            )
            novelty_rgbs.append(_novelty_map_rgb(d_train, valid_mask, bg, novelty_cfg))
            if rel_novelty_rgbs is not None and rel_map is not None:
                rel_novelty_rgbs.append(_relative_novelty_map_rgb(rel_map, valid_mask, bg, novelty_cfg))
            if novelty_summaries is not None:
                summary = _novelty_summary(d_train, valid_mask)
                if rel_map is not None:
                    summary.update(_relative_novelty_summary(rel_map, valid_mask))
                novelty_summaries.append(summary)

        if ft_cfg is None:
            continue
        if ft_rgbs is None and ft_color_rgbs is None:
            continue
        if bool(ft_cfg["skip_train"]) and i == int(train_slide_index):
            logger.info("[%s] xfer+ft: skip finetune on train slide (reuse xfer)", group_name)
            if ft_rgbs is not None:
                ft_rgbs.append(xfer_rgb)
            if ft_color_rgbs is not None and anchor is not None:
                ft_color_rgbs.append(
                    _predict_rgb_with_optional_anchor(
                        model,
                        msi,
                        anchor,
                        anchor_mode=str(ft_cfg["color_anchor_mode"]),
                        anchor_blend=float(ft_cfg["color_anchor_blend"]),
                    )
                )
            continue
        logger.info("[%s] xfer+ft on target %d/%d", group_name, i + 1, len(target_msis))
        ft_cache = None
        if target_sampling_caches is not None and i < len(target_sampling_caches):
            ft_cache = target_sampling_caches[i]
        ft_model = _clone_mics_for_finetune(model)
        _finetune_model_on_target(ft_model, msi, ft_cfg, pixel_sampling_cache=ft_cache)
        if ft_rgbs is not None:
            ft_rgb = _predict_rgb_with_optional_anchor(
                ft_model,
                msi,
                anchor,
                anchor_mode=str(ft_cfg["display_anchor_mode"]),
                anchor_blend=float(ft_cfg["display_anchor_blend"]),
            )
            ft_rgbs.append(ft_rgb)
        if ft_color_rgbs is not None:
            ft_color_rgbs.append(
                _predict_rgb_with_optional_anchor(
                    ft_model,
                    msi,
                    anchor,
                    anchor_mode=str(ft_cfg["color_anchor_mode"]),
                    anchor_blend=float(ft_cfg["color_anchor_blend"]),
                )
            )
        del ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return xfer_rgbs, ft_rgbs, ft_color_rgbs, novelty_rgbs, rel_novelty_rgbs, novelty_summaries, anchor


def _train_and_predict_self_modes_row(
    *,
    group_name: str,
    group_overrides: dict[str, Any],
    target_msis: list[np.ndarray],
    slide_keys: list[str],
    bench_cfg: DictConfig,
    cfg: DictConfig,
    seed: int,
    sampling_mode: str,
    run_dir: Path,
    target_sampling_caches: list[dict[str, Any] | None] | None = None,
    include_self: bool,
    include_self_aligned: bool,
    train_color_anchor: list[tuple[float, float]] | None = None,
    self_aligned_anchor_mode: str = "chroma",
    self_aligned_anchor_blend: float = 1.0,
) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None]:
    """Self-trained MiCS; optionally render with train-slide color anchors (self+align)."""
    if not include_self and not include_self_aligned:
        return None, None
    if include_self_aligned and train_color_anchor is None:
        raise ValueError("self+align requires train_color_anchor from the transfer train slide")

    self_row: list[np.ndarray] | None = [] if include_self else None
    self_aligned_row: list[np.ndarray] | None = [] if include_self_aligned else None
    for i, (msi, slide_key) in enumerate(zip(target_msis, slide_keys, strict=True)):
        mode_bits = []
        if include_self:
            mode_bits.append("self")
        if include_self_aligned:
            mode_bits.append("self+align")
        logger.info(
            "[%s] %s fit+predict on %s (%d/%d)",
            group_name,
            "+".join(mode_bits),
            slide_key,
            i + 1,
            len(target_msis),
        )
        prebuilt = None
        if target_sampling_caches is not None and i < len(target_sampling_caches):
            prebuilt = target_sampling_caches[i]
        run_seed = int(seed)
        overrides = _merged_group_overrides(cfg, group_overrides)
        model_kw = _mics_kwargs(bench_cfg, sampling_mode, run_seed, overrides)
        ga = cfg.gallery
        cache = prebuilt
        cache_subdir = f"{group_name}/self/{slide_key}"
        if cache is None and bool(OmegaConf.select(ga, "share_pixel_sampling", default=True)):
            cache_path_raw = OmegaConf.select(ga, "pixel_sampling_cache_path")
            cache_path = _abs_path(str(cache_path_raw)) if cache_path_raw else None
            cache = resolve_pixel_sampling_cache(
                msi,
                model_kw,
                out_dir=run_dir / "caches" / cache_subdir,
                share=True,
                cache_path=cache_path,
                strip_cluster_labels=False,
            )
        model = MSIParametricMiCSLMC(**model_kw)
        model.fit(msi, pixel_sampling_cache=cache)
        if include_self and self_row is not None:
            self_row.append(_to_uint8_rgb(model.predict(msi)))
        if include_self_aligned and self_aligned_row is not None:
            self_aligned_row.append(
                _predict_rgb_with_optional_anchor(
                    model,
                    msi,
                    train_color_anchor,
                    anchor_mode=self_aligned_anchor_mode,
                    anchor_blend=self_aligned_anchor_blend,
                )
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return self_row, self_aligned_row


def _interleave_transfer_columns(
    transfer_row: list[np.ndarray],
    finetune_row: list[np.ndarray] | None,
    finetune_color_row: list[np.ndarray] | None,
    self_row: list[np.ndarray] | None,
    self_aligned_row: list[np.ndarray] | None,
    novelty_row: list[np.ndarray] | None = None,
    rel_novelty_row: list[np.ndarray] | None = None,
    he_row: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i, xfer in enumerate(transfer_row):
        if he_row is not None:
            out.append(he_row[i])
        out.append(xfer)
        if finetune_row is not None:
            out.append(finetune_row[i])
        if finetune_color_row is not None:
            out.append(finetune_color_row[i])
        if self_row is not None:
            out.append(self_row[i])
        if self_aligned_row is not None:
            out.append(self_aligned_row[i])
        if novelty_row is not None:
            out.append(novelty_row[i])
        if rel_novelty_row is not None:
            out.append(rel_novelty_row[i])
    return out


def _column_labels_for_modes(
    header: str,
    *,
    xfer_label: str,
    ft_label: str,
    ft_color_label: str,
    self_label: str,
    self_aligned_label: str,
    novelty_label: str,
    rel_novelty_label: str,
    include_finetune: bool,
    include_finetune_color: bool,
    include_self: bool,
    include_self_aligned: bool,
    include_novelty: bool,
    include_relative_novelty: bool,
) -> list[str]:
    labels = [f"{header}\n{xfer_label}"]
    if include_finetune:
        labels.append(f"{header}\n{ft_label}")
    if include_finetune_color:
        labels.append(f"{header}\n{ft_color_label}")
    if include_self:
        labels.append(f"{header}\n{self_label}")
    if include_self_aligned:
        labels.append(f"{header}\n{self_aligned_label}")
    if include_novelty:
        labels.append(f"{header}\n{novelty_label}")
    if include_relative_novelty:
        labels.append(f"{header}\n{rel_novelty_label}")
    return labels


def _append_mode_cells(
    out: list[np.ndarray],
    *,
    xfer: np.ndarray,
    finetune: np.ndarray | None,
    finetune_color: np.ndarray | None,
    self_rgb: np.ndarray | None,
    self_aligned_rgb: np.ndarray | None,
    novelty_rgb: np.ndarray | None,
    rel_novelty_rgb: np.ndarray | None,
    include_finetune: bool,
    include_finetune_color: bool,
    include_self: bool,
    include_self_aligned: bool,
    include_novelty: bool,
    include_relative_novelty: bool,
) -> None:
    out.append(xfer)
    if include_finetune and finetune is not None:
        out.append(finetune)
    if include_finetune_color and finetune_color is not None:
        out.append(finetune_color)
    if include_self and self_rgb is not None:
        out.append(self_rgb)
    if include_self_aligned and self_aligned_rgb is not None:
        out.append(self_aligned_rgb)
    if include_novelty and novelty_rgb is not None:
        out.append(novelty_rgb)
    if include_relative_novelty and rel_novelty_rgb is not None:
        out.append(rel_novelty_rgb)


GroupResult = tuple[
    str,
    list[np.ndarray],
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
]

TrainSlideResult = tuple[
    str,
    int,
    str,
    list[np.ndarray],
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
]


def _novelty_cfg(cfg: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.select(cfg, "gallery.novelty")
    if raw is None:
        enabled = bool(OmegaConf.select(cfg, "gallery.include_novelty_columns", default=False))
        return {
            "enabled": enabled,
            "relative_enabled": enabled,
            "space": "embedding",
            "reference": "coreset",
            "metric": "euclidean",
            "top_k": 5,
            "epsilon": 1e-6,
            "colormap": "magma",
            "relative_colormap": "coolwarm",
            "clip_percentile": 99.0,
            "relative_clip_percentile": 99.0,
            "chunk_size": 8192,
        }
    return {
        "enabled": bool(OmegaConf.select(raw, "enabled", default=True)),
        "relative_enabled": bool(OmegaConf.select(raw, "relative_enabled", default=True)),
        "space": str(OmegaConf.select(raw, "space", default="embedding")).strip().lower(),
        "reference": str(OmegaConf.select(raw, "reference", default="coreset")).strip().lower(),
        "metric": str(OmegaConf.select(raw, "metric", default="euclidean")).strip().lower(),
        "top_k": max(1, int(OmegaConf.select(raw, "top_k", default=5))),
        "epsilon": float(OmegaConf.select(raw, "epsilon", default=1e-6)),
        "colormap": str(OmegaConf.select(raw, "colormap", default="magma")).strip().lower(),
        "relative_colormap": str(
            OmegaConf.select(raw, "relative_colormap", default="coolwarm")
        ).strip().lower(),
        "clip_percentile": float(OmegaConf.select(raw, "clip_percentile", default=99.0)),
        "relative_clip_percentile": float(
            OmegaConf.select(raw, "relative_clip_percentile", default=99.0)
        ),
        "chunk_size": int(OmegaConf.select(raw, "chunk_size", default=8192)),
    }


def _build_train_reference_matrix(
    model: MSIParametricMiCSLMC,
    novelty_cfg: dict[str, Any],
) -> np.ndarray:
    return _reference_matrix_from_sampled(getattr(model, "sampled_data", None), model, novelty_cfg)


def _build_target_self_reference_matrix(
    model: MSIParametricMiCSLMC,
    target_msi: np.ndarray,
    target_cache: dict[str, Any] | None,
    novelty_cfg: dict[str, Any],
) -> np.ndarray:
    flat = np.asarray(target_msi, dtype=np.float32).reshape(-1, target_msi.shape[-1])
    if target_cache is not None:
        sampled_idx = np.asarray(target_cache["sampled_flat_indices"], dtype=np.int64).ravel()
        spectra = flat[sampled_idx]
        if str(novelty_cfg["reference"]).lower() == "coreset":
            core_idx = np.asarray(target_cache["indices"], dtype=np.int64).ravel()
            spectra = spectra[core_idx]
    else:
        valid = flat.sum(axis=-1) > 0
        tissue = flat[valid]
        if tissue.shape[0] == 0:
            raise ValueError("target self reference requires tissue pixels")
        step = max(1, tissue.shape[0] // 5000)
        spectra = tissue[::step]
    return _reference_matrix_from_spectra(np.asarray(spectra, dtype=np.float32), model, novelty_cfg)


def _reference_matrix_from_sampled(
    sampled: np.ndarray | None,
    model: MSIParametricMiCSLMC,
    novelty_cfg: dict[str, Any],
) -> np.ndarray:
    if sampled is None:
        raise ValueError("novelty reference requires sampled_data")
    ref_kind = str(novelty_cfg["reference"]).lower()
    if ref_kind == "coreset":
        idx = np.asarray(model.indices, dtype=np.int64).ravel()
        spectra = np.asarray(sampled, dtype=np.float32)[idx]
    else:
        spectra = np.asarray(sampled, dtype=np.float32)
    return _reference_matrix_from_spectra(spectra, model, novelty_cfg)


def _reference_matrix_from_spectra(
    spectra: np.ndarray,
    model: MSIParametricMiCSLMC,
    novelty_cfg: dict[str, Any],
) -> np.ndarray:
    if str(novelty_cfg["space"]).lower() == "spectrum":
        return spectra
    return model._infer_embedding_flat(spectra)


def _topk_mean_distances(dists: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 1 or dists.shape[1] <= 1:
        return dists.min(axis=1)
    k = min(int(top_k), int(dists.shape[1]))
    part = np.partition(dists, k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


def _novelty_distance_maps(
    model: MSIParametricMiCSLMC,
    target_msi: np.ndarray,
    ref_train: np.ndarray,
    ref_self: np.ndarray | None,
    novelty_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-pixel top-k mean distance to train and (optional) self reference banks."""
    valid = target_msi.sum(axis=-1) > 0
    h, w = target_msi.shape[:2]
    space = str(novelty_cfg["space"]).lower()
    metric = str(novelty_cfg["metric"]).lower()
    if metric == "l2":
        metric = "euclidean"
    top_k = max(1, int(novelty_cfg.get("top_k", 1)))
    chunk = max(256, int(novelty_cfg["chunk_size"]))

    if space == "spectrum":
        query = np.asarray(target_msi, dtype=np.float64).reshape(-1, target_msi.shape[-1])
    else:
        query = np.asarray(model.predict_embedding(target_msi), dtype=np.float64).reshape(
            -1, ref_train.shape[-1]
        )

    ref_a = np.asarray(ref_train, dtype=np.float64)
    ref_b = np.asarray(ref_self, dtype=np.float64) if ref_self is not None else None
    valid_flat = valid.ravel()
    q_valid = query[valid_flat]
    d_train_flat = np.empty(q_valid.shape[0], dtype=np.float64)
    d_self_flat = np.empty(q_valid.shape[0], dtype=np.float64) if ref_b is not None else None
    for start in range(0, q_valid.shape[0], chunk):
        end = min(start + chunk, q_valid.shape[0])
        block = q_valid[start:end]
        d_train_flat[start:end] = _topk_mean_distances(cdist(block, ref_a, metric=metric), top_k)
        if ref_b is not None and d_self_flat is not None:
            d_self_flat[start:end] = _topk_mean_distances(cdist(block, ref_b, metric=metric), top_k)

    out_train = np.full(h * w, np.nan, dtype=np.float64)
    out_train[valid_flat] = d_train_flat
    out_self = None
    if d_self_flat is not None:
        out_self = np.full(h * w, np.nan, dtype=np.float64)
        out_self[valid_flat] = d_self_flat
    return out_train.reshape(h, w), out_self.reshape(h, w) if out_self is not None else None


def _relative_novelty_map(
    d_train: np.ndarray,
    d_self: np.ndarray,
    novelty_cfg: dict[str, Any],
) -> np.ndarray:
    eps = float(novelty_cfg.get("epsilon", 1e-6))
    return np.log((np.asarray(d_self, dtype=np.float64) + eps) / (np.asarray(d_train, dtype=np.float64) + eps))


def _novelty_map_rgb(
    novelty: np.ndarray,
    valid_mask: np.ndarray,
    bg: tuple[int, int, int],
    novelty_cfg: dict[str, Any],
) -> np.ndarray:
    cmap = plt.get_cmap(str(novelty_cfg.get("colormap", "magma")))
    vals = np.asarray(novelty, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    tissue = mask & np.isfinite(vals)
    rgb = np.full((*vals.shape, 3), bg, dtype=np.uint8)
    if not np.any(tissue):
        return rgb
    pct = float(novelty_cfg.get("clip_percentile", 99.0))
    vmax = float(np.nanpercentile(vals[tissue], pct))
    vmax = max(vmax, 1e-6)
    norm = np.clip(vals / vmax, 0.0, 1.0)
    colored = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[mask] = colored[mask]
    return rgb


def _relative_novelty_map_rgb(
    rel_novelty: np.ndarray,
    valid_mask: np.ndarray,
    bg: tuple[int, int, int],
    novelty_cfg: dict[str, Any],
) -> np.ndarray:
    cmap = plt.get_cmap(str(novelty_cfg.get("relative_colormap", "coolwarm")))
    vals = np.asarray(rel_novelty, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    tissue = mask & np.isfinite(vals)
    rgb = np.full((*vals.shape, 3), bg, dtype=np.uint8)
    if not np.any(tissue):
        return rgb
    pct = float(novelty_cfg.get("relative_clip_percentile", 99.0))
    vmax = float(np.nanpercentile(np.abs(vals[tissue]), pct))
    vmax = max(vmax, 1e-6)
    norm = np.clip((vals + vmax) / (2.0 * vmax), 0.0, 1.0)
    colored = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[mask] = colored[mask]
    return rgb


def _novelty_summary(novelty: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    vals = np.asarray(novelty, dtype=np.float64)[np.asarray(valid_mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"novelty_mean": float("nan"), "novelty_p95": float("nan")}
    return {
        "novelty_mean": float(np.mean(vals)),
        "novelty_p95": float(np.percentile(vals, 95.0)),
    }


def _relative_novelty_summary(rel_novelty: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    vals = np.asarray(rel_novelty, dtype=np.float64)[np.asarray(valid_mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"rel_novelty_mean": float("nan"), "rel_novelty_p95": float("nan")}
    return {
        "rel_novelty_mean": float(np.mean(vals)),
        "rel_novelty_p95": float(np.percentile(vals, 95.0)),
    }


def _color_emd_cfg(cfg: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.select(cfg, "metrics.color_emd")
    if raw is None:
        return {"enabled": False, "bins": 32, "colorspace": "lab"}
    return {
        "enabled": bool(OmegaConf.select(raw, "enabled", default=True)),
        "bins": int(OmegaConf.select(raw, "bins", default=32)),
        "colorspace": str(OmegaConf.select(raw, "colorspace", default="lab")).strip().lower(),
    }


def _masked_color_pixels(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    colorspace: str,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(valid_mask, dtype=bool)
    if rgb.shape[:2] != mask.shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if colorspace == "rgb":
        pixels = rgb[mask]
    else:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        pixels = lab[mask]
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(pixels, dtype=np.float64)


def _earth_movers_color_histogram(
    rgb_a: np.ndarray,
    mask_a: np.ndarray,
    rgb_b: np.ndarray,
    mask_b: np.ndarray,
    *,
    bins: int,
    colorspace: str,
) -> float:
    """Mean 1D EMD (Wasserstein) over color channels on tissue pixels."""
    px_a = _masked_color_pixels(rgb_a, mask_a, colorspace=colorspace)
    px_b = _masked_color_pixels(rgb_b, mask_b, colorspace=colorspace)
    if px_a.size == 0 or px_b.size == 0:
        return float("nan")
    bins = max(8, int(bins))
    edges = np.linspace(0.0, 256.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    emds: list[float] = []
    for ch in range(min(3, px_a.shape[1])):
        ha, _ = np.histogram(px_a[:, ch], bins=edges, density=True)
        hb, _ = np.histogram(px_b[:, ch], bins=edges, density=True)
        ha = ha + 1e-12
        hb = hb + 1e-12
        ha = ha / ha.sum()
        hb = hb / hb.sum()
        emds.append(float(wasserstein_distance(centers, centers, ha, hb)))
    return float(np.mean(emds)) if emds else float("nan")


def _is_train_sweep_layout(layout: str) -> bool:
    return str(layout).strip().lower() in ("by_train_slide", "train_sweep", "train_matrix")


def _assemble_gallery_grid(
    *,
    group_results: list[GroupResult],
    train_results: list[TrainSlideResult] | None,
    infer_items: list[tuple[Path, dict[str, Any]]],
    train_path_s: str,
    layout: str,
    include_he: bool,
    he_label: str,
    include_finetune: bool,
    include_finetune_color: bool,
    include_self: bool,
    include_self_aligned: bool,
    include_novelty: bool,
    include_relative_novelty: bool,
    xfer_label: str,
    ft_label: str,
    ft_color_label: str,
    self_label: str,
    self_aligned_label: str,
    novelty_label: str,
    rel_novelty_label: str,
    he_row: list[np.ndarray] | None = None,
) -> tuple[list[list[np.ndarray]], list[str], list[str], set[int]]:
    he_column_indices: set[int] = set()
    layout_l = str(layout).strip().lower()
    if layout_l in ("by_setting", "setting", "group"):
        col_labels: list[str] = []
        for slide_i, (path, meta) in enumerate(infer_items):
            key_line = meta["slide_key"] + (" *" if str(path.resolve()) == train_path_s else "")
            col_base = len(col_labels)
            if include_he and he_row is not None:
                col_labels.append(f"{key_line}\n{he_label}")
                he_column_indices.add(col_base)
            col_labels.extend(
                _column_labels_for_modes(
                    key_line,
                    xfer_label=xfer_label,
                    ft_label=ft_label,
                    ft_color_label=ft_color_label,
                    self_label=self_label,
                    self_aligned_label=self_aligned_label,
                    novelty_label=novelty_label,
                    rel_novelty_label=rel_novelty_label,
                    include_finetune=include_finetune,
                    include_finetune_color=include_finetune_color,
                    include_self=include_self,
                    include_self_aligned=include_self_aligned,
                    include_novelty=include_novelty,
                    include_relative_novelty=include_relative_novelty,
                )
            )
        grid_rgb: list[list[np.ndarray]] = []
        row_labels: list[str] = []
        for (
            group_name,
            xfer_row,
            ft_row,
            ft_color_row,
            self_row,
            self_aligned_row,
            novelty_row,
            rel_novelty_row,
        ) in group_results:
            row_labels.append(group_name)
            grid_rgb.append(
                _interleave_transfer_columns(
                    xfer_row,
                    ft_row,
                    ft_color_row,
                    self_row,
                    self_aligned_row,
                    novelty_row,
                    rel_novelty_row,
                    he_row=he_row,
                )
            )
        return grid_rgb, row_labels, col_labels, he_column_indices

    if layout_l in ("by_slide", "slide", "image"):
        col_labels: list[str] = []
        if include_he and he_row is not None:
            col_labels.append(he_label)
            he_column_indices.add(0)
        for group_name, _, _, _, _, _, _, _ in group_results:
            col_labels.extend(
                _column_labels_for_modes(
                    group_name,
                    xfer_label=xfer_label,
                    ft_label=ft_label,
                    ft_color_label=ft_color_label,
                    self_label=self_label,
                    self_aligned_label=self_aligned_label,
                    novelty_label=novelty_label,
                    rel_novelty_label=rel_novelty_label,
                    include_finetune=include_finetune,
                    include_finetune_color=include_finetune_color,
                    include_self=include_self,
                    include_self_aligned=include_self_aligned,
                    include_novelty=include_novelty,
                    include_relative_novelty=include_relative_novelty,
                )
            )
        grid_rgb = []
        row_labels = []
        for slide_i, (path, meta) in enumerate(infer_items):
            row: list[np.ndarray] = []
            if include_he and he_row is not None:
                row.append(he_row[slide_i])
            for (
                _,
                xfer_row,
                ft_row,
                ft_color_row,
                self_row,
                self_aligned_row,
                novelty_row,
                rel_novelty_row,
            ) in group_results:
                _append_mode_cells(
                    row,
                    xfer=xfer_row[slide_i],
                    finetune=ft_row[slide_i] if ft_row is not None else None,
                    finetune_color=ft_color_row[slide_i] if ft_color_row is not None else None,
                    self_rgb=self_row[slide_i] if self_row is not None else None,
                    self_aligned_rgb=self_aligned_row[slide_i] if self_aligned_row is not None else None,
                    novelty_rgb=novelty_row[slide_i] if novelty_row is not None else None,
                    rel_novelty_rgb=rel_novelty_row[slide_i] if rel_novelty_row is not None else None,
                    include_finetune=include_finetune,
                    include_finetune_color=include_finetune_color,
                    include_self=include_self,
                    include_self_aligned=include_self_aligned,
                    include_novelty=include_novelty,
                    include_relative_novelty=include_relative_novelty,
                )
            grid_rgb.append(row)
            row_labels.append(meta["slide_key"] + (" *" if str(path.resolve()) == train_path_s else ""))
        return grid_rgb, row_labels, col_labels, he_column_indices

    if _is_train_sweep_layout(layout_l):
        if not train_results:
            raise ValueError("by_train_slide layout requires train_results")
        col_labels = []
        for slide_i, (_path, meta) in enumerate(infer_items):
            col_base = len(col_labels)
            if include_he and he_row is not None:
                col_labels.append(f"{meta['slide_key']}\n{he_label}")
                he_column_indices.add(col_base)
            col_labels.extend(
                _column_labels_for_modes(
                    str(meta["slide_key"]),
                    xfer_label=xfer_label,
                    ft_label=ft_label,
                    ft_color_label=ft_color_label,
                    self_label=self_label,
                    self_aligned_label=self_aligned_label,
                    novelty_label=novelty_label,
                    rel_novelty_label=rel_novelty_label,
                    include_finetune=include_finetune,
                    include_finetune_color=include_finetune_color,
                    include_self=include_self,
                    include_self_aligned=include_self_aligned,
                    include_novelty=include_novelty,
                    include_relative_novelty=include_relative_novelty,
                )
            )
        grid_rgb = []
        row_labels = []
        for (
            train_key,
            _train_i,
            _group,
            xfer_row,
            ft_row,
            ft_color_row,
            self_row,
            self_aligned_row,
            novelty_row,
            rel_novelty_row,
        ) in train_results:
            row: list[np.ndarray] = []
            for slide_i in range(len(infer_items)):
                if include_he and he_row is not None:
                    row.append(he_row[slide_i])
                _append_mode_cells(
                    row,
                    xfer=xfer_row[slide_i],
                    finetune=ft_row[slide_i] if ft_row is not None else None,
                    finetune_color=ft_color_row[slide_i] if ft_color_row is not None else None,
                    self_rgb=self_row[slide_i] if self_row is not None else None,
                    self_aligned_rgb=self_aligned_row[slide_i] if self_aligned_row is not None else None,
                    novelty_rgb=novelty_row[slide_i] if novelty_row is not None else None,
                    rel_novelty_rgb=rel_novelty_row[slide_i] if rel_novelty_row is not None else None,
                    include_finetune=include_finetune,
                    include_finetune_color=include_finetune_color,
                    include_self=include_self,
                    include_self_aligned=include_self_aligned,
                    include_novelty=include_novelty,
                    include_relative_novelty=include_relative_novelty,
                )
            grid_rgb.append(row)
            row_labels.append(f"train: {train_key}")
        return grid_rgb, row_labels, col_labels, he_column_indices

    raise ValueError(
        f"gallery.layout must be by_setting, by_slide, or by_train_slide, got {layout!r}"
    )


def _load_rank_cfg(cfg: DictConfig) -> DictConfig:
    raw = OmegaConf.select(cfg, "metrics.rank_config") or OmegaConf.select(
        cfg, "rank_config", default="scripts/configs/hd_methods_gallery.yaml"
    )
    rank_cfg = OmegaConf.load(str(_abs_path(str(raw))))
    rank_cfg.edge_detection.hd_method = "soft_landmark_contrast"
    return rank_cfg


def _hd_edge_sigmas(rank_cfg: DictConfig) -> tuple[list[float], str, bool]:
    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    ms_enabled = bool(getattr(multi, "enabled", False)) if multi is not None else False
    if not ms_enabled:
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    spatial_enabled = bool(getattr(rank_cfg.spatial_normalization, "enabled", True))
    return sigmas, aggregation, spatial_enabled


def _build_eval_contexts(
    msis: list[np.ndarray],
    infer_items: list[tuple[Path, dict[str, Any]]],
    rank_cfg: DictConfig,
    *,
    skip_hd_edges: bool = False,
) -> list[dict[str, Any]]:
    sigmas, aggregation, spatial_enabled = _hd_edge_sigmas(rank_cfg)
    contexts: list[dict[str, Any]] = []
    for msi, (path, meta) in zip(msis, infer_items, strict=True):
        valid_mask = msi.sum(axis=-1) > 0
        if not np.any(valid_mask):
            raise ValueError(f"Empty valid mask for {path}")
        hd_edge_n = None
        if not skip_hd_edges:
            hd_edge_n = _compute_hd_edge_n_for_method(
                rank_cfg,
                msi,
                valid_mask,
                getattr(rank_cfg, "hd_edges", None),
                "soft_landmark_contrast",
                sigmas,
                aggregation,
                spatial_enabled,
            )
        contexts.append(
            {
                "slide_key": str(meta["slide_key"]),
                "npy_path": str(path),
                "polarity": _slide_polarity(str(meta["slide_key"]), path),
                "valid_mask": valid_mask,
                "hd_edge_n": hd_edge_n,
            }
        )
    return contexts


def _composite_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for key, w in weights.items():
        if w <= 0:
            continue
        val = metrics.get(key)
        if val is None or not np.isfinite(val):
            continue
        num += float(w) * float(val)
        den += float(w)
    return num / den if den > 0 else float("nan")


def _score_viz_rgb(
    viz_rgb: np.ndarray,
    ctx: dict[str, Any],
    rank_cfg: DictConfig,
    *,
    square: bool,
    metric_weights: dict[str, float],
) -> dict[str, float]:
    valid_mask = ctx["valid_mask"]
    hd_edge_n = ctx.get("hd_edge_n")
    rgb = np.asarray(viz_rgb, dtype=np.uint8)
    dice = float("nan")
    if hd_edge_n is not None:
        viz_edge_n = _compute_viz_edge_n(rgb, valid_mask, rank_cfg)
        hd_d, viz_d = _edges_for_continuous_metrics(hd_edge_n, viz_edge_n, square=square)
        dice = float(_compute_continuous_dice(hd_d, viz_d, valid_mask))
    cq = _compute_viz_contrast_color_metrics(rgb, valid_mask)
    out = {
        "continuous_dice": dice,
        "luminance_rms_contrast": float(cq["luminance_rms_contrast"]),
        "lab_chroma_entropy": float(cq["lab_chroma_entropy"]),
    }
    out["composite_score"] = _composite_score(out, metric_weights)
    return out


def _short_slide_label(slide_key: str) -> str:
    return (
        str(slide_key)
        .replace("_MSimaging_", "\n")
        .replace("_RMS", "")
        .replace("_", " ")
    )


def _metric_weights_from_cfg(cfg: DictConfig) -> dict[str, float]:
    raw = OmegaConf.to_container(OmegaConf.select(cfg, "metrics.weights"), resolve=True)
    if isinstance(raw, dict) and raw:
        return {str(k): float(v) for k, v in raw.items()}
    return {"continuous_dice": 1.0, "luminance_rms_contrast": 0.0, "lab_chroma_entropy": 0.0}


def _write_metrics_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    fields = [
        "mics_group",
        "train_slide_key",
        "slide_key",
        "mode",
        "polarity",
        "is_train_slide",
        "continuous_dice",
        "luminance_rms_contrast",
        "lab_chroma_entropy",
        "composite_score",
        "color_emd_to_train_ref",
        "novelty_mean",
        "novelty_p95",
        "rel_novelty_mean",
        "rel_novelty_p95",
        "delta_self_minus_transfer",
        "delta_self_minus_transfer_finetune",
        "delta_transfer_finetune_minus_transfer",
        "cell_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _append_gallery_cell_records(
    *,
    group_name: str,
    train_slide_key: str | None,
    infer_items: list[tuple[Path, dict[str, Any]]],
    xfer_row: list[np.ndarray],
    ft_row: list[np.ndarray] | None,
    ft_color_row: list[np.ndarray] | None,
    self_row: list[np.ndarray] | None,
    self_aligned_row: list[np.ndarray] | None,
    novelty_row: list[np.ndarray] | None,
    rel_novelty_row: list[np.ndarray] | None,
    novelty_summaries: list[dict[str, float]] | None,
    train_ref_rgb: np.ndarray | None,
    train_ref_mask: np.ndarray | None,
    cells_dir: Path,
    eval_ctxs: list[dict[str, Any]] | None,
    rank_cfg: DictConfig | None,
    square_dice: bool,
    metric_weights: dict[str, float],
    color_emd_cfg: dict[str, Any],
    include_finetune: bool,
    include_finetune_color: bool,
    include_self: bool,
    include_self_aligned: bool,
    include_novelty: bool,
    include_relative_novelty: bool,
    novelty_label: str,
    rel_novelty_label: str,
    train_path_s: str,
    cell_name_prefix: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    emd_enabled = bool(color_emd_cfg.get("enabled")) and train_ref_rgb is not None and train_ref_mask is not None

    for slide_i, (path, meta) in enumerate(infer_items):
        target_key = str(meta["slide_key"])
        is_train_target = str(path.resolve()) == train_path_s
        prefix = cell_name_prefix or f"{group_name}__"

        def _emd(rgb: np.ndarray) -> float | None:
            if not emd_enabled or eval_ctxs is None:
                return None
            return _earth_movers_color_histogram(
                rgb,
                eval_ctxs[slide_i]["valid_mask"],
                train_ref_rgb,
                train_ref_mask,
                bins=int(color_emd_cfg["bins"]),
                colorspace=str(color_emd_cfg["colorspace"]),
            )

        xfer_rgb = xfer_row[slide_i]
        xfer_name = f"{prefix}{target_key}__transfer.png"
        xfer_path = cells_dir / xfer_name
        Image.fromarray(xfer_rgb).save(xfer_path)
        xfer_entry: dict[str, Any] = {
            "mics_group": group_name,
            "train_slide_key": train_slide_key,
            "slide_key": target_key,
            "npy_path": str(path),
            "mode": "transfer",
            "polarity": eval_ctxs[slide_i]["polarity"] if eval_ctxs else None,
            "is_train_slide": is_train_target,
            "cell_path": str(xfer_path),
        }
        if eval_ctxs is not None and rank_cfg is not None:
            xfer_entry.update(
                _score_viz_rgb(
                    xfer_rgb,
                    eval_ctxs[slide_i],
                    rank_cfg,
                    square=square_dice,
                    metric_weights=metric_weights,
                )
            )
        emd_v = _emd(xfer_rgb)
        if emd_v is not None:
            xfer_entry["color_emd_to_train_ref"] = emd_v
        if novelty_summaries is not None:
            xfer_entry.update(novelty_summaries[slide_i])
        manifest_rows.append(xfer_entry)
        metric_records.append(dict(xfer_entry))

        if include_finetune and ft_row is not None:
            ft_rgb = ft_row[slide_i]
            ft_name = f"{prefix}{target_key}__transfer_finetune.png"
            ft_path = cells_dir / ft_name
            Image.fromarray(ft_rgb).save(ft_path)
            ft_entry: dict[str, Any] = {
                "mics_group": group_name,
                "train_slide_key": train_slide_key,
                "slide_key": target_key,
                "npy_path": str(path),
                "mode": "transfer_finetune",
                "polarity": eval_ctxs[slide_i]["polarity"] if eval_ctxs else None,
                "is_train_slide": is_train_target,
                "cell_path": str(ft_path),
            }
            if eval_ctxs is not None and rank_cfg is not None:
                ft_entry.update(
                    _score_viz_rgb(
                        ft_rgb,
                        eval_ctxs[slide_i],
                        rank_cfg,
                        square=square_dice,
                        metric_weights=metric_weights,
                    )
                )
            emd_v = _emd(ft_rgb)
            if emd_v is not None:
                ft_entry["color_emd_to_train_ref"] = emd_v
            manifest_rows.append(ft_entry)
            metric_records.append(dict(ft_entry))

        if include_finetune_color and ft_color_row is not None:
            ft_color_rgb = ft_color_row[slide_i]
            ft_color_name = f"{prefix}{target_key}__transfer_finetune_color.png"
            ft_color_path = cells_dir / ft_color_name
            Image.fromarray(ft_color_rgb).save(ft_color_path)
            ft_color_entry: dict[str, Any] = {
                "mics_group": group_name,
                "train_slide_key": train_slide_key,
                "slide_key": target_key,
                "npy_path": str(path),
                "mode": "transfer_finetune_color",
                "polarity": eval_ctxs[slide_i]["polarity"] if eval_ctxs else None,
                "is_train_slide": is_train_target,
                "cell_path": str(ft_color_path),
            }
            if eval_ctxs is not None and rank_cfg is not None:
                ft_color_entry.update(
                    _score_viz_rgb(
                        ft_color_rgb,
                        eval_ctxs[slide_i],
                        rank_cfg,
                        square=square_dice,
                        metric_weights=metric_weights,
                    )
                )
            emd_v = _emd(ft_color_rgb)
            if emd_v is not None:
                ft_color_entry["color_emd_to_train_ref"] = emd_v
            manifest_rows.append(ft_color_entry)
            metric_records.append(dict(ft_color_entry))

        if include_self and self_row is not None:
            self_rgb = self_row[slide_i]
            self_name = f"{prefix}{target_key}__self.png"
            self_path = cells_dir / self_name
            Image.fromarray(self_rgb).save(self_path)
            self_entry: dict[str, Any] = {
                "mics_group": group_name,
                "train_slide_key": train_slide_key,
                "slide_key": target_key,
                "npy_path": str(path),
                "mode": "self",
                "polarity": eval_ctxs[slide_i]["polarity"] if eval_ctxs else None,
                "is_train_slide": is_train_target,
                "cell_path": str(self_path),
            }
            if eval_ctxs is not None and rank_cfg is not None:
                self_entry.update(
                    _score_viz_rgb(
                        self_rgb,
                        eval_ctxs[slide_i],
                        rank_cfg,
                        square=square_dice,
                        metric_weights=metric_weights,
                    )
                )
            emd_v = _emd(self_rgb)
            if emd_v is not None:
                self_entry["color_emd_to_train_ref"] = emd_v
            manifest_rows.append(self_entry)
            metric_records.append(dict(self_entry))

        if include_self_aligned and self_aligned_row is not None:
            aligned_rgb = self_aligned_row[slide_i]
            aligned_name = f"{prefix}{target_key}__self_aligned.png"
            aligned_path = cells_dir / aligned_name
            Image.fromarray(aligned_rgb).save(aligned_path)
            aligned_entry: dict[str, Any] = {
                "mics_group": group_name,
                "train_slide_key": train_slide_key,
                "slide_key": target_key,
                "npy_path": str(path),
                "mode": "self_aligned",
                "polarity": eval_ctxs[slide_i]["polarity"] if eval_ctxs else None,
                "is_train_slide": is_train_target,
                "cell_path": str(aligned_path),
            }
            if eval_ctxs is not None and rank_cfg is not None:
                aligned_entry.update(
                    _score_viz_rgb(
                        aligned_rgb,
                        eval_ctxs[slide_i],
                        rank_cfg,
                        square=square_dice,
                        metric_weights=metric_weights,
                    )
                )
            emd_v = _emd(aligned_rgb)
            if emd_v is not None:
                aligned_entry["color_emd_to_train_ref"] = emd_v
            manifest_rows.append(aligned_entry)
            metric_records.append(dict(aligned_entry))

        if include_novelty and novelty_row is not None:
            novelty_rgb = novelty_row[slide_i]
            novelty_name = f"{prefix}{target_key}__novelty.png"
            novelty_path = cells_dir / novelty_name
            Image.fromarray(novelty_rgb).save(novelty_path)
            novelty_entry: dict[str, Any] = {
                "mics_group": group_name,
                "train_slide_key": train_slide_key,
                "slide_key": target_key,
                "npy_path": str(path),
                "mode": "novelty",
                "polarity": eval_ctxs[slide_i]["polarity"] if eval_ctxs else None,
                "is_train_slide": is_train_target,
                "cell_path": str(novelty_path),
            }
            if eval_ctxs is not None and novelty_summaries is not None:
                novelty_entry.update(novelty_summaries[slide_i])
            manifest_rows.append(novelty_entry)
            metric_records.append(dict(novelty_entry))

        if include_relative_novelty and rel_novelty_row is not None:
            rel_rgb = rel_novelty_row[slide_i]
            rel_name = f"{prefix}{target_key}__rel_novelty.png"
            rel_path = cells_dir / rel_name
            Image.fromarray(rel_rgb).save(rel_path)
            rel_entry: dict[str, Any] = {
                "mics_group": group_name,
                "train_slide_key": train_slide_key,
                "slide_key": target_key,
                "npy_path": str(path),
                "mode": "rel_novelty",
                "polarity": eval_ctxs[slide_i]["polarity"] if eval_ctxs else None,
                "is_train_slide": is_train_target,
                "cell_path": str(rel_path),
            }
            if novelty_summaries is not None:
                rel_entry.update(
                    {
                        k: novelty_summaries[slide_i][k]
                        for k in ("rel_novelty_mean", "rel_novelty_p95")
                        if k in novelty_summaries[slide_i]
                    }
                )
            manifest_rows.append(rel_entry)
            metric_records.append(dict(rel_entry))

    return manifest_rows, metric_records


def _save_color_emd_figures(
    records: list[dict[str, Any]],
    run_dir: Path,
    *,
    infer_items: list[tuple[Path, dict[str, Any]]],
) -> dict[str, str]:
    if not records or not any("color_emd_to_train_ref" in r for r in records):
        return {}
    out_paths: dict[str, str] = {}
    train_keys = sorted({str(r["train_slide_key"]) for r in records if r.get("train_slide_key")})
    target_keys = [str(m["slide_key"]) for _, m in infer_items]
    short_targets = [_short_slide_label(k) for k in target_keys]
    has_finetune = any(str(r["mode"]) == "transfer_finetune" for r in records)
    has_ft_color = any(str(r["mode"]) == "transfer_finetune_color" for r in records)
    has_self_aligned = any(str(r["mode"]) == "self_aligned" for r in records)
    mode_specs: list[tuple[str, str]] = [("transfer", "xfer")]
    if has_finetune:
        mode_specs.append(("transfer_finetune", "xfer+ft"))
    if has_ft_color:
        mode_specs.append(("transfer_finetune_color", "xfer+ft+align"))
    mode_specs.append(("self", "self"))
    if has_self_aligned:
        mode_specs.append(("self_aligned", "self+align"))

    for mode, label in mode_specs:
        mat = np.full((len(train_keys), len(target_keys)), np.nan, dtype=np.float64)
        for ti, tk in enumerate(train_keys):
            for si, sk in enumerate(target_keys):
                rec = next(
                    (
                        r
                        for r in records
                        if str(r.get("train_slide_key")) == tk
                        and str(r["slide_key"]) == sk
                        and str(r["mode"]) == mode
                    ),
                    None,
                )
                if rec is None:
                    continue
                v = rec.get("color_emd_to_train_ref")
                if v is not None and np.isfinite(float(v)):
                    mat[ti, si] = float(v)
        fig, ax = plt.subplots(
            figsize=(max(10, len(target_keys) * 0.55), max(3.5, len(train_keys) * 0.55)),
            dpi=160,
        )
        vmax = float(np.nanpercentile(mat, 95)) if np.any(np.isfinite(mat)) else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=vmax)
        ax.set_yticks(range(len(train_keys)), train_keys, fontsize=7)
        ax.set_xticks(range(len(target_keys)), short_targets, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"Color EMD to train reference ({label})")
        ax.set_xlabel("Target slide")
        ax.set_ylabel("Train slide")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="EMD (LAB hist.)")
        fig.tight_layout()
        p = run_dir / f"metrics_color_emd_{label.replace('+', '_')}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        out_paths[f"color_emd_{label}"] = str(p)

    # Per-target bars: mean EMD over train slides
    fig, ax = plt.subplots(figsize=(max(12, len(target_keys) * 0.45), 4.5), dpi=160)
    x = np.arange(len(target_keys))
    n_modes = len(mode_specs)
    w = 0.8 / max(1, n_modes)
    colors = ("#4C72B0", "#C44E52", "#55A868")
    for mi, (mode, label) in enumerate(mode_specs):
        means = []
        for sk in target_keys:
            vals = [
                float(r["color_emd_to_train_ref"])
                for r in records
                if str(r["slide_key"]) == sk
                and str(r["mode"]) == mode
                and r.get("color_emd_to_train_ref") is not None
                and np.isfinite(float(r["color_emd_to_train_ref"]))
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
        off = (mi - (n_modes - 1) / 2.0) * w
        ax.bar(x + off, means, width=w, label=label, color=colors[mi % len(colors)], alpha=0.9)
    ax.set_xticks(x, short_targets, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean color EMD to train ref")
    ax.set_title("Mean LAB histogram EMD vs train-slide reference (lower = closer palette)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    p_bar = run_dir / "metrics_color_emd_mean_by_target.png"
    fig.savefig(p_bar, bbox_inches="tight")
    plt.close(fig)
    out_paths["color_emd_mean_bars"] = str(p_bar)
    return out_paths


def _attach_delta_records(records: list[dict[str, Any]]) -> None:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rec in records:
        train_k = str(rec.get("train_slide_key") or "")
        key = (str(rec["mics_group"]), train_k, str(rec["slide_key"]))
        index.setdefault(key, {})[str(rec["mode"])] = rec
    for rec in records:
        rec["delta_self_minus_transfer"] = float("nan")
        rec["delta_self_minus_transfer_finetune"] = float("nan")
        rec["delta_transfer_finetune_minus_transfer"] = float("nan")
        train_k = str(rec.get("train_slide_key") or "")
        pair = index.get((str(rec["mics_group"]), train_k, str(rec["slide_key"])), {})
        xfer = pair.get("transfer")
        xfer_ft = pair.get("transfer_finetune")
        self_rec = pair.get("self")
        if rec["mode"] == "self" and xfer is not None:
            xd = float(xfer.get("continuous_dice", float("nan")))
            sd = float(rec.get("continuous_dice", float("nan")))
            if np.isfinite(xd) and np.isfinite(sd):
                rec["delta_self_minus_transfer"] = sd - xd
        if rec["mode"] == "self" and xfer_ft is not None:
            xfd = float(xfer_ft.get("continuous_dice", float("nan")))
            sd = float(rec.get("continuous_dice", float("nan")))
            if np.isfinite(xfd) and np.isfinite(sd):
                rec["delta_self_minus_transfer_finetune"] = sd - xfd
        if rec["mode"] == "transfer_finetune" and xfer is not None:
            xd = float(xfer.get("continuous_dice", float("nan")))
            xfd = float(rec.get("continuous_dice", float("nan")))
            if np.isfinite(xd) and np.isfinite(xfd):
                rec["delta_transfer_finetune_minus_transfer"] = xfd - xd


def _save_metrics_figures(
    records: list[dict[str, Any]],
    run_dir: Path,
    *,
    train_slide_key: str,
) -> dict[str, str]:
    if not records:
        return {}
    out_paths: dict[str, str] = {}
    groups = sorted({str(r["mics_group"]) for r in records})
    slides = sorted({str(r["slide_key"]) for r in records})
    short_slides = [_short_slide_label(s) for s in slides]
    has_finetune = any(str(r["mode"]) == "transfer_finetune" for r in records)

    # --- Heatmap: self - transfer continuous Dice ---
    delta_mat = np.full((len(groups), len(slides)), np.nan, dtype=np.float64)
    for gi, g in enumerate(groups):
        for si, s in enumerate(slides):
            self_rec = next(
                (r for r in records if r["mics_group"] == g and r["slide_key"] == s and r["mode"] == "self"),
                None,
            )
            if self_rec is not None:
                d = self_rec.get("delta_self_minus_transfer")
                if d is not None and np.isfinite(float(d)):
                    delta_mat[gi, si] = float(d)

    fig, ax = plt.subplots(figsize=(max(10, len(slides) * 0.55), max(3.5, len(groups) * 0.65)), dpi=160)
    im = ax.imshow(delta_mat, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.15)
    ax.set_yticks(range(len(groups)), groups)
    ax.set_xticks(range(len(slides)), short_slides, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Self − transfer continuous Dice (train: {train_slide_key})")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Δ Dice")
    fig.tight_layout()
    p_delta = run_dir / "metrics_delta_self_minus_transfer.png"
    fig.savefig(p_delta, bbox_inches="tight")
    plt.close(fig)
    out_paths["delta_heatmap"] = str(p_delta)

    # --- Per-setting paired bars (continuous Dice) ---
    n_g = len(groups)
    fig, axes = plt.subplots(n_g, 1, figsize=(max(12, len(slides) * 0.45), 2.8 * n_g), dpi=160, sharex=True)
    if n_g == 1:
        axes = [axes]
    x = np.arange(len(slides))
    if has_finetune:
        w = 0.24
        offsets = (-w, 0.0, w)
        colors = ("#4C72B0", "#C44E52", "#55A868")
        labels = ("transfer", "xfer+ft", "self")
    else:
        w = 0.36
        offsets = (-w / 2, w / 2)
        colors = ("#4C72B0", "#55A868")
        labels = ("transfer", "self")
    for gi, g in enumerate(groups):
        ax = axes[gi]
        series: list[list[float]] = []
        for mode in ("transfer", "transfer_finetune", "self") if has_finetune else ("transfer", "self"):
            vals = []
            for s in slides:
                rec = next(
                    (r for r in records if r["mics_group"] == g and r["slide_key"] == s and r["mode"] == mode),
                    None,
                )
                vals.append(float(rec["continuous_dice"]) if rec else np.nan)
            series.append(vals)
        for off, ys, color, label in zip(offsets, series, colors, labels, strict=True):
            ax.bar(x + off, ys, width=w, label=label, color=color, alpha=0.9)
        ax.set_ylabel("Dice")
        ax.set_title(g)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    axes[-1].set_xticks(x, short_slides, rotation=45, ha="right", fontsize=7)
    title = "Continuous Dice: transfer vs xfer+ft vs self" if has_finetune else "Continuous Dice: transfer vs self-trained"
    fig.suptitle(title, y=1.01, fontsize=12)
    fig.tight_layout()
    p_bars = run_dir / "metrics_dice_transfer_vs_self_by_slide.png"
    fig.savefig(p_bars, bbox_inches="tight")
    plt.close(fig)
    out_paths["dice_bars"] = str(p_bars)

    # --- Mean over slides per setting ---
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.4), 4.2), dpi=160)
    xg = np.arange(len(groups))
    if has_finetune:
        w = 0.24
        offsets = (-w, 0.0, w)
        colors = ("#4C72B0", "#C44E52", "#55A868")
        labels = ("transfer (mean)", "xfer+ft (mean)", "self (mean)")
        modes = ("transfer", "transfer_finetune", "self")
    else:
        w = 0.36
        offsets = (-w / 2, w / 2)
        colors = ("#4C72B0", "#55A868")
        labels = ("transfer (mean)", "self (mean)")
        modes = ("transfer", "self")
    for off, mode, color, label in zip(offsets, modes, colors, labels, strict=True):
        means = []
        stds = []
        for g in groups:
            vals = [
                float(r["continuous_dice"])
                for r in records
                if r["mics_group"] == g and r["mode"] == mode and np.isfinite(float(r["continuous_dice"]))
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        ax.bar(xg + off, means, width=w, yerr=stds, label=label, color=color, capsize=3)
    ax.set_xticks(xg, groups, rotation=20, ha="right")
    ax.set_ylabel("Continuous Dice (mean ± std over slides)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"Aggregate comparison (train slide: {train_slide_key})")
    fig.tight_layout()
    p_mean = run_dir / "metrics_dice_mean_by_setting.png"
    fig.savefig(p_mean, bbox_inches="tight")
    plt.close(fig)
    out_paths["mean_bars"] = str(p_mean)

    # --- Win rate summary text ---
    n_self_better = sum(
        1
        for r in records
        if r["mode"] == "self"
        and np.isfinite(float(r.get("delta_self_minus_transfer", np.nan)))
        and float(r["delta_self_minus_transfer"]) > 0
    )
    n_pairs = sum(1 for r in records if r["mode"] == "self")
    n_ft_better = sum(
        1
        for r in records
        if r["mode"] == "transfer_finetune"
        and np.isfinite(float(r.get("delta_transfer_finetune_minus_transfer", np.nan)))
        and float(r["delta_transfer_finetune_minus_transfer"]) > 0
    )
    n_ft_pairs = sum(1 for r in records if r["mode"] == "transfer_finetune")
    summary_lines = [
        f"Train slide: {train_slide_key}",
        f"Metric pairs (self vs transfer): {n_pairs}",
        f"Self better (Δ Dice > 0): {n_self_better} ({100.0 * n_self_better / max(n_pairs, 1):.1f}%)",
    ]
    if has_finetune:
        summary_lines.extend(
            [
                f"Finetune pairs (xfer+ft vs transfer): {n_ft_pairs}",
                f"xfer+ft better (Δ Dice > 0): {n_ft_better} ({100.0 * n_ft_better / max(n_ft_pairs, 1):.1f}%)",
            ]
        )
    summary_lines.extend(
        [
            "",
            "Per setting (mean Δ Dice = mean(self) - mean(transfer)):",
        ]
    )
    for g in groups:
        deltas = [
            float(r["delta_self_minus_transfer"])
            for r in records
            if r["mics_group"] == g and r["mode"] == "self" and np.isfinite(float(r["delta_self_minus_transfer"]))
        ]
        md = float(np.mean(deltas)) if deltas else float("nan")
        summary_lines.append(f"  {g}: {md:+.4f}")
    if has_finetune:
        summary_lines.append("")
        summary_lines.append("Per setting (mean Δ Dice = mean(xfer+ft) - mean(transfer)):")
        for g in groups:
            deltas = [
                float(r["delta_transfer_finetune_minus_transfer"])
                for r in records
                if r["mics_group"] == g
                and r["mode"] == "transfer_finetune"
                and np.isfinite(float(r["delta_transfer_finetune_minus_transfer"]))
            ]
            md = float(np.mean(deltas)) if deltas else float("nan")
            summary_lines.append(f"  {g}: {md:+.4f}")
    p_txt = run_dir / "metrics_transfer_vs_self_summary.txt"
    p_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    out_paths["summary_txt"] = str(p_txt)

    return out_paths


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_transfer_carstenhopf")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ga = cfg.gallery
    seed = int(cfg.seed)
    _seed_everything(seed)

    npy_paths = _resolve_npy_paths(cfg)
    compatible = _filter_compatible_slides(npy_paths, ga.spectral_filter)
    infer_pol_raw = OmegaConf.select(ga, "infer_polarity")
    if infer_pol_raw is not None and str(infer_pol_raw).strip().lower() not in (
        "",
        "any",
        "all",
        "null",
        "none",
        "~",
    ):
        compatible = _filter_by_polarity(compatible, str(infer_pol_raw))
        logger.info(
            "Infer polarity %s: %d compatible slide(s)",
            infer_pol_raw,
            len(compatible),
        )
    if len(compatible) < 2:
        raise ValueError(f"Need at least 2 compatible slides, found {len(compatible)}")

    train_idx, train_path, train_meta = _resolve_train_slide(compatible, ga)
    infer_items = list(compatible)
    max_inf = OmegaConf.select(ga, "max_slides")
    layout = str(OmegaConf.select(ga, "layout", default="by_setting"))
    train_sweep = _is_train_sweep_layout(layout)
    if max_inf is not None and not train_sweep:
        max_inf = int(max_inf)
        others = [item for i, item in enumerate(infer_items) if i != train_idx]
        infer_items = [compatible[train_idx]] + others[: max(0, max_inf - 1)]
    elif max_inf is not None and train_sweep:
        infer_items = infer_items[: int(max_inf)]

    norm_mode = _resolve_normalization_mode(str(cfg.data.normalization))
    transpose_msi = bool(cfg.data.transpose_msi)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    include_self = bool(OmegaConf.select(ga, "include_self_columns", default=True))
    include_self_aligned = bool(OmegaConf.select(ga, "include_self_aligned_columns", default=False))
    include_finetune = bool(OmegaConf.select(ga, "include_finetune_columns", default=False))
    include_finetune_color = bool(OmegaConf.select(ga, "include_finetune_color_columns", default=False))
    novelty_cfg = _novelty_cfg(cfg)
    include_novelty = bool(novelty_cfg.get("enabled"))
    include_relative_novelty = include_novelty and bool(novelty_cfg.get("relative_enabled"))
    xfer_label = str(OmegaConf.select(ga, "transfer_column_label", default="xfer"))
    ft_label = str(OmegaConf.select(ga, "finetune_column_label", default="xfer+ft"))
    ft_color_label = str(OmegaConf.select(ga, "finetune_color_column_label", default="xfer+ft+align"))
    self_label = str(OmegaConf.select(ga, "self_column_label", default="self"))
    self_aligned_label = str(
        OmegaConf.select(ga, "self_aligned_column_label", default="self+align")
    )
    self_aligned_anchor_mode = str(
        OmegaConf.select(ga, "self_aligned_color_anchor", default="chroma")
    ).strip().lower()
    if self_aligned_anchor_mode not in ("full", "chroma"):
        raise ValueError(
            f"gallery.self_aligned_color_anchor must be 'full' or 'chroma', "
            f"got {self_aligned_anchor_mode!r}"
        )
    novelty_label = str(OmegaConf.select(ga, "novelty_column_label", default="d_train"))
    rel_novelty_label = str(
        OmegaConf.select(ga, "relative_novelty_column_label", default="log d_self/d_train")
    )
    include_he = _he_enabled(cfg)
    he_label = str(OmegaConf.select(ga, "he_column_label", default="H&E"))
    bg = tuple(int(x) for x in OmegaConf.to_container(ga.background_rgb, resolve=True))
    train_path_s = str(train_path.resolve()) if not train_sweep else ""

    if train_sweep:
        logger.info("Gallery layout=by_train_slide, %d target slides (train sweep)", len(infer_items))
    else:
        logger.info("Train slide: %s (%s)", train_meta["slide_key"], train_path)
        logger.info("Gallery layout=%s, %d compatible slides", layout, len(infer_items))

    msis: list[np.ndarray] = []
    slide_keys: list[str] = []
    train_slide_index = 0
    if not train_sweep:
        train_slide_index = next(
            i for i, (p, _) in enumerate(infer_items) if str(p.resolve()) == train_path_s
        )
    for path, meta in infer_items:
        msis.append(_load_msi(path, transpose_msi=transpose_msi, normalization=norm_mode))
        slide_keys.append(str(meta["slide_key"]))

    train_msi = msis[train_slide_index] if not train_sweep else msis[0]
    param_groups = _load_parameter_groups(cfg)
    bench_cfg = _bench_cfg(cfg)
    sampling_mode = str(OmegaConf.select(ga, "sampling_mode", default="superpixel"))

    group_results: list[GroupResult] = []
    train_results: list[TrainSlideResult] = []
    manifest_rows: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []

    metrics_enabled = bool(OmegaConf.select(cfg, "metrics.enabled", default=True))
    rank_cfg = _load_rank_cfg(cfg) if metrics_enabled else None
    metric_weights = _metric_weights_from_cfg(cfg) if metrics_enabled else {}
    color_emd_cfg = _color_emd_cfg(cfg) if metrics_enabled else {"enabled": False, "bins": 32, "colorspace": "lab"}
    square_dice = bool(
        OmegaConf.select(cfg, "metrics.square_before_continuous_dice", default=False)
    )
    skip_hd_edges = bool(OmegaConf.select(cfg, "metrics.skip_hd_edges", default=False))
    eval_ctxs = (
        _build_eval_contexts(msis, infer_items, rank_cfg, skip_hd_edges=skip_hd_edges)
        if metrics_enabled and rank_cfg is not None
        else None
    )

    target_sampling_caches: list[dict[str, Any] | None] | None = None
    if param_groups and (include_self or include_self_aligned or include_finetune or include_finetune_color):
        group_name0, group_overrides0 = param_groups[0]
        target_sampling_caches = _prebuild_target_sampling_caches(
            cfg=cfg,
            bench_cfg=bench_cfg,
            group_overrides=group_overrides0,
            seed=seed,
            sampling_mode=sampling_mode,
            run_dir=run_dir,
            msis=msis,
            slide_keys=slide_keys,
        )

    cells_dir = run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    he_row: list[np.ndarray] | None = None
    if include_he:
        he_row, he_metas = _build_he_row(infer_items, cfg, bg=bg)
        for slide_i, (path, meta) in enumerate(infer_items):
            slide_key = str(meta["slide_key"])
            he_path = cells_dir / f"{slide_key}__he.png"
            Image.fromarray(he_row[slide_i]).save(he_path)
            manifest_rows.append(
                {
                    "mics_group": None,
                    "train_slide_key": None,
                    "slide_key": slide_key,
                    "npy_path": str(path),
                    "mode": "he",
                    "polarity": _slide_polarity(slide_key, path),
                    "is_train_slide": str(path.resolve()) == train_path_s if train_path_s else False,
                    "cell_path": str(he_path),
                    **he_metas[slide_i],
                }
            )

    if train_sweep:
        if len(param_groups) != 1:
            logger.warning(
                "by_train_slide uses one MiCS setting; got %d groups, using %s",
                len(param_groups),
                param_groups[0][0],
            )
        group_name, group_overrides = param_groups[0]
        self_row = None
        if include_self:
            self_row, _ = _train_and_predict_self_modes_row(
                group_name=group_name,
                group_overrides=group_overrides,
                target_msis=msis,
                slide_keys=slide_keys,
                bench_cfg=bench_cfg,
                cfg=cfg,
                seed=seed,
                sampling_mode=sampling_mode,
                run_dir=run_dir,
                target_sampling_caches=target_sampling_caches,
                include_self=True,
                include_self_aligned=False,
                train_color_anchor=None,
            )
        train_indices = list(range(len(infer_items)))
        max_train_raw = OmegaConf.select(ga, "max_train_slides")
        if max_train_raw is not None:
            train_indices = train_indices[: int(max_train_raw)]
        for sweep_n, train_i in enumerate(train_indices, start=1):
            train_path_i, train_meta_i = infer_items[train_i]
            train_key = str(train_meta_i["slide_key"])
            logger.info(
                "[train sweep %d/%d] fit on %s, predict all targets",
                sweep_n,
                len(train_indices),
                train_key,
            )
            xfer_row, ft_row, ft_color_row, novelty_row, rel_novelty_row, novelty_summaries, train_color_anchor = _train_and_predict_transfer_row(
                group_name=group_name,
                group_overrides=group_overrides,
                train_msi=msis[train_i],
                target_msis=msis,
                bench_cfg=bench_cfg,
                cfg=cfg,
                seed=seed,
                sampling_mode=sampling_mode,
                run_dir=run_dir,
                train_slide_index=train_i,
                cache_subdir=f"transfer/{train_key}",
                target_sampling_caches=target_sampling_caches,
                need_train_color_anchor=include_self_aligned or include_finetune_color,
                include_finetune_color=include_finetune_color,
            )
            self_aligned_row = None
            if include_self_aligned:
                _, self_aligned_row = _train_and_predict_self_modes_row(
                    group_name=group_name,
                    group_overrides=group_overrides,
                    target_msis=msis,
                    slide_keys=slide_keys,
                    bench_cfg=bench_cfg,
                    cfg=cfg,
                    seed=seed,
                    sampling_mode=sampling_mode,
                    run_dir=run_dir,
                    target_sampling_caches=target_sampling_caches,
                    include_self=False,
                    include_self_aligned=True,
                    train_color_anchor=train_color_anchor,
                    self_aligned_anchor_mode=self_aligned_anchor_mode,
                )
            train_results.append(
                (
                    train_key,
                    train_i,
                    group_name,
                    xfer_row,
                    ft_row,
                    ft_color_row,
                    self_row,
                    self_aligned_row,
                    novelty_row,
                    rel_novelty_row,
                )
            )
            train_ref_rgb = xfer_row[train_i]
            train_ref_mask = eval_ctxs[train_i]["valid_mask"] if eval_ctxs is not None else None
            row_manifest, row_metrics = _append_gallery_cell_records(
                group_name=group_name,
                train_slide_key=train_key,
                infer_items=infer_items,
                xfer_row=xfer_row,
                ft_row=ft_row,
                ft_color_row=ft_color_row,
                self_row=self_row,
                self_aligned_row=self_aligned_row,
                novelty_row=novelty_row,
                rel_novelty_row=rel_novelty_row,
                novelty_summaries=novelty_summaries,
                train_ref_rgb=train_ref_rgb,
                train_ref_mask=train_ref_mask,
                cells_dir=cells_dir,
                eval_ctxs=eval_ctxs,
                rank_cfg=rank_cfg,
                square_dice=square_dice,
                metric_weights=metric_weights,
                color_emd_cfg=color_emd_cfg,
                include_finetune=include_finetune,
                include_finetune_color=include_finetune_color,
                include_self=include_self,
                include_self_aligned=include_self_aligned,
                include_novelty=include_novelty,
                include_relative_novelty=include_relative_novelty,
                novelty_label=novelty_label,
                rel_novelty_label=rel_novelty_label,
                train_path_s=str(train_path_i.resolve()),
                cell_name_prefix=f"{group_name}__train_{train_key}__",
            )
            manifest_rows.extend(row_manifest)
            metric_records.extend(row_metrics)
    else:
        for group_name, group_overrides in param_groups:
            xfer_row, ft_row, ft_color_row, novelty_row, rel_novelty_row, novelty_summaries, train_color_anchor = _train_and_predict_transfer_row(
                group_name=group_name,
                group_overrides=group_overrides,
                train_msi=train_msi,
                target_msis=msis,
                bench_cfg=bench_cfg,
                cfg=cfg,
                seed=seed,
                sampling_mode=sampling_mode,
                run_dir=run_dir,
                train_slide_index=train_slide_index,
                target_sampling_caches=target_sampling_caches,
                need_train_color_anchor=include_self_aligned or include_finetune_color,
                include_finetune_color=include_finetune_color,
            )
            self_row = None
            self_aligned_row = None
            if include_self or include_self_aligned:
                self_row, self_aligned_row = _train_and_predict_self_modes_row(
                    group_name=group_name,
                    group_overrides=group_overrides,
                    target_msis=msis,
                    slide_keys=slide_keys,
                    bench_cfg=bench_cfg,
                    cfg=cfg,
                    seed=seed,
                    sampling_mode=sampling_mode,
                    run_dir=run_dir,
                    target_sampling_caches=target_sampling_caches,
                    include_self=include_self,
                    include_self_aligned=include_self_aligned,
                    train_color_anchor=train_color_anchor,
                    self_aligned_anchor_mode=self_aligned_anchor_mode,
                )
            group_results.append(
                (
                    group_name,
                    xfer_row,
                    ft_row,
                    ft_color_row,
                    self_row,
                    self_aligned_row,
                    novelty_row,
                    rel_novelty_row,
                )
            )
            train_ref_rgb = xfer_row[train_slide_index]
            train_ref_mask = (
                eval_ctxs[train_slide_index]["valid_mask"] if eval_ctxs is not None else None
            )
            row_manifest, row_metrics = _append_gallery_cell_records(
                group_name=group_name,
                train_slide_key=str(train_meta["slide_key"]),
                infer_items=infer_items,
                xfer_row=xfer_row,
                ft_row=ft_row,
                ft_color_row=ft_color_row,
                self_row=self_row,
                self_aligned_row=self_aligned_row,
                novelty_row=novelty_row,
                rel_novelty_row=rel_novelty_row,
                novelty_summaries=novelty_summaries,
                train_ref_rgb=train_ref_rgb,
                train_ref_mask=train_ref_mask,
                cells_dir=cells_dir,
                eval_ctxs=eval_ctxs,
                rank_cfg=rank_cfg,
                square_dice=square_dice,
                metric_weights=metric_weights,
                color_emd_cfg=color_emd_cfg,
                include_finetune=include_finetune,
                include_finetune_color=include_finetune_color,
                include_self=include_self,
                include_self_aligned=include_self_aligned,
                include_novelty=include_novelty,
                include_relative_novelty=include_relative_novelty,
                novelty_label=novelty_label,
                rel_novelty_label=rel_novelty_label,
                train_path_s=train_path_s,
            )
            manifest_rows.extend(row_manifest)
            metric_records.extend(row_metrics)

    grid_rgb, row_labels, col_labels, he_column_indices = _assemble_gallery_grid(
        group_results=group_results,
        train_results=train_results if train_sweep else None,
        infer_items=infer_items,
        train_path_s=train_path_s,
        layout=layout,
        include_he=include_he,
        he_label=he_label,
        include_finetune=include_finetune,
        include_finetune_color=include_finetune_color,
        include_self=include_self,
        include_self_aligned=include_self_aligned,
        include_novelty=include_novelty,
        include_relative_novelty=include_relative_novelty,
        xfer_label=xfer_label,
        ft_label=ft_label,
        ft_color_label=ft_color_label,
        self_label=self_label,
        self_aligned_label=self_aligned_label,
        novelty_label=novelty_label,
        rel_novelty_label=rel_novelty_label,
        he_row=he_row,
    )

    flat_viz = [v for row in grid_rgb for v in row]
    cell_max = OmegaConf.to_container(OmegaConf.select(ga, "cell_max_size"), resolve=True)
    if cell_max:
        flat_viz = [_maybe_downscale_rgb(v, list(cell_max)) for v in flat_viz]
    cell_size = _compute_cell_size(flat_viz, ga)

    padded_grid: list[list[np.ndarray]] = []
    padded_grid_raw: list[list[np.ndarray]] = []
    for row in grid_rgb:
        padded_row: list[np.ndarray] = []
        padded_row_raw: list[np.ndarray] = []
        for col_i, viz in enumerate(row):
            ds = _maybe_downscale_rgb(viz, list(cell_max) if cell_max else None)
            padded_row_raw.append(_letterbox_rgb(ds, cell_size, bg=bg))
            if col_i in he_column_indices:
                padded_row.append(_letterbox_rgb(ds, cell_size, bg=bg))
            else:
                eq = _maybe_equalize_rgb(ds, ga)
                padded_row.append(_letterbox_rgb(eq, cell_size, bg=bg))
        padded_grid.append(padded_row)
        padded_grid_raw.append(padded_row_raw)

    equalize = bool(OmegaConf.select(ga, "equalize_visualization", default=True))
    save_raw = bool(OmegaConf.select(ga, "save_raw_panel", default=True))
    mosaic_eq = _build_labeled_grid(
        padded_grid,
        row_labels,
        col_labels,
        cell_size=cell_size,
        row_label_width=int(ga.row_label_width),
        col_label_height=int(ga.col_label_height),
        gap=int(ga.gap_px),
        bg=bg,
    )

    out_name = str(ga.output_filename)
    if equalize:
        out_path = run_dir / out_name
        Image.fromarray(mosaic_eq).save(out_path)
        if save_raw:
            raw_name = str(OmegaConf.select(ga, "raw_output_filename", default="gallery_mics_transfer_raw.png"))
            raw_path = run_dir / raw_name
            mosaic_raw = _build_labeled_grid(
                padded_grid_raw,
                row_labels,
                col_labels,
                cell_size=cell_size,
                row_label_width=int(ga.row_label_width),
                col_label_height=int(ga.col_label_height),
                gap=int(ga.gap_px),
                bg=bg,
            )
            Image.fromarray(mosaic_raw).save(raw_path)
        else:
            raw_path = None
    else:
        mosaic_raw = mosaic_eq
        out_path = run_dir / out_name
        Image.fromarray(mosaic_raw).save(out_path)
        raw_path = None

    metric_figure_paths: dict[str, str] = {}
    metrics_csv_path: str | None = None
    manifest_train_key = "train_sweep" if train_sweep else str(train_meta["slide_key"])
    if metrics_enabled and metric_records:
        _attach_delta_records(metric_records)
        for rec in manifest_rows:
            rec_group = rec.get("mics_group")
            rec_mode = rec.get("mode")
            if rec_group is None or rec_mode in ("he",):
                continue
            match = next(
                (
                    m
                    for m in metric_records
                    if m.get("mics_group") == rec_group
                    and m.get("train_slide_key") == rec.get("train_slide_key")
                    and m.get("slide_key") == rec.get("slide_key")
                    and m.get("mode") == rec_mode
                ),
                None,
            )
            if match is None:
                continue
            for key in (
                "delta_self_minus_transfer",
                "delta_self_minus_transfer_finetune",
                "delta_transfer_finetune_minus_transfer",
                "color_emd_to_train_ref",
                "novelty_mean",
                "novelty_p95",
                "rel_novelty_mean",
                "rel_novelty_p95",
            ):
                if key in match:
                    rec[key] = match.get(key)
        csv_path = run_dir / "transfer_self_metrics.csv"
        _write_metrics_csv(metric_records, csv_path)
        metrics_csv_path = str(csv_path)
        if not train_sweep:
            metric_figure_paths = _save_metrics_figures(
                metric_records,
                run_dir,
                train_slide_key=manifest_train_key,
            )
        if color_emd_cfg.get("enabled"):
            metric_figure_paths.update(
                _save_color_emd_figures(metric_records, run_dir, infer_items=infer_items)
            )
        n_self = sum(1 for r in metric_records if r["mode"] == "self")
        n_wins = sum(
            1
            for r in metric_records
            if r["mode"] == "self"
            and np.isfinite(float(r.get("delta_self_minus_transfer", np.nan)))
            and float(r["delta_self_minus_transfer"]) > 0
        )
        logger.info("Metrics: self better on %d/%d slide×setting pairs", n_wins, n_self)

    manifest = {
        "train_slide_key": manifest_train_key,
        "train_npy_path": None if train_sweep else str(train_path),
        "train_polarity": None if train_sweep else _slide_polarity(train_meta["slide_key"], train_path),
        "train_index": None if train_sweep else train_idx,
        "train_sweep": train_sweep,
        "n_train_rows": len(train_results) if train_sweep else 1,
        "n_compatible_slides": len(compatible),
        "n_slide_columns": len(infer_items),
        "n_grid_columns": len(col_labels),
        "include_self_columns": include_self,
        "include_self_aligned_columns": include_self_aligned,
        "self_aligned_color_anchor": self_aligned_anchor_mode if include_self_aligned else None,
        "include_finetune_columns": include_finetune,
        "include_novelty_columns": include_novelty,
        "include_relative_novelty_columns": include_relative_novelty,
        "include_he_columns": include_he,
        "he_settings": OmegaConf.to_container(OmegaConf.select(cfg, "gallery.he"), resolve=True)
        if include_he
        else None,
        "novelty_settings": novelty_cfg if include_novelty else None,
        "infer_polarity": str(infer_pol_raw) if infer_pol_raw is not None else None,
        "finetune_settings": OmegaConf.to_container(
            OmegaConf.create(_finetune_settings(ga)), resolve=True
        )
        if include_finetune
        else None,
        "layout": layout,
        "n_mics_groups": len(param_groups),
        "n_rows": len(row_labels),
        "n_columns": len(col_labels),
        "spectral_filter": OmegaConf.to_container(ga.spectral_filter, resolve=True),
        "cell_size": list(cell_size),
        "equalize_visualization": equalize,
        "equalize_method": str(OmegaConf.select(ga, "equalize_method", default="lab_l")),
        "rows": manifest_rows,
        "mosaic_path": str(out_path),
        "raw_mosaic_path": str(raw_path) if raw_path else None,
        "metrics_csv": metrics_csv_path,
        "metrics_figures": metric_figure_paths,
        "color_emd": color_emd_cfg if metrics_enabled else None,
    }
    (run_dir / "gallery_mics_transfer_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    col_modes: list[str] = ["xfer"]
    if include_he:
        col_modes.insert(0, "H&E")
    if include_finetune:
        col_modes.append("xfer+ft")
    if include_finetune_color:
        col_modes.append("xfer+ft+align")
    if include_self:
        col_modes.append("self")
    if include_self_aligned:
        col_modes.append("self+align")
    if include_novelty:
        col_modes.append("d_train")
    if include_relative_novelty:
        col_modes.append("log(d_self/d_train)")
    if train_sweep:
        logger.info(
            "Wrote %s (%d rows × %d cols = %d train slides × %d targets × [%s])",
            out_path,
            len(row_labels),
            len(col_labels),
            len(train_results),
            len(infer_items),
            "+".join(col_modes),
        )
    elif layout.strip().lower() in ("by_slide", "slide", "image"):
        logger.info(
            "Wrote %s (%d rows × %d cols = %d slides × %d settings × [%s])",
            out_path,
            len(row_labels),
            len(col_labels),
            len(infer_items),
            len(param_groups),
            "+".join(col_modes),
        )
    else:
        logger.info(
            "Wrote %s (%d rows × %d cols = %d settings × %d slides × [%s])",
            out_path,
            len(row_labels),
            len(col_labels),
            len(param_groups),
            len(infer_items),
            "+".join(col_modes),
        )


if __name__ == "__main__":
    main()
