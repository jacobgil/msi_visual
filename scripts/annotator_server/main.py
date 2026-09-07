#!/usr/bin/env python3
"""
FastAPI server for the interactive ROI annotator.
Loads config from YAML, serves initial visualization, and recomputes with ROI mask on request.

Run from project root:
  PYTHONPATH=. uvicorn scripts.annotator_server.main:app --reload
Or from scripts/:
  uvicorn annotator_server.main:app --reload

Config path: set CONFIG_PATH env to a YAML path, or defaults to
  scripts/annotator_server/configs/default.yaml
  (import/export list only that folder — not scripts/configs/).
Log file: set ANNOTATOR_LOG to a path, or logs go to annotator_server.log in cwd (and stderr).
CPU threads: set MAX_CPU_JOBS=N (e.g. 4) to limit numpy/sklearn/torch thread count; avoids maxing all cores.

Classical visualizations (model_type in YAML / Settings): top3, percentile_ratio, nmf_3d, kmeans_segmentation,
nmf_segmentation, nonparametric_umap — implemented via msi_visual (no MiCS training). ROI recompute runs the same full-image method.

After a sudden crash (freeze/OOM/kernel):
  - Inspect the last lines of the log file to see which step ran last and memory at that point.
  - Check system OOM/kernel: dmesg | tail -100  or  journalctl -b -1 -n 200  (previous boot).
"""
import os
import sys

# Limit CPU threads before importing numpy/sklearn (PCA, KMeans, pairwise_distances, BLAS)
_max_jobs = os.environ.get("MAX_CPU_JOBS")
if _max_jobs:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_var, _max_jobs)

"""
Fix for PyTorch DLL loading on Windows (c10.dll or dependencies not found):
https://github.com/pytorch/pytorch/issues/115812#issuecomment-2502093614
"""
import platform
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

import asyncio
import base64
import hashlib
import io
import itertools
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from omegaconf import OmegaConf
from pydantic import BaseModel
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression

NMF_ROI_FIT_MAX_SAMPLES = 5000


class _SimpleVizModelPlaceholder:
    """Marker for classical msi_visual methods (no sampled_flat_indices / LMC reference points)."""


SIMPLE_VIZ_PLACEHOLDER = _SimpleVizModelPlaceholder()

# Classical / non-parametric visualizations from msi_visual (no MiCS / parametric UMAP).
SIMPLE_VIZ_MODEL_TYPES = frozenset(
    {
        "top3",
        "percentile_ratio",
        "nmf_3d",
        "kmeans_segmentation",
        "nmf_segmentation",
        "nonparametric_umap",
    }
)


def _merged_model_value(cfg: Any, overrides: Optional[dict], key: str, default: Any) -> Any:
    if overrides and isinstance(overrides.get("model"), dict):
        v = overrides["model"].get(key)
        if v is not None:
            return v
    if hasattr(cfg.model, key):
        val = getattr(cfg.model, key)
        if val is not None:
            return val
    return default


def _ensure_uint8_rgb_viz(viz: Any) -> np.ndarray:
    if isinstance(viz, list):
        viz = viz[0]
    arr = np.asarray(viz)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        if mx <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255.0).astype(np.float32)
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8, copy=False)


def _compute_simple_visualization(
    img: np.ndarray,
    model_type: str,
    cfg: Any,
    overrides: Optional[dict],
) -> np.ndarray:
    """Run TOP3, NMF3D, k-means / NMF segmentation, percentile ratio, or nonparametric UMAP."""
    from msi_visual.kmeans_segmentation import KmeansSegmentation
    from msi_visual.nmf_3d import NMF3D
    from msi_visual.nmf_segmentation import NMFSegmentation
    from msi_visual.nonparametric_umap import MSINonParametricUMAP
    from msi_visual.percentile_ratio import TOP3, PercentileRatio

    m = lambda key, default: _merged_model_value(cfg, overrides, key, default)

    if model_type == "top3":
        viz = TOP3(
            low=float(m("top3_low", 99.9)),
            norm_percentile=float(m("top3_norm_percentile", 99.9)),
        )(img)
    elif model_type == "percentile_ratio":
        viz = PercentileRatio()(img)
    elif model_type == "nmf_3d":
        k = int(m("nmf_3d_n_components", m("number_of_components", 3)))
        max_iter = int(m("nmf_3d_max_iter", 200))
        viz = NMF3D(n_components=max(1, k), max_iter=max(1, max_iter))(img)
    elif model_type == "kmeans_segmentation":
        k = int(m("simple_k", 8))
        viz = KmeansSegmentation(max(2, k))(img)
    elif model_type == "nmf_segmentation":
        k = int(m("simple_k", 8))
        max_iter = int(m("nmf_segmentation_max_iter", 2000))
        viz = NMFSegmentation(max(2, k), max_iter=max(1, max_iter))(img)
    elif model_type == "nonparametric_umap":
        model = MSINonParametricUMAP(
            min_dist=float(m("umap_np_min_dist", 0.5)),
            n_neighbors=int(m("umap_np_n_neighbors", 100)),
            metric=str(m("umap_np_metric", "euclidean")),
        )
        viz = model(img)
    else:
        raise ValueError(f"Unknown simple viz model_type: {model_type}")

    return _ensure_uint8_rgb_viz(viz)


# Add scripts dir so we can import train_parametric_mics_lmc
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from .state import get_state

# --- Logging: file + stderr, with memory hints for crash debugging ---
def _memory_str():
    """Return a short string with process and GPU memory (if available) for logs."""
    parts = []
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        parts.append(f"RSS={rss_mb:.0f}MB")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            parts.append(f"GPU_alloc={alloc_mb:.0f}MB GPU_reserved={reserved_mb:.0f}MB")
    except Exception:
        pass
    return " ".join(parts) if parts else ""


def _setup_logging():
    log_path = os.environ.get("ANNOTATOR_LOG", "annotator_server.log")
    log_file = Path(log_path)
    if not log_file.is_absolute():
        log_file = Path.cwd() / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "[%(asctime)s] %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    return logging.getLogger("annotator")


logger = _setup_logging()


def _log(level, msg, with_memory=True):
    if with_memory:
        mem = _memory_str()
        msg = f"{msg} | {mem}" if mem else msg
    getattr(logger, level)(msg)

# Lazy imports for heavy deps
def _load_train_helpers():
    from train_parametric_mics_lmc import (
        find_files_with_glob,
        load_and_preprocess_image,
        load_trained_model,
    )
    return load_and_preprocess_image, load_trained_model, find_files_with_glob


class RecomputeBody(BaseModel):
    roi: list[list[float]]  # [[x1,y1], [x2,y2], ...] in image pixel coords
    category_annotations: Optional[list[dict[str, Any]]] = None  # [{"polygon": [[x,y],...], "category": 0}, ...]


class ModelOverridesBody(BaseModel):
    model_type: Optional[str] = None  # mics_lmc | parametric_umap | top3 | nmf_3d | ...
    number_of_points: Optional[int] = None
    num_samples: Optional[int] = None
    pixel_sampling: Optional[str] = None  # "superpixel" or "random"
    pca_fit_step: Optional[int] = None  # subsample step for PCA fit (1 = all pixels)
    cluster_on_pca: Optional[bool] = None  # if true, KMeans on PCA-reduced data
    cluster_pca_dims: Optional[int] = None  # PCA dims when cluster_on_pca is true
    num_epochs: Optional[int] = None
    number_of_components: Optional[int] = None
    num_layers: Optional[int] = None
    factor: Optional[float] = None
    lr: Optional[float] = None
    warmup_epochs: Optional[int] = None
    clusters: Optional[str] = None  # e.g. "16-32-64-128"
    batch_size: Optional[int] = None
    beta: Optional[float] = None
    cluster_loss_weight: Optional[float] = None
    category_loss_weight: Optional[float] = None
    umap_n_neighbors: Optional[int] = None
    umap_n_training_epochs: Optional[int] = None
    umap_n_epochs: Optional[int] = None
    # Classical msi_visual methods (see SIMPLE_VIZ_MODEL_TYPES)
    simple_k: Optional[int] = None
    nmf_3d_max_iter: Optional[int] = None
    nmf_3d_n_components: Optional[int] = None
    top3_low: Optional[float] = None
    top3_norm_percentile: Optional[float] = None
    nmf_segmentation_max_iter: Optional[int] = None
    umap_np_min_dist: Optional[float] = None
    umap_np_n_neighbors: Optional[int] = None
    umap_np_metric: Optional[str] = None


class DataOverridesBody(BaseModel):
    input_path: Optional[str] = None


class ConfigOverridesBody(BaseModel):
    model: Optional[ModelOverridesBody] = None
    data: Optional[DataOverridesBody] = None


class ReloadVizBody(BaseModel):
    roi: Optional[list[list[float]]] = None  # if provided, recompute with ROI instead of full image
    category_annotations: Optional[list[dict[str, Any]]] = None  # [{"polygon": [[x,y],...], "category": 0}, ...]
    input_path: Optional[str] = None  # optional .npy path to load before compute


class LoadImageBody(BaseModel):
    path: str  # path to .npy file to load (must be in config's input_files)


class ImportConfigBody(BaseModel):
    filename: str


class RegionNMFScoreBody(BaseModel):
    roi: list[list[float]]  # [[x,y], ...]
    n_components: Optional[int] = 32


class ComposePanelBody(BaseModel):
    ids: list[str]
    rows: int = 1
    cols: int = 3
    caption_keys: Optional[list[str]] = None  # empty / omitted = no YAML text under cells
    cell_max: int = 480
    save: bool = False
    output_path: Optional[str] = None  # optional override for where Save writes


class PanelListPathBody(BaseModel):
    path: str  # absolute folder (or file → parent) to list PNGs from


class SweepStartBody(BaseModel):
    """Hydra-style multi-value overrides for model.* keys (cartesian product)."""
    params: dict[str, str]  # key -> "a,b,c" multi-value string
    max_runs: int = 32
    panel_cols: int = 4
    build_panel: bool = True
    keep_last_as_current: bool = True
    input_path: Optional[str] = None  # client path; used to re-load after server reload


class SweepConfigSaveBody(BaseModel):
    name: Optional[str] = None  # stem; empty → timestamped
    model_type: Optional[str] = None
    max_runs: int = 32
    panel_cols: int = 4
    build_panel: bool = True
    fields: list[dict[str, Any]] = []


class SweepConfigNameBody(BaseModel):
    filename: str


class OutputPathsBody(BaseModel):
    app_output: Optional[str] = None  # empty / null = reset to default
    panel_output: Optional[str] = None




def _configs_dir() -> Path:
    """Annotator-owned configs only (not scripts/configs training YAMLs)."""
    return Path(__file__).resolve().parent / "configs"


def _sweep_configs_dir() -> Path:
    d = _configs_dir() / "sweeps"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_sweep_config_stem(name: Optional[str]) -> str:
    from datetime import datetime

    raw = (name or "").strip()
    if raw.lower().endswith(".yaml"):
        raw = raw[:-5]
    if raw.lower().endswith(".yml"):
        raw = raw[:-4]
    tok = _safe_name_token(raw, fallback="", max_len=60) if raw else ""
    if not tok:
        tok = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return tok


def _config_path() -> Path:
    p = _configs_dir() / "default.yaml"
    return Path(os.environ.get("CONFIG_PATH", str(p)))


def _repo_root() -> Path:
    # scripts/annotator_server/main.py → repo root
    return Path(__file__).resolve().parents[2]


def _scrub_hydra_for_plain_omegaconf(obj: Any, repo_root: Optional[Path] = None) -> Any:
    """Annotator loads YAML with OmegaConf only (no Hydra compose).

    Training configs may include Hydra ``defaults`` and ``${hydra:runtime.cwd}``
    interpolations that break ``OmegaConf.to_container(..., resolve=True)``.
    """
    root = str(repo_root or _repo_root()).replace("\\", "/")
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "defaults":
                continue  # Hydra-only; not composed here
            out[k] = _scrub_hydra_for_plain_omegaconf(v, Path(root))
        return out
    if isinstance(obj, list):
        return [_scrub_hydra_for_plain_omegaconf(v, Path(root)) for v in obj]
    if isinstance(obj, str) and "${hydra:runtime.cwd}" in obj:
        return obj.replace("${hydra:runtime.cwd}", root)
    return obj


def _load_config(path: Optional[Path] = None) -> dict:
    path = path or _config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    # UTF-8 required: default configs contain Unicode in comments (—, →, “ ”).
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _scrub_hydra_for_plain_omegaconf(cfg)
    return OmegaConf.create(cfg)


def _handle_input_path_change(state, prev_input_path: Path, new_input_path: Path) -> None:
    if str(prev_input_path) != str(new_input_path):
        state.last_roi = None
        state.last_category_annotations = None
        state.current_input_path = None
        state.img = None
        state.current_viz = None


def _merge_config_with_overrides(cfg: Any, overrides: dict) -> dict:
    base = OmegaConf.to_container(cfg, resolve=True)
    if overrides.get("model"):
        base.setdefault("model", {}).update(overrides["model"])
    if overrides.get("data"):
        base.setdefault("data", {}).update(overrides["data"])
    return base


def _normalize_model_type(value: Optional[str]) -> str:
    if not value:
        return "mics_lmc"
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if v in ("mics+lmc", "mics_lmc", "mics-lmc", "mics"):
        return "mics_lmc"
    if v in ("parametric_umap", "pumap"):
        return "parametric_umap"
    if v in ("top3",):
        return "top3"
    if v in ("percentile_ratio", "percentile", "pr"):
        return "percentile_ratio"
    if v in ("nmf_3d", "nmf3d"):
        return "nmf_3d"
    if v in ("kmeans_segmentation", "kmeans", "segmentation_kmeans", "kmeans_seg"):
        return "kmeans_segmentation"
    if v in ("nmf_segmentation", "segmentation_nmf", "nmf_seg"):
        return "nmf_segmentation"
    if v in ("nonparametric_umap", "umap_3d", "umap3d", "np_umap"):
        return "nonparametric_umap"
    return v


def _get_model_type(cfg: Any, overrides: Optional[dict]) -> str:
    override = None
    if overrides and isinstance(overrides.get("model"), dict):
        override = overrides["model"].get("model_type")
    cfg_value = getattr(cfg.model, "model_type", None)
    return _normalize_model_type(override or cfg_value)


def _build_roi_mask(height: int, width: int, roi: list[list[float]]) -> np.ndarray:
    """Build boolean (H, W) mask from polygon. roi: list of [x, y] (col, row)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(roi, dtype=np.int32)
    if pts.size < 6:  # need at least 3 points
        return mask
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _viz_to_png_bytes(viz: np.ndarray) -> bytes:
    """Apply per-channel equalization and return PNG bytes."""
    if viz is None or viz.size == 0:
        raise ValueError("Empty visualization")
    viz_u8 = np.clip(viz, 0, 255).astype(np.uint8)
    if viz_u8.ndim == 2:
        viz_u8 = cv2.equalizeHist(viz_u8)
    else:
        out = np.zeros_like(viz_u8)
        for c in range(viz_u8.shape[2]):
            out[:, :, c] = cv2.equalizeHist(viz_u8[:, :, c])
        viz_u8 = out
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(viz_u8).save(buf, format="PNG")
    return buf.getvalue()


def _viz_to_png_base64(viz: np.ndarray) -> str:
    """Apply per-channel equalization and return PNG as base64."""
    return base64.b64encode(_viz_to_png_bytes(viz)).decode("utf-8")


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_vals = values[order]
    n = len(values)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_std = a.std()
    b_std = b.std()
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_pearson(_rankdata_average(a), _rankdata_average(b))


def _compute_nmf_factors_for_valid_pixels(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    n_components: int,
    fit_mask: Optional[np.ndarray] = None,
    max_fit_samples: int = NMF_ROI_FIT_MAX_SAMPLES,
) -> tuple[np.ndarray, int]:
    x_valid = msi[valid_mask]
    if x_valid.size == 0:
        raise ValueError("No valid MSI pixels for NMF")
    k = int(n_components) if n_components is not None else 32
    k = max(1, min(k, 512))
    if x_valid.shape[0] < k:
        k = max(1, x_valid.shape[0])
    if fit_mask is None:
        fit_mask_valid = valid_mask
    else:
        fit_mask_valid = np.asarray(fit_mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    x_fit = msi[fit_mask_valid]
    if x_fit.size == 0:
        raise ValueError("No fit pixels for NMF (fit_mask ∩ valid_mask is empty)")
    n_fit = int(x_fit.shape[0])
    fit_count = min(n_fit, int(max_fit_samples))
    if fit_count < n_fit:
        rng = np.random.default_rng(42)
        fit_idx = rng.choice(n_fit, size=fit_count, replace=False)
        x_fit = x_fit[fit_idx]
    _log(
        "info",
        f"nmf_ranking fitting on ROI subset fit_count={int(x_fit.shape[0])}/{n_fit} "
        f"(max_fit_samples={int(max_fit_samples)}) components={k}",
    )

    nmf = NMF(
        n_components=k,
        init="nndsvda",
        max_iter=500,
        tol=1e-4,
        random_state=42,
    )
    nmf.fit(x_fit)
    # Project all valid pixels into learned NMF basis.
    w_nmf = nmf.transform(x_valid).astype(np.float32, copy=False)
    return w_nmf, int(k)


def _compute_region_nmf_lr_spearman(roi: list[list[float]], n_components: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    state = get_state()
    if state.img is None:
        raise ValueError("No image loaded")
    if state.current_viz is None:
        raise ValueError("No visualization available")
    _log(
        "info",
        f"region_nmf_ranking START roi_points={len(roi) if roi is not None else 0} "
        f"requested_components={n_components}",
    )

    msi = np.asarray(state.img, dtype=np.float32)
    viz = np.asarray(state.current_viz, dtype=np.float32)
    if viz.ndim == 2:
        viz = np.stack([viz, viz, viz], axis=-1)
    if viz.shape[-1] > 3:
        viz = viz[:, :, :3]
    if np.nanmax(viz) > 1.0:
        viz = viz / 255.0

    h, w, _ = msi.shape
    if not roi or len(roi) < 3:
        raise ValueError("roi must have at least 3 points")
    roi_mask = _build_roi_mask(h, w, roi)
    valid_mask = msi.sum(axis=-1) > 0
    region_mask = roi_mask & valid_mask
    n_region = int(region_mask.sum())
    _log(
        "info",
        f"region_nmf_ranking masks HxW={h}x{w} valid_pixels={int(valid_mask.sum())} "
        f"roi_pixels={int(roi_mask.sum())} region_valid_pixels={n_region}",
    )
    if n_region < 16:
        raise ValueError("Selected region is too small; need at least 16 valid pixels")

    k = int(n_components) if n_components is not None else 32
    t_nmf0 = time.perf_counter()
    w_nmf, k = _compute_nmf_factors_for_valid_pixels(
        msi,
        valid_mask,
        k,
        fit_mask=region_mask,
        max_fit_samples=NMF_ROI_FIT_MAX_SAMPLES,
    )
    _log(
        "info",
        f"region_nmf_ranking NMF fit/project done elapsed_s={time.perf_counter() - t_nmf0:.3f} "
        f"components={k} w_nmf_shape={tuple(w_nmf.shape)}",
    )

    rgb_valid = viz[valid_mask].astype(np.float32, copy=False)
    region_valid = region_mask[valid_mask]
    x_region = rgb_valid[region_valid]
    _log("info", f"region_nmf_ranking x_region_shape={tuple(x_region.shape)}")
    if x_region.shape[0] < 8:
        raise ValueError("Selected region is too small after masking")

    component_rows: list[dict[str, Any]] = []
    for comp_idx in range(k):
        y_valid = w_nmf[:, comp_idx]
        y_region = y_valid[region_valid]
        if y_region.size < 8:
            score = float("nan")
        else:
            lr = LinearRegression()
            lr.fit(x_region, y_region)
            y_pred = lr.predict(x_region)
            score = float(_safe_spearman(y_region, y_pred))

        comp_map = np.zeros((h, w), dtype=np.float32)
        comp_map[valid_mask] = y_valid
        comp_u8 = np.zeros((h, w), dtype=np.uint8)
        vals = comp_map[valid_mask]
        if vals.size > 0:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if vmax > vmin:
                comp_u8 = np.uint8(np.clip((comp_map - vmin) / (vmax - vmin) * 255.0, 0, 255))
        comp_u8[~valid_mask] = 0

        component_rows.append(
            {
                "component": int(comp_idx),
                "lr_spearman": score,
                "image_base64": _viz_to_png_base64(comp_u8),
            }
        )

    component_rows.sort(
        key=lambda r: (np.isfinite(float(r["lr_spearman"])), float(r["lr_spearman"])),
        reverse=True,
    )
    for rank_idx, row in enumerate(component_rows, start=1):
        row["rank"] = int(rank_idx)

    finite_scores = [float(r["lr_spearman"]) for r in component_rows if np.isfinite(float(r["lr_spearman"]))]
    if finite_scores:
        _log(
            "info",
            f"region_nmf_ranking scores top={max(finite_scores):.4f} "
            f"median={float(np.median(np.asarray(finite_scores, dtype=np.float64))):.4f} "
            f"bottom={min(finite_scores):.4f} finite={len(finite_scores)}/{len(component_rows)}",
        )
    _log("info", f"region_nmf_ranking DONE elapsed_s={time.perf_counter() - t0:.3f}")

    return {
        "n_components": int(k),
        "n_region_pixels": n_region,
        "components": component_rows,
    }


def _default_app_output_dir() -> Path:
    return (Path(__file__).resolve().parents[2] / "output-MiCS-app").resolve()


def _default_panel_output_dir() -> Path:
    return (_default_app_output_dir() / "panels").resolve()


def _paths_settings_file() -> Path:
    return Path(__file__).resolve().parent / "paths.json"


def _load_paths_settings() -> dict[str, str]:
    path = _paths_settings_file()
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_paths_settings(app_output: Optional[str], panel_output: Optional[str]) -> None:
    data = _load_paths_settings()
    if app_output is None or not str(app_output).strip():
        data.pop("app_output", None)
    else:
        data["app_output"] = str(Path(str(app_output).strip()).expanduser().resolve())
    if panel_output is None or not str(panel_output).strip():
        data.pop("panel_output", None)
    else:
        data["panel_output"] = str(Path(str(panel_output).strip()).expanduser().resolve())
    path = _paths_settings_file()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _apply_paths_settings_to_state() -> None:
    state = get_state()
    data = _load_paths_settings()
    state.app_output_dir = (str(data["app_output"]).strip() or None) if data.get("app_output") else None
    state.panel_output_dir = (str(data["panel_output"]).strip() or None) if data.get("panel_output") else None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _output_dir() -> Path:
    """App results root (saved viz PNG/YAML, sweeps)."""
    state = get_state()
    if getattr(state, "app_output_dir", None):
        return _ensure_dir(Path(state.app_output_dir).expanduser())
    data = _load_paths_settings()
    if data.get("app_output"):
        state.app_output_dir = str(data["app_output"])
        return _ensure_dir(Path(data["app_output"]).expanduser())
    return _ensure_dir(_default_app_output_dir())


def _panel_output_dir() -> Path:
    """Where Compose panel Save writes PNGs."""
    state = get_state()
    if getattr(state, "panel_output_dir", None):
        return _ensure_dir(Path(state.panel_output_dir).expanduser())
    data = _load_paths_settings()
    if data.get("panel_output"):
        state.panel_output_dir = str(data["panel_output"])
        return _ensure_dir(Path(data["panel_output"]).expanduser())
    # Default under current app output so changing app output also moves panels unless overridden
    return _ensure_dir(_output_dir() / "panels")


def _saved_images_dir() -> Path:
    return _output_dir()


def _output_paths_payload() -> dict:
    app_out = _output_dir()
    panel_out = _panel_output_dir()
    return {
        "app_output": str(app_out),
        "panel_output": str(panel_out),
        "defaults": {
            "app_output": str(_default_app_output_dir()),
            "panel_output": str(_default_panel_output_dir()),
        },
    }


def _safe_name_token(value: Any, *, fallback: str = "unknown", max_len: int = 80) -> str:
    s = str(value or "").strip().replace("\\", "/").replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("._-")
    return (s or fallback)[:max_len]


def _parse_sample_binning_region(input_path: Optional[str | Path]) -> dict[str, str]:
    """Extract sample / binning / region tokens from an MSI .npy path."""
    import re

    sample = "unknown_sample"
    binning = "unknown_bins"
    region = "r?"
    if not input_path:
        return {"sample": sample, "binning": binning, "region": region}
    p = Path(str(input_path))
    parts = [x for x in p.parts if x not in ("/", "\\")]
    # Drop drive letter on Windows ("E:")
    if parts and len(parts[0]) == 2 and parts[0][1] == ":":
        parts = parts[1:]
    stem = p.stem if p.suffix.lower() == ".npy" else p.name
    region = _safe_name_token(stem, fallback="r0", max_len=32)

    bin_re = re.compile(r"^(\d+)[_-]?bins?$", re.IGNORECASE)
    bin_idx = None
    for i, part in enumerate(parts):
        if bin_re.match(part):
            bin_idx = i
            binning = _safe_name_token(part, fallback="bins", max_len=32)
            break
    if bin_idx is not None and bin_idx > 0:
        sample = _safe_name_token(parts[bin_idx - 1], fallback=sample, max_len=80)
    elif p.suffix.lower() == ".npy" and len(parts) >= 2:
        sample = _safe_name_token(parts[-2], fallback=sample, max_len=80)
    elif parts:
        sample = _safe_name_token(parts[-1], fallback=sample, max_len=80)
    return {"sample": sample, "binning": binning, "region": region}


def _current_input_path_str() -> str:
    state = get_state()
    if state.current_input_path:
        return str(state.current_input_path)
    cfg = state.config
    overrides = getattr(state, "config_overrides", {}) or {}
    if cfg is not None:
        try:
            return str(_effective_input_path(cfg, overrides))
        except Exception:
            pass
    return ""


def _result_timestamp(with_seconds: bool = True) -> str:
    from datetime import datetime

    fmt = "%Y%m%d_%H%M%S" if with_seconds else "%Y%m%d_%H%M"
    return datetime.now().strftime(fmt)


def _result_stem(kind: str, *, input_path: Optional[str | Path] = None, with_seconds: bool = True) -> str:
    """Standard result stem: ``{sample}__{binning}__{region}__{kind}__{timestamp}``."""
    meta = _parse_sample_binning_region(input_path if input_path is not None else _current_input_path_str())
    kind_tok = _safe_name_token(kind, fallback="out", max_len=40)
    ts = _result_timestamp(with_seconds=with_seconds)
    return f"{meta['sample']}__{meta['binning']}__{meta['region']}__{kind_tok}__{ts}"


def _save_current_viz() -> Path:
    state = get_state()
    if state.current_viz is None:
        raise ValueError("No visualization available")
    cfg = state.config or _load_config()
    overrides = getattr(state, "config_overrides", {}) or {}
    if state.current_input_path:
        input_path = str(Path(state.current_input_path))
    else:
        input_path = str(_effective_input_path(cfg, overrides))
    model_type = _get_model_type(cfg, overrides)
    stem = _result_stem(f"viz_{model_type}", input_path=input_path, with_seconds=True)
    out_dir = _output_dir()
    out_path = out_dir / f"{stem}.png"
    out_path.write_bytes(_viz_to_png_bytes(state.current_viz))

    merged = _merge_config_with_overrides(cfg, overrides)
    merged.setdefault("data", {})["input_path"] = input_path
    yaml_path = out_dir / f"{stem}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    return out_path


def _flatten_yaml(obj: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_yaml(v, key))
        return out
    if isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = ", ".join(str(x) for x in obj)
        else:
            for i, v in enumerate(obj):
                out.update(_flatten_yaml(v, f"{prefix}[{i}]"))
        return out
    if obj is None:
        out[prefix] = "null"
    else:
        out[prefix] = str(obj)
    return out


def _shorten_caption_value(key: str, value: str) -> str:
    if key.endswith("input_path") or key.endswith("_path") or key.endswith("_folder"):
        p = Path(str(value).replace("\\", "/"))
        return p.name or str(value)
    if len(value) > 80:
        return value[:77] + "..."
    return value


def _caption_label(key: str) -> str:
    return key.split(".")[-1]


def _panel_roots() -> dict[str, Path]:
    root = _repo_root()
    return {
        "results": _output_dir().resolve(),
        "panel_out": _panel_output_dir().resolve(),
        "panels": (root / "outputs" / "viz_panel_run").resolve(),
        "hydra_panels": (root / "outputs" / "generate_visualization_panel").resolve(),
    }


def _b64url_encode_path(path: Path) -> str:
    raw = str(path.resolve()).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode_path(token: str) -> Path:
    pad = "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode((token or "") + pad).decode("utf-8")
    except Exception as e:
        raise ValueError("invalid custom path token") from e
    if not raw.strip():
        raise ValueError("empty custom path")
    return Path(raw).expanduser().resolve()


def _parse_source_rel(item_id: str) -> tuple[str, str]:
    if not item_id or "/" not in item_id:
        raise ValueError("invalid panel id")
    source, rel = item_id.split("/", 1)
    if source == "custom":
        rel = rel.replace("\\", "/").lstrip("/")
        if not rel or "/" in rel:
            raise ValueError("invalid custom panel path")
        return source, rel
    if source not in _panel_roots():
        raise ValueError(f"unknown source: {source}")
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel or ".." in Path(rel).parts:
        raise ValueError("invalid panel path")
    return source, rel


_PANEL_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _is_panel_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _PANEL_IMAGE_EXTS


def _parse_panel_id(item_id: str) -> tuple[str, Path]:
    source, rel = _parse_source_rel(item_id)
    if source == "custom":
        full = _b64url_decode_path(rel)
        if not _is_panel_image(full):
            raise ValueError("image not found")
        return source, full
    root = _panel_roots()[source]
    full = (root / rel).resolve()
    try:
        full.relative_to(root)
    except ValueError as e:
        raise ValueError("path escapes source root") from e
    if not _is_panel_image(full):
        raise ValueError("image not found")
    return source, full


def _resolve_folder(folder_id: str) -> tuple[str, Path]:
    source, _, rest = (folder_id or "").partition("/")
    if source == "custom":
        if not rest:
            raise ValueError("custom folder requires a path token")
        folder = _b64url_decode_path(rest.replace("\\", "/").lstrip("/"))
        if not folder.is_dir():
            raise ValueError("folder not found")
        return "custom", folder
    roots = _panel_roots()
    if source not in roots:
        raise ValueError(f"unknown folder: {folder_id}")
    root = roots[source]
    if not rest:
        return source, root
    rel = Path(rest.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("invalid folder path")
    full = (root / rel).resolve()
    try:
        full.relative_to(root)
    except ValueError as e:
        raise ValueError("path escapes source root") from e
    if not full.is_dir():
        raise ValueError("folder not found")
    return source, full


def _resolve_custom_dir(raw_path: str) -> Path:
    s = str(raw_path or "").strip().strip('"').strip("'")
    if not s:
        raise ValueError("empty path")
    folder = Path(s).expanduser()
    # Allow pasting an image / .yaml file → use its parent folder
    try:
        resolved = folder.resolve()
    except OSError as e:
        raise ValueError(f"invalid path: {e}") from e
    if resolved.is_file():
        resolved = resolved.parent
    if not resolved.is_dir():
        raise ValueError(f"folder not found: {resolved}")
    return resolved


def _iter_panel_images(folder: Path, *, recursive: bool, max_files: int = 2500) -> list[tuple[Path, float]]:
    entries: list[tuple[Path, float]] = []
    if recursive:
        candidates = []
        for ext in _PANEL_IMAGE_EXTS:
            candidates.extend(folder.rglob(f"*{ext}"))
            # Also match uppercase on case-sensitive FS
            candidates.extend(folder.rglob(f"*{ext.upper()}"))
        iterator = candidates
    else:
        iterator = (p for p in folder.iterdir() if _is_panel_image(p))
    seen: set[Path] = set()
    for p in iterator:
        try:
            key = p.resolve()
        except OSError:
            continue
        if key in seen or not _is_panel_image(p):
            continue
        seen.add(key)
        try:
            entries.append((p, p.stat().st_mtime))
        except OSError:
            continue
        if len(entries) >= max_files:
            break
    entries.sort(key=lambda t: t[1], reverse=True)
    return entries


def _list_png_items_in_dir(
    source: str,
    root: Path,
    folder: Path,
    folder_id: str,
    *,
    recursive: bool = False,
) -> dict:
    """List PNGs in one directory (optionally recursive for custom paths)."""
    items: list[dict] = []
    caption_keys: set[str] = set(_CAPTION_KEY_PREFERRED)
    caption_keys.update({"label", "file"})

    entries = _iter_panel_images(folder, recursive=recursive)
    # If a flat folder is empty but has subdirs, auto-recurse once for custom loads
    if not entries and not recursive:
        try:
            has_subdir = any(p.is_dir() and not p.name.startswith(".") for p in folder.iterdir())
        except OSError:
            has_subdir = False
        if has_subdir:
            entries = _iter_panel_images(folder, recursive=True)

    sample_n = 40
    for i, (png, mtime) in enumerate(entries):
        yml = _sidecar_yaml_path(png)
        has_yaml = yml is not None
        model_type = None
        if has_yaml and i < sample_n:
            data = _load_sidecar_yaml(png)
            caption_keys.update(_flatten_yaml(data).keys())
            model = data.get("model")
            if isinstance(model, dict) and model.get("model_type") is not None:
                model_type = str(model.get("model_type"))
        item = _panel_item_payload(
            source, root, png, mtime=mtime, has_yaml=has_yaml, model_type=model_type
        )
        item["folder"] = folder_id
        items.append(item)

    ordered_keys = [k for k in _CAPTION_KEY_PREFERRED if k in caption_keys]
    ordered_keys.extend(sorted(k for k in caption_keys if k not in ordered_keys))
    return {
        "folder": folder_id,
        "path": str(folder),
        "items": items,
        "caption_keys": ordered_keys,
        "recursive": bool(recursive or (len(entries) > 0 and any(
            (e[0].parent.resolve() != folder.resolve()) for e in entries[:20]
        ))),
    }


def _list_folder_items(folder_id: str) -> dict:
    """List PNGs in one folder. Avoid parsing every sidecar YAML (results can be 400+)."""
    source, folder = _resolve_folder(folder_id)
    root = folder if source == "custom" else _panel_roots()[source]
    return _list_png_items_in_dir(source, root, folder, folder_id, recursive=False)


def _list_custom_path_items(raw_path: str) -> dict:
    folder = _resolve_custom_dir(raw_path)
    folder_id = f"custom/{_b64url_encode_path(folder)}"
    # Custom paths: include nested PNGs so Visualization trees work
    return _list_png_items_in_dir("custom", folder, folder, folder_id, recursive=True)


def _count_panel_images_here(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if _is_panel_image(p))


def _list_panel_folders() -> list[dict]:
    roots = _panel_roots()
    out: list[dict] = []

    def add(folder_id: str, label: str, path: Path) -> None:
        n = _count_panel_images_here(path)
        if n <= 0 and folder_id not in ("results", "panels", "panel_out"):
            return
        out.append({
            "id": folder_id,
            "label": f"{label}  ({n})",
            "n": n,
        })

    add("results", f"app results ({roots['results']})", roots["results"])
    composed = roots["panel_out"]
    add("panel_out", f"composed panels ({composed})", composed)
    nested = roots["results"] / "panels"
    if nested.is_dir() and nested.resolve() != composed.resolve():
        add("results/panels", f"app results/panels ({nested})", nested)
    sweeps_root = roots["results"] / "sweeps"
    if sweeps_root.is_dir():
        sweep_dirs = []
        for p in sweeps_root.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                try:
                    sweep_dirs.append((p.stat().st_mtime, p))
                except OSError:
                    continue
        sweep_dirs.sort(key=lambda t: t[0], reverse=True)
        for _, p in sweep_dirs[:40]:
            add(f"results/sweeps/{p.name}", f"sweep/{p.name}", p)
    add("panels", "outputs/viz_panel_run", roots["panels"])
    add("panels/eq", "outputs/viz_panel_run/eq", roots["panels"] / "eq")

    hydra = roots["hydra_panels"]
    if hydra.is_dir():
        runs: list[tuple[float, Path]] = []
        for date_dir in hydra.iterdir():
            if not date_dir.is_dir() or date_dir.name.startswith("."):
                continue
            for run_dir in date_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name.startswith("."):
                    continue
                n = _count_panel_images_here(run_dir)
                if n <= 0:
                    continue
                try:
                    mtime = run_dir.stat().st_mtime
                except OSError:
                    mtime = 0.0
                runs.append((mtime, run_dir))
        runs.sort(key=lambda t: t[0], reverse=True)
        for _, run_dir in runs[:80]:
            rel = run_dir.relative_to(hydra).as_posix()
            add(f"hydra_panels/{rel}", f"generate_visualization_panel/{rel}", run_dir)
    return out


def _thumb_cache_path(png_path: Path) -> Path:
    digest = hashlib.sha1(str(png_path.resolve()).encode("utf-8")).hexdigest()
    return _repo_root() / "outputs" / ".annotator_thumbs" / f"{digest}.jpg"


def _jpeg_thumb_bytes(png_path: Path, max_side: int = 160, quality: int = 55) -> bytes:
    """Decode PNG to a small JPEG. Cache under outputs/.annotator_thumbs (not the results folder)."""
    from PIL import Image

    cache_path = _thumb_cache_path(png_path)
    try:
        if cache_path.is_file() and cache_path.stat().st_mtime >= png_path.stat().st_mtime:
            return cache_path.read_bytes()
    except OSError:
        pass
    with Image.open(png_path) as im:
        # Results viz PNGs are often RGBA; JPEG needs RGB.
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    except OSError:
        pass
    return data


def _sidecar_yaml_path(png_path: Path) -> Optional[Path]:
    yml = png_path.with_suffix(".yaml")
    if yml.is_file():
        return yml
    yml = png_path.with_suffix(".yml")
    return yml if yml.is_file() else None


def _load_sidecar_yaml(png_path: Path) -> dict:
    yml = _sidecar_yaml_path(png_path)
    if yml is None:
        return {}
    try:
        with open(yml, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _panel_item_payload(
    source: str,
    root: Path,
    png: Path,
    *,
    mtime: Optional[float] = None,
    has_yaml: bool = False,
    model_type: Optional[str] = None,
) -> dict:
    if source == "custom":
        rel = _b64url_encode_path(png)
    else:
        rel = png.relative_to(root).as_posix()
    tag = png.stem
    if "__" in tag:
        tag = tag.split("__", 1)[-1]
    if mtime is None:
        mtime = png.stat().st_mtime
    return {
        "id": f"{source}/{rel}",
        "source": source,
        "name": png.name,
        "rel": rel,
        "tag": tag,
        "has_yaml": bool(has_yaml),
        "model_type": model_type,
        "mtime": mtime,
    }


_CAPTION_KEY_PREFERRED = [
    "model.model_type",
    "model.num_epochs",
    "model.num_samples",
    "model.number_of_points",
    "model.clusters",
    "model.pixel_sampling",
    "model.num_layers",
    "model.number_of_components",
    "label",
    "data.input_path",
]


def _caption_font(size: int):
    from PIL import ImageFont

    candidates = [
        Path(r"C:\Windows\Fonts\consola.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _text_wh(draw, text: str, font) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return int(r - l), int(b - t)
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return int(r - l), int(b - t)
    return font.getsize(text)


def _fit_rgb(im, cell_w: int, cell_h: int) -> "Image.Image":
    from PIL import Image

    im = im.convert("RGB")
    w, h = im.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
    scale = min(cell_w / w, cell_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
    canvas.paste(resized, ((cell_w - nw) // 2, (cell_h - nh) // 2))
    return canvas


def _compose_panel_png(
    ids: list[str],
    rows: int,
    cols: int,
    caption_keys: Optional[list[str]],
    cell_max: int,
) -> bytes:
    from PIL import Image, ImageDraw

    rows = max(1, int(rows))
    cols = max(1, int(cols))
    cell_max = max(64, min(int(cell_max), 1600))
    n_slots = rows * cols
    use_ids = list(ids)[:n_slots]
    if not use_ids:
        raise ValueError("select at least one image")

    keys = [k.strip() for k in (caption_keys or []) if str(k).strip()]
    loaded: list[tuple[Any, list[str]]] = []
    max_w = 1
    max_h = 1
    for item_id in use_ids:
        _, path = _parse_panel_id(item_id)
        from PIL import Image as PILImage

        im = PILImage.open(path).convert("RGB")
        max_w = max(max_w, im.size[0])
        max_h = max(max_h, im.size[1])
        lines: list[str] = []
        if keys:
            values = _flatten_yaml(_load_sidecar_yaml(path))
            if not values:
                values = {"label": path.stem.split("__")[-1], "file": path.name}
            for key in keys:
                if key in values:
                    lines.append(f"{_caption_label(key)}: {_shorten_caption_value(key, values[key])}")
                else:
                    lines.append(f"{_caption_label(key)}: —")
        loaded.append((im, lines))

    cell_w = min(cell_max, max_w)
    cell_h = min(cell_max, max_h)
    gap = 10
    pad = 16
    font = _caption_font(13)
    probe = Image.new("RGB", (8, 8))
    probe_draw = ImageDraw.Draw(probe)
    line_h = max(14, _text_wh(probe_draw, "Ag", font)[1] + 3)
    caption_h = (len(keys) * line_h + 8) if keys else 0
    tile_h = cell_h + caption_h

    canvas_w = pad * 2 + cols * cell_w + (cols - 1) * gap
    canvas_h = pad * 2 + rows * tile_h + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (8, 8, 10))
    draw = ImageDraw.Draw(canvas)

    for i, (im, lines) in enumerate(loaded):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + gap)
        y = pad + r * (tile_h + gap)
        tile = _fit_rgb(im, cell_w, cell_h)
        canvas.paste(tile, (x, y))
        im.close()
        for j, line in enumerate(lines):
            draw.text((x + 2, y + cell_h + 4 + j * line_h), line, fill=(226, 226, 232), font=font)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _save_composed_panel(png_bytes: bytes) -> Path:
    out_dir = _panel_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _result_stem("panel", with_seconds=True)
    path = out_dir / f"{stem}.png"
    path.write_bytes(png_bytes)
    return path


def _effective_input_path(cfg: Any, overrides: Optional[dict] = None) -> Path:
    if overrides and isinstance(overrides.get("data"), dict):
        override_path = overrides["data"].get("input_path")
        if override_path:
            return Path(str(override_path))
    return Path(cfg.data.input_path)


def _get_input_files(cfg: Any, overrides: Optional[dict] = None) -> list:
    """Resolve list of .npy input paths from config/overrides. Returns list of Path."""
    is_inference = (
        getattr(cfg, "inference", None)
        and getattr(cfg.inference, "model_path", None)
        and getattr(cfg.inference, "inference_folder", None)
    )
    if is_inference:
        _, _, find_files_with_glob = _load_train_helpers()
        pattern = cfg.inference.inference_folder
        return find_files_with_glob(pattern)
    input_path = _effective_input_path(cfg, overrides)
    if input_path.is_file() and input_path.suffix == ".npy":
        return [input_path]
    if input_path.is_dir():
        return sorted(Path(p) for p in Path(input_path).rglob("*.npy"))
    raise ValueError(f"Invalid data.input_path: {input_path}")


def _compute_initial_viz(requested_path: Optional[str] = None) -> None:
    """Load config, load image (only if path changed), train or load model, compute viz. Runs in thread."""
    import gc

    _max_jobs = os.environ.get("MAX_CPU_JOBS")
    if _max_jobs:
        try:
            import torch
            torch.set_num_threads(int(_max_jobs))
        except Exception:
            pass

    from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC
    from msi_visual.parametric_umap import UMAPVirtualStain

    _log("info", "compute_initial_viz START")
    state = get_state()
    state.busy = True
    state.error = None
    try:
        # Release existing model first so we don't hold two in memory (e.g. on "Apply and reload")
        if state.model is not None:
            try:
                _log("info", f"releasing previous model type={type(state.model).__name__}")
                if hasattr(state.model, "release_resources"):
                    state.model.release_resources()
            except Exception:
                pass
            state.model = None
            state.current_viz = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        cfg_path = getattr(state, "config_override_path", None)
        cfg = _load_config(cfg_path) if cfg_path else _load_config()
        state.config = cfg
        _log("info", "config loaded")
        load_and_preprocess_image, load_trained_model, find_files_with_glob = _load_train_helpers()

        overrides = getattr(state, "config_overrides", None)
        input_files = _get_input_files(cfg, overrides)
        model_type = _get_model_type(cfg, overrides)
        _log("info", f"model_type resolved to '{model_type}'")
        is_inference = (
            model_type == "mics_lmc"
            and getattr(cfg, "inference", None)
            and getattr(cfg.inference, "model_path", None)
            and getattr(cfg.inference, "inference_folder", None)
        )
        state.mode = "inference" if is_inference else "train"
        if not input_files:
            raise ValueError("No .npy files found")

        resolved = [p.resolve() for p in input_files]
        current_resolved = Path(state.current_input_path).resolve() if state.current_input_path else None

        # Resolve which path to load: requested, else current (keep same image), else first
        if requested_path and Path(requested_path).resolve() in resolved:
            path_to_load = Path(requested_path).resolve()
        elif current_resolved and current_resolved in resolved:
            path_to_load = current_resolved
        else:
            path_to_load = input_files[0].resolve()

        # Reload .npy only if path changed (or no image yet)
        if current_resolved == path_to_load and state.img is not None:
            _log("info", f"reusing cached image (path unchanged): {path_to_load}")
            img = state.img
        else:
            img = load_and_preprocess_image(path_to_load, transpose=cfg.data.transpose)
            state.img = img
            state.current_input_path = str(path_to_load)
            H, W = img.shape[0], img.shape[1]
            _log("info", f"image loaded shape=({H},{W},{img.shape[2]}) from {path_to_load}")

        if model_type in SIMPLE_VIZ_MODEL_TYPES:
            _log("info", f"classical msi_visual method ({model_type})")
            viz = _compute_simple_visualization(img, model_type, cfg, overrides)
            state.model = SIMPLE_VIZ_PLACEHOLDER
            state.current_viz = viz
            _log("info", "compute_initial_viz DONE (classical)")
            return

        if state.mode == "inference":
            _log("info", "loading trained model")
            model = load_trained_model(Path(cfg.inference.model_path), cfg)
            state.model = model
            _log("info", "running predict (inference)")
            viz = model.predict(img)
        else:
            _log("info", f"creating new model and fitting ({model_type})")
            if model_type == "parametric_umap":
                t0 = time.perf_counter()
                _log("info", "building UMAP kwargs")
                umap_kwargs = _umap_kwargs_from_config(cfg, overrides)
                _log("info", f"UMAP kwargs built in {time.perf_counter() - t0:.2f}s")
                _log("info", f"UMAP kwargs: {umap_kwargs}")
                model = UMAPVirtualStain(**umap_kwargs)
                _log("info", "starting UMAP.fit (train)")
                t1 = time.perf_counter()
                model.fit([img], keras_fit_kwargs={}, roi_mask=None)
                _log("info", f"UMAP.fit completed in {time.perf_counter() - t1:.2f}s")
                state.model = model
                _log("info", "running UMAP.predict (train)")
                viz = model.predict(img)
                _log("info", f"UMAP.predict done (viz shape={getattr(viz, 'shape', None)}, dtype={getattr(viz, 'dtype', None)})")
                if isinstance(viz, np.ndarray) and viz.dtype != np.uint8 and np.nanmax(viz) <= 1.0:
                    viz = (viz * 255.0).clip(0, 255).astype(np.uint8)
                elif isinstance(viz, np.ndarray) and viz.dtype != np.uint8:
                    viz = np.clip(viz, 0, 255).astype(np.uint8)
                _log("info", f"UMAP viz normalized (shape={getattr(viz, 'shape', None)}, dtype={getattr(viz, 'dtype', None)})")
            else:
                t0 = time.perf_counter()
                _log("info", "building model kwargs")
                model_kwargs = _model_kwargs_from_config(cfg, overrides)
                _log("info", f"model kwargs built in {time.perf_counter() - t0:.2f}s")
                _log("info", f"MiCS+LMC kwargs: {model_kwargs}")

                _log("info", "initializing model")
                t1 = time.perf_counter()
                model = MSIParametricMiCSLMC(**model_kwargs)
                _log("info", f"model init done in {time.perf_counter() - t1:.2f}s")

                _log("info", "starting model.fit (train)")
                t2 = time.perf_counter()
                model.fit(img)
                _log("info", f"model.fit completed in {time.perf_counter() - t2:.2f}s")
                state.model = model
                _log("info", "running predict (train)")
                viz = model.predict(img)
        state.current_viz = viz
        _log("info", "compute_initial_viz DONE")
    except Exception as e:
        _log("error", f"compute_initial_viz FAILED: {e}", with_memory=True)
        state.error = str(e)
        raise
    finally:
        state.busy = False


def _effective_model_config(cfg: Any, overrides: dict) -> dict:
    """Merged model config for display (YAML + UI overrides)."""
    base = {
        "model_type": _normalize_model_type(getattr(cfg.model, "model_type", None)),
        "number_of_points": int(cfg.model.number_of_points),
        "num_samples": int(cfg.model.num_samples),
        "num_epochs": int(cfg.model.num_epochs),
        "number_of_components": int(cfg.model.number_of_components),
        "num_layers": int(getattr(cfg.model, "num_layers", 2)),
        "factor": float(cfg.model.factor),
        "lr": float(cfg.model.lr),
        "warmup_epochs": int(cfg.model.warmup_epochs),
        "clusters": str(cfg.model.clusters),
        "batch_size": int(cfg.model.batch_size),
        "beta": float(cfg.model.beta),
        "cluster_loss_weight": float(getattr(cfg.model, "cluster_loss_weight", 1.0)),
        "category_loss_weight": float(getattr(cfg.model, "category_loss_weight", 1.0)),
        "pixel_sampling": str(getattr(cfg.model, "pixel_sampling", "superpixel")),
        "pca_fit_step": int(getattr(cfg.model, "pca_fit_step", 4)),
        "cluster_on_pca": bool(getattr(cfg.model, "cluster_on_pca", False)),
        "cluster_pca_dims": int(getattr(cfg.model, "cluster_pca_dims", 50)),
        "umap_n_neighbors": int(getattr(cfg.model, "umap_n_neighbors", 50)),
        "umap_n_training_epochs": int(getattr(cfg.model, "umap_n_training_epochs", getattr(cfg.model, "num_epochs", 20))),
        "umap_n_epochs": (int(getattr(cfg.model, "umap_n_epochs")) if getattr(cfg.model, "umap_n_epochs", None) is not None else None),
        "simple_k": int(getattr(cfg.model, "simple_k", 8)),
        "nmf_3d_max_iter": int(getattr(cfg.model, "nmf_3d_max_iter", 200)),
        "nmf_3d_n_components": int(
            getattr(cfg.model, "nmf_3d_n_components", getattr(cfg.model, "number_of_components", 3))
        ),
        "top3_low": float(getattr(cfg.model, "top3_low", 99.9)),
        "top3_norm_percentile": float(getattr(cfg.model, "top3_norm_percentile", 99.9)),
        "nmf_segmentation_max_iter": int(getattr(cfg.model, "nmf_segmentation_max_iter", 2000)),
        "umap_np_min_dist": float(getattr(cfg.model, "umap_np_min_dist", 0.5)),
        "umap_np_n_neighbors": int(getattr(cfg.model, "umap_np_n_neighbors", 100)),
        "umap_np_metric": str(getattr(cfg.model, "umap_np_metric", "euclidean")),
    }
    if overrides and isinstance(overrides.get("model"), dict):
        for k in base:
            if k in overrides["model"] and overrides["model"][k] is not None:
                base[k] = overrides["model"][k]
    return base


def _model_kwargs_from_config(cfg, overrides: Optional[dict] = None):
    """Build model_kwargs from config; overrides (from UI) take precedence."""
    kwargs = {
        "number_of_points": cfg.model.number_of_points,
        "sampling": cfg.model.sampling,
        "num_epochs": cfg.model.num_epochs,
        "number_of_components": cfg.model.number_of_components,
        "num_layers": int(getattr(cfg.model, "num_layers", 2)),
        "lab_to_rgb": cfg.model.lab_to_rgb,
        "cluster": cfg.model.cluster,
        "clusters": [int(x) for x in str(cfg.model.clusters).split("-")] if cfg.model.cluster else None,
        "num_samples": cfg.model.num_samples,
        "beta": cfg.model.beta,
        "k_epoch": cfg.model.k_epoch,
        "temperature": cfg.model.temperature,
        "lr": cfg.model.lr,
        "batch_size": cfg.model.batch_size,
        "warmup_epochs": cfg.model.warmup_epochs,
        "factor": cfg.model.factor,
        "random_state": cfg.model.random_state,
        "verbose": getattr(cfg.model, "verbose", False),
        "cluster_loss_weight": float(getattr(cfg.model, "cluster_loss_weight", 1.0)),
        "category_loss_weight": float(getattr(cfg.model, "category_loss_weight", 1.0)),
        "pixel_sampling": str(getattr(cfg.model, "pixel_sampling", "superpixel")),
        "pca_fit_step": int(getattr(cfg.model, "pca_fit_step", 4)),
        "cluster_on_pca": bool(getattr(cfg.model, "cluster_on_pca", False)),
        "cluster_pca_dims": int(getattr(cfg.model, "cluster_pca_dims", 50)),
    }
    if overrides and isinstance(overrides.get("model"), dict):
        m = overrides["model"]
        if m.get("number_of_points") is not None:
            kwargs["number_of_points"] = int(m["number_of_points"])
        if m.get("num_samples") is not None:
            kwargs["num_samples"] = int(m["num_samples"])
        if m.get("num_epochs") is not None:
            kwargs["num_epochs"] = int(m["num_epochs"])
        if m.get("number_of_components") is not None:
            kwargs["number_of_components"] = int(m["number_of_components"])
        if m.get("num_layers") is not None:
            kwargs["num_layers"] = int(m["num_layers"])
        if m.get("factor") is not None:
            kwargs["factor"] = float(m["factor"])
        if m.get("lr") is not None:
            kwargs["lr"] = float(m["lr"])
        if m.get("warmup_epochs") is not None:
            kwargs["warmup_epochs"] = int(m["warmup_epochs"])
        if m.get("clusters") is not None:
            s = str(m["clusters"]).strip()
            kwargs["clusters"] = [int(x) for x in s.split("-") if x.strip()] if cfg.model.cluster and s else kwargs["clusters"]
        if m.get("batch_size") is not None:
            kwargs["batch_size"] = int(m["batch_size"])
        if m.get("beta") is not None:
            kwargs["beta"] = float(m["beta"])
        if m.get("cluster_loss_weight") is not None:
            kwargs["cluster_loss_weight"] = float(m["cluster_loss_weight"])
        if m.get("category_loss_weight") is not None:
            kwargs["category_loss_weight"] = float(m["category_loss_weight"])
        if m.get("pixel_sampling") is not None:
            kwargs["pixel_sampling"] = str(m["pixel_sampling"]).strip().lower() or "superpixel"
        if m.get("pca_fit_step") is not None:
            kwargs["pca_fit_step"] = max(1, int(m["pca_fit_step"]))
        if m.get("cluster_on_pca") is not None:
            kwargs["cluster_on_pca"] = bool(m["cluster_on_pca"])
        if m.get("cluster_pca_dims") is not None:
            kwargs["cluster_pca_dims"] = max(1, int(m["cluster_pca_dims"]))
    return kwargs


def _umap_kwargs_from_config(cfg, overrides: Optional[dict] = None) -> dict:
    model_overrides = overrides.get("model", {}) if overrides and isinstance(overrides.get("model"), dict) else {}
    n_components = int(model_overrides.get("number_of_components", cfg.model.number_of_components))
    n_neighbors = int(model_overrides.get("umap_n_neighbors", getattr(cfg.model, "umap_n_neighbors", 50)))
    n_training_epochs = int(model_overrides.get("umap_n_training_epochs", getattr(cfg.model, "umap_n_training_epochs", getattr(cfg.model, "num_epochs", 20))))
    n_epochs = model_overrides.get("umap_n_epochs", getattr(cfg.model, "umap_n_epochs", None))
    pixel_sampling = str(model_overrides.get("pixel_sampling", getattr(cfg.model, "pixel_sampling", "superpixel"))).strip().lower() or "superpixel"
    num_samples = int(model_overrides.get("num_samples", cfg.model.num_samples))
    pca_fit_step = int(model_overrides.get("pca_fit_step", getattr(cfg.model, "pca_fit_step", 4)))
    random_state = int(model_overrides.get("random_state", getattr(cfg.model, "random_state", 42)))
    return {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "n_training_epochs": n_training_epochs,
        "n_epochs": (int(n_epochs) if n_epochs is not None else None),
        "pixel_sampling": pixel_sampling,
        "num_samples": num_samples,
        "pca_fit_step": pca_fit_step,
        "random_state": random_state,
        "start_bin": 0,
        "end_bin": None,
    }


def _recompute_viz(roi: list[list[float]], category_annotations: Optional[list] = None) -> None:
    """Delete previous model and clustering; create a new model from scratch with ROI sampling (and optional category annotations), train, predict."""
    import gc
    import tqdm

    from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC
    from msi_visual.parametric_umap import UMAPVirtualStain

    _log("info", "recompute_viz START")
    state = get_state()
    if state.img is None or state.config is None:
        raise ValueError("Not initialized; call GET /api/viz or POST /api/init first")
    state.busy = True
    state.error = None
    try:
        # Completely delete previous model and clustering to avoid crashes and memory growth
        _log("info", "releasing previous model (GPU + large tensors)")
        old_model = state.model
        state.model = None
        state.current_viz = None
        if old_model is not None:
            try:
                _log("info", f"previous model type={type(old_model).__name__}")
                if hasattr(old_model, "release_resources"):
                    old_model.release_resources()
            except Exception as e:
                _log("warning", f"release_resources: {e}", with_memory=False)
            del old_model
        gc.collect()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _log("info", "after release + gc + cuda empty_cache")

        H, W = state.img.shape[0], state.img.shape[1]
        roi_mask = _build_roi_mask(H, W, roi)
        if not np.any(roi_mask):
            raise ValueError("ROI mask is empty")
        roi_pixels = int(np.sum(roi_mask))
        _log("info", f"ROI mask built H={H} W={W} roi_pixels={roi_pixels}")

        model_type = _get_model_type(state.config, getattr(state, "config_overrides", None))
        _log("info", f"model_type resolved to '{model_type}'")
        if model_type in SIMPLE_VIZ_MODEL_TYPES:
            _log(
                "info",
                "classical viz: Recompute uses full image (ROI is not used for these methods).",
            )
            overrides = getattr(state, "config_overrides", None)
            viz = _compute_simple_visualization(state.img, model_type, state.config, overrides)
            state.model = SIMPLE_VIZ_PLACEHOLDER
            state.current_viz = viz
            _log("info", "recompute_viz DONE (classical)")
            return
        if model_type == "parametric_umap":
            _log("info", "creating new Parametric UMAP model")
            umap_kwargs = _umap_kwargs_from_config(state.config, getattr(state, "config_overrides", None))
            _log("info", f"UMAP kwargs: {umap_kwargs}")
            model = UMAPVirtualStain(**umap_kwargs)
            _log("info", "starting UMAP.fit (train) on ROI-sampled pixels")
            model.fit([state.img], keras_fit_kwargs={}, roi_mask=roi_mask)
            _log("info", "running UMAP.predict")
            viz = model.predict(state.img)
            _log("info", f"UMAP.predict done (viz shape={getattr(viz, 'shape', None)}, dtype={getattr(viz, 'dtype', None)})")
            if isinstance(viz, np.ndarray) and viz.dtype != np.uint8 and np.nanmax(viz) <= 1.0:
                viz = (viz * 255.0).clip(0, 255).astype(np.uint8)
            elif isinstance(viz, np.ndarray) and viz.dtype != np.uint8:
                viz = np.clip(viz, 0, 255).astype(np.uint8)
            _log("info", f"UMAP viz normalized (shape={getattr(viz, 'shape', None)}, dtype={getattr(viz, 'dtype', None)})")
            state.model = model
            state.current_viz = viz
        else:
            # Create a new model from scratch (embedding + clustering)
            _log("info", "creating new MSIParametricMiCSLMC")
            model_kwargs = _model_kwargs_from_config(state.config, getattr(state, "config_overrides", None))
            _log("info", f"MiCS+LMC kwargs: {model_kwargs}")
            model = MSIParametricMiCSLMC(**model_kwargs)
            # Normalize category_annotations to list of dicts with "polygon" and "category"
            cat_ann = None
            if category_annotations and len(category_annotations) > 0:
                cat_ann = [
                    {"polygon": a.get("polygon", a.get("points", [])), "category": int(a.get("category", 0))}
                    for a in category_annotations
                    if a.get("polygon") or a.get("points")
                ]
                if cat_ann:
                    _log("info", f"category_annotations: {len(cat_ann)} polygons")
            _log("info", "calling set_image(roi_mask=..., category_annotations=...)")
            model.set_image(state.img, roi_mask=roi_mask, category_annotations=cat_ann)
            _log("info", f"training {model.num_epochs} epochs")
            for epoch in range(model.num_epochs):
                model.step()
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    _log("info", f"epoch {epoch + 1}/{model.num_epochs}")
            _log("info", "running predict")
            state.model = model
            state.current_viz = model.predict(state.img)
        _log("info", "recompute_viz DONE")
    except Exception as e:
        _log("error", f"recompute_viz FAILED: {e}", with_memory=True)
        state.error = str(e)
        raise
    finally:
        state.busy = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# --- FastAPI app ---
app = FastAPI(title="MSI-VISUAL API", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.middleware("http")
async def _no_cache_annotator_ui(request: Request, call_next):
    """Avoid stale index.html / app.js when the annotator UI is updated."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".ico")):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


@app.get("/api/config")
async def api_config():
    """Return config for the UI (image shape, mode, editable model fields). Loads YAML if needed; does not run initial viz."""
    state = get_state()
    if state.config is None:
        cfg_path = getattr(state, "config_override_path", None)
        state.config = _load_config(cfg_path) if cfg_path else _load_config()
    overrides = getattr(state, "config_overrides", {}) or {}
    shape = state.image_shape
    model_config = _effective_model_config(state.config, overrides)
    return JSONResponse({
        "image_shape": list(shape) if shape else None,
        "mode": state.mode,
        "data": {"input_path": str(_effective_input_path(state.config, overrides))},
        "model": model_config,
    })


@app.patch("/api/config")
async def api_patch_config(body: ConfigOverridesBody):
    """Apply UI config overrides (model/data). Merged with YAML on next compute/recompute."""
    state = get_state()
    if state.config is None:
        cfg_path = getattr(state, "config_override_path", None)
        state.config = _load_config(cfg_path) if cfg_path else _load_config()
    overrides = getattr(state, "config_overrides", {}) or {}
    prev_input_path = _effective_input_path(state.config, overrides)
    if "model" not in overrides:
        overrides["model"] = {}
    if "data" not in overrides:
        overrides["data"] = {}
    if body.data is not None:
        if body.data.input_path is not None:
            value = body.data.input_path.strip()
            if value:
                overrides["data"]["input_path"] = value
            else:
                overrides["data"].pop("input_path", None)
    if body.model is not None:
        if body.model.model_type is not None:
            overrides["model"]["model_type"] = _normalize_model_type(body.model.model_type)
        if body.model.number_of_points is not None:
            overrides["model"]["number_of_points"] = body.model.number_of_points
        if body.model.num_samples is not None:
            overrides["model"]["num_samples"] = body.model.num_samples
        if body.model.num_epochs is not None:
            overrides["model"]["num_epochs"] = body.model.num_epochs
        if body.model.number_of_components is not None:
            overrides["model"]["number_of_components"] = body.model.number_of_components
        if body.model.num_layers is not None:
            overrides["model"]["num_layers"] = max(1, body.model.num_layers)
        if body.model.factor is not None:
            overrides["model"]["factor"] = body.model.factor
        if body.model.lr is not None:
            overrides["model"]["lr"] = body.model.lr
        if body.model.warmup_epochs is not None:
            overrides["model"]["warmup_epochs"] = body.model.warmup_epochs
        if body.model.clusters is not None:
            overrides["model"]["clusters"] = body.model.clusters.strip()
        if body.model.batch_size is not None:
            overrides["model"]["batch_size"] = body.model.batch_size
        if body.model.beta is not None:
            overrides["model"]["beta"] = body.model.beta
        if body.model.cluster_loss_weight is not None:
            overrides["model"]["cluster_loss_weight"] = body.model.cluster_loss_weight
        if body.model.category_loss_weight is not None:
            overrides["model"]["category_loss_weight"] = body.model.category_loss_weight
        if body.model.umap_n_neighbors is not None:
            overrides["model"]["umap_n_neighbors"] = max(1, body.model.umap_n_neighbors)
        if body.model.umap_n_training_epochs is not None:
            overrides["model"]["umap_n_training_epochs"] = max(1, body.model.umap_n_training_epochs)
        if body.model.umap_n_epochs is not None:
            overrides["model"]["umap_n_epochs"] = max(1, body.model.umap_n_epochs)
        if body.model.pixel_sampling is not None:
            overrides["model"]["pixel_sampling"] = body.model.pixel_sampling.strip().lower() or "superpixel"
        if body.model.pca_fit_step is not None:
            overrides["model"]["pca_fit_step"] = max(1, body.model.pca_fit_step)
        if body.model.cluster_on_pca is not None:
            overrides["model"]["cluster_on_pca"] = body.model.cluster_on_pca
        if body.model.cluster_pca_dims is not None:
            overrides["model"]["cluster_pca_dims"] = max(1, body.model.cluster_pca_dims)
        if body.model.simple_k is not None:
            overrides["model"]["simple_k"] = max(2, body.model.simple_k)
        if body.model.nmf_3d_max_iter is not None:
            overrides["model"]["nmf_3d_max_iter"] = max(1, body.model.nmf_3d_max_iter)
        if body.model.nmf_3d_n_components is not None:
            overrides["model"]["nmf_3d_n_components"] = max(1, body.model.nmf_3d_n_components)
        if body.model.top3_low is not None:
            overrides["model"]["top3_low"] = float(body.model.top3_low)
        if body.model.top3_norm_percentile is not None:
            overrides["model"]["top3_norm_percentile"] = float(body.model.top3_norm_percentile)
        if body.model.nmf_segmentation_max_iter is not None:
            overrides["model"]["nmf_segmentation_max_iter"] = max(1, body.model.nmf_segmentation_max_iter)
        if body.model.umap_np_min_dist is not None:
            overrides["model"]["umap_np_min_dist"] = float(body.model.umap_np_min_dist)
        if body.model.umap_np_n_neighbors is not None:
            overrides["model"]["umap_np_n_neighbors"] = max(2, body.model.umap_np_n_neighbors)
        if body.model.umap_np_metric is not None:
            overrides["model"]["umap_np_metric"] = str(body.model.umap_np_metric).strip()
    state.config_overrides = overrides
    new_input_path = _effective_input_path(state.config, overrides)
    _handle_input_path_change(state, prev_input_path, new_input_path)
    model_config = _effective_model_config(state.config, state.config_overrides)
    return JSONResponse({
        "image_shape": list(state.image_shape) if state.image_shape else None,
        "mode": state.mode,
        "data": {"input_path": str(_effective_input_path(state.config, getattr(state, "config_overrides", None)))},
        "model": model_config,
    })


@app.get("/api/configs/list")
async def api_list_configs():
    """List yaml configs from scripts/annotator_server/configs."""
    cfg_dir = _configs_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p.name for p in cfg_dir.glob("*.yaml")])
    return JSONResponse({"files": files, "dir": str(cfg_dir)})


@app.get("/api/saved_images/list")
async def api_list_saved_images():
    """List yaml configs saved alongside images in output-MiCS-app."""
    out_dir = _saved_images_dir()
    files = sorted([p.name for p in out_dir.glob("*.yaml")])
    return JSONResponse({"files": files})


@app.post("/api/configs/import")
async def api_import_config(body: ImportConfigBody):
    """Import a yaml config from scripts/annotator_server/configs (no recompute)."""
    filename = body.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")
    if not filename.lower().endswith(".yaml"):
        raise HTTPException(status_code=400, detail="filename must be a .yaml file")
    cfg_dir = _configs_dir().resolve()
    candidate = (cfg_dir / filename).resolve()
    if candidate.parent != cfg_dir or not candidate.exists():
        raise HTTPException(status_code=404, detail="config not found")
    state = get_state()
    state.config_override_path = candidate
    state.config_overrides = {}
    try:
        state.config = _load_config(candidate)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    overrides = getattr(state, "config_overrides", {}) or {}
    model_config = _effective_model_config(state.config, overrides)
    return JSONResponse({
        "image_shape": list(state.image_shape) if state.image_shape else None,
        "mode": state.mode,
        "data": {"input_path": str(_effective_input_path(state.config, overrides))},
        "model": model_config,
    })


@app.post("/api/saved_images/import")
async def api_import_saved_image_config(body: ImportConfigBody):
    """Import a yaml config from output-MiCS-app (no recompute)."""
    filename = body.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")
    if not filename.lower().endswith(".yaml"):
        raise HTTPException(status_code=400, detail="filename must be a .yaml file")
    out_dir = _saved_images_dir().resolve()
    candidate = (out_dir / filename).resolve()
    if candidate.parent != out_dir or not candidate.exists():
        raise HTTPException(status_code=404, detail="config not found")
    state = get_state()
    state.config_override_path = candidate
    state.config_overrides = {}
    try:
        state.config = _load_config(candidate)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    overrides = getattr(state, "config_overrides", {}) or {}
    model_config = _effective_model_config(state.config, overrides)
    return JSONResponse({
        "image_shape": list(state.image_shape) if state.image_shape else None,
        "mode": state.mode,
        "data": {"input_path": str(_effective_input_path(state.config, overrides))},
        "model": model_config,
    })


@app.get("/api/config/export")
async def api_export_config():
    """Export merged config (YAML) to scripts/annotator_server/configs."""
    state = get_state()
    if state.config is None:
        cfg_path = getattr(state, "config_override_path", None)
        state.config = _load_config(cfg_path) if cfg_path else _load_config()
    overrides = getattr(state, "config_overrides", {}) or {}
    merged = _merge_config_with_overrides(state.config, overrides)
    merged.setdefault("data", {})["input_path"] = str(_effective_input_path(state.config, overrides))
    stem = _result_stem("config_export", with_seconds=True)
    filename = f"{stem}.yaml"
    configs_dir = _configs_dir()
    configs_dir.mkdir(parents=True, exist_ok=True)
    out_path = (configs_dir / filename).resolve()
    if out_path.parent != configs_dir.resolve():
        raise HTTPException(status_code=400, detail="invalid export path")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return JSONResponse({"saved_path": str(out_path), "filename": filename})




@app.get("/api/status")
async def api_status():
    """Return whether server is busy (computing)."""
    state = get_state()
    return JSONResponse({"busy": state.busy, "error": state.error})


@app.get("/api/output_paths")
async def api_get_output_paths():
    """Current app / panel output directories (with defaults)."""
    return JSONResponse(_output_paths_payload())


@app.post("/api/output_paths")
async def api_set_output_paths(body: OutputPathsBody):
    """Set app results and/or composed-panel output directories. Empty string resets to default."""
    state = get_state()
    # None = leave unchanged; "" = reset to default
    data = _load_paths_settings()
    cur_app = data.get("app_output")
    cur_panel = data.get("panel_output")

    if body.app_output is not None:
        cur_app = body.app_output.strip() or None
        if cur_app is not None:
            try:
                p = Path(cur_app).expanduser().resolve()
                p.mkdir(parents=True, exist_ok=True)
                cur_app = str(p)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"invalid app_output: {e}")
    if body.panel_output is not None:
        cur_panel = body.panel_output.strip() or None
        if cur_panel is not None:
            try:
                p = Path(cur_panel).expanduser().resolve()
                p.mkdir(parents=True, exist_ok=True)
                cur_panel = str(p)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"invalid panel_output: {e}")

    _save_paths_settings(cur_app, cur_panel)
    state.app_output_dir = cur_app
    state.panel_output_dir = cur_panel
    return JSONResponse(_output_paths_payload())


def _get_sampled_and_reference_points():
    """Return sampled and reference point coords as [x,y] (col, row) in image pixel space."""
    state = get_state()
    if state.img is None or state.model is None:
        return None
    H, W = state.img.shape[0], state.img.shape[1]
    flat = getattr(state.model, "sampled_flat_indices", None)
    if flat is None or len(flat) == 0:
        return {"sampled": [], "reference": []}
    rows, cols = np.unravel_index(flat, (H, W))
    sampled = [[int(c), int(r)] for r, c in zip(rows, cols)]
    indices = getattr(state.model, "indices", None)
    if indices is None or len(indices) == 0:
        return {"sampled": sampled, "reference": []}
    ref_flat = flat[indices]
    ref_rows, ref_cols = np.unravel_index(ref_flat, (H, W))
    reference = [[int(c), int(r)] for r, c in zip(ref_rows, ref_cols)]
    return {"sampled": sampled, "reference": reference}


@app.get("/api/input_files")
async def api_input_files():
    """Return list of .npy paths from config and the currently loaded path (for image selector)."""
    state = get_state()
    if state.config is None:
        state.config = _load_config()
    try:
        input_files = _get_input_files(state.config, getattr(state, "config_overrides", None))
        paths = [str(p.resolve()) for p in input_files]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({
        "paths": paths,
        "current": state.current_input_path,
    })


@app.get("/api/viz")
async def api_viz(input_path: Optional[str] = None):
    """Ensure initial computation is done; return current viz as base64 PNG. Optional input_path to load a specific .npy."""
    state = get_state()
    if state.config is None:
        state.config = _load_config()
    need_compute = (
        state.current_viz is None
        or (input_path is not None and (state.current_input_path is None or Path(state.current_input_path).resolve() != Path(input_path).resolve()))
    )
    if need_compute:
        await asyncio.to_thread(_compute_initial_viz, input_path)
    state = get_state()
    if state.error:
        raise HTTPException(status_code=500, detail=state.error)
    if state.current_viz is None:
        raise HTTPException(status_code=503, detail="No visualization available")
    b64 = _viz_to_png_base64(state.current_viz)
    return JSONResponse({"image_base64": b64, "shape": list(state.current_viz.shape)})


@app.post("/api/load_image")
async def api_load_image(body: LoadImageBody):
    """Load a specific .npy file and compute its viz. Returns new viz as base64. Reloads .npy only if path changed."""
    state = get_state()
    if state.config is None:
        state.config = _load_config()
    try:
        await asyncio.to_thread(_compute_initial_viz, body.path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    state = get_state()
    if state.error:
        raise HTTPException(status_code=500, detail=state.error)
    if state.current_viz is None:
        raise HTTPException(status_code=503, detail="No visualization available")
    b64 = _viz_to_png_base64(state.current_viz)
    return JSONResponse({"image_base64": b64, "shape": list(state.current_viz.shape)})


@app.post("/api/reload_viz")
async def api_reload_viz(body: ReloadVizBody = Body(default=ReloadVizBody())):
    """Re-run computation with current config (YAML + overrides). Uses roi from body, else last_roi from previous recompute, else full image. Returns new viz as base64."""
    state = get_state()
    if state.config is None:
        state.config = _load_config()
    if body.input_path:
        try:
            load_and_preprocess_image, _, _ = _load_train_helpers()
            img = load_and_preprocess_image(Path(body.input_path), transpose=state.config.data.transpose)
            state.img = img
            state.current_input_path = str(Path(body.input_path).resolve())
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    roi = None
    if body.roi and len(body.roi) >= 3:
        roi = body.roi
    elif getattr(state, "last_roi", None) and len(state.last_roi) >= 3:
        roi = state.last_roi
    if roi and len(roi) >= 3:
        _log("info", f"reload_viz using ROI ({len(roi)} points)")
        state.last_roi = roi
        cat_ann = body.category_annotations if (body.category_annotations and len(body.category_annotations) > 0) else getattr(state, "last_category_annotations", None)
        await asyncio.to_thread(_recompute_viz, roi, cat_ann)
    else:
        _log("info", "reload_viz using full image (no ROI)")
        state.last_roi = None
        state.last_category_annotations = None
        await asyncio.to_thread(_compute_initial_viz)
    state = get_state()
    if state.error:
        raise HTTPException(status_code=500, detail=state.error)
    if state.current_viz is None:
        raise HTTPException(status_code=503, detail="No visualization available")
    b64 = _viz_to_png_base64(state.current_viz)
    return JSONResponse({"image_base64": b64, "shape": list(state.current_viz.shape)})


@app.get("/api/sampled_points")
async def api_sampled_points():
    """Return centers of sampled points and LMC reference points (image pixel coords [x,y])."""
    state = get_state()
    if state.config is None or state.current_viz is None:
        raise HTTPException(status_code=503, detail="No visualization loaded yet")
    out = _get_sampled_and_reference_points()
    if out is None:
        raise HTTPException(status_code=503, detail="No sampled points available")
    return JSONResponse(out)


@app.post("/api/recompute")
async def api_recompute(body: RecomputeBody):
    """Recompute visualization restricted to the given ROI polygon."""
    if not body.roi or len(body.roi) < 3:
        raise HTTPException(status_code=400, detail="roi must have at least 3 points")
    state = get_state()
    if state.img is None:
        await asyncio.to_thread(_compute_initial_viz)
        state = get_state()
    if state.error:
        raise HTTPException(status_code=500, detail=state.error)
    try:
        state.last_roi = body.roi
        state.last_category_annotations = body.category_annotations
        await asyncio.to_thread(_recompute_viz, body.roi, body.category_annotations)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    state = get_state()
    if state.current_viz is None:
        raise HTTPException(status_code=500, detail="Recompute produced no viz")
    b64 = _viz_to_png_base64(state.current_viz)
    return JSONResponse({"image_base64": b64, "shape": list(state.current_viz.shape)})


@app.post("/api/region_nmf_ranking")
async def api_region_nmf_ranking(body: RegionNMFScoreBody):
    """Score NMF components in selected ROI by LR-Spearman and return ranked component maps."""
    _log(
        "info",
        f"/api/region_nmf_ranking request roi_points={len(body.roi) if body.roi else 0} "
        f"n_components={int(body.n_components or 32)}",
    )
    state = get_state()
    if state.img is None or state.current_viz is None:
        _log("info", "/api/region_nmf_ranking no image/viz loaded; computing initial viz")
        await asyncio.to_thread(_compute_initial_viz)
        state = get_state()
    if state.error:
        raise HTTPException(status_code=500, detail=state.error)
    try:
        result = await asyncio.to_thread(
            _compute_region_nmf_lr_spearman,
            body.roi,
            int(body.n_components or 32),
        )
        _log(
            "info",
            f"/api/region_nmf_ranking success n_components={result.get('n_components')} "
            f"n_region_pixels={result.get('n_region_pixels')}",
        )
    except Exception as e:
        _log("error", f"/api/region_nmf_ranking FAILED: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(result)


@app.post("/api/save_image")
async def api_save_image():
    """Save current visualization PNG to output-MiCS-app."""
    state = get_state()
    if state.current_viz is None:
        raise HTTPException(status_code=503, detail="No visualization available")
    try:
        saved_path = await asyncio.to_thread(_save_current_viz)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"saved_path": str(saved_path)})


@app.get("/api/panel/folders")
async def api_panel_folders():
    """List selectable image folders (results + generated-panel runs)."""
    try:
        folders = await asyncio.to_thread(_list_panel_folders)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"folders": folders})


@app.get("/api/panel/list")
async def api_panel_list(
    folder: str = Query("results"),
    path: Optional[str] = Query(None, description="Absolute folder path (any local dir with PNGs)"),
    custom_path: Optional[str] = Query(None, description="Alias for path"),
):
    """List PNGs in one selected folder, or in an arbitrary local path."""
    raw = (custom_path or path or "").strip()
    try:
        if raw:
            data = await asyncio.to_thread(_list_custom_path_items, raw)
        else:
            data = await asyncio.to_thread(_list_folder_items, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(data)


@app.post("/api/panel/list_path")
async def api_panel_list_path(body: PanelListPathBody):
    """List PNGs under an arbitrary local path (POST avoids Windows path query-encoding issues)."""
    try:
        data = await asyncio.to_thread(_list_custom_path_items, body.path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(data)


@app.get("/api/panel/thumb/{source}/{rel:path}")
async def api_panel_thumb(source: str, rel: str):
    """Serve a cached JPEG thumbnail for a PNG under a known root."""
    try:
        _, path = _parse_panel_id(f"{source}/{rel}")
        data = await asyncio.to_thread(_jpeg_thumb_bytes, path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=data, media_type="image/jpeg")


@app.post("/api/panel/compose")
async def api_panel_compose(body: ComposePanelBody):
    """Compose a rows×cols panel from chosen images; optional YAML captions under each cell."""
    if body.rows < 1 or body.cols < 1:
        raise HTTPException(status_code=400, detail="rows and cols must be >= 1")
    if not body.ids:
        raise HTTPException(status_code=400, detail="ids is required")
    try:
        png = await asyncio.to_thread(
            _compose_panel_png,
            body.ids,
            body.rows,
            body.cols,
            body.caption_keys,
            body.cell_max,
        )
        saved_path = None
        if body.save:
            if body.output_path and str(body.output_path).strip():
                # Persist panel output from compose modal so next saves match
                state = get_state()
                p = Path(str(body.output_path).strip()).expanduser().resolve()
                p.mkdir(parents=True, exist_ok=True)
                data = _load_paths_settings()
                _save_paths_settings(data.get("app_output"), str(p))
                state.panel_output_dir = str(p)
            saved_path = str(await asyncio.to_thread(_save_composed_panel, png))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({
        "image_base64": base64.b64encode(png).decode("utf-8"),
        "saved_path": saved_path,
        "n": min(len(body.ids), body.rows * body.cols),
        "truncated": max(0, len(body.ids) - body.rows * body.cols),
    })


# --- Hydra-style parameter sweep (in-app cartesian multirun) ---

def _resolve_npy_for_load(preferred: Optional[str] = None) -> Path:
    """Pick a .npy path from preferred / current / config (file or first in folder)."""
    state = get_state()
    if state.config is None:
        state.config = _load_config()
    overrides = getattr(state, "config_overrides", {}) or {}
    for raw in (preferred, state.current_input_path):
        if not raw:
            continue
        p = Path(str(raw))
        if p.is_file() and p.suffix.lower() == ".npy":
            return p.resolve()
    # Prefer an explicit file under config input list
    files = _get_input_files(state.config, overrides)
    if preferred:
        pref = Path(preferred).resolve()
        for f in files:
            if f.resolve() == pref:
                return f.resolve()
    if state.current_input_path:
        cur = Path(state.current_input_path).resolve()
        for f in files:
            if f.resolve() == cur:
                return f.resolve()
    eff = _effective_input_path(state.config, overrides)
    if eff.is_file() and eff.suffix.lower() == ".npy":
        return eff.resolve()
    if files:
        return files[0].resolve()
    raise ValueError("No .npy path available to load (set input path / Apply first)")


def _ensure_img_loaded(preferred: Optional[str] = None) -> Path:
    """Ensure ``state.img`` is loaded. Reloads after uvicorn --reload when the browser still has a path."""
    state = get_state()
    path = _resolve_npy_for_load(preferred)
    if (
        state.img is not None
        and state.current_input_path
        and Path(state.current_input_path).resolve() == path.resolve()
    ):
        return path
    if state.config is None:
        state.config = _load_config()
    load_and_preprocess_image, _, _ = _load_train_helpers()
    _log("info", f"ensure_img_loaded loading {path}")
    img = load_and_preprocess_image(path, transpose=state.config.data.transpose)
    state.img = img
    state.current_input_path = str(path.resolve())
    overrides = getattr(state, "config_overrides", None)
    if overrides is None:
        state.config_overrides = {}
        overrides = state.config_overrides
    overrides.setdefault("data", {})["input_path"] = str(path.resolve())
    return path


_SWEEP_FIELDS: dict[str, list[dict[str, Any]]] = {
    "mics_lmc": [
        {"key": "batch_size", "kind": "int", "hint": "1024,2048"},
        {"key": "beta", "kind": "float", "hint": "1:20:5 or 1.0,20.0"},
        {"key": "cluster_on_pca", "kind": "bool", "hint": "true,false"},
        {"key": "cluster_pca_dims", "kind": "int", "hint": "20:64:22 or 20,64"},
        {"key": "clusters", "kind": "str", "hint": "16-32-64,16-32-64-1024"},
        {"key": "factor", "kind": "float", "hint": "1:2:0.5 or 1,2"},
        {"key": "lr", "kind": "float", "hint": "0.1:1.0:0.3 or 0.5,1.0"},
        {"key": "num_epochs", "kind": "int", "hint": "20:50:10 or 20,30,50"},
        {"key": "num_layers", "kind": "int", "hint": "2:7:1 or 2,4,7"},
        {"key": "num_samples", "kind": "int", "hint": "2000:5000:1000 or 2000,5000"},
        {"key": "number_of_points", "kind": "int", "hint": "500:1500:500 or 500,1000"},
        {"key": "pca_fit_step", "kind": "int", "hint": "4:32:4 or 4,16,32"},
        {"key": "pixel_sampling", "kind": "str", "hint": "random,superpixel"},
    ],
    "parametric_umap": [
        {"key": "num_samples", "kind": "int", "hint": "2000:5000:1000 or 2000,5000"},
        {"key": "number_of_components", "kind": "int", "hint": "3"},
        {"key": "pca_fit_step", "kind": "int", "hint": "4:16:4 or 4,16"},
        {"key": "pixel_sampling", "kind": "str", "hint": "random,superpixel"},
        {"key": "umap_n_epochs", "kind": "int", "hint": "2:10:2 or 2,10"},
        {"key": "umap_n_neighbors", "kind": "int", "hint": "15:50:5 or 15,50"},
        {"key": "umap_n_training_epochs", "kind": "int", "hint": "1:3:1 or 1,2"},
    ],
    "top3": [
        {"key": "top3_low", "kind": "float", "hint": "99.0:99.9:0.3 or 99.0,99.9"},
        {"key": "top3_norm_percentile", "kind": "float", "hint": "99.0:99.9:0.3 or 99.0,99.9"},
    ],
    "percentile_ratio": [
        {"key": "top3_low", "kind": "float", "hint": "99.0:99.9:0.3 or 99.0,99.9"},
        {"key": "top3_norm_percentile", "kind": "float", "hint": "99.0:99.9:0.3 or 99.0,99.9"},
    ],
    "nmf_3d": [
        {"key": "nmf_3d_max_iter", "kind": "int", "hint": "100:300:50 or 100,200"},
        {"key": "nmf_3d_n_components", "kind": "int", "hint": "3:5:1 or 3,5"},
    ],
    "kmeans_segmentation": [
        {"key": "simple_k", "kind": "int", "hint": "4:16:4 or 4,8,16"},
    ],
    "nmf_segmentation": [
        {"key": "nmf_segmentation_max_iter", "kind": "int", "hint": "500:2000:500 or 500,2000"},
        {"key": "simple_k", "kind": "int", "hint": "4:16:4 or 4,8,16"},
    ],
    "nonparametric_umap": [
        {"key": "umap_np_metric", "kind": "str", "hint": "euclidean,cosine"},
        {"key": "umap_np_min_dist", "kind": "float", "hint": "0.1:0.5:0.1 or 0.1,0.5"},
        {"key": "umap_np_n_neighbors", "kind": "int", "hint": "50:150:25 or 50,100"},
    ],
}


def _parse_sweep_token(raw: str, kind: str) -> Any:
    s = str(raw).strip().strip("'\"")
    if kind == "int":
        return int(float(s))
    if kind == "float":
        return float(s)
    if kind == "bool":
        v = s.lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"invalid bool: {raw}")
    return s


def _parse_sweep_range(raw: str, kind: str) -> list[Any]:
    """Parse ``start:stop`` or ``start:stop:step`` (inclusive stop) for int/float."""
    parts = [p.strip() for p in str(raw).split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(f"range needs start:stop or start:stop:step, got {raw!r}")
    start = _parse_sweep_token(parts[0], kind)
    stop = _parse_sweep_token(parts[1], kind)
    if len(parts) == 3:
        step = _parse_sweep_token(parts[2], kind)
    else:
        step = 1 if kind == "int" else 1.0
    if float(step) == 0:
        raise ValueError("range step cannot be 0")

    values: list[Any] = []
    if kind == "int":
        start_i, stop_i, step_i = int(start), int(stop), int(step)
        if step_i == 0:
            raise ValueError("range step cannot be 0")
        if stop_i < start_i and step_i > 0:
            step_i = -step_i
        elif stop_i > start_i and step_i < 0:
            step_i = -step_i
        x = start_i
        if step_i > 0:
            while x <= stop_i:
                values.append(x)
                x += step_i
                if len(values) > 500:
                    raise ValueError("range too large (>500 values)")
        else:
            while x >= stop_i:
                values.append(x)
                x += step_i
                if len(values) > 500:
                    raise ValueError("range too large (>500 values)")
    else:
        start_f, stop_f, step_f = float(start), float(stop), float(step)
        if stop_f < start_f and step_f > 0:
            step_f = -step_f
        elif stop_f > start_f and step_f < 0:
            step_f = -step_f
        eps = abs(step_f) * 1e-9 + 1e-12
        x = start_f
        if step_f > 0:
            while x <= stop_f + eps:
                values.append(float(f"{x:.10g}"))
                x += step_f
                if len(values) > 500:
                    raise ValueError("range too large (>500 values)")
        else:
            while x >= stop_f - eps:
                values.append(float(f"{x:.10g}"))
                x += step_f
                if len(values) > 500:
                    raise ValueError("range too large (>500 values)")
    if not values:
        raise ValueError(f"empty range: {raw!r}")
    return values


def _parse_sweep_values(raw: str, kind: str) -> list[Any]:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("empty sweep values")
    # Numeric range: start:stop[:step]  (no commas — comma means explicit list)
    if kind in ("int", "float") and ":" in s and "," not in s:
        return _parse_sweep_range(s, kind)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty sweep values")
    return [_parse_sweep_token(p, kind) for p in parts]


def _sweep_tag(combo: dict[str, Any]) -> str:
    """Human-readable combo tag for logs / runs_index (not embedded in filenames)."""
    parts = []
    for k, v in combo.items():
        short = k.replace("number_of_", "n_").replace("umap_", "").replace("nmf_", "")
        val = str(v).replace(" ", "")
        if len(val) > 24:
            val = val[:24]
        parts.append(f"{short}={val}")
    tag = "__".join(parts) if parts else "base"
    return "".join(ch if ch.isalnum() or ch in "._-=" else "_" for ch in tag)[:120]


def _sweep_id_prefix(input_path: str) -> str:
    """``{sample}__{binning}__{region}`` for sweep file/folder stems."""
    meta = _parse_sample_binning_region(input_path)
    sample = _safe_name_token(meta.get("sample") or "sample", fallback="sample", max_len=80)
    binning = _safe_name_token(meta.get("binning") or "bins", fallback="bins", max_len=32)
    region = _safe_name_token(meta.get("region") or "0", fallback="0", max_len=32)
    return f"{sample}__{binning}__{region}"


def _sweep_run_stem(id_prefix: str, index: int, timestamp: str) -> str:
    """
    Per-run stem: sample + binning + region + run index + timestamp.
    Full parameter tags stay in runs_index.yaml / per-run YAML (avoids Windows MAX_PATH).
    """
    return f"{id_prefix}__run_{int(index):04d}__{timestamp}"


def _sweep_dir_stem(model_type: str, input_path: str, timestamp: str) -> str:
    """Sweep folder under sweeps/: sample + binning + region + model + timestamp."""
    id_prefix = _sweep_id_prefix(input_path)
    mt = _safe_name_token(model_type or "model", fallback="model", max_len=24)
    return f"{id_prefix}__sweep_{mt}__{timestamp}"


def _expand_sweep_grid(
    params: dict[str, str],
    model_type: str,
    max_runs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = {f["key"]: f for f in _SWEEP_FIELDS.get(model_type, [])}
    axes: list[tuple[str, list[Any]]] = []
    field_meta: list[dict[str, Any]] = []
    for key, raw in params.items():
        key = str(key).strip()
        if not key or not str(raw).strip():
            continue
        meta = fields.get(key)
        kind = meta["kind"] if meta else "str"
        values = _parse_sweep_values(raw, kind)
        axes.append((key, values))
        field_meta.append({"key": key, "kind": kind, "values": values})
    if not axes:
        raise ValueError("select at least one parameter with multi-values")
    keys = [k for k, _ in axes]
    value_lists = [vals for _, vals in axes]
    combos = [dict(zip(keys, prod)) for prod in itertools.product(*value_lists)]
    if len(combos) > max_runs:
        raise ValueError(f"sweep has {len(combos)} runs (limit {max_runs}); reduce values")
    return combos, field_meta


def _fit_predict_on_image(img: np.ndarray, cfg: Any, overrides: Optional[dict]) -> np.ndarray:
    """Train/predict one visualization for ``img`` without mutating annotator state.model."""
    import gc

    from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC
    from msi_visual.parametric_umap import UMAPVirtualStain

    model_type = _get_model_type(cfg, overrides)
    model = None
    try:
        if model_type in SIMPLE_VIZ_MODEL_TYPES:
            return _compute_simple_visualization(img, model_type, cfg, overrides)
        if model_type == "parametric_umap":
            model = UMAPVirtualStain(**_umap_kwargs_from_config(cfg, overrides))
            model.fit([img], keras_fit_kwargs={}, roi_mask=None)
            viz = model.predict(img)
            return _ensure_uint8_rgb_viz(viz)
        model = MSIParametricMiCSLMC(**_model_kwargs_from_config(cfg, overrides))
        model.fit(img)
        viz = model.predict(img)
        return _ensure_uint8_rgb_viz(viz)
    finally:
        if model is not None and hasattr(model, "release_resources"):
            try:
                model.release_resources()
            except Exception:
                pass
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _write_hydra_sweep_yaml(
    out_dir: Path,
    *,
    model_type: str,
    input_path: str,
    field_meta: list[dict[str, Any]],
    combos: list[dict[str, Any]],
    base_model: dict,
) -> None:
    """Write Hydra-compatible sweep docs for reproducibility / CLI multirun."""
    sweeper_params = {}
    for f in field_meta:
        sweeper_params[f"model.{f['key']}"] = ", ".join(str(v) for v in f["values"])
    hydra_cfg = {
        "defaults": ["_self_"],
        "hydra": {
            "mode": "MULTIRUN",
            "run": {"dir": str(out_dir.as_posix())},
            "sweep": {"dir": str(out_dir.as_posix()), "subdir": "${hydra.job.num}"},
            "sweeper": {"params": sweeper_params},
        },
        "mode": "train",
        "data": {"input_path": input_path, "transpose": False},
        "model": {**base_model, "model_type": model_type},
        "annotator_sweep": {
            "n_runs": len(combos),
            "combinations": combos,
        },
    }
    with open(out_dir / "sweep.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(hydra_cfg, f, sort_keys=False)
    with open(out_dir / "multirun.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"overrides": combos}, f, sort_keys=False)


def _run_parameter_sweep(
    params: dict[str, str],
    max_runs: int,
    panel_cols: int,
    build_panel: bool,
    keep_last_as_current: bool,
) -> None:
    state = get_state()
    if state.img is None or state.config is None:
        raise ValueError("Load an image first (Apply / Load viz)")
    if state.busy and (state.sweep or {}).get("status") == "running":
        raise ValueError("A sweep is already running")

    base_overrides = dict(getattr(state, "config_overrides", {}) or {})
    base_model = _effective_model_config(state.config, base_overrides)
    model_type = str(base_model.get("model_type") or "mics_lmc")
    combos, field_meta = _expand_sweep_grid(params, model_type, max(1, int(max_runs)))

    input_path = str(state.current_input_path or _effective_input_path(state.config, base_overrides))
    sweep_ts = _result_timestamp(with_seconds=True)
    id_prefix = _sweep_id_prefix(input_path)
    sweep_stem = _sweep_dir_stem(model_type, input_path, sweep_ts)
    out_dir = _output_dir() / "sweeps" / sweep_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_hydra_sweep_yaml(
        out_dir,
        model_type=model_type,
        input_path=input_path,
        field_meta=field_meta,
        combos=combos,
        base_model=base_model,
    )
    # Full param tags live here / in each YAML — filenames keep sample+binning+run+timestamp only
    runs_index: list[dict[str, Any]] = []
    with open(out_dir / "runs_index.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "input_path": input_path,
                "model_type": model_type,
                "id_prefix": id_prefix,
                "timestamp": sweep_ts,
                "n_runs": len(combos),
                "runs": [],
            },
            f,
            sort_keys=False,
        )

    state.sweep = {
        "status": "running",
        "cancel": False,
        "current": 0,
        "total": len(combos),
        "run_dir": str(out_dir),
        "model_type": model_type,
        "error": None,
        "results": [],
        "panel_path": None,
    }
    state.busy = True
    state.error = None
    last_viz = None
    panel_ids: list[str] = []

    try:
        _log("info", f"sweep START n={len(combos)} dir={out_dir}")
        for i, combo in enumerate(combos):
            if (state.sweep or {}).get("cancel"):
                state.sweep["status"] = "cancelled"
                _log("info", "sweep CANCELLED")
                break
            state.sweep["current"] = i + 1
            tag = _sweep_tag(combo)
            stem = _sweep_run_stem(id_prefix, i, sweep_ts)
            run_overrides = {
                "model": {**(base_overrides.get("model") or {}), **combo},
                "data": dict(base_overrides.get("data") or {}),
            }
            _log("info", f"sweep {i + 1}/{len(combos)} {stem} {combo}")
            viz = _fit_predict_on_image(state.img, state.config, run_overrides)
            last_viz = viz
            png_path = out_dir / f"{stem}.png"
            yaml_path = out_dir / f"{stem}.yaml"
            png_path.write_bytes(_viz_to_png_bytes(viz))
            merged = _merge_config_with_overrides(state.config, run_overrides)
            merged.setdefault("data", {})["input_path"] = input_path
            merged.setdefault("model", {})["model_type"] = model_type
            merged["annotator_sweep_run"] = {"index": i, "tag": tag, "params": combo, "timestamp": sweep_ts}
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(merged, f, sort_keys=False)
            # Register under results/sweeps/... for panel compose
            rel = png_path.relative_to(_panel_roots()["results"]).as_posix()
            panel_ids.append(f"results/{rel}")
            entry = {
                "index": i,
                "stem": stem,
                "tag": tag,
                "params": combo,
                "png": str(png_path),
                "yaml": str(yaml_path),
                "id": f"results/{rel}",
            }
            runs_index.append({
                "index": i,
                "stem": stem,
                "tag": tag,
                "params": {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in combo.items()},
                "png": f"{stem}.png",
                "yaml": f"{stem}.yaml",
            })
            state.sweep["results"].append(entry)
            # Rewrite index after each run so a crash still leaves a usable map
            with open(out_dir / "runs_index.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "input_path": input_path,
                        "model_type": model_type,
                        "id_prefix": id_prefix,
                        "timestamp": sweep_ts,
                        "n_runs": len(combos),
                        "completed": len(runs_index),
                        "runs": runs_index,
                    },
                    f,
                    sort_keys=False,
                )

        if build_panel and panel_ids and (state.sweep or {}).get("status") == "running":
            cols = max(1, int(panel_cols))
            rows = max(1, int(np.ceil(len(panel_ids) / cols)))
            caption = [f"model.{m['key']}" for m in field_meta][:6]
            try:
                panel_bytes = _compose_panel_png(panel_ids, rows, cols, caption, 360)
                panel_path = out_dir / f"{id_prefix}__sweep_panel__{sweep_ts}.png"
                panel_path.write_bytes(panel_bytes)
                state.sweep["panel_path"] = str(panel_path)
            except Exception as e:
                _log("error", f"sweep panel failed: {e}")

        if keep_last_as_current and last_viz is not None and (state.sweep or {}).get("status") != "cancelled":
            state.current_viz = last_viz
            if combos:
                state.config_overrides = {
                    "model": {**(base_overrides.get("model") or {}), **combos[-1]},
                    "data": dict(base_overrides.get("data") or {}),
                }

        if (state.sweep or {}).get("status") == "running":
            state.sweep["status"] = "done"
            _log("info", f"sweep DONE dir={out_dir}")
    except Exception as e:
        _log("error", f"sweep FAILED: {e}")
        if state.sweep is not None:
            state.sweep["status"] = "error"
            state.sweep["error"] = str(e)
        state.error = str(e)
        raise
    finally:
        state.busy = False


@app.get("/api/sweep/schema")
async def api_sweep_schema(input_path: Optional[str] = Query(None)):
    """Sweepable parameters for the current model_type (Hydra-style multi-values)."""
    state = get_state()
    if state.config is None:
        cfg_path = getattr(state, "config_override_path", None)
        state.config = _load_config(cfg_path) if cfg_path else _load_config()
    overrides = getattr(state, "config_overrides", {}) or {}
    model = _effective_model_config(state.config, overrides)
    model_type = str(model.get("model_type") or "mics_lmc")
    fields = []
    for f in _SWEEP_FIELDS.get(model_type, []):
        cur = model.get(f["key"])
        fields.append({
            "key": f["key"],
            "kind": f["kind"],
            "hint": f["hint"],
            "current": cur,
            "values": "" if cur is None else str(cur),
        })
    resolved = None
    resolve_error = None
    try:
        resolved = str(_resolve_npy_for_load(input_path))
    except Exception as e:
        resolve_error = str(e)
    return JSONResponse({
        "model_type": model_type,
        "input_path": resolved or str(state.current_input_path or _effective_input_path(state.config, overrides)),
        "has_image": state.img is not None,
        "can_load": resolved is not None,
        "resolve_error": resolve_error,
        "fields": fields,
        "max_runs_default": 32,
        "hydra_note": "Numeric fields: list (1,2,3) or range from:to:step (1:10:2, inclusive). Str/bool: comma list only.",
    })


@app.post("/api/sweep/start")
async def api_sweep_start(body: SweepStartBody):
    """Start an in-app Hydra-style parameter sweep on the loaded image."""
    state = get_state()
    if state.busy:
        raise HTTPException(status_code=409, detail="Server busy")
    try:
        await asyncio.to_thread(_ensure_img_loaded, body.input_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Load an image first: {e}")
    state = get_state()
    if state.img is None:
        raise HTTPException(status_code=400, detail="Load an image first")
    try:
        # Validate expansion before starting thread
        if state.config is None:
            state.config = _load_config()
        model_type = _get_model_type(state.config, getattr(state, "config_overrides", None))
        combos, _ = _expand_sweep_grid(body.params or {}, model_type, max(1, int(body.max_runs)))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Kick off in background thread (same pattern as other heavy work via to_thread)
    async def _runner():
        try:
            await asyncio.to_thread(
                _run_parameter_sweep,
                body.params or {},
                int(body.max_runs),
                int(body.panel_cols),
                bool(body.build_panel),
                bool(body.keep_last_as_current),
            )
        except Exception:
            pass

    asyncio.create_task(_runner())
    # Give the worker a moment to set state.sweep
    await asyncio.sleep(0.05)
    return JSONResponse({
        "started": True,
        "n_runs": len(combos),
        "sweep": state.sweep,
        "input_path": state.current_input_path,
    })


@app.get("/api/sweep/configs")
async def api_sweep_configs_list():
    """List saved sweep YAML configs under configs/sweeps/."""
    d = _sweep_configs_dir()
    files = sorted([p.name for p in d.glob("*.yaml")] + [p.name for p in d.glob("*.yml")])
    # unique preserve order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return JSONResponse({"files": uniq, "dir": str(d)})


@app.post("/api/sweep/configs/save")
async def api_sweep_configs_save(body: SweepConfigSaveBody):
    """Save current sweep UI settings as YAML."""
    stem = _safe_sweep_config_stem(body.name)
    d = _sweep_configs_dir().resolve()
    out_path = (d / f"{stem}.yaml").resolve()
    if out_path.parent != d:
        raise HTTPException(status_code=400, detail="invalid name")
    payload = {
        "name": stem,
        "model_type": body.model_type,
        "max_runs": int(body.max_runs),
        "panel_cols": int(body.panel_cols),
        "build_panel": bool(body.build_panel),
        "fields": body.fields or [],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return JSONResponse({"saved_path": str(out_path), "filename": out_path.name})


@app.get("/api/sweep/configs/load")
async def api_sweep_configs_load(filename: str = Query(...)):
    """Load a saved sweep config YAML."""
    name = (filename or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="filename is required")
    if not (name.lower().endswith(".yaml") or name.lower().endswith(".yml")):
        name = f"{name}.yaml"
    d = _sweep_configs_dir().resolve()
    path = (d / name).resolve()
    if path.parent != d or not path.is_file():
        raise HTTPException(status_code=404, detail="sweep config not found")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="invalid sweep config")
    return JSONResponse({"filename": path.name, "config": data})


@app.post("/api/sweep/configs/delete")
async def api_sweep_configs_delete(body: SweepConfigNameBody):
    """Delete a saved sweep config YAML."""
    name = (body.filename or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="filename is required")
    if not (name.lower().endswith(".yaml") or name.lower().endswith(".yml")):
        name = f"{name}.yaml"
    d = _sweep_configs_dir().resolve()
    path = (d / name).resolve()
    if path.parent != d or not path.is_file():
        raise HTTPException(status_code=404, detail="sweep config not found")
    path.unlink()
    return JSONResponse({"ok": True, "filename": name})


@app.get("/api/sweep/status")
async def api_sweep_status():
    state = get_state()
    sweep = getattr(state, "sweep", None) or {}
    status = sweep.get("status")
    results = sweep.get("results") or []
    out = {
        "busy": state.busy,
        "status": status,
        "current": sweep.get("current"),
        "total": sweep.get("total"),
        "run_dir": sweep.get("run_dir"),
        "panel_path": sweep.get("panel_path"),
        "model_type": sweep.get("model_type"),
        "error": sweep.get("error"),
        "n_results": len(results),
    }
    # Full result list only when finished (UI does not need it while running)
    if status in ("done", "error", "cancelled"):
        out["results"] = results
    if status == "done" and state.current_viz is not None:
        out["image_base64"] = _viz_to_png_base64(state.current_viz)
    return JSONResponse(out)


@app.post("/api/sweep/cancel")
async def api_sweep_cancel():
    state = get_state()
    if state.sweep is None:
        return JSONResponse({"ok": True, "status": None})
    state.sweep["cancel"] = True
    return JSONResponse({"ok": True, "status": state.sweep.get("status")})


# Mount static files (frontend) last so /api/* routes take precedence
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
