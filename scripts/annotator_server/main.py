#!/usr/bin/env python3
"""
FastAPI server for the interactive ROI annotator.
Loads config from YAML, serves initial visualization, and recomputes with ROI mask on request.

Run from project root:
  PYTHONPATH=. uvicorn scripts.annotator_server.main:app --reload
Or from scripts/:
  uvicorn annotator_server.main:app --reload

Config path: set CONFIG_PATH env to a YAML path, or defaults to scripts/configs/train_parametric_mics_lmc.yaml
Log file: set ANNOTATOR_LOG to a path, or logs go to annotator_server.log in cwd (and stderr).
CPU threads: set MAX_CPU_JOBS=N (e.g. 4) to limit numpy/sklearn/torch thread count; avoids maxing all cores.

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

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pydantic import BaseModel

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


class CategoryAnnotation(BaseModel):
    polygon: list[list[float]]  # [[x1,y1], [x2,y2], ...] in image pixel coords
    category: int  # 0, 1, 2, ... class id


class RecomputeBody(BaseModel):
    roi: list[list[float]]  # [[x1,y1], [x2,y2], ...] in image pixel coords
    category_annotations: Optional[list[dict[str, Any]]] = None  # [{"polygon": [[x,y],...], "category": 0}, ...]


class ModelOverridesBody(BaseModel):
    number_of_points: Optional[int] = None
    num_samples: Optional[int] = None
    num_epochs: Optional[int] = None
    clusters: Optional[str] = None  # e.g. "16-32-64-128"
    batch_size: Optional[int] = None
    beta: Optional[float] = None
    cluster_loss_weight: Optional[float] = None
    category_loss_weight: Optional[float] = None


class ConfigOverridesBody(BaseModel):
    model: Optional[ModelOverridesBody] = None


class ReloadVizBody(BaseModel):
    roi: Optional[list[list[float]]] = None  # if provided, recompute with ROI instead of full image
    category_annotations: Optional[list[dict[str, Any]]] = None  # [{"polygon": [[x,y],...], "category": 0}, ...]


class LoadImageBody(BaseModel):
    path: str  # path to .npy file to load (must be in config's input_files)


def _config_path() -> Path:
    p = Path(__file__).resolve().parent.parent / "configs" / "train_parametric_mics_lmc.yaml"
    return Path(os.environ.get("CONFIG_PATH", str(p)))


def _load_config() -> dict:
    path = _config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return OmegaConf.create(cfg)


def _build_roi_mask(height: int, width: int, roi: list[list[float]]) -> np.ndarray:
    """Build boolean (H, W) mask from polygon. roi: list of [x, y] (col, row)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(roi, dtype=np.int32)
    if pts.size < 6:  # need at least 3 points
        return mask
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _viz_to_png_base64(viz: np.ndarray) -> str:
    """Apply per-channel equalization and return PNG as base64."""
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
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_input_files(cfg: Any) -> list:
    """Resolve list of .npy input paths from config. Returns list of Path."""
    is_inference = (
        getattr(cfg, "inference", None)
        and getattr(cfg.inference, "model_path", None)
        and getattr(cfg.inference, "inference_folder", None)
    )
    if is_inference:
        _, _, find_files_with_glob = _load_train_helpers()
        pattern = cfg.inference.inference_folder
        return find_files_with_glob(pattern)
    input_path = Path(cfg.data.input_path)
    if input_path.is_file() and input_path.suffix == ".npy":
        return [input_path]
    if input_path.is_dir():
        return sorted(Path(p) for p in Path(input_path).glob("*.npy"))
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

    _log("info", "compute_initial_viz START")
    state = get_state()
    state.busy = True
    state.error = None
    try:
        # Release existing model first so we don't hold two in memory (e.g. on "Apply and reload")
        if state.model is not None:
            try:
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
        cfg = _load_config()
        state.config = cfg
        _log("info", "config loaded")
        load_and_preprocess_image, load_trained_model, find_files_with_glob = _load_train_helpers()

        input_files = _get_input_files(cfg)
        is_inference = (
            getattr(cfg, "inference", None)
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

        if state.mode == "inference":
            _log("info", "loading trained model")
            model = load_trained_model(Path(cfg.inference.model_path), cfg)
            state.model = model
            _log("info", "running predict (inference)")
            viz = model.predict(img)
        else:
            _log("info", "creating new model and fitting")
            model_kwargs = _model_kwargs_from_config(cfg, getattr(state, "config_overrides", None))
            model = MSIParametricMiCSLMC(**model_kwargs)
            model.fit(img)
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
        "number_of_points": int(cfg.model.number_of_points),
        "num_samples": int(cfg.model.num_samples),
        "num_epochs": int(cfg.model.num_epochs),
        "clusters": str(cfg.model.clusters),
        "batch_size": int(cfg.model.batch_size),
        "beta": float(cfg.model.beta),
        "cluster_loss_weight": float(getattr(cfg.model, "cluster_loss_weight", 1.0)),
        "category_loss_weight": float(getattr(cfg.model, "category_loss_weight", 1.0)),
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
    }
    if overrides and isinstance(overrides.get("model"), dict):
        m = overrides["model"]
        if m.get("number_of_points") is not None:
            kwargs["number_of_points"] = int(m["number_of_points"])
        if m.get("num_samples") is not None:
            kwargs["num_samples"] = int(m["num_samples"])
        if m.get("num_epochs") is not None:
            kwargs["num_epochs"] = int(m["num_epochs"])
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
    return kwargs


def _recompute_viz(roi: list[list[float]], category_annotations: Optional[list] = None) -> None:
    """Delete previous model and clustering; create a new model from scratch with ROI sampling (and optional category annotations), train, predict."""
    import gc
    import tqdm

    from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC

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

        # Create a new model from scratch (embedding + clustering)
        _log("info", "creating new MSIParametricMiCSLMC")
        model_kwargs = _model_kwargs_from_config(state.config, getattr(state, "config_overrides", None))
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


@app.get("/api/config")
async def api_config():
    """Return config for the UI (image shape, mode, editable model fields). Loads YAML if needed; does not run initial viz."""
    state = get_state()
    if state.config is None:
        state.config = _load_config()
    overrides = getattr(state, "config_overrides", {}) or {}
    shape = state.image_shape
    model_config = _effective_model_config(state.config, overrides)
    return JSONResponse({
        "image_shape": list(shape) if shape else None,
        "mode": state.mode,
        "model": model_config,
    })


@app.patch("/api/config")
async def api_patch_config(body: ConfigOverridesBody):
    """Apply UI config overrides (model fields). Merged with YAML on next compute/recompute."""
    state = get_state()
    if state.config is None:
        state.config = _load_config()
    overrides = getattr(state, "config_overrides", {}) or {}
    if "model" not in overrides:
        overrides["model"] = {}
    if body.model is not None:
        if body.model.number_of_points is not None:
            overrides["model"]["number_of_points"] = body.model.number_of_points
        if body.model.num_samples is not None:
            overrides["model"]["num_samples"] = body.model.num_samples
        if body.model.num_epochs is not None:
            overrides["model"]["num_epochs"] = body.model.num_epochs
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
    state.config_overrides = overrides
    model_config = _effective_model_config(state.config, state.config_overrides)
    return JSONResponse({
        "image_shape": list(state.image_shape) if state.image_shape else None,
        "mode": state.mode,
        "model": model_config,
    })


@app.get("/api/status")
async def api_status():
    """Return whether server is busy (computing)."""
    state = get_state()
    return JSONResponse({"busy": state.busy, "error": state.error})


def _get_sampled_and_reference_points():
    """Return sampled and reference point coords as [x,y] (col, row) in image pixel space."""
    state = get_state()
    if state.img is None or state.model is None:
        return None
    H, W = state.img.shape[0], state.img.shape[1]
    flat = state.model.sampled_flat_indices
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
        input_files = _get_input_files(state.config)
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


# Mount static files (frontend) last so /api/* routes take precedence
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
