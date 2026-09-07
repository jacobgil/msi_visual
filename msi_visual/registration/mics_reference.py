"""Resolve MiCS RGB maps for use as registration references."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

MicsChannel = Literal["l", "edges", "chroma"]


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


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


def _composite_score(metrics: dict[str, float], tune_cfg_path: Path) -> float:
    from omegaconf import OmegaConf
    from scripts.bayes_tune_mics_lmc import (
        _parse_metric_bounds,
        _parse_objective_weights,
        _weighted_linear_objective,
    )

    tune_cfg = OmegaConf.load(str(tune_cfg_path))
    weights = _parse_objective_weights(tune_cfg)
    bounds = _parse_metric_bounds(tune_cfg)
    means = {
        "continuous_dice": float(metrics.get("continuous_dice", float("nan"))),
        "luminance_rms_contrast": float(metrics.get("luminance_rms_contrast", float("nan"))),
        "lab_chroma_entropy": float(metrics.get("lab_chroma_entropy", float("nan"))),
    }
    score, _ = _weighted_linear_objective(means, weights, bounds)
    return float(score)


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
            try:
                v = float(vals[img_idx])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                out[key] = v
    return out


def load_best_mics_rgb_from_tune_run(
    tune_run_dir: Path,
    img_idx: int,
) -> tuple[np.ndarray | None, str | None]:
    """Load top-ranked per-slide MiCS PNG from a bayes_tune_mics_lmc run."""
    csv_path = tune_run_dir / "bayes_mics_trials.csv"
    cfg_path = tune_run_dir / "config_resolved.yaml"
    if not csv_path.is_file():
        return None, None

    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    scored: list[tuple[float, Path]] = []
    for row in rows:
        metrics = _metrics_for_img(row, img_idx)
        if not metrics:
            continue
        mics_path = _pick_path_for_img(_parse_pipe_paths(row.get("image_paths", "")), img_idx)
        if mics_path is None or not mics_path.is_file():
            continue
        score = _composite_score(metrics, cfg_path) if cfg_path.is_file() else metrics.get("continuous_dice", 0.0)
        if np.isfinite(score):
            scored.append((score, mics_path))
    if not scored:
        return None, None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_path = scored[0][1]
    return _load_rgb(best_path), str(best_path)


def find_mics_rgb_near_npy(npy_path: Path, glob_pattern: str) -> tuple[np.ndarray | None, str | None]:
    """Search for a MiCS PNG under the MSI extraction folder (e.g. batch_extract viz)."""
    root = npy_path.parent.parent if npy_path.parent.name in ("5_bins", "5bins") else npy_path.parent
    hits = sorted(root.glob(glob_pattern))
    if not hits:
        hits = sorted(root.rglob(glob_pattern))
    for hit in hits:
        if hit.is_file() and hit.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return _load_rgb(hit), str(hit)
    return None, None


def resolve_mics_rgb_for_registration(
    *,
    npy_path: Path,
    img_idx: int,
    tune_run_dir: Path | None,
    explicit_path: Path | None,
    viz_glob: str | None,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray | None, str | None]:
    """Resolve MiCS RGB in MSI grid size for registration reference stages."""
    rgb: np.ndarray | None = None
    source: str | None = None

    if explicit_path is not None and explicit_path.is_file():
        rgb, source = _load_rgb(explicit_path), str(explicit_path)
    elif tune_run_dir is not None:
        rgb, source = load_best_mics_rgb_from_tune_run(tune_run_dir, img_idx)
    elif viz_glob:
        rgb, source = find_mics_rgb_near_npy(npy_path, viz_glob)

    if rgb is None:
        return None, None

    th, tw = target_hw
    if rgb.shape[:2] != (th, tw):
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    return rgb, source


def mics_reference_gray(
    mics_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    channel: MicsChannel = "l",
) -> np.ndarray:
    """Convert MiCS RGB to a uint8 grayscale reference for ECC vs H&E L."""
    rgb = np.asarray(mics_rgb, dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    if channel == "l":
        gray = l_ch.astype(np.float32)
    elif channel == "chroma":
        a = lab[:, :, 1].astype(np.float32) - 128.0
        b = lab[:, :, 2].astype(np.float32) - 128.0
        gray = np.sqrt(a * a + b * b)
    elif channel == "edges":
        gx = cv2.Sobel(l_ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l_ch, cv2.CV_32F, 0, 1, ksize=3)
        gray = cv2.magnitude(gx, gy)
    else:
        raise ValueError(f"Unknown mics channel: {channel!r}")

    gray = np.where(valid_mask, gray, 0.0)
    vals = gray[valid_mask]
    if vals.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    lo, hi = float(np.percentile(vals, 2.0)), float(np.percentile(vals, 98.0))
    span = max(hi - lo, 1e-6)
    out = np.clip((gray - lo) / span, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def mics_structure_mask(mics_rgb: np.ndarray, valid_mask: np.ndarray, *, channel: MicsChannel = "l") -> np.ndarray:
    gray = mics_reference_gray(mics_rgb, valid_mask, channel=channel)
    vals = gray[valid_mask]
    if vals.size == 0:
        return valid_mask
    thr = float(np.percentile(vals, 25.0))
    return valid_mask & (gray > thr)
