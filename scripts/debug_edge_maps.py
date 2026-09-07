#!/usr/bin/env python3
from __future__ import annotations

import logging
import time
import hashlib
import os
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import NMF, PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

try:
    import numba
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    njit = lambda **kw: lambda f: f
    prange = range

logger = logging.getLogger(__name__)


def _viz_output_tag(viz_path: Path, index: int) -> str:
    stem = viz_path.stem
    short = hashlib.sha1(str(viz_path).encode("utf-8")).hexdigest()[:8]
    return f"v{index:02d}__{stem}__{short}"


def _tic_normalize(msi: np.ndarray, *, chunk_rows: int = 32) -> np.ndarray:
    """TIC per pixel; processes row chunks to limit peak RAM on large HxWxM arrays."""
    src = np.asarray(msi)
    h, w, c = int(src.shape[0]), int(src.shape[1]), int(src.shape[2])
    if h <= 0 or w <= 0 or c <= 0:
        raise ValueError(f"Invalid MSI shape for TIC: {src.shape}")
    out = np.empty((h, w, c), dtype=np.float32)
    step = max(1, int(chunk_rows))
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        block = np.asarray(src[y0:y1], dtype=np.float32)
        sums = block.sum(axis=-1, keepdims=True)
        np.divide(block, sums + 1e-8, out=out[y0:y1])
    return out


def _spatial_tic_normalize(msi: np.ndarray, *, chunk_rows: int = 32) -> np.ndarray:
    """TIC per pixel, then channel-wise spatial scaling and clipping."""
    processed = _tic_normalize(msi, chunk_rows=chunk_rows)
    channel_max = np.percentile(processed, 100, axis=(0, 1), keepdims=True)
    np.divide(processed, channel_max + 1e-8, out=processed)
    return np.clip(processed, 0.0, 1.0).astype(np.float32, copy=False)


def _resolve_normalization_mode(value: object) -> str:
    """Normalize config values to one of: none, tic, spatial_tic."""
    if isinstance(value, bool):
        return "tic" if value else "none"
    s = str(value).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    if s in {"none", "off", "false", "0"}:
        return "none"
    if s in {"tic", "total_ion_count", "true", "1", "on"}:
        return "tic"
    if s in {"spatial_tic", "spatialtic", "sp_tic"}:
        return "spatial_tic"
    raise ValueError(
        f"Unknown normalization mode: {value!r}. Use one of: none, tic, spatial_tic."
    )


def _load_msi(path: Path, transpose_msi: bool, normalization: str = "tic") -> np.ndarray:
    img = np.load(path, mmap_mode="r")
    if img.ndim == 2:
        # Single map (e.g. one bin / projection saved as HxW)
        img = img[..., np.newaxis]
    elif img.ndim != 3:
        raise ValueError(f"Expected MSI shape HxWxC (or CxHxW with transpose), got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    norm_mode = _resolve_normalization_mode(normalization)
    if norm_mode == "spatial_tic":
        img = _spatial_tic_normalize(img)
    elif norm_mode == "tic":
        img = _tic_normalize(img)
    elif norm_mode == "none":
        img = np.asarray(img, dtype=np.float32)
    else:
        raise ValueError(
            f"Unknown normalization mode after parsing: {normalization!r}. "
            "Use one of: none, tic, spatial_tic."
        )
    return img


def _load_valid_mask(path: Path, *, transpose_msi: bool) -> np.ndarray:
    """Tissue mask from disk without materializing the full normalized HxWxM cube."""
    img = np.load(path, mmap_mode="r")
    if img.ndim == 2:
        img = img[..., np.newaxis]
    elif img.ndim != 3:
        raise ValueError(f"Expected MSI shape HxWxC (or CxHxW with transpose), got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    return np.asarray(img.sum(axis=-1) > 0, dtype=bool)


def _to_uint8_rgb(viz: np.ndarray) -> np.ndarray:
    """Convert visualization to uint8 RGB [0,255]. Handles: uint8, float [0,1], float [0,M] for M>1."""
    arr = np.asarray(viz)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr
    out = arr.astype(np.float32)
    mx = float(np.nanmax(out))
    mn = float(np.nanmin(out))
    if mx <= 1.0:
        # Float [0,1]: scale to 0-255
        out = out * 255.0
    elif mx > mn + 1e-8:
        # Float with max>1 (e.g. 0-100): stretch to full 0-255 so display and edges match
        out = (out - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_visualization(path: Path, target_hw: Tuple[int, int], allow_resize: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        viz = np.load(path, mmap_mode=None)
        viz_rgb = _to_uint8_rgb(viz)
    else:
        viz_rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = target_hw
    if viz_rgb.shape[:2] != (h, w):
        if not allow_resize:
            raise ValueError(
                f"Visualization shape mismatch for {path}: got {viz_rgb.shape[:2]}, expected {(h, w)}"
            )
        viz_rgb = cv2.resize(viz_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return viz_rgb


def _spatial_hw_npy(path: Path) -> Tuple[int, int]:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim < 2:
        raise ValueError(f"Expected ndim>=2 for {path}, got shape {arr.shape}")
    return int(arr.shape[0]), int(arr.shape[1])


def _spatial_hw_msi_cube_file(path: Path, transpose_msi: bool) -> Tuple[int, int]:
    """
    Spatial (H, W) for MSI .npy on disk, consistent with _load_msi(..., transpose_msi=...).

    - transpose_msi false: array is H x W x C -> (shape[0], shape[1]).
    - transpose_msi true: array on disk is C x H x W -> spatial (shape[1], shape[2]).
    """
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim != 3:
        raise ValueError(f"Expected MSI shape HxWxC or CxHxW (with transpose), got {arr.shape}")
    if transpose_msi:
        return int(arr.shape[1]), int(arr.shape[2])
    return int(arr.shape[0]), int(arr.shape[1])


def _spatial_hw_png(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return int(h), int(w)


def _viz_path_spatial_hw(path: Path) -> Tuple[int, int]:
    """(H, W) for a .npy array or image file."""
    if path.suffix.lower() == ".npy":
        return _spatial_hw_npy(path)
    return _spatial_hw_png(path)


def _pairing_unset_for_inference(pairing: str) -> bool:
    """YAML null / unset often becomes '', 'none', or 'null' after str()."""
    return pairing in ("", "auto", "none", "null", "~")


def _pick_msi_npy_for_shape_pairing(npy_dir: Path, png_dir: Path, *, transpose_msi: bool = False) -> Path:
    """
    Choose which .npy in npy_dir to load as MSI when input.npy_path is unset:
    among spatial sizes that appear in both dirs (npy + png), pick the largest H*W,
    then prefer 0.npy / 1.npy in that bucket, else first by name.
    """
    from collections import defaultdict

    buckets: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    for p in sorted(npy_dir.glob("*.npy"), key=lambda x: x.name.lower()):
        try:
            hw = _spatial_hw_msi_cube_file(p, transpose_msi)
        except Exception as e:
            logger.warning("skip %s when picking MSI | %s", p, e)
            continue
        buckets[hw].append(p)
    png_sizes: set[Tuple[int, int]] = set()
    for p in png_dir.glob("*.png"):
        try:
            png_sizes.add(_spatial_hw_png(p))
        except Exception:
            continue
    shared = set(buckets.keys()) & png_sizes
    if not shared:
        raise ValueError(
            f"No common spatial (H,W) between .npy in {npy_dir} and .png in {png_dir}. "
            "Cannot auto-pick MSI.",
        )
    hw_best = max(shared, key=lambda hw: int(hw[0]) * int(hw[1]))
    cands = sorted(buckets[hw_best], key=lambda p: p.name.lower())
    for name in ("0.npy", "1.npy"):
        for p in cands:
            if p.name.lower() == name:
                logger.info(
                    "input.npy_path unset - using MSI file %s (shared size %s with png dir)",
                    p,
                    hw_best,
                )
                return p
    chosen = cands[0]
    logger.info(
        "input.npy_path unset - using MSI file %s (shared size %s with png dir)",
        chosen,
        hw_best,
    )
    return chosen


def _build_npy_png_shape_matching_report(
    npy_dir: Path,
    png_dir: Path,
    *,
    msi_npy_path: Path | None = None,
    target_hw: Tuple[int, int] | None = None,
    transpose_msi: bool = False,
) -> str:
    """
    Summarize every *.npy / *.png under the two dirs by spatial (H, W), and show greedy pairs
    (sorted filename, 1:1) within each size — same rule as _pair_npy_png_paths_by_shape.

    For visualization_pairing=by_shape, the edge pipeline only scores those greedy pairs, not every
    file listed in a bucket. Use visualization_pairing=matched_dirs for one MSI .npy : many images.
    """
    from collections import defaultdict

    npy_by_hw: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    npy_errors: List[str] = []
    for p in sorted(npy_dir.glob("*.npy"), key=lambda x: x.name.lower()):
        try:
            hw = _spatial_hw_msi_cube_file(p, transpose_msi)
        except Exception as e:
            npy_errors.append(f"  {p.name}: {e}")
            continue
        npy_by_hw[hw].append(p)

    png_by_hw: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    png_errors: List[str] = []
    for p in sorted(png_dir.glob("*.png"), key=lambda x: x.name.lower()):
        try:
            hw = _spatial_hw_png(p)
        except Exception as e:
            png_errors.append(f"  {p.name}: {e}")
            continue
        png_by_hw[hw].append(p)

    lines: List[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("NPY <-> PNG shape matching (grouped by spatial H x W)")
    lines.append(
        "  Note (by_shape): edge metrics use greedy 1:1 pairs per size (min(n_npy, n_png)), "
        "not every file listed in a bucket. For 1 cube : many PNGs use visualization_pairing=matched_dirs.",
    )
    lines.append(f"  MSI .npy on-disk layout: {'C x H x W (transpose_msi)' if transpose_msi else 'H x W x C'}")
    lines.append(f"  visualization_npy_dir: {npy_dir}")
    lines.append(f"  visualization_png_dir: {png_dir}")
    if msi_npy_path is not None:
        lines.append(f"  MSI / HD edges loaded from: {msi_npy_path}")
    if target_hw is not None:
        lines.append(
            f"  Target grid for pairing this run (MSI H x W): {target_hw[0]} x {target_hw[1]}",
        )
    lines.append("=" * 78)

    if npy_errors:
        lines.append("\n.npy files that could not be read:")
        lines.extend(npy_errors)
    if png_errors:
        lines.append("\n.png files that could not be read:")
        lines.extend(png_errors)

    all_hw = sorted(
        set(npy_by_hw.keys()) | set(png_by_hw.keys()),
        key=lambda x: (x[0], x[1]),
    )
    if not all_hw and not npy_errors and not png_errors:
        lines.append("\n(no .npy or .png found in the directories)")
        return "\n".join(lines)

    for hw in all_hw:
        h, w = int(hw[0]), int(hw[1])
        npys = sorted(npy_by_hw.get(hw, []), key=lambda p: p.name.lower())
        pngs = sorted(png_by_hw.get(hw, []), key=lambda p: p.name.lower())
        tag = ""
        if target_hw is not None and (h, w) == (int(target_hw[0]), int(target_hw[1])):
            tag = "  [this size is used for MSI-matched viz pairs in this run]"
        lines.append("")
        lines.append(f"--- Size H x W = {h} x {w}{tag} ---")
        lines.append(f"  .npy count: {len(npys)}")
        for p in npys:
            lines.append(f"    npy  {p.name}  ->  shape ({h}, {w})  |  {p}")
        if not npys:
            lines.append("    (no .npy at this size)")
        lines.append(f"  .png count: {len(pngs)}")
        for p in pngs:
            lines.append(f"    png  {p.name}  ->  size W x H = {w} x {h}  |  {p}")
        if not pngs:
            lines.append("    (no .png at this size)")

        if npys and pngs:
            n_pair = min(len(npys), len(pngs))
            lines.append(f"  Greedy pairs (sorted filename, 1:1, first {n_pair}):")
            for i in range(n_pair):
                lines.append(f"    {npys[i].name}  <->  {pngs[i].name}")
            if len(npys) != len(pngs):
                lines.append(
                    f"  Note: unpaired extras - {max(0, len(npys) - n_pair)} npy, "
                    f"{max(0, len(pngs) - n_pair)} png",
                )
        elif npys and not pngs:
            lines.append("  (no pairing: missing .png at this size)")
        elif pngs and not npys:
            lines.append("  (no pairing: missing .npy at this size)")

    lines.append("")
    lines.append("=" * 78)
    lines.append("")
    return "\n".join(lines)


def _filter_viz_paths_to_msi_shape(
    viz_paths: List[Path],
    target_hw: Tuple[int, int],
) -> List[Path]:
    """Drop paths whose spatial size differs from MSI (when allow_resize is false)."""
    tgt = tuple(int(x) for x in target_hw)
    out: List[Path] = []
    for p in viz_paths:
        try:
            hw = _viz_path_spatial_hw(p)
        except Exception as e:
            logger.warning("skip %s: could not read spatial size | %s", p, e)
            continue
        if hw != tgt:
            logger.warning(
                "skip %s: spatial size %s != MSI %s (set input.npy_path to match, or allow_resize=true)",
                p,
                hw,
                tgt,
            )
            continue
        out.append(p)
    return out


def _pair_npy_png_paths_by_shape(
    npy_dir: Path,
    png_dir: Path,
    target_hw: Tuple[int, int] | None = None,
    transpose_msi: bool = False,
) -> List[Path]:
    """
    Match *.npy under npy_dir with *.png under png_dir by equal (H, W).
    Within each bucket, pair by sorted filename (greedy 1:1). Unmatched files are skipped with a warning.

    If target_hw is set, only the bucket with that (H, W) is paired (must match MSI / reference grid).

    Returns a flat list [npy1, png1, npy2, png2, ...] so each pair is adjacent.
    """
    from collections import defaultdict

    buckets: Dict[Tuple[int, int], Dict[str, List[Path]]] = defaultdict(lambda: {"npy": [], "png": []})
    for p in sorted(npy_dir.glob("*.npy"), key=lambda x: x.name.lower()):
        try:
            hw = _spatial_hw_msi_cube_file(p, transpose_msi)
        except Exception as e:
            logger.warning("skip npy (shape read failed): %s | %s", p, e)
            continue
        buckets[hw]["npy"].append(p)
    for p in sorted(png_dir.glob("*.png"), key=lambda x: x.name.lower()):
        try:
            hw = _spatial_hw_png(p)
        except Exception as e:
            logger.warning("skip png (size read failed): %s | %s", p, e)
            continue
        buckets[hw]["png"].append(p)

    out: List[Path] = []
    for hw in sorted(buckets.keys(), key=lambda x: (x[0], x[1])):
        if target_hw is not None and hw != tuple(target_hw):
            n_npy, n_png = len(buckets[hw]["npy"]), len(buckets[hw]["png"])
            if n_npy or n_png:
                logger.info(
                    "by_shape: skipping spatial size %s (MSI / target is %s) - %d .npy, %d .png",
                    hw,
                    target_hw,
                    n_npy,
                    n_png,
                )
            continue
        npys = sorted(buckets[hw]["npy"], key=lambda p: p.name.lower())
        pngs = sorted(buckets[hw]["png"], key=lambda p: p.name.lower())
        if not npys and not pngs:
            continue
        if not npys:
            logger.warning(
                "by_shape: no .npy for size %s - skipping %d png(s) (e.g. %s)",
                hw,
                len(pngs),
                pngs[0].name if pngs else "",
            )
            continue
        if not pngs:
            logger.warning(
                "by_shape: no .png for size %s - skipping %d npy(s) (e.g. %s)",
                hw,
                len(npys),
                npys[0].name if npys else "",
            )
            continue
        n = min(len(npys), len(pngs))
        if len(npys) != len(pngs):
            logger.warning(
                "by_shape: size %s has %d npy vs %d png - pairing first %d",
                hw,
                len(npys),
                len(pngs),
                n,
            )
        for i in range(n):
            out.extend([npys[i], pngs[i]])
    return out


def _sobel_edge_energy(img2d: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(img2d, dtype=np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)
    edge[~valid_mask] = 0.0
    return edge


def _nmf_component_preprocess_channel(
    ch: np.ndarray,
    valid_mask: np.ndarray,
    *,
    smooth_sigma: float,
    hist_equalize: bool,
) -> np.ndarray:
    """
    Optional Gaussian smoothing then OpenCV histogram equalization on each 2D component map
    (before percentile normalization and Sobel). Invalid pixels stay zero.
    """
    out = np.asarray(ch, dtype=np.float32)
    s = float(smooth_sigma)
    if s > 0:
        ksize = int(max(3, 2 * int(round(3 * s)) + 1))
        out = cv2.GaussianBlur(out, (ksize, ksize), sigmaX=s, sigmaY=s)
    if hist_equalize:
        vm = np.asarray(valid_mask, dtype=bool)
        v = out[vm]
        if v.size == 0:
            return out
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi <= lo + 1e-12:
            return out
        u8 = np.zeros(out.shape, dtype=np.uint8)
        u8[vm] = np.clip((out[vm] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        u8_eq = cv2.equalizeHist(u8)
        out = np.asarray(u8_eq, dtype=np.float32) / 255.0
        out[~vm] = 0.0
    return out


def _normalize_channel_percentile(
    img2d: np.ndarray,
    valid_mask: np.ndarray,
    low_pct: float,
    high_pct: float,
    eps: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(img2d, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[valid_mask]
    if vals.size == 0:
        return out
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    if hi <= lo + eps:
        return out
    out = (x - lo) / (hi - lo + eps)
    out = np.clip(out, 0.0, 1.0)
    out[~valid_mask] = 0.0
    return out.astype(np.float32, copy=False)


def _normalize_channel_zscore(
    img2d: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-channel z-score on valid pixels; invalid pixels set to 0."""
    x = np.asarray(img2d, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[valid_mask]
    if vals.size == 0:
        return out
    mu = float(np.mean(vals))
    std = float(np.std(vals))
    if std < eps:
        return out
    out = (x - mu) / (std + eps)
    out[~valid_mask] = 0.0
    return out.astype(np.float32, copy=False)


def _highd_edge_maps_base(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    hd_cfg: object,
) -> Tuple[np.ndarray, np.ndarray]:
    normalize_channels = bool(getattr(hd_cfg, "normalize_channels", False)) if hd_cfg is not None else False
    channel_low = float(getattr(hd_cfg, "channel_low_percentile", 1.0)) if hd_cfg is not None else 1.0
    channel_high = float(getattr(hd_cfg, "channel_high_percentile", 99.0)) if hd_cfg is not None else 99.0
    use_log1p = bool(getattr(hd_cfg, "log1p", False)) if hd_cfg is not None else False
    agg_mode = str(getattr(hd_cfg, "aggregation", "mean")).strip().lower() if hd_cfg is not None else "mean"
    topk = int(getattr(hd_cfg, "topk", 5)) if hd_cfg is not None else 5
    nb_cfg = getattr(hd_cfg, "neighborhood", None) if hd_cfg is not None else None
    nb_enabled = bool(getattr(nb_cfg, "enabled", False)) if nb_cfg is not None else False
    nb_radius = int(getattr(nb_cfg, "radius", 1)) if nb_cfg is not None else 1
    nb_sigma = float(getattr(nb_cfg, "sigma", 1.0)) if nb_cfg is not None else 1.0
    nb_metric = str(getattr(nb_cfg, "metric", "l2")).strip().lower() if nb_cfg is not None else "l2"

    c = int(msi.shape[2])
    channels = np.zeros((*msi.shape[:2], c), dtype=np.float32)
    for i in range(c):
        ch = np.asarray(msi[:, :, i], dtype=np.float32)
        if use_log1p:
            ch = np.log1p(np.clip(ch, 0.0, None)).astype(np.float32, copy=False)
        if normalize_channels:
            ch = _normalize_channel_percentile(ch, valid_mask, channel_low, channel_high)
        channels[:, :, i] = ch

    if nb_enabled:
        h, w, _ = channels.shape
        r = max(1, int(nb_radius))
        sigma = float(nb_sigma)
        metric = "sam" if nb_metric == "sam" else "l2"
        edge_sum = np.zeros((h, w), dtype=np.float32)
        edge_max = np.zeros((h, w), dtype=np.float32)
        eps = 1e-8

        for dy in range(0, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx <= 0:
                    continue
                if dy == 0 and dx == 0:
                    continue
                if dy > 0 or (dy == 0 and dx > 0):
                    y0a, y1a = dy, h
                    y0b, y1b = 0, h - dy
                    if dx >= 0:
                        x0a, x1a = dx, w
                        x0b, x1b = 0, w - dx
                    else:
                        x0a, x1a = 0, w + dx
                        x0b, x1b = -dx, w

                    a = channels[y0a:y1a, x0a:x1a, :]
                    b = channels[y0b:y1b, x0b:x1b, :]
                    m = valid_mask[y0a:y1a, x0a:x1a] & valid_mask[y0b:y1b, x0b:x1b]
                    if not np.any(m):
                        continue

                    if metric == "sam":
                        dot = np.sum(a * b, axis=-1)
                        na = np.linalg.norm(a, axis=-1)
                        nb = np.linalg.norm(b, axis=-1)
                        cos = np.clip(dot / (na * nb + eps), -1.0, 1.0)
                        dist = np.arccos(cos).astype(np.float32, copy=False) / np.pi
                    else:
                        dist = np.linalg.norm(a - b, axis=-1).astype(np.float32, copy=False)

                    if sigma > 0:
                        w_sp = float(np.exp(-((dx * dx + dy * dy) / (2.0 * sigma * sigma))))
                    else:
                        w_sp = 1.0
                    d = np.where(m, dist * w_sp, 0.0).astype(np.float32, copy=False)

                    edge_sum[y0a:y1a, x0a:x1a] += d
                    edge_sum[y0b:y1b, x0b:x1b] += d
                    edge_max[y0a:y1a, x0a:x1a] = np.maximum(edge_max[y0a:y1a, x0a:x1a], d)
                    edge_max[y0b:y1b, x0b:x1b] = np.maximum(edge_max[y0b:y1b, x0b:x1b], d)

        edge_sum[~valid_mask] = 0.0
        edge_max[~valid_mask] = 0.0
        return edge_sum.astype(np.float32, copy=False), edge_max.astype(np.float32, copy=False)

    edges = np.zeros((*msi.shape[:2], c), dtype=np.float32)
    for i in range(c):
        edges[:, :, i] = _sobel_edge_energy(channels[:, :, i], valid_mask)

    if agg_mode == "max":
        out = np.max(edges, axis=2)
    elif agg_mode == "topk_mean":
        k = max(1, min(int(topk), c))
        top = np.partition(edges, kth=c - k, axis=2)[:, :, c - k :]
        out = np.mean(top, axis=2)
    else:
        out = np.mean(edges, axis=2)

    out_linf = np.max(edges, axis=2)
    out = np.asarray(out, dtype=np.float32)
    out_linf = np.asarray(out_linf, dtype=np.float32)
    out[~valid_mask] = 0.0
    out_linf[~valid_mask] = 0.0
    return out, out_linf


def _pca_project_msi_for_edges(msi: np.ndarray, valid_mask: np.ndarray, hd_cfg: object) -> np.ndarray:
    pca_cfg = getattr(hd_cfg, "pca", None) if hd_cfg is not None else None
    enabled = bool(getattr(pca_cfg, "enabled", False)) if pca_cfg is not None else False
    if not enabled:
        return np.asarray(msi, dtype=np.float32)

    x = np.asarray(msi[valid_mask], dtype=np.float32)
    if x.size == 0:
        return np.asarray(msi, dtype=np.float32)
    n_components = int(getattr(pca_cfg, "n_components", 16))
    random_state = int(getattr(pca_cfg, "random_state", 42))
    k = max(1, min(int(n_components), int(x.shape[0]), int(x.shape[1])))
    pca = PCA(n_components=k, random_state=random_state, svd_solver="randomized")
    z = pca.fit_transform(x).astype(np.float32, copy=False)
    out = np.zeros((msi.shape[0], msi.shape[1], k), dtype=np.float32)
    out[valid_mask] = z
    return out


def _compute_tic_sobel_edge_map(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rgb_color_space: str = "gray",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Sobel edges on the total-ion-current map (sum over m/z channels).

    Expects **unnormalized** intensities; TIC-normalized cubes have ~constant sum and are unsuitable.
    """
    tic = np.asarray(msi, dtype=np.float32).sum(axis=-1)
    out = np.zeros(valid_mask.shape, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(mask):
        return out
    tiss = tic[mask]
    lo = float(np.percentile(tiss, float(low_percentile)))
    hi = float(np.percentile(tiss, float(high_percentile)))
    tic_n = np.clip((tic - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32, copy=False)
    tic_u8 = (tic_n * 255.0).astype(np.uint8)
    rgb = np.stack([tic_u8, tic_u8, tic_u8], axis=-1)
    edge = _viz_edge_map(rgb, mask, color_space=str(rgb_color_space), already_01=False)
    out[mask] = edge[mask]
    return out


def _viz_edge_map(rgb_uint8: np.ndarray, valid_mask: np.ndarray, color_space: str = "gray", already_01: bool = False) -> np.ndarray:
    """Edge map from viz. color_space: gray (Luminance) | lab (max over L,A,B).
    already_01: if True, input is float [0,1]; must convert to uint8 for cvtColor (OpenCV LAB output scale differs by dtype)."""
    cs = str(color_space).strip().lower()
    if cs == "lab":
        if already_01:
            rgb_u8 = (np.clip(rgb_uint8, 0.0, 1.0) * 255.0).astype(np.uint8)
            lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
            lab01 = lab.astype(np.float32) / 255.0
        else:
            lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
            lab01 = lab.astype(np.float32) / 255.0
        edges = np.zeros((*lab01.shape[:2], 3), dtype=np.float32)
        for i in range(3):
            edges[:, :, i] = _sobel_edge_energy(lab01[:, :, i], valid_mask)
        out = np.max(edges, axis=2).astype(np.float32, copy=False)
        out[~valid_mask] = 0.0
        return out
    if already_01:
        gray_u8 = (np.clip(rgb_uint8, 0.0, 1.0) * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(gray_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return _sobel_edge_energy(gray, valid_mask)


def _viz_edge_map_linf(rgb_uint8: np.ndarray, valid_mask: np.ndarray, color_space: str = "rgb", already_01: bool = False) -> np.ndarray:
    """L-inf edge map. color_space: rgb (max over R,G,B) | lab (max over L,A,B).
    already_01: if True, input is float [0,1]; convert to uint8 for cvtColor to get correct LAB scale."""
    cs = str(color_space).strip().lower()
    if cs == "lab":
        if already_01:
            rgb_u8 = (np.clip(rgb_uint8, 0.0, 1.0) * 255.0).astype(np.uint8)
            lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
            lab01 = lab.astype(np.float32) / 255.0
        else:
            lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
            lab01 = lab.astype(np.float32) / 255.0
        edges = np.zeros((*lab01.shape[:2], 3), dtype=np.float32)
        for i in range(3):
            edges[:, :, i] = _sobel_edge_energy(lab01[:, :, i], valid_mask)
    else:
        if already_01:
            rgb01 = np.clip(rgb_uint8, 0.0, 1.0).astype(np.float32)
        else:
            rgb01 = rgb_uint8.astype(np.float32) / 255.0
        edges = np.zeros((*rgb01.shape[:2], 3), dtype=np.float32)
        for i in range(3):
            edges[:, :, i] = _sobel_edge_energy(rgb01[:, :, i], valid_mask)
    out = np.max(edges, axis=2).astype(np.float32, copy=False)
    out[~valid_mask] = 0.0
    return out


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(arr, dtype=np.float32)
    ksize = int(max(3, 2 * int(round(3 * sigma)) + 1))
    return cv2.GaussianBlur(np.asarray(arr, dtype=np.float32), (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))


def _to_u8_from01(x01: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x01, dtype=np.float32), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _aggregate_channel_stack(stack: np.ndarray, mode: str, topk: int) -> np.ndarray:
    if stack.ndim != 3 or stack.shape[2] == 0:
        return np.zeros(stack.shape[:2], dtype=np.float32)
    m = str(mode).strip().lower()
    if m == "max":
        return np.max(stack, axis=2).astype(np.float32, copy=False)
    if m == "topk_mean":
        c = int(stack.shape[2])
        k = max(1, min(int(topk), c))
        top = np.partition(stack, kth=c - k, axis=2)[:, :, c - k :]
        return np.mean(top, axis=2).astype(np.float32, copy=False)
    return np.mean(stack, axis=2).astype(np.float32, copy=False)


def _canny_edge_from_u8(
    img_u8: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    aperture_size: int,
    l2gradient: bool,
) -> np.ndarray:
    e = cv2.Canny(
        img_u8,
        threshold1=float(low_threshold),
        threshold2=float(high_threshold),
        apertureSize=int(aperture_size),
        L2gradient=bool(l2gradient),
    )
    return (e.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def _canny_stack_from_channels(
    channels01: np.ndarray,
    sigma: float,
    low_threshold: float,
    high_threshold: float,
    aperture_size: int,
    l2gradient: bool,
    n_jobs: int,
) -> np.ndarray:
    h, w, c = channels01.shape
    stack = np.zeros((h, w, c), dtype=np.float32)
    sigma_f = float(sigma)

    def _one_channel(i: int) -> Tuple[int, np.ndarray]:
        ch = np.asarray(channels01[:, :, i], dtype=np.float32)
        if sigma_f > 0:
            ch = _gaussian_blur(ch, sigma_f)
        ch_u8 = _to_u8_from01(ch)
        return i, _canny_edge_from_u8(ch_u8, low_threshold, high_threshold, aperture_size, l2gradient)

    nj = int(n_jobs)
    if nj <= 0:
        nj = max(1, (os.cpu_count() or 1))
    nj = min(nj, c)
    if nj <= 1 or c <= 2:
        for i in range(c):
            _, e = _one_channel(i)
            stack[:, :, i] = e
        return stack

    with ThreadPoolExecutor(max_workers=nj) as ex:
        for i, e in ex.map(_one_channel, range(c)):
            stack[:, :, i] = e
    return stack


def _normalize_on_mask(arr: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[mask]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo + eps:
        out = np.zeros_like(x, dtype=np.float32)
        out[~mask] = 0.0
        return out
    out = (x - lo) / (hi - lo + eps)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    out[~mask] = 0.0
    return out


def _aggregate_multiscale_maps(maps: List[np.ndarray], valid_mask: np.ndarray, mode: str, normalize: bool = True) -> np.ndarray:
    if len(maps) == 0:
        out = np.zeros(valid_mask.shape, dtype=np.float32)
        out[~valid_mask] = 0.0
        return out
    if normalize:
        normalized = [_normalize_on_mask(m, valid_mask) for m in maps]
    else:
        normalized = [np.asarray(m, dtype=np.float32) for m in maps]
    stack = np.stack(normalized, axis=0)
    if str(mode).strip().lower() == "max":
        out = np.max(stack, axis=0)
    else:
        out = np.mean(stack, axis=0)
    out = np.asarray(out, dtype=np.float32)
    out[~valid_mask] = 0.0
    return out


def _neighbor_dist(a: np.ndarray, b: np.ndarray, metric: str, eps: float = 1e-8) -> np.ndarray:
    """Compute distance between pixel spectral vectors a and b (HxWxC)."""
    metric = str(metric).strip().lower()
    if metric == "l2":
        return np.linalg.norm(a - b, axis=-1).astype(np.float32, copy=False)
    if metric == "sam":
        dot = np.sum(a * b, axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cos = np.clip(dot / (na * nb + eps), -1.0, 1.0)
        return (np.arccos(cos) / np.pi).astype(np.float32, copy=False)
    if metric == "cosine":
        dot = np.sum(a * b, axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cos = np.clip(dot / (na * nb + eps), -1.0, 1.0)
        return (1.0 - cos).astype(np.float32, copy=False)
    if metric == "chebyshev":
        return np.max(np.abs(a - b), axis=-1).astype(np.float32, copy=False)
    if metric == "manhattan":
        return np.sum(np.abs(a - b), axis=-1).astype(np.float32, copy=False)
    if metric == "braycurtis":
        diff = np.abs(a - b)
        total = np.abs(a) + np.abs(b) + eps
        return np.sum(diff, axis=-1) / np.sum(total, axis=-1)
    return np.linalg.norm(a - b, axis=-1).astype(np.float32, copy=False)


def _compute_neighborhood_edge_map(
    channels: np.ndarray,
    valid_mask: np.ndarray,
    metric: str,
    radius: int = 1,
    sigma: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Edge map from max spatial-neighbor distance per pixel."""
    h, w, _ = channels.shape
    r = max(1, int(radius))
    edge_max = np.zeros((h, w), dtype=np.float32)
    for dy in range(0, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx <= 0:
                continue
            if dy == 0 and dx == 0:
                continue
            if dy > 0 or (dy == 0 and dx > 0):
                y0a, y1a = dy, h
                y0b, y1b = 0, h - dy
                if dx >= 0:
                    x0a, x1a = dx, w
                    x0b, x1b = 0, w - dx
                else:
                    x0a, x1a = 0, w + dx
                    x0b, x1b = -dx, w
                a = channels[y0a:y1a, x0a:x1a, :]
                b = channels[y0b:y1b, x0b:x1b, :]
                m = valid_mask[y0a:y1a, x0a:x1a] & valid_mask[y0b:y1b, x0b:x1b]
                if not np.any(m):
                    continue
                dist = _neighbor_dist(a, b, metric, eps)
                if sigma > 0:
                    w_sp = float(np.exp(-((dx * dx + dy * dy) / (2.0 * sigma * sigma))))
                else:
                    w_sp = 1.0
                d = np.where(m, dist * w_sp, 0.0).astype(np.float32, copy=False)
                edge_max[y0a:y1a, x0a:x1a] = np.maximum(edge_max[y0a:y1a, x0a:x1a], d)
                edge_max[y0b:y1b, x0b:x1b] = np.maximum(edge_max[y0b:y1b, x0b:x1b], d)
    edge_max[~valid_mask] = 0.0
    return edge_max


def _zscore_map(arr: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score on valid pixels; invalid pixels set to 0."""
    out = np.zeros_like(arr, dtype=np.float32)
    vals = arr[mask]
    if vals.size == 0:
        return out
    mu = float(np.mean(vals))
    std = float(np.std(vals))
    if std < eps:
        return out
    out = (arr - mu) / (std + eps)
    out[~mask] = 0.0
    return out.astype(np.float32, copy=False)


def _highd_edge_maps_distance_multi(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    hd_cfg: object,
) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    """
    Compute edge maps with multiple distance functions, z-score each, mean-aggregate.
    Returns (merged, merged_linf, individual_maps_dict). individual_maps_dict is None unless
    save_individual_distance_maps is True in hd_cfg.
    """
    distances = list(getattr(hd_cfg, "distance_functions", ["l2", "sam", "cosine", "chebyshev", "manhattan"]) or ["l2"])
    nb_cfg = getattr(hd_cfg, "neighborhood", None) or {}
    radius = int(getattr(nb_cfg, "radius", 1))
    sigma = float(getattr(nb_cfg, "sigma", 1.0))
    normalize_channels = bool(getattr(hd_cfg, "normalize_channels", False))
    channel_low = float(getattr(hd_cfg, "channel_low_percentile", 1.0))
    channel_high = float(getattr(hd_cfg, "channel_high_percentile", 99.0))
    use_log1p = bool(getattr(hd_cfg, "log1p", False))
    save_individual = bool(getattr(hd_cfg, "save_individual_distance_maps", False))
    metrics = [str(m).strip().lower() for m in distances]
    # Percentile scaling to [0, 1] makes Chebyshev (L∞) neighbor distance saturate at 1 for
    # high-dimensional MSI cubes, yielding a constant map and a useless z-score.
    use_zscore_channels = normalize_channels and metrics and all(m == "chebyshev" for m in metrics)

    h, w, c = msi.shape
    channels = np.zeros((h, w, c), dtype=np.float32)
    for i in range(c):
        ch = np.asarray(msi[:, :, i], dtype=np.float32)
        if use_log1p:
            ch = np.log1p(np.clip(ch, 0.0, None)).astype(np.float32, copy=False)
        if normalize_channels:
            if use_zscore_channels:
                ch = _normalize_channel_zscore(ch, valid_mask)
            else:
                ch = _normalize_channel_percentile(ch, valid_mask, channel_low, channel_high)
        channels[:, :, i] = ch

    edge_maps: List[np.ndarray] = []
    for metric in distances:
        m = str(metric).strip().lower()
        em = _compute_neighborhood_edge_map(channels, valid_mask, m, radius=radius, sigma=sigma)
        z = _zscore_map(em, valid_mask)
        if m == "chebyshev" and np.any(valid_mask):
            em_v = em[valid_mask]
            z_v = z[valid_mask]
            # Use print so output is visible under Hydra (often hides INFO from other modules).
            print(
                f"[chebyshev] raw neighbor-distance (channel-norm spectra) | mean={float(np.mean(em_v)):.6g} "
                f"max={float(np.max(em_v)):.6g}",
                flush=True,
            )
            print(
                f"[chebyshev] after z-score | mean={float(np.mean(z_v)):.6g} max={float(np.max(z_v)):.6g}",
                flush=True,
            )
        edge_maps.append(z)

    if not edge_maps:
        out = np.zeros(valid_mask.shape, dtype=np.float32)
        out[~valid_mask] = 0.0
        return out, out.copy(), None

    out = np.mean(np.stack(edge_maps, axis=0), axis=0).astype(np.float32, copy=False)
    out[~valid_mask] = 0.0
    logger.info(
        "distance_multi: metrics=%s radius=%d -> z-scored mean",
        distances,
        radius,
    )
    individual = None
    if save_individual:
        individual = {str(distances[i]): edge_maps[i].copy() for i in range(len(distances))}
    return out, out.copy(), individual


def _cluster_boundary_strength_from_labels(labels_hw: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fraction of 4-neighbors with a different discrete label (0..1); invalid = -1 internally."""
    L = np.where(valid_mask, labels_hw.astype(np.int32), -1)
    h, w = L.shape
    acc = np.zeros((h, w), dtype=np.float32)
    m = valid_mask[:-1, :] & valid_mask[1:, :]
    ne = (L[:-1] != L[1:]).astype(np.float32)
    ne = np.where(m, ne, 0.0)
    acc[:-1] += ne
    acc[1:] += ne
    m2 = valid_mask[:, :-1] & valid_mask[:, 1:]
    ne2 = (L[:, :-1] != L[:, 1:]).astype(np.float32)
    ne2 = np.where(m2, ne2, 0.0)
    acc[:, :-1] += ne2
    acc[:, 1:] += ne2
    acc = np.where(valid_mask, acc / 4.0, 0.0)
    return acc.astype(np.float32, copy=False)


def _highd_edge_maps_ensemble_cluster(
    data: np.ndarray,
    valid_mask: np.ndarray,
    hd_cfg: object,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Consensus *boundary* map from many clusterings (not a single clustering objective).

    Each experiment: subsample valid pixels -> PCA on subset -> fit KMeans on transformed
    subset -> predict labels for all valid pixels -> boundary strength from label disagreements
    with 4-neighbors. Mean-aggregate boundary maps across experiments; ``linf`` is the per-pixel
    max across experiments (analogous to other HD methods).

    Optional (``hd_edges.ensemble_cluster``): ``output_gamma`` (default 1): apply ``x**gamma`` on
    valid pixels to emphasize strong boundaries (try 1.5–3). ``suppress_below_percentile`` in (0,100)
    zeros values below that percentile of the map on valid tissue (after gamma).
    ``per_run_normalize_by_sum``: if true, divide each experiment's boundary map by
    ``sum(bmap[valid]) + per_run_normalize_epsilon`` before mean/max aggregation so each run has
    comparable total mass (reduces dominance of high-``k`` partitions with more edge pixels).
    """
    ec = getattr(hd_cfg, "ensemble_cluster", None) if hd_cfg is not None else None
    n_exp = int(getattr(ec, "n_experiments", 10)) if ec is not None else 10
    n_exp = max(1, n_exp)
    subsample_frac = float(getattr(ec, "subsample_fraction", 0.5)) if ec is not None else 0.5
    subsample_frac = float(np.clip(subsample_frac, 0.05, 1.0))
    pca_dim = int(getattr(ec, "pca_n_components", 16)) if ec is not None else 16
    k_list = list(getattr(ec, "n_clusters_list", [6, 8, 10, 12])) if ec is not None else [6, 8, 10, 12]
    if not k_list:
        k_single = int(getattr(ec, "n_clusters", 8)) if ec is not None else 8
        k_list = [max(2, k_single)]
    k_list = [max(2, int(k)) for k in k_list]
    kmeans_n_init = int(getattr(ec, "kmeans_n_init", 8)) if ec is not None else 8
    kmeans_n_init = max(1, kmeans_n_init)
    random_state_base = int(getattr(ec, "random_state", 42)) if ec is not None else 42
    per_run_norm_sum = bool(getattr(ec, "per_run_normalize_by_sum", False)) if ec is not None else False
    try:
        per_run_eps = float(getattr(ec, "per_run_normalize_epsilon", 1e-12)) if ec is not None else 1e-12
    except (TypeError, ValueError):
        per_run_eps = 1e-12
    per_run_eps = max(per_run_eps, 0.0)

    x = np.asarray(data[valid_mask], dtype=np.float64)
    n_pix, n_chan = x.shape
    h, w = valid_mask.shape
    if n_pix < 4 or n_chan < 1:
        z = np.zeros((h, w), dtype=np.float32)
        return z, z.copy()

    sum_b = np.zeros((h, w), dtype=np.float32)
    max_b = np.zeros((h, w), dtype=np.float32)

    t0 = time.perf_counter()
    for e in range(n_exp):
        rng = np.random.default_rng(random_state_base + e)
        k = k_list[e % len(k_list)]
        n_fit = max(int(k * 2), int(subsample_frac * n_pix))
        n_fit = min(n_fit, n_pix)
        fit_idx = rng.choice(n_pix, size=n_fit, replace=False)
        x_fit = x[fit_idx]

        n_comp = min(pca_dim, n_fit - 1, n_chan)
        n_comp = max(1, n_comp)
        pca = PCA(n_components=n_comp, random_state=random_state_base + e)
        pca.fit(x_fit)
        z_all = pca.transform(x).astype(np.float32, copy=False)
        z_fit = z_all[fit_idx]
        k_use = min(k, n_fit)
        k_use = max(2, k_use)
        km = KMeans(
            n_clusters=k_use,
            n_init=kmeans_n_init,
            random_state=random_state_base + e,
        )
        km.fit(z_fit)
        lab_flat = km.predict(z_all).astype(np.int32, copy=False)

        labels = np.full((h, w), -1, dtype=np.int32)
        labels[valid_mask] = lab_flat
        bmap = _cluster_boundary_strength_from_labels(labels, valid_mask)
        if per_run_norm_sum:
            vm_arr = np.asarray(valid_mask, dtype=bool)
            s = float(np.sum(bmap[vm_arr]))
            denom = s + per_run_eps
            bmap = (bmap / denom).astype(np.float32)
        sum_b += bmap
        max_b = np.maximum(max_b, bmap)

    mean_b = (sum_b / float(n_exp)).astype(np.float32, copy=False)
    mean_b[~valid_mask] = 0.0
    max_b[~valid_mask] = 0.0

    # Optional power-law (gamma > 1): weak boundary responses are in [0,1]; x^gamma pulls small
    # values toward 0 → sparser-looking maps before spatial normalization downstream.
    gamma = float(getattr(ec, "output_gamma", 1.0)) if ec is not None else 1.0
    if gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
        m = np.asarray(valid_mask, dtype=bool)
        mean_b = np.asarray(mean_b, dtype=np.float32).copy()
        max_b = np.asarray(max_b, dtype=np.float32).copy()
        mean_b[m] = np.power(np.clip(mean_b[m], 0.0, None), gamma).astype(np.float32)
        max_b[m] = np.power(np.clip(max_b[m], 0.0, None), gamma).astype(np.float32)

    supp = getattr(ec, "suppress_below_percentile", None) if ec is not None else None
    if supp is not None:
        try:
            sp = float(supp)
        except (TypeError, ValueError):
            sp = None
        if sp is not None and 0.0 < sp < 100.0 and np.any(valid_mask):
            vm = np.asarray(valid_mask, dtype=bool)
            for arr in (mean_b, max_b):
                t = float(np.percentile(arr[vm], sp))
                arr[vm & (arr < t)] = 0.0

    logger.info(
        "ensemble_cluster: n_exp=%d subsample_frac=%g pca_dim<=%d k_list=%s per_run_L1=%s eps=%g gamma=%s suppress_pct=%s | %.2fs",
        n_exp,
        subsample_frac,
        pca_dim,
        k_list,
        per_run_norm_sum,
        per_run_eps,
        gamma,
        supp,
        time.perf_counter() - t0,
    )
    return mean_b, max_b.astype(np.float32, copy=False)


def _highd_edge_maps_nmf_component_edges(
    data: np.ndarray,
    valid_mask: np.ndarray,
    hd_cfg: object,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each ``k`` in ``hd_edges.nmf_component_edges.k_list``: fit NMF on a random pixel subset,
    ``transform`` all valid pixels to obtain H×W×k nonnegative coefficients. Per component, optional
    Gaussian smoothing and histogram equalization (``smooth_sigma``, ``hist_equalize``), then normalize
    to [0,1] with ``_normalize_channel_percentile``, then Sobel gradient magnitude. Concatenate edge
    maps from every component of every ``k``; aggregate with ``aggregate`` (``mean`` or ``max``) for
    the primary map and pixelwise ``max`` for the L∞-style map.
    """
    nc = getattr(hd_cfg, "nmf_component_edges", None) if hd_cfg is not None else None
    k_list = list(getattr(nc, "k_list", [8, 16, 32, 64])) if nc is not None else [8, 16, 32, 64]
    k_list = list(dict.fromkeys(max(2, int(k)) for k in k_list))
    fit_frac = float(getattr(nc, "fit_fraction", 0.2)) if nc is not None else 0.2
    fit_frac = float(np.clip(fit_frac, 0.01, 1.0))
    random_state = int(getattr(nc, "random_state", 42)) if nc is not None else 42
    init = str(getattr(nc, "init", "nndsvda")) if nc is not None else "nndsvda"
    max_iter = int(getattr(nc, "max_iter", 300)) if nc is not None else 300
    tol = float(getattr(nc, "tol", 1e-4)) if nc is not None else 1e-4
    agg = str(getattr(nc, "aggregate", "mean")).strip().lower() if nc is not None else "mean"
    ch_lo = float(getattr(nc, "channel_low_percentile", 1.0)) if nc is not None else 1.0
    ch_hi = float(getattr(nc, "channel_high_percentile", 99.0)) if nc is not None else 99.0
    smooth_sigma = float(getattr(nc, "smooth_sigma", 0.0)) if nc is not None else 0.0
    hist_equalize = bool(getattr(nc, "hist_equalize", False)) if nc is not None else False

    h, w = valid_mask.shape
    x = np.maximum(np.asarray(data[valid_mask], dtype=np.float64), 0.0)
    n_pix, n_chan = x.shape
    if n_pix < 4 or n_chan < 1:
        z = np.zeros((h, w), dtype=np.float32)
        return z, z.copy()

    t0 = time.perf_counter()
    edge_maps: List[np.ndarray] = []
    rng = np.random.default_rng(random_state)

    for k in k_list:
        if n_pix < k:
            logger.warning("nmf_component_edges: skip k=%d (need >=k valid pixels)", k)
            continue
        n_fit = max(int(fit_frac * n_pix), k)
        n_fit = min(max(n_fit, k), n_pix)
        fit_idx = rng.choice(n_pix, size=n_fit, replace=False)
        x_fit = x[fit_idx]
        nmf = NMF(
            n_components=k,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        nmf.fit(x_fit)
        W = nmf.transform(x).astype(np.float32, copy=False)
        cub = np.zeros((h, w, k), dtype=np.float32)
        cub[valid_mask] = W
        for j in range(k):
            ch = cub[:, :, j]
            ch = _nmf_component_preprocess_channel(
                ch,
                valid_mask,
                smooth_sigma=smooth_sigma,
                hist_equalize=hist_equalize,
            )
            norm = _normalize_channel_percentile(ch, valid_mask, ch_lo, ch_hi)
            e = _sobel_edge_energy(norm, valid_mask)
            edge_maps.append(e)

    if not edge_maps:
        z = np.zeros((h, w), dtype=np.float32)
        return z, z.copy()

    stack = np.stack(edge_maps, axis=0)
    base_linf = np.max(stack, axis=0).astype(np.float32, copy=False)
    if agg == "max":
        base_primary = np.max(stack, axis=0).astype(np.float32, copy=False)
    else:
        base_primary = np.mean(stack, axis=0).astype(np.float32, copy=False)

    logger.info(
        "nmf_component_edges: k_list=%s n_edge_maps=%d aggregate=%s fit_frac=%g smooth_sigma=%g hist_equalize=%s | %.2fs",
        k_list,
        len(edge_maps),
        agg,
        fit_frac,
        smooth_sigma,
        hist_equalize,
        time.perf_counter() - t0,
    )
    return base_primary, base_linf


def _get_landmark_indices(
    x_norm: np.ndarray,
    n_landmarks: int,
    sampling: str,
    random_state: int,
    tic: np.ndarray | None = None,
    flat_valid: np.ndarray | None = None,
    img_width: int = 0,
) -> np.ndarray:
    """Sample landmark indices: random | stratified | kmeans | tic | spatial."""
    n_valid = x_norm.shape[0]
    n_landmarks = min(n_landmarks, n_valid)
    rng = np.random.default_rng(random_state)
    sampling = str(sampling).strip().lower()

    if sampling == "random":
        return rng.choice(n_valid, size=n_landmarks, replace=False).astype(np.int64)

    if sampling == "spatial" and flat_valid is not None and img_width > 0:
        # Grid-based: divide image into ~n_landmarks cells, pick one per cell
        coords = np.stack(
            (flat_valid // img_width, flat_valid % img_width),
            axis=1,
        ).astype(np.float64)
        n_cells = min(n_landmarks, n_valid)
        km = KMeans(n_clusters=n_cells, random_state=random_state, n_init=3, max_iter=30)
        km.fit(coords)
        # Closest pixel to each cell center
        dist = np.sum((coords[:, None, :] - km.cluster_centers_[None, :, :]) ** 2, axis=2)
        closest = np.argmin(dist, axis=0)
        idx = np.unique(closest).astype(np.int64)
        if len(idx) < n_landmarks:
            mask = np.ones(n_valid, dtype=bool)
            mask[idx] = False
            remaining = np.where(mask)[0].astype(np.int64)
            n_need = n_landmarks - len(idx)
            if remaining.size >= n_need:
                extra = rng.choice(remaining, size=n_need, replace=False)
                idx = np.concatenate([idx, extra])
        return idx[:n_landmarks]

    if sampling == "stratified":
        order = np.arange(n_valid, dtype=np.int64)
        rng.shuffle(order)
        stride = max(1, n_valid // n_landmarks)
        idx = order[::stride][:n_landmarks]
        if len(idx) < n_landmarks:
            extra = rng.choice(n_valid, size=n_landmarks - len(idx), replace=False)
            idx = np.concatenate([idx, extra])
        return idx[:n_landmarks]

    if sampling == "kmeans":
        km = KMeans(n_clusters=n_landmarks, random_state=random_state, n_init=3, max_iter=50)
        km.fit(x_norm)
        centers = km.cluster_centers_.astype(np.float32)
        sim = x_norm @ centers.T
        closest = np.argmax(sim, axis=0)
        idx = np.unique(closest).astype(np.int64)
        if len(idx) < n_landmarks:
            mask = np.ones(n_valid, dtype=bool)
            mask[idx] = False
            remaining = np.where(mask)[0].astype(np.int64)
            n_need = n_landmarks - len(idx)
            if remaining.size >= n_need:
                extra = rng.choice(remaining, size=n_need, replace=False)
                idx = np.concatenate([idx, extra])
        return idx[:n_landmarks]

    if sampling == "tic" and tic is not None and tic.size == n_valid:
        # TIC-weighted: higher intensity pixels more likely
        w = np.asarray(tic, dtype=np.float64).ravel()
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()
        return rng.choice(n_valid, size=n_landmarks, replace=False, p=w).astype(np.int64)

    return rng.choice(n_valid, size=n_landmarks, replace=False).astype(np.int64)


def _jaccard_dice_loop_python(
    knn_indices: np.ndarray,
    flat_valid: np.ndarray,
    rc_to_flat: np.ndarray,
    offset_dr: np.ndarray,
    offset_dc: np.ndarray,
    offset_weight: np.ndarray,
    n_landmarks: int,
    use_dice: bool,
    agg_mode: int,
    agg_percentile: float,
    edge_out: np.ndarray,
    dissim_buffer: np.ndarray,
    x_norm: np.ndarray | None = None,
    use_spectral_weight: bool = False,
) -> None:
    """
    Python fallback: compute set dissimilarity (Jaccard or Dice) for each pixel vs spatial neighbors.
    agg_mode: 0=max, 1=mean, 2=percentile
    """
    n_valid = knn_indices.shape[0]
    k = knn_indices.shape[1]
    n_offsets = offset_dr.shape[0]
    h, w = rc_to_flat.shape

    for ev_i in range(n_valid):
        flat_idx = flat_valid[ev_i]
        r = flat_idx // w
        c = flat_idx % w
        set_i = np.zeros(n_landmarks, dtype=np.uint8)
        for j in range(k):
            lidx = knn_indices[ev_i, j]
            if 0 <= lidx < n_landmarks:
                set_i[lidx] = 1

        sum_w = 0.0
        sum_wd = 0.0
        max_d = 0.0
        buf_idx = 0

        for o in range(n_offsets):
            dr = offset_dr[o]
            dc = offset_dc[o]
            rn = r + dr
            cn = c + dc
            if 0 <= rn < h and 0 <= cn < w:
                ev_j = rc_to_flat[rn, cn]
                if ev_j >= 0:
                    set_j = np.zeros(n_landmarks, dtype=np.uint8)
                    for j in range(k):
                        lidx = knn_indices[ev_j, j]
                        if 0 <= lidx < n_landmarks:
                            set_j[lidx] = 1
                    inter = 0
                    union = 0
                    sum_i = int(np.sum(set_i))
                    sum_j = int(np.sum(set_j))
                    for L in range(n_landmarks):
                        if set_i[L] or set_j[L]:
                            union += 1
                        if set_i[L] and set_j[L]:
                            inter += 1
                    if union > 0:
                        if use_dice:
                            dissim = 1.0 - 2.0 * inter / (sum_i + sum_j + 1e-12)
                        else:
                            dissim = 1.0 - inter / union
                        dissim = max(0.0, min(1.0, dissim))
                        wght = offset_weight[o]
                        if use_spectral_weight and x_norm is not None:
                            cos_sim = float(np.dot(x_norm[ev_i], x_norm[ev_j]))
                            wght = wght * max(0.0, 1.0 - cos_sim)
                        sum_w += wght
                        sum_wd += wght * dissim
                        max_d = max(max_d, dissim)
                        if buf_idx < dissim_buffer.shape[1]:
                            dissim_buffer[ev_i, buf_idx] = dissim
                            buf_idx += 1

        if agg_mode == 0:
            edge_out[r, c] = max_d
        elif agg_mode == 1 and sum_w > 1e-12:
            edge_out[r, c] = sum_wd / sum_w
        elif agg_mode == 2 and buf_idx > 0:
            vals = dissim_buffer[ev_i, :buf_idx]
            p = min(100.0, max(0.0, agg_percentile))
            edge_out[r, c] = float(np.percentile(vals, p))


if _NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _jaccard_dice_loop_numba_impl(
        knn_indices: np.ndarray,
        flat_valid: np.ndarray,
        rc_to_flat: np.ndarray,
        offset_dr: np.ndarray,
        offset_dc: np.ndarray,
        offset_weight: np.ndarray,
        n_landmarks: int,
        use_dice: bool,
        agg_mode: int,
        agg_percentile: float,
        edge_out: np.ndarray,
        dissim_buffer: np.ndarray,
        x_norm: np.ndarray,
        use_spectral_weight: bool,
    ) -> None:
        n_valid = knn_indices.shape[0]
        k = knn_indices.shape[1]
        n_offsets = offset_dr.shape[0]
        h, w = rc_to_flat.shape

        for ev_i in prange(n_valid):
            flat_idx = flat_valid[ev_i]
            r = flat_idx // w
            c = flat_idx % w
            set_i = np.zeros(n_landmarks, dtype=np.uint8)
            for j in range(k):
                lidx = knn_indices[ev_i, j]
                if 0 <= lidx < n_landmarks:
                    set_i[lidx] = 1

            sum_w = 0.0
            sum_wd = 0.0
            max_d = 0.0
            buf_idx = 0

            for o in range(n_offsets):
                dr = offset_dr[o]
                dc = offset_dc[o]
                rn = r + dr
                cn = c + dc
                if 0 <= rn < h and 0 <= cn < w:
                    ev_j = rc_to_flat[rn, cn]
                    if ev_j >= 0:
                        set_j = np.zeros(n_landmarks, dtype=np.uint8)
                        for j in range(k):
                            lidx = knn_indices[ev_j, j]
                            if 0 <= lidx < n_landmarks:
                                set_j[lidx] = 1
                        inter = 0
                        union = 0
                        sum_i = 0
                        sum_j = 0
                        for L in range(n_landmarks):
                            if set_i[L]:
                                sum_i += 1
                            if set_j[L]:
                                sum_j += 1
                            if set_i[L] or set_j[L]:
                                union += 1
                            if set_i[L] and set_j[L]:
                                inter += 1
                        if union > 0:
                            if use_dice:
                                denom = sum_i + sum_j + 1e-12
                                dissim = 1.0 - 2.0 * inter / denom
                            else:
                                dissim = 1.0 - inter / union
                            if dissim < 0.0:
                                dissim = 0.0
                            if dissim > 1.0:
                                dissim = 1.0
                            wght = offset_weight[o]
                            if use_spectral_weight:
                                sim_ij = 0.0
                                for d in range(x_norm.shape[1]):
                                    sim_ij += x_norm[ev_i, d] * x_norm[ev_j, d]
                                wght = wght * max(0.0, 1.0 - sim_ij)
                            sum_w += wght
                            sum_wd += wght * dissim
                            if dissim > max_d:
                                max_d = dissim
                            if buf_idx < dissim_buffer.shape[1]:
                                dissim_buffer[ev_i, buf_idx] = dissim
                                buf_idx += 1

            if agg_mode == 0:
                edge_out[r, c] = max_d
            elif agg_mode == 1 and sum_w > 1e-12:
                edge_out[r, c] = sum_wd / sum_w
            elif agg_mode == 2 and buf_idx > 0:
                # Inline percentile for Numba
                nv = buf_idx
                arr = dissim_buffer[ev_i, :buf_idx].copy()
                for i in range(nv - 1):
                    for j in range(i + 1, nv):
                        if arr[j] < arr[i]:
                            arr[i], arr[j] = arr[j], arr[i]
                idx = int((agg_percentile / 100.0) * (nv - 1) + 0.5)
                idx = min(idx, nv - 1)
                edge_out[r, c] = arr[idx]


def _normalize_landmark_k_list(raw: object) -> List[int]:
    """Scalar or list of positive ints; duplicates removed, sorted. Default [15]."""
    default: List[int] = [15]
    if raw is None:
        return default
    if OmegaConf.is_config(raw):
        if OmegaConf.is_list(raw):
            raw = list(raw)
        else:
            try:
                v = int(raw)
                return [max(1, v)]
            except (TypeError, ValueError):
                return default
    elif not isinstance(raw, (list, tuple)):
        try:
            v = int(raw)
            return [max(1, v)]
        except (TypeError, ValueError):
            return default
    out: List[int] = []
    for x in raw:
        try:
            v = int(x)
            if v > 0:
                out.append(v)
        except (TypeError, ValueError):
            continue
    out = sorted(set(out))
    return out if out else default


def _landmark_knn_pair_dissimilarity(
    members: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    k_use: int,
    use_dice: bool,
    batch_size: int,
) -> np.ndarray:
    out = np.zeros(idx_a.shape[0], dtype=np.float32)
    batch = max(1, int(batch_size))
    for start in range(0, idx_a.shape[0], batch):
        end = min(start + batch, idx_a.shape[0])
        ma = members[idx_a[start:end]]
        mb = members[idx_b[start:end]]
        inter = np.logical_and(ma, mb).sum(axis=1).astype(np.float32)
        if use_dice:
            dissim = 1.0 - inter / max(1.0, float(k_use))
        else:
            union = float(2 * k_use) - inter
            dissim = 1.0 - inter / np.maximum(union, 1.0)
        out[start:end] = np.clip(dissim, 0.0, 1.0).astype(np.float32)
    return out


def _landmark_knn_directional_contrast_edge(
    knn_indices: np.ndarray,
    flat_valid: np.ndarray,
    rc_to_flat: np.ndarray,
    directions: List[Tuple[int, int]],
    n_landmarks: int,
    use_dice: bool,
    texture_weight: float,
    aggregation: str,
    batch_size: int,
) -> np.ndarray:
    h, w = rc_to_flat.shape
    n_valid, k_use = knn_indices.shape
    members = np.zeros((n_valid, n_landmarks), dtype=bool)
    row_idx = np.repeat(np.arange(n_valid), k_use)
    members[row_idx, knn_indices.reshape(-1)] = True

    edge_values = np.zeros(n_valid, dtype=np.float32)
    counts = np.zeros(n_valid, dtype=np.float32)
    use_mean = str(aggregation).strip().lower() in {"mean", "avg", "average"}
    tw = max(0.0, float(texture_weight))

    for dr, dc in directions:
        center_idx: List[int] = []
        neg_idx: List[int] = []
        pos_idx: List[int] = []
        for ev_i, flat_idx in enumerate(flat_valid):
            r = int(flat_idx // w)
            c = int(flat_idx % w)
            rn = r - dr
            cn = c - dc
            rp = r + dr
            cp = c + dc
            if 0 <= rn < h and 0 <= cn < w and 0 <= rp < h and 0 <= cp < w:
                ev_neg = int(rc_to_flat[rn, cn])
                ev_pos = int(rc_to_flat[rp, cp])
                if ev_neg >= 0 and ev_pos >= 0:
                    center_idx.append(ev_i)
                    neg_idx.append(ev_neg)
                    pos_idx.append(ev_pos)
        if not center_idx:
            continue

        center = np.asarray(center_idx, dtype=np.int64)
        neg = np.asarray(neg_idx, dtype=np.int64)
        pos = np.asarray(pos_idx, dtype=np.int64)
        across = _landmark_knn_pair_dissimilarity(members, neg, pos, k_use, use_dice, batch_size)
        within_neg = _landmark_knn_pair_dissimilarity(members, center, neg, k_use, use_dice, batch_size)
        within_pos = _landmark_knn_pair_dissimilarity(members, center, pos, k_use, use_dice, batch_size)
        local_texture = 0.5 * (within_neg + within_pos)
        score = np.maximum(0.0, across - tw * local_texture).astype(np.float32)
        if use_mean:
            edge_values[center] += score
            counts[center] += 1.0
        else:
            edge_values[center] = np.maximum(edge_values[center], score)

    if use_mean:
        ok = counts > 0
        edge_values[ok] /= counts[ok]

    edge_map = np.zeros((h, w), dtype=np.float32)
    rr = flat_valid // w
    cc = flat_valid % w
    edge_map[rr, cc] = edge_values
    return edge_map


def _landmark_knn_soft_signatures(
    sim: np.ndarray,
    soft_topk: int,
    temperature: float,
) -> np.ndarray:
    n_valid, n_landmarks = sim.shape
    topk = min(max(1, int(soft_topk)), n_landmarks)
    temp = max(1e-3, float(temperature))
    probs = np.zeros((n_valid, n_landmarks), dtype=np.float32)
    if topk >= n_landmarks:
        logits = (sim.astype(np.float32, copy=True) / temp)
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=1, keepdims=True) + 1e-12
        return logits.astype(np.float32, copy=False)

    top_idx = np.argpartition(-sim, topk - 1, axis=1)[:, :topk]
    top_vals = np.take_along_axis(sim, top_idx, axis=1).astype(np.float32, copy=False)
    logits = top_vals / temp
    logits -= logits.max(axis=1, keepdims=True)
    np.exp(logits, out=logits)
    logits /= logits.sum(axis=1, keepdims=True) + 1e-12
    rows = np.repeat(np.arange(n_valid), topk)
    probs[rows, top_idx.reshape(-1)] = logits.reshape(-1)
    return probs


def _landmark_knn_soft_patch_contrast_edge(
    signatures: np.ndarray,
    flat_valid: np.ndarray,
    valid_mask: np.ndarray,
    radii: List[int],
    spatial_connectivity: str,
    patch_radius: int,
    texture_weight: float,
    direction_aggregation: str,
    scale_aggregation: str,
    chunk_size: int,
) -> np.ndarray:
    h, w = valid_mask.shape
    rr = flat_valid // w
    cc = flat_valid % w
    n_landmarks = signatures.shape[1]
    valid_float = valid_mask.astype(np.float32)
    patch = max(0, int(patch_radius))
    ksize = (2 * patch + 1, 2 * patch + 1)
    counts = cv2.boxFilter(valid_float, ddepth=-1, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    counts = np.maximum(counts, 1e-6)

    direction_mean = str(direction_aggregation).strip().lower() in {"mean", "avg", "average"}
    scale_mode = str(scale_aggregation).strip().lower()
    tw = max(0.0, float(texture_weight))
    chunk = max(1, int(chunk_size))
    scale_maps: List[np.ndarray] = []

    for radius in radii:
        r = max(1, int(radius))
        directions: List[Tuple[int, int]] = [(r, 0), (0, r)]
        if str(spatial_connectivity).strip().lower() in ("8", "8_connected", "eight"):
            directions.extend([(r, r), (r, -r)])

        dir_values: List[np.ndarray] = []
        for dr, dc in directions:
            center_mask = np.zeros((h, w), dtype=bool)
            row_start = max(dr, -dr)
            row_end = h - max(dr, -dr)
            col_start = max(dc, -dc)
            col_end = w - max(dc, -dc)
            if row_start >= row_end or col_start >= col_end:
                continue
            center_mask[row_start:row_end, col_start:col_end] = True
            center_mask &= valid_mask
            center_mask &= np.roll(valid_mask, shift=(dr, dc), axis=(0, 1))
            center_mask &= np.roll(valid_mask, shift=(-dr, -dc), axis=(0, 1))
            idx = np.flatnonzero(center_mask.ravel())
            if idx.size == 0:
                continue
            cr = idx // w
            cc2 = idx % w
            nr = cr - dr
            nc = cc2 - dc
            pr = cr + dr
            pc = cc2 + dc
            across = np.zeros(idx.size, dtype=np.float32)
            texture = np.zeros(idx.size, dtype=np.float32)

            for start in range(0, n_landmarks, chunk):
                end = min(start + chunk, n_landmarks)
                signature_img = np.zeros((h, w, end - start), dtype=np.float32)
                signature_img[rr, cc, :] = signatures[:, start:end]
                patch_sum = cv2.boxFilter(
                    signature_img,
                    ddepth=-1,
                    ksize=ksize,
                    normalize=False,
                    borderType=cv2.BORDER_CONSTANT,
                )
                if patch_sum.ndim == 2:
                    patch_sum = patch_sum[:, :, None]
                patch_sig = patch_sum / counts[:, :, None]
                center = patch_sig[cr, cc2, :]
                neg = patch_sig[nr, nc, :]
                pos = patch_sig[pr, pc, :]
                across += 0.5 * np.abs(neg - pos).sum(axis=1).astype(np.float32)
                texture += 0.25 * (
                    np.abs(center - neg).sum(axis=1) + np.abs(center - pos).sum(axis=1)
                ).astype(np.float32)

            vals = np.maximum(0.0, across - tw * texture).astype(np.float32)
            dir_map = np.zeros((h, w), dtype=np.float32)
            dir_map[cr, cc2] = vals
            dir_values.append(dir_map)

        if not dir_values:
            continue
        stack = np.stack(dir_values, axis=0)
        scale_maps.append(stack.mean(axis=0) if direction_mean else stack.max(axis=0))

    if not scale_maps:
        return np.zeros((h, w), dtype=np.float32)
    stack = np.stack(scale_maps, axis=0)
    if scale_mode in {"geo", "geomean", "geometric_mean"}:
        edge_map = np.exp(np.mean(np.log(np.maximum(stack, 1e-6)), axis=0)).astype(np.float32)
    elif scale_mode in {"min", "minimum"}:
        edge_map = stack.min(axis=0).astype(np.float32)
    elif scale_mode in {"max", "maximum"}:
        edge_map = stack.max(axis=0).astype(np.float32)
    else:
        edge_map = stack.mean(axis=0).astype(np.float32)
    edge_map[~valid_mask] = 0.0
    return edge_map


def _landmark_knn_edge(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    k: int | Sequence[int] = 15,
    n_landmarks: int = 1000,
    spatial_resolutions: List[str] | None = None,
    spatial_radii: List[int] | None = None,
    spatial_connectivity: str = "8",
    random_state: int = 42,
    aggregation: str = "max",
    percentile: float = 90.0,
    spatial_weight: str = "uniform",
    spatial_weight_sigma: float = 2.0,
    landmark_sampling: str = "random",
    similarity: str = "jaccard",
    post_process: str = "none",
    pca_enabled: bool = False,
    pca_n_components: int = 64,
    pca_random_state: int = 42,
    min_edge_percentile: float = 0.0,
    spectral_weight: bool = False,
    hysteresis_low_percentile: float = 70.0,
    hysteresis_high_percentile: float = 95.0,
    score_mode: str = "neighbor",
    contrast_texture_weight: float = 0.75,
    contrast_aggregation: str = "max",
    contrast_batch_size: int = 8192,
    soft_topk: int = 128,
    soft_temperature: float = 0.08,
    soft_patch_radius: int = 1,
    soft_scale_aggregation: str = "mean",
    soft_chunk_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Landmark KNN edge: for each pixel, K nearest neighbors restricted to N landmarks.
    Uses exact cosine similarity (dot product on L2-normalized vectors).
    Set dissimilarity (Jaccard or Dice) between landmark index sets of pixel vs
    spatial neighbors at multiple resolutions; aggregate (max/mean/percentile).
    ``k`` may be an int or a sequence of ints; maps are averaged across k values.
    ``score_mode=contrast`` compares opposite-side signatures and subtracts
    center-to-neighbor texture to favor localized transitions over filled regions.
    ``score_mode=soft_patch_contrast`` uses soft landmark signatures and compares
    opposite patch signatures, which is smoother than hard top-k set overlap.
    """
    t0 = time.perf_counter()
    x = np.asarray(msi[valid_mask], dtype=np.float32)
    if x.size == 0:
        out = np.zeros(valid_mask.shape, dtype=np.float32)
        out[~valid_mask] = 0.0
        return out, out.copy()

    h, w = valid_mask.shape
    n_valid = int(x.shape[0])
    dim = int(x.shape[1])

    if pca_enabled and dim > pca_n_components:
        t_pca = time.perf_counter()
        n_comp = min(pca_n_components, dim, n_valid - 1)
        pca = PCA(n_components=n_comp, random_state=pca_random_state)
        x = pca.fit_transform(x).astype(np.float32)
        dim = x.shape[1]
        logger.info("landmark_knn: PCA %d->%d dims | %.2fs", int(msi.shape[2]), dim, time.perf_counter() - t_pca)
    flat_valid = np.where(valid_mask.ravel())[0].astype(np.int64)
    rc_to_flat = np.full((h, w), -1, dtype=np.int64)
    rc_to_flat[valid_mask] = np.arange(n_valid, dtype=np.int64)
    logger.info(
        "landmark_knn: start | n_valid=%d dim=%d h=%d w=%d | %.2fs",
        n_valid, dim, h, w, time.perf_counter() - t0,
    )

    t1 = time.perf_counter()
    x_norm = x.astype(np.float32)
    norms = np.linalg.norm(x_norm, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    x_norm = (x_norm / norms).astype(np.float32)
    logger.info("landmark_knn: normalize | %.2fs", time.perf_counter() - t1)

    tic = None
    if str(landmark_sampling).strip().lower() == "tic":
        tic = np.asarray(msi[valid_mask].sum(axis=1), dtype=np.float64)

    t2 = time.perf_counter()
    n_landmarks = min(n_landmarks, n_valid)
    landmark_idx = _get_landmark_indices(
        x_norm,
        n_landmarks,
        landmark_sampling,
        random_state,
        tic,
        flat_valid=flat_valid if str(landmark_sampling).strip().lower() == "spatial" else None,
        img_width=w if str(landmark_sampling).strip().lower() == "spatial" else 0,
    )
    if landmark_idx.size < n_landmarks:
        n_landmarks = landmark_idx.size
    landmark_vectors = x_norm[landmark_idx]
    logger.info(
        "landmark_knn: sample landmarks n=%d sampling=%s | %.2fs",
        n_landmarks, landmark_sampling, time.perf_counter() - t2,
    )

    t3 = time.perf_counter()
    sim = x_norm @ landmark_vectors.T
    ks = _normalize_landmark_k_list(k)
    logger.info(
        "landmark_knn: matrix multiply done | k values=%s | %.2fs",
        ks,
        time.perf_counter() - t3,
    )

    def _get_offsets_for_radius(radius: int, connectivity: str) -> List[Tuple[int, int]]:
        r = max(1, int(radius))
        if str(connectivity).strip().lower() in ("8", "8_connected", "eight"):
            return [(-r, -r), (-r, 0), (-r, r), (0, -r), (0, r), (r, -r), (r, 0), (r, r)]
        return [(-r, 0), (r, 0), (0, -r), (0, r)]

    offset_tuples: List[Tuple[int, int, float]] = []
    if spatial_radii:
        for radius in spatial_radii:
            r = max(1, int(radius))
            for dr, dc in _get_offsets_for_radius(r, spatial_connectivity):
                offset_tuples.append((dr, dc, float(r)))
    else:
        for res in (spatial_resolutions or ["4", "8"]):
            res = str(res).strip().lower()
            for dr, dc in _get_offsets_for_radius(1, res):
                offset_tuples.append((dr, dc, 1.0))

    n_offsets = len(offset_tuples)
    offset_dr = np.array([t[0] for t in offset_tuples], dtype=np.int32)
    offset_dc = np.array([t[1] for t in offset_tuples], dtype=np.int32)
    radii = np.array([t[2] for t in offset_tuples], dtype=np.float32)

    sw = str(spatial_weight).strip().lower()
    if sw in ("inv", "inverse", "1/r"):
        offset_weight = np.where(radii > 1e-12, 1.0 / radii, 1.0).astype(np.float32)
    elif sw in ("gaussian", "gauss"):
        sigma = max(0.1, float(spatial_weight_sigma))
        offset_weight = np.exp(-0.5 * (radii / sigma) ** 2).astype(np.float32)
    else:
        offset_weight = np.ones(n_offsets, dtype=np.float32)

    agg = str(aggregation).strip().lower()
    agg_mode = 0 if agg in ("max", "maximum") else (1 if agg in ("mean", "avg") else 2)
    use_dice = str(similarity).strip().lower() in ("dice", "sorensen")
    mode = str(score_mode).strip().lower()
    use_contrast = mode in {"contrast", "directional_contrast", "opposite"}
    use_soft_patch = mode in {"soft_patch_contrast", "soft_contrast", "soft_patch"}
    contrast_dirs: List[Tuple[int, int]] = []
    if use_contrast:
        radii_for_contrast = spatial_radii if spatial_radii else [1]
        for radius in radii_for_contrast:
            r = max(1, int(radius))
            contrast_dirs.extend([(r, 0), (0, r)])
            if str(spatial_connectivity).strip().lower() in ("8", "8_connected", "eight"):
                contrast_dirs.extend([(r, r), (r, -r)])

    dissim_buffer = np.zeros((n_valid, n_offsets), dtype=np.float32)

    if use_soft_patch:
        t_soft = time.perf_counter()
        signatures = _landmark_knn_soft_signatures(sim, int(soft_topk), float(soft_temperature))
        radii_for_soft = [int(r) for r in spatial_radii] if spatial_radii else [1, 2]
        edge_map = _landmark_knn_soft_patch_contrast_edge(
            signatures,
            flat_valid,
            valid_mask,
            radii_for_soft,
            spatial_connectivity,
            int(soft_patch_radius),
            float(contrast_texture_weight),
            str(contrast_aggregation),
            str(soft_scale_aggregation),
            int(soft_chunk_size),
        )
        logger.info(
            "landmark_knn: soft patch contrast topk=%d temp=%.4f radii=%s patch=%d | %.2fs",
            min(max(1, int(soft_topk)), n_landmarks),
            float(soft_temperature),
            radii_for_soft,
            int(soft_patch_radius),
            time.perf_counter() - t_soft,
        )
    else:
        edge_acc = np.zeros((h, w), dtype=np.float64)
        use_spec = bool(spectral_weight)
        x_for_spec = x_norm if use_spec else np.zeros((1, 1), dtype=np.float32)
        for k_try in ks:
            k_use = min(int(k_try), n_landmarks)
            k_use = max(1, k_use)
            kth = n_landmarks - k_use
            t_part = time.perf_counter()
            topk_idx = np.argpartition(-sim, kth, axis=1)[:, kth:]
            knn_indices = topk_idx.astype(np.int32)
            if use_contrast:
                edge_map_k = _landmark_knn_directional_contrast_edge(
                    knn_indices,
                    flat_valid,
                    rc_to_flat,
                    contrast_dirs,
                    n_landmarks,
                    use_dice,
                    float(contrast_texture_weight),
                    str(contrast_aggregation),
                    int(contrast_batch_size),
                )
            else:
                edge_map_k = np.zeros((h, w), dtype=np.float32)
            if (not use_contrast) and _NUMBA_AVAILABLE:
                _jaccard_dice_loop_numba_impl(
                    knn_indices,
                    flat_valid,
                    rc_to_flat,
                    offset_dr,
                    offset_dc,
                    offset_weight,
                    n_landmarks,
                    use_dice,
                    agg_mode,
                    float(percentile),
                    edge_map_k,
                    dissim_buffer,
                    x_for_spec,
                    use_spec,
                )
            elif not use_contrast:
                _jaccard_dice_loop_python(
                    knn_indices,
                    flat_valid,
                    rc_to_flat,
                    offset_dr,
                    offset_dc,
                    offset_weight,
                    n_landmarks,
                    use_dice,
                    agg_mode,
                    float(percentile),
                    edge_map_k,
                    dissim_buffer,
                    x_norm if use_spec else None,
                    use_spec,
                )
            edge_acc += edge_map_k.astype(np.float64)
            logger.info(
                "landmark_knn: %s score k=%d | %.2fs",
                "contrast" if use_contrast else "neighbor",
                k_use,
                time.perf_counter() - t_part,
            )
        edge_map = (edge_acc / float(len(ks))).astype(np.float32)

    pp = str(post_process).strip().lower()
    if pp in ("thin", "thinning", "morphological"):
        try:
            from skimage.morphology import skeletonize
            p50 = float(np.percentile(edge_map[valid_mask], 50)) if np.any(valid_mask) else 0.0
            binary = (edge_map > p50).astype(np.uint8)
            skel = skeletonize(binary)
            edge_map = np.where(skel, edge_map, 0.0).astype(np.float32)
        except Exception as e:
            logger.warning("landmark_knn: skeletonize failed: %s", e)
    elif pp in ("nms", "non_max_suppression"):
        try:
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(edge_map, size=3, mode="constant", cval=0)
            is_max = (edge_map >= local_max - 1e-9) & (edge_map > 0)
            edge_map = np.where(is_max & valid_mask, edge_map, 0.0).astype(np.float32)
        except Exception as e:
            logger.warning("landmark_knn: NMS failed: %s", e)
    elif pp in ("hysteresis", "canny_hysteresis"):
        try:
            from scipy.ndimage import binary_dilation
            vals = edge_map[valid_mask]
            if vals.size > 0:
                lo = float(np.percentile(vals, hysteresis_low_percentile))
                hi = float(np.percentile(vals, hysteresis_high_percentile))
                strong = (edge_map >= hi) & valid_mask
                weak = (edge_map >= lo) & (edge_map < hi) & valid_mask
                # Propagate: weak pixels adjacent to strong are kept; repeat until stable
                for _ in range(5):
                    grown = binary_dilation(strong) & valid_mask
                    added = grown & weak
                    if not np.any(added):
                        break
                    strong = strong | added
                edge_map = np.where(strong, edge_map, 0.0).astype(np.float32)
        except Exception as e:
            logger.warning("landmark_knn: hysteresis failed: %s", e)
    elif pp in ("open", "opening", "cleanup"):
        try:
            from skimage.morphology import opening, disk
            binary = (edge_map > np.percentile(edge_map[valid_mask], 50)) if np.any(valid_mask) else (edge_map > 0)
            opened = opening(binary.astype(np.uint8), disk(1))
            edge_map = np.where(opened, edge_map, 0.0).astype(np.float32)
        except Exception as e:
            logger.warning("landmark_knn: opening failed: %s", e)

    if min_edge_percentile > 0 and np.any(valid_mask):
        thresh = float(np.percentile(edge_map[valid_mask], min_edge_percentile))
        edge_map = np.where(edge_map >= thresh, edge_map, 0.0).astype(np.float32)

    edge_map[~valid_mask] = 0.0
    res_info = f"radii={spatial_radii}" if spatial_radii else f"resolutions={spatial_resolutions}"
    logger.info(
        "landmark_knn: done | n_landmarks=%d k=%s agg=%s sim=%s %s | total %.2fs",
        n_landmarks, ks, aggregation, similarity, res_info, time.perf_counter() - t0,
    )
    return edge_map, edge_map.copy()


def _parse_edge_detection_cfg(cfg: DictConfig) -> dict:
    edge_cfg = getattr(cfg, "edge_detection", None)
    canny_cfg = getattr(edge_cfg, "canny", None) if edge_cfg is not None else None
    gmm_cfg = getattr(edge_cfg, "gmm", None) if edge_cfg is not None else None
    out = {
        "hd_method": str(getattr(edge_cfg, "hd_method", "gradient")).strip().lower() if edge_cfg is not None else "gradient",
        "rgb_method": str(getattr(edge_cfg, "rgb_method", "gradient")).strip().lower() if edge_cfg is not None else "gradient",
        "rgb_color_space": str(getattr(edge_cfg, "rgb_color_space", "gray")).strip().lower() if edge_cfg is not None else "gray",
        "canny_low_threshold": float(getattr(canny_cfg, "low_threshold", 40.0)) if canny_cfg is not None else 40.0,
        "canny_high_threshold": float(getattr(canny_cfg, "high_threshold", 120.0)) if canny_cfg is not None else 120.0,
        "canny_aperture_size": int(getattr(canny_cfg, "aperture_size", 3)) if canny_cfg is not None else 3,
        "canny_l2gradient": bool(getattr(canny_cfg, "l2gradient", True)) if canny_cfg is not None else True,
        "canny_n_jobs": int(getattr(canny_cfg, "n_jobs", 0)) if canny_cfg is not None else 0,
        "canny_rgb_mode": str(getattr(canny_cfg, "rgb_mode", "max_channel")).strip().lower() if canny_cfg is not None else "max_channel",
        "gmm_n_components": int(getattr(gmm_cfg, "n_components", 12)) if gmm_cfg is not None else 12,
        "gmm_covariance_type": str(getattr(gmm_cfg, "covariance_type", "diag")).strip().lower() if gmm_cfg is not None else "diag",
        "gmm_reg_covar": float(getattr(gmm_cfg, "reg_covar", 1e-6)) if gmm_cfg is not None else 1e-6,
        "gmm_max_iter": int(getattr(gmm_cfg, "max_iter", 200)) if gmm_cfg is not None else 200,
        "gmm_random_state": int(getattr(gmm_cfg, "random_state", 42)) if gmm_cfg is not None else 42,
        "gmm_max_samples": int(getattr(gmm_cfg, "max_samples", 200000)) if gmm_cfg is not None else 200000,
        "gmm_pca_enabled": bool(getattr(gmm_cfg, "pca_enabled", False)) if gmm_cfg is not None else False,
        "gmm_pca_n_components": int(getattr(gmm_cfg, "pca_n_components", 64)) if gmm_cfg is not None else 64,
        "gmm_pca_random_state": int(getattr(gmm_cfg, "pca_random_state", 42)) if gmm_cfg is not None else 42,
    }
    lk_cfg = None
    if edge_cfg is not None:
        lk_cfg = getattr(edge_cfg, "soft_landmark_contrast", None)
        if lk_cfg is None:
            lk_cfg = getattr(edge_cfg, "landmark_knn", None)
    raw_lk_k = getattr(lk_cfg, "k", 15) if lk_cfg is not None else 15
    lk_ks = _normalize_landmark_k_list(raw_lk_k)
    out["lk_k_list"] = lk_ks
    out["lk_k"] = int(lk_ks[0])
    out["lk_n_landmarks"] = int(getattr(lk_cfg, "n_landmarks", 1000)) if lk_cfg is not None else 1000
    out["lk_spatial_resolutions"] = list(getattr(lk_cfg, "spatial_resolutions", ["4", "8"])) if lk_cfg is not None else ["4", "8"]
    out["lk_spatial_radii"] = list(getattr(lk_cfg, "spatial_radii", [])) if lk_cfg is not None else []
    out["lk_spatial_connectivity"] = str(getattr(lk_cfg, "spatial_connectivity", "8")).strip().lower() if lk_cfg is not None else "8"
    out["lk_random_state"] = int(getattr(lk_cfg, "random_state", 42)) if lk_cfg is not None else 42
    out["lk_aggregation"] = str(getattr(lk_cfg, "aggregation", "max")).strip().lower() if lk_cfg is not None else "max"
    out["lk_percentile"] = float(getattr(lk_cfg, "percentile", 90.0)) if lk_cfg is not None else 90.0
    out["lk_spatial_weight"] = str(getattr(lk_cfg, "spatial_weight", "uniform")).strip().lower() if lk_cfg is not None else "uniform"
    out["lk_spatial_weight_sigma"] = float(getattr(lk_cfg, "spatial_weight_sigma", 2.0)) if lk_cfg is not None else 2.0
    out["lk_landmark_sampling"] = str(getattr(lk_cfg, "landmark_sampling", "random")).strip().lower() if lk_cfg is not None else "random"
    out["lk_similarity"] = str(getattr(lk_cfg, "similarity", "jaccard")).strip().lower() if lk_cfg is not None else "jaccard"
    out["lk_post_process"] = str(getattr(lk_cfg, "post_process", "none")).strip().lower() if lk_cfg is not None else "none"
    out["lk_pca_enabled"] = bool(getattr(lk_cfg, "pca_enabled", False)) if lk_cfg is not None else False
    out["lk_pca_n_components"] = int(getattr(lk_cfg, "pca_n_components", 64)) if lk_cfg is not None else 64
    out["lk_pca_random_state"] = int(getattr(lk_cfg, "pca_random_state", 42)) if lk_cfg is not None else 42
    out["lk_min_edge_percentile"] = float(getattr(lk_cfg, "min_edge_percentile", 0.0)) if lk_cfg is not None else 0.0
    out["lk_spectral_weight"] = bool(getattr(lk_cfg, "spectral_weight", False)) if lk_cfg is not None else False
    out["lk_hysteresis_low_percentile"] = float(getattr(lk_cfg, "hysteresis_low_percentile", 70.0)) if lk_cfg is not None else 70.0
    out["lk_hysteresis_high_percentile"] = float(getattr(lk_cfg, "hysteresis_high_percentile", 95.0)) if lk_cfg is not None else 95.0
    out["lk_score_mode"] = str(getattr(lk_cfg, "score_mode", "neighbor")).strip().lower() if lk_cfg is not None else "neighbor"
    out["lk_contrast_texture_weight"] = float(getattr(lk_cfg, "contrast_texture_weight", 0.75)) if lk_cfg is not None else 0.75
    out["lk_contrast_aggregation"] = str(getattr(lk_cfg, "contrast_aggregation", "max")).strip().lower() if lk_cfg is not None else "max"
    out["lk_contrast_batch_size"] = int(getattr(lk_cfg, "contrast_batch_size", 8192)) if lk_cfg is not None else 8192
    out["lk_soft_topk"] = int(getattr(lk_cfg, "soft_topk", 128)) if lk_cfg is not None else 128
    out["lk_soft_temperature"] = float(getattr(lk_cfg, "soft_temperature", 0.08)) if lk_cfg is not None else 0.08
    out["lk_soft_patch_radius"] = int(getattr(lk_cfg, "soft_patch_radius", 1)) if lk_cfg is not None else 1
    out["lk_soft_scale_aggregation"] = str(getattr(lk_cfg, "soft_scale_aggregation", "mean")).strip().lower() if lk_cfg is not None else "mean"
    out["lk_soft_chunk_size"] = int(getattr(lk_cfg, "soft_chunk_size", 64)) if lk_cfg is not None else 64
    if out["hd_method"] not in {
        "gradient",
        "canny",
        "gmm_kl",
        "distance_multi",
        "landmark_knn",
        "soft_landmark_contrast",
        "ensemble_cluster",
        "nmf_component_edges",
        "top3_nmf_max",
        "top3_nmf_mean",
    }:
        out["hd_method"] = "gradient"
    if out["rgb_method"] not in {"gradient", "canny"}:
        out["rgb_method"] = "gradient"
    if out["rgb_color_space"] not in {"gray", "rgb", "lab"}:
        out["rgb_color_space"] = "gray"
    if out["canny_aperture_size"] not in {3, 5, 7}:
        out["canny_aperture_size"] = 3
    if out["canny_rgb_mode"] not in {"gray", "max_channel", "mean_channel", "any_channel"}:
        out["canny_rgb_mode"] = "max_channel"
    if out["gmm_covariance_type"] not in {"full", "tied", "diag", "spherical"}:
        out["gmm_covariance_type"] = "diag"
    out["gmm_n_components"] = max(2, int(out["gmm_n_components"]))
    out["gmm_max_iter"] = max(10, int(out["gmm_max_iter"]))
    out["gmm_max_samples"] = max(0, int(out["gmm_max_samples"]))
    out["gmm_pca_n_components"] = max(1, int(out["gmm_pca_n_components"]))
    return out


def _gmm_kl_apply_optional_pca(
    x: np.ndarray,
    fit_x: np.ndarray,
    edge_method_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optionally project spectra with PCA before GMM (fit on ``fit_x``, transform both ``x`` and ``fit_x``).
    """
    if not bool(edge_method_cfg.get("gmm_pca_enabled", False)):
        return x, fit_x
    if fit_x.shape[0] < 2 or x.shape[1] < 1:
        logger.warning(
            "gmm_kl: PCA skipped (need >=2 fit rows and >=1 channel); using raw spectra",
        )
        return x, fit_x
    pca_dim = int(edge_method_cfg.get("gmm_pca_n_components", 64))
    rs = int(edge_method_cfg.get("gmm_pca_random_state", 42))
    n_chan = int(x.shape[1])
    n_fit = int(fit_x.shape[0])
    n_comp = min(pca_dim, max(1, n_fit - 1), n_chan)
    n_comp = max(1, n_comp)
    pca = PCA(n_components=n_comp, random_state=rs)
    pca.fit(fit_x.astype(np.float64, copy=False))
    x_t = pca.transform(x.astype(np.float64, copy=False)).astype(np.float32)
    fit_t = pca.transform(fit_x.astype(np.float64, copy=False)).astype(np.float32)
    logger.info(
        "gmm_kl: PCA | n_components=%d (requested %d) n_chan=%d n_fit=%d",
        n_comp,
        pca_dim,
        n_chan,
        n_fit,
    )
    return x_t, fit_t


def _normalize_map_percentile(arr: np.ndarray, mask: np.ndarray, low_pct: float, high_pct: float, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[mask]
    if vals.size == 0:
        return out
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    if hi <= lo + eps:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmax > vmin + 1e-12:
            lo, hi = vmin, vmax
        else:
            return out
    out = (x - lo) / (hi - lo + eps)
    out = np.clip(out, 0.0, 1.0)
    out[~mask] = 0.0
    return out.astype(np.float32, copy=False)


def _hd_edge_power_from_cfg(hd_cfg: object) -> float:
    """``hd_edges.edge_power`` (default 1.0). Values <= 0 are ignored (treated as 1.0)."""
    if hd_cfg is None:
        return 1.0
    try:
        p = float(getattr(hd_cfg, "edge_power", 1.0))
    except (TypeError, ValueError):
        return 1.0
    if p <= 0.0:
        return 1.0
    return p


def _apply_hd_edge_power(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    hd_cfg: object,
) -> np.ndarray:
    """
    ``out = clip(x,0)**gamma`` on valid pixels before percentile normalization (``hd_edges.edge_power``).
    ``gamma=1`` is identity; ``gamma>1`` (e.g. 2) suppresses weak responses relative to strong ones.
    """
    gamma = _hd_edge_power_from_cfg(hd_cfg)
    if abs(gamma - 1.0) <= 1e-9:
        return arr
    vm = np.asarray(valid_mask, dtype=bool)
    x = np.asarray(arr, dtype=np.float32).copy()
    if not np.any(vm):
        return x
    x[vm] = np.power(np.clip(x[vm], 0.0, None), gamma).astype(np.float32)
    x[~vm] = 0.0
    return x


def _edge_maps_multiscale(
    kind: str,
    data: np.ndarray,
    valid_mask: np.ndarray,
    sigmas: List[float],
    aggregation: str,
    hd_cfg: object = None,
    edge_method_cfg: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    edge_method_cfg = edge_method_cfg or {}
    edge_maps = []
    edge_linf_maps = []
    individual_distance_maps: dict[str, np.ndarray] | None = None
    if kind == "hd":
        method = str(edge_method_cfg.get("hd_method", "gradient")).strip().lower()
        if method == "gmm_kl":
            t_gmm0 = time.perf_counter()
            x = np.asarray(data[valid_mask], dtype=np.float32)
            if x.size == 0:
                logger.warning("gmm_kl: no valid pixels, returning empty edge maps")
                base_edge = np.zeros(valid_mask.shape, dtype=np.float32)
                base_edge_linf = np.zeros(valid_mask.shape, dtype=np.float32)
            else:
                # Optional subsampling for fitting speed on very large images.
                max_samples = int(edge_method_cfg.get("gmm_max_samples", 0))
                fit_x = x
                logger.info(
                    "gmm_kl: start | valid_pixels=%d dims=%d max_samples=%d",
                    int(x.shape[0]),
                    int(x.shape[1]),
                    int(max_samples),
                )
                if 0 < max_samples < x.shape[0]:
                    rng = np.random.default_rng(int(edge_method_cfg.get("gmm_random_state", 42)))
                    idx = rng.choice(x.shape[0], size=max_samples, replace=False)
                    fit_x = x[idx]
                    logger.info("gmm_kl: subsampled fit set to %d pixels", int(fit_x.shape[0]))
                x, fit_x = _gmm_kl_apply_optional_pca(x, fit_x, edge_method_cfg)
                n_components = min(int(edge_method_cfg.get("gmm_n_components", 12)), fit_x.shape[0])
                gmm = GaussianMixture(
                    n_components=max(2, n_components),
                    covariance_type=str(edge_method_cfg.get("gmm_covariance_type", "diag")),
                    reg_covar=float(edge_method_cfg.get("gmm_reg_covar", 1e-6)),
                    max_iter=int(edge_method_cfg.get("gmm_max_iter", 200)),
                    random_state=int(edge_method_cfg.get("gmm_random_state", 42)),
                )
                logger.info(
                    "gmm_kl: fitting GMM | n_components=%d covariance_type=%s reg_covar=%g max_iter=%d",
                    int(gmm.n_components),
                    str(gmm.covariance_type),
                    float(gmm.reg_covar),
                    int(gmm.max_iter),
                )
                t_fit0 = time.perf_counter()
                gmm.fit(fit_x)
                logger.info(
                    "gmm_kl: fit complete | converged=%s n_iter=%d lower_bound=%.6f fit_time=%.2fs",
                    str(bool(getattr(gmm, "converged_", False))),
                    int(getattr(gmm, "n_iter_", -1)),
                    float(getattr(gmm, "lower_bound_", float("nan"))),
                    float(time.perf_counter() - t_fit0),
                )
                t_pred0 = time.perf_counter()
                probs = gmm.predict_proba(x).astype(np.float32, copy=False)
                logger.info(
                    "gmm_kl: predict_proba complete | shape=%s time=%.2fs",
                    tuple(probs.shape),
                    float(time.perf_counter() - t_pred0),
                )
                eps = 1e-8
                p_map = np.zeros((valid_mask.shape[0], valid_mask.shape[1], probs.shape[1]), dtype=np.float32)
                p_map[valid_mask] = probs

                # Symmetric KL between neighboring assignment distributions.
                # D_sym(p,q) = 0.5*(KL(p||q)+KL(q||p)).
                t_kl0 = time.perf_counter()
                def _sym_kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                    aa = np.clip(a, eps, 1.0)
                    bb = np.clip(b, eps, 1.0)
                    return 0.5 * (
                        np.sum(aa * (np.log(aa) - np.log(bb)), axis=-1)
                        + np.sum(bb * (np.log(bb) - np.log(aa)), axis=-1)
                    )

                h, w, _ = p_map.shape
                sum_edge = np.zeros((h, w), dtype=np.float32)
                max_edge = np.zeros((h, w), dtype=np.float32)

                # Vertical neighbors (r, c) <-> (r+1, c)
                m_v = valid_mask[:-1, :] & valid_mask[1:, :]
                if np.any(m_v):
                    d_v = _sym_kl(p_map[:-1, :, :], p_map[1:, :, :]).astype(np.float32, copy=False)
                    d_v = np.where(m_v, d_v, 0.0)
                    sum_edge[:-1, :] += d_v
                    sum_edge[1:, :] += d_v
                    max_edge[:-1, :] = np.maximum(max_edge[:-1, :], d_v)
                    max_edge[1:, :] = np.maximum(max_edge[1:, :], d_v)

                # Horizontal neighbors (r, c) <-> (r, c+1)
                m_h = valid_mask[:, :-1] & valid_mask[:, 1:]
                if np.any(m_h):
                    d_h = _sym_kl(p_map[:, :-1, :], p_map[:, 1:, :]).astype(np.float32, copy=False)
                    d_h = np.where(m_h, d_h, 0.0)
                    sum_edge[:, :-1] += d_h
                    sum_edge[:, 1:] += d_h
                    max_edge[:, :-1] = np.maximum(max_edge[:, :-1], d_h)
                    max_edge[:, 1:] = np.maximum(max_edge[:, 1:], d_h)

                base_edge = sum_edge.astype(np.float32, copy=False)
                base_edge_linf = max_edge.astype(np.float32, copy=False)
                base_edge[~valid_mask] = 0.0
                base_edge_linf[~valid_mask] = 0.0
                logger.info(
                    "gmm_kl: KL edge maps complete | kl_time=%.2fs total_time=%.2fs",
                    float(time.perf_counter() - t_kl0),
                    float(time.perf_counter() - t_gmm0),
                )

            for s in sigmas:
                sigma = float(s)
                if sigma <= 0:
                    edge_maps.append(base_edge)
                    edge_linf_maps.append(base_edge_linf)
                else:
                    edge_maps.append(_gaussian_blur(base_edge, sigma))
                    edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
            logger.info(
                "gmm_kl: multi-scale aggregation prepared | n_scales=%d",
                int(len(sigmas)),
            )
        elif method in {"landmark_knn", "soft_landmark_contrast"}:
            logger.info("%s: entering (data shape=%s)", method, data.shape)
            lk_radii = list(edge_method_cfg.get("lk_spatial_radii", []))
            lk_ks = edge_method_cfg.get("lk_k_list")
            if lk_ks is None:
                lk_ks = _normalize_landmark_k_list(edge_method_cfg.get("lk_k", 15))
            base_edge, base_edge_linf = _landmark_knn_edge(
                data,
                valid_mask,
                k=lk_ks,
                n_landmarks=int(edge_method_cfg.get("lk_n_landmarks", 1000)),
                spatial_resolutions=list(edge_method_cfg.get("lk_spatial_resolutions", ["4", "8"])),
                spatial_radii=[int(r) for r in lk_radii] if lk_radii else None,
                spatial_connectivity=str(edge_method_cfg.get("lk_spatial_connectivity", "8")).strip().lower(),
                random_state=int(edge_method_cfg.get("lk_random_state", 42)),
                aggregation=str(edge_method_cfg.get("lk_aggregation", "max")).strip().lower(),
                percentile=float(edge_method_cfg.get("lk_percentile", 90.0)),
                spatial_weight=str(edge_method_cfg.get("lk_spatial_weight", "uniform")).strip().lower(),
                spatial_weight_sigma=float(edge_method_cfg.get("lk_spatial_weight_sigma", 2.0)),
                landmark_sampling=str(edge_method_cfg.get("lk_landmark_sampling", "random")).strip().lower(),
                similarity=str(edge_method_cfg.get("lk_similarity", "jaccard")).strip().lower(),
                post_process=str(edge_method_cfg.get("lk_post_process", "none")).strip().lower(),
                pca_enabled=bool(edge_method_cfg.get("lk_pca_enabled", False)),
                pca_n_components=int(edge_method_cfg.get("lk_pca_n_components", 64)),
                pca_random_state=int(edge_method_cfg.get("lk_pca_random_state", 42)),
                min_edge_percentile=float(edge_method_cfg.get("lk_min_edge_percentile", 0.0)),
                spectral_weight=bool(edge_method_cfg.get("lk_spectral_weight", False)),
                hysteresis_low_percentile=float(edge_method_cfg.get("lk_hysteresis_low_percentile", 70.0)),
                hysteresis_high_percentile=float(edge_method_cfg.get("lk_hysteresis_high_percentile", 95.0)),
                score_mode=str(edge_method_cfg.get("lk_score_mode", "neighbor")).strip().lower(),
                contrast_texture_weight=float(edge_method_cfg.get("lk_contrast_texture_weight", 0.75)),
                contrast_aggregation=str(edge_method_cfg.get("lk_contrast_aggregation", "max")).strip().lower(),
                contrast_batch_size=int(edge_method_cfg.get("lk_contrast_batch_size", 8192)),
                soft_topk=int(edge_method_cfg.get("lk_soft_topk", 128)),
                soft_temperature=float(edge_method_cfg.get("lk_soft_temperature", 0.08)),
                soft_patch_radius=int(edge_method_cfg.get("lk_soft_patch_radius", 1)),
                soft_scale_aggregation=str(edge_method_cfg.get("lk_soft_scale_aggregation", "mean")).strip().lower(),
                soft_chunk_size=int(edge_method_cfg.get("lk_soft_chunk_size", 64)),
            )
            for s in sigmas:
                sigma = float(s)
                if sigma <= 0:
                    edge_maps.append(base_edge)
                    edge_linf_maps.append(base_edge_linf)
                else:
                    edge_maps.append(_gaussian_blur(base_edge, sigma))
                    edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
        elif method == "distance_multi":
            base_edge, base_edge_linf, individual_distance_maps = _highd_edge_maps_distance_multi(data, valid_mask, hd_cfg)
            for s in sigmas:
                sigma = float(s)
                if sigma <= 0:
                    edge_maps.append(base_edge)
                    edge_linf_maps.append(base_edge_linf)
                else:
                    edge_maps.append(_gaussian_blur(base_edge, sigma))
                    edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
        elif method == "ensemble_cluster":
            base_edge, base_edge_linf = _highd_edge_maps_ensemble_cluster(data, valid_mask, hd_cfg)
            for s in sigmas:
                sigma = float(s)
                if sigma <= 0:
                    edge_maps.append(base_edge)
                    edge_linf_maps.append(base_edge_linf)
                else:
                    edge_maps.append(_gaussian_blur(base_edge, sigma))
                    edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
        elif method == "nmf_component_edges":
            base_edge, base_edge_linf = _highd_edge_maps_nmf_component_edges(data, valid_mask, hd_cfg)
            for s in sigmas:
                sigma = float(s)
                if sigma <= 0:
                    edge_maps.append(base_edge)
                    edge_linf_maps.append(base_edge_linf)
                else:
                    edge_maps.append(_gaussian_blur(base_edge, sigma))
                    edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
        elif method == "canny":
            normalize_channels = bool(getattr(hd_cfg, "normalize_channels", False)) if hd_cfg is not None else False
            channel_low = float(getattr(hd_cfg, "channel_low_percentile", 1.0)) if hd_cfg is not None else 1.0
            channel_high = float(getattr(hd_cfg, "channel_high_percentile", 99.0)) if hd_cfg is not None else 99.0
            use_log1p = bool(getattr(hd_cfg, "log1p", False)) if hd_cfg is not None else False
            channel_agg = str(getattr(hd_cfg, "aggregation", "mean")).strip().lower() if hd_cfg is not None else "mean"
            topk = int(getattr(hd_cfg, "topk", 5)) if hd_cfg is not None else 5
            msi_proc = np.asarray(data, dtype=np.float32)
            channels = np.zeros_like(msi_proc, dtype=np.float32)
            for i in range(msi_proc.shape[2]):
                ch = np.asarray(msi_proc[:, :, i], dtype=np.float32)
                if use_log1p:
                    ch = np.log1p(np.clip(ch, 0.0, None)).astype(np.float32, copy=False)
                if normalize_channels:
                    ch = _normalize_channel_percentile(ch, valid_mask, channel_low, channel_high)
                else:
                    # Keep Canny robust when upstream normalization (e.g. TIC) yields tiny ranges.
                    ch = _normalize_on_mask(ch, valid_mask)
                channels[:, :, i] = ch
            for s in sigmas:
                canny_stack = _canny_stack_from_channels(
                    channels01=channels,
                    sigma=float(s),
                    low_threshold=float(edge_method_cfg.get("canny_low_threshold", 40.0)),
                    high_threshold=float(edge_method_cfg.get("canny_high_threshold", 120.0)),
                    aperture_size=int(edge_method_cfg.get("canny_aperture_size", 3)),
                    l2gradient=bool(edge_method_cfg.get("canny_l2gradient", True)),
                    n_jobs=int(edge_method_cfg.get("canny_n_jobs", 0)),
                )
                edge_maps.append(_aggregate_channel_stack(canny_stack, mode=channel_agg, topk=topk))
                edge_linf_maps.append(np.max(canny_stack, axis=2).astype(np.float32, copy=False))
        else:
            base_edge, base_edge_linf = _highd_edge_maps_base(data, valid_mask, hd_cfg=hd_cfg)
            for s in sigmas:
                sigma = float(s)
                if sigma <= 0:
                    edge_maps.append(base_edge)
                    edge_linf_maps.append(base_edge_linf)
                else:
                    edge_maps.append(_gaussian_blur(base_edge, sigma))
                    edge_linf_maps.append(_gaussian_blur(base_edge_linf, sigma))
    else:
        method = str(edge_method_cfg.get("rgb_method", "gradient")).strip().lower()
        rgb_scale_01 = bool(edge_method_cfg.get("rgb_scale_01", False))
        rgb = np.asarray(data, dtype=np.float32)
        if method == "canny":
            rgb_mode = str(edge_method_cfg.get("canny_rgb_mode", "max_channel")).strip().lower()
            for s in sigmas:
                rgb_s = _gaussian_blur(rgb, float(s))
                rgb_s01 = rgb_s if rgb_scale_01 else np.clip(rgb_s / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
                if rgb_mode == "gray":
                    if rgb_scale_01:
                        gray = cv2.cvtColor(np.clip(rgb_s, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
                    else:
                        gray = cv2.cvtColor(np.clip(rgb_s, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                    e = _canny_edge_from_u8(
                        _to_u8_from01(gray),
                        low_threshold=float(edge_method_cfg.get("canny_low_threshold", 40.0)),
                        high_threshold=float(edge_method_cfg.get("canny_high_threshold", 120.0)),
                        aperture_size=int(edge_method_cfg.get("canny_aperture_size", 3)),
                        l2gradient=bool(edge_method_cfg.get("canny_l2gradient", True)),
                    )
                    edge_maps.append(e)
                    edge_linf_maps.append(e)
                else:
                    stack = _canny_stack_from_channels(
                        channels01=rgb_s01,
                        sigma=0.0,
                        low_threshold=float(edge_method_cfg.get("canny_low_threshold", 40.0)),
                        high_threshold=float(edge_method_cfg.get("canny_high_threshold", 120.0)),
                        aperture_size=int(edge_method_cfg.get("canny_aperture_size", 3)),
                        l2gradient=bool(edge_method_cfg.get("canny_l2gradient", True)),
                        n_jobs=int(edge_method_cfg.get("canny_n_jobs", 0)),
                    )
                    if rgb_mode == "mean_channel":
                        e = np.mean(stack, axis=2).astype(np.float32, copy=False)
                    elif rgb_mode == "any_channel":
                        e = (np.max(stack, axis=2) > 0).astype(np.float32)
                    else:
                        e = np.max(stack, axis=2).astype(np.float32, copy=False)
                    edge_maps.append(e)
                    edge_linf_maps.append(np.max(stack, axis=2).astype(np.float32, copy=False))
        else:
            for s in sigmas:
                sigma = float(s)
                rgb_cs = str(edge_method_cfg.get("rgb_color_space", "gray")).strip().lower()
                linf_cs = "lab" if rgb_cs == "lab" else "rgb"
                if sigma <= 0:
                    edge_maps.append(_viz_edge_map(data, valid_mask, color_space=rgb_cs, already_01=rgb_scale_01))
                    edge_linf_maps.append(_viz_edge_map_linf(data, valid_mask, color_space=linf_cs, already_01=rgb_scale_01))
                else:
                    rgb_s = _gaussian_blur(rgb, sigma)
                    rgb_for_viz = np.clip(rgb_s, 0, 1).astype(np.float32) if rgb_scale_01 else np.clip(rgb_s, 0, 255).astype(np.uint8)
                    edge_maps.append(_viz_edge_map(rgb_for_viz, valid_mask, color_space=rgb_cs, already_01=rgb_scale_01))
                    edge_linf_maps.append(_viz_edge_map_linf(rgb_for_viz, valid_mask, color_space=linf_cs, already_01=rgb_scale_01))
    rgb_no_normalize = bool(edge_method_cfg.get("rgb_no_normalize", False))
    merge_normalize = not rgb_no_normalize
    merged = _aggregate_multiscale_maps(edge_maps, valid_mask, aggregation, normalize=merge_normalize)
    merged_linf = _aggregate_multiscale_maps(edge_linf_maps, valid_mask, aggregation, normalize=merge_normalize)
    if kind == "hd":
        _gp = _hd_edge_power_from_cfg(hd_cfg)
        if abs(_gp - 1.0) > 1e-9:
            logger.info("hd_edges: edge_power=%g before percentile norm (see hd_edges.edge_power)", _gp)
        merged = _apply_hd_edge_power(merged, valid_mask, hd_cfg)
        merged_linf = _apply_hd_edge_power(merged_linf, valid_mask, hd_cfg)
        if individual_distance_maps is not None:
            individual_distance_maps = {
                k: _apply_hd_edge_power(v, valid_mask, hd_cfg) for k, v in individual_distance_maps.items()
            }
    if individual_distance_maps is not None:
        return (merged, merged_linf, individual_distance_maps)
    return (merged, merged_linf)


def _to_uint8_gray01(arr01: np.ndarray) -> np.ndarray:
    x = np.asarray(arr01, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _gray_u8_to_rgb_colormap(
    gray_u8: np.ndarray,
    cmap_name: str,
    *,
    valid_mask: np.ndarray | None = None,
    background_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Map luminance 0–255 to RGB using a matplotlib colormap name (e.g. magma, viridis).

    If ``valid_mask`` is provided, pixels outside the mask are set to ``background_rgb``
    (default white) so non-tissue areas are not painted with the colormap.
    """
    try:
        from matplotlib import colormaps

        cmap = colormaps[cmap_name]
    except (KeyError, ImportError, AttributeError):
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)
    g = np.clip(gray_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    rgba = cmap(g)
    rgb = (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        if m.shape == rgb.shape[:2]:
            rgb[~m] = np.asarray(background_rgb, dtype=np.uint8)
    return rgb


def _equalize_gray_u8(
    gray_u8: np.ndarray,
    enabled: bool,
    method: str,
    clahe_clip_limit: float,
    clahe_tile_grid_size: int,
) -> np.ndarray:
    if not enabled:
        return gray_u8
    m = str(method).strip().lower()
    if m == "clahe":
        tgs = max(2, int(clahe_tile_grid_size))
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(tgs, tgs))
        return clahe.apply(gray_u8)
    # default to global histogram equalization
    return cv2.equalizeHist(gray_u8)


def _apply_equalize_rgb_uint8(
    rgb_uint8: np.ndarray,
    method: str,
    clip_limit: float,
    tile_grid_size: int,
) -> np.ndarray:
    """Histogram-equalize luminance in RGB→LAB: equalize L, then LAB→RGB. Grayscale / non-RGB unchanged."""
    rgb = np.asarray(rgb_uint8, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return rgb
    m = str(method).strip().lower()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    if m == "clahe":
        tgs = max(2, int(tile_grid_size))
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tgs, tgs))
        lab[:, :, 0] = clahe.apply(l_ch)
    else:
        lab[:, :, 0] = cv2.equalizeHist(l_ch)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.asarray(out, dtype=np.uint8)


def _save_map_png_and_npy(
    out_base: Path,
    arr01: np.ndarray,
    save_npy: bool,
    equalize_png: bool,
    equalize_method: str,
    clahe_clip_limit: float,
    clahe_tile_grid_size: int,
    orig_for_context: np.ndarray | None = None,
    raw_max_scale: float | None = None,
    display_gamma: float | None = None,
    colormap: str | None = None,
) -> None:
    """Save edge map PNG (and optional npy). If orig_for_context is given, save side-by-side (orig | edge).
    raw_max_scale: if set (e.g. 1.414 for Sobel on 0-1 input), scale arr by this before uint8 conversion.
    display_gamma: if set (<1 expands mid-tones, e.g. 0.5), apply gamma before save to reduce binary look.
    colormap: matplotlib colormap name (e.g. magma); None or empty = grayscale L PNG as before."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr01, dtype=np.float32)
    if raw_max_scale is not None and raw_max_scale > 1e-8:
        arr = arr / float(raw_max_scale)
    gray_u8 = _to_uint8_gray01(arr)
    if display_gamma is not None and 0 < display_gamma < 1:
        gray_f = np.clip(gray_u8.astype(np.float32) / 255.0, 0.0, 1.0)
        gray_u8 = (np.power(gray_f, display_gamma) * 255.0).astype(np.uint8)
    gray_u8 = _equalize_gray_u8(
        gray_u8,
        enabled=equalize_png,
        method=equalize_method,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )
    cmap_name = (str(colormap).strip() if colormap is not None else "") or ""
    use_cmap = bool(cmap_name and cmap_name.lower() not in ("none", "null", "~", ""))
    edge_rgb_out: np.ndarray | None
    if use_cmap:
        try:
            edge_rgb_out = _gray_u8_to_rgb_colormap(gray_u8, cmap_name)
        except Exception as e:
            logger.warning("edge colormap %r failed (%s); saving grayscale", cmap_name, e)
            edge_rgb_out = None
    else:
        edge_rgb_out = None

    if orig_for_context is not None:
        orig = np.asarray(orig_for_context)
        if orig.dtype == np.float32 or orig.dtype == np.float64:
            orig_u8 = (np.clip(orig, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            orig_u8 = np.asarray(orig, dtype=np.uint8)
        if orig_u8.ndim == 2:
            orig_rgb = np.stack([orig_u8] * 3, axis=-1)
        else:
            orig_rgb = orig_u8[..., :3]
        if edge_rgb_out is not None:
            edge_panel = edge_rgb_out
        else:
            edge_panel = np.stack([gray_u8] * 3, axis=-1)
        sep = np.full((orig_rgb.shape[0], 2, 3), 255, dtype=np.uint8)
        side_by_side = np.concatenate([orig_rgb, sep, edge_panel], axis=1)
        Image.fromarray(side_by_side, mode="RGB").save(out_base.with_suffix(".png"))
    elif edge_rgb_out is not None:
        Image.fromarray(edge_rgb_out, mode="RGB").save(out_base.with_suffix(".png"))
    else:
        Image.fromarray(gray_u8, mode="L").save(out_base.with_suffix(".png"))
    if save_npy:
        np.save(out_base.with_suffix(".npy"), np.asarray(arr01, dtype=np.float32))


def _save_side_by_side_rgb(
    left_rgb: np.ndarray,
    right_img: np.ndarray,
    out_path: Path,
) -> None:
    """Save side-by-side PNG: [left | separator | right]. left/right as RGB (HxWx3) or gray (HxW)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    left = np.asarray(left_rgb, dtype=np.uint8)
    if left.ndim == 2:
        left = np.stack([left] * 3, axis=-1)
    else:
        left = left[..., :3]
    right = np.asarray(right_img, dtype=np.uint8)
    if right.ndim == 2:
        right = np.stack([right] * 3, axis=-1)
    else:
        right = right[..., :3]
    sep = np.full((left.shape[0], 2, 3), 255, dtype=np.uint8)
    combined = np.concatenate([left, sep, right], axis=1)
    Image.fromarray(combined, mode="RGB").save(out_path)


def _compute_prf1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, int, int, int]:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    fn = int(np.sum(yt & (~yp)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    if (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0
    return precision, recall, f1, tp, fp, fn


def _edges_for_continuous_metrics(
    hd_edge: np.ndarray,
    viz_edge: np.ndarray,
    *,
    square: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare nonnegative edge maps for continuous Dice / soft precision-recall.

    If square is True, use element-wise squares (emphasizes stronger gradients). Maps are clipped
    to nonnegative before squaring.
    """
    h = np.asarray(hd_edge, dtype=np.float64)
    v = np.asarray(viz_edge, dtype=np.float64)
    if not square:
        return h, v
    return np.square(np.maximum(h, 0.0)), np.square(np.maximum(v, 0.0))


def _compute_continuous_dice(hd_edge: np.ndarray, viz_edge: np.ndarray, valid_mask: np.ndarray) -> float:
    """Continuous Dice: 2 * sum(h*v) / (sum(h) + sum(v)) on valid pixels. No binarization."""
    h = np.asarray(hd_edge, dtype=np.float64).ravel()
    v = np.asarray(viz_edge, dtype=np.float64).ravel()
    m = np.asarray(valid_mask, dtype=bool).ravel()
    if m.size != h.size or m.size != v.size:
        return float("nan")
    h, v = h[m], v[m]
    shv = float(np.sum(h * v))
    sh, sv = float(np.sum(h)), float(np.sum(v))
    denom = sh + sv
    if denom < 1e-12:
        return 0.0
    return float(2.0 * shv / denom)


def _compute_soft_precision_recall(
    hd_edge: np.ndarray,
    viz_edge: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """
    Soft precision / recall on nonnegative edge maps over valid_mask (same pixels as continuous Dice).

    Let h = HD/reference edges, v = visualization edges (both nonnegative).
    - soft_precision = sum(h*v) / sum(v)  — reduces to TP/(TP+FP) for binary masks.
    - soft_recall    = sum(h*v) / sum(h)  — reduces to TP/(TP+FN) for binary masks.

    If sum(v) <= eps or sum(h) <= eps, the corresponding ratio is undefined (nan).
    """
    h = np.asarray(hd_edge, dtype=np.float64).ravel()
    v = np.asarray(viz_edge, dtype=np.float64).ravel()
    m = np.asarray(valid_mask, dtype=bool).ravel()
    if m.size != h.size or m.size != v.size:
        return float("nan"), float("nan")
    h, v = h[m], v[m]
    h = np.maximum(h, 0.0)
    v = np.maximum(v, 0.0)
    shv = float(np.sum(h * v))
    sh, sv = float(np.sum(h)), float(np.sum(v))
    soft_prec = float(shv / (sv + eps)) if sv > eps else float("nan")
    soft_rec = float(shv / (sh + eps)) if sh > eps else float("nan")
    return soft_prec, soft_rec


def _compute_weighted_gradient_alignment(
    hd_edge: np.ndarray,
    viz_edge: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Experimental edge alignment: spatial gradients (gx, gy) of scalar HD and viz edge maps,
    cosine alignment per pixel, weighted by (|grad_hd| + |grad_viz|).

    Returns a scalar in [-1, 1]: 1 = parallel gradients everywhere (weighted), -1 = opposite.
    Uses np.gradient; only valid_mask pixels contribute to the weighted mean.
    """
    h = np.asarray(hd_edge, dtype=np.float64)
    v = np.asarray(viz_edge, dtype=np.float64)
    m = np.asarray(valid_mask, dtype=bool)
    if h.shape != v.shape or m.shape != h.shape:
        return float("nan")
    # axis 0 = y, axis 1 = x
    ghy, ghx = np.gradient(h)
    gvy, gvx = np.gradient(v)
    mag_h = np.sqrt(ghx * ghx + ghy * ghy)
    mag_v = np.sqrt(gvx * gvx + gvy * gvy)
    dot = ghx * gvx + ghy * gvy
    cos_align = dot / (mag_h * mag_v + eps)
    cos_align = np.clip(cos_align, -1.0, 1.0)
    w = mag_h + mag_v
    mm = m & np.isfinite(cos_align) & np.isfinite(w)
    if not np.any(mm):
        return float("nan")
    num = float(np.sum(w[mm] * cos_align[mm]))
    den = float(np.sum(w[mm]))
    if den < eps:
        return float("nan")
    return float(num / den)


def _bbox_tight_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return None
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1


def _luminance_from_rgb01(rgb: np.ndarray) -> np.ndarray:
    """RGB float [0,1] or uint8 → scalar luminance HxW."""
    x = np.asarray(rgb)
    if x.ndim == 2:
        return x.astype(np.float32, copy=False)
    if x.ndim != 3 or x.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3 RGB, got {x.shape}")
    if np.issubdtype(x.dtype, np.integer):
        r = x[..., 0].astype(np.float32) / 255.0
        g = x[..., 1].astype(np.float32) / 255.0
        b = x[..., 2].astype(np.float32) / 255.0
    else:
        r = x[..., 0].astype(np.float32)
        g = x[..., 1].astype(np.float32)
        b = x[..., 2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32, copy=False)


def _compute_viz_contrast_color_metrics(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lab_entropy_bins_a: int = 16,
    lab_entropy_bins_b: int = 16,
) -> dict[str, float]:
    """Per-visualization contrast and color variety (no reference map).

    - luminance_rms_contrast: std of Rec.601 luminance on ``valid_mask``.
    - luminance_gradient_mean: mean Sobel gradient magnitude on luminance.
    - lab_chroma_entropy: Shannon entropy of joint histogram of OpenCV LAB a,b on ``valid_mask``.
    - lab_mean_chroma: mean sqrt((a-128)^2 + (b-128)^2) in OpenCV LAB units.

    ``rgb``: HxWx3 uint8 or float [0,1].
    """
    vm = np.asarray(valid_mask, dtype=bool)
    out_nan = {
        "luminance_rms_contrast": float("nan"),
        "luminance_gradient_mean": float("nan"),
        "lab_chroma_entropy": float("nan"),
        "lab_mean_chroma": float("nan"),
    }
    if not np.any(vm):
        return out_nan

    x = np.asarray(rgb)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] < 3:
        return out_nan

    if np.issubdtype(x.dtype, np.integer):
        rgb_u8 = np.clip(x[..., :3], 0, 255).astype(np.uint8)
        rgb01 = rgb_u8.astype(np.float32) / 255.0
    else:
        rgb01 = np.clip(x[..., :3].astype(np.float32), 0.0, 1.0)
        rgb_u8 = _to_uint8_rgb(rgb01)

    y = _luminance_from_rgb01(rgb01)
    yv = y[vm]
    lum_rms = float(np.std(yv))

    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(np.mean(mag[vm]))

    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    a = lab[..., 1].astype(np.float32)
    bch = lab[..., 2].astype(np.float32)
    chroma = np.sqrt((a - 128.0) ** 2 + (bch - 128.0) ** 2)
    mean_chroma = float(np.mean(chroma[vm]))

    ba = max(2, int(lab_entropy_bins_a))
    bb = max(2, int(lab_entropy_bins_b))
    a_idx = np.clip((a * ba / 256.0).astype(np.int32), 0, ba - 1)
    b_idx = np.clip((bch * bb / 256.0).astype(np.int32), 0, bb - 1)
    joint = a_idx * bb + b_idx
    vals = joint[vm].astype(np.int64, copy=False)
    counts = np.bincount(vals, minlength=ba * bb).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        entropy = float("nan")
    else:
        p = counts[counts > 0] / total
        entropy = float(-np.sum(p * np.log(p + 1e-15)))

    return {
        "luminance_rms_contrast": lum_rms,
        "luminance_gradient_mean": grad_mean,
        "lab_chroma_entropy": entropy,
        "lab_mean_chroma": mean_chroma,
    }


def _compute_gradient_ssim_mean(
    hd_scalar: np.ndarray,
    viz_scalar: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gaussian_weights: bool = True,
    sigma: float = 1.5,
) -> tuple[float, float, float, float]:
    """
    Experimental: np.gradient on HD scalar map vs viz scalar map → gx, gy, magnitude each;
    SSIM between corresponding maps; return (mean_ssim, ssim_gx, ssim_gy, ssim_mag).

    hd_scalar / viz_scalar: same HxW. valid_mask: tissue mask (bbox crop used for SSIM).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    h = np.asarray(hd_scalar, dtype=np.float64)
    v = np.asarray(viz_scalar, dtype=np.float64)
    m = np.asarray(valid_mask, dtype=bool)
    if h.shape != v.shape or m.shape != h.shape:
        return (float("nan"),) * 4
    bb = _bbox_tight_from_mask(m)
    if bb is None:
        return (float("nan"),) * 4
    y0, y1, x0, x1 = bb
    mc = m[y0:y1, x0:x1]
    if not np.any(mc):
        return (float("nan"),) * 4
    h0 = np.where(mc, h[y0:y1, x0:x1], 0.0)
    v0 = np.where(mc, v[y0:y1, x0:x1], 0.0)

    ghy, ghx = np.gradient(h0)
    gvy, gvx = np.gradient(v0)
    mag_h = np.sqrt(ghx * ghx + ghy * ghy)
    mag_v = np.sqrt(gvx * gvx + gvy * gvy)

    def _dr(a: np.ndarray, b: np.ndarray) -> float:
        av = a[mc]
        bv = b[mc]
        if av.size == 0:
            return 1.0
        return float(max(np.ptp(av), np.ptp(bv), 1e-8))

    def _one(a: np.ndarray, b: np.ndarray) -> float:
        dr = _dr(a, b)
        if dr < 1e-15:
            return 1.0
        try:
            return float(
                ssim(
                    a,
                    b,
                    data_range=dr,
                    gaussian_weights=gaussian_weights,
                    sigma=sigma,
                    use_sample_covariance=False,
                )
            )
        except Exception:
            return float("nan")

    sx = _one(ghx, gvx)
    sy = _one(ghy, gvy)
    sm = _one(mag_h, mag_v)
    vals = [x for x in (sx, sy, sm) if np.isfinite(x)]
    if not vals:
        mean_ssim = float("nan")
    else:
        mean_ssim = float(sum(vals) / len(vals))
    return (mean_ssim, sx, sy, sm)


def _agreement_overlay_rgb(
    hd_bin: np.ndarray,
    viz_bin: np.ndarray,
    valid_mask: np.ndarray,
    base_rgb: np.ndarray | None = None,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Create RGB overlay: green=agreement (both edge), red=viz missed (HD only),
    blue=viz added (viz only), dark gray=no edge. Optionally blend over base_rgb.
    """
    h, w = valid_mask.shape
    # Colors: TP=green, FN=red, FP=blue, neither=dark gray
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[~valid_mask] = 40  # dark for invalid
    tp = hd_bin & viz_bin
    fn = hd_bin & ~viz_bin
    fp = ~hd_bin & viz_bin
    neither = ~hd_bin & ~viz_bin
    overlay[valid_mask & tp] = [0, 200, 0]  # green - agreement
    overlay[valid_mask & fn] = [220, 50, 50]  # red - viz missed
    overlay[valid_mask & fp] = [50, 50, 220]  # blue - viz added
    overlay[valid_mask & neither] = [60, 60, 60]  # dark gray - no edge
    if base_rgb is not None and base_rgb.shape[:2] == (h, w):
        base_u8 = np.asarray(base_rgb, dtype=np.uint8)
        if base_u8.ndim == 2:
            base_u8 = np.stack([base_u8] * 3, axis=-1)
        blend = (alpha * overlay.astype(np.float32) + (1 - alpha) * base_u8.astype(np.float32)).astype(np.uint8)
        blend[~valid_mask] = base_u8[~valid_mask]
        return blend
    return overlay


def _add_agreement_legend(img: Image.Image, bar_height: int = 28) -> Image.Image:
    """Add a legend strip at the bottom: green=agreement, red=viz missed, blue=viz added, gray=no edge."""
    w, h = img.size
    new_h = h + bar_height
    out = Image.new("RGB", (w, new_h), (30, 30, 30))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    def _text_width(f: ImageFont.ImageFont, s: str) -> float:
        if hasattr(f, "getlength"):
            return float(f.getlength(s))
        bbox = f.getbbox(s)
        return float(bbox[2] - bbox[0]) if bbox else 0.0

    items = [
        ((0, 200, 0), "Agreement (both edge)"),
        ((220, 50, 50), "Viz missed (HD only)"),
        ((50, 50, 220), "Viz added (viz only)"),
        ((60, 60, 60), "No edge"),
    ]
    x = 8
    for rgb, label in items:
        draw.rectangle([x, h + 4, x + 20, h + bar_height - 4], fill=rgb, outline=(100, 100, 100))
        draw.text((x + 24, h + 6), label, fill=(255, 255, 255), font=font)
        x += 24 + _text_width(font, label) + 20
    return out


def _binary_from_threshold(
    arr01: np.ndarray,
    valid_mask: np.ndarray,
    mode: str,
    percentile: float,
    value: float,
    otsu_pre_equalize: bool = False,
    otsu_equalize_method: str = "hist",
    otsu_clahe_clip_limit: float = 2.0,
    otsu_clahe_tile_grid_size: int = 8,
) -> np.ndarray:
    x = np.asarray(arr01, dtype=np.float32)
    vals = x[valid_mask]
    if vals.size == 0:
        out = np.zeros_like(valid_mask, dtype=bool)
        return out
    m = str(mode).strip().lower()
    if m == "value":
        thr = float(value)
    elif m == "otsu":
        x_u8 = _to_uint8_gray01(x)
        if bool(otsu_pre_equalize):
            x_u8 = _equalize_gray_u8(
                x_u8,
                enabled=True,
                method=str(otsu_equalize_method),
                clahe_clip_limit=float(otsu_clahe_clip_limit),
                clahe_tile_grid_size=int(otsu_clahe_tile_grid_size),
            )
        vals_u8 = x_u8[valid_mask]
        thr_u8, _ = cv2.threshold(vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(thr_u8 / 255.0)
    else:
        p = min(max(float(percentile), 0.0), 100.0)
        thr = float(np.percentile(vals, p))
    out = np.zeros_like(valid_mask, dtype=bool)
    out[valid_mask] = x[valid_mask] >= thr
    return out


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _print_metrics_table(rows: List[dict]) -> None:
    if not rows:
        return
    base = ["visualization", "metric", "precision", "recall", "f1", "tp", "fp", "fn"]
    headers = ["msi_npy"] + base if any("msi_npy" in r for r in rows) else base
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))
    sep = " | "
    header_line = sep.join(h.ljust(widths[h]) for h in headers)
    divider = "-+-".join("-" * widths[h] for h in headers)
    print("\nEdge agreement PR/F1 table")
    print(header_line)
    print(divider)
    for r in rows:
        print(sep.join(str(r.get(h, "")).ljust(widths[h]) for h in headers))


def _parse_viz_exts(s: object | None) -> set[str]:
    if s is None or str(s).strip() == "":
        return {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
    return {"." + x.strip().lstrip(".").lower() for x in str(s).split(",") if x.strip()}


def _safe_subdir(stem: str) -> str:
    import re

    s = re.sub(r'[<>:"/\\|?*]', "_", stem).strip()
    return s[:120] or "msi"


def _viz_name_matches_npy_stem(viz_name: str, npy_stem: str) -> bool:
    s = npy_stem.lower()
    n = viz_name.lower()
    return n.startswith(s + "_") or n.startswith(s + ".")


def _one_to_many_mapping(
    npys_at_hw: List[Path],
    viz_at_hw: List[Path],
) -> Tuple[Dict[Path, List[Path]], List[Path]]:
    if len(npys_at_hw) == 1:
        return {npys_at_hw[0]: list(viz_at_hw)}, []

    stems_by_len = sorted({p.stem for p in npys_at_hw}, key=len, reverse=True)
    stem_to_npy = {p.stem: p for p in npys_at_hw}

    viz_to_npy: Dict[Path, Path] = {}
    for v in viz_at_hw:
        best_stem: str | None = None
        best_len = -1
        for st in stems_by_len:
            if _viz_name_matches_npy_stem(v.name, st) and len(st) > best_len:
                best_stem = st
                best_len = len(st)
        if best_stem is not None:
            viz_to_npy[v] = stem_to_npy[best_stem]

    by_npy: Dict[Path, List[Path]] = {n: [] for n in npys_at_hw}
    for v, npy in viz_to_npy.items():
        by_npy[npy].append(v)
    for n in by_npy:
        by_npy[n].sort(key=lambda x: x.name.lower())
    unassigned = [v for v in viz_at_hw if v not in viz_to_npy]
    unassigned.sort(key=lambda x: x.name.lower())
    return by_npy, unassigned


def _bucket_paths_by_hw_match(paths: List[Path]) -> Dict[Tuple[int, int], List[Path]]:
    from collections import defaultdict

    buckets: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    for p in paths:
        try:
            hw = _viz_path_spatial_hw(p)
        except Exception as e:
            logger.warning("skip %s | %s", p, e)
            continue
        buckets[hw].append(p)
    for k in buckets:
        buckets[k].sort(key=lambda x: x.name.lower())
    return dict(buckets)


def _bucket_msi_npys_by_hw(npy_paths: List[Path], transpose_msi: bool) -> Dict[Tuple[int, int], List[Path]]:
    from collections import defaultdict

    buckets: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    for p in npy_paths:
        try:
            hw = _spatial_hw_msi_cube_file(p, transpose_msi)
        except Exception as e:
            logger.warning("skip %s | %s", p, e)
            continue
        buckets[hw].append(p)
    for k in buckets:
        buckets[k].sort(key=lambda x: x.name.lower())
    return dict(buckets)


def _build_match_groups_matched_dirs(
    npy_dir: Path,
    viz_dir: Path,
    viz_exts: set[str],
    *,
    transpose_msi: bool = False,
) -> List[Tuple[Path, List[Path]]]:
    npy_paths = sorted([p for p in npy_dir.glob("*.npy") if p.is_file()], key=lambda x: x.name.lower())
    viz_paths: List[Path] = []
    for p in sorted(viz_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in viz_exts:
            viz_paths.append(p)
    if not npy_paths:
        raise ValueError(f"matched_dirs: no .npy files in {npy_dir}")
    if not viz_paths:
        raise ValueError(f"matched_dirs: no visualization files (extensions {sorted(viz_exts)}) in {viz_dir}")

    npy_b = _bucket_msi_npys_by_hw(npy_paths, transpose_msi)
    viz_b = _bucket_paths_by_hw_match(viz_paths)

    groups: List[Tuple[Path, List[Path]]] = []
    unassigned_all: List[Path] = []

    for hw in sorted(set(npy_b) & set(viz_b)):
        na = npy_b[hw]
        va = viz_b[hw]
        by_npy, unassigned = _one_to_many_mapping(na, va)
        unassigned_all.extend(unassigned)
        for npy in sorted(by_npy, key=lambda p: p.name.lower()):
            vlist = by_npy[npy]
            if vlist:
                groups.append((npy, vlist))

    if unassigned_all:
        sample = [p.name for p in unassigned_all[:24]]
        if len(unassigned_all) > 24:
            sample.append("...")
        logger.warning(
            "matched_dirs: %d viz file(s) unassigned (no stem match at their spatial size): %s",
            len(unassigned_all),
            sample,
        )
    for hw in sorted(set(npy_b) - set(viz_b)):
        logger.warning(
            "matched_dirs: spatial size %s has %d .npy but no viz at this size",
            hw,
            len(npy_b[hw]),
        )
    for hw in sorted(set(viz_b) - set(npy_b)):
        logger.warning(
            "matched_dirs: spatial size %s has %d viz but no .npy at this size",
            hw,
            len(viz_b[hw]),
        )

    groups.sort(key=lambda x: x[0].name.lower())
    return groups


def _format_match_groups_report(groups: List[Tuple[Path, List[Path]]]) -> str:
    lines = [
        "matched_dirs: same (rows, cols) as list_dir_shapes.py; 1 MSI .npy : many viz; "
        "with multiple npy at one size, viz assigned by <stem>_ or <stem>.<ext> (longest stem wins).",
        "",
    ]
    for npy, vlist in groups:
        lines.append(f"{npy}  ->  {len(vlist)} visualization(s)")
        for v in vlist:
            lines.append(f"  {v.name}")
        lines.append("")
    return "\n".join(lines)


def _run_edge_debug_group(
    cfg: DictConfig,
    *,
    npy_path: Path,
    viz_paths: List[Path],
    out_dir: Path,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    msi_label: str,
) -> List[dict]:
    """HD edges + per-visualization RGB edges and optional PR/F1 for one loaded MSI cube."""
    _ = npy_path
    allow_resize = bool(getattr(cfg.input, "allow_resize", False))
    ms_cfg = getattr(cfg, "edge_maps", None)
    hd_cfg = getattr(cfg, "hd_edges", None)
    edge_method_cfg = _parse_edge_detection_cfg(cfg)
    ms_enabled = bool(getattr(getattr(ms_cfg, "multi_scale", None), "enabled", False))
    sigmas = [float(v) for v in list(getattr(getattr(ms_cfg, "multi_scale", None), "sigmas", [0.0]))]
    if (not ms_enabled) or len(sigmas) == 0:
        sigmas = [0.0]
    aggregation = str(getattr(getattr(ms_cfg, "multi_scale", None), "aggregation", "mean"))

    t_pca0 = time.perf_counter()
    msi_for_edges = _pca_project_msi_for_edges(msi, valid_mask, hd_cfg)
    logger.info("hd edge PCA/project | shape=%s | %.2fs", msi_for_edges.shape, time.perf_counter() - t_pca0)
    if int(msi_for_edges.shape[2]) != int(msi.shape[2]):
        logger.info("hd edge PCA enabled | channels %d -> %d", int(msi.shape[2]), int(msi_for_edges.shape[2]))

    t_hd0 = time.perf_counter()
    hd_result = _edge_maps_multiscale(
        "hd",
        msi_for_edges,
        valid_mask,
        sigmas=sigmas,
        aggregation=aggregation,
        hd_cfg=hd_cfg,
        edge_method_cfg=edge_method_cfg,
    )
    hd_edge, hd_edge_linf = hd_result[0], hd_result[1]
    individual_hd_maps = hd_result[2] if len(hd_result) > 2 else None
    logger.info("hd edge multiscale done | %.2fs", time.perf_counter() - t_hd0)
    hd_edge_n = _normalize_map_percentile(
        hd_edge,
        valid_mask,
        low_pct=float(cfg.spatial_normalization.low_percentile),
        high_pct=float(cfg.spatial_normalization.high_percentile),
    )
    hd_edge_linf_n = _normalize_map_percentile(
        hd_edge_linf,
        valid_mask,
        low_pct=float(cfg.spatial_normalization.low_percentile),
        high_pct=float(cfg.spatial_normalization.high_percentile),
    )

    flatten_outputs = bool(getattr(cfg.output, "flatten_outputs", False))
    hd_dir = out_dir if flatten_outputs else (out_dir / str(getattr(cfg.output, "hd_subdir", "hd_edges")))
    vals_hd = hd_edge[valid_mask]
    if vals_hd.size > 0 and float(np.max(vals_hd)) <= 1e-8:
        logger.warning("hd edge map is all zeros; saving raw NPY for debug (load and inspect min/max)")
        hd_dir.mkdir(parents=True, exist_ok=True)
        np.save(hd_dir / "hd_edge_raw_debug.npy", hd_edge.astype(np.float32))
    save_npy = bool(getattr(cfg.output, "save_npy", True))
    equalize_png = bool(getattr(cfg.output, "equalize_png", False))
    equalize_method = str(getattr(cfg.output, "equalize_method", "clahe"))
    clahe_clip_limit = float(getattr(cfg.output, "clahe_clip_limit", 2.0))
    clahe_tile_grid_size = int(getattr(cfg.output, "clahe_tile_grid_size", 8))
    edge_colormap = getattr(cfg.output, "edge_colormap", None)
    if edge_colormap is None and getattr(cfg, "debug", None) is not None:
        edge_colormap = getattr(cfg.debug, "edge_colormap", None)
    eval_cfg = getattr(cfg, "evaluation", None)
    eval_enabled = bool(getattr(eval_cfg, "enabled", False)) if eval_cfg is not None else False
    eval_mode = str(getattr(eval_cfg, "threshold_mode", "percentile")) if eval_cfg is not None else "percentile"
    eval_percentile = float(getattr(eval_cfg, "threshold_percentile", 85.0)) if eval_cfg is not None else 85.0
    eval_value = float(getattr(eval_cfg, "threshold_value", 0.5)) if eval_cfg is not None else 0.5
    eval_include_linf = bool(getattr(eval_cfg, "include_linf", True)) if eval_cfg is not None else True
    eval_otsu_pre_equalize = bool(getattr(eval_cfg, "otsu_pre_equalize", True)) if eval_cfg is not None else True
    eval_otsu_equalize_method = str(getattr(eval_cfg, "otsu_equalize_method", "hist")) if eval_cfg is not None else "hist"
    eval_otsu_clahe_clip_limit = (
        float(getattr(eval_cfg, "otsu_clahe_clip_limit", 2.0)) if eval_cfg is not None else 2.0
    )
    eval_otsu_clahe_tile_grid_size = (
        int(getattr(eval_cfg, "otsu_clahe_tile_grid_size", 8)) if eval_cfg is not None else 8
    )
    save_agreement_overlay = bool(getattr(eval_cfg, "save_agreement_overlay", False)) if eval_cfg is not None else False
    eval_rows: List[dict] = []

    _save_map_png_and_npy(
        hd_dir / "hd_edge_norm",
        hd_edge_n,
        save_npy=save_npy,
        equalize_png=equalize_png,
        equalize_method=equalize_method,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
        colormap=edge_colormap,
    )
    if bool(getattr(cfg.output, "save_linf", True)):
        _save_map_png_and_npy(
            hd_dir / "hd_edge_linf_norm",
            hd_edge_linf_n,
            save_npy=save_npy,
            equalize_png=equalize_png,
            equalize_method=equalize_method,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            colormap=edge_colormap,
        )
    if individual_hd_maps:
        hd_dir.mkdir(parents=True, exist_ok=True)
        _save_map_png_and_npy(
            hd_dir / "hd_edge_final_norm",
            hd_edge_n,
            save_npy=save_npy,
            equalize_png=equalize_png,
            equalize_method=equalize_method,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            colormap=edge_colormap,
        )
        for metric_name, arr in individual_hd_maps.items():
            arr_n = _normalize_map_percentile(
                arr,
                valid_mask,
                low_pct=float(cfg.spatial_normalization.low_percentile),
                high_pct=float(cfg.spatial_normalization.high_percentile),
            )
            _save_map_png_and_npy(
                hd_dir / f"hd_edge_{metric_name}_norm",
                arr_n,
                save_npy=save_npy,
                equalize_png=equalize_png,
                equalize_method=equalize_method,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid_size=clahe_tile_grid_size,
                colormap=edge_colormap,
            )
        logger.info("saved %d individual distance metric edge maps to %s", len(individual_hd_maps), hd_dir)
    logger.info("saved hd edge maps to %s", hd_dir)

    viz_dir = out_dir if flatten_outputs else (out_dir / str(getattr(cfg.output, "viz_subdir", "viz_edges")))
    for i, viz_path in enumerate(viz_paths):
        tv = time.perf_counter()
        viz_rgb = _load_visualization(
            viz_path,
            target_hw=(msi.shape[0], msi.shape[1]),
            allow_resize=allow_resize,
        )
        v_edge, v_edge_linf = _edge_maps_multiscale(
            "rgb",
            viz_rgb,
            valid_mask,
            sigmas=sigmas,
            aggregation=aggregation,
            hd_cfg=hd_cfg,
            edge_method_cfg=edge_method_cfg,
        )
        v_edge_n = _normalize_map_percentile(
            v_edge,
            valid_mask,
            low_pct=float(cfg.spatial_normalization.low_percentile),
            high_pct=float(cfg.spatial_normalization.high_percentile),
        )
        v_edge_linf_n = _normalize_map_percentile(
            v_edge_linf,
            valid_mask,
            low_pct=float(cfg.spatial_normalization.low_percentile),
            high_pct=float(cfg.spatial_normalization.high_percentile),
        )
        tag = _viz_output_tag(viz_path, i)
        _save_map_png_and_npy(
            viz_dir / f"{tag}__viz_edge_norm",
            v_edge_n,
            save_npy=save_npy,
            equalize_png=equalize_png,
            equalize_method=equalize_method,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            orig_for_context=viz_rgb,
            colormap=edge_colormap,
        )
        if bool(getattr(cfg.output, "save_linf", True)):
            _save_map_png_and_npy(
                viz_dir / f"{tag}__viz_edge_linf_norm",
                v_edge_linf_n,
                save_npy=save_npy,
                equalize_png=equalize_png,
                equalize_method=equalize_method,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid_size=clahe_tile_grid_size,
                orig_for_context=viz_rgb,
                colormap=edge_colormap,
            )
        if eval_enabled:
            if equalize_png:
                hd_eval = _equalize_gray_u8(
                    _to_uint8_gray01(hd_edge_n),
                    enabled=True,
                    method=equalize_method,
                    clahe_clip_limit=clahe_clip_limit,
                    clahe_tile_grid_size=clahe_tile_grid_size,
                ).astype(np.float32) / 255.0
                viz_eval = _equalize_gray_u8(
                    _to_uint8_gray01(v_edge_n),
                    enabled=True,
                    method=equalize_method,
                    clahe_clip_limit=clahe_clip_limit,
                    clahe_tile_grid_size=clahe_tile_grid_size,
                ).astype(np.float32) / 255.0
                hd_eval[~valid_mask] = 0.0
                viz_eval[~valid_mask] = 0.0
            else:
                hd_eval = hd_edge_n
                viz_eval = v_edge_n

            hd_bin = _binary_from_threshold(
                hd_eval,
                valid_mask,
                eval_mode,
                eval_percentile,
                eval_value,
                otsu_pre_equalize=eval_otsu_pre_equalize,
                otsu_equalize_method=eval_otsu_equalize_method,
                otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
                otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
            )
            viz_bin = _binary_from_threshold(
                viz_eval,
                valid_mask,
                eval_mode,
                eval_percentile,
                eval_value,
                otsu_pre_equalize=eval_otsu_pre_equalize,
                otsu_equalize_method=eval_otsu_equalize_method,
                otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
                otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
            )
            p, r, f1, tp, fp, fn = _compute_prf1(hd_bin[valid_mask], viz_bin[valid_mask])
            eval_rows.append(
                {
                    "msi_npy": msi_label,
                    "visualization": viz_path.name,
                    "metric": "edge_norm",
                    "precision": f"{p:.4f}",
                    "recall": f"{r:.4f}",
                    "f1": f"{f1:.4f}",
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
            if save_agreement_overlay:
                overlay_rgb = _agreement_overlay_rgb(hd_bin, viz_bin, valid_mask)
                overlay_img = Image.fromarray(overlay_rgb, mode="RGB")
                overlay_img = _add_agreement_legend(overlay_img)
                overlay_path = viz_dir / f"{tag}__agreement_overlay.png"
                viz_dir.mkdir(parents=True, exist_ok=True)
                legend_height = 28
                viz_u8 = np.asarray(viz_rgb, dtype=np.uint8)
                if viz_u8.ndim == 2:
                    viz_u8 = np.stack([viz_u8] * 3, axis=-1)
                pad = np.full((legend_height, viz_u8.shape[1], 3), 30, dtype=np.uint8)
                orig_padded = np.concatenate([viz_u8[..., :3], pad], axis=0)
                _save_side_by_side_rgb(orig_padded, np.asarray(overlay_img), overlay_path)
            if eval_include_linf and bool(getattr(cfg.output, "save_linf", True)):
                if equalize_png:
                    hd_linf_eval = _equalize_gray_u8(
                        _to_uint8_gray01(hd_edge_linf_n),
                        enabled=True,
                        method=equalize_method,
                        clahe_clip_limit=clahe_clip_limit,
                        clahe_tile_grid_size=clahe_tile_grid_size,
                    ).astype(np.float32) / 255.0
                    viz_linf_eval = _equalize_gray_u8(
                        _to_uint8_gray01(v_edge_linf_n),
                        enabled=True,
                        method=equalize_method,
                        clahe_clip_limit=clahe_clip_limit,
                        clahe_tile_grid_size=clahe_tile_grid_size,
                    ).astype(np.float32) / 255.0
                    hd_linf_eval[~valid_mask] = 0.0
                    viz_linf_eval[~valid_mask] = 0.0
                else:
                    hd_linf_eval = hd_edge_linf_n
                    viz_linf_eval = v_edge_linf_n
                hd_linf_bin = _binary_from_threshold(
                    hd_linf_eval,
                    valid_mask,
                    eval_mode,
                    eval_percentile,
                    eval_value,
                    otsu_pre_equalize=eval_otsu_pre_equalize,
                    otsu_equalize_method=eval_otsu_equalize_method,
                    otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
                    otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
                )
                viz_linf_bin = _binary_from_threshold(
                    viz_linf_eval,
                    valid_mask,
                    eval_mode,
                    eval_percentile,
                    eval_value,
                    otsu_pre_equalize=eval_otsu_pre_equalize,
                    otsu_equalize_method=eval_otsu_equalize_method,
                    otsu_clahe_clip_limit=eval_otsu_clahe_clip_limit,
                    otsu_clahe_tile_grid_size=eval_otsu_clahe_tile_grid_size,
                )
                p, r, f1, tp, fp, fn = _compute_prf1(hd_linf_bin[valid_mask], viz_linf_bin[valid_mask])
                eval_rows.append(
                    {
                        "msi_npy": msi_label,
                        "visualization": viz_path.name,
                        "metric": "edge_linf_norm",
                        "precision": f"{p:.4f}",
                        "recall": f"{r:.4f}",
                        "f1": f"{f1:.4f}",
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )
        logger.info("saved viz edges for %s as %s | time=%.2fs", viz_path.name, tag, float(time.perf_counter() - tv))

    return eval_rows


@hydra.main(version_base=None, config_path="configs", config_name="debug_edge_maps")
def main(cfg: DictConfig) -> None:
    log_level = str(getattr(getattr(cfg, "logging", None), "level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    npy_dir_r = OmegaConf.select(cfg, "input.visualization_npy_dir")
    png_dir_r = OmegaConf.select(cfg, "input.visualization_png_dir")
    vpaths = OmegaConf.select(cfg, "input.visualization_paths")
    pairing = str(OmegaConf.select(cfg, "input.visualization_pairing") or "").strip().lower()
    try:
        n_vpaths = len(vpaths) if vpaths is not None else 0
    except TypeError:
        n_vpaths = 1 if vpaths else 0
    matched_dirs_mode = pairing in ("matched_dirs", "one_to_many")
    use_shape_pairing = pairing in ("by_shape", "shape", "size", "by_size")
    if (
        not matched_dirs_mode
        and not use_shape_pairing
        and _pairing_unset_for_inference(pairing)
        and npy_dir_r
        and png_dir_r
        and n_vpaths == 0
    ):
        use_shape_pairing = True
        logger.info(
            "visualization_pairing inferred as by_shape (both npy/png dirs set, visualization_paths empty)",
        )
    npy_dir_pair: Path | None = None
    png_dir_pair: Path | None = None
    viz_paths: List[Path] | None = None
    if matched_dirs_mode:
        if not npy_dir_r or not png_dir_r:
            raise ValueError(
                "visualization_pairing=matched_dirs requires input.visualization_npy_dir and "
                "input.visualization_png_dir (both set to existing directories).",
            )
        npy_dir_pair = Path(str(npy_dir_r))
        png_dir_pair = Path(str(png_dir_r))
        if not npy_dir_pair.is_dir():
            raise FileNotFoundError(f"visualization_npy_dir is not a directory: {npy_dir_pair}")
        if not png_dir_pair.is_dir():
            raise FileNotFoundError(f"visualization_png_dir is not a directory: {png_dir_pair}")
    elif use_shape_pairing:
        if not npy_dir_r or not png_dir_r:
            raise ValueError(
                "Shape pairing requires input.visualization_npy_dir and input.visualization_png_dir "
                "(both non-empty), or set input.visualization_paths instead.",
            )
        npy_dir_pair = Path(str(npy_dir_r))
        png_dir_pair = Path(str(png_dir_r))
        if not npy_dir_pair.is_dir():
            raise FileNotFoundError(f"visualization_npy_dir is not a directory: {npy_dir_pair}")
        if not png_dir_pair.is_dir():
            raise FileNotFoundError(f"visualization_png_dir is not a directory: {png_dir_pair}")
    else:
        if not vpaths or n_vpaths == 0:
            raise ValueError(
                "input.visualization_paths must contain at least one path, "
                "or set both input.visualization_npy_dir and input.visualization_png_dir for shape pairing "
                "(optionally set visualization_pairing=by_shape or matched_dirs).",
            )
        viz_paths = [Path(str(p)) for p in vpaths]
        for p in viz_paths:
            if not p.exists():
                raise FileNotFoundError(f"Visualization file not found: {p}")

    npy_raw = OmegaConf.select(cfg, "input.npy_path")
    npy_path: Path | None = None
    if matched_dirs_mode:
        pass
    elif npy_raw is not None and str(npy_raw).strip() != "":
        npy_path = Path(str(npy_raw))
    elif use_shape_pairing and npy_dir_pair is not None and png_dir_pair is not None:
        npy_path = _pick_msi_npy_for_shape_pairing(
            npy_dir_pair,
            png_dir_pair,
            transpose_msi=bool(cfg.data.transpose_msi),
        )
    else:
        raise ValueError(
            "input.npy_path is required unless you use shape pairing with visualization_npy_dir + "
            "visualization_png_dir (then MSI is auto-picked from a .npy that matches a .png size), "
            "or use visualization_pairing=matched_dirs. "
            "Set input.npy_path in scripts/configs/debug_edge_maps.yaml or pass input.npy_path=E:/path/to/0.npy",
        )
    if not matched_dirs_mode:
        assert npy_path is not None
        if not npy_path.exists():
            raise FileNotFoundError(f"MSI file not found: {npy_path}")

    out_dir = Path(str(cfg.output.out_dir)) if cfg.output.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    t0 = time.perf_counter()
    normalization_raw = getattr(cfg.data, "normalization", None)
    if normalization_raw is None:
        # Backward compatibility with older configs.
        normalization_raw = bool(getattr(cfg.data, "tic_normalize", False))
    normalization = _resolve_normalization_mode(normalization_raw)

    eval_cfg_top = getattr(cfg, "evaluation", None)
    eval_enabled = bool(getattr(eval_cfg_top, "enabled", False)) if eval_cfg_top is not None else False

    if matched_dirs_mode:
        assert npy_dir_pair is not None and png_dir_pair is not None
        exts = _parse_viz_exts(OmegaConf.select(cfg, "input.visualization_viz_ext"))
        match_groups = _build_match_groups_matched_dirs(
            npy_dir_pair,
            png_dir_pair,
            exts,
            transpose_msi=bool(cfg.data.transpose_msi),
        )
        if not match_groups:
            raise ValueError(
                "matched_dirs: no MSI .npy with at least one matched visualization. "
                "Check spatial (rows, cols) and <stem>_ filenames (see scripts/list_dir_shapes.py).",
            )
        try:
            (out_dir / "matched_dirs_report.txt").write_text(
                _format_match_groups_report(match_groups),
                encoding="utf-8",
            )
            logger.info("Wrote matched_dirs_report.txt under %s", out_dir)
        except OSError as exc:
            logger.warning("Could not write matched_dirs_report.txt: %s", exc)

        eval_rows_all: List[dict] = []
        allow_resize_global = bool(getattr(cfg.input, "allow_resize", False))
        for gi, (msi_path, vlist) in enumerate(match_groups):
            gout = out_dir if len(match_groups) == 1 else (out_dir / _safe_subdir(msi_path.stem))
            gout.mkdir(parents=True, exist_ok=True)
            logger.info(
                "matched_dirs | group %d/%d | MSI=%s | %d viz",
                gi + 1,
                len(match_groups),
                msi_path.name,
                len(vlist),
            )
            msi = _load_msi(
                msi_path,
                transpose_msi=bool(cfg.data.transpose_msi),
                normalization=normalization,
            )
            valid_mask = msi.sum(axis=-1) > 0
            if not np.any(valid_mask):
                logger.warning("skip %s: MSI valid mask empty", msi_path)
                continue
            logger.info("loaded msi shape=%s valid_pixels=%d", tuple(msi.shape), int(valid_mask.sum()))
            v_work = list(vlist)
            if not allow_resize_global:
                tgt_hw = (int(msi.shape[0]), int(msi.shape[1]))
                n_before = len(v_work)
                v_work = _filter_viz_paths_to_msi_shape(v_work, tgt_hw)
                if len(v_work) < n_before:
                    logger.info(
                        "dropped %d visualization(s) not matching MSI size %dx%d",
                        n_before - len(v_work),
                        tgt_hw[0],
                        tgt_hw[1],
                    )
                if not v_work:
                    logger.warning("skip %s: no viz at MSI spatial size after filter", msi_path)
                    continue
            eval_rows_all.extend(
                _run_edge_debug_group(
                    cfg,
                    npy_path=msi_path,
                    viz_paths=v_work,
                    out_dir=gout,
                    msi=msi,
                    valid_mask=valid_mask,
                    msi_label=msi_path.name,
                )
            )

        if eval_enabled and eval_rows_all:
            metrics_csv = out_dir / "edge_prf1_table.csv"
            _write_csv(metrics_csv, eval_rows_all)
            _print_metrics_table(eval_rows_all)
            logger.info("saved edge PR/F1 table: %s", metrics_csv)

        logger.info("done | total_time=%.2fs | out_dir=%s", float(time.perf_counter() - t0), out_dir)
        print(f"Saved edge debug outputs to: {out_dir}")
        return

    assert npy_path is not None
    msi = _load_msi(
        npy_path,
        transpose_msi=bool(cfg.data.transpose_msi),
        normalization=normalization,
    )
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty.")
    logger.info("loaded msi shape=%s valid_pixels=%d", tuple(msi.shape), int(valid_mask.sum()))

    if use_shape_pairing:
        assert npy_dir_pair is not None and png_dir_pair is not None
        report = _build_npy_png_shape_matching_report(
            npy_dir_pair,
            png_dir_pair,
            msi_npy_path=npy_path,
            target_hw=(int(msi.shape[0]), int(msi.shape[1])),
            transpose_msi=bool(cfg.data.transpose_msi),
        )
        print(report, flush=True)
        logger.info("%s", report)
        try:
            (out_dir / "shape_matching_report.txt").write_text(report, encoding="utf-8")
            logger.info("Wrote shape_matching_report.txt under %s", out_dir)
        except OSError as exc:
            logger.warning("Could not write shape_matching_report.txt: %s", exc)

        viz_paths = _pair_npy_png_paths_by_shape(
            npy_dir_pair,
            png_dir_pair,
            target_hw=(int(msi.shape[0]), int(msi.shape[1])),
            transpose_msi=bool(cfg.data.transpose_msi),
        )
        if not viz_paths:
            raise ValueError(
                "No .npy/.png pairs at MSI spatial size. "
                f"MSI is {msi.shape[0]}x{msi.shape[1]}; the viz directories had no matching pair at that "
                f"(H,W). Use input.npy_path for the same slide/grid as the visualizations, or set "
                f"input.allow_resize=true with explicit input.visualization_paths.",
            )
        n_pairs = len(viz_paths) // 2
        logger.info(
            "visualization_pairing=by_shape | %d pairs (%d files) at MSI size %dx%d from %s and %s",
            n_pairs,
            len(viz_paths),
            int(msi.shape[0]),
            int(msi.shape[1]),
            npy_dir_pair,
            png_dir_pair,
        )
        for p in viz_paths:
            if not p.exists():
                raise FileNotFoundError(f"Visualization file not found: {p}")

        n_pair_files = len(viz_paths)
        viz_paths = [p for p in viz_paths if p.suffix.lower() != ".npy"]
        if len(viz_paths) < n_pair_files:
            logger.info(
                "by_shape: removed %d paired .npy path(s) from viz list; edge branch scores image exports only",
                n_pair_files - len(viz_paths),
            )

    assert viz_paths is not None

    allow_resize = bool(getattr(cfg.input, "allow_resize", False))
    if not allow_resize:
        tgt_hw = (int(msi.shape[0]), int(msi.shape[1]))
        n_before = len(viz_paths)
        viz_paths = _filter_viz_paths_to_msi_shape(viz_paths, tgt_hw)
        if len(viz_paths) < n_before:
            logger.info(
                "dropped %d visualization(s) not matching MSI size %dx%d",
                n_before - len(viz_paths),
                tgt_hw[0],
                tgt_hw[1],
            )
        if not viz_paths:
            raise ValueError(
                "No visualizations left at MSI spatial size with allow_resize=false. "
                f"MSI is {tgt_hw[0]}x{tgt_hw[1]}. Point input.npy_path at the same slide/grid as the viz files, "
                "or set input.allow_resize=true.",
            )

    eval_rows = _run_edge_debug_group(
        cfg,
        npy_path=npy_path,
        viz_paths=viz_paths,
        out_dir=out_dir,
        msi=msi,
        valid_mask=valid_mask,
        msi_label=npy_path.name,
    )

    if eval_enabled:
        metrics_csv = out_dir / "edge_prf1_table.csv"
        _write_csv(metrics_csv, eval_rows)
        _print_metrics_table(eval_rows)
        logger.info("saved edge PR/F1 table: %s", metrics_csv)

    logger.info("done | total_time=%.2fs | out_dir=%s", float(time.perf_counter() - t0), out_dir)
    print(f"Saved edge debug outputs to: {out_dir}")


if __name__ == "__main__":
    main()

