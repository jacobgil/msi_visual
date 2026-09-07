#!/usr/bin/env python3
"""
Gallery: TOP3 | SoLaCE | automatic segmentation from the SoLaCE edge field.

Segmentation methods (``segmentation.method``):
  * ``landmark_graphcut`` — soft landmark α + k-means + α-expansion Potts (default)
    or random walker (``lgc_optimizer=random_walker``)
  * ``graph_twostage`` — SoLaCE Felzenszwalb leaves + TOP3 RAG merge
  * ``graph`` — Felzenszwalb-style pixel graph with SoLaCE edge weights only
  * ``hierarchical`` — SLIC/watershed/closed-edge leaves + SoLaCE-weighted merges
  * ``mgac`` — morphological geodesic active contours stopped by SoLaCE
  * ``watershed`` — marker-controlled watershed on SoLaCE ridges

Graph / hierarchical gallery row:
  TOP3 | SoLaCE | fine | medium | coarse

Outputs (Hydra run dir):
  * ``{stem}__top3.png``, ``{stem}__solace.png``, ``{stem}__segmentation.png``
  * ``{stem}__labels.npy`` — primary cut labels (0 = background / invalid)
  * ``{stem}__labels_{level}.npy`` — per hierarchy level (hierarchical mode)
  * ``{stem}__panel.png`` — side-by-side row (or stacked rows for multiple cubes)
  * ``manifest.json``

Run from repo root:
  python scripts/solace_segmentation_gallery.py
  python scripts/solace_segmentation_gallery.py data.npy_path=D:/data/0.npy
  python scripts/solace_segmentation_gallery.py segmentation.method=mgac

Config: ``scripts/configs/solace_segmentation_gallery.yaml``
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    import random

    np.random.seed(seed)
    random.seed(seed)


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI shape H*W*C (or C*H*W with transpose), got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI contains negative values; this script expects non-negative inputs.")
    if tic_normalize:
        sums = img.sum(axis=-1, keepdims=True) + 1e-8
        img = img / sums
    return img


def _resolve_npy_paths(cfg: DictConfig) -> list[Path]:
    raw_list = OmegaConf.select(cfg, "data.npy_paths", default=None)
    paths: list[Path] = []
    if raw_list is not None:
        container = OmegaConf.to_container(raw_list, resolve=True)
        if isinstance(container, (list, tuple)):
            paths = [Path(to_absolute_path(str(p))) for p in container if str(p).strip()]
    if not paths:
        single = OmegaConf.select(cfg, "data.npy_path", default=None)
        if single is None or not str(single).strip():
            raise ValueError("Set data.npy_path or data.npy_paths.")
        paths = [Path(to_absolute_path(str(single)))]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing MSI cubes:\n  " + "\n  ".join(str(p) for p in missing))
    return paths


def _top3_rgb(msi: np.ndarray, cfg: DictConfig) -> np.ndarray:
    from msi_visual.percentile_ratio import TOP3

    t = cfg.top3
    rgb = TOP3(
        low=float(OmegaConf.select(t, "low", default=99.9)),
        norm_percentile=float(OmegaConf.select(t, "norm_percentile", default=99.9)),
    )(msi, to_lab=bool(OmegaConf.select(t, "to_lab", default=True)))
    return np.asarray(rgb, dtype=np.uint8)


def _compute_solace(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rank_config_path: str,
    hd_method: str,
) -> np.ndarray:
    from solace_edge_aware_smoothing import _compute_solace_guide

    return _compute_solace_guide(
        msi,
        valid_mask,
        rank_config_path=rank_config_path,
        hd_method=hd_method,
    )


def _solace_rgb(guide: np.ndarray, valid_mask: np.ndarray, colormap: str) -> np.ndarray:
    from solace_edge_aware_smoothing import _solace_edges_rgb

    return _solace_edges_rgb(guide, valid_mask, colormap)


def _equalize_edge_map(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    method: str = "clahe",
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
) -> np.ndarray:
    """Equalize SoLaCE strengths on tissue so weak/strong ridges are more comparable.

    Methods:
      * ``hist``  — global histogram equalization
      * ``clahe`` — contrast-limited adaptive hist eq (default; better for local ridges)
      * ``rank``  — rank-transform / percentile stretch of tissue pixels to [0,1]
    """
    import cv2

    mask = np.asarray(valid_mask, dtype=bool)
    src = np.asarray(edge, dtype=np.float32)
    out = np.zeros_like(src, dtype=np.float32)
    if not np.any(mask):
        return out

    mode = str(method).strip().lower()
    vals = src[mask]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin + 1e-12:
        out[mask] = 0.0
        return out

    norm01 = np.zeros_like(src, dtype=np.float32)
    norm01[mask] = (src[mask] - vmin) / (vmax - vmin)

    if mode in {"rank", "percentile", "cdf"}:
        # Empirical CDF on tissue → uniform [0,1]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, num=vals.size, dtype=np.float32)
        out[mask] = ranks
        return out

    u8 = np.zeros(src.shape, dtype=np.uint8)
    u8[mask] = np.clip(np.rint(norm01[mask] * 255.0), 0, 255).astype(np.uint8)

    if mode in {"clahe", "adaptive"}:
        grid = max(2, int(clahe_grid))
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(grid, grid))
        eq = clahe.apply(u8)
    else:
        # Global hist eq only on tissue: equalize a compact tissue crop then paste back.
        eq = cv2.equalizeHist(u8)

    out[mask] = eq[mask].astype(np.float32) / 255.0
    return out


def _smooth_edge(edge: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or float(sigma) <= 0:
        return np.asarray(edge, dtype=np.float32)
    from skimage.filters import gaussian

    return gaussian(np.asarray(edge, dtype=np.float32), sigma=float(sigma), preserve_range=True).astype(
        np.float32
    )


def _seed_coords(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    edge_percentile: float,
    min_distance: int,
) -> np.ndarray:
    """Local maxima of distance-to-ridge inside tissue (basin centers)."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    mask = np.asarray(valid_mask, dtype=bool)
    vals = edge[mask]
    if vals.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    thr = float(np.percentile(vals, float(edge_percentile)))
    basins = mask & (edge < thr)
    if not np.any(basins):
        basins = mask
    dist = ndi.distance_transform_edt(basins)
    dist = dist.astype(np.float32)
    dist[~mask] = 0.0
    coords = peak_local_max(
        dist,
        min_distance=max(1, int(min_distance)),
        labels=mask.astype(np.int32),
        exclude_border=False,
    )
    if coords.size == 0:
        # Fallback: densest tissue pixel.
        flat = int(np.argmax(dist))
        coords = np.array([[flat // dist.shape[1], flat % dist.shape[1]]], dtype=np.int64)
    return np.asarray(coords, dtype=np.int64)


def _segment_mgac(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    edge_percentile: float,
    min_seed_distance: int,
    iterations: int,
    smoothing: int,
    balloon: float,
    threshold: float,
    edge_scale: float,
    seed_radius: int,
) -> np.ndarray:
    from skimage.segmentation import morphological_geodesic_active_contour

    mask = np.asarray(valid_mask, dtype=bool)
    scale = max(1e-6, float(edge_scale))
    gimage = 1.0 / (1.0 + (np.asarray(edge, dtype=np.float32) / scale) ** 2)
    gimage = np.clip(gimage, 0.0, 1.0).astype(np.float32)
    gimage[~mask] = 0.0

    coords = _seed_coords(
        edge,
        mask,
        edge_percentile=edge_percentile,
        min_distance=min_seed_distance,
    )
    init = np.zeros(edge.shape, dtype=np.int8)
    rr = max(1, int(seed_radius))
    h, w = edge.shape
    for y, x in coords:
        y0, y1 = max(0, int(y) - rr), min(h, int(y) + rr + 1)
        x0, x1 = max(0, int(x) - rr), min(w, int(x) + rr + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - int(y)) ** 2 + (xx - int(x)) ** 2 <= rr * rr
        init[y0:y1, x0:x1][disk] = 1
    init[~mask] = 0

    level_set = morphological_geodesic_active_contour(
        gimage,
        num_iter=max(1, int(iterations)),
        init_level_set=init.astype(np.float64),
        smoothing=max(0, int(smoothing)),
        threshold=float(threshold),
        balloon=float(balloon),
    )
    fg = (np.asarray(level_set) > 0.5) & mask

    from skimage.measure import label

    labels = label(fg, connectivity=1).astype(np.int32)
    labels[~mask] = 0
    return labels


def _segment_watershed(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    edge_percentile: float,
    min_distance: int,
    compactness: float,
) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    mask = np.asarray(valid_mask, dtype=bool)
    vals = edge[mask]
    thr = float(np.percentile(vals, float(edge_percentile))) if vals.size else 0.5
    basins = mask & (edge < thr)
    if not np.any(basins):
        basins = mask
    dist = ndi.distance_transform_edt(basins).astype(np.float32)
    dist[~mask] = 0.0
    coords = peak_local_max(
        dist,
        min_distance=max(1, int(min_distance)),
        labels=mask.astype(np.int32),
        exclude_border=False,
    )
    markers = np.zeros(edge.shape, dtype=np.int32)
    if coords.size == 0:
        flat = int(np.argmax(dist))
        markers.flat[flat] = 1
    else:
        for i, (y, x) in enumerate(coords, start=1):
            markers[int(y), int(x)] = i
    labels = watershed(
        np.asarray(edge, dtype=np.float64),
        markers=markers,
        mask=mask,
        compactness=float(compactness),
    )
    return np.asarray(labels, dtype=np.int32)


def _hysteresis_ridges(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    """Binary SoLaCE ridges via hysteresis (strong edges seed, weak edges keep connectivity)."""
    from skimage.measure import label

    mask = np.asarray(valid_mask, dtype=bool)
    vals = np.asarray(edge, dtype=np.float32)[mask]
    if vals.size == 0:
        return np.zeros(edge.shape, dtype=bool)
    lo_p = float(min(low_percentile, high_percentile))
    hi_p = float(max(low_percentile, high_percentile))
    t_lo = float(np.percentile(vals, lo_p))
    t_hi = float(np.percentile(vals, hi_p))
    strong = mask & (edge >= t_hi)
    weak = mask & (edge >= t_lo)
    if not np.any(strong):
        return weak
    # Keep weak components that touch at least one strong pixel.
    comp = label(weak, connectivity=2)
    keep = np.unique(comp[strong])
    keep = keep[keep > 0]
    ridges = np.isin(comp, keep)
    return ridges


def _segment_closed_edges(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
    close_radius: int,
    dilate_radius: int,
    min_area: int,
    ridges: np.ndarray | None = None,
) -> np.ndarray:
    """Turn SoLaCE ridges into object walls: close gaps → label basins → fill ridges."""
    from scipy import ndimage as ndi
    from skimage.measure import label
    from skimage.segmentation import watershed

    mask = np.asarray(valid_mask, dtype=bool)
    if ridges is None:
        ridges = _closed_edge_ridges(
            edge,
            mask,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            close_radius=close_radius,
            dilate_radius=dilate_radius,
        )
    else:
        ridges = np.asarray(ridges, dtype=bool) & mask

    basins = mask & ~ridges
    markers = label(basins, connectivity=1).astype(np.int32)
    if int(markers.max()) == 0:
        # Fallback: whole tissue is one region.
        out = mask.astype(np.int32)
        return out

    # Drop tiny fragments (noise pockets inside texture).
    min_a = max(1, int(min_area))
    counts = np.bincount(markers.ravel())
    keep = np.zeros(counts.shape[0], dtype=bool)
    keep[1:] = counts[1:] >= min_a
    markers_kept = markers.copy()
    markers_kept[~keep[markers]] = 0
    if int(markers_kept.max()) == 0:
        markers_kept = markers

    # Watershed on SoLaCE assigns ridge pixels to nearest basin.
    labels = watershed(
        np.asarray(edge, dtype=np.float64),
        markers=markers_kept,
        mask=mask,
    )
    labels = np.asarray(labels, dtype=np.int32)

    # Relabel densely 1..K
    uniq = np.unique(labels[mask])
    uniq = uniq[uniq > 0]
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for i, u in enumerate(uniq, start=1):
        remap[int(u)] = i
    out = remap[labels]
    out[~mask] = 0
    # Any leftover tissue holes: nearest-label fill
    if np.any(mask & (out == 0)):
        dist, (iy, ix) = ndi.distance_transform_edt(out == 0, return_indices=True)
        filled = out.copy()
        miss = mask & (out == 0)
        filled[miss] = out[iy[miss], ix[miss]]
        out = filled
        out[~mask] = 0
    return out


def _rag_boundary_weights(
    labels: np.ndarray,
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    agg: str = "p90",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjacent region pairs (a,b) with boundary SoLaCE aggregate (mean | max | p90)."""
    lab = np.asarray(labels, dtype=np.int32)
    e = np.asarray(edge, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    max_l = int(lab.max())
    mode = str(agg).strip().lower()
    if max_l <= 1:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )

    lo_parts: list[np.ndarray] = []
    hi_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []

    a = lab[:, :-1]
    b = lab[:, 1:]
    ma = mask[:, :-1] & mask[:, 1:]
    diff = ma & (a > 0) & (b > 0) & (a != b)
    if np.any(diff):
        aa = a[diff]
        bb = b[diff]
        ww = 0.5 * (e[:, :-1][diff] + e[:, 1:][diff])
        lo_parts.append(np.minimum(aa, bb))
        hi_parts.append(np.maximum(aa, bb))
        w_parts.append(ww.astype(np.float32, copy=False))

    a = lab[:-1, :]
    b = lab[1:, :]
    ma = mask[:-1, :] & mask[1:, :]
    diff = ma & (a > 0) & (b > 0) & (a != b)
    if np.any(diff):
        aa = a[diff]
        bb = b[diff]
        ww = 0.5 * (e[:-1, :][diff] + e[1:, :][diff])
        lo_parts.append(np.minimum(aa, bb))
        hi_parts.append(np.maximum(aa, bb))
        w_parts.append(ww.astype(np.float32, copy=False))

    if not lo_parts:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )

    lo = np.concatenate(lo_parts)
    hi = np.concatenate(hi_parts)
    ww = np.concatenate(w_parts)
    stride = max_l + 1
    keys = lo.astype(np.int64) * stride + hi.astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    n_u = int(uniq.size)
    if mode in {"max", "maximum"}:
        acc = np.full(n_u, -np.inf, dtype=np.float64)
        np.maximum.at(acc, inv, ww.astype(np.float64))
        score = acc.astype(np.float32)
    elif mode in {"min", "minimum", "saddle", "waterfall"}:
        # Lowest pass / saddle on the shared boundary (classical watershed hierarchy).
        acc = np.full(n_u, np.inf, dtype=np.float64)
        np.minimum.at(acc, inv, ww.astype(np.float64))
        score = acc.astype(np.float32)
    elif mode in {"p90", "percentile90", "q90"}:
        score = np.zeros(n_u, dtype=np.float32)
        for i in range(n_u):
            score[i] = float(np.percentile(ww[inv == i], 90.0))
    else:
        sums = np.bincount(inv, weights=ww.astype(np.float64), minlength=n_u)
        cnts = np.bincount(inv, minlength=n_u)
        score = (sums / np.maximum(cnts, 1)).astype(np.float32)
    a_out = (uniq // stride).astype(np.int32)
    b_out = (uniq % stride).astype(np.int32)
    return a_out, b_out, score


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def remap_labels(self, labels: np.ndarray) -> np.ndarray:
        lab = np.asarray(labels, dtype=np.int32)
        max_l = int(lab.max())
        root = np.arange(max_l + 1, dtype=np.int32)
        for i in range(1, max_l + 1):
            root[i] = self.find(i)
        # Dense positive ids for live roots
        used = np.unique(root[lab[lab > 0]]) if np.any(lab > 0) else np.zeros(0, dtype=np.int32)
        dense = np.zeros(max_l + 1, dtype=np.int32)
        for i, r in enumerate(used, start=1):
            dense[int(r)] = i
        out = dense[root[lab]]
        out[lab == 0] = 0
        return out.astype(np.int32, copy=False)


def _cut_hierarchy(
    leaf_labels: np.ndarray,
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_regions_targets: list[int],
    level_names: list[str],
    boundary_agg: str = "p90",
) -> dict[str, np.ndarray]:
    """Agglomerative merge of leaf regions by ascending boundary SoLaCE strength."""
    leaves = np.asarray(leaf_labels, dtype=np.int32)
    n_leaves = int(leaves.max())
    if n_leaves <= 0:
        empty = np.zeros_like(leaves)
        return {name: empty.copy() for name in level_names}

    a, b, w = _rag_boundary_weights(leaves, edge, valid_mask, agg=boundary_agg)
    order = np.argsort(w, kind="mergesort")
    a, b, w = a[order], b[order], w[order]

    # Clamp targets into [1, n_leaves], keep unique descending then restore names.
    raw_targets = [max(1, min(int(t), n_leaves)) for t in n_regions_targets]
    if len(level_names) < len(raw_targets):
        level_names = list(level_names) + [f"cut{i}" for i in range(len(level_names), len(raw_targets))]
    level_names = list(level_names[: len(raw_targets)])

    wanted = {t for t in raw_targets}
    uf = _UnionFind(n_leaves + 1)
    n_comp = n_leaves
    snapshots: dict[int, np.ndarray] = {}
    if n_comp in wanted:
        snapshots[n_comp] = uf.remap_labels(leaves)

    min_wanted = min(wanted) if wanted else 1
    for ai, bi in zip(a.tolist(), b.tolist()):
        if n_comp <= min_wanted:
            break
        if not uf.union(int(ai), int(bi)):
            continue
        n_comp -= 1
        if n_comp in wanted:
            snapshots[n_comp] = uf.remap_labels(leaves)

    if n_comp not in snapshots:
        snapshots[n_comp] = uf.remap_labels(leaves)
    available = sorted(snapshots.keys(), reverse=True)

    levels: dict[str, np.ndarray] = {}
    for name, target in zip(level_names, raw_targets):
        if target in snapshots:
            levels[name] = snapshots[target]
            continue
        # Nearest count <= target if possible, else nearest overall.
        below = [c for c in available if c <= target]
        pick = max(below) if below else min(available, key=lambda c: abs(c - target))
        levels[name] = snapshots[pick]
    return levels


@dataclass
class SegResult:
    method: str
    primary: np.ndarray
    levels: dict[str, np.ndarray] = field(default_factory=dict)
    n_leaves: int = 0
    ridges: np.ndarray | None = None

    @property
    def n_regions(self) -> int:
        return int(self.primary.max()) if self.primary.size else 0


def _closed_edge_ridges(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
    close_radius: int,
    dilate_radius: int,
) -> np.ndarray:
    from skimage.morphology import binary_closing, binary_dilation, disk

    mask = np.asarray(valid_mask, dtype=bool)
    ridges = _hysteresis_ridges(
        edge,
        mask,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    r_close = max(0, int(close_radius))
    if r_close > 0:
        ridges = binary_closing(ridges, footprint=disk(r_close))
    r_dil = max(0, int(dilate_radius))
    if r_dil > 0:
        ridges = binary_dilation(ridges, footprint=disk(r_dil))
    return ridges & mask


def _segment_slic_leaves(
    top3_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_segments: int,
    compactness: float,
    sigma: float,
) -> np.ndarray:
    from skimage.segmentation import slic

    mask = np.asarray(valid_mask, dtype=bool)
    img = np.asarray(top3_rgb, dtype=np.float32) / 255.0
    labels = slic(
        img,
        n_segments=max(2, int(n_segments)),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=1,
        mask=mask,
        channel_axis=-1,
    )
    labels = np.asarray(labels, dtype=np.int32)
    labels[~mask] = 0
    return labels


def _segment_hierarchical(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    seg_cfg: DictConfig,
    *,
    top3_rgb: np.ndarray | None = None,
) -> SegResult:
    leaf_mode = str(OmegaConf.select(seg_cfg, "hierarchical_leaf_mode", default="slic")).strip().lower()
    ridges = None
    if leaf_mode in {"slic", "superpixel", "superpixels"}:
        if top3_rgb is None:
            raise ValueError("hierarchical_leaf_mode=slic requires TOP3 RGB")
        leaves = _segment_slic_leaves(
            top3_rgb,
            valid_mask,
            n_segments=int(OmegaConf.select(seg_cfg, "hierarchical_slic_segments", default=220)),
            compactness=float(OmegaConf.select(seg_cfg, "hierarchical_slic_compactness", default=8.0)),
            sigma=float(OmegaConf.select(seg_cfg, "hierarchical_slic_sigma", default=0.0)),
        )
        logger.info("Hierarchical leaves (SLIC on TOP3): %d regions", int(leaves.max()))
    elif leaf_mode in {"closed_edges", "edges", "ridge", "ridges"}:
        low_p = float(OmegaConf.select(seg_cfg, "hierarchical_ridge_low", default=72.0))
        high_p = float(OmegaConf.select(seg_cfg, "hierarchical_ridge_high", default=90.0))
        close_r = int(OmegaConf.select(seg_cfg, "hierarchical_close_radius", default=2))
        dil_r = int(OmegaConf.select(seg_cfg, "hierarchical_dilate_radius", default=1))
        ridges = _closed_edge_ridges(
            edge,
            valid_mask,
            low_percentile=low_p,
            high_percentile=high_p,
            close_radius=close_r,
            dilate_radius=dil_r,
        )
        leaves = _segment_closed_edges(
            edge,
            valid_mask,
            low_percentile=low_p,
            high_percentile=high_p,
            close_radius=close_r,
            dilate_radius=dil_r,
            min_area=int(OmegaConf.select(seg_cfg, "hierarchical_min_area", default=40)),
            ridges=ridges,
        )
        logger.info(
            "Hierarchical leaves (closed_edges): %d regions (ridge_px=%d)",
            int(leaves.max()),
            int(np.count_nonzero(ridges)),
        )
    else:
        min_dist = int(OmegaConf.select(seg_cfg, "hierarchical_min_distance", default=5))
        edge_pct = float(
            OmegaConf.select(
                seg_cfg,
                "hierarchical_edge_percentile",
                default=OmegaConf.select(seg_cfg, "edge_percentile", default=55.0),
            )
        )
        compactness = float(OmegaConf.select(seg_cfg, "hierarchical_compactness", default=0.001))
        leaves = _segment_watershed(
            edge,
            valid_mask,
            edge_percentile=edge_pct,
            min_distance=min_dist,
            compactness=compactness,
        )
        logger.info(
            "Hierarchical leaves (watershed): %d regions (min_distance=%d)",
            int(leaves.max()),
            min_dist,
        )

    n_leaves = int(leaves.max())

    raw_ns = OmegaConf.select(seg_cfg, "hierarchical_n_regions", default=[100, 35, 12])
    n_targets = [int(x) for x in list(OmegaConf.to_container(raw_ns, resolve=True))]
    raw_names = OmegaConf.select(seg_cfg, "hierarchical_level_names", default=["fine", "medium", "coarse"])
    names = [str(x) for x in list(OmegaConf.to_container(raw_names, resolve=True))]
    if len(names) < len(n_targets):
        names = names + [f"cut{i}" for i in range(len(names), len(n_targets))]
    names = names[: len(n_targets)]

    boundary_agg = str(OmegaConf.select(seg_cfg, "hierarchical_boundary_agg", default="p90"))
    levels = _cut_hierarchy(
        leaves,
        edge,
        valid_mask,
        n_regions_targets=n_targets,
        level_names=names,
        boundary_agg=boundary_agg,
    )
    # Ensure finest requested cut is at least the leaf map when target >= n_leaves.
    for name, target in zip(names, n_targets):
        if target >= n_leaves:
            levels[name] = leaves.copy()

    primary_name = str(OmegaConf.select(seg_cfg, "hierarchical_primary", default=names[min(1, len(names) - 1)]))
    if primary_name not in levels:
        primary_name = names[min(1, len(names) - 1)] if names else "fine"
        if primary_name not in levels:
            primary_name = next(iter(levels))
    primary = levels[primary_name]

    for name in names:
        logger.info("  hierarchy[%s]: %d regions", name, int(levels[name].max()))

    return SegResult(
        method="hierarchical",
        primary=primary,
        levels=levels,
        n_leaves=n_leaves,
        ridges=ridges,
    )


def _pixel_graph_edges(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    connectivity: int = 4,
    weight_mode: str = "max",
    top3_rgb: np.ndarray | None = None,
    color_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4/8-connected tissue graph. Returns (node_u, node_v, weights).

    Edge weight is SoLaCE barrier strength (high = hard to merge across).
    Optional TOP3 color distance is added when ``color_weight > 0``.
    """
    mask = np.asarray(valid_mask, dtype=bool)
    e = np.asarray(edge, dtype=np.float32)
    h, w = mask.shape
    node = -np.ones((h, w), dtype=np.int32)
    ys, xs = np.nonzero(mask)
    node[ys, xs] = np.arange(ys.size, dtype=np.int32)
    if ys.size == 0:
        z = np.zeros(0, dtype=np.int32)
        return z, z, np.zeros(0, dtype=np.float32)

    mode = str(weight_mode).strip().lower()
    cw = float(max(0.0, color_weight))
    color = None
    if cw > 0.0 and top3_rgb is not None:
        color = np.asarray(top3_rgb, dtype=np.float32) / 255.0

    offsets = [(0, 1), (1, 0)]
    if int(connectivity) >= 8:
        offsets.extend([(1, 1), (1, -1)])

    u_parts: list[np.ndarray] = []
    v_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    for dy, dx in offsets:
        y0 = max(0, -dy)
        y1 = h - max(0, dy)
        x0 = max(0, -dx)
        x1 = w - max(0, dx)
        a_ids = node[y0:y1, x0:x1]
        b_ids = node[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        ea = e[y0:y1, x0:x1]
        eb = e[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        ok = (a_ids >= 0) & (b_ids >= 0)
        if not np.any(ok):
            continue
        aa = a_ids[ok]
        bb = b_ids[ok]
        if mode in {"mean", "avg", "average"}:
            ww = 0.5 * (ea[ok] + eb[ok])
        elif mode in {"min", "minimum"}:
            ww = np.minimum(ea[ok], eb[ok])
        else:
            ww = np.maximum(ea[ok], eb[ok])
        # Scale SoLaCE [0,1] → ~[0,255] so Felzenszwalb k matches classic ranges.
        ww = (np.asarray(ww, dtype=np.float32) * np.float32(255.0))
        if color is not None:
            ca = color[y0:y1, x0:x1][ok]
            cb = color[y0 + dy : y1 + dy, x0 + dx : x1 + dx][ok]
            # Color distance in ~[0,255] units as well.
            d = np.linalg.norm(ca - cb, axis=-1).astype(np.float32) * np.float32(255.0)
            ww = ww + np.float32(cw) * d
        u_parts.append(aa.astype(np.int32, copy=False))
        v_parts.append(bb.astype(np.int32, copy=False))
        w_parts.append(ww)

    if not u_parts:
        z = np.zeros(0, dtype=np.int32)
        return z, z, np.zeros(0, dtype=np.float32)
    return np.concatenate(u_parts), np.concatenate(v_parts), np.concatenate(w_parts)


def _labels_from_parents(
    parent: np.ndarray,
    find,
    mask: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    n = int(parent.size)
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int32, count=n)
    _, inv = np.unique(roots, return_inverse=True)
    labels = np.zeros(shape, dtype=np.int32)
    ys, xs = np.nonzero(mask)
    labels[ys, xs] = inv.astype(np.int32) + 1
    return labels


def _felzenszwalb_solace_multiscale(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    *,
    scales: list[float],
    level_names: list[str],
    min_size: int,
    connectivity: int = 4,
    weight_mode: str = "max",
    top3_rgb: np.ndarray | None = None,
    color_weight: float = 0.0,
) -> dict[str, np.ndarray]:
    """One Kruskal pass; snapshot labels at each Felzenszwalb scale k (nested cuts).

    Uses the largest k for the merge predicate while recording when each merge first
    becomes legal for smaller k values — implemented as separate passes over the same
    sorted edges for clarity/correctness (still O(E log E) once for sort).
    """
    u, v, w = _pixel_graph_edges(
        edge,
        valid_mask,
        connectivity=connectivity,
        weight_mode=weight_mode,
        top3_rgb=top3_rgb,
        color_weight=color_weight,
    )
    mask = np.asarray(valid_mask, dtype=bool)
    n = int(mask.sum())
    shape = edge.shape
    if n == 0:
        empty = np.zeros(shape, dtype=np.int32)
        return {name: empty.copy() for name in level_names}

    order = np.argsort(w, kind="mergesort")
    min_sz = max(1, int(min_size))
    levels: dict[str, np.ndarray] = {}

    # Process from largest k → smallest so we can also offer coarsest first; store by name.
    # Each scale is a full FH run sharing the same sorted edges (correct FH semantics).
    for name, scale in zip(level_names, scales):
        parent = np.arange(n, dtype=np.int32)
        size = np.ones(n, dtype=np.int32)
        inten = np.zeros(n, dtype=np.float32)
        k = float(max(1e-6, scale))

        def find(x: int, _p=parent) -> int:
            while _p[x] != x:
                _p[x] = _p[_p[x]]
                x = int(_p[x])
            return x

        for idx in order:
            a = find(int(u[idx]))
            b = find(int(v[idx]))
            if a == b:
                continue
            wij = float(w[idx])
            if wij <= min(inten[a] + k / float(size[a]), inten[b] + k / float(size[b])):
                if size[a] < size[b]:
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]
                inten[a] = max(inten[a], inten[b], wij)

        if min_sz > 1 and u.size:
            for idx in order:
                a = find(int(u[idx]))
                b = find(int(v[idx]))
                if a == b:
                    continue
                if size[a] < min_sz or size[b] < min_sz:
                    if size[a] < size[b]:
                        a, b = b, a
                    parent[b] = a
                    size[a] += size[b]
                    inten[a] = max(inten[a], inten[b], float(w[idx]))

        levels[name] = _labels_from_parents(parent, find, mask, shape)

    return levels


def _segment_graph(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    seg_cfg: DictConfig,
    *,
    top3_rgb: np.ndarray | None = None,
) -> SegResult:
    """Multi-scale SoLaCE graph segmentation (Felzenszwalb k ladder)."""
    raw_scales = OmegaConf.select(seg_cfg, "graph_scales", default=[5, 30, 150])
    scales = [float(x) for x in list(OmegaConf.to_container(raw_scales, resolve=True))]
    raw_names = OmegaConf.select(seg_cfg, "graph_level_names", default=["fine", "medium", "coarse"])
    names = [str(x) for x in list(OmegaConf.to_container(raw_names, resolve=True))]
    if len(names) < len(scales):
        names = names + [f"k{i}" for i in range(len(names), len(scales))]
    names = names[: len(scales)]

    min_size = int(OmegaConf.select(seg_cfg, "graph_min_size", default=3))
    connectivity = int(OmegaConf.select(seg_cfg, "graph_connectivity", default=4))
    weight_mode = str(OmegaConf.select(seg_cfg, "graph_weight", default="max"))
    color_weight = float(OmegaConf.select(seg_cfg, "graph_color_weight", default=0.0))

    levels = _felzenszwalb_solace_multiscale(
        edge,
        valid_mask,
        scales=scales,
        level_names=names,
        min_size=min_size,
        connectivity=connectivity,
        weight_mode=weight_mode,
        top3_rgb=top3_rgb,
        color_weight=color_weight,
    )
    for name, scale in zip(names, scales):
        logger.info("  graph[%s] k=%.1f: %d regions", name, scale, int(levels[name].max()))

    primary_name = str(OmegaConf.select(seg_cfg, "graph_primary", default=names[0] if names else "fine"))
    if primary_name not in levels:
        primary_name = names[0]
    fine_n = int(levels[names[0]].max()) if names else 0
    return SegResult(
        method="graph",
        primary=levels[primary_name],
        levels=levels,
        n_leaves=fine_n,
    )


def _region_mean_rgb(
    labels: np.ndarray,
    rgb: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Mean RGB in [0,1] for labels 0..max (row 0 unused)."""
    lab = np.asarray(labels, dtype=np.int32)
    img = np.asarray(rgb, dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    mask = np.asarray(valid_mask, dtype=bool)
    n = int(lab.max())
    means = np.zeros((n + 1, 3), dtype=np.float32)
    if n <= 0:
        return means
    flat_l = lab[mask]
    for c in range(3):
        sums = np.bincount(flat_l, weights=img[:, :, c][mask], minlength=n + 1)
        means[:, c] = sums.astype(np.float32)
    counts = np.bincount(flat_l, minlength=n + 1).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    means /= counts[:, None]
    return means


def _rag_merge_by_color_and_barrier(
    leaf_labels: np.ndarray,
    edge: np.ndarray,
    top3_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_regions_targets: list[int],
    level_names: list[str],
    barrier_agg: str,
    barrier_max: float,
    barrier_weight: float,
) -> dict[str, np.ndarray]:
    """Stage-2 RAG: merge SoLaCE leaves if boundary is weak and TOP3 colors are similar."""
    leaves = np.asarray(leaf_labels, dtype=np.int32)
    n_leaves = int(leaves.max())
    if n_leaves <= 0:
        empty = np.zeros_like(leaves)
        return {name: empty.copy() for name in level_names}

    a, b, barrier = _rag_boundary_weights(leaves, edge, valid_mask, agg=barrier_agg)
    means = _region_mean_rgb(leaves, top3_rgb, valid_mask)
    if a.size == 0:
        return {name: leaves.copy() for name in level_names}

    color_d = np.linalg.norm(means[a] - means[b], axis=1).astype(np.float32)
    # Soft SoLaCE prior: prefer weak barriers + similar TOP3.
    # Optional hard gate (barrier_max < 1) excludes very strong walls from merging.
    bmax = float(barrier_max)
    if bmax < 0.999:
        weak = barrier <= bmax
        if not np.any(weak):
            logger.warning(
                "RAG: no weak boundaries (barrier_max=%.3f); disabling hard gate",
                bmax,
            )
            weak = np.ones(barrier.shape, dtype=bool)
    else:
        weak = np.ones(barrier.shape, dtype=bool)

    a_w, b_w = a[weak], b[weak]
    cost = color_d[weak] + float(barrier_weight) * barrier[weak].astype(np.float32)
    order = np.argsort(cost, kind="mergesort")
    a_w, b_w, cost = a_w[order], b_w[order], cost[order]

    raw_targets = []
    for t in n_regions_targets:
        ti = int(t)
        if ti <= 0 or ti >= n_leaves:
            raw_targets.append(n_leaves)
        else:
            raw_targets.append(max(1, ti))

    names = list(level_names[: len(raw_targets)])
    wanted = {t for t in raw_targets}
    uf = _UnionFind(n_leaves + 1)
    n_comp = n_leaves
    snapshots: dict[int, np.ndarray] = {}
    if n_comp in wanted:
        snapshots[n_comp] = uf.remap_labels(leaves)

    min_wanted = min(wanted) if wanted else 1
    for ai, bi in zip(a_w.tolist(), b_w.tolist()):
        if n_comp <= min_wanted:
            break
        if not uf.union(int(ai), int(bi)):
            continue
        n_comp -= 1
        if n_comp in wanted:
            snapshots[n_comp] = uf.remap_labels(leaves)

    if n_comp not in snapshots:
        snapshots[n_comp] = uf.remap_labels(leaves)
    available = sorted(snapshots.keys(), reverse=True)

    levels: dict[str, np.ndarray] = {}
    for name, target in zip(names, raw_targets):
        if target in snapshots:
            levels[name] = snapshots[target]
            continue
        below = [c for c in available if c <= target]
        pick = max(below) if below else min(available, key=lambda c: abs(c - target))
        levels[name] = snapshots[pick]
    return levels


def _segment_graph_twostage(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    seg_cfg: DictConfig,
    *,
    top3_rgb: np.ndarray | None = None,
) -> SegResult:
    """Stage1: dense SoLaCE Felzenszwalb objects. Stage2: TOP3 RAG across weak walls."""
    if top3_rgb is None:
        raise ValueError("graph_twostage requires TOP3 RGB for RAG color merges")

    # Stage 1 — keep every ridge-separated pocket (small k).
    raw_scales = OmegaConf.select(seg_cfg, "graph_scales", default=[5])
    scales = [float(x) for x in list(OmegaConf.to_container(raw_scales, resolve=True))]
    stage1_k = float(scales[0]) if scales else 5.0
    leaves = _felzenszwalb_solace_multiscale(
        edge,
        valid_mask,
        scales=[stage1_k],
        level_names=["leaves"],
        min_size=int(OmegaConf.select(seg_cfg, "graph_min_size", default=3)),
        connectivity=int(OmegaConf.select(seg_cfg, "graph_connectivity", default=4)),
        weight_mode=str(OmegaConf.select(seg_cfg, "graph_weight", default="max")),
        top3_rgb=None,
        color_weight=0.0,
    )["leaves"]
    n_leaves = int(leaves.max())
    logger.info("Two-stage graph: stage1 SoLaCE leaves = %d (k=%.1f)", n_leaves, stage1_k)

    raw_names = OmegaConf.select(seg_cfg, "graph_level_names", default=["fine", "medium", "coarse"])
    names = [str(x) for x in list(OmegaConf.to_container(raw_names, resolve=True))]
    raw_ns = OmegaConf.select(seg_cfg, "graph_rag_n_regions", default=[0, 400, 100])
    n_targets = [int(x) for x in list(OmegaConf.to_container(raw_ns, resolve=True))]
    if len(names) < len(n_targets):
        names = names + [f"cut{i}" for i in range(len(names), len(n_targets))]
    names = names[: len(n_targets)]

    barrier_max = float(OmegaConf.select(seg_cfg, "graph_rag_barrier_max", default=0.35))
    barrier_weight = float(OmegaConf.select(seg_cfg, "graph_rag_barrier_weight", default=1.0))
    barrier_agg = str(OmegaConf.select(seg_cfg, "graph_rag_barrier_agg", default="p90"))

    levels = _rag_merge_by_color_and_barrier(
        leaves,
        edge,
        top3_rgb,
        valid_mask,
        n_regions_targets=n_targets,
        level_names=names,
        barrier_agg=barrier_agg,
        barrier_max=barrier_max,
        barrier_weight=barrier_weight,
    )
    # Always expose raw leaves if fine target kept all.
    for name in names:
        logger.info("  graph_twostage[%s]: %d regions", name, int(levels[name].max()))

    primary_name = str(OmegaConf.select(seg_cfg, "graph_primary", default=names[min(1, len(names) - 1)]))
    if primary_name not in levels:
        primary_name = names[0]
    return SegResult(
        method="graph_twostage",
        primary=levels[primary_name],
        levels=levels,
        n_leaves=n_leaves,
    )


def _compute_soft_landmark_signatures(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rank_config_path: str,
) -> np.ndarray:
    """Soft landmark signatures α(x) as float32 array (n_valid, n_landmarks), SoLaCE front-end."""
    from hydra.utils import to_absolute_path
    from sklearn.decomposition import PCA

    from debug_edge_maps import _get_landmark_indices, _landmark_knn_soft_signatures

    rank_path = Path(to_absolute_path(str(rank_config_path)))
    if not rank_path.is_file():
        raise FileNotFoundError(f"rank_config not found: {rank_path}")
    rank_cfg = OmegaConf.load(rank_path)
    edge_cfg = getattr(rank_cfg, "edge_detection", None)
    lk = getattr(edge_cfg, "soft_landmark_contrast", None) if edge_cfg is not None else None
    if lk is None:
        lk = OmegaConf.create({})

    mask = np.asarray(valid_mask, dtype=bool)
    x = np.asarray(msi[mask], dtype=np.float32)
    if x.size == 0:
        return np.zeros((0, 1), dtype=np.float32)

    pca_enabled = bool(OmegaConf.select(lk, "pca_enabled", default=True))
    pca_n = int(OmegaConf.select(lk, "pca_n_components", default=64))
    pca_seed = int(OmegaConf.select(lk, "pca_random_state", default=42))
    if pca_enabled and x.shape[1] > pca_n:
        n_comp = min(pca_n, x.shape[1], x.shape[0] - 1)
        x = PCA(n_components=n_comp, random_state=pca_seed).fit_transform(x).astype(np.float32)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    x_norm = (x / norms).astype(np.float32)

    n_landmarks = min(int(OmegaConf.select(lk, "n_landmarks", default=1000)), int(x_norm.shape[0]))
    sampling = str(OmegaConf.select(lk, "landmark_sampling", default="random"))
    rng = int(OmegaConf.select(lk, "random_state", default=42))
    landmark_idx = _get_landmark_indices(x_norm, n_landmarks, sampling, rng, None)
    landmark_vectors = x_norm[landmark_idx]
    sim = x_norm @ landmark_vectors.T
    soft_topk = int(OmegaConf.select(lk, "soft_topk", default=80))
    soft_temp = float(OmegaConf.select(lk, "soft_temperature", default=0.06))
    signatures = _landmark_knn_soft_signatures(sim, soft_topk, soft_temp)
    logger.info(
        "Landmark signatures: n_valid=%d n_landmarks=%d topk=%d temp=%.3f",
        int(signatures.shape[0]),
        int(signatures.shape[1]),
        soft_topk,
        soft_temp,
    )
    return np.asarray(signatures, dtype=np.float32)


def _signatures_to_feature_volume(
    signatures: np.ndarray,
    valid_mask: np.ndarray,
    *,
    pca_dims: int,
    random_state: int,
) -> np.ndarray:
    """Pack α into H×W×C feature volume (PCA-reduced) for random walker."""
    from sklearn.decomposition import PCA

    mask = np.asarray(valid_mask, dtype=bool)
    h, w = mask.shape
    n_valid, n_feat = signatures.shape
    dims = int(pca_dims)
    feats = signatures
    if dims > 0 and n_feat > dims and n_valid > dims + 1:
        n_comp = min(dims, n_feat, n_valid - 1)
        feats = PCA(n_components=n_comp, random_state=int(random_state)).fit_transform(signatures)
        feats = feats.astype(np.float32)
    # Standardize channels for RW beta scale.
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    feats = (feats - mu) / sd

    vol = np.zeros((h, w, feats.shape[1]), dtype=np.float32)
    vol[mask] = feats
    return vol


def _kmeans_labels_on_signatures(
    signatures: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels HxW in 1..K, centers (K, F) in signature space)."""
    from sklearn.cluster import MiniBatchKMeans

    mask = np.asarray(valid_mask, dtype=bool)
    n_valid = int(signatures.shape[0])
    k = max(2, min(int(n_clusters), n_valid))
    sample_n = min(n_valid, max(k * 20, int(sample_size)))
    rng = np.random.default_rng(int(random_state))
    if sample_n < n_valid:
        idx = rng.choice(n_valid, size=sample_n, replace=False)
        fit_x = signatures[idx]
    else:
        fit_x = signatures
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(random_state),
        batch_size=min(8192, max(4608, fit_x.shape[0] // 4)),
        n_init=3,
        max_iter=200,
    )
    km.fit(fit_x)
    pred = km.predict(signatures).astype(np.int32) + 1  # 1..K
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels[mask] = pred
    centers = np.asarray(km.cluster_centers_, dtype=np.float32)
    return labels, centers


def _seeds_from_labels(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    seed_frac: float,
    min_distance: int,
) -> np.ndarray:
    """Interior seeds per class via distance transform (0 = unlabeled for random walker)."""
    from scipy import ndimage as ndi

    mask = np.asarray(valid_mask, dtype=bool)
    lab = np.asarray(labels, dtype=np.int32)
    seeds = np.zeros(lab.shape, dtype=np.int32)
    frac = float(np.clip(seed_frac, 0.02, 0.8))
    min_d = max(1, int(min_distance))

    for k in range(1, int(lab.max()) + 1):
        region = (lab == k) & mask
        n_reg = int(region.sum())
        if n_reg == 0:
            continue
        dist = ndi.distance_transform_edt(region)
        # Keep the most interior fraction as seeds.
        vals = dist[region]
        if vals.size == 0:
            continue
        thr = float(np.quantile(vals, 1.0 - frac))
        thr = max(thr, float(min_d) * 0.5)
        core = region & (dist >= thr)
        if not np.any(core):
            # Fallback: farthest pixel(s)
            flat = np.argmax(dist)
            cy, cx = divmod(int(flat), dist.shape[1])
            seeds[cy, cx] = k
            continue
        # Thin seeds with min distance via greedy peaks on dist.
        dist_core = dist.copy()
        dist_core[~core] = 0.0
        coords = []
        dist_work = dist_core.copy()
        while np.any(dist_work > 0):
            flat = int(np.argmax(dist_work))
            y, x = divmod(flat, dist_work.shape[1])
            if dist_work[y, x] <= 0:
                break
            coords.append((y, x))
            y0 = max(0, y - min_d)
            y1 = min(dist_work.shape[0], y + min_d + 1)
            x0 = max(0, x - min_d)
            x1 = min(dist_work.shape[1], x + min_d + 1)
            dist_work[y0:y1, x0:x1] = 0.0
            if len(coords) >= max(1, n_reg // 50):
                break
        for y, x in coords:
            seeds[y, x] = k
        # Ensure at least one seed
        if not np.any(seeds == k):
            flat = int(np.argmax(dist))
            y, x = divmod(flat, dist.shape[1])
            seeds[y, x] = k

    seeds[~mask] = 0
    return seeds


def _affinity_graph_laplacian(
    feature_vol: np.ndarray,
    mask: np.ndarray,
    *,
    beta: float,
):
    """4-connected sparse Laplacian on tissue pixels; returns (L, ys, xs)."""
    from scipy import sparse

    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    node = -np.ones((h, w), dtype=np.int32)
    node[ys, xs] = np.arange(ys.size, dtype=np.int32)
    n = int(ys.size)
    feat = np.asarray(feature_vol, dtype=np.float64)
    beta_f = float(max(1e-6, beta))
    n_ch = max(int(feat.shape[-1]), 1)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for dy, dx in ((0, 1), (1, 0)):
        y0, y1 = max(0, -dy), h - max(0, dy)
        x0, x1 = max(0, -dx), w - max(0, dx)
        a = node[y0:y1, x0:x1]
        b = node[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        ok = (a >= 0) & (b >= 0)
        if not np.any(ok):
            continue
        ia = a[ok].astype(np.int32)
        ib = b[ok].astype(np.int32)
        fa = feat[y0:y1, x0:x1][ok]
        fb = feat[y0 + dy : y1 + dy, x0 + dx : x1 + dx][ok]
        dist2 = np.sum((fa - fb) ** 2, axis=-1)
        wij = np.exp(-beta_f * dist2 / float(n_ch)).astype(np.float64)
        # L_ij = -w, L_ii += w
        rows.extend([ia, ib, ia, ib])
        cols.extend([ib, ia, ia, ib])
        vals.extend([-wij, -wij, wij, wij])

    if not rows:
        return sparse.csr_matrix((n, n)), ys, xs
    L = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    return L, ys, xs


def _random_walker_refine(
    feature_vol: np.ndarray,
    seeds: np.ndarray,
    valid_mask: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    """Multi-label random walker on landmark features (Grady-style, scipy CG)."""
    from scipy.sparse.linalg import cg

    mask = np.asarray(valid_mask, dtype=bool)
    seed_img = np.asarray(seeds, dtype=np.int32).copy()
    seed_img[~mask] = 0
    if not np.any(seed_img > 0):
        return np.zeros(mask.shape, dtype=np.int32)

    labels_present = np.unique(seed_img[seed_img > 0])
    L, ys, xs = _affinity_graph_laplacian(feature_vol, mask, beta=beta)
    seed_flat = seed_img[ys, xs]
    is_seed = seed_flat > 0
    is_unlab = ~is_seed
    if not np.any(is_unlab):
        out = seed_img.copy()
        out[~mask] = 0
        return out

    from scipy import sparse as sp

    probs = np.zeros((int(is_unlab.sum()), int(labels_present.size)), dtype=np.float64)
    Luu = L[is_unlab][:, is_unlab]
    # Stabilize CG on weakly connected components.
    Luu = Luu + sp.eye(Luu.shape[0], format="csr") * 1e-6
    Lub = L[is_unlab][:, is_seed]
    seed_labs = seed_flat[is_seed]

    for j, lab in enumerate(labels_present):
        rhs = -Lub @ (seed_labs == int(lab)).astype(np.float64)
        x0 = np.zeros(int(is_unlab.sum()), dtype=np.float64)
        try:
            sol, info = cg(Luu, rhs, x0=x0, rtol=1e-3, atol=0.0, maxiter=400)
        except TypeError:
            sol, info = cg(Luu, rhs, x0=x0, tol=1e-3, maxiter=400)
        if info != 0:
            try:
                sol, _ = cg(Luu, rhs, x0=sol, rtol=1e-2, atol=0.0, maxiter=800)
            except TypeError:
                sol, _ = cg(Luu, rhs, x0=sol, tol=1e-2, maxiter=800)
        probs[:, j] = np.clip(sol, 0.0, 1.0)

    winners = labels_present[np.argmax(probs, axis=1)]
    out = np.zeros(mask.shape, dtype=np.int32)
    out[ys[is_seed], xs[is_seed]] = seed_flat[is_seed]
    out[ys[is_unlab], xs[is_unlab]] = winners.astype(np.int32)
    out[~mask] = 0
    return out


def _tissue_neighbor_edges(valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4-connected tissue edges as flat node indices (u, v) and pixel coords helper."""
    mask = np.asarray(valid_mask, dtype=bool)
    h, w = mask.shape
    node = -np.ones((h, w), dtype=np.int32)
    ys, xs = np.nonzero(mask)
    node[ys, xs] = np.arange(ys.size, dtype=np.int32)
    u_parts: list[np.ndarray] = []
    v_parts: list[np.ndarray] = []
    for dy, dx in ((0, 1), (1, 0)):
        a = node[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
        b = node[max(0, dy) : h - max(0, -dy), max(0, dx) : w - max(0, -dx)]
        ok = (a >= 0) & (b >= 0)
        if np.any(ok):
            u_parts.append(a[ok].astype(np.int32, copy=False))
            v_parts.append(b[ok].astype(np.int32, copy=False))
    if not u_parts:
        z = np.zeros(0, dtype=np.int32)
        return z, z, node
    return np.concatenate(u_parts), np.concatenate(v_parts), node


def _pairwise_weights_from_features(
    feature_vol: np.ndarray,
    valid_mask: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    beta: float,
    pairwise_scale: float,
    edge: np.ndarray | None = None,
    edge_gamma: float = 1.5,
) -> np.ndarray:
    """Contrast-sensitive Potts weights.

    High weight → prefer same label (smooth interiors).
    Optional SoLaCE ``edge`` lowers weight across ridges so cuts follow anatomy.
    """
    mask = np.asarray(valid_mask, dtype=bool)
    feats = np.asarray(feature_vol[mask], dtype=np.float64)
    n_ch = max(int(feats.shape[1]), 1)
    d2 = np.sum((feats[u] - feats[v]) ** 2, axis=1)
    w = np.exp(-float(beta) * d2 / float(n_ch))
    if edge is not None:
        e = np.clip(np.asarray(edge, dtype=np.float64), 0.0, 1.0)
        ys, xs = np.nonzero(mask)
        # u,v are flat tissue indices matching ys/xs order from _tissue_neighbor_edges
        barrier = np.maximum(e[ys[u], xs[u]], e[ys[v], xs[v]])
        gate = np.power(np.maximum(1.0 - barrier, 0.0), float(max(0.0, edge_gamma)))
        # Keep a small floor so the graph stays connected.
        w = w * np.maximum(gate, 0.05)
    return (float(pairwise_scale) * w).astype(np.float64)


def _relabel_consecutive(labels: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Remap positive labels to 1..K densely."""
    lab = np.asarray(labels, dtype=np.int32).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    uniq = np.unique(lab[mask])
    uniq = uniq[uniq > 0]
    mapping = np.zeros(int(lab.max()) + 1, dtype=np.int32)
    for i, u in enumerate(uniq, start=1):
        mapping[int(u)] = i
    out = np.zeros_like(lab)
    out[mask] = mapping[lab[mask]]
    return out


def _cleanup_label_map(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    majority_radius: int = 2,
    min_region_area: int = 40,
    trim_rim_px: int = 0,
) -> np.ndarray:
    """Spatial cleanup: modal filter, absorb tiny islands, optional rim reassignment."""
    from scipy import ndimage as ndi
    from skimage.filters.rank import modal
    from skimage.morphology import disk

    lab = np.asarray(labels, dtype=np.int32).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(mask) or int(lab.max()) < 1:
        return lab

    # Modal filter over a disk — kills salt-and-pepper (labels must fit uint8).
    r = max(0, int(majority_radius))
    if r > 0 and int(lab.max()) <= 255:
        footprint = disk(r)
        u8 = lab.astype(np.uint8, copy=False)
        filtered = modal(u8, footprint=footprint, mask=mask).astype(np.int32)
        lab = np.where(mask, filtered, 0)

    # Absorb connected components smaller than min_region_area into neighbor majority.
    min_area = max(1, int(min_region_area))
    if min_area > 1 and int(lab.max()) >= 1:
        out = lab.copy()
        structure = np.ones((3, 3), dtype=bool)
        for lbl in range(1, int(lab.max()) + 1):
            comp = out == lbl
            if not np.any(comp):
                continue
            labeled, ncc = ndi.label(comp, structure=structure)
            if ncc == 0:
                continue
            sizes = ndi.sum(comp, labeled, index=np.arange(1, ncc + 1))
            for ci, sz in enumerate(sizes, start=1):
                if float(sz) >= min_area:
                    continue
                island = labeled == ci
                dil = ndi.binary_dilation(island, structure=structure) & mask & ~island
                neigh = out[dil]
                neigh = neigh[neigh > 0]
                if neigh.size == 0:
                    continue
                out[island] = int(np.bincount(neigh).argmax())
        lab = out

    # Thin tissue rim often forms a fake "capsule" class — map to nearest interior.
    rim = max(0, int(trim_rim_px))
    if rim > 0:
        dist_in = ndi.distance_transform_edt(mask)
        rim_mask = mask & (dist_in <= float(rim))
        interior = mask & ~rim_mask
        if np.any(rim_mask) and np.any(interior):
            _, (iy, ix) = ndi.distance_transform_edt(~interior, return_indices=True)
            lab[rim_mask] = lab[iy[rim_mask], ix[rim_mask]]

    lab[~mask] = 0
    return _relabel_consecutive(lab, mask)


def _unary_from_centers(
    signatures: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Unary costs (n_valid, K): squared distance to each landmark-cluster center."""
    # ||x||^2 + ||c||^2 - 2 x·c
    x2 = np.sum(signatures.astype(np.float64) ** 2, axis=1, keepdims=True)
    c2 = np.sum(centers.astype(np.float64) ** 2, axis=1)[None, :]
    xc = signatures.astype(np.float64) @ centers.astype(np.float64).T
    return np.maximum(x2 + c2 - 2.0 * xc, 0.0)


def _add_pairwise_potts_term(graph, p: int, q: int, e00: float, e01: float, e10: float, e11: float) -> None:
    """Add pairwise binary term (Boykov–Kolmogorov). source=0, sink=1; needs submodularity."""
    # Matches Energy::add_term2 from the BK maxflow energy minimization code.
    graph.add_tedge(p, float(e11), float(e00))
    e01 = float(e01) - float(e00)
    e10 = float(e10) - float(e11)
    if e01 < 0.0:
        graph.add_tedge(p, 0.0, e01)
        graph.add_tedge(q, 0.0, -e01)
        e10 += e01
        e01 = 0.0
    elif e10 < 0.0:
        graph.add_tedge(p, -e10, 0.0)
        graph.add_tedge(q, -e10, 0.0)
        e01 += e10
        e10 = 0.0
    # Now e01 >= 0, e10 >= 0
    if e01 > 0.0 or e10 > 0.0:
        graph.add_edge(p, q, e01, e10)


def _alpha_expansion_move(
    labels: np.ndarray,
    alpha: int,
    unary: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """One α-expansion move (Potts) via s-t min-cut. labels are 0..K-1.

    Only non-α pixels are graph nodes (α pixels are fixed to the sink label).
    """
    import maxflow

    labels = np.asarray(labels, dtype=np.int32)
    n = int(labels.shape[0])
    active = np.flatnonzero(labels != alpha).astype(np.int32, copy=False)
    if active.size == 0:
        return labels.copy()

    node_of = np.full(n, -1, dtype=np.int32)
    node_of[active] = np.arange(active.size, dtype=np.int32)

    g = maxflow.Graph[float](int(active.size), int(u.size) + int(active.size))
    nodes = g.add_nodes(int(active.size))

    # t-links: source=keep current, sink=take alpha
    # E(sink)=unary[α], E(source)=unary[fp]  => add_tedge(cap_source, cap_sink)
    fp_active = labels[active]
    src_caps = unary[active, alpha].astype(np.float64, copy=False)
    snk_caps = unary[active, fp_active].astype(np.float64, copy=False)
    for i in range(active.size):
        g.add_tedge(int(nodes[i]), float(src_caps[i]), float(snk_caps[i]))

    # Neighbor terms. Reduce edges touching fixed-α nodes into unary on the active side.
    wu = weights.astype(np.float64, copy=False)
    keep = wu > 1e-15
    if np.any(keep):
        uu = u[keep].astype(np.int32, copy=False)
        vv = v[keep].astype(np.int32, copy=False)
        ww = wu[keep]
        nu = node_of[uu]
        nv = node_of[vv]
        both = (nu >= 0) & (nv >= 0)
        only_u = (nu >= 0) & (nv < 0)
        only_v = (nu < 0) & (nv >= 0)

        if np.any(both):
            ub = uu[both]
            vb = vv[both]
            wb = ww[both]
            same = labels[ub] == labels[vb]
            # same non-α: binary edge; different: Potts E00=E01=E10=w, E11=0
            for p_i, q_i, w, same_lab in zip(
                nu[both].tolist(),
                nv[both].tolist(),
                wb.tolist(),
                same.tolist(),
            ):
                w = float(w)
                if same_lab:
                    g.add_edge(int(nodes[p_i]), int(nodes[q_i]), w, w)
                else:
                    _add_pairwise_potts_term(g, int(nodes[p_i]), int(nodes[q_i]), w, w, w, 0.0)

        # Neighbor already α: keeping current costs w, taking α costs 0
        # => add to E(source) by increasing cap_sink
        if np.any(only_u):
            for p_i, w in zip(nu[only_u].tolist(), ww[only_u].tolist()):
                g.add_tedge(int(nodes[p_i]), 0.0, float(w))
        if np.any(only_v):
            for q_i, w in zip(nv[only_v].tolist(), ww[only_v].tolist()):
                g.add_tedge(int(nodes[q_i]), 0.0, float(w))

    g.maxflow()
    out = labels.copy()
    take = np.fromiter(
        (g.get_segment(int(nodes[i])) == 1 for i in range(active.size)),
        dtype=bool,
        count=active.size,
    )
    out[active[take]] = alpha
    return out


def _alpha_expansion_potts(
    init_labels_0: np.ndarray,
    unary: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    *,
    max_iter: int = 4,
) -> np.ndarray:
    """Boykov α-expansion for Potts energy until convergence or max_iter."""
    labels = np.asarray(init_labels_0, dtype=np.int32).copy()
    n_labels = int(unary.shape[1])
    for it in range(max(1, int(max_iter))):
        changed = False
        order = np.arange(n_labels)
        np.random.shuffle(order)
        for alpha in order:
            new_lab = _alpha_expansion_move(labels, int(alpha), unary, u, v, weights)
            if np.any(new_lab != labels):
                labels = new_lab
                changed = True
        logger.info("  alpha-expansion iter %d/%d changed=%s", it + 1, max_iter, changed)
        if not changed:
            break
    return labels


def _segment_landmark_graphcut(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    seg_cfg: DictConfig,
    *,
    rank_config_path: str,
    random_state: int = 17,
    edge: np.ndarray | None = None,
) -> SegResult:
    """Landmark soft signatures → k-means types → RW or α-expansion Potts refine."""
    signatures = _compute_soft_landmark_signatures(
        msi,
        valid_mask,
        rank_config_path=rank_config_path,
    )
    pca_dims = int(OmegaConf.select(seg_cfg, "lgc_pca_dims", default=32))
    feat_vol = _signatures_to_feature_volume(
        signatures,
        valid_mask,
        pca_dims=pca_dims,
        random_state=random_state,
    )

    raw_ks = OmegaConf.select(seg_cfg, "lgc_n_clusters", default=[40, 18, 8])
    n_clusters_list = [int(x) for x in list(OmegaConf.to_container(raw_ks, resolve=True))]
    raw_names = OmegaConf.select(seg_cfg, "lgc_level_names", default=["fine", "medium", "coarse"])
    names = [str(x) for x in list(OmegaConf.to_container(raw_names, resolve=True))]
    if len(names) < len(n_clusters_list):
        names = names + [f"k{i}" for i in range(len(names), len(n_clusters_list))]
    names = names[: len(n_clusters_list)]

    beta_rw = float(OmegaConf.select(seg_cfg, "lgc_beta", default=90.0))
    seed_frac = float(OmegaConf.select(seg_cfg, "lgc_seed_frac", default=0.12))
    seed_min_d = int(OmegaConf.select(seg_cfg, "lgc_seed_min_distance", default=3))
    sample_size = int(OmegaConf.select(seg_cfg, "lgc_kmeans_sample", default=20000))
    optimizer = str(OmegaConf.select(seg_cfg, "lgc_optimizer", default="alpha_expansion")).strip().lower()
    pairwise_scale = float(OmegaConf.select(seg_cfg, "lgc_pairwise_scale", default=100.0))
    # RW uses a sharp affinity (high beta). α-expansion needs a milder beta or
    # almost all Potts weights collapse to ~0 and labels stay speckled.
    pairwise_beta = float(
        OmegaConf.select(seg_cfg, "lgc_pairwise_beta", default=2.0)
    )
    edge_gamma = float(OmegaConf.select(seg_cfg, "lgc_edge_gamma", default=1.5))
    use_edge_barrier = bool(OmegaConf.select(seg_cfg, "lgc_use_edge_barrier", default=True))
    alpha_iters = int(OmegaConf.select(seg_cfg, "lgc_alpha_iterations", default=4))
    skip_rw = bool(OmegaConf.select(seg_cfg, "lgc_skip_random_walker", default=False))
    cleanup = bool(OmegaConf.select(seg_cfg, "lgc_cleanup", default=True))
    majority_r = int(OmegaConf.select(seg_cfg, "lgc_majority_radius", default=3))
    min_area = int(OmegaConf.select(seg_cfg, "lgc_min_region_area", default=250))
    trim_rim = int(OmegaConf.select(seg_cfg, "lgc_trim_rim_px", default=3))
    cleanup_passes = int(OmegaConf.select(seg_cfg, "lgc_cleanup_passes", default=2))

    # Precompute graph topology once (flat tissue indexing).
    u_e, v_e, _node = _tissue_neighbor_edges(valid_mask)
    # Features on tissue for unary in PCA space (match pairwise space).
    feats_flat = np.asarray(feat_vol[np.asarray(valid_mask, dtype=bool)], dtype=np.float32)
    edge_for_pw = edge if (use_edge_barrier and edge is not None) else None

    levels: dict[str, np.ndarray] = {}
    method_tag = "landmark_graphcut"
    for name, n_cl in zip(names, n_clusters_list):
        km_labels, _centers_sig = _kmeans_labels_on_signatures(
            signatures,
            valid_mask,
            n_clusters=n_cl,
            random_state=random_state + n_cl,
            sample_size=sample_size,
        )
        if skip_rw or optimizer in {"none", "kmeans"}:
            lab = km_labels
            opt_used = "kmeans"
        elif optimizer in {"alpha_expansion", "alphacut", "expansion", "graphcut", "boykov"}:
            # Unary in PCA feature space for consistency with pairwise affinities.
            k_eff = int(km_labels.max())
            centers_f = np.zeros((k_eff, feats_flat.shape[1]), dtype=np.float64)
            flat_lab = km_labels[np.asarray(valid_mask, dtype=bool)].astype(np.int32) - 1
            for ki in range(k_eff):
                sel = flat_lab == ki
                if np.any(sel):
                    centers_f[ki] = feats_flat[sel].mean(axis=0)
                else:
                    centers_f[ki] = feats_flat.mean(axis=0)
            unary = _unary_from_centers(feats_flat, centers_f.astype(np.float32))
            # Normalize unary so median assigned cost ≈ 1; pairwise_scale then
            # has a stable meaning (boundary cost in units of unary).
            cur = unary[np.arange(unary.shape[0]), flat_lab]
            med = float(np.median(cur))
            if med > 1e-12:
                unary = unary / med
            pw = _pairwise_weights_from_features(
                feat_vol,
                valid_mask,
                u_e,
                v_e,
                beta=pairwise_beta,
                pairwise_scale=pairwise_scale,
                edge=edge_for_pw,
                edge_gamma=edge_gamma,
            )
            logger.info(
                "  alphacut[%s]: unary_med=%.4g pairwise mean/p50/p90=%.3g/%.3g/%.3g edge_barrier=%s",
                name,
                med,
                float(pw.mean()),
                float(np.median(pw)),
                float(np.percentile(pw, 90)),
                edge_for_pw is not None,
            )
            init0 = flat_lab.copy()
            np.random.seed(int(random_state) + n_cl)
            refined0 = _alpha_expansion_potts(
                init0, unary, u_e, v_e, pw, max_iter=alpha_iters
            )
            lab = np.zeros(valid_mask.shape, dtype=np.int32)
            lab[np.asarray(valid_mask, dtype=bool)] = refined0.astype(np.int32) + 1
            opt_used = "alpha_expansion"
            method_tag = "landmark_alphacut"
        else:
            seeds = _seeds_from_labels(
                km_labels,
                valid_mask,
                seed_frac=seed_frac,
                min_distance=seed_min_d,
            )
            lab = _random_walker_refine(feat_vol, seeds, valid_mask, beta=beta_rw)
            if int(lab.max()) < 2:
                logger.warning("landmark_graphcut[%s]: RW collapsed; using k-means", name)
                lab = km_labels
            opt_used = "random_walker"

        if cleanup:
            before = int(lab.max())
            # Scale min area mildly with coarseness (fewer clusters → larger islands OK).
            area_k = max(min_area, int(min_area * (40.0 / max(n_cl, 1))))
            for _pass in range(max(1, cleanup_passes)):
                # Rim trim only on the last pass (avoids eating anatomy early).
                lab = _cleanup_label_map(
                    lab,
                    valid_mask,
                    majority_radius=majority_r,
                    min_region_area=area_k,
                    trim_rim_px=trim_rim if _pass == max(1, cleanup_passes) - 1 else 0,
                )
            logger.info(
                "  cleanup[%s]: %d -> %d regions (min_area=%d rim=%d passes=%d)",
                name,
                before,
                int(lab.max()),
                area_k,
                trim_rim,
                max(1, cleanup_passes),
            )

        levels[name] = lab
        logger.info(
            "  landmark_graphcut[%s]: K=%d -> %d regions (%s)",
            name,
            n_cl,
            int(lab.max()),
            opt_used,
        )

    primary_name = str(OmegaConf.select(seg_cfg, "lgc_primary", default=names[min(1, len(names) - 1)]))
    if primary_name not in levels:
        primary_name = names[0]
    return SegResult(
        method=method_tag,
        primary=levels[primary_name],
        levels=levels,
        n_leaves=int(levels[names[0]].max()) if names else 0,
    )


def _segment_from_solace(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    seg_cfg: DictConfig,
    *,
    top3_rgb: np.ndarray | None = None,
    msi: np.ndarray | None = None,
    rank_config_path: str | None = None,
    random_state: int = 17,
) -> SegResult:
    method = str(OmegaConf.select(seg_cfg, "method", default="landmark_graphcut")).strip().lower()
    edge_s = _smooth_edge(edge, float(OmegaConf.select(seg_cfg, "edge_smooth_sigma", default=1.0)))
    edge_pct = float(OmegaConf.select(seg_cfg, "edge_percentile", default=70.0))
    min_dist = int(OmegaConf.select(seg_cfg, "min_seed_distance", default=12))

    if method in {"landmark_graphcut", "lgc", "landmark_rw", "graphcut"}:
        if msi is None or rank_config_path is None:
            raise ValueError("landmark_graphcut requires msi and solace.rank_config")
        return _segment_landmark_graphcut(
            msi,
            valid_mask,
            seg_cfg,
            rank_config_path=rank_config_path,
            random_state=random_state,
            edge=edge_s,
        )

    if method in {"graph_twostage", "twostage", "graph_rag", "rag_color"}:
        return _segment_graph_twostage(edge_s, valid_mask, seg_cfg, top3_rgb=top3_rgb)

    if method in {"graph", "felzenszwalb", "fh", "gbis"}:
        return _segment_graph(edge_s, valid_mask, seg_cfg, top3_rgb=top3_rgb)

    if method in {"hierarchical", "hierarchy", "rag", "waterfall"}:
        return _segment_hierarchical(edge_s, valid_mask, seg_cfg, top3_rgb=top3_rgb)

    if method in {"mgac", "active_contour", "active_contours", "gac"}:
        labels = _segment_mgac(
            edge_s,
            valid_mask,
            edge_percentile=edge_pct,
            min_seed_distance=min_dist,
            iterations=int(OmegaConf.select(seg_cfg, "mgac_iterations", default=80)),
            smoothing=int(OmegaConf.select(seg_cfg, "mgac_smoothing", default=1)),
            balloon=float(OmegaConf.select(seg_cfg, "mgac_balloon", default=-1.0)),
            threshold=float(OmegaConf.select(seg_cfg, "mgac_threshold", default=0.35)),
            edge_scale=float(OmegaConf.select(seg_cfg, "mgac_edge_scale", default=0.25)),
            seed_radius=int(OmegaConf.select(seg_cfg, "mgac_seed_radius", default=4)),
        )
        return SegResult(method="mgac", primary=labels, levels={"segmentation": labels})

    if method in {"watershed", "ws"}:
        labels = _segment_watershed(
            edge_s,
            valid_mask,
            edge_percentile=edge_pct,
            min_distance=int(OmegaConf.select(seg_cfg, "watershed_min_distance", default=min_dist)),
            compactness=float(OmegaConf.select(seg_cfg, "watershed_compactness", default=0.01)),
        )
        return SegResult(method="watershed", primary=labels, levels={"segmentation": labels})

    raise ValueError(
        f"Unknown segmentation.method: {method!r} "
        "(use landmark_graphcut, graph_twostage, graph, hierarchical, mgac, or watershed)"
    )


def _label_colors(n_labels: int, seed: int) -> np.ndarray:
    """High-contrast distinct RGB fills for labels 1..n (row 0 = black)."""
    import colorsys

    colors = np.zeros((n_labels + 1, 3), dtype=np.uint8)
    if n_labels <= 0:
        return colors
    rng = np.random.default_rng(int(seed))
    # Golden-angle hues + jittered sat/val so neighbors stay separable.
    hues = (np.arange(n_labels, dtype=np.float64) * 0.61803398875 + float(rng.random()) * 0.07) % 1.0
    for i in range(n_labels):
        sat = 0.72 + 0.22 * float(rng.random())
        val = 0.92 + 0.08 * float(rng.random())
        # Alternate slightly cooler/warmer value bands every other label.
        if i % 2 == 0:
            val = min(1.0, val + 0.04)
        else:
            sat = min(1.0, sat + 0.06)
        r, g, b = colorsys.hsv_to_rgb(float(hues[i]), sat, val)
        colors[i + 1] = (int(255 * r), int(255 * g), int(255 * b))
    return colors


def _segmentation_rgb(
    labels: np.ndarray,
    top3_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    fill_mode: str,
    overlay_on_top3: bool,
    region_alpha: float,
    boundary_color: tuple[int, int, int],
    draw_boundaries: bool,
    seed: int,
) -> np.ndarray:
    from skimage.segmentation import find_boundaries

    lab = np.asarray(labels, dtype=np.int32)
    mask = np.asarray(valid_mask, dtype=bool)
    n = int(lab.max()) if lab.size else 0
    colors = _label_colors(n, seed)
    color_img = colors[np.clip(lab, 0, n)]
    color_img[~mask] = 0

    mode = str(fill_mode).strip().lower()
    use_overlay = mode in {"overlay", "blend"} or (
        mode in {"auto", ""} and bool(overlay_on_top3)
    )
    if mode in {"solid", "flat", "fill"}:
        use_overlay = False

    if use_overlay:
        base = np.asarray(top3_rgb, dtype=np.float32)
        a = float(np.clip(region_alpha, 0.0, 1.0))
        out = base.copy()
        soft = mask & (lab > 0)
        out[soft] = (1.0 - a) * base[soft] + a * color_img[soft].astype(np.float32)
        out = np.clip(out, 0, 255).astype(np.uint8)
    else:
        out = color_img.copy()

    # Outer boundaries on speckled label maps paint almost every tissue pixel
    # dark; keep them for overlay mode only unless explicitly requested.
    if draw_boundaries and use_overlay:
        bounds = find_boundaries(lab, mode="outer", connectivity=1)
        bounds &= mask
        out[bounds] = np.asarray(boundary_color, dtype=np.uint8)
    out[~mask] = 0
    return out


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _titled_panel(rgb: np.ndarray, title: str, title_h: int) -> np.ndarray:
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    band = max(28, int(title_h))
    canvas = Image.new("RGB", (im.size[0], im.size[1] + band), (0, 0, 0))
    canvas.paste(im, (0, band))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(14, min(28, band - 12)))
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(0, (im.size[0] - tw) // 2)
    y = max(0, (band - th) // 2)
    draw.text((x, y), title, fill=(245, 245, 245), font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _hstack(blocks: list[np.ndarray], gap: int) -> np.ndarray:
    g = max(0, int(gap))
    max_h = max(b.shape[0] for b in blocks)
    total_w = sum(b.shape[1] for b in blocks) + g * (len(blocks) - 1)
    canvas = Image.new("RGB", (total_w, max_h), (0, 0, 0))
    x = 0
    for i, block in enumerate(blocks):
        im = Image.fromarray(np.asarray(block, dtype=np.uint8), mode="RGB")
        y = (max_h - im.size[1]) // 2
        canvas.paste(im, (x, y))
        x += im.size[0] + (g if i < len(blocks) - 1 else 0)
    return np.asarray(canvas, dtype=np.uint8)


def _vstack(blocks: list[np.ndarray], gap: int) -> np.ndarray:
    g = max(0, int(gap))
    max_w = max(b.shape[1] for b in blocks)
    total_h = sum(b.shape[0] for b in blocks) + g * (len(blocks) - 1)
    canvas = Image.new("RGB", (max_w, total_h), (0, 0, 0))
    y = 0
    for i, block in enumerate(blocks):
        im = Image.fromarray(np.asarray(block, dtype=np.uint8), mode="RGB")
        x = (max_w - im.size[0]) // 2
        canvas.paste(im, (x, y))
        y += im.size[1] + (g if i < len(blocks) - 1 else 0)
    return np.asarray(canvas, dtype=np.uint8)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def _boundary_color(cfg: DictConfig) -> tuple[int, int, int]:
    raw = OmegaConf.select(cfg, "gallery.boundary_color", default=[255, 255, 0])
    vals = list(OmegaConf.to_container(raw, resolve=True))
    if len(vals) != 3:
        return (255, 255, 0)
    return (int(vals[0]), int(vals[1]), int(vals[2]))


def _process_one(
    npy_path: Path,
    cfg: DictConfig,
    run_dir: Path,
    seed: int,
) -> dict[str, Any]:
    normalization = str(OmegaConf.select(cfg, "data.normalization", default="tic")).lower()
    msi = _load_msi(
        npy_path,
        bool(OmegaConf.select(cfg, "data.transpose_msi", default=False)),
        tic_normalize=(normalization == "tic"),
    )
    valid_mask = np.asarray(msi.sum(axis=-1) > 0, dtype=bool)
    if not np.any(valid_mask):
        raise ValueError(f"Empty tissue mask for {npy_path}")

    stem = npy_path.stem
    logger.info("Loaded %s shape=%s tissue=%d", npy_path.name, msi.shape, int(valid_mask.sum()))

    logger.info("TOP3 ...")
    top3 = _top3_rgb(msi, cfg)

    logger.info("SoLaCE ...")
    solace_raw = _compute_solace(
        msi,
        valid_mask,
        rank_config_path=str(cfg.solace.rank_config),
        hd_method=str(OmegaConf.select(cfg, "solace.hd_method", default="soft_landmark_contrast")),
    )
    cmap = str(OmegaConf.select(cfg, "solace.edge_colormap", default="PuBu"))
    eq_enabled = bool(OmegaConf.select(cfg, "solace.equalize", default=True))
    eq_method = str(OmegaConf.select(cfg, "solace.equalize_method", default="clahe"))
    if eq_enabled:
        solace = _equalize_edge_map(
            solace_raw,
            valid_mask,
            method=eq_method,
            clahe_clip=float(OmegaConf.select(cfg, "solace.clahe_clip", default=2.0)),
            clahe_grid=int(OmegaConf.select(cfg, "solace.clahe_grid", default=8)),
        )
        logger.info("SoLaCE equalized (%s) for segmentation", eq_method)
    else:
        solace = solace_raw

    solace_rgb = _solace_rgb(solace, valid_mask, cmap)
    solace_raw_rgb = _solace_rgb(solace_raw, valid_mask, cmap) if eq_enabled else None
    show_raw_solace = bool(OmegaConf.select(cfg, "gallery.show_raw_solace", default=True)) and eq_enabled

    logger.info(
        "Segmentation (%s) ...",
        OmegaConf.select(cfg, "segmentation.method", default="hierarchical"),
    )
    seg = _segment_from_solace(
        solace,
        valid_mask,
        cfg.segmentation,
        top3_rgb=top3,
        msi=msi,
        rank_config_path=str(cfg.solace.rank_config),
        random_state=seed,
    )
    logger.info(
        "Segmentation done: method=%s primary_regions=%d leaves=%d",
        seg.method,
        seg.n_regions,
        seg.n_leaves,
    )

    overlay = bool(OmegaConf.select(cfg, "gallery.overlay_on_top3", default=False))
    region_alpha = float(OmegaConf.select(cfg, "gallery.region_alpha", default=0.45))
    fill_mode = str(OmegaConf.select(cfg, "gallery.fill_mode", default="solid"))
    draw_boundaries = bool(OmegaConf.select(cfg, "gallery.draw_boundaries", default=True))
    bcolor = _boundary_color(cfg)

    # Preserve level order from config when present.
    if seg.method in {
        "hierarchical",
        "graph",
        "graph_twostage",
        "landmark_graphcut",
        "landmark_alphacut",
    }:
        if seg.method in {"landmark_graphcut", "landmark_alphacut"}:
            key = "segmentation.lgc_level_names"
        elif seg.method in {"graph", "graph_twostage"}:
            key = "segmentation.graph_level_names"
        else:
            key = "segmentation.hierarchical_level_names"
        raw_names = OmegaConf.select(cfg, key, default=["fine", "medium", "coarse"])
        ordered_names = [str(x) for x in list(OmegaConf.to_container(raw_names, resolve=True))]
        level_items = [(n, seg.levels[n]) for n in ordered_names if n in seg.levels]
        for n, lab in seg.levels.items():
            if n not in dict(level_items):
                level_items.append((n, lab))
    else:
        level_items = [("segmentation", seg.primary)]

    level_rgbs: dict[str, np.ndarray] = {}
    for i, (name, lab) in enumerate(level_items):
        level_rgbs[name] = _segmentation_rgb(
            lab,
            top3,
            valid_mask,
            fill_mode=fill_mode,
            overlay_on_top3=overlay,
            region_alpha=region_alpha,
            boundary_color=bcolor,
            draw_boundaries=draw_boundaries,
            seed=seed + i * 17,
        )

    if seg.method in {"landmark_graphcut", "landmark_alphacut"}:
        primary_name = str(OmegaConf.select(cfg, "segmentation.lgc_primary", default="medium"))
    elif seg.method in {"graph", "graph_twostage"}:
        primary_name = str(OmegaConf.select(cfg, "segmentation.graph_primary", default="medium"))
    elif seg.method == "hierarchical":
        primary_name = str(OmegaConf.select(cfg, "segmentation.hierarchical_primary", default="medium"))
    else:
        primary_name = "segmentation"
    if primary_name not in level_rgbs:
        primary_name = next(iter(level_rgbs))
    seg_rgb = level_rgbs[primary_name]

    labels_cfg = OmegaConf.select(cfg, "gallery.labels", default={}) or {}
    title_top3 = str(OmegaConf.select(labels_cfg, "top3", default="TOP3"))
    title_solace = str(OmegaConf.select(labels_cfg, "solace", default="SoLaCE"))
    if eq_enabled:
        title_solace = f"{title_solace} ({eq_method})"
    title_h = int(OmegaConf.select(cfg, "gallery.title_h", default=48))
    gap = int(OmegaConf.select(cfg, "gallery.gap_px", default=8))

    panels = [_titled_panel(top3, title_top3, title_h)]
    if show_raw_solace and solace_raw_rgb is not None:
        panels.append(_titled_panel(solace_raw_rgb, "SoLaCE (raw)", title_h))
    panels.append(_titled_panel(solace_rgb, title_solace, title_h))
    if bool(OmegaConf.select(cfg, "gallery.show_ridges", default=True)) and seg.ridges is not None:
        ridge_rgb = np.zeros((*valid_mask.shape, 3), dtype=np.uint8)
        ridge_rgb[valid_mask] = (235, 235, 235)
        ridge_rgb[np.asarray(seg.ridges, dtype=bool)] = (20, 40, 160)
        ridge_rgb[~valid_mask] = 0
        panels.append(_titled_panel(ridge_rgb, "Ridges (walls)", title_h))
        if bool(OmegaConf.select(cfg, "output.save_individual", default=True)):
            p_ridge = run_dir / f"{stem}__ridges.png"
            _save_rgb(p_ridge, ridge_rgb)
            # files dict filled below; stash path for later
            _ridge_path = str(p_ridge)
        else:
            _ridge_path = None
    else:
        _ridge_path = None

    for name, rgb in level_rgbs.items():
        nreg = int(seg.levels[name].max()) if name in seg.levels else seg.n_regions
        if seg.method in {
            "hierarchical",
            "graph",
            "graph_twostage",
            "landmark_graphcut",
            "landmark_alphacut",
        }:
            title = f"{name} ({nreg})"
        elif seg.method == "mgac":
            title = f"Segmentation (MGAC, {nreg})"
        elif seg.method == "watershed":
            title = f"Segmentation (watershed, {nreg})"
        else:
            title = f"Segmentation ({nreg})"
        panels.append(_titled_panel(rgb, title, title_h))
    row = _hstack(panels, gap)

    out: dict[str, Any] = {
        "npy_path": str(npy_path),
        "stem": stem,
        "shape": list(msi.shape),
        "tissue_pixels": int(valid_mask.sum()),
        "segmentation_method": seg.method,
        "n_regions": seg.n_regions,
        "n_leaves": seg.n_leaves,
        "levels": {name: int(lab.max()) for name, lab in seg.levels.items()},
        "primary_level": primary_name,
        "files": {},
    }

    if bool(OmegaConf.select(cfg, "output.save_individual", default=True)):
        p_top3 = run_dir / f"{stem}__top3.png"
        p_solace = run_dir / f"{stem}__solace.png"
        p_seg = run_dir / f"{stem}__segmentation.png"
        _save_rgb(p_top3, top3)
        _save_rgb(p_solace, solace_rgb)
        _save_rgb(p_seg, seg_rgb)
        out["files"]["top3"] = str(p_top3)
        out["files"]["solace"] = str(p_solace)
        out["files"]["segmentation"] = str(p_seg)
        if solace_raw_rgb is not None:
            p_raw = run_dir / f"{stem}__solace_raw.png"
            _save_rgb(p_raw, solace_raw_rgb)
            out["files"]["solace_raw"] = str(p_raw)
        np.save(run_dir / f"{stem}__solace_edge.npy", solace)
        out["files"]["solace_edge"] = str(run_dir / f"{stem}__solace_edge.npy")
        if eq_enabled:
            np.save(run_dir / f"{stem}__solace_edge_raw.npy", solace_raw)
            out["files"]["solace_edge_raw"] = str(run_dir / f"{stem}__solace_edge_raw.npy")
        if _ridge_path is not None:
            out["files"]["ridges"] = _ridge_path
        for name, rgb in level_rgbs.items():
            p = run_dir / f"{stem}__seg_{name}.png"
            _save_rgb(p, rgb)
            out["files"][f"seg_{name}"] = str(p)

    if bool(OmegaConf.select(cfg, "output.save_labels_npy", default=True)):
        p_lab = run_dir / f"{stem}__labels.npy"
        np.save(p_lab, seg.primary)
        out["files"]["labels"] = str(p_lab)
        for name, lab in seg.levels.items():
            p = run_dir / f"{stem}__labels_{name}.npy"
            np.save(p, lab)
            out["files"][f"labels_{name}"] = str(p)

    p_row = run_dir / f"{stem}__panel.png"
    _save_rgb(p_row, row)
    out["files"]["panel"] = str(p_row)
    out["_row_rgb"] = row
    return out


@hydra.main(version_base=None, config_path="configs", config_name="solace_segmentation_gallery")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed = int(OmegaConf.select(cfg, "seed", default=17))
    _seed_everything(seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

    paths = _resolve_npy_paths(cfg)
    results: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []

    for npy_path in paths:
        info = _process_one(npy_path, cfg, run_dir, seed)
        rows.append(info.pop("_row_rgb"))
        results.append(info)

    gap = int(OmegaConf.select(cfg, "gallery.gap_px", default=8))
    if bool(OmegaConf.select(cfg, "output.panel", default=True)):
        gallery = rows[0] if len(rows) == 1 else _vstack(rows, gap)
        gallery_path = run_dir / "gallery.png"
        _save_rgb(gallery_path, gallery)
        # Matplotlib panel with consistent DPI metadata (optional companion).
        dpi = int(OmegaConf.select(cfg, "output.dpi", default=160))
        fig, ax = plt.subplots(figsize=(gallery.shape[1] / dpi, gallery.shape[0] / dpi), dpi=dpi)
        ax.imshow(gallery)
        ax.axis("off")
        fig.savefig(run_dir / "gallery_mpl.png", dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="black")
        plt.close(fig)
    else:
        gallery_path = None

    manifest = {
        "run_dir": str(run_dir),
        "n_cubes": len(results),
        "results": results,
        "gallery": str(gallery_path) if gallery_path is not None else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote gallery under %s", run_dir)


if __name__ == "__main__":
    main()
