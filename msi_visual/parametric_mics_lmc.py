import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
import cv2
import tqdm
from typing import Optional, List, Any
import random
import time
import logging
from sklearn.decomposition import PCA
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import silhouette_score
from msi_visual.utils import cluster_id_labels_to_rgb_u8, cluster_palette_rgb_u8, normalize

logger = logging.getLogger(__name__)


def _spatial_gaussian_smooth_channels(embed: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur on float H×W×C PCC / embedding maps (before percentile scaling to uint8)."""
    s = float(sigma)
    if s <= 0.0:
        return embed
    out = np.asarray(embed, dtype=np.float32).copy()
    k = int(max(3, round(6 * s) | 1))
    kk = (k, k)
    fs = float(s)
    for c in range(out.shape[2]):
        out[:, :, c] = cv2.GaussianBlur(out[:, :, c], kk, sigmaX=fs, sigmaY=fs)
    return out


def _shift2d(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """Shift a 2D/3D array by (dy, dx) with constant fill (no wraparound)."""
    out = np.full_like(a, fill)
    h, w = a.shape[:2]
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    yd0, yd1 = max(0, -dy), min(h, h - dy)
    xd0, xd1 = max(0, -dx), min(w, w - dx)
    if ys1 > ys0 and xs1 > xs0:
        out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    return out


def _normalize_edge_guide(edge_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robustly rescale an edge-strength map to ~[0, 1] using the tissue 99th percentile."""
    e = np.asarray(edge_map, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    vals = e[m] if m.any() else e.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(e)
    lo = float(vals.min())
    hi = float(np.percentile(vals, 99.0))
    if hi <= lo + 1e-8:
        hi = lo + 1e-3
    out = (e - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def solace_guided_bilateral_smooth(
    embed: np.ndarray,
    edge_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    spatial_sigma: float = 1.5,
    edge_scale: float = 0.15,
    iterations: int = 4,
    radius: Optional[int] = None,
    range_sigma: float = 0.0,
    sharpen_amount: float = 0.0,
    sharpen_sigma: float = 1.0,
) -> np.ndarray:
    """Edge-aware (joint-bilateral) spatial smoothing of an H×W×C map guided by SoLaCE edges.

    Unlike an isotropic Gaussian blur, mixing between two pixels is attenuated whenever a
    strong SoLaCE boundary lies between them, so tissue types do not bleed across edges.

    The per-neighbor weight is::

        w = exp(-dist² / 2·spatial_sigma²)                  # spatial proximity
            · exp(-barrier² / 2·edge_scale²)                # SoLaCE edge barrier
            · [exp(-‖Δembed‖² / 2·range_sigma²)]            # optional range term
            · tissue_mask(neighbor)

    where ``barrier = max(edge_center, edge_neighbor)`` (a boundary on either endpoint
    blocks the exchange). A small ``radius`` applied over several ``iterations`` behaves
    like anisotropic diffusion, propagating within regions but not across boundaries.

    Args:
        embed: H×W×C float map to smooth (e.g. the MiCS embedding before RGB scaling).
        edge_map: H×W SoLaCE edge-strength map (higher = stronger boundary). Rescaled
            internally to ~[0, 1] via the tissue 99th percentile.
        mask: Optional H×W tissue mask; background pixels are excluded and left untouched.
        spatial_sigma: Gaussian spatial weight sigma in pixels.
        edge_scale: Edge-barrier sigma; smaller values block mixing more aggressively.
        iterations: Number of diffusion passes (each pass mixes a ``radius`` neighborhood).
        radius: Neighborhood radius per pass. Defaults to ``max(1, ceil(2·spatial_sigma))``
            capped at 5 to bound cost.
        range_sigma: Optional intensity-range sigma (classic bilateral term). ``<= 0`` disables it.
        sharpen_amount: Edge-guided unsharp strength applied after smoothing. ``> 0`` boosts
            local contrast *only* where SoLaCE is strong, sharpening boundaries while interiors
            stay smoothed. ``<= 0`` disables it.
        sharpen_sigma: Gaussian sigma for the unsharp low-pass (larger = broader edge halo).

    Returns:
        Smoothed H×W×C float32 array (background pixels preserved from the input).
    """
    x = np.asarray(embed, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, :, None]
    h, w, _c = x.shape

    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    ss = float(spatial_sigma)
    es = float(edge_scale)
    iters = int(iterations)
    if ss <= 0.0 or iters < 1 or es <= 0.0:
        return np.asarray(embed, dtype=np.float32)

    if radius is None:
        radius = int(min(5, max(1, np.ceil(2.0 * ss))))
    radius = int(max(1, radius))

    edge = _normalize_edge_guide(edge_map, mask)
    maskf = mask.astype(np.float32)

    offsets: list[tuple[int, int, float]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            spatial_w = float(np.exp(-(dy * dy + dx * dx) / (2.0 * ss * ss)))
            if spatial_w < 1e-4:
                continue
            offsets.append((dy, dx, spatial_w))

    rs2 = 2.0 * range_sigma * range_sigma if range_sigma > 0.0 else 0.0

    cur = x.copy()
    cur[~mask] = 0.0
    for _ in range(iters):
        acc = cur * maskf[:, :, None]  # center contributes with weight 1
        wsum = maskf.copy()
        for dy, dx, spatial_w in offsets:
            n_embed = _shift2d(cur, dy, dx, 0.0)
            n_edge = _shift2d(edge, dy, dx, 1.0)
            n_mask = _shift2d(maskf, dy, dx, 0.0)
            barrier = np.maximum(edge, n_edge)
            w = spatial_w * np.exp(-(barrier * barrier) / (2.0 * es * es)) * n_mask
            if rs2 > 0.0:
                diff = cur - n_embed
                w = w * np.exp(-np.sum(diff * diff, axis=2) / rs2)
            acc += w[:, :, None] * n_embed
            wsum += w
        safe = wsum > 1e-8
        nxt = cur.copy()
        nxt[safe] = acc[safe] / wsum[safe, None]
        nxt[~mask] = 0.0
        cur = nxt

    if float(sharpen_amount) > 0.0:
        # Edge-guided unsharp mask: add high-frequency detail scaled by SoLaCE strength so
        # only region boundaries get sharpened while smoothed interiors are left alone.
        blurred = _spatial_gaussian_smooth_channels(cur, float(sharpen_sigma))
        high = cur - blurred
        cur = cur + float(sharpen_amount) * edge[:, :, None] * high
        cur[~mask] = 0.0

    out = np.asarray(embed, dtype=np.float32).copy()
    if out.ndim == 2:
        out[mask] = cur[..., 0][mask]
    else:
        out[mask] = cur[mask]
    return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            torch.nn.GELU()
        )

    def forward(self, x):
        return x + self.layer(x)


def _build_pcc_sequential(
    input_dim: int,
    n_components: int = 3,
    factor: float = 1.0,
    num_layers: int = 2,
) -> torch.nn.Sequential:
    width = int(2048 * factor)
    n_lyr = max(1, int(num_layers))
    layers: list[torch.nn.Module] = [
        torch.nn.Linear(input_dim, width),
        torch.nn.BatchNorm1d(width),
        torch.nn.GELU(),
        ResidualBlock(width),
    ]
    for _ in range(n_lyr - 1):
        layers.extend([torch.nn.BatchNorm1d(width), torch.nn.GELU(), ResidualBlock(width)])
    layers.append(torch.nn.Linear(width, int(n_components)))
    return torch.nn.Sequential(*layers)


class PCCWithLearnedInputMask(torch.nn.Module):
    """PCC backbone with per-m/z sigmoid mask: x' = x ⊙ σ(logits) ⊙ fixed_mask."""

    def __init__(
        self,
        input_dim: int,
        n_components: int = 3,
        factor: float = 1.0,
        num_layers: int = 2,
        mask_logits_init: float = 4.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        # Large positive logits → σ≈1 (all-pass); L1 pulls effective mask down over stages.
        init = float(mask_logits_init)
        self.mask_logits = torch.nn.Parameter(
            torch.full((self.input_dim,), init, dtype=torch.float32)
        )
        self.register_buffer(
            "fixed_mask",
            torch.ones(self.input_dim, dtype=torch.float32),
            persistent=True,
        )
        self.backbone = _build_pcc_sequential(
            self.input_dim, n_components, factor, num_layers
        )

    def set_fixed_mask(self, mask_1d: np.ndarray | torch.Tensor) -> None:
        m = torch.as_tensor(mask_1d, dtype=torch.float32).reshape(-1)
        if m.numel() != self.input_dim:
            raise ValueError(f"fixed_mask length {m.numel()} != input_dim {self.input_dim}")
        self.fixed_mask.copy_(m.to(self.fixed_mask.device))

    def stage_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits)

    def effective_mask(self) -> torch.Tensor:
        return self.stage_mask() * self.fixed_mask

    def l1_mask_penalty(self) -> torch.Tensor:
        return self.effective_mask().mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.effective_mask()
        return self.backbone(x * m)


def create_pcc_model(input_dim, n_components=3, factor=1.0, num_layers=2):
    """
    Create a PCC model with the architecture from supervised_dr.ipynb.
    
    Args:
        input_dim: Input dimension (number of features)
        n_components: Output dimension (number of components)
        factor: Scaling factor for model size (default 1.0)
        
    Returns:
        torch.nn.Module: The model
    """
    model = _build_pcc_sequential(input_dim, n_components, factor, num_layers)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def correlation(
    pred: torch.Tensor, target: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    Compute correlation between two tensors.

    Args:
        pred: Prediction tensor
        target: Target tensor
        dim: Dimension along which to compute correlation. If None, treats tensors as 1D.

    Returns:
        Correlation coefficient as a tensor
    """
    if dim is None:
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return (pred * target).sum()
    else:
        pred = pred - pred.mean(dim=dim)[:, None]
        pred = pred / pred.norm(dim=dim)[:, None]
        target = target - target.mean(dim=dim)[:, None]
        target = target / target.norm(dim=dim)[:, None]
        return (pred * target).sum(dim=dim).mean()


def random_pixel_sampling(X_raw, approx_budget, height, width, roi_mask=None, rng_seed=42):
    """
    Fast path: sample random valid pixel indices (no PCA/SLIC). Use when superpixel is too slow.
    Returns (sampled_coords, sampled_flat_indices, None) for API compatibility.
    """
    mask = X_raw.max(axis=-1) > 0
    valid_flat = np.where(mask.ravel())[0]
    if roi_mask is not None:
        roi_flat = np.asarray(roi_mask, dtype=bool).reshape(height, width).ravel()
        valid_flat = valid_flat[roi_flat[valid_flat]]
    n = len(valid_flat)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), None
    rng = np.random.default_rng(rng_seed)
    k = min(approx_budget, n)
    chosen_flat = rng.choice(valid_flat, size=k, replace=False)
    rows, cols = np.unravel_index(chosen_flat, (height, width))
    sampled_coords = np.column_stack((rows, cols))
    return sampled_coords, chosen_flat, None


def weighted_pixel_sampling(
    X_raw,
    approx_budget,
    height,
    width,
    weight_map,
    roi_mask=None,
    rng_seed=42,
    uniform_mix=0.2,
):
    """Sample valid pixels using a per-pixel weight map, with uniform mixing for coverage."""
    mask = X_raw.max(axis=-1) > 0
    valid_flat = np.where(mask.ravel())[0]
    if roi_mask is not None:
        roi_flat = np.asarray(roi_mask, dtype=bool).reshape(height, width).ravel()
        valid_flat = valid_flat[roi_flat[valid_flat]]
    n = len(valid_flat)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), None

    w = np.asarray(weight_map, dtype=np.float64)
    if w.shape[:2] != (height, width):
        raise ValueError(f"weight_map shape {w.shape[:2]} != image shape {(height, width)}")
    weights = np.clip(w.reshape(-1)[valid_flat], 0.0, np.inf)
    if not np.any(np.isfinite(weights)):
        weights = np.ones_like(weights, dtype=np.float64)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    mix = float(np.clip(uniform_mix, 0.0, 1.0))
    if mix > 0.0:
        weights = (1.0 - mix) * weights + mix * (np.ones_like(weights) / len(weights))
        weights = weights / (weights.sum() + 1e-12)

    rng = np.random.default_rng(rng_seed)
    k = min(int(approx_budget), n)
    chosen_flat = rng.choice(valid_flat, size=k, replace=False, p=weights)
    rows, cols = np.unravel_index(chosen_flat, (height, width))
    sampled_coords = np.column_stack((rows, cols))
    return sampled_coords, chosen_flat, None


def edge_stratified_pixel_sampling(
    X_raw,
    approx_budget,
    height,
    width,
    weight_map,
    roi_mask=None,
    rng_seed=42,
    grid_cells=16,
    edge_floor=0.05,
    uniform_mix_within_cell=0.15,
    quota_by_weight: bool = False,
    min_quota_per_cell: int = 1,
):
    """Sample pixels with edge-biased weights per spatial grid cell for coverage.

    Default: each cell quota ∝ valid pixel count (``min_quota_per_cell`` ≥ 1 per occupied cell).
    ``quota_by_weight=True``: quota ∝ sum of weights in cell (for miss maps; no forced per-cell sample).
    Within a cell, weights are (1 - mix) * normalized_edge + mix * uniform.
    """
    mask = X_raw.max(axis=-1) > 0
    valid_pixel_bool = mask.reshape(height, width).astype(bool)
    if roi_mask is not None:
        valid_pixel_bool = valid_pixel_bool & np.asarray(roi_mask, dtype=bool).reshape(height, width)

    w = np.asarray(weight_map, dtype=np.float64)
    if w.shape[:2] != (height, width):
        raise ValueError(f"weight_map shape {w.shape[:2]} != image shape {(height, width)}")
    edge = np.clip(w, 0.0, None)
    edge = np.nan_to_num(edge, nan=0.0, posinf=0.0, neginf=0.0)

    valid_flat = np.where(valid_pixel_bool.ravel())[0]
    n_valid = len(valid_flat)
    if n_valid == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), None

    rows, cols = np.unravel_index(valid_flat, (height, width))
    n_grid = max(2, int(grid_cells))
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())
    r_span = max(1, r_max - r_min + 1)
    c_span = max(1, c_max - c_min + 1)
    cell_h = max(1, int(np.ceil(r_span / n_grid)))
    cell_w = max(1, int(np.ceil(c_span / n_grid)))

    cell_ids = ((rows - r_min) // cell_h) * n_grid + ((cols - c_min) // cell_w)
    cell_ids = np.clip(cell_ids, 0, n_grid * n_grid - 1)
    unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_cells))

    k_total = min(int(approx_budget), n_valid)
    n_cells = len(unique_cells)
    cell_weight_sums = np.zeros(n_cells, dtype=np.float64)
    for ci in range(n_cells):
        in_cell = np.where(inverse == ci)[0]
        cell_weight_sums[ci] = float(edge.ravel()[valid_flat[in_cell]].sum())

    if quota_by_weight and float(cell_weight_sums.sum()) > 1e-12:
        probs = cell_weight_sums / cell_weight_sums.sum()
        raw_q = k_total * probs
        quotas = np.floor(raw_q).astype(int)
        deficit = int(k_total - int(quotas.sum()))
        if deficit > 0:
            frac = raw_q - quotas
            for j in np.argsort(-frac)[:deficit]:
                quotas[int(j)] += 1
        elif deficit < 0:
            for j in np.argsort(frac)[:(-deficit)]:
                if quotas[int(j)] > 0:
                    quotas[int(j)] -= 1
    else:
        min_q = max(0, int(min_quota_per_cell))
        quotas = np.maximum(min_q, np.round(k_total * counts / counts.sum()).astype(int))
        while int(quotas.sum()) > k_total:
            j = int(np.argmax(quotas))
            if quotas[j] > min_q:
                quotas[j] -= 1
            else:
                break
        while int(quotas.sum()) < k_total:
            quotas[int(np.argmax(counts))] += 1

    floor = float(max(0.0, edge_floor))
    mix = float(np.clip(uniform_mix_within_cell, 0.0, 1.0))
    rng = np.random.default_rng(rng_seed)
    chosen: list[int] = []

    for ci, cell in enumerate(unique_cells):
        k_cell = min(int(quotas[ci]), int(counts[ci]))
        if k_cell <= 0:
            continue
        in_cell = np.where(inverse == ci)[0]
        flat_cell = valid_flat[in_cell]
        ew = edge.ravel()[flat_cell] + floor
        if not np.any(np.isfinite(ew)) or float(ew.sum()) <= 1e-12:
            ew = np.ones_like(ew, dtype=np.float64)
        ew = ew / (ew.sum() + 1e-12)
        if mix > 0.0:
            ew = (1.0 - mix) * ew + mix * (np.ones_like(ew) / len(ew))
            ew = ew / (ew.sum() + 1e-12)
        if k_cell >= len(flat_cell):
            chosen.extend(flat_cell.tolist())
        else:
            chosen.extend(
                rng.choice(flat_cell, size=k_cell, replace=False, p=ew).tolist()
            )

    chosen_flat = np.asarray(chosen[:k_total], dtype=np.int64)
    if chosen_flat.size < k_total:
        remaining = np.setdiff1d(valid_flat, chosen_flat, assume_unique=False)
        n_need = k_total - chosen_flat.size
        if remaining.size > 0:
            extra = rng.choice(
                remaining,
                size=min(n_need, remaining.size),
                replace=False,
            )
            chosen_flat = np.concatenate([chosen_flat, extra])

    rows_o, cols_o = np.unravel_index(chosen_flat, (height, width))
    sampled_coords = np.column_stack((rows_o, cols_o))
    return sampled_coords, chosen_flat, None


def build_miss_stratified_weight_map(
    hd_edge: np.ndarray,
    quality: np.ndarray,
    valid_mask: np.ndarray,
    *,
    quality_gamma: float = 2.0,
    hd_edge_floor: float = 0.05,
    uniformity_floor: float = 0.02,
) -> np.ndarray:
    """Sampling weights for structural misses: HD structure × low viz agreement.

    weight ∝ (uniformity_floor + (1 - quality)^γ) × (hd_edge + hd_edge_floor) on valid pixels.
    """
    m = np.asarray(valid_mask, dtype=bool)
    hd = np.clip(np.nan_to_num(np.asarray(hd_edge, dtype=np.float64), nan=0.0), 0.0, None)
    q = np.clip(np.nan_to_num(np.asarray(quality, dtype=np.float32), nan=0.0), 0.0, 1.0)
    bad = np.power(np.clip(1.0 - q, 0.0, 1.0), max(0.1, float(quality_gamma)))
    w = bad * (hd + float(hd_edge_floor))
    return np.where(m, w + float(uniformity_floor), 0.0).astype(np.float64)


def gate_miss_weight_map_for_sampling(
    miss_weight: np.ndarray,
    valid_mask: np.ndarray,
    weight_percentile: float | None,
) -> np.ndarray:
    """Zero out weights below a valid-mask percentile so sampling stays on worst misses."""
    w = np.asarray(miss_weight, dtype=np.float64).copy()
    if w.ndim != 2:
        raise ValueError(f"miss_weight must be 2D (H,W), got shape {w.shape}")
    m = np.asarray(valid_mask, dtype=bool)
    if m.shape != w.shape:
        raise ValueError(
            f"valid_mask shape {m.shape} != miss_weight shape {w.shape}"
        )
    if weight_percentile is None:
        return w
    p = float(weight_percentile)
    if p <= 0.0 or p >= 100.0:
        return w
    v = w[m]
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return w
    thr = float(np.percentile(v, p))
    w[(w < thr) | ~m] = 0.0
    return w


def topk_miss_pixel_sampling(
    X_raw,
    height: int,
    width: int,
    weight_map: np.ndarray,
    top_k: int,
    roi_mask=None,
    exclude_flat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, None]:
    """Pick exactly the ``top_k`` valid pixels with highest miss weight (deterministic)."""
    H, W = int(height), int(width)
    mask = X_raw.max(axis=-1) > 0
    valid_flat = np.where(mask.ravel())[0]
    if roi_mask is not None:
        roi_flat = np.asarray(roi_mask, dtype=bool).reshape(H, W).ravel()
        valid_flat = valid_flat[roi_flat[valid_flat]]
    if exclude_flat is not None and len(exclude_flat) > 0:
        ex = np.unique(np.asarray(exclude_flat, dtype=np.int64).ravel())
        keep = ~np.isin(valid_flat, ex)
        valid_flat = valid_flat[keep]
    n = len(valid_flat)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), None

    w = np.asarray(weight_map, dtype=np.float64)
    if w.shape[:2] != (H, W):
        raise ValueError(f"weight_map shape {w.shape[:2]} != image shape {(H, W)}")
    scores = np.nan_to_num(w.reshape(-1)[valid_flat], nan=0.0, posinf=0.0, neginf=0.0)
    k = max(1, min(int(top_k), n))
    order = np.argsort(-scores, kind="stable")
    chosen_flat = valid_flat[order[:k]]
    rows, cols = np.unravel_index(chosen_flat, (H, W))
    return np.column_stack((rows, cols)), chosen_flat, None


_CLUSTERS_LADDER_TEMPLATES = frozenset(
    {"legacy", "geo3", "geo4", "geo3_fine", "geo4_fine"}
)


def build_clusters_ladder_from_template(
    base_k: int,
    n_samples: int,
    template: str,
    k_fine: int = 2048,
    clamp_n_clusters=None,
) -> list[int]:
    """Build MiCS cluster level list from base k₀ and a named ladder template."""
    if clamp_n_clusters is None:
        clamp_n_clusters = MSIParametricMiCSLMC._clamp_n_clusters
    tpl = str(template).strip().lower()
    if tpl not in _CLUSTERS_LADDER_TEMPLATES:
        raise ValueError(
            f"clusters_auto_tune_ladder must be one of {sorted(_CLUSTERS_LADDER_TEMPLATES)} (got {template!r})"
        )
    k0_i = max(2, int(base_k))
    multipliers: list[int | str] = []
    if tpl == "legacy":
        multipliers = [1, 2, 4, 8]
    elif tpl == "geo3":
        multipliers = [1, 2, 4]
    elif tpl == "geo4":
        multipliers = [1, 2, 4, 8]
    elif tpl == "geo3_fine":
        multipliers = [1, 2, 4, "fine"]
    elif tpl == "geo4_fine":
        multipliers = [1, 2, 4, 8, "fine"]

    raw: list[int] = []
    for m in multipliers:
        if m == "fine":
            raw.append(int(k_fine))
        else:
            raw.append(int(k0_i) * int(m))

    seen: set[int] = set()
    out: list[int] = []
    n_samples = int(n_samples)
    for kk in raw:
        k_clamped = int(clamp_n_clusters(int(kk), n_samples))
        if k_clamped >= 2 and k_clamped not in seen:
            seen.add(k_clamped)
            out.append(k_clamped)
    if not out:
        out = [max(2, int(clamp_n_clusters(k0_i, n_samples)))]
    return out


def uniform_superpixel_sampling(X, X_raw, approx_budget=5000, 
                               init_superpixels=256, 
                               init_points_per_superpixel=8, 
                               compactness=10, 
                               max_iters=8, 
                               verbose=False, 
                               visualize=False,
                               rng_seed=42,
                               roi_mask=None,
                               pca_fit_step=4):
    """
    Uniformly sample points from superpixels with a target total sample budget.
    Will adaptively increase superpixels or points per superpixel if not enough points.
    
    Args:
        X: (H, W, C) array, main image stack
        X_raw: (H*W, C), flattened image stack
        approx_budget: int, target number of sampled points
        init_superpixels: initial n_superpixels to start from
        init_points_per_superpixel: initial samples per superpixel
        compactness: passed to slic
        max_iters: number of tries before giving up
        verbose: print search log
        visualize: show matplotlib overlay
        rng_seed: for random generator
        roi_mask: optional (H, W) boolean array; if provided, sampling is restricted to ROI.
        pca_fit_step: int, subsample step for PCA fit (X_raw[::pca_fit_step, :]); 1 = use all pixels.

    Returns:
        sampled_coords: (N, 2) array of sampled coordinates (row, col)
        sampled_flat_indices: (N, ) array for X_raw selection
        mask_labels: SLIC label map
    """
    mask = X.sum(axis=-1) > 0
    height, width = X.shape[:2]
    pca_fit_step = max(1, int(pca_fit_step))

    # 1. PCA for RGB mapping for visualization and SLIC basis
    t0 = time.perf_counter()
    pca = PCA(n_components=3)
    pca.fit(X_raw[::pca_fit_step, :])
    X_pca = pca.transform(X_raw)
    X_rgb = X_pca - X_pca.min(axis=0)
    X_rgb = X_rgb / (X_rgb.max(axis=0) + 1e-8)
    X_rgb = (X_rgb * 255).clip(0, 255).astype(np.uint8)
    X_rgb_img = X_rgb.reshape(height, width, 3)
    X_rgb_float = img_as_float(X_rgb_img)
    t_pca = time.perf_counter() - t0
    print(f"[Superpixel] PCA (fit step={pca_fit_step}): {t_pca:.2f}s")

    mask_img = mask.reshape(height, width)
    valid_pixel_bool = mask_img.astype(bool)
    if roi_mask is not None:
        valid_pixel_bool = valid_pixel_bool & np.asarray(roi_mask, dtype=bool).reshape(height, width)
    sampled_coords = []

    n_superpixels = init_superpixels
    points_per_superpixel = init_points_per_superpixel

    # Adaptive search for enough points
    for attempt in range(max_iters):
        if verbose:
            print(f"[Superpixel Select] Attempt {attempt+1}, n_segments={n_superpixels}, pts_per_superpixel={points_per_superpixel}")
        
        # 2. Superpixel segmentation
        t0 = time.perf_counter()
        mask_labels = slic(X_rgb_float, n_segments=n_superpixels, compactness=compactness, channel_axis=2, start_label=0)
        t_slic = time.perf_counter() - t0
        print(f"[Superpixel] SLIC (n_segments={n_superpixels}): {t_slic:.2f}s")
        unique_labels = np.unique(mask_labels)
        # Vectorized: one pass over valid pixels and their labels (avoids N_superpixels full-image argwhere)
        valid_flat = np.where(valid_pixel_bool.ravel())[0]
        labels_valid = mask_labels.ravel()[valid_flat]
        rows, cols = np.unravel_index(valid_flat, (height, width))
        result_coords = []
        rng = np.random.default_rng(rng_seed)
        for lbl in unique_labels:
            idx = labels_valid == lbl
            n = idx.sum()
            if n == 0:
                continue
            coords_valid = np.column_stack((rows[idx], cols[idx]))
            if n <= points_per_superpixel:
                chosen = coords_valid
            else:
                idx_sort = np.lexsort((coords_valid[:, 1], coords_valid[:, 0]))
                coords_valid_sorted = coords_valid[idx_sort]
                steps = max(1, n // points_per_superpixel)
                chosen = coords_valid_sorted[::steps][:points_per_superpixel]
            result_coords.append(chosen)
            
        if len(result_coords) == 0:
            sampled_coords = np.empty((0, 2), dtype=int)
        else:
            sampled_coords = np.concatenate(result_coords, axis=0)
        
        if verbose:
            print(f"    Collected {sampled_coords.shape[0]} points (budget={approx_budget})")
        
        # If meets budget, return
        if sampled_coords.shape[0] >= approx_budget:
            break
        # else, adaptively increase superpixels or points per superpixel
        if attempt % 2 == 0:
            # alternate between increasing points per superpixel
            points_per_superpixel = int(points_per_superpixel * 1.5)
        else:
            n_superpixels = int(n_superpixels * 1.5) + 8  # +8 to ensure new superpixels appear

    # Final to flat indices for X_raw
    if sampled_coords.shape[0] == 0:
        sampled_flat_indices = np.empty((0,), dtype=int)
    else:
        sampled_flat_indices = np.ravel_multi_index((sampled_coords[:, 0], sampled_coords[:, 1]), (height, width))

    if verbose:
        print("Returned", sampled_flat_indices.shape)

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))
        img_with_boundaries = mark_boundaries(X_rgb_img, mask_labels, color=(1, 0, 0), mode='thick')
        ax.imshow(img_with_boundaries)
        if len(sampled_coords) > 0:
            ax.scatter(sampled_coords[:, 1], sampled_coords[:, 0], s=8, c='yellow', alpha=0.7, label='Superpixel Samples')
        ax.axis('off')
        ax.set_title(f"Superpixel Overlay: Boundaries, Superpixel Uniform Samples (yellow);\nTotal points: {sampled_coords.shape[0]}")
        ax.legend()
        plt.show()


    return sampled_coords, sampled_flat_indices, mask_labels


def _build_category_labels(
    height: int,
    width: int,
    sampled_flat_indices: np.ndarray,
    category_annotations: List[dict],
) -> np.ndarray:
    """
    Assign a category to each sampled point from polygon annotations.
    category_annotations: list of {"polygon": [[x, y], ...], "category": int}
    with (x, y) = (col, row). Points inside no polygon get -1 (ignore).
    First matching polygon wins.
    """
    n_sampled = len(sampled_flat_indices)
    out = np.full(n_sampled, -1, dtype=np.int64)
    if not category_annotations:
        return out
    # Build one mask per annotation (order matters: first match wins)
    for ann in category_annotations:
        poly = ann.get("polygon") or ann.get("points")
        cat = ann.get("category", 0)
        if not poly or len(poly) < 3:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        mask_bool = mask.astype(bool)
        rows = sampled_flat_indices // width
        cols = sampled_flat_indices % width
        inside = mask_bool[rows, cols]
        # Only assign where still unlabeled
        out[(out == -1) & inside] = cat
    return out


class MSIParametricMiCSLMC:
    def __init__(
            self,
            model: torch.nn.Module = None,
            number_of_points=500,
            sampling="coreset",
            num_epochs=200,
            number_of_components=3,
            lab_to_rgb=True,
            cluster=False,
            clusters=None,
            num_samples=5000,
            beta=5.0,
            k_epoch=1,
            temperature=10.0,
            lr=1.0,
            batch_size=4096 * 2,
            warmup_epochs=10,
            factor=1.0,
            num_layers=2,
            random_state=42,
            verbose=False,
            cluster_loss_weight=1.0,
            category_loss_weight=1.0,
            pixel_sampling="superpixel",
            pca_fit_step=4,
            cluster_on_pca=False,
            cluster_pca_dims=50,
            sampling_weight_map=None,
            weighted_sampling_uniform_mix=0.2,
            predict_percentile_low: float = 0.001,
            predict_percentile_high: float = 99.999,
            predict_spatial_smooth_sigma: float = 0.0,
            predict_edge_smooth: bool = False,
            predict_edge_spatial_sigma: float = 1.5,
            predict_edge_scale: float = 0.15,
            predict_edge_iterations: int = 4,
            predict_edge_radius: Optional[int] = None,
            predict_edge_range_sigma: float = 0.0,
            predict_edge_sharpen_amount: float = 0.0,
            predict_edge_sharpen_sigma: float = 1.0,
            clusters_auto_tune: Optional[str] = None,
            clusters_auto_tune_k_min: int = 2,
            clusters_auto_tune_k_max: int = 256,
            clusters_auto_tune_n_candidates: int = 20,
            clusters_auto_tune_k_step: Optional[int] = None,
            clusters_auto_tune_ladder: str = "legacy",
            clusters_auto_tune_k_fine: int = 2048,
            hd_edge_map_for_sampling: Optional[np.ndarray] = None,
            spatial_quality_map_for_sampling: Optional[np.ndarray] = None,
            edge_stratified_grid_cells: int = 16,
            edge_stratified_edge_floor: float = 0.05,
            edge_stratified_uniform_mix: float = 0.15,
            miss_stratified_quality_gamma: float = 2.0,
            miss_stratified_hd_edge_floor: float = 0.05,
            miss_stratified_uniformity_floor: float = 0.02,
            miss_stratified_uniform_mix: float = 0.0,
            miss_stratified_quota_by_weight: bool = True,
            miss_stratified_sampler: str = "weighted",
            miss_stratified_weight_percentile: float = 65.0,
            miss_stratified_top_k: Optional[int] = None,
            miss_stratified_additive: bool = True,
            prior_sampled_flat_indices: Optional[np.ndarray] = None,
            finetune_miss_weighted_batches: bool = False,
            learn_input_mask: bool = False,
            input_mask_l1_weight: float = 0.0,
            input_mask_logits_init: float = 4.0,
    ):
        self.model = model
        self.verbose = verbose
        self.factor = factor
        self.num_layers = max(1, int(num_layers))
        self.num_epochs = num_epochs
        self.sampling = sampling
        self.number_of_points = number_of_points
        self.number_of_components = number_of_components
        self.lab_to_rgb = lab_to_rgb
        self.cluster = cluster
        self.clusters = self._normalize_clusters(clusters)
        self.num_samples = num_samples
        self.beta = beta
        self.k_epoch = k_epoch
        self.temperature = temperature
        self.lr = lr
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.random_state = random_state
        self.step_counter = 0
        self.scheduler = None
        self.cluster_loss_weight = float(cluster_loss_weight)
        self.category_loss_weight = float(category_loss_weight)
        self.pixel_sampling = str(pixel_sampling).lower() if pixel_sampling else "superpixel"
        self.pca_fit_step = max(1, int(pca_fit_step))
        self.cluster_on_pca = bool(cluster_on_pca)
        self.cluster_pca_dims = max(1, int(cluster_pca_dims))
        self.sampling_weight_map = sampling_weight_map
        self.weighted_sampling_uniform_mix = float(weighted_sampling_uniform_mix)
        self.predict_percentile_low = float(predict_percentile_low)
        self.predict_percentile_high = float(predict_percentile_high)
        self.predict_spatial_smooth_sigma = max(0.0, float(predict_spatial_smooth_sigma))
        self.predict_edge_smooth = bool(predict_edge_smooth)
        self.predict_edge_spatial_sigma = float(predict_edge_spatial_sigma)
        self.predict_edge_scale = float(predict_edge_scale)
        self.predict_edge_iterations = int(predict_edge_iterations)
        self.predict_edge_radius = predict_edge_radius
        self.predict_edge_range_sigma = float(predict_edge_range_sigma)
        self.predict_edge_sharpen_amount = float(predict_edge_sharpen_amount)
        self.predict_edge_sharpen_sigma = float(predict_edge_sharpen_sigma)
        # SoLaCE edge-strength guide (H×W). Set by the caller to enable edge-aware smoothing.
        self.predict_edge_map: Optional[np.ndarray] = None

        cat = clusters_auto_tune
        if cat is not None:
            s_mode = str(cat).strip().lower()
            if s_mode in ("", "null", "~", "none", "false"):
                cat = None
            elif s_mode == "silhouette":
                cat = s_mode
            elif s_mode == "bic":
                raise ValueError(
                    "clusters_auto_tune='bic' was removed; use 'silhouette' only."
                )
            else:
                raise ValueError(
                    "clusters_auto_tune must be None or 'silhouette' "
                    f"(got {clusters_auto_tune!r})"
                )
        self.clusters_auto_tune = cat
        self.clusters_auto_tune_k_min = max(2, int(clusters_auto_tune_k_min))
        self.clusters_auto_tune_k_max = max(
            self.clusters_auto_tune_k_min, int(clusters_auto_tune_k_max)
        )
        self.clusters_auto_tune_n_candidates = max(3, int(clusters_auto_tune_n_candidates))
        _kstp = clusters_auto_tune_k_step
        if _kstp is not None and isinstance(_kstp, str) and _kstp.strip().lower() in (
            "",
            "none",
            "null",
        ):
            _kstp = None
        try:
            step_i = int(_kstp) if _kstp is not None else None
        except (TypeError, ValueError):
            step_i = None
        self.clusters_auto_tune_k_step = step_i if (step_i is not None and int(step_i) >= 1) else None
        tpl = str(clusters_auto_tune_ladder).strip().lower()
        if tpl not in _CLUSTERS_LADDER_TEMPLATES:
            raise ValueError(
                f"clusters_auto_tune_ladder must be one of {sorted(_CLUSTERS_LADDER_TEMPLATES)} "
                f"(got {clusters_auto_tune_ladder!r})"
            )
        self.clusters_auto_tune_ladder = tpl
        self.clusters_auto_tune_k_fine = max(2, int(clusters_auto_tune_k_fine))
        self.hd_edge_map_for_sampling = (
            None
            if hd_edge_map_for_sampling is None
            else np.asarray(hd_edge_map_for_sampling, dtype=np.float64)
        )
        self.spatial_quality_map_for_sampling = (
            None
            if spatial_quality_map_for_sampling is None
            else np.asarray(spatial_quality_map_for_sampling, dtype=np.float64)
        )
        self.edge_stratified_grid_cells = max(2, int(edge_stratified_grid_cells))
        self.edge_stratified_edge_floor = float(edge_stratified_edge_floor)
        self.edge_stratified_uniform_mix = float(edge_stratified_uniform_mix)
        self.miss_stratified_quality_gamma = float(miss_stratified_quality_gamma)
        self.miss_stratified_hd_edge_floor = float(miss_stratified_hd_edge_floor)
        self.miss_stratified_uniformity_floor = float(miss_stratified_uniformity_floor)
        self.miss_stratified_uniform_mix = float(miss_stratified_uniform_mix)
        self.miss_stratified_quota_by_weight = bool(miss_stratified_quota_by_weight)
        self.miss_stratified_sampler = str(miss_stratified_sampler).strip().lower()
        _wp = miss_stratified_weight_percentile
        self.miss_stratified_weight_percentile = (
            None if _wp is None else float(_wp)
        )
        _tk = miss_stratified_top_k
        if _tk is not None and isinstance(_tk, str) and _tk.strip().lower() in (
            "",
            "none",
            "null",
        ):
            _tk = None
        self.miss_stratified_top_k = None if _tk is None else max(1, int(_tk))
        self.miss_stratified_additive = bool(miss_stratified_additive)
        _prior = prior_sampled_flat_indices
        self.prior_sampled_flat_indices = (
            None
            if _prior is None
            else np.asarray(_prior, dtype=np.int64).ravel().copy()
        )
        self.finetune_miss_weighted_batches = bool(finetune_miss_weighted_batches)
        self._sample_batch_weights: np.ndarray | None = None
        self.residual_base_hwc: np.ndarray | None = None
        self.residual_base_sampled: np.ndarray | None = None
        self.learn_input_mask = bool(learn_input_mask)
        self.input_mask_l1_weight = max(0.0, float(input_mask_l1_weight))
        self.input_mask_logits_init = float(input_mask_logits_init)
        self._fixed_input_mask_1d: np.ndarray | None = None
        self.clusters_auto_tune_base_k: Optional[int] = None
        self.clusters_auto_tune_ladder_effective: Optional[list[int]] = None

        # Seed everything for reproducibility
        self.seed_everything()

    def seed_everything(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __repr__(self):
        return f"Parametric PCC Optimization: num_epochs: {self.num_epochs} \
            sampling: {self.sampling} number_of_points:{self.number_of_points} cluster: {self.cluster}"

    def release_resources(self):
        """Release GPU and large tensors so the object can be collected and memory freed. Call before dropping the model reference."""
        try:
            if self.model is not None:
                self.model.cpu()
            if getattr(self, "reference_points", None) is not None:
                del self.reference_points
                self.reference_points = None
            if getattr(self, "euclidean", None) is not None:
                del self.euclidean
                self.euclidean = None
            if getattr(self, "mask", None) is not None:
                del self.mask
                self.mask = None
            if getattr(self, "visualiation_to_cluster", None):
                for layer in self.visualiation_to_cluster:
                    try:
                        layer.cpu()
                    except Exception:
                        pass
                self.visualiation_to_cluster = []
            if getattr(self, "category_head", None) is not None:
                try:
                    self.category_head.cpu()
                except Exception:
                    pass
                self.category_head = None
            if getattr(self, "category_labels", None) is not None:
                self.category_labels = None
            if getattr(self, "optim", None) is not None:
                self.optim = None
            if getattr(self, "scheduler", None) is not None:
                self.scheduler = None
            # Drop large numpy arrays so they can be collected
            if getattr(self, "sampled_data", None) is not None:
                self.sampled_data = None
            if getattr(self, "sampled_flat_indices", None) is not None:
                self.sampled_flat_indices = None
        except Exception:
            pass

    def set_residual_base(self, base_hwc: np.ndarray) -> None:
        """Fixed cumulative embedding (H,W,C) added to each forward pass (residual ladder)."""
        base = np.asarray(base_hwc, dtype=np.float32)
        if base.ndim != 3:
            raise ValueError(f"residual base must be HxWxC, got shape {base.shape}")
        if self.number_of_components != int(base.shape[-1]):
            raise ValueError(
                f"residual base channels {base.shape[-1]} != number_of_components "
                f"{self.number_of_components}"
            )
        self.residual_base_hwc = base.copy()
        self._sync_residual_base_sampled()

    def clear_residual_base(self) -> None:
        self.residual_base_hwc = None
        self.residual_base_sampled = None

    def _sync_residual_base_sampled(self) -> None:
        if self.residual_base_hwc is None:
            self.residual_base_sampled = None
            return
        flat = getattr(self, "sampled_flat_indices", None)
        if flat is None:
            self.residual_base_sampled = None
            return
        r = self.residual_base_hwc.reshape(-1, self.number_of_components)
        idx = np.asarray(flat, dtype=np.int64).ravel()
        self.residual_base_sampled = np.asarray(r[idx], dtype=np.float32)

    def _forward_training_embed(
        self, x: torch.Tensor, sample_row_indices: np.ndarray
    ) -> torch.Tensor:
        out = self.model(x)
        base = getattr(self, "residual_base_sampled", None)
        if base is not None:
            b = torch.from_numpy(np.asarray(base[sample_row_indices], dtype=np.float32)).to(
                device=out.device, dtype=out.dtype
            )
            out = out + b
        return out

    def _reference_embeddings(self) -> torch.Tensor:
        ref = self.model(self.reference_points)
        base = getattr(self, "residual_base_sampled", None)
        idx = getattr(self, "indices", None)
        if base is not None and idx is not None:
            b = torch.from_numpy(np.asarray(base[idx], dtype=np.float32)).to(
                device=ref.device, dtype=ref.dtype
            )
            ref = ref + b
        return ref

    def _infer_embedding_flat(self, reshaped: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Model (+ residual base) embedding for all pixels, shape (H*W, C)."""
        if batch_size is None:
            batch_size = 1024 * 32
        self.model.eval()
        base_flat = None
        if self.residual_base_hwc is not None:
            base_flat = self.residual_base_hwc.reshape(-1, self.number_of_components)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch in torch.split(torch.from_numpy(reshaped).float(), batch_size):
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                emb = self.model(x_batch).cpu().numpy()
                if base_flat is not None:
                    n0 = sum(c.shape[0] for c in chunks)
                    n1 = n0 + emb.shape[0]
                    emb = emb + base_flat[n0:n1]
                chunks.append(emb.astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0)

    def predict_delta(self, img: np.ndarray) -> np.ndarray:
        """Residual stage output only (no cumulative base), H×W×C float32."""
        if self.model is None:
            raise ValueError("Model must be initialized before predict_delta.")
        reshaped = np.asarray(img, dtype=np.float32).reshape(-1, img.shape[-1])
        self.model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch in torch.split(torch.from_numpy(reshaped).float(), 1024 * 32):
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                chunks.append(self.model(x_batch).cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0).reshape(img.shape[0], img.shape[1], -1)

    def predict_embedding(self, img: np.ndarray) -> np.ndarray:
        """Cumulative embedding (H,W,C) float32 before percentile stretch / LAB→RGB."""
        if self.model is None:
            raise ValueError("Model must be initialized before predict_embedding.")
        reshaped = np.asarray(img, dtype=np.float32).reshape(-1, img.shape[-1])
        flat = self._infer_embedding_flat(reshaped)
        return flat.reshape(img.shape[0], img.shape[1], -1)

    def _embedding_to_display_rgb(self, embedding_hwc: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Percentile stretch (+ optional LAB→RGB) shared by predict paths."""
        result = np.asarray(embedding_hwc, dtype=np.float32)
        mask = img.sum(axis=-1) > 0
        norm_mask = None
        if getattr(self, "roi_mask", None) is not None:
            norm_mask = mask & self.roi_mask
        out = np.uint8(
            255.0
            * normalize(
                result,
                low=self.predict_percentile_low,
                high=self.predict_percentile_high,
                mask=norm_mask,
            )
        )
        out[mask == 0] = 0
        if self.lab_to_rgb and self.number_of_components == 3:
            out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
        out[mask == 0] = 0
        return out

    def set_fixed_input_mask(self, mask_1d: np.ndarray) -> None:
        """Prior m/z gate multiplied before the learned sigmoid mask (mask-ladder stages)."""
        m = np.asarray(mask_1d, dtype=np.float32).reshape(-1)
        self._fixed_input_mask_1d = m.copy()
        if isinstance(self.model, PCCWithLearnedInputMask):
            self.model.set_fixed_mask(m)

    def get_input_mask_arrays(self) -> dict[str, np.ndarray]:
        """Effective, stage-only, and fixed m/z masks (numpy)."""
        if not isinstance(self.model, PCCWithLearnedInputMask):
            n = int(self.reshaped.shape[1]) if getattr(self, "reshaped", None) is not None else 0
            ones = np.ones(n, dtype=np.float32)
            return {"effective": ones, "stage": ones, "fixed": ones}
        with torch.no_grad():
            eff = self.model.effective_mask().detach().cpu().numpy().astype(np.float32)
            stg = self.model.stage_mask().detach().cpu().numpy().astype(np.float32)
            fix = self.model.fixed_mask.detach().cpu().numpy().astype(np.float32)
        return {"effective": eff, "stage": stg, "fixed": fix}

    def _create_default_model(self, input_dim: int) -> torch.nn.Module:
        """Pixel-wise MLP (PCC). Subclasses may override for spatial encoders (e.g. U-Net)."""
        if self.learn_input_mask:
            model = PCCWithLearnedInputMask(
                int(input_dim),
                self.number_of_components,
                self.factor,
                self.num_layers,
                mask_logits_init=self.input_mask_logits_init,
            )
            if self._fixed_input_mask_1d is not None:
                model.set_fixed_mask(self._fixed_input_mask_1d)
            if torch.cuda.is_available():
                model = model.cuda()
            return model
        return create_pcc_model(
            input_dim,
            self.number_of_components,
            self.factor,
            self.num_layers,
        )

    @staticmethod
    def _normalize_clusters(clusters: Any) -> list[int]:
        """Parse Hydra-style ``1024-2048-4096`` or list into ``[1024, 2048, 4096]``."""
        if clusters is None:
            return []
        if isinstance(clusters, (list, tuple)):
            return [int(x) for x in clusters]
        s = str(clusters).strip()
        if not s or s.lower() in ("", "null", "none", "~"):
            return []
        if "-" in s:
            return [int(x) for x in s.split("-") if x.strip()]
        return [int(s)]

    @staticmethod
    def _clamp_n_clusters(n_clusters: int, n_samples: int) -> int:
        """Sklearn KMeans requires n_samples >= n_clusters when n_clusters > 1."""
        k = int(n_clusters)
        n = int(n_samples)
        if n <= 0:
            return max(1, k)
        return max(1, min(k, n))

    def _clusters_auto_tune_active(self) -> bool:
        m = getattr(self, "clusters_auto_tune", None)
        return isinstance(m, str) and m == "silhouette"

    def _prepare_cluster_feature_matrix_for_clustering(self) -> np.ndarray:
        """Same feature matrix as MiCS KMeans (raw sampled spectra or PCA if ``cluster_on_pca``)."""
        X = np.asarray(self.sampled_data, dtype=np.float64)
        if not getattr(self, "cluster_on_pca", False):
            return X
        n_comp = min(
            int(self.cluster_pca_dims),
            int(X.shape[1]),
            max(1, int(X.shape[0]) - 1),
        )
        pca_cluster = PCA(n_components=n_comp, random_state=self.random_state)
        out = pca_cluster.fit_transform(X)
        if self.verbose:
            print(f"[set_image] Clustering on PCA-reduced data: {X.shape[1]} -> {n_comp} dims")
        return out

    def _clusters_auto_tune_candidate_ks(self, n_samples: int) -> list[int]:
        n = int(n_samples)
        k_lo = int(self.clusters_auto_tune_k_min)
        k_hi_cap = min(int(self.clusters_auto_tune_k_max), max(k_lo, n - 1))
        if k_hi_cap < k_lo:
            return [k_lo]
        step_ov = getattr(self, "clusters_auto_tune_k_step", None)
        if step_ov is not None and int(step_ov) >= 1:
            step_i = int(step_ov)
            out: list[int] = []
            k = k_lo
            while k <= k_hi_cap:
                out.append(int(k))
                k += step_i
            return out if out else [min(k_lo, k_hi_cap)]
        span = k_hi_cap - k_lo
        n_lin = max(
            3,
            min(int(self.clusters_auto_tune_n_candidates), span + 1),
        )
        xs = np.linspace(k_lo, k_hi_cap, num=n_lin if span > 0 else 1)
        ks_sorted = sorted({int(round(float(x))) for x in xs})
        return [k for k in ks_sorted if k_lo <= k <= k_hi_cap]

    def _clusters_auto_tune_select_base_k(self, X: np.ndarray) -> int:
        """Pick base k ∈ candidates using KMeans silhouette on clustering features."""
        mode = str(getattr(self, "clusters_auto_tune", "")).strip().lower()
        n = int(X.shape[0])
        ks = self._clusters_auto_tune_candidate_ks(n)
        if not ks:
            return max(2, min(int(self.clusters_auto_tune_k_min), n))
        rng = int(self.random_state)

        def _silhouette_pick() -> int:
            best_k, best_s = ks[0], -1.0
            ss = None
            if n > 4096:
                ss = min(2048, n - 1)
                if ss < 2:
                    ss = None
            for k in ks:
                if k >= n:
                    continue
                try:
                    lab = KMeans(
                        n_clusters=k, random_state=rng, n_init="auto"
                    ).fit_predict(X)
                    if len(np.unique(lab)) < 2:
                        continue
                    try:
                        s = float(
                            silhouette_score(
                                X,
                                lab,
                                metric="euclidean",
                                sample_size=ss,
                                random_state=rng if ss else None,
                            )
                        )
                    except TypeError:
                        s = float(silhouette_score(X, lab, metric="euclidean"))
                    if s > best_s:
                        best_s, best_k = s, int(k)
                except Exception:
                    continue
            if best_s <= -1.0 + 1e-12:
                return ks[len(ks) // 2]
            return int(best_k)

        if mode == "silhouette":
            return _silhouette_pick()
        raise RuntimeError(f"unexpected clusters_auto_tune mode {mode!r}")

    def _clusters_auto_tune_ladder(self, base_k: int, n_samples: int) -> list[int]:
        """Build ladder from ``clusters_auto_tune_ladder`` template and ``clusters_auto_tune_k_fine``."""
        return build_clusters_ladder_from_template(
            base_k,
            n_samples,
            getattr(self, "clusters_auto_tune_ladder", "legacy"),
            k_fine=int(getattr(self, "clusters_auto_tune_k_fine", 2048)),
            clamp_n_clusters=self._clamp_n_clusters,
        )

    def _maybe_apply_clusters_auto_tune(self, cluster_feature_matrix: np.ndarray) -> None:
        if not self._clusters_auto_tune_active():
            return
        X = np.asarray(cluster_feature_matrix, dtype=np.float64)
        n = int(X.shape[0])
        if n < 3:
            msg = (
                "[MiCS clusters_auto_tune] skipped: need >= 3 samples for criterion search "
                f"(got n={n})"
            )
            print(msg)
            logger.warning(msg)
            return
        mode = str(self.clusters_auto_tune)
        base_k_unc = self._clusters_auto_tune_select_base_k(X)
        ladder_eff = self._clusters_auto_tune_ladder(base_k_unc, n)
        self.clusters = ladder_eff
        self.clusters_auto_tune_base_k = int(base_k_unc)
        self.clusters_auto_tune_ladder_effective = list(ladder_eff)
        cand = self._clusters_auto_tune_candidate_ks(n)
        tpl = str(getattr(self, "clusters_auto_tune_ladder", "legacy"))
        k_fine = int(getattr(self, "clusters_auto_tune_k_fine", 2048))
        msg = (
            f"[MiCS clusters_auto_tune={mode}] template={tpl} k_fine={k_fine} n={n} "
            f"candidates k in {cand}, base_k(selected)={base_k_unc} "
            f"-> MiCS cluster levels {ladder_eff}"
        )
        print(msg)
        logger.info(msg)

    def get_reference_points(self, data, Np):
        """Reduces (NxD) data matrix from N to Np data points.

        Args:
            data: ndarray of shape [N, D]
            Np: number of data points in the coreset
        Returns:
            coreset: ndarray of shape [Np, D]
            weights: 1darray of shape [Np, 1]
        """
        N = data.shape[0]
        method = self.sampling
        rng = np.random.RandomState(self.random_state)

        if method == "random":
            return rng.choice(N, Np, replace=False)

        elif method == "kmeans++":
            _, indices = kmeans_plusplus(data, n_clusters=Np, random_state=self.random_state)
            return indices

        elif method == "kmeans":
            kmeans = KMeans(n_clusters=Np, random_state=self.random_state, n_init="auto").fit(data)
            return pairwise_distances(data, kmeans.cluster_centers_).argmin(axis=0)

        elif method == "coreset":
            # compute mean
            u = np.mean(data, axis=0)
            # compute proposal distribution
            q = np.linalg.norm(data - u, axis=1)**2
            q_sum = np.sum(q)
            d = q / q_sum
            q = 0.5 * (d + 1.0 / N)
            # get sample and fill coreset
            return rng.choice(N, Np, p=q)

    def export_pixel_sampling_cache(self) -> dict[str, Any]:
        """Export sampled pixels + reference coreset + HD distances for gallery reuse."""
        if getattr(self, "sampled_flat_indices", None) is None:
            raise ValueError("export_pixel_sampling_cache requires set_image/fit first")
        if getattr(self, "indices", None) is None:
            raise ValueError("export_pixel_sampling_cache requires resample/coreset indices")
        if getattr(self, "euclidean", None) is None:
            raise ValueError("export_pixel_sampling_cache requires euclidean distance matrix")
        euc = self.euclidean
        if isinstance(euc, torch.Tensor):
            euc_np = euc.detach().cpu().numpy().astype(np.float32, copy=True)
        else:
            euc_np = np.asarray(euc, dtype=np.float32).copy()
        labels = None
        if self.cluster and getattr(self, "cluster_labels", None):
            labels = [np.asarray(x, dtype=np.int32).copy() for x in self.cluster_labels]
        return {
            "sampled_flat_indices": np.asarray(self.sampled_flat_indices, dtype=np.int64).copy(),
            "indices": np.asarray(self.indices, dtype=np.int64).copy(),
            "euclidean": euc_np,
            "number_of_points": int(len(np.asarray(self.indices).reshape(-1))),
            "cluster_labels": labels,
        }

    def _apply_pixel_sampling_cache(self, cache: dict[str, Any]) -> None:
        """Reuse precomputed sample indices, coreset refs, and MSI distance matrix."""
        flat = np.asarray(cache["sampled_flat_indices"], dtype=np.int64).ravel()
        n_pix = int(self.reshaped.shape[0])
        if flat.size == 0 or flat.max() >= n_pix or flat.min() < 0:
            raise ValueError(
                f"cached sampled_flat_indices invalid for reshaped n={n_pix}"
            )
        self.sampled_flat_indices = flat
        self.sampled_data = self.reshaped[flat]
        self._sync_residual_base_sampled()

        self.indices = np.asarray(cache["indices"], dtype=np.int64).ravel()
        ref_pts = self.sampled_data[self.indices, :]
        self.reference_points = torch.from_numpy(ref_pts).float()
        if torch.cuda.is_available():
            self.reference_points = self.reference_points.cuda()

        euc_np = np.asarray(cache["euclidean"], dtype=np.float32)
        if euc_np.shape[0] != self.sampled_data.shape[0]:
            raise ValueError(
                f"cached euclidean rows {euc_np.shape[0]} != n_samples {self.sampled_data.shape[0]}"
            )
        self.euclidean = torch.from_numpy(euc_np).float()
        if torch.cuda.is_available():
            self.euclidean = self.euclidean.cuda()

        self._preset_cluster_labels = cache.get("cluster_labels")
        if self.verbose:
            print(
                f"[set_image] Reused pixel sampling cache: "
                f"n_samples={self.sampled_data.shape[0]} n_ref={self.indices.size}"
            )

    def set_image(
        self,
        img,
        roi_mask=None,
        category_annotations=None,
        pixel_sampling_cache: dict[str, Any] | None = None,
    ):
        timing_enabled = logger.isEnabledFor(logging.INFO)
        if timing_enabled:
            t0 = time.perf_counter()
            t_last = t0
            logger.info(
                "[set_image] start: shape=%s, roi_mask=%s, category_annotations=%s",
                getattr(img, "shape", None),
                roi_mask is not None,
                bool(category_annotations),
            )
        if self.verbose:
            print(f"[set_image] Starting image setup: shape={img.shape}, roi_mask={roi_mask is not None}, category_annotations={bool(category_annotations)}")
        self.img = img
        self.reshaped = self.img.reshape(
            self.img.shape[0] * self.img.shape[1], -1)
        self.img_mask = self.img.max(axis=-1) > 0
        if timing_enabled:
            logger.info("[set_image] reshape+mask done in %.2fs", time.perf_counter() - t_last)
            t_last = time.perf_counter()
        if self.verbose:
            print(f"[set_image] Reshaped image: {self.reshaped.shape}")
        
        # Pixel sampling: "superpixel" (slower, spread over segments) or "random" (fast)
        H, W = self.img.shape[0], self.img.shape[1]
        self._preset_cluster_labels = None
        if pixel_sampling_cache is not None:
            self._apply_pixel_sampling_cache(pixel_sampling_cache)
            if timing_enabled:
                logger.info("[set_image] sampling cache applied in %.2fs", time.perf_counter() - t_last)
                t_last = time.perf_counter()
        else:
            if self.pixel_sampling == "weighted":
                if self.sampling_weight_map is None:
                    raise ValueError("pixel_sampling='weighted' requires sampling_weight_map")
                _, sampled_flat_indices, _ = weighted_pixel_sampling(
                    self.reshaped,
                    self.num_samples,
                    H,
                    W,
                    self.sampling_weight_map,
                    roi_mask=roi_mask,
                    rng_seed=self.random_state,
                    uniform_mix=self.weighted_sampling_uniform_mix,
                )
            elif self.pixel_sampling in ("edge_stratified", "edge_stratified_hd"):
                edge_map = self.hd_edge_map_for_sampling
                if edge_map is None:
                    edge_map = self.sampling_weight_map
                if edge_map is None:
                    raise ValueError(
                        "pixel_sampling='edge_stratified' requires hd_edge_map_for_sampling "
                        "(or sampling_weight_map) before set_image/fit"
                    )
                _, sampled_flat_indices, _ = edge_stratified_pixel_sampling(
                    self.reshaped,
                    self.num_samples,
                    H,
                    W,
                    edge_map,
                    roi_mask=roi_mask,
                    rng_seed=self.random_state,
                    grid_cells=self.edge_stratified_grid_cells,
                    edge_floor=self.edge_stratified_edge_floor,
                    uniform_mix_within_cell=self.edge_stratified_uniform_mix,
                )
            elif self.pixel_sampling in ("miss_stratified", "miss"):
                sampled_flat_indices = self._sample_miss_stratified_flat(H, W, roi_mask=roi_mask)
            elif self.pixel_sampling == "random":
                _, sampled_flat_indices, _ = random_pixel_sampling(
                    self.reshaped, self.num_samples, H, W,
                    roi_mask=roi_mask, rng_seed=self.random_state
                )
            else:
                _, sampled_flat_indices, _ = uniform_superpixel_sampling(
                    self.img,
                    self.reshaped,
                    approx_budget=self.num_samples,
                    init_superpixels=512,
                    init_points_per_superpixel=8,
                    compactness=10,
                    max_iters=8,
                    verbose=self.verbose,
                    visualize=self.verbose,
                    rng_seed=self.random_state,
                    roi_mask=roi_mask,
                    pca_fit_step=getattr(self, "pca_fit_step", 4),
                )
            self.sampled_data = self.reshaped[sampled_flat_indices]
            self.sampled_flat_indices = sampled_flat_indices
            self._sync_residual_base_sampled()
            if timing_enabled:
                logger.info(
                    "[set_image] sampling done in %.2fs (method=%s, sampled=%s)",
                    time.perf_counter() - t_last,
                    self.pixel_sampling,
                    self.sampled_data.shape[0],
                )
                t_last = time.perf_counter()
            if self.verbose:
                print(
                    f"[set_image] Pixel sampling completed: sampled {self.sampled_data.shape[0]} "
                    f"points from {self.reshaped.shape[0]} total"
                )

        self.roi_mask = roi_mask  # store for ROI-based normalization in predict
        
        # Initialize model if not provided
        if self.model is None:
            input_dim = self.reshaped.shape[1]
            self.model = self._create_default_model(input_dim)
            if timing_enabled:
                logger.info(
                    "[set_image] model init done in %.2fs (input_dim=%s, n_components=%s)",
                    time.perf_counter() - t_last,
                    input_dim,
                    self.number_of_components,
                )
                t_last = time.perf_counter()
            if self.verbose:
                print(f"[set_image] Model initialized: input_dim={input_dim}, n_components={self.number_of_components}, factor={self.factor}")
        else:
            if self.verbose:
                print(f"[set_image] Using provided model")
        
        # Resample operates on the sampled_data, not the entire image
        # Validate that number_of_points doesn't exceed sampled data size
        max_points = min(self.number_of_points, self.sampled_data.shape[0])
        if max_points < self.number_of_points:
            if self.verbose:
                print(f"Warning: number_of_points ({self.number_of_points}) exceeds sampled data size ({self.sampled_data.shape[0]}). Using {max_points} points.")
        if pixel_sampling_cache is None:
            self.resample(number_of_points=max_points)
            if timing_enabled:
                logger.info(
                    "[set_image] resample done in %.2fs (points=%s)",
                    time.perf_counter() - t_last,
                    max_points,
                )
                t_last = time.perf_counter()
            if self.verbose:
                print(
                    f"[set_image] Reference points resampled: {max_points} points "
                    f"using '{self.sampling}' method"
                )
        
        self.mask_np = self.reshaped.max(axis=-1) > 0
        self.mask = torch.from_numpy(self.mask_np).float()
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

        # Category annotations: assign labels from polygon masks (supervised)
        height, width = self.img.shape[0], self.img.shape[1]
        self.category_labels = None
        self.category_head = None
        if category_annotations and len(category_annotations) > 0:
            self.category_labels = _build_category_labels(
                height, width, self.sampled_flat_indices, category_annotations
            )
            num_categories = max(ann.get("category", 0) for ann in category_annotations) + 1
            self.category_head = torch.nn.Sequential(
                torch.nn.Linear(self.number_of_components, num_categories)
            )
            if torch.cuda.is_available():
                self.category_head = self.category_head.cuda()
            if self.verbose:
                n_labeled = np.sum(self.category_labels >= 0)
                print(f"[set_image] Category annotations: {len(category_annotations)} polygons, {num_categories} classes, {n_labeled}/{len(self.category_labels)} points labeled")
        else:
            self.category_labels = None
            self.category_head = None
        if timing_enabled:
            logger.info("[set_image] category labels done in %.2fs", time.perf_counter() - t_last)
            t_last = time.perf_counter()

        self._init_clustering_heads()
        if timing_enabled:
            if self.visualiation_to_cluster:
                logger.info(
                    "[set_image] clustering init done in %.2fs (heads=%s)",
                    time.perf_counter() - t_last,
                    len(self.visualiation_to_cluster),
                )
            else:
                logger.info("[set_image] clustering skipped in %.2fs", time.perf_counter() - t_last)
            t_last = time.perf_counter()

        # Create optimizer with model and classifier parameters
        params = [{"params": self.model.parameters(), "weight_decay": 0}]
        if self.visualiation_to_cluster:
            for layer in self.visualiation_to_cluster:
                params.append({"params": layer.parameters(), "weight_decay": 0})
        if self.category_head is not None:
            params.append({"params": self.category_head.parameters(), "weight_decay": 0})

        self.optim = torch.optim.AdamW(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=self._lr_lambda
        )
        if timing_enabled:
            logger.info(
                "[set_image] optimizer init done in %.2fs (lr=%s, warmup=%s)",
                time.perf_counter() - t_last,
                self.lr,
                self.warmup_epochs,
            )
            logger.info("[set_image] total setup time %.2fs", time.perf_counter() - t0)
        if self.verbose:
            print(f"[set_image] Optimizer and scheduler initialized: lr={self.lr}, warmup_epochs={self.warmup_epochs}")
            print(f"[set_image] Image setup completed successfully")

    def _init_clustering_heads(self) -> None:
        """Build ``visualiation_to_cluster`` + ``cluster_labels`` from ``self.sampled_data`` (single hierarchy)."""
        self._cluster_head_expert_index = None
        if not self.cluster:
            self.visualiation_to_cluster = []
            self.cluster_labels = []
            return

        preset = getattr(self, "_preset_cluster_labels", None)
        if preset is not None and len(preset) > 0:
            self.cluster_labels = [np.asarray(x, dtype=np.int32) for x in preset]
            self.visualiation_to_cluster = []
            for labels in self.cluster_labels:
                k_eff = int(np.max(labels)) + 1 if labels.size else 2
                layer = torch.nn.Sequential(torch.nn.Linear(self.number_of_components, k_eff))
                if torch.cuda.is_available():
                    layer = layer.cuda()
                self.visualiation_to_cluster.append(layer)
            if self.verbose:
                print(
                    f"[set_image] Cluster heads from cache ({len(self.visualiation_to_cluster)} levels)"
                )
            self._preset_cluster_labels = None
            return

        data_for_cluster = self._prepare_cluster_feature_matrix_for_clustering()
        self._maybe_apply_clusters_auto_tune(data_for_cluster)

        if len(self.clusters) == 0:
            self.visualiation_to_cluster = []
            self.cluster_labels = []
            return

        if self.verbose:
            print(
                f"[set_image] Initializing clustering with {len(self.clusters)} cluster levels: {self.clusters}"
            )
        self.cluster_labels = []
        self.visualiation_to_cluster = []
        n_samples_dc = int(data_for_cluster.shape[0])
        for k in self.clusters:
            k_eff = self._clamp_n_clusters(k, n_samples_dc)
            if k_eff != int(k):
                msg = f"n_clusters reduced from {k} to {k_eff} (n_samples={n_samples_dc} for clustering)"
                if self.verbose:
                    print(f"[set_image] {msg}")
                else:
                    logger.warning("[set_image] %s", msg)
            layer = torch.nn.Sequential(torch.nn.Linear(self.number_of_components, k_eff))
            if torch.cuda.is_available():
                layer = layer.cuda()
            self.visualiation_to_cluster.append(layer)
            kmeans = KMeans(n_clusters=k_eff, random_state=self.random_state, n_init="auto")
            labels = kmeans.fit_predict(data_for_cluster)
            self.cluster_labels.append(labels)
        if self.verbose:
            print(f"[set_image] Clustering initialization completed")

    def _embed_for_cluster_head(self, x: torch.Tensor, head_idx: int) -> torch.Tensor:
        """Embedding vector fed to ``visualiation_to_cluster[head_idx]`` (default: fused / single PCC output)."""
        del head_idx
        return self.model(x)

    def _attach_sample_batch_weights(self, sampled_flat: np.ndarray) -> None:
        """Per-sample weights for weighted minibatches during finetune (from miss map)."""
        miss_map = self.spatial_quality_map_for_sampling
        if miss_map is None:
            self._sample_batch_weights = None
            return
        flat = np.asarray(sampled_flat, dtype=np.int64).ravel()
        w = np.asarray(miss_map, dtype=np.float64).reshape(-1)[flat]
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        if float(w.sum()) <= 1e-12:
            w = np.ones(flat.shape[0], dtype=np.float64)
        self._sample_batch_weights = w

    def _sample_miss_stratified_flat(
        self,
        height: int,
        width: int,
        roi_mask=None,
        prior_flat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample flat indices for miss_stratified (weighted, grid, or top-k by miss weight)."""
        H, W = int(height), int(width)
        miss_map = self.spatial_quality_map_for_sampling
        if miss_map is None:
            miss_map = self.sampling_weight_map
        if miss_map is None:
            raise ValueError(
                "pixel_sampling='miss_stratified' requires spatial_quality_map_for_sampling "
                "(miss weight map from (1-quality)^γ × hd_edge)"
            )
        valid_roi = np.asarray(self.img_mask, dtype=bool)
        if roi_mask is not None:
            valid_roi = valid_roi & np.asarray(roi_mask, dtype=bool).reshape(H, W)
        miss_map = np.asarray(miss_map, dtype=np.float64)
        if miss_map.shape[:2] != (H, W):
            miss_map = miss_map.reshape(H, W)

        sampler = str(self.miss_stratified_sampler).strip().lower()
        use_topk = sampler in ("topk", "top_k", "top") or self.miss_stratified_top_k is not None
        if use_topk:
            top_k = self.miss_stratified_top_k if self.miss_stratified_top_k is not None else 1000
            prior = None
            if self.miss_stratified_additive:
                if prior_flat is not None:
                    prior = np.asarray(prior_flat, dtype=np.int64).ravel()
                elif self.prior_sampled_flat_indices is not None:
                    prior = np.asarray(self.prior_sampled_flat_indices, dtype=np.int64).ravel()
            exclude = prior if (prior is not None and prior.size > 0) else None
            _, new_flat, _ = topk_miss_pixel_sampling(
                self.reshaped,
                H,
                W,
                miss_map,
                top_k,
                roi_mask=roi_mask,
                exclude_flat=exclude,
            )
            if prior is not None and prior.size > 0:
                flat = np.unique(np.concatenate([prior, new_flat]))
                w_new = miss_map.reshape(-1)[new_flat]
                print(
                    f"[train] miss_stratified sample: sampler=topk additive "
                    f"prior={prior.size} + new={new_flat.size} (requested={top_k}) "
                    f"-> total={flat.size} new_weight[min,med,max]="
                    f"({float(np.min(w_new)):.4g}, {float(np.median(w_new)):.4g}, {float(np.max(w_new)):.4g})"
                )
            else:
                flat = new_flat
                w_at = miss_map.reshape(-1)[flat]
                print(
                    f"[train] miss_stratified sample: sampler=topk top_k={len(flat)} "
                    f"(requested={top_k}) weight[min,med,max]="
                    f"({float(np.min(w_at)):.4g}, {float(np.median(w_at)):.4g}, {float(np.max(w_at)):.4g})"
                )
            self._attach_sample_batch_weights(flat)
            return flat

        miss_map = gate_miss_weight_map_for_sampling(
            miss_map,
            valid_roi,
            self.miss_stratified_weight_percentile,
        )
        elig = int(np.count_nonzero((miss_map > 0) & valid_roi))
        n_roi = int(np.count_nonzero(valid_roi))
        pct = 100.0 * elig / max(n_roi, 1)
        print(
            f"[train] miss_stratified sample: sampler={sampler} "
            f"weight_percentile={self.miss_stratified_weight_percentile} "
            f"eligible_pixels={pct:.1f}% ({elig}/{n_roi}) n={self.num_samples}"
        )
        if sampler in ("weighted", "global", "prob"):
            _, flat, _ = weighted_pixel_sampling(
                self.reshaped,
                self.num_samples,
                H,
                W,
                miss_map,
                roi_mask=roi_mask,
                rng_seed=self.random_state,
                uniform_mix=self.miss_stratified_uniform_mix,
            )
        else:
            _, flat, _ = edge_stratified_pixel_sampling(
                self.reshaped,
                self.num_samples,
                H,
                W,
                miss_map,
                roi_mask=roi_mask,
                rng_seed=self.random_state,
                grid_cells=self.edge_stratified_grid_cells,
                edge_floor=0.0,
                uniform_mix_within_cell=self.miss_stratified_uniform_mix,
                quota_by_weight=self.miss_stratified_quota_by_weight,
                min_quota_per_cell=0,
            )
        self._attach_sample_batch_weights(flat)
        return flat

    def _sample_pixel_indices(
        self,
        height: int,
        width: int,
        roi_mask=None,
    ) -> np.ndarray:
        """Draw ``num_samples`` flat pixel indices for the current ``pixel_sampling`` mode."""
        H, W = int(height), int(width)
        ps = str(self.pixel_sampling).lower()
        if ps == "weighted":
            if self.sampling_weight_map is None:
                raise ValueError("pixel_sampling='weighted' requires sampling_weight_map")
            _, flat, _ = weighted_pixel_sampling(
                self.reshaped,
                self.num_samples,
                H,
                W,
                self.sampling_weight_map,
                roi_mask=roi_mask,
                rng_seed=self.random_state,
                uniform_mix=self.weighted_sampling_uniform_mix,
            )
            return flat
        if ps in ("edge_stratified", "edge_stratified_hd"):
            edge_map = self.hd_edge_map_for_sampling
            if edge_map is None:
                edge_map = self.sampling_weight_map
            if edge_map is None:
                raise ValueError(
                    "pixel_sampling='edge_stratified' requires hd_edge_map_for_sampling"
                )
            _, flat, _ = edge_stratified_pixel_sampling(
                self.reshaped,
                self.num_samples,
                H,
                W,
                edge_map,
                roi_mask=roi_mask,
                rng_seed=self.random_state,
                grid_cells=self.edge_stratified_grid_cells,
                edge_floor=self.edge_stratified_edge_floor,
                uniform_mix_within_cell=self.edge_stratified_uniform_mix,
            )
            return flat
        if ps in ("miss_stratified", "miss"):
            return self._sample_miss_stratified_flat(H, W, roi_mask=roi_mask)
        if ps == "random":
            _, flat, _ = random_pixel_sampling(
                self.reshaped,
                self.num_samples,
                H,
                W,
                roi_mask=roi_mask,
                rng_seed=self.random_state,
            )
            return flat
        _, flat, _ = uniform_superpixel_sampling(
            self.img,
            self.reshaped,
            approx_budget=self.num_samples,
            init_superpixels=512,
            init_points_per_superpixel=8,
            compactness=10,
            max_iters=8,
            verbose=self.verbose,
            visualize=False,
            rng_seed=self.random_state,
            roi_mask=roi_mask,
            pca_fit_step=self.pca_fit_step,
        )
        return flat

    def _rebuild_optimizer(self, lr: float, warmup_epochs: int) -> None:
        """Recreate AdamW + scheduler after resampling / new cluster heads."""
        params = [{"params": self.model.parameters(), "weight_decay": 0}]
        if self.visualiation_to_cluster:
            for layer in self.visualiation_to_cluster:
                params.append({"params": layer.parameters(), "weight_decay": 0})
        if self.category_head is not None:
            params.append({"params": self.category_head.parameters(), "weight_decay": 0})
        self.lr = float(lr)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.optim = torch.optim.AdamW(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=self._lr_lambda
        )

    def finetune_miss_stratified(
        self,
        img,
        roi_mask=None,
        *,
        finetune_epochs: int = 10,
        finetune_lr_scale: float = 0.1,
        finetune_warmup_epochs: int = 0,
    ) -> "MSIParametricMiCSLMC":
        """Keep bootstrap PCC weights; resample on low-recall pixels and train with lower LR."""
        if self.model is None:
            raise ValueError("finetune_miss_stratified requires a bootstrap fit first")
        self.img = img
        self.reshaped = self.img.reshape(self.img.shape[0] * self.img.shape[1], -1)
        self.img_mask = self.img.max(axis=-1) > 0
        H, W = self.img.shape[0], self.img.shape[1]
        self.pixel_sampling = "miss_stratified"
        if self.verbose:
            print(
                f"[finetune_miss_stratified] resample + {int(finetune_epochs)} epochs "
                f"at lr={float(self.lr) * float(finetune_lr_scale):.4g} "
                f"(scale={finetune_lr_scale})"
            )
        prior_flat = None
        if getattr(self, "sampled_flat_indices", None) is not None:
            prior_flat = np.asarray(self.sampled_flat_indices, dtype=np.int64).copy()
        sampled_flat = self._sample_miss_stratified_flat(
            H, W, roi_mask=roi_mask, prior_flat=prior_flat
        )
        self.sampled_data = self.reshaped[sampled_flat]
        self.sampled_flat_indices = sampled_flat
        self._sync_residual_base_sampled()
        self._attach_sample_batch_weights(sampled_flat)
        max_points = min(self.number_of_points, self.sampled_data.shape[0])
        self.indices = None
        self.resample(number_of_points=max_points)
        if self.cluster:
            self._init_clustering_heads()
        finetune_lr = float(self.lr) * float(finetune_lr_scale)
        if self.finetune_miss_weighted_batches and self._sample_batch_weights is not None:
            print(
                "[finetune_miss_stratified] weighted minibatches ON "
                f"(n={self.sampled_data.shape[0]})"
            )
        self._rebuild_optimizer(finetune_lr, finetune_warmup_epochs)
        self.step_counter = 0
        n_ep = max(1, int(finetune_epochs))
        for epoch in range(n_ep):
            self.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.verbose and ((epoch + 1) % 5 == 0 or epoch + 1 == n_ep):
                print(f"[finetune_miss_stratified] epoch {epoch + 1}/{n_ep}")
        if self.verbose:
            print("[finetune_miss_stratified] done")
        return self

    def finetune_transfer(
        self,
        img,
        roi_mask=None,
        *,
        finetune_epochs: int = 10,
        finetune_lr_scale: float = 0.1,
        finetune_warmup_epochs: int = 0,
        pixel_sampling_cache: dict[str, Any] | None = None,
    ) -> "MSIParametricMiCSLMC":
        """Adapt source-trained weights on a new slide (resample + re-init heads, lower LR).

        Unlike ``finetune_miss_stratified``, keeps the original ``pixel_sampling`` mode
        (e.g. superpixel) and does not require a miss-stratified quality map.

        When ``pixel_sampling_cache`` is provided (e.g. gallery prebuild), reuses
        superpixel/coreset indices and cluster labels instead of rebuilding them.
        """
        if self.model is None:
            raise ValueError("finetune_transfer requires a bootstrap fit first")
        finetune_lr = float(self.lr) * float(finetune_lr_scale)
        n_ep = max(1, int(finetune_epochs))
        if pixel_sampling_cache is not None:
            if self.verbose:
                print(
                    f"[finetune_transfer] cached sampling + {n_ep} epochs at lr="
                    f"{finetune_lr:.4g} (scale={finetune_lr_scale})"
                )
            self._sample_batch_weights = None
            self.set_image(img, roi_mask=roi_mask, pixel_sampling_cache=pixel_sampling_cache)
            self._rebuild_optimizer(finetune_lr, finetune_warmup_epochs)
            self.step_counter = 0
            for epoch in range(n_ep):
                self.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.verbose and ((epoch + 1) % 5 == 0 or epoch + 1 == n_ep):
                    print(f"[finetune_transfer] epoch {epoch + 1}/{n_ep}")
            if self.verbose:
                print("[finetune_transfer] done")
            return self

        self.img = img
        self.reshaped = self.img.reshape(self.img.shape[0] * self.img.shape[1], -1)
        self.img_mask = self.img.max(axis=-1) > 0
        if self.verbose:
            print(
                f"[finetune_transfer] resample ({self.pixel_sampling}) + "
                f"{n_ep} epochs at lr={finetune_lr:.4g} (scale={finetune_lr_scale})"
            )
        self._sample_batch_weights = None
        self.update_roi_sampling(roi_mask)
        self._rebuild_optimizer(finetune_lr, finetune_warmup_epochs)
        self.step_counter = 0
        for epoch in range(n_ep):
            self.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.verbose and ((epoch + 1) % 5 == 0 or epoch + 1 == n_ep):
                print(f"[finetune_transfer] epoch {epoch + 1}/{n_ep}")
        if self.verbose:
            print("[finetune_transfer] done")
        return self

    def update_roi_sampling(self, roi_mask):
        """
        Update sampling to ROI only: resample points, re-fit cluster labels, re-initialize cluster heads.
        The embedding model is kept (not re-initialized) so full-image structure is preserved;
        only the cluster heads are re-initialized to learn the new ROI cluster assignments.
        Optimizer is recreated so it tracks the new head parameters.
        """
        if self.img is None or self.reshaped is None:
            raise ValueError("Call set_image() first before update_roi_sampling()")
        if self.verbose:
            print(f"[update_roi_sampling] Updating sampling to ROI (keep model, resample + re-init heads)")
        H, W = self.img.shape[0], self.img.shape[1]
        if getattr(self, "pixel_sampling", "superpixel") == "random":
            _, sampled_flat_indices, _ = random_pixel_sampling(
                self.reshaped, self.num_samples, H, W,
                roi_mask=roi_mask, rng_seed=self.random_state
            )
        else:
            _, sampled_flat_indices, _ = uniform_superpixel_sampling(
                self.img,
                self.reshaped,
                approx_budget=self.num_samples,
                init_superpixels=512,
                init_points_per_superpixel=8,
                compactness=10,
                max_iters=8,
                verbose=self.verbose,
                visualize=False,
                rng_seed=self.random_state,
                roi_mask=roi_mask,
                pca_fit_step=getattr(self, "pca_fit_step", 4),
            )
        self.sampled_data = self.reshaped[sampled_flat_indices]
        self.sampled_flat_indices = sampled_flat_indices
        self._sync_residual_base_sampled()
        max_points = min(self.number_of_points, self.sampled_data.shape[0])
        self.indices = None  # force resample to recompute from new sampled_data
        self.resample(number_of_points=max_points)
        if self.cluster:
            self._init_clustering_heads()
            params = [{"params": self.model.parameters(), "weight_decay": 0}]
            for layer in self.visualiation_to_cluster:
                params.append({"params": layer.parameters(), "weight_decay": 0})
            self.optim = torch.optim.AdamW(params, lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optim, lr_lambda=self._lr_lambda
            )
            if self.verbose:
                print(
                    f"[update_roi_sampling] Cluster labels and heads re-initialized "
                    f"({len(self.visualiation_to_cluster)} head(s))"
                )
    
    def _lr_lambda(self, epoch: int) -> float:
        """ Learning rate scheduler with warmup """
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)
        return 1.0

    def fit(self, X, pixel_sampling_cache: dict[str, Any] | None = None):
        """
        Fit the model on image data X.
        
        This method combines set_image() and training in one call, following
        the sklearn-like API pattern.
        
        Args:
            X: numpy array of shape (H, W, C) - the image data to train on
            pixel_sampling_cache: optional cache from ``export_pixel_sampling_cache``
            verbose: if True, print progress updates during training
            
        Returns:
            self: Returns self for method chaining
        """
        # Initialize the model with the image
        if logger.isEnabledFor(logging.INFO):
            t0 = time.perf_counter()
            logger.info("[fit] set_image starting: shape=%s", getattr(X, "shape", None))
        self.set_image(X, pixel_sampling_cache=pixel_sampling_cache)
        if logger.isEnabledFor(logging.INFO):
            logger.info("[fit] set_image done in %.2fs", time.perf_counter() - t0)
        if self.verbose:
            print(f"[fit] Starting training for {self.num_epochs} epochs...")
        
        # Train the model
        if self.verbose:
            print(f"Training ParametricPCC for {self.num_epochs} epochs...")
        if logger.isEnabledFor(logging.INFO):
            t_train = time.perf_counter()
            logger.info("[fit] training loop starting: epochs=%s", self.num_epochs)
        
        for epoch in tqdm.tqdm(range(self.num_epochs)):
            self.step()
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs} completed")
        
        if self.verbose:
            print("Training completed!")
        if self.verbose:
            print(f"[fit] Training completed after {self.num_epochs} epochs")
        if logger.isEnabledFor(logging.INFO):
            logger.info("[fit] training loop finished in %.2fs", time.perf_counter() - t_train)
        
        return self

    def resample(self, number_of_points):
        # Following the pattern from parametric_pcc.py
        # Operate on sampled_data, not the entire image
        if not hasattr(self, 'indices') or self.indices is None:
            sampled_indices = self.get_reference_points(
                self.sampled_data, number_of_points)
            self.indices = sampled_indices
            if self.verbose:
                print(f"[resample] Selected {len(sampled_indices)} reference points using '{self.sampling}' method")

        reference_points = self.sampled_data[self.indices, :]
        self.reference_points = torch.from_numpy(reference_points)
        if torch.cuda.is_available():
            self.reference_points = self.reference_points.cuda()
        if self.verbose:
            print(f"[resample] Reference points shape: {self.reference_points.shape}")

        # Use chebyshev/euclidean computation from parametric_pcc.py
        # Compute distances from sampled data to reference points
        if self.verbose:
            print(f"[resample] Computing distance matrices for {self.sampled_data.shape[0]} points...")
        chebyshev = pairwise_distances(
            self.sampled_data, reference_points, metric='chebyshev')
        chebyshev = chebyshev.argsort().argsort()

        euclidean = pairwise_distances(
            self.sampled_data, reference_points, metric='cosine')
        euclidean = euclidean.argsort().argsort()
        euclidean = (euclidean + chebyshev) / 2
        
        self.euclidean = torch.from_numpy(euclidean)
        if torch.cuda.is_available():
            self.euclidean = self.euclidean.cuda()
        if self.verbose:
            print(f"[resample] Distance matrices computed: shape={self.euclidean.shape}")

    def step(self):
        """
        Compute one optimization epoch using mini-batches.
        Following the pattern from parametric_pcc.py
        """
        batch_size = self.batch_size
        n_samples = self.sampled_data.shape[0]

        num_batches = (n_samples + batch_size - 1) // batch_size
        if self.step_counter == 0 and self.verbose:
            print(f"[step] Starting optimization: {num_batches} batches, batch_size={batch_size}, n_samples={n_samples}")

        sbw = getattr(self, "_sample_batch_weights", None)
        use_weighted = (
            sbw is not None
            and getattr(self, "finetune_miss_weighted_batches", False)
            and len(sbw) == n_samples
        )
        if use_weighted:
            p = np.asarray(sbw, dtype=np.float64)
            p = p / (p.sum() + 1e-12)
            epoch_order = np.random.default_rng(
                int(self.random_state) + int(self.step_counter)
            ).choice(n_samples, size=n_samples, replace=True, p=p)
        else:
            epoch_order = np.random.permutation(n_samples)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_mask = epoch_order[start_idx:end_idx]

            x = torch.from_numpy(self.sampled_data[batch_mask])
            if torch.cuda.is_available():
                x = x.cuda()

            # Must be a tensor (not Python 0): with cluster/category off and correlation gated
            # by k_epoch, some steps would otherwise leave loss as int and break backward().
            loss = torch.zeros((), device=x.device, dtype=torch.float32)

            batch_outputs = self._forward_training_embed(x, batch_mask)
            batch_reference_points = self._reference_embeddings()

            # Get reference points for this batch
            low_d_distances = torch.cdist(batch_outputs, batch_reference_points)

            high_d_distances = self.euclidean[batch_mask]

            # Clustering loss (weighted)
            if self.cluster and len(self.visualiation_to_cluster) > 0:
                cluster_loss_sum = 0.0
                for i in range(len(self.visualiation_to_cluster)):
                    layer = self.visualiation_to_cluster[i]
                    clusters = self.cluster_labels[i]
                    batch_clusters = torch.from_numpy(clusters[batch_mask]).long()
                    if torch.cuda.is_available():
                        batch_clusters = batch_clusters.cuda()

                    emb_i = batch_outputs
                    layer_output = layer(emb_i)
                    cluster_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(
                        layer_output / self.temperature, batch_clusters
                    )
                    cluster_loss_sum = cluster_loss_sum + cluster_loss
                loss = loss + self.cluster_loss_weight * (cluster_loss_sum / len(self.visualiation_to_cluster))

            # Category annotation loss / supervised loss (weighted)
            if self.category_head is not None and self.category_labels is not None:
                batch_cats = torch.from_numpy(self.category_labels[batch_mask]).long()
                if torch.cuda.is_available():
                    batch_cats = batch_cats.cuda()
                cat_output = self.category_head(batch_outputs)
                cat_loss = torch.nn.functional.cross_entropy(
                    cat_output / self.temperature, batch_cats, ignore_index=-1
                )
                loss = loss + self.category_loss_weight * cat_loss

            # Correlation loss (computed every k_epoch steps)
            if self.step_counter % self.k_epoch == self.k_epoch - 1:
                correlation_loss = -correlation(low_d_distances, high_d_distances).mean()

                if correlation_loss > 0 and self.verbose:
                    print(f"Correlation loss is positive: {correlation_loss}")

                alpha = -float(correlation_loss.detach().cpu().numpy())
                alpha = max(alpha, 1e-6)

                if self.cluster and len(self.visualiation_to_cluster) > 0:
                    loss = loss + correlation_loss * self.beta / alpha
                elif self.category_head is not None:
                    loss = loss + correlation_loss * self.beta / alpha
                else:
                    loss = correlation_loss

            # If nothing contributed a graph-connected term (e.g. cluster/category off and
            # correlation step not active this epoch), loss would be a constant with no grad.
            if not getattr(loss, "requires_grad", False):
                loss = loss + (batch_outputs.sum() * 0.0)

            l1w = float(getattr(self, "input_mask_l1_weight", 0.0))
            if l1w > 0.0 and isinstance(self.model, PCCWithLearnedInputMask):
                loss = loss + l1w * self.model.l1_mask_penalty()

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optim.step()

        # Update scheduler once per epoch (not per batch)
        # if self.scheduler is not None:
        #     self.scheduler.step()

        # Update k_epoch similar to parametric_pcc.py
        if self.step_counter % self.k_epoch == self.k_epoch - 1:
            # Linearly reduce k_epoch to 1 after 90% of epochs
            old_k_epoch = self.k_epoch
            self.k_epoch = max(1, int(self.k_epoch * 0.92))
            if old_k_epoch != self.k_epoch and self.verbose:
                print(f"[step] Updated k_epoch: {old_k_epoch} -> {self.k_epoch}")

        self.step_counter += 1

    def predict(self, img):
        """
        Predict visualization for an image using the trained model.
        Following the pattern from visualize function in supervised_dr.ipynb
        """
        if self.model is None:
            raise ValueError("Model must be initialized. Call set_image() first or provide a model in __init__.")
        
        # Set up image data without redoing clustering
        if self.verbose:
            print(f"[predict] Starting prediction for image shape: {img.shape}")
        self.img = img
        reshaped = self.img.reshape(
            self.img.shape[0] * self.img.shape[1], -1)
        if self.verbose:
            print(f"[predict] Reshaped to: {reshaped.shape}")
        
        result = self.predict_embedding(img)
        if self.verbose:
            print(f"[predict] Model inference completed: result shape={result.shape}")

        edge_map = getattr(self, "predict_edge_map", None)
        if bool(getattr(self, "predict_edge_smooth", False)) and edge_map is not None:
            tissue_mask = np.asarray(img).sum(axis=-1) > 0
            if self.verbose:
                print("[predict] Applying SoLaCE-guided edge-aware smoothing")
            result = solace_guided_bilateral_smooth(
                result,
                np.asarray(edge_map, dtype=np.float32),
                mask=tissue_mask,
                spatial_sigma=self.predict_edge_spatial_sigma,
                edge_scale=self.predict_edge_scale,
                iterations=self.predict_edge_iterations,
                radius=self.predict_edge_radius,
                range_sigma=self.predict_edge_range_sigma,
                sharpen_amount=getattr(self, "predict_edge_sharpen_amount", 0.0),
                sharpen_sigma=getattr(self, "predict_edge_sharpen_sigma", 1.0),
            )
        else:
            result = _spatial_gaussian_smooth_channels(result, self.predict_spatial_smooth_sigma)
        global_contrast = self._embedding_to_display_rgb(result, img)
        if self.verbose:
            print(f"[predict] Prediction completed: output shape={global_contrast.shape}")

        return global_contrast

    def get_mics_kmeans_metadata(self) -> dict:
        """
        MiCS cluster loss uses KMeans on ``sampled_data`` (or PCA-reduced data if
        ``cluster_on_pca``), one fit per entry in ``self.clusters``. Labels exist **only**
        at sampled pixel indices (``num_samples``), not on every image pixel.

        Returns:
            dict with keys ``clusters`` (list of k), ``sampled_flat_indices`` (1D int64),
            ``cluster_labels`` (list of 1D int label arrays, same length as indices each),
            ``cluster_on_pca``, ``cluster_pca_dims``, and ``image_shape`` (H, W) if ``self.img`` is set.
        """
        if self.img is None:
            raise ValueError("Call fit() first so image shape is known.")
        out: dict = {
            "clusters": list(self.clusters) if self.clusters is not None else [],
            "sampled_flat_indices": (
                np.asarray(self.sampled_flat_indices, dtype=np.int64)
                if getattr(self, "sampled_flat_indices", None) is not None
                else None
            ),
            "cluster_labels": (
                [np.asarray(x, dtype=np.int32) for x in self.cluster_labels]
                if getattr(self, "cluster_labels", None)
                else []
            ),
            "cluster_on_pca": bool(getattr(self, "cluster_on_pca", False)),
            "cluster_pca_dims": int(getattr(self, "cluster_pca_dims", 0)),
            "image_shape": (int(self.img.shape[0]), int(self.img.shape[1])),
        }
        return out

    def rasterize_mics_kmeans_labels(
        self,
        level_idx: int = 0,
        fill_value: int = -1,
    ) -> np.ndarray:
        """
        Map MiCS KMeans labels at ``level_idx`` (same order as ``self.clusters``) to a
        full H×W image. Non-sampled pixels are set to ``fill_value`` (default ``-1``).

        This is the **initialization** clustering used for cluster-loss targets, not the
        argmax of trained cluster heads on every pixel (use forward passes on
        ``visualiation_to_cluster`` for that instead).

        Requires ``cluster=True``, non-empty ``clusters``, and ``fit()`` completed.
        """
        if self.img is None:
            raise ValueError("Call fit() first.")
        if not getattr(self, "cluster_labels", None) or len(self.cluster_labels) == 0:
            raise ValueError(
                "MiCS KMeans labels are missing; use cluster=True and a non-empty clusters list."
            )
        if level_idx < 0 or level_idx >= len(self.cluster_labels):
            raise IndexError(
                f"level_idx must be in [0, {len(self.cluster_labels) - 1}], got {level_idx}"
            )
        H, W = int(self.img.shape[0]), int(self.img.shape[1])
        flat_idx = np.asarray(self.sampled_flat_indices, dtype=np.int64)
        labels = np.asarray(self.cluster_labels[level_idx], dtype=np.int32)
        if flat_idx.shape[0] != labels.shape[0]:
            raise ValueError(
                f"sampled_flat_indices length {flat_idx.shape[0]} != cluster_labels length {labels.shape[0]}"
            )
        out = np.full((H * W,), int(fill_value), dtype=np.int32)
        out[flat_idx] = labels
        return out.reshape(H, W)

    def rasterize_mics_kmeans_rgb_u8(
        self,
        level_idx: int = 0,
        *,
        fill_value: int = -1,
        background: tuple[int, int, int] = (0, 0, 0),
        color_scheme: str = "gist_rainbow",
    ) -> np.ndarray:
        """Same as ``rasterize_mics_kmeans_labels`` then ``cluster_id_labels_to_rgb_u8`` (see ``msi_visual.utils``)."""
        r = self.rasterize_mics_kmeans_labels(level_idx=level_idx, fill_value=fill_value)
        return cluster_id_labels_to_rgb_u8(
            r,
            fill_value=fill_value,
            background=background,
            color_scheme=color_scheme,
        )

    def predict_cluster_argmax_labels(self, img=None, level_idx: int = 0) -> np.ndarray:
        """
        Dense H×W cluster ids (``int32``) from the **trained** cluster head: full-image PCC
        embedding then ``argmax`` logits. Invalid (zero-intensity) pixels are ``-1``.

        Unlike ``rasterize_mics_kmeans_labels``, this assigns every valid pixel using the
        learned classifier, not the initialization KMeans labels on sampled pixels only.
        """
        if self.model is None:
            raise ValueError("Model must be initialized. Call fit() first.")
        if not getattr(self, "visualiation_to_cluster", None) or len(self.visualiation_to_cluster) == 0:
            raise ValueError("No cluster heads; use cluster=True and non-empty clusters.")
        if level_idx < 0 or level_idx >= len(self.visualiation_to_cluster):
            raise IndexError(
                f"level_idx must be in [0, {len(self.visualiation_to_cluster) - 1}], got {level_idx}"
            )
        if img is None:
            img = self.img
        if img is None:
            raise ValueError("Pass img or call fit() so self.img is set.")
        img = np.asarray(img, dtype=np.float32)
        H, W = int(img.shape[0]), int(img.shape[1])
        mask = img.sum(axis=-1) > 0
        reshaped = img.reshape(-1, img.shape[-1])
        device = next(self.model.parameters()).device
        classifier = self.visualiation_to_cluster[level_idx].to(device)
        self.model.eval()
        classifier.eval()
        batch_size = 1024 * 32
        flat_ids: list[np.ndarray] = []
        with torch.no_grad():
            x_batches = torch.split(torch.from_numpy(reshaped).float(), batch_size)
            for x_batch in x_batches:
                x_batch = x_batch.to(device)
                y_batch = self._embed_for_cluster_head(x_batch, level_idx)
                logits = classifier(y_batch)
                pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)
                flat_ids.append(pred)
        out = np.concatenate(flat_ids, axis=0).reshape(H, W)
        out = np.where(mask, out, -1).astype(np.int32)
        return out

    def predict_cluster_argmax_rgb_u8(
        self,
        img=None,
        level_idx: int = 0,
        *,
        fill_value: int = -1,
        background: tuple[int, int, int] = (0, 0, 0),
        color_scheme: str = "gist_rainbow",
    ) -> np.ndarray:
        """``predict_cluster_argmax_labels`` then ``cluster_id_labels_to_rgb_u8``."""
        r = self.predict_cluster_argmax_labels(img=img, level_idx=level_idx)
        return cluster_id_labels_to_rgb_u8(
            r,
            fill_value=fill_value,
            background=background,
            color_scheme=color_scheme,
        )

    def predict_cluster_soft_probs(self, img=None, level_idx: int = 0) -> np.ndarray:
        """
        Dense H×W×K float32 softmax probabilities from a trained cluster head.
        Invalid (zero-intensity) pixels are all-zero along K.
        """
        if self.model is None:
            raise ValueError("Model must be initialized. Call fit() first.")
        if not getattr(self, "visualiation_to_cluster", None) or len(self.visualiation_to_cluster) == 0:
            raise ValueError("No cluster heads; use cluster=True and non-empty clusters.")
        if level_idx < 0 or level_idx >= len(self.visualiation_to_cluster):
            raise IndexError(
                f"level_idx must be in [0, {len(self.visualiation_to_cluster) - 1}], got {level_idx}"
            )
        if img is None:
            img = self.img
        if img is None:
            raise ValueError("Pass img or call fit() so self.img is set.")
        img = np.asarray(img, dtype=np.float32)
        h, w = int(img.shape[0]), int(img.shape[1])
        mask = img.sum(axis=-1) > 0
        reshaped = img.reshape(-1, img.shape[-1])
        device = next(self.model.parameters()).device
        classifier = self.visualiation_to_cluster[level_idx].to(device)
        n_classes = int(classifier[-1].out_features)
        self.model.eval()
        classifier.eval()
        batch_size = 1024 * 32
        flat_probs = np.zeros((reshaped.shape[0], n_classes), dtype=np.float32)
        with torch.no_grad():
            x_batches = torch.split(torch.from_numpy(reshaped).float(), batch_size)
            offset = 0
            for x_batch in x_batches:
                x_batch = x_batch.to(device)
                emb = self._embed_for_cluster_head(x_batch, level_idx)
                logits = classifier(emb).detach().cpu().numpy().astype(np.float64, copy=False)
                logits = logits - np.max(logits, axis=1, keepdims=True)
                exps = np.exp(logits)
                probs = exps / np.sum(exps, axis=1, keepdims=True)
                n0 = probs.shape[0]
                flat_probs[offset : offset + n0] = probs.astype(np.float32, copy=False)
                offset += n0
        out = flat_probs.reshape(h, w, n_classes)
        out[~mask] = 0.0
        return out

    def predict_cluster_soft_rgb_u8(
        self,
        img=None,
        level_idx: int = 0,
        *,
        background: tuple[int, int, int] = (0, 0, 0),
        color_scheme: str = "gist_rainbow",
    ) -> np.ndarray:
        """Soft cluster map: RGB = sum_k softmax(logits)_k * palette[k] (not argmax)."""
        if self.model is None:
            raise ValueError("Model must be initialized. Call fit() first.")
        if not getattr(self, "visualiation_to_cluster", None) or len(self.visualiation_to_cluster) == 0:
            raise ValueError("No cluster heads; use cluster=True and non-empty clusters.")
        if level_idx < 0 or level_idx >= len(self.visualiation_to_cluster):
            raise IndexError(
                f"level_idx must be in [0, {len(self.visualiation_to_cluster) - 1}], got {level_idx}"
            )
        if img is None:
            img = self.img
        if img is None:
            raise ValueError("Pass img or call fit() so self.img is set.")
        img = np.asarray(img, dtype=np.float32)
        h, w = int(img.shape[0]), int(img.shape[1])
        mask = img.sum(axis=-1) > 0
        reshaped = img.reshape(-1, img.shape[-1])
        device = next(self.model.parameters()).device
        classifier = self.visualiation_to_cluster[level_idx].to(device)
        n_classes = int(classifier[-1].out_features)
        palette = cluster_palette_rgb_u8(n_classes, color_scheme).astype(np.float64)
        self.model.eval()
        classifier.eval()
        batch_size = 1024 * 32
        rgb_flat = np.zeros((reshaped.shape[0], 3), dtype=np.float64)
        with torch.no_grad():
            x_batches = torch.split(torch.from_numpy(reshaped).float(), batch_size)
            offset = 0
            for x_batch in x_batches:
                x_batch = x_batch.to(device)
                emb = self._embed_for_cluster_head(x_batch, level_idx)
                logits = classifier(emb).detach().cpu().numpy().astype(np.float64, copy=False)
                logits = logits - np.max(logits, axis=1, keepdims=True)
                exps = np.exp(logits)
                probs = exps / np.sum(exps, axis=1, keepdims=True)
                n0 = probs.shape[0]
                rgb_flat[offset : offset + n0] = probs @ palette
                offset += n0
        rgb = np.clip(rgb_flat, 0.0, 255.0).astype(np.uint8).reshape(h, w, 3)
        bg = np.asarray(background, dtype=np.uint8)
        rgb[~mask] = bg
        return rgb

    def uncertainty2(self, img):
        """
        Computes the average entropy per pixel across all classifiers.
        Returns an entropy map of shape (H, W).
        """
        if self.model is None:
            raise ValueError("Model must be initialized. Call set_image() first or provide a model in __init__.")
        if not hasattr(self, "visualiation_to_cluster") or len(self.visualiation_to_cluster) == 0:
            raise ValueError("No classifiers available - enable clustering and call set_image() first.")

        if self.verbose:
            print(f"[uncertainty] Starting uncertainty prediction for image shape: {img.shape}")
        self.img = img
        reshaped = self.img.reshape(
            self.img.shape[0] * self.img.shape[1], -1)
        if self.verbose:
            print(f"[uncertainty] Reshaped to: {reshaped.shape}")

        # Per-classifier embeddings (MoE routes each head to its expert; single PCC uses fused output).
        self.model.eval()
        device = next(self.model.parameters()).device
        batch_size = 1024 * 32
        x_torch = torch.from_numpy(reshaped).float()
        x_batches = torch.split(x_torch, batch_size)

        entropies = []
        for idx, classifier in enumerate(self.visualiation_to_cluster):
            classifier.eval()
            classifier = classifier.to(device)
            with torch.no_grad():
                batch_outs = []
                for x_batch in x_batches:
                    x_batch = x_batch.to(device)
                    emb = self._embed_for_cluster_head(x_batch, idx)
                    logits = classifier(emb).cpu().numpy()
                    # Turn logits into probabilities with softmax
                    # Avoid overflow: subtract max for numerical stability
                    logits = logits - np.max(logits, axis=1, keepdims=True)
                    exps = np.exp(logits)
                    probs = exps / np.sum(exps, axis=1, keepdims=True)
                    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                    batch_outs.append(entropy)
                all_entropy = np.concatenate(batch_outs, axis=0)
                entropies.append(all_entropy)

        # Average entropy over all classifiers
        mean_entropy = np.mean(np.stack(entropies, axis=1), axis=1)  # (num_pixels,)

        # Reshape back to image shape
        H, W = img.shape[0], img.shape[1]
        entropy_map = mean_entropy.reshape((H, W))

        # Mask zero-intensity pixels as zero entropy if desired (optional)
        mask = img.sum(axis=-1) > 0
        entropy_map[~mask] = 0.0

        if self.verbose:
            print(f"[uncertainty] Average entropy map computed: shape={entropy_map.shape}")

        # Scale the entropy map to [0, 1]
        min_val = np.min(entropy_map)
        max_val = np.max(entropy_map)
        if max_val > min_val:
            entropy_map = (entropy_map - min_val) / (max_val - min_val)
        else:
            entropy_map = np.zeros_like(entropy_map)

        return entropy_map


    def __call__(self, img):
        return self.predict(img)
