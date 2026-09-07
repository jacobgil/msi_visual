"""
Product-of-experts fusion for Spatial MiCS cluster soft assignments (α).

Two experts (e.g. coarse context vs fine bandpass) are cluster-aligned via
Hungarian matching on spatial correlation, then fused:

    α_fused_k ∝ α_coarse_k^w_c · α_fine_k^w_f   (renormalized per pixel)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from msi_visual.utils import cluster_palette_rgb_u8


def spatial_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation of two H×W maps on tissue."""
    m = np.asarray(mask, dtype=bool)
    va = np.asarray(a, dtype=np.float64)[m].ravel()
    vb = np.asarray(b, dtype=np.float64)[m].ravel()
    if va.size < 2:
        return 0.0
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = float(np.sqrt((va * va).sum() * (vb * vb).sum())) + 1e-12
    return float((va * vb).sum() / denom)


def align_cluster_alpha(
    alpha_ref: np.ndarray,
    alpha_other: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Permute ``alpha_other`` cluster axis to best match ``alpha_ref``.

    Returns
    -------
    perm : (K,) int — ``alpha_other[..., perm]`` aligns to ``alpha_ref``.
    alpha_aligned : H×W×K
    meta : alignment diagnostics
    """
    ref = np.asarray(alpha_ref, dtype=np.float32)
    other = np.asarray(alpha_other, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if ref.shape != other.shape:
        raise ValueError(f"alpha shapes must match: {ref.shape} vs {other.shape}")
    if ref.ndim != 3:
        raise ValueError(f"Expected H×W×K alpha, got {ref.shape}")

    k = int(ref.shape[-1])
    cost = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            cost[i, j] = 1.0 - spatial_correlation(ref[:, :, i], other[:, :, j], m)

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.asarray(col_ind[np.argsort(row_ind)], dtype=np.int64)
    aligned = other[:, :, perm]
    mean_corr = float(np.mean([spatial_correlation(ref[:, :, i], aligned[:, :, i], m) for i in range(k)]))
    meta = {
        "perm_other_to_ref": perm.tolist(),
        "mean_aligned_correlation": mean_corr,
        "pairwise_cost": cost.tolist(),
    }
    return perm, aligned, meta


def product_of_experts_alpha(
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
    mask: np.ndarray,
    *,
    w_a: float = 0.5,
    w_b: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Fuse two aligned H×W×K soft assignment maps."""
    a = np.asarray(alpha_a, dtype=np.float64)
    b = np.asarray(alpha_b, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"alpha shapes must match: {a.shape} vs {b.shape}")

    wa = float(w_a)
    wb = float(w_b)
    if wa < 0.0 or wb < 0.0:
        raise ValueError("PoE weights must be non-negative")
    if wa == 0.0 and wb == 0.0:
        raise ValueError("At least one PoE weight must be > 0")

    a = np.clip(a, eps, 1.0)
    b = np.clip(b, eps, 1.0)
    log_p = wa * np.log(a) + wb * np.log(b)
    log_p -= np.max(log_p, axis=-1, keepdims=True)
    fused = np.exp(log_p)
    fused_sum = np.sum(fused, axis=-1, keepdims=True)
    fused = np.divide(fused, fused_sum, out=np.zeros_like(fused), where=fused_sum > eps)
    fused = fused.astype(np.float32, copy=False)
    fused[~m] = 0.0
    return fused


def alpha_to_soft_rgb_u8(
    alpha: np.ndarray,
    mask: np.ndarray,
    *,
    color_scheme: str = "gist_rainbow",
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """RGB visualization: sum_k α_k · palette[k]."""
    a = np.asarray(alpha, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    k = int(a.shape[-1])
    palette = cluster_palette_rgb_u8(k, color_scheme).astype(np.float64)
    rgb = np.clip(a @ palette, 0.0, 255.0).astype(np.uint8)
    bg = np.asarray(background, dtype=np.uint8)
    rgb[~m] = bg
    return rgb
