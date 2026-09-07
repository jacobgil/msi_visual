"""
SpatialMiCS input features: exactly 3× channel count.

Default (bandpass, sharper):
    X_spatial = concat( X ,  |X − blur(X, σ_regional)| ,  |X − blur(X, σ_fine)| )

Legacy (context, tends to blur MiCS output):
    X_spatial = concat( X ,  blur(X, σ_regional) ,  |X − blur(X, σ_fine)| )

All operations are mask-aware. Blocks are optionally percentile-normalized per channel.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import cv2
import numpy as np

DetailMode = Literal["abs", "signed"]
BlockMode = Literal["bandpass", "context"]
SingleBlockName = Literal[
    "raw",
    "regional_blur",
    "regional_contrast",
    "fine_contrast",
    "fine_detail",
]


def masked_gaussian_blur_channels(
    x: np.ndarray,
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Masked mean via normalized Gaussian blur on H×W×C float cube."""
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if x.ndim != 3:
        raise ValueError(f"Expected H×W×C, got shape {x.shape}")
    if mask.shape != x.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != spatial {x.shape[:2]}")
    s = float(sigma)
    if s <= 0.0:
        out = x.copy()
        out[~mask] = 0.0
        return out

    m = mask.astype(np.float32)
    k = int(max(3, round(6 * s) | 1))
    kk = (k, k)
    fs = s
    out = np.zeros_like(x, dtype=np.float32)
    den = cv2.GaussianBlur(m, kk, sigmaX=fs, sigmaY=fs)
    for ci in range(x.shape[2]):
        num = cv2.GaussianBlur(x[:, :, ci] * m, kk, sigmaX=fs, sigmaY=fs)
        out[:, :, ci] = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-8)
    out[~mask] = 0.0
    return out


def normalize_block_percentile(
    block: np.ndarray,
    mask: np.ndarray,
    *,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """Per-channel percentile stretch on tissue; output in [0, 1], background zero."""
    block = np.asarray(block, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(block, dtype=np.float32)
    flat_mask = mask.ravel()
    for c in range(block.shape[2]):
        ch = block[:, :, c]
        vals = ch.ravel()[flat_mask]
        if vals.size < 2:
            out[:, :, c] = np.where(mask, np.clip(ch, 0.0, None), 0.0)
            continue
        lo = float(np.percentile(vals, low_pct))
        hi = float(np.percentile(vals, high_pct))
        scaled = (ch - lo) / (hi - lo + 1e-8)
        out[:, :, c] = np.clip(scaled, 0.0, 1.0)
    out[~mask] = 0.0
    return out


def _detail_from_delta(delta: np.ndarray, mask: np.ndarray, detail_mode: DetailMode) -> np.ndarray:
    if str(detail_mode).strip().lower() == "signed":
        out = np.asarray(delta, dtype=np.float32)
    else:
        out = np.abs(delta).astype(np.float32, copy=False)
    out = out.copy()
    out[~mask] = 0.0
    return out


def build_spatial_mics_cube(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    sigma_regional: float = 3.0,
    sigma_fine: float = 0.7,
    block_mode: BlockMode = "bandpass",
    detail_mode: DetailMode = "abs",
    block_normalize: bool = True,
    block_low_pct: float = 1.0,
    block_high_pct: float = 99.0,
    block_gains: Sequence[float] = (1.0, 1.0, 1.25),
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build H×W×(3C) spatial feature cube from H×W×C spectra.

    bandpass (default, sharper):
      1. raw X
      2. |X − blur(X, σ_regional)|
      3. |X − blur(X, σ_fine)|

    context (legacy; can over-smooth MiCS):
      1. raw X
      2. blur(X, σ_regional)
      3. |X − blur(X, σ_fine)|
    """
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if x.ndim != 3:
        raise ValueError(f"Expected H×W×C, got shape {x.shape}")
    if not np.any(mask):
        raise ValueError("Empty valid mask for spatial features.")

    mode = str(block_mode).strip().lower()
    regional_blur = masked_gaussian_blur_channels(x, mask, float(sigma_regional))
    fine_blur = masked_gaussian_blur_channels(x, mask, float(sigma_fine))
    regional_detail = _detail_from_delta(x - regional_blur, mask, detail_mode)
    fine_detail = _detail_from_delta(x - fine_blur, mask, detail_mode)

    if mode == "context":
        blocks: list[np.ndarray] = [x, regional_blur, fine_detail]
        block_names = ["raw", "regional_blur", "fine_detail"]
    elif mode == "bandpass":
        blocks = [x, regional_detail, fine_detail]
        block_names = ["raw", "regional_contrast", "fine_contrast"]
    else:
        raise ValueError(f"block_mode must be 'bandpass' or 'context'; got {block_mode!r}")

    gains = tuple(float(g) for g in block_gains)
    if len(gains) != 3:
        raise ValueError(f"block_gains must have length 3; got {len(gains)}")

    if block_normalize:
        blocks = [
            normalize_block_percentile(
                b,
                mask,
                low_pct=float(block_low_pct),
                high_pct=float(block_high_pct),
            )
            for b in blocks
        ]

    blocks = [np.clip(b * float(g), 0.0, 1.0) for b, g in zip(blocks, gains)]
    cube = np.concatenate(blocks, axis=-1).astype(np.float32, copy=False)
    c = int(x.shape[-1])
    meta = {
        "input_channels": c,
        "sigma_regional": float(sigma_regional),
        "sigma_fine": float(sigma_fine),
        "block_mode": mode,
        "detail_mode": str(detail_mode),
        "block_normalize": bool(block_normalize),
        "block_low_pct": float(block_low_pct),
        "block_high_pct": float(block_high_pct),
        "block_gains": list(gains),
        "block_channels": c,
        "output_channels": int(cube.shape[-1]),
        "blocks": block_names,
        "feature_multiplier": 3,
    }
    return cube, meta


def _spatial_blocks_dict(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    sigma_regional: float,
    sigma_fine: float,
    detail_mode: DetailMode,
) -> dict[str, np.ndarray]:
    regional_blur = masked_gaussian_blur_channels(x, mask, float(sigma_regional))
    fine_blur = masked_gaussian_blur_channels(x, mask, float(sigma_fine))
    return {
        "raw": x,
        "regional_blur": regional_blur,
        "regional_contrast": _detail_from_delta(x - regional_blur, mask, detail_mode),
        "fine_contrast": _detail_from_delta(x - fine_blur, mask, detail_mode),
        "fine_detail": _detail_from_delta(x - fine_blur, mask, detail_mode),
    }


def build_spatial_mics_single_block(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    block: SingleBlockName = "fine_contrast",
    sigma_regional: float = 3.0,
    sigma_fine: float = 0.7,
    detail_mode: DetailMode = "abs",
    block_normalize: bool = True,
    block_low_pct: float = 1.0,
    block_high_pct: float = 99.0,
    block_gain: float = 1.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build H×W×C cube from one spatial block (for residual MiCS stages)."""
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    name = str(block).strip().lower()
    blocks = _spatial_blocks_dict(
        x,
        mask,
        sigma_regional=sigma_regional,
        sigma_fine=sigma_fine,
        detail_mode=detail_mode,
    )
    if name not in blocks:
        raise ValueError(f"block must be one of {sorted(blocks)}; got {block!r}")

    selected = blocks[name]
    if block_normalize:
        selected = normalize_block_percentile(
            selected,
            mask,
            low_pct=float(block_low_pct),
            high_pct=float(block_high_pct),
        )
    selected = np.clip(selected * float(block_gain), 0.0, 1.0).astype(np.float32, copy=False)
    c = int(x.shape[-1])
    meta = {
        "input_channels": c,
        "sigma_regional": float(sigma_regional),
        "sigma_fine": float(sigma_fine),
        "block": name,
        "detail_mode": str(detail_mode),
        "block_normalize": bool(block_normalize),
        "block_low_pct": float(block_low_pct),
        "block_high_pct": float(block_high_pct),
        "block_gain": float(block_gain),
        "output_channels": c,
        "feature_multiplier": 1,
    }
    return selected, meta


def masked_local_zscore_channels(
    x: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    *,
    eps: float = 1e-6,
    clip_z: float | None = 5.0,
) -> np.ndarray:
    """
    Mask-aware local z-score per channel:

        z = (X − blur(X, σ)) / sqrt(blur(X², σ) − blur(X, σ)² + ε)
    """
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    local_mean = masked_gaussian_blur_channels(x, mask, float(sigma))
    local_mean_sq = masked_gaussian_blur_channels(x * x, mask, float(sigma))
    var = np.clip(local_mean_sq - local_mean * local_mean, 0.0, None)
    std = np.sqrt(var + float(eps))
    z = (x - local_mean) / std
    if clip_z is not None:
        z = np.clip(z, -float(clip_z), float(clip_z))
    z = z.astype(np.float32, copy=False)
    z[~mask] = 0.0
    return z


def build_local_zscore_mics_input(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float = 0.7,
    eps: float = 1e-6,
    clip_z: float | None = 5.0,
    block_normalize: bool = True,
    block_low_pct: float = 1.0,
    block_high_pct: float = 99.0,
    block_gain: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build H×W×C local z-score cube for MiCS."""
    z = masked_local_zscore_channels(
        x,
        mask,
        sigma,
        eps=eps,
        clip_z=clip_z,
    )
    if block_normalize:
        z = normalize_block_percentile(
            z,
            mask,
            low_pct=float(block_low_pct),
            high_pct=float(block_high_pct),
        )
    z = np.clip(z * float(block_gain), 0.0, 1.0).astype(np.float32, copy=False)
    meta = {
        "feature_type": "local_zscore",
        "sigma": float(sigma),
        "eps": float(eps),
        "clip_z": None if clip_z is None else float(clip_z),
        "block_normalize": bool(block_normalize),
        "block_low_pct": float(block_low_pct),
        "block_high_pct": float(block_high_pct),
        "block_gain": float(block_gain),
        "input_channels": int(x.shape[-1]),
        "output_channels": int(z.shape[-1]),
    }
    return z, meta


def build_log_ratio_mics_input(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float = 3.0,
    eps: float = 1e-8,
    clip_log_ratio: float | None = 3.0,
    block_normalize: bool = True,
    block_low_pct: float = 1.0,
    block_high_pct: float = 99.0,
    block_gain: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Local log-enrichment vs neighborhood:

        log X − log blur(X, σ)
    """
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    blur = masked_gaussian_blur_channels(x, mask, float(sigma))
    ratio = np.log(x + float(eps)) - np.log(blur + float(eps))
    if clip_log_ratio is not None:
        ratio = np.clip(ratio, -float(clip_log_ratio), float(clip_log_ratio))
    ratio = ratio.astype(np.float32, copy=False)
    ratio[~mask] = 0.0
    if block_normalize:
        ratio = normalize_block_percentile(
            ratio,
            mask,
            low_pct=float(block_low_pct),
            high_pct=float(block_high_pct),
        )
    ratio = np.clip(ratio * float(block_gain), 0.0, 1.0).astype(np.float32, copy=False)
    meta = {
        "feature_type": "log_ratio",
        "sigma": float(sigma),
        "eps": float(eps),
        "clip_log_ratio": None if clip_log_ratio is None else float(clip_log_ratio),
        "block_normalize": bool(block_normalize),
        "block_low_pct": float(block_low_pct),
        "block_high_pct": float(block_high_pct),
        "block_gain": float(block_gain),
        "input_channels": int(x.shape[-1]),
        "output_channels": int(ratio.shape[-1]),
    }
    return ratio, meta


def build_spatial_mics_input(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    sigma_regional: float = 3.0,
    sigma_fine: float = 0.7,
    block_mode: BlockMode = "bandpass",
    detail_mode: DetailMode = "abs",
    block_normalize: bool = True,
    block_low_pct: float = 1.0,
    block_high_pct: float = 99.0,
    block_gains: Sequence[float] = (1.0, 1.0, 1.25),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build 3× spatial feature cube from raw H×W×C MSI input."""
    return build_spatial_mics_cube(
        x,
        mask,
        sigma_regional=sigma_regional,
        sigma_fine=sigma_fine,
        block_mode=block_mode,
        detail_mode=detail_mode,
        block_normalize=block_normalize,
        block_low_pct=block_low_pct,
        block_high_pct=block_high_pct,
        block_gains=block_gains,
    )
