"""H&E mosaic + MiCS overlay helpers for pathology visualization panels."""

from __future__ import annotations

import cv2
import numpy as np


def resize_rgb(rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.shape[0] == target_h and rgb.shape[1] == target_w:
        return rgb
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def alpha_blend_rgb(
    base_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    alpha: float,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Semi-transparent overlay; optional mask limits blend to tissue cells."""
    base = np.asarray(base_rgb, dtype=np.float32)
    over = np.asarray(overlay_rgb, dtype=np.float32)
    if base.shape[:2] != over.shape[:2]:
        over = resize_rgb(over, base.shape[0], base.shape[1]).astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    if mask is None:
        out = (1.0 - a) * base + a * over
    else:
        m = np.asarray(mask, dtype=bool)
        if m.shape != base.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        out = base.copy()
        out[m] = (1.0 - a) * base[m] + a * over[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def side_by_side_rgb(left: np.ndarray, right: np.ndarray, *, gap: int = 8) -> np.ndarray:
    left = np.asarray(left, dtype=np.uint8)
    right = np.asarray(right, dtype=np.uint8)
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        left = resize_rgb(left, h, int(round(left.shape[1] * h / left.shape[0])))
    if right.shape[0] != h:
        right = resize_rgb(right, h, int(round(right.shape[1] * h / right.shape[0])))
    if gap > 0:
        pad = np.zeros((h, gap, 3), dtype=np.uint8)
        return np.concatenate([left, pad, right], axis=1)
    return np.concatenate([left, right], axis=1)
