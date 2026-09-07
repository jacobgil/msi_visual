"""LAB fusion: H&E luminance + MiCS chroma."""

from __future__ import annotations

import cv2
import numpy as np


def lab_overlay_he_l_mics_ab(
    he_rgb: np.ndarray,
    mics_rgb: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    background_rgb: tuple[int, int, int] = (14, 14, 16),
) -> np.ndarray:
    """
    Build RGB where L comes from H&E and a/b come from MiCS (OpenCV LAB, uint8).

    Pixels outside ``mask`` (when provided) are filled with ``background_rgb``.
    """
    he = np.asarray(he_rgb, dtype=np.uint8)
    mics = np.asarray(mics_rgb, dtype=np.uint8)
    if he.shape[:2] != mics.shape[:2]:
        raise ValueError(f"H&E/MiCS size mismatch: {he.shape[:2]} vs {mics.shape[:2]}")
    he_lab = cv2.cvtColor(he, cv2.COLOR_RGB2LAB)
    mics_lab = cv2.cvtColor(mics, cv2.COLOR_RGB2LAB)
    merged = cv2.merge([he_lab[:, :, 0], mics_lab[:, :, 1], mics_lab[:, :, 2]])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != out.shape[:2]:
            raise ValueError(f"Mask shape {m.shape} != image shape {out.shape[:2]}")
        bg = np.full_like(out, background_rgb, dtype=np.uint8)
        bg[m] = out[m]
        out = bg
    return out
