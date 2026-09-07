"""Lightweight MSI I/O + seeding for visualization scripts (no torch/faiss imports)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    sums = msi.sum(axis=-1, keepdims=True)
    return msi / (sums + 1e-8)


def load_msi(path: Path, *, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI shape HxWxC (or CxHxW with transpose), got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI contains negative values; expected non-negative inputs.")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
