#!/usr/bin/env python3
"""Smoke test for WSL GPU: PyTorch, TensorFlow, and Parametric UMAP fit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    import tensorflow as tf

    print("=== GPU smoke test ===")
    try:
        import torch

        print("PyTorch CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  device:", torch.cuda.get_device_name(0))
    except ImportError:
        print("PyTorch: not installed")
    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("TF GPUs:", gpus)

    from msi_visual.parametric_umap import UMAPVirtualStain

    m = UMAPVirtualStain(
        n_components=3,
        n_training_epochs=1,
        n_epochs=2,
        n_neighbors=50,
        num_samples=5000,
        pixel_sampling="random",
    )
    x = np.random.randn(120, 120, 200).astype(np.float32)
    x[x < 0] = 0
    print("=== Parametric UMAP fit (small) ===")
    m.fit([x], keras_fit_kwargs={})
    print("pUMAP fit OK")
    pred = m.predict(x)
    print("pUMAP predict shape:", np.asarray(pred).shape)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
