#!/usr/bin/env python3
"""Quick check: TensorFlow and PyTorch GPU visibility in one process."""
import tensorflow as tf

print("TF GPUs:", tf.config.list_physical_devices("GPU"))

import torch

print("torch CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(100, device="cuda")
    print("torch matmul OK:", tuple((x @ x.T).shape))
