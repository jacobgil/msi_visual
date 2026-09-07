#!/usr/bin/env python3
"""Write ~/wsl_gpu_env.sh for split TF (cuDNN 8) + PyTorch (cuDNN 9) CUDA libs."""
from pathlib import Path

# TensorFlow 2.15 needs cuDNN 8 from site-packages; PyTorch 2.5 needs cuDNN 9
# from ~/local/torchlibs. Torch also loads NCCL via RPATH from site-packages, so
# install once: pip install nvidia-nccl-cu12==2.21.5
content = """#!/usr/bin/env bash
set -eu
ENV_PREFIX="/home/pahnkelab/miniconda3/envs/maldi-wsl"
_SP="${ENV_PREFIX}/lib/python3.11/site-packages/nvidia"
_TL="${HOME}/local/torchlibs/nvidia"
paths=""
for d in "${_TL}"/*/lib; do
  if [ -d "$d" ]; then paths="${paths}:$d"; fi
done
if [ -d "${_SP}/cudnn/lib" ]; then paths="${paths}:${_SP}/cudnn/lib"; fi
export LD_LIBRARY_PATH="${paths#:}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"""
out = Path.home() / "wsl_gpu_env.sh"
out.write_text(content, newline="\n")
print(f"wrote {out}")
print("Run once in maldi-wsl: pip install nvidia-nccl-cu12==2.21.5")

