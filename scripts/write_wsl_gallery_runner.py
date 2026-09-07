#!/usr/bin/env python3
"""Write ~/run_wsl_gallery.sh (LF line endings) for WSL GPU gallery runs."""
from pathlib import Path

content = """#!/usr/bin/env bash
set -eu
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate maldi-wsl
source "$HOME/wsl_gpu_env.sh"
cd /mnt/c/Users/PahnkeLab/Desktop/maldi_msi
DATA=/mnt/e/MSImaging-data/_msi_visual/Extractions/NRL4485-s2/5_bins
exec python -u scripts/mics_cross_infer_gallery.py \\
  "gallery.npy_paths=[\\"${DATA}/0.npy\\",\\"${DATA}/1.npy\\",\\"${DATA}/2.npy\\",\\"${DATA}/3.npy\\"]"
"""
out = Path.home() / "run_wsl_gallery.sh"
out.write_text(content, newline="\n")
out.chmod(0o755)
print(f"wrote {out}")
