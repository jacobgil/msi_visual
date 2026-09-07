#!/usr/bin/env python3
"""
Register H&E thumbnails to MSI pixel grids (multi-stage coarse-to-fine ECC).

Writes ``registration.json`` (+ preview PNGs) next to each H&E thumbnail folder.

Run from repo root:
  python scripts/register_he_msi.py
  python scripts/register_he_msi.py input.npy_dir="E:/.../MSimaging" input.npy_glob="**/5_bins/0.npy"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msi_visual.registration.he_msi import (
    apply_registration_to_he,
    load_registration,
    register_he_to_msi,
    resolve_registration_path,
    save_registration,
)
from msi_visual.registration.hydra_cfg import registration_kwargs_from_cfg, resolve_mics_from_registration_cfg
from msi_visual.registration.paths import msi_sample_id, resolve_he_thumbnail, slide_key_from_npy

logger = logging.getLogger(__name__)


def _resolve_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _resolve_npy_paths(cfg: DictConfig) -> list[Path]:
    from scripts.bayes_tune_mics_lmc import _resolve_npy_paths

    stub = OmegaConf.create(
        {
            "input": {
                "npy_path": OmegaConf.select(cfg, "input.npy_path"),
                "npy_paths": OmegaConf.select(cfg, "input.npy_paths", default=[]),
                "npy_dir": OmegaConf.select(cfg, "input.npy_dir"),
                "npy_glob": OmegaConf.select(cfg, "input.npy_glob", default="**/5_bins/0.npy"),
            }
        }
    )
    return _resolve_npy_paths(stub)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _msi_shape_hw(npy_path: Path, *, transpose_msi: bool) -> tuple[int, int]:
    arr = np.load(str(npy_path), mmap_mode="r")
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        if transpose_msi:
            h, w = int(arr.shape[1]), int(arr.shape[2])
        else:
            h, w = int(arr.shape[0]), int(arr.shape[1])
    else:
        raise ValueError(f"Unexpected MSI shape {arr.shape} for {npy_path}")
    return h, w


@hydra.main(version_base=None, config_path="configs", config_name="register_he_msi")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    npy_paths = _resolve_npy_paths(cfg)
    rank_cfg = _resolve_path(str(cfg.registration.rank_config)) if cfg.registration.rank_config else None
    skip_existing = bool(cfg.output.skip_existing)
    transpose_msi = bool(cfg.data.transpose_msi)
    manifest: list[dict[str, Any]] = []

    for img_idx, npy_path in enumerate(npy_paths):
        slide_key = slide_key_from_npy(npy_path)
        sample_id = msi_sample_id(slide_key) or slide_key
        he_path = resolve_he_thumbnail(slide_key, cfg.he, resolve_path=_resolve_path)
        if he_path is None:
            logger.warning("[%s] no H&E thumbnail — skip", slide_key)
            manifest.append({"slide_key": slide_key, "status": "no_he"})
            continue

        reg_path = resolve_registration_path(he_path, str(cfg.output.registration_filename), slide_key=slide_key)
        if skip_existing and reg_path.is_file():
            logger.info("[%s] registration exists — skip", slide_key)
            manifest.append({"slide_key": slide_key, "status": "skipped", "registration_path": str(reg_path)})
            continue

        logger.info("[%d/%d] Registering %s", img_idx + 1, len(npy_paths), slide_key)
        try:
            target_hw = _msi_shape_hw(npy_path, transpose_msi=transpose_msi)
            strategy = str(OmegaConf.select(cfg, "registration.strategy", default="mask_only")).strip().lower()
            mics_rgb, mics_source = None, None
            if strategy == "multistage":
                mics_rgb, mics_source = resolve_mics_from_registration_cfg(
                    cfg,
                    npy_path=npy_path,
                    img_idx=img_idx,
                    target_hw=target_hw,
                    resolve_path=_resolve_path,
                )
            result = register_he_to_msi(
                npy_path=npy_path,
                he_path=he_path,
                slide_key=slide_key,
                sample_id=sample_id,
                **registration_kwargs_from_cfg(
                    cfg,
                    rank_path=rank_cfg,
                    mics_rgb=mics_rgb,
                    mics_source=mics_source,
                ),
            )
            save_registration(result, reg_path)

            he_reg = apply_registration_to_he(_load_rgb(he_path), result, tuple(result.msi_shape_hw))
            he_reg_path = he_path.parent / str(cfg.output.registered_he_filename)
            Image.fromarray(he_reg).save(he_reg_path)

            manifest.append(
                {
                    "slide_key": slide_key,
                    "status": "ok",
                    "registration_path": str(reg_path),
                    "success": result.success,
                    "correlation": result.correlation,
                    "method": result.method,
                    "rotation_deg": result.rotation_deg,
                    "mics_source": result.mics_source,
                    "stages": result.stages,
                    "he_registered": str(he_reg_path),
                }
            )
        except Exception as exc:
            logger.exception("Failed registration for %s", slide_key)
            manifest.append({"slide_key": slide_key, "status": "error", "error": str(exc)})

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "register_he_msi_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    logger.info("Finished %d slide(s)", len(manifest))


if __name__ == "__main__":
    main()
