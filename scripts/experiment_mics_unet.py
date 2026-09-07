#!/usr/bin/env python3
"""
MiCS-U-Net experiment: spatial encoder on full MSI cube vs per-pixel MiCS PCC.

Trains ``MSIParametricMiCSUnet`` with soft Dice + contrast vs HD reference edges,
using the same pixel sampling / MiCS cluster+correlation losses as the LMC trainer.

Run (repo root):
    python scripts/experiment_mics_unet.py
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_mics_gnn import (  # noqa: E402
    _global_continuous_dice,
    _load_msi,
    _mics_kwargs_from_cfg,
    _parse_clusters,
    _rgb_diff_heatmap,
    _save_edge_comparison_panel,
    _sampling_cache_kwargs,
)
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from msi_visual.parametric_mics_unet import MSIParametricMiCSUnet  # noqa: E402
from train_parametric_mics_lmc import _build_hd_edge_map_for_train  # noqa: E402

logger = logging.getLogger(__name__)


def _unet_kwargs_from_cfg(cfg: DictConfig, mics_kw: dict[str, Any]) -> dict[str, Any]:
    un = OmegaConf.to_container(getattr(cfg, "unet", {}), resolve=True)
    if not isinstance(un, dict):
        un = {}
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    out = dict(mics_kw)
    out["num_epochs"] = int(un.get("num_epochs", merged.get("num_epochs", 40)))
    out["unet_base_channels"] = int(un.get("unet_base_channels", merged.get("unet_base_channels", 32)))
    out["dice_loss_weight"] = float(un.get("dice_loss_weight", merged.get("dice_loss_weight", 0.01)))
    out["contrast_loss_weight"] = float(
        un.get("contrast_loss_weight", merged.get("contrast_loss_weight", 0.0))
    )
    out["predict_spatial_smooth_sigma"] = float(
        un.get("predict_spatial_smooth_sigma", mics_kw.get("predict_spatial_smooth_sigma", 0.0))
    )
    if "lr" in un:
        out["lr"] = float(un["lr"])
    if "k_epoch" in un:
        out["k_epoch"] = int(un["k_epoch"])
    out["clusters_auto_tune"] = None
    for sk in list(out):
        if str(sk).startswith("clusters_auto_tune_"):
            out.pop(sk, None)
    return out


@hydra.main(version_base=None, config_path="configs", config_name="experiment_mics_unet")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    data = getattr(cfg, "data", None)
    pipe = getattr(cfg, "pipeline", None)
    bl = getattr(cfg, "baseline", None)
    if data is None:
        raise ValueError("Config must define data:")

    npy_path = Path(to_absolute_path(str(data.npy_path))).expanduser().resolve()
    if not npy_path.is_file():
        raise FileNotFoundError(f"data.npy_path not found: {npy_path}")

    msi = _load_msi(npy_path, bool(data.transpose_msi), bool(data.tic_normalize))
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    seed = int(OmegaConf.select(pipe, "seed", default=42))
    baseline_epochs = int(OmegaConf.select(bl, "num_epochs", default=30))
    mics_kw = _mics_kwargs_from_cfg(cfg, seed, baseline_epochs)
    unet_kw = _unet_kwargs_from_cfg(cfg, mics_kw)

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )

    share = bool(OmegaConf.select(pipe, "share_pixel_sampling", default=True))
    cache_raw = OmegaConf.select(pipe, "pixel_sampling_cache_path", default=None)
    cache_path: Path | None = None
    if cache_raw is not None and str(cache_raw).strip().lower() not in ("", "~", "null", "none"):
        cache_path = Path(to_absolute_path(str(cache_raw))).expanduser().resolve()

    pixel_cache = resolve_pixel_sampling_cache(
        msi,
        _sampling_cache_kwargs(mics_kw),
        out_dir=out_dir,
        share=share,
        cache_path=cache_path,
    )

    ckpt_raw = OmegaConf.select(pipe, "mics_baseline_checkpoint", default=None)
    ckpt_path: Path | None = None
    if ckpt_raw is not None and str(ckpt_raw).strip().lower() not in ("", "~", "null", "none"):
        ckpt_path = Path(to_absolute_path(str(ckpt_raw))).expanduser().resolve()

    mics: MSIParametricMiCSLMC | None = None
    unet: MSIParametricMiCSUnet | None = None

    try:
        logger.info("Building HD edge map (soft_landmark_contrast)...")
        hd_edges = _build_hd_edge_map_for_train(cfg, msi).astype(np.float32)
        np.save(out_dir / "hd_edges.npy", hd_edges)

        mics = MSIParametricMiCSLMC(**mics_kw)
        if ckpt_path is not None and ckpt_path.is_file():
            logger.info("Loading MiCS baseline checkpoint: %s", ckpt_path)
            blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            mics.set_image(msi, pixel_sampling_cache=pixel_cache)
            if mics.model is None:
                raise ValueError("Checkpoint load failed: MiCS model not initialized.")
            if "mics_state" in blob:
                mics.model.load_state_dict(blob["mics_state"])
        else:
            logger.info("Training MiCS PCC baseline (%d epochs)", baseline_epochs)
            mics.fit(msi, pixel_sampling_cache=pixel_cache)
            if mics.model is not None:
                torch.save(
                    {
                        "mics_state": mics.model.state_dict(),
                        "mics_base_emb": mics.predict_embedding(msi),
                    },
                    out_dir / "baseline_mics.pt",
                )

        mics_viz = mics.predict(msi)
        Image.fromarray(mics_viz, mode="RGB").save(out_dir / "baseline_mics_viz.png")
        dice_baseline = _global_continuous_dice(cfg, mics_viz, hd_edges, valid_mask)
        logger.info("Baseline MiCS PCC Dice: %.4f", dice_baseline)

        logger.info(
            "Training MiCS-U-Net (epochs=%s, dice_w=%s, contrast_w=%s, base_ch=%s)",
            unet_kw["num_epochs"],
            unet_kw["dice_loss_weight"],
            unet_kw["contrast_loss_weight"],
            unet_kw["unet_base_channels"],
        )
        unet = MSIParametricMiCSUnet(
            hd_edges=hd_edges,
            unet_base_channels=unet_kw.pop("unet_base_channels"),
            dice_loss_weight=unet_kw.pop("dice_loss_weight"),
            contrast_loss_weight=unet_kw.pop("contrast_loss_weight"),
            **unet_kw,
        )
        unet.fit(msi, pixel_sampling_cache=pixel_cache)

        unet_viz = unet.predict(msi)
        unet_emb = unet.predict_embedding(msi)
        Image.fromarray(unet_viz, mode="RGB").save(out_dir / "unet_viz.png")
        np.save(out_dir / "unet_embedding.npy", unet_emb)

        if unet.model is not None:
            torch.save(
                {
                    "model_state_dict": unet.model.state_dict(),
                    "unet_base_channels": int(
                        OmegaConf.select(cfg, "unet.unet_base_channels", default=32)
                    ),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                out_dir / "unet_model.pt",
            )

        dice_unet = _global_continuous_dice(cfg, unet_viz, hd_edges, valid_mask)
        logger.info("MiCS-U-Net Dice: %.4f", dice_unet)
        logger.info("Dice delta (unet - baseline): %+.4f", dice_unet - dice_baseline)

        if bool(OmegaConf.select(pipe, "save_comparison", default=True)):
            rgb_diff = _rgb_diff_heatmap(mics_viz, unet_viz, valid_mask)
            fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)
            axes[0].imshow(mics_viz)
            axes[0].set_title(f"MiCS PCC (baseline)\nDice {dice_baseline:.4f}")
            axes[0].axis("off")
            axes[1].imshow(unet_viz)
            axes[1].set_title(f"MiCS-U-Net\nDice {dice_unet:.4f}")
            axes[1].axis("off")
            im = axes[2].imshow(rgb_diff, cmap="inferno", vmin=0, vmax=255)
            axes[2].set_title("|ΔRGB| x8")
            axes[2].axis("off")
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            fig.tight_layout()
            cmp_path = out_dir / "comparison_mics_vs_unet.png"
            fig.savefig(cmp_path, bbox_inches="tight")
            plt.close(fig)
            Image.fromarray(rgb_diff, mode="L").save(out_dir / "comparison_rgb_diff.png")
            logger.info("Comparison: %s", cmp_path)

        if bool(OmegaConf.select(pipe, "save_edge_panel", default=True)):
            edge_path = _save_edge_comparison_panel(
                out_dir,
                mics_viz=mics_viz,
                refined_viz=unet_viz,
                hd_edges=hd_edges,
                valid_mask=valid_mask,
                cfg=cfg,
                dice_stage1=dice_baseline,
                dice_stage2=dice_unet,
            )
            logger.info("Edge comparison: %s", edge_path)

        summary = out_dir / "run_summary.txt"
        with summary.open("w", encoding="utf-8") as fh:
            fh.write(f"npy_path: {npy_path}\n")
            fh.write(f"output_dir: {out_dir}\n")
            fh.write(f"baseline_epochs: {baseline_epochs}\n")
            fh.write(f"unet_epochs: {unet_kw['num_epochs']}\n")
            fh.write(f"continuous_dice_baseline: {dice_baseline:.6f}\n")
            fh.write(f"continuous_dice_unet: {dice_unet:.6f}\n")
            fh.write(f"continuous_dice_delta: {dice_unet - dice_baseline:+.6f}\n")

        logger.info("Done. Outputs in %s", out_dir)
    finally:
        if unet is not None:
            try:
                unet.release_resources()
            except Exception:
                pass
        if mics is not None:
            try:
                mics.release_resources()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
