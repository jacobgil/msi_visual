#!/usr/bin/env python3
"""
Edge-aware spatial smoothing of an MSI visualization, guided by SoLaCE edges.

Combines two things this repo already produces:
  * a **SoLaCE** (``soft_landmark_contrast``) edge-strength map, computed from the full
    high-dimensional spectra, and
  * a **visualization** (a MiCS embedding by default, or any RGB image you provide),

into a single **joint-bilateral** smoother (see ``solace_guided_bilateral_smooth``). Unlike an
isotropic Gaussian blur, mixing between two pixels is attenuated whenever a strong SoLaCE
boundary lies between them, so noise inside a tissue region is averaged out while the region
boundaries stay crisp. Running a small neighborhood over several iterations behaves like
anisotropic (Perona-Malik) diffusion, but driven by SoLaCE instead of a per-channel gradient.

Outputs (under the Hydra run dir):
  * ``baseline.png``        - the raw visualization (no smoothing)
  * ``gaussian.png``        - isotropic Gaussian smoothing (comparison)
  * ``solace_edges.png``    - the SoLaCE edge map (colormap)
  * ``edge_aware.png``      - SoLaCE-guided edge-aware smoothing
  * ``panel.png``           - side-by-side comparison
  * ``manifest.json``       - resolved parameters and file paths

Run from repo root:
  python scripts/solace_edge_aware_smoothing.py
  python scripts/solace_edge_aware_smoothing.py data.npy_path=D:/data/0.npy
  # Smooth an existing RGB visualization instead of training MiCS:
  python scripts/solace_edge_aware_smoothing.py visualization.source=image \
      visualization.image_path=D:/some_visualization.png

Config: ``scripts/configs/solace_edge_aware_smoothing.yaml``
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from msi_visual.parametric_mics_lmc import (
    MSIParametricMiCSLMC,
    _spatial_gaussian_smooth_channels,
    solace_guided_bilateral_smooth,
)

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    import random

    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI shape H*W*C (or C*H*W with transpose), got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI contains negative values; this script expects non-negative inputs.")
    if tic_normalize:
        sums = img.sum(axis=-1, keepdims=True) + 1e-8
        img = img / sums
    return img


def _to_uint8_rgb(viz: np.ndarray) -> np.ndarray:
    if isinstance(viz, list):
        viz = viz[0]
    arr = np.asarray(viz)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr
    out = arr.astype(np.float32)
    mx = float(np.nanmax(out))
    if mx <= 1.0 + 1e-6:
        out = (out / mx) * 255.0 if mx > 1e-12 else np.zeros_like(out)
    return np.clip(out, 0, 255).astype(np.uint8)


def _compute_solace_guide(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rank_config_path: str,
    hd_method: str = "soft_landmark_contrast",
) -> np.ndarray:
    """SoLaCE edge-strength map as a float H*W guide (higher = stronger boundary)."""
    from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method

    rank_path = Path(to_absolute_path(str(rank_config_path)))
    if not rank_path.is_file():
        raise FileNotFoundError(f"solace.rank_config not found: {rank_path}")
    rank_cfg = OmegaConf.load(rank_path)

    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    if multi is None or not bool(getattr(multi, "enabled", False)):
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    spatial_enabled = bool(getattr(rank_cfg.spatial_normalization, "enabled", True))

    mask = np.asarray(valid_mask, dtype=bool)
    hd_edge_n = _compute_hd_edge_n_for_method(
        rank_cfg,
        msi,
        mask,
        getattr(rank_cfg, "hd_edges", None),
        str(hd_method),
        sigmas,
        aggregation,
        spatial_enabled,
    )
    guide = np.asarray(hd_edge_n, dtype=np.float32)
    guide[~mask] = 0.0
    return guide


def _solace_edges_rgb(guide: np.ndarray, valid_mask: np.ndarray, colormap: str) -> np.ndarray:
    from debug_edge_maps import _gray_u8_to_rgb_colormap, _to_uint8_gray01

    mask = np.asarray(valid_mask, dtype=bool)
    gray = _to_uint8_gray01(guide)
    cmap = str(colormap).strip()
    if cmap.lower() in ("", "null", "none", "~"):
        rgb = np.stack([gray] * 3, axis=-1)
    else:
        rgb = _gray_u8_to_rgb_colormap(gray, cmap)
    rgb[~mask] = 0
    return rgb


def _mics_kwargs_from_cfg(cfg: DictConfig, seed: int) -> dict[str, Any]:
    m = cfg.mics
    clusters = OmegaConf.to_container(m.clusters, resolve=True)
    if not isinstance(clusters, (list, tuple)):
        clusters = [int(clusters)]
    return {
        "clusters": [int(x) for x in clusters],
        "cluster": bool(OmegaConf.select(m, "cluster", default=True)),
        "cluster_on_pca": bool(OmegaConf.select(m, "cluster_on_pca", default=False)),
        "beta": float(OmegaConf.select(m, "beta", default=20.0)),
        "num_layers": int(OmegaConf.select(m, "num_layers", default=3)),
        "num_epochs": int(OmegaConf.select(m, "num_epochs", default=30)),
        "number_of_points": int(OmegaConf.select(m, "number_of_points", default=1000)),
        "sampling": str(OmegaConf.select(m, "sampling", default="coreset")),
        "pixel_sampling": str(OmegaConf.select(m, "pixel_sampling", default="superpixel")),
        "num_samples": int(OmegaConf.select(m, "num_samples", default=5000)),
        "k_epoch": int(OmegaConf.select(m, "k_epoch", default=20)),
        "temperature": float(OmegaConf.select(m, "temperature", default=100.0)),
        "lr": float(OmegaConf.select(m, "lr", default=1.0)),
        "batch_size": int(OmegaConf.select(m, "batch_size", default=1024)),
        "warmup_epochs": int(OmegaConf.select(m, "warmup_epochs", default=10)),
        "factor": float(OmegaConf.select(m, "factor", default=1.0)),
        "lab_to_rgb": bool(OmegaConf.select(m, "lab_to_rgb", default=True)),
        "clusters_auto_tune": None,
        "random_state": int(OmegaConf.select(m, "random_state", default=seed)),
        "verbose": bool(OmegaConf.select(m, "verbose", default=False)),
    }


def _load_visualization_image(path: str, shape_hw: tuple[int, int]) -> np.ndarray:
    p = Path(to_absolute_path(str(path)))
    if not p.is_file():
        raise FileNotFoundError(f"visualization.image_path not found: {p}")
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        arr = np.asarray(Image.open(p).convert("RGB"))
    arr = np.asarray(arr)
    if arr.shape[:2] != tuple(shape_hw):
        raise ValueError(
            f"Visualization image shape {arr.shape[:2]} does not match MSI grid {shape_hw}."
        )
    return arr


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def _save_panel(items: list[tuple[str, np.ndarray]], path: Path, dpi: int = 200) -> None:
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), dpi=dpi)
    fig.patch.set_facecolor("black")
    if n == 1:
        axes = [axes]
    for ax, (title, rgb) in zip(axes, items):
        ax.imshow(np.asarray(rgb, dtype=np.uint8))
        ax.set_title(title, color="white", fontsize=14)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="solace_edge_aware_smoothing")
def main(cfg: DictConfig) -> None:
    seed = int(OmegaConf.select(cfg, "seed", default=17))
    _seed_everything(seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    npy_path = Path(to_absolute_path(str(cfg.data.npy_path)))
    if not npy_path.is_file():
        raise FileNotFoundError(f"data.npy_path not found: {npy_path}")
    normalization = str(OmegaConf.select(cfg, "data.normalization", default="tic")).lower()
    msi = _load_msi(
        npy_path,
        bool(OmegaConf.select(cfg, "data.transpose_msi", default=False)),
        tic_normalize=(normalization == "tic"),
    )
    valid_mask = np.asarray(msi.sum(axis=-1) > 0, dtype=bool)
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty.")
    logger.info("Loaded MSI %s (H*W*C = %s), tissue pixels = %d", npy_path.name, msi.shape, int(valid_mask.sum()))

    logger.info("Computing SoLaCE (%s) edge guide ...", str(OmegaConf.select(cfg, "solace.hd_method", default="soft_landmark_contrast")))
    guide = _compute_solace_guide(
        msi,
        valid_mask,
        rank_config_path=str(cfg.solace.rank_config),
        hd_method=str(OmegaConf.select(cfg, "solace.hd_method", default="soft_landmark_contrast")),
    )
    edges_rgb = _solace_edges_rgb(guide, valid_mask, str(OmegaConf.select(cfg, "solace.edge_colormap", default="PuBu")))

    sm = cfg.smoothing
    spatial_sigma = float(OmegaConf.select(sm, "spatial_sigma", default=1.5))
    edge_scale = float(OmegaConf.select(sm, "edge_scale", default=0.15))
    iterations = int(OmegaConf.select(sm, "iterations", default=4))
    _radius = OmegaConf.select(sm, "radius", default=None)
    radius = int(_radius) if _radius is not None else None
    range_sigma = float(OmegaConf.select(sm, "range_sigma", default=0.0))
    sharpen_amount = float(OmegaConf.select(sm, "sharpen_amount", default=0.0))
    sharpen_sigma = float(OmegaConf.select(sm, "sharpen_sigma", default=1.0))
    gaussian_sigma = float(OmegaConf.select(sm, "gaussian_sigma", default=spatial_sigma))

    source = str(OmegaConf.select(cfg, "visualization.source", default="mics")).strip().lower()

    if source == "mics":
        mk = _mics_kwargs_from_cfg(cfg, seed)
        logger.info("Training MiCS: clusters=%s beta=%s layers=%s epochs=%s", mk["clusters"], mk["beta"], mk["num_layers"], mk["num_epochs"])
        model = MSIParametricMiCSLMC(**mk)
        model.fit(msi)

        # Baseline (no smoothing) and isotropic Gaussian, both via the same trained model.
        model.predict_edge_smooth = False
        model.predict_edge_map = None
        model.predict_spatial_smooth_sigma = 0.0
        baseline_rgb = _to_uint8_rgb(model.predict(msi))

        model.predict_spatial_smooth_sigma = gaussian_sigma
        gaussian_rgb = _to_uint8_rgb(model.predict(msi))

        # SoLaCE-guided edge-aware smoothing (on the embedding, before RGB scaling).
        model.predict_spatial_smooth_sigma = 0.0
        model.predict_edge_smooth = True
        model.predict_edge_map = guide
        model.predict_edge_spatial_sigma = spatial_sigma
        model.predict_edge_scale = edge_scale
        model.predict_edge_iterations = iterations
        model.predict_edge_radius = radius
        model.predict_edge_range_sigma = range_sigma
        model.predict_edge_sharpen_amount = sharpen_amount
        model.predict_edge_sharpen_sigma = sharpen_sigma
        edge_aware_rgb = _to_uint8_rgb(model.predict(msi))
    elif source == "image":
        image_path = OmegaConf.select(cfg, "visualization.image_path", default=None)
        if image_path is None:
            raise ValueError("visualization.source=image requires visualization.image_path")
        baseline_rgb = _to_uint8_rgb(_load_visualization_image(str(image_path), msi.shape[:2]))
        base_f = baseline_rgb.astype(np.float32)
        gaussian_rgb = _to_uint8_rgb(_spatial_gaussian_smooth_channels(base_f, gaussian_sigma))
        smoothed = solace_guided_bilateral_smooth(
            base_f,
            guide,
            mask=valid_mask,
            spatial_sigma=spatial_sigma,
            edge_scale=edge_scale,
            iterations=iterations,
            radius=radius,
            range_sigma=range_sigma,
            sharpen_amount=sharpen_amount,
            sharpen_sigma=sharpen_sigma,
        )
        edge_aware_rgb = _to_uint8_rgb(smoothed)
    else:
        raise ValueError(f"Unknown visualization.source={source!r} (expected 'mics' or 'image')")

    for rgb in (baseline_rgb, gaussian_rgb, edge_aware_rgb):
        rgb[~valid_mask] = 0

    save_individual = bool(OmegaConf.select(cfg, "output.save_individual", default=True))
    paths: dict[str, str] = {}
    if save_individual:
        _save_rgb(baseline_rgb, run_dir / "baseline.png")
        _save_rgb(gaussian_rgb, run_dir / "gaussian.png")
        _save_rgb(edges_rgb, run_dir / "solace_edges.png")
        _save_rgb(edge_aware_rgb, run_dir / "edge_aware.png")
        paths.update(
            baseline=str(run_dir / "baseline.png"),
            gaussian=str(run_dir / "gaussian.png"),
            solace_edges=str(run_dir / "solace_edges.png"),
            edge_aware=str(run_dir / "edge_aware.png"),
        )

    if bool(OmegaConf.select(cfg, "output.panel", default=True)):
        panel_items = [
            ("Baseline", baseline_rgb),
            (f"Gaussian (sigma={gaussian_sigma:g})", gaussian_rgb),
            ("SoLaCE edges", edges_rgb),
            ("SoLaCE edge-aware", edge_aware_rgb),
        ]
        _save_panel(panel_items, run_dir / "panel.png", dpi=int(OmegaConf.select(cfg, "output.dpi", default=200)))
        paths["panel"] = str(run_dir / "panel.png")
        logger.info("Panel: %s", paths["panel"])

    manifest = {
        "npy_path": str(npy_path),
        "visualization_source": source,
        "smoothing": {
            "spatial_sigma": spatial_sigma,
            "edge_scale": edge_scale,
            "iterations": iterations,
            "radius": radius,
            "range_sigma": range_sigma,
            "sharpen_amount": sharpen_amount,
            "sharpen_sigma": sharpen_sigma,
            "gaussian_sigma": gaussian_sigma,
        },
        "solace": {
            "rank_config": str(cfg.solace.rank_config),
            "hd_method": str(OmegaConf.select(cfg, "solace.hd_method", default="soft_landmark_contrast")),
        },
        "outputs": paths,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
