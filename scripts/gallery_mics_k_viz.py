#!/usr/bin/env python3
"""
MiCS visualization gallery by cluster scale.

Train separate MiCS models for each entry in ``gallery.k_ladder`` (default K=8, 32, 128),
then mosaic **only** the main MiCS RGB visualization (``model.predict``) per model.

``gallery.cluster_ladder_mode``:
- ``cumulative``: prefixes [8], [8,32], [8,32,128]
- ``single``: independent [8], [32], [128]
"""

from __future__ import annotations

import csv
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

from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402

logger = logging.getLogger(__name__)

_MICS_KEYS: tuple[str, ...] = (
    "number_of_points",
    "sampling",
    "num_epochs",
    "number_of_components",
    "lab_to_rgb",
    "cluster",
    "clusters",
    "num_samples",
    "beta",
    "k_epoch",
    "temperature",
    "lr",
    "batch_size",
    "warmup_epochs",
    "factor",
    "random_state",
    "verbose",
    "cluster_loss_weight",
    "category_loss_weight",
    "pixel_sampling",
    "pca_fit_step",
    "cluster_on_pca",
    "cluster_pca_dims",
    "predict_percentile_low",
    "predict_percentile_high",
    "predict_spatial_smooth_sigma",
    "clusters_auto_tune",
)


def _abs_path_str(path: Path) -> str:
    return str(path.expanduser().resolve())


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    s = msi.sum(axis=-1, keepdims=True)
    return msi / (s + 1e-8)


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(str(path), mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI HxWxC, got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI must be nonnegative.")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def _mics_kwargs_from_cfg(cfg: DictConfig, clusters: list[int], seed: int) -> dict[str, Any]:
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    kw: dict[str, Any] = {}
    for name in _MICS_KEYS:
        if name == "clusters":
            continue
        if name in merged:
            kw[name] = merged[name]
    kw.setdefault("number_of_points", int(merged.get("number_of_points", 1000)))
    kw.setdefault("sampling", str(merged.get("sampling", "coreset")))
    kw.setdefault("num_epochs", int(merged.get("num_epochs", 30)))
    kw.setdefault("number_of_components", int(merged.get("number_of_components", 3)))
    kw.setdefault("lab_to_rgb", bool(merged.get("lab_to_rgb", True)))
    kw.setdefault("cluster", True)
    kw.setdefault("num_samples", int(merged.get("num_samples", 5000)))
    kw.setdefault("beta", float(merged.get("beta", 20.0)))
    kw.setdefault("k_epoch", int(merged.get("k_epoch", 20)))
    kw.setdefault("temperature", float(merged.get("temperature", 100.0)))
    kw.setdefault("lr", float(merged.get("lr", 1.0)))
    kw.setdefault("batch_size", int(merged.get("batch_size", 1024)))
    kw.setdefault("warmup_epochs", int(merged.get("warmup_epochs", 10)))
    kw.setdefault("factor", float(merged.get("factor", 1.0)))
    kw.setdefault("verbose", bool(merged.get("verbose", False)))
    kw.setdefault("pixel_sampling", str(merged.get("pixel_sampling", "superpixel")))
    kw.setdefault(
        "predict_spatial_smooth_sigma", float(merged.get("predict_spatial_smooth_sigma", 0.8))
    )
    kw.setdefault("num_layers", int(merged.get("num_layers", 2)))
    kw["clusters"] = [int(k) for k in clusters]
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw["random_state"] = int(seed)
    kw["learn_input_mask"] = False
    kw["input_mask_l1_weight"] = 0.0
    return kw


def _parse_k_ladder(ga: Any) -> list[int]:
    raw = OmegaConf.select(ga, "k_ladder", default=[8, 32, 128])
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("gallery.k_ladder must be a non-empty list of integers")
    out: list[int] = []
    for x in raw:
        k = int(x)
        if k < 2:
            raise ValueError(f"k_ladder entries must be >= 2 (got {k})")
        out.append(k)
    return out


def _cluster_specs(k_ladder: list[int], mode: str) -> list[tuple[int, list[int]]]:
    """Return (panel_k_label, clusters_list) for each model."""
    mode = str(mode).strip().lower()
    specs: list[tuple[int, list[int]]] = []
    if mode == "single":
        for k in k_ladder:
            specs.append((k, [k]))
        return specs
    if mode == "cumulative":
        acc: list[int] = []
        for k in k_ladder:
            acc.append(k)
            specs.append((k, list(acc)))
        return specs
    raise ValueError(f"gallery.cluster_ladder_mode must be cumulative | single, got {mode!r}")


def _mics_rgb(model: MSIParametricMiCSLMC, msi: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgb = np.asarray(model.predict(msi), dtype=np.uint8)[..., :3].copy()
    rgb[~valid_mask] = 0
    return rgb


def _save_mosaic(
    panel_rgbs: list[np.ndarray],
    titles: list[str],
    *,
    out_path: Path,
    ncols: int,
    fig_w: float,
    dpi: int,
    title_fs: int,
) -> None:
    n_p = len(panel_rgbs)
    nrows = int(np.ceil(n_p / ncols))
    mh = float(max(fig_w * 0.35, fig_w * 0.28 * max(1, nrows)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, mh))
    fig.patch.set_facecolor("black")
    if nrows == 1 and ncols == 1:
        ax_grid = np.array([[axes]])
    elif nrows == 1:
        ax_grid = np.asarray(axes).reshape(1, -1)
    elif ncols == 1:
        ax_grid = np.asarray(axes).reshape(-1, 1)
    else:
        ax_grid = np.asarray(axes)

    for idx, rgb in enumerate(panel_rgbs):
        r, c = divmod(idx, ncols)
        ax = ax_grid[r, c]
        ax.set_facecolor("black")
        ax.imshow(rgb)
        ax.set_title(titles[idx], fontsize=title_fs, color="white", pad=6)
        ax.axis("off")
    for idx in range(n_p, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax_grid[r, c].set_facecolor("black")
        ax_grid[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_k_viz")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ga = getattr(cfg, "gallery", None)
    if ga is None:
        raise ValueError("Config must define gallery:")

    npy_path = Path(to_absolute_path(str(ga.npy_path))).expanduser().resolve()
    if not npy_path.is_file():
        raise FileNotFoundError(f"gallery.npy_path not found: {npy_path}")

    msi = _load_msi(npy_path, bool(ga.transpose_msi), bool(ga.tic_normalize))
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    k_ladder = _parse_k_ladder(ga)
    mode = str(OmegaConf.select(ga, "cluster_ladder_mode", default="cumulative"))
    specs = _cluster_specs(k_ladder, mode)
    seed = int(OmegaConf.select(ga, "seed", default=42))
    ncols = max(1, int(OmegaConf.select(ga, "ncols", default=3)))
    fig_w = float(OmegaConf.select(ga, "figure_size_inches", default=14))
    dpi = int(OmegaConf.select(ga, "dpi", default=300))
    title_fs = int(OmegaConf.select(ga, "panel_title_fontsize", default=10))

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    pan_dir = out_dir / "panels"
    pan_dir.mkdir(parents=True, exist_ok=True)

    share_sampling = bool(OmegaConf.select(ga, "share_pixel_sampling", default=True))
    cache_path_raw = OmegaConf.select(ga, "pixel_sampling_cache_path", default=None)
    cache_path: Path | None = None
    if cache_path_raw is not None and str(cache_path_raw).strip().lower() not in (
        "",
        "~",
        "null",
        "none",
    ):
        cache_path = Path(to_absolute_path(str(cache_path_raw))).expanduser().resolve()

    # Pixel sampling cache depends on model kwargs except clusters; use first spec for cache build.
    cache_kw = _mics_kwargs_from_cfg(cfg, specs[0][1], seed)
    pixel_cache = resolve_pixel_sampling_cache(
        msi,
        cache_kw,
        out_dir=out_dir,
        share=share_sampling,
        cache_path=cache_path,
    )

    panel_rgbs: list[np.ndarray] = []
    panel_titles: list[str] = []
    rows: list[dict[str, Any]] = []

    for idx, (k_label, clusters) in enumerate(specs):
        lbl = "-".join(str(k) for k in clusters)
        kw = _mics_kwargs_from_cfg(cfg, clusters, seed + idx)
        logger.info(
            "Training MiCS panel %d/%d | k=%d | clusters=%s",
            idx + 1,
            len(specs),
            k_label,
            clusters,
        )
        model: MSIParametricMiCSLMC | None = None
        try:
            model = MSIParametricMiCSLMC(**kw)
            model.fit(msi, pixel_sampling_cache=pixel_cache)
            rgb = _mics_rgb(model, msi, valid_mask)
            viz_path = pan_dir / f"panel__k{k_label}__mics_viz.png"
            Image.fromarray(rgb, mode="RGB").save(viz_path)
            panel_rgbs.append(rgb)
            panel_titles.append(f"k={k_label}")
            rows.append(
                {
                    "panel": f"k={k_label}",
                    "clusters": lbl,
                    "viz_path": _abs_path_str(viz_path),
                }
            )
            logger.info("MiCS viz k=%d: %s", k_label, _abs_path_str(viz_path))
        finally:
            if model is not None:
                try:
                    model.release_resources()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mosaic = out_dir / "gallery_mics_k_viz.png"
    _save_mosaic(
        panel_rgbs,
        panel_titles,
        out_path=mosaic,
        ncols=ncols,
        fig_w=fig_w,
        dpi=dpi,
        title_fs=title_fs,
    )

    csv_path = out_dir / "mics_k_viz_panels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["panel", "clusters", "viz_path"])
        writer.writeheader()
        writer.writerows(rows)

    summary = out_dir / "mics_k_viz_summary.txt"
    with summary.open("w", encoding="utf-8") as fh:
        fh.write(f"npy_path: {npy_path}\n")
        fh.write(f"k_ladder: {k_ladder}\n")
        fh.write(f"cluster_ladder_mode: {mode}\n")
        fh.write(f"num_panels: {len(specs)}\n\n")
        for row in rows:
            fh.write(f"{row['panel']}: clusters={row['clusters']}\n")
            fh.write(f"  viz: {row['viz_path']}\n")
        fh.write(f"\nmosaic: {_abs_path_str(mosaic)}\n")

    logger.info("Gallery mosaic: %s", _abs_path_str(mosaic))
    logger.info("Summary: %s", _abs_path_str(summary))


if __name__ == "__main__":
    main()
