#!/usr/bin/env python3
"""
MiCS cluster-head gallery: soft softmax-weighted cluster maps.

Train one MiCS model (``cluster: true``), then one panel per cluster level
(e.g. k=1024, 2048, 4096) using **soft** predictions:

    RGB(x,y) = sum_k p_k(x,y) * palette(k)

Not argmax. Optional ``gallery.include_argmax_panels=true`` adds hard maps for comparison.
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
from matplotlib.patches import ConnectionPatch
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


def _parse_clusters(val: Any) -> list[int]:
    if val is None:
        return [1024, 2048, 4096]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip()]
    return [int(s)]


def _resolve_num_layers(cfg: DictConfig, ga: Any) -> int:
    raw = OmegaConf.select(ga, "num_layers", default=None)
    if raw is None or str(raw).strip().lower() in ("", "~", "null", "none"):
        return max(1, int(cfg.model.num_layers))
    return max(1, int(raw))


def _mics_kwargs_from_cfg(cfg: DictConfig, num_layers: int, seed: int) -> dict[str, Any]:
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    kw: dict[str, Any] = {}
    for name in _MICS_KEYS:
        if name in ("num_layers", "clusters"):
            continue
        if name in merged:
            kw[name] = merged[name]
    kw.setdefault("number_of_points", int(merged.get("number_of_points", 1000)))
    kw.setdefault("sampling", str(merged.get("sampling", "coreset")))
    kw.setdefault("num_epochs", int(merged.get("num_epochs", 30)))
    kw.setdefault("number_of_components", int(merged.get("number_of_components", 3)))
    kw.setdefault("lab_to_rgb", bool(merged.get("lab_to_rgb", True)))
    kw.setdefault("cluster", bool(merged.get("cluster", True)))
    kw.setdefault("clusters", _parse_clusters(merged.get("clusters", [1024, 2048, 4096])))
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
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw["num_layers"] = max(1, int(num_layers))
    kw["random_state"] = int(seed)
    kw["learn_input_mask"] = False
    kw["input_mask_l1_weight"] = 0.0
    if not kw.get("cluster"):
        raise ValueError("gallery_mics_cluster_soft requires model.cluster=true")
    return kw


def _save_mosaic(
    panel_rgbs: list[np.ndarray],
    titles: list[str],
    *,
    out_path: Path,
    ncols: int,
    fig_w: float,
    dpi: int,
    title_fs: int,
    suptitle: str | None = None,
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

    if suptitle:
        fig.suptitle(suptitle, fontsize=title_fs + 2, color="white", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)


def _save_hub_mosaic(
    cluster_rgbs: list[np.ndarray],
    cluster_titles: list[str],
    mics_rgb: np.ndarray,
    mics_title: str,
    *,
    out_path: Path,
    ncols: int,
    fig_w: float,
    dpi: int,
    title_fs: int,
    arrow_color: str = "white",
    arrow_lw: float = 1.8,
) -> None:
    """Cluster panels on top row(s); MiCS centered on bottom row; arrows → MiCS."""
    n_cluster = len(cluster_rgbs)
    if n_cluster == 0:
        raise ValueError("hub layout requires at least one cluster panel")
    ncols = max(1, int(ncols))
    cluster_rows = int(np.ceil(n_cluster / ncols))
    nrows = cluster_rows + 1
    center_col = ncols // 2
    mh = float(max(fig_w * 0.38, fig_w * 0.30 * nrows))
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

    cluster_axes: list[Any] = []
    for idx, (rgb, title) in enumerate(zip(cluster_rgbs, cluster_titles, strict=True)):
        r, c = divmod(idx, ncols)
        ax = ax_grid[r, c]
        ax.set_facecolor("black")
        ax.imshow(rgb)
        ax.set_title(title, fontsize=title_fs, color="white", pad=6)
        ax.axis("off")
        cluster_axes.append(ax)

    for idx in range(n_cluster, cluster_rows * ncols):
        r, c = divmod(idx, ncols)
        ax_grid[r, c].set_facecolor("black")
        ax_grid[r, c].axis("off")

    mics_row = cluster_rows
    for c in range(ncols):
        if c != center_col:
            ax_grid[mics_row, c].set_facecolor("black")
            ax_grid[mics_row, c].axis("off")
    mics_ax = ax_grid[mics_row, center_col]
    mics_ax.set_facecolor("black")
    mics_ax.imshow(mics_rgb)
    mics_ax.text(
        0.5,
        -0.06,
        mics_title,
        transform=mics_ax.transAxes,
        ha="center",
        va="top",
        fontsize=title_fs,
        color="white",
        clip_on=False,
    )
    mics_ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.42)
    fig.canvas.draw()
    for c_ax in cluster_axes:
        con = ConnectionPatch(
            xyA=(0.5, 0.04),
            xyB=(0.5, 0.96),
            coordsA="axes fraction",
            coordsB="axes fraction",
            axesA=c_ax,
            axesB=mics_ax,
            color=arrow_color,
            linewidth=arrow_lw,
            arrowstyle="-|>",
            mutation_scale=14,
            shrinkA=10,
            shrinkB=12,
            zorder=1,
        )
        fig.add_artist(con)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_cluster_soft")
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

    num_layers = _resolve_num_layers(cfg, ga)
    seed = int(OmegaConf.select(ga, "seed", default=42))
    kw = _mics_kwargs_from_cfg(cfg, num_layers, seed)
    color_scheme = str(OmegaConf.select(ga, "color_scheme", default="gist_rainbow"))
    include_argmax = bool(OmegaConf.select(ga, "include_argmax_panels", default=False))
    include_mics_viz = bool(OmegaConf.select(ga, "include_mics_viz_panel", default=True))
    layout = str(OmegaConf.select(ga, "layout", default="hub")).strip().lower()
    arrow_color = str(OmegaConf.select(ga, "arrow_color", default="white"))
    save_probs = bool(OmegaConf.select(ga, "save_soft_probs", default=False))
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
    pixel_cache = resolve_pixel_sampling_cache(
        msi,
        kw,
        out_dir=out_dir,
        share=share_sampling,
        cache_path=cache_path,
    )

    mics_label = "mics2-equivalent" if num_layers >= 7 else "MiCS"
    logger.info(
        "Training %s (layers=%d, clusters=%s) on %s",
        mics_label,
        num_layers,
        kw.get("clusters"),
        npy_path.name,
    )

    model: MSIParametricMiCSLMC | None = None
    try:
        model = MSIParametricMiCSLMC(**kw)
        model.fit(msi, pixel_sampling_cache=pixel_cache)

        heads = getattr(model, "visualiation_to_cluster", None) or []
        if not heads:
            raise ValueError("No cluster heads after fit; check model.cluster and model.clusters.")

        levels = list(getattr(model, "clusters", []) or [])
        soft_rgbs: list[np.ndarray] = []
        soft_titles: list[str] = []
        argmax_rgbs: list[np.ndarray] = []
        argmax_titles: list[str] = []

        for idx in range(len(heads)):
            k_cfg = int(levels[idx]) if idx < len(levels) else idx
            n_cls = int(heads[idx][-1].out_features)

            if save_probs:
                probs = model.predict_cluster_soft_probs(msi, level_idx=idx)
                prob_path = pan_dir / f"panel__cluster_k{k_cfg}__soft_probs.npy"
                np.save(prob_path, probs)
                logger.info("Saved soft probs: %s", _abs_path_str(prob_path))

            rgb_soft = model.predict_cluster_soft_rgb_u8(
                msi, level_idx=idx, color_scheme=color_scheme
            )
            soft_path = pan_dir / f"panel__cluster_k{k_cfg}__soft.png"
            Image.fromarray(rgb_soft, mode="RGB").save(soft_path)
            logger.info(
                "Soft cluster panel k=%s (%d classes): %s",
                k_cfg,
                n_cls,
                _abs_path_str(soft_path),
            )
            soft_rgbs.append(rgb_soft)
            soft_titles.append(f"k={k_cfg}")

            if include_argmax:
                rgb_hard = model.predict_cluster_argmax_rgb_u8(
                    msi, level_idx=idx, color_scheme=color_scheme
                )
                hard_path = pan_dir / f"panel__cluster_k{k_cfg}__argmax.png"
                Image.fromarray(rgb_hard, mode="RGB").save(hard_path)
                logger.info("Argmax cluster panel k=%s: %s", k_cfg, _abs_path_str(hard_path))
                argmax_rgbs.append(rgb_hard)
                argmax_titles.append(f"k={k_cfg}")

        mics_rgb: np.ndarray | None = None
        if include_mics_viz:
            mics_rgb = np.asarray(model.predict(msi), dtype=np.uint8)[..., :3].copy()
            mics_rgb[~valid_mask] = 0
            mics_path = pan_dir / "panel__mics_viz.png"
            Image.fromarray(mics_rgb, mode="RGB").save(mics_path)
            logger.info("MiCS visualization panel: %s", _abs_path_str(mics_path))

        mosaic_soft = out_dir / "gallery_mics_cluster_soft.png"
        if include_mics_viz and layout in ("hub", "mics_center", "center"):
            _save_hub_mosaic(
                soft_rgbs,
                soft_titles,
                mics_rgb,
                "MiCS",
                out_path=mosaic_soft,
                ncols=ncols,
                fig_w=fig_w,
                dpi=dpi,
                title_fs=title_fs,
                arrow_color=arrow_color,
            )
        else:
            if include_mics_viz and mics_rgb is not None:
                soft_rgbs.append(mics_rgb)
                soft_titles.append("MiCS")
            _save_mosaic(
                soft_rgbs,
                soft_titles,
                out_path=mosaic_soft,
                ncols=ncols,
                fig_w=fig_w,
                dpi=dpi,
                title_fs=title_fs,
            )
        logger.info("Gallery mosaic: %s", _abs_path_str(mosaic_soft))

        mosaic_argmax: Path | None = None
        if include_argmax and argmax_rgbs:
            mosaic_argmax = out_dir / "gallery_mics_cluster_argmax.png"
            if include_mics_viz and mics_rgb is not None and layout in ("hub", "mics_center", "center"):
                _save_hub_mosaic(
                    argmax_rgbs,
                    argmax_titles,
                    mics_rgb,
                    "MiCS",
                    out_path=mosaic_argmax,
                    ncols=ncols,
                    fig_w=fig_w,
                    dpi=dpi,
                    title_fs=title_fs,
                    arrow_color=arrow_color,
                )
            else:
                if include_mics_viz and mics_rgb is not None:
                    argmax_rgbs.append(mics_rgb)
                    argmax_titles.append("MiCS")
                _save_mosaic(
                    argmax_rgbs,
                    argmax_titles,
                    out_path=mosaic_argmax,
                    ncols=ncols,
                    fig_w=fig_w,
                    dpi=dpi,
                    title_fs=title_fs,
                )
            logger.info("Argmax mosaic: %s", _abs_path_str(mosaic_argmax))

        summary = out_dir / "cluster_soft_summary.txt"
        with summary.open("w", encoding="utf-8") as fh:
            fh.write(f"npy_path: {npy_path}\n")
            fh.write(f"num_layers: {num_layers}\n")
            fh.write(f"clusters: {levels}\n")
            fh.write(f"color_scheme: {color_scheme}\n")
            fh.write(f"layout: {layout}\n")
            fh.write(f"include_argmax_panels: {include_argmax}\n")
            fh.write(f"include_mics_viz_panel: {include_mics_viz}\n")
            fh.write(f"save_soft_probs: {save_probs}\n")
            fh.write(f"\nmosaic_soft: {_abs_path_str(mosaic_soft)}\n")
            if mosaic_argmax is not None:
                fh.write(f"mosaic_argmax: {_abs_path_str(mosaic_argmax)}\n")
        logger.info("Summary: %s", _abs_path_str(summary))

    finally:
        if model is not None:
            try:
                model.release_resources()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
