#!/usr/bin/env python3
"""
Train MiCS with an edge-preservation objective on the RGB visualization.

Loss = standard MiCS (correlation + cluster) +
       edge_head(embedding) vs edge target on **sampled pixels** (default, fast).

Gallery: MiCS viz | edge target | predicted edge magnitude
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

from msi_visual.parametric_mics_edge import (  # noqa: E402
    MSIParametricMiCSEdge,
    _compute_edge_target_from_msi,
)
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402

logger = logging.getLogger(__name__)

_MICS_KEYS: tuple[str, ...] = (
    "number_of_points",
    "sampling",
    "num_epochs",
    "number_of_components",
    "num_layers",
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
    "edge_loss_weight",
    "edge_mse_weight",
    "edge_loss_mode",
    "edge_binary_threshold",
    "edge_label_smoothing",
    "edge_sample_weight_mode",
    "edge_sample_weight_alpha",
    "edge_sample_weight_floor",
    "edge_loss_every_n_steps",
    "edge_spatial_stride",
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
        return []
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if not s or s.lower() in ("", "null", "none", "~"):
        return []
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip()]
    return [int(s)]


def _mics_kwargs_from_cfg(cfg: DictConfig, seed: int) -> dict[str, Any]:
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    kw: dict[str, Any] = {}
    for name in _MICS_KEYS:
        if name in merged:
            kw[name] = merged[name]
    kw.setdefault("number_of_points", int(merged.get("number_of_points", 1000)))
    kw.setdefault("sampling", str(merged.get("sampling", "coreset")))
    kw.setdefault("num_epochs", int(merged.get("num_epochs", 30)))
    kw.setdefault("number_of_components", int(merged.get("number_of_components", 3)))
    kw.setdefault("num_layers", int(merged.get("num_layers", 2)))
    kw.setdefault("lab_to_rgb", bool(merged.get("lab_to_rgb", True)))
    kw.setdefault("cluster", bool(merged.get("cluster", True)))
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
        "predict_spatial_smooth_sigma", float(merged.get("predict_spatial_smooth_sigma", 0.0))
    )
    kw.setdefault("edge_loss_weight", float(merged.get("edge_loss_weight", 1.0)))
    kw.setdefault("edge_mse_weight", float(merged.get("edge_mse_weight", 0.0)))
    kw.setdefault("edge_loss_mode", str(merged.get("edge_loss_mode", "sampled")))
    thr = merged.get("edge_binary_threshold", None)
    if thr is None or str(thr).strip().lower() in ("", "~", "null", "none"):
        kw["edge_binary_threshold"] = None
    else:
        kw["edge_binary_threshold"] = float(thr)
    kw.setdefault("edge_label_smoothing", float(merged.get("edge_label_smoothing", 0.0)))
    kw.setdefault(
        "edge_sample_weight_mode", str(merged.get("edge_sample_weight_mode", "none"))
    )
    kw.setdefault(
        "edge_sample_weight_alpha", float(merged.get("edge_sample_weight_alpha", 1.0))
    )
    kw.setdefault(
        "edge_sample_weight_floor", float(merged.get("edge_sample_weight_floor", 0.25))
    )
    kw.setdefault("edge_loss_every_n_steps", int(merged.get("edge_loss_every_n_steps", 5)))
    kw.setdefault("edge_spatial_stride", int(merged.get("edge_spatial_stride", 4)))
    if kw.get("cluster"):
        kw["clusters"] = _parse_clusters(kw.get("clusters", merged.get("clusters", [8, 32, 128])))
    else:
        kw["clusters"] = []
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw["random_state"] = int(seed)
    kw["learn_input_mask"] = False
    kw["input_mask_l1_weight"] = 0.0
    return kw


def _resolve_edge_target(
    msi: np.ndarray, cfg: DictConfig, ga: Any
) -> tuple[np.ndarray, str]:
    mode = str(OmegaConf.select(ga, "edge_target", default="msi_simple")).strip().lower()
    hp = OmegaConf.select(cfg.model, "hd_edge_map_path", default=None)
    if hp is not None and str(hp).strip().lower() not in ("", "~", "null", "none"):
        pth = Path(to_absolute_path(str(hp))).expanduser().resolve()
        if not pth.is_file():
            raise FileNotFoundError(f"hd_edge_map_path not found: {pth}")
        edge = np.squeeze(np.load(str(pth))).astype(np.float32)
        if edge.shape != msi.shape[:2]:
            raise ValueError(f"hd_edge_map shape {edge.shape} != MSI {msi.shape[:2]}")
        return np.clip(edge, 0.0, 1.0), f"file:{pth.name}"

    if mode in ("rank_config", "rank", "hd"):
        from train_parametric_mics_lmc import _build_hd_edge_map_for_train

        edge = _build_hd_edge_map_for_train(cfg, msi).astype(np.float32)
        m = msi.sum(axis=-1) > 0
        mx = float(edge[m].max()) if np.any(m) else 0.0
        if mx > 1e-8:
            edge = edge / mx
        rank_sel = OmegaConf.select(cfg, "rank_config", default=None)
        hd_method = "soft_landmark_contrast"
        if rank_sel is not None and str(rank_sel).strip():
            from hydra.utils import to_absolute_path

            rank_cfg = OmegaConf.load(to_absolute_path(str(rank_sel)))
            hd_method = str(
                OmegaConf.select(rank_cfg, "edge_detection.hd_method", default=hd_method)
            )
        return np.clip(edge, 0.0, 1.0), hd_method

    return _compute_edge_target_from_msi(msi), "msi_simple"


def _edge_to_heatmap(edge: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    import cv2

    m = np.asarray(valid_mask, dtype=bool)
    u8 = np.zeros(edge.shape, dtype=np.uint8)
    v = np.asarray(edge, dtype=np.float32).copy()
    v[~m] = 0.0
    vals = v[m]
    if vals.size:
        p99 = float(np.percentile(vals, 99.0))
        u8[m] = np.clip(v[m] / max(p99, 1e-8) * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb[~m] = 0
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


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_edge")
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

    edge_target, edge_label = _resolve_edge_target(msi, cfg, ga)
    np.save(out_dir / "edge_target.npy", edge_target.astype(np.float32))

    kw = _mics_kwargs_from_cfg(cfg, seed)
    kw["hd_edges"] = edge_target

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

    logger.info(
        "Training MSIParametricMiCSEdge | edge_target=%s | edge_loss_weight=%s",
        edge_label,
        kw.get("edge_loss_weight"),
    )

    model: MSIParametricMiCSEdge | None = None
    try:
        model = MSIParametricMiCSEdge(**kw)
        model.fit(msi, pixel_sampling_cache=pixel_cache)

        mics_rgb = np.asarray(model.predict(msi), dtype=np.uint8)[..., :3]
        mics_rgb[~valid_mask] = 0
        pred_edge = model.predict_edge_probability_map(msi)

        edge_rgb = _edge_to_heatmap(edge_target, valid_mask)
        pred_edge_rgb = _edge_to_heatmap(pred_edge, valid_mask)

        mics_path = pan_dir / "panel__mics_viz.png"
        edge_path = pan_dir / "panel__edge_target.png"
        pred_edge_path = pan_dir / "panel__viz_edge_mag.png"
        Image.fromarray(mics_rgb, mode="RGB").save(mics_path)
        Image.fromarray(edge_rgb, mode="RGB").save(edge_path)
        Image.fromarray(pred_edge_rgb, mode="RGB").save(pred_edge_path)

        titles = ["MiCS", f"edge target ({edge_label})", "predicted edge prob"]
        rgbs = [mics_rgb, edge_rgb, pred_edge_rgb]
        mosaic = out_dir / "gallery_mics_edge.png"
        _save_mosaic(
            rgbs,
            titles,
            out_path=mosaic,
            ncols=ncols,
            fig_w=fig_w,
            dpi=dpi,
            title_fs=title_fs,
        )

        m = valid_mask
        p = pred_edge[m].astype(np.float64)
        t = edge_target[m].astype(np.float64)
        inter = float((p * t).sum())
        dice = (2.0 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)

        summary = out_dir / "mics_edge_summary.txt"
        with summary.open("w", encoding="utf-8") as fh:
            fh.write(f"npy_path: {npy_path}\n")
            fh.write(f"edge_target: {edge_label}\n")
            fh.write(f"edge_loss_mode: {kw.get('edge_loss_mode')}\n")
            fh.write(f"edge_binary_threshold: {kw.get('edge_binary_threshold')}\n")
            fh.write(f"edge_label_smoothing: {kw.get('edge_label_smoothing')}\n")
            fh.write(f"edge_sample_weight_mode: {kw.get('edge_sample_weight_mode')}\n")
            fh.write(f"edge_loss_weight: {kw.get('edge_loss_weight')}\n")
            fh.write(f"edge_mse_weight: {kw.get('edge_mse_weight')}\n")
            fh.write(f"cluster: {kw.get('cluster')}\n")
            fh.write(f"final_edge_dice: {dice:.4f}\n\n")
            fh.write(f"MiCS: {_abs_path_str(mics_path)}\n")
            fh.write(f"edge target: {_abs_path_str(edge_path)}\n")
            fh.write(f"viz edges: {_abs_path_str(pred_edge_path)}\n")
            fh.write(f"\nmosaic: {_abs_path_str(mosaic)}\n")

        logger.info("Final edge Dice (viz vs target): %.4f", dice)
        logger.info("Gallery: %s", _abs_path_str(mosaic))
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
