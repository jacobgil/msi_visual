#!/usr/bin/env python3
"""
Gallery for experimental residual-cluster MiCS.

Trains one ``MSIParametricMiCSResidualCluster`` model, then builds a mosaic.

Default (viz-only): ``shared | k=8 | k=16 | … | k=256``

Set ``gallery.show_delta_panels=true`` to include Δ panels;
``gallery.show_final_panel=true`` adds final (same as deepest k=).
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

from msi_visual.parametric_mics_residual_cluster import MSIParametricMiCSResidualCluster  # noqa: E402
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
    "residual_factor",
    "residual_num_layers",
    "residual_l2_weight",
    "residual_l2_relative",
    "level_wise_freeze",
    "epochs_per_level",
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
        return [8, 16, 32, 64, 128, 256]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
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
    kw.setdefault("num_layers", int(merged.get("num_layers", 2)))
    kw.setdefault("verbose", bool(merged.get("verbose", False)))
    kw.setdefault("pixel_sampling", str(merged.get("pixel_sampling", "superpixel")))
    kw.setdefault(
        "predict_spatial_smooth_sigma", float(merged.get("predict_spatial_smooth_sigma", 0.8))
    )
    kw.setdefault("residual_factor", float(merged.get("residual_factor", 0.25)))
    kw.setdefault("residual_num_layers", int(merged.get("residual_num_layers", 2)))
    kw.setdefault("residual_l2_weight", float(merged.get("residual_l2_weight", 0.05)))
    kw.setdefault("residual_l2_relative", bool(merged.get("residual_l2_relative", True)))
    kw.setdefault("level_wise_freeze", bool(merged.get("level_wise_freeze", False)))
    ep_level = merged.get("epochs_per_level", None)
    if ep_level is None or str(ep_level).strip().lower() in ("", "~", "null", "none"):
        kw["epochs_per_level"] = None
    else:
        kw["epochs_per_level"] = max(1, int(ep_level))
    kw["clusters"] = _parse_clusters(kw.get("clusters", merged.get("clusters", [8, 16, 32, 64, 128, 256])))
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw["random_state"] = int(seed)
    kw["learn_input_mask"] = False
    kw["input_mask_l1_weight"] = 0.0
    return kw


def _compute_norm_anchor(
    emb: np.ndarray,
    mask: np.ndarray,
    low: float,
    high: float,
) -> list[tuple[float, float]]:
    anchor: list[tuple[float, float]] = []
    m = np.asarray(mask, dtype=bool)
    for c in range(int(emb.shape[-1])):
        vals = np.asarray(emb[..., c], dtype=np.float64)[m]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            anchor.append((0.0, 1.0))
        else:
            anchor.append((float(np.percentile(vals, low)), float(np.percentile(vals, high))))
    return anchor


def _embedding_to_rgb_with_anchor(
    emb: np.ndarray,
    mask: np.ndarray,
    anchor: list[tuple[float, float]],
    *,
    lab_to_rgb: bool,
    n_components: int,
    smooth_sigma: float,
) -> np.ndarray:
    import cv2

    from msi_visual.parametric_mics_lmc import _spatial_gaussian_smooth_channels

    result = np.asarray(emb, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=bool)
    for c, (p_lo, p_hi) in enumerate(anchor):
        ch = result[..., c]
        ch = (ch - p_lo) / max(p_hi - p_lo, 1e-8)
        ch = np.clip(ch, 0.0, 1.0)
        result[..., c] = ch
    result[~m] = 0.0
    result = _spatial_gaussian_smooth_channels(result, smooth_sigma)
    out = np.uint8(255.0 * result)
    out[~m] = 0
    if lab_to_rgb and n_components == 3:
        out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    out[~m] = 0
    return out


def _embedding_to_rgb_per_panel(
    emb: np.ndarray,
    mask: np.ndarray,
    *,
    low: float,
    high: float,
    lab_to_rgb: bool,
    n_components: int,
    smooth_sigma: float,
) -> np.ndarray:
    anchor = _compute_norm_anchor(emb, mask, low, high)
    return _embedding_to_rgb_with_anchor(
        emb,
        mask,
        anchor,
        lab_to_rgb=lab_to_rgb,
        n_components=n_components,
        smooth_sigma=smooth_sigma,
    )


def _resolve_normalization(ga: Any) -> str:
    raw = OmegaConf.select(ga, "normalization", default=None)
    if raw is not None and str(raw).strip().lower() not in ("", "~", "null", "none"):
        mode = str(raw).strip().lower()
        if mode not in ("per_panel", "shared"):
            raise ValueError("gallery.normalization must be per_panel | shared")
        return mode
    if bool(OmegaConf.select(ga, "shared_normalization", default=False)):
        return "shared"
    return "per_panel"


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


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_residual_cluster")
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
    ncols = max(1, int(OmegaConf.select(ga, "ncols", default=4)))
    fig_w = float(OmegaConf.select(ga, "figure_size_inches", default=16))
    dpi = int(OmegaConf.select(ga, "dpi", default=300))
    title_fs = int(OmegaConf.select(ga, "panel_title_fontsize", default=9))
    normalization = _resolve_normalization(ga)
    show_deltas = bool(OmegaConf.select(ga, "show_delta_panels", default=False))
    show_final = bool(OmegaConf.select(ga, "show_final_panel", default=False))
    include_steps = bool(OmegaConf.select(ga, "include_fused_increments", default=False))

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    pan_dir = out_dir / "panels"
    pan_dir.mkdir(parents=True, exist_ok=True)

    kw = _mics_kwargs_from_cfg(cfg, seed)
    clusters = _parse_clusters(kw["clusters"])
    pred_lo = float(kw.get("predict_percentile_low", 1.0))
    pred_hi = float(kw.get("predict_percentile_high", 99.0))
    lab2rgb = bool(kw.get("lab_to_rgb", True))
    n_ch = int(kw.get("number_of_components", 3))
    smooth_sigma = float(kw.get("predict_spatial_smooth_sigma", 0.0))

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
        "Training MSIParametricMiCSResidualCluster | clusters=%s | residual_factor=%s | "
        "residual_l2_weight=%s",
        clusters,
        kw.get("residual_factor"),
        kw.get("residual_l2_weight"),
    )
    model: MSIParametricMiCSResidualCluster | None = None
    try:
        model = MSIParametricMiCSResidualCluster(**kw)
        model.fit(msi, pixel_sampling_cache=pixel_cache)

        shared_emb = model.predict_shared_embedding(msi)
        final_emb = model.predict_embedding(msi)
        np.save(out_dir / "shared_embedding.npy", shared_emb)
        np.save(out_dir / "final_embedding.npy", final_emb)

        shared_anchor: list[tuple[float, float]] | None = None
        if normalization == "shared":
            shared_anchor = _compute_norm_anchor(shared_emb, valid_mask, pred_lo, pred_hi)

        panel_rgbs: list[np.ndarray] = []
        panel_titles: list[str] = []
        step_rgbs: list[np.ndarray] = []
        step_titles: list[str] = []
        rows: list[dict[str, str]] = []

        def _to_rgb(
            emb: np.ndarray,
            title: str,
            fname: str,
            *,
            per_panel: bool = False,
            to_mosaic: bool = True,
            mosaic_list: list[np.ndarray] | None = None,
            title_list: list[str] | None = None,
        ) -> np.ndarray:
            if per_panel or shared_anchor is None:
                rgb = _embedding_to_rgb_per_panel(
                    emb,
                    valid_mask,
                    low=pred_lo,
                    high=pred_hi,
                    lab_to_rgb=lab2rgb,
                    n_components=n_ch,
                    smooth_sigma=smooth_sigma,
                )
            else:
                rgb = _embedding_to_rgb_with_anchor(
                    emb,
                    valid_mask,
                    shared_anchor,
                    lab_to_rgb=lab2rgb,
                    n_components=n_ch,
                    smooth_sigma=smooth_sigma,
                )
            path = pan_dir / fname
            Image.fromarray(rgb, mode="RGB").save(path)
            if to_mosaic:
                target_rgbs = mosaic_list if mosaic_list is not None else panel_rgbs
                target_titles = title_list if title_list is not None else panel_titles
                target_rgbs.append(rgb)
                target_titles.append(title)
            rows.append({"panel": title, "path": _abs_path_str(path)})
            return rgb

        _to_rgb(
            shared_emb,
            "shared",
            "panel__shared.png",
            per_panel=(normalization == "per_panel"),
        )

        n_levels = len(clusters)
        prev_fused = shared_emb
        for li, k_label in enumerate(clusters):
            resid_emb = model.predict_level_residual_embedding(msi, li)
            fused_emb = model.predict_level_fused_embedding(msi, li)
            np.save(pan_dir / f"residual_k{k_label}__embedding.npy", resid_emb)
            np.save(pan_dir / f"fused_k{k_label}__embedding.npy", fused_emb)
            if show_deltas:
                _to_rgb(
                    resid_emb,
                    f"Δ k={k_label}",
                    f"panel__delta_k{k_label}.png",
                    per_panel=True,
                )
            _to_rgb(
                fused_emb,
                f"k={k_label}",
                f"panel__fused_k{k_label}.png",
                per_panel=(normalization == "per_panel"),
            )
            if include_steps:
                step_emb = (fused_emb - prev_fused).astype(np.float32, copy=False)
                np.save(pan_dir / f"step_k{k_label}__embedding.npy", step_emb)
                _to_rgb(
                    step_emb,
                    f"step k={k_label}",
                    f"panel__step_k{k_label}.png",
                    per_panel=True,
                    to_mosaic=False,
                    mosaic_list=step_rgbs,
                    title_list=step_titles,
                )
            prev_fused = fused_emb

        if show_final:
            _to_rgb(
                final_emb,
                "final",
                "panel__final.png",
                per_panel=(normalization == "per_panel"),
            )

        delta_stats = model.compute_residual_delta_stats(msi, valid_mask)
        logger.info(
            "Delta RMS vs shared: final=%.3f shared_rms=%.4g",
            delta_stats["final_minus_shared_over_shared"],
            delta_stats["rms_shared"],
        )
        for lv in delta_stats["levels"]:
            logger.info(
                "  level %d: |Δ|/|shared|=%.3f |fused-shared|/|shared|=%.3f",
                lv["level"],
                lv["delta_over_shared"],
                lv["fused_minus_shared_over_shared"],
            )

        mosaic = out_dir / "gallery_mics_residual_cluster.png"
        _save_mosaic(
            panel_rgbs,
            panel_titles,
            out_path=mosaic,
            ncols=ncols,
            fig_w=fig_w,
            dpi=dpi,
            title_fs=title_fs,
        )

        steps_mosaic = out_dir / "gallery_mics_residual_cluster_steps.png"
        if include_steps and step_rgbs:
            _save_mosaic(
                [panel_rgbs[0], *step_rgbs],
                [panel_titles[0], *step_titles],
                out_path=steps_mosaic,
                ncols=min(ncols, 1 + len(step_rgbs)),
                fig_w=fig_w,
                dpi=dpi,
                title_fs=title_fs,
            )

        csv_path = out_dir / "residual_cluster_panels.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["panel", "path"])
            writer.writeheader()
            writer.writerows(rows)

        summary = out_dir / "residual_cluster_summary.txt"
        with summary.open("w", encoding="utf-8") as fh:
            fh.write(f"npy_path: {npy_path}\n")
            fh.write(f"clusters: {clusters}\n")
            fh.write(f"residual_factor: {kw.get('residual_factor')}\n")
            fh.write(f"residual_num_layers: {kw.get('residual_num_layers')}\n")
            fh.write(f"residual_l2_weight: {kw.get('residual_l2_weight')}\n")
            fh.write(f"residual_l2_relative: {kw.get('residual_l2_relative')}\n")
            fh.write(f"level_wise_freeze: {kw.get('level_wise_freeze')}\n")
            fh.write(f"epochs_per_level: {kw.get('epochs_per_level')}\n")
            fh.write(f"normalization: {normalization}\n")
            fh.write(f"show_delta_panels: {show_deltas}\n")
            fh.write(f"show_final_panel: {show_final}\n")
            fh.write(f"include_fused_increments: {include_steps}\n\n")
            fh.write("delta_stats (RMS on valid tissue):\n")
            fh.write(f"  rms_shared: {delta_stats['rms_shared']:.6g}\n")
            fh.write(f"  rms_final: {delta_stats['rms_final']:.6g}\n")
            fh.write(
                f"  |final-shared|/|shared|: "
                f"{delta_stats['final_minus_shared_over_shared']:.4f}\n"
            )
            for lv in delta_stats["levels"]:
                fh.write(
                    f"  level {lv['level']}: |Δ|/|shared|={lv['delta_over_shared']:.4f} "
                    f"|fused-shared|/|shared|={lv['fused_minus_shared_over_shared']:.4f}\n"
                )
            fh.write("\n")
            for row in rows:
                fh.write(f"{row['panel']}: {row['path']}\n")
            fh.write(f"\nmosaic: {_abs_path_str(mosaic)}\n")
            if include_steps and step_rgbs:
                fh.write(f"steps_mosaic: {_abs_path_str(steps_mosaic)}\n")

        logger.info("Gallery mosaic: %s", _abs_path_str(mosaic))
        if include_steps and step_rgbs:
            logger.info("Step increment mosaic: %s", _abs_path_str(steps_mosaic))
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
