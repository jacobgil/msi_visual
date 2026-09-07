#!/usr/bin/env python3
"""
Cumulative MiCS cluster-ladder gallery: for ladder k0, k1, ... k(L-1), train MiCS independently
with clusters k0 only, then k0,k1, ..., save each viz and a mosaic figure.

Uses ``gallery.*`` overrides + merged ``train_parametric_mics_lmc`` defaults (model.*, data transpose/tic semantics mirrored for loading only).
Requires ``model.model_type=mics_lmc``.

Optional ``gallery.metrics``: per-panel titles with continuous Dice vs HD landmark edges and
viz-only ``luminance_rms_contrast`` / ``lab_chroma_entropy`` (same trio as Bayesian tuning). Disable via
``gallery.metrics.enabled=false``.
"""

from __future__ import annotations

import gc
import logging
import sys
import csv
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

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402

from debug_edge_maps import (  # noqa: E402
    _compute_continuous_dice,
    _compute_viz_contrast_color_metrics,
    _edge_maps_multiscale,
    _edges_for_continuous_metrics,
    _normalize_map_percentile,
    _normalize_on_mask,
    _parse_edge_detection_cfg,
)
from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method  # noqa: E402

logger = logging.getLogger(__name__)


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    s = msi.sum(axis=-1, keepdims=True)
    return msi / (s + 1e-8)


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(str(path), mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI HxWxC (or transpose to HxWxC), got shape {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI must be nonnegative for this gallery script.")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def _to_uint8_rgb(viz: Any) -> np.ndarray:
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
    mx = float(np.nanmax(out)) if out.size else 0.0
    if mx <= 1.0 + 1e-6:
        if mx > 1e-12:
            out = (out / mx) * 255.0
        else:
            out = np.zeros_like(out)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _hist_equalize_rgb(viz_rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2

        out = np.asarray(viz_rgb, dtype=np.uint8).copy()
        for ch in range(min(3, out.shape[-1])):
            out[:, :, ch] = cv2.equalizeHist(out[:, :, ch])
        return out
    except Exception:
        return viz_rgb


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
)


def _mics_kwargs_from_train_cfg(cfg: DictConfig, clusters: list[int], seed: int) -> dict[str, Any]:
    mt = str(OmegaConf.select(cfg.model, "model_type", default="mics_lmc")).strip().lower()
    if mt != "mics_lmc":
        raise ValueError(
            f"gallery_mics_cluster_ladder supports model.model_type=mics_lmc only (got {mt!r})"
        )

    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}

    kw: dict[str, Any] = {}
    for name in _MICS_KEYS:
        if name == "clusters":
            continue
        if name in merged:
            kw[name] = merged[name]

    defaults_num_layers = int(merged.get("num_layers", 2))
    defaults_num_points = int(merged.get("number_of_points", 1000))
    kw.setdefault("number_of_points", defaults_num_points)
    kw.setdefault("sampling", str(merged.get("sampling", "coreset")))
    kw.setdefault("num_epochs", int(merged.get("num_epochs", 100)))
    kw.setdefault("number_of_components", int(merged.get("number_of_components", 3)))
    kw.setdefault("num_layers", defaults_num_layers)
    kw.setdefault("lab_to_rgb", bool(merged.get("lab_to_rgb", True)))
    kw.setdefault("cluster", True)
    kw.setdefault("num_samples", int(merged.get("num_samples", 5000)))
    kw.setdefault("beta", float(merged.get("beta", 1.0)))
    kw.setdefault("k_epoch", int(merged.get("k_epoch", 20)))
    kw.setdefault("temperature", float(merged.get("temperature", 100.0)))
    kw.setdefault("lr", float(merged.get("lr", 1.0)))
    kw.setdefault("batch_size", int(merged.get("batch_size", 1024)))
    kw.setdefault("warmup_epochs", int(merged.get("warmup_epochs", 10)))
    kw.setdefault("factor", float(merged.get("factor", 1.0)))
    kw.setdefault("verbose", bool(merged.get("verbose", False)))
    kw.setdefault("cluster_loss_weight", float(merged.get("cluster_loss_weight", 1.0)))
    kw.setdefault("category_loss_weight", float(merged.get("category_loss_weight", 0.0)))
    kw.setdefault("pixel_sampling", str(merged.get("pixel_sampling", "superpixel")))
    kw.setdefault("pca_fit_step", int(merged.get("pca_fit_step", 4)))
    kw.setdefault("cluster_on_pca", bool(merged.get("cluster_on_pca", False)))
    kw.setdefault("cluster_pca_dims", int(merged.get("cluster_pca_dims", 50)))
    kw.setdefault(
        "predict_percentile_low",
        float(merged.get("predict_percentile_low", 0.001)),
    )
    kw.setdefault(
        "predict_percentile_high",
        float(merged.get("predict_percentile_high", 99.999)),
    )
    kw.setdefault(
        "predict_spatial_smooth_sigma",
        float(merged.get("predict_spatial_smooth_sigma", 0.0)),
    )

    kw["clusters"] = [int(k) for k in clusters]
    kw["clusters_auto_tune"] = None

    strip_prefix = tuple(x for x in kw if str(x).startswith("clusters_auto_tune_"))
    for sk in strip_prefix:
        kw.pop(sk, None)

    kw["random_state"] = int(seed)
    kw["cluster"] = True

    return kw


def _cumulative_prefixes(ladder: list[int]) -> list[list[int]]:
    if len(ladder) < 1:
        raise ValueError("gallery.ladder must be non-empty")
    prev: int | None = None
    out: list[list[int]] = []
    acc: list[int] = []
    for k in ladder:
        kk = int(k)
        if kk < 2:
            raise ValueError(f"Ladder entries must be >= 2 (got {kk})")
        if prev is not None and kk == prev:
            logger.warning("Adjacent duplicate ladder entry %s — successive panels may look very similar.", kk)
        prev = kk
        acc.append(kk)
        out.append(list(acc))
    return out


def _gallery_compute_viz_edge_n(
    viz_rgb: np.ndarray, valid_mask: np.ndarray, rank_cfg: DictConfig
) -> np.ndarray:
    """Same normalized RGB viz-edge map pipeline as bayes_tune_mics_lmc._compute_viz_edge_n."""
    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    ms_enabled = bool(getattr(multi, "enabled", False)) if multi is not None else False
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    if (not ms_enabled) or not sigmas:
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    edge_cfg = dict(_parse_edge_detection_cfg(rank_cfg))
    v_edge, _v_edge_linf = _edge_maps_multiscale(
        "rgb",
        viz_rgb,
        valid_mask,
        sigmas=sigmas,
        aggregation=aggregation,
        hd_cfg=None,
        edge_method_cfg=edge_cfg,
    )
    sn = getattr(rank_cfg, "spatial_normalization", None)
    if sn is not None and bool(getattr(sn, "enabled", True)):
        return _normalize_map_percentile(
            v_edge,
            valid_mask,
            low_pct=float(getattr(sn, "low_percentile", 1.0)),
            high_pct=float(getattr(sn, "high_percentile", 99.0)),
        )
    return _normalize_on_mask(v_edge, valid_mask)


def _fmt_metric(x: float, nd_dec: int) -> str:
    xf = float(x)
    if not np.isfinite(xf):
        return "nan"
    return f"{xf:.{nd_dec}f}"


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_cluster_ladder")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ga = getattr(cfg, "gallery", None)
    if ga is None:
        raise ValueError("Config must define a top-level gallery: section.")

    npy_path = Path(to_absolute_path(str(ga.npy_path))).expanduser().resolve()
    if not npy_path.is_file():
        raise FileNotFoundError(f"gallery.npy_path not found: {npy_path}")

    transpose = bool(getattr(ga, "transpose_msi", False))
    ticnorm = bool(getattr(ga, "tic_normalize", True))
    msi_vol = _load_msi(npy_path, transpose, ticnorm)
    valid_mask = msi_vol.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty (no pixels with positive total intensity).")

    ladder_raw = OmegaConf.to_container(getattr(ga, "ladder", None), resolve=True)
    if isinstance(ladder_raw, (list, tuple)):
        ladder = [int(x) for x in ladder_raw]
    else:
        raise ValueError("gallery.ladder must be a non-empty list of integers")

    max_panels = OmegaConf.select(ga, "max_panels", default=None)
    prefixes_full = _cumulative_prefixes(ladder)
    if max_panels is not None and int(max_panels) >= 1:
        prefixes_full = prefixes_full[: int(max_panels)]

    base_seed = int(getattr(ga, "seed", 42))
    eq = bool(getattr(ga, "equalize_visualization", False))
    ncols = max(1, int(getattr(ga, "ncols", 3)))
    dpi = float(getattr(ga, "dpi", 160))
    fig_w = float(getattr(ga, "figure_size_inches", 14))
    title_fs = float(getattr(ga, "panel_title_fontsize", 11))

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    pan_dir = out_dir / "panels"
    pan_dir.mkdir(parents=True, exist_ok=True)
    resolved = out_dir / "config_resolved.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    metrics_enabled = bool(OmegaConf.select(ga, "metrics.enabled", default=False))
    square_cd = bool(OmegaConf.select(ga, "metrics.square_before_continuous_dice", default=False))
    rank_cfg_live: DictConfig | None = None
    hd_edge_n: np.ndarray | None = None
    metric_rows: list[dict[str, Any]] = []

    if metrics_enabled:
        rcp_raw = OmegaConf.select(ga, "metrics.rank_config", default=None)
        if rcp_raw is None or str(rcp_raw).strip().lower() in ("~", "null", "none", ""):
            rc_path = (_SCRIPTS / "configs" / "hd_methods_gallery.yaml").resolve()
        else:
            rc_path = Path(to_absolute_path(str(rcp_raw))).expanduser().resolve()
        if not rc_path.is_file():
            logger.warning(
                "gallery.metrics.rank_config path not found (%s); disabling per-panel metrics.",
                rc_path,
            )
            metrics_enabled = False
        else:
            rank_cfg_live = OmegaConf.load(str(rc_path))
            ms_cfg = getattr(rank_cfg_live, "edge_maps", None)
            multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
            sigmas_mt = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
            ms_enabled_mt = bool(getattr(multi, "enabled", False)) if multi is not None else False
            if (not ms_enabled_mt) or not sigmas_mt:
                sigmas_mt = [0.0]
            agg_mt = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
            sn_live = getattr(rank_cfg_live, "spatial_normalization", None)
            spatial_mt = bool(getattr(sn_live, "enabled", True)) if sn_live is not None else True
            hd_method = str(
                OmegaConf.select(rank_cfg_live, "edge_detection.hd_method", default="soft_landmark_contrast")
            )
            if hd_method != "soft_landmark_contrast":
                logger.warning(
                    "Gallery metrics forces edge_detection.hd_method=soft_landmark_contrast (config had %r).",
                    hd_method,
                )
                rank_cfg_live.edge_detection.hd_method = "soft_landmark_contrast"
            try:
                hd_edge_n = _compute_hd_edge_n_for_method(
                    rank_cfg_live,
                    msi_vol,
                    valid_mask,
                    getattr(rank_cfg_live, "hd_edges", None),
                    "soft_landmark_contrast",
                    sigmas_mt,
                    agg_mt,
                    spatial_mt,
                )
            except Exception as ex:
                logger.warning("Could not compute HD reference edges (%s); metrics disabled.", ex)
                hd_edge_n = None
                rank_cfg_live = None

    hd_ready = bool(metrics_enabled and rank_cfg_live is not None and hd_edge_n is not None)
    if hd_ready:
        logger.info("Per-panel metrics enabled (continuous Dice vs HD edges + viz color duo).")

    n_p = len(prefixes_full)
    nrows = int(np.ceil(n_p / ncols))
    mh = float(max(fig_w * 0.35, fig_w * 0.26 * float(max(1, nrows))))
    if hd_ready:
        mh *= 1.12
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, mh))
    if nrows == 1 and ncols == 1:
        ax_list = np.array([[axes]])
    elif nrows == 1:
        ax_list = np.asarray(axes).reshape(1, -1)
    elif ncols == 1:
        ax_list = np.asarray(axes).reshape(-1, 1)
    else:
        ax_list = np.asarray(axes)

    for idx, ks in enumerate(prefixes_full):
        lbl = "-".join(str(x) for x in ks)
        kw = _mics_kwargs_from_train_cfg(cfg, ks, seed=base_seed)
        logger.info("Panel %d/%d | clusters=%s | seed=%s", idx + 1, n_p, ks, base_seed)

        model: MSIParametricMiCSLMC | None = None
        try:
            model = MSIParametricMiCSLMC(**kw)
            model.fit(msi_vol)
            viz = model.predict(msi_vol)
            rgb = _to_uint8_rgb(viz)
            if eq:
                rgb = _hist_equalize_rgb(rgb)
            raw_path = pan_dir / f"panel_{idx:02d}__clusters__{lbl}__viz.png"
            Image.fromarray(rgb, mode="RGB").save(raw_path)

            dice_f = float("nan")
            lum_f = float("nan")
            lab_h = float("nan")
            if hd_ready and rank_cfg_live is not None and hd_edge_n is not None:
                try:
                    viz_edge_n = _gallery_compute_viz_edge_n(rgb, valid_mask, rank_cfg_live)
                    hd_d, viz_d = _edges_for_continuous_metrics(hd_edge_n, viz_edge_n, square=square_cd)
                    dice_f = float(_compute_continuous_dice(hd_d, viz_d, valid_mask))
                    cq = _compute_viz_contrast_color_metrics(rgb, valid_mask)
                    lum_f = float(cq.get("luminance_rms_contrast", float("nan")))
                    lab_h = float(cq.get("lab_chroma_entropy", float("nan")))
                except Exception as ex:
                    logger.warning("Panel %d metric computation failed: %s", idx + 1, ex)

            ks_joined = ", ".join(str(x) for x in ks)
            if hd_ready:
                tit = (
                    f"clusters = {ks_joined}\n"
                    f"cDice={_fmt_metric(dice_f, 3)}\n"
                    f"L_rms={_fmt_metric(lum_f, 4)}  LAB_H={_fmt_metric(lab_h, 3)}"
                )
                subtitle_fs = max(8.0, title_fs - 1.5)
                metric_rows.append(
                    {
                        "panel_idx": idx,
                        "clusters": ks_joined,
                        "continuous_dice": dice_f,
                        "luminance_rms_contrast": lum_f,
                        "lab_chroma_entropy": lab_h,
                    }
                )
            else:
                tit = f"clusters = {ks_joined}"
                subtitle_fs = title_fs

            r, cdiv = divmod(idx, ncols)
            ax = ax_list[r, cdiv]
            ax.imshow(rgb)
            ax.set_title(tit, fontsize=subtitle_fs, pad=8)
            ax.axis("off")
        finally:
            if model is not None:
                try:
                    model.release_resources()
                except Exception:
                    pass
                del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    used = len(prefixes_full)
    for blank in range(used, nrows * ncols):
        r, cdiv = divmod(blank, ncols)
        ax_list[r, cdiv].axis("off")

    ladder_tip = prefixes_full[-1] if prefixes_full else []
    k_seq = ", ".join(str(x) for x in ladder_tip)
    fig.suptitle(
        f"MiCS cumulative cluster ladder  ·  k: {k_seq}",
        fontsize=title_fs + 2,
        y=0.98,
    )
    adj_top = 0.88 if hd_ready else 0.91
    adj_h = 0.52 if hd_ready else 0.38
    # Avoid plt.tight_layout + bbox_inches="tight": they frequently clip subplot titles on lower rows.
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=adj_top,
        wspace=0.08,
        hspace=adj_h,
    )
    gallery_path = out_dir / "cluster_ladder_gallery.png"
    fig.savefig(gallery_path, dpi=dpi)
    plt.close(fig)

    if metric_rows:
        csv_p = out_dir / "panel_metrics.csv"
        with csv_p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "panel_idx",
                    "clusters",
                    "continuous_dice",
                    "luminance_rms_contrast",
                    "lab_chroma_entropy",
                ],
                extrasaction="ignore",
            )
            w.writeheader()
            for row in metric_rows:
                out_row = dict(row)
                for k in ("continuous_dice", "luminance_rms_contrast", "lab_chroma_entropy"):
                    v = out_row[k]
                    if isinstance(v, float) and np.isfinite(v):
                        out_row[k] = f"{v:.8g}"
                    else:
                        out_row[k] = ""
                w.writerow(out_row)

    ladder_readme = " | ".join(", ".join(str(x) for x in p) for p in prefixes_full)
    readme_lines = [
                "MiCS cluster-ladder cumulative gallery.",
                f"MSI: {npy_path}",
                f"Ladder prefixes (cumulative): {ladder_readme}",
                f"Mosaic: {gallery_path.relative_to(out_dir)}",
                f"Panels: {pan_dir.relative_to(out_dir)}/",
            ]
    if metric_rows:
        readme_lines.append("Per-panel metrics (CSV): panel_metrics.csv")
    readme_lines.append("")
    (out_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"[gallery_mics_cluster_ladder] Saved mosaic: {gallery_path}", flush=True)
    logger.info("Done | %s", gallery_path)


if __name__ == "__main__":
    main()
