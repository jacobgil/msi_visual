#!/usr/bin/env python3
"""
Train MiCS (frozen), learn per-pixel concept weights (torch.zeros + backprop),
reconstruct the visualization, and explain colors in homogeneous regions.
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
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

from benchmark_parametric_methods import _seed_everything  # noqa: E402
from debug_edge_maps import _load_msi, _resolve_normalization_mode, _to_uint8_rgb  # noqa: E402
from gallery_mics_concept_distill import (  # noqa: E402
    _load_teacher_cache,
    _masked_mse,
    _mics_kwargs,
    _pixel_sampling_cache_path,
    _save_teacher_cache,
    _teacher_cache_path,
)
from gallery_mics_sampling_cache import (  # noqa: E402
    resolve_pixel_sampling_cache,
    save_pixel_sampling_cache,
)
from msi_visual.mics_concept_explain import (  # noqa: E402
    ConceptExplainConfig,
    FrozenConceptExplainer,
    _rgb_color_name,
    attach_region_alphas,
    build_color_explanation,
    build_region_overview,
    concept_colors_precomputed,
    save_concept_color_palette,
    save_reconstruction_mosaic,
    save_region_explanations,
    select_homogeneous_regions,
)
from msi_visual.mics_concept_viz import save_concept_gallery  # noqa: E402
from msi_visual.mics_concept_explain_viz import (  # noqa: E402
    save_explanation_visualizations,
    save_interpretability_overview_gallery,
)
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402

logger = logging.getLogger(__name__)


def _abs_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _concepts_cfg(cfg: DictConfig, model_kw: dict[str, Any]) -> ConceptExplainConfig:
    raw = OmegaConf.select(cfg, "concepts") or {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ConceptExplainConfig(
        n_concepts=int(OmegaConf.select(raw, "n_concepts", default=32)),
        top_k=int(OmegaConf.select(raw, "top_k", default=4)),
        nmf_max_iter=int(OmegaConf.select(raw, "nmf_max_iter", default=200)),
        nmf_init_pixels=int(OmegaConf.select(raw, "nmf_init_pixels", default=12000)),
        ridge_lambda=float(OmegaConf.select(raw, "ridge_lambda", default=1e-2)),
        normalize_alpha=bool(OmegaConf.select(raw, "normalize_alpha", default=False)),
        lab_to_rgb=bool(model_kw.get("lab_to_rgb", True)),
        percentile_low=float(model_kw.get("predict_percentile_low", 1.0)),
        percentile_high=float(model_kw.get("predict_percentile_high", 99.0)),
        nnls_batch_size=int(OmegaConf.select(raw, "nnls_batch_size", default=8192)),
        fit_method=str(OmegaConf.select(raw, "fit_method", default="backprop")),
        weight_mode=str(OmegaConf.select(raw, "weight_mode", default="sparse_softmax")),
        softmax_temperature=float(OmegaConf.select(raw, "softmax_temperature", default=1.0)),
        train_epochs=int(OmegaConf.select(raw, "train_epochs", default=300)),
        train_lr=float(OmegaConf.select(raw, "train_lr", default=0.05)),
        sparsity_weight=float(OmegaConf.select(raw, "sparsity_weight", default=0.0)),
        log_every=int(OmegaConf.select(raw, "log_every", default=50)),
        mics_input_batch_size=int(OmegaConf.select(raw, "mics_input_batch_size", default=2048)),
        mics_input_tic_normalize=bool(
            OmegaConf.select(raw, "mics_input_tic_normalize", default=True)
        ),
        hard_concrete_l0_weight=float(
            OmegaConf.select(raw, "hard_concrete_l0_weight", default=0.0)
        ),
        hard_concrete_temperature=float(
            OmegaConf.select(raw, "hard_concrete_temperature", default=2.0 / 3.0)
        ),
        hard_concrete_gamma=float(
            OmegaConf.select(raw, "hard_concrete_gamma", default=-0.1)
        ),
        hard_concrete_zeta=float(
            OmegaConf.select(raw, "hard_concrete_zeta", default=1.1)
        ),
        hard_concrete_init_prob=float(
            OmegaConf.select(raw, "hard_concrete_init_prob", default=0.1)
        ),
        device=str(OmegaConf.select(raw, "device", default=device)),
        seed=int(cfg.seed),
    )


def _compute_solace_rgb(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rank_config_path: str,
    edge_colormap: str = "PuBu",
    cache_path: Path | None = None,
) -> np.ndarray:
    """SoLaCE (soft_landmark_contrast) edge map as RGB, matching hd_methods gallery."""
    from debug_edge_maps import _gray_u8_to_rgb_colormap, _to_uint8_gray01
    from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method

    if cache_path is not None and cache_path.is_file():
        return np.asarray(Image.open(cache_path).convert("RGB"), dtype=np.uint8)

    rank_cfg = OmegaConf.load(to_absolute_path(rank_config_path))
    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    if not bool(getattr(multi, "enabled", False)):
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    spatial_enabled = bool(getattr(rank_cfg.spatial_normalization, "enabled", True))

    mask = np.asarray(valid_mask, dtype=bool)
    hd_edge_n = _compute_hd_edge_n_for_method(
        rank_cfg,
        msi,
        mask,
        getattr(rank_cfg, "hd_edges", None),
        "soft_landmark_contrast",
        sigmas,
        aggregation,
        spatial_enabled,
    )
    rgb = _gray_u8_to_rgb_colormap(_to_uint8_gray01(hd_edge_n), str(edge_colormap))
    rgb[~mask] = 0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(cache_path)
    return rgb


def _cosine_stretched_rgb(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    percentile_low: float,
    percentile_high: float,
    gamma: float,
    cmap: str = "magma",
) -> np.ndarray:
    from debug_edge_maps import _gray_u8_to_rgb_colormap, _to_uint8_gray01

    mask = np.asarray(valid_mask, dtype=bool)
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = _tissue_percentile_limits(arr, mask, percentile_low, percentile_high)
    norm = np.zeros_like(arr, dtype=np.float32)
    norm[mask] = np.clip((arr[mask] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    if float(gamma) != 1.0:
        norm[mask] = norm[mask] ** float(gamma)
    rgb = _gray_u8_to_rgb_colormap(_to_uint8_gray01(norm), str(cmap))
    rgb[~mask] = 0
    return rgb


def _tissue_percentile_limits(
    values: np.ndarray,
    valid_mask: np.ndarray,
    percentile_low: float,
    percentile_high: float,
) -> tuple[float, float]:
    tissue = np.asarray(values, dtype=np.float64)[np.asarray(valid_mask, dtype=bool)]
    if tissue.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(tissue, float(percentile_low)))
    hi = float(np.percentile(tissue, float(percentile_high)))
    if hi <= lo + 1e-8:
        hi = lo + 1e-3
    return lo, hi


def _save_scalar_map(
    values: np.ndarray,
    valid_mask: np.ndarray,
    path: Path,
    *,
    title: str = "",
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    show_title: bool = False,
    show_colorbar: bool = False,
) -> str:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    arr = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    shown = np.ma.array(arr, mask=~mask)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    if show_title and title:
        ax.set_title(title, color="white")
    ax.axis("off")
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor("white")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return str(path)


def _save_cosine_distance_maps(
    values: np.ndarray,
    valid_mask: np.ndarray,
    run_dir: Path,
    *,
    stem: str,
    title: str,
    percentile_low: float,
    percentile_high: float,
    gamma: float = 1.0,
) -> dict[str, Any]:
    """Save linear 0-1 and percentile-stretched cosine-distance maps."""
    mask = np.asarray(valid_mask, dtype=bool)
    arr = np.asarray(values, dtype=np.float32)
    linear_png = _save_scalar_map(
        arr,
        mask,
        run_dir / f"{stem}.png",
        title=title,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    lo, hi = _tissue_percentile_limits(arr, mask, percentile_low, percentile_high)
    stretched = arr.copy()
    norm = np.clip((stretched - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    if float(gamma) != 1.0:
        norm = norm ** float(gamma)
    stretched[mask] = norm[mask]
    stretched_png = _save_scalar_map(
        stretched,
        mask,
        run_dir / f"{stem}_stretched.png",
        title=f"{title} (p{percentile_low:g}-p{percentile_high:g} stretch)",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    return {
        "png": linear_png,
        "stretched_png": stretched_png,
        "percentile_low": float(percentile_low),
        "percentile_high": float(percentile_high),
        "stretch_gamma": float(gamma),
        "stretch_vmin": float(lo),
        "stretch_vmax": float(hi),
    }


def _keep_topk_rows(values: np.ndarray, top_k: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    n, k = arr.shape
    keep = max(1, min(int(top_k), k))
    out = np.zeros_like(arr, dtype=np.float32)
    idx = np.argpartition(-arr, keep - 1, axis=1)[:, :keep]
    rows = np.arange(n)[:, None]
    out[rows, idx] = arr[rows, idx]
    return out


def _row_cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    cosine = np.zeros(aa.shape[0], dtype=np.float32)
    good = denom > 1e-8
    cosine[good] = np.sum(aa[good] * bb[good], axis=1) / denom[good]
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def _save_nmf_deviation_diagnostics(
    msi: np.ndarray,
    explainer: FrozenConceptExplainer,
    valid_mask: np.ndarray,
    run_dir: Path,
    *,
    active_threshold: float = 1e-4,
    sparse_top_k: int = 4,
    cosine_percentile_low: float = 15.0,
    cosine_percentile_high: float = 90.0,
    cosine_gamma: float = 2.0,
) -> dict[str, Any]:
    mask = np.asarray(valid_mask, dtype=bool)
    alpha = np.asarray(explainer.alpha_hwk, dtype=np.float32)
    h, w, k = alpha.shape
    scores = FrozenConceptExplainer.concept_scores(msi[mask], explainer.concepts)
    alpha_flat = alpha.reshape(-1, k)[mask.ravel()]
    cosine_distance_flat = _row_cosine_distance(alpha_flat, scores)
    alpha_sparse = _keep_topk_rows(alpha_flat, int(sparse_top_k))
    scores_sparse = _keep_topk_rows(scores, int(sparse_top_k))
    sparse_cosine_distance_flat = _row_cosine_distance(alpha_sparse, scores_sparse)

    cosine_distance = np.zeros((h, w), dtype=np.float32)
    cosine_distance[mask] = cosine_distance_flat
    sparse_cosine_distance = np.zeros((h, w), dtype=np.float32)
    sparse_cosine_distance[mask] = sparse_cosine_distance_flat
    active_count = (alpha > float(active_threshold)).sum(axis=-1).astype(np.float32)
    active_count[~mask] = 0.0

    cosine_npy = run_dir / "nmf_projection_cosine_distance.npy"
    sparse_cosine_npy = run_dir / f"nmf_projection_sparse_top{int(sparse_top_k)}_cosine_distance.npy"
    active_npy = run_dir / "active_concept_count.npy"
    np.save(cosine_npy, cosine_distance)
    np.save(sparse_cosine_npy, sparse_cosine_distance)
    np.save(active_npy, active_count)

    cosine_maps = _save_cosine_distance_maps(
        cosine_distance,
        mask,
        run_dir,
        stem="nmf_projection_cosine_distance",
        title="Cosine distance: explanation weights vs raw NMF projection",
        percentile_low=float(cosine_percentile_low),
        percentile_high=float(cosine_percentile_high),
        gamma=float(cosine_gamma),
    )
    sparse_cosine_maps = _save_cosine_distance_maps(
        sparse_cosine_distance,
        mask,
        run_dir,
        stem=f"nmf_projection_sparse_top{int(sparse_top_k)}_cosine_distance",
        title=f"Sparse top-{int(sparse_top_k)} cosine distance: MiCS explanation vs NMF abundance",
        percentile_low=float(cosine_percentile_low),
        percentile_high=float(cosine_percentile_high),
        gamma=float(cosine_gamma),
    )
    active_png = _save_scalar_map(
        active_count,
        mask,
        run_dir / "active_concept_count.png",
        title=f"Active explanation concepts (alpha > {active_threshold:g})",
        cmap="viridis",
        vmin=0.0,
        vmax=float(max(1, int(np.nanmax(active_count)))),
        show_title=True,
        show_colorbar=True,
    )

    return {
        "cosine_distance_png": cosine_maps["png"],
        "cosine_distance_stretched_png": cosine_maps["stretched_png"],
        "cosine_distance_npy": str(cosine_npy),
        "cosine_stretch": {
            k: cosine_maps[k]
            for k in (
                "percentile_low",
                "percentile_high",
                "stretch_gamma",
                "stretch_vmin",
                "stretch_vmax",
            )
        },
        "sparse_top_k": int(sparse_top_k),
        "sparse_cosine_distance_png": sparse_cosine_maps["png"],
        "sparse_cosine_distance_stretched_png": sparse_cosine_maps["stretched_png"],
        "sparse_cosine_distance_npy": str(sparse_cosine_npy),
        "sparse_cosine_stretch": {
            k: sparse_cosine_maps[k]
            for k in (
                "percentile_low",
                "percentile_high",
                "stretch_gamma",
                "stretch_vmin",
                "stretch_vmax",
            )
        },
        "active_count_png": active_png,
        "active_count_npy": str(active_npy),
        "active_threshold": float(active_threshold),
        "mean_cosine_distance": float(np.mean(cosine_distance_flat)) if cosine_distance_flat.size else 0.0,
        "median_cosine_distance": float(np.median(cosine_distance_flat)) if cosine_distance_flat.size else 0.0,
        "mean_sparse_cosine_distance": (
            float(np.mean(sparse_cosine_distance_flat)) if sparse_cosine_distance_flat.size else 0.0
        ),
        "median_sparse_cosine_distance": (
            float(np.median(sparse_cosine_distance_flat)) if sparse_cosine_distance_flat.size else 0.0
        ),
        "mean_active_concepts": float(np.mean(active_count[mask])) if np.any(mask) else 0.0,
    }


@hydra.main(version_base=None, config_path="configs", config_name="explain_mics_concepts")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ga = cfg.gallery
    seed = int(cfg.seed)
    _seed_everything(seed)

    npy_path = _abs_path(str(ga.npy_path))
    if not npy_path.is_file():
        raise FileNotFoundError(f"gallery.npy_path not found: {npy_path}")

    norm_mode = _resolve_normalization_mode(str(cfg.data.normalization))
    msi = _load_msi(npy_path, transpose_msi=bool(cfg.data.transpose_msi), normalization=norm_mode)
    rot_cw = int(OmegaConf.select(cfg, "data.rotate_clockwise_deg", default=0)) % 360
    if rot_cw:
        if rot_cw not in (90, 180, 270):
            raise ValueError(
                f"data.rotate_clockwise_deg must be 0/90/180/270; got {rot_cw} "
                f"(from {OmegaConf.select(cfg, 'data.rotate_clockwise_deg', default=0)!r})"
            )
        # np.rot90 is counter-clockwise; map clockwise degrees → k.
        k_ccw = {90: 3, 180: 2, 270: 1}[rot_cw]
        before = tuple(int(x) for x in msi.shape)
        msi = np.ascontiguousarray(np.rot90(msi, k=k_ccw, axes=(0, 1)))
        logger.info(
            "Rotated MSI clockwise %d deg (np.rot90 k=%d): %s -> %s",
            rot_cw,
            k_ccw,
            before,
            tuple(int(x) for x in msi.shape),
        )
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_kw = _mics_kwargs(cfg)
    mics_group = str(OmegaConf.select(cfg, "mics.group", default="custom"))
    cache_teacher = bool(OmegaConf.select(ga, "cache_teacher", default=True))
    teacher_cache_path = _teacher_cache_path(cfg, npy_path, model_kw, mics_group)

    teacher_emb: np.ndarray | None = None
    mics_rgb: np.ndarray | None = None
    teacher: MSIParametricMiCSLMC | None = None
    concepts_cfg = _concepts_cfg(cfg, model_kw)
    needs_teacher_model = str(concepts_cfg.fit_method).strip().lower() in {
        "mics_input",
        "through_mics",
        "input_backprop",
    }
    if cache_teacher and not needs_teacher_model:
        cached = _load_teacher_cache(teacher_cache_path)
        if cached is not None:
            teacher_emb, mics_rgb = cached
            logger.info("Loaded cached MiCS teacher (%s) from %s", mics_group, teacher_cache_path)

    if teacher_emb is None or needs_teacher_model:
        logger.info("Training MiCS teacher on %s ...", npy_path.name)
        cache = None
        if bool(OmegaConf.select(ga, "share_pixel_sampling", default=True)):
            sampling_cache_path = _pixel_sampling_cache_path(cfg, npy_path, model_kw, mics_group)
            cache = resolve_pixel_sampling_cache(
                msi,
                model_kw,
                out_dir=run_dir / "caches" / "mics",
                share=True,
                cache_path=sampling_cache_path if sampling_cache_path.is_file() else None,
                strip_cluster_labels=False,
            )
            if cache is not None and not sampling_cache_path.is_file():
                save_pixel_sampling_cache(sampling_cache_path, cache)
        teacher = MSIParametricMiCSLMC(**model_kw)
        teacher.fit(msi, pixel_sampling_cache=cache)
        teacher_emb = np.asarray(teacher.predict_embedding(msi), dtype=np.float32)
        mics_rgb = _to_uint8_rgb(teacher.predict(msi))
        if cache_teacher and not needs_teacher_model:
            _save_teacher_cache(teacher_cache_path, teacher_emb, mics_rgb)

    assert teacher_emb is not None and mics_rgb is not None
    mics_rgb = np.asarray(mics_rgb, dtype=np.uint8)
    mics_rgb[~valid_mask] = 0
    teacher_emb = np.asarray(teacher_emb, dtype=np.float32)
    teacher_emb[~valid_mask] = 0.0

    logger.info(
        "Frozen MiCS -> NMF %d concepts, learn logits (%s/%s) ...",
        concepts_cfg.n_concepts,
        concepts_cfg.fit_method,
        concepts_cfg.weight_mode,
    )
    explainer = FrozenConceptExplainer.from_mics(
        msi,
        teacher_emb,
        valid_mask,
        concepts_cfg,
        mics_model=teacher.model if teacher is not None else None,
    )
    if teacher is not None:
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    recon_emb = explainer.predict_embedding()
    recon_rgb = explainer.predict_rgb(valid_mask)

    embedding_mse = _masked_mse(teacher_emb, recon_emb, valid_mask)
    pixel_mse = _masked_mse(mics_rgb, recon_rgb, valid_mask)
    logger.info(
        "Reconstruction - embedding MSE: %.6f | pixel RGB MSE: %.2f | fit: %s",
        embedding_mse,
        pixel_mse,
        explainer.fit_stats,
    )

    Image.fromarray(mics_rgb).save(run_dir / "mics_teacher.png")
    Image.fromarray(recon_rgb).save(run_dir / "mics_reconstruction.png")
    save_reconstruction_mosaic(
        mics_rgb,
        recon_rgb,
        valid_mask,
        run_dir / "reconstruction_check.png",
        gap_px=int(OmegaConf.select(ga, "mosaic_gap_px", default=2)),
    )

    np.save(run_dir / "concepts.npy", explainer.concepts)
    np.save(run_dir / "embed_w.npy", explainer.embed_w)
    np.save(run_dir / "alpha_hwk.npy", explainer.alpha_hwk)
    diagnostics_summary = _save_nmf_deviation_diagnostics(
        msi,
        explainer,
        valid_mask,
        run_dir,
        active_threshold=float(OmegaConf.select(cfg, "concepts.active_threshold", default=1e-4)),
        sparse_top_k=int(OmegaConf.select(cfg, "concepts.diagnostic_sparse_top_k", default=4)),
        cosine_percentile_low=float(
            OmegaConf.select(cfg, "concepts.diagnostic_cosine_percentile_low", default=15.0)
        ),
        cosine_percentile_high=float(
            OmegaConf.select(cfg, "concepts.diagnostic_cosine_percentile_high", default=90.0)
        ),
        cosine_gamma=float(OmegaConf.select(cfg, "concepts.diagnostic_cosine_gamma", default=2.0)),
    )

    concept_rgbs = concept_colors_precomputed(explainer, valid_mask=valid_mask)
    palette_path = save_concept_color_palette(
        concept_rgbs,
        run_dir / "concept_colors.png",
        cols=int(OmegaConf.select(cfg, "explain.concept_color_cols", default=8)),
    )
    np.save(run_dir / "concept_colors.npy", concept_rgbs)

    viz_summary: dict[str, Any] | None = None
    overview_summary: dict[str, Any] | None = None
    vz = OmegaConf.select(cfg, "viz") or {}
    if bool(OmegaConf.select(cfg, "viz.enabled", default=True)):
        cv = OmegaConf.select(cfg, "concepts_viz") or {}
        viz_summary = save_explanation_visualizations(
            mics_rgb,
            recon_rgb,
            explainer.alpha_hwk,
            concept_rgbs,
            valid_mask,
            run_dir,
            teacher_emb=teacher_emb,
            n_concepts=int(concepts_cfg.n_concepts),
            top_k=int(concepts_cfg.top_k),
            pixel_rgb_mse=float(pixel_mse),
            embedding_mse=float(embedding_mse),
            n_example_pixels=int(OmegaConf.select(vz, "n_example_pixels", default=5)),
            dpi=int(OmegaConf.select(vz, "dpi", default=200)),
            seed=seed,
            concept_node_style=str(OmegaConf.select(vz, "concept_node_style", default="heatmap")),
            concept_thumb_px=int(OmegaConf.select(vz, "concept_thumb_px", default=112)),
            grid_offset_px=float(OmegaConf.select(vz, "grid_offset_px", default=170.0)),
            grid_spacing_px=(
                None
                if OmegaConf.select(vz, "grid_spacing_px", default=None) is None
                else float(OmegaConf.select(vz, "grid_spacing_px"))
            ),
            map_fig_width=float(OmegaConf.select(vz, "map_fig_width", default=24.0)),
            map_fig_height=float(OmegaConf.select(vz, "map_fig_height", default=22.0)),
            map_marker_fontsize=float(OmegaConf.select(vz, "map_marker_fontsize", default=18.0)),
            map_marker_size=float(OmegaConf.select(vz, "map_marker_size", default=14.0)),
            equalize_map_display=bool(OmegaConf.select(vz, "equalize_map_display", default=True)),
            equalize_thumb_activation=bool(
                OmegaConf.select(vz, "equalize_thumb_activation", default=False)
            ),
            save_both_display_variants=bool(
                OmegaConf.select(vz, "save_both_display_variants", default=False)
            ),
            example_pixel_spatial_weight=float(
                OmegaConf.select(vz, "example_pixel_spatial_weight", default=2.5)
            ),
            example_pixel_interior_min_px=float(
                OmegaConf.select(vz, "example_pixel_interior_min_px", default=10.0)
            ),
            example_pixel_interior_weight=float(
                OmegaConf.select(vz, "example_pixel_interior_weight", default=3.0)
            ),
            activation_percentile=float(
                OmegaConf.select(vz, "activation_percentile", default=OmegaConf.select(cv, "activation_percentile", default=98.0))
            ),
            activation_gamma=float(
                OmegaConf.select(vz, "activation_gamma", default=OmegaConf.select(cv, "activation_gamma", default=0.85))
            ),
            activation_colormap=str(
                OmegaConf.select(vz, "activation_colormap", default=OmegaConf.select(cv, "activation_colormap", default="magma"))
            ),
            concept_rank_mode=str(OmegaConf.select(vz, "concept_rank_mode", default="relative")),
            min_concept_weight=float(
                OmegaConf.select(vz, "min_concept_weight", default=OmegaConf.select(cfg, "explain.min_concept_weight", default=0.04))
            ),
            top_terms=int(OmegaConf.select(vz, "top_terms", default=OmegaConf.select(cfg, "explain.top_concepts", default=4))),
            export_interpretability=bool(
                OmegaConf.select(vz, "export_interpretability", default=True)
            ),
            interpretability_subdir=str(
                OmegaConf.select(vz, "interpretability_subdir", default="interpretability")
            ),
            export_dpi=int(OmegaConf.select(vz, "export_dpi", default=300)),
            export_thumb_px=int(OmegaConf.select(vz, "export_thumb_px", default=480)),
        )
        logger.info(
            "Sparsity: %.2f active concepts/pixel (cap %d), mean top weight %.0f%%",
            viz_summary["sparsity"].get("mean_nnz", float("nan")),
            int(concepts_cfg.top_k),
            100.0 * viz_summary["sparsity"].get("mean_top1_weight", 0.0),
        )

        if bool(OmegaConf.select(vz, "overview_gallery.enabled", default=True)):
            og = OmegaConf.select(vz, "overview_gallery") or {}
            map_variants = viz_summary.get("concept_arrows_on_map_variants") or {}
            concept_path = (
                map_variants.get("equalized")
                or map_variants.get("primary")
                or viz_summary.get("concept_arrows_on_map_png")
            )
            if concept_path is None:
                logger.warning("Overview gallery skipped: concept map path missing.")
            else:
                sparse_top_k = int(OmegaConf.select(cfg, "concepts.diagnostic_sparse_top_k", default=4))
                use_sparse = str(OmegaConf.select(og, "cosine_map", default="sparse")).strip().lower() != "dense"
                cosine_npy = (
                    run_dir / f"nmf_projection_sparse_top{sparse_top_k}_cosine_distance.npy"
                    if use_sparse
                    else run_dir / "nmf_projection_cosine_distance.npy"
                )
                cosine_vals = np.load(cosine_npy)
                cosine_rgb = _cosine_stretched_rgb(
                    cosine_vals,
                    valid_mask,
                    percentile_low=float(
                        OmegaConf.select(cfg, "concepts.diagnostic_cosine_percentile_low", default=15.0)
                    ),
                    percentile_high=float(
                        OmegaConf.select(cfg, "concepts.diagnostic_cosine_percentile_high", default=90.0)
                    ),
                    gamma=float(OmegaConf.select(cfg, "concepts.diagnostic_cosine_gamma", default=2.0)),
                )
                rank_config = str(
                    OmegaConf.select(
                        og,
                        "rank_config",
                        default=str(_REPO / "scripts" / "configs" / "hd_methods_gallery.yaml"),
                    )
                )
                solace_cache = run_dir / "solace_reference.png"
                solace_rgb = _compute_solace_rgb(
                    msi,
                    valid_mask,
                    rank_config_path=rank_config,
                    edge_colormap=str(OmegaConf.select(og, "edge_colormap", default="PuBu")),
                    cache_path=solace_cache,
                )
                overview_path = save_interpretability_overview_gallery(
                    mics_rgb,
                    valid_mask,
                    run_dir / "interpretability_overview_gallery.png",
                    concept_map_path=Path(concept_path),
                    solace_rgb=solace_rgb,
                    cosine_stretched_rgb=cosine_rgb,
                    equalize_mics=bool(OmegaConf.select(vz, "equalize_map_display", default=True)),
                    gap_px=int(OmegaConf.select(og, "gap_px", default=0)),
                )
                overview_summary = {
                    "png": overview_path,
                    "concept_map_png": str(concept_path),
                    "solace_reference_png": str(solace_cache),
                    "cosine_map": "sparse" if use_sparse else "dense",
                    "cosine_npy": str(cosine_npy),
                }
                logger.info("Overview gallery: %s", overview_path)

    alpha_hwk = explainer.alpha_hwk
    ex = OmegaConf.select(cfg, "explain") or {}
    regions = select_homogeneous_regions(
        teacher_emb,
        mics_rgb,
        valid_mask,
        n_regions=int(OmegaConf.select(ex, "n_regions", default=6)),
        n_clusters=int(OmegaConf.select(ex, "n_clusters", default=12)),
        min_pixels=int(OmegaConf.select(ex, "min_pixels", default=600)),
        max_emb_std=float(OmegaConf.select(ex, "max_emb_std", default=0.35)),
        crop_size=int(OmegaConf.select(ex, "crop_size", default=72)),
        seed=seed,
    )
    attach_region_alphas(regions, alpha_hwk, valid_mask)

    explanations: list[dict[str, Any]] = []
    for reg in regions:
        color_name = _rgb_color_name(reg.mean_rgb)
        expl = build_color_explanation(
            reg.mean_alpha,
            concept_rgbs,
            top_n=int(OmegaConf.select(ex, "top_concepts", default=5)),
            min_weight=float(OmegaConf.select(ex, "min_concept_weight", default=0.04)),
            color_label=f"MiCS {color_name}",
        )
        explanations.append(expl)
        logger.info("%s (%d px): %s", reg.label, reg.n_pixels, expl["formula"])

    overview = build_region_overview(mics_rgb, regions)
    region_summary = save_region_explanations(
        regions,
        explanations,
        mics_rgb,
        recon_rgb,
        concept_rgbs,
        run_dir,
        overview_rgb=overview,
    )

    concepts_viz_summary: dict[str, Any] | None = None
    if bool(OmegaConf.select(cfg, "concepts_viz.enabled", default=True)):
        cv = OmegaConf.select(cfg, "concepts_viz") or {}
        gallery_student = explainer.as_gallery_student()
        concepts_viz_summary = save_concept_gallery(
            gallery_student,
            msi,
            valid_mask=valid_mask,
            out_dir=run_dir,
            npy_path=npy_path,
            alpha_hwk=alpha_hwk,
            max_concepts=int(OmegaConf.select(cv, "max_concepts", default=16)),
            spectrum_top_peaks=int(OmegaConf.select(cv, "spectrum_top_peaks", default=25)),
            dpi=int(OmegaConf.select(cv, "dpi", default=200)),
            cols=int(OmegaConf.select(cv, "cols", default=4)),
            gap_px=int(OmegaConf.select(cv, "gap_px", default=1)),
            activation_percentile=float(OmegaConf.select(cv, "activation_percentile", default=98.0)),
            activation_gamma=float(OmegaConf.select(cv, "activation_gamma", default=0.85)),
            activation_colormap=str(OmegaConf.select(cv, "activation_colormap", default="assigned")),
            dedupe=bool(OmegaConf.select(cv, "dedupe", default=True)),
            dedupe_min_color_dist=float(OmegaConf.select(cv, "dedupe_min_color_dist", default=35.0)),
            dedupe_max_spatial_corr=float(OmegaConf.select(cv, "dedupe_max_spatial_corr", default=0.88)),
            dedupe_max_mz_cosine=float(OmegaConf.select(cv, "dedupe_max_mz_cosine", default=0.92)),
        )

    manifest: dict[str, Any] = {
        "npy_path": str(npy_path),
        "shape": list(msi.shape),
        "mics_group": mics_group,
        "method": f"frozen_mics_nmf_{concepts_cfg.fit_method}",
        "reconstruction": {
            "embedding_mse": embedding_mse,
            "pixel_rgb_mse": pixel_mse,
            "teacher_png": str(run_dir / "mics_teacher.png"),
            "reconstruction_png": str(run_dir / "mics_reconstruction.png"),
            "check_mosaic_png": str(run_dir / "reconstruction_check.png"),
        },
        "concept_fit": explainer.fit_stats,
        "concept_colors_png": palette_path,
        "concept_colors_npy": str(run_dir / "concept_colors.npy"),
        "concepts_npy": str(run_dir / "concepts.npy"),
        "embed_w_npy": str(run_dir / "embed_w.npy"),
        "alpha_hwk_npy": str(run_dir / "alpha_hwk.npy"),
        "diagnostic_maps": diagnostics_summary,
        "concepts_mz": explainer.concept_mz_summary(
            top_peaks=int(OmegaConf.select(cfg, "concepts.top_peaks", default=5))
        ),
        "region_explanations": region_summary,
        "mics_kwargs": {k: model_kw[k] for k in sorted(model_kw) if k != "verbose"},
        "concepts_config": asdict(concepts_cfg),
    }
    if concepts_viz_summary is not None:
        manifest["concepts_viz"] = concepts_viz_summary
    if viz_summary is not None:
        manifest["explanation_viz"] = viz_summary
    if overview_summary is not None:
        manifest["overview_gallery"] = overview_summary
    (run_dir / "explain_mics_concepts_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
