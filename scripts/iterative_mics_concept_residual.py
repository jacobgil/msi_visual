#!/usr/bin/env python3
"""
Iteratively peel MiCS-explained m/z signatures from an MSI cube.

Each stage:
  1. Train MiCS on the current residual spectra.
  2. Fit sparse per-pixel concept weights to the MiCS embedding.
  3. Convert weights + component spectra into a per-pixel m/z signature.
  4. Suppress that signature and continue with the residual spectra.

The output gallery is exploratory: it shows what structure remains after
successively removing the component-level signal that explains each stage.
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from dataclasses import asdict, replace
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

from benchmark_parametric_methods import _seed_everything  # noqa: E402
from debug_edge_maps import _load_msi, _resolve_normalization_mode, _to_uint8_rgb  # noqa: E402
from explain_mics_concepts import (  # noqa: E402
    _compute_solace_rgb,
    _concepts_cfg,
    _cosine_stretched_rgb,
    _save_nmf_deviation_diagnostics,
)
from gallery_mics_concept_distill import _masked_mse, _mics_kwargs  # noqa: E402
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402
from msi_visual.mics_concept_explain import FrozenConceptExplainer, concept_colors_precomputed  # noqa: E402
from msi_visual.mics_concept_explain_viz import (  # noqa: E402
    _equalize_rgb_display,
    save_explanation_visualizations,
    save_interpretability_overview_gallery,
)
from msi_visual.mics_concept_viz import save_concept_gallery  # noqa: E402
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402

logger = logging.getLogger(__name__)


def _abs_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _tic_normalize_valid(msi: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(msi, dtype=np.float32).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    sums = out.sum(axis=-1, keepdims=True)
    np.divide(out, sums + 1e-8, out=out)
    out[~mask] = 0.0
    return out


def _safe_fraction(numer: float, denom: float) -> float:
    return float(numer / denom) if abs(float(denom)) > 1e-12 else 0.0


def _stage_mics_kwargs(base: dict[str, Any], seed: int) -> dict[str, Any]:
    kw = dict(base)
    kw["random_state"] = int(seed)
    return kw


def _fit_mics_stage(
    msi: np.ndarray,
    model_kw: dict[str, Any],
    *,
    pixel_sampling_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, MSIParametricMiCSLMC]:
    teacher = MSIParametricMiCSLMC(**model_kw)
    teacher.fit(msi, pixel_sampling_cache=pixel_sampling_cache)
    emb = np.asarray(teacher.predict_embedding(msi), dtype=np.float32)
    rgb = _to_uint8_rgb(teacher.predict(msi))
    return emb, rgb, teacher


def _release_teacher(teacher: MSIParametricMiCSLMC | None) -> None:
    if teacher is None:
        return
    try:
        teacher.release_resources()
    except Exception:
        pass
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _run_stage_interpretability(
    cfg: DictConfig,
    stage_dir: Path,
    *,
    residual_msi: np.ndarray,
    mics_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    explainer: FrozenConceptExplainer,
    teacher_emb: np.ndarray,
    valid_mask: np.ndarray,
    npy_path: Path,
    stage_seed: int,
    concepts_cfg: Any,
    pixel_mse: float,
    embedding_mse: float,
) -> dict[str, Any]:
    """Diagnostics, concept maps, and 4-panel overview gallery for one stage."""
    vz = OmegaConf.select(cfg, "viz") or {}
    cv = OmegaConf.select(cfg, "concepts_viz") or {}
    concept_rgbs = concept_colors_precomputed(explainer, valid_mask=valid_mask)
    np.save(stage_dir / "concept_colors.npy", concept_rgbs)

    diagnostics_summary = _save_nmf_deviation_diagnostics(
        residual_msi,
        explainer,
        valid_mask,
        stage_dir,
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

    viz_summary = save_explanation_visualizations(
        mics_rgb,
        recon_rgb,
        explainer.alpha_hwk,
        concept_rgbs,
        valid_mask,
        stage_dir,
        teacher_emb=teacher_emb,
        n_concepts=int(concepts_cfg.n_concepts),
        top_k=int(concepts_cfg.top_k),
        pixel_rgb_mse=float(pixel_mse),
        embedding_mse=float(embedding_mse),
        n_example_pixels=int(OmegaConf.select(vz, "n_example_pixels", default=5)),
        dpi=int(OmegaConf.select(vz, "dpi", default=200)),
        seed=int(stage_seed),
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
        equalize_thumb_activation=bool(OmegaConf.select(vz, "equalize_thumb_activation", default=False)),
        save_both_display_variants=bool(OmegaConf.select(vz, "save_both_display_variants", default=False)),
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
        top_terms=int(OmegaConf.select(vz, "top_terms", default=4)),
        export_interpretability=bool(OmegaConf.select(vz, "export_interpretability", default=True)),
        interpretability_subdir=str(
            OmegaConf.select(vz, "interpretability_subdir", default="interpretability")
        ),
        export_dpi=int(OmegaConf.select(vz, "export_dpi", default=300)),
        export_thumb_px=int(OmegaConf.select(vz, "export_thumb_px", default=480)),
    )

    overview_summary: dict[str, Any] | None = None
    if bool(OmegaConf.select(vz, "overview_gallery.enabled", default=True)):
        og = OmegaConf.select(vz, "overview_gallery") or {}
        map_variants = viz_summary.get("concept_arrows_on_map_variants") or {}
        concept_path = (
            map_variants.get("equalized")
            or map_variants.get("primary")
            or viz_summary.get("concept_arrows_on_map_png")
        )
        if concept_path is None:
            logger.warning("Stage overview gallery skipped: concept map path missing.")
        else:
            sparse_top_k = int(OmegaConf.select(cfg, "concepts.diagnostic_sparse_top_k", default=4))
            use_sparse = str(OmegaConf.select(og, "cosine_map", default="sparse")).strip().lower() != "dense"
            cosine_npy = (
                stage_dir / f"nmf_projection_sparse_top{sparse_top_k}_cosine_distance.npy"
                if use_sparse
                else stage_dir / "nmf_projection_cosine_distance.npy"
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
            solace_cache = stage_dir / "solace_reference.png"
            solace_rgb = _compute_solace_rgb(
                residual_msi,
                valid_mask,
                rank_config_path=rank_config,
                edge_colormap=str(OmegaConf.select(og, "edge_colormap", default="PuBu")),
                cache_path=solace_cache,
            )
            overview_path = save_interpretability_overview_gallery(
                mics_rgb,
                valid_mask,
                stage_dir / "interpretability_overview_gallery.png",
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
            }

    if bool(OmegaConf.select(cv, "enabled", default=True)):
        save_concept_gallery(
            explainer.as_gallery_student(),
            residual_msi,
            valid_mask=valid_mask,
            out_dir=stage_dir,
            npy_path=npy_path,
            alpha_hwk=explainer.alpha_hwk,
            max_concepts=int(OmegaConf.select(cv, "max_concepts", default=12)),
            spectrum_top_peaks=int(OmegaConf.select(cv, "spectrum_top_peaks", default=25)),
            dpi=int(OmegaConf.select(cv, "dpi", default=180)),
            cols=int(OmegaConf.select(cv, "cols", default=4)),
            gap_px=int(OmegaConf.select(cv, "gap_px", default=1)),
            activation_percentile=float(OmegaConf.select(cv, "activation_percentile", default=98.0)),
            activation_gamma=float(OmegaConf.select(cv, "activation_gamma", default=0.85)),
            activation_colormap=str(OmegaConf.select(cv, "activation_colormap", default="magma")),
            dedupe=bool(OmegaConf.select(cv, "dedupe", default=True)),
            dedupe_min_color_dist=float(OmegaConf.select(cv, "dedupe_min_color_dist", default=35.0)),
            dedupe_max_spatial_corr=float(OmegaConf.select(cv, "dedupe_max_spatial_corr", default=0.88)),
            dedupe_max_mz_cosine=float(OmegaConf.select(cv, "dedupe_max_mz_cosine", default=0.92)),
        )

    return {
        "diagnostics": diagnostics_summary,
        "explanation_viz": viz_summary,
        "overview_gallery": overview_summary,
    }


def _load_mz_axis(path_raw: Any, n_mz: int) -> np.ndarray:
    if path_raw is None or str(path_raw).strip().lower() in {"", "none", "null", "~"}:
        return np.arange(n_mz, dtype=np.float64)
    path = _abs_path(str(path_raw))
    arr = np.loadtxt(path, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size != n_mz:
        raise ValueError(f"mz_axis_path has {arr.size} entries, expected {n_mz}")
    return arr


def _concept_signature(alpha_hwk: np.ndarray, concepts: np.ndarray) -> np.ndarray:
    return np.einsum("hwk,km->hwm", alpha_hwk, concepts, optimize=True).astype(np.float32)


def _scaled_signature(
    msi: np.ndarray,
    signature: np.ndarray,
    valid_mask: np.ndarray,
    *,
    scale_mode: str,
    max_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(scale_mode).strip().lower()
    sig = np.asarray(signature, dtype=np.float32)
    x = np.asarray(msi, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    scale = np.zeros(x.shape[:2], dtype=np.float32)

    if mode in {"none", "off", "false"}:
        scale[mask] = 1.0
    elif mode in {"least_squares", "lsq", "projection"}:
        numer = np.sum(x * sig, axis=-1)
        denom = np.sum(sig * sig, axis=-1)
        np.divide(numer, denom + 1e-8, out=scale)
        scale = np.maximum(scale, 0.0)
    elif mode in {"tic", "sum"}:
        numer = np.sum(x, axis=-1)
        denom = np.sum(sig, axis=-1)
        np.divide(numer, denom + 1e-8, out=scale)
        scale = np.maximum(scale, 0.0)
    else:
        raise ValueError("residual.scale_mode must be one of: least_squares, tic, none")

    if np.isfinite(max_scale) and float(max_scale) > 0:
        scale = np.minimum(scale, float(max_scale))
    scale[~mask] = 0.0
    return (sig * scale[..., None]).astype(np.float32), scale.astype(np.float32)


def _suppress_signature(
    msi: np.ndarray,
    scaled_signature: np.ndarray,
    valid_mask: np.ndarray,
    *,
    strength: float,
    mode: str,
    renormalize_tic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(msi, dtype=np.float32)
    sig = np.asarray(scaled_signature, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    amount = float(np.clip(strength, 0.0, 1.0))
    method = str(mode).strip().lower()

    if method in {"subtract", "residual", "sub"}:
        removed = np.minimum(x, amount * sig).astype(np.float32)
        residual = np.maximum(x - removed, 0.0).astype(np.float32)
    elif method in {"attenuate", "multiply", "gate"}:
        max_sig = np.max(sig, axis=-1, keepdims=True)
        gate = np.divide(sig, max_sig + 1e-8)
        gate = np.clip(gate, 0.0, 1.0)
        residual = (x * (1.0 - amount * gate)).astype(np.float32)
        removed = np.maximum(x - residual, 0.0).astype(np.float32)
    else:
        raise ValueError("residual.suppression_mode must be one of: subtract, attenuate")

    residual[~mask] = 0.0
    removed[~mask] = 0.0
    if bool(renormalize_tic):
        residual = _tic_normalize_valid(residual, mask)
    return residual, removed


def _save_heatmap(values_hw: np.ndarray, out_path: Path, *, title: str, dpi: int) -> None:
    arr = np.asarray(values_hw, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=int(dpi))
    im = ax.imshow(arr, cmap="magma")
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _gallery_panel(
    mics_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    panel_source: str,
    overview_path: Path | None,
) -> np.ndarray:
    source = str(panel_source).strip().lower()
    if source == "overview" and overview_path is not None and overview_path.is_file():
        return _load_rgb_image(overview_path)
    if source in {"mics_equalized", "equalized", "overview"}:
        return _equalize_rgb_display(mics_rgb, valid_mask)
    return np.asarray(mics_rgb, dtype=np.uint8)


def _save_gallery(
    panels: list[np.ndarray],
    titles: list[str],
    out_path: Path,
    *,
    ncols: int,
    dpi: int,
    gap_px: int = 0,
    show_titles: bool = False,
) -> None:
    if not panels:
        return
    if not show_titles:
        ncols = max(1, int(ncols))
        gap = max(0, int(gap_px))
        first = np.asarray(panels[0], dtype=np.uint8)
        h, w = first.shape[:2]
        nrows = int(np.ceil(len(panels) / ncols))
        canvas_h = nrows * h + max(0, nrows - 1) * gap
        canvas_w = ncols * w + max(0, ncols - 1) * gap
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for i, panel in enumerate(panels):
            row, col = divmod(i, ncols)
            y0 = row * (h + gap)
            x0 = col * (w + gap)
            rgb = np.asarray(panel, dtype=np.uint8)
            if rgb.shape[:2] != (h, w):
                raise ValueError(f"Gallery panel {i} has shape {rgb.shape[:2]}, expected {(h, w)}")
            canvas[y0 : y0 + h, x0 : x0 + w] = rgb[:, :, :3]
        Image.fromarray(canvas).save(out_path)
        return

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 4.3 * nrows),
        dpi=int(dpi),
        squeeze=False,
    )
    fig.patch.set_facecolor("black")
    for ax in axes.ravel():
        ax.set_facecolor("black")
        ax.set_axis_off()
    for ax, rgb, title in zip(axes.ravel(), panels, titles):
        ax.imshow(np.asarray(rgb, dtype=np.uint8))
        ax.set_title(title, fontsize=10, color="white")
        ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _top_removed_mz(
    removed: np.ndarray,
    valid_mask: np.ndarray,
    mz_axis: np.ndarray,
    *,
    top_n: int,
) -> list[dict[str, float | int]]:
    mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(mask):
        return []
    mean_removed = np.asarray(removed, dtype=np.float64)[mask].mean(axis=0)
    n = max(1, min(int(top_n), int(mean_removed.size)))
    idx = np.argsort(-mean_removed)[:n]
    return [
        {
            "mz_index": int(i),
            "mz": float(mz_axis[int(i)]),
            "mean_removed": float(mean_removed[int(i)]),
        }
        for i in idx
    ]


@hydra.main(version_base=None, config_path="configs", config_name="iterative_mics_concept_residual")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    seed = int(cfg.seed)
    _seed_everything(seed)

    ga = cfg.gallery
    npy_path = _abs_path(str(ga.npy_path))
    if not npy_path.is_file():
        raise FileNotFoundError(f"gallery.npy_path not found: {npy_path}")

    norm_mode = _resolve_normalization_mode(str(cfg.data.normalization))
    msi0 = _load_msi(npy_path, transpose_msi=bool(cfg.data.transpose_msi), normalization=norm_mode)
    valid_mask = msi0.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = run_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    model_kw_base = _mics_kwargs(cfg)
    concepts_cfg = _concepts_cfg(cfg, model_kw_base)
    iterations = max(1, int(OmegaConf.select(cfg, "residual.iterations", default=4)))
    strength = float(OmegaConf.select(cfg, "residual.strength", default=0.75))
    scale_mode = str(OmegaConf.select(cfg, "residual.scale_mode", default="least_squares"))
    suppression_mode = str(OmegaConf.select(cfg, "residual.suppression_mode", default="subtract"))
    renormalize_tic = bool(OmegaConf.select(cfg, "residual.renormalize_tic", default=True))
    max_scale = float(OmegaConf.select(cfg, "residual.max_scale", default=10.0))
    save_arrays = bool(OmegaConf.select(cfg, "residual.save_arrays", default=False))
    save_signatures = bool(OmegaConf.select(cfg, "residual.save_signatures", default=False))
    top_removed_mz = int(OmegaConf.select(cfg, "residual.top_removed_mz", default=25))
    mz_axis = _load_mz_axis(OmegaConf.select(cfg, "gallery.mz_axis_path", default=None), msi0.shape[-1])

    share_cache = bool(OmegaConf.select(ga, "share_pixel_sampling", default=True))
    strip_cluster_labels = bool(OmegaConf.select(ga, "strip_sampling_cluster_labels", default=True))
    sampling_cache = resolve_pixel_sampling_cache(
        msi0,
        model_kw_base,
        out_dir=run_dir / "caches" / "mics",
        share=share_cache,
        cache_path=None,
        strip_cluster_labels=strip_cluster_labels,
    )

    residual_msi = np.asarray(msi0, dtype=np.float32).copy()
    residual_msi[~valid_mask] = 0.0
    initial_tic = float(np.sum(msi0[valid_mask], dtype=np.float64))
    panels: list[np.ndarray] = []
    mics_panels: list[np.ndarray] = []
    titles: list[str] = []
    stage_summaries: list[dict[str, Any]] = []
    panel_source = str(OmegaConf.select(cfg, "gallery.panel_source", default="overview"))
    viz_enabled = bool(OmegaConf.select(cfg, "viz.enabled", default=True))

    for stage in range(iterations):
        stage_dir = stages_dir / f"stage_{stage:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_seed = seed + stage
        model_kw = _stage_mics_kwargs(model_kw_base, stage_seed)
        logger.info("Stage %d/%d: fitting MiCS on current residual ...", stage + 1, iterations)
        teacher: MSIParametricMiCSLMC | None = None
        interpretability_summary: dict[str, Any] | None = None
        overview_path = stage_dir / "interpretability_overview_gallery.png"
        try:
            teacher_emb, mics_rgb, teacher = _fit_mics_stage(
                residual_msi,
                model_kw,
                pixel_sampling_cache=sampling_cache,
            )
            teacher_emb[~valid_mask] = 0.0
            mics_rgb[~valid_mask] = 0
            Image.fromarray(mics_rgb).save(stage_dir / "mics_residual_rgb.png")
            mics_panels.append(np.asarray(mics_rgb, dtype=np.uint8))

            residual_total = float(np.sum(residual_msi[valid_mask], dtype=np.float64))
            title = f"stage {stage}: residual {100.0 * _safe_fraction(residual_total, initial_tic):.1f}%"
            titles.append(title)

            logger.info("Stage %d/%d: fitting sparse concept explanation (%s) ...", stage + 1, iterations, concepts_cfg.fit_method)
            stage_concepts_cfg = replace(concepts_cfg, seed=int(stage_seed))
            explainer = FrozenConceptExplainer.from_mics(
                residual_msi,
                teacher_emb,
                valid_mask,
                stage_concepts_cfg,
                mics_model=teacher.model,
            )
            recon_emb = explainer.predict_embedding()
            recon_rgb = explainer.predict_rgb(valid_mask)
            Image.fromarray(recon_rgb).save(stage_dir / "mics_concept_reconstruction.png")
            embedding_mse = _masked_mse(teacher_emb, recon_emb, valid_mask)
            pixel_mse = _masked_mse(mics_rgb, recon_rgb, valid_mask)
            logger.info(
                "Stage %d reconstruction - embedding MSE: %.6f | pixel RGB MSE: %.2f",
                stage,
                embedding_mse,
                pixel_mse,
            )

            interpretability_summary = None
            if viz_enabled:
                interpretability_summary = _run_stage_interpretability(
                    cfg,
                    stage_dir,
                    residual_msi=residual_msi,
                    mics_rgb=mics_rgb,
                    recon_rgb=recon_rgb,
                    explainer=explainer,
                    teacher_emb=teacher_emb,
                    valid_mask=valid_mask,
                    npy_path=npy_path,
                    stage_seed=int(stage_seed),
                    concepts_cfg=stage_concepts_cfg,
                    pixel_mse=float(pixel_mse),
                    embedding_mse=float(embedding_mse),
                )
        finally:
            _release_teacher(teacher)

        panels.append(
            _gallery_panel(
                mics_rgb,
                valid_mask,
                panel_source=panel_source,
                overview_path=overview_path if viz_enabled else None,
            )
        )

        signature = _concept_signature(explainer.alpha_hwk, explainer.concepts)
        scaled_signature, scale_hw = _scaled_signature(
            residual_msi,
            signature,
            valid_mask,
            scale_mode=scale_mode,
            max_scale=max_scale,
        )
        next_residual, removed = _suppress_signature(
            residual_msi,
            scaled_signature,
            valid_mask,
            strength=strength,
            mode=suppression_mode,
            renormalize_tic=renormalize_tic,
        )

        removed_intensity = removed.sum(axis=-1)
        residual_intensity = next_residual.sum(axis=-1)
        _save_heatmap(
            removed_intensity,
            stage_dir / "removed_intensity.png",
            title=f"stage {stage} removed intensity",
            dpi=int(OmegaConf.select(cfg, "gallery.dpi", default=200)),
        )
        _save_heatmap(
            residual_intensity,
            stage_dir / "next_residual_tic.png",
            title=f"stage {stage} next residual TIC",
            dpi=int(OmegaConf.select(cfg, "gallery.dpi", default=200)),
        )

        mean_signature = np.asarray(removed, dtype=np.float64)[valid_mask].mean(axis=0)
        np.save(stage_dir / "mean_removed_signature.npy", mean_signature.astype(np.float32))
        if save_arrays:
            np.save(stage_dir / "concepts.npy", explainer.concepts.astype(np.float32))
            np.save(stage_dir / "alpha_hwk.npy", explainer.alpha_hwk.astype(np.float32))
            np.save(stage_dir / "scale_hw.npy", scale_hw.astype(np.float32))
            np.save(stage_dir / "residual_input.npy", residual_msi.astype(np.float32))
            np.save(stage_dir / "residual_next.npy", next_residual.astype(np.float32))
        if save_signatures:
            np.save(stage_dir / "scaled_signature_hw_mz.npy", scaled_signature.astype(np.float32))
            np.save(stage_dir / "removed_hw_mz.npy", removed.astype(np.float32))

        top_mz = _top_removed_mz(removed, valid_mask, mz_axis, top_n=top_removed_mz)
        (stage_dir / "top_removed_mz.json").write_text(json.dumps(top_mz, indent=2), encoding="utf-8")

        removed_total = float(np.sum(removed[valid_mask], dtype=np.float64))
        next_total = float(np.sum(next_residual[valid_mask], dtype=np.float64))
        stage_summary = {
            "stage": int(stage),
            "stage_seed": int(stage_seed),
            "residual_input_total": residual_total,
            "removed_total": removed_total,
            "residual_next_total": next_total,
            "removed_fraction_of_stage_input": _safe_fraction(removed_total, residual_total),
            "residual_fraction_of_initial": _safe_fraction(next_total, initial_tic),
            "alpha_mean_active": float(np.mean(np.sum(explainer.alpha_hwk[valid_mask] > 1e-6, axis=-1))),
            "scale_mean": float(np.mean(scale_hw[valid_mask])),
            "scale_p95": float(np.percentile(scale_hw[valid_mask], 95)),
            "concept_fit": explainer.fit_stats,
            "reconstruction": {
                "embedding_mse": float(embedding_mse),
                "pixel_rgb_mse": float(pixel_mse),
            },
            "top_removed_mz": top_mz,
        }
        if interpretability_summary is not None:
            stage_summary["interpretability"] = interpretability_summary
        stage_summaries.append(stage_summary)
        (stage_dir / "stage_summary.json").write_text(json.dumps(stage_summary, indent=2), encoding="utf-8")

        logger.info(
            "Stage %d removed %.2f%% of current residual; next residual %.2f%% of initial.",
            stage,
            100.0 * stage_summary["removed_fraction_of_stage_input"],
            100.0 * stage_summary["residual_fraction_of_initial"],
        )
        residual_msi = next_residual

    _save_gallery(
        panels,
        titles,
        run_dir / "iterative_residual_gallery.png",
        ncols=int(OmegaConf.select(cfg, "gallery.ncols", default=4)),
        dpi=int(OmegaConf.select(cfg, "gallery.dpi", default=200)),
        gap_px=int(OmegaConf.select(cfg, "gallery.gap_px", default=0)),
        show_titles=bool(OmegaConf.select(cfg, "gallery.show_titles", default=False)),
    )
    equalized_gallery_path: Path | None = None
    if bool(OmegaConf.select(cfg, "gallery.save_equalized", default=True)):
        equalized_gallery_path = run_dir / "iterative_residual_gallery_equalized.png"
        equalized_panels = [_equalize_rgb_display(panel, valid_mask) for panel in mics_panels]
        _save_gallery(
            equalized_panels,
            titles,
            equalized_gallery_path,
            ncols=int(OmegaConf.select(cfg, "gallery.ncols", default=4)),
            dpi=int(OmegaConf.select(cfg, "gallery.dpi", default=200)),
            gap_px=int(OmegaConf.select(cfg, "gallery.gap_px", default=0)),
            show_titles=bool(OmegaConf.select(cfg, "gallery.show_titles", default=False)),
        )
    if save_arrays:
        np.save(run_dir / "final_residual.npy", residual_msi.astype(np.float32))

    manifest = {
        "npy_path": str(npy_path),
        "shape": list(msi0.shape),
        "method": "iterative_mics_concept_residual",
        "iterations": int(iterations),
        "residual": {
            "strength": float(strength),
            "scale_mode": scale_mode,
            "suppression_mode": suppression_mode,
            "renormalize_tic": bool(renormalize_tic),
            "max_scale": float(max_scale),
        },
        "mics_kwargs": {k: model_kw_base[k] for k in sorted(model_kw_base) if k != "verbose"},
        "concepts_config": asdict(concepts_cfg),
        "gallery_panel_source": panel_source,
        "gallery_png": str(run_dir / "iterative_residual_gallery.png"),
        "gallery_equalized_png": str(equalized_gallery_path) if equalized_gallery_path is not None else None,
        "stages": stage_summaries,
    }
    (run_dir / "iterative_mics_concept_residual_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
