#!/usr/bin/env python3
"""
Train MiCS on one slide, distill to sparse m/z concepts + router + linear RGB,
and write a 2-panel gallery: MiCS | distilled MiCS.

Run:
  python scripts/gallery_mics_concept_distill.py --config-name=gallery_mics_concept_distill
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import sys
from pathlib import Path
from dataclasses import asdict
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

from benchmark_parametric_methods import (  # noqa: E402
    _resolve_mics_parameter_groups,
    _seed_everything,
)
from debug_edge_maps import _load_msi, _resolve_normalization_mode, _to_uint8_rgb  # noqa: E402
from gallery_mics_sampling_cache import (  # noqa: E402
    load_pixel_sampling_cache,
    resolve_pixel_sampling_cache,
    save_pixel_sampling_cache,
)
from msi_visual.mics_concept_distill import (  # noqa: E402
    ConceptDistillConfig,
    SparseConceptDistiller,
    embedding_to_display_rgb,
)
from msi_visual.mics_concept_viz import save_concept_gallery  # noqa: E402

from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402

logger = logging.getLogger(__name__)


def _abs_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _mics_group_overrides(cfg: DictConfig) -> dict[str, Any]:
    group_name = str(OmegaConf.select(cfg, "mics.group", default=""))
    if not group_name:
        return {}
    groups_path = OmegaConf.select(cfg, "mics.parameter_groups_config")
    if not groups_path:
        raise ValueError("mics.group set but mics.parameter_groups_config is missing")
    groups_cfg = OmegaConf.load(str(_abs_path(str(groups_path))))
    stub = OmegaConf.create({"mics_lmc": groups_cfg.get("mics_lmc", groups_cfg)})
    for name, overrides in _resolve_mics_parameter_groups(stub):
        if name == group_name:
            return dict(overrides)
    raise ValueError(f"Unknown mics.group={group_name!r}")


def _mics_kwargs(cfg: DictConfig) -> dict[str, Any]:
    defaults = OmegaConf.load(str(_abs_path(str(cfg.mics.defaults_yaml))))
    model_cfg = OmegaConf.to_container(defaults.model, resolve=True)
    model_cfg.update(_mics_group_overrides(cfg))
    raw_overrides = OmegaConf.select(cfg, "mics.overrides")
    if raw_overrides is None:
        overrides = {}
    else:
        overrides = OmegaConf.to_container(OmegaConf.create(raw_overrides), resolve=True)
    if isinstance(overrides, dict):
        model_cfg.update(overrides)
    ga = cfg.gallery
    return {
        "number_of_points": int(model_cfg.get("number_of_points", 1000)),
        "sampling": str(model_cfg.get("sampling", "coreset")),
        "num_epochs": int(model_cfg.get("num_epochs", 30)),
        "number_of_components": int(model_cfg.get("number_of_components", 3)),
        "num_layers": int(model_cfg.get("num_layers", 3)),
        "lab_to_rgb": bool(model_cfg.get("lab_to_rgb", True)),
        "cluster": bool(model_cfg.get("cluster", True)),
        "clusters": model_cfg.get("clusters", [128, 256, 512]),
        "num_samples": int(OmegaConf.select(ga, "num_samples", default=5000)),
        "beta": float(model_cfg.get("beta", 20.0)),
        "k_epoch": int(model_cfg.get("k_epoch", 20)),
        "temperature": float(model_cfg.get("temperature", 100.0)),
        "lr": float(model_cfg.get("lr", 1.0)),
        "batch_size": int(model_cfg.get("batch_size", 1024)),
        "warmup_epochs": int(model_cfg.get("warmup_epochs", 10)),
        "factor": float(model_cfg.get("factor", 1.0)),
        "random_state": int(cfg.seed),
        "verbose": bool(model_cfg.get("verbose", False)),
        "pixel_sampling": str(OmegaConf.select(ga, "sampling_mode", default="superpixel")),
        "pca_fit_step": int(model_cfg.get("pca_fit_step", 4)),
        "cluster_on_pca": bool(model_cfg.get("cluster_on_pca", False)),
        "cluster_pca_dims": int(model_cfg.get("cluster_pca_dims", 50)),
        "predict_percentile_low": float(model_cfg.get("predict_percentile_low", 1.0)),
        "predict_percentile_high": float(model_cfg.get("predict_percentile_high", 99.0)),
        "predict_spatial_smooth_sigma": float(model_cfg.get("predict_spatial_smooth_sigma", 0.0)),
    }


def _distill_cfg(cfg: DictConfig) -> ConceptDistillConfig:
    raw = OmegaConf.select(cfg, "distill")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if raw is None:
        return ConceptDistillConfig(device=device, seed=int(cfg.seed))
    return ConceptDistillConfig(
        n_concepts=int(OmegaConf.select(raw, "n_concepts", default=128)),
        top_k=int(OmegaConf.select(raw, "top_k", default=12)),
        epochs=int(OmegaConf.select(raw, "epochs", default=120)),
        steps_per_epoch=int(OmegaConf.select(raw, "steps_per_epoch", default=4)),
        lr=float(OmegaConf.select(raw, "lr", default=1e-2)),
        sparsity_weight=float(OmegaConf.select(raw, "sparsity_weight", default=0.005)),
        recon_weight=float(OmegaConf.select(raw, "recon_weight", default=0.01)),
        concept_l1_weight=float(OmegaConf.select(raw, "concept_l1_weight", default=1e-5)),
        batch_size=int(OmegaConf.select(raw, "batch_size", default=16384)),
        nmf_max_iter=int(OmegaConf.select(raw, "nmf_max_iter", default=200)),
        router_temperature=float(OmegaConf.select(raw, "router_temperature", default=1.0)),
        lab_to_rgb=bool(OmegaConf.select(raw, "lab_to_rgb", default=True)),
        percentile_low=float(OmegaConf.select(raw, "percentile_low", default=1.0)),
        percentile_high=float(OmegaConf.select(raw, "percentile_high", default=99.0)),
        display_bounds=str(OmegaConf.select(raw, "display_bounds", default="teacher")),
        readout_ridge_fit=bool(OmegaConf.select(raw, "readout_ridge_fit", default=True)),
        readout_ridge_lambda=float(OmegaConf.select(raw, "readout_ridge_lambda", default=1e-3)),
        readout_ridge_pixels=int(OmegaConf.select(raw, "readout_ridge_pixels", default=40000)),
        readout_epochs=int(OmegaConf.select(raw, "readout_epochs", default=40)),
        readout_steps_per_epoch=int(OmegaConf.select(raw, "readout_steps_per_epoch", default=4)),
        readout_refine_pixels=int(OmegaConf.select(raw, "readout_refine_pixels", default=40000)),
        readout_lr=float(OmegaConf.select(raw, "readout_lr", default=0.05)),
        embed_w_lr=float(OmegaConf.select(raw, "embed_w_lr", default=0.05)),
        router_mode=str(OmegaConf.select(raw, "router_mode", default="spatial_conv")),
        router_conv_kernel=int(OmegaConf.select(raw, "router_conv_kernel", default=3)),
        spatial_crop_size=int(OmegaConf.select(raw, "spatial_crop_size", default=192)),
        predict_embedding_smooth_sigma=float(
            OmegaConf.select(raw, "predict_embedding_smooth_sigma", default=0.75)
        ),
        train_objective=str(OmegaConf.select(raw, "train_objective", default="distill")),
        mics_number_of_points=int(OmegaConf.select(raw, "mics_number_of_points", default=1000)),
        mics_sampling=str(OmegaConf.select(raw, "mics_sampling", default="coreset")),
        mics_num_samples=int(OmegaConf.select(raw, "mics_num_samples", default=5000)),
        mics_beta=float(OmegaConf.select(raw, "mics_beta", default=20.0)),
        mics_k_epoch=int(OmegaConf.select(raw, "mics_k_epoch", default=1)),
        mics_teacher_blend=float(OmegaConf.select(raw, "mics_teacher_blend", default=0.0)),
        embed_diversity_weight=float(OmegaConf.select(raw, "embed_diversity_weight", default=1e-3)),
        concept_diversity_weight=float(OmegaConf.select(raw, "concept_diversity_weight", default=1e-4)),
        seed=int(cfg.seed),
        device=device,
    )


def _save_mosaic(
    rgbs: list[np.ndarray],
    titles: list[str],
    *,
    out_path: Path,
    dpi: int,
    figsize: tuple[float, float],
    gap_px: int = 1,
) -> None:
    del titles, dpi, figsize  # mosaic is pixel-stitched; labels live in manifest
    gap = max(0, int(gap_px))
    h, w = rgbs[0].shape[:2]
    n = len(rgbs)
    canvas = np.zeros((h, n * w + max(0, n - 1) * gap, 3), dtype=np.uint8)
    for i, rgb in enumerate(rgbs):
        x0 = i * (w + gap)
        canvas[:, x0 : x0 + w] = np.asarray(rgb, dtype=np.uint8)
    Image.fromarray(canvas).save(out_path)


def _masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return float("nan")
    diff = np.asarray(a, dtype=np.float64)[m] - np.asarray(b, dtype=np.float64)[m]
    return float(np.mean(diff * diff))


def _pixel_sampling_cache_path(cfg: DictConfig, npy_path: Path, model_kw: dict[str, Any], mics_group: str) -> Path:
    cache_root = _abs_path(
        str(OmegaConf.select(cfg, "gallery.persistent_cache_dir", default="outputs/gallery_mics_concept_distill/_caches"))
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "npy_path": str(npy_path),
        "mics_group": mics_group,
        "pixel_sampling": model_kw.get("pixel_sampling"),
        "num_samples": model_kw.get("num_samples"),
        "sampling": model_kw.get("sampling"),
        "number_of_points": model_kw.get("number_of_points"),
        "transpose_msi": bool(OmegaConf.select(cfg, "data.transpose_msi", default=False)),
        "rotate_clockwise_deg": int(OmegaConf.select(cfg, "data.rotate_clockwise_deg", default=0)) % 360,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return cache_root / f"{npy_path.stem}__sampling__{digest}.npz"


def _teacher_cache_path(cfg: DictConfig, npy_path: Path, model_kw: dict[str, Any], mics_group: str) -> Path:
    cache_root = _abs_path(
        str(OmegaConf.select(cfg, "gallery.teacher_cache_dir", default="outputs/gallery_mics_concept_distill/_teacher_cache"))
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "npy_path": str(npy_path),
        "mics_group": mics_group,
        "seed": int(cfg.seed),
        "model_kw": {k: model_kw[k] for k in sorted(model_kw) if k not in {"verbose"}},
        "normalization": str(cfg.data.normalization),
        "transpose_msi": bool(cfg.data.transpose_msi),
        "rotate_clockwise_deg": int(OmegaConf.select(cfg, "data.rotate_clockwise_deg", default=0)) % 360,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return cache_root / f"{npy_path.stem}__{mics_group}__{digest}.npz"


def _load_teacher_cache(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    data = np.load(path)
    return np.asarray(data["teacher_emb"], dtype=np.float32), np.asarray(data["teacher_rgb"], dtype=np.uint8)


def _save_teacher_cache(path: Path, teacher_emb: np.ndarray, teacher_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, teacher_emb=teacher_emb.astype(np.float32), teacher_rgb=teacher_rgb.astype(np.uint8))


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_concept_distill")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ga = cfg.gallery
    seed = int(cfg.seed)
    _seed_everything(seed)

    npy_path = _abs_path(str(ga.npy_path))
    if not npy_path.is_file():
        raise FileNotFoundError(f"gallery.npy_path not found: {npy_path}")

    norm_mode = _resolve_normalization_mode(str(cfg.data.normalization))
    transpose_msi = bool(cfg.data.transpose_msi)
    msi = _load_msi(npy_path, transpose_msi=transpose_msi, normalization=norm_mode)
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    model_kw = _mics_kwargs(cfg)
    mics_group = str(OmegaConf.select(cfg, "mics.group", default="custom"))
    cache_teacher = bool(OmegaConf.select(ga, "cache_teacher", default=True))
    teacher_cache_path = _teacher_cache_path(cfg, npy_path, model_kw, mics_group)

    teacher_emb: np.ndarray | None = None
    mics_rgb: np.ndarray | None = None
    if cache_teacher:
        cached = _load_teacher_cache(teacher_cache_path)
        if cached is not None:
            teacher_emb, mics_rgb = cached
            logger.info("Loaded cached MiCS teacher from %s", teacher_cache_path)

    if teacher_emb is None:
        logger.info("Training MiCS teacher on %s …", npy_path.name)
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
        if cache_teacher:
            _save_teacher_cache(teacher_cache_path, teacher_emb, mics_rgb)
            logger.info("Saved MiCS teacher cache: %s", teacher_cache_path)
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert teacher_emb is not None and mics_rgb is not None
    mics_rgb = np.asarray(mics_rgb, dtype=np.uint8)
    mics_rgb[~valid_mask] = 0

    distill_cfg = _distill_cfg(cfg)
    distill_cfg.percentile_low = float(model_kw["predict_percentile_low"])
    distill_cfg.percentile_high = float(model_kw["predict_percentile_high"])
    distill_cfg.lab_to_rgb = bool(model_kw["lab_to_rgb"])
    distill_cfg.mics_number_of_points = int(model_kw["number_of_points"])
    distill_cfg.mics_sampling = str(model_kw["sampling"])
    distill_cfg.mics_num_samples = int(model_kw["num_samples"])
    distill_cfg.mics_beta = float(model_kw["beta"])
    if int(distill_cfg.mics_k_epoch) <= 0:
        distill_cfg.mics_k_epoch = int(model_kw["k_epoch"])

    train_objective = str(distill_cfg.train_objective).lower()
    mics_cache: dict[str, Any] | None = None
    if train_objective in {"mics", "hybrid"} and bool(OmegaConf.select(ga, "share_pixel_sampling", default=True)):
        sampling_cache_path = _pixel_sampling_cache_path(cfg, npy_path, model_kw, mics_group)
        if sampling_cache_path.is_file():
            mics_cache = load_pixel_sampling_cache(sampling_cache_path)
            logger.info("Loaded MiCS pixel sampling cache for student training: %s", sampling_cache_path)
        else:
            mics_cache = resolve_pixel_sampling_cache(
                msi,
                model_kw,
                out_dir=run_dir / "caches" / "mics",
                share=True,
                cache_path=None,
                strip_cluster_labels=True,
            )
            if mics_cache is not None:
                save_pixel_sampling_cache(sampling_cache_path, mics_cache)
                logger.info("Built MiCS pixel sampling cache for student: %s", sampling_cache_path)

    tissue = msi[valid_mask]
    teacher_emb_tissue = teacher_emb[valid_mask]
    max_fit_raw = OmegaConf.select(cfg, "distill.max_fit_pixels", default=None)
    if max_fit_raw is None or int(max_fit_raw) <= 0:
        tissue_fit = tissue
        teacher_fit = teacher_emb_tissue
    else:
        max_fit = int(max_fit_raw)
        if tissue.shape[0] > max_fit:
            rng = np.random.default_rng(seed)
            pick = rng.choice(tissue.shape[0], size=max_fit, replace=False)
            tissue_fit = tissue[pick]
            teacher_fit = teacher_emb_tissue[pick]
        else:
            tissue_fit = tissue
            teacher_fit = teacher_emb_tissue

    nmf_init_raw = OmegaConf.select(cfg, "distill.nmf_init_pixels", default=24000)
    nmf_init = int(nmf_init_raw) if nmf_init_raw is not None else 0
    if nmf_init > 0 and tissue.shape[0] > nmf_init:
        rng_nmf = np.random.default_rng(seed + 1)
        nmf_pick = rng_nmf.choice(tissue.shape[0], size=nmf_init, replace=False)
        tissue_nmf = tissue[nmf_pick]
    else:
        tissue_nmf = tissue

    logger.info(
        "Initializing %d concepts (NMF on %d px) + training sparse student (%s) from %d pixels …",
        distill_cfg.n_concepts,
        tissue_nmf.shape[0],
        train_objective,
        tissue_fit.shape[0],
    )
    concepts0 = SparseConceptDistiller.init_concepts_nmf(tissue_nmf, distill_cfg)
    student = SparseConceptDistiller(msi.shape[-1], distill_cfg, concepts_init=concepts0)
    fit_stats = student.fit(
        tissue_fit,
        teacher_fit if train_objective in {"distill", "hybrid"} else None,
        teacher_display_embedding=teacher_emb if str(distill_cfg.display_bounds) == "teacher" else None,
        spec_image=msi,
        teacher_image=teacher_emb if train_objective == "distill" else None,
        mask_image=valid_mask,
        mics_cache=mics_cache,
    )
    student_emb = student.predict_embedding(msi, valid_mask=valid_mask)
    bounds = student._display_channel_bounds if str(distill_cfg.display_bounds) == "teacher" else None
    distill_rgb = embedding_to_display_rgb(
        student_emb,
        tissue_mask=valid_mask,
        lab_to_rgb=bool(distill_cfg.lab_to_rgb),
        percentile_low=float(distill_cfg.percentile_low),
        percentile_high=float(distill_cfg.percentile_high),
        channel_bounds=bounds,
    )

    mics_path = cells_dir / "mics.png"
    distill_path = cells_dir / "distilled_mics.png"
    Image.fromarray(mics_rgb).save(mics_path)
    Image.fromarray(distill_rgb).save(distill_path)

    mics_label = str(OmegaConf.select(ga, "mics_column_label", default="MiCS"))
    distill_label = str(OmegaConf.select(ga, "distill_column_label", default="distilled MiCS"))
    mosaic_path = run_dir / str(ga.output_filename)
    _save_mosaic(
        [mics_rgb, distill_rgb],
        [mics_label, distill_label],
        out_path=mosaic_path,
        dpi=int(ga.dpi),
        figsize=tuple(float(x) for x in OmegaConf.to_container(ga.figure_size_inches, resolve=True)),
        gap_px=int(OmegaConf.select(ga, "mosaic_gap_px", default=1)),
    )

    pixel_mse = _masked_mse(mics_rgb, distill_rgb, valid_mask)
    embedding_mse = _masked_mse(teacher_emb, student_emb, valid_mask)

    concepts_viz_summary: dict[str, Any] | None = None
    if bool(OmegaConf.select(cfg, "concepts_viz.enabled", default=True)):
        cv = OmegaConf.select(cfg, "concepts_viz") or {}
        concepts_viz_summary = save_concept_gallery(
            student,
            msi,
            valid_mask=valid_mask,
            out_dir=run_dir,
            npy_path=npy_path,
            max_concepts=int(OmegaConf.select(cv, "max_concepts", default=16)),
            spectrum_top_peaks=int(OmegaConf.select(cv, "spectrum_top_peaks", default=25)),
            dpi=int(OmegaConf.select(cv, "dpi", default=int(ga.dpi))),
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

    manifest = {
        "npy_path": str(npy_path),
        "shape": list(msi.shape),
        "mics_group": mics_group,
        "mics_rgb": str(mics_path),
        "distilled_rgb": str(distill_path),
        "mosaic_path": str(mosaic_path),
        "distill_fit": fit_stats,
        "pixel_rgb_mse_vs_teacher": pixel_mse,
        "embedding_mse_vs_teacher": embedding_mse,
        "concepts": student.concept_mz_summary(top_peaks=int(OmegaConf.select(cfg, "distill.top_peaks", default=5))),
        "mics_kwargs": {k: model_kw[k] for k in sorted(model_kw) if k != "verbose"},
        "distill_config": asdict(distill_cfg),
    }
    if concepts_viz_summary is not None:
        manifest["concepts_viz"] = concepts_viz_summary
    (run_dir / "gallery_mics_concept_distill_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    logger.info("Embedding MSE vs teacher: %.6f", embedding_mse)
    logger.info("Pixel RGB MSE vs teacher: %.2f", pixel_mse)
    logger.info("Wrote %s", mosaic_path)

    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
