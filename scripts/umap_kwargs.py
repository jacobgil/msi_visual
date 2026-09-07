"""Shared Parametric UMAP kwargs from Hydra config (no heavy benchmark imports)."""
from __future__ import annotations

from typing import Dict

from omegaconf import DictConfig, OmegaConf


def umap_kwargs(cfg: DictConfig, sampling_mode: str, run_seed: int) -> Dict[str, object]:
    shared_sampling = getattr(cfg.benchmark, "shared_sampling", None)
    enforce_same_pca_fit_step = (
        bool(getattr(shared_sampling, "enforce_same_pca_fit_step", True))
        if shared_sampling is not None
        else False
    )
    shared_pca_fit_step = (
        int(getattr(shared_sampling, "pca_fit_step", cfg.parametric_umap.pca_fit_step))
        if shared_sampling is not None
        else int(cfg.parametric_umap.pca_fit_step)
    )
    enforce_same_sampling_seed = (
        bool(getattr(shared_sampling, "enforce_same_sampling_seed", True))
        if shared_sampling is not None
        else False
    )
    ne_raw = OmegaConf.select(cfg, "parametric_umap.n_epochs")
    return {
        "n_components": int(cfg.parametric_umap.n_components),
        "n_neighbors": int(cfg.parametric_umap.n_neighbors),
        "n_training_epochs": int(cfg.parametric_umap.n_training_epochs),
        "n_epochs": int(ne_raw) if ne_raw is not None else None,
        "pixel_sampling": str(sampling_mode),
        "num_samples": int(cfg.benchmark.num_samples),
        "pca_fit_step": (
            int(shared_pca_fit_step)
            if enforce_same_pca_fit_step
            else int(cfg.parametric_umap.pca_fit_step)
        ),
        "random_state": (
            int(run_seed)
            if enforce_same_sampling_seed
            else int(cfg.parametric_umap.random_state)
        ),
    }
