#!/usr/bin/env python3
"""Multi-image MiCS tuning: OFAT + hill on clusters, sampling, and training hyperparameters."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bayes_tune_mics_lmc import (  # noqa: E402
    _build_datasets_from_cfg,
    _resolve_npy_paths,
    _search_space_items,
    run_mics_hyperparameter_tune,
)

logger = logging.getLogger(__name__)


def _dict_from_cfg_node(cfg: DictConfig, key: str) -> dict[str, Any]:
    raw = OmegaConf.to_container(OmegaConf.select(cfg, key, default={}), resolve=True)
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def _merge_model_base_into_fixed(cfg: DictConfig) -> None:
    """Push ``model_base`` into ``tuning.fixed_params`` when not a search axis."""
    base = _dict_from_cfg_node(cfg, "model_base")
    if not base:
        return
    space_names = {name for name, _ in _search_space_items(cfg)}
    fixed_raw = OmegaConf.select(cfg, "tuning.fixed_params", default={})
    fixed = OmegaConf.to_container(fixed_raw, resolve=True) if fixed_raw is not None else {}
    if not isinstance(fixed, dict):
        fixed = {}
    for k, v in base.items():
        if k not in space_names:
            fixed[k] = v
    OmegaConf.update(cfg, "tuning.fixed_params", fixed, merge=False)


def _apply_input_data_aliases(cfg: DictConfig) -> None:
    if OmegaConf.select(cfg, "data.normalization", default=None) is None:
        norm = OmegaConf.select(cfg, "input.normalization", default="tic")
        OmegaConf.update(cfg, "data.normalization", norm, merge=True)
    if OmegaConf.select(cfg, "data.transpose_msi", default=None) is None:
        tr = OmegaConf.select(cfg, "input.transpose_msi", default=False)
        OmegaConf.update(cfg, "data.transpose_msi", tr, merge=True)


@hydra.main(version_base=None, config_path="configs", config_name="study_mics_clusters_sampling")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(getattr(cfg.logging, "level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    _apply_input_data_aliases(cfg)
    _merge_model_base_into_fixed(cfg)

    npy_paths = _resolve_npy_paths(cfg)
    max_n = OmegaConf.select(cfg, "study.max_images", default=None)
    if max_n is not None and int(max_n) >= 1:
        npy_paths = npy_paths[: int(max_n)]
        OmegaConf.update(cfg, "input.npy_paths", [str(p) for p in npy_paths], merge=True)

    logger.info(
        "Study: %d images, up to %d tuning iterations (OFAT + hill)",
        len(npy_paths),
        int(cfg.tuning.n_iterations),
    )

    datasets = _build_datasets_from_cfg(cfg, run_dir)
    result = run_mics_hyperparameter_tune(cfg, run_dir, datasets)
    best = result.get("best_row") or {}
    logger.info(
        "Done. Best score=%s | ladder=%s | sampling=%s | features=%s | "
        "beta=%s layers=%s factor=%s samples=%s k_epoch=%s | CSV=%s",
        best.get("score"),
        best.get("cluster_ladder", best.get("clusters")),
        best.get("pixel_sampling"),
        best.get("cluster_features"),
        best.get("beta"),
        best.get("num_layers"),
        best.get("factor"),
        best.get("num_samples"),
        best.get("k_epoch"),
        result.get("csv_path"),
    )


if __name__ == "__main__":
    main()
