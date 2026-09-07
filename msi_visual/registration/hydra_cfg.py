"""Parse Hydra registration config into register_he_to_msi kwargs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from msi_visual.registration.he_msi import EccStageConfig


def parse_ecc_stages(raw: Any) -> list[EccStageConfig] | None:
    if raw is None:
        return None
    items = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(items, list) or not items:
        return None
    out: list[EccStageConfig] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            EccStageConfig(
                name=str(item.get("name", "stage")),
                reference=str(item.get("reference", "tic")),  # type: ignore[arg-type]
                max_side=item.get("max_side"),
                motion=str(item.get("motion", "affine")),  # type: ignore[arg-type]
                ecc_iterations=int(item.get("ecc_iterations", 3000)),
                ecc_epsilon=float(item.get("ecc_epsilon", 1e-6)),
                gauss_filt_size=int(item.get("gauss_filt_size", 5)),
            )
        )
    return out or None


def registration_kwargs_from_cfg(
    cfg: DictConfig,
    *,
    rank_path: Path | None,
    tune_cfg: DictConfig | None = None,
    mics_rgb: Any = None,
    mics_source: str | None = None,
) -> dict[str, Any]:
    tune = tune_cfg if tune_cfg is not None else cfg
    reg = cfg.registration
    rot_raw = OmegaConf.select(reg, "rotation_search")
    rotation_search = None
    if rot_raw is not None:
        rot_items = OmegaConf.to_container(rot_raw, resolve=True)
        if isinstance(rot_items, list):
            rotation_search = [int(v) for v in rot_items]

    mics_channel = str(OmegaConf.select(reg, "mics.channel", default="l"))
    strategy = str(OmegaConf.select(reg, "strategy", default="mask_only")).strip().lower()
    if strategy not in ("mask_only", "multistage"):
        strategy = "mask_only"

    angle_range = OmegaConf.to_container(OmegaConf.select(reg, "refine.angle_deg_range", default=[-35, 35]), resolve=True)
    scale_range = OmegaConf.to_container(OmegaConf.select(reg, "refine.scale_range", default=[0.82, 1.18]), resolve=True)

    return {
        "transpose_msi": bool(OmegaConf.select(tune, "data.transpose_msi", default=False)),
        "normalization": str(OmegaConf.select(tune, "data.normalization", default="tic")),
        "reference": str(OmegaConf.select(reg, "reference", default="tic")),
        "rank_config_path": rank_path,
        "motion": str(OmegaConf.select(reg, "motion", default="affine")),
        "ecc_iterations": int(OmegaConf.select(reg, "ecc_iterations", default=5000)),
        "ecc_epsilon": float(OmegaConf.select(reg, "ecc_epsilon", default=1e-6)),
        "gauss_filt_size": int(OmegaConf.select(reg, "gauss_filt_size", default=5)),
        "stages": parse_ecc_stages(OmegaConf.select(reg, "stages")),
        "rotation_search": rotation_search,
        "init_method": str(OmegaConf.select(reg, "init_method", default="mask_similarity")),
        "mics_rgb": mics_rgb,
        "mics_channel": mics_channel,
        "mics_source": mics_source,
        "strategy": strategy,
        "refine_enabled": bool(OmegaConf.select(reg, "refine.enabled", default=True)),
        "refine_max_side": int(OmegaConf.select(reg, "refine.max_side", default=384)),
        "refine_angle_deg_range": (float(angle_range[0]), float(angle_range[1])),
        "refine_angle_deg_step": float(OmegaConf.select(reg, "refine.angle_deg_step", default=5.0)),
        "refine_scale_range": (float(scale_range[0]), float(scale_range[1])),
        "refine_scale_step": float(OmegaConf.select(reg, "refine.scale_step", default=0.04)),
        "refine_shift_frac": float(OmegaConf.select(reg, "refine.shift_frac", default=0.12)),
        "refine_shift_step": float(OmegaConf.select(reg, "refine.shift_step", default=0.04)),
    }


def resolve_mics_from_registration_cfg(
    cfg: DictConfig,
    *,
    npy_path: Path,
    img_idx: int,
    target_hw: tuple[int, int],
    resolve_path,
) -> tuple[Any, str | None]:
    """Load MiCS RGB for registration from tune run, viz glob, or explicit path."""
    from msi_visual.registration.mics_reference import resolve_mics_rgb_for_registration

    if not bool(OmegaConf.select(cfg, "registration.mics.enabled", default=True)):
        return None, None

    tune_raw = OmegaConf.select(cfg, "registration.mics.tune_run_dir")
    tune_run_dir = resolve_path(str(tune_raw)) if tune_raw else None

    # Fallback: diagnostic panels pass tune_run_dir at top level
    if tune_run_dir is None:
        top_tune = OmegaConf.select(cfg, "tune_run_dir")
        if top_tune:
            tune_run_dir = resolve_path(str(top_tune))

    png_raw = OmegaConf.select(cfg, "registration.mics.png_path")
    explicit = resolve_path(str(png_raw)) if png_raw else None

    viz_glob = OmegaConf.select(cfg, "registration.mics.viz_glob")
    viz_glob_s = str(viz_glob).strip() if viz_glob else None
    if viz_glob_s in ("", "null", "none", "~"):
        viz_glob_s = None

    return resolve_mics_rgb_for_registration(
        npy_path=npy_path,
        img_idx=img_idx,
        tune_run_dir=tune_run_dir,
        explicit_path=explicit,
        viz_glob=viz_glob_s,
        target_hw=target_hw,
    )
