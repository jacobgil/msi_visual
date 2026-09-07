#!/usr/bin/env python3
"""
Remap any MSI RGB visualization with a perceptual display transform.

Accepts a PNG/JPG/.npy RGB map (or a folder of them), optionally pairs with an MSI .npy
for tissue mask + HD reference edges, and optimizes interpretable LAB/RGB knobs to improve
luminance contrast, chromatic variety, and (when MSI is given) continuous edge Dice.

Examples (repo root):
  python scripts/remap_perceptual_display.py input.visualization_path=outputs/.../viz.png
  python scripts/remap_perceptual_display.py input.visualization_dir=path/to/pngs remap.mode=fixed
  python scripts/remap_perceptual_display.py input.visualization_path=viz.png input.msi_npy_path=cube.npy
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from debug_edge_maps import (  # noqa: E402
    _compute_continuous_dice,
    _edge_maps_multiscale,
    _edges_for_continuous_metrics,
    _load_msi,
    _load_visualization,
    _parse_edge_detection_cfg,
    _pca_project_msi_for_edges,
    _resolve_normalization_mode,
)
from msi_visual.perceptual_display import (  # noqa: E402
    PerceptualDisplayParams,
    _lut_grid_sizes,
    apply_perceptual_display,
    compute_ab_shuffle_distance,
    compute_color_metrics,
    param_bounds_for,
    save_lut_npy,
    save_params_json,
    tissue_mask_from_rgb,
)

logger = logging.getLogger(__name__)

_VIZ_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".npy"}


def _clickable_path(path: Path) -> str:
    """Absolute path string (clickable in most terminals / Cursor)."""
    return str(path.expanduser().resolve())


def _lut_grids_from_cfg(cfg: DictConfig) -> tuple[int, int]:
    ab = int(OmegaConf.select(cfg, "remap.lut_ab_grid", default=5))
    lg = int(OmegaConf.select(cfg, "remap.lut_l_grid", default=1))
    return ab, lg


def _opt_path(raw: Any) -> Path | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None
    return Path(s)


def _collect_viz_paths(cfg: DictConfig) -> list[Path]:
    single = _opt_path(OmegaConf.select(cfg, "input.visualization_path"))
    if single is not None:
        if not single.is_file():
            raise FileNotFoundError(f"visualization_path not found: {single}")
        return [single]

    folder = _opt_path(OmegaConf.select(cfg, "input.visualization_dir"))
    if folder is not None:
        if not folder.is_dir():
            raise NotADirectoryError(f"visualization_dir not found: {folder}")
        paths = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _VIZ_EXT
        )
        if not paths:
            raise FileNotFoundError(f"No visualization files under {folder}")
        return paths

    raise ValueError("Set input.visualization_path or input.visualization_dir")


def _load_mask(cfg: DictConfig, hw: tuple[int, int]) -> np.ndarray | None:
    mp = _opt_path(OmegaConf.select(cfg, "input.mask_path"))
    if mp is None:
        return None
    arr = np.load(mp, mmap_mode=None)
    if arr.shape[:2] != hw:
        raise ValueError(f"mask shape {arr.shape[:2]} != visualization {hw}")
    return np.asarray(arr, dtype=bool) if arr.dtype == bool else arr > 0.5


def _load_rgb_viz(path: Path, hw: tuple[int, int], allow_resize: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return _load_visualization(path, hw, allow_resize)
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = hw
    if rgb.shape[:2] != (h, w):
        if not allow_resize:
            raise ValueError(f"Shape mismatch {rgb.shape[:2]} vs {(h, w)} for {path}")
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb


def _normalization_mode(cfg: DictConfig) -> str:
    return _resolve_normalization_mode(
        OmegaConf.select(cfg, "data.normalization", default="tic")
    )


def _resolve_hw(cfg: DictConfig, viz_path: Path) -> tuple[int, int]:
    msi = _opt_path(OmegaConf.select(cfg, "input.msi_npy_path"))
    if msi is not None:
        transpose = bool(OmegaConf.select(cfg, "input.transpose_msi", default=False))
        cube = _load_msi(msi, transpose_msi=transpose, normalization=_normalization_mode(cfg))
        return int(cube.shape[0]), int(cube.shape[1])
    if viz_path.suffix.lower() == ".npy":
        arr = np.load(viz_path, mmap_mode="r")
        if arr.ndim < 2:
            raise ValueError(f"Bad npy shape {arr.shape}")
        return int(arr.shape[0]), int(arr.shape[1])
    with Image.open(viz_path) as im:
        w, h = im.size
    return int(h), int(w)


def _build_hd_context(cfg: DictConfig, hw: tuple[int, int]) -> dict[str, Any] | None:
    msi_path = _opt_path(OmegaConf.select(cfg, "input.msi_npy_path"))
    if msi_path is None:
        return None

    transpose = bool(OmegaConf.select(cfg, "input.transpose_msi", default=False))
    norm_mode = _normalization_mode(cfg)
    cube = _load_msi(msi_path, transpose_msi=transpose, normalization=norm_mode)
    if cube.shape[:2] != hw:
        raise ValueError(f"MSI shape {cube.shape[:2]} != visualization {hw}")

    valid_mask = cube.sum(axis=-1) > 0
    edge_cfg = _parse_edge_detection_cfg(cfg)
    pca_proj = _pca_project_msi_for_edges(cube, cfg, valid_mask)

    ms_cfg = OmegaConf.select(cfg, "edge_maps.multi_scale", default=None)
    ms_enabled = bool(OmegaConf.select(ms_cfg, "enabled", default=False)) if ms_cfg else False
    sigmas_raw = OmegaConf.select(ms_cfg, "sigmas", default=[0.0]) if ms_cfg else [0.0]
    sigmas = tuple(float(v) for v in list(sigmas_raw)) if ms_enabled and sigmas_raw else (0.0,)
    aggregation = str(
        OmegaConf.select(ms_cfg, "aggregation", default="mean") if ms_cfg else "mean"
    )

    hd_edge, _ = _edge_maps_multiscale(
        "hd",
        cube,
        valid_mask,
        sigmas=sigmas,
        aggregation=aggregation,
        hd_cfg=pca_proj,
        edge_method_cfg=edge_cfg,
    )
    spatial_norm = bool(OmegaConf.select(cfg, "spatial_normalization.enabled", default=True))
    if spatial_norm:
        from debug_edge_maps import _normalize_map_percentile

        low = float(OmegaConf.select(cfg, "spatial_normalization.low_percentile", default=1.0))
        high = float(OmegaConf.select(cfg, "spatial_normalization.high_percentile", default=99.0))
        hd_n = _normalize_map_percentile(hd_edge, valid_mask, low_pct=low, high_pct=high)
    else:
        from debug_edge_maps import _normalize_on_mask

        hd_n = _normalize_on_mask(hd_edge, valid_mask)

    return {
        "valid_mask": valid_mask,
        "hd_edge_norm": hd_n,
    }


def _compute_viz_edge_norm(
    viz_rgb: np.ndarray,
    valid_mask: np.ndarray,
    cfg: DictConfig,
) -> np.ndarray:
    from debug_edge_maps import _normalize_map_percentile, _normalize_on_mask, _to_uint8_rgb

    ms_cfg = OmegaConf.select(cfg, "edge_maps.multi_scale", default=None)
    ms_enabled = bool(OmegaConf.select(ms_cfg, "enabled", default=False)) if ms_cfg else False
    sigmas_raw = OmegaConf.select(ms_cfg, "sigmas", default=[0.0]) if ms_cfg else [0.0]
    sigmas = [float(v) for v in list(sigmas_raw)] if ms_enabled and sigmas_raw else [0.0]
    aggregation = str(
        OmegaConf.select(ms_cfg, "aggregation", default="mean") if ms_cfg else "mean"
    )
    edge_cfg = _parse_edge_detection_cfg(cfg)
    viz_u8 = _to_uint8_rgb(viz_rgb)
    v_edge, _ = _edge_maps_multiscale(
        "rgb",
        viz_u8,
        valid_mask,
        sigmas=sigmas,
        aggregation=aggregation,
        hd_cfg=None,
        edge_method_cfg=edge_cfg,
    )
    sn = OmegaConf.select(cfg, "spatial_normalization", default=None)
    if sn is not None and bool(OmegaConf.select(sn, "enabled", default=True)):
        return _normalize_map_percentile(
            v_edge,
            valid_mask,
            low_pct=float(OmegaConf.select(sn, "low_percentile", default=1.0)),
            high_pct=float(OmegaConf.select(sn, "high_percentile", default=99.0)),
        )
    return _normalize_on_mask(v_edge, valid_mask)


def _continuous_dice_rgb(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    hd_ctx: dict[str, Any],
    cfg: DictConfig,
) -> float:
    square = bool(
        OmegaConf.select(cfg, "evaluation.square_before_continuous_metrics", default=False)
    )
    viz_edge_n = _compute_viz_edge_norm(rgb, valid_mask, cfg)
    hd_d, v_d = _edges_for_continuous_metrics(
        hd_ctx["hd_edge_norm"],
        viz_edge_n,
        square=square,
    )
    return float(_compute_continuous_dice(hd_d, v_d, valid_mask))


def _objective_score(
    rgb_in: np.ndarray,
    rgb_out: np.ndarray,
    valid_mask: np.ndarray,
    weights: dict[str, float],
    baseline_metrics: dict[str, float],
    baseline_dice: float | None,
    dice_out: float | None,
    *,
    relative: bool,
    dice_floor_ratio: float,
    lab_bins_a: int,
    lab_bins_b: int,
    luminance_preservation_weight: float,
) -> float:
    m_out = compute_color_metrics(
        rgb_out,
        valid_mask,
        lab_entropy_bins_a=lab_bins_a,
        lab_entropy_bins_b=lab_bins_b,
    )
    score = 0.0
    w_sum = 0.0

    for key in (
        "luminance_rms_contrast",
        "lab_chroma_entropy",
        "lab_mean_chroma",
        "luminance_gradient_mean",
    ):
        w = float(weights.get(key, 0.0) or 0.0)
        if abs(w) < 1e-12:
            continue
        v_out = m_out[key]
        if not np.isfinite(v_out):
            continue
        if relative:
            v_in = baseline_metrics.get(key, float("nan"))
            if not np.isfinite(v_in) or v_in < 1e-8:
                term = v_out
            else:
                term = v_out / v_in
        else:
            term = v_out
        score += w * term
        w_sum += abs(w)

    w_ab = float(weights.get("ab_shuffle_distance", 0.0) or 0.0)
    if abs(w_ab) > 1e-12:
        ab_dist = compute_ab_shuffle_distance(rgb_in, rgb_out, valid_mask)
        if np.isfinite(ab_dist):
            if relative:
                ab_in = float(baseline_metrics.get("ab_shuffle_distance", 0.0) or 0.0)
                term_ab = ab_dist / max(ab_in, 1.0) if ab_in > 1e-8 else ab_dist
            else:
                term_ab = ab_dist
            score += w_ab * term_ab
            w_sum += abs(w_ab)

    w_dice = float(weights.get("continuous_dice", 0.0) or 0.0)
    if abs(w_dice) > 1e-12 and dice_out is not None and np.isfinite(dice_out):
        dice_term = dice_out
        if baseline_dice is not None and np.isfinite(baseline_dice) and baseline_dice > 1e-8:
            floor = float(dice_floor_ratio) * baseline_dice
            if dice_out < floor:
                dice_term = dice_out - 2.0 * (floor - dice_out)
        score += w_dice * dice_term
        w_sum += abs(w_dice)

    w_lum = float(luminance_preservation_weight)
    if w_lum > 1e-12:
        lum_in = baseline_metrics.get("luminance_rms_contrast", float("nan"))
        lum_out = m_out.get("luminance_rms_contrast", float("nan"))
        if np.isfinite(lum_in) and np.isfinite(lum_out) and lum_in > 1e-8:
            rel_change = abs(lum_out - lum_in) / lum_in
            score -= w_lum * rel_change
            w_sum += w_lum

    if w_sum < 1e-12:
        return 0.0
    return score / w_sum


def _evaluate_params(
    params: PerceptualDisplayParams,
    rgb_in: np.ndarray,
    valid_mask: np.ndarray,
    hd_ctx: dict[str, Any] | None,
    obj_cfg: dict[str, Any],
    baseline_metrics: dict[str, float],
    baseline_dice: float | None,
    cfg: DictConfig,
) -> tuple[float, dict[str, float], np.ndarray, float | None]:
    rgb_out = apply_perceptual_display(rgb_in, params, valid_mask)
    dice_out = None
    if hd_ctx is not None:
        dice_out = _continuous_dice_rgb(rgb_out, valid_mask, hd_ctx, cfg)
    score = _objective_score(
        rgb_in,
        rgb_out,
        valid_mask,
        obj_cfg["weights"],
        baseline_metrics,
        baseline_dice,
        dice_out,
        relative=bool(obj_cfg.get("relative_to_input", True)),
        dice_floor_ratio=float(obj_cfg.get("dice_floor_ratio", 0.92)),
        lab_bins_a=int(obj_cfg.get("lab_entropy_bins_a", 16)),
        lab_bins_b=int(obj_cfg.get("lab_entropy_bins_b", 16)),
        luminance_preservation_weight=float(
            obj_cfg.get("luminance_preservation_weight", 0.0) or 0.0
        ),
    )
    metrics_out = compute_color_metrics(
        rgb_out,
        valid_mask,
        lab_entropy_bins_a=int(obj_cfg.get("lab_entropy_bins_a", 16)),
        lab_entropy_bins_b=int(obj_cfg.get("lab_entropy_bins_b", 16)),
    )
    metrics_out["ab_shuffle_distance"] = compute_ab_shuffle_distance(
        rgb_in, rgb_out, valid_mask
    )
    if dice_out is not None:
        metrics_out["continuous_dice"] = dice_out
    return score, metrics_out, rgb_out, dice_out


def _optimize_params(
    rgb_in: np.ndarray,
    valid_mask: np.ndarray,
    hd_ctx: dict[str, Any] | None,
    obj_cfg: dict[str, Any],
    opt_cfg: dict[str, Any],
    initial: PerceptualDisplayParams,
    baseline_metrics: dict[str, float],
    baseline_dice: float | None,
    cfg: DictConfig,
) -> tuple[PerceptualDisplayParams, float, dict[str, float], np.ndarray]:
    method = str(opt_cfg.get("method", "differential_evolution")).strip().lower()
    seed = int(opt_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    strategy = initial.normalized_strategy()
    lut_ab, lut_l = _lut_grids_from_cfg(cfg)
    bounds = list(
        param_bounds_for(strategy, lut_ab_grid=lut_ab, lut_l_grid=lut_l)
    )
    if strategy == "color_shuffle" and not bool(
        OmegaConf.select(cfg, "remap.optimize_rgb_perm", default=False)
    ):
        # Lock RGB channel order; hue rotation in LAB preserves L.
        bounds[2] = (0.0, 0.01)

    best_params = initial
    best_score, best_metrics, best_rgb, _ = _evaluate_params(
        best_params,
        rgb_in,
        valid_mask,
        hd_ctx,
        obj_cfg,
        baseline_metrics,
        baseline_dice,
        cfg,
    )

    def objective_vec(x: np.ndarray) -> float:
        try:
            p = PerceptualDisplayParams.from_vector(
                x,
                strategy=strategy,
                lut_ab_grid=lut_ab,
                lut_l_grid=lut_l,
            )
            if strategy == "contrast" and p.percentile_high <= p.percentile_low + 0.5:
                return 1e6
            sc, _, _, _ = _evaluate_params(
                p,
                rgb_in,
                valid_mask,
                hd_ctx,
                obj_cfg,
                baseline_metrics,
                baseline_dice,
                cfg,
            )
            return -sc
        except Exception:
            return 1e6

    if method == "differential_evolution":
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            objective_vec,
            bounds=bounds,
            maxiter=int(opt_cfg.get("max_iterations", 45)),
            popsize=max(2, int(opt_cfg.get("population_size", 14)) // len(bounds)),
            seed=seed,
            polish=bool(opt_cfg.get("polish", True)),
            workers=1,
            updating="immediate",
        )
        cand = PerceptualDisplayParams.from_vector(
            result.x,
            strategy=strategy,
            lut_ab_grid=lut_ab,
            lut_l_grid=lut_l,
        )
        sc, met, rgb_o, _ = _evaluate_params(
            cand,
            rgb_in,
            valid_mask,
            hd_ctx,
            obj_cfg,
            baseline_metrics,
            baseline_dice,
            cfg,
        )
        if sc > best_score:
            best_params, best_score, best_metrics, best_rgb = cand, sc, met, rgb_o
    else:
        n_trials = int(opt_cfg.get("random_trials", 400))
        for _ in range(n_trials):
            x = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float64)
            cand = PerceptualDisplayParams.from_vector(
                x,
                strategy=strategy,
                lut_ab_grid=lut_ab,
                lut_l_grid=lut_l,
            )
            if strategy == "contrast" and cand.percentile_high <= cand.percentile_low + 0.5:
                continue
            sc, met, rgb_o, _ = _evaluate_params(
                cand,
                rgb_in,
                valid_mask,
                hd_ctx,
                obj_cfg,
                baseline_metrics,
                baseline_dice,
                cfg,
            )
            if sc > best_score:
                best_params, best_score, best_metrics, best_rgb = cand, sc, met, rgb_o

    return best_params, best_score, best_metrics, best_rgb


def _side_by_side(left: np.ndarray, right: np.ndarray, pad: int) -> np.ndarray:
    p = max(0, int(pad))
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1] + p
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    x1 = left.shape[1] + p
    canvas[: right.shape[0], x1 : x1 + right.shape[1]] = right
    return canvas


def _process_one(
    viz_path: Path,
    cfg: DictConfig,
    out_dir: Path,
) -> dict[str, Any]:
    hw = _resolve_hw(cfg, viz_path)
    allow_resize = bool(OmegaConf.select(cfg, "input.allow_resize", default=True))
    rgb_in = _load_rgb_viz(viz_path, hw, allow_resize)

    hd_ctx = _build_hd_context(cfg, hw)
    if hd_ctx is not None:
        valid_mask = hd_ctx["valid_mask"]
    else:
        mask = _load_mask(cfg, hw)
        if mask is not None:
            valid_mask = mask
        else:
            thr = float(OmegaConf.select(cfg, "input.background_threshold", default=0))
            valid_mask = tissue_mask_from_rgb(rgb_in)
            if thr > 0:
                valid_mask = valid_mask & (np.sum(rgb_in, axis=-1) > thr)

    obj_raw = OmegaConf.to_container(cfg.objective, resolve=True)
    assert isinstance(obj_raw, dict)
    obj_cfg = dict(obj_raw)
    obj_cfg["weights"] = dict(obj_cfg.get("weights") or {})

    baseline_metrics = compute_color_metrics(
        rgb_in,
        valid_mask,
        lab_entropy_bins_a=int(obj_cfg.get("lab_entropy_bins_a", 16)),
        lab_entropy_bins_b=int(obj_cfg.get("lab_entropy_bins_b", 16)),
    )
    baseline_metrics["ab_shuffle_distance"] = 0.0
    baseline_dice = None
    if hd_ctx is not None:
        baseline_dice = _continuous_dice_rgb(rgb_in, valid_mask, hd_ctx, cfg)
        baseline_metrics["continuous_dice"] = baseline_dice

    remap_raw = OmegaConf.to_container(cfg.remap, resolve=True)
    assert isinstance(remap_raw, dict)
    initial = PerceptualDisplayParams.from_mapping(remap_raw)
    mode = str(remap_raw.get("mode", "optimize")).strip().lower()

    if mode == "fixed":
        params = initial
        score, metrics_out, rgb_out, dice_out = _evaluate_params(
            params,
            rgb_in,
            valid_mask,
            hd_ctx,
            obj_cfg,
            baseline_metrics,
            baseline_dice,
            cfg,
        )
    else:
        opt_raw = OmegaConf.to_container(cfg.optimize, resolve=True)
        assert isinstance(opt_raw, dict)
        params, score, metrics_out, rgb_out = _optimize_params(
            rgb_in,
            valid_mask,
            hd_ctx,
            obj_cfg,
            opt_raw,
            initial,
            baseline_metrics,
            baseline_dice,
            cfg,
        )

    stem = viz_path.stem
    out_png = out_dir / f"{stem}__perceptual_remap.png"
    Image.fromarray(rgb_out, mode="RGB").save(out_png)

    if bool(OmegaConf.select(cfg, "output.save_comparison", default=True)):
        pad = int(OmegaConf.select(cfg, "output.comparison_pad", default=8))
        cmp_path = out_dir / f"{stem}__before_after.png"
        Image.fromarray(_side_by_side(rgb_in, rgb_out, pad), mode="RGB").save(cmp_path)
    else:
        cmp_path = None

    params_path = out_dir / f"{stem}__perceptual_params.json"
    save_params_json(params_path, params)
    lut_path = out_dir / f"{stem}__perceptual_lut.npy"
    if params.normalized_strategy() == "perceptual_lut":
        save_lut_npy(lut_path, params)
        lg, g = _lut_grid_sizes(params.lut_ab_grid, params.lut_l_grid)
        logger.info(
            "LUT grid: L=%d x a,b=%dx%d (%d learnable deltas)",
            lg,
            g,
            g,
            lg * g * g * 2,
        )
    else:
        lut_path = None

    row = {
        "visualization": str(viz_path),
        "output": str(out_png),
        "mode": mode,
        "objective_score": score,
        **{f"in_{k}": v for k, v in baseline_metrics.items()},
        **{f"out_{k}": v for k, v in metrics_out.items()},
        **{f"param_{k}": v for k, v in params.to_json_dict().items()},
    }
    logger.info(
        "Remapped %s [%s] | score=%.4f | entropy %.3f->%.3f | chroma %.1f->%.1f | lum %.4f->%.4f%s",
        viz_path.name,
        params.normalized_strategy(),
        score,
        baseline_metrics.get("lab_chroma_entropy", float("nan")),
        metrics_out.get("lab_chroma_entropy", float("nan")),
        baseline_metrics.get("lab_mean_chroma", float("nan")),
        metrics_out.get("lab_mean_chroma", float("nan")),
        baseline_metrics.get("luminance_rms_contrast", float("nan")),
        metrics_out.get("luminance_rms_contrast", float("nan")),
        (
            f" | dice {baseline_dice:.4f}->{metrics_out.get('continuous_dice', float('nan')):.4f}"
            if baseline_dice is not None
            else ""
        ),
    )
    logger.info("Input visualization: %s", _clickable_path(viz_path))
    logger.info("Output visualization: %s", _clickable_path(out_png))
    if cmp_path is not None:
        logger.info("Before/after comparison: %s", _clickable_path(cmp_path))
    logger.info("Params JSON: %s", _clickable_path(params_path))
    row["comparison"] = _clickable_path(cmp_path) if cmp_path else ""
    row["params_json"] = _clickable_path(params_path)
    row["lut_npy"] = _clickable_path(lut_path) if lut_path is not None else ""
    row["output"] = _clickable_path(out_png)
    row["visualization"] = _clickable_path(viz_path)
    if lut_path is not None:
        logger.info("LUT array: %s", _clickable_path(lut_path))
    return row


@hydra.main(version_base=None, config_path="configs", config_name="remap_perceptual_display")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )

    paths = _collect_viz_paths(cfg)
    rows: list[dict[str, Any]] = []
    for viz_path in paths:
        rows.append(_process_one(viz_path, cfg, out_dir))

    if bool(OmegaConf.select(cfg, "output.save_metrics_csv", default=True)) and rows:
        csv_path = out_dir / "remap_metrics.csv"
        fieldnames: list[str] = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info("Saved metrics: %s", _clickable_path(csv_path))

    logger.info("Output directory: %s", _clickable_path(out_dir))
    logger.info("Done - %d visualization(s)", len(rows))


if __name__ == "__main__":
    main()
