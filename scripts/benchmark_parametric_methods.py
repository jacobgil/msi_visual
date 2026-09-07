#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import hashlib
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from msi_visual.metrics import MSIVisualizationMetrics
from msi_visual.nmf_3d import NMF3D
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC
from msi_visual.parametric_umap import UMAPVirtualStain
from msi_visual.pca_3d import PCA3D
from msi_visual.percentile_ratio import TOP3
from scripts.debug_edge_maps import _compute_viz_contrast_color_metrics
from scripts.debug_visualization_metrics import DebugNMFEvaluator
from scripts.umap_kwargs import umap_kwargs as _umap_kwargs

logger = logging.getLogger(__name__)

_BF_STAGES = 4


def _bf_stage(n: int, msg: str) -> None:
    print(f"[backfill_edge_metrics {n}/{_BF_STAGES}] {msg}", flush=True)


# Edge agreement metrics (F1 vs continuous Dice, min_viz_edge_percentile) are computed in
# DebugNMFEvaluator; configure under edge_evaluation in benchmark_parametric_methods.yaml.
# Rows also get luminance RMS contrast + LAB chroma entropy from the saved viz RGB (see
# _compute_visualization_quality_metrics); ok rows expose continuous_dice as a mirror of
# edge_agreement_continuous_dice for alignment with bayes_tune column names.


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    sums = msi.sum(axis=-1, keepdims=True)
    sums = sums + 1e-8
    return msi / sums


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI shape HxWxC (or CxHxW with transpose), got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI contains negative values; this benchmark expects non-negative inputs.")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def _load_raw_msi(path: Path, transpose_msi: bool) -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI shape HxWxC (or CxHxW with transpose), got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI contains negative values; this benchmark expects non-negative inputs.")
    return img


def _to_uint8_rgb(viz: np.ndarray) -> np.ndarray:
    if isinstance(viz, list):
        viz = viz[0]
    arr = np.asarray(viz)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        out = arr
    else:
        out = arr.astype(np.float32)
        mx = float(np.nanmax(out))
        # Float in ~[0,1] (or very dim): scale so max -> 255. Plain *255 would map max=0.001 to ~0 (black).
        if mx <= 1.0 + 1e-6:
            if mx > 1e-12:
                out = (out / mx) * 255.0
            else:
                out = np.zeros_like(out)
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _load_existing_viz_rgb(viz_path: Path, expected_hw: Tuple[int, int]) -> np.ndarray:
    viz_rgb = np.asarray(Image.open(viz_path).convert("RGB"), dtype=np.uint8)
    if viz_rgb.shape[:2] != expected_hw:
        raise ValueError(
            f"Existing visualization shape mismatch for {viz_path}: "
            f"got {viz_rgb.shape[:2]}, expected {expected_hw}"
        )
    return viz_rgb


def _maybe_lab_to_rgb(viz_rgb: np.ndarray, method_name: str, cfg: DictConfig) -> np.ndarray:
    methods = getattr(cfg.benchmark, "lab_to_rgb_methods", [])
    if methods is None:
        return viz_rgb
    method_set = {str(m) for m in methods}
    if method_name not in method_set:
        return viz_rgb
    if viz_rgb.ndim != 3 or viz_rgb.shape[-1] != 3:
        return viz_rgb
    return cv2.cvtColor(viz_rgb, cv2.COLOR_LAB2RGB)


def _histogram_equalize_rgb(viz_rgb: np.ndarray) -> np.ndarray:
    if viz_rgb.ndim != 3 or viz_rgb.shape[-1] < 3:
        return viz_rgb
    out = viz_rgb.copy()
    for ch in range(3):
        out[:, :, ch] = cv2.equalizeHist(out[:, :, ch])
    return out


def _to_uint8_gray01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    return np.uint8(np.round(x * 255.0))


def _save_nmf_component_visualizations(
    out_dir: Path,
    stem: str,
    valid_mask: np.ndarray,
    eval_positions: np.ndarray,
    y_true: np.ndarray,
    y_pred_oof: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = valid_mask.shape
    valid_flat = np.flatnonzero(valid_mask.ravel())
    eval_positions = np.asarray(eval_positions, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred_oof = np.asarray(y_pred_oof, dtype=np.float32)
    if y_true.ndim != 2 or y_pred_oof.ndim != 2 or y_true.shape != y_pred_oof.shape:
        return
    if len(eval_positions) != y_true.shape[0]:
        return

    n_components = y_true.shape[1]

    def _to_colormap(values: np.ndarray) -> np.ndarray:
        mapped = np.zeros(h * w, dtype=np.float32)
        if len(eval_positions) > 0:
            pixel_idx = valid_flat[eval_positions]
            mapped[pixel_idx] = values
        mapped = mapped.reshape(h, w)
        present = mapped[valid_mask]
        if present.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        lo = float(np.min(present))
        hi = float(np.max(present))
        if hi > lo:
            norm = (mapped - lo) / (hi - lo + 1e-8)
        else:
            norm = np.zeros_like(mapped, dtype=np.float32)
        gray = _to_uint8_gray01(norm)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
        color[~valid_mask] = 0
        return color

    for comp_idx in range(n_components):
        gt_img = _to_colormap(y_true[:, comp_idx])
        pred_img = _to_colormap(y_pred_oof[:, comp_idx])
        gt_path = out_dir / f"{stem}__nmf_gt_c{comp_idx:02d}.png"
        pred_path = out_dir / f"{stem}__nmf_pred_c{comp_idx:02d}.png"
        Image.fromarray(gt_img, mode="RGB").save(gt_path)
        Image.fromarray(pred_img, mode="RGB").save(pred_path)


def _save_edge_maps_visualizations(
    out_dir: Path,
    stem: str,
    edge_maps: Dict[str, object],
) -> None:
    if not edge_maps:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in edge_maps.items():
        try:
            x = np.asarray(arr, dtype=np.float32)
            img = _to_uint8_gray01(x)
            Image.fromarray(img, mode="L").save(out_dir / f"{stem}__{key}.png")
        except Exception:
            continue


def _path_identifier(path: Path) -> str:
    # Include parent folders + stem, plus short hash for collision resistance.
    parts = [p for p in path.parts if p not in ("/", "\\", ":", "")]
    core = "__".join(parts[-4:]) if len(parts) >= 4 else "__".join(parts)
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in core)
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{safe}__{digest}"


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value))
    return token.strip("_") or "default"


def _resolve_mics_parameter_groups(cfg: DictConfig) -> List[Tuple[str, Dict[str, Any]]]:
    groups_cfg = getattr(cfg.mics_lmc, "parameter_groups", None)
    groups: List[Tuple[str, Dict[str, Any]]] = []

    if groups_cfg:
        for idx, group in enumerate(groups_cfg, start=1):
            group_name = _safe_token(getattr(group, "name", f"group_{idx}"))
            group_overrides_cfg = getattr(group, "overrides", {})
            group_overrides = OmegaConf.to_container(group_overrides_cfg, resolve=True) if group_overrides_cfg else {}
            if group_overrides is None:
                group_overrides = {}
            if not isinstance(group_overrides, dict):
                raise ValueError(f"mics_lmc.parameter_groups[{idx - 1}].overrides must be a mapping")
            groups.append((group_name, group_overrides))
        return groups

    # Backward-compatible single-group behavior.
    legacy_overrides_cfg = getattr(cfg.mics_lmc, "overrides", {})
    legacy_overrides = OmegaConf.to_container(legacy_overrides_cfg, resolve=True) if legacy_overrides_cfg else {}
    if legacy_overrides is None:
        legacy_overrides = {}
    if not isinstance(legacy_overrides, dict):
        raise ValueError("mics_lmc.overrides must be a mapping")
    groups.append(("default", legacy_overrides))
    return groups


def _build_metric_mask_from_tic(raw_msi_img: np.ndarray, cfg: DictConfig) -> np.ndarray:
    mm_cfg = getattr(cfg.benchmark, "metric_mask", None)
    if mm_cfg is None or not bool(getattr(mm_cfg, "enabled", False)):
        return np.ones(raw_msi_img.shape[:2], dtype=bool)

    tic = raw_msi_img.sum(axis=-1).astype(np.float32)
    nonzero = tic > 0
    if not np.any(nonzero):
        return np.zeros(raw_msi_img.shape[:2], dtype=bool)
    p = float(getattr(mm_cfg, "drop_bottom_tic_percentile", 30.0))
    p = min(max(p, 0.0), 100.0)
    thr = float(np.percentile(tic[nonzero], p))
    mask = (tic > thr) & nonzero
    return mask


def _compute_correlation_metrics(
    msi_img: np.ndarray,
    viz_rgb: np.ndarray,
    cfg: DictConfig,
    metric_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    # User-requested benchmark set: rely on MSIVisualizationMetrics output,
    # which includes ZADU metrics and correlation/trustworthiness metrics.
    out: Dict[str, float] = {}
    pair = MSIVisualizationMetrics(
        msi_img,
        viz_rgb,
        mask=metric_mask,
        num_samples=int(cfg.pairwise_num_samples),
    ).get_metrics()
    skip_saliency_smoothness = bool(getattr(cfg, "skip_saliency_smoothness", False))
    for key, value in pair.items():
        try:
            metric_key = str(key).replace(" ", "_").replace("∞", "inf").replace("%", "pct").lower()
            if skip_saliency_smoothness and ("saliency" in metric_key or "smoothness" in metric_key):
                continue
            out[metric_key] = float(value)
        except Exception:
            continue
    return out


def _flatten_debug_metrics(result: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    summary = result.get("summary", {})
    if isinstance(summary, dict):
        for k, v in summary.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
    for bin_row in result.get("frequency_bins", []):
        if not isinstance(bin_row, dict):
            continue
        bin_name = str(bin_row.get("bin", "unknown"))
        for k, v in bin_row.items():
            if k == "bin":
                continue
            key = f"bin_{bin_name}__{k}"
            try:
                out[key] = float(v)
            except Exception:
                continue
    return out


def _compute_visualization_quality_metrics(viz_rgb: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    q = _compute_viz_contrast_color_metrics(viz_rgb, valid_mask)
    return {
        "luminance_rms_contrast": float(q.get("luminance_rms_contrast", np.nan)),
        "lab_chroma_entropy": float(q.get("lab_chroma_entropy", np.nan)),
    }


def _finalize_core_benchmark_metrics(row: Dict[str, object], cfg: DictConfig) -> None:
    """Match bayes_tune naming and verify primary objective scalars are present on successful rows.

    - ``continuous_dice`` mirrors ``edge_agreement_continuous_dice`` (from ``edge_evaluation``).
    - ``luminance_rms_contrast`` / ``lab_chroma_entropy`` come from the viz RGB (LAB contrast/entropy).
    """
    if str(row.get("status", "")).strip().lower() != "ok":
        return
    ee_metric = str(OmegaConf.select(cfg, "edge_evaluation.metric", default="continuous_dice")).strip().lower()
    if ee_metric == "continuous_dice":
        row["continuous_dice"] = _to_float_or_nan(row.get("edge_agreement_continuous_dice"))
    else:
        row["continuous_dice"] = float("nan")
    row["luminance_rms_contrast"] = _to_float_or_nan(row.get("luminance_rms_contrast"))
    row["lab_chroma_entropy"] = _to_float_or_nan(row.get("lab_chroma_entropy"))

    ee_on = bool(OmegaConf.select(cfg, "edge_evaluation.enabled", default=True))
    if ee_on and ee_metric == "continuous_dice":
        if np.isnan(float(row["continuous_dice"])):
            logger.warning(
                "Missing finite continuous Dice for %s | method=%s "
                "(expect edge_agreement_continuous_dice when edge_evaluation.metric is continuous_dice)",
                row.get("path"),
                row.get("method"),
            )
    for name in ("luminance_rms_contrast", "lab_chroma_entropy"):
        if np.isnan(float(row[name])):
            logger.warning(
                "Missing finite %s for %s | method=%s",
                name,
                row.get("path"),
                row.get("method"),
            )


def _mics_kwargs(
    cfg: DictConfig,
    sampling_mode: str,
    run_seed: int,
    group_overrides: Dict[str, Any] | None = None,
) -> Dict[str, object]:
    defaults_path = Path(str(cfg.mics_lmc.defaults_yaml))
    if not defaults_path.exists():
        raise FileNotFoundError(f"MiCS-LMC defaults yaml not found: {defaults_path}")
    defaults_cfg = OmegaConf.load(defaults_path)
    model_cfg = OmegaConf.to_container(defaults_cfg.model, resolve=True)
    overrides = dict(group_overrides or {})
    model_cfg.update(overrides)

    shared_sampling = getattr(cfg.benchmark, "shared_sampling", None)
    enforce_same_pca_fit_step = bool(getattr(shared_sampling, "enforce_same_pca_fit_step", True)) if shared_sampling is not None else False
    shared_pca_fit_step = int(getattr(shared_sampling, "pca_fit_step", model_cfg.get("pca_fit_step", 4))) if shared_sampling is not None else int(model_cfg.get("pca_fit_step", 4))
    enforce_same_sampling_seed = bool(getattr(shared_sampling, "enforce_same_sampling_seed", True)) if shared_sampling is not None else False

    kwargs = {
        "number_of_points": int(model_cfg.get("number_of_points", 1000)),
        "sampling": str(model_cfg.get("sampling", "coreset")),
        "num_epochs": int(model_cfg.get("num_epochs", 100)),
        "number_of_components": int(model_cfg.get("number_of_components", 3)),
        "num_layers": int(model_cfg.get("num_layers", 2)),
        "lab_to_rgb": bool(model_cfg.get("lab_to_rgb", True)),
        "cluster": bool(model_cfg.get("cluster", False)),
        "clusters": model_cfg.get("clusters", None),
        "num_samples": int(cfg.benchmark.num_samples),
        "beta": float(model_cfg.get("beta", 1.0)),
        "k_epoch": int(model_cfg.get("k_epoch", 1)),
        "temperature": float(model_cfg.get("temperature", 10.0)),
        "lr": float(model_cfg.get("lr", 1.0)),
        "batch_size": int(model_cfg.get("batch_size", 2048)),
        "warmup_epochs": int(model_cfg.get("warmup_epochs", 10)),
        "factor": float(model_cfg.get("factor", 1.0)),
        "random_state": int(run_seed) if enforce_same_sampling_seed else int(model_cfg.get("random_state", 42)),
        "verbose": bool(model_cfg.get("verbose", False)),
        "cluster_loss_weight": float(model_cfg.get("cluster_loss_weight", 1.0)),
        "category_loss_weight": float(model_cfg.get("category_loss_weight", 1.0)),
        "pixel_sampling": str(sampling_mode),
        "pca_fit_step": int(shared_pca_fit_step) if enforce_same_pca_fit_step else int(model_cfg.get("pca_fit_step", 4)),
        "cluster_on_pca": bool(model_cfg.get("cluster_on_pca", False)),
        "cluster_pca_dims": int(model_cfg.get("cluster_pca_dims", 50)),
        "predict_percentile_low": float(model_cfg.get("predict_percentile_low", 0.001)),
        "predict_percentile_high": float(model_cfg.get("predict_percentile_high", 99.999)),
        "predict_spatial_smooth_sigma": float(
            model_cfg.get("predict_spatial_smooth_sigma", 0.0),
        ),
    }
    cat = model_cfg.get("clusters_auto_tune", None)
    if cat is not None and str(cat).strip().lower() not in (
        "",
        "null",
        "~",
        "none",
        "false",
    ):
        kwargs["clusters_auto_tune"] = str(cat).strip().lower()
        kwargs["clusters_auto_tune_k_min"] = int(
            model_cfg.get("clusters_auto_tune_k_min", 2),
        )
        kwargs["clusters_auto_tune_k_max"] = int(
            model_cfg.get("clusters_auto_tune_k_max", 256),
        )
        kwargs["clusters_auto_tune_n_candidates"] = int(
            model_cfg.get("clusters_auto_tune_n_candidates", 20),
        )
        k_step = model_cfg.get("clusters_auto_tune_k_step", None)
        try:
            k_step_i = int(k_step) if k_step is not None else None
        except (TypeError, ValueError):
            k_step_i = None
        if k_step_i is not None and int(k_step_i) >= 1:
            kwargs["clusters_auto_tune_k_step"] = int(k_step_i)

    if kwargs["cluster"] and isinstance(kwargs["clusters"], str):
        kwargs["clusters"] = [int(x) for x in kwargs["clusters"].split("-") if x]
    return kwargs


def _write_rows_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows_csv(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _row_key(path: str, sampling_mode: str, method: str, repeat_idx: int, seed: int) -> Tuple[str, str, str, int, int]:
    return (path, sampling_mode, method, int(repeat_idx), int(seed))


def _row_needs_edge_backfill(row: Dict[str, object], backfill_columns: List[str]) -> bool:
    if str(row.get("status", "")).strip().lower() != "ok":
        return False
    vp = row.get("viz_path")
    if not vp or not Path(str(vp)).exists():
        return False
    for col in backfill_columns:
        if col not in row:
            return True
        val = row[col]
        if val is None or (isinstance(val, str) and not str(val).strip()):
            return True
        try:
            if np.isnan(float(val)):
                return True
        except (TypeError, ValueError):
            return True
    return False


def _backfill_edge_evaluation_metrics(rows: List[Dict[str, object]], cfg: DictConfig) -> List[Dict[str, object]]:
    """
    For rows missing edge_evaluation columns, load MSI once per path (HD edges in memory),
    then load each viz and run evaluate_edge_evaluation_only (no NMF).
    """
    backfill_cols = [
        str(c)
        for c in         getattr(
            cfg.benchmark,
            "backfill_edge_columns",
            [
                "edge_agreement_continuous_dice",
                "luminance_rms_contrast",
                "lab_chroma_entropy",
            ],
        )
    ]
    _bf_stage(1, f"Scanning {len(rows)} CSV row(s) for missing columns {backfill_cols}")
    by_path: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        if not _row_needs_edge_backfill(row, backfill_cols):
            continue
        p = str(row.get("path", "")).strip()
        if p:
            by_path.setdefault(p, []).append(i)
    if not by_path:
        _bf_stage(2, f"No rows need updates (all present for {backfill_cols})")
        logger.info("backfill_edge_metrics: no rows need updates for columns %s", backfill_cols)
        return rows

    n_rows_to_touch = sum(len(ix) for ix in by_path.values())
    _bf_stage(
        2,
        f"Will update {n_rows_to_touch} row(s) across {len(by_path)} MSI file(s); columns={backfill_cols}",
    )

    n_updated = 0
    for msi_i, (npy_path_str, row_indices) in enumerate(by_path.items(), start=1):
        npy_path = Path(npy_path_str)
        if not npy_path.exists():
            logger.warning("backfill_edge_metrics: MSI missing, skip %s", npy_path)
            _bf_stage(3, f"[{msi_i}/{len(by_path)}] SKIP — MSI not found: {npy_path}")
            continue
        _bf_stage(
            3,
            f"[{msi_i}/{len(by_path)}] Loading MSI + building HD-edge evaluator (once per MSI): {npy_path.name}",
        )
        try:
            raw_msi_img = _load_raw_msi(npy_path, bool(cfg.data.transpose_msi))
            msi_img = _load_msi(npy_path, bool(cfg.data.transpose_msi), bool(cfg.data.tic_normalize))
            metric_mask = _build_metric_mask_from_tic(raw_msi_img, cfg)
            metric_mask_effective = metric_mask & (msi_img.sum(axis=-1) > 0)
            edge_eval = DebugNMFEvaluator(msi_img, cfg, edge_only=True)
        except Exception as e:
            logger.warning("backfill_edge_metrics: failed to build evaluator for %s: %s", npy_path, e)
            _bf_stage(3, f"  FAILED to build evaluator: {e}")
            continue

        for j, row_idx in enumerate(row_indices, start=1):
            row = rows[row_idx]
            try:
                viz_path = Path(str(row["viz_path"]))
                _bf_stage(
                    3,
                    f"  viz {j}/{len(row_indices)} row#{row_idx} | {viz_path.name}",
                )
                viz_rgb = _load_existing_viz_rgb(viz_path, expected_hw=msi_img.shape[:2])
                updates = edge_eval.evaluate_edge_evaluation_only(viz_rgb)
                updates.update(_compute_visualization_quality_metrics(viz_rgb, metric_mask_effective))
                for k, v in updates.items():
                    if isinstance(v, str):
                        row[k] = v
                    else:
                        row[k] = float(v)
                _finalize_core_benchmark_metrics(row, cfg)
                n_updated += 1
            except Exception as e:
                logger.warning("backfill_edge_metrics: row %s viz %s: %s", row_idx, row.get("viz_path"), e)
                _bf_stage(3, f"  ERROR row#{row_idx}: {e}")

    _bf_stage(4, f"Finished edge backfill | {n_updated} row(s) updated")
    logger.info("backfill_edge_metrics: updated %d rows (columns driven by edge_evaluation.metric)", n_updated)
    return rows


def _existing_keys(rows: List[Dict[str, object]]) -> Set[Tuple[str, str, str, int, int]]:
    keys: Set[Tuple[str, str, str, int, int]] = set()
    for row in rows:
        try:
            status = str(row.get("status", "")).strip().lower()
            if status not in {"ok", "failed"}:
                continue
            keys.add(
                _row_key(
                    str(row["path"]),
                    str(row["sampling_mode"]),
                    str(row["method"]),
                    int(row.get("repeat_idx", 0)),
                    int(row.get("seed", 0)),
                )
            )
        except Exception:
            continue
    return keys


def _to_float_or_nan(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _write_aggregate_csv(rows: List[Dict[str, object]], aggregate_csv_path: Path) -> None:
    excluded_keys = {
        "path",
        "file_id",
        "sampling_mode",
        "method",
        "status",
        "error",
        "viz_path",
        "repeat_idx",
        "seed",
    }
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        key = (str(row.get("path", "")), str(row.get("sampling_mode", "")), str(row.get("method", "")))
        grouped.setdefault(key, []).append(row)

    metric_keys: List[str] = []
    for grows in grouped.values():
        for r in grows:
            for k, v in r.items():
                if k in excluded_keys:
                    continue
                fv = _to_float_or_nan(v)
                if not np.isnan(fv) and k not in metric_keys:
                    metric_keys.append(k)

    out_rows: List[Dict[str, object]] = []
    for (path, sampling_mode, method), grows in grouped.items():
        out: Dict[str, object] = {
            "path": path,
            "sampling_mode": sampling_mode,
            "method": method,
            "n_repeats": len(grows),
        }
        for mk in metric_keys:
            vals = np.array([_to_float_or_nan(r.get(mk)) for r in grows], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            out[f"{mk}_mean"] = float(vals.mean())
            out[f"{mk}_std"] = float(vals.std())
        out_rows.append(out)

    _write_rows_csv(out_rows, aggregate_csv_path)


@hydra.main(version_base=None, config_path="configs", config_name="benchmark_parametric_methods")
def main(cfg: DictConfig) -> None:
    if cfg.output.out_dir:
        output_dir = Path(cfg.output.out_dir)
    else:
        from hydra.core.hydra_config import HydraConfig
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(getattr(cfg.benchmark, "backfill_edge_metrics_only", False)):
        csv_path = output_dir / str(cfg.output.results_csv)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"backfill_edge_metrics_only: results CSV not found: {csv_path}. "
                "Set output.out_dir to the run folder that contains benchmark_parametric_methods.csv"
            )
        print(f"[backfill_edge_metrics] mode=backfill_edge_metrics_only | csv={csv_path}", flush=True)
        rows = _load_rows_csv(csv_path)
        rows = _backfill_edge_evaluation_metrics(rows, cfg)
        print("[backfill_edge_metrics] Writing results CSV and aggregate...", flush=True)
        _write_rows_csv(rows, csv_path)
        aggregate_csv_path = output_dir / str(cfg.output.aggregate_csv)
        _write_aggregate_csv(rows, aggregate_csv_path)
        print(f"Backfill complete. Updated {csv_path} and {aggregate_csv_path}")
        return

    if not cfg.data.npy_paths:
        raise ValueError("data.npy_paths is empty. Provide a list of .npy files.")
    viz_dir = output_dir / str(cfg.output.visualizations_subdir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    tic_dir = output_dir / str(cfg.output.tic_subdir)
    nmf_components_dir = output_dir / str(getattr(cfg.output, "nmf_components_subdir", "nmf_components"))
    edge_maps_dir = output_dir / str(getattr(cfg.output, "edge_maps_subdir", "edge_maps"))
    if bool(getattr(cfg.output, "save_tic_image", True)):
        tic_dir.mkdir(parents=True, exist_ok=True)
    if bool(getattr(cfg.output, "save_nmf_component_visualizations", True)):
        nmf_components_dir.mkdir(parents=True, exist_ok=True)
    if bool(getattr(cfg.output, "save_edge_maps_visualizations", False)):
        edge_maps_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / str(cfg.output.results_csv)
    aggregate_csv_path = output_dir / str(cfg.output.aggregate_csv)
    resolved_path = output_dir / "config_resolved.yaml"
    with resolved_path.open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    rows: List[Dict[str, object]] = []
    if bool(cfg.benchmark.resume_from_existing_csv):
        rows = _load_rows_csv(csv_path)
    completed_keys = _existing_keys(rows)
    has_mics_parameter_groups = bool(getattr(cfg.mics_lmc, "parameter_groups", None))
    mics_groups = _resolve_mics_parameter_groups(cfg)
    mics_overrides_by_group = {group_name: overrides for group_name, overrides in mics_groups}

    total_paths = len(cfg.data.npy_paths)
    for path_idx, npy_path_str in enumerate(cfg.data.npy_paths, start=1):
        npy_path = Path(str(npy_path_str))
        print(f"[next {path_idx}/{total_paths}] {npy_path}")
        if not npy_path.exists():
            rows.append(
                {
                    "status": "failed",
                    "path": str(npy_path),
                    "method": "all",
                    "sampling_mode": "all",
                    "error": "MSI path not found",
                }
            )
            _write_rows_csv(rows, csv_path)
            continue

        try:
            raw_msi_img = _load_raw_msi(npy_path, bool(cfg.data.transpose_msi))
            msi_img = _load_msi(npy_path, bool(cfg.data.transpose_msi), bool(cfg.data.tic_normalize))
            metric_mask = _build_metric_mask_from_tic(raw_msi_img, cfg)
            debug_eval = DebugNMFEvaluator(msi_img, cfg)
            msi_mask = msi_img.sum(axis=-1) > 0
            metric_mask_effective = metric_mask & msi_mask
            metric_mask_fraction = float(metric_mask_effective.mean())
            metric_mask_pixels = int(metric_mask_effective.sum())
            if bool(getattr(cfg.output, "save_tic_image", True)):
                file_id = _path_identifier(npy_path)
                # TIC visualization should be computed from raw MSI (before TIC normalization).
                tic = raw_msi_img.sum(axis=-1).astype(np.float32)
                tic_min = float(np.min(tic))
                tic_max = float(np.max(tic))
                if tic_max > tic_min:
                    tic01 = (tic - tic_min) / (tic_max - tic_min)
                else:
                    tic01 = np.zeros_like(tic, dtype=np.float32)
                tic_img = _to_uint8_gray01(tic01)
                Image.fromarray(tic_img, mode="L").save(tic_dir / f"{file_id}__tic01.png")
        except Exception as e:
            rows.append(
                {
                    "status": "failed",
                    "path": str(npy_path),
                    "method": "all",
                    "sampling_mode": "all",
                    "error": f"data_prepare_failed: {e}",
                }
            )
            _write_rows_csv(rows, csv_path)
            if not bool(cfg.benchmark.continue_on_error):
                raise
            continue

        for repeat_idx in range(int(cfg.benchmark.repeats)):
            seed = int(cfg.benchmark.seed_start) + repeat_idx * int(cfg.benchmark.seed_step)
            _seed_everything(seed)

            for sampling_mode in cfg.benchmark.sampling_modes:
                methods: List[Tuple[str, str]] = [
                    ("parametric_mics_lmc", group_name) for group_name, _ in mics_groups
                ]
                if bool(getattr(cfg.parametric_umap, "enabled", True)):
                    methods.append(("parametric_umap", "default"))
                if bool(getattr(cfg.pca, "enabled", True)):
                    methods.append(("pca", "default"))
                if bool(getattr(cfg.top3, "enabled", False)):
                    methods.append(("top3", "default"))
                if bool(getattr(cfg.nmf_3d, "enabled", False)):
                    methods.append(("nmf_3d", "default"))

                for method_name, method_variant in methods:
                    method_tag = method_name
                    if method_name == "parametric_mics_lmc" and has_mics_parameter_groups:
                        method_tag = f"{method_name}__{method_variant}"
                    method_file_tag = _safe_token(method_tag)
                    key = _row_key(str(npy_path), str(sampling_mode), method_tag, repeat_idx, seed)
                    if key in completed_keys:
                        print(
                            f"[skip] {npy_path.name} | repeat={repeat_idx} seed={seed} | "
                            f"{sampling_mode} | {method_tag}"
                        )
                        continue

                    row: Dict[str, object] = {
                        "path": str(npy_path),
                        "file_id": _path_identifier(npy_path),
                        "sampling_mode": str(sampling_mode),
                        "method": method_tag,
                        "repeat_idx": int(repeat_idx),
                        "seed": int(seed),
                        "sampling_num_samples": int(cfg.benchmark.num_samples),
                        "metric_mask_fraction": metric_mask_fraction,
                        "metric_mask_pixels": metric_mask_pixels,
                    }
                    if method_name == "parametric_mics_lmc":
                        row["mics_group"] = method_variant
                    t0 = time.perf_counter()
                    model = None
                    try:
                        viz_path = viz_dir / f"{row['file_id']}__r{repeat_idx}__s{seed}__{sampling_mode}__{method_file_tag}.png"
                        reuse_existing_viz = bool(getattr(cfg.output, "reuse_existing_visualizations", True))
                        equalize_visualizations = bool(getattr(cfg.output, "equalize_visualizations", True))
                        if method_name == "parametric_mics_lmc":
                            mics_kwargs = _mics_kwargs(
                                cfg,
                                str(sampling_mode),
                                seed,
                                group_overrides=mics_overrides_by_group.get(method_variant, {}),
                            )
                            row["sampling_seed_used"] = int(mics_kwargs.get("random_state", seed))
                            row["sampling_pca_fit_step_used"] = int(mics_kwargs.get("pca_fit_step", -1))
                        elif method_name == "parametric_umap":
                            umap_kwargs = _umap_kwargs(cfg, str(sampling_mode), seed)
                            row["sampling_seed_used"] = int(umap_kwargs.get("random_state", seed))
                            row["sampling_pca_fit_step_used"] = int(umap_kwargs.get("pca_fit_step", -1))
                        elif method_name == "pca":
                            row["sampling_seed_used"] = int(seed)
                            row["sampling_pca_fit_step_used"] = -1
                        elif method_name == "top3":
                            row["sampling_seed_used"] = int(seed)
                            row["sampling_pca_fit_step_used"] = -1
                        elif method_name == "nmf_3d":
                            row["sampling_seed_used"] = int(seed)
                            row["sampling_pca_fit_step_used"] = -1
                        else:
                            raise ValueError(f"Unknown method: {method_name}")

                        if reuse_existing_viz and viz_path.exists():
                            viz_rgb = _load_existing_viz_rgb(viz_path, expected_hw=msi_img.shape[:2])
                            row["viz_reused"] = 1
                            row["time_seconds"] = 0.0
                        else:
                            if method_name == "parametric_mics_lmc":
                                model = MSIParametricMiCSLMC(**mics_kwargs)
                                model.fit(msi_img)
                                viz = model.predict(msi_img)
                            elif method_name == "parametric_umap":
                                model = UMAPVirtualStain(**umap_kwargs)
                                model.fit([msi_img], keras_fit_kwargs={})
                                viz = model.predict(msi_img)
                            elif method_name == "pca":
                                model = PCA3D()
                                model.fit([msi_img])
                                viz = model.predict(msi_img)
                            elif method_name == "top3":
                                model = TOP3()
                                viz = model(msi_img)
                            elif method_name == "nmf_3d":
                                nmf_cfg = getattr(cfg, "nmf_3d", None)
                                model = NMF3D(
                                    n_components=int(getattr(nmf_cfg, "n_components", 3)),
                                    init=str(getattr(nmf_cfg, "init", "nndsvda")),
                                    max_iter=int(getattr(nmf_cfg, "max_iter", 200)),
                                    random_state=int(seed),
                                )
                                model.fit([msi_img])
                                viz = model.predict(msi_img)
                            else:
                                raise ValueError(f"Unknown method: {method_name}")

                            viz_rgb = _to_uint8_rgb(viz)
                            viz_rgb = _maybe_lab_to_rgb(viz_rgb, method_name, cfg)
                            if equalize_visualizations:
                                viz_rgb = _histogram_equalize_rgb(viz_rgb)
                            if method_name in {"parametric_umap", "pca", "nmf_3d", "top3"}:
                                viz_rgb = viz_rgb.copy()
                                viz_rgb[~msi_mask] = 0
                            Image.fromarray(viz_rgb, mode="RGB").save(viz_path)
                            row["viz_reused"] = 0
                            row["time_seconds"] = float(time.perf_counter() - t0)

                        row["status"] = "ok"
                        row["viz_path"] = str(viz_path)

                        corr_metrics = _compute_correlation_metrics(
                            msi_img,
                            viz_rgb,
                            cfg.correlation,
                            metric_mask=metric_mask_effective,
                        )
                        row.update(corr_metrics)
                        row.update(_compute_visualization_quality_metrics(viz_rgb, metric_mask_effective))

                        save_edge_maps = bool(getattr(cfg.output, "save_edge_maps_visualizations", False))
                        dbg_result, _ = debug_eval.evaluate_visualization(
                            viz_rgb,
                            return_component_maps=False,
                            return_edge_maps=save_edge_maps,
                        )
                        row.update(_flatten_debug_metrics(dbg_result))
                        if save_edge_maps:
                            edge_stem = f"{row['file_id']}__r{repeat_idx}__{sampling_mode}__{method_file_tag}"
                            _save_edge_maps_visualizations(
                                edge_maps_dir,
                                edge_stem,
                                dbg_result.get("edge_maps", {}) if isinstance(dbg_result, dict) else {},
                            )
                            row["edge_maps_dir"] = str(edge_maps_dir)
                        _finalize_core_benchmark_metrics(row, cfg)
                    except Exception as e:
                        row["status"] = "failed"
                        row["error"] = str(e)
                        row["time_seconds"] = float(time.perf_counter() - t0)
                        if not bool(cfg.benchmark.continue_on_error):
                            rows.append(row)
                            completed_keys.add(key)
                            _write_rows_csv(rows, csv_path)
                            _write_aggregate_csv(rows, aggregate_csv_path)
                            raise
                    finally:
                        if model is not None and hasattr(model, "release_resources"):
                            try:
                                model.release_resources()
                            except Exception:
                                pass
                        del model
                        gc.collect()

                    rows.append(row)
                    completed_keys.add(key)
                    _write_rows_csv(rows, csv_path)
                    _write_aggregate_csv(rows, aggregate_csv_path)
                    print(
                        f"[{row['status']}] {npy_path.name} | repeat={repeat_idx} seed={seed} | "
                        f"{sampling_mode} | {method_tag} | time={row['time_seconds']:.2f}s"
                    )

    _write_aggregate_csv(rows, aggregate_csv_path)
    print(f"Benchmark complete. Results CSV: {csv_path}")
    print(f"Aggregate CSV: {aggregate_csv_path}")


if __name__ == "__main__":
    main()

