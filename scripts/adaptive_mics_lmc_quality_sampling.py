#!/usr/bin/env python3
"""
Adaptive MiCS-LMC sampling driven by local quality.

Loop:
  1. Train a MiCS2-style MiCS-LMC model.
  2. Score its visualization with local patch quality:
     weighted continuous Dice against HD edges + luminance RMS contrast.
  3. Convert low-quality regions into the next iteration's weighted sampling map.
  4. Save per-iteration visualization, local quality image, sampling map, CSV.
  5. Build a gallery greedily: first global-best image, then images maximizing
     spatial coverage via max(selected_quality_maps).
"""
from __future__ import annotations

import csv
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark_parametric_methods import _resolve_mics_parameter_groups  # noqa: E402
from bayes_tune_mics_lmc import (  # noqa: E402
    _base_mics_kwargs,
    _compute_viz_edge_n,
    _propose_params,
    _search_space_items,
    _to_uint8_rgb,
)
from debug_edge_maps import (  # noqa: E402
    _compute_continuous_dice,
    _compute_viz_contrast_color_metrics,
    _edges_for_continuous_metrics,
    _gray_u8_to_rgb_colormap,
    _load_msi,
    _normalize_map_percentile,
    _normalize_on_mask,
    _parse_edge_detection_cfg,
    _pca_project_msi_for_edges,
    _resolve_normalization_mode,
    _to_uint8_gray01,
)
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method  # noqa: E402

logger = logging.getLogger(__name__)


def _stage(msg: str) -> None:
    print(f"[adaptive_mics] {msg}", flush=True)


def _resolve_path(p: str | Path) -> Path:
    return Path(to_absolute_path(str(p)))


def _global_score(viz_rgb: np.ndarray, hd_edge_n: np.ndarray, valid_mask: np.ndarray, rank_cfg: DictConfig) -> dict[str, float]:
    viz_edge_n = _compute_viz_edge_n(viz_rgb, valid_mask, rank_cfg)
    square = bool(OmegaConf.select(rank_cfg, "evaluation.square_before_continuous_metrics", default=False))
    hd_d, v_d = _edges_for_continuous_metrics(hd_edge_n, viz_edge_n, square=square)
    dice = float(_compute_continuous_dice(hd_d, v_d, valid_mask))
    cq = _compute_viz_contrast_color_metrics(viz_rgb, valid_mask)
    return {
        "continuous_dice": dice,
        "luminance_rms_contrast": float(cq["luminance_rms_contrast"]),
    }


def _patch_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, max(1, length - patch + 1), stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def _minmax_on_mask(x: np.ndarray, mask: np.ndarray, default: float = 0.5) -> np.ndarray:
    out = np.full_like(x, default, dtype=np.float32)
    vals = np.asarray(x, dtype=np.float32)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo + 1e-12:
        out[mask] = default
    else:
        out[mask] = np.clip((x[mask] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _rank_scores_higher_better(values: list[float]) -> np.ndarray:
    """Tie-aware average-rank scores in [0,1], matching rank_visualizations composite semantics."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    idx = np.nonzero(finite)[0]
    sub = arr[finite]
    order = np.argsort(-sub, kind="mergesort")
    sorted_vals = sub[order]
    ranks_ordered = np.empty_like(sorted_vals, dtype=np.float64)
    n = int(sorted_vals.shape[0])
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks_ordered[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    inv = np.empty_like(order)
    inv[order] = np.arange(n, dtype=np.int64)
    ranks = ranks_ordered[inv]
    denom = float(max(n - 1, 1))
    out[idx] = 1.0 - (ranks - 1.0) / denom
    return out


def _apply_final_global_composite(rows: list[dict[str, Any]], dice_weight: float, contrast_weight: float) -> None:
    dice_scores = _rank_scores_higher_better([float(r["global_continuous_dice"]) for r in rows])
    contrast_scores = _rank_scores_higher_better([float(r["global_luminance_rms_contrast"]) for r in rows])
    denom = max(1e-12, float(abs(dice_weight) + abs(contrast_weight)))
    for i, r in enumerate(rows):
        if not np.isfinite(dice_scores[i]) or not np.isfinite(contrast_scores[i]):
            r["global_composite"] = float("nan")
            continue
        r["global_composite"] = (
            float(dice_weight) * float(dice_scores[i])
            + float(contrast_weight) * float(contrast_scores[i])
        ) / denom


def _combined_score_from_raw(dice: float, contrast: float, contrast_scale: float, dice_weight: float, contrast_weight: float) -> float:
    denom = max(1e-12, float(abs(dice_weight) + abs(contrast_weight)))
    c = float(contrast) / max(float(contrast_scale), 1e-12)
    return (float(dice_weight) * float(dice) + float(contrast_weight) * c) / denom


def _refresh_bayes_observation_scores(
    observations: list[dict[str, Any]],
    dice_weight: float,
    contrast_weight: float,
) -> None:
    if not observations:
        return
    cmax = max([float(o.get("global_luminance_rms_contrast", 0.0)) for o in observations] + [1e-12])
    for o in observations:
        o["score"] = _combined_score_from_raw(
            float(o["global_continuous_dice"]),
            float(o["global_luminance_rms_contrast"]),
            cmax,
            dice_weight,
            contrast_weight,
        )


def _local_quality_map(
    viz_rgb: np.ndarray,
    hd_edge_n: np.ndarray,
    valid_mask: np.ndarray,
    rank_cfg: DictConfig,
    *,
    patch_size: int,
    patch_stride: int,
    min_valid_fraction: float,
    dice_weight: float,
    contrast_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (combined_quality, local_dice, local_contrast_norm), each HxW float32."""
    H, W = valid_mask.shape
    patch = max(1, int(patch_size))
    stride = max(1, int(patch_stride))
    viz_edge_n = _compute_viz_edge_n(viz_rgb, valid_mask, rank_cfg)
    square = bool(OmegaConf.select(rank_cfg, "evaluation.square_before_continuous_metrics", default=False))
    hd_d, v_d = _edges_for_continuous_metrics(hd_edge_n, viz_edge_n, square=square)
    lum = np.asarray(viz_rgb, dtype=np.float32)
    if lum.dtype == np.uint8:
        lum = lum / 255.0
    elif float(np.nanmax(lum)) > 1.0 + 1e-6:
        lum = lum / 255.0
    y = (0.299 * lum[..., 0] + 0.587 * lum[..., 1] + 0.114 * lum[..., 2]).astype(np.float32)

    dice_acc = np.zeros((H, W), dtype=np.float32)
    contrast_acc = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    min_valid = max(1, int(round(patch * patch * float(min_valid_fraction))))

    for y0 in _patch_starts(H, patch, stride):
        y1 = min(H, y0 + patch)
        for x0 in _patch_starts(W, patch, stride):
            x1 = min(W, x0 + patch)
            m = valid_mask[y0:y1, x0:x1]
            if int(m.sum()) < min_valid:
                continue
            d = float(_compute_continuous_dice(hd_d[y0:y1, x0:x1], v_d[y0:y1, x0:x1], m))
            c = float(np.std(y[y0:y1, x0:x1][m]))
            dice_acc[y0:y1, x0:x1] += d
            contrast_acc[y0:y1, x0:x1] += c
            count[y0:y1, x0:x1] += 1.0

    covered = (count > 0) & valid_mask
    local_dice = np.zeros((H, W), dtype=np.float32)
    local_contrast = np.zeros((H, W), dtype=np.float32)
    local_dice[covered] = dice_acc[covered] / count[covered]
    local_contrast[covered] = contrast_acc[covered] / count[covered]
    contrast_norm = _minmax_on_mask(local_contrast, covered, default=0.5)

    denom = max(1e-12, float(abs(dice_weight) + abs(contrast_weight)))
    quality = (float(dice_weight) * local_dice + float(contrast_weight) * contrast_norm) / denom
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)
    quality[~valid_mask] = 0.0
    return quality, local_dice, contrast_norm


def _badness_sampling_map(
    quality: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gamma: float,
    floor: float,
) -> np.ndarray:
    bad = np.power(np.clip(1.0 - quality, 0.0, 1.0), max(0.1, float(gamma))).astype(np.float32)
    bad = np.where(valid_mask, bad + float(floor), 0.0).astype(np.float32)
    return bad


def _save_map_png(path: Path, arr: np.ndarray, valid_mask: np.ndarray, colormap: str = "viridis") -> None:
    x = np.asarray(arr, dtype=np.float32).copy()
    x[~valid_mask] = 0.0
    u8 = _to_uint8_gray01(x)
    try:
        rgb = _gray_u8_to_rgb_colormap(u8, colormap)
    except Exception:
        rgb = np.stack([u8] * 3, axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def _save_viz(path: Path, viz_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8_rgb(viz_rgb), mode="RGB").save(path)


def _copy_preview(src: Path, dst: Path) -> None:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as exc:
        logger.warning("Could not copy preview %s -> %s: %s", src, dst, exc)


def _append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _resolve_mics2_kwargs(
    cfg: DictConfig,
    seed: int,
    sampling_map: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bench_cfg = OmegaConf.load(_resolve_path(str(cfg.mics.benchmark_config)))
    if hasattr(bench_cfg.mics_lmc, "defaults_yaml"):
        bench_cfg.mics_lmc.defaults_yaml = to_absolute_path(str(bench_cfg.mics_lmc.defaults_yaml))
    groups = dict(_resolve_mics_parameter_groups(bench_cfg))
    group_name = str(cfg.mics.parameter_group)
    if group_name not in groups:
        raise ValueError(f"mics.parameter_group={group_name!r} not found. Available: {sorted(groups)}")
    train_cfg = OmegaConf.load(_resolve_path(str(bench_cfg.mics_lmc.defaults_yaml)))
    overrides = dict(groups[group_name])
    if params:
        overrides.update(params)
    extra = OmegaConf.to_container(getattr(cfg.mics, "overrides", {}), resolve=True) or {}
    overrides.update(extra)
    kwargs = _base_mics_kwargs(train_cfg, overrides, seed)
    if sampling_map is not None:
        kwargs["pixel_sampling"] = "weighted"
        kwargs["sampling_weight_map"] = sampling_map
        kwargs["weighted_sampling_uniform_mix"] = float(cfg.adaptive.uniform_sampling_mix)
    return kwargs


def _compute_hd_edge(cfg: DictConfig, rank_cfg: DictConfig, msi: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    edge_cfg = _parse_edge_detection_cfg(rank_cfg)
    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    if multi is None or not bool(getattr(multi, "enabled", False)):
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    hd_cfg = getattr(rank_cfg, "hd_edges", None)
    spatial_enabled = bool(getattr(rank_cfg.spatial_normalization, "enabled", True))
    return _compute_hd_edge_n_for_method(
        rank_cfg,
        msi,
        valid_mask,
        hd_cfg,
        str(edge_cfg.get("hd_method", "gradient")),
        sigmas,
        aggregation,
        spatial_enabled,
    )


def _select_coverage_gallery(rows: list[dict[str, Any]], maps: list[np.ndarray], valid_mask: np.ndarray, n: int) -> list[int]:
    valid_ids = [i for i, r in enumerate(rows) if np.isfinite(float(r["global_composite"]))]
    if not valid_ids:
        return []
    first = max(valid_ids, key=lambda i: float(rows[i]["global_composite"]))
    selected = [first]
    current = maps[first].copy()
    valid = valid_mask.astype(bool)
    while len(selected) < min(int(n), len(valid_ids)):
        best_i = None
        best_score = -np.inf
        for i in valid_ids:
            if i in selected:
                continue
            merged = np.maximum(current, maps[i])
            score = float(np.mean(merged[valid])) if np.any(valid) else float("-inf")
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        current = np.maximum(current, maps[best_i])
    return selected


def _save_gallery(rows: list[dict[str, Any]], selected: list[int], out_path: Path, columns: int = 4) -> None:
    if not selected:
        return
    imgs: list[tuple[Image.Image, str]] = []
    for rank, idx in enumerate(selected, start=1):
        r = rows[idx]
        with Image.open(r["viz_path"]) as im:
            label = f"#{rank} it={r['iteration']} comp={float(r['global_composite']):.4f}"
            imgs.append((im.convert("RGB"), label))
    cell_w = max(im.width for im, _ in imgs)
    cell_h = max(im.height for im, _ in imgs)
    scale = min(360 / cell_w, 360 / cell_h, 1.0)
    label_h = 24
    pad = 6
    resized = []
    for im, label in imgs:
        if scale < 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.LANCZOS)
        resized.append((im, label))
    cell_w = max(im.width for im, _ in resized) + 2 * pad
    cell_h = max(im.height for im, _ in resized) + 2 * pad + label_h
    cols = max(1, int(columns))
    n_rows = int(np.ceil(len(resized) / cols))
    canvas = Image.new("RGB", (cols * cell_w, n_rows * cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for i, (im, label) in enumerate(resized):
        rr, cc = divmod(i, cols)
        x0, y0 = cc * cell_w, rr * cell_h
        x = x0 + (cell_w - im.width) // 2
        y = y0 + pad
        canvas.paste(im, (x, y))
        draw.text((x0 + pad, y + im.height + 2), label, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


@hydra.main(version_base=None, config_path="configs", config_name="adaptive_mics_lmc_quality_sampling")
def main(cfg: DictConfig) -> None:
    log_level = str(getattr(getattr(cfg, "logging", None), "level", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    out_dir = Path(str(cfg.output.out_dir)) if cfg.output.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    _stage(f"Output directory: {out_dir.resolve()}")

    rank_cfg = OmegaConf.load(_resolve_path(str(cfg.rank_config)))
    npy_path = _resolve_path(str(cfg.input.npy_path))
    normalization = _resolve_normalization_mode(getattr(rank_cfg.data, "normalization", "tic"))
    msi = _load_msi(npy_path, transpose_msi=bool(rank_cfg.data.transpose_msi), normalization=normalization)
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty.")
    _stage(f"Loaded MSI {tuple(msi.shape)} valid={int(valid_mask.sum())}")

    _stage("Computing HD edge target...")
    hd_edge_n = _compute_hd_edge(cfg, rank_cfg, msi, valid_mask)
    np.save(out_dir / "hd_edge_n.npy", hd_edge_n.astype(np.float32))
    hd_png = out_dir / "hd_edge_n.png"
    _save_map_png(hd_png, hd_edge_n, valid_mask, colormap=str(cfg.output.map_colormap))
    _stage(f"Saved HD edge image: {hd_png}")

    fieldnames = [
        "iteration",
        "seed",
        "bayes_score",
        "proposed_pixel_sampling",
        "global_continuous_dice",
        "global_luminance_rms_contrast",
        "global_composite",
        "mean_local_quality",
        "mean_sampling_weight",
        "viz_path",
        "quality_map_path",
        "sampling_map_path",
        "params",
    ]
    rows: list[dict[str, Any]] = []
    quality_maps: list[np.ndarray] = []
    sampling_map: np.ndarray | None = None
    t0 = time.perf_counter()
    base_seed = int(cfg.adaptive.seed)
    w_dice = float(cfg.adaptive.weights.continuous_dice)
    w_contrast = float(cfg.adaptive.weights.luminance_rms_contrast)
    bayes_cfg = OmegaConf.load(_resolve_path(str(cfg.bayes.config)))
    search_space = _search_space_items(bayes_cfg)
    rng = np.random.default_rng(int(cfg.bayes.random_state))
    observations: list[dict[str, Any]] = []
    n_initial = int(cfg.bayes.n_initial_random)
    n_candidates = int(cfg.bayes.n_candidates)

    for it in range(int(cfg.adaptive.iterations)):
        seed = base_seed + it
        _refresh_bayes_observation_scores(observations, w_dice, w_contrast)
        params = _propose_params(
            observations,
            search_space,
            rng,
            n_initial_random=n_initial,
            n_candidates=n_candidates,
        )
        proposed_pixel_sampling = str(params.get("pixel_sampling", ""))
        _stage(
            f"Iteration {it + 1}/{int(cfg.adaptive.iterations)} seed={seed} "
            f"bayes_params={params}"
        )
        kwargs = _resolve_mics2_kwargs(cfg, seed=seed, sampling_map=sampling_map, params=params)
        model = MSIParametricMiCSLMC(**kwargs)
        model.fit(msi)
        viz = model.predict(msi)
        if hasattr(model, "release_resources"):
            model.release_resources()

        viz_path = out_dir / "visualizations" / f"iter_{it:03d}.png"
        _save_viz(viz_path, viz)
        _copy_preview(viz_path, out_dir / "latest_viz.png")
        _stage(f"Saved visualization: {viz_path}")

        g = _global_score(viz, hd_edge_n, valid_mask, rank_cfg)
        quality, local_dice, local_contrast = _local_quality_map(
            viz,
            hd_edge_n,
            valid_mask,
            rank_cfg,
            patch_size=int(cfg.adaptive.patch_size),
            patch_stride=int(cfg.adaptive.patch_stride),
            min_valid_fraction=float(cfg.adaptive.min_valid_fraction),
            dice_weight=w_dice,
            contrast_weight=w_contrast,
        )
        q_path = out_dir / "quality_maps" / f"iter_{it:03d}_quality.png"
        s_path = out_dir / "sampling_maps" / f"iter_{it:03d}_sampling.png"
        _save_map_png(q_path, quality, valid_mask, colormap=str(cfg.output.map_colormap))
        _copy_preview(q_path, out_dir / "latest_quality.png")
        np.save(q_path.with_suffix(".npy"), quality.astype(np.float32))
        np.save(out_dir / "quality_maps" / f"iter_{it:03d}_local_dice.npy", local_dice.astype(np.float32))
        np.save(out_dir / "quality_maps" / f"iter_{it:03d}_local_contrast_norm.npy", local_contrast.astype(np.float32))

        sampling_map = _badness_sampling_map(
            quality,
            valid_mask,
            gamma=float(cfg.adaptive.low_quality_gamma),
            floor=float(cfg.adaptive.sampling_floor),
        )
        _save_map_png(s_path, _normalize_on_mask(sampling_map, valid_mask), valid_mask, colormap=str(cfg.output.map_colormap))
        _copy_preview(s_path, out_dir / "latest_sampling.png")
        _stage(f"Saved quality/sampling maps: {q_path} | {s_path}")
        np.save(s_path.with_suffix(".npy"), sampling_map.astype(np.float32))

        # Online composite for progress only; final CSV/gallery recompute rank-normalized composite after all iterations.
        contrast_vals = [float(r["global_luminance_rms_contrast"]) for r in rows] + [g["luminance_rms_contrast"]]
        cmax = max(max(contrast_vals), 1e-12)
        for r in rows:
            r["global_composite"] = (w_dice * float(r["global_continuous_dice"]) + w_contrast * (float(r["global_luminance_rms_contrast"]) / cmax)) / max(1e-12, w_dice + w_contrast)
        comp = (w_dice * g["continuous_dice"] + w_contrast * (g["luminance_rms_contrast"] / cmax)) / max(1e-12, w_dice + w_contrast)
        observations.append(
            {
                "params": dict(params),
                "global_continuous_dice": g["continuous_dice"],
                "global_luminance_rms_contrast": g["luminance_rms_contrast"],
                "score": comp,
            }
        )
        row = {
            "iteration": it,
            "seed": seed,
            "bayes_score": comp,
            "proposed_pixel_sampling": proposed_pixel_sampling,
            "global_continuous_dice": g["continuous_dice"],
            "global_luminance_rms_contrast": g["luminance_rms_contrast"],
            "global_composite": comp,
            "mean_local_quality": float(np.mean(quality[valid_mask])),
            "mean_sampling_weight": float(np.mean(sampling_map[valid_mask])),
            "viz_path": str(viz_path),
            "quality_map_path": str(q_path),
            "sampling_map_path": str(s_path),
            "params": repr(dict(params)),
        }
        rows.append(row)
        quality_maps.append(quality)
        _append_csv(out_dir / str(cfg.output.csv_name), row, fieldnames)

    # Rewrite final CSV with final average-rank normalization after all rows are known.
    _apply_final_global_composite(rows, w_dice, w_contrast)
    csv_path = out_dir / str(cfg.output.csv_name)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    selected = _select_coverage_gallery(rows, quality_maps, valid_mask, int(cfg.adaptive.gallery_size))
    _save_gallery(rows, selected, out_dir / "coverage_gallery.png", columns=int(cfg.output.gallery_columns))
    _stage(f"Saved coverage gallery: {out_dir / 'coverage_gallery.png'}")
    _stage(f"Done in {time.perf_counter() - t0:.2f}s | outputs in {out_dir}")


if __name__ == "__main__":
    main()
