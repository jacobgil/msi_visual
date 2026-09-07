#!/usr/bin/env python3
"""
MiCS m/z interpretability gallery.

Train one MiCS model, rank m/z bins by importance, then build ablation panels (no retraining).

``gallery.gradient_rank_scope``:
- ``global``: one top-k m/z list for the whole slice (default).
- ``per_pixel``: each pixel keeps its own top-k m/z (gradient or mean_intensity only).
"""

from __future__ import annotations

import csv
import gc
import logging
import sys
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

from msi_visual.parametric_mics_lmc import (  # noqa: E402
    MSIParametricMiCSLMC,
    PCCWithLearnedInputMask,
)
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402

logger = logging.getLogger(__name__)

_MICS_KEYS: tuple[str, ...] = (
    "number_of_points",
    "sampling",
    "num_epochs",
    "number_of_components",
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


def _abs_path_str(path: Path) -> str:
    return str(path.expanduser().resolve())


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    s = msi.sum(axis=-1, keepdims=True)
    return msi / (s + 1e-8)


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(str(path), mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI HxWxC, got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI must be nonnegative.")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def _parse_clusters(val: Any) -> list[int]:
    if val is None:
        return [1024, 2048, 4096]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip()]
    return [int(s)]


def _mics_kwargs_from_cfg(
    cfg: DictConfig,
    num_layers: int,
    seed: int,
    ga: Any,
) -> dict[str, Any]:
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    kw: dict[str, Any] = {}
    for name in _MICS_KEYS:
        if name in ("num_layers", "clusters"):
            continue
        if name in merged:
            kw[name] = merged[name]
    kw.setdefault("number_of_points", int(merged.get("number_of_points", 1000)))
    kw.setdefault("sampling", str(merged.get("sampling", "coreset")))
    kw.setdefault("num_epochs", int(merged.get("num_epochs", 30)))
    kw.setdefault("number_of_components", int(merged.get("number_of_components", 3)))
    kw.setdefault("lab_to_rgb", bool(merged.get("lab_to_rgb", True)))
    kw.setdefault("cluster", bool(merged.get("cluster", True)))
    kw.setdefault("clusters", _parse_clusters(merged.get("clusters", [1024, 2048, 4096])))
    kw.setdefault("num_samples", int(merged.get("num_samples", 5000)))
    kw.setdefault("beta", float(merged.get("beta", 20.0)))
    kw.setdefault("k_epoch", int(merged.get("k_epoch", 20)))
    kw.setdefault("temperature", float(merged.get("temperature", 100.0)))
    kw.setdefault("lr", float(merged.get("lr", 1.0)))
    kw.setdefault("batch_size", int(merged.get("batch_size", 1024)))
    kw.setdefault("warmup_epochs", int(merged.get("warmup_epochs", 10)))
    kw.setdefault("factor", float(merged.get("factor", 1.0)))
    kw.setdefault("verbose", bool(merged.get("verbose", False)))
    kw.setdefault("pixel_sampling", str(merged.get("pixel_sampling", "superpixel")))
    kw.setdefault(
        "predict_spatial_smooth_sigma", float(merged.get("predict_spatial_smooth_sigma", 0.8))
    )
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw["num_layers"] = max(1, int(num_layers))
    kw["random_state"] = int(seed)

    plain_pcc = bool(OmegaConf.select(ga, "plain_pcc", default=False))
    if plain_pcc:
        kw["learn_input_mask"] = False
        kw["input_mask_l1_weight"] = 0.0
    else:
        kw["learn_input_mask"] = bool(OmegaConf.select(ga, "learn_input_mask", default=True))
        l1w = OmegaConf.select(ga, "input_mask_l1_weight", default=None)
        if l1w is not None:
            kw["input_mask_l1_weight"] = max(0.0, float(l1w))
        else:
            kw.setdefault("input_mask_l1_weight", 0.0)
        init = OmegaConf.select(ga, "input_mask_logits_init", default=None)
        if init is not None:
            kw["input_mask_logits_init"] = float(init)
        else:
            kw.setdefault("input_mask_logits_init", float(merged.get("input_mask_logits_init", 4.0)))
    return kw


def _parse_topk_ladder(ga: Any, n_mz: int) -> list[int | None]:
    raw = OmegaConf.select(ga, "topk_ladder", default=[3, 30, 100, 300])
    if raw is None or (
        isinstance(raw, str) and raw.strip().lower() in ("", "~", "null", "none")
    ):
        raw = [3, 30, 100, 300]
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    ladder: list[int | None] = []
    if bool(OmegaConf.select(ga, "include_full_panel", default=True)):
        ladder.append(None)
    seen: set[int | None] = {None} if None in ladder else set()
    for x in raw:
        k = int(x)
        if k <= 0 or k >= n_mz:
            continue
        if k in seen:
            continue
        ladder.append(k)
        seen.add(k)
    if not ladder:
        ladder = [None]
    return ladder


def _resolve_gradient_rank_scope(ga: Any, rank_by: str) -> str:
    scope = str(OmegaConf.select(ga, "gradient_rank_scope", default="global")).strip().lower()
    if scope not in ("global", "per_pixel"):
        raise ValueError(f"gallery.gradient_rank_scope must be global | per_pixel, got {scope!r}")
    if scope == "per_pixel" and rank_by == "effective_mask":
        logger.warning(
            "gradient_rank_scope=per_pixel ignored for rank_by=effective_mask; using global."
        )
        return "global"
    return scope


def _max_ladder_k(ladder: list[int | None], n_mz: int) -> int:
    ks = [int(k) for k in ladder if k is not None]
    if not ks:
        return min(int(n_mz), 1)
    return min(int(n_mz), max(ks))


def _topk_binary_mask(n_mz: int, rank_order: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(int(n_mz), dtype=np.float32)
    keep = np.asarray(rank_order[: int(k)], dtype=np.int64)
    mask[keep] = 1.0
    return mask


def _rank_by_mean_intensity(msi: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    m = np.asarray(valid_mask, dtype=bool)
    x = np.asarray(msi[m], dtype=np.float64)
    if x.size == 0:
        return np.arange(msi.shape[-1], dtype=np.int64)
    scores = np.mean(np.abs(x), axis=0)
    return np.argsort(-scores)


def _rank_by_gradient(
    model: MSIParametricMiCSLMC,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_pixels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model.model is None:
        raise ValueError("Model not initialized.")
    reshaped = np.asarray(msi, dtype=np.float32).reshape(-1, msi.shape[-1])
    valid_flat = np.where(valid_mask.ravel())[0]
    n_mz = int(msi.shape[-1])
    if valid_flat.size == 0:
        order = np.arange(n_mz, dtype=np.int64)
        return order, np.zeros(n_mz, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    n_take = min(int(n_pixels), int(valid_flat.size))
    chosen = rng.choice(valid_flat, size=n_take, replace=False)
    x_np = reshaped[chosen].copy()
    device = next(model.model.parameters()).device
    model.model.eval()
    x = torch.from_numpy(x_np).to(device)
    x.requires_grad_(True)
    out = model.model(x)
    loss = out.pow(2).sum(dim=-1).mean()
    model.model.zero_grad(set_to_none=True)
    loss.backward()
    if x.grad is None:
        order = np.arange(n_mz, dtype=np.int64)
        return order, np.zeros(n_mz, dtype=np.float32)
    sal = x.grad.detach().abs().mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
    order = np.argsort(-sal)
    return order, sal


def _compute_per_pixel_gradient_topk(
    model: MSIParametricMiCSLMC,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    k_max: int,
    batch_size: int,
) -> np.ndarray:
    """Top-k m/z indices per pixel from gradient saliency, shape (H,W,k_max); -1 off-tissue."""
    if model.model is None:
        raise ValueError("Model not initialized.")
    h, w, n_mz = msi.shape
    k_max = max(1, min(int(k_max), int(n_mz)))
    flat = np.asarray(msi, dtype=np.float32).reshape(-1, n_mz)
    valid_flat = np.where(valid_mask.ravel())[0]
    topk_flat = np.full((h * w, k_max), -1, dtype=np.int32)
    if valid_flat.size == 0:
        return topk_flat.reshape(h, w, k_max)

    device = next(model.model.parameters()).device
    model.model.eval()
    bs = max(1, int(batch_size))
    n_valid = int(valid_flat.size)
    logger.info(
        "Per-pixel gradient ranking: %d valid pixels, k_max=%d, batch=%d",
        n_valid,
        k_max,
        bs,
    )
    for start in range(0, n_valid, bs):
        end = min(start + bs, n_valid)
        idx = valid_flat[start:end]
        x_np = flat[idx].copy()
        x = torch.from_numpy(x_np).to(device)
        x.requires_grad_(True)
        out = model.model(x)
        loss = (out.pow(2).sum(dim=-1)).sum()
        model.model.zero_grad(set_to_none=True)
        loss.backward()
        if x.grad is None:
            continue
        sal = x.grad.detach().abs()
        kk = min(k_max, int(n_mz))
        top_idx = torch.topk(sal, k=kk, dim=1).indices.cpu().numpy().astype(np.int32, copy=False)
        topk_flat[idx] = top_idx
    return topk_flat.reshape(h, w, k_max)


def _compute_per_pixel_intensity_topk(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    k_max: int,
) -> np.ndarray:
    """Top-k m/z by |intensity| at each pixel, shape (H,W,k_max); -1 off-tissue."""
    h, w, n_mz = msi.shape
    k_max = max(1, min(int(k_max), int(n_mz)))
    flat = np.asarray(msi, dtype=np.float32).reshape(-1, n_mz)
    topk_flat = np.full((h * w, k_max), -1, dtype=np.int32)
    valid_flat = np.where(valid_mask.ravel())[0]
    if valid_flat.size == 0:
        return topk_flat.reshape(h, w, k_max)
    abs_spec = np.abs(flat[valid_flat])
    kk = min(k_max, int(n_mz))
    # argpartition for top-k without full sort
    part = np.argpartition(-abs_spec, kk - 1, axis=1)[:, :kk]
    row_order = np.argsort(-np.take_along_axis(abs_spec, part, axis=1), axis=1)
    top_idx = np.take_along_axis(part, row_order, axis=1).astype(np.int32, copy=False)
    topk_flat[valid_flat] = top_idx
    return topk_flat.reshape(h, w, k_max)


def _global_order_from_per_pixel_topk(
    topk_hwk: np.ndarray,
    valid_mask: np.ndarray,
    n_mz: int,
    *,
    k_use: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-pixel top-k into a global rank (by frequency in top-k lists)."""
    m = np.asarray(valid_mask, dtype=bool)
    kk = max(1, min(int(k_use), int(topk_hwk.shape[-1])))
    idx = np.asarray(topk_hwk[m, :kk], dtype=np.int64)
    counts = np.zeros(int(n_mz), dtype=np.float64)
    if idx.size:
        np.add.at(counts, idx.ravel(), 1.0)
    order = np.argsort(-counts)
    scores = counts[order].astype(np.float32)
    return order, scores


def _spectrum_ablated_msi_per_pixel(
    msi: np.ndarray, topk_hwk: np.ndarray, k: int
) -> np.ndarray:
    """Zero all but per-pixel top-k m/z bins."""
    h, w, n_c = msi.shape
    k = max(1, min(int(k), int(topk_hwk.shape[-1]), int(n_c)))
    flat = np.asarray(msi, dtype=np.float32).reshape(-1, n_c)
    n_rows = int(flat.shape[0])
    idx = np.asarray(topk_hwk[..., :k], dtype=np.int64).reshape(n_rows, k)
    out = np.zeros_like(flat)
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), k)
    col_ids = idx.reshape(-1)
    good = col_ids >= 0
    out[row_ids[good], col_ids[good]] = flat[row_ids[good], col_ids[good]]
    return out.reshape(h, w, n_c)


def _rank_mz_indices(
    model: MSIParametricMiCSLMC,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    ga: Any,
    *,
    rank_by_override: str | None = None,
    ladder: list[int | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str, np.ndarray | None]:
    """Returns (rank_order, scores_sorted, rank_by, scope, per_pixel_topk_or_none)."""
    rank_by = (
        str(rank_by_override).strip().lower()
        if rank_by_override is not None
        else str(OmegaConf.select(ga, "rank_by", default="gradient")).strip().lower()
    )
    n_mz = int(msi.shape[-1])
    scope = _resolve_gradient_rank_scope(ga, rank_by)
    k_max = _max_ladder_k(ladder or [3, 30, 100, 300], n_mz)
    per_pixel_topk: np.ndarray | None = None

    if scope == "per_pixel" and rank_by == "mean_intensity":
        per_pixel_topk = _compute_per_pixel_intensity_topk(msi, valid_mask, k_max=k_max)
        order, scores = _global_order_from_per_pixel_topk(
            per_pixel_topk, valid_mask, n_mz, k_use=k_max
        )
        rank_label = "mean_intensity_per_pixel"
        return order, scores, rank_label, scope, per_pixel_topk

    if scope == "per_pixel" and rank_by == "gradient":
        per_pixel_topk = _compute_per_pixel_gradient_topk(
            model,
            msi,
            valid_mask,
            k_max=k_max,
            batch_size=int(OmegaConf.select(ga, "gradient_rank_batch_size", default=512)),
        )
        order, scores = _global_order_from_per_pixel_topk(
            per_pixel_topk, valid_mask, n_mz, k_use=k_max
        )
        rank_label = "gradient_per_pixel"
        return order, scores, rank_label, scope, per_pixel_topk

    if rank_by == "mean_intensity":
        order = _rank_by_mean_intensity(msi, valid_mask)
        scores = np.zeros(n_mz, dtype=np.float32)
        m = np.asarray(valid_mask, dtype=bool)
        scores[:] = np.mean(np.abs(msi[m]), axis=0) if np.any(m) else 0.0
        return order, scores[order], rank_by, scope, None

    if rank_by == "gradient":
        order, sal = _rank_by_gradient(
            model,
            msi,
            valid_mask,
            n_pixels=int(OmegaConf.select(ga, "gradient_rank_n_pixels", default=2048)),
            seed=int(OmegaConf.select(ga, "seed", default=42)),
        )
        return order, sal[order], rank_by, scope, None

    # effective_mask
    masks = model.get_input_mask_arrays()
    eff = np.asarray(masks["effective"], dtype=np.float64).ravel()
    order = np.argsort(-eff)
    return order, eff[order].astype(np.float32), "effective_mask", scope, None


def _compute_norm_anchor(
    emb: np.ndarray, mask: np.ndarray, low: float, high: float
) -> list[tuple[float, float]]:
    anchor: list[tuple[float, float]] = []
    m = np.asarray(mask, dtype=bool)
    for c in range(int(emb.shape[-1])):
        vals = np.asarray(emb[..., c], dtype=np.float64)[m]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            anchor.append((0.0, 1.0))
        else:
            anchor.append((float(np.percentile(vals, low)), float(np.percentile(vals, high))))
    return anchor


def _match_embedding_intensity(
    emb: np.ndarray,
    ref: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gain: float = 1.0,
) -> np.ndarray:
    out = np.asarray(emb, dtype=np.float32).copy()
    m = np.asarray(valid_mask, dtype=bool)
    if not np.any(m):
        return out
    g = float(gain)
    for c in range(int(out.shape[-1])):
        e = np.asarray(out[..., c], dtype=np.float64)[m]
        r = np.asarray(ref[..., c], dtype=np.float64)[m]
        e = e[np.isfinite(e)]
        r = r[np.isfinite(r)]
        if e.size == 0 or r.size == 0:
            continue
        std_e = float(np.std(e))
        std_r = float(np.std(r))
        if std_e < 1e-12:
            continue
        out[..., c] *= g * (std_r / std_e)
    return out


def _embedding_to_rgb_with_anchor(
    emb: np.ndarray,
    valid_mask: np.ndarray,
    anchor: list[tuple[float, float]],
    *,
    lab_to_rgb: bool,
    n_components: int,
    smooth_sigma: float,
) -> np.ndarray:
    import cv2

    from msi_visual.parametric_mics_lmc import _spatial_gaussian_smooth_channels

    result = np.asarray(emb, dtype=np.float32).copy()
    m = np.asarray(valid_mask, dtype=bool)
    for c, (p_lo, p_hi) in enumerate(anchor):
        ch = result[..., c]
        ch = (ch - p_lo) / max(p_hi - p_lo, 1e-8)
        ch = np.clip(ch, 0.0, 1.0)
        result[..., c] = ch
    result[~m] = 0.0
    result = _spatial_gaussian_smooth_channels(result, smooth_sigma)
    out = np.uint8(255.0 * result)
    out[~m] = 0
    if lab_to_rgb and n_components == 3:
        out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    out[~m] = 0
    return out


def _embedding_rmse_vs_ref(
    emb: np.ndarray, ref: np.ndarray, valid_mask: np.ndarray
) -> float:
    m = np.asarray(valid_mask, dtype=bool)
    if not np.any(m):
        return float("nan")
    d = np.asarray(emb[m], dtype=np.float64) - np.asarray(ref[m], dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=-1))))


def _save_ranking_diagnostic(
    path: Path,
    scores_by_rank: np.ndarray,
    rank_order: np.ndarray,
    scores_by_index: np.ndarray,
    *,
    rank_by: str,
    top_k: int,
) -> None:
    by_idx = np.asarray(scores_by_index, dtype=np.float64).ravel()
    by_rank = np.asarray(scores_by_rank, dtype=np.float64).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(by_idx, lw=0.6)
    axes[0].set_title(f"m/z importance ({rank_by})")
    axes[0].set_xlabel("m/z index")
    axes[0].set_ylabel("score")

    k = min(int(top_k), by_rank.size)
    if k > 0:
        top_idx = rank_order[:k]
        axes[1].bar(np.arange(k), by_rank[:k], width=0.85)
        axes[1].set_title(f"top {k} m/z by rank")
        axes[1].set_xlabel("rank")
        axes[1].set_ylabel("score")
        tick_step = max(1, k // 12)
        axes[1].set_xticks(np.arange(0, k, tick_step))
        axes[1].set_xticklabels([str(int(top_idx[i])) for i in range(0, k, tick_step)], rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _panel_label(k: int | None) -> str:
    if k is None:
        return "full"
    return f"top{k}"


def _resolve_num_layers(cfg: DictConfig, ga: Any) -> int:
    raw = OmegaConf.select(ga, "num_layers", default=None)
    if raw is None or str(raw).strip().lower() in ("", "~", "null", "none"):
        return max(1, int(cfg.model.num_layers))
    return max(1, int(raw))


def _resolve_normalization(ga: Any) -> str:
    norm = str(OmegaConf.select(ga, "normalization", default="")).strip().lower()
    if norm not in ("", "~", "null", "none"):
        if norm not in ("per_panel", "shared", "both"):
            raise ValueError(f"gallery.normalization must be per_panel | shared | both, got {norm!r}")
        return norm
    shared = bool(OmegaConf.select(ga, "shared_normalization", default=False))
    return "shared" if shared else "per_panel"


def _spectrum_ablated_msi(msi: np.ndarray, ablation_mask: np.ndarray) -> np.ndarray:
    """Hard zero on non-kept m/z bins (works for plain PCC and masked PCC)."""
    m = np.asarray(ablation_mask, dtype=np.float32).reshape(1, 1, -1)
    return (np.asarray(msi, dtype=np.float32) * m).astype(np.float32, copy=False)


def _save_mosaic(
    panel_rgbs: list[np.ndarray],
    panel_records: list[dict[str, Any]],
    *,
    out_path: Path,
    n_mz: int,
    rank_by: str,
    rank_scope: str,
    num_layers: int,
    norm_label: str,
    ncols: int,
    fig_w: float,
    dpi: int,
    title_fs: int,
) -> None:
    n_p = len(panel_rgbs)
    nrows = int(np.ceil(n_p / ncols))
    mh = float(max(fig_w * 0.35, fig_w * 0.28 * max(1, nrows)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, mh))
    if nrows == 1 and ncols == 1:
        ax_grid = np.array([[axes]])
    elif nrows == 1:
        ax_grid = np.asarray(axes).reshape(1, -1)
    elif ncols == 1:
        ax_grid = np.asarray(axes).reshape(-1, 1)
    else:
        ax_grid = np.asarray(axes)

    for idx, rgb in enumerate(panel_rgbs):
        r, c = divmod(idx, ncols)
        ax = ax_grid[r, c]
        ax.imshow(rgb)
        rec = panel_records[idx]
        k_lbl = rec["panel"]
        rmse = rec["embedding_rmse_vs_full"]
        if k_lbl == "full":
            tit = f"full ({n_mz} m/z)\nreference"
        else:
            tit = f"{k_lbl} m/z\nRMSE vs full={rmse:.3g}"
        ax.set_title(tit, fontsize=title_fs)
        ax.axis("off")
    for idx in range(n_p, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax_grid[r, c].axis("off")

    fig.suptitle(
        f"MiCS top-k m/z ablation (rank={rank_by}, scope={rank_scope}, L={num_layers}, norm={norm_label})",
        fontsize=title_fs + 2,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_topk_mz")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ga = getattr(cfg, "gallery", None)
    if ga is None:
        raise ValueError("Config must define gallery:")

    npy_path = Path(to_absolute_path(str(ga.npy_path))).expanduser().resolve()
    if not npy_path.is_file():
        raise FileNotFoundError(f"gallery.npy_path not found: {npy_path}")

    msi = _load_msi(npy_path, bool(ga.transpose_msi), bool(ga.tic_normalize))
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")
    n_mz = int(msi.shape[-1])

    num_layers = _resolve_num_layers(cfg, ga)
    seed = int(OmegaConf.select(ga, "seed", default=42))
    kw = _mics_kwargs_from_cfg(cfg, num_layers, seed, ga)
    ladder = _parse_topk_ladder(ga, n_mz)
    norm_mode = _resolve_normalization(ga)

    rank_by_cfg = str(OmegaConf.select(ga, "rank_by", default="gradient")).strip().lower()
    if not kw.get("learn_input_mask") and rank_by_cfg == "effective_mask":
        logger.warning(
            "rank_by=effective_mask needs learn_input_mask; using gradient saliency instead."
        )
        rank_by_cfg = "gradient"

    intensity_match = bool(OmegaConf.select(ga, "display_intensity_match", default=True))
    intensity_gain = float(OmegaConf.select(ga, "display_intensity_gain", default=1.0))
    pred_lo = float(kw.get("predict_percentile_low", 1.0))
    pred_hi = float(kw.get("predict_percentile_high", 99.0))
    top_k_plot = int(OmegaConf.select(ga, "mask_plot_top_k", default=48))
    ncols = max(1, int(OmegaConf.select(ga, "ncols", default=3)))
    fig_w = float(OmegaConf.select(ga, "figure_size_inches", default=14))
    dpi = int(OmegaConf.select(ga, "dpi", default=300))
    title_fs = int(OmegaConf.select(ga, "panel_title_fontsize", default=10))

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    pan_dir = out_dir / "panels"
    pan_dir.mkdir(parents=True, exist_ok=True)

    share_sampling = bool(OmegaConf.select(ga, "share_pixel_sampling", default=True))
    cache_path_raw = OmegaConf.select(ga, "pixel_sampling_cache_path", default=None)
    cache_path: Path | None = None
    if cache_path_raw is not None and str(cache_path_raw).strip().lower() not in (
        "",
        "~",
        "null",
        "none",
    ):
        cache_path = Path(to_absolute_path(str(cache_path_raw))).expanduser().resolve()
    pixel_cache = resolve_pixel_sampling_cache(
        msi,
        kw,
        out_dir=out_dir,
        share=share_sampling,
        cache_path=cache_path,
    )

    mics_label = "mics2-equivalent" if num_layers >= 7 and kw.get("learn_input_mask") is False else "MiCS"
    logger.info(
        "Training %s (layers=%d, learn_input_mask=%s, L1=%g, rank=%s, scope=%s) on %s",
        mics_label,
        num_layers,
        kw.get("learn_input_mask"),
        float(kw.get("input_mask_l1_weight", 0.0)),
        rank_by_cfg,
        _resolve_gradient_rank_scope(ga, rank_by_cfg),
        npy_path.name,
    )

    model: MSIParametricMiCSLMC | None = None
    try:
        model = MSIParametricMiCSLMC(**kw)
        model.fit(msi, pixel_sampling_cache=pixel_cache)

        rank_order, rank_scores_sorted, rank_by, rank_scope, per_pixel_topk = _rank_mz_indices(
            model, msi, valid_mask, ga, rank_by_override=rank_by_cfg, ladder=ladder
        )
        rank_scores_full = np.zeros(n_mz, dtype=np.float32)
        rank_scores_full[rank_order] = rank_scores_sorted

        np.save(out_dir / "mz_rank_order.npy", rank_order.astype(np.int64))
        np.save(out_dir / "mz_rank_scores.npy", rank_scores_full)
        if per_pixel_topk is not None:
            np.save(out_dir / "mz_rank_per_pixel_topk.npy", per_pixel_topk)
            logger.info(
                "Saved per-pixel top-k indices: %s",
                _abs_path_str(out_dir / "mz_rank_per_pixel_topk.npy"),
            )

        masks_after_train = model.get_input_mask_arrays()
        if isinstance(model.model, PCCWithLearnedInputMask):
            np.save(out_dir / "effective_mask_after_train.npy", masks_after_train["effective"])

        rank_plot = out_dir / "gallery_mics_topk_mz_ranking.png"
        _save_ranking_diagnostic(
            rank_plot,
            rank_scores_sorted,
            rank_order,
            rank_scores_full,
            rank_by=rank_by,
            top_k=top_k_plot,
        )
        logger.info("Ranking plot: %s", _abs_path_str(rank_plot))

        full_mask = np.ones(n_mz, dtype=np.float32)
        model.set_fixed_input_mask(full_mask)
        ref_emb = model.predict_embedding(msi).astype(np.float32, copy=False)
        full_anchor = _compute_norm_anchor(ref_emb, valid_mask, pred_lo, pred_hi)

        rgb_kw = dict(
            lab_to_rgb=bool(kw.get("lab_to_rgb", True)),
            n_components=int(kw.get("number_of_components", 3)),
            smooth_sigma=float(kw.get("predict_spatial_smooth_sigma", 0.0)),
        )

        panel_records: list[dict[str, Any]] = []
        panel_rgbs_per: list[np.ndarray] = []
        panel_rgbs_shared: list[np.ndarray] = []
        for k in ladder:
            if k is None:
                ablation_mask = full_mask
                msi_in = msi
            elif per_pixel_topk is not None:
                ablation_mask = None
                msi_in = _spectrum_ablated_msi_per_pixel(msi, per_pixel_topk, int(k))
            else:
                ablation_mask = _topk_binary_mask(n_mz, rank_order, int(k))
                msi_in = _spectrum_ablated_msi(msi, ablation_mask)
            emb = model.predict_embedding(msi_in).astype(np.float32, copy=False)
            rmse = _embedding_rmse_vs_ref(emb, ref_emb, valid_mask)

            anchor_panel = full_anchor if k is None else _compute_norm_anchor(
                emb, valid_mask, pred_lo, pred_hi
            )
            rgb_per = _embedding_to_rgb_with_anchor(
                emb, valid_mask, anchor_panel, **rgb_kw
            )
            panel_rgbs_per.append(rgb_per)

            emb_shared = emb
            if intensity_match and k is not None:
                emb_shared = _match_embedding_intensity(
                    emb, ref_emb, valid_mask, gain=intensity_gain
                )
            rgb_shared = _embedding_to_rgb_with_anchor(
                emb_shared, valid_mask, full_anchor, **rgb_kw
            )
            panel_rgbs_shared.append(rgb_shared)

            lbl = _panel_label(k)
            if norm_mode == "shared":
                rgb_primary = rgb_shared
            else:
                rgb_primary = rgb_per
            viz_path = pan_dir / f"panel__{lbl}__viz.png"
            Image.fromarray(rgb_primary, mode="RGB").save(viz_path)
            if norm_mode == "both":
                Image.fromarray(rgb_shared, mode="RGB").save(
                    pan_dir / f"panel__{lbl}__viz__shared_norm.png"
                )
            np.save(pan_dir / f"panel__{lbl}__embedding.npy", emb)
            if ablation_mask is not None:
                np.save(pan_dir / f"panel__{lbl}__ablation_mask.npy", ablation_mask)
            elif per_pixel_topk is not None and k is not None:
                np.save(
                    pan_dir / f"panel__{lbl}__ablation_topk.npy",
                    per_pixel_topk[..., : int(k)],
                )

            n_kept = int(n_mz) if k is None else int(k)
            scope_lbl = "per-pixel" if per_pixel_topk is not None and k is not None else "global"
            panel_records.append(
                {
                    "panel": lbl,
                    "top_k": "" if k is None else int(k),
                    "n_mz_kept": n_kept,
                    "embedding_rmse_vs_full": rmse,
                    "viz_path": _abs_path_str(viz_path),
                }
            )
            logger.info(
                "Panel %s | scope=%s | kept=%d/%d | embed RMSE vs full=%.4g",
                lbl,
                scope_lbl,
                n_kept,
                n_mz,
                rmse,
            )

        mosaic = out_dir / "gallery_mics_topk_mz.png"
        primary_rgbs = panel_rgbs_shared if norm_mode == "shared" else panel_rgbs_per
        _save_mosaic(
            primary_rgbs,
            panel_records,
            out_path=mosaic,
            n_mz=n_mz,
            rank_by=rank_by,
            rank_scope=rank_scope,
            num_layers=num_layers,
            norm_label=norm_mode if norm_mode != "both" else "per_panel",
            ncols=ncols,
            fig_w=fig_w,
            dpi=dpi,
            title_fs=title_fs,
        )
        mosaic_shared: Path | None = None
        if norm_mode == "both":
            mosaic_shared = out_dir / "gallery_mics_topk_mz__shared_norm.png"
            _save_mosaic(
                panel_rgbs_shared,
                panel_records,
                out_path=mosaic_shared,
                n_mz=n_mz,
                rank_by=rank_by,
                rank_scope=rank_scope,
                num_layers=num_layers,
                norm_label="shared",
                ncols=ncols,
                fig_w=fig_w,
                dpi=dpi,
                title_fs=title_fs,
            )

        csv_path = out_dir / "topk_mz_panels.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "panel",
                    "top_k",
                    "n_mz_kept",
                    "embedding_rmse_vs_full",
                    "viz_path",
                ],
            )
            writer.writeheader()
            writer.writerows(panel_records)

        summary = out_dir / "topk_mz_summary.txt"
        with summary.open("w", encoding="utf-8") as fh:
            fh.write(f"npy_path: {npy_path}\n")
            fh.write(f"n_mz: {n_mz}\n")
            fh.write(f"num_layers: {num_layers}\n")
            fh.write(f"learn_input_mask: {kw.get('learn_input_mask')}\n")
            fh.write(f"input_mask_l1_weight: {kw.get('input_mask_l1_weight')}\n")
            fh.write(f"mics_label: {mics_label}\n")
            fh.write(f"rank_by: {rank_by}\n")
            fh.write(f"gradient_rank_scope: {rank_scope}\n")
            fh.write(f"topk_ladder: {ladder}\n")
            fh.write(
                "ablation: spectrum hard-zero (msi * top-k mask"
                + (", per-pixel top-k" if per_pixel_topk is not None else ", global top-k")
                + ")\n"
            )
            fh.write(f"normalization: {norm_mode}\n")
            fh.write(f"display_intensity_match: {intensity_match}\n")
            fh.write("\n")
            for rec in panel_records:
                fh.write(
                    f"{rec['panel']}: kept={rec['n_mz_kept']} "
                    f"rmse={rec['embedding_rmse_vs_full']:.6g}\n"
                )
                fh.write(f"  viz: {rec['viz_path']}\n")
            fh.write(f"\nmosaic: {_abs_path_str(mosaic)}\n")
            if mosaic_shared is not None:
                fh.write(f"mosaic_shared_norm: {_abs_path_str(mosaic_shared)}\n")
            fh.write(f"ranking: {_abs_path_str(rank_plot)}\n")
            fh.write(f"csv: {_abs_path_str(csv_path)}\n")

        logger.info("Gallery mosaic: %s", _abs_path_str(mosaic))
        if mosaic_shared is not None:
            logger.info("Gallery mosaic (shared norm): %s", _abs_path_str(mosaic_shared))
        logger.info("Summary: %s", _abs_path_str(summary))
        logger.info("CSV: %s", _abs_path_str(csv_path))

    finally:
        if model is not None:
            try:
                model.release_resources()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
