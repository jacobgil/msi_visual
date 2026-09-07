#!/usr/bin/env python3
"""
Train peak-token MiCS (hybrid m/z embedding + small transformer).

Multi-slide discovery (all compatible slides under data roots):
  python scripts/train_peak_token_mics.py

Foundation model (all catalog slides, mixed spectral grids):
  python scripts/train_peak_token_mics.py --config-name train_peak_token_mics_foundation

Explicit slide list:
  python scripts/train_peak_token_mics.py data.npy_paths=[path/a.npy,path/b.npy]

Long run (~12 h default): checkpoints + previews written under outputs/peak_token_mics/...
"""

from __future__ import annotations

import gc
import json
import logging
import re
import sys
from argparse import Namespace
from collections import defaultdict
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

from debug_edge_maps import _load_msi, _load_valid_mask, _resolve_normalization_mode  # noqa: E402
from msi_visual.extraction import resolve_mass_axis  # noqa: E402
from msi_visual.peak_token_mics import (  # noqa: E402
    PeakTokenFitOptions,
    PeakTokenMiCSConfig,
    PeakTokenMiCSTrainer,
    SlideRecord,
    mass_axis_is_dalton,
)

logger = logging.getLogger(__name__)


def _abs_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _read_slide_meta(npy_path: Path) -> dict[str, Any]:
    arr = np.load(str(npy_path), mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxM array, got {arr.shape} for {npy_path}")
    args_path = npy_path.parent / "args.txt"
    meta: dict[str, Any] = {
        "npy_path": str(npy_path),
        "slide_key": npy_path.parent.parent.name if npy_path.parent.name in ("5_bins", "5bins") else npy_path.parent.name,
        "spectral_bins": int(arr.shape[-1]),
        "start_mz": None,
        "end_mz": None,
        "bins_per_mz": None,
    }
    if args_path.is_file():
        try:
            text = args_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = args_path.read_text(encoding="latin-1")
        try:
            ns: Namespace = eval(text)  # noqa: S307
            meta["start_mz"] = int(ns.start_mz)
            meta["end_mz"] = int(ns.end_mz)
            meta["bins_per_mz"] = int(ns.bins)
        except Exception as exc:
            logger.warning("Could not parse %s: %s", args_path, exc)
    return meta


def _spectral_group_key(meta: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(meta["spectral_bins"]),
        meta.get("start_mz"),
        meta.get("end_mz"),
        meta.get("bins_per_mz"),
    )


def _discover_from_roots(cfg: DictConfig) -> list[Path]:
    roots_raw = OmegaConf.select(cfg, "data.discover.roots")
    globs_raw = OmegaConf.select(cfg, "data.discover.npy_globs")
    if roots_raw is None:
        return []

    roots = [_abs_path(str(r)) for r in OmegaConf.to_container(roots_raw, resolve=True)]
    if globs_raw is None:
        globs = [str(OmegaConf.select(cfg, "data.discover.npy_glob", default="**/5_bins/*.npy"))]
    else:
        globs = [str(g) for g in OmegaConf.to_container(globs_raw, resolve=True)]

    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            logger.warning("Discovery root missing: %s", root)
            continue
        for pattern in globs:
            for p in sorted(root.glob(pattern)):
                if not p.is_file():
                    continue
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                found.append(p.resolve())
    return found


def _load_benchmark_catalog_paths(cfg: DictConfig) -> list[Path]:
    catalog_raw = OmegaConf.select(cfg, "data.benchmark_catalog")
    if catalog_raw is None or not str(catalog_raw).strip():
        return []

    catalog_path = _abs_path(str(catalog_raw))
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Benchmark catalog not found: {catalog_path}")

    include_comments = bool(OmegaConf.select(cfg, "data.include_comment_catalog", default=True))
    require_exists = bool(OmegaConf.select(cfg, "data.catalog_require_exists", default=True))
    found: list[Path] = []
    seen: set[str] = set()
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and not include_comments:
            continue
        for match in re.finditer(r'"([^"]+\.npy)"', line):
            candidate = _abs_path(match.group(1))
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                found.append(candidate)
            elif require_exists:
                logger.warning("Catalog path missing on disk: %s", candidate)
            else:
                logger.debug("Catalog path missing (skipped): %s", candidate)
    logger.info("Loaded %d catalog .npy path(s) from %s", len(found), catalog_path)
    return found


def _discover_npy_paths(cfg: DictConfig) -> list[Path]:
    foundation_mode = bool(OmegaConf.select(cfg, "data.foundation_mode", default=False))
    catalog_paths = _load_benchmark_catalog_paths(cfg) if (
        foundation_mode or OmegaConf.select(cfg, "data.benchmark_catalog") is not None
    ) else []

    explicit = OmegaConf.select(cfg, "data.npy_paths")
    if explicit is not None:
        items = OmegaConf.to_container(explicit, resolve=True)
        if isinstance(items, (list, tuple)) and items:
            paths = [_abs_path(str(p)) for p in items]
            if foundation_mode and catalog_paths:
                merged = list(catalog_paths)
                seen = {str(p) for p in merged}
                for p in paths:
                    key = str(p)
                    if key not in seen:
                        seen.add(key)
                        merged.append(p)
                return merged
            return paths

    single = OmegaConf.select(cfg, "data.npy_path")
    if single is not None and str(single).strip():
        return [_abs_path(str(single))]

    if foundation_mode:
        discovered = list(catalog_paths)
        if bool(OmegaConf.select(cfg, "data.discover.merge_with_catalog", default=True)):
            for p in _discover_from_roots(cfg):
                key = str(p)
                if key not in {str(x) for x in discovered}:
                    discovered.append(p)
        if not discovered:
            raise FileNotFoundError(
                "foundation_mode: no slides from benchmark catalog or discover.roots"
            )
        return discovered

    discovered = _discover_from_roots(cfg)
    if not discovered:
        raise FileNotFoundError("Set data.npy_path, data.npy_paths, or data.discover.roots")
    return discovered


def _select_compatible_slides(npy_paths: list[Path], cfg: DictConfig) -> tuple[list[Path], dict[str, Any]]:
    metas: list[tuple[Path, dict[str, Any]]] = []
    min_tissue = int(OmegaConf.select(cfg, "data.min_tissue_pixels", default=500))
    for p in npy_paths:
        try:
            meta = _read_slide_meta(p)
            arr = np.load(str(p), mmap_mode="r")
            tissue = int((arr.sum(axis=-1) > 0).sum()) if arr.ndim == 3 else 0
            if tissue < min_tissue:
                logger.info("Skip %s (tissue=%d < %d)", p.name, tissue, min_tissue)
                continue
            meta["tissue_pixels"] = tissue
            metas.append((p, meta))
        except Exception as exc:
            logger.warning("Skip %s: %s", p, exc)

    if not metas:
        raise ValueError("No slides passed tissue / readability filters")

    filt = OmegaConf.select(cfg, "data.spectral_filter") or {}
    if bool(OmegaConf.select(filt, "auto", default=True)):
        groups: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
        for p, meta in metas:
            groups[_spectral_group_key(meta)].append((p, meta))
        best_key = max(groups, key=lambda k: len(groups[k]))
        chosen = groups[best_key]
        group_info = {
            "mode": "auto",
            "spectral_group": best_key,
            "n_groups_found": len(groups),
            "n_slides_in_group": len(chosen),
        }
        # Require full spectral metadata (skip slides with unreadable args.txt).
        full_meta = [
            (p, m)
            for p, m in chosen
            if m.get("start_mz") is not None
            and m.get("end_mz") is not None
            and m.get("bins_per_mz") is not None
        ]
        if full_meta:
            chosen = full_meta
            group_info["n_slides_in_group"] = len(chosen)
        logger.info(
            "Auto spectral group %s -> %d slide(s) (of %d total candidates, %d groups)",
            best_key,
            len(chosen),
            len(metas),
            len(groups),
        )
    else:
        chosen = []
        for p, meta in metas:
            if int(meta["spectral_bins"]) != int(filt.spectral_bins):
                continue
            if int(meta.get("start_mz", filt.start_mz)) != int(filt.start_mz):
                continue
            if int(meta.get("end_mz", filt.end_mz)) != int(filt.end_mz):
                continue
            if int(meta.get("bins_per_mz", filt.bins_per_mz)) != int(filt.bins_per_mz):
                continue
            chosen.append((p, meta))
        group_info = {"mode": "explicit", "n_slides_in_group": len(chosen)}

    max_slides = OmegaConf.select(cfg, "data.max_slides")
    if max_slides is not None:
        chosen = chosen[: int(max_slides)]

    if len(chosen) < 1:
        raise ValueError("No compatible slides after spectral filtering")

    paths = [p for p, _ in chosen]
    slide_manifest = {
        **group_info,
        "slides": [
            {
                "npy_path": str(p),
                "slide_key": m["slide_key"],
                "tissue_pixels": int(m.get("tissue_pixels", 0)),
                "spectral_bins": int(m["spectral_bins"]),
            }
            for p, m in chosen
        ],
    }
    return paths, slide_manifest


def _select_foundation_slides(npy_paths: list[Path], cfg: DictConfig) -> tuple[list[Path], dict[str, Any]]:
    metas: list[tuple[Path, dict[str, Any]]] = []
    min_tissue = int(OmegaConf.select(cfg, "data.min_tissue_pixels", default=500))
    skipped_non_dalton = 0
    for p in npy_paths:
        try:
            meta = _read_slide_meta(p)
            arr = np.load(str(p), mmap_mode="r")
            tissue = int((arr.sum(axis=-1) > 0).sum()) if arr.ndim == 3 else 0
            if tissue < min_tissue:
                logger.info("Skip %s (tissue=%d < %d)", p.name, tissue, min_tissue)
                continue
            mass_axis = resolve_mass_axis(p, int(meta["spectral_bins"]))
            if not mass_axis_is_dalton(mass_axis):
                skipped_non_dalton += 1
                logger.info("Skip %s (non-Dalton mass axis)", p.name)
                continue
            meta["tissue_pixels"] = tissue
            meta["mass_min_da"] = float(mass_axis.min())
            meta["mass_max_da"] = float(mass_axis.max())
            metas.append((p, meta))
        except Exception as exc:
            logger.warning("Skip %s: %s", p, exc)

    if not metas:
        raise ValueError("No foundation slides passed tissue / Dalton m/z filters")

    max_slides = OmegaConf.select(cfg, "data.max_slides")
    if max_slides is not None:
        metas = metas[: int(max_slides)]

    paths = [p for p, _ in metas]
    slide_manifest = {
        "mode": "foundation",
        "n_candidates": len(npy_paths),
        "n_slides": len(metas),
        "skipped_non_dalton": int(skipped_non_dalton),
        "slides": [
            {
                "npy_path": str(p),
                "slide_key": m["slide_key"],
                "tissue_pixels": int(m.get("tissue_pixels", 0)),
                "spectral_bins": int(m["spectral_bins"]),
                "mass_min_da": float(m.get("mass_min_da", 0.0)),
                "mass_max_da": float(m.get("mass_max_da", 0.0)),
            }
            for p, m in metas
        ],
    }
    logger.info(
        "Foundation selection: %d slide(s) from %d candidates (%d skipped non-Dalton)",
        len(metas),
        len(npy_paths),
        skipped_non_dalton,
    )
    return paths, slide_manifest


def _select_training_slides(npy_paths: list[Path], cfg: DictConfig) -> tuple[list[Path], dict[str, Any]]:
    if bool(OmegaConf.select(cfg, "data.foundation_mode", default=False)):
        return _select_foundation_slides(npy_paths, cfg)
    return _select_compatible_slides(npy_paths, cfg)


def _trainer_cfg(cfg: DictConfig) -> PeakTokenMiCSConfig:
    m = OmegaConf.select(cfg, "model") or {}
    return PeakTokenMiCSConfig(
        top_k=int(OmegaConf.select(m, "top_k", default=128)),
        embed_dim=int(OmegaConf.select(m, "embed_dim", default=3)),
        d_model=int(OmegaConf.select(m, "d_model", default=128)),
        nhead=int(OmegaConf.select(m, "nhead", default=4)),
        num_layers=int(OmegaConf.select(m, "num_layers", default=2)),
        dim_feedforward=int(OmegaConf.select(m, "dim_feedforward", default=192)),
        dropout=float(OmegaConf.select(m, "dropout", default=0.05)),
        mass_window_size=float(OmegaConf.select(m, "mass_window_size", default=0.0)),
        mass_scale=float(OmegaConf.select(m, "mass_scale", default=0.0)),
        max_mass_windows=int(OmegaConf.select(m, "max_mass_windows", default=512)),
        sort_peaks_by_mass=bool(OmegaConf.select(m, "sort_peaks_by_mass", default=True)),
        num_epochs=int(OmegaConf.select(m, "num_epochs", default=60)),
        batch_size=int(OmegaConf.select(m, "batch_size", default=512)),
        lr=float(OmegaConf.select(m, "lr", default=8e-4)),
        weight_decay=float(OmegaConf.select(m, "weight_decay", default=1e-4)),
        warmup_epochs=int(OmegaConf.select(m, "warmup_epochs", default=3)),
        num_samples=int(OmegaConf.select(m, "num_samples", default=5000)),
        number_of_points=int(OmegaConf.select(m, "number_of_points", default=1000)),
        sampling=str(OmegaConf.select(m, "sampling", default="coreset")),
        beta=float(OmegaConf.select(m, "beta", default=20.0)),
        k_epoch=int(OmegaConf.select(m, "k_epoch", default=20)),
        temperature=float(OmegaConf.select(m, "temperature", default=100.0)),
        cluster=bool(OmegaConf.select(m, "cluster", default=True)),
        clusters=OmegaConf.select(m, "clusters", default="16-32-64-1024"),
        cluster_loss_weight=float(OmegaConf.select(m, "cluster_loss_weight", default=1.0)),
        cluster_on_pca=bool(OmegaConf.select(m, "cluster_on_pca", default=False)),
        cluster_pca_dims=int(OmegaConf.select(m, "cluster_pca_dims", default=64)),
        lab_to_rgb=bool(OmegaConf.select(m, "lab_to_rgb", default=True)),
        percentile_low=float(OmegaConf.select(m, "predict_percentile_low", default=1.0)),
        percentile_high=float(OmegaConf.select(m, "predict_percentile_high", default=99.0)),
        predict_spatial_smooth_sigma=float(OmegaConf.select(m, "predict_spatial_smooth_sigma", default=0.0)),
        seed=int(cfg.seed),
        device="cuda" if torch.cuda.is_available() else "cpu",
        foundation_mode=bool(OmegaConf.select(cfg, "data.foundation_mode", default=False)),
        canonical_mz_step=float(OmegaConf.select(m, "canonical_mz_step", default=5.0)),
        mass_range_padding_da=float(OmegaConf.select(m, "mass_range_padding_da", default=20.0)),
    )


def _fit_opts(cfg: DictConfig, run_dir: Path) -> PeakTokenFitOptions:
    tr = OmegaConf.select(cfg, "training") or {}
    ckpt_dir = run_dir / str(OmegaConf.select(tr, "checkpoint_subdir", default="checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    max_hours_raw = OmegaConf.select(tr, "max_hours")
    return PeakTokenFitOptions(
        max_epochs=int(OmegaConf.select(tr, "max_epochs", default=OmegaConf.select(cfg, "model.num_epochs", default=60))),
        max_hours=None if max_hours_raw is None else float(max_hours_raw),
        pool_refresh_epochs=int(OmegaConf.select(tr, "pool_refresh_epochs", default=0)),
        adaptive_sampling=str(OmegaConf.select(tr, "adaptive_sampling", default="proportional")),
        cosine_lr=bool(OmegaConf.select(tr, "cosine_lr", default=True)),
        min_lr_ratio=float(OmegaConf.select(tr, "min_lr_ratio", default=0.05)),
        release_slide_memory=bool(OmegaConf.select(tr, "release_slide_memory", default=True)),
        refresh_clusters_on_pool_refresh=bool(
            OmegaConf.select(tr, "refresh_clusters_on_pool_refresh", default=True)
        ),
        checkpoint_every_epochs=int(OmegaConf.select(tr, "checkpoint_every_epochs", default=5)),
        checkpoint_dir=ckpt_dir,
    )


def _letterbox_rgb(rgb: np.ndarray, *, target_h: int, target_w: int, bg: tuple[int, int, int] = (14, 14, 16)) -> np.ndarray:
    """Resize RGB into a fixed cell while preserving aspect ratio."""
    img = np.asarray(rgb, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(float(target_h) / max(h, 1), float(target_w) / max(w, 1))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.Resampling.NEAREST))
    cell = np.zeros((int(target_h), int(target_w), 3), dtype=np.uint8)
    cell[:] = np.asarray(bg, dtype=np.uint8)
    y0 = (int(target_h) - nh) // 2
    x0 = (int(target_w) - nw) // 2
    cell[y0 : y0 + nh, x0 : x0 + nw] = resized
    return cell


def _stitch_rgb(rgbs: list[np.ndarray], gap_px: int = 0) -> np.ndarray:
    if not rgbs:
        raise ValueError("No images to stitch")
    gap = max(0, int(gap_px))
    target_h = max(int(r.shape[0]) for r in rgbs)
    target_w = max(int(r.shape[1]) for r in rgbs)
    cells = [_letterbox_rgb(r, target_h=target_h, target_w=target_w) for r in rgbs]
    canvas = np.zeros((target_h, len(cells) * target_w + max(0, len(cells) - 1) * gap, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        x0 = i * (target_w + gap)
        canvas[:, x0 : x0 + target_w] = cell
    return canvas


def _save_training_curve(history: list[dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    epochs = [int(h["epoch"]) for h in history]
    corr = [float(h["correlation"]) for h in history]
    best = [float(h["best_correlation"]) for h in history]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#111")
    ax.plot(epochs, corr, label="epoch correlation", color="#58a6ff", linewidth=1.5)
    ax.plot(epochs, best, label="best so far", color="#3fb950", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation")
    ax.set_title("Peak-token MiCS training")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_previews(
    trainer: PeakTokenMiCSTrainer,
    preview_paths: list[Path],
    *,
    run_dir: Path,
    tag: str,
    norm_mode: str,
    transpose_msi: bool,
    gap_px: int,
) -> dict[str, Any]:
    preview_dir = run_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cells: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for npy_path in preview_paths:
        msi = _load_msi(npy_path, transpose_msi=transpose_msi, normalization=norm_mode)
        valid_mask = msi.sum(axis=-1) > 0
        mass_axis = resolve_mass_axis(npy_path, msi.shape[-1])
        rgb = trainer.predict_rgb(msi, valid_mask=valid_mask, mass_axis=mass_axis)
        out_png = preview_dir / f"{tag}__{npy_path.stem}.png"
        Image.fromarray(rgb).save(out_png)
        cells.append(rgb)
        records.append({"npy_path": str(npy_path), "preview_png": str(out_png)})
        del msi
        gc.collect()
    mosaic_path = None
    if cells:
        mosaic = _stitch_rgb(cells, gap_px=gap_px)
        mosaic_path = preview_dir / f"{tag}__mosaic.png"
        Image.fromarray(mosaic).save(mosaic_path)
    return {"tag": tag, "slides": records, "mosaic_path": str(mosaic_path) if mosaic_path else None}


@hydra.main(version_base=None, config_path="configs", config_name="train_peak_token_mics")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = run_dir / "training_log.jsonl"

    discovered = _discover_npy_paths(cfg)
    npy_paths, slide_manifest = _select_training_slides(discovered, cfg)
    logger.info("Training on %d slide(s)", len(npy_paths))

    norm_mode = _resolve_normalization_mode(str(cfg.data.normalization))
    transpose_msi = bool(cfg.data.transpose_msi)
    trainer_cfg = _trainer_cfg(cfg)
    samples_per_slide = int(OmegaConf.select(cfg, "model.samples_per_slide", default=0))
    if samples_per_slide > 0:
        trainer_cfg.num_samples = max(int(trainer_cfg.num_samples), samples_per_slide * len(npy_paths))
        logger.info("Scaled num_samples -> %d (%d/slide)", trainer_cfg.num_samples, samples_per_slide)
    trainer = PeakTokenMiCSTrainer(trainer_cfg)
    fit_opts = _fit_opts(cfg, run_dir)

    preview_indices = OmegaConf.to_container(
        OmegaConf.select(cfg, "output.preview_slide_indices", default=[]),
        resolve=True,
    )
    if not isinstance(preview_indices, list) or not preview_indices:
        n_prev = min(int(OmegaConf.select(cfg, "output.max_preview_slides", default=6)), len(npy_paths))
        preview_indices = list(range(n_prev))
    preview_paths = [npy_paths[int(i)] for i in preview_indices if 0 <= int(i) < len(npy_paths)]

    def _load_slide_from_disk(slide: SlideRecord) -> tuple[np.ndarray, np.ndarray]:
        p = Path(slide.npy_path)
        msi = _load_msi(p, transpose_msi=transpose_msi, normalization=norm_mode)
        mask = msi.sum(axis=-1) > 0
        return msi, mask

    trainer.set_slide_loader(_load_slide_from_disk)

    lazy_register = bool(fit_opts.release_slide_memory) or len(npy_paths) > 1
    for npy_path in npy_paths:
        arr = np.load(str(npy_path), mmap_mode="r")
        n_mz = int(arr.shape[-1])
        mass_axis = resolve_mass_axis(npy_path, n_mz)
        if lazy_register:
            valid_mask = _load_valid_mask(npy_path, transpose_msi=transpose_msi)
            if not np.any(valid_mask):
                raise ValueError(f"Empty tissue mask: {npy_path}")
            trainer.add_slide(
                None,
                valid_mask=valid_mask,
                mass_axis=mass_axis,
                name=npy_path.stem,
                npy_path=str(npy_path),
            )
        else:
            msi = _load_msi(npy_path, transpose_msi=transpose_msi, normalization=norm_mode)
            valid_mask = msi.sum(axis=-1) > 0
            if not np.any(valid_mask):
                raise ValueError(f"Empty tissue mask: {npy_path}")
            trainer.add_slide(
                msi,
                valid_mask=valid_mask,
                mass_axis=mass_axis,
                name=npy_path.stem,
                npy_path=str(npy_path),
            )
            del msi
        gc.collect()

    viz_every = int(OmegaConf.select(cfg, "training.visualize_every_epochs", default=10))
    gap_px = int(OmegaConf.select(cfg, "output.mosaic_gap_px", default=2))
    preview_log: list[dict[str, Any]] = []

    def _on_epoch_end(epoch: int, rec: dict[str, float], tr: PeakTokenMiCSTrainer) -> None:
        line = json.dumps(rec)
        with log_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        curve_path = run_dir / "training_curve.png"
        _save_training_curve(tr._training_history, curve_path)
        if viz_every > 0 and (epoch + 1) % viz_every == 0:
            summary = _save_previews(
                tr,
                preview_paths,
                run_dir=run_dir,
                tag=f"epoch_{epoch + 1:04d}",
                norm_mode=norm_mode,
                transpose_msi=transpose_msi,
                gap_px=gap_px,
            )
            preview_log.append(summary)
            logger.info("Wrote preview mosaic: %s", summary.get("mosaic_path"))

    fit_opts.on_epoch_end = _on_epoch_end

    stats = trainer.fit(fit_opts)
    ckpt_path = run_dir / "peak_token_mics.pt"
    trainer.save(ckpt_path, epoch=int(stats.get("epochs_completed", 0)), tag="best_final")

    final_preview = _save_previews(
        trainer,
        preview_paths,
        run_dir=run_dir,
        tag="final",
        norm_mode=norm_mode,
        transpose_msi=transpose_msi,
        gap_px=gap_px,
    )

    manifest = {
        "checkpoint": str(ckpt_path),
        "checkpoints_dir": str(fit_opts.checkpoint_dir),
        "fit_stats": stats,
        "slide_manifest": slide_manifest,
        "n_slides_trained": len(npy_paths),
        "training_history": trainer._training_history,
        "preview_runs": preview_log,
        "final_preview": final_preview,
        "training_log_jsonl": str(log_jsonl),
        "training_curve_png": str(run_dir / "training_curve.png"),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (run_dir / "train_peak_token_mics_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    logger.info("Done. checkpoint=%s  slides=%d  best_corr=%.4f  elapsed=%.2fh",
                ckpt_path, len(npy_paths), stats.get("best_correlation", 0.0), stats.get("elapsed_hours", 0.0))


if __name__ == "__main__":
    main()
