"""
Batch H&E tile embedding + MiCS visualization pipeline.

For each whole-slide H&E image:
  1. ATD (automatic tissue detection) on a downsampled overview
  2. Extract 256×256 tiles at ~20× magnification on a non-overlapping grid
  3. Encode tiles with a pathology foundation model (UNI2-h or OpenMidnight)
  4. Scatter embeddings to a 2-D grid, run MiCS, build a visualization panel

Run from repo root:
  python scripts/batch_he_embed_mics_viz.py
  python scripts/batch_he_embed_mics_viz.py foundation.model=openmidnight
  python scripts/batch_he_embed_mics_viz.py extract.dry_run=true
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msi_visual.extract.he_thumbnail import DEFAULT_WSI_EXTENSIONS, find_wsi_files
from msi_visual.extract.he_tiles_atd import (
    build_he_mosaic,
    grid_shape_from_plan,
    extract_tiles_from_slide,
    save_thumbnail_overview,
)
from msi_visual.he_foundation.embedding_map import (
    prepare_mics_input,
    save_embedding_artifacts,
    scatter_embeddings,
    tile_coords_array,
)
from msi_visual.viz.he_overlay import alpha_blend_rgb, resize_rgb

logger = logging.getLogger(__name__)


def _resolve_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _parse_extensions(raw) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_WSI_EXTENSIONS
    items = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(items, (list, tuple)) or not items:
        return DEFAULT_WSI_EXTENSIONS
    out: list[str] = []
    for item in items:
        s = str(item).strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = f".{s}"
        out.append(s)
    return tuple(out) if out else DEFAULT_WSI_EXTENSIONS


def _output_dir_for_slide(
    input_root: Path,
    output_root: Path,
    slide_path: Path,
    *,
    preserve_relative_dirs: bool,
) -> Path:
    stem = slide_path.stem
    if preserve_relative_dirs:
        rel_parent = slide_path.parent.relative_to(input_root)
        if str(rel_parent) in (".", ""):
            return output_root / stem
        return output_root / rel_parent / stem
    return output_root / stem


def _build_mics_kwargs(cfg: DictConfig, seed: int) -> dict[str, Any]:
    from scripts.benchmark_parametric_methods import _mics_kwargs

    train_path = _resolve_path(str(cfg.mics.train_config))
    train_cfg = OmegaConf.load(str(train_path))
    bench_stub = OmegaConf.create(
        {
            "mics_lmc": {"defaults_yaml": str(train_path)},
            "benchmark": OmegaConf.to_container(cfg.mics.benchmark, resolve=True),
        }
    )
    overrides = OmegaConf.to_container(cfg.mics.model_overrides, resolve=True) if cfg.mics.model_overrides else {}
    mk = _mics_kwargs(
        bench_stub,
        str(cfg.mics.benchmark.sampling_mode),
        seed,
        group_overrides=dict(overrides or {}),
    )
    return mk


def _run_mics_on_embedding(mics_input: np.ndarray, cfg: DictConfig, seed: int) -> np.ndarray:
    from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC
    from scripts.benchmark_parametric_methods import _to_uint8_rgb

    mk = _build_mics_kwargs(cfg, seed)
    logger.info(
        "MiCS on H&E embeddings: clusters=%s beta=%s epochs=%s channels=%s",
        mk.get("clusters"),
        mk.get("beta"),
        mk.get("num_epochs"),
        mics_input.shape[-1],
    )
    model = MSIParametricMiCSLMC(**mk)
    model.fit(mics_input)
    viz = model.predict(mics_input)
    return _to_uint8_rgb(viz)


def _process_slide(
    slide_path: Path,
    out_dir: Path,
    cfg: DictConfig,
    *,
    encoder_bundle,
    device,
) -> dict[str, Any]:
    ex = cfg.extract
    fd = cfg.foundation
    viz_cfg = cfg.visualization
    seed = int(cfg.seed)

    thumb_path = out_dir / str(viz_cfg.thumbnail_filename)
    embed_dir = out_dir / str(cfg.output.embeddings_subdir)
    viz_dir = out_dir / str(cfg.output.viz_subdir)
    panel_path = viz_dir / str(viz_cfg.panel_filename)

    if bool(ex.skip_existing) and panel_path.is_file():
        logger.info("Skip existing panel: %s", panel_path)
        return {"slide": str(slide_path), "status": "skipped", "panel": str(panel_path)}

    stride_px = int(ex.tile_stride_px) if ex.tile_stride_px is not None else int(ex.tile_px)

    plan, mask_l0, atd_thumb, tiles = extract_tiles_from_slide(
        slide_path,
        target_mpp=float(ex.target_mpp),
        tile_px=int(ex.tile_px),
        tile_stride_px=int(ex.tile_stride_px) if ex.tile_stride_px is not None else None,
        atd_max_size=int(ex.atd_max_size),
        min_tissue_frac=float(ex.min_tissue_frac),
    )
    gh, gw = grid_shape_from_plan(plan)
    logger.info(
        "Slide %s | grid=%dx%d | tiles=%d | tile=%d stride=%d (l0 step=%d) mpp=%.3f",
        slide_path.name,
        gh,
        gw,
        len(tiles),
        int(ex.tile_px),
        stride_px,
        plan.step_l0,
        plan.mpp,
    )
    if len(tiles) > 50000:
        logger.warning(
            "Large tile count (%d). Consider extract.tile_stride_px=%d or dry_run first.",
            len(tiles),
            min(stride_px * 2, int(ex.tile_px)),
        )

    if bool(ex.dry_run):
        return {
            "slide": str(slide_path),
            "status": "dry_run",
            "n_tiles": len(tiles),
            "grid_h": gh,
            "grid_w": gw,
        }

    from msi_visual.he_foundation.encoder import encode_tiles

    model, transform, embed_dim, model_tag = encoder_bundle
    tile_rgbs = [t.rgb for t in tiles]
    features = encode_tiles(
        model,
        transform,
        tile_rgbs,
        device=device,
        batch_size=int(fd.batch_size),
        model_input_px=int(fd.model_input_px),
        tile_px=int(ex.tile_px),
    )
    embed_map, tissue_mask = scatter_embeddings(tiles, features, plan)
    coords = tile_coords_array(tiles)

    meta = {
        "slide": str(slide_path),
        "foundation_model": model_tag,
        "target_mpp": float(ex.target_mpp),
        "tile_px": int(ex.tile_px),
        "tile_stride_px": stride_px,
        "grid_h": gh,
        "grid_w": gw,
        "n_tiles": len(tiles),
        "embed_dim": embed_dim,
        "tile_size_l0": plan.tile_size_l0,
        "read_level": plan.read_level,
        "mpp": plan.mpp,
    }
    mics_input, mics_meta = prepare_mics_input(
        embed_map,
        tissue_mask,
        pca_dims=int(cfg.mics.pca_dims) if cfg.mics.pca_dims else None,
        random_state=seed,
    )
    meta.update(mics_meta)

    save_embedding_artifacts(
        embed_dir,
        embed_map=embed_map,
        tissue_mask=tissue_mask,
        features=features,
        coords=coords,
        meta=meta,
    )
    np.save(out_dir / "mics_input.npy", mics_input.astype(np.float32))

    from scripts.benchmark_parametric_methods import _seed_everything

    _seed_everything(seed)
    mics_rgb = _run_mics_on_embedding(mics_input, cfg, seed)

    mosaic_raw = OmegaConf.select(viz_cfg, "mosaic_tile_px")
    display_px = stride_px if mosaic_raw is None else int(mosaic_raw)
    he_mosaic = build_he_mosaic(tiles, gh, gw, tile_px=int(ex.tile_px), display_px=display_px)
    thumb_rgb = save_thumbnail_overview(
        atd_thumb,
        mask_l0,
        thumb_path,
        max_size=int(viz_cfg.thumbnail_max_size),
    )

    # Align MiCS map to mosaic resolution for overlay / panel rows.
    mics_up = resize_rgb(mics_rgb, he_mosaic.shape[0], he_mosaic.shape[1])
    mask_up = resize_rgb(
        (tissue_mask.astype(np.uint8) * 255)[..., None].repeat(3, axis=-1),
        he_mosaic.shape[0],
        he_mosaic.shape[1],
    )[:, :, 0] > 0

    panel_items: list[tuple[str, np.ndarray]] = [("H&E mosaic", he_mosaic)]
    if bool(viz_cfg.include_thumbnail):
        panel_items.insert(0, ("H&E overview", thumb_rgb))
    panel_items.append(("MiCS (HE embeddings)", mics_up))

    if bool(viz_cfg.overlay.enabled):
        blended = alpha_blend_rgb(
            he_mosaic,
            mics_up,
            float(viz_cfg.overlay.alpha),
            mask=mask_up if bool(viz_cfg.overlay.mask_to_tissue) else None,
        )
        panel_items.append((f"H&E + MiCS overlay (α={viz_cfg.overlay.alpha})", blended))

    viz_dir.mkdir(parents=True, exist_ok=True)
    from scripts.generate_visualization_panel import save_panel_png

    save_panel_png(
        panel_items,
        panel_path,
        ncols=int(viz_cfg.panel_columns),
        cell_max_size=int(viz_cfg.cell_max_size),
        dpi=int(viz_cfg.dpi),
        figsize_scale=float(viz_cfg.figsize_scale),
        panel_full_size=bool(viz_cfg.panel_full_size),
    )

    with (viz_dir / "panel_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return {
        "slide": str(slide_path),
        "status": "ok",
        "n_tiles": len(tiles),
        "grid_h": gh,
        "grid_w": gw,
        "panel": str(panel_path),
        "embeddings_dir": str(embed_dir),
    }


@hydra.main(version_base=None, config_path="configs", config_name="batch_he_embed_mics_viz")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ex = cfg.extract

    input_root = _resolve_path(ex.input_root)
    output_root = _resolve_path(ex.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    slides = find_wsi_files(
        input_root,
        recursive=bool(ex.recursive),
        extensions=_parse_extensions(ex.extensions),
        include_glob=ex.include_glob,
    )
    if not slides:
        raise FileNotFoundError(f"No WSI files under {input_root}")

    logger.info("Found %d slide(s) under %s", len(slides), input_root)

    encoder_bundle = None
    device = None
    if not bool(ex.dry_run):
        import torch
        from msi_visual.he_foundation.encoder import load_he_encoder

        device = torch.device(str(cfg.foundation.device)) if cfg.foundation.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        encoder_bundle = load_he_encoder(
            str(cfg.foundation.model),
            device=str(device),
            local_checkpoint=OmegaConf.select(cfg, "foundation.local_checkpoint"),
            dinov2_repo=OmegaConf.select(cfg, "foundation.dinov2_repo"),
        )

    rows: list[dict[str, Any]] = []
    n_fail = 0
    for i, slide_path in enumerate(slides, start=1):
        out_dir = _output_dir_for_slide(
            input_root,
            output_root,
            slide_path,
            preserve_relative_dirs=bool(ex.preserve_relative_dirs),
        )
        logger.info("[%d/%d] %s", i, len(slides), slide_path.name)
        try:
            row = _process_slide(slide_path, out_dir, cfg, encoder_bundle=encoder_bundle, device=device)
            rows.append(row)
        except Exception as exc:
            n_fail += 1
            logger.exception("Failed: %s", slide_path)
            rows.append({"slide": str(slide_path), "status": "error", "error": str(exc)})

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    if bool(ex.write_manifest):
        manifest_path = run_dir / "batch_he_embed_mics_manifest.csv"
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote manifest: %s", manifest_path)

    cfg_path = run_dir / "config_resolved.yaml"
    cfg_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
