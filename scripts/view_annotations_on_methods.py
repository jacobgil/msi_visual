#!/usr/bin/env python3
"""
For each benchmark-style method (TOP3 / PCA3D / NMF3D / Parametric UMAP / MiCS-LMC), compute
the RGB visualization on an MSI cube, then draw the same VIA annotation boundaries as
``view_annotation_boundaries.py``.

TOP3 is ``msi_visual.percentile_ratio.TOP3`` (per-pixel three largest intensities).
Use ``annotation.top3_to_lab: false`` for B/G/R channel mapping (usually easier to see edges);
``true`` matches the benchmark’s LAB→RGB path. Overlays use a dark halo behind white boundaries.

When ``edge_metrics.enabled`` is true, loads parts of ``debug_visualization_metrics.yaml`` (not
``edge_detection`` / ``hd_edges``; those come from this run's YAML), runs
``DebugNMFEvaluator.evaluate_edge_agreement_and_maps`` (AUC-ROC/PR, Spearman, SSIM, multi-percentile AUC,
plus threshold metrics from ``edge_evaluation``), saves ``edge_metrics.csv``, and writes annotation overlays
on each normalized edge map under ``edge_metrics.maps_subdir``.
If ``edge_detection.hd_method`` is a list, each HD method is evaluated (one CSV row per method;
multi-method runs add ``__hd_<method>__`` to edge-map PNG filenames).

With ``annotation.paths``, use ``annotation.slide: 0`` for one row (default), ``slide: all`` or
``slides: [0,1,2]`` to process multiple ``.npy`` slides in one run (separate outputs per file).

If VIA polygons were drawn on a transformed view of the cube, set ``annotation.transpose_spatial: true``
or ``annotation.transpose_spatial_mode: chain_flip`` (default when the bool is true). ``chain_flip``
matches ``data.transpose().transpose(1, 2, 0)[::-1, :, :]`` on ``H×W×C``; use ``swap_hw`` for a plain
``(1,0,2)`` swap only. Combine with ``data.transpose_msi`` when the file is ``C×H×W``.

``annotation.enhance_overlay_rgb`` (default on) stretches luminance in LAB before ``render_overlay`` so
the MSI RGB is not a flat gray after float conversion. Edge-map PNGs under ``edge_metrics.maps_subdir``
stay grayscale by design (single-channel edge maps). When ``edge_metrics.save_maps_separate`` is true,
raw edge arrays and preview PNGs (no annotations) are also written under ``edge_metrics.separate_subdir``.
"""
from __future__ import annotations

import csv
import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from msi_visual.nmf_3d import NMF3D
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC
from msi_visual.parametric_umap import UMAPVirtualStain
from msi_visual.pca_3d import PCA3D
from msi_visual.percentile_ratio import TOP3
from scripts.debug_visualization_metrics import DebugNMFEvaluator, normalize_hd_methods_list
from scripts.benchmark_parametric_methods import (
    _histogram_equalize_rgb,
    _maybe_lab_to_rgb,
    _mics_kwargs,
    _path_identifier,
    _resolve_mics_parameter_groups,
    _safe_token,
    _seed_everything,
    _to_uint8_rgb,
    _umap_kwargs,
    _load_msi,
)
from scripts.view_annotation_boundaries import (
    annotation_index_from_npy_path,
    apply_annotation_spatial_transform,
    load_annotation_polygons,
    render_overlay,
    resolve_transpose_spatial_mode,
    _infer_hw_from_polygons,
    _keep_from_cfg,
    _npy_from_paths_list,
    _opt_path,
)
from hydra.core.hydra_config import HydraConfig


def _stretch_lab_l_percentile(
    rgb_u8: np.ndarray,
    tissue_mask: np.ndarray,
    lo_p: float = 1.0,
    hi_p: float = 99.0,
) -> np.ndarray:
    """
    Stretch luminance (L in LAB) using percentiles on masked tissue; A/B unchanged so color is kept.
    Helps when float→uint8 left the image flat gray or nearly black.
    """
    rgb = np.asarray(rgb_u8, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32)
    vm = tissue_mask & np.isfinite(l_f)
    if not np.any(vm):
        return rgb
    lo = float(np.percentile(l_f[vm], lo_p))
    hi = float(np.percentile(l_f[vm], hi_p))
    if hi <= lo:
        return rgb
    l2 = np.clip((l_f - lo) / (hi - lo + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    out[~tissue_mask] = 0
    return out


def _scalar_map_to_rgb_u8(z: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Float edge map [H,W] → uint8 RGB for ``render_overlay``.

    Uses 1–99% percentiles; if that range is empty (common for sparse edges), falls back to
    min–max on valid pixels so the image is not all black.
    """
    z = np.asarray(z, dtype=np.float32)
    vm = valid_mask & np.isfinite(z)
    out = np.zeros((*z.shape[:2], 3), dtype=np.uint8)
    if not np.any(vm):
        return out
    vals = z[vm]
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.0))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        g = np.zeros_like(z, dtype=np.float32)
    else:
        g = np.clip((z - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    g = (g * 255.0).astype(np.uint8)
    g[~valid_mask] = 0
    return np.stack([g, g, g], axis=-1)


def _ann_category_keys(ann: Any) -> Tuple[str, ...]:
    raw = OmegaConf.select(ann, "category_keys")
    if raw is None:
        return ("main_cat1", "Allen_name", "sub_cat2")
    items = OmegaConf.to_container(raw, resolve=True)
    if not items:
        return ("main_cat1", "Allen_name", "sub_cat2")
    return tuple(str(x) for x in items)


def _expand_method_jobs(cfg: DictConfig) -> List[Tuple[str, str]]:
    raw = OmegaConf.to_container(cfg.methods, resolve=True)
    if not raw:
        return [
            ("top3", "default"),
            ("pca", "default"),
            ("parametric_umap", "default"),
            ("nmf_3d", "default"),
        ]
    mics_resolved = _resolve_mics_parameter_groups(cfg)
    out: List[Tuple[str, str]] = []
    for m in raw:
        ms = str(m).strip()
        if ms == "parametric_mics_lmc":
            for group_name, _ in mics_resolved:
                out.append(("parametric_mics_lmc", group_name))
        elif ms.startswith("parametric_mics_lmc__"):
            out.append(("parametric_mics_lmc", ms.split("__", 1)[1]))
        else:
            out.append((ms, "default"))
    return out


def _method_tag(method_name: str, method_variant: str, has_mics_parameter_groups: bool) -> str:
    if method_name == "parametric_mics_lmc" and has_mics_parameter_groups:
        return f"{method_name}__{method_variant}"
    return method_name


def _compute_viz_rgb(
    method_name: str,
    method_variant: str,
    cfg: DictConfig,
    msi_img: np.ndarray,
    msi_mask: np.ndarray,
    sampling_mode: str,
    seed: int,
    equalize: bool,
    mics_overrides_by_group: dict[str, dict[str, Any]],
) -> np.ndarray:
    model = None
    try:
        if method_name == "parametric_mics_lmc":
            mics_kwargs = _mics_kwargs(
                cfg,
                str(sampling_mode),
                seed,
                group_overrides=mics_overrides_by_group.get(method_variant, {}),
            )
            model = MSIParametricMiCSLMC(**mics_kwargs)
            model.fit(msi_img)
            viz = model.predict(msi_img)
        elif method_name == "parametric_umap":
            umap_kwargs = _umap_kwargs(cfg, str(sampling_mode), seed)
            model = UMAPVirtualStain(**umap_kwargs)
            model.fit([msi_img], keras_fit_kwargs={})
            viz = model.predict(msi_img)
        elif method_name == "pca":
            model = PCA3D()
            model.fit([msi_img])
            viz = model.predict(msi_img)
        elif method_name == "top3":
            model = TOP3()
            sel_lab = OmegaConf.select(cfg, "annotation.top3_to_lab")
            use_lab = True if sel_lab is None else bool(sel_lab)
            viz = model(msi_img, to_lab=use_lab)
            if not use_lab:
                viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
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
        if equalize:
            viz_rgb = _histogram_equalize_rgb(viz_rgb)
        # Re-apply in case any step returned float01; _to_uint8_rgb maps max→255 for dim floats.
        viz_rgb = _to_uint8_rgb(viz_rgb)
        viz_rgb = np.asarray(viz_rgb, dtype=np.uint8).copy()
        viz_rgb[~msi_mask] = 0
        return viz_rgb
    finally:
        if model is not None and hasattr(model, "release_resources"):
            try:
                model.release_resources()
            except Exception:
                pass
        del model
        gc.collect()


def _slide_indices_from_cfg(ann: Any, n_paths: int) -> List[int]:
    """Which ``annotation.paths`` rows to run. Default ``slide: 0``; ``slides: [...]`` or ``slide: all`` for many."""
    if n_paths <= 0:
        return [0]
    slides_sel = OmegaConf.select(ann, "slides")
    if slides_sel is not None:
        raw = OmegaConf.to_container(slides_sel, resolve=True)
        if isinstance(raw, (list, tuple)):
            out: List[int] = []
            for x in raw:
                try:
                    i = int(x)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < n_paths:
                    out.append(i)
            if out:
                return list(dict.fromkeys(out))
            raise ValueError("annotation.slides: no valid indices for paths (check slide indices vs len(paths))")
    slide_raw = OmegaConf.select(ann, "slide")
    if slide_raw is None:
        return [0]
    if isinstance(slide_raw, str) and slide_raw.strip().lower() in ("all", "*"):
        return list(range(n_paths))
    if isinstance(slide_raw, (list, tuple, ListConfig)):
        raw = OmegaConf.to_container(slide_raw, resolve=True)
        if isinstance(raw, (list, tuple)):
            out = []
            for x in raw:
                try:
                    i = int(x)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < n_paths:
                    out.append(i)
            if out:
                return list(dict.fromkeys(out))
    slide = int(slide_raw)
    if slide < 0 or slide >= n_paths:
        raise IndexError(f"slide={slide} out of range for paths (len={n_paths})")
    return [slide]


def _resolve_annotation_runs(ann: Any) -> List[Tuple[Path, str, list[str], Any, Tuple[str, ...]]]:
    """One or more (npy_path, index_str, ignore, keep, category_keys) from ``annotation``."""
    annotation_json = _opt_path(OmegaConf.select(ann, "annotation_json"))
    if annotation_json is None or not annotation_json.is_file():
        raise FileNotFoundError(f"annotation_json not found: {annotation_json}")

    paths_raw = OmegaConf.select(ann, "paths")
    paths_rows: list[Any] | None = None
    if paths_raw is not None:
        paths_rows = OmegaConf.to_container(paths_raw, resolve=True)
        if not isinstance(paths_rows, list):
            paths_rows = None

    npy_opt = _opt_path(OmegaConf.select(ann, "npy"))
    ignore = list(OmegaConf.to_container(OmegaConf.select(ann, "ignore"), resolve=True) or [])
    keep = _keep_from_cfg(OmegaConf.select(ann, "keep"))
    category_keys = _ann_category_keys(ann)

    index_raw = OmegaConf.select(ann, "index")
    fixed_index = str(index_raw).strip() if index_raw is not None and str(index_raw).strip() != "" else None

    runs: List[Tuple[Path, str, list[str], Any, Tuple[str, ...]]] = []

    def _append_run(npy: Path | None) -> None:
        if npy is None or not npy.is_file():
            raise FileNotFoundError(f"MSI .npy not found: {npy}")
        if fixed_index is not None:
            index_str = fixed_index
        else:
            index_str = annotation_index_from_npy_path(npy)
        runs.append((npy, index_str, ignore, keep, category_keys))

    if npy_opt is not None:
        _append_run(npy_opt)
        return runs

    if paths_rows:
        for si in _slide_indices_from_cfg(ann, len(paths_rows)):
            npy = _npy_from_paths_list(paths_rows, si)
            if npy is None:
                raise ValueError(f"paths[{si}] must be [npy_path, optional_label] or {{npy: path}}")
            _append_run(npy)
        return runs

    raise ValueError("Set annotation.npy or annotation.paths (+ slide / slides)")


def _merge_debug_edge_metrics_cfg(cfg: DictConfig) -> DictConfig:
    """Load spatial_normalization / edge_agreement / clahe / … from debug_visualization_metrics.yaml.

    ``edge_detection`` and ``hd_edges`` are *not* merged here so this script's Hydra config
    (benchmark + ``view_annotations_on_methods.yaml``) controls HD/RGB edge methods.
    """
    dbg_path = Path(__file__).resolve().parent / "configs" / "debug_visualization_metrics.yaml"
    if not dbg_path.is_file():
        return cfg
    dbg = OmegaConf.load(dbg_path)
    keys = (
        "spatial_normalization",
        "edge_agreement",
        "clahe",
        "edge_evaluation",
    )
    patch = OmegaConf.create({k: dbg[k] for k in keys if OmegaConf.select(dbg, k) is not None})
    return OmegaConf.merge(cfg, patch)


@hydra.main(version_base=None, config_path="configs", config_name="view_annotations_on_methods")
def main(cfg: DictConfig) -> None:
    cfg = _merge_debug_edge_metrics_cfg(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    ann = cfg.annotation
    try:
        runs = _resolve_annotation_runs(ann)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    annotation_json = _opt_path(ann.annotation_json)
    assert annotation_json is not None

    boundary_width = int(OmegaConf.select(ann, "boundary_width") or 2)
    continue_on_error = bool(OmegaConf.select(ann, "continue_on_method_error") or True)

    transpose_msi = bool(cfg.data.transpose_msi)
    tic_normalize = bool(cfg.data.tic_normalize)
    transpose_spatial_mode = resolve_transpose_spatial_mode(ann)

    modes = OmegaConf.to_container(cfg.benchmark.sampling_modes, resolve=True)
    sampling_mode = str(modes[0]) if modes else "random"
    repeat_idx = 0
    seed = int(cfg.benchmark.seed_start) + repeat_idx * int(cfg.benchmark.seed_step)
    _seed_everything(seed)

    equalize = bool(getattr(cfg.output, "equalize_visualizations", True))
    has_mics_parameter_groups = bool(getattr(cfg.mics_lmc, "parameter_groups", None))
    mics_groups = _resolve_mics_parameter_groups(cfg)
    mics_overrides_by_group = {group_name: overrides for group_name, overrides in mics_groups}

    jobs = _expand_method_jobs(cfg)
    out_sub = Path(str(getattr(cfg.output, "annotated_subdir", "annotated_visualizations")))
    out_dir = output_dir / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_metrics_cfg = getattr(cfg, "edge_metrics", None)
    run_edge = bool(getattr(edge_metrics_cfg, "enabled", False)) if edge_metrics_cfg is not None else False
    edge_rows: List[Dict[str, Any]] = []

    print(
        f"[view_annotations_on_methods] {len(runs)} MSI file(s) | "
        f"transpose_spatial_mode={transpose_spatial_mode}",
        flush=True,
    )

    for npy_path, index_str, ignore, keep, category_keys in runs:
        polys, cats = load_annotation_polygons(
            annotation_json,
            index_str,
            ignore=ignore,
            keep=keep,
            category_keys=category_keys,
        )
        if not polys:
            print(
                f"[skip] no regions for index prefix {index_str!r} ({npy_path})",
                file=sys.stderr,
                flush=True,
            )
            continue

        h0, w0 = _infer_hw_from_polygons(polys)
        height = OmegaConf.select(ann, "height")
        width = OmegaConf.select(ann, "width")

        msi_img = _load_msi(npy_path, transpose_msi, tic_normalize)
        if transpose_spatial_mode != "none":
            msi_img = apply_annotation_spatial_transform(msi_img, transpose_spatial_mode)

        if height is not None or width is not None:
            shape_hw = (int(height or h0), int(width or w0))
        else:
            shape_hw = (int(msi_img.shape[0]), int(msi_img.shape[1]))

        if msi_img.shape[:2] != shape_hw:
            print(
                f"Warning: MSI shape {msi_img.shape[:2]} != annotation canvas {shape_hw}; "
                "check data.transpose_msi, annotation.transpose_spatial_mode, height/width.",
                file=sys.stderr,
            )

        msi_mask = msi_img.sum(axis=-1) > 0
        file_id = _path_identifier(npy_path)

        edge_evaluator: DebugNMFEvaluator | None = None
        hd_methods: List[str] = []
        multi_hd_edge = False
        if run_edge:
            hd_methods = normalize_hd_methods_list(getattr(cfg, "edge_detection", None))
            multi_hd_edge = len(hd_methods) > 1
            edge_evaluator = DebugNMFEvaluator(msi_img, cfg, edge_only=True)

        for method_name, method_variant in jobs:
            method_tag = _method_tag(method_name, method_variant, has_mics_parameter_groups)
            method_file_tag = _safe_token(method_tag)
            t0 = time.perf_counter()
            try:
                viz_rgb = _compute_viz_rgb(
                    method_name,
                    method_variant,
                    cfg,
                    msi_img,
                    msi_mask,
                    sampling_mode,
                    seed,
                    equalize,
                    mics_overrides_by_group,
                )
                if (
                    method_name == "top3"
                    and bool(OmegaConf.select(cfg, "annotation.equalize_top3_for_overlay") is not False)
                    and not equalize
                ):
                    viz_rgb = _histogram_equalize_rgb(viz_rgb)
                    viz_rgb[~msi_mask] = 0

                if run_edge and edge_evaluator is not None:
                    em_sub = str(getattr(edge_metrics_cfg, "maps_subdir", "edge_map_annotations"))
                    maps_dir = output_dir / em_sub
                    maps_dir.mkdir(parents=True, exist_ok=True)
                    primary_hd = str(edge_evaluator.edge_method_cfg.get("hd_method", "gradient"))
                    vm_edge = edge_evaluator.valid_mask
                    for hd_m in hd_methods:
                        if hd_m == primary_hd:
                            hde = edge_evaluator.highd_edge
                            hdel = edge_evaluator.highd_edge_linf
                        else:
                            hde, hdel = edge_evaluator.compute_highd_edge_maps_for_hd_method(hd_m)
                        edge_summary, edge_maps = edge_evaluator.evaluate_edge_agreement_and_maps(
                            viz_rgb,
                            highd_edge_override=hde,
                            highd_edge_linf_override=hdel,
                        )
                        row = {
                            "file_id": file_id,
                            "method": method_tag,
                            "edge_agreement_hd_method": hd_m,
                            **edge_summary,
                        }
                        edge_rows.append(row)
                        hd_tag = _safe_token(hd_m)
                        for map_name, emap in edge_maps.items():
                            em = np.asarray(emap, dtype=np.float32)
                            if em.shape[:2] != shape_hw:
                                em = cv2.resize(
                                    em, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR
                                )
                                vm_use = (
                                    cv2.resize(
                                        vm_edge.astype(np.uint8),
                                        (shape_hw[1], shape_hw[0]),
                                        interpolation=cv2.INTER_NEAREST,
                                    ).astype(bool)
                                )
                            else:
                                vm_use = vm_edge

                            safe_map = _safe_token(str(map_name))

                            if bool(getattr(edge_metrics_cfg, "save_maps_separate", True)):
                                sep_sub = str(getattr(edge_metrics_cfg, "separate_subdir", "edge_maps_separate"))
                                sep_dir = output_dir / sep_sub
                                sep_dir.mkdir(parents=True, exist_ok=True)
                                stem_prefix = f"{file_id}__{method_file_tag}"
                                if multi_hd_edge:
                                    stem_prefix += f"__hd_{hd_tag}"
                                stem_prefix += f"__{safe_map}"
                                np.save(sep_dir / f"{stem_prefix}.npy", em.astype(np.float32))
                                prev = _scalar_map_to_rgb_u8(em, vm_use)
                                cv2.imwrite(
                                    str(sep_dir / f"{stem_prefix}.png"),
                                    cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
                                )
                            if bool(getattr(edge_metrics_cfg, "print_map_stats", True)):
                                n_v = int(np.sum(vm_use))
                                if n_v > 0:
                                    vv = em[vm_use]
                                    print(
                                        f"[edge-map] {safe_map} | {file_id} | shape={tuple(em.shape)} "
                                        f"n_valid={n_v} "
                                        f"min={float(np.min(vv)):.6g} max={float(np.max(vv)):.6g} "
                                        f"mean={float(np.mean(vv)):.6g}",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"[edge-map] {safe_map} | {file_id} | shape={tuple(em.shape)} "
                                        "n_valid=0 (mask empty; map may be all black)",
                                        flush=True,
                                    )

                            bg = _scalar_map_to_rgb_u8(em, vm_use)
                            overlay_e = render_overlay(
                                bg,
                                polys,
                                cats,
                                shape_hw,
                                boundary_width=boundary_width,
                                fill_alpha=float(OmegaConf.select(cfg, "annotation.fill_alpha") or 0.45),
                                boundary_halo=int(OmegaConf.select(cfg, "annotation.boundary_halo") or 2),
                            )
                            if multi_hd_edge:
                                edge_png = (
                                    maps_dir
                                    / f"{file_id}__{method_file_tag}__hd_{hd_tag}__{safe_map}__annotated.png"
                                )
                            else:
                                edge_png = maps_dir / f"{file_id}__{method_file_tag}__{safe_map}__annotated.png"
                            cv2.imwrite(str(edge_png), cv2.cvtColor(overlay_e, cv2.COLOR_RGB2BGR))

                if viz_rgb.shape[:2] != shape_hw:
                    viz_rgb = cv2.resize(
                        viz_rgb, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR
                    )
                    if msi_mask.shape[:2] != shape_hw:
                        msk = cv2.resize(
                            msi_mask.astype(np.uint8),
                            (shape_hw[1], shape_hw[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    else:
                        msk = msi_mask
                    viz_rgb[~msk] = 0
                else:
                    msk = msi_mask

                if bool(OmegaConf.select(cfg, "annotation.enhance_overlay_rgb") is not False):
                    lo_e = float(OmegaConf.select(cfg, "annotation.enhance_overlay_l_lo_percentile") or 1.0)
                    hi_e = float(OmegaConf.select(cfg, "annotation.enhance_overlay_l_hi_percentile") or 99.0)
                    viz_rgb = _stretch_lab_l_percentile(viz_rgb, msk, lo_p=lo_e, hi_p=hi_e)

                fill_alpha = float(OmegaConf.select(cfg, "annotation.fill_alpha") or 0.45)
                boundary_halo = int(OmegaConf.select(cfg, "annotation.boundary_halo") or 2)
                overlay = render_overlay(
                    viz_rgb,
                    polys,
                    cats,
                    shape_hw,
                    boundary_width=boundary_width,
                    fill_alpha=fill_alpha,
                    boundary_halo=boundary_halo,
                )
                out_path = out_dir / f"{file_id}__{method_file_tag}__annotated.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(
                    f"[ok] {file_id} | {method_tag} | {out_path.name} | {time.perf_counter() - t0:.2f}s",
                    flush=True,
                )
            except Exception as e:
                print(f"[fail] {method_tag} ({npy_path}): {e}", file=sys.stderr, flush=True)
                if not continue_on_error:
                    raise

    if run_edge and edge_rows:
        emc = getattr(cfg, "edge_metrics", None)
        csv_name = str(getattr(emc, "csv_name", "edge_metrics.csv") if emc is not None else "edge_metrics.csv")
        csv_path = output_dir / csv_name
        keys: List[str] = []
        for row in edge_rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(edge_rows)
        print(f"Wrote edge metrics CSV: {csv_path}", flush=True)

    print(f"Wrote under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
