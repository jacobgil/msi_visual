#!/usr/bin/env python3
"""
Gallery (multi-row): K-means–style maps, MiCS RGB, optional multi-cluster MiCS, optional
``mics.sweep_rows`` (``num_layers``, ``beta``).

**pairing: per_k** (default): For each ``k`` in ``kmeans.k_list`` (= ``mics.k_list``), one
MiCS fit with ``cluster: true`` and a **single** cluster level ``k``. Column ``k``: top =
MiCS cluster panel (``kmeans.mics_cluster_viz``: **dense** = trained head argmax on all
pixels; **sparse** = init K-means on sampled sites only), bottom = ``predict()`` for the
**same** model. Optional ``mics.sweep_rows`` adds full-width multi-cluster MiCS rows
(``mics.combined`` supplies hyperparameters and ``predict_viz``).

**pairing: legacy**: Row 1 = ``kmeans.source`` (standalone ``KmeansSegmentation`` or
``mics_kmeans`` multi-fit rasters); row 2 = separate MiCS runs with ``number_of_components``
= each entry and ``cluster: false`` (fast ablation). No standalone combined-only gallery row.

Precomputed: ``external`` (see YAML). ``kmeans.source=mics_kmeans`` only applies with
``pairing: legacy``.

Optional ``mics.equalize_display``: LAB L histogram or CLAHE on saved MiCS RGB (display only).

Optional ``mics.sweep_rows``: extra full-width rows of multi-cluster MiCS—``num_layers`` /
``beta`` over ``values``, or ``clusters_cumulative`` to add one cluster level at a time
(prefixes of ``mics.cluster_levels`` / ``k_list``).

Config: ``scripts/configs/gallery_kmeans_mics.yaml``
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Tuple

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for p in (_REPO, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_parametric_methods import _mics_kwargs, _resolve_mics_parameter_groups, _seed_everything
from debug_edge_maps import (
    _apply_equalize_rgb_uint8,
    _load_msi,
    _load_visualization,
    _resolve_normalization_mode,
    _to_uint8_rgb,
)
from gallery_top3_nmf_edges import (
    _maybe_downscale,
    _paste_grid,
    _slide_name_from_args_txt,
    _slide_prefixed_basename,
    _slug_filename,
    _vstack_images,
)
from msi_visual.kmeans_segmentation import KmeansSegmentation
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC

logger = logging.getLogger(__name__)


def _maybe_equalize_mics_display_rgb(
    cfg: DictConfig,
    rgb: np.ndarray,
    *,
    is_cluster_panel: bool,
) -> np.ndarray:
    """LAB L-channel equalization for gallery display (hist or CLAHE); does not affect training."""
    sub = OmegaConf.select(cfg, "mics.equalize_display")
    if sub is None or not bool(getattr(sub, "enabled", False)):
        return rgb
    if is_cluster_panel and not bool(getattr(sub, "cluster_panels", False)):
        return rgb
    return _apply_equalize_rgb_uint8(
        np.asarray(rgb, dtype=np.uint8),
        str(getattr(sub, "method", "hist")),
        float(getattr(sub, "clahe_clip_limit", 2.0)),
        int(getattr(sub, "clahe_tile_grid_size", 8)),
    )


def _group_overrides_for_name(bench_cfg: DictConfig, group_name: str | None) -> dict[str, Any]:
    if group_name is None:
        return {}
    raw = str(group_name).strip()
    if not raw or raw.lower() in ("none", "null", "~"):
        return {}
    wanted = raw
    groups = _resolve_mics_parameter_groups(bench_cfg)
    for name, ov in groups:
        if str(name) == wanted:
            return dict(ov)
    available = [n for n, _ in groups]
    raise ValueError(f"mics.mics_parameter_group {wanted!r} not found. Available: {available}")


def _kmeans_rgb_u8(msi: np.ndarray, k: int) -> np.ndarray:
    km = KmeansSegmentation(max(2, int(k)))
    out = km(msi)
    if isinstance(out, list):
        out = out[0]
    arr = np.asarray(out)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        if mx <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255.0).astype(np.float32)
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8, copy=False)


def _resolve_path_str(p: Any) -> Path | None:
    if p is None:
        return None
    s = str(p).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None
    return Path(to_absolute_path(s))


def _find_external_in_section(
    section: Any,
    k: int,
    *,
    dir_path: Path | None,
    filename_fmts: tuple[str, ...],
) -> Path | None:
    """Resolve ``section[k]`` / ``section[str(k)]``, then ``dir_path / fmt.format(k=k)`` for each fmt."""
    if section is not None:
        cont = OmegaConf.to_container(section, resolve=True) if OmegaConf.is_config(section) else section
        if isinstance(cont, dict):
            for key in (str(k), k):
                if key in cont:
                    p = _resolve_path_str(cont[key])
                    if p is not None and p.is_file():
                        return p
    if dir_path is not None and dir_path.is_dir():
        for fmt in filename_fmts:
            cand = dir_path / fmt.format(k=k)
            if cand.is_file():
                return cand
    return None


def _load_viz_external(path: Path, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return _load_visualization(path, (h, w), allow_resize=True)


def _pad_to_width_center(img: np.ndarray, target_w: int, bg: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Pad a uint8 image horizontally so width matches ``target_w`` (centered)."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    pad = target_w - w
    pl = pad // 2
    out = Image.new("RGB", (target_w, h), bg)
    pil = Image.fromarray(img) if img.ndim == 3 else Image.fromarray(img).convert("RGB")
    out.paste(pil, (pl, 0))
    return np.asarray(out, dtype=np.uint8)


def _bench_cfg_for_mics(cfg: DictConfig) -> DictConfig:
    m = cfg.mics
    bench_path = Path(to_absolute_path(str(m.benchmark_config)))
    if not bench_path.is_file():
        raise FileNotFoundError(f"mics.benchmark_config not found: {bench_path}")
    bench_cfg = OmegaConf.load(bench_path)
    if hasattr(bench_cfg.mics_lmc, "defaults_yaml"):
        bench_cfg.mics_lmc.defaults_yaml = to_absolute_path(str(bench_cfg.mics_lmc.defaults_yaml))
    bench_cfg = OmegaConf.merge(
        bench_cfg,
        OmegaConf.create({"benchmark": OmegaConf.to_container(m.benchmark, resolve=True)}),
    )
    return bench_cfg


def _merged_mics_overrides(bench_cfg: DictConfig, cfg: DictConfig) -> dict[str, Any]:
    m = cfg.mics
    group_key = OmegaConf.select(m, "mics_parameter_group")
    group_overrides = _group_overrides_for_name(bench_cfg, group_key)
    raw_panel = OmegaConf.select(m, "mics_model", default={})
    panel_model = OmegaConf.to_container(raw_panel, resolve=True) if raw_panel else {}
    raw_cli = OmegaConf.select(m, "mics_model_overrides", default={})
    cli_overrides = OmegaConf.to_container(raw_cli, resolve=True) if raw_cli else {}
    return {**group_overrides, **panel_model, **cli_overrides}


def _run_mics(
    msi: np.ndarray,
    cfg: DictConfig,
    bench_cfg: DictConfig,
    seed: int,
    extra_overrides: dict[str, Any],
) -> np.ndarray:
    rgb, _model = _run_mics_train(msi, cfg, bench_cfg, seed, extra_overrides)
    return rgb


def _run_mics_train(
    msi: np.ndarray,
    cfg: DictConfig,
    bench_cfg: DictConfig,
    seed: int,
    extra_overrides: dict[str, Any],
) -> Tuple[np.ndarray, MSIParametricMiCSLMC]:
    m = cfg.mics
    base = _merged_mics_overrides(bench_cfg, cfg)
    merged = {**base, **extra_overrides}
    mk = _mics_kwargs(
        bench_cfg,
        str(m.benchmark.sampling_mode),
        seed,
        group_overrides=merged,
    )
    model = MSIParametricMiCSLMC(**mk)
    model.fit(msi)
    viz = model.predict(msi)
    return _to_uint8_rgb(viz), model


def _parse_combined_clusters_ints(comb: Any) -> list[int]:
    raw = getattr(comb, "clusters", [])
    if isinstance(raw, str):
        return [int(x) for x in raw.split("-") if str(x).strip()]
    return [int(x) for x in list(raw)]


def _cluster_levels_match_k_list(comb: Any, ks: list[int]) -> None:
    parsed = _parse_combined_clusters_ints(comb)
    if parsed != ks:
        raise ValueError(
            "kmeans.source=mics_kmeans requires mics.combined.clusters to match kmeans.k_list "
            f"(same length and order). Got clusters={parsed}, kmeans.k_list={ks}."
        )


def _combined_cluster_argmax_display_rgb(model: MSIParametricMiCSLMC, msi: np.ndarray, comb: Any) -> np.ndarray:
    """One trained cluster head’s argmax map (see ``combined.predict_viz: cluster_argmax``)."""
    raw_idx = getattr(comb, "cluster_argmax_level_idx", "last")
    levels = list(getattr(model, "clusters", []) or [])
    n = len(levels)
    if n < 1:
        raise ValueError("cluster_argmax display requires non-empty model.clusters")
    if isinstance(raw_idx, str):
        s = raw_idx.strip().lower()
        if s in ("last", "max", "finest", "highest"):
            level_idx = n - 1
        elif s in ("first", "min", "coarsest", "lowest"):
            level_idx = 0
        else:
            level_idx = int(s)
    else:
        level_idx = int(raw_idx)
    if level_idx < 0:
        level_idx = n + level_idx
    level_idx = int(np.clip(level_idx, 0, n - 1))
    logger.info(
        "Combined MiCS display: cluster_argmax level_idx=%s (k=%s of %s)",
        level_idx,
        levels[level_idx],
        levels,
    )
    return model.predict_cluster_argmax_rgb_u8(msi, level_idx=level_idx)


def _fit_combined_mics_model(
    msi: np.ndarray,
    cfg: DictConfig,
    bench_cfg: DictConfig,
    seed: int,
    comb: Any,
    *,
    clusters_joined: str | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> Tuple[np.ndarray, MSIParametricMiCSLMC]:
    comb_model = OmegaConf.to_container(getattr(comb, "mics_model", {}), resolve=True) or {}
    if clusters_joined is not None:
        clusters_val = clusters_joined
    else:
        raw_cl = getattr(comb, "clusters", None)
        if raw_cl is None:
            raw_cl = [8, 16, 32, 64]
        if isinstance(raw_cl, str):
            clusters_val = raw_cl
        else:
            clusters_val = "-".join(str(int(x)) for x in list(raw_cl))
    n_comp = int(getattr(comb, "number_of_components", 3))
    # comb_model then model_overrides; clusters/cluster/n_comp last so ``cluster_levels`` wins.
    extra = {
        **comb_model,
        **(model_overrides or {}),
        "number_of_components": n_comp,
        "cluster": True,
        "clusters": clusters_val,
    }
    rgb, model = _run_mics_train(msi, cfg, bench_cfg, seed, extra)
    cl = getattr(model, "clusters", None)
    if cl is not None and len(cl) > 0:
        logger.info("Combined MiCS trained with cluster levels: %s", list(cl))
    if model_overrides:
        logger.info("Combined MiCS model overrides: %s", model_overrides)
    mode = str(getattr(comb, "predict_viz", "embedding") or "embedding").strip().lower()
    if mode in ("cluster_argmax", "argmax", "argmax_rgb", "cluster"):
        rgb = _combined_cluster_argmax_display_rgb(model, msi, comb)
    # else: rgb is PCC embedding from ``predict()`` (does not render cluster heads).
    return rgb, model


def _coerce_sweep_value(param_key: str, raw: Any) -> Any:
    if param_key == "beta":
        return float(raw)
    if param_key == "num_layers":
        return int(raw)
    return raw


def _sweep_combined_row(
    msi: np.ndarray,
    cfg: DictConfig,
    bench_cfg: DictConfig,
    seed: int,
    comb: Any,
    *,
    clusters_joined: str | None,
    param_key: str,
    values: list[Any],
    cell_title_fmt: str,
    row_title: str,
    ncols: int,
    col_pad: int,
    cell_gap: int,
    row_gap: int,
    row_lab: int,
    cell_max_seq: Any,
) -> np.ndarray:
    """One gallery row: multi-cluster MiCS with ``param_key`` swept over ``values``."""
    cells: list[np.ndarray] = []
    titles: list[str] = []
    for raw in values:
        v = _coerce_sweep_value(param_key, raw)
        t = (
            cell_title_fmt.replace("{value}", str(v))
            .replace("{param}", param_key)
            .replace("{row_title}", row_title)
        )
        titles.append(t)
        logger.info("%s: training combined MiCS with %s=%s", row_title, param_key, v)
        img = _fit_combined_mics_model(
            msi,
            cfg,
            bench_cfg,
            seed,
            comb,
            clusters_joined=clusters_joined,
            model_overrides={param_key: v},
        )[0]
        img = _maybe_equalize_mics_display_rgb(cfg, img, is_cluster_panel=False)
        img = _maybe_downscale(img, cell_max_seq)
        cells.append(img)
    return _paste_grid(
        cells,
        ncols=min(ncols, len(cells)),
        column_padding=col_pad,
        cell_gap=cell_gap,
        row_gap=row_gap,
        row_label_height=row_lab,
        titles=titles,
    )


def _sweep_combined_clusters_cumulative_row(
    msi: np.ndarray,
    cfg: DictConfig,
    bench_cfg: DictConfig,
    seed: int,
    comb: Any,
    *,
    ks: list[int],
    cell_title_fmt: str,
    row_title: str,
    ncols: int,
    col_pad: int,
    cell_gap: int,
    row_gap: int,
    row_lab: int,
    cell_max_seq: Any,
) -> np.ndarray:
    """Multi-cluster MiCS: column *i* uses ``clusters`` = first *i* levels of ``ks`` (1 … len(ks))."""
    cells: list[np.ndarray] = []
    titles: list[str] = []
    if len(ks) < 1:
        raise ValueError("clusters_cumulative sweep needs a non-empty cluster level list (cluster_levels or k_list)")
    for i in range(1, len(ks) + 1):
        prefix = ks[:i]
        cj = "-".join(str(x) for x in prefix)
        t = (
            cell_title_fmt.replace("{value}", cj)
            .replace("{param}", "clusters")
            .replace("{row_title}", row_title)
        )
        titles.append(t)
        logger.info("%s: training combined MiCS with clusters=%s", row_title, cj)
        img = _fit_combined_mics_model(
            msi,
            cfg,
            bench_cfg,
            seed,
            comb,
            clusters_joined=cj,
            model_overrides=None,
        )[0]
        img = _maybe_equalize_mics_display_rgb(cfg, img, is_cluster_panel=False)
        img = _maybe_downscale(img, cell_max_seq)
        cells.append(img)
    return _paste_grid(
        cells,
        ncols=min(ncols, len(cells)),
        column_padding=col_pad,
        cell_gap=cell_gap,
        row_gap=row_gap,
        row_label_height=row_lab,
        titles=titles,
    )


def _fit_mics_single_cluster_k(
    msi: np.ndarray,
    cfg: DictConfig,
    bench_cfg: DictConfig,
    seed: int,
    k: int,
) -> Tuple[np.ndarray, MSIParametricMiCSLMC]:
    """One MiCS model with a single cluster level ``k`` (MiCS K-means + cluster head)."""
    per_k = OmegaConf.select(cfg, "mics.per_k", default={})
    per_k_ov = OmegaConf.to_container(getattr(per_k, "mics_model", {}), resolve=True) if per_k else {}
    n_comp = OmegaConf.select(cfg, "mics.per_k.number_of_components")
    if n_comp is None:
        n_comp = OmegaConf.select(cfg, "mics.combined.number_of_components")
    n_comp = int(n_comp) if n_comp is not None else 3
    # per_k_ov first; then force single cluster level k (do not let YAML re-disable cluster).
    extra = {
        **per_k_ov,
        "number_of_components": n_comp,
        "cluster": True,
        "clusters": str(int(k)),
    }
    return _run_mics_train(msi, cfg, bench_cfg, seed, extra)


def _mics_cluster_kmeans_rgb(
    model: MSIParametricMiCSLMC,
    msi: np.ndarray,
    level_idx: int,
    mode: str,
) -> np.ndarray:
    """
    Top-row MiCS “K-means” visualization: ``dense`` = trained cluster head argmax on every
    pixel; ``sparse`` = initialization K-means labels on sampled pixels only (raster).
    """
    m = str(mode).strip().lower()
    if m in ("dense", "full", "predict", "head", "all_pixels", "all"):
        return model.predict_cluster_argmax_rgb_u8(msi, level_idx=level_idx)
    if m in ("sparse", "sampled", "init", "raster"):
        return model.rasterize_mics_kmeans_rgb_u8(level_idx=level_idx)
    raise ValueError(f"kmeans.mics_cluster_viz must be dense|sparse; got {mode!r}")


@hydra.main(version_base=None, config_path="configs", config_name="gallery_kmeans_mics")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(cfg.logging.level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    npy_path = Path(to_absolute_path(str(cfg.input.npy_path)))
    norm = _resolve_normalization_mode(getattr(cfg.data, "normalization", "tic"))
    msi = _load_msi(npy_path, transpose_msi=bool(cfg.data.transpose_msi), normalization=norm)
    h, w = msi.shape[:2]
    hw = (h, w)

    seed = int(getattr(cfg, "seed", 42))
    _seed_everything(seed)

    base_out = Path(cfg.output.out_dir) if cfg.output.out_dir else Path(HydraConfig.get().run.dir)
    use_slide = bool(getattr(cfg.output, "use_slide_name_from_args_txt", True))
    args_txt_override = OmegaConf.select(cfg, "input.args_txt_path")
    slide_raw = _slide_name_from_args_txt(npy_path, args_txt_override) if use_slide else None
    slide_tag = _slug_filename(slide_raw) if slide_raw else ""
    out_root = base_out / slide_tag if slide_tag else base_out
    out_root.mkdir(parents=True, exist_ok=True)
    if slide_raw:
        logger.info("Slide label from args.txt: %r → output subdir %s", slide_raw, slide_tag)

    combined_clusters_joined: str | None = None
    raw_levels = OmegaConf.select(cfg, "mics.cluster_levels")
    if raw_levels is not None:
        cont = OmegaConf.to_container(raw_levels, resolve=True) if OmegaConf.is_config(raw_levels) else raw_levels
        if not isinstance(cont, (list, tuple)) or len(cont) < 1:
            raise ValueError("mics.cluster_levels must be a non-empty list when set")
        ks = [int(x) for x in list(cont)]
        mics_ks = list(ks)
        combined_clusters_joined = "-".join(str(x) for x in ks)
        logger.info(
            "Using mics.cluster_levels=%s for columns and combined MiCS (clusters=%s)",
            ks,
            combined_clusters_joined,
        )
    else:
        ks = [int(x) for x in list(cfg.kmeans.k_list)]
        mics_ks = [int(x) for x in list(cfg.mics.k_list)]
        if len(ks) < 1 or len(mics_ks) < 1:
            raise ValueError("Set mics.cluster_levels or both kmeans.k_list and mics.k_list (non-empty)")

    pairing = str(getattr(cfg.mics, "pairing", "per_k")).strip().lower()
    pairing_per_k = pairing in ("per_k", "paired", "paired_columns", "columns")
    if pairing_per_k and ks != mics_ks:
        raise ValueError(
            f"mics.pairing=per_k requires kmeans.k_list == mics.k_list; got {ks} vs {mics_ks}"
        )

    o = cfg.output
    ncols = int(getattr(o, "gallery_columns", 4))
    cell_max = getattr(o, "cell_max_size", None)
    if OmegaConf.is_config(cell_max):
        cell_max = OmegaConf.to_container(cell_max, resolve=True)
    cell_max_seq = cell_max if isinstance(cell_max, (list, tuple)) else None
    col_pad = int(getattr(o, "column_padding", 8))
    cell_gap = int(getattr(o, "cell_gap", 8))
    row_gap = int(getattr(o, "row_gap", 12))
    row_lab = int(getattr(o, "row_label_height", 34))
    split_gap = int(getattr(o, "split_gallery_gap", 24))

    km_sec = OmegaConf.select(cfg, "external.kmeans")
    km_dir = _resolve_path_str(OmegaConf.select(cfg, "external.kmeans_dir"))
    mics_sec = OmegaConf.select(cfg, "external.mics")
    mics_dir = _resolve_path_str(OmegaConf.select(cfg, "external.mics_dir"))

    bench_cfg = _bench_cfg_for_mics(cfg)
    km_cluster_viz = str(getattr(cfg.kmeans, "mics_cluster_viz", "dense")).strip().lower()

    km_source = str(getattr(cfg.kmeans, "source", "standalone")).strip().lower()
    if pairing_per_k and km_source in ("mics_kmeans", "mics", "from_mics"):
        raise ValueError(
            "kmeans.source=mics_kmeans only works with mics.pairing: legacy (one multi-fit for the top row). "
            "With pairing: per_k, each column uses its own MiCS single-k fit. "
            "Set kmeans.source: standalone or omit mics_kmeans."
        )

    comb_cfg = getattr(cfg.mics, "combined", None)
    combined_mics_model: MSIParametricMiCSLMC | None = None

    if (not pairing_per_k) and km_source in ("mics_kmeans", "mics", "from_mics"):
        if comb_cfg is None or not bool(getattr(comb_cfg, "enabled", True)):
            raise ValueError("kmeans.source=mics_kmeans requires mics.combined.enabled: true")
        if combined_clusters_joined is None:
            _cluster_levels_match_k_list(comb_cfg, ks)
        p_multi_block = _resolve_path_str(OmegaConf.select(cfg, "external.mics_multi_path"))
        if p_multi_block is not None and p_multi_block.is_file():
            raise ValueError(
                "kmeans.source=mics_kmeans is incompatible with external.mics_multi_path "
                "(rasterized MiCS K-means needs one trained combined model). "
                "Remove external.mics_multi_path or use kmeans.source: standalone."
            )
        logger.info(
            "kmeans.source=mics_kmeans: training combined MiCS once; top row = rasterized MiCS K-means from that model",
        )
        _, combined_mics_model = _fit_combined_mics_model(
            msi, cfg, bench_cfg, seed, comb_cfg,
            clusters_joined=combined_clusters_joined,
        )
    elif (not pairing_per_k) and km_source not in ("standalone", "kmeans_segmentation", "sklearn", ""):
        raise ValueError(
            f"Unknown kmeans.source={km_source!r}; use standalone | mics_kmeans",
        )

    # --- Row 1 + 2: paired per k (MiCS single cluster level) or legacy split ---
    kmeans_cells: list[np.ndarray] = []
    kmeans_titles: list[str] = []
    mics_cells: list[np.ndarray] = []
    mics_titles: list[str] = []

    if pairing_per_k:
        pair_fmt = str(getattr(cfg.mics, "title_fmt_pair", None) or getattr(cfg.mics, "title_fmt", "MiCS k={k}"))
        for k in ks:
            kmeans_titles.append(
                str(getattr(cfg.kmeans, "title_fmt_mics", "MiCS K-means k={k}")).replace("{k}", str(k))
            )
            mics_titles.append(pair_fmt.replace("{n}", str(k)).replace("{k}", str(k)))
            p_km = _find_external_in_section(
                km_sec,
                k,
                dir_path=km_dir,
                filename_fmts=("kmeans_k{k}.npy", "kmeans_k{k}.png"),
            )
            p_m = _find_external_in_section(
                mics_sec,
                k,
                dir_path=mics_dir,
                filename_fmts=("mics_n{k}.npy", "mics_n{k}.png", "mics_k{k}.npy", "mics_k{k}.png"),
            )
            if p_km is not None and p_m is not None:
                logger.info("Column k=%s: loading external kmeans + MiCS", k)
                img_km = _load_viz_external(p_km, hw)
                img_m = _load_viz_external(p_m, hw)
            elif p_km is not None:
                logger.info("Column k=%s: external kmeans; training MiCS (cluster k=%s)", k, k)
                rgb, model = _fit_mics_single_cluster_k(msi, cfg, bench_cfg, seed, k)
                img_km = _load_viz_external(p_km, hw)
                img_m = rgb
            elif p_m is not None:
                logger.info("Column k=%s: external MiCS; training MiCS for K-means panel (cluster k=%s)", k, k)
                rgb, model = _fit_mics_single_cluster_k(msi, cfg, bench_cfg, seed, k)
                img_km = _mics_cluster_kmeans_rgb(model, msi, 0, km_cluster_viz)
                img_m = _load_viz_external(p_m, hw)
            else:
                logger.info(
                    "Column k=%s: MiCS train (single cluster level), K-means viz=%s + predict",
                    k,
                    km_cluster_viz,
                )
                rgb, model = _fit_mics_single_cluster_k(msi, cfg, bench_cfg, seed, k)
                img_km = _mics_cluster_kmeans_rgb(model, msi, 0, km_cluster_viz)
                img_m = rgb
            img_km = _maybe_equalize_mics_display_rgb(cfg, img_km, is_cluster_panel=True)
            img_m = _maybe_equalize_mics_display_rgb(cfg, img_m, is_cluster_panel=False)
            kmeans_cells.append(_maybe_downscale(img_km, cell_max_seq))
            mics_cells.append(_maybe_downscale(img_m, cell_max_seq))
    else:
        # --- Row 1: K-means (standalone) or MiCS K-means rasters (mics_kmeans) ---
        for i, k in enumerate(ks):
            if km_source in ("mics_kmeans", "mics", "from_mics"):
                title = str(getattr(cfg.kmeans, "title_fmt_mics", "MiCS K-means k={k}")).replace("{k}", str(k))
            else:
                title = str(getattr(cfg.kmeans, "title_fmt", "K-means k={k}")).replace("{k}", str(k))
            kmeans_titles.append(title)
            p = _find_external_in_section(
                km_sec,
                k,
                dir_path=km_dir,
                filename_fmts=("kmeans_k{k}.npy", "kmeans_k{k}.png"),
            )
            if p is not None:
                logger.info("Top row k=%s: loading external %s", k, p)
                kmeans_cells.append(_load_viz_external(p, hw))
            elif km_source in ("mics_kmeans", "mics", "from_mics"):
                assert combined_mics_model is not None
                logger.info(
                    "Top row k=%s: MiCS K-means panel (viz=%s, level_idx=%s)",
                    k,
                    km_cluster_viz,
                    i,
                )
                kmeans_cells.append(
                    _mics_cluster_kmeans_rgb(combined_mics_model, msi, i, km_cluster_viz)
                )
            else:
                logger.info("K-means k=%s: fitting KmeansSegmentation", k)
                kmeans_cells.append(_kmeans_rgb_u8(msi, k))
            top_is_mics_cluster = km_source in ("mics_kmeans", "mics", "from_mics")
            kmeans_cells[-1] = _maybe_equalize_mics_display_rgb(
                cfg, kmeans_cells[-1], is_cluster_panel=top_is_mics_cluster
            )
            kmeans_cells[-1] = _maybe_downscale(kmeans_cells[-1], cell_max_seq)

        # --- Row 2: MiCS one component count per cell (legacy) ---
        fmt = str(getattr(cfg.mics, "title_fmt", "MiCS n={n}"))
        for comp_k in mics_ks:
            mics_titles.append(fmt.replace("{n}", str(comp_k)).replace("{k}", str(comp_k)))
            p = _find_external_in_section(
                mics_sec,
                comp_k,
                dir_path=mics_dir,
                filename_fmts=("mics_n{k}.npy", "mics_n{k}.png", "mics_k{k}.npy", "mics_k{k}.png"),
            )
            if p is not None:
                logger.info("MiCS n=%s: loading external %s", comp_k, p)
                mics_cells.append(_load_viz_external(p, hw))
            else:
                logger.info("MiCS n=%s: training + predict (cluster off)", comp_k)
                extra = {
                    "number_of_components": int(comp_k),
                    "cluster": False,
                    "clusters": None,
                }
                mics_cells.append(_run_mics(msi, cfg, bench_cfg, seed, extra))
            mics_cells[-1] = _maybe_equalize_mics_display_rgb(cfg, mics_cells[-1], is_cluster_panel=False)
            mics_cells[-1] = _maybe_downscale(mics_cells[-1], cell_max_seq)

    grid_km = _paste_grid(
        kmeans_cells,
        ncols=min(ncols, len(kmeans_cells)),
        column_padding=col_pad,
        cell_gap=cell_gap,
        row_gap=row_gap,
        row_label_height=row_lab,
        titles=kmeans_titles,
    )

    grid_mics = _paste_grid(
        mics_cells,
        ncols=min(ncols, len(mics_cells)),
        column_padding=col_pad,
        cell_gap=cell_gap,
        row_gap=row_gap,
        row_label_height=row_lab,
        titles=mics_titles,
    )

    composite = _vstack_images(grid_km, grid_mics, split_gap)

    comb = getattr(cfg.mics, "combined", None)

    sweep_block = OmegaConf.select(cfg, "mics.sweep_rows")
    if sweep_block is None:
        sweep_block = getattr(cfg.mics, "sweep_rows", None)

    sweep_any = False
    if sweep_block is not None:
        for _pk, _sub in sweep_block.items():
            if bool(getattr(_sub, "enabled", False)):
                sweep_any = True
                break

    # Sweeps do not require ``mics.combined.enabled`` (that flag only mattered for the removed single combined row).
    if sweep_block is not None and sweep_any:
        if comb is None:
            raise ValueError(
                "mics.sweep_rows is enabled but mics.combined is missing; add a ``combined:`` block "
                "(number_of_components, predict_viz, mics_model, …) for multi-cluster sweep fits.",
            )
        logger.info(
            "Appending mics.sweep_rows (combined.enabled=%s does not gate sweeps)",
            getattr(comb, "enabled", None),
        )
        for param_key, sub in sweep_block.items():
            if not bool(getattr(sub, "enabled", False)):
                continue
            pk = str(param_key).strip()
            if pk == "clusters_cumulative":
                row_title = str(getattr(sub, "row_title", "MiCS multi-cluster · clusters (cumulative)"))
                cell_fmt = str(getattr(sub, "cell_title_fmt", "clusters={value}"))
                grid_sw = _sweep_combined_clusters_cumulative_row(
                    msi,
                    cfg,
                    bench_cfg,
                    seed,
                    comb,
                    ks=ks,
                    cell_title_fmt=cell_fmt,
                    row_title=row_title,
                    ncols=ncols,
                    col_pad=col_pad,
                    cell_gap=cell_gap,
                    row_gap=row_gap,
                    row_lab=row_lab,
                    cell_max_seq=cell_max_seq,
                )
                if grid_sw.shape[1] < composite.shape[1]:
                    grid_sw = _pad_to_width_center(grid_sw, composite.shape[1])
                composite = _vstack_images(composite, grid_sw, split_gap)
                continue
            raw_vals = OmegaConf.select(sub, "values")
            if raw_vals is None:
                logger.warning("mics.sweep_rows.%s: missing values; skipped", param_key)
                continue
            cont = OmegaConf.to_container(raw_vals, resolve=True) if OmegaConf.is_config(raw_vals) else raw_vals
            if not isinstance(cont, (list, tuple)) or len(cont) < 1:
                logger.warning("mics.sweep_rows.%s: values must be a non-empty list; skipped", param_key)
                continue
            if pk not in ("num_layers", "beta"):
                raise ValueError(
                    f"mics.sweep_rows key {param_key!r} is not supported; "
                    f"use num_layers, beta, or clusters_cumulative"
                )
            row_title = str(getattr(sub, "row_title", pk))
            cell_fmt = str(getattr(sub, "cell_title_fmt", "{param}={value}"))
            grid_sw = _sweep_combined_row(
                msi,
                cfg,
                bench_cfg,
                seed,
                comb,
                clusters_joined=combined_clusters_joined,
                param_key=pk,
                values=list(cont),
                cell_title_fmt=cell_fmt,
                row_title=row_title,
                ncols=ncols,
                col_pad=col_pad,
                cell_gap=cell_gap,
                row_gap=row_gap,
                row_lab=row_lab,
                cell_max_seq=cell_max_seq,
            )
            if grid_sw.shape[1] < composite.shape[1]:
                grid_sw = _pad_to_width_center(grid_sw, composite.shape[1])
            composite = _vstack_images(composite, grid_sw, split_gap)

    gallery_name = str(getattr(o, "gallery_png", "gallery_kmeans_mics.png"))
    gallery_path = out_root / _slide_prefixed_basename(slide_tag, gallery_name)
    Image.fromarray(composite).save(gallery_path)
    logger.info("Wrote gallery: %s", gallery_path)

    if bool(getattr(o, "save_individual_panels", False)):
        pdir = out_root / str(getattr(o, "panels_dir", "panels"))
        pdir.mkdir(parents=True, exist_ok=True)
        for im, t in zip(kmeans_cells, kmeans_titles):
            Image.fromarray(im).save(pdir / f"panel_{_slug_filename(t)}.png")
        for im, t in zip(mics_cells, mics_titles):
            Image.fromarray(im).save(pdir / f"panel_{_slug_filename(t)}.png")
        logger.info("Wrote panels under %s", pdir)


if __name__ == "__main__":
    main()
