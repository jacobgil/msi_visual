#!/usr/bin/env python3
"""
Generate multiple MSI visualizations (MiCS+LMC, TOP3, NMF3D, PR3D, TOP3-on-NMF-coeffs, PCA3D,
parametric UMAP, nonparametric UMAP) and maintain a ``panel.png`` grid (6 columns).

Each method writes ``<stem>__<tag>.png`` and refreshes ``panel.png`` with all panels so far.
If ``output.save_equalized`` is true, LAB luminance–equalized copies go under ``output.dir/<output.equalized_subdir>/``
(same filenames plus ``panel.png`` there).

``output.panel_full_size`` (default true): panel uses a black background, white titles, and native image
resolution (no ``cell_max_size`` downscale). Set false to use letterboxing and ``figsize_scale`` as before.

Run from repo root:
  python scripts/generate_visualization_panel.py
  python scripts/generate_visualization_panel.py data.npy_path=D:/data/0.npy output.dir=./my_out

Config: ``scripts/configs/generate_visualization_panel.yaml``

Controls **which** methods run and **in what order** via the ``methods`` list (top-to-bottom, then
left-to-right in the matplotlib grid). Omit any name you do not need; repeat entries are allowed
if you want the same method more than once.

MiCS+LMC merge order: ``train_parametric_mics_lmc.yaml`` (via benchmark) →
``mics_model`` (panel defaults) → ``mics_parameter_group`` / ``mics_groups`` overrides →
``mics_model_overrides`` (CLI). Pixel sample count is always ``benchmark.num_samples``.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from msi_visual.nmf_3d import NMF3D
from msi_visual.nonparametric_umap import MSINonParametricUMAP
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC
from msi_visual.parametric_umap import UMAPVirtualStain
from msi_visual.pca_3d import PCA3D
from msi_visual.percentile_ratio import TOP3, PercentileRatio
from scripts.benchmark_parametric_methods import (
    _load_msi,
    _mics_kwargs,
    _resolve_mics_parameter_groups,
    _seed_everything,
    _to_uint8_rgb,
    _umap_kwargs,
)

logger = logging.getLogger(__name__)


def _group_overrides_for_name(bench_cfg: DictConfig, group_name: str | None) -> Dict[str, Any]:
    """Benchmark ``parameter_groups`` entry, or {} if null / none / disabled."""
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
    raise ValueError(f"mics_parameter_group {wanted!r} not found. Available: {available}")


def _letterbox_u8(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(max_side / max(h, w), 1.0)
    if s < 1.0:
        nh, nw = int(round(h * s)), int(round(w * s))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


def save_panel_png(
    items: List[Tuple[str, np.ndarray]],
    out_path: Path,
    *,
    ncols: int,
    cell_max_size: int,
    dpi: int,
    figsize_scale: float,
    panel_full_size: bool = True,
) -> None:
    n = len(items)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))

    def _style_axes(ax) -> None:
        ax.set_facecolor("black")

    if panel_full_size:
        # Native image dimensions: no downscale; figure size ~ sum(pixels)/dpi so cells map ~1:1.
        prepared: List[Tuple[str, np.ndarray, bool]] = []
        for title, im in items:
            im = np.asarray(im)
            if im.ndim == 2:
                prepared.append((title, im, True))
            else:
                prepared.append((title, _to_uint8_rgb(im), False))

        height_ratios: List[int] = []
        for r in range(nrows):
            hs = [
                prepared[r * ncols + c][1].shape[0]
                for c in range(ncols)
                if r * ncols + c < n
            ]
            height_ratios.append(max(hs) if hs else 1)
        width_ratios: List[int] = []
        for c in range(ncols):
            ws = [
                prepared[r * ncols + c][1].shape[1]
                for r in range(nrows)
                if r * ncols + c < n
            ]
            width_ratios.append(max(ws) if ws else 1)
        height_ratios = [max(h, 1) for h in height_ratios]
        width_ratios = [max(w, 1) for w in width_ratios]

        w_in = sum(width_ratios) / float(dpi) + 0.6
        h_in = sum(height_ratios) / float(dpi) + 0.6
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(w_in, h_in),
            facecolor="black",
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
        )
        fig.patch.set_facecolor("black")
        axes_arr = np.atleast_2d(np.asarray(axes))
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                ax = axes_arr[r, c]
                _style_axes(ax)
                if idx < n:
                    title, im, is_gray = prepared[idx]
                    if is_gray:
                        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                    else:
                        ax.imshow(im, interpolation="nearest")
                    ax.set_title(str(title)[:80], fontsize=8, color="white")
                    idx += 1
                ax.axis("off")
        fig.tight_layout()
    else:
        fig_w = max(6.0, ncols * figsize_scale)
        fig_h = max(4.0, nrows * figsize_scale)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), facecolor="black")
        fig.patch.set_facecolor("black")
        axes_arr = np.atleast_2d(np.asarray(axes))
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                ax = axes_arr[r, c]
                _style_axes(ax)
                if idx < n:
                    title, im = items[idx]
                    im = np.asarray(im)
                    if im.ndim == 2:
                        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
                    else:
                        ax.imshow(_letterbox_u8(_to_uint8_rgb(im), cell_max_size))
                    ax.set_title(str(title)[:80], fontsize=8, color="white")
                    idx += 1
                ax.axis("off")
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)
    logger.info("Wrote panel: %s (%d cells)", out_path, n)


def _safe_tag(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("|", "-")
    )


def _equalize_rgb_u8_lab_l(rgb: np.ndarray) -> np.ndarray:
    """Histogram-equalize the L channel in LAB (uint8 RGB in/out)."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.equalizeHist(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def _run_parametric_mics_lmc(msi: np.ndarray, cfg: DictConfig, seed: int) -> List[Tuple[str, np.ndarray]]:
    bench_path = Path(to_absolute_path(str(cfg.mics_benchmark_config)))
    if not bench_path.is_file():
        raise FileNotFoundError(f"mics_benchmark_config not found: {bench_path}")
    bench_cfg = OmegaConf.load(bench_path)
    # Resolve train-yaml path vs Hydra cwd (same issue as benchmark runs from outputs/...).
    if hasattr(bench_cfg.mics_lmc, "defaults_yaml"):
        bench_cfg.mics_lmc.defaults_yaml = to_absolute_path(str(bench_cfg.mics_lmc.defaults_yaml))
    # Let this script's ``benchmark:`` block override the benchmark YAML (e.g. num_samples, sampling_mode).
    bench_cfg = OmegaConf.merge(
        bench_cfg,
        OmegaConf.create({"benchmark": OmegaConf.to_container(cfg.benchmark, resolve=True)}),
    )
    group_key = OmegaConf.select(cfg, "mics_parameter_group")
    group_overrides = _group_overrides_for_name(bench_cfg, group_key)
    raw_panel = OmegaConf.select(cfg, "mics_model", default={})
    panel_model = OmegaConf.to_container(raw_panel, resolve=True) if raw_panel else {}
    raw_cli = OmegaConf.select(cfg, "mics_model_overrides", default={})
    cli_overrides = OmegaConf.to_container(raw_cli, resolve=True) if raw_cli else {}
    # Order: train yaml < panel mics_model defaults < named group < CLI overrides.
    # Named groups must win over panel defaults so ``mics_groups=[...]`` actually applies
    # each preset (otherwise generate_visualization_panel.yaml mics_model clobbers them).
    if group_overrides:
        merged_overrides = {**panel_model, **group_overrides, **cli_overrides}
    else:
        merged_overrides = {**group_overrides, **panel_model, **cli_overrides}
    mk = _mics_kwargs(
        bench_cfg,
        str(cfg.benchmark.sampling_mode),
        seed,
        group_overrides=merged_overrides,
    )
    logger.info(
        "MiCS fit: group=%s clusters=%s beta=%s layers=%s epochs=%s pixel_sampling=%s",
        group_key or "(default)",
        mk.get("clusters"),
        mk.get("beta"),
        mk.get("num_layers"),
        mk.get("num_epochs"),
        mk.get("pixel_sampling"),
    )
    model = MSIParametricMiCSLMC(**mk)
    model.fit(msi)
    viz = model.predict(msi)
    g = str(group_key).strip() if group_key is not None else ""
    if not g or g.lower() in ("none", "null", "~"):
        label = "MiCS+LMC"
    else:
        label = f"MiCS+LMC__{g}"
    return [(label, _to_uint8_rgb(viz))]


def _run_top3(msi: np.ndarray, cfg: DictConfig, _seed: int) -> List[Tuple[str, np.ndarray]]:
    t = cfg.top3
    viz = TOP3(
        low=float(t.low),
        norm_percentile=float(t.norm_percentile),
    )(msi, to_lab=bool(t.to_lab))
    return [("TOP3", _to_uint8_rgb(viz))]


def _run_nmf_3d(msi: np.ndarray, cfg: DictConfig, seed: int) -> List[Tuple[str, np.ndarray]]:
    n = cfg.nmf_3d
    rs_raw = getattr(n, "random_state", 42)
    rs = int(rs_raw) if rs_raw is not None else None
    if rs is not None and rs < 0:
        rs = None
    model = NMF3D(
        n_components=int(n.n_components),
        init=str(n.init),
        max_iter=int(n.max_iter),
        random_state=rs,
    )
    viz = model(msi)
    return [(f"NMF3D_k{int(n.n_components)}", _to_uint8_rgb(viz))]


def _run_pr3d(msi: np.ndarray, cfg: DictConfig, _seed: int) -> List[Tuple[str, np.ndarray]]:
    pct = OmegaConf.to_container(cfg.pr3d.percentiles, resolve=True)
    if not isinstance(pct, (list, tuple)) or len(pct) != 6:
        raise ValueError("pr3d.percentiles must be a list of six numbers")
    pr = PercentileRatio(percentiles=[float(x) for x in pct])
    viz = pr(msi)
    return [("PR3D", _to_uint8_rgb(viz))]


def _nmf_coefficient_cube(msi: np.ndarray, model: NMF3D) -> np.ndarray:
    """Nonnegative NMF coefficients H×W×k (float32), same as sklearn ``transform`` on spectra."""
    model._ensure_legacy_attrs()
    sb, eb = int(model.start_bin), int(model.end_bin)
    n_bins = eb - sb
    flat = np.ascontiguousarray(
        np.clip(msi[:, :, sb:eb], 0.0, None).astype(np.float32)
    ).reshape(-1, n_bins)
    out = model.nmf.transform(flat)
    k = int(model.n_components)
    return out.reshape(msi.shape[0], msi.shape[1], k)


def _run_top3_nmf(msi: np.ndarray, cfg: DictConfig, seed: int) -> List[Tuple[str, np.ndarray]]:
    """Fit NMF on nonnegative spectra, then TOP3 pseudo-RGB on the coefficient cube (gallery style).

    Ranks: ``top3_nmf.k_list`` if set (like ``gallery_top3_nmf_edges`` ``nmf.k_list``), else a single
    ``nmf_3d.n_components``.
    """
    del seed  # NMF randomness from nmf_3d.random_state only
    t = cfg.top3_nmf
    n = cfg.nmf_3d
    raw_kl = OmegaConf.select(cfg, "top3_nmf.k_list")
    if raw_kl is None:
        ks = [int(n.n_components)]
    else:
        kl = OmegaConf.to_container(raw_kl, resolve=True)
        if not isinstance(kl, (list, tuple)) or len(kl) == 0:
            ks = [int(n.n_components)]
        else:
            ks = sorted({int(x) for x in kl})
    for k in ks:
        if k < 3:
            raise ValueError(f"top3_nmf: each k must be >= 3, got {k}")
    rs = int(n.random_state) if int(n.random_state) >= 0 else None
    out: List[Tuple[str, np.ndarray]] = []
    for k in ks:
        model = NMF3D(
            n_components=k,
            init=str(n.init),
            max_iter=int(n.max_iter),
            random_state=rs,
        )
        model.fit([msi])
        coeffs = _nmf_coefficient_cube(msi, model)
        if bool(getattr(t, "use_top3_cfg", True)):
            top = TOP3(
                low=float(cfg.top3.low),
                norm_percentile=float(cfg.top3.norm_percentile),
            )(coeffs, to_lab=bool(cfg.top3.to_lab))
        else:
            top = TOP3()(coeffs, to_lab=bool(cfg.top3.to_lab))
        out.append((f"TOP3_on_NMF_k{k}", _to_uint8_rgb(top)))
    return out


def _run_pca(msi: np.ndarray, cfg: DictConfig, _seed: int) -> List[Tuple[str, np.ndarray]]:
    p = cfg.pca
    model = PCA3D(max_iter=int(p.max_iter))
    model.fit([msi])
    viz = model.predict(msi)
    return [("PCA3D", _to_uint8_rgb(viz))]


def _run_soft_edges(msi: np.ndarray, cfg: DictConfig, _seed: int) -> List[Tuple[str, np.ndarray]]:
    """HD soft landmark-contrast edge map (same pipeline as MiCS edge-stratified training)."""
    from debug_edge_maps import _gray_u8_to_rgb_colormap, _to_uint8_gray01
    from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method

    se = cfg.soft_edges
    rank_path = Path(to_absolute_path(str(se.rank_config)))
    if not rank_path.is_file():
        raise FileNotFoundError(f"soft_edges.rank_config not found: {rank_path}")
    rank_cfg = OmegaConf.load(rank_path)

    valid_mask = np.asarray(msi.sum(axis=-1) > 0, dtype=bool)
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty; cannot compute soft HD edges")

    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    if multi is None or not bool(getattr(multi, "enabled", False)):
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    spatial_enabled = bool(getattr(rank_cfg.spatial_normalization, "enabled", True))

    hd_method = str(getattr(se, "hd_method", "soft_landmark_contrast")).strip().lower()
    hd_edge_n = _compute_hd_edge_n_for_method(
        rank_cfg,
        msi,
        valid_mask,
        getattr(rank_cfg, "hd_edges", None),
        hd_method,
        sigmas,
        aggregation,
        spatial_enabled,
    )
    cmap = str(getattr(se, "colormap", "copper")).strip()
    if cmap.lower() in ("null", "none", "~", ""):
        rgb = np.stack([_to_uint8_gray01(hd_edge_n)] * 3, axis=-1)
    else:
        rgb = _gray_u8_to_rgb_colormap(_to_uint8_gray01(hd_edge_n), cmap)
    label = str(getattr(se, "label", "SoftEdges"))
    return [(label, rgb)]


def _run_parametric_umap(msi: np.ndarray, cfg: DictConfig, seed: int) -> List[Tuple[str, np.ndarray]]:
    mk = _umap_kwargs(cfg, str(cfg.benchmark.sampling_mode), seed)
    model = UMAPVirtualStain(**mk, start_bin=0, end_bin=None)
    model.fit([msi], keras_fit_kwargs={}, roi_mask=None)
    viz = model.predict(msi)
    return [("parametric_UMAP", _to_uint8_rgb(viz))]


def _run_nonparametric_umap(msi: np.ndarray, cfg: DictConfig, _seed: int) -> List[Tuple[str, np.ndarray]]:
    u = cfg.nonparametric_umap
    model = MSINonParametricUMAP(
        n_components=int(getattr(u, "n_components", 3)),
        min_dist=float(u.min_dist),
        n_neighbors=int(u.n_neighbors),
        metric=str(u.metric),
    )
    viz = model(msi)
    return [("nonparametric_UMAP", _to_uint8_rgb(viz))]


METHODS: Dict[str, Callable[..., List[Tuple[str, np.ndarray]]]] = {}
MICS_METHOD_KEYS = frozenset({"parametric_mics_lmc", "mics_lmc"})


def _register() -> None:
    global METHODS
    METHODS = {
        "parametric_mics_lmc": _run_parametric_mics_lmc,
        "mics_lmc": _run_parametric_mics_lmc,
        "top3": _run_top3,
        "nmf_3d": _run_nmf_3d,
        "pr3d": _run_pr3d,
        "top3_nmf": _run_top3_nmf,
        "pca": _run_pca,
        "soft_edges": _run_soft_edges,
        "hd_soft_edges": _run_soft_edges,
        "soft_landmark_contrast": _run_soft_edges,
        "parametric_umap": _run_parametric_umap,
        "nonparametric_umap": _run_nonparametric_umap,
    }


def _normalize_mics_group(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in ("", "null", "none", "~"):
        return None
    return s


def resolve_method_specs(cfg: DictConfig) -> List[Tuple[str, str | None]]:
    """Build ordered (method_key, mics_parameter_group) specs for one MSI slice."""
    _register()
    specs: List[Tuple[str, str | None]] = []
    methods_list = OmegaConf.to_container(cfg.methods, resolve=True)
    if not isinstance(methods_list, list):
        raise ValueError("config `methods` must be a list of method names")

    for raw_name in methods_list:
        key = str(raw_name).strip().lower()
        if key in MICS_METHOD_KEYS:
            if key not in METHODS:
                raise ValueError(f"Unknown method {raw_name!r}")
            specs.append(
                ("parametric_mics_lmc", _normalize_mics_group(OmegaConf.select(cfg, "mics_parameter_group")))
            )
        elif key not in METHODS:
            raise ValueError(
                f"Unknown method {raw_name!r}. Choose from: {sorted(METHODS.keys())}"
            )
        else:
            specs.append((key, None))

    raw_groups = OmegaConf.select(cfg, "mics_groups")
    if raw_groups is not None:
        groups = OmegaConf.to_container(raw_groups, resolve=True)
        if groups is not None:
            if not isinstance(groups, list):
                raise ValueError("config `mics_groups` must be a list")
            for group in groups:
                specs.append(("parametric_mics_lmc", _normalize_mics_group(group)))

    if not specs:
        raise ValueError("No visualization methods configured (methods / mics_groups empty)")
    return specs


def run_visualization_specs(
    msi: np.ndarray,
    cfg: DictConfig,
    specs: List[Tuple[str, str | None]],
    seed: int,
) -> List[Tuple[str, np.ndarray]]:
    """Run visualization methods and return (title, rgb_u8) pairs."""
    _register()
    accumulated: List[Tuple[str, np.ndarray]] = []
    for key, mics_group in specs:
        run_cfg = cfg
        if key == "parametric_mics_lmc":
            run_cfg = OmegaConf.merge(cfg, {"mics_parameter_group": mics_group})
        fn = METHODS[key]
        logger.info("--- Computing: %s (mics_group=%s) ---", key, mics_group)
        for title, arr in fn(msi, run_cfg, seed):
            accumulated.append((title, _to_uint8_rgb(arr)))
    return accumulated


def visualization_panel_path(out_dir: Path, cfg: DictConfig) -> Path:
    panel_name = str(OmegaConf.select(cfg, "output.panel_png", default="panel.png"))
    return out_dir / panel_name


def _expected_viz_tags(cfg: DictConfig, specs: List[Tuple[str, str | None]]) -> List[str]:
    tags: List[str] = []
    for key, mics_group in specs:
        if key == "parametric_mics_lmc":
            label = f"MiCS+LMC__{mics_group}" if mics_group else "MiCS+LMC"
            tags.append(_safe_tag(label))
        elif key == "top3":
            tags.append(_safe_tag("TOP3"))
        elif key == "pca":
            tags.append(_safe_tag("PCA3D"))
        elif key in ("soft_edges", "hd_soft_edges", "soft_landmark_contrast"):
            label = str(OmegaConf.select(cfg, "soft_edges.label", default="SoftEdges"))
            tags.append(_safe_tag(label))
        elif key == "pr3d":
            tags.append(_safe_tag("PR3D"))
        elif key == "parametric_umap":
            tags.append(_safe_tag("parametric_UMAP"))
        elif key == "nonparametric_umap":
            tags.append(_safe_tag("nonparametric_UMAP"))
        elif key == "nmf_3d":
            k = int(OmegaConf.select(cfg, "nmf_3d.n_components", default=3))
            tags.append(_safe_tag(f"NMF3D_k{k}"))
        elif key == "top3_nmf":
            n = cfg.nmf_3d
            raw_kl = OmegaConf.select(cfg, "top3_nmf.k_list")
            if raw_kl is None:
                ks = [int(n.n_components)]
            else:
                kl = OmegaConf.to_container(raw_kl, resolve=True)
                ks = [int(n.n_components)] if not isinstance(kl, (list, tuple)) or not kl else sorted({int(x) for x in kl})
            for k in ks:
                tags.append(_safe_tag(f"TOP3_on_NMF_k{k}"))
    return tags


def visualizations_complete(out_dir: Path, stem: str, cfg: DictConfig) -> bool:
    """True when all expected per-method PNGs exist (and panel.png if save_panel)."""
    if not out_dir.is_dir():
        return False
    try:
        specs = resolve_method_specs(cfg)
    except ValueError:
        return False
    tags = _expected_viz_tags(cfg, specs)
    if not tags:
        return False
    if not all((out_dir / f"{stem}__{tag}.png").is_file() for tag in tags):
        return False
    save_panel = bool(OmegaConf.select(cfg, "output.save_panel", default=True))
    if save_panel and not visualization_panel_path(out_dir, cfg).is_file():
        return False
    return True


def save_visualization_outputs(
    pairs: List[Tuple[str, np.ndarray]],
    out_dir: Path,
    stem: str,
    cfg: DictConfig,
    *,
    hydra_mirror_dir: Path | None = None,
) -> Path | None:
    """Write per-method PNGs and optional panel / equalized copies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_full_size = bool(OmegaConf.select(cfg, "output.panel_full_size", default=True))
    save_eq = bool(OmegaConf.select(cfg, "output.save_equalized", default=False))
    save_panel = bool(OmegaConf.select(cfg, "output.save_panel", default=True))
    eq_sub = str(OmegaConf.select(cfg, "output.equalized_subdir", default="eq")).strip() or "eq"
    eq_dir = out_dir / eq_sub if save_eq else None
    accumulated_eq: List[Tuple[str, np.ndarray]] = []
    if eq_dir is not None:
        eq_dir.mkdir(parents=True, exist_ok=True)

    for title, u8 in pairs:
        tag = _safe_tag(title)
        png_path = out_dir / f"{stem}__{tag}.png"
        cv2.imwrite(str(png_path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
        logger.info("Saved %s", png_path)
        if eq_dir is not None:
            eq_u8 = _equalize_rgb_u8_lab_l(u8)
            eq_png = eq_dir / f"{stem}__{tag}.png"
            cv2.imwrite(str(eq_png), cv2.cvtColor(eq_u8, cv2.COLOR_RGB2BGR))
            accumulated_eq.append((title, eq_u8))

    panel_path: Path | None = None
    if save_panel and pairs:
        panel_path = visualization_panel_path(out_dir, cfg)
        save_panel_png(
            pairs,
            panel_path,
            ncols=int(cfg.output.columns),
            cell_max_size=int(cfg.output.cell_max_size),
            dpi=int(cfg.output.dpi),
            figsize_scale=float(cfg.output.figsize_scale),
            panel_full_size=panel_full_size,
        )
        if hydra_mirror_dir is not None and hydra_mirror_dir.resolve() != out_dir.resolve():
            hydra_mirror_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(panel_path, hydra_mirror_dir / panel_path.name)

    if eq_dir is not None and accumulated_eq and save_panel:
        eq_panel = eq_dir / visualization_panel_path(out_dir, cfg).name
        save_panel_png(
            accumulated_eq,
            eq_panel,
            ncols=int(cfg.output.columns),
            cell_max_size=int(cfg.output.cell_max_size),
            dpi=int(cfg.output.dpi),
            figsize_scale=float(cfg.output.figsize_scale),
            panel_full_size=panel_full_size,
        )

    return panel_path


@hydra.main(version_base=None, config_path="configs", config_name="generate_visualization_panel")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    seed = int(cfg.seed)
    _seed_everything(seed)

    npy = Path(to_absolute_path(str(cfg.data.npy_path)))
    if not npy.is_file():
        raise FileNotFoundError(f"data.npy_path not found: {npy}")

    out_dir = Path(to_absolute_path(str(cfg.output.dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(cfg.output.stem)

    msi = _load_msi(
        npy,
        transpose_msi=bool(cfg.data.transpose_msi),
        tic_normalize=bool(cfg.data.tic_normalize),
    )
    logger.info("Loaded MSI shape=%s from %s", tuple(msi.shape), npy)

    specs = resolve_method_specs(cfg)
    hydra_run = Path(HydraConfig.get().runtime.output_dir)
    pairs = run_visualization_specs(msi, cfg, specs, seed)
    panel_path = save_visualization_outputs(pairs, out_dir, stem, cfg, hydra_mirror_dir=hydra_run)
    if panel_path is not None:
        logger.info("Done. Panel: %s", panel_path)
    else:
        logger.info("Done. Wrote %d visualization PNG(s) to %s", len(pairs), out_dir)


if __name__ == "__main__":
    main()
