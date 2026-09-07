#!/usr/bin/env python3
"""
Publication heatmaps: sparse MiCS explanation weights (α) vs NMF abundances.

Computes several abundance-aware (and baseline) disagreement maps and saves
clean, paper-ready heatmaps with clear sequential color schemes.

Requires an existing ``explain_mics_concepts`` run directory containing
``alpha_hwk.npy`` and ``concepts.npy``.

Run from repo root:
  python scripts/paper_alpha_nmf_heatmaps.py
  python scripts/paper_alpha_nmf_heatmaps.py \\
    data.interpretability_dir=outputs/interpretability/CarstenHopf_ID9_neg

Config: ``scripts/configs/paper_alpha_nmf_heatmaps.yaml``
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig, OmegaConf

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from msi_visual.mics_concept_explain import FrozenConceptExplainer

logger = logging.getLogger(__name__)

# Short paper labels (figure captions can expand).
METRIC_LABELS: dict[str, str] = {
    "weighted_l1_nmf": r"Abundance-weighted $L_1$ (by NMF)",
    "weighted_l1_joint": r"Abundance-weighted $L_1$ (joint)",
    "jensen_shannon": "Jensen–Shannon divergence",
    "sparse_topk_cosine": r"Sparse top-$k$ cosine distance",
    "cosine": "Cosine distance",
    "topk_set_distance": r"Top-$k$ set distance",
    "nmf_entropy": "NMF abundance entropy",
    "nmf_eff_concepts": r"Effective # NMF concepts ($e^{H}$)",
    "alpha_entropy": r"Explanation-weight entropy ($\alpha$)",
}

METRIC_COLORBAR: dict[str, str] = {
    "weighted_l1_nmf": r"Weighted $L_1$",
    "weighted_l1_joint": r"Weighted $L_1$",
    "jensen_shannon": "JS divergence",
    "sparse_topk_cosine": r"$1-\cos$",
    "cosine": r"$1-\cos$",
    "topk_set_distance": r"$1-$ Jaccard",
    "nmf_entropy": r"$H(s)$ (nats)",
    "nmf_eff_concepts": r"$N_{\mathrm{eff}}=e^{H(s)}$",
    "alpha_entropy": r"$H(\alpha)$ (nats)",
}


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected H×W×C MSI, got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if tic_normalize:
        sums = img.sum(axis=-1, keepdims=True) + 1e-8
        img = img / sums
    return img


def _row_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    s = a.sum(axis=1, keepdims=True)
    return a / np.maximum(s, eps)


def _keep_topk_rows(values: np.ndarray, top_k: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    n, k = arr.shape
    keep = max(1, min(int(top_k), k))
    out = np.zeros_like(arr, dtype=np.float32)
    idx = np.argpartition(-arr, keep - 1, axis=1)[:, :keep]
    rows = np.arange(n)[:, None]
    out[rows, idx] = arr[rows, idx]
    return out


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    out = np.ones(aa.shape[0], dtype=np.float64)
    good = denom > 1e-12
    cos = np.sum(aa[good] * bb[good], axis=1) / denom[good]
    out[good] = 1.0 - np.clip(cos, -1.0, 1.0)
    return out.astype(np.float32)


def _weighted_l1(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Row-wise Σ w_i |a_i - b_i| with w already row-normalized."""
    aa = _row_normalize(a)
    bb = _row_normalize(b)
    ww = _row_normalize(np.asarray(weights, dtype=np.float64))
    return np.sum(ww * np.abs(aa - bb), axis=1).astype(np.float32)


def _jensen_shannon(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(_row_normalize(a), eps, 1.0)
    q = np.clip(_row_normalize(b), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    # JS in nats; normalize by log(2) → [0, 1]
    js = 0.5 * (kl_pm + kl_qm) / np.log(2.0)
    return np.clip(js, 0.0, 1.0).astype(np.float32)


def _topk_set_distance(a: np.ndarray, b: np.ndarray, top_k: int) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    n, k = aa.shape
    keep = max(1, min(int(top_k), k))
    ia = np.argpartition(-aa, keep - 1, axis=1)[:, :keep]
    ib = np.argpartition(-bb, keep - 1, axis=1)[:, :keep]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        sa = set(ia[i].tolist())
        sb = set(ib[i].tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        jacc = inter / max(union, 1)
        out[i] = 1.0 - float(jacc)
    return out


def _shannon_entropy(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise Shannon entropy (nats) of a non-negative mixture after simplex normalize."""
    p = np.clip(_row_normalize(x), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)


def _compute_metric_maps(
    alpha_flat: np.ndarray,
    scores_flat: np.ndarray,
    *,
    top_k: int,
) -> dict[str, np.ndarray]:
    alpha_s = _keep_topk_rows(alpha_flat, top_k)
    scores_s = _keep_topk_rows(scores_flat, top_k)
    nmf_H = _shannon_entropy(scores_flat)
    alpha_H = _shannon_entropy(alpha_flat)
    return {
        "weighted_l1_nmf": _weighted_l1(alpha_flat, scores_flat, scores_flat),
        "weighted_l1_joint": _weighted_l1(
            alpha_flat, scores_flat, 0.5 * (alpha_flat + scores_flat)
        ),
        "jensen_shannon": _jensen_shannon(alpha_flat, scores_flat),
        "sparse_topk_cosine": _cosine_distance(alpha_s, scores_s),
        "cosine": _cosine_distance(alpha_flat, scores_flat),
        "topk_set_distance": _topk_set_distance(alpha_flat, scores_flat, top_k),
        # Mixture complexity (high = many co-abundant concepts → hard for 3D viz).
        "nmf_entropy": nmf_H,
        "nmf_eff_concepts": np.exp(nmf_H).astype(np.float32),
        "alpha_entropy": alpha_H,
    }


def _colormap(name: str):
    """Paper-oriented sequential schemes (higher = more disagreement)."""
    key = str(name).strip().lower()
    presets = {
        # Pale cream → deep teal (print-friendly, non-rainbow)
        "paper_teal": [
            "#F7FBF9",
            "#D5EDE6",
            "#9FCFC0",
            "#5FA08C",
            "#2F6F5E",
            "#0B3D34",
        ],
        # Pale → deep indigo
        "paper_indigo": [
            "#F7F7FB",
            "#D9D8F0",
            "#A9A6D8",
            "#6B66B3",
            "#3C3880",
            "#1A1740",
        ],
        # Soft gray → muted red (classic "error" without screaming)
        "paper_grayred": [
            "#F5F5F5",
            "#E6D5D5",
            "#D09A9A",
            "#B85C5C",
            "#8B2E2E",
            "#4A1212",
        ],
    }
    if key in presets:
        return LinearSegmentedColormap.from_list(key, presets[key], N=256)
    if key in ("paper_magma", "magma"):
        return plt.get_cmap("magma")
    if key in ("paper_cividis", "cividis"):
        return plt.get_cmap("cividis")
    if key in ("inferno", "viridis", "plasma", "Greys"):
        return plt.get_cmap(key)
    raise ValueError(f"Unknown colormap {name!r}")


def _off_tissue_rgba(mode: str) -> tuple[float, float, float, float]:
    m = str(mode).strip().lower()
    if m in ("black", "k"):
        return (0.0, 0.0, 0.0, 1.0)
    if m in ("lightgray", "grey", "gray"):
        return (0.92, 0.92, 0.92, 1.0)
    return (1.0, 1.0, 1.0, 1.0)


def _tissue_limits(
    values: np.ndarray, mask: np.ndarray, plo: float, phi: float
) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, float(plo)))
    hi = float(np.percentile(v, float(phi)))
    if hi <= lo + 1e-12:
        hi = lo + 1e-3
    return lo, hi


def _save_heatmap(
    values: np.ndarray,
    mask: np.ndarray,
    out_stem: Path,
    *,
    cmap_name: str,
    vmin: float,
    vmax: float,
    title: str,
    colorbar_label: str,
    dpi: int,
    figsize: tuple[float, float],
    show_title: bool,
    show_colorbar: bool,
    show_colorbar_label: bool,
    off_tissue: str,
    save_pdf: bool,
    save_png: bool,
) -> dict[str, str]:
    arr = np.asarray(values, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    cmap = _colormap(cmap_name)
    cmap = cmap.copy()
    # Masked pixels use facecolor, not the colormap's under value.
    shown = np.ma.array(arr, mask=~m)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_off_tissue_rgba(off_tissue)[:3])
    im = ax.imshow(
        shown,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        interpolation="nearest",
        origin="upper",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    if show_title and title:
        ax.set_title(title, fontsize=11, pad=8, color="#1a1a1a")
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors="#333333")
        if show_colorbar_label and colorbar_label:
            cbar.set_label(colorbar_label, fontsize=9, color="#333333")
        cbar.outline.set_edgecolor("#888888")
        cbar.outline.set_linewidth(0.6)

    fig.tight_layout(pad=0.15)
    paths: dict[str, str] = {}
    if save_png:
        png = out_stem.with_suffix(".png")
        fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        paths["png"] = str(png)
    if save_pdf:
        pdf = out_stem.with_suffix(".pdf")
        fig.savefig(pdf, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        paths["pdf"] = str(pdf)
    plt.close(fig)
    return paths


def _save_panel(
    items: list[tuple[str, np.ndarray, float, float]],
    mask: np.ndarray,
    out_stem: Path,
    *,
    cmap_name: str,
    dpi: int,
    show_title: bool,
    off_tissue: str,
    save_pdf: bool,
    save_png: bool,
) -> dict[str, str]:
    """Panel with per-metric p2–p98 stretch mapped to a shared [0, 1] display scale."""
    n = len(items)
    fig_w = 3.2 * n + 0.9
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fig_w, 3.5),
        dpi=dpi,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    cmap = _colormap(cmap_name)
    m = np.asarray(mask, dtype=bool)
    last_im = None
    for ax, (title, vals, vmin, vmax) in zip(axes, items):
        ax.set_facecolor(_off_tissue_rgba(off_tissue)[:3])
        raw = np.asarray(vals, dtype=np.float32)
        # Independent stretch → shared [0,1] so one colorbar is honest.
        norm = np.zeros_like(raw, dtype=np.float32)
        span = max(float(vmax) - float(vmin), 1e-12)
        norm[m] = np.clip((raw[m] - float(vmin)) / span, 0.0, 1.0)
        shown = np.ma.array(norm, mask=~m)
        last_im = ax.imshow(
            shown,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            origin="upper",
        )
        ax.set_aspect("equal")
        ax.axis("off")
        if show_title:
            ax.set_title(title, fontsize=8.5, pad=5, color="#1a1a1a")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=list(axes), fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Relative disagreement (per-metric p2–p98)", fontsize=8)
    paths: dict[str, str] = {}
    if save_png:
        png = out_stem.with_suffix(".png")
        fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05)
        paths["png"] = str(png)
    if save_pdf:
        pdf = out_stem.with_suffix(".pdf")
        fig.savefig(pdf, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05)
        paths["pdf"] = str(pdf)
    plt.close(fig)
    return paths


def _resolve_npy_path(cfg: DictConfig, interp_dir: Path) -> Path:
    raw = OmegaConf.select(cfg, "data.npy_path", default=None)
    if raw is not None and str(raw).strip().lower() not in ("", "null", "none", "~"):
        return Path(to_absolute_path(str(raw)))
    manifest = interp_dir / "explain_mics_concepts_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"npy_path not set and no manifest at {manifest}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    npy = payload.get("npy_path")
    if not npy:
        raise ValueError(f"manifest missing npy_path: {manifest}")
    return Path(npy)


@hydra.main(version_base=None, config_path="configs", config_name="paper_alpha_nmf_heatmaps")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    interp_dir = Path(to_absolute_path(str(cfg.data.interpretability_dir)))
    alpha_path = interp_dir / "alpha_hwk.npy"
    concepts_path = interp_dir / "concepts.npy"
    if not alpha_path.is_file() or not concepts_path.is_file():
        raise FileNotFoundError(
            f"Need alpha_hwk.npy and concepts.npy in {interp_dir}"
        )

    npy_path = _resolve_npy_path(cfg, interp_dir)
    if not npy_path.is_file():
        raise FileNotFoundError(f"MSI npy not found: {npy_path}")

    alpha = np.asarray(np.load(alpha_path), dtype=np.float32)
    concepts = np.asarray(np.load(concepts_path), dtype=np.float32)
    msi = _load_msi(
        npy_path,
        bool(OmegaConf.select(cfg, "data.transpose_msi", default=False)),
        tic_normalize=str(OmegaConf.select(cfg, "data.normalization", default="tic")).lower()
        == "tic",
    )
    if msi.shape[:2] != alpha.shape[:2]:
        raise ValueError(
            f"MSI spatial shape {msi.shape[:2]} != alpha shape {alpha.shape[:2]}"
        )
    if concepts.shape[0] != alpha.shape[2]:
        raise ValueError(
            f"concepts K={concepts.shape[0]} != alpha K={alpha.shape[2]}"
        )

    mask = np.asarray(msi.sum(axis=-1) > 0, dtype=bool)
    h, w, k = alpha.shape
    alpha_flat = alpha.reshape(-1, k)[mask.ravel()]
    logger.info(
        "Computing NMF concept scores for %d tissue pixels (K=%d) ...",
        int(mask.sum()),
        k,
    )
    scores_flat = FrozenConceptExplainer.concept_scores(msi[mask], concepts)

    top_k = int(OmegaConf.select(cfg, "metrics.sparse_top_k", default=4))
    metric_maps_flat = _compute_metric_maps(alpha_flat, scores_flat, top_k=top_k)

    names = OmegaConf.to_container(cfg.metrics.names, resolve=True)
    if not isinstance(names, list) or not names:
        names = list(metric_maps_flat.keys())
    names = [str(n) for n in names]

    plo = float(OmegaConf.select(cfg, "metrics.percentile_low", default=2.0))
    phi = float(OmegaConf.select(cfg, "metrics.percentile_high", default=98.0))
    primary_cmap = str(OmegaConf.select(cfg, "style.colormap", default="paper_teal"))
    also = OmegaConf.select(cfg, "style.also_export_schemes", default=[]) or []
    if OmegaConf.is_config(also):
        also = OmegaConf.to_container(also, resolve=True)
    also_list = [str(x) for x in (also or [])]
    schemes = [primary_cmap] + [c for c in also_list if c != primary_cmap]

    st = cfg.style
    dpi = int(OmegaConf.select(st, "dpi", default=300))
    panel_dpi = int(OmegaConf.select(st, "panel_dpi", default=300))
    figsize = OmegaConf.to_container(st.figsize_single, resolve=True)
    figsize_t = (float(figsize[0]), float(figsize[1]))
    show_title = bool(OmegaConf.select(st, "show_title", default=False))
    show_cbar = bool(OmegaConf.select(st, "colorbar", default=True))
    show_cbar_label = bool(OmegaConf.select(st, "show_colorbar_label", default=True))
    off_tissue = str(OmegaConf.select(st, "off_tissue", default="white"))
    save_pdf = bool(OmegaConf.select(st, "save_pdf", default=True))
    save_png = bool(OmegaConf.select(st, "save_png", default=True))
    save_npy = bool(OmegaConf.select(st, "save_npy", default=True))

    maps_hw: dict[str, np.ndarray] = {}
    limits: dict[str, dict[str, float]] = {}
    summary: dict[str, Any] = {
        "interpretability_dir": str(interp_dir),
        "npy_path": str(npy_path),
        "sparse_top_k": top_k,
        "percentile_low": plo,
        "percentile_high": phi,
        "metrics": {},
        "outputs": {},
    }

    for name in names:
        if name not in metric_maps_flat:
            raise ValueError(f"Unknown metric {name!r}; choose from {sorted(metric_maps_flat)}")
        hw = np.zeros((h, w), dtype=np.float32)
        hw[mask] = metric_maps_flat[name]
        maps_hw[name] = hw
        lo, hi = _tissue_limits(hw, mask, plo, phi)
        limits[name] = {"vmin": lo, "vmax": hi}
        tissue = hw[mask]
        summary["metrics"][name] = {
            "mean": float(np.mean(tissue)),
            "median": float(np.median(tissue)),
            "p02": float(np.percentile(tissue, 2)),
            "p98": float(np.percentile(tissue, 98)),
            "stretch_vmin": lo,
            "stretch_vmax": hi,
        }
        if save_npy:
            np.save(run_dir / f"{name}.npy", hw)

    for cmap_name in schemes:
        scheme_dir = run_dir / "schemes" / cmap_name
        scheme_dir.mkdir(parents=True, exist_ok=True)
        panel_items: list[tuple[str, np.ndarray, float, float]] = []
        for name in names:
            stem = scheme_dir / name
            paths = _save_heatmap(
                maps_hw[name],
                mask,
                stem,
                cmap_name=cmap_name,
                vmin=limits[name]["vmin"],
                vmax=limits[name]["vmax"],
                title=METRIC_LABELS.get(name, name),
                colorbar_label=METRIC_COLORBAR.get(name, "Value"),
                dpi=dpi,
                figsize=figsize_t,
                show_title=show_title,
                show_colorbar=show_cbar,
                show_colorbar_label=show_cbar_label,
                off_tissue=off_tissue,
                save_pdf=save_pdf,
                save_png=save_png,
            )
            summary["outputs"].setdefault(cmap_name, {})[name] = paths
            panel_items.append(
                (
                    METRIC_LABELS.get(name, name),
                    maps_hw[name],
                    limits[name]["vmin"],
                    limits[name]["vmax"],
                )
            )
            logger.info("Wrote %s / %s", cmap_name, name)

        if bool(OmegaConf.select(st, "panel", default=True)) and panel_items:
            panel_paths = _save_panel(
                panel_items,
                mask,
                scheme_dir / "panel",
                cmap_name=cmap_name,
                dpi=panel_dpi,
                show_title=True,
                off_tissue=off_tissue,
                save_pdf=save_pdf,
                save_png=save_png,
            )
            summary["outputs"].setdefault(cmap_name, {})["panel"] = panel_paths

        # Compact paper panel: disagreement + mixture complexity.
        paper_names = OmegaConf.select(cfg, "metrics.paper_panel", default=None)
        if paper_names is not None:
            paper_list = [str(n) for n in OmegaConf.to_container(paper_names, resolve=True)]
            paper_items: list[tuple[str, np.ndarray, float, float]] = []
            for name in paper_list:
                if name not in maps_hw:
                    raise ValueError(
                        f"paper_panel metric {name!r} not in metrics.names "
                        f"(available: {sorted(maps_hw)})"
                    )
                paper_items.append(
                    (
                        METRIC_LABELS.get(name, name),
                        maps_hw[name],
                        limits[name]["vmin"],
                        limits[name]["vmax"],
                    )
                )
            if paper_items:
                paper_paths = _save_panel(
                    paper_items,
                    mask,
                    scheme_dir / "panel_paper",
                    cmap_name=cmap_name,
                    dpi=panel_dpi,
                    show_title=True,
                    off_tissue=off_tissue,
                    save_pdf=save_pdf,
                    save_png=save_png,
                )
                summary["outputs"].setdefault(cmap_name, {})["panel_paper"] = paper_paths

    # Convenience copies of the primary scheme at the run-dir root.
    primary_dir = run_dir / "schemes" / primary_cmap
    if primary_dir.is_dir():
        for p in primary_dir.glob("*"):
            if p.is_file():
                target = run_dir / p.name
                target.write_bytes(p.read_bytes())

    (run_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done. Paper heatmaps in %s (primary scheme: %s)", run_dir, primary_cmap)


if __name__ == "__main__":
    main()
