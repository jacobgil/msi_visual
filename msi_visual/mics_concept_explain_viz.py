"""Publication-style figures for MiCS concept explanations."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

from msi_visual.mics_concept_explain import build_color_explanation
from msi_visual.mics_concept_viz import _activation_rgb, _stitch_grid

logger = logging.getLogger(__name__)

_MAP_ARROW_COLOR = "#e6edf3"
_MAP_ARROW_LW = 1.6
# Concept panel slots: TL, top-center, center-left, top-right (D), center-right (E).
_PANEL_GS = ((0, 0), (0, 1), (1, 0), (0, 2), (1, 2))


# 2x2 thumb layout — large cells filling the panel (square, flush).
_PANEL_GRID_XY = ((0.5, 1.5), (1.5, 1.5), (0.5, 0.5), (1.5, 0.5))
_PANEL_CELL_HALF = 0.495


def _panel_thumb_cells() -> list[tuple[float, float, float, float]]:
    """Return (cx, cy, half_w, half_h) per cell — top row first, left to right."""
    h = _PANEL_CELL_HALF
    return [(cx, cy, h, h) for cx, cy in _PANEL_GRID_XY]


def _draw_thumb_grid_border(
    ax,
    cells: list[tuple[float, float, float, float]],
    *,
    linewidth: float = 1.2,
) -> None:
    """Single outer frame + cross dividers (flush 2x2, no gutter)."""
    xs = [c[0] - c[2] for c in cells] + [c[0] + c[2] for c in cells]
    ys = [c[1] - c[3] for c in cells] + [c[1] + c[3] for c in cells]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="white",
            linewidth=linewidth,
            zorder=3,
        )
    )
    mid_x = 0.5 * (x0 + x1)
    mid_y = 0.5 * (y0 + y1)
    ax.plot([mid_x, mid_x], [y0, y1], color="white", linewidth=linewidth, zorder=3)
    ax.plot([x0, x1], [mid_y, mid_y], color="white", linewidth=linewidth, zorder=3)



def _equalize_rgb_display(rgb: np.ndarray, fg_mask: np.ndarray | None = None) -> np.ndarray:
    """Tissue-only RGB->LAB L-channel histogram equalization; keeps background black."""
    out = np.asarray(rgb, dtype=np.uint8).copy()
    if fg_mask is None:
        fg = np.any(out > 0, axis=-1)
    else:
        fg = np.asarray(fg_mask, dtype=bool)
    if not np.any(fg):
        return out
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    vals = l_ch[fg]
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    nz = cdf > 0
    if np.any(nz):
        cdf_min = float(cdf[nz][0])
        denom = float(cdf[-1] - cdf_min)
        if denom > 1e-8:
            lut = np.clip((cdf - cdf_min) / denom * 255.0, 0.0, 255.0).astype(np.uint8)
            l_ch[fg] = lut[l_ch[fg]]
            lab[:, :, 0] = l_ch
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    out[~fg] = 0
    return out


def _equalize_map_display(mics_rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Per-channel histogram equalization on tissue (display-only, for map figure)."""
    return _equalize_rgb_display(mics_rgb, np.asarray(valid_mask, dtype=bool))


def _fill_pixel_terms(
    alpha_yx: np.ndarray,
    palette: np.ndarray,
    expl: dict[str, Any],
    *,
    map_top_n: int,
    renormalize_terms: bool,
) -> list[dict[str, Any]]:
    terms = list(expl.get("terms", []))
    if len(terms) < map_top_n:
        used = {int(t["concept"]) for t in terms}
        order = np.argsort(-np.asarray(alpha_yx, dtype=np.float64))
        for ki in order:
            ki = int(ki)
            if ki in used:
                continue
            terms.append(
                {
                    "concept": ki,
                    "weight": float(alpha_yx[ki]),
                    "weight_norm": float(alpha_yx[ki]),
                    "color_rgb": palette[ki].astype(np.uint8).tolist(),
                    "color_name": "",
                }
            )
            used.add(ki)
            if len(terms) >= map_top_n:
                break
    terms = terms[:map_top_n]
    if bool(renormalize_terms):
        total = sum(float(t.get("weight", 0.0)) for t in terms) or 1.0
        for t in terms:
            t["weight_norm"] = float(t.get("weight", 0.0)) / total
    else:
        for t in terms:
            t["weight_norm"] = float(t.get("weight", 0.0))
    return terms


def _thumb_extent_in_cell(
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
    img_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    """Fit an HxW image inside a cell while preserving aspect ratio."""
    img_h = max(1, int(img_shape[0]))
    img_w = max(1, int(img_shape[1]))
    cell_w = 2.0 * half_w
    cell_h = 2.0 * half_h
    img_ar = img_w / img_h
    cell_ar = cell_w / cell_h
    if img_ar >= cell_ar:
        draw_w = cell_w
        draw_h = cell_w / img_ar
    else:
        draw_h = cell_h
        draw_w = cell_h * img_ar
    half_draw_w = 0.5 * draw_w
    half_draw_h = 0.5 * draw_h
    return (
        cx - half_draw_w,
        cx + half_draw_w,
        cy - half_draw_h,
        cy + half_draw_h,
    )


def _draw_map_pixel_marker(
    ax,
    x: float,
    y: float,
    label: str,
    *,
    marker_size: float = 14.0,
    marker_fontsize: float = 18.0,
) -> None:
    """Circle marker on the tissue map with its letter label tucked below."""
    ax.plot(
        x,
        y,
        marker="o",
        markersize=float(marker_size),
        markerfacecolor="none",
        markeredgecolor="#ffe066",
        markeredgewidth=2.8,
        zorder=10,
    )
    label_off = max(5.0, float(marker_size) * 0.48)
    ax.text(
        x,
        y + label_off,
        str(label),
        color="#ffe066",
        fontsize=float(marker_fontsize),
        fontweight="bold",
        ha="center",
        va="top",
        zorder=11,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="#0f1117",
            alpha=0.85,
            edgecolor="#ffe066",
            linewidth=1.1,
        ),
    )


def _draw_pixel_panel_title(ax, pixel_label: str) -> None:
    """Prominent boxed title for a per-pixel concept panel."""
    ax.text(
        1.0,
        1.92,
        f"Pixel {pixel_label}",
        ha="center",
        va="center",
        color="#ffe066",
        fontsize=16,
        fontweight="bold",
        clip_on=False,
        zorder=20,
        bbox=dict(
            boxstyle="round,pad=0.42",
            facecolor="#1a1d26",
            edgecolor="#ffe066",
            linewidth=1.8,
            alpha=0.96,
        ),
    )


def _draw_pixel_concept_panel(
    ax,
    terms: list[dict[str, Any]],
    *,
    pixel_label: str,
    thumb_fn,
    renormalize: bool,
    use_heatmap: bool,
) -> None:
    """Draw a compact 2x2 concept grid beside the tissue map."""
    ax.set_facecolor("#0f1117")
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    cells = _panel_thumb_cells()
    for term, (cx, cy, half_w, half_h) in zip(terms, cells):
        _, weight_label = _format_term_weight(term, renormalize_terms=renormalize)
        ki = int(term["concept"])
        rgb = term["color_rgb"]
        if use_heatmap:
            thumb = _annotate_concept_thumb(thumb_fn(ki), ki, weight_label)
            ext = _thumb_extent_in_cell(cx, cy, half_w, half_h, thumb.shape)
            ax.imshow(
                thumb,
                extent=ext,
                aspect="equal",
                interpolation="nearest",
                zorder=2,
            )
        else:
            r = min(half_w, half_h) * 0.92
            ax.add_patch(
                Circle((cx, cy), r, facecolor=_rgb01(rgb), edgecolor="white", linewidth=1.2)
            )
            ax.text(
                cx,
                cy,
                f"c{ki}\n{weight_label}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
    if use_heatmap:
        _draw_thumb_grid_border(ax, cells)
    _draw_pixel_panel_title(ax, pixel_label)


def compute_sparsity_stats(
    alpha_hwk: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_concepts: int,
    top_k: int | None = None,
    weight_threshold: float = 1e-5,
) -> dict[str, float]:
    """Summarize how sparse per-pixel concept weights are."""
    alpha = np.asarray(alpha_hwk, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    tissue = alpha[mask]
    if tissue.size == 0:
        return {}
    eps = 1e-10
    nnz = (tissue > float(weight_threshold)).sum(axis=1).astype(np.float64)
    ent = -(tissue * np.log(tissue + eps)).sum(axis=1)
    k = int(n_concepts)
    top1 = tissue.max(axis=1)
    order = np.argsort(-tissue, axis=1)
    top2 = tissue[np.arange(tissue.shape[0]), order[:, 1]] if k > 1 else np.zeros(tissue.shape[0])
    max_k = int(top_k) if top_k is not None else k
    return {
        "n_tissue_pixels": float(tissue.shape[0]),
        "n_concepts": float(k),
        "top_k_cap": float(max_k),
        "mean_nnz": float(nnz.mean()),
        "median_nnz": float(np.median(nnz)),
        "min_nnz": float(nnz.min()),
        "max_nnz": float(nnz.max()),
        "mean_top1_weight": float(top1.mean()),
        "mean_top2_weight": float(top2.mean()),
        "mean_entropy_nats": float(ent.mean()),
        "max_entropy_nats": float(np.log(max(max_k, 2))),
        "entropy_fraction_of_uniform": float(ent.mean() / max(np.log(max(max_k, 2)), eps)),
        "pct_pixels_with_top1_above_0.5": float((top1 > 0.5).mean() * 100.0),
        "pct_pixels_with_top1_above_0.75": float((top1 > 0.75).mean() * 100.0),
    }


def _interior_candidate_indices(
    dists: np.ndarray,
    *,
    n_needed: int,
    min_interior_px: float,
) -> np.ndarray:
    """Indices of tissue pixels at least min_interior_px from background (with fallback)."""
    n_needed = max(1, int(n_needed))
    pos = dists[dists > 0]
    thresholds = [float(min_interior_px)]
    if pos.size:
        for pct in (65.0, 50.0, 35.0, 20.0):
            thresholds.append(float(np.percentile(pos, pct)))
    seen: set[float] = set()
    for thresh in thresholds:
        t = max(0.0, float(thresh))
        if t in seen:
            continue
        seen.add(t)
        keep = np.flatnonzero(dists >= t)
        if keep.size >= n_needed:
            return keep
    return np.arange(dists.size, dtype=np.int64)


def _pick_example_pixels(
    mics_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    teacher_emb: np.ndarray | None = None,
    n: int,
    seed: int,
    spatial_weight: float = 2.0,
    interior_min_px: float = 10.0,
    interior_weight: float = 3.0,
) -> list[tuple[int, int, str]]:
    """Diverse interior tissue pixels via greedy max-min on feature + spatial distance."""
    rgb = np.asarray(mics_rgb, dtype=np.uint8)
    mask = np.asarray(valid_mask, dtype=bool)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return []
    n_pick = max(1, min(int(n), ys.size))

    dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dists = dist_map[ys, xs]
    cand = _interior_candidate_indices(dists, n_needed=n_pick, min_interior_px=float(interior_min_px))
    ys = ys[cand]
    xs = xs[cand]
    dists = dists[cand]

    if teacher_emb is not None:
        feats = np.asarray(teacher_emb, dtype=np.float64)[mask].reshape(-1, 3)[cand]
    else:
        flat_rgb = rgb[mask].reshape(-1, 1, 3)[cand]
        feats = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)

    feat = feats.astype(np.float64, copy=True)
    feat -= feat.mean(axis=0)
    fstd = feat.std(axis=0)
    fstd[fstd < 1e-6] = 1.0
    feat /= fstd

    coords = np.stack([ys.astype(np.float64), xs.astype(np.float64)], axis=1)
    coords -= coords.mean(axis=0)
    cstd = coords.std(axis=0)
    cstd[cstd < 1e-6] = 1.0
    coords /= cstd
    sw = max(0.0, float(spatial_weight))
    iw = max(0.0, float(interior_weight))
    idist = dists.astype(np.float64)
    idist -= idist.mean()
    istd = idist.std()
    if istd < 1e-6:
        istd = 1.0
    idist /= istd
    parts = [feat]
    if sw > 0.0:
        parts.append(sw * coords)
    if iw > 0.0:
        parts.append(iw * idist.reshape(-1, 1))
    combined = np.concatenate(parts, axis=1)

    rng = np.random.default_rng(int(seed))
    if dists.size == 0:
        start = int(rng.integers(0, combined.shape[0]))
    else:
        dist_floor = float(np.percentile(dists, 80.0))
        interior = np.flatnonzero(dists >= dist_floor)
        pool = interior if interior.size else np.arange(combined.shape[0])
        start = int(rng.choice(pool))
    selected = [start]
    dist_max = max(float(dists.max()), 1e-6)
    while len(selected) < n_pick:
        best_i = -1
        best_score = -1.0
        for i in range(combined.shape[0]):
            if i in selected:
                continue
            d_feat = min(float(np.linalg.norm(combined[i] - combined[j])) for j in selected)
            d_int = float(dists[i]) / dist_max
            score = d_feat * (0.2 + 0.8 * d_int)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)

    def _example_label(i: int) -> str:
        # A..Z, then AA..AZ, BA.. (Excel-style) so n>26 stays short and sortable-ish
        i = int(i)
        if i < 26:
            return chr(ord("A") + i)
        chars: list[str] = []
        n = i
        while True:
            chars.append(chr(ord("A") + (n % 26)))
            n = n // 26 - 1
            if n < 0:
                break
        return "".join(reversed(chars))

    labels = [_example_label(i) for i in range(len(selected))]
    return [(int(ys[i]), int(xs[i]), labels[j]) for j, i in enumerate(selected)]


def _format_term_weight(term: dict[str, Any], *, renormalize_terms: bool) -> tuple[float, str]:
    """Return (value for arrow width, label text) using true mixture fraction when possible."""
    w = float(term.get("weight", 0.0))
    if renormalize_terms:
        frac = float(term.get("weight_norm", w))
    else:
        frac = w
    return frac, f"{frac:.1%}"


def _weight_label_slug(label: str) -> str:
    """Filesystem-safe suffix for annotated concept thumbnail weights."""
    s = str(label).strip().replace("%", "pct")
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in {".", "-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "weight"


def _annotate_concept_thumb(thumb: np.ndarray, concept_idx: int, weight_label: str) -> np.ndarray:
    """Tiny label overlaid on the bottom of a concept thumbnail."""
    out = np.asarray(thumb, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    scale = max(0.22, w / 340.0)
    thick = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"c{int(concept_idx)} {weight_label}"
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    pad = max(2, int(round(0.04 * h)))
    x = max(pad, (w - tw) // 2)
    y = h - pad - baseline
    cv2.rectangle(out, (x - pad, y - th - pad), (min(w - 1, x + tw + pad), min(h - 1, y + baseline + pad)), (0, 0, 0), -1)
    cv2.putText(out, label, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def _rgb01(rgb: np.ndarray) -> tuple[float, float, float]:
    c = np.asarray(rgb, dtype=np.float64).reshape(3) / 255.0
    return float(c[0]), float(c[1]), float(c[2])


def _concept_activation_thumb(
    alpha_hwk: np.ndarray,
    concept_idx: int,
    valid_mask: np.ndarray,
    concept_rgb: np.ndarray,
    *,
    thumb_px: int = 48,
    activation_percentile: float = 98.0,
    activation_gamma: float = 0.85,
    activation_colormap: str = "assigned",
    activation_vmax: float | None = None,
    equalize_activation: bool = False,
) -> np.ndarray:
    """Mini heatmap for one concept (tissue bbox crop, aspect-preserving resize)."""
    act = np.asarray(alpha_hwk, dtype=np.float32)[:, :, int(concept_idx)]
    mask = np.asarray(valid_mask, dtype=bool)
    rgb = _activation_rgb(
        act,
        mask,
        percentile=float(activation_percentile),
        gamma=float(activation_gamma),
        colormap=str(activation_colormap),
        concept_rgb=np.asarray(concept_rgb, dtype=np.uint8),
        vmax=activation_vmax,
        equalize_activation=bool(equalize_activation),
    )
    ys, xs = np.where(mask)
    if ys.size == 0:
        px = max(8, int(thumb_px))
        return np.zeros((px, px, 3), dtype=np.uint8)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    px = max(8, int(thumb_px))
    scale = px / max(ch, cw, 1)
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)


def _make_concept_thumb_fn(
    alpha_hwk: np.ndarray,
    valid_mask: np.ndarray,
    concept_rgbs: np.ndarray,
    *,
    thumb_px: int,
    activation_percentile: float,
    activation_gamma: float,
    activation_colormap: str,
    equalize_activation: bool,
) -> Any:
    """Cached concept activation thumbnail factory."""
    alpha = np.asarray(alpha_hwk, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    palette = np.asarray(concept_rgbs, dtype=np.uint8)
    px = max(8, int(thumb_px))
    cache: dict[int, np.ndarray] = {}

    def _thumb_for(ki: int) -> np.ndarray:
        if ki not in cache:
            cache[ki] = _concept_activation_thumb(
                alpha,
                ki,
                mask,
                palette[ki],
                thumb_px=px,
                activation_percentile=float(activation_percentile),
                activation_gamma=float(activation_gamma),
                activation_colormap=str(activation_colormap),
                equalize_activation=bool(equalize_activation),
            )
        return cache[ki]

    return _thumb_for


def _display_variant_path(out_path: Path, variant: str) -> Path:
    """``concept_arrows_on_map.png`` -> ``concept_arrows_on_map_raw.png``."""
    p = Path(out_path)
    return p.with_name(f"{p.stem}_{variant}{p.suffix}")


def _equalize_display_jobs(
    out_path: Path,
    *,
    equalize: bool,
    save_both: bool,
) -> list[tuple[bool, Path]]:
    """Return ``(equalize_flag, output_path)`` jobs for map or thumb rendering."""
    if bool(save_both):
        return [
            (False, _display_variant_path(out_path, "raw")),
            (True, _display_variant_path(out_path, "equalized")),
        ]
    return [(bool(equalize), Path(out_path))]


def _copy_primary_display_variant(
    out_path: Path,
    *,
    equalize: bool,
    variant_paths: dict[bool, str],
) -> str:
    """Copy the configured primary variant onto the legacy filename."""
    primary = variant_paths.get(bool(equalize))
    if primary is None:
        raise KeyError(f"Missing display variant for equalize={equalize!r}")
    shutil.copy2(primary, out_path)
    return str(out_path)



def _save_standalone_pixel_panel(
    terms: list[dict[str, Any]],
    *,
    pixel_label: str,
    thumb_fn,
    renormalize: bool,
    use_heatmap: bool,
    out_path: Path,
    dpi: int,
) -> str:
    """Save one pixel's 2x2 concept panel as a standalone high-DPI figure."""
    fig, ax = plt.subplots(figsize=(4.2, 4.4), facecolor="#0f1117")
    _draw_pixel_concept_panel(
        ax,
        terms,
        pixel_label=str(pixel_label),
        thumb_fn=thumb_fn,
        renormalize=bool(renormalize),
        use_heatmap=bool(use_heatmap),
    )
    out_path = Path(out_path)
    fig.savefig(
        out_path,
        dpi=int(dpi),
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    return str(out_path)


def _save_single_pixel_concept_arrows_map(
    mics_rgb: np.ndarray,
    alpha_hwk: np.ndarray,
    concept_rgbs: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    *,
    y: int,
    x: int,
    lab: str,
    tissue_mean_alpha: np.ndarray | None = None,
    concept_rank_mode: str = "relative",
    top_terms: int = 4,
    min_weight: float = 0.04,
    concept_node_style: str = "heatmap",
    concept_thumb_px: int = 360,
    activation_percentile: float = 98.0,
    activation_gamma: float = 0.85,
    activation_colormap: str = "magma",
    equalize_map_display: bool = True,
    equalize_thumb_activation: bool = False,
    map_marker_fontsize: float = 16.0,
    map_marker_size: float = 12.0,
    dpi: int = 200,
) -> str:
    """Tissue map with one labeled pixel + its concept panel (for manual figure assembly)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mics = np.asarray(mics_rgb, dtype=np.uint8)
    if bool(equalize_map_display):
        mics = _equalize_map_display(mics, valid_mask)
    alpha = np.asarray(alpha_hwk, dtype=np.float32)
    palette = np.asarray(concept_rgbs, dtype=np.uint8)
    mask = np.asarray(valid_mask, dtype=bool)
    h, w = mics.shape[:2]
    use_heatmap = str(concept_node_style).strip().lower() in {"heatmap", "activation", "thumb"}
    map_top_n = max(4, int(top_terms))
    thumb_px = max(8, int(concept_thumb_px))

    expl = build_color_explanation(
        alpha[int(y), int(x)],
        palette,
        top_n=map_top_n,
        min_weight=float(min_weight),
        color_label=f"Pixel {lab}",
        tissue_mean_alpha=tissue_mean_alpha,
        rank_mode=str(concept_rank_mode),
    )
    renormalize = bool(expl.get("renormalize_terms", True))
    terms = _fill_pixel_terms(
        alpha[int(y), int(x)],
        palette,
        expl,
        map_top_n=map_top_n,
        renormalize_terms=renormalize,
    )
    thumb_fn = _make_concept_thumb_fn(
        alpha,
        mask,
        palette,
        thumb_px=thumb_px,
        activation_percentile=float(activation_percentile),
        activation_gamma=float(activation_gamma),
        activation_colormap=str(activation_colormap),
        equalize_activation=bool(equalize_thumb_activation),
    )

    tissue_aspect = float(w) / max(float(h), 1.0)
    fig = plt.figure(figsize=(11.5, 5.8), facecolor="#0f1117")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.04, left=0.02, right=0.98, bottom=0.04, top=0.96)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])
    ax_map.set_facecolor("#0f1117")
    ax_map.imshow(mics)
    ax_map.set_xlim(-0.5, w - 0.5)
    ax_map.set_ylim(h - 0.5, -0.5)
    ax_map.set_box_aspect(tissue_aspect)
    ax_map.axis("off")
    _draw_map_pixel_marker(
        ax_map,
        float(x),
        float(y),
        str(lab),
        marker_size=float(map_marker_size),
        marker_fontsize=float(map_marker_fontsize),
    )
    ax_panel.set_box_aspect(1.0)
    _draw_pixel_concept_panel(
        ax_panel,
        terms,
        pixel_label=str(lab),
        thumb_fn=thumb_fn,
        renormalize=renormalize,
        use_heatmap=use_heatmap,
    )
    fig.savefig(out_path, dpi=int(dpi), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return str(out_path)


def save_pixel_interpretability_exports(
    mics_rgb: np.ndarray,
    alpha_hwk: np.ndarray,
    concept_rgbs: np.ndarray,
    valid_mask: np.ndarray,
    out_dir: Path,
    *,
    example_pixels: list[tuple[int, int, str]],
    tissue_mean_alpha: np.ndarray | None = None,
    concept_rank_mode: str = "relative",
    top_terms: int = 4,
    min_weight: float = 0.04,
    concept_node_style: str = "heatmap",
    export_thumb_px: int = 480,
    export_dpi: int = 300,
    activation_percentile: float = 98.0,
    activation_gamma: float = 0.85,
    activation_colormap: str = "magma",
    equalize_map_display: bool = True,
    equalize_thumb_activation: bool = False,
    save_both_display_variants: bool = False,
    map_fig_width: float = 24.0,
    map_fig_height: float = 22.0,
    map_marker_fontsize: float = 18.0,
    map_marker_size: float = 14.0,
    subdir: str = "interpretability",
) -> dict[str, Any]:
    """High-DPI map overlay plus per-pixel concept thumbnail folders."""
    interp_dir = Path(out_dir) / str(subdir)
    interp_dir.mkdir(parents=True, exist_ok=True)
    alpha = np.asarray(alpha_hwk, dtype=np.float32)
    palette = np.asarray(concept_rgbs, dtype=np.uint8)
    use_heatmap = str(concept_node_style).strip().lower() in {"heatmap", "activation", "thumb"}
    map_top_n = max(4, int(top_terms))
    export_px = max(64, int(export_thumb_px))

    # Shared overview map keeps a small panel set; per-pixel maps are saved below.
    map_pixels = list(example_pixels[: len(_PANEL_GS)])
    map_path = _save_map_concept_overlay_variants(
        mics_rgb,
        alpha_hwk,
        concept_rgbs,
        valid_mask,
        interp_dir / "concept_arrows_on_map.png",
        example_pixels=map_pixels,
        tissue_mean_alpha=tissue_mean_alpha,
        concept_rank_mode=str(concept_rank_mode),
        top_terms=int(top_terms),
        min_weight=float(min_weight),
        concept_node_style=str(concept_node_style),
        concept_thumb_px=int(export_px),
        map_fig_width=float(map_fig_width),
        map_fig_height=float(map_fig_height),
        map_marker_fontsize=float(map_marker_fontsize),
        map_marker_size=float(map_marker_size),
        equalize_map_display=bool(equalize_map_display),
        equalize_thumb_activation=bool(equalize_thumb_activation),
        save_both_display_variants=bool(save_both_display_variants),
        activation_percentile=float(activation_percentile),
        activation_gamma=float(activation_gamma),
        activation_colormap=str(activation_colormap),
        dpi=int(export_dpi),
    )

    thumb_jobs = _equalize_display_jobs(
        Path("panel.png"),
        equalize=bool(equalize_thumb_activation),
        save_both=False,
    )
    thumb_fns: dict[bool, Any] = {
        eq: _make_concept_thumb_fn(
            alpha_hwk,
            valid_mask,
            concept_rgbs,
            thumb_px=export_px,
            activation_percentile=float(activation_percentile),
            activation_gamma=float(activation_gamma),
            activation_colormap=str(activation_colormap),
            equalize_activation=bool(eq),
        )
        for eq, _ in thumb_jobs
    }

    pixel_records: list[dict[str, Any]] = []
    for y, x, lab in example_pixels:
        expl = build_color_explanation(
            alpha[y, x],
            palette,
            top_n=map_top_n,
            min_weight=float(min_weight),
            color_label=f"Pixel {lab}",
            tissue_mean_alpha=tissue_mean_alpha,
            rank_mode=str(concept_rank_mode),
        )
        renormalize = bool(expl.get("renormalize_terms", True))
        terms = _fill_pixel_terms(
            alpha[y, x],
            palette,
            expl,
            map_top_n=map_top_n,
            renormalize_terms=renormalize,
        )
        pixel_dir = interp_dir / f"pixel_{lab}"
        pixel_dir.mkdir(parents=True, exist_ok=True)

        panel_jobs = _equalize_display_jobs(
            pixel_dir / "panel.png",
            equalize=bool(equalize_thumb_activation),
            save_both=False,
        )
        panel_paths: dict[str, str] = {}
        for eq_panel, panel_out in panel_jobs:
            panel_paths["equalized" if eq_panel else "raw"] = _save_standalone_pixel_panel(
                terms,
                pixel_label=str(lab),
                thumb_fn=thumb_fns[eq_panel],
                renormalize=renormalize,
                use_heatmap=use_heatmap,
                out_path=panel_out,
                dpi=int(export_dpi),
            )
        panel_path = panel_paths["equalized" if bool(equalize_thumb_activation) else "raw"]

        map_out = _save_single_pixel_concept_arrows_map(
            mics_rgb,
            alpha_hwk,
            concept_rgbs,
            valid_mask,
            pixel_dir / "concept_arrows_on_map.png",
            y=int(y),
            x=int(x),
            lab=str(lab),
            tissue_mean_alpha=tissue_mean_alpha,
            concept_rank_mode=str(concept_rank_mode),
            top_terms=int(top_terms),
            min_weight=float(min_weight),
            concept_node_style=str(concept_node_style),
            concept_thumb_px=int(export_px),
            activation_percentile=float(activation_percentile),
            activation_gamma=float(activation_gamma),
            activation_colormap=str(activation_colormap),
            equalize_map_display=bool(equalize_map_display),
            equalize_thumb_activation=bool(equalize_thumb_activation),
            map_marker_fontsize=float(map_marker_fontsize),
            map_marker_size=float(map_marker_size),
            dpi=int(export_dpi),
        )

        thumb_records: list[dict[str, Any]] = []
        for slot, term in enumerate(terms):
            ki = int(term["concept"])
            _, weight_label = _format_term_weight(term, renormalize_terms=renormalize)
            slug = _weight_label_slug(weight_label)
            slot_records: dict[str, Any] = {
                "slot": int(slot + 1),
                "concept": ki,
                "weight_label": weight_label,
            }
            for eq_thumb, variant_key in [(bool(equalize_thumb_activation), "primary")]:
                raw = np.asarray(thumb_fns[eq_thumb](ki), dtype=np.uint8)
                raw_path = pixel_dir / f"slot{slot + 1:02d}_concept_c{ki:03d}_raw.png"
                ann_path = pixel_dir / f"slot{slot + 1:02d}_concept_c{ki:03d}_{slug}.png"
                Image.fromarray(raw).save(raw_path)
                Image.fromarray(_annotate_concept_thumb(raw, ki, weight_label)).save(ann_path)
                slot_records[f"{variant_key}_thumb_png"] = str(raw_path)
                slot_records[f"{variant_key}_annotated_png"] = str(ann_path)
            thumb_records.append(slot_records)

        pixel_records.append(
            {
                "label": str(lab),
                "y": int(y),
                "x": int(x),
                "dir": str(pixel_dir),
                "panel_png": panel_path,
                "concept_arrows_on_map_png": map_out,
                "thumbnails": thumb_records,
                "explanation": expl,
            }
        )

    manifest_path = interp_dir / "pixels_manifest.json"
    manifest_path.write_text(json.dumps(pixel_records, indent=2), encoding="utf-8")
    logger.info(
        "Wrote interpretability exports: %s (%d pixels, dpi=%d, thumb_px=%d)",
        interp_dir,
        len(pixel_records),
        int(export_dpi),
        export_px,
    )
    return {
        "interpretability_dir": str(interp_dir),
        "concept_arrows_on_map_png": map_path.get("primary"),
        "concept_arrows_on_map_variants": map_path,
        "pixels_manifest_json": str(manifest_path),
        "pixels": pixel_records,
        "export_dpi": int(export_dpi),
        "export_thumb_px": int(export_px),
    }


def save_original_reconstruction_panel(
    mics_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    *,
    sparsity: dict[str, float],
    pixel_rgb_mse: float,
    embedding_mse: float,
    example_pixels: list[tuple[int, int, str]] | None = None,
    dpi: int = 200,
) -> str:
    """Two-panel Original | Reconstruction with sparsity summary."""
    out_path = Path(out_path)
    mics = np.asarray(mics_rgb, dtype=np.uint8)
    recon = np.asarray(recon_rgb, dtype=np.uint8)
    mask = np.asarray(valid_mask, dtype=bool)

    fig = plt.figure(figsize=(13.5, 6.2), facecolor="#0f1117")
    gs = fig.add_gridspec(2, 2, height_ratios=[5.0, 0.85], hspace=0.12, wspace=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    for ax, img, title in (
        (ax0, mics, "Original MiCS"),
        (ax1, recon, "Concept reconstruction"),
    ):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)
        if example_pixels:
            for y, x, lab in example_pixels:
                _draw_map_pixel_marker(
                    ax,
                    float(x),
                    float(y),
                    str(lab),
                    marker_size=9.0,
                    marker_fontsize=11.0,
                )

    mean_nnz = sparsity.get("mean_nnz", float("nan"))
    top1 = sparsity.get("mean_top1_weight", float("nan"))
    ent_frac = sparsity.get("entropy_fraction_of_uniform", float("nan"))
    cap = sparsity.get("top_k_cap", float("nan"))
    stats_line = (
        f"RGB MSE {pixel_rgb_mse:.1f}   |   embedding MSE {embedding_mse:.1f}   |   "
        f"active concepts {mean_nnz:.2f} / {cap:.0f} per pixel   |   "
        f"mean top weight {top1:.0%}   |   entropy {ent_frac:.0%} of uniform"
    )
    ax_bar.set_facecolor("#0f1117")
    ax_bar.axis("off")
    ax_bar.text(
        0.5,
        0.55,
        stats_line,
        ha="center",
        va="center",
        color="#d8dee9",
        fontsize=11,
        transform=ax_bar.transAxes,
    )
    ax_bar.text(
        0.5,
        0.12,
        "Sparse softmax: at most top-k concepts per pixel; weights sum to 1.",
        ha="center",
        va="center",
        color="#8b949e",
        fontsize=10,
        transform=ax_bar.transAxes,
    )

    fig.savefig(out_path, dpi=int(dpi), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote side-by-side panel: %s", out_path)
    return str(out_path)


def save_map_concept_overlay(
    mics_rgb: np.ndarray,
    alpha_hwk: np.ndarray,
    concept_rgbs: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    *,
    example_pixels: list[tuple[int, int, str]],
    tissue_mean_alpha: np.ndarray | None = None,
    concept_rank_mode: str = "relative",
    top_terms: int = 4,
    min_weight: float = 0.04,
    concept_node_style: str = "heatmap",
    concept_thumb_px: int = 112,
    activation_percentile: float = 98.0,
    activation_gamma: float = 0.85,
    activation_colormap: str = "magma",
    grid_offset_px: float = 170.0,
    grid_spacing_px: float | None = None,
    map_fig_width: float = 24.0,
    map_fig_height: float = 22.0,
    map_marker_fontsize: float = 18.0,
    map_marker_size: float = 14.0,
    equalize_map_display: bool = True,
    equalize_thumb_activation: bool = False,
    dpi: int = 200,
) -> str:
    """Labeled tissue map with compact concept panels (no arrows)."""
    del grid_offset_px, grid_spacing_px  # legacy
    out_path = Path(out_path)
    mics = np.asarray(mics_rgb, dtype=np.uint8)
    if bool(equalize_map_display):
        mics = _equalize_map_display(mics, valid_mask)
    alpha = np.asarray(alpha_hwk, dtype=np.float32)
    palette = np.asarray(concept_rgbs, dtype=np.uint8)
    mask = np.asarray(valid_mask, dtype=bool)
    h, w = mics.shape[:2]
    use_heatmap = str(concept_node_style).strip().lower() in {"heatmap", "activation", "thumb"}
    thumb_px = max(8, int(concept_thumb_px))
    map_top_n = max(4, int(top_terms))

    fig = plt.figure(figsize=(float(map_fig_width), float(map_fig_height)), facecolor="#0f1117")
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.95, 1.95, 1.95],
        height_ratios=[1.0, 1.0],
        wspace=0.0,
        hspace=-0.08,
        left=0.008,
        right=0.992,
        bottom=0.008,
        top=0.992,
    )
    ax_tissue = fig.add_subplot(gs[1, 1])
    panel_axes = [fig.add_subplot(gs[r, c]) for r, c in _PANEL_GS]
    # Unused slots (fewer than 5 pixels) stay dark/hidden instead of default white boxes.
    for ax in panel_axes:
        ax.set_facecolor("#0f1117")
        ax.axis("off")
    tissue_box_aspect = float(w) / max(float(h), 1.0)
    for pi, ax in enumerate(panel_axes):
        row, _col = _PANEL_GS[pi]
        if row == 0:
            ax.set_anchor("S")
        else:
            ax.set_anchor("C")
            ax.set_box_aspect(1.0)
    ax_tissue.set_anchor("C")
    ax_tissue.set_box_aspect(tissue_box_aspect)

    ax_tissue.imshow(mics)
    ax_tissue.set_xlim(-0.5, w - 0.5)
    ax_tissue.set_ylim(h - 0.5, -0.5)
    ax_tissue.axis("off")

    _thumb_for = _make_concept_thumb_fn(
        alpha,
        mask,
        palette,
        thumb_px=thumb_px,
        activation_percentile=float(activation_percentile),
        activation_gamma=float(activation_gamma),
        activation_colormap=str(activation_colormap),
        equalize_activation=bool(equalize_thumb_activation),
    )

    for pi, (y, x, lab) in enumerate(example_pixels[: len(_PANEL_GS)]):
        expl = build_color_explanation(
            alpha[y, x],
            palette,
            top_n=map_top_n,
            min_weight=float(min_weight),
            color_label=f"Pixel {lab}",
            tissue_mean_alpha=tissue_mean_alpha,
            rank_mode=str(concept_rank_mode),
        )
        terms = _fill_pixel_terms(
            alpha[y, x],
            palette,
            expl,
            map_top_n=map_top_n,
            renormalize_terms=bool(expl.get("renormalize_terms", True)),
        )
        renormalize = bool(expl.get("renormalize_terms", True))
        _draw_pixel_concept_panel(
            panel_axes[pi],
            terms,
            pixel_label=str(lab),
            thumb_fn=_thumb_for,
            renormalize=renormalize,
            use_heatmap=use_heatmap,
        )
        _draw_map_pixel_marker(
            ax_tissue,
            float(x),
            float(y),
            str(lab),
            marker_size=float(map_marker_size),
            marker_fontsize=float(map_marker_fontsize),
        )

    # Mark remaining example pixels on the tissue map (no side panels beyond _PANEL_GS).
    for y, x, lab in example_pixels[len(_PANEL_GS) :]:
        _draw_map_pixel_marker(
            ax_tissue,
            float(x),
            float(y),
            str(lab),
            marker_size=float(map_marker_size),
            marker_fontsize=float(map_marker_fontsize),
        )

    fig.savefig(out_path, dpi=int(dpi), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("Wrote map concept overlay: %s", out_path)
    return str(out_path)


def _save_map_concept_overlay_variants(
    mics_rgb: np.ndarray,
    alpha_hwk: np.ndarray,
    concept_rgbs: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    *,
    equalize_map_display: bool,
    equalize_thumb_activation: bool,
    save_both_display_variants: bool,
    **kwargs: Any,
) -> dict[str, str]:
    """Save map overlay; optionally write raw and equalized variants."""
    jobs = _equalize_display_jobs(
        out_path,
        equalize=bool(equalize_map_display),
        save_both=bool(save_both_display_variants),
    )
    paths: dict[str, str] = {}
    for eq_map, job_path in jobs:
        paths["equalized" if eq_map else "raw"] = save_map_concept_overlay(
            mics_rgb,
            alpha_hwk,
            concept_rgbs,
            valid_mask,
            job_path,
            equalize_map_display=eq_map,
            equalize_thumb_activation=bool(equalize_thumb_activation),
            **kwargs,
        )
    if bool(save_both_display_variants):
        primary_key = "equalized" if bool(equalize_map_display) else "raw"
        paths["primary"] = _copy_primary_display_variant(
            out_path,
            equalize=bool(equalize_map_display),
            variant_paths={True: paths["equalized"], False: paths["raw"]},
        )
    else:
        paths["primary"] = paths["equalized" if bool(equalize_map_display) else "raw"]
    return paths


def _draw_mixture_row(
    ax,
    *,
    pixel_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    terms: list[dict[str, Any]],
    label: str,
    yx: tuple[int, int],
    alpha_hwk: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    concept_node_style: str = "heatmap",
    concept_thumb_px: int = 112,
    activation_percentile: float = 98.0,
    activation_gamma: float = 0.85,
    activation_colormap: str = "magma",
    renormalize_terms: bool = True,
    equalize_activation: bool = False,
) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.set_facecolor("#0f1117")
    ax.axis("off")

    px, py = 0.55, 1.1
    rx, ry = 9.35, 1.1
    r_node = 0.34

    ax.add_patch(Circle((px, py), r_node, facecolor=_rgb01(pixel_rgb), edgecolor="white", linewidth=1.5))
    ax.text(px, py - 0.72, f"Pixel {label}\n({yx[0]}, {yx[1]})", ha="center", va="top", color="#c9d1d9", fontsize=9)

    n = max(1, len(terms))
    cx_start, cx_end = 2.4, 7.0
    xs = np.linspace(cx_start, cx_end, n) if n > 1 else np.array([(cx_start + cx_end) / 2.0])
    cy = 1.1
    use_heatmap = (
        str(concept_node_style).strip().lower() in {"heatmap", "activation", "thumb"}
        and alpha_hwk is not None
        and valid_mask is not None
    )
    node_r = 0.34 if not use_heatmap else 0.62

    for i, (term, cx) in enumerate(zip(terms, xs)):
        rgb = term["color_rgb"]
        ki = int(term["concept"])
        _, weight_label = _format_term_weight(term, renormalize_terms=renormalize_terms)
        if use_heatmap:
            thumb = _annotate_concept_thumb(
                _concept_activation_thumb(
                    alpha_hwk,
                    ki,
                    valid_mask,
                    rgb,
                    thumb_px=int(concept_thumb_px),
                    activation_percentile=float(activation_percentile),
                    activation_gamma=float(activation_gamma),
                    activation_colormap=str(activation_colormap),
                    equalize_activation=bool(equalize_activation),
                ),
                ki,
                weight_label,
            )
            half = node_r * 0.95
            ext = _thumb_extent_in_cell(cx, cy, half, half, thumb.shape)
            ax.imshow(
                thumb,
                extent=ext,
                aspect="equal",
                zorder=7,
                interpolation="nearest",
            )
            ax.add_patch(
                plt.Rectangle(
                    (ext[0], ext[2]),
                    ext[1] - ext[0],
                    ext[3] - ext[2],
                    fill=False,
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=8,
                )
            )
        else:
            ax.add_patch(Circle((cx, cy), node_r * 0.92, facecolor=_rgb01(rgb), edgecolor="#555", linewidth=1.0))
            ax.text(
                cx,
                cy,
                f"c{ki}\n{weight_label}",
                ha="center",
                va="center",
                color="#e6edf3",
                fontsize=8.5,
                fontweight="bold",
                zorder=9,
            )

        arr_in = FancyArrowPatch(
            (px + r_node, py),
            (cx - node_r * 0.95, cy),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=_MAP_ARROW_LW,
            color=_MAP_ARROW_COLOR,
            alpha=0.85,
        )
        ax.add_patch(arr_in)

        arr_out = FancyArrowPatch(
            (cx + node_r * 0.95, cy),
            (rx - r_node, ry),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=_MAP_ARROW_LW,
            color=_MAP_ARROW_COLOR,
            alpha=0.55,
        )
        ax.add_patch(arr_out)

    ax.add_patch(Circle((rx, ry), r_node, facecolor=_rgb01(recon_rgb), edgecolor="white", linewidth=1.5))
    ax.text(rx, ry - 0.72, "Recon", ha="center", va="top", color="#c9d1d9", fontsize=9)

    arr_final = FancyArrowPatch(
        (px + r_node, py),
        (rx - r_node, ry),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.0,
        color="#8b949e",
        linestyle="--",
        alpha=0.35,
    )
    ax.add_patch(arr_final)


def save_pixel_mixture_figure(
    mics_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    alpha_hwk: np.ndarray,
    concept_rgbs: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    *,
    example_pixels: list[tuple[int, int, str]] | None = None,
    n_examples: int = 4,
    top_terms: int = 4,
    min_weight: float = 0.04,
    concept_rank_mode: str = "relative",
    seed: int = 42,
    dpi: int = 180,
    concept_node_style: str = "heatmap",
    concept_thumb_px: int = 112,
    activation_percentile: float = 98.0,
    activation_gamma: float = 0.85,
    activation_colormap: str = "magma",
    tissue_mean_alpha: np.ndarray | None = None,
    equalize_thumb_activation: bool = False,
    save_both_display_variants: bool = False,
) -> dict[str, Any]:
    """Arrow diagram: pixel -> weighted concepts -> reconstruction color."""
    out_path = Path(out_path)
    jobs = _equalize_display_jobs(
        out_path,
        equalize=bool(equalize_thumb_activation),
        save_both=False,
    )
    variant_paths: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for eq, job_path in jobs:
        job_summary = _save_pixel_mixture_figure_once(
            mics_rgb,
            recon_rgb,
            alpha_hwk,
            concept_rgbs,
            valid_mask,
            job_path,
            example_pixels=example_pixels,
            n_examples=int(n_examples),
            top_terms=int(top_terms),
            min_weight=float(min_weight),
            concept_rank_mode=str(concept_rank_mode),
            seed=int(seed),
            concept_node_style=str(concept_node_style),
            concept_thumb_px=int(concept_thumb_px),
            activation_percentile=float(activation_percentile),
            activation_gamma=float(activation_gamma),
            activation_colormap=str(activation_colormap),
            tissue_mean_alpha=tissue_mean_alpha,
            equalize_thumb_activation=bool(eq),
            dpi=int(dpi),
        )
        variant_paths["equalized" if eq else "raw"] = str(job_summary["mixture_figure_png"])
        if not records:
            records = list(job_summary.get("examples", []))

    variant_paths["primary"] = variant_paths["equalized" if bool(equalize_thumb_activation) else "raw"]

    return {
        "mixture_figure_png": variant_paths.get("primary"),
        "mixture_figure_variants": variant_paths,
        "examples": records,
    }


def _save_pixel_mixture_figure_once(
    mics_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    alpha_hwk: np.ndarray,
    concept_rgbs: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    *,
    example_pixels: list[tuple[int, int, str]] | None = None,
    n_examples: int = 4,
    top_terms: int = 4,
    min_weight: float = 0.04,
    concept_rank_mode: str = "relative",
    seed: int = 42,
    dpi: int = 180,
    concept_node_style: str = "heatmap",
    concept_thumb_px: int = 112,
    activation_percentile: float = 98.0,
    activation_gamma: float = 0.85,
    activation_colormap: str = "magma",
    tissue_mean_alpha: np.ndarray | None = None,
    equalize_thumb_activation: bool = False,
) -> dict[str, Any]:
    """Arrow diagram: pixel -> weighted concepts -> reconstruction color."""
    out_path = Path(out_path)
    mics = np.asarray(mics_rgb, dtype=np.uint8)
    recon = np.asarray(recon_rgb, dtype=np.uint8)
    alpha = np.asarray(alpha_hwk, dtype=np.float32)
    palette = np.asarray(concept_rgbs, dtype=np.uint8)
    mask = np.asarray(valid_mask, dtype=bool)

    if not example_pixels:
        example_pixels = _pick_example_pixels(mics, mask, n=int(n_examples), seed=int(seed))

    n_rows = len(example_pixels)
    if n_rows == 0:
        logger.warning("No example pixels for mixture figure.")
        return {"mixture_figure_png": None, "examples": []}

    fig_h = 1.55 * n_rows + 0.4
    fig = plt.figure(figsize=(12.5, fig_h), facecolor="#0f1117")
    gs = fig.add_gridspec(n_rows, 1, hspace=0.45)

    records: list[dict[str, Any]] = []
    for i, (y, x, lab) in enumerate(example_pixels):
        ax = fig.add_subplot(gs[i, 0])
        expl = build_color_explanation(
            alpha[y, x],
            palette,
            top_n=int(top_terms),
            min_weight=float(min_weight),
            color_label=f"Pixel {lab}",
            tissue_mean_alpha=tissue_mean_alpha,
            rank_mode=str(concept_rank_mode),
        )
        renormalize = bool(expl.get("renormalize_terms", True))
        _draw_mixture_row(
            ax,
            pixel_rgb=mics[y, x],
            recon_rgb=recon[y, x],
            terms=expl["terms"],
            label=lab,
            yx=(y, x),
            alpha_hwk=alpha,
            valid_mask=mask,
            concept_node_style=str(concept_node_style),
            concept_thumb_px=int(concept_thumb_px),
            activation_percentile=float(activation_percentile),
            activation_gamma=float(activation_gamma),
            activation_colormap=str(activation_colormap),
            renormalize_terms=renormalize,
            equalize_activation=bool(equalize_thumb_activation),
        )
        records.append({"label": lab, "yx": [y, x], "explanation": expl})

    fig.suptitle(
        "Interpretability with concept mixtures",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(out_path, dpi=int(dpi), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote pixel mixture figure: %s (%d examples)", out_path, n_rows)
    return {"mixture_figure_png": str(out_path), "examples": records}


def save_explanation_visualizations(
    mics_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    alpha_hwk: np.ndarray,
    concept_rgbs: np.ndarray,
    valid_mask: np.ndarray,
    out_dir: Path,
    *,
    teacher_emb: np.ndarray | None = None,
    n_concepts: int,
    top_k: int,
    pixel_rgb_mse: float,
    embedding_mse: float,
    n_example_pixels: int = 5,
    dpi: int = 200,
    seed: int = 42,
    concept_node_style: str = "heatmap",
    concept_thumb_px: int = 112,
    grid_offset_px: float = 170.0,
    grid_spacing_px: float | None = None,
    map_fig_width: float = 24.0,
    map_fig_height: float = 22.0,
    map_marker_fontsize: float = 18.0,
    map_marker_size: float = 14.0,
    equalize_map_display: bool = True,
    equalize_thumb_activation: bool = False,
    save_both_display_variants: bool = False,
    example_pixel_spatial_weight: float = 2.5,
    example_pixel_interior_min_px: float = 10.0,
    example_pixel_interior_weight: float = 3.0,
    activation_percentile: float = 98.0,
    activation_gamma: float = 0.85,
    activation_colormap: str = "magma",
    concept_rank_mode: str = "relative",
    min_concept_weight: float = 0.04,
    top_terms: int = 4,
    export_interpretability: bool = True,
    interpretability_subdir: str = "interpretability",
    export_dpi: int = 300,
    export_thumb_px: int = 480,
) -> dict[str, Any]:
    """Side-by-side panel + map overlay arrows + sparsity JSON."""
    out_dir = Path(out_dir)
    alpha = np.asarray(alpha_hwk, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    tissue_mean_alpha = alpha[mask].mean(axis=0) if np.any(mask) else alpha.mean(axis=(0, 1))
    sparsity = compute_sparsity_stats(
        alpha_hwk,
        valid_mask,
        n_concepts=int(n_concepts),
        top_k=int(top_k),
    )
    examples = _pick_example_pixels(
        mics_rgb,
        valid_mask,
        teacher_emb=teacher_emb,
        n=int(n_example_pixels),
        seed=int(seed),
        spatial_weight=float(example_pixel_spatial_weight),
        interior_min_px=float(example_pixel_interior_min_px),
        interior_weight=float(example_pixel_interior_weight),
    )

    side_path = save_original_reconstruction_panel(
        mics_rgb,
        recon_rgb,
        valid_mask,
        out_dir / "original_vs_reconstruction.png",
        sparsity=sparsity,
        pixel_rgb_mse=float(pixel_rgb_mse),
        embedding_mse=float(embedding_mse),
        example_pixels=examples,
        dpi=int(dpi),
    )
    map_overlay_paths = _save_map_concept_overlay_variants(
        mics_rgb,
        alpha_hwk,
        concept_rgbs,
        valid_mask,
        out_dir / "concept_arrows_on_map.png",
        example_pixels=examples,
        tissue_mean_alpha=tissue_mean_alpha,
        concept_rank_mode=str(concept_rank_mode),
        top_terms=int(top_terms),
        min_weight=float(min_concept_weight),
        concept_node_style=str(concept_node_style),
        concept_thumb_px=int(concept_thumb_px),
        grid_offset_px=float(grid_offset_px),
        grid_spacing_px=grid_spacing_px,
        map_fig_width=float(map_fig_width),
        map_fig_height=float(map_fig_height),
        map_marker_fontsize=float(map_marker_fontsize),
        map_marker_size=float(map_marker_size),
        equalize_map_display=bool(equalize_map_display),
        equalize_thumb_activation=bool(equalize_thumb_activation),
        save_both_display_variants=bool(save_both_display_variants),
        activation_percentile=float(activation_percentile),
        activation_gamma=float(activation_gamma),
        activation_colormap=str(activation_colormap),
        dpi=int(dpi),
    )
    mixture_summary = save_pixel_mixture_figure(
        mics_rgb,
        recon_rgb,
        alpha_hwk,
        concept_rgbs,
        valid_mask,
        out_dir / "pixel_concept_mixtures_schematic.png",
        example_pixels=examples,
        n_examples=int(n_example_pixels),
        top_terms=int(top_terms),
        min_weight=float(min_concept_weight),
        concept_rank_mode=str(concept_rank_mode),
        tissue_mean_alpha=tissue_mean_alpha,
        seed=int(seed),
        concept_node_style=str(concept_node_style),
        concept_thumb_px=int(concept_thumb_px),
        activation_percentile=float(activation_percentile),
        activation_gamma=float(activation_gamma),
        activation_colormap=str(activation_colormap),
        equalize_thumb_activation=bool(equalize_thumb_activation),
        save_both_display_variants=bool(save_both_display_variants),
        dpi=int(dpi),
    )
    sparsity_path = out_dir / "sparsity_stats.json"
    sparsity_path.write_text(json.dumps(sparsity, indent=2), encoding="utf-8")

    interpretability_summary: dict[str, Any] | None = None
    if bool(export_interpretability):
        interpretability_summary = save_pixel_interpretability_exports(
            mics_rgb,
            alpha_hwk,
            concept_rgbs,
            valid_mask,
            out_dir,
            example_pixels=examples,
            tissue_mean_alpha=tissue_mean_alpha,
            concept_rank_mode=str(concept_rank_mode),
            top_terms=int(top_terms),
            min_weight=float(min_concept_weight),
            concept_node_style=str(concept_node_style),
            export_thumb_px=int(export_thumb_px),
            export_dpi=int(export_dpi),
            activation_percentile=float(activation_percentile),
            activation_gamma=float(activation_gamma),
            activation_colormap=str(activation_colormap),
            equalize_map_display=bool(equalize_map_display),
            equalize_thumb_activation=bool(equalize_thumb_activation),
            save_both_display_variants=bool(save_both_display_variants),
            map_fig_width=float(map_fig_width),
            map_fig_height=float(map_fig_height),
            map_marker_fontsize=float(map_marker_fontsize),
            map_marker_size=float(map_marker_size),
            subdir=str(interpretability_subdir),
        )

    summary = {
        "original_vs_reconstruction_png": side_path,
        "concept_arrows_on_map_png": map_overlay_paths.get("primary"),
        "concept_arrows_on_map_variants": map_overlay_paths,
        "pixel_concept_mixtures_png": mixture_summary.get("mixture_figure_png"),
        "pixel_concept_mixtures_variants": mixture_summary.get("mixture_figure_variants"),
        "sparsity_stats_json": str(sparsity_path),
        "sparsity": sparsity,
        "mixture_examples": mixture_summary.get("examples", []),
        "example_pixels": [{"label": lab, "y": y, "x": x} for y, x, lab in examples],
    }
    if interpretability_summary is not None:
        summary["interpretability"] = interpretability_summary
    return summary


def _letterbox_rgb(
    rgb: np.ndarray,
    target_h: int,
    target_w: int,
    *,
    bg: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize RGB to fit inside target size with black letterboxing."""
    img = np.asarray(rgb, dtype=np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    ih, iw = img.shape[:2]
    if ih == target_h and iw == target_w:
        return img
    scale = min(float(target_w) / max(float(iw), 1.0), float(target_h) / max(float(ih), 1.0))
    nh = max(1, int(round(ih * scale)))
    nw = max(1, int(round(iw * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(bg, dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def save_interpretability_overview_gallery(
    mics_rgb: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    *,
    concept_map_path: Path | str,
    solace_rgb: np.ndarray,
    cosine_stretched_rgb: np.ndarray,
    equalize_mics: bool = True,
    gap_px: int = 0,
) -> str:
    """Gapless 1x4 mosaic: MiCS | concept overlay | SoLaCE | stretched cosine distance."""
    out_path = Path(out_path)
    mask = np.asarray(valid_mask, dtype=bool)
    mics = np.asarray(mics_rgb, dtype=np.uint8)
    if bool(equalize_mics):
        mics = _equalize_rgb_display(mics, mask)
    else:
        mics = mics.copy()
        mics[~mask] = 0

    concept_path = Path(concept_map_path)
    if not concept_path.is_file():
        raise FileNotFoundError(f"Concept map not found: {concept_path}")

    h, w = mics.shape[:2]
    panels = [
        mics,
        _letterbox_rgb(_load_rgb_image(concept_path), h, w),
        _letterbox_rgb(np.asarray(solace_rgb, dtype=np.uint8), h, w),
        _letterbox_rgb(np.asarray(cosine_stretched_rgb, dtype=np.uint8), h, w),
    ]
    mosaic = _stitch_grid(panels, cols=4, gap_px=int(gap_px))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mosaic).save(out_path)
    logger.info("Wrote interpretability overview gallery: %s", out_path)
    return str(out_path)
