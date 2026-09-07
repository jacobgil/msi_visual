"""Visualize sparse m/z concepts from distilled MiCS."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from msi_visual.extraction import resolve_mass_axis
from msi_visual.mics_concept_distill import (
    SparseConceptDistiller,
    embedding_to_display_rgb,
)

logger = logging.getLogger(__name__)


def _concept_palette_bounds(
    embed_w: np.ndarray,
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Per-channel stretch bounds from all concept readout weights (not full-image stats)."""
    w = np.asarray(embed_w, dtype=np.float64)
    bounds = np.zeros((w.shape[0], 2), dtype=np.float64)
    for ch in range(w.shape[0]):
        bounds[ch, 0] = float(np.percentile(w[ch], float(percentile_low)))
        bounds[ch, 1] = float(np.percentile(w[ch], float(percentile_high)))
        if bounds[ch, 1] <= bounds[ch, 0]:
            bounds[ch, 1] = bounds[ch, 0] + 1e-6
    return bounds


def _all_concept_rgbs(
    student: SparseConceptDistiller,
    *,
    tissue_mask: np.ndarray,
) -> np.ndarray:
    """RGB for every concept using palette bounds shared across concepts."""
    cfg = student.cfg
    w = np.asarray(student.embed_w.detach().cpu().numpy(), dtype=np.float32)
    bounds = _concept_palette_bounds(
        w,
        percentile_low=float(cfg.percentile_low),
        percentile_high=float(cfg.percentile_high),
    )
    rgbs = np.zeros((w.shape[1], 3), dtype=np.uint8)
    for ki in range(w.shape[1]):
        lab = w[:, ki].reshape(1, 1, 3)
        rgbs[ki] = embedding_to_display_rgb(
            lab,
            tissue_mask=np.ones((1, 1), dtype=bool),
            lab_to_rgb=bool(cfg.lab_to_rgb),
            percentile_low=float(cfg.percentile_low),
            percentile_high=float(cfg.percentile_high),
            channel_bounds=bounds,
        )[0, 0]
    return rgbs


def _concept_pure_rgb(
    student: SparseConceptDistiller,
    concept_idx: int,
    *,
    tissue_mask: np.ndarray,
    palette_rgbs: np.ndarray | None = None,
) -> np.ndarray:
    if palette_rgbs is not None:
        return np.asarray(palette_rgbs[int(concept_idx)], dtype=np.uint8)
    return _all_concept_rgbs(student, tissue_mask=tissue_mask)[int(concept_idx)]


def predict_concept_alpha(
    student: SparseConceptDistiller,
    msi: np.ndarray,
    *,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Router weights ``(H, W, K)``; zero outside tissue."""
    spec = np.asarray(msi, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    h, w, m = spec.shape
    student.eval()
    with torch.inference_mode():
        spec_t = torch.from_numpy(spec).to(student.device)
        if str(student.cfg.router_mode) == "spatial_conv":
            flat = spec.reshape(-1, m)
            scores = student._concept_scores_flat(flat).reshape(h, w, -1)
            scores_khw = student._spatial_mix_scores(scores.permute(2, 0, 1))
            scores = scores_khw.permute(1, 2, 0)
            alpha = student._router_alpha_from_scores(scores).cpu().numpy()
        else:
            flat = spec.reshape(-1, m)
            alpha = student._router_alpha(torch.from_numpy(flat).to(student.device)).cpu().numpy()
            alpha = alpha.reshape(h, w, -1)
    alpha = np.asarray(alpha, dtype=np.float32)
    alpha[~mask] = 0.0
    return alpha


def _rank_concepts_by_activation(alpha: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    tissue = alpha[mask]
    mean_act = tissue.mean(axis=0) if tissue.size else alpha.mean(axis=(0, 1))
    order = np.argsort(-mean_act)
    return order[: max(1, min(int(k), order.size))]


def _spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _mz_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _select_diverse_concepts(
    alpha: np.ndarray,
    mask: np.ndarray,
    concepts: np.ndarray,
    palette_rgbs: np.ndarray,
    k: int,
    *,
    min_color_dist: float = 35.0,
    max_spatial_corr: float = 0.88,
    max_mz_cosine: float = 0.92,
) -> np.ndarray:
    """Greedy top-activation pick with color / spatial / m/z deduplication."""
    tissue = alpha[mask]
    mean_act = tissue.mean(axis=0) if tissue.size else alpha.mean(axis=(0, 1))
    order = np.argsort(-mean_act)
    selected: list[int] = []

    for ki in order:
        ki = int(ki)
        if len(selected) >= int(k):
            break
        duplicate = False
        act_i = tissue[:, ki] if tissue.size else alpha[:, :, ki].ravel()
        for sj in selected:
            color_dist = float(np.linalg.norm(palette_rgbs[ki].astype(np.float32) - palette_rgbs[sj].astype(np.float32)))
            if color_dist < float(min_color_dist):
                duplicate = True
                break
            if _spatial_corr(act_i, tissue[:, sj] if tissue.size else alpha[:, :, sj].ravel()) > float(max_spatial_corr):
                duplicate = True
                break
            if _mz_cosine(concepts[ki], concepts[sj]) > float(max_mz_cosine):
                duplicate = True
                break
        if not duplicate:
            selected.append(ki)

    if len(selected) < int(k):
        for ki in order:
            ki = int(ki)
            if ki in selected:
                continue
            selected.append(ki)
            if len(selected) >= int(k):
                break
    return np.asarray(selected[: int(k)], dtype=np.int64)


def _equalize_activation(act: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Histogram-equalize activation on tissue pixels to [0, 1] (before colormap)."""
    act = np.asarray(act, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(act, dtype=np.float32)
    if not np.any(mask):
        return out
    vals = act[mask].astype(np.float64)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo:
        out[mask] = 0.0
        return out
    hist, bin_edges = np.histogram(vals, bins=256, range=(lo, hi))
    cdf = hist.cumsum().astype(np.float64)
    if cdf[-1] <= 0.0:
        out[mask] = np.clip((vals - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
        return out
    cdf /= cdf[-1]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    eq = np.interp(vals, bin_centers, cdf)
    out[mask] = np.clip(eq, 0.0, 1.0).astype(np.float32)
    return out


def _normalize_activation(
    act: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float = 98.0,
    gamma: float = 0.85,
    vmax: float | None = None,
) -> np.ndarray:
    act = np.asarray(act, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if vmax is None:
        vmax = float(np.percentile(act[mask], percentile)) if np.any(mask) else float(act.max())
    vmax = max(float(vmax), 1e-6)
    norm = np.clip(act / vmax, 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, gamma)
    return norm


def _overlay_dashed_tissue_border(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    dash_on: int | None = None,
    dash_off: int | None = None,
    thickness: int = 1,
) -> np.ndarray:
    """Paint a dashed border along the tissue mask outline (white by default)."""
    import cv2

    out = np.asarray(rgb, dtype=np.uint8).copy()
    m = np.asarray(mask, dtype=np.uint8)
    if m.ndim != 2 or m.shape[:2] != out.shape[:2] or not np.any(m):
        return out

    diag = float(np.hypot(float(m.shape[0]), float(m.shape[1])))
    on = int(dash_on) if dash_on is not None else max(3, int(round(diag / 90.0)))
    off = int(dash_off) if dash_off is not None else max(2, int(round(on * 0.65)))
    th = max(1, int(thickness))
    col = (int(color[0]), int(color[1]), int(color[2]))

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if pts.shape[0] < 2:
            continue
        drawing = True
        seg_start = 0
        acc = 0.0
        for i in range(1, len(pts)):
            dx = float(pts[i, 0] - pts[i - 1, 0])
            dy = float(pts[i, 1] - pts[i - 1, 1])
            acc += float(np.hypot(dx, dy))
            limit = float(on if drawing else off)
            if acc >= limit:
                if drawing and i > seg_start:
                    cv2.polylines(
                        out,
                        [pts[seg_start : i + 1]],
                        isClosed=False,
                        color=col,
                        thickness=th,
                        lineType=cv2.LINE_AA,
                    )
                seg_start = i
                acc = 0.0
                drawing = not drawing
        if drawing and seg_start < len(pts) - 1:
            cv2.polylines(
                out,
                [pts[seg_start:]],
                isClosed=False,
                color=col,
                thickness=th,
                lineType=cv2.LINE_AA,
            )
    return out


def _activation_rgb(
    act: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float = 98.0,
    gamma: float = 0.85,
    colormap: str = "assigned",
    concept_rgb: np.ndarray | None = None,
    vmax: float | None = None,
    equalize_activation: bool = False,
    tissue_border: bool = True,
) -> np.ndarray:
    """Pixel-sharp activation map.

    ``assigned``: black → concept readout color (links to distilled MiCS).
    ``heat`` / ``magma``: shared heatmap (easier cross-panel strength comparison).
    With ``tissue_border=True`` (default), draw a dashed white outline of the tissue mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if bool(equalize_activation):
        norm = _equalize_activation(act, mask)
    else:
        norm = _normalize_activation(act, mask, percentile=percentile, gamma=gamma, vmax=vmax)
    if bool(equalize_activation) and gamma != 1.0:
        norm = np.power(np.clip(norm, 0.0, 1.0), float(gamma))
    mode = str(colormap).lower()
    if mode in {"heat", "magma", "heatmap"}:
        rgb = (plt.cm.magma(norm)[:, :, :3] * 255.0).astype(np.uint8)
    elif mode in {"assigned", "concept", "readout"}:
        if concept_rgb is None:
            raise ValueError("concept_rgb required for assigned activation colormap")
        tint = np.asarray(concept_rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
        rgb = np.clip(norm[..., None] * tint * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown activation colormap: {colormap!r}")
    out = np.zeros_like(rgb)
    out[mask] = rgb[mask]
    if bool(tissue_border):
        out = _overlay_dashed_tissue_border(out, mask)
    return out


def _stitch_grid(images: list[np.ndarray], *, cols: int, gap_px: int = 0) -> np.ndarray:
    if not images:
        raise ValueError("No images to stitch.")
    gap = max(0, int(gap_px))
    ncols = max(1, int(cols))
    nrows = int(np.ceil(len(images) / ncols))
    h, w = images[0].shape[:2]
    ch = images[0].shape[2] if images[0].ndim == 3 else 1
    canvas = np.zeros((nrows * h + max(0, nrows - 1) * gap, ncols * w + max(0, ncols - 1) * gap, ch), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        y0 = r * (h + gap)
        x0 = c * (w + gap)
        canvas[y0 : y0 + h, x0 : x0 + w] = img
    if canvas.shape[-1] == 1:
        canvas = np.repeat(canvas, 3, axis=-1)
    return canvas


def save_concept_gallery(
    student: SparseConceptDistiller,
    msi: np.ndarray,
    *,
    valid_mask: np.ndarray,
    out_dir: Path,
    npy_path: Path | None = None,
    alpha_hwk: np.ndarray | None = None,
    max_concepts: int = 16,
    spectrum_top_peaks: int = 25,
    dpi: int = 200,
    cols: int = 4,
    gap_px: int = 1,
    activation_percentile: float = 98.0,
    activation_gamma: float = 0.85,
    activation_colormap: str = "assigned",
    dedupe: bool = True,
    dedupe_min_color_dist: float = 35.0,
    dedupe_max_spatial_corr: float = 0.88,
    dedupe_max_mz_cosine: float = 0.92,
) -> dict[str, Any]:
    """Write concept activation maps, spectra, colors, and summary mosaic."""
    out_dir = Path(out_dir)
    cells_dir = out_dir / "concepts"
    cells_dir.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(valid_mask, dtype=bool)
    if alpha_hwk is not None:
        alpha = np.asarray(alpha_hwk, dtype=np.float32).copy()
        alpha[~mask] = 0.0
    else:
        alpha = predict_concept_alpha(student, msi, valid_mask=valid_mask)
    concepts = np.asarray(student.concepts.detach().cpu().numpy(), dtype=np.float32)
    n_mz = concepts.shape[1]
    mz_axis = resolve_mass_axis(npy_path, n_mz) if npy_path is not None else np.arange(n_mz, dtype=np.float64)

    palette_rgbs = _all_concept_rgbs(student, tissue_mask=mask)
    if bool(dedupe):
        ranked = _select_diverse_concepts(
            alpha,
            mask,
            concepts,
            palette_rgbs,
            max_concepts,
            min_color_dist=float(dedupe_min_color_dist),
            max_spatial_corr=float(dedupe_max_spatial_corr),
            max_mz_cosine=float(dedupe_max_mz_cosine),
        )
    else:
        ranked = _rank_concepts_by_activation(alpha, mask, max_concepts)
    tissue = alpha[mask]
    mean_act = {int(i): float(tissue[:, i].mean()) for i in ranked} if tissue.size else {}

    saved: list[dict[str, Any]] = []
    mosaic_cells: list[np.ndarray] = []
    ncols = max(1, int(cols))

    for ki in ranked:
        ki = int(ki)
        act = alpha[:, :, ki]
        color_rgb = _concept_pure_rgb(student, ki, tissue_mask=mask, palette_rgbs=palette_rgbs)
        act_u8 = _activation_rgb(
            act,
            mask,
            percentile=float(activation_percentile),
            gamma=float(activation_gamma),
            colormap=str(activation_colormap),
            concept_rgb=color_rgb,
        )
        act_path = cells_dir / f"concept_{ki:03d}_activation.png"
        Image.fromarray(act_u8).save(act_path)

        row = concepts[ki]
        peak_idx = np.argsort(-row)[: max(1, int(spectrum_top_peaks))]
        spec_path = cells_dir / f"concept_{ki:03d}_spectrum.png"
        fig_s, ax_s = plt.subplots(figsize=(3.0, 1.6), facecolor="black")
        ax_s.set_facecolor("black")
        ax_s.vlines(mz_axis[peak_idx], 0.0, row[peak_idx], color="#7ec8ff", linewidth=1.2)
        ax_s.set_xlim(float(mz_axis.min()), float(mz_axis.max()))
        ax_s.set_ylim(0.0, float(row[peak_idx].max()) * 1.08 + 1e-6)
        ax_s.set_title(f"concept {ki}", color="white", fontsize=9)
        ax_s.tick_params(colors="white", labelsize=7)
        for spine in ax_s.spines.values():
            spine.set_color("#555555")
        ax_s.set_xlabel("m/z (or bin)", color="white", fontsize=7)
        fig_s.tight_layout()
        fig_s.savefig(spec_path, dpi=dpi, facecolor="black", edgecolor="none")
        plt.close(fig_s)

        color_path = cells_dir / f"concept_{ki:03d}_color.png"
        Image.fromarray(np.broadcast_to(color_rgb, (32, 32, 3)).astype(np.uint8)).save(color_path)

        mosaic_cells.append(act_u8)

        saved.append(
            {
                "concept": ki,
                "mean_alpha": mean_act.get(ki, 0.0),
                "activation_png": str(act_path),
                "spectrum_png": str(spec_path),
                "color_rgb": [int(x) for x in color_rgb.tolist()],
                "top_mz_indices": [int(i) for i in peak_idx.tolist()],
                "top_mz_values": [float(mz_axis[i]) for i in peak_idx],
            }
        )

    nrows = int(np.ceil(len(ranked) / ncols))
    while len(mosaic_cells) < nrows * ncols:
        h, w = mosaic_cells[0].shape[:2]
        mosaic_cells.append(np.zeros((h, w, 3), dtype=np.uint8))

    mosaic_rgb = _stitch_grid(mosaic_cells[: nrows * ncols], cols=ncols, gap_px=gap_px)
    mosaic_path = out_dir / "gallery_concepts.png"
    Image.fromarray(mosaic_rgb).save(mosaic_path)

    summary = {
        "gallery_concepts_png": str(mosaic_path),
        "concepts_dir": str(cells_dir),
        "n_concepts_shown": len(saved),
        "activation_colormap": str(activation_colormap),
        "dedupe": bool(dedupe),
        "ranked_concepts": saved,
    }
    (out_dir / "concepts_viz.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote concept gallery: %s (%d concepts)", mosaic_path, len(saved))
    return summary
