#!/usr/bin/env python3
"""
Compare soft edge methods against VIA region-boundary annotations.

Primary metrics (robust to incomplete annotations when interpreted relatively):
  - boundary_enrichment = mean(edge|boundary) / mean(edge|annotated interior)
  - surrounding_enrichment = mean(edge|boundary) / mean(edge|local exterior ring)
  - average_precision   = AP ranking boundary vs annotated-interior pixels
  - top_pct_f1          = F1 after selecting top-p% edge pixels on tissue

Surrounding enrichment asks whether borders stand out from neighboring tissue
(appropriate for plaques and, in general, for all region classes).

Run:
  python scripts/annotation_edge_metrics.py
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from solace_annotation_gallery import (  # noqa: E402
    _find_cached_solace,
    _opt_path,
    _resolve_slide_rows,
)
from view_annotation_boundaries import (  # noqa: E402
    _category_ignored,
    _category_string,
    _region_main_cat1,
    _region_to_polygon,
    apply_annotation_spatial_transform,
    resolve_transpose_spatial_mode,
    spatial_hw_after_transpose_mode,
    spatial_hw_from_npy,
)

logger = logging.getLogger(__name__)

METHODS_DEFAULT = (
    "soft_landmark_contrast",
    "distance_multi",
    "tic_sobel",
)


def _rank_normalize_on_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(x.shape, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    vals = np.asarray(x, dtype=np.float64)[m]
    n = int(vals.size)
    if n == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64) / float(n)
    out[m] = ranks
    return out


def _load_polygons_for_stem(
    annotation_json: Path,
    stem: str,
    *,
    ignore: list[str],
    keep: set[str] | None,
    category_keys: tuple[str, ...],
) -> tuple[list[np.ndarray], list[str], list[str], int]:
    """
    Load regions from all VIA keys for ``stem``, then caller should dedupe.

    Multiple keys are visualization copies of the same slide; duplicates are removed
    by coordinate hashing so boundaries are not artificially thickened.

    Returns (polygons, full_categories, main_cat1_labels, n_via_keys).
    """
    import json

    with annotation_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = raw.get("_via_img_metadata", raw)
    keys = [k for k in meta if str(k).startswith(f"{stem}_")]
    polygons: list[np.ndarray] = []
    categories: list[str] = []
    main_cats: list[str] = []
    for via_key in keys:
        for r in meta[via_key].get("regions") or []:
            cat = _category_string(r, category_keys)
            main = _region_main_cat1(r)
            if _category_ignored(cat, ignore, main_cat1=main):
                continue
            if keep is not None and cat not in keep:
                continue
            poly = _region_to_polygon(r)
            if poly is None or len(poly) < 3:
                continue
            polygons.append(poly.astype(np.float32))
            categories.append(cat)
            main_cats.append(main)
    return polygons, categories, main_cats, len(keys)


def _dedupe_polygons(
    polygons: list[np.ndarray],
    categories: list[str],
    main_cats: list[str] | None = None,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Drop exact duplicate polygons (same rounded coordinates)."""
    seen: set[bytes] = set()
    out_p: list[np.ndarray] = []
    out_c: list[str] = []
    out_m: list[str] = []
    mains = main_cats if main_cats is not None else [""] * len(polygons)
    for poly, cat, main in zip(polygons, categories, mains):
        p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        key = np.round(p, 1).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out_p.append(poly)
        out_c.append(cat)
        out_m.append(main)
    return out_p, out_c, out_m


def _rasterize_annotation_masks(
    polygons: list[np.ndarray],
    shape_hw: tuple[int, int],
    *,
    boundary_dilate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (boundary, interior, annotated_union) boolean masks.

    Interior = filled polygons minus boundary. Negatives for AP use interior only
    (safer under incomplete annotations than all non-boundary tissue).
    """
    h, w = int(shape_hw[0]), int(shape_hw[1])
    filled = np.zeros((h, w), dtype=np.uint8)
    boundary = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        p = np.int32(np.round(np.asarray(poly).reshape(-1, 2)))
        if p.shape[0] < 2:
            continue
        cv2.fillPoly(filled, [p], 255)
        cv2.polylines(boundary, [p], isClosed=True, color=255, thickness=1, lineType=cv2.LINE_8)

    dil = max(0, int(boundary_dilate))
    if dil > 0:
        k = 2 * dil + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        boundary = cv2.dilate(boundary, kernel, iterations=1)

    boundary_b = boundary.astype(bool)
    filled_b = filled.astype(bool)
    if dil > 0:
        expand = cv2.dilate(filled, kernel, iterations=1).astype(bool)
        boundary_b &= expand
    else:
        # Undilated stroke pixels may sit on the polygon rim; keep those on/near fill.
        boundary_b &= filled_b | boundary_b

    interior = filled_b & ~boundary_b
    annotated = filled_b | boundary_b
    return boundary_b, interior, annotated


def _enrichment_only(
    scores: np.ndarray,
    *,
    boundary: np.ndarray,
    interior: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (enrichment, AP, n_boundary, n_negatives)."""
    boundary = np.asarray(boundary, dtype=bool)
    interior = np.asarray(interior, dtype=bool)
    b_vals = scores[boundary]
    i_vals = scores[interior]
    mean_b = float(np.mean(b_vals)) if b_vals.size else float("nan")
    mean_i = float(np.mean(i_vals)) if i_vals.size else float("nan")
    enrichment = mean_b / mean_i if (np.isfinite(mean_i) and mean_i > 1e-12) else float("nan")
    eval_m = boundary | interior
    y = boundary[eval_m].astype(np.int32)
    s = scores[eval_m]
    if y.size == 0 or int(y.sum()) == 0 or int(y.sum()) == int(y.size):
        ap = float("nan")
    else:
        ap = float(average_precision_score(y, s))
    return enrichment, ap, float(boundary.sum()), float(interior.sum())


def _filled_mask(polygons: list[np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    filled = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        p = np.int32(np.round(np.asarray(poly).reshape(-1, 2)))
        if p.shape[0] >= 2:
            cv2.fillPoly(filled, [p], 255)
    return filled


def _surrounding_ring_from_filled(
    filled_u8: np.ndarray,
    *,
    ring_width: int,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Exterior band around filled polygons (does not include the fill itself)."""
    rw = max(1, int(ring_width))
    k = 2 * rw + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(filled_u8, kernel, iterations=1)
    ring = dilated.astype(bool) & ~filled_u8.astype(bool)
    if exclude is not None:
        ring &= ~np.asarray(exclude, dtype=bool)
    return ring


def _rasterize_category_masks(
    polygons: list[np.ndarray],
    shape_hw: tuple[int, int],
    *,
    boundary_dilate: int,
    min_negatives: int = 8,
    ring_width: int = 3,
    negative_mode: str = "surrounding",
) -> tuple[np.ndarray, np.ndarray, str, int] | None:
    """
    Build (boundary, negatives) for one anatomical category.

    ``negative_mode``:
      - ``surrounding``: exterior ring around the filled ROIs (vs neighborhood)
      - ``interior``: filled ROI minus boundary (vs own interior)
      - ``auto``: interior when thick enough, else surrounding ring
    """
    mode = str(negative_mode).strip().lower()
    if mode in ("ring", "exterior", "neighborhood", "surround"):
        mode = "surrounding"

    filled = _filled_mask(polygons, shape_hw)
    if int(filled.sum()) == 0:
        return None

    def _boundary(dil: int) -> np.ndarray:
        b, _, _ = _rasterize_annotation_masks(polygons, shape_hw, boundary_dilate=dil)
        return b

    if mode == "surrounding":
        # Thin stroke + exterior neighborhood (plaques and large regions alike).
        b = _boundary(0 if int(boundary_dilate) <= 0 else min(1, int(boundary_dilate)))
        # Prefer dilate=0 stroke so the ring is truly outside the ROI.
        b0 = _boundary(0)
        if int(b0.sum()) > 0:
            b = b0
        ring = _surrounding_ring_from_filled(filled, ring_width=ring_width, exclude=b)
        if int(b.sum()) == 0 or int(ring.sum()) < min_negatives:
            return None
        return b, ring, "surrounding", 0

    # interior / auto
    dil_candidates: list[int] = []
    for d in (int(boundary_dilate), 0):
        if d not in dil_candidates:
            dil_candidates.append(d)

    best: tuple[np.ndarray, np.ndarray, str, int] | None = None
    for dil in dil_candidates:
        b, i, _ = _rasterize_annotation_masks(polygons, shape_hw, boundary_dilate=dil)
        if int(b.sum()) == 0:
            continue
        if int(i.sum()) >= min_negatives:
            if mode == "interior":
                return b, i, "interior", dil
            return b, i, "interior", dil
        if best is None or int(i.sum()) > int(best[1].sum()):
            best = (b, i, "interior", dil)

    if mode == "interior":
        return best if best is not None and int(best[1].sum()) >= 1 else None

    # auto fallback -> surrounding
    b0 = _boundary(0)
    ring = _surrounding_ring_from_filled(filled, ring_width=ring_width, exclude=b0)
    if int(b0.sum()) == 0 or int(ring.sum()) < min_negatives:
        return best
    return b0, ring, "surrounding", 0


_MAIN_CAT_DISPLAY = {
    "WM": "WM",
    "CTX": "CTX",
    "BST": "BST",
    "CNU": "CNU",
    "PLQ": "PLQ (plaques)",
    "PLX": "PLX",
    "BG": "BG",
}


def _load_edge_gray_aligned(
    npy: Path,
    *,
    raw_root: Path,
    method_key: str,
    spatial_mode: str,
) -> np.ndarray:
    png = _find_cached_solace(raw_root, npy, method_key)
    if png is None:
        raise FileNotFoundError(f"Missing cached edge map {method_key} for {npy} under {raw_root}")
    gray = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
    if spatial_mode not in ("none", "off", "identity", ""):
        gray = apply_annotation_spatial_transform(gray, spatial_mode)
    file_hw = spatial_hw_from_npy(npy)
    if file_hw is None:
        raise ValueError(f"Cannot read shape from {npy}")
    target_hw = spatial_hw_after_transpose_mode(file_hw, spatial_mode)
    if gray.shape[:2] != target_hw:
        gray = cv2.resize(gray, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return gray


def _topk_mask(scores: np.ndarray, tissue: np.ndarray, top_pct: float) -> np.ndarray:
    m = np.asarray(tissue, dtype=bool)
    vals = np.asarray(scores, dtype=np.float64)[m]
    n = int(vals.size)
    out = np.zeros_like(m, dtype=bool)
    if n == 0:
        return out
    pct = float(np.clip(top_pct, 0.1, 100.0))
    k = max(1, int(np.ceil(n * pct / 100.0)))
    if k >= n:
        out[m] = True
        return out
    thr = np.partition(vals, n - k)[n - k]
    cand = m & (np.asarray(scores, dtype=np.float64) >= thr)
    # Cap ties to exactly top-k
    if int(cand.sum()) > k:
        idx = np.flatnonzero(m)
        order = np.argsort(-np.asarray(scores, dtype=np.float64)[m], kind="mergesort")
        keep = np.zeros(n, dtype=bool)
        keep[order[:k]] = True
        out = np.zeros_like(m, dtype=bool)
        out.flat[idx[keep]] = True
        return out
    return cand


def _binary_prf(pred: np.ndarray, gt_pos: np.ndarray, eval_mask: np.ndarray) -> tuple[float, float, float]:
    e = np.asarray(eval_mask, dtype=bool)
    p = np.asarray(pred, dtype=bool) & e
    g = np.asarray(gt_pos, dtype=bool) & e
    tp = float(np.sum(p & g))
    fp = float(np.sum(p & ~g))
    fn = float(np.sum(~p & g))
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if not np.isfinite(prec) or not np.isfinite(rec) or (prec + rec) <= 0:
        f1 = float("nan")
    else:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1


def _metrics_for_edge(
    edge_u8: np.ndarray,
    *,
    boundary: np.ndarray,
    interior: np.ndarray,
    tissue: np.ndarray,
    rank_normalize: bool,
    top_pcts: list[float],
    surrounding: np.ndarray | None = None,
) -> dict[str, float]:
    tissue = np.asarray(tissue, dtype=bool)
    boundary = np.asarray(boundary, dtype=bool) & tissue
    interior = np.asarray(interior, dtype=bool) & tissue
    scores = np.asarray(edge_u8, dtype=np.float64)
    if rank_normalize:
        scores = _rank_normalize_on_mask(scores, tissue)

    b_vals = scores[boundary]
    i_vals = scores[interior]
    mean_b = float(np.mean(b_vals)) if b_vals.size else float("nan")
    mean_i = float(np.mean(i_vals)) if i_vals.size else float("nan")
    enrichment = mean_b / mean_i if (np.isfinite(mean_i) and mean_i > 1e-12) else float("nan")

    # AP / ROC on annotated pixels only: boundary = positive, interior = negative
    eval_m = boundary | interior
    y = boundary[eval_m].astype(np.int32)
    s = scores[eval_m]
    if y.size == 0 or int(y.sum()) == 0 or int(y.sum()) == int(y.size):
        ap = float("nan")
        auc = float("nan")
    else:
        ap = float(average_precision_score(y, s))
        try:
            auc = float(roc_auc_score(y, s))
        except ValueError:
            auc = float("nan")

    annotated = boundary | interior
    out: dict[str, float] = {
        "mean_boundary": mean_b,
        "mean_interior": mean_i,
        "boundary_enrichment": enrichment,
        "average_precision": ap,
        "roc_auc": auc,
        "n_boundary": float(boundary.sum()),
        "n_interior": float(interior.sum()),
        "n_tissue": float(tissue.sum()),
    }

    if surrounding is not None:
        surr = np.asarray(surrounding, dtype=bool) & tissue & ~boundary
        surr_enr, surr_ap, _, n_surr = _enrichment_only(scores, boundary=boundary, interior=surr)
        out["surrounding_enrichment"] = surr_enr
        out["surrounding_ap"] = surr_ap
        out["n_surrounding"] = n_surr
        surr_vals = scores[surr]
        out["mean_surrounding"] = float(np.mean(surr_vals)) if surr_vals.size else float("nan")
    else:
        out["surrounding_enrichment"] = float("nan")
        out["surrounding_ap"] = float("nan")
        out["n_surrounding"] = 0.0
        out["mean_surrounding"] = float("nan")

    f1s: list[float] = []
    for pct in top_pcts:
        pred = _topk_mask(scores, annotated, float(pct))
        prec, rec, f1 = _binary_prf(pred, boundary, annotated)
        tag = _pct_tag(pct)
        out[f"f1_at_{tag}"] = f1
        out[f"precision_at_{tag}"] = prec
        out[f"recall_at_{tag}"] = rec
        if np.isfinite(f1):
            f1s.append(float(f1))
    out["f1_mean_over_pcts"] = float(np.mean(f1s)) if f1s else float("nan")
    # Backward-compatible aliases at first / 10% if present
    if 10.0 in top_pcts or 10 in top_pcts:
        out["top_pct_f1"] = out.get("f1_at_10", out["f1_mean_over_pcts"])
        out["top_pct_precision"] = out.get("precision_at_10", float("nan"))
        out["top_pct_recall"] = out.get("recall_at_10", float("nan"))
    else:
        out["top_pct_f1"] = out["f1_mean_over_pcts"]
        out["top_pct_precision"] = float("nan")
        out["top_pct_recall"] = float("nan")
    return out


def _pct_tag(pct: float) -> str:
    x = float(pct)
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:g}".replace(".", "p")


def _resolve_top_pcts(met_cfg: Any) -> list[float]:
    raw = getattr(met_cfg, "top_pcts", None)
    if raw is not None:
        vals = OmegaConf.to_container(raw, resolve=True)
        if isinstance(vals, (list, tuple)) and vals:
            return [float(x) for x in vals]
    return [float(getattr(met_cfg, "top_pct", 10.0))]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _aggregate_table(
    rows: list[dict[str, Any]],
    methods: list[str],
    labels: dict[str, str],
    top_pcts: list[float],
) -> list[dict[str, Any]]:
    keys = [
        "boundary_enrichment",
        "surrounding_enrichment",
        "average_precision",
        "roc_auc",
        "f1_mean_over_pcts",
    ]
    for pct in top_pcts:
        keys.append(f"f1_at_{_pct_tag(pct)}")

    out: list[dict[str, Any]] = []
    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        if not sub:
            continue
        row: dict[str, Any] = {
            "method": method,
            "method_label": labels.get(method, method),
            "n_slides": len(sub),
        }
        for k in keys:
            vals = np.asarray([float(r[k]) for r in sub], dtype=np.float64)
            row[f"{k}_mean"] = float(np.nanmean(vals))
            row[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
        out.append(row)
    return out


def _write_markdown_tables(
    per_slide: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    path: Path,
    *,
    top_pcts: list[float],
    by_cat_summary: list[dict[str, Any]] | None = None,
) -> None:
    pct_txt = ", ".join(f"{p:g}%" for p in top_pcts)
    lines: list[str] = []
    lines.append("# Annotation-boundary edge metrics")
    lines.append("")
    lines.append(
        "Evaluated on VIA region borders (incomplete labels; **BG / background ignored**). "
        "**Enrichment (interior)** = boundary vs annotated interior; "
        "**Enrichment (surround.)** = boundary vs local exterior ring. "
        "Prefer enrichment and **AP** for relative method comparison. "
        f"Density-matched F1 at {{{pct_txt}}} inside the annotated ROI; "
        "**mean F1** averages those percentiles."
    )
    lines.append("")
    lines.append("## Mean over slides")
    lines.append("")
    headers = ["Method", "Enrich↑", "Surround↑", "AP ↑", "ROC-AUC ↑", "Mean F1 ↑"]
    headers.extend([f"F1@{p:g}%" for p in top_pcts])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in summary:
        cells = [
            str(r["method_label"]),
            f"{r['boundary_enrichment_mean']:.3f} ± {r['boundary_enrichment_std']:.3f}",
            f"{r['surrounding_enrichment_mean']:.3f} ± {r['surrounding_enrichment_std']:.3f}",
            f"{r['average_precision_mean']:.3f} ± {r['average_precision_std']:.3f}",
            f"{r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}",
            f"{r['f1_mean_over_pcts_mean']:.3f} ± {r['f1_mean_over_pcts_std']:.3f}",
        ]
        for pct in top_pcts:
            k = f"f1_at_{_pct_tag(pct)}"
            cells.append(f"{r[f'{k}_mean']:.3f} ± {r[f'{k}_std']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per slide (mean F1 over percentiles)")
    lines.append("")
    headers2 = ["Slide", "Hemi", "Method", "Enrich", "Surround", "AP", "ROC-AUC", "Mean F1", "n_boundary"]
    lines.append("| " + " | ".join(headers2) + " |")
    lines.append("| " + " | ".join("---" for _ in headers2) + " |")
    for r in per_slide:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["slide"]),
                    str(r.get("hemisphere", "")),
                    str(r["method_label"]),
                    f"{float(r['boundary_enrichment']):.3f}",
                    f"{float(r['surrounding_enrichment']):.3f}",
                    f"{float(r['average_precision']):.3f}",
                    f"{float(r['roc_auc']):.3f}",
                    f"{float(r['f1_mean_over_pcts']):.3f}",
                    str(int(r["n_boundary"])),
                ]
            )
            + " |"
        )
    lines.append("")
    if by_cat_summary:
        lines.append("## Enrichment vs surrounding by anatomical category (`main_cat1`)")
        lines.append("")
        lines.append(
            "Boundary vs local exterior ring (same definition for all categories, including plaques). "
            "Mean ± SD over slides with enough boundary / surrounding pixels."
        )
        lines.append("")
        headers3 = ["Category", "Method", "Enrichment ↑", "AP ↑", "n_slides", "mean n_boundary", "negatives"]
        lines.append("| " + " | ".join(headers3) + " |")
        lines.append("| " + " | ".join("---" for _ in headers3) + " |")
        for r in by_cat_summary:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _MAIN_CAT_DISPLAY.get(str(r["main_cat1"]), str(r["main_cat1"])),
                        str(r["method_label"]),
                        f"{r['boundary_enrichment_mean']:.3f} ± {r['boundary_enrichment_std']:.3f}",
                        f"{r['average_precision_mean']:.3f} ± {r['average_precision_std']:.3f}",
                        str(int(r["n_slides"])),
                        f"{r['n_boundary_mean']:.0f}",
                        str(r.get("negative_mode", "interior")),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate_by_category(
    by_cat_rows: list[dict[str, Any]],
    methods: list[str],
    labels: dict[str, str],
    *,
    min_slides: int = 1,
) -> list[dict[str, Any]]:
    cats = sorted({str(r["main_cat1"]) for r in by_cat_rows})
    out: list[dict[str, Any]] = []
    for cat in cats:
        for method in methods:
            sub = [r for r in by_cat_rows if r["main_cat1"] == cat and r["method"] == method]
            if len(sub) < min_slides:
                continue
            enr = np.asarray([float(r["boundary_enrichment"]) for r in sub], dtype=np.float64)
            ap = np.asarray([float(r["average_precision"]) for r in sub], dtype=np.float64)
            nb = np.asarray([float(r["n_boundary"]) for r in sub], dtype=np.float64)
            if not np.isfinite(enr).any():
                continue
            modes = [str(r.get("negative_mode", "interior")) for r in sub]
            mode = max(set(modes), key=modes.count)
            out.append(
                {
                    "main_cat1": cat,
                    "method": method,
                    "method_label": labels.get(method, method),
                    "n_slides": int(np.isfinite(enr).sum()),
                    "boundary_enrichment_mean": float(np.nanmean(enr)),
                    "boundary_enrichment_std": float(np.nanstd(enr[np.isfinite(enr)], ddof=1))
                    if int(np.isfinite(enr).sum()) > 1
                    else 0.0,
                    "average_precision_mean": float(np.nanmean(ap)),
                    "average_precision_std": float(np.nanstd(ap[np.isfinite(ap)], ddof=1))
                    if int(np.isfinite(ap).sum()) > 1
                    else 0.0,
                    "n_boundary_mean": float(np.nanmean(nb)),
                    "negative_mode": mode,
                }
            )
    return out


def _write_latex_summary(summary: list[dict[str, Any]], path: Path, *, top_pcts: list[float]) -> None:
    pct_txt = "/".join(f"{p:g}" for p in top_pcts)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Edge methods vs VIA region boundaries (NRL4485; BG ignored; mean$\pm$SD over sections). "
            r"Enrich = boundary vs annotated interior; Surround = boundary vs local exterior ring. "
            f"Mean F1 averages density-matched F1 at {{{pct_txt}}}"
            + r"\% inside the annotated region union.}"
        ),
        r"\label{tab:annotation_edge_metrics}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Enrich $\uparrow$ & Surround $\uparrow$ & AP $\uparrow$ & ROC-AUC $\uparrow$ & Mean F1 $\uparrow$ \\",
        r"\midrule",
    ]
    for r in summary:
        lines.append(
            " & ".join(
                [
                    str(r["method_label"]),
                    f"{r['boundary_enrichment_mean']:.3f}$\\pm${r['boundary_enrichment_std']:.3f}",
                    f"{r['surrounding_enrichment_mean']:.3f}$\\pm${r['surrounding_enrichment_std']:.3f}",
                    f"{r['average_precision_mean']:.3f}$\\pm${r['average_precision_std']:.3f}",
                    f"{r['roc_auc_mean']:.3f}$\\pm${r['roc_auc_std']:.3f}",
                    f"{r['f1_mean_over_pcts_mean']:.3f}$\\pm${r['f1_mean_over_pcts_std']:.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


_METHOD_COLORS = {
    "soft_landmark_contrast": "#1B4965",
    "distance_multi": "#BC6C25",
    "tic_sobel": "#2A9D8F",
}


def _save_paper_figure(
    per_slide: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    methods: list[str],
    labels: dict[str, str],
    top_pcts: list[float],
    out_dir: Path,
    *,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#D0D5DD",
            "legend.framealpha": 0.95,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.05, 1.15]})
    ax_f1, ax_sum = axes

    # Left: F1 vs top-% with mean±SD across slides
    x = np.asarray(top_pcts, dtype=np.float64)
    for method in methods:
        means, stds = [], []
        for pct in top_pcts:
            key = f"f1_at_{_pct_tag(pct)}"
            vals = np.asarray(
                [float(r[key]) for r in per_slide if r["method"] == method],
                dtype=np.float64,
            )
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0)
        color = _METHOD_COLORS.get(method, "#444444")
        ax_f1.plot(
            x,
            means,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=color,
            label=labels.get(method, method),
        )
        ax_f1.fill_between(
            x,
            np.asarray(means) - np.asarray(stds),
            np.asarray(means) + np.asarray(stds),
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    ax_f1.set_xlabel("Top %")
    ax_f1.set_ylabel("F1")
    ax_f1.set_title("Boundary F1")
    ax_f1.set_xticks(list(top_pcts))
    ax_f1.set_xlim(min(top_pcts) - 1.5, max(top_pcts) + 1.5)
    ax_f1.set_ylim(0.0, max(0.45, ax_f1.get_ylim()[1] * 1.05))
    ax_f1.grid(True, axis="y", linestyle="-", alpha=0.45)
    ax_f1.spines["top"].set_visible(False)
    ax_f1.spines["right"].set_visible(False)
    ax_f1.legend(frameon=True, loc="lower right", fontsize=9)

    metric_keys = [
        ("boundary_enrichment", "Enrichment\nvs interior"),
        ("surrounding_enrichment", "Enrichment\nvs surrounding"),
        ("average_precision", "AP"),
        ("f1_mean_over_pcts", "Mean F1"),
    ]

    n_m = len(methods)
    n_met = len(metric_keys)
    x_idx = np.arange(n_met, dtype=np.float64)
    width = 0.22
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width
    by_method = {r["method"]: r for r in summary}

    for i, method in enumerate(methods):
        sr = by_method[method]
        vals = []
        errs = []
        for key, _name in metric_keys:
            vals.append(float(sr[f"{key}_mean"]))
            errs.append(float(sr[f"{key}_std"]))
        color = _METHOD_COLORS.get(method, "#444444")
        bars = ax_sum.bar(
            x_idx + offsets[i],
            vals,
            width=width * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=labels.get(method, method),
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"},
        )
        for b, v in zip(bars, vals):
            ax_sum.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="#222222",
            )

    ax_sum.set_xticks(x_idx)
    ax_sum.set_xticklabels([name for _k, name in metric_keys], fontsize=9)
    ax_sum.set_ylabel("Score")
    ax_sum.set_title("MSI edge evaluation metrics")
    ax_sum.set_ylim(0.0, max(1.55, ax_sum.get_ylim()[1] * 1.12))
    ax_sum.grid(True, axis="y", linestyle="-", alpha=0.45)
    ax_sum.spines["top"].set_visible(False)
    ax_sum.spines["right"].set_visible(False)
    ax_sum.legend(frameon=True, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    saved: list[Path] = []
    png = out_dir / "annotation_edge_metrics_figure.png"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    saved.append(png)
    if save_pdf:
        pdf = out_dir / "annotation_edge_metrics_figure.pdf"
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        saved.append(pdf)
    plt.close(fig)
    return saved


def _save_by_category_figure(
    by_cat_summary: list[dict[str, Any]],
    methods: list[str],
    labels: dict[str, str],
    out_dir: Path,
    *,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt

    if not by_cat_summary:
        return []

    cats = sorted({str(r["main_cat1"]) for r in by_cat_summary})
    # Prefer categories present for all methods; keep order by mean SoLaCE enrichment
    sol = "soft_landmark_contrast"
    def _sol_enr(cat: str) -> float:
        for r in by_cat_summary:
            if r["main_cat1"] == cat and r["method"] == sol:
                return float(r["boundary_enrichment_mean"])
        return -1.0

    cats = sorted(cats, key=_sol_enr, reverse=True)

    fig, ax = plt.subplots(figsize=(max(7.5, 1.2 * len(cats) + 2.5), 4.2))
    n_m = len(methods)
    x_idx = np.arange(len(cats), dtype=np.float64)
    width = min(0.24, 0.75 / max(n_m, 1))
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width
    lookup = {(r["main_cat1"], r["method"]): r for r in by_cat_summary}

    for i, method in enumerate(methods):
        vals, errs = [], []
        for cat in cats:
            r = lookup.get((cat, method))
            if r is None:
                vals.append(0.0)
                errs.append(0.0)
            else:
                vals.append(float(r["boundary_enrichment_mean"]))
                errs.append(float(r["boundary_enrichment_std"]))
        color = _METHOD_COLORS.get(method, "#444444")
        ax.bar(
            x_idx + offsets[i],
            vals,
            width=width * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=labels.get(method, method),
            yerr=errs,
            capsize=2.5,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
        )

    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([_MAIN_CAT_DISPLAY.get(c, c) for c in cats])
    ax.set_ylabel("Enrichment vs surrounding")
    ax.set_xlabel("Anatomical category")
    ax.set_title("Boundary enrichment vs local surrounding")
    ax.set_ylim(0.0, max(1.6, ax.get_ylim()[1] * 1.08))
    ax.grid(True, axis="y", linestyle="-", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    fig.tight_layout()

    saved: list[Path] = []
    png = out_dir / "enrichment_by_main_cat1.png"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    saved.append(png)
    if save_pdf:
        pdf = out_dir / "enrichment_by_main_cat1.pdf"
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        saved.append(pdf)
    plt.close(fig)
    return saved


@hydra.main(version_base=None, config_path="configs", config_name="annotation_edge_metrics")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ann = cfg.annotation
    edges_cfg = cfg.edges
    met_cfg = cfg.metrics

    annotation_json = _opt_path(ann.annotation_json)
    if annotation_json is None or not annotation_json.is_file():
        raise FileNotFoundError(f"annotation_json not found: {annotation_json}")
    raw_root = _opt_path(edges_cfg.raw_root)
    if raw_root is None or not raw_root.is_dir():
        raise FileNotFoundError(f"edges.raw_root not found: {raw_root}")

    spatial_mode = resolve_transpose_spatial_mode(ann)
    slides = _resolve_slide_rows(ann)
    methods = [str(m) for m in OmegaConf.to_container(edges_cfg.methods, resolve=True)]
    if not methods:
        methods = list(METHODS_DEFAULT)
    labels_raw = OmegaConf.to_container(getattr(edges_cfg, "method_labels", {}), resolve=True) or {}
    labels = {str(k): str(v) for k, v in dict(labels_raw).items()}

    ignore = list(OmegaConf.to_container(getattr(ann, "ignore", []), resolve=True) or [])
    keep_raw = getattr(ann, "keep", None)
    keep = set(OmegaConf.to_container(keep_raw, resolve=True)) if keep_raw is not None else None
    cat_keys = tuple(
        str(x) for x in (OmegaConf.to_container(getattr(ann, "category_keys", None), resolve=True) or [])
    ) or ("main_cat1", "Allen_name", "sub_cat2")

    boundary_dilate = int(getattr(met_cfg, "boundary_dilate", 1))
    top_pcts = _resolve_top_pcts(met_cfg)
    rank_normalize = bool(getattr(met_cfg, "rank_normalize", True))
    min_cat_boundary = int(getattr(met_cfg, "min_category_boundary_pixels", 40))
    min_cat_interior = int(getattr(met_cfg, "min_category_interior_pixels", 8))
    cat_ring_width = int(getattr(met_cfg, "category_ring_width", 3))
    cat_neg_mode = str(getattr(met_cfg, "category_negative_mode", "surrounding"))
    pooled_ring_width = int(getattr(met_cfg, "pooled_ring_width", cat_ring_width))
    plot_cfg = getattr(met_cfg, "plot", None)
    plot_dpi = int(getattr(plot_cfg, "dpi", 300)) if plot_cfg is not None else 300
    plot_pdf = bool(getattr(plot_cfg, "save_pdf", True)) if plot_cfg is not None else True

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    per_slide: list[dict[str, Any]] = []
    by_cat_rows: list[dict[str, Any]] = []
    for npy, index_str, path_label in slides:
        polys, cats, main_cats, n_keys = _load_polygons_for_stem(
            annotation_json,
            index_str,
            ignore=ignore,
            keep=keep,
            category_keys=cat_keys,
        )
        n_raw = len(polys)
        polys, cats, main_cats = _dedupe_polygons(polys, cats, main_cats)
        if not polys:
            logger.warning("No regions for %s — skip", npy.name)
            continue
        logger.info(
            "stem %s: %d VIA keys, %d raw regions -> %d unique (ignore=%s)",
            index_str,
            n_keys,
            n_raw,
            len(polys),
            ignore,
        )
        file_hw = spatial_hw_from_npy(npy)
        if file_hw is None:
            raise ValueError(f"Cannot read shape: {npy}")
        shape_hw = spatial_hw_after_transpose_mode(file_hw, spatial_mode)
        boundary, interior, annotated = _rasterize_annotation_masks(
            polys, shape_hw, boundary_dilate=boundary_dilate
        )
        filled_all = _filled_mask(polys, shape_hw)
        surrounding = _surrounding_ring_from_filled(
            filled_all, ring_width=pooled_ring_width, exclude=boundary
        )
        logger.info(
            "%s | boundary=%d interior=%d surrounding=%d annotated=%d",
            npy.stem,
            int(boundary.sum()),
            int(interior.sum()),
            int(surrounding.sum()),
            int(annotated.sum()),
        )

        # Per main_cat1: default = boundary vs local surrounding (all categories).
        uniq_mains = sorted({m for m in main_cats if m and m != "ARTEFACT"})
        cat_masks: dict[str, tuple[np.ndarray, np.ndarray, str, int]] = {}
        for main in uniq_mains:
            sub_polys = [p for p, m in zip(polys, main_cats) if m == main]
            if not sub_polys:
                continue
            built = _rasterize_category_masks(
                sub_polys,
                shape_hw,
                boundary_dilate=boundary_dilate,
                min_negatives=min_cat_interior,
                ring_width=cat_ring_width,
                negative_mode=cat_neg_mode,
            )
            if built is None:
                continue
            b_c, neg_c, neg_mode, dil_used = built
            if int(b_c.sum()) < min_cat_boundary:
                continue
            cat_masks[main] = (b_c, neg_c, neg_mode, dil_used)
            logger.info(
                "  cat %-4s  boundary=%d  negatives=%d  mode=%s  dilate=%d  n_poly=%d",
                main,
                int(b_c.sum()),
                int(neg_c.sum()),
                neg_mode,
                dil_used,
                len(sub_polys),
            )

        for method in methods:
            gray = _load_edge_gray_aligned(
                npy, raw_root=raw_root, method_key=method, spatial_mode=spatial_mode
            )
            tissue = gray > 0
            tissue_eval = tissue
            m = _metrics_for_edge(
                gray,
                boundary=boundary,
                interior=interior,
                tissue=tissue_eval,
                rank_normalize=rank_normalize,
                top_pcts=top_pcts,
                surrounding=surrounding,
            )
            row = {
                "slide": npy.stem,
                "hemisphere": path_label if path_label is not None else "",
                "method": method,
                "method_label": labels.get(method, method),
                "top_pcts": ";".join(f"{p:g}" for p in top_pcts),
                "boundary_dilate": boundary_dilate,
                "rank_normalize": rank_normalize,
                "pooled_ring_width": pooled_ring_width,
                **m,
            }
            per_slide.append(row)
            logger.info(
                "  %-22s enrich=%.3f surround=%.3f AP=%.3f meanF1=%.3f",
                labels.get(method, method),
                m["boundary_enrichment"],
                m["surrounding_enrichment"],
                m["average_precision"],
                m["f1_mean_over_pcts"],
            )

            scores = np.asarray(gray, dtype=np.float64)
            if rank_normalize:
                scores = _rank_normalize_on_mask(scores, tissue_eval)
            for main, (b_c, neg_c, neg_mode, dil_used) in cat_masks.items():
                enr, ap, n_b, n_neg = _enrichment_only(
                    scores,
                    boundary=b_c & tissue_eval,
                    interior=neg_c & tissue_eval,
                )
                if not np.isfinite(enr) or n_neg < 1:
                    continue
                by_cat_rows.append(
                    {
                        "slide": npy.stem,
                        "hemisphere": path_label if path_label is not None else "",
                        "main_cat1": main,
                        "method": method,
                        "method_label": labels.get(method, method),
                        "boundary_enrichment": enr,
                        "average_precision": ap,
                        "n_boundary": n_b,
                        "n_interior": n_neg,
                        "negative_mode": neg_mode,
                        "category_boundary_dilate": dil_used,
                    }
                )

    if not per_slide:
        raise SystemExit("No metrics computed.")

    # Keep summary method order aligned with `methods`
    summary = _aggregate_table(per_slide, methods, labels, top_pcts)
    by_cat_summary = _aggregate_by_category(by_cat_rows, methods, labels)
    _write_csv(per_slide, out_dir / "metrics_by_slide.csv")
    _write_csv(summary, out_dir / "metrics_summary.csv")
    if by_cat_rows:
        _write_csv(by_cat_rows, out_dir / "metrics_by_main_cat1_slide.csv")
    if by_cat_summary:
        _write_csv(by_cat_summary, out_dir / "metrics_by_main_cat1_summary.csv")
    _write_markdown_tables(
        per_slide,
        summary,
        out_dir / "metrics_table.md",
        top_pcts=top_pcts,
        by_cat_summary=by_cat_summary,
    )
    _write_latex_summary(summary, out_dir / "metrics_table.tex", top_pcts=top_pcts)
    fig_paths = _save_paper_figure(
        per_slide,
        summary,
        methods,
        labels,
        top_pcts,
        out_dir,
        dpi=plot_dpi,
        save_pdf=plot_pdf,
    )
    fig_paths.extend(
        _save_by_category_figure(
            by_cat_summary,
            methods,
            labels,
            out_dir,
            dpi=plot_dpi,
            save_pdf=plot_pdf,
        )
    )

    tsv_path = out_dir / "metrics_summary.tsv"
    if summary:
        keys = list(summary[0].keys())
        tsv_path.write_text(
            "\t".join(keys)
            + "\n"
            + "\n".join("\t".join(str(r[k]) for k in keys) for r in summary)
            + "\n",
            encoding="utf-8",
        )

    print(f"[annotation_edge_metrics] Wrote {out_dir / 'metrics_table.md'}", flush=True)
    for p in fig_paths:
        print(f"[annotation_edge_metrics] Wrote {p}", flush=True)
    print("", flush=True)
    print(f"{'Method':22s}  {'Enrich':>8s}  {'Surround':>8s}  {'AP':>8s}  {'AUC':>8s}  {'MeanF1':>8s}", flush=True)
    for r in summary:
        print(
            f"{str(r['method_label']):22s}  "
            f"{r['boundary_enrichment_mean']:8.3f}  "
            f"{r['surrounding_enrichment_mean']:8.3f}  "
            f"{r['average_precision_mean']:8.3f}  "
            f"{r['roc_auc_mean']:8.3f}  "
            f"{r['f1_mean_over_pcts_mean']:8.3f}",
            flush=True,
        )
    if by_cat_summary:
        print("", flush=True)
        print("Enrichment vs surrounding by main_cat1 (mean over slides):", flush=True)
        cats = sorted({r["main_cat1"] for r in by_cat_summary})
        for cat in cats:
            bits = []
            for method in methods:
                hit = next((r for r in by_cat_summary if r["main_cat1"] == cat and r["method"] == method), None)
                if hit:
                    bits.append(f"{labels.get(method, method)}={hit['boundary_enrichment_mean']:.2f}")
            print(f"  {cat:6s}  " + "  ".join(bits), flush=True)


if __name__ == "__main__":
    main()