#!/usr/bin/env python3
"""
Gallery from bayes_tune diagnostic_panels hires tiles.

One row per slide:
  H&E | SoLaCE | MiCS #1 … #6  (8 columns)

MiCS tiles optionally LAB-L equalized (SoLaCE and H&E unchanged).

Run (repo root):
  python scripts/gallery_diagnostic_hires_summary.py \\
    --panels-dir outputs/bayes_tune_mics_lmc_carstenhopf/2026-06-19/18-10-42/diagnostic_panels
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_BG = (14, 14, 18)
_GAP = 0
_ROW_GAP = 0
_HEADER_H = 16
_SLIDE_BAR_H = 14
_DICE_TAG_H = 14
_DEFAULT_TOP_MICS = 6


def _equalize_rgb_u8_lab_l(rgb: np.ndarray) -> np.ndarray:
    """Histogram-equalize LAB L channel (matches diagnostic_panel equalized MiCS)."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.equalizeHist(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _slide_sort_key(slide_key: str) -> tuple:
    m = re.search(r"ID(\d+[a-z]?)_MSimaging_(neg|pos)", slide_key, re.I)
    if not m:
        return (2, slide_key)
    id_part, pol = m.group(1), m.group(2).lower()
    num = int(re.match(r"\d+", id_part).group())
    suffix = id_part[len(str(num)) :]
    pol_ord = 0 if pol == "neg" else 1
    return (pol_ord, num, suffix, slide_key)


def _resize_to_height(rgb: np.ndarray, target_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == target_h:
        return rgb
    scale = target_h / float(h)
    tw = max(1, int(round(w * scale)))
    return cv2.resize(rgb, (tw, target_h), interpolation=cv2.INTER_AREA)


def _pad_center(rgb: np.ndarray, out_h: int, out_w: int, bg: tuple[int, int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    y0 = max(0, (out_h - h) // 2)
    x0 = max(0, (out_w - w) // 2)
    y1 = min(out_h, y0 + h)
    x1 = min(out_w, x0 + w)
    canvas[y0:y1, x0:x1] = rgb[: y1 - y0, : x1 - x0]
    return canvas


def _draw_bar(
    width: int,
    height: int,
    text: str,
    *,
    bg: tuple[int, int, int] = (28, 28, 36),
    fg: tuple[int, int, int] = (230, 230, 235),
    font_size: int = 14,
) -> np.ndarray:
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2 - 1), text, fill=fg, font=font)
    return np.asarray(img, dtype=np.uint8)


@dataclass
class SlideRow:
    slide_key: str
    he: np.ndarray
    solace: np.ndarray
    mics: list[np.ndarray]
    dice_labels: list[str | None]


def _load_slide_row(
    entry: dict,
    panels_dir: Path,
    *,
    top_mics: int,
    equalize_mics: bool,
) -> SlideRow:
    slide_key = str(entry["slide_key"])
    tiles = panels_dir / slide_key / "tiles_hires"
    he_p = tiles / "he_hires.png"
    sol_p = tiles / "solace_hires.png"
    if not he_p.is_file() or not sol_p.is_file():
        raise FileNotFoundError(f"Missing H&E or SoLaCE tile for {slide_key}")

    mics_paths = [tiles / f"mics_rank{r:02d}_hires.png" for r in range(1, top_mics + 1)]
    for p in mics_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing tile for {slide_key}: {p}")

    trials = entry.get("top_trials") or []
    dice_labels: list[str | None] = []
    for r in range(top_mics):
        if r < len(trials) and trials[r].get("continuous_dice") is not None:
            dice_labels.append(f"dice={float(trials[r]['continuous_dice']):.3f}")
        else:
            dice_labels.append(None)

    mics_rgbs: list[np.ndarray] = []
    for p in mics_paths:
        rgb = _load_rgb(p)
        if equalize_mics:
            rgb = _equalize_rgb_u8_lab_l(rgb)
        mics_rgbs.append(rgb)

    return SlideRow(
        slide_key=slide_key,
        he=_load_rgb(he_p),
        solace=_load_rgb(sol_p),
        mics=mics_rgbs,
        dice_labels=dice_labels,
    )


def _column_labels(top_mics: int) -> tuple[str, ...]:
    return ("H&E", "SoLaCE", *tuple(f"MiCS #{i}" for i in range(1, top_mics + 1)))


def build_gallery(
    panels_dir: Path,
    *,
    top_mics: int = _DEFAULT_TOP_MICS,
    cell_h: int = 420,
    gap: int = _GAP,
    row_gap: int = _ROW_GAP,
    equalize_mics: bool = True,
    out_path: Path | None = None,
) -> Path:
    manifest_path = panels_dir / "diagnostic_panels_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    top_mics = max(1, int(top_mics))
    ncols = 2 + top_mics
    gap = max(0, int(gap))
    row_gap = max(0, int(row_gap))

    entries = sorted(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        key=lambda e: _slide_sort_key(str(e["slide_key"])),
    )
    slides = [
        _load_slide_row(e, panels_dir, top_mics=top_mics, equalize_mics=equalize_mics)
        for e in entries
    ]

    # Uniform cell width across all tiles (after height resize).
    widths: list[int] = []
    for slide in slides:
        for rgb in (slide.he, slide.solace, *slide.mics):
            widths.append(int(_resize_to_height(rgb, cell_h).shape[1]))
    cell_w = max(widths) if widths else cell_h

    canvas_w = ncols * cell_w + max(0, ncols - 1) * gap
    col_labels = _column_labels(top_mics)

    header_row = np.full((_HEADER_H, canvas_w, 3), _BG, dtype=np.uint8)
    for i, lab in enumerate(col_labels):
        x0 = i * (cell_w + gap)
        header_row[:, x0 : x0 + cell_w] = _draw_bar(cell_w, _HEADER_H, lab, font_size=10)

    canvas_parts: list[np.ndarray] = [header_row]

    for slide in slides:
        short = slide.slide_key.replace("_MSimaging_", " / ").replace("_RMS", "")
        canvas_parts.append(_draw_bar(canvas_w, _SLIDE_BAR_H, short, font_size=9))

        tile_row = np.full((cell_h, canvas_w, 3), _BG, dtype=np.uint8)
        tiles = [slide.he, slide.solace, *slide.mics]
        dice_for_col = [None, None, *slide.dice_labels]

        for col, (rgb, dice) in enumerate(zip(tiles, dice_for_col)):
            cell = _pad_center(_resize_to_height(rgb, cell_h), cell_h, cell_w, _BG)
            x0 = col * (cell_w + gap)
            tile_row[:, x0 : x0 + cell_w] = cell
            if dice:
                tag = _draw_bar(cell_w, _DICE_TAG_H, dice, bg=(20, 20, 28), font_size=9)
                tile_row[cell_h - _DICE_TAG_H : cell_h, x0 : x0 + cell_w] = tag

        canvas_parts.append(tile_row)
        if row_gap > 0:
            canvas_parts.append(np.full((row_gap, canvas_w, 3), _BG, dtype=np.uint8))

    if canvas_parts and row_gap > 0 and canvas_parts[-1].shape[0] == row_gap:
        canvas_parts.pop()

    gallery = np.vstack(canvas_parts)
    if out_path is None:
        eq_tag = "_mics_equalized" if equalize_mics else ""
        out_path = panels_dir / f"gallery_hires_he_solace_top{top_mics}_{ncols}col{eq_tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gallery).save(out_path, dpi=(150, 150))
    logger.info("Wrote gallery %dx%d (%d slides) -> %s", gallery.shape[1], gallery.shape[0], len(slides), out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description="H&E + SoLaCE + top MiCS hires gallery (1 row/slide).")
    ap.add_argument(
        "--panels-dir",
        type=Path,
        default=Path(
            "outputs/bayes_tune_mics_lmc_carstenhopf/2026-06-19/18-10-42/diagnostic_panels"
        ),
    )
    ap.add_argument("--top-mics", type=int, default=_DEFAULT_TOP_MICS)
    ap.add_argument("--cell-h", type=int, default=420)
    ap.add_argument("--gap", type=int, default=_GAP)
    ap.add_argument("--row-gap", type=int, default=_ROW_GAP)
    ap.add_argument(
        "--equalize-mics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LAB-L equalize MiCS tiles only (not SoLaCE or H&E)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    panels_dir = args.panels_dir.expanduser().resolve()
    out = args.out.expanduser().resolve() if args.out else None
    build_gallery(
        panels_dir,
        top_mics=int(args.top_mics),
        cell_h=int(args.cell_h),
        gap=int(args.gap),
        row_gap=int(args.row_gap),
        equalize_mics=bool(args.equalize_mics),
        out_path=out,
    )


if __name__ == "__main__":
    main()
