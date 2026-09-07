#!/usr/bin/env python3
"""
Rebuild the cross-inference mosaic from saved panels (no retrain).

Adds dashed Train highlights on the diagonal and boxed method titles.

Example:
  python scripts/restyle_mics_cross_infer_gallery.py \\
    --run-dir outputs/mics_cross_infer_gallery/2026-06-07/07-45-43

  # Histogram-equalized panels → restyled_equalized/
  python scripts/restyle_mics_cross_infer_gallery.py \\
    --run-dir outputs/mics_cross_infer_gallery/2026-06-07/07-45-43 --equalize
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mics_cross_infer_gallery import (  # noqa: E402
    METHOD_LABELS,
    _build_cross_infer_mosaic,
    _combine_method_blocks,
    _hist_equalize_rgb,
    _prepare_display_grid,
    _wrap_block_title,
)

_PANEL_RE = re.compile(
    r"^(?P<prefix>pmics|pumap|pca)__train_(?P<train>[^_]+)__infer_(?P<infer>[^_]+)\.png$",
    re.I,
)
_PREFIX_TO_METHOD = {
    "pmics": "mics",
    "pumap": "parametric_umap",
    "pca": "pca",
}


def _load_grids(panels_dir: Path) -> tuple[list[str], dict[str, list[list[np.ndarray]]]]:
    files = sorted(panels_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No panels in {panels_dir}")

    by_method: dict[str, dict[tuple[str, str], Path]] = {}
    stems: set[str] = set()
    for p in files:
        m = _PANEL_RE.match(p.name)
        if not m:
            continue
        method = _PREFIX_TO_METHOD[m.group("prefix").lower()]
        train, infer = m.group("train"), m.group("infer")
        stems.add(train)
        stems.add(infer)
        by_method.setdefault(method, {})[(train, infer)] = p

    # Hemisphere order from README of the June run: stems [2,0,1,3] → hemi 0..3
    # Prefer numeric stem sort only if that matches file set; else use order of
    # first appearance in train_* filenames for mics.
    stem_list = sorted(stems, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
    # Reconstruct display order: train row order as in pmics train_* files if present
    mics_trains = sorted(
        {t for (t, _) in by_method.get("mics", {})},
        key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s),
    )
    # Actual June order was hemisphere-sorted file stems [2,0,1,3]. Detect from
    # which train/infer pairs exist with consistent diagonal completeness.
    # Use config_resolved / README if available; else hemisphere map.
    hemi_order = _resolve_stem_order(panels_dir.parent, stem_list, mics_trains)

    grids: dict[str, list[list[np.ndarray]]] = {}
    for method, pairs in by_method.items():
        n = len(hemi_order)
        grid: list[list[np.ndarray | None]] = [[None] * n for _ in range(n)]
        for r, train in enumerate(hemi_order):
            for c, infer in enumerate(hemi_order):
                path = pairs.get((train, infer))
                if path is None:
                    raise FileNotFoundError(f"Missing panel {method} train={train} infer={infer}")
                grid[r][c] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        grids[method] = grid  # type: ignore[assignment]
    return hemi_order, grids


def _resolve_stem_order(run_dir: Path, stems: list[str], fallback: list[str]) -> list[str]:
    readme = run_dir / "README.txt"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        # Paths: ['...\\2.npy', '...\\0.npy', ...]
        m = re.search(r"Paths:\s*\[([^\]]+)\]", text)
        if m:
            parts = re.findall(r"[\\/](\d+)\.npy", m.group(1))
            if parts and set(parts) == set(stems):
                return parts
        m = re.search(r"hemisphere ids:\s*\[([^\]]+)\]", text)
        # Not enough alone; try config
    cfg = run_dir / "config_resolved.yaml"
    if cfg.is_file():
        text = cfg.read_text(encoding="utf-8", errors="replace")
        # Prefer paths order after slice_order already applied — README Paths is best.
        paths = re.findall(r'-\s*".*?[\\/](\d+)\.npy"', text)
        if not paths:
            paths = re.findall(r"-\s+'.*?[\\/](\d+)\.npy'", text)
        hemi = re.search(r"hemisphere_ids:\s*\[([^\]]+)\]", text)
        if paths and hemi:
            hemi_ids = [int(x.strip()) for x in hemi.group(1).split(",")]
            if len(paths) == len(hemi_ids) == len(stems):
                # Apply hemisphere sort like the gallery script
                order = sorted(range(len(paths)), key=lambda i: hemi_ids[i])
                ordered = [paths[i] for i in order]
                if set(ordered) == set(stems):
                    return ordered
    # NRL4485 default map
    hemi_map = {"0": 1, "1": 2, "2": 0, "3": 3}
    if set(stems) <= set(hemi_map):
        return sorted(stems, key=lambda s: hemi_map[s])
    return fallback if fallback else stems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Hydra run directory containing panels/",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/restyled)",
    )
    p.add_argument("--method-layout", default="horizontal", choices=("horizontal", "vertical"))
    p.add_argument("--method-block-gap", type=int, default=12)
    p.add_argument("--cell-gap", type=int, default=2)
    p.add_argument("--mosaic-margin", type=int, default=2)
    p.add_argument("--row-label-width", type=int, default=22)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument(
        "--mosaic-max-panel-width",
        type=int,
        default=None,
        help="Downscale panels for mosaic only (null/omit = native)",
    )
    p.add_argument(
        "--equalize",
        action="store_true",
        help="Apply per-channel histogram equalization to each panel (same as gallery.equalize_visualization)",
    )
    p.add_argument(
        "--no-text",
        action="store_true",
        help="Omit method titles, row/col labels, and Train annotations",
    )
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    panels_dir = run_dir / "panels"
    if args.out_dir is not None:
        default_name = None
        out_dir = args.out_dir.resolve()
    elif args.equalize and args.no_text:
        out_dir = (run_dir / "restyled_equalized_notext").resolve()
    elif args.equalize:
        out_dir = (run_dir / "restyled_equalized").resolve()
    elif args.no_text:
        out_dir = (run_dir / "restyled_notext").resolve()
    else:
        out_dir = (run_dir / "restyled").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stems, grids = _load_grids(panels_dir)
    if args.equalize:
        grids = {
            method: [[_hist_equalize_rgb(cell) for cell in row] for row in grid]
            for method, grid in grids.items()
        }
    n = len(stems)
    if args.no_text:
        row_labels = [""] * n
        col_labels = None
        row_label_w = 0
        mark_train = False
    else:
        hemi_labels = [str(i) for i in range(n)]
        row_labels = list(hemi_labels)
        col_labels = list(hemi_labels)
        row_label_w = args.row_label_width
        mark_train = True

    method_order = [k for k in ("mics", "parametric_umap", "pca") if k in grids]
    blocks: list[np.ndarray] = []
    for method_key in method_order:
        disp = _prepare_display_grid(
            grids[method_key],
            pad_val=0,
            pad_align="top_left",
            mosaic_max_w=args.mosaic_max_panel_width,
        )
        block = _build_cross_infer_mosaic(
            disp,
            row_labels=row_labels,
            col_labels=col_labels,
            cell_labels=None,
            cell_gap=args.cell_gap,
            margin=args.mosaic_margin,
            row_label_w=row_label_w,
            mark_train=mark_train,
        )
        if args.no_text:
            titled = block
        else:
            title = METHOD_LABELS.get(method_key, method_key)
            titled = _wrap_block_title(block, title)
        single = out_dir / f"cross_infer_{method_key}.png"
        Image.fromarray(titled, mode="RGB").save(single, dpi=(args.dpi, args.dpi))
        print(f"Wrote {single}")
        blocks.append(titled)

    mosaic = _combine_method_blocks(blocks, args.method_block_gap, args.method_layout)
    mosaic_path = out_dir / "mics_cross_infer_gallery.png"
    Image.fromarray(mosaic, mode="RGB").save(mosaic_path, dpi=(args.dpi, args.dpi))
    print(f"Wrote {mosaic_path} ({mosaic.shape[1]}x{mosaic.shape[0]})")
    print(f"Stem order: {stems}")
    print(f"Equalize: {bool(args.equalize)}  no_text: {bool(args.no_text)}  cell_gap: {args.cell_gap}")


if __name__ == "__main__":
    main()
