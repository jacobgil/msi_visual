#!/usr/bin/env python3
"""
List spatial/array shapes for files under two directories, then match by (rows, cols).

- npy-dir: all *.npy files (shape = numpy array shape).
- viz-dir: files with common image or .npy extensions (image shape = HxWxC from decoded pixels).

Matching uses the first two dimensions as (H, W) = (rows, cols).

**1 : many** — Each ``.npy`` is matched to **all** visualization files at the same spatial size
when there is **only one** ``.npy`` at that size. If **several** ``.npy`` files share the same
(H, W), PNGs are assigned by **filename prefix**: ``<npy_stem>_...`` or ``<npy_stem>.<ext>``
(case-insensitive). Any viz file at that size that does not match any stem is listed as unassigned.

Example:
  python scripts/list_dir_shapes.py --npy-dir "D:/data/cubes" --viz-dir "D:/data/pngs"
  python scripts/list_dir_shapes.py ... --no-map   # bucket lists only, no 1:n mapping
  python scripts/list_dir_shapes.py ... --no-match  # shapes only
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _shape_tuple_npy(path: Path) -> tuple[int, ...]:
    arr = np.load(path, mmap_mode="r")
    return tuple(int(x) for x in arr.shape)


def _shape_tuple_image(path: Path) -> tuple[int, ...]:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im)
        return tuple(int(x) for x in arr.shape)


def _shape_tuple(path: Path) -> tuple[int, ...]:
    suf = path.suffix.lower()
    if suf == ".npy":
        return _shape_tuple_npy(path)
    return _shape_tuple_image(path)


def _spatial_hw(shape: tuple[int, ...]) -> tuple[int, int] | None:
    """Rows, cols from the leading two dimensions (same as MSI / image height x width)."""
    if len(shape) < 2:
        return None
    return int(shape[0]), int(shape[1])


def _iter_files(root: Path, extensions: set[str]) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in extensions:
            out.append(p)
    return out


def _print_section(title: str, root: Path, paths: list[Path]) -> None:
    print(f"\n=== {title} ===")
    print(f"path: {root.resolve()}")
    print(f"count: {len(paths)}")
    for p in paths:
        try:
            sh = _shape_tuple(p)
            line = str(sh)
        except Exception as exc:
            line = f"<error: {exc}>"
        print(f"  {p.name}\t{line}")


def _bucket_by_hw(paths: list[Path]) -> dict[tuple[int, int], list[Path]]:
    buckets: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for p in paths:
        try:
            sh = _shape_tuple(p)
            hw = _spatial_hw(sh)
        except Exception:
            continue
        if hw is None:
            continue
        buckets[hw].append(p)
    for k in buckets:
        buckets[k].sort(key=lambda x: x.name.lower())
    return dict(buckets)


def _viz_name_matches_npy_stem(viz_name: str, npy_stem: str) -> bool:
    """True if viz filename is ``stem_...`` or ``stem.ext`` (e.g. 0_Foo.png, 2.png)."""
    s = npy_stem.lower()
    n = viz_name.lower()
    return n.startswith(s + "_") or n.startswith(s + ".")


def _one_to_many_mapping(
    npys_at_hw: list[Path],
    viz_at_hw: list[Path],
) -> tuple[dict[Path, list[Path]], list[Path]]:
    """
    Each viz file is assigned to at most one npy.

    - Single npy at this (H,W): all viz files map to it (1 : N).
    - Multiple npys: assign by ``<stem>_`` / ``<stem>.`` prefix; if several stems match,
      the **longest** stem wins so ``10_`` beats ``1_``.
    """
    if len(npys_at_hw) == 1:
        return {npys_at_hw[0]: list(viz_at_hw)}, []

    stems_by_len = sorted({p.stem for p in npys_at_hw}, key=len, reverse=True)
    stem_to_npy = {p.stem: p for p in npys_at_hw}

    viz_to_npy: dict[Path, Path] = {}
    for v in viz_at_hw:
        best_stem: str | None = None
        best_len = -1
        for st in stems_by_len:
            if _viz_name_matches_npy_stem(v.name, st) and len(st) > best_len:
                best_stem = st
                best_len = len(st)
        if best_stem is not None:
            viz_to_npy[v] = stem_to_npy[best_stem]

    by_npy: dict[Path, list[Path]] = {n: [] for n in npys_at_hw}
    for v, npy in viz_to_npy.items():
        by_npy[npy].append(v)
    for n in by_npy:
        by_npy[n].sort(key=lambda x: x.name.lower())
    unassigned = [v for v in viz_at_hw if v not in viz_to_npy]
    unassigned.sort(key=lambda x: x.name.lower())
    return by_npy, unassigned


def _print_match_section(
    npy_buckets: dict[tuple[int, int], list[Path]],
    viz_buckets: dict[tuple[int, int], list[Path]],
    *,
    show_map: bool,
) -> None:
    print("\n=== Match by (rows, cols) - 1 numpy : many viz ===")
    all_hw = sorted(set(npy_buckets) | set(viz_buckets))
    shared = set(npy_buckets) & set(viz_buckets)

    for hw in all_hw:
        h, w = hw
        na = npy_buckets.get(hw, [])
        va = viz_buckets.get(hw, [])
        print(f"\n  (rows, cols) = ({h}, {w})")
        print(f"    npy-dir: {len(na)} file(s)")
        for p in na:
            print(f"      {p.name}")
        print(f"    viz-dir: {len(va)} file(s)")
        for p in va:
            print(f"      {p.name}")

        if not na or not va:
            print("    -> no cross-dir match (missing files on one side).")
            continue

        if not show_map:
            continue

        if len(na) == 1:
            print(f"    -> single .npy at this size: all {len(va)} viz file(s) map to it (1 : N).")
        else:
            print(
                "    -> multiple .npy at this size: each viz maps to one stem by prefix "
                "``<stem>_`` / ``<stem>.<ext>``; longest stem wins if several match.",
            )

        by_npy, unassigned = _one_to_many_mapping(na, va)
        for npy in sorted(by_npy, key=lambda p: p.name.lower()):
            mapped = by_npy[npy]
            print(f"    -> {npy.name}  ->  {len(mapped)} viz file(s):")
            if not mapped:
                print("         (none matched this stem prefix)")
            for v in mapped:
                print(f"         {v.name}")

        if unassigned:
            print(f"    -> unassigned viz at this size ({len(unassigned)}), no matching npy stem:")
            for v in unassigned:
                print(f"         {v.name}")

    print(
        f"\n  Summary: {len(shared)} spatial size(s) (rows, cols) appear in both directories; "
        f"{len(all_hw) - len(shared)} size(s) are only on one side.",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy-dir", type=Path, required=True, help="Directory of .npy files to inspect")
    ap.add_argument(
        "--viz-dir",
        type=Path,
        required=True,
        help="Directory of visualizations (.png, .jpg, …, and optionally .npy)",
    )
    ap.add_argument(
        "--viz-ext",
        type=str,
        default="png,jpg,jpeg,tif,tiff,webp,bmp,npy",
        help="Comma-separated extensions to include under viz-dir (default: common images + npy)",
    )
    ap.add_argument(
        "--no-match",
        action="store_true",
        help="Only print per-file shapes; do not group or pair by (rows, cols).",
    )
    ap.add_argument(
        "--no-map",
        action="store_true",
        help="List files per (H,W) bucket but do not print 1:n numpy->viz mapping.",
    )
    args = ap.parse_args()
    show_map = not bool(args.no_map)

    npy_dir = args.npy_dir
    viz_dir = args.viz_dir
    if not npy_dir.is_dir():
        print(f"npy-dir is not a directory: {npy_dir}", file=sys.stderr)
        sys.exit(1)
    if not viz_dir.is_dir():
        print(f"viz-dir is not a directory: {viz_dir}", file=sys.stderr)
        sys.exit(1)

    viz_exts = {"." + x.strip().lstrip(".").lower() for x in args.viz_ext.split(",") if x.strip()}

    npy_paths = _iter_files(npy_dir, {".npy"})
    viz_paths = _iter_files(viz_dir, viz_exts)

    _print_section("npy-dir (*.npy)", npy_dir, npy_paths)
    _print_section("viz-dir", viz_dir, viz_paths)

    if not args.no_match:
        npy_b = _bucket_by_hw(npy_paths)
        viz_b = _bucket_by_hw(viz_paths)
        _print_match_section(npy_b, viz_b, show_map=show_map)


if __name__ == "__main__":
    main()
