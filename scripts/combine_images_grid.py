#!/usr/bin/env python3
"""Combine images from a folder into a single grid image."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required to run this script. Install it with: pip install pillow"
    ) from exc


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def split_prefix_and_row_key(path: Path) -> tuple[int | None, str]:
    """Extract leading integer prefix and row key from filename stem."""
    match = re.match(r"^\s*(\d+)\s*[_\-\s]\s*(.*)$", path.stem)
    if not match:
        return None, path.stem.strip().lower()
    prefix = int(match.group(1))
    remainder = match.group(2).strip().lower()
    row_key = remainder if remainder else path.stem.strip().lower()
    return prefix, row_key


def image_sort_key(path: Path) -> tuple[str, int, str]:
    """Sort by row key first, then prefix, then full name."""
    prefix, row_key = split_prefix_and_row_key(path)
    prefix_sort = prefix if prefix is not None else 10**9
    return row_key, prefix_sort, path.name.lower()


def find_images(folder: Path, sort: bool = True) -> list[Path]:
    """Return image files from folder, optionally sorted alphabetically by row key."""
    paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if sort:
        return sorted(paths, key=image_sort_key)
    return list(paths)


def open_images(paths: Iterable[Path]) -> list[tuple[Path, Image.Image]]:
    """Open files as PIL images (RGB), preserving source path."""
    images: list[tuple[Path, Image.Image]] = []
    for p in paths:
        with Image.open(p) as im:
            images.append((p, im.convert("RGB")))
    return images


def combine_grid(
    images_with_paths: list[tuple[Path, Image.Image]],
    columns: int,
    cell_padding: int = 8,
    background: tuple[int, int, int] = (0, 0, 0),
    simple_layout: bool = False,
) -> Image.Image:
    """Create a grid image.

    If simple_layout is True: fill row by row, left to right, using the given
    number of columns. Works with any column count and preserves image order.

    If simple_layout is False: use prefix-based columns and name-based rows
    (original behavior for filenames like "1_foo.png").
    """
    if not images_with_paths:
        raise ValueError("No images were provided.")
    if columns <= 0:
        raise ValueError("Columns must be greater than 0.")

    max_w = max(im.width for _, im in images_with_paths)
    max_h = max(im.height for _, im in images_with_paths)

    cell_w = max_w + 2 * cell_padding
    cell_h = max_h + 2 * cell_padding

    if simple_layout:
        # Fill row by row, left to right; any number of columns
        n = len(images_with_paths)
        rows = (n + columns - 1) // columns
        grid_w = columns * cell_w
        grid_h = rows * cell_h
        canvas = Image.new("RGB", (grid_w, grid_h), color=background)

        for i, (path, im) in enumerate(images_with_paths):
            row, col = divmod(i, columns)
            x0 = col * cell_w
            y0 = row * cell_h
            x = x0 + (cell_w - im.width) // 2
            y = y0 + (cell_h - im.height) // 2
            canvas.paste(im, (x, y))

        return canvas

    # Original prefix-based layout
    row_to_images: dict[str, dict[int, Image.Image]] = {}
    for path, im in images_with_paths:
        prefix, row_key = split_prefix_and_row_key(path)
        if prefix is None:
            col = 0
        else:
            col = prefix % columns
        if row_key not in row_to_images:
            row_to_images[row_key] = {}
        row_to_images[row_key][col] = im

    row_keys = sorted(row_to_images.keys())
    rows = len(row_keys)
    if rows == 0:
        raise ValueError("No images were provided.")

    grid_w = columns * cell_w
    grid_h = rows * cell_h

    canvas = Image.new("RGB", (grid_w, grid_h), color=background)

    for row, row_key in enumerate(row_keys):
        for col in range(columns):
            im = row_to_images[row_key].get(col)
            if im is None:
                continue
            x0 = col * cell_w
            y0 = row * cell_h

            # Center image in each cell without distortion.
            x = x0 + (cell_w - im.width) // 2
            y = y0 + (cell_h - im.height) // 2
            canvas.paste(im, (x, y))

    return canvas


def parse_args() -> argparse.Namespace:
    default_input = Path(r"C:\Users\PahnkeLab\Desktop\maldi_msi\annotation-images")

    parser = argparse.ArgumentParser(
        description="Combine all images in a folder into one grid image."
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=default_input,
        help=f"Folder containing images (default: {default_input})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output image path (default: same as input folder, combined_grid_<cols>cols_<n>images.png)",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of columns in the output grid (default: 4)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Padding (pixels) around each image in its cell (default: 8)",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Preserve filesystem order instead of alphabetical sorting",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple row-by-row layout (fill left to right, any column count); ignores filename prefix parsing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_folder: Path = args.input_folder

    if not input_folder.exists():
        raise SystemExit(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise SystemExit(f"Input path is not a folder: {input_folder}")

    image_paths = find_images(input_folder, sort=not args.no_sort)
    if not image_paths:
        raise SystemExit(f"No supported image files found in: {input_folder}")

    output_path = args.output_path
    if output_path is None:
        output_path = input_folder / f"combined_grid_{args.columns}cols_{len(image_paths)}images.png"
    elif not output_path.is_absolute():
        output_path = input_folder / output_path

    images_with_paths = open_images(image_paths)
    combined = combine_grid(
        images_with_paths,
        columns=args.columns,
        cell_padding=args.padding,
        simple_layout=args.simple,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)
    print(f"Saved combined image: {output_path}")
    print(f"Images combined: {len(image_paths)} | columns: {args.columns}")


if __name__ == "__main__":
    main()
