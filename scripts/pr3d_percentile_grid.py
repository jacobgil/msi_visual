#!/usr/bin/env python3
"""
Run Percentile Ratio 3-channel (PR3D) visualization on an MSI ``.npy`` cube for 16 percentile
combinations and save **two** figure grids (4x4): non-equalized and equalized (LAB L channel).

Uses ``msi_visual.percentile_ratio.PercentileRatio``: three ratios (p0/p1, p2/p3, p4/p5) mapped to
RGB after LAB-style assembly (see that module).

The 16 presets vary the first two ratio pairs on a 4x4 grid; the third pair (p4, p5) is fixed at
(98.0, 85.0) so each cell is distinct but comparable.

Outputs (default ``--out`` stem):
  - ``<stem>.png`` — raw PR3D RGB
  - ``<stem>_equalized.png`` — same panels after histogram equalization on L in LAB

Example:
  python scripts/pr3d_percentile_grid.py path/to/cube.npy --out pr3d_grid.png
  python scripts/pr3d_percentile_grid.py path/to/cube.npy --transpose-msi --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from msi_visual.percentile_ratio import PercentileRatio

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


# Numerator/denominator pairs for ratio a = p0/p1 (higher percentiles = brighter bins in sorted spectrum).
_GRID_A: list[tuple[float, float]] = [
    (99.99, 99.9),
    (99.9, 99.5),
    (99.5, 99.0),
    (99.0, 97.5),
]
# Pairs for ratio b = p2/p3
_GRID_B: list[tuple[float, float]] = [
    (99.9, 99.0),
    (99.0, 98.0),
    (98.0, 96.0),
    (96.0, 92.0),
]
# Fixed third ratio c = p4/p5 (same in all 16 panels for a stable chroma channel).
_P4_P5: tuple[float, float] = (98.0, 85.0)


def _percentile_presets_16() -> list[list[float]]:
    out: list[list[float]] = []
    for r in range(4):
        for c in range(4):
            p0, p1 = _GRID_A[r]
            p2, p3 = _GRID_B[c]
            p4, p5 = _P4_P5
            out.append([p0, p1, p2, p3, p4, p5])
    return out


def _load_msi(path: Path, *, transpose_msi: bool) -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    elif img.ndim != 3:
        raise ValueError(f"Expected HxWxC (or CxHxW with --transpose-msi), got shape {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    return np.asarray(img, dtype=np.float32)


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    s = msi.sum(axis=-1, keepdims=True) + 1e-8
    return (msi / s).astype(np.float32, copy=False)


def _equalize_rgb_u8(vis: np.ndarray) -> np.ndarray:
    """Histogram-equalize luminance in LAB; keeps color channels, improves contrast."""
    import cv2

    arr = np.asarray(vis)
    if arr.ndim == 2:
        return cv2.equalizeHist(arr.astype(np.uint8))
    rgb = arr.astype(np.uint8)
    if rgb.shape[-1] != 3:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.equalizeHist(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return out


def _render_grid_figure(
    *,
    img: np.ndarray,
    presets: list[list[float]],
    frames: list[np.ndarray | None],
    panel_errors: list[str | None],
    npy_path: Path,
    normalization: str,
    figsize: float,
    dpi: int,
    suptitle_suffix: str,
) -> object:
    import matplotlib.pyplot as plt

    n = len(presets)
    if n == 16:
        nrows, ncols = 4, 4
    else:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    fig_w = float(figsize)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_w * nrows / ncols),
        facecolor="black",
        subplot_kw={"facecolor": "black"},
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for i, pct in enumerate(presets):
        ax = axes_flat[i]
        vis = frames[i]
        if vis is None:
            err = panel_errors[i] if i < len(panel_errors) else ""
            ax.text(
                0.5,
                0.5,
                f"error:\n{err}" if err else "missing",
                ha="center",
                va="center",
                fontsize=7,
                color="lightcoral",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        if vis.ndim == 2:
            ax.imshow(vis, cmap="gray")
        else:
            ax.imshow(np.asarray(vis))
        title = (
            f"[{i}]  "
            f"{pct[0]:.2f}/{pct[1]:.2f} | "
            f"{pct[2]:.2f}/{pct[3]:.2f} | "
            f"{pct[4]:.2f}/{pct[5]:.2f}"
        )
        ax.set_title(title, fontsize=7, color="0.85", pad=2)
        ax.axis("off")
        ax.margins(0)

    for j in range(len(presets), len(axes_flat)):
        axes_flat[j].set_axis_off()
        axes_flat[j].set_facecolor("black")

    fig.suptitle(
        f"PercentileRatio (PR3D) {suptitle_suffix} | {npy_path.name} | shape={tuple(img.shape)} | norm={normalization}",
        fontsize=11,
        color="white",
        y=0.995,
    )
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.935, wspace=0.004, hspace=0.07)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npy_path", type=Path, help="Path to MSI .npy (HxWxC or CxHxW with --transpose-msi)")
    ap.add_argument("--out", type=Path, default=Path("pr3d_percentile_grid.png"), help="Output PNG path")
    ap.add_argument("--transpose-msi", action="store_true", help="Treat array on disk as CxHxW")
    ap.add_argument(
        "--normalization",
        choices=("tic", "none"),
        default="tic",
        help="Per-pixel TIC normalize before PR3D, or use raw float32 (default: tic)",
    )
    ap.add_argument("--dpi", type=int, default=120, help="Figure DPI when saving PNG")
    ap.add_argument("--figsize", type=float, default=18.0, help="Figure width/height in inches (square)")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    npy_path = args.npy_path
    if not npy_path.is_file():
        logger.error("File not found: %s", npy_path)
        sys.exit(1)

    logger.info("Loading %s", npy_path.resolve())
    img = _load_msi(npy_path, transpose_msi=args.transpose_msi)
    logger.info("Array shape=%s dtype=%s transpose_msi=%s", tuple(img.shape), img.dtype, args.transpose_msi)
    if args.normalization == "tic":
        img = _tic_normalize(img)
        logger.info("Applied TIC normalization")
    else:
        logger.info("No intensity normalization (raw float32)")

    presets = _percentile_presets_16()
    logger.info("Rendering %d PercentileRatio panels (raw + equalized)", len(presets))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required: pip install matplotlib")
        raise SystemExit(1) from None

    raw_frames: list[np.ndarray | None] = []
    eq_frames: list[np.ndarray | None] = []
    panel_errors: list[str | None] = []

    for i, pct in enumerate(tqdm(presets, desc="PR3D panels", unit="panel")):
        pr = PercentileRatio(percentiles=pct)
        try:
            vis = pr(img)
            raw_frames.append(np.asarray(vis))
            eq_frames.append(_equalize_rgb_u8(vis))
            panel_errors.append(None)
            logger.debug("panel %d percentiles=%s ok", i, pct)
        except Exception as exc:
            err_s = str(exc)
            logger.warning("panel %d failed: %s", i, err_s)
            raw_frames.append(None)
            eq_frames.append(None)
            panel_errors.append(err_s)

    out_path = args.out
    out_eq = out_path.parent / f"{out_path.stem}_equalized{out_path.suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for path, suffix, frames in (
        (out_path, "(raw)", raw_frames),
        (out_eq, "(equalized L)", eq_frames),
    ):
        fig = _render_grid_figure(
            img=img,
            presets=presets,
            frames=frames,
            panel_errors=panel_errors,
            npy_path=npy_path,
            normalization=args.normalization,
            figsize=args.figsize,
            dpi=args.dpi,
            suptitle_suffix=suffix,
        )
        logger.info("Saving %s dpi=%d -> %s", suffix.strip("()"), args.dpi, path.resolve())
        fig.savefig(
            path,
            dpi=args.dpi,
            bbox_inches="tight",
            facecolor="black",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info("Wrote %s", path.resolve())


if __name__ == "__main__":
    main()
