from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MASS_NPY_NAMES = ("masses.npy", "mzs.npy", "mz.npy", "mass_axis.npy")


def _read_args_namespace(args_path: Path) -> Namespace | None:
    if not args_path.is_file():
        return None
    try:
        text = args_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = args_path.read_text(encoding="latin-1")
    try:
        return eval(text)  # noqa: S307
    except Exception as exc:
        logger.warning("Could not parse %s: %s", args_path, exc)
        return None


def mz_list_from_namespace(ns: Namespace) -> list[float]:
    """Build the full m/z list stored in an extraction ``args.txt`` namespace."""
    mzs = getattr(ns, "mzs", None)
    if mzs is not None:
        return [float(m) for m in mzs]
    start = float(ns.start_mz)
    end = float(ns.end_mz)
    bins = int(ns.bins)
    return [float(f"{mz:.3f}") for mz in np.arange(start, end + 1.0, 1.0 / float(bins))]


def mass_axis_from_args_txt(args_path: Path, n_mz: int) -> np.ndarray | None:
    """Load m/z axis from ``args.txt`` when length matches ``n_mz``."""
    ns = _read_args_namespace(args_path)
    if ns is None:
        return None
    axis = np.asarray(mz_list_from_namespace(ns), dtype=np.float64).reshape(-1)
    if axis.size != int(n_mz):
        logger.warning(
            "args.txt m/z length %d != spectrum width %d (%s)",
            axis.size,
            int(n_mz),
            args_path,
        )
        return None
    logger.info(
        "Loaded m/z axis from %s (%d bins, %.1f–%.1f Da)",
        args_path,
        axis.size,
        float(axis.min()),
        float(axis.max()),
    )
    return axis


def resolve_mass_axis(npy_path: Path | str | None, n_mz: int) -> np.ndarray:
    """Load m/z axis beside ``npy``: ``*.npy`` sidecars, then ``args.txt``, else bin indices."""
    if npy_path is None:
        return np.arange(int(n_mz), dtype=np.float64)
    npy_path = Path(npy_path)
    for name in _MASS_NPY_NAMES:
        candidate = npy_path.parent / name
        if candidate.is_file():
            axis = np.asarray(np.load(candidate), dtype=np.float64).reshape(-1)
            if axis.size == int(n_mz):
                return axis
    axis = mass_axis_from_args_txt(npy_path.parent / "args.txt", int(n_mz))
    if axis is not None:
        return axis
    logger.warning(
        "No m/z axis beside %s — using bin indices 0..%d",
        npy_path,
        int(n_mz) - 1,
    )
    return np.arange(int(n_mz), dtype=np.float64)


def get_extraction_mz_list(extraction_folder) -> list[float]:
    extraction_folder = Path(extraction_folder)
    ns = _read_args_namespace(extraction_folder / "args.txt")
    if ns is None:
        raise FileNotFoundError(f"Missing or invalid args.txt in {extraction_folder}")
    return mz_list_from_namespace(ns)
