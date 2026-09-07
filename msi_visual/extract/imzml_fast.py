"""
Fast imzML → dense HxWxC numpy extraction.

Optimizations vs ``PymzmlToNumpy`` / ``BaseMSIToNumpy.to_numpy``:
  * Stream spectra (no full peak-list materialization)
  * Vectorized scatter-add via ``np.add.at``
  * Optional row slice (``max_rows``) for fast iteration
  * Optional process pool over disjoint pixels using ``PortableSpectrumReader``
    (XML parsed once; workers only open the ``.ibd``)

Public API mirrors the dense-binning path of ``BaseMSIToNumpy`` so outputs can
be compared with ``np.allclose`` to the legacy extractor.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser, PortableSpectrumReader
from tqdm import tqdm

try:
    from multiprocessing import shared_memory
except ImportError:  # pragma: no cover
    shared_memory = None  # type: ignore

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ibd_path_for(imzml_path: str | Path) -> Path:
    p = Path(imzml_path)
    ibd = p.with_suffix(".ibd")
    if not ibd.is_file():
        # Some datasets use mismatched casing / alternate suffix naming.
        alt = p.parent / (p.stem + ".ibd")
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"No .ibd beside imzML: {p}")
    return ibd


def compute_bins(mzs, min_mz: float, bins_per_mz: float) -> np.ndarray:
    """Match legacy ``BaseMSIToNumpy`` binning: ``int32(round((float32(mzs)-min_mz)*bins))``."""
    return np.int32(np.round((np.float32(mzs) - float(min_mz)) * float(bins_per_mz)))


def accumulate_spectrum(
    img: np.ndarray,
    y: int,
    x: int,
    mzs,
    intensities,
    min_mz: float,
    bins_per_mz: float,
) -> None:
    """Accumulate one spectrum into ``img[y, x]`` (in-place)."""
    bins = compute_bins(mzs, min_mz, bins_per_mz)
    n = img.shape[-1]
    valid = (bins >= 0) & (bins < n)
    if not np.any(valid):
        return
    intens = np.asarray(intensities, dtype=img.dtype)
    np.add.at(img[y, x], bins[valid], intens[valid])


def accumulate_spectrum_slow(
    img: np.ndarray,
    y: int,
    x: int,
    mzs,
    intensities,
    min_mz: float,
    bins_per_mz: float,
) -> None:
    """Legacy per-peak Python loop (for accuracy / speed baselines)."""
    bins = compute_bins(mzs, min_mz, bins_per_mz)
    intens = np.asarray(intensities)
    n = img.shape[-1]
    for b, intensity in zip(bins, intens):
        bi = int(b)
        if 0 <= bi < n:
            img[y, x, bi] = img[y, x, bi] + intensity


def mz_axis(min_mz: float, max_mz: float, bins_per_mz: int, n_channels: int) -> list[float]:
    """Match legacy axis formatting in ``BaseMSIToNumpy.to_numpy``."""
    mzs = np.arange(float(min_mz), float(max_mz) + 1, 1.0 / float(bins_per_mz))
    return [float(f"{mzs[i]:.6f}") for i in range(n_channels)]


def n_channels_for(min_mz: float, max_mz: float, bins_per_mz: int) -> int:
    num_mzs = round(float(max_mz) - float(min_mz) + 1)
    return int(bins_per_mz) * int(num_mzs)


def discover_mz_range(
    reader: PortableSpectrumReader,
    ibd: Path,
    indices: Optional[Sequence[int]] = None,
) -> tuple[float, float]:
    """Stream spectra to find floor(min mz) / ceil(max mz), matching legacy auto bounds."""
    mn = np.inf
    mx = -np.inf
    idxs: Iterable[int]
    if indices is None:
        idxs = range(len(reader.coordinates))
    else:
        idxs = indices
    with open(ibd, "rb") as f:
        for idx in idxs:
            mzs, _ = reader.read_spectrum_from_file(f, int(idx))
            if mzs is None or len(mzs) == 0:
                continue
            mn = min(mn, float(np.min(mzs)))
            mx = max(mx, float(np.max(mzs)))
    if not np.isfinite(mn) or not np.isfinite(mx):
        raise ValueError("Could not discover m/z range (empty spectra)")
    return float(np.floor(mn)), float(np.ceil(mx))


@dataclass(frozen=True)
class ImzmlGeometry:
    coordinates: list[tuple[int, int, int]]
    xs0: np.ndarray  # original x
    ys0: np.ndarray  # original y
    x_min: int
    y_min: int
    width: int
    height: int
    indices: np.ndarray  # spectrum indices included (after max_rows filter)


def select_geometry(
    coordinates: Sequence[tuple[int, int, int]],
    *,
    max_rows: Optional[int] = None,
) -> ImzmlGeometry:
    xs0 = np.int32([c[0] for c in coordinates])
    ys0 = np.int32([c[1] for c in coordinates])
    x_min = int(xs0.min())
    y_min = int(ys0.min())
    width = int(xs0.max() - x_min + 1)
    height_full = int(ys0.max() - y_min + 1)

    if max_rows is None:
        indices = np.arange(len(coordinates), dtype=np.int64)
        height = height_full
    else:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("max_rows must be > 0")
        row = ys0 - y_min
        indices = np.flatnonzero(row < max_rows).astype(np.int64)
        height = min(max_rows, height_full)

    return ImzmlGeometry(
        coordinates=list(coordinates),
        xs0=xs0,
        ys0=ys0,
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
        indices=indices,
    )


def _norm_yx(geom: ImzmlGeometry, idx: int) -> tuple[int, int]:
    y = int(geom.ys0[idx] - geom.y_min)
    x = int(geom.xs0[idx] - geom.x_min)
    return y, x


def _worker_fill_shm(payload: dict) -> None:
    """Worker: read assigned spectra from .ibd into a shared-memory cube."""
    shm = shared_memory.SharedMemory(name=payload["shm_name"])
    try:
        img = np.ndarray(payload["shape"], dtype=np.dtype(payload["dtype"]), buffer=shm.buf)
        reader: PortableSpectrumReader = payload["reader"]
        geom_xs0 = payload["xs0"]
        geom_ys0 = payload["ys0"]
        x_min = payload["x_min"]
        y_min = payload["y_min"]
        min_mz = payload["min_mz"]
        bins_per_mz = payload["bins_per_mz"]
        indices = payload["indices"]
        with open(payload["ibd_path"], "rb") as f:
            for idx in indices:
                mzs, intensities = reader.read_spectrum_from_file(f, int(idx))
                y = int(geom_ys0[idx] - y_min)
                x = int(geom_xs0[idx] - x_min)
                accumulate_spectrum(img, y, x, mzs, intensities, min_mz, bins_per_mz)
    finally:
        shm.close()


class ImzmlFastExtractor:
    """
    Dense-binning imzML extractor.

    Parameters
    ----------
    min_mz, max_mz, bins_per_mz
        Same semantics as ``BaseMSIToNumpy`` dense path.
    num_workers
        Process count for pixel partitioning. ``1`` = single-process.
    max_rows
        If set, only pixels with ``(y - y_min) < max_rows`` are extracted into an
        image of height ``max_rows`` (or less if the slide is shorter).
    id
        Written into ``args.txt`` for compatibility with the legacy layout.
    """

    def __init__(
        self,
        *,
        min_mz: Optional[float],
        max_mz: Optional[float],
        bins_per_mz: int = 1,
        num_workers: int = 1,
        max_rows: Optional[int] = None,
        id: str = "",
    ):
        self.min_mz = None if min_mz is None else float(min_mz)
        self.max_mz = None if max_mz is None else float(max_mz)
        self.bins_per_mz = int(bins_per_mz)
        self.num_workers = max(1, int(num_workers))
        self.max_rows = None if max_rows is None else int(max_rows)
        self.id = str(id)

    def get_img_type(self):
        return np.float32

    def n_channels(self) -> int:
        if self.min_mz is None or self.max_mz is None:
            raise ValueError("min_mz/max_mz not set")
        return n_channels_for(self.min_mz, self.max_mz, self.bins_per_mz)

    def parse(self, input_path: str | Path) -> tuple[PortableSpectrumReader, Path]:
        _configure_logging()
        input_path = Path(input_path)
        logger.info("Parsing imzML (can take ~10–30s for large slides): %s", input_path)
        parser = ImzMLParser(str(input_path))
        reader = parser.portable_spectrum_reader()
        ibd = ibd_path_for(input_path)
        logger.info("Parsed %d spectra; ibd=%s", len(reader.coordinates), ibd)
        return reader, ibd

    def extract_array(
        self,
        input_path: str | Path,
        *,
        reader: Optional[PortableSpectrumReader] = None,
        ibd: Optional[Path] = None,
        mode: str = "fast",
    ) -> tuple[np.ndarray, list[float]]:
        """
        Return ``(img, mzs)`` without writing disk.

        mode:
          * ``fast`` — ``np.add.at`` (+ optional workers)
          * ``slow`` — legacy Python peak loop, single-process (baseline)
        """
        _configure_logging()
        input_path = Path(input_path)
        if reader is None or ibd is None:
            reader, ibd = self.parse(input_path)

        geom = select_geometry(reader.coordinates, max_rows=self.max_rows)

        if self.min_mz is None or self.max_mz is None:
            logger.info("Discovering m/z range from spectra…")
            mn, mx = discover_mz_range(reader, ibd, indices=geom.indices)
            if self.min_mz is None:
                self.min_mz = mn
            if self.max_mz is None:
                self.max_mz = mx
            logger.info("m/z range %.1f–%.1f", self.min_mz, self.max_mz)

        shape = (geom.height, geom.width, self.n_channels())
        dtype = self.get_img_type()
        n_bytes = int(shape[0]) * int(shape[1]) * int(shape[2]) * int(np.dtype(dtype).itemsize)
        logger.info(
            "Extract mode=%s workers=%s pixels=%d shape=%s (%.2f GiB)",
            mode,
            self.num_workers,
            len(geom.indices),
            shape,
            n_bytes / (1024**3),
        )
        if n_bytes > 8 * (1024**3):
            logger.warning(
                "Large output cube (%.1f GiB). Expect long SHM alloc/copy + save; "
                "consider fewer bins or more RAM.",
                n_bytes / (1024**3),
            )

        if mode == "slow":
            img = self._extract_slow(reader, ibd, geom, shape, dtype)
        elif mode == "fast":
            if self.num_workers == 1 or len(geom.indices) < 2:
                img = self._extract_fast_single(reader, ibd, geom, shape, dtype)
            else:
                img = self._extract_fast_pool(reader, ibd, geom, shape, dtype)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        logger.info("Building m/z axis (%d channels)…", shape[-1])
        mzs = mz_axis(self.min_mz, self.max_mz, self.bins_per_mz, shape[-1])
        return np.float32(img), mzs

    def _extract_slow(
        self,
        reader: PortableSpectrumReader,
        ibd: Path,
        geom: ImzmlGeometry,
        shape: tuple[int, int, int],
        dtype,
    ) -> np.ndarray:
        img = np.zeros(shape, dtype=dtype)
        with open(ibd, "rb") as f:
            for idx in tqdm(geom.indices, desc="bin spectra (slow)", mininterval=1.0):
                mzs, intensities = reader.read_spectrum_from_file(f, int(idx))
                y, x = _norm_yx(geom, int(idx))
                accumulate_spectrum_slow(
                    img, y, x, mzs, intensities, self.min_mz, self.bins_per_mz
                )
        return img

    def _extract_fast_single(
        self,
        reader: PortableSpectrumReader,
        ibd: Path,
        geom: ImzmlGeometry,
        shape: tuple[int, int, int],
        dtype,
    ) -> np.ndarray:
        logger.info("Single-process binning…")
        img = np.zeros(shape, dtype=dtype)
        with open(ibd, "rb") as f:
            for idx in tqdm(geom.indices, desc="bin spectra", mininterval=1.0):
                mzs, intensities = reader.read_spectrum_from_file(f, int(idx))
                y, x = _norm_yx(geom, int(idx))
                accumulate_spectrum(
                    img, y, x, mzs, intensities, self.min_mz, self.bins_per_mz
                )
        return img

    def _extract_fast_pool(
        self,
        reader: PortableSpectrumReader,
        ibd: Path,
        geom: ImzmlGeometry,
        shape: tuple[int, int, int],
        dtype,
    ) -> np.ndarray:
        if shared_memory is None:
            return self._extract_fast_single(reader, ibd, geom, shape, dtype)

        # Spawn + SHM overhead dominates on small slices; fall back.
        if len(geom.indices) < 2000 or self.num_workers <= 1:
            return self._extract_fast_single(reader, ibd, geom, shape, dtype)

        # Windows/numpy: np.prod on int32 overflows for large HxWxC cubes (~5GB+).
        n_bytes = (
            int(shape[0]) * int(shape[1]) * int(shape[2]) * int(np.dtype(dtype).itemsize)
        )
        logger.info("Allocating shared memory (%.2f GiB)…", n_bytes / (1024**3))
        shm = shared_memory.SharedMemory(create=True, size=n_bytes)
        try:
            img = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            logger.info("Zero-filling shared cube…")
            img.fill(0)

            indices = geom.indices
            n_workers = min(self.num_workers, max(1, len(indices)))
            chunks = np.array_split(indices, n_workers)
            payloads = []
            for chunk in chunks:
                if len(chunk) == 0:
                    continue
                payloads.append(
                    {
                        "shm_name": shm.name,
                        "shape": shape,
                        "dtype": np.dtype(dtype).str,
                        "reader": reader,
                        "ibd_path": str(ibd),
                        "xs0": geom.xs0,
                        "ys0": geom.ys0,
                        "x_min": geom.x_min,
                        "y_min": geom.y_min,
                        "min_mz": self.min_mz,
                        "bins_per_mz": self.bins_per_mz,
                        "indices": np.asarray(chunk, dtype=np.int64),
                    }
                )

            logger.info("Spawning %d workers…", len(payloads))
            ctx = get_context("spawn")
            with ctx.Pool(processes=len(payloads)) as pool:
                pool.map(_worker_fill_shm, payloads)
            logger.info("Workers finished; copying cube out of shared memory…")

            # Copy out before shm teardown.
            return np.array(img, dtype=dtype, copy=True)
        finally:
            shm.close()
            shm.unlink()

    def save_numpy(self, img: np.ndarray, output_path: str | Path, region: int = 0) -> Path:
        _configure_logging()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        out = output_path / f"{region}.npy"
        n_giB = img.nbytes / (1024**3)
        logger.info("Saving %s (%.2f GiB)…", out, n_giB)
        np.save(out, img)
        logger.info("Saved %s", out)
        return out

    def save_extraction_args(
        self,
        mzs: list[float],
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        _configure_logging()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        extraction_args = Namespace(
            path=str(input_path),
            start_mz=self.min_mz,
            end_mz=self.max_mz,
            bins=self.bins_per_mz,
            nonzero=False,
            mzs=mzs,
            id=self.id,
        )
        args_path = output_path / "args.txt"
        logger.info("Writing %s (%d m/z labels)…", args_path, len(mzs))
        with open(args_path, "w", encoding="utf-8") as f:
            f.write(str(extraction_args))
        logger.info("Wrote %s", args_path)

    def __call__(self, input_path: str | Path, output_path: str | Path, region: int = 0):
        _configure_logging()
        logger.info("=== imzML fast extract start ===")
        img, mzs = self.extract_array(input_path, mode="fast")
        self.save_numpy(img, output_path, region=region)
        self.save_extraction_args(mzs, input_path, output_path)
        logger.info("=== imzML fast extract done ===")
        return img, mzs


def extract_imzml(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    min_mz: float,
    max_mz: float,
    bins_per_mz: int = 1,
    num_workers: int = 1,
    max_rows: Optional[int] = None,
    id: str = "",
    mode: str = "fast",
) -> tuple[np.ndarray, list[float]]:
    """Convenience wrapper around ``ImzmlFastExtractor``."""
    ex = ImzmlFastExtractor(
        min_mz=min_mz,
        max_mz=max_mz,
        bins_per_mz=bins_per_mz,
        num_workers=num_workers,
        max_rows=max_rows,
        id=id,
    )
    img, mzs = ex.extract_array(input_path, mode=mode)
    if output_path is not None:
        ex.save_numpy(img, output_path, region=0)
        ex.save_extraction_args(mzs, input_path, output_path)
    return img, mzs
