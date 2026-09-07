"""
Fast Bruker TIMS/TSF → dense HxWxC numpy extraction.

Optimizations vs ``BrukerTimsToNumpy`` / ``BrukerTsfToNumpy``:
  * Stream frames (no full peak-list materialization)
  * Vectorized scatter-add via ``np.add.at``
  * TIMS: numpy concatenate over mobility scans (no Python ``list.extend``)
  * Optional ``max_rows`` slice and process pool over disjoint frames
  * Each worker opens its own SDK handle (DLL is not shared across processes)

Dense-binning semantics match ``BaseMSIToNumpy`` so outputs can be compared to
legacy extractors / existing ``.npy`` goldens.
"""

from __future__ import annotations

import os
import sqlite3
from argparse import Namespace
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from msi_visual.extract.imzml_fast import (
    accumulate_spectrum,
    accumulate_spectrum_slow,
    mz_axis,
    n_channels_for,
)

try:
    from multiprocessing import shared_memory
except ImportError:  # pragma: no cover
    shared_memory = None  # type: ignore

BrukerKind = Literal["tims", "tsf"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_bruker_cwd() -> None:
    """timsdata/tsfdata load ``./timsdata.dll`` from CWD."""
    root = _repo_root()
    dll = root / "timsdata.dll"
    if dll.is_file():
        os.chdir(root)


@dataclass(frozen=True)
class BrukerRegionTable:
    frames: np.ndarray  # int64 Frame ids
    xs0: np.ndarray
    ys0: np.ndarray
    x_min: int
    y_min: int
    width: int
    height: int
    indices: np.ndarray  # positions into frames/xs0/ys0 after max_rows filter


def load_region_table(
    analysis_dir: str | Path,
    region: int,
    *,
    max_rows: Optional[int] = None,
) -> BrukerRegionTable:
    """Read MaldiFrameInfo for one region (uses Frame column, not fragile offsets)."""
    analysis_dir = Path(analysis_dir)
    # analysis.tdf / analysis.tsf are sqlite
    db = analysis_dir / "analysis.tdf"
    if not db.is_file():
        db = analysis_dir / "analysis.tsf"
    if not db.is_file():
        raise FileNotFoundError(f"No analysis.tdf/tsf in {analysis_dir}")

    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT Frame, XIndexPos, YIndexPos FROM MaldiFrameInfo "
            "WHERE RegionNumber=? ORDER BY Frame",
            (int(region),),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"No MaldiFrameInfo rows for region={region}")

    frames = np.int64([r[0] for r in rows])
    xs0 = np.int32([r[1] for r in rows])
    ys0 = np.int32([r[2] for r in rows])
    x_min = int(xs0.min())
    y_min = int(ys0.min())
    width = int(xs0.max() - x_min + 1)
    height_full = int(ys0.max() - y_min + 1)

    if max_rows is None:
        indices = np.arange(len(frames), dtype=np.int64)
        height = height_full
    else:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("max_rows must be > 0")
        row = ys0 - y_min
        indices = np.flatnonzero(row < max_rows).astype(np.int64)
        height = min(max_rows, height_full)

    return BrukerRegionTable(
        frames=frames,
        xs0=xs0,
        ys0=ys0,
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
        indices=indices,
    )


def list_regions(analysis_dir: str | Path) -> list[int]:
    analysis_dir = Path(analysis_dir)
    db = analysis_dir / "analysis.tdf"
    if not db.is_file():
        db = analysis_dir / "analysis.tsf"
    conn = sqlite3.connect(str(db))
    try:
        regs = [int(r[0]) for r in conn.execute("SELECT DISTINCT RegionNumber FROM MaldiFrameInfo")]
    finally:
        conn.close()
    return sorted(regs)


def detect_kind(analysis_dir: str | Path) -> BrukerKind:
    analysis_dir = Path(analysis_dir)
    if (analysis_dir / "analysis.tdf").is_file():
        return "tims"
    if (analysis_dir / "analysis.tsf").is_file():
        return "tsf"
    raise FileNotFoundError(f"Not a Bruker .d folder (no analysis.tdf/tsf): {analysis_dir}")


def _open_reader(kind: BrukerKind, analysis_dir: str | Path):
    _ensure_bruker_cwd()
    if kind == "tims":
        from msi_visual.extract import timsdata

        return timsdata.TimsData(str(analysis_dir))
    from msi_visual.extract import tsfdata

    return tsfdata.TsfData(str(analysis_dir))


def read_frame_spectrum(kind: BrukerKind, reader, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mzs, intensities) for one MALDI frame."""
    if kind == "tsf":
        indices, intensities = reader.readLineSpectrum(int(frame_id))
        mzs = reader.indexToMz(int(frame_id), indices)
        return np.asarray(mzs), np.asarray(intensities, dtype=np.float32)

    # TIMS: sum all mobility scans (legacy semantics), but with numpy concat.
    q = reader.conn.execute("SELECT NumScans FROM Frames WHERE Id={0}".format(int(frame_id)))
    num_scans = int(q.fetchone()[0])
    mz_parts: list[np.ndarray] = []
    int_parts: list[np.ndarray] = []
    for scan in reader.readScans(int(frame_id), 0, num_scans):
        index = np.asarray(scan[0], dtype=np.float64)
        mz_parts.append(np.asarray(reader.indexToMz(int(frame_id), index)))
        int_parts.append(np.asarray(scan[1]))
    if not mz_parts:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.concatenate(mz_parts), np.concatenate(int_parts)


def _worker_fill_shm(payload: dict) -> None:
    shm = shared_memory.SharedMemory(name=payload["shm_name"])
    try:
        img = np.ndarray(payload["shape"], dtype=np.dtype(payload["dtype"]), buffer=shm.buf)
        kind: BrukerKind = payload["kind"]
        reader = _open_reader(kind, payload["analysis_dir"])
        try:
            frames = payload["frames"]
            xs0 = payload["xs0"]
            ys0 = payload["ys0"]
            x_min = payload["x_min"]
            y_min = payload["y_min"]
            min_mz = payload["min_mz"]
            bins_per_mz = payload["bins_per_mz"]
            for i in payload["indices"]:
                i = int(i)
                y = int(ys0[i] - y_min)
                x = int(xs0[i] - x_min)
                mzs, intens = read_frame_spectrum(kind, reader, int(frames[i]))
                accumulate_spectrum(img, y, x, mzs, intens, min_mz, bins_per_mz)
        finally:
            # TimsData/TsfData close on del / context if available
            if hasattr(reader, "close"):
                reader.close()
            del reader
    finally:
        shm.close()


class BrukerFastExtractor:
    """
    Dense-binning Bruker extractor for TIMS (``analysis.tdf``) or TSF (``analysis.tsf``).
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
        kind: Optional[BrukerKind] = None,
    ):
        self.min_mz = None if min_mz is None else float(min_mz)
        self.max_mz = None if max_mz is None else float(max_mz)
        self.bins_per_mz = int(bins_per_mz)
        self.num_workers = max(1, int(num_workers))
        self.max_rows = None if max_rows is None else int(max_rows)
        self.id = str(id)
        self.kind_override = kind

    def get_img_type(self, kind: BrukerKind):
        # Match legacy: TIMS accumulates uint32 then cast to float32 on save;
        # TSF uses float32 throughout. We accumulate in float32 for both so
        # SharedMemory workers stay simple; values remain integral for TIMS counts.
        return np.float32

    def n_channels(self) -> int:
        if self.min_mz is None or self.max_mz is None:
            raise ValueError("min_mz/max_mz not set")
        return n_channels_for(self.min_mz, self.max_mz, self.bins_per_mz)

    def extract_region_array(
        self,
        input_path: str | Path,
        region: int = 0,
        *,
        mode: str = "fast",
    ) -> tuple[np.ndarray, list[float]]:
        input_path = Path(input_path)
        kind = self.kind_override or detect_kind(input_path)
        table = load_region_table(input_path, region, max_rows=self.max_rows)

        if self.min_mz is None or self.max_mz is None:
            raise ValueError("BrukerFastExtractor requires min_mz and max_mz for dense binning")

        shape = (table.height, table.width, self.n_channels())
        dtype = self.get_img_type(kind)

        if mode == "slow":
            img = self._extract_single(kind, input_path, table, shape, dtype, slow=True)
        elif mode == "fast":
            # TIMS is SDK-bound: parallelize even on modest slices.
            min_for_pool = 200 if kind == "tims" else 2000
            if self.num_workers <= 1 or len(table.indices) < min_for_pool:
                img = self._extract_single(kind, input_path, table, shape, dtype, slow=False)
            else:
                img = self._extract_pool(kind, input_path, table, shape, dtype)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        mzs = mz_axis(self.min_mz, self.max_mz, self.bins_per_mz, shape[-1])
        return np.float32(img), mzs

    def _extract_single(
        self,
        kind: BrukerKind,
        input_path: Path,
        table: BrukerRegionTable,
        shape: tuple[int, int, int],
        dtype,
        *,
        slow: bool,
    ) -> np.ndarray:
        img = np.zeros(shape, dtype=dtype)
        reader = _open_reader(kind, input_path)
        try:
            for i in table.indices:
                i = int(i)
                y = int(table.ys0[i] - table.y_min)
                x = int(table.xs0[i] - table.x_min)
                mzs, intens = read_frame_spectrum(kind, reader, int(table.frames[i]))
                if slow:
                    accumulate_spectrum_slow(
                        img, y, x, mzs, intens, self.min_mz, self.bins_per_mz
                    )
                else:
                    accumulate_spectrum(
                        img, y, x, mzs, intens, self.min_mz, self.bins_per_mz
                    )
        finally:
            if hasattr(reader, "close"):
                reader.close()
            del reader
        return img

    def _extract_pool(
        self,
        kind: BrukerKind,
        input_path: Path,
        table: BrukerRegionTable,
        shape: tuple[int, int, int],
        dtype,
    ) -> np.ndarray:
        if shared_memory is None:
            return self._extract_single(kind, input_path, table, shape, dtype, slow=False)

        n_bytes = int(shape[0]) * int(shape[1]) * int(shape[2]) * int(np.dtype(dtype).itemsize)
        shm = shared_memory.SharedMemory(create=True, size=n_bytes)
        try:
            img = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            img.fill(0)

            indices = table.indices
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
                        "kind": kind,
                        "analysis_dir": str(input_path),
                        "frames": table.frames,
                        "xs0": table.xs0,
                        "ys0": table.ys0,
                        "x_min": table.x_min,
                        "y_min": table.y_min,
                        "min_mz": self.min_mz,
                        "bins_per_mz": self.bins_per_mz,
                        "indices": np.asarray(chunk, dtype=np.int64),
                    }
                )

            ctx = get_context("spawn")
            with ctx.Pool(processes=len(payloads)) as pool:
                pool.map(_worker_fill_shm, payloads)
            return np.array(img, dtype=dtype, copy=True)
        finally:
            shm.close()
            shm.unlink()

    def save_numpy(self, img: np.ndarray, output_path: str | Path, region: int = 0) -> Path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        out = output_path / f"{region}.npy"
        np.save(out, img)
        return out

    def save_extraction_args(
        self,
        mzs: list[float],
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
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
        with open(output_path / "args.txt", "w", encoding="utf-8") as f:
            f.write(str(extraction_args))

    def __call__(self, input_path: str | Path, output_path: str | Path):
        input_path = Path(input_path)
        regions = list_regions(input_path)
        last = None
        for region in regions:
            img, mzs = self.extract_region_array(input_path, region=region, mode="fast")
            self.save_numpy(img, output_path, region=region)
            self.save_extraction_args(mzs, input_path, output_path)
            last = (img, mzs)
        return last
