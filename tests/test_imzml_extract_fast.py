"""Accuracy tests for fast imzML extraction (synthetic; no large files)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msi_visual.extract.imzml_fast import (
    ImzmlFastExtractor,
    accumulate_spectrum,
    accumulate_spectrum_slow,
    compute_bins,
    mz_axis,
    n_channels_for,
    select_geometry,
)


def test_compute_bins_matches_legacy_formula():
    mzs = np.array([50.0, 50.2, 50.199, 999.9], dtype=np.float64)
    min_mz, bins_per_mz = 50.0, 5
    legacy = np.int32(np.round((np.float32(mzs) - min_mz) * bins_per_mz))
    assert np.array_equal(compute_bins(mzs, min_mz, bins_per_mz), legacy)


def test_accumulate_fast_matches_slow():
    rng = np.random.default_rng(0)
    n_bins = 200
    img_fast = np.zeros((3, 4, n_bins), dtype=np.float32)
    img_slow = np.zeros_like(img_fast)
    for _ in range(20):
        y, x = int(rng.integers(0, 3)), int(rng.integers(0, 4))
        mzs = rng.uniform(50, 80, size=500)
        intens = rng.random(500).astype(np.float64) * 1000
        accumulate_spectrum(img_fast, y, x, mzs, intens, 50.0, 5)
        accumulate_spectrum_slow(img_slow, y, x, mzs, intens, 50.0, 5)
    assert np.allclose(img_fast, img_slow, rtol=1e-5, atol=1e-2)


def test_select_geometry_max_rows():
    coords = [(10 + x, 5 + y, 1) for y in range(8) for x in range(5)]
    geom = select_geometry(coords, max_rows=3)
    assert geom.height == 3
    assert geom.width == 5
    assert len(geom.indices) == 3 * 5
    # all selected rows are in [0, 3)
    for idx in geom.indices:
        assert 0 <= int(geom.ys0[idx] - geom.y_min) < 3


def test_mz_axis_length():
    min_mz, max_mz, bins = 50.0, 100.0, 5
    n = n_channels_for(min_mz, max_mz, bins)
    axis = mz_axis(min_mz, max_mz, bins, n)
    assert len(axis) == n
    assert axis[0] == 50.0


def _write_minimal_imzml_ibd(tmp_path: Path) -> Path:
    """
    Build a tiny continuous imzML + ibd with 2x3 pixels of known spectra.

    Layout (x,y):
      (1,1) (2,1) (3,1)
      (1,2) (2,2) (3,2)
    Each spectrum: peaks at mz={10.0, 10.2} with intensity = 10*x + y and 1.0.
    """
    coords = [(x, y, 1) for y in (1, 2) for x in (1, 2, 3)]
    spectra = []
    for x, y, _ in coords:
        mzs = np.array([10.0, 10.2], dtype=np.float64)
        intens = np.array([10.0 * x + y, 1.0], dtype=np.float32)
        spectra.append((mzs, intens))

    ibd = tmp_path / "toy.ibd"
    mz_offsets, mz_lengths = [], []
    int_offsets, int_lengths = [], []
    with open(ibd, "wb") as f:
        # 16-byte UUID placeholder header often present; offsets absolute.
        f.write(b"\x00" * 16)
        for mzs, intens in spectra:
            mz_offsets.append(f.tell())
            mz_bytes = mzs.astype("<f8").tobytes()
            f.write(mz_bytes)
            mz_lengths.append(len(mzs))
            int_offsets.append(f.tell())
            int_bytes = intens.astype("<f4").tobytes()
            f.write(int_bytes)
            int_lengths.append(len(intens))

    # Minimal imzML that pyimzml can parse is heavy; instead unit-test through
    # PortableSpectrumReader + extract_array internals without XML when possible.
    # Store sidecar for optional integration; return ibd path and reader via pickle-less construction.
    meta = {
        "coords": coords,
        "mz_offsets": mz_offsets,
        "mz_lengths": mz_lengths,
        "int_offsets": int_offsets,
        "int_lengths": int_lengths,
        "ibd": ibd,
    }
    np.save(tmp_path / "toy_meta.npy", meta, allow_pickle=True)
    return tmp_path


def test_fast_extractor_on_portable_reader(tmp_path: Path):
    from pyimzml.ImzMLParser import PortableSpectrumReader

    _write_minimal_imzml_ibd(tmp_path)
    meta = np.load(tmp_path / "toy_meta.npy", allow_pickle=True).item()
    reader = PortableSpectrumReader(
        meta["coords"],
        "d",
        meta["mz_offsets"],
        meta["mz_lengths"],
        "f",
        meta["int_offsets"],
        meta["int_lengths"],
    )
    ibd = Path(meta["ibd"])

    ex = ImzmlFastExtractor(min_mz=10.0, max_mz=12.0, bins_per_mz=5, num_workers=1)
    img_fast, mzs = ex.extract_array("unused.imzML", reader=reader, ibd=ibd, mode="fast")
    img_slow, _ = ex.extract_array("unused.imzML", reader=reader, ibd=ibd, mode="slow")

    assert img_fast.shape == (2, 3, n_channels_for(10.0, 12.0, 5))
    assert np.allclose(img_fast, img_slow, rtol=0, atol=1e-5)

    # Spot-check: pixel (x=3,y=2) -> norm (y=1,x=2), intensity 32 at mz 10.0
    b0 = int(compute_bins([10.0], 10.0, 5)[0])
    assert img_fast[1, 2, b0] == pytest.approx(32.0, abs=1e-5)


def test_multiprocess_matches_single(tmp_path: Path):
    from pyimzml.ImzMLParser import PortableSpectrumReader

    _write_minimal_imzml_ibd(tmp_path)
    meta = np.load(tmp_path / "toy_meta.npy", allow_pickle=True).item()
    reader = PortableSpectrumReader(
        meta["coords"],
        "d",
        meta["mz_offsets"],
        meta["mz_lengths"],
        "f",
        meta["int_offsets"],
        meta["int_lengths"],
    )
    ibd = Path(meta["ibd"])

    ex1 = ImzmlFastExtractor(min_mz=10.0, max_mz=12.0, bins_per_mz=5, num_workers=1)
    ex4 = ImzmlFastExtractor(min_mz=10.0, max_mz=12.0, bins_per_mz=5, num_workers=4)
    a, _ = ex1.extract_array("x", reader=reader, ibd=ibd, mode="fast")
    b, _ = ex4.extract_array("x", reader=reader, ibd=ibd, mode="fast")
    assert np.allclose(a, b, rtol=0, atol=1e-5)


def test_max_rows_slice(tmp_path: Path):
    from pyimzml.ImzMLParser import PortableSpectrumReader

    _write_minimal_imzml_ibd(tmp_path)
    meta = np.load(tmp_path / "toy_meta.npy", allow_pickle=True).item()
    reader = PortableSpectrumReader(
        meta["coords"],
        "d",
        meta["mz_offsets"],
        meta["mz_lengths"],
        "f",
        meta["int_offsets"],
        meta["int_lengths"],
    )
    ibd = Path(meta["ibd"])
    ex = ImzmlFastExtractor(min_mz=10.0, max_mz=12.0, bins_per_mz=5, max_rows=1)
    img, _ = ex.extract_array("x", reader=reader, ibd=ibd, mode="fast")
    assert img.shape[0] == 1
    assert img.shape[1] == 3
