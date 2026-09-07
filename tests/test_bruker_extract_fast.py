"""Unit tests for fast Bruker extraction helpers (no large .d required)."""

from __future__ import annotations

import numpy as np

from msi_visual.extract.bruker_fast import BrukerRegionTable, load_region_table
from msi_visual.extract.imzml_fast import accumulate_spectrum, accumulate_spectrum_slow


def test_tims_accumulate_matches_slow_uint_like():
    rng = np.random.default_rng(1)
    img_f = np.zeros((2, 2, 100), dtype=np.float32)
    img_s = np.zeros_like(img_f)
    mzs = rng.uniform(300, 320, size=200)
    intens = rng.integers(1, 50, size=200).astype(np.float64)
    accumulate_spectrum(img_f, 0, 1, mzs, intens, 300.0, 5)
    accumulate_spectrum_slow(img_s, 0, 1, mzs, intens, 300.0, 5)
    assert np.allclose(img_f, img_s, rtol=0, atol=1e-3)


def test_load_region_table_max_rows_on_real_if_present():
    path = r"E:\MSImaging-data\MSI_slides\2024-02-15_NRL4485_slide2_lipid_neg_timsON\Slide2_lipid_neg_TIMS_on.d"
    import os

    if not os.path.isdir(path):
        return
    full = load_region_table(path, 0, max_rows=None)
    sliced = load_region_table(path, 0, max_rows=20)
    assert sliced.height == 20
    assert sliced.width == full.width
    assert len(sliced.indices) < len(full.indices)
    assert np.all((sliced.ys0[sliced.indices] - sliced.y_min) < 20)
