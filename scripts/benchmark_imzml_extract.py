#!/usr/bin/env python3
"""
Benchmark / correctness harness for fast imzML extraction.

Default target (Dataset18, first 50 rows)::

  python scripts/benchmark_imzml_extract.py

Reports:
  * parse time (XML once)
  * slow (legacy loop) vs fast single vs fast multiprocess wall times
  * max abs / rtol agreement
  * speedup factors (read+bin only, excluding parse)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from msi_visual.extract.imzml_fast import ImzmlFastExtractor  # noqa: E402

DEFAULT_IMZML = (
    r"E:\MSImaging-data\MSI_slides\Metaspace_datasets"
    r"\Dataset18 - brainmouse-sag-20um\ppkg_midres_20250529-total ion count.imzML"
)
DEFAULT_GOLDEN = (
    r"E:\MSImaging-data\_msi_visual\Extractions"
    r"\Metaspace_datasets\Dataset18_brainmouse20um\5bins\0.npy"
)


def _peak_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imzml", type=str, default=DEFAULT_IMZML)
    ap.add_argument("--min-mz", type=float, default=50.0)
    ap.add_argument("--max-mz", type=float, default=1000.0)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, (os_cpu := __import__("os").cpu_count() or 4) // 2))
    ap.add_argument("--skip-slow", action="store_true", help="Skip legacy-loop baseline")
    ap.add_argument("--golden", type=str, default="", help="Full-slide golden .npy to compare (ignores max-rows)")
    ap.add_argument("--full", action="store_true", help="Full slide (sets max-rows to None)")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--target-speedup", type=float, default=10.0)
    args = ap.parse_args()

    max_rows = None if args.full else args.max_rows
    imzml = Path(args.imzml)
    if not imzml.is_file():
        print(f"ERROR: imzML not found: {imzml}", file=sys.stderr)
        return 2

    print(f"imzML: {imzml}")
    print(f"max_rows={max_rows} bins={args.bins} workers={args.workers}")

    ex = ImzmlFastExtractor(
        min_mz=args.min_mz,
        max_mz=args.max_mz,
        bins_per_mz=args.bins,
        num_workers=1,
        max_rows=max_rows,
        id="bench",
    )

    t0 = time.perf_counter()
    reader, ibd = ex.parse(imzml)
    parse_s = time.perf_counter() - t0
    print(f"parse_s={parse_s:.2f}  n_coords={len(reader.coordinates)}  ibd={ibd}")

    results: dict = {
        "imzml": str(imzml),
        "max_rows": max_rows,
        "bins": args.bins,
        "workers": args.workers,
        "parse_s": parse_s,
        "n_coords": len(reader.coordinates),
    }

    slow_img = None
    slow_s = None
    if not args.skip_slow:
        rss0 = _peak_rss_mb()
        t0 = time.perf_counter()
        slow_img, _ = ex.extract_array(imzml, reader=reader, ibd=ibd, mode="slow")
        slow_s = time.perf_counter() - t0
        rss1 = _peak_rss_mb()
        print(f"slow_loop_s={slow_s:.2f}  shape={slow_img.shape}  rss_mb={rss0}->{rss1}")
        results["slow_s"] = slow_s
        results["shape"] = list(slow_img.shape)

    # Fast single-process
    ex1 = ImzmlFastExtractor(
        min_mz=args.min_mz,
        max_mz=args.max_mz,
        bins_per_mz=args.bins,
        num_workers=1,
        max_rows=max_rows,
    )
    t0 = time.perf_counter()
    fast1, _ = ex1.extract_array(imzml, reader=reader, ibd=ibd, mode="fast")
    fast1_s = time.perf_counter() - t0
    print(f"fast_workers=1_s={fast1_s:.2f}  shape={fast1.shape}")
    results["fast1_s"] = fast1_s

    if slow_img is not None:
        max_abs = float(np.max(np.abs(fast1 - slow_img)))
        ok = bool(np.allclose(fast1, slow_img, rtol=0, atol=1e-3))
        print(f"fast1_vs_slow: allclose={ok} max_abs={max_abs:.6g}")
        results["fast1_vs_slow_allclose"] = ok
        results["fast1_vs_slow_max_abs"] = max_abs

    # Fast multiprocess
    exw = ImzmlFastExtractor(
        min_mz=args.min_mz,
        max_mz=args.max_mz,
        bins_per_mz=args.bins,
        num_workers=args.workers,
        max_rows=max_rows,
    )
    t0 = time.perf_counter()
    fastw, _ = exw.extract_array(imzml, reader=reader, ibd=ibd, mode="fast")
    fastw_s = time.perf_counter() - t0
    print(f"fast_workers={args.workers}_s={fastw_s:.2f}")
    results["fastw_s"] = fastw_s

    max_abs_w = float(np.max(np.abs(fastw - fast1)))
    ok_w = bool(np.allclose(fastw, fast1, rtol=0, atol=1e-3))
    print(f"fastw_vs_fast1: allclose={ok_w} max_abs={max_abs_w:.6g}")
    results["fastw_vs_fast1_allclose"] = ok_w
    results["fastw_vs_fast1_max_abs"] = max_abs_w

    baseline = slow_s if slow_s is not None else None
    if baseline and baseline > 0:
        sp1 = baseline / fast1_s if fast1_s > 0 else float("inf")
        spw = baseline / fastw_s if fastw_s > 0 else float("inf")
        print(f"speedup_vs_slow: single={sp1:.2f}x  workers={args.workers}:{spw:.2f}x  target={args.target_speedup}x")
        results["speedup_single"] = sp1
        results["speedup_workers"] = spw
        results["hit_target"] = bool(spw >= args.target_speedup or sp1 >= args.target_speedup)
        if results["hit_target"]:
            print("TARGET HIT")
        else:
            print("TARGET NOT YET HIT — iterate (more workers / profile I/O)")

    # Optional full-slide golden compare (only meaningful with --full)
    golden = args.golden or (DEFAULT_GOLDEN if args.full else "")
    if golden and max_rows is None:
        gpath = Path(golden)
        if gpath.is_file():
            print(f"Comparing to golden: {gpath}")
            gold = np.load(gpath, mmap_mode="r")
            # Compare in chunks to limit RAM
            max_abs_g = 0.0
            for y0 in range(0, fastw.shape[0], 32):
                y1 = min(y0 + 32, fastw.shape[0])
                diff = np.max(np.abs(fastw[y0:y1] - gold[y0:y1]))
                max_abs_g = max(max_abs_g, float(diff))
            ok_g = max_abs_g <= 1e-2
            print(f"fastw_vs_golden: max_abs={max_abs_g:.6g} ok={ok_g}")
            results["golden_max_abs"] = max_abs_g
            results["golden_ok"] = ok_g
        else:
            print(f"WARNING: golden not found: {gpath}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    # Non-zero exit if accuracy failed
    if results.get("fast1_vs_slow_allclose") is False:
        return 3
    if results.get("fastw_vs_fast1_allclose") is False:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
