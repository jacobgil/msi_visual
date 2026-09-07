#!/usr/bin/env python3
"""
Benchmark fast Bruker TIMS/TSF extraction.

Default: NRL4485 slide2 TIMS region 0, first 20 rows::

  python scripts/benchmark_bruker_extract.py
  python scripts/benchmark_bruker_extract.py --full --region 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from msi_visual.extract.bruker_fast import BrukerFastExtractor, detect_kind  # noqa: E402

DEFAULT_D = (
    r"E:\MSImaging-data\MSI_slides\2024-02-15_NRL4485_slide2_lipid_neg_timsON"
    r"\Slide2_lipid_neg_TIMS_on.d"
)
DEFAULT_GOLDEN = r"E:\MSImaging-data\_msi_visual\Extractions\NRL4485-s2\5_bins\0.npy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=DEFAULT_D)
    ap.add_argument("--region", type=int, default=0)
    ap.add_argument("--min-mz", type=float, default=300.0)
    ap.add_argument("--max-mz", type=float, default=1350.0)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--skip-slow", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--golden", type=str, default="")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--target-speedup", type=float, default=10.0)
    args = ap.parse_args()

    max_rows = None if args.full else args.max_rows
    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"ERROR: missing {input_path}", file=sys.stderr)
        return 2

    kind = detect_kind(input_path)
    print(f"input={input_path}")
    print(f"kind={kind} region={args.region} max_rows={max_rows} bins={args.bins} workers={args.workers}")

    results: dict = {
        "input": str(input_path),
        "kind": kind,
        "region": args.region,
        "max_rows": max_rows,
        "bins": args.bins,
        "workers": args.workers,
    }

    slow_img = None
    slow_s = None
    if not args.skip_slow:
        ex = BrukerFastExtractor(
            min_mz=args.min_mz,
            max_mz=args.max_mz,
            bins_per_mz=args.bins,
            num_workers=1,
            max_rows=max_rows,
            kind=kind,
        )
        t0 = time.perf_counter()
        slow_img, _ = ex.extract_region_array(input_path, region=args.region, mode="slow")
        slow_s = time.perf_counter() - t0
        print(f"slow_loop_s={slow_s:.2f} shape={slow_img.shape}")
        results["slow_s"] = slow_s
        results["shape"] = list(slow_img.shape)

    ex1 = BrukerFastExtractor(
        min_mz=args.min_mz,
        max_mz=args.max_mz,
        bins_per_mz=args.bins,
        num_workers=1,
        max_rows=max_rows,
        kind=kind,
    )
    t0 = time.perf_counter()
    fast1, _ = ex1.extract_region_array(input_path, region=args.region, mode="fast")
    fast1_s = time.perf_counter() - t0
    print(f"fast_workers=1_s={fast1_s:.2f} shape={fast1.shape}")
    results["fast1_s"] = fast1_s

    if slow_img is not None:
        max_abs = float(np.max(np.abs(fast1.astype(np.float64) - slow_img.astype(np.float64))))
        ok = bool(np.allclose(fast1, slow_img, rtol=0, atol=1e-3))
        print(f"fast1_vs_slow: allclose={ok} max_abs={max_abs:.6g}")
        results["fast1_vs_slow_allclose"] = ok
        results["fast1_vs_slow_max_abs"] = max_abs

    exw = BrukerFastExtractor(
        min_mz=args.min_mz,
        max_mz=args.max_mz,
        bins_per_mz=args.bins,
        num_workers=args.workers,
        max_rows=max_rows,
        kind=kind,
    )
    t0 = time.perf_counter()
    fastw, _ = exw.extract_region_array(input_path, region=args.region, mode="fast")
    fastw_s = time.perf_counter() - t0
    print(f"fast_workers={args.workers}_s={fastw_s:.2f}")
    results["fastw_s"] = fastw_s
    max_abs_w = float(np.max(np.abs(fastw.astype(np.float64) - fast1.astype(np.float64))))
    ok_w = bool(np.allclose(fastw, fast1, rtol=0, atol=1e-3))
    print(f"fastw_vs_fast1: allclose={ok_w} max_abs={max_abs_w:.6g}")
    results["fastw_vs_fast1_allclose"] = ok_w

    if slow_s and slow_s > 0:
        sp1 = slow_s / fast1_s if fast1_s > 0 else float("inf")
        spw = slow_s / fastw_s if fastw_s > 0 else float("inf")
        print(f"speedup_vs_slow: single={sp1:.2f}x workers={args.workers}:{spw:.2f}x target={args.target_speedup}x")
        results["speedup_single"] = sp1
        results["speedup_workers"] = spw
        results["hit_target"] = bool(max(sp1, spw) >= args.target_speedup)
        print("TARGET HIT" if results["hit_target"] else "TARGET NOT YET HIT")

    golden = args.golden or (DEFAULT_GOLDEN if args.full and args.region == 0 else "")
    if golden and max_rows is None:
        gpath = Path(golden)
        if gpath.is_file():
            gold = np.load(gpath, mmap_mode="r")
            max_abs_g = 0.0
            for y0 in range(0, fast1.shape[0], 32):
                y1 = min(y0 + 32, fast1.shape[0])
                max_abs_g = max(max_abs_g, float(np.max(np.abs(fast1[y0:y1] - gold[y0:y1]))))
            print(f"golden_max_abs={max_abs_g:.6g} ok={max_abs_g <= 1e-2} shape_g={gold.shape}")
            results["golden_max_abs"] = max_abs_g
            results["golden_ok"] = bool(max_abs_g <= 1e-2)
        else:
            print(f"WARNING: golden missing {gpath}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    if results.get("fast1_vs_slow_allclose") is False:
        return 3
    if results.get("fastw_vs_fast1_allclose") is False:
        return 4
    if results.get("golden_ok") is False:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
