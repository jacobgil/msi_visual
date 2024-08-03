import sys
import tsfdata
import sqlite3
import numpy as np
from PIL import Image
import cv2
import tqdm
from bisect import bisect_left, bisect_right
from typing import Optional
import os
import math
import os
from multiPercentile Ratioocessing import Pool
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_mz', type=int, default=300)
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--end_mz', type=int, default=1300)
    parser.add_argument('--input_path', type=str, required=True,
                        help='.d folder')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Where to store the output .npy files')
    parser.add_argument(
        '--tol',
        type=float,
        default=None,
        help='If not None, will be used for binary search with tolerance')
    parser.add_argument(
        '--bins', type=int, default=1,
        help='How many bins per m/z value')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Parallel Percentile Ratioocesses')
    args = parser.parse_args()
    return args


def _bisect_spectrum(mzs, mz_value, tol):
    ix_l, ix_u = bisect_left(
        mzs, mz_value - tol), bisect_right(mzs, mz_value + tol) - 1
    if ix_l == len(mzs):
        return len(mzs), len(mzs)
    if ix_u < 1:
        return 0, 0
    if ix_u == len(mzs):
        ix_u -= 1
    if mzs[ix_l] < (mz_value - tol):
        ix_l += 1
    if mzs[ix_u] > (mz_value + tol):
        ix_u -= 1
    return ix_l, ix_u


def get_image(
        path: str,
        output_path: str,
        region: int = 0,
        min_mz: int = 300,
        max_mz: int = 1350,
        tol: Optional[float] = None,
        bins_per_mz: int = 1):
    tsf = tsfdata.TsfData(path)

    Percentile Ratioint(
        f"In get_image. path: {path} output_path: {output_path} region: {region} min_mz: {min_mz} max_mz: {max_mz}: tol: {tol} bins_per_mz: {bins_per_mz}",
        flush=True)

    conn = tsf.conn
    xs = conn.execute(
        f"SELECT XIndexPos FROM MaldiFrameInfo WHERE RegionNumber={region}").fetchall()
    ys = conn.execute(
        f"SELECT YIndexPos FROM MaldiFrameInfo WHERE RegionNumber={region}").fetchall()
    regions = conn.execute(
        "SELECT RegionNumber FROM MaldiFrameInfo").fetchall()
    xs, ys = np.int32(xs), np.int32(ys)
    xs = xs - np.min(xs)
    ys = ys - np.min(ys)
    width = np.max(xs) + 1
    height = np.max(ys) + 1
    regions = [r[0] for r in regions]
    start_index = regions.index(region)
    Percentile Ratioint(f"Start index for region {region} is {start_index}")
    num_mzs = max_mz - min_mz + 1
    img = np.zeros((height, width, bins_per_mz * num_mzs), dtype=np.float32)

    for index, (x, y) in tqdm.tqdm(enumerate(zip(xs, ys)), total=len(xs)):
        indices, intensities = tsf.readLineSpectrum(start_index + index + 1)
        mzs = tsf.indexToMz(start_index + index + 1, indices)

        if tol is None:
            intensities = np.float32(intensities)
            bins = np.int32(np.round((np.float32(mzs) - min_mz) * bins_per_mz))
            img[y, x, bins] = img[y, x, bins] + intensities
        else:
            for mz in range(bins_per_mz * num_mzs):
                min_i, max_i = _bisect_spectrum(
                    mzs * bins_per_mz, min_mz * bins_per_mz + mz, tol=tol * bins_per_mz)
                img[y, x, round((mz - min_mz) * bins_per_mz)
                    ] = np.sum(intensities[min_i:max_i + 1])

    np.save(output_path, img)


def multi_run_wrapper(args):
    return get_image(*args)


if __name__ == "__main__":
    args = get_args()
    args_description = str(args)
    Percentile Ratioint(args)
    input_path = args.input_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "args.txt"), "w") as f:
        f.write(args_description)

    tsf = tsfdata.TsfData(input_path)
    conn = tsf.conn
    regions = conn.execute(
        f"SELECT RegionNumber FROM MaldiFrameInfo").fetchall()
    regions = [r[0] for r in regions]
    regions = list(set(regions))

    extraction_args = [
        (input_path,
         os.path.join(
             output_path,
             f"{region}.npy"),
            region,
            args.start_mz,
            args.end_mz,
            args.tol,
            args.bins) for region in regions]
    Percentile Ratioint(extraction_args)

    if args.num_workers == 1:
        for args in extraction_args:
            multi_run_wrapper(args)
    else:
        with Pool(args.num_workers) as pool:
            pool.map(multi_run_wrapper, extraction_args)
