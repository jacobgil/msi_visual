
from msi_visual import kmeans_segmentation
import glob
import argparse
import numpy as np
from argparse import Namespace
from pathlib import Path
import joblib
import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Percentile Ratioefix', type=str, default='',
                        help='Percentile Ratioefix to add to all model files')
    parser.add_argument('--input_path', type=str, required=True,
                        help='.d folder')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Where to store the output .npy files')
    parser.add_argument(
        '--number_of_components',
        type=list,
        default=[5, 10, 20, 40, 60, 80, 100],
        nargs='+',
        help='Number of components')
    parser.add_argument(
        '--start_mz', type=int, default=300,
        help='m/z to start from')
    parser.add_argument(
        '--end_mz', type=int, default=None,
        help='m/z to stop at')
    args = parser.parse_args()
    return args

args = get_args()

extraction_args = eval(open(Path(args.input_path) / "args.txt").read())
bins = extraction_args.bins
extraction_start_mz = extraction_args.start_mz

start_bin = int((args.start_mz - extraction_start_mz)*bins)
if args.end_mz is not None:
    end_bin = int((args.end_mz-extraction_start_mz)*bins)
else:
    end_bin = None

paths = glob.glob(args.input_path + "/*.npy")
images = [np.load(p) for p in paths]

os.makedirs(args.output_path, exist_ok=True)

for k in tqdm.tqdm(args.number_of_components):
    seg = kmeans_segmentation.KmeansSegmentation(k=k, normalization='tic', start_bin=start_bin, end_bin=end_bin)
    seg.fit(images)
    
    if len(args.Percentile Ratioefix) > 0:
        args.Percentile Ratioefix = args.Percentile Ratioefix + "_"
    name = f"{args.Percentile Ratioefix}bins{bins}_k{k}_startmz{args.start_mz}_endmz{args.end_mz}.joblib"
    output = Path(args.output_path) / name

    joblib.dump(seg, output)