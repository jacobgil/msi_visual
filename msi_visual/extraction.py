from argparse import Namespace
from pathlib import Path
import numpy as np

def get_extraction_mz_list(extraction_folder):
        extraction_args = eval(
            open(
                Path(extraction_folder) /
                "args.txt").read())
        bins = extraction_args.bins
        extraction_start_mz = extraction_args.start_mz
        extraction_end_mz = extraction_args.end_mz
        try:
            extraction_mzs = extraction_args.mzs
        except Exception as e:
            extraction_mzs = list(
                np.arange(
                    extraction_start_mz,
                    extraction_end_mz + 1,
                    1.0 / bins))
            extraction_mzs = [float(f"{mz:.3f}") for mz in extraction_mzs]
        
        return extraction_mzs
