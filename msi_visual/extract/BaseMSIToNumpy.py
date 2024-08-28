import numpy as np
import os
from argparse import Namespace
from typing import Optional

from abc import ABC, abstractmethod

class BaseMSIToNumpy(ABC):
    def __init__(self,
        min_mz: Optional[float] = None,
        max_mz: Optional[float] = None,
        bins_per_mz: int = 1,
        nonzero=False,
        num_workers=1):
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bins_per_mz = bins_per_mz
        self.nonzero = nonzero
        self.num_workers = num_workers

    def save_numpy(self, img: np.ndarray, mzs: list[float], input_path: str, output_path: str, region: int):
        os.makedirs(output_path, exist_ok=True)
        np.save(os.path.join(output_path, f"{region}.npy"), img)

    def save_extraction_args(self, mzs, input_path, output_path):
        extraction_args = Namespace(path=input_path, min_mz=self.min_mz, max_mz=self.max_mz, bins_per_mz=self.bins_per_mz, nonzero=self.nonzero, mzs=mzs)
        extraction_args = str(extraction_args)
        with open(os.path.join(output_path, "args.txt"), "w") as f:
            f.write(extraction_args)

    def extract_region(self, input_path, output_path, region):
        img, mzs = self.to_numpy(input_path, region)
        self.save_numpy(img, mzs, input_path, output_path, region)
        self.save_extraction_args(mzs, input_path, output_path)

    def __call__(self, input_path, output_path):
        regions = self.get_regions()
        for region in regions:
            self.extract_region(input_path, output_path, region)

    @abstractmethod
    def get_regions(self):
        pass

    @abstractmethod
    def to_numpy(self, input_path: str, region: int=0):   
        pass