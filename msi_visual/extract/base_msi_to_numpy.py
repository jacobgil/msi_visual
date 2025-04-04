import numpy as np
import os
from argparse import Namespace
from typing import Optional
import tqdm
from abc import ABC, abstractmethod

class BaseMSIToNumpy(ABC):
    def __init__(self,
        min_mz: Optional[float] = None,
        max_mz: Optional[float] = None,
        bins_per_mz: int = 1,
        mz_list: Optional[list[float]] = None,
        mz_list_tolerance: Optional[float] = 0.01,
        nonzero=False,
        id=''):
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bins_per_mz = bins_per_mz
        if mz_list is not None:
            self.mz_list = np.array(mz_list)
        else:
            self.mz_list = None
        self.mz_list_tolerance = mz_list_tolerance
        self.nonzero = nonzero
        self.id = id
        
        if self.mz_list is not None:
            self.bins_per_mz = 1

        self.discrete_set_of_mzs = self.nonzero or (self.mz_list is not None)
        print("self.discrete_set_of_mzs", self.discrete_set_of_mzs, self.nonzero, self.mz_list)


    def save_numpy(self, img: np.ndarray, mzs: list[float], input_path: str, output_path: str, region: int):
        os.makedirs(output_path, exist_ok=True)
        np.save(os.path.join(output_path, f"{region}.npy"), img)

    def save_extraction_args(self, mzs, input_path, output_path):
        extraction_args = Namespace(path=input_path, start_mz=self.min_mz, end_mz=self.max_mz, bins=self.bins_per_mz, nonzero=self.nonzero, mzs=mzs, id=self.id)
        extraction_args = str(extraction_args)
        with open(os.path.join(output_path, "args.txt"), "w") as f:
            f.write(extraction_args)

    def extract_region(self, input_path, output_path, region):
        img, mzs = self.to_numpy(input_path, region)
        self.save_numpy(img, mzs, input_path, output_path, region)
        self.save_extraction_args(mzs, input_path, output_path)

    def __call__(self, input_path, output_path):
        regions = self.get_regions(input_path)
        for region in regions:
            self.extract_region(input_path, output_path, region)

    @abstractmethod
    def get_img_type(self):
        pass

    @abstractmethod
    def read_all_point_data(self, input_path, region):
        pass
    

    def to_numpy(self, input_path, region=0):   
        xs, ys, all_mzs, all_intensities = self.read_all_point_data(input_path, region)
        if self.discrete_set_of_mzs:
            set_of_mzs = set()
            for mz_list in all_mzs:
                set_of_mzs.update(mz_list)
            set_of_mzs = np.float32(sorted(list(set_of_mzs)))

            if self.nonzero:
                set_of_mzs_quantized = sorted(list(set(list(np.int32(np.round(set_of_mzs * self.bins_per_mz))))))
            else:
                set_of_mzs_quantized = self.mz_list
            mz_to_index = {mz: i for i, mz in enumerate(set_of_mzs_quantized)}

        xs, ys = np.int32(xs), np.int32(ys)
        
        if self.max_mz is None or self.min_mz is None:
            set_of_mzs = set()
            for mz_list in all_mzs:
                set_of_mzs.update(mz_list)
            set_of_mzs = np.float32(list(set_of_mzs))
            self.max_mz = np.max(set_of_mzs)
            self.min_mz = np.min(set_of_mzs)
        else:
            self.max_mz, self.min_mz = float(self.max_mz), float(self.min_mz)
        xs = xs - np.min(xs)
        ys = ys - np.min(ys)
        width = np.max(xs) + 1
        height = np.max(ys) + 1
        num_mzs = round(self.max_mz - self.min_mz + 1)
        if self.discrete_set_of_mzs:
            img = np.zeros((height, width, len(set_of_mzs_quantized)), dtype=self.get_img_type())
        else:
            img = np.zeros((height, width, self.bins_per_mz * num_mzs), dtype=self.get_img_type())
        print("shape", img.shape)
        for x, y, mzs, intensities in tqdm.tqdm(zip(xs, ys, all_mzs, all_intensities)):
            intensities = np.array(intensities)
            if self.discrete_set_of_mzs:
                mz_indices = [mz_to_index[mz] for mz in mzs]

                for bin, intensity in zip(mz_indices, intensities):
                    img[y, x, bin] = img[y, x, bin] + intensity
            else:
                bins = np.int32(np.round((np.float32(mzs) - self.min_mz) * self.bins_per_mz))
                for bin, intensity in zip(bins, intensities):
                    img[y, x, bin] = img[y, x, bin] + intensity

        if self.discrete_set_of_mzs:
            if self.nonzero:
                mzs = [float(f"{(mz/self.bins_per_mz):.6f}") for mz in set_of_mzs_quantized]
            else:
                mzs = self.mz_list            
        else:
            mzs = np.arange(self.min_mz, self.max_mz + 1, 1.0/self.bins_per_mz)
            mzs = [float(f"{mzs[i]:.6f}") for i in range(img.shape[-1])]

        return np.float32(img), mzs


    @abstractmethod
    def get_regions(self, input_path: str):
        pass