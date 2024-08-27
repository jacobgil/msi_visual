import numpy as np
import cv2
import tqdm
import os
import math
import os
from argparse import Namespace
from typing import Optional
from pyimzml.ImzMLParser import ImzMLParser

class PymzmlToNumpy:
    def __init__(self,
        min_mz: Optional[float] = None,
        max_mz: Optional[float] = None,
        bins_per_mz: int = 1,
        nonzero=False):
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bins_per_mz = bins_per_mz
        self.nonzero = nonzero
    
    def save(self, img, mzs, input_path, output_path):
        os.makedirs(output_path, exist_ok=True)

        np.save(os.path.join(output_path, "0.npy"), img)
        extraction_args = Namespace(path=input_path, min_mz=self.min_mz, max_mz=self.max_mz, bins_per_mz=self.bins_per_mz, nonzero=self.nonzero, mzs=mzs)
        extraction_args = str(extraction_args)
        with open(os.path.join(output_path, "args.txt"), "w") as f:
            f.write(extraction_args)

    def __call__(self, input_path, output_path):
        img, mzs = self.to_numpy(input_path)
        self.save(img, mzs, input_path, output_path)

    def to_numpy(self, input_path):   
        p = ImzMLParser(input_path)
        xs = []
        ys = []
        all_mzs = []
        all_intensities = []
        for idx, (x,y,z) in tqdm.tqdm(enumerate(p.coordinates), total=len(p.coordinates)):
            mzs, intensities = p.getspectrum(idx)
            xs.append(x)
            ys.append(y)
            all_mzs.append(mzs)
            all_intensities.append(intensities)

        if self.nonzero:
            set_of_mzs = set()
            for mz_list in all_mzs:
                set_of_mzs.update(mz_list)
            set_of_mzs = np.float32(sorted(list(set_of_mzs)))
            set_of_mzs_quantized = sorted(list(set(list(np.int32(np.round(set_of_mzs * self.bins_per_mz))))))
            mz_to_index = {mz: i for i, mz in enumerate(set_of_mzs_quantized)}

        xs, ys = np.int32(xs), np.int32(ys)
        if self.max_mz is None or self.min_mz is None:
            set_of_mzs = set()
            for mz_list in all_mzs:
                set_of_mzs.update(mz_list)
            set_of_mzs = np.float32(list(set_of_mzs))
            self.max_mz = np.max(set_of_mzs)
            self.min_mz = np.min(set_of_mzs)


        xs = xs - np.min(xs)
        ys = ys - np.min(ys)
        width = np.max(xs) + 1
        height = np.max(ys) + 1

        num_mzs = int(self.max_mz - self.min_mz + 1)
        if self.nonzero:
            img = np.zeros((height, width, len(set_of_mzs_quantized)), dtype=np.float32)
        else:
            img = np.zeros((height, width, self.bins_per_mz * num_mzs), dtype=np.float32)

        for x, y, mzs, intensities in tqdm.tqdm(zip(xs, ys, all_mzs, all_intensities)):
            intensities = np.float32(intensities)
            
            if self.nonzero:
                mzs = np.float32(mzs)
                mz_indices = [mz_to_index[mz] for mz in list(np.int32(np.round(mzs * self.bins_per_mz)))]
                img[y, x, mz_indices] = img[y, x, mz_indices] + intensities
            else:
                bins = np.int32(np.round((np.float32(mzs) - self.min_mz) * self.bins_per_mz))
                img[y, x, bins] = img[y, x, bins] + intensities
        mzs = np.arange(self.min_mz, self.max_mz + 1, 1.0/self.bins_per_mz)
        indices = list(range(img.shape[-1]))
        if self.nonzero:
            mzs = [float(f"{(mz/self.bins_per_mz):.6f}") for mz in set_of_mzs_quantized]
        else:
            mzs = [float(f"{mzs[i]:.6f}") for i in indices]

        return img, mzs