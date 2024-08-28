import numpy as np
import tqdm
import math
from pyimzml.ImzMLParser import ImzMLParser

from msi_visual.extract import BaseMSIToNumpy
        
class PymzmlToNumpy(BaseMSIToNumpy):
    def get_regions(self):
        return [0]

    def to_numpy(self, input_path, region=0):   
        p = ImzMLParser(input_path)
        xs, ys, all_mzs, all_intensities = [], [], [], []
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