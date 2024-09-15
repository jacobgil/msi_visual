import numpy as np
import tqdm
import math
from pyimzml.ImzMLParser import ImzMLParser
from msi_visual.extract.base_msi_to_numpy import BaseMSIToNumpy
        
class PymzmlToNumpy(BaseMSIToNumpy):
    def get_regions(self, input_path):
        return [0]

    def get_img_type(self):
        return np.float32

    def read_all_point_data(self, input_path: str, region: int=0):
        p = ImzMLParser(input_path)
        xs, ys, all_mzs, all_intensities = [], [], [], []
        for idx, (x,y,z) in tqdm.tqdm(enumerate(p.coordinates), total=len(p.coordinates)):
            mzs, intensities = p.getspectrum(idx)
            xs.append(x)
            ys.append(y)
            all_mzs.append(mzs)
            all_intensities.append(intensities)
        print("IMS slide max mz", np.max([np.max(m) for m in all_mzs]))
        print("IMS slide min mz", np.min([np.min(m) for m in all_mzs]))
        return xs, ys, all_mzs, all_intensities