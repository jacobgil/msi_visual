import numpy as np
import tqdm
import math
from msi_visual.extract import tsfdata
import sqlite3
from msi_visual.extract.base_msi_to_numpy import BaseMSIToNumpy
        
class BrukerTsfToNumpy(BaseMSIToNumpy):
    def get_regions(self, input_path: str):
        tsf = tsfdata.TsfData(input_path)
        conn = tsf.conn
        regions = conn.execute(
            f"SELECT RegionNumber FROM MaldiFrameInfo").fetchall()
        regions = [int(r[0]) for r in regions]
        regions = sorted(list(set(regions)))
        return regions

    def get_img_type(self):
        return np.float32

    def read_all_point_data(self, input_path: str, region: int = 0):
        tsf = tsfdata.TsfData(input_path)
        conn = tsf.conn
        xs = conn.execute(
            f"SELECT XIndexPos FROM MaldiFrameInfo WHERE RegionNumber={region}").fetchall()
        ys = conn.execute(
            f"SELECT YIndexPos FROM MaldiFrameInfo WHERE RegionNumber={region}").fetchall()
        regions = conn.execute(
            "SELECT RegionNumber FROM MaldiFrameInfo").fetchall()
        regions = [r[0] for r in regions]
        xs, ys = np.int32(xs), np.int32(ys)

        start_index = regions.index(region)
        all_xs, all_ys, all_mzs, all_intensities = [], [], [], []
        for index, (x, y) in tqdm.tqdm(enumerate(zip(xs, ys)), total=len(xs)):
            indices, intensities = tsf.readLineSpectrum(start_index + index + 1)
            mzs = tsf.indexToMz(start_index + index + 1, indices)
            
            intensities = np.float32(intensities)

            all_xs.append(x)
            all_ys.append(y)
            all_mzs.append(mzs)
            all_intensities.append(intensities)
        return all_xs, all_ys, all_mzs, all_intensities