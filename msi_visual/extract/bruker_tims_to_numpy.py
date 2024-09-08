import numpy as np
import tqdm
import math
from msi_visual.extract import timsdata
import sqlite3
from msi_visual.extract.base_msi_to_numpy import BaseMSIToNumpy
        
class BrukerTimsToNumpy(BaseMSIToNumpy):
    def get_regions(self, input_path: str):
        td = timsdata.TimsData(input_path)
        conn = td.conn
        regions = conn.execute(
            f"SELECT RegionNumber FROM MaldiFrameInfo").fetchall()
        regions = [int(r[0]) for r in regions]
        regions = sorted(list(set(regions)))
        return regions

    def get_img_type(self):
        return np.uint32


    def read_all_point_data(self, input_path: str, region: int = 0):
        td = timsdata.TimsData(input_path)
        conn = td.conn
        xs = conn.execute(
            f"SELECT XIndexPos FROM MaldiFrameInfo WHERE RegionNumber={region}").fetchall()
        ys = conn.execute(
            f"SELECT YIndexPos FROM MaldiFrameInfo WHERE RegionNumber={region}").fetchall()
        regions = conn.execute(
            "SELECT RegionNumber FROM MaldiFrameInfo").fetchall()
        regions = [int(r[0]) for r in regions]
        

        xs, ys = np.int32(xs), np.int32(ys)
        start_index = regions.index(region)
        print("start index", start_index)
        
        all_xs, all_ys, all_mzs, all_intensities = [], [], [], []
        for point_index, (x, y) in tqdm.tqdm(enumerate(zip(xs, ys)), total=len(xs)):
            frame_id = start_index+point_index+1
            all_xs.append(x)
            all_ys.append(y)
            q = conn.execute("SELECT NumScans FROM Frames WHERE Id={0}".format(frame_id))
            num_scans = q.fetchone()[0]
            mzs, intensities = [], []
            for scan in td.readScans(frame_id, 0, num_scans):
                
                index = np.array(scan[0], dtype=np.float64)
                mz = td.indexToMz(frame_id, index)
                intensity = scan[1]

                mzs.extend(list(mz))
                intensities.extend(list(intensity))
            all_mzs.append(mzs)
            all_intensities.append(intensities)

        return all_xs, all_ys, all_mzs, all_intensities