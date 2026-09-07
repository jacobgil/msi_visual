import numpy as np
import cv2
import time
import numba
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion


class TOP3:
    def __init__(self, low=99.9, norm_percentile=99.9):
        self.low = low
        self.norm_percentile = norm_percentile

    def __repr__(self):
        return f"TOP3 Intensities. low={self.low}"


    def __call__(self, img: np.ndarray, power=1, to_lab=True):
        # Numba reshape() requires C-contiguous arrays; transpose / views are not contiguous.
        img = np.ascontiguousarray(np.asarray(img, dtype=np.float32))

        @numba.njit(parallel=True, fastmath=True)
        def get_p0p1p2(img):
            H, W, C = img.shape
            flat = img.reshape(-1, C)
            N = flat.shape[0]

            # Allocate output arrays
            p0 = np.empty(N, dtype=np.float32)
            p1 = np.empty(N, dtype=np.float32)
            p2 = np.empty(N, dtype=np.float32)

            # Loop through each pixel and find the top 3 values
            for i in numba.prange(N):  # Parallelized loop
                row = flat[i]
                v0, v1, v2 = -np.inf, -np.inf, -np.inf

                for v in row:
                    if v > v0:
                        v2, v1, v0 = v1, v0, v  # Shift down
                    elif v > v1:
                        v2, v1 = v1, v
                    elif v > v2:
                        v2 = v
                
                p0[i], p1[i], p2[i] = v0, v1, v2
            p0, p1, p2 = p0.reshape(H, W), p1.reshape(H, W), p2.reshape(H, W)
            return p0, p1, p2
        
        p0, p1, p2 = get_p0p1p2(img)
        if power != 1:
            p0 = np.power(p0, power)
            p1 = np.power(p1, power)
            p2 = np.power(p2, power)
        # Normalize
        percentiles = np.percentile([p0, p1, p2], self.norm_percentile, axis=(1, 2))

        p0 /= percentiles[0]
        p1 /= percentiles[1]
        p2 /= percentiles[2]

        # Clip to [0,1]
        np.clip(p0, 0, 1, out=p0)
        np.clip(p1, 0, 1, out=p1)
        np.clip(p2, 0, 1, out=p2)

        # p = np.concatenate( (p0[..., None], p1[..., None], p2[..., None]), axis=-1)
        # p = p / (1e-6 + np.sum(p, axis=-1)[..., None])
        # visualization = np.uint8(255 * p)


        # Merge into RGB
        visualization = cv2.merge([(np.uint8(255 * p0)), (np.uint8(255 * p1)), (np.uint8(255 * p2))])
        if to_lab:
            visualization = cv2.cvtColor(visualization, cv2.COLOR_LAB2RGB)
        visualization[p0  == 0] = 0
        return visualization

class PercentileRatio:
    def __init__(self, percentiles=[99.99, 99.9, 99.9, 99, 98, 85]):
        self.percentiles = percentiles

    def __repr__(self):
        return "Percentile Ratio"

    def __call__(self, img):
        sorted_image = np.sort(img, axis=-1)
        N = sorted_image.shape[-1]

        p0 = sorted_image[:, :, int(N * self.percentiles[0] / 100)]
        p1 = sorted_image[:, :, int(N * self.percentiles[1] / 100)]
        p2 = sorted_image[:, :, int(N * self.percentiles[2] / 100)]
        p3 = sorted_image[:, :, int(N * self.percentiles[3] / 100)]
        p4 = sorted_image[:, :, int(N * self.percentiles[4] / 100)]
        p5 = sorted_image[:, :, int(N * self.percentiles[5] / 100)]

        a = p0 / (1e-5 + p1)
        a = a / np.percentile(a, 99.9)
        a[a > 1] = 1

        b = p2 / (1e-5 + p3)
        b = b / np.percentile(b, 99.9)
        b[b > 1] = 1

        c = p4 / (1e-5 + p5)
        c = c / np.percentile(c, 99.9)
        c[c > 1] = 1


        visualization = cv2.merge([(np.uint8(255 * a)),
                                    (np.uint8(255 * b)),
                                    (np.uint8(255 * c))])
        visualization = cv2.cvtColor(visualization, cv2.COLOR_LAB2LRGB)
        visualization[img.max(axis=-1) == 0] = 0
        return visualization
