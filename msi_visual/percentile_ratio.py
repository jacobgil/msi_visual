import numpy as np
import cv2

from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion


class TOP3:
    def __init__(self, low=99.9):
        self.low = low

    def __repr__(self):
        return f"TOP-3 Intensities. low={self.low}"

    def __call__(self, img):
        sorted = np.sort(img, axis=-1)
        p0 = sorted[:, :, -1]
        p1 = sorted[:, :, -2]
        p2 = sorted[:, :, -3]
        a = p0
        a = a / np.percentile(a, 99.9)
        a[a > 1] = 1
        b = p1
        b = b / np.percentile(b, 99.9)
        b[b > 1] = 1
        c = p2
        c = c / np.percentile(c, 99.9)
        c[c > 1] = 1

        visualization = cv2.merge([(np.uint8(255 * a)), (np.uint8(255 * b)), (np.uint8(255 * c))])
        visualization = cv2.cvtColor(visualization, cv2.COLOR_LAB2RGB)
        visualization[img.max(axis=-1) == 0] = 0
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
