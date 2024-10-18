from PIL import Image
import phate
import cv2
import numpy as np
import joblib
from msi_visual.utils import normalize

class PHATE3D:
    def __init__(self, start_bin=0, end_bin=None):
        self._trained = False
        self.start_bin = start_bin
        self.end_bin = end_bin


    def __repr__(self):
        return f"PHATE3D"

    def __call__(self, img):
        vector = img[:, :, self.start_bin:self.end_bin].reshape(
            (-1, img.shape[-1]))

        phate_operator = phate.PHATE(n_components=3, n_jobs=5)
        result = phate_operator.fit_transform(vector)
        result = result.reshape(img.shape[0], img.shape[1], result.shape[-1])
        return np.uint8(255 * normalize(result))

    def save(self, path):
        joblib.dump(self, path)
