from msi_visual.base_dim_reduction import BaseDimReduction
import pacmap
import numpy as np
import joblib
from msi_visual.utils import normalize


class PACMAC3D():
    def __init__(self):
        self.model = pacmap.PaCMAP(n_components=3)

    def __repr__(self):
        return f"PACMAC-3D"
        
    def __call__(self, img):
        vector = img.reshape((-1, img.shape[-1]))
        result = self.model.fit_transform(vector)
        print(result.shape)
        result = result.reshape(img.shape[0], img.shape[1], result.shape[-1])
        return np.uint8(255 * normalize(result))

    def save(self, path):
        joblib.dump(self, path)

