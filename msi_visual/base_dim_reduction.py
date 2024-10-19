import numpy as np
import joblib
from msi_visual.utils import normalize


class BaseDimReduction:
    def __init__(self, model, start_bin=0, end_bin=None):
        self.k = 3
        self.model = model
        self.start_bin = start_bin
        self.end_bin = end_bin
        self._trained = False


    def fit(self, images):
        if self.end_bin is None:
            self.end_bin = images[0].shape[-1]

        vector = np.concatenate([img[:, :, self.start_bin:self.end_bin].reshape(
            -1, images[0][:, :, self.start_bin:self.end_bin].shape[-1]) for img in images], axis=0)
        vector = vector.reshape((-1, vector.shape[-1]))
        self.model.fit(vector)
        self._trained = True


    def predict(self, img):
        vector = img[:, :, self.start_bin:self.end_bin].reshape(
            (-1, img.shape[-1]))
        result = self.model.transform(vector)
        result = result.reshape(img.shape[0], img.shape[1], result.shape[-1])
        return np.uint8(255 * normalize(result))

    def __call__(self, img):
        if not self._trained:
            self.fit([img])
        return self.predict(img)

    def save(self, path):
        joblib.dump(self, path)

class BaseDimReductionWithoutFit(BaseDimReduction):
    def __init__(self, model, name):
        super().__init__(model=model)
        self.name = name

    def __repr__(self):
        return self.name
        
    def __call__(self, img):
        vector = img.reshape((-1, img.shape[-1]))
        result = self.model.fit_transform(vector)
        result = result.reshape(img.shape[0], img.shape[1], result.shape[-1])
        return np.uint8(255 * normalize(result))

    def save(self, path):
        joblib.dump(self, path)

