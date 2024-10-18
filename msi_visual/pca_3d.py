from PIL import Image
from sklearn.decomposition import PCA

import cv2
import numpy as np
from matplotlib import pyplot as plt
import joblib
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from msi_visual.visualizations import visualizations_from_explanations
from msi_visual.utils import normalize, segment_visualization


class PCA3D:
    def __init__(self, start_bin=0, end_bin=None, max_iter=2000):
        self.k = 3
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.max_iter = max_iter
        self._trained = False

    def __repr__(self):
        return f"PCA-3D max_iter={self.max_iter}"

    def fit(self, images):
        if self.end_bin is None:
            self.end_bin = images[0].shape[-1]

        vector = np.concatenate([img[:, :, self.start_bin:self.end_bin].reshape(
            -1, images[0][:, :, self.start_bin:self.end_bin].shape[-1]) for img in images], axis=0)
        vector = vector.reshape((-1, vector.shape[-1]))

        # # Normalize the data
        # vector_mean = np.mean(vector, axis=0)
        # vector_std = np.std(vector, axis=0)
        # self.mean, self.std = vector_mean, vector_std
        # vector_normalized = (vector - vector_mean) / (1e-6 + vector_std)
        # vector_normalized[:, vector_std == 0] = 0

        # Transform the data using PCA
        self.pca = PCA(n_components=self.k)
        vector_transformed = self.pca.fit_transform(vector)
        # Save the PCA transform
        self.pca_transform = self.pca.transform

        self._trained = True


    def predict(self, img):
        vector = img[:, :, self.start_bin:self.end_bin].reshape(
            (-1, img.shape[-1]))

        #transformed_vector = (vector - self.mean) / (1e-6 + self.std)
        result = self.pca_transform(vector)
        result = result.reshape(img.shape[0], img.shape[1], result.shape[-1])
        return np.uint8(255 * normalize(result))

    def __call__(self, img):
        if not self._trained:
            self.fit([img])
        return self.predict(img)

    def segment_visualization(self,
                              img,
                              visualization,
                              method='spatial_norm'):

        return segment_visualization(visualization)

    def save(self, path):
        joblib.dump(self, path)
