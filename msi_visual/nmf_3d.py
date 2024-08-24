from PIL import Image
from sklearn.decomposition import NMF, non_negative_factorization
import cv2
import numpy as np
from matplotlib import pyplot as plt
import joblib
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from msi_visual.visualizations import visualizations_from_explanations
from msi_visual.utils import normalize, segment_visualization


class NMF3D:
    def __init__(self, start_bin=0, end_bin=None, max_iter=2000):
        self.k = 3
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.max_iter = max_iter

    def __repr__(self):
        return f"NMF-3D max_iter={self.max_iter}"

    def fit(self, images):
        if self.end_bin is None:
            self.end_bin = images[0].shape[-1]

        vector = np.concatenate([img[:, :, self.start_bin:self.end_bin].reshape(
            -1, images[0][:, :, self.start_bin:self.end_bin].shape[-1]) for img in images], axis=0)
        vector = vector.reshape((-1, vector.shape[-1]))
        self.model = NMF(
            n_components=self.k,
            init='random',
            random_state=0,
            max_iter=self.max_iter)
        self.W = self.model.fit_transform(vector)
        self.H = self.model.components_
        self.train_image_shapes = [img.shape[:2] for img in images]

    def visualize_training_components(self):
        result = []
        elements = 0
        for index, shape in enumerate(self.train_image_shapes):
            img_elements = shape[0] * shape[1]
            w = self.W[elements: elements + img_elements, :].copy()
            elements = elements + img_elements
            explanations = w.transpose().reshape(self.k, shape[0], shape[1])
            explanations = explanations.transpose((1, 2, 0))
            explanations = normalize(explanations)
            result.append(explanations)
        return result

    def predict(self, img):
        vector = img[:, :, self.start_bin:self.end_bin].reshape(
            (-1, img.shape[-1]))
        w_new, h_new, n_iter = non_negative_factorization(
            vector, H=self.H, W=None, n_components=self.k, update_H=False, random_state=0)
        result = w_new.transpose().reshape(self.k, img.shape[0], img.shape[1])
        return np.uint8(255 * normalize(result.transpose((1, 2, 0))))

    def __call__(self, img):
        self.fit([img])
        return self.predict(img)

    def segment_visualization(self,
                              img,
                              visualization,
                              method='spatial_norm'):

        return segment_visualization(visualization)

    def save(self, path):
        joblib.dump(self, path)
