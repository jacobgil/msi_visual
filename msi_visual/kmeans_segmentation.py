from PIL import Image
from sklearn.cluster import KMeans
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from msi_visual.visualizations import visualizations_from_explanations
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import cmapy
from msi_visual.utils import get_certainty


class KmeansSegmentation:
    def __init__(
            self,
            k,
            start_bin=0,
            end_bin=None,
            max_iter=200,
            color_scheme='gist_rainbow', method='spatial_norm'):

        self.k = k
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.max_iter = max_iter
        self.color_scheme = color_scheme
        self.method = method
        self._trained = False

    def __repr__(self):
        return f"KmeansSegmentation k:{self.k} max_iter:{self.max_iter} color_scheme:{self.color_scheme}"


    def fit(self, images):
        if self.end_bin is None:
            self.end_bin = images[0].shape[-1]
        vector = np.concatenate([img[:, :, self.start_bin:self.end_bin].reshape(
            (img.shape[0] * img.shape[1]), -1) for img in images], axis=0)
        vector = vector.reshape((-1, vector.shape[-1]))
        self.model = KMeans(n_clusters=self.k, init='random', random_state=0)
        self.model = self.model.fit(vector)
        centroids = self.model.cluster_centers_
        similarity = cosine_similarity(np.array(centroids), np.array(vector))

        self.training_components = similarity
        self.train_image_shapes = [img.shape[:2] for img in images]

    def get_colors(self, color_scheme='gist_rainbow'):
        _cmap = plt.cm.get_cmap(color_scheme)
        return [
            np.array(
                _cmap(i)) for i in np.arange(
                0,
                1,
                1.0 /
                self.k)]

    def visualize_training_components(self):
        result = []
        elements = 0
        for index, shape in enumerate(self.train_image_shapes):
            img_elements = shape[0] * shape[1]
            explanations = self.training_components[:,
                                                    elements: elements + img_elements].copy()
            explanations = explanations.reshape(self.k, shape[0], shape[1])
            elements = elements + img_elements

            spatial_sum_visualization, global_percentile_visualization, _, _ = visualizations_from_explanations(
                shape, explanations, self.get_colors())
            result.append(global_percentile_visualization)
        return result

    def predict(self, img):
        if not self._trained:
            self.fit([img])
            self._trained = True

        img = img[:, :, self.start_bin:self.end_bin]
        vector = img.reshape((-1, img.shape[-1]))
        centroids = self.model.cluster_centers_
        similarity = cosine_similarity(np.array(centroids), np.array(vector))
        segmentation = similarity.reshape(self.k, img.shape[0], img.shape[1])
        segmentation = torch.nn.Softmax(
            dim=0)(
            torch.from_numpy(
                segmentation *
                50)).numpy()
        return segmentation

    def __call__(self, img):
        return self.visualize(img, color_scheme=self.color_scheme, method=self.method)


    def segment_visualization(self,
                              img,
                              visualization,
                              color_scheme='gist_rainbow',
                              method='spatial_norm',
                              certainty_image=None,
                              region_factors=None):
        spatial_sum_visualization, global_percentile_visualization, normalized_sum, normalized_percentile = \
            visualizations_from_explanations(img.shape,
                                             visualization,
                                             self.get_colors(color_scheme),
                                             intensity_image=certainty_image,
                                             factors=region_factors)
        if method == 'spatial_norm':
            return normalized_sum, spatial_sum_visualization
        else:
            return normalized_percentile, global_percentile_visualization

    def visualize(
            self,
            img,
            color_scheme='gist_rainbow',
            method='spatial_norm'):

        if not self._trained:
            self.fit([img])

        factorization = self.predict(img)
        certainty = get_certainty(factorization)

        certainty = certainty / (1e-6 + np.max(certainty))
        certainty = np.uint8(255 * certainty)

        certainty = cv2.applyColorMap(certainty, cmapy.cmap('viridis'))[:, :, ::-1].copy()

        result = [self.segment_visualization(
            img, factorization, color_scheme, method=method)[1]]
        result.append(certainty)
        return result
