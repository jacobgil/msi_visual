from umap import UMAP
import tensorflow as tf
import numpy as np
import os
import cv2
from pathlib import Path
import joblib
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion


def norm_umap_channel(channel, low=0.01, high=99.99):
    channel = channel - np.percentile(channel, low)
    channel = channel / np.percentile(channel, high)
    channel[channel < 0] = 0
    channel[channel > 1] = 1
    return channel


def embeddings_to_image(embeddings, rows, cols):
    color_image = embeddings.reshape((rows, cols, embeddings.shape[-1]))
    color_image[:, :, 0] = norm_umap_channel(color_image[:, :, 0])
    color_image[:, :, 1] = norm_umap_channel(color_image[:, :, 1])
    color_image[:, :, 2] = norm_umap_channel(color_image[:, :, 2])
    return np.uint8(color_image * 255)


class MSINonParametricUMAP:
    def __init__(
            self,
            n_components=3,
            min_dist=0.1,
            n_neighbors=15,
            metric='euclidean',
            start_bin=0,
            end_bin=None):
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.n_components = n_components
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors
        self.metric = metric

    def __repr__(self):
        return f"MSINonParametricUMAP min_dist: {self.min_dist} n_neighbors: {self.n_neighbors} metric: {self.metric}"

    def predict(self, img):
        vector = img.reshape(-1, img.shape[-1])[:, self.start_bin:self.end_bin]
        output = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric=self.metric).fit_transform(vector)
        output = output.reshape(img.shape[0], img.shape[1], output.shape[-1])
        visualization = embeddings_to_image(output, img.shape[0], img.shape[1])
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2LAB)
        visualization[img.max(axis=-1) == 0] = 0

        return visualization

    def __call__(self, img):
        return self.predict(img)
