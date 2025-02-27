import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
import cv2
import scipy
import math
import tqdm
import time
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import cmapy
from PIL import Image
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from msi_visual.utils import set_region_importance


def show_factorization_on_image(img: np.ndarray,
                                explanations: np.ndarray,
                                colors: list[np.ndarray] = None,
                                image_weight: float = 0.5,
                                concept_labels: list = None,
                                intensity_image: np.ndarray = None) -> np.ndarray:
    """ Color code the different component heatmaps on top of the image.
        Every component color code will be magnified according to the heatmap itensity
        (by modifying the V channel in the HSV color space),
        and optionally create a lagend that shows the labels.

        Since different factorization component heatmaps can overlap in principle,
        we need a strategy to decide how to deal with the overlaps.
        This keeps the component that has a higher value in it's heatmap.

    :param img: The base image RGB format.
    :param explanations: A tensor of shape num_componetns x height x width, with the component visualizations.
    :param colors: List of R, G, B colors to be used for the components.
                   If None, will use the gist_rainbow cmap as a default.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * visualization.
    :concept_labels: A list of strings for every component. If this is paseed, a legend that shows
                     the labels and their colors will be added to the image.
    :returns: The visualized image.
    """
    n_components = explanations.shape[0]
    if colors is None:
        # taken from https://github.com/edocollins/DFF/blob/master/utils.py
        _cmap = plt.cm.get_cmap('gist_rainbow')
        colors = [
            np.array(
                _cmap(i)) for i in np.arange(
                0,
                1,
                1.0 /
                n_components)]
    concept_per_pixel = explanations.argmax(axis=0)
    masks = []
    for i in range(n_components):
        mask = np.zeros(shape=(img.shape[0], img.shape[1], 3))
        mask[:, :, :] = colors[i][:3]
        if intensity_image is not None:
            explanation = intensity_image.copy()
        else:
            explanation = explanations[i]
        explanation[concept_per_pixel != i] = 0
        mask = np.uint8(mask * 255)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
        mask[:, :, 2] = np.uint8(255 * explanation)
        mask = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
        mask = np.float32(mask) / 255
        masks.append(mask)

    mask = np.sum(np.float32(masks), axis=0)
    result = img * image_weight + mask * (1 - image_weight)
    result = np.uint8(result * 255)
    return result


def get_colors(number, colormap='gist_rainbow'):
    _cmap = plt.cm.get_cmap(colormap)
    colors_for_components = [
        np.array(
            _cmap(i)) for i in np.arange(
            0,
            1,
            1.0 /
            number)]
    return colors_for_components


def visualizations_from_explanations(
        shape,
        explanations,
        colors,
        intensity_image=None,
        factors=None):
    normalized_by_spatial_sum = explanations / \
        (1e-6 + explanations.sum(axis=(1, 2))[:, None, None])
    normalized_by_spatial_sum = normalized_by_spatial_sum / \
        (1e-6 + np.percentile(normalized_by_spatial_sum, 99, axis=(1, 2))[:, None, None])
    normalized_by_global_percentile = explanations / \
        np.percentile(explanations, 99, axis=(0, 1, 2))

    if factors is not None:
        normalized_by_spatial_sum = set_region_importance(
            normalized_by_spatial_sum, factors)
        normalized_by_global_percentile = set_region_importance(
            normalized_by_global_percentile, factors)

    
    spatial_sum_visualization = show_factorization_on_image(
        np.zeros(
            shape=(
                (shape[0],
                 shape[1],
                 3))),
        normalized_by_spatial_sum.copy(),
        image_weight=0.0,
        colors=colors,
        intensity_image=intensity_image)
    global_percentile_visualization = show_factorization_on_image(
        np.zeros(
            shape=(
                (shape[0],
                 shape[1],
                 3))),
        normalized_by_global_percentile.copy(),
        image_weight=0.0,
        colors=colors,
        intensity_image=intensity_image)
    return spatial_sum_visualization, global_percentile_visualization, normalized_by_spatial_sum, normalized_by_global_percentile


def create_ion_img(img, mz_index):
    ion = img[:, :, mz_index]
    ion = ion / np.percentile(ion[:], 99)
    ion[ion > 1] = 1
    return ion

def create_ion_heatmap(img, mz_index):
    ion = create_ion_img(img, mz_index)
    ion = np.uint8(255 * ion)
    # ion = np.uint8(255 * raw_ion)
    # ion = cv2.applyColorMap(ion, cmapy.cmap('viridis'))[:, :, ::-1].copy()
    
    mask = img.max(axis=-1) > 0
    # Convert grayscale to RGB IHC-like coloring
    # Create RGB image with brown for high values and light pink for low values
    rgb = np.zeros((ion.shape[0], ion.shape[1], 3), dtype=np.uint8)
    
    # Brown color (RGB: 139, 69, 19) for high values
    # Light pink (RGB: 255, 228, 225) for low values
    rgb[:,:,0] = np.uint8(255 - ion * 0.45)  # R channel 
    rgb[:,:,1] = np.uint8(228 - ion * 0.62)  # G channel
    rgb[:,:,2] = np.uint8(225 - ion * 0.81)  # B channel


    # Create a colormap from white to brown
    white = np.array([255, 255, 255])
    brown = np.array([139, 69, 19]) 
    
    # Create normalized intensity values between 0 and 1
    norm_ion = ion.astype(float) / 255
    
    # For each pixel, interpolate between white and brown based on intensity
    for i in range(3):  # RGB channels
        rgb[:,:,i] = np.uint8(white[i] + (brown[i] - white[i]) * norm_ion)

    rgb[mask == 0] = 0



    #mz_img = visualizations.create_ion_image(img, mz_index) * 1
    ion = rgb


    return ion

def get_mask(visualization, segmentation_mask, x, y):
    label = segmentation_mask[y, x]
    result = visualization.copy()
    mask = np.uint8(255 * (segmentation_mask == label))
    result[mask == 0] = 0
    return result



class RegionComparison:
    def __init__(
            self,
            img,
            mzs):
        self.img = img
        self.mzs = mzs

    def ranking_comparison(self, mask_a, mask_b, peak_minimum=0, method="U-Test"):
        values_a = self.img[mask_a > 0]
        values_b = self.img[mask_b > 0]
        t0 = time.time()

        #peaks = list(range(self.img.shape[-1]))

        peaks_a30, _ = find_peaks(
            np.percentile(
                values_a, 30, axis=0), height=(
                peak_minimum, None))
        peaks_b30, _ = find_peaks(
            np.percentile(
                values_b, 30, axis=0), height=(
                peak_minimum, None))

        peaks = list(peaks_a30) + list(peaks_b30)
        peaks = sorted(list(set(peaks)))

        if method == "U-Test":
            result = scipy.stats.mannwhitneyu(
                values_a[:, peaks], values_b[:, peaks], keepdims=True)
            us = result.statistic[0, :]
            p = result.pvalue[0, :]
            print("u test took", time.time() - t0)

            result = us / (values_a.shape[0] * values_b.shape[0])
            result = dict(zip(range(len(result)), result))

            result = {self.mzs[peaks[index]]: u for index,
                    u in result.items() if p[index] < 0.05}

        else:
            ion_images = self.img[:, :, peaks]
            ion_images = ion_images / np.max(ion_images, axis=(0, 1))[None, None, :]
            ion_images_a = ion_images[mask_a > 0]
            ion_images_b = ion_images[mask_b > 0]
            score_a = ion_images_a.mean(axis=0)
            score_b = ion_images_b.mean(axis=0)
            score = score_a / (1e-5 + score_b)
            result = {self.mzs[peaks[i]] : score[i] for i in range(len(peaks))}

            norm = np.max(list(result.values()))
            result = {mz: result[mz]/norm for mz in result}

        return result

    def compare_one_point(self, point, size=3):
        x, y = point
        x, y = int(x), int(y)
        mask_b = np.zeros(
            (self.img.shape[0],
             self.img.shape[1]),
            dtype=np.uint8)
        mask_b = cv2.circle(mask_b, (x, y), size, 255, -1)
        mask_b[y, x] = 0

        mask_a = np.zeros(
            (self.img.shape[0],
             self.img.shape[1]),
            dtype=np.uint8)
        mask_a[y, x] = 255
        aucs = self.ranking_comparison(mask_a, mask_b)
        return aucs, mask_a, mask_b

    def visualize_object_comparison(self, point, size=3):
        x, y = point
        aucs, mask_a, mask_b = self.compare_one_point(point, size=size)
        color_a = self.visualization[y, x]
        color_b = np.int32([0, 0, 0])

        image = self._create_auc_visualization(aucs, color_a, color_b)
        return image

    def compare_two_points(self, point_a, point_b, top_mzs=200):
        x1, y1 = point_a
        x2, y2 = point_b

        values_a = self.img[y1, x1, :]
        values_b = self.img[y2, x2, :]

        indices_a = list(np.argsort(values_a)[-top_mzs:])
        indices_b = list(np.argsort(values_b)[-top_mzs:])
        indices = sorted(indices_a + indices_b)

        x = self.img[:, :, indices]
        x = x.reshape((x.shape[0] * x.shape[1], -1))
        ranks = x.argsort(
            axis=0).argsort(
            axis=0).reshape(
            (self.img.shape[0], self.img.shape[1], -1))

        scores = (-ranks[y1, x1] + ranks[y2, x2]) / \
            (ranks.shape[0] * ranks.shape[0])
        scores = (scores + 1) / 2
        indices = [self.mzs[mz_index] for mz_index in indices]
        return dict(zip(indices, scores))

    def compare_two_regions(self, point_a, point_b):
        x1, y1 = point_a
        x2, y2 = point_b
        label_a = self.segmentation_mask[y1, x1]
        label_b = self.segmentation_mask[y2, x2]
        return self.compare_two_labels(label_a, label_b), label_a, label_b

    def visualize_comparison_between_regions(self, point_a, point_b):
        (aucs, mask_a, mask_b), label_a, label_b = self.compare_two_regions(
            point_a, point_b)
        color_a = self.visualization[mask_a > 0]
        color_a = color_a.mean(axis=0)
        color_b = self.visualization[mask_b > 0].mean(axis=0)

        aucs = {self.mzs[mz_index]: aucs[mz_index] for mz_index in aucs}

        image = self._create_auc_visualization(aucs, color_a, color_b)

        return image, label_a, label_b, (color_a, color_b, aucs)
