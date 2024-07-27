from PIL import Image
from sklearn.decomposition import NMF, non_negative_factorization
import cv2
import numpy as np
from matplotlib import pyplot as plt
import joblib
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from msi_visual.visualizations import visualizations_from_explanations
from msi_visual.percentile_ratio import percentile_ratio_rgb


class PercentileRatioSegmentation:
    def __init__(self, equalize=False):
        self.equalize = equalize

    def predict(self, img):
        return np.float32(
            percentile_ratio_rgb(
                img, equalize=self.equalize)) / 255

    def segment_visualization(self,
                              img,
                              contributions,
                              method='spatial_norm'):

        number_of_bins_for_comparison = 5
        bins = np.linspace(0, 1, number_of_bins_for_comparison)

        digitized_a = np.digitize(contributions[:, :, 0], bins)
        digitized_b = np.digitize(contributions[:, :, 1], bins)
        digitized_c = np.digitize(contributions[:, :, 2], bins)

        digitized = digitized_a * number_of_bins_for_comparison * \
            number_of_bins_for_comparison + digitized_b * number_of_bins_for_comparison + digitized_c

        return digitized, np.uint8(255 * contributions)
