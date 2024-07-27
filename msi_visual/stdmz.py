import cv2
import cmapy
from PIL import Image
import numpy as np
from msi_visual.utils import normalize_image_grayscale, image_histogram_equalization


class AvgMZVisualization:
    def __init__(self):
        pass

    def predict(self, img):
        processed_img = img / img.sum(axis=-1)[:, :, None]
        mz = np.arange(processed_img.shape[-1])
        mz = mz[None, None, :]
        self.avgmz = np.mean((processed_img * mz), axis=-1)
        return self.avgmz

    def get_subsegmentation(self, img, roi_mask, avgmz,
                            number_of_bins_for_comparison):
        region_heatmap = avgmz.copy()
        region_heatmap[roi_mask == 0] = 0
        region_heatmap = normalize_image_grayscale(
            region_heatmap, high_percentile=99)
        num_bins = 2048
        region_heatmap = image_histogram_equalization(
            region_heatmap, roi_mask, num_bins) / (num_bins - 1)
        bins = np.linspace(0, 1, number_of_bins_for_comparison)
        digitized = np.digitize(region_heatmap, bins)
        return digitized, region_heatmap

    def segment_visualization(self,
                              img,
                              avgmz,
                              roi_mask,
                              color_scheme,
                              method='spatial_norm',
                              region_factors=None,
                              number_of_bins_for_comparison=5):

        sub_segmentation, heatmap = self.get_subsegmentation(
            img, roi_mask, avgmz, number_of_bins_for_comparison)
        heatmap = np.uint8(heatmap * 255)
        visualizations = cv2.applyColorMap(heatmap, cmapy.cmap(color_scheme))
        return sub_segmentation, visualizations
