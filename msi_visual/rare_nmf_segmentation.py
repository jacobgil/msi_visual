import cv2
import cmapy
from PIL import Image
import numpy as np
from msi_visual.utils import normalize_image_grayscale, image_histogram_equalization
from msi_visual.normalization import spatial_total_ion_count


def get_pr(normalized, high=99.9, low=99):
    novel = np.percentile(normalized, high, axis=-1) / \
        (1e-5 + np.percentile(normalized, low, axis=-1))
    novel = novel / np.percentile(novel, 99)
    novel[novel > 1] = 1
    return np.uint8(255 * novel)


class SegmentationPercentileRatio:
    def __init__(self, segmentation_model):
        self.segmentation_model = segmentation_model
        self.k = self.segmentation_model.k

    def predict(self, img):
        self.segmentation = self.segmentation_model.predict(img)
        rare = get_pr(img)
        return self.segmentation, rare

    def get_subsegmentation(
            self,
            img,
            roi_mask,
            segmentation,
            avgmz,
            number_of_bins_for_comparison):
        segmentation_argmax = segmentation.argmax(axis=0)

        regions = np.unique(segmentation_argmax)
        num_regions = len(regions)
        sub_segmentation = np.zeros(
            shape=segmentation_argmax.shape[:2], dtype=np.int32)
        region_heatmaps = np.zeros(
            shape=segmentation_argmax.shape,
            dtype=np.float32)

        for region in regions:
            region_mask = np.uint8(segmentation_argmax == region) * 255
            region_mask[roi_mask == 0] = 0
            region_heatmap = avgmz.copy()
            region_heatmap[region_mask == 0] = 0

            region_heatmap = normalize_image_grayscale(
                region_heatmap, high_percentile=99)
            num_bins = 2048
            region_heatmap = image_histogram_equalization(
                region_heatmap, region_mask, num_bins) / (num_bins - 1)
            region_heatmaps[region_mask > 0] = region_heatmap[region_mask > 0]

            bins = np.linspace(0, 1, number_of_bins_for_comparison)

            digitized = np.digitize(region_heatmap, bins)
            sub_segmentation[region_mask > 0] = digitized[region_mask >
                                                          0] + (number_of_bins_for_comparison + 2) * region
        return sub_segmentation, region_heatmaps

    def segment_visualization(self,
                              img,
                              segmentation,
                              visualization,
                              roi_mask,
                              color_scheme_per_region,
                              method='spatial_norm',
                              region_factors=None,
                              number_of_bins_for_comparison=5):

        segmentation, _ = self.segmentation_model.segment_visualization(
            img, segmentation, method=method, region_factors=region_factors)
        segmentation_argmax = segmentation.argmax(axis=0)
        sub_segmentation, region_UMAP s = self.get_subsegmentation(
            img, roi_mask, segmentation, visualization, number_of_bins_for_comparison)
        regions = np.unique(segmentation_argmax)
        # Create the colorful image
        visualizations = []
        for region in regions:
            region_mask = np.uint8(segmentation_argmax == region) * 255

            if roi_mask is not None:
                region_mask[roi_mask == 0] = 0

            region_UMAP  = region_UMAP s.copy()
            region_UMAP  = np.uint8(region_UMAP  * 255)
            region_visualization = cv2.applyColorMap(
                region_UMAP , cmapy.cmap(color_scheme_per_region[region]))
            region_visualization = region_visualization[:, :, ::-1]
            region_visualization[region_mask == 0] = 0
            visualizations.append(region_visualization)

        visualizations = np.array(visualizations)
        visualizations = visualizations.max(axis=0)
        return segmentation, sub_segmentation, visualizations
