import cv2
import cmapy
from PIL import Image
import numpy as np

from msi_visual.utils import brain_nmf_semantic_segmentation, normalize_image_grayscale, image_histogram_equalization

class SegmentationUMAPVisualization:
    def __init__(self, umap_model, segmentation_model):
        self.umap_model = umap_model
        self.segmentation_model = segmentation_model
        self.k = self.segmentation_model.k

    def compute(self, img, method):
        self.umap_1d = self.umap_model.predict(img)[:, :, 0]
        self.segmentation,  _, self.raw_segmentation  = self.segmentation_model.predict(img, method=method)

    def factorize(self, img):
        self.umap_1d = self.umap_model.predict(img)[:, :, 0]
        self.segmentation = self.segmentation_model.factorize(img)
        return self.segmentation

    def get_subsegmentation(self, img, roi_mask, segmentation, number_of_bins_for_comparison):
        segmentation_argmax = segmentation.argmax(axis=0)

        regions = np.unique(segmentation_argmax)
        num_regions = len(regions)
        sub_segmentation = np.zeros(shape=segmentation_argmax.shape[:2], dtype=np.int32)
        region_umaps = np.zeros(shape=segmentation_argmax.shape, dtype=np.float32)

        for region in regions:
            region_mask = np.uint8(segmentation_argmax == region) * 255
            region_mask[roi_mask == 0] = 0
            region_umap = self.umap_1d.copy()
            region_umap[region_mask == 0] = 0

            region_umap = normalize_image_grayscale(region_umap, high_percentile=99)
            num_bins = 2048
            region_umap = image_histogram_equalization(region_umap, region_mask, num_bins) / (num_bins - 1)
            region_umaps[region_mask > 0] = region_umap[region_mask > 0]

            bins = np.linspace(0, 1, number_of_bins_for_comparison)

            digitized = np.digitize(region_umap, bins)
            sub_segmentation[region_mask > 0] = digitized[region_mask > 0] + (number_of_bins_for_comparison + 2) * region
        return sub_segmentation, region_umaps

    def visualize_factorization(self,
                                img,
                                segmentation,
                                roi_mask,
                                color_scheme_per_region,
                                method='spatial_norm',
                                region_factors=None,
                                number_of_bins_for_comparison=5):
        
        segmentation, _, _, = self.segmentation_model.visualize_factorization(img,
                                                        segmentation,
                                                        method=method,
                                                        region_factors=region_factors)
        segmentation_argmax = segmentation.argmax(axis=0)
        sub_segmentation, region_umaps = self.get_subsegmentation(img, roi_mask, segmentation, number_of_bins_for_comparison)
        regions = np.unique(segmentation_argmax)
        # Create the colorful image
        visualizations = []
        for region in regions:
            region_mask = np.uint8(segmentation_argmax == region) * 255

            if roi_mask is not None:
                region_mask[roi_mask == 0] = 0

            region_umap = region_umaps.copy()
            region_umap = np.uint8(region_umap * 255)
            region_visualization = cv2.applyColorMap(region_umap, cmapy.cmap(color_scheme_per_region[region]))
            region_visualization = region_visualization[:, :, ::-1]
            region_visualization[region_mask == 0] = 0
            visualizations.append(region_visualization)
        
        visualizations = np.array(visualizations)
        visualizations = visualizations.max(axis=0)
        return segmentation, sub_segmentation, visualizations