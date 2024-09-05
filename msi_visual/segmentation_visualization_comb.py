from pathlib import Path
import msi_visual
import json
import glob
import cmapy
import cv2
from PIL import Image
from msi_visual.nmf_segmentation import NMFSegmentation
from msi_visual.auto_colorization import AutoColorizeRandom
from msi_visual.utils import normalize_image_grayscale, image_histogram_equalization
import numpy as np
from msi_visual.auto_colorization import AutoColorizeRandom


class SegmentationAndGuidingImageFolder:
    def __init__(self, seg_model, folder, number_colors=1):
        self.seg_model = seg_model
        self.folder = folder
        self.colors = []
        for _ in range(number_colors):
            self.colors.append(AutoColorizeRandom(json.load(open(Path(msi_visual.__path__[0] ) / "auto_color_schemes.json"))).colorize(0.5, seg_model.k))
        
        
    def __repr__(self):
        return f"SegmentationAndGuidingImageFolder: {str(self.seg_model)}_{Path(self.folder).stem}"

    def __call__(self, img):
        paths = glob.glob(self.folder + "/*.png" )
        print("paths", paths)
        seg = self.seg_model.predict(img).argmax(axis=0)
        result = []
        for path in paths:
            guiding_img = np.array(Image.open(path))
            if guiding_img.shape[:2] != img.shape[: 2]:
                continue

            guiding_img = cv2.cvtColor(guiding_img, cv2.COLOR_RGB2GRAY)
            guiding_img = np.float32(guiding_img) / 255
            for colors in self.colors:
                viz = self.predict(seg, colors, guiding_img)
                viz[img.max(axis=-1) == 0] = 0
                result.append(viz)
        return result

    def predict(self, seg, colors, guiding_img):
        result = np.zeros(
            shape=(seg.shape[0], seg.shape[1], 3),
            dtype=np.uint8)
        for region in np.unique(seg):
            region_mask = np.uint8(seg == region) * 255
            region_img = guiding_img.copy()
            region_img[region_mask > 0] = normalize_image_grayscale(
                region_img[region_mask > 0], 0.1, 99.9)
            num_bins = 2048
            region_img = image_histogram_equalization(
                region_img, region_mask, num_bins) / (num_bins - 1)
            region_img  = np.uint8(region_img  * 255)
            region_img = cv2.applyColorMap(
                region_img , cmapy.cmap(colors[region]))
            result[region_mask > 0] = region_img[region_mask > 0]
        return result