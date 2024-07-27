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

    if concept_labels is not None:
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(result.shape[1] * px, result.shape[0] * px))
        plt.rcParams['legend.fontsize'] = int(
            14 * result.shape[0] / 256 / max(1, n_components / 6))
        lw = 5 * result.shape[0] / 256
        lines = [Line2D([0], [0], color=colors[i], lw=lw)
                 for i in range(n_components)]
        plt.legend(lines,
                   concept_labels,
                   mode="expand",
                   fancybox=True,
                   shadow=True)

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.axis('off')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt.close(fig=fig)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.resize(data, (result.shape[1], result.shape[0]))
        result = np.hstack((result, data))
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

    print(shape, "shape")
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


def analyze_region_differences(ion_image: np.ndarray,
                               region_gray_image: np.ndarray,
                               mzs_per_bin: int,
                               number_of_bins=3) -> dict[float]:
    _, bins = np.histogram(region_gray_image[:], bins=number_of_bins)
    digitized = np.digitize(region_gray_image, bins) - 1
    bin_values = []
    bins = sorted(np.unique(digitized[:]))
    for bin in bins:
        digitized_mask = (digitized == bin)
        digitized_mask[ion_image.max(axis=-1) == 0] = 0
        bin_values.append(ion_image[digitized_mask > 0])
    region_pair_aucs = {}
    for i in range(len(bin_values)):
        for j in range(i + 1, len(bin_values)):
            values_a, values_b = bin_values[i], bin_values[j]
            both = np.concatenate((values_a, values_b), axis=0)
            peaks_a = np.percentile(values_a, 30, axis=0)
            peaks_a, _ = find_peaks(peaks_a, height=(0.01 * 1e10, None))
            peaks_b = np.percentile(values_b, 30, axis=0)
            peaks_b, _ = find_peaks(peaks_b, height=(0.01 * 1e10, None))

            labels = [0] * len(values_a) + [1] * len(values_b)
            peaks = list(peaks_a) + list(peaks_b)
            aucs = defaultdict(float)
            full_range_aucs = defaultdict(float)
            for mz in tqdm.tqdm(peaks):
                auc = roc_auc_score(labels, both[:, mz])
                auc = 2 * auc - 1
                full_range_aucs[mz] += auc
                mz = int(mz * mzs_per_bin / 5)
                aucs[mz] += auc
            region_pair_aucs[(i, j)] = (aucs, full_range_aucs)

    return bins, digitized, region_pair_aucs


def get_qr_images(
        region_gray_image,
        bins,
        digitized,
        region_pair_aucs,
        mzs_per_bin,
        color_scheme):
    qr_images = []
    qr_cols, qr_rows = int(1 + (5005 * mzs_per_bin / 5)
                           ** 0.5), int(1 + (5005 * mzs_per_bin / 5)**0.5)
    qr_cell_size = 10

    for (i, j), (aucs, _) in region_pair_aucs.items():
        color_b = cv2.applyColorMap(np.uint8(np.ones((qr_cell_size,
                                                      qr_cell_size)) * region_gray_image[digitized == bins[j]].mean()),
                                    cmapy.cmap(color_scheme))[:,
                                                              :,
                                                              ::-1]
        color_a = cv2.applyColorMap(np.uint8(np.ones((qr_cell_size,
                                                      qr_cell_size)) * region_gray_image[digitized == bins[i]].mean()),
                                    cmapy.cmap(color_scheme))[:,
                                                              :,
                                                              ::-1]
        qr_image = np.zeros(
            (qr_rows *
             qr_cell_size +
             qr_cell_size,
             qr_cols *
             qr_cell_size +
             qr_cell_size,
             3),
            dtype=np.uint8)

        for mz, auc in aucs.items():
            if auc > 0:
                color = color_b.copy()
            else:
                color = color_a.copy()
            auc = abs(auc)
            auc = auc**3
            lab = cv2.cvtColor(color, cv2.COLOR_RGB2Lab)
            l, a, b = cv2.split(lab)
            l = np.uint8(l * auc)
            lab = cv2.merge([l, a, b])
            color = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
            row, col = int(mz / qr_rows), int(mz % qr_cols)
            qr_image[row *
                     qr_cell_size: row *
                     qr_cell_size +
                     qr_cell_size, col *
                     qr_cell_size: col *
                     qr_cell_size +
                     qr_cell_size, :] = color
        qr_image[-qr_cell_size:, -2 *
                 qr_cell_size:] = np.hstack((color_a, color_b))
        qr_images.append(qr_image)
    return qr_images


def get_difference_summary_table(
        region_gray_image,
        digitized,
        bins,
        top_mzs,
        color_scheme):
    rows = []
    for i, j in top_mzs:
        color_b = cv2.applyColorMap(np.uint8(np.ones(
            (100, 100)) * region_gray_image[digitized == bins[j]].mean()), cmapy.cmap(color_scheme))[:, :, ::-1]
        color_a = cv2.applyColorMap(np.uint8(np.ones(
            (100, 100)) * region_gray_image[digitized == bins[i]].mean()), cmapy.cmap(color_scheme))[:, :, ::-1]
        combination = np.hstack((color_a, color_b))
        cells = []
        top_pos_auc = {k: v for k, v in sorted(
            top_mzs[i, j].items(), key=lambda item: item[1])[::-1][:10]}
        top_mzs_from_both = {}
        top_neg_auc = {k: v for k, v in sorted(
            top_mzs[i, j].items(), key=lambda item: -item[1])[::-1][:10]}

        for k, v in top_neg_auc.items():
            if v < 0:
                top_mzs_from_both[k] = v

        for k, v in top_pos_auc.items():
            if v > 0:
                top_mzs_from_both[k] = v

        for mz, auc in top_mzs_from_both.items():
            if abs(auc) < 0.7:
                continue
            mz = 300 + mz / 5
            cell = np.ones((100, 100, 3), dtype=np.uint8) * 255

            if auc > 0:
                icon = cv2.applyColorMap(np.uint8(np.ones(
                    (8, 8)) * region_gray_image[digitized == bins[j]].mean()), cmapy.cmap(color_scheme))[:, :, ::-1]
            else:
                icon = cv2.applyColorMap(np.uint8(np.ones(
                    (8, 8)) * region_gray_image[digitized == bins[i]].mean()), cmapy.cmap(color_scheme))[:, :, ::-1]

            cell[20: 20 + icon.shape[0], 10: 10 + icon.shape[1], :] = icon

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 50)
            fontScale = 0.5
            fontColor = (0, 0, 00)
            thickness = 1
            lineType = 1
            cell = cv2.putText(cell, f"{mz:.3f}",
                               bottomLeftCornerOfText,
                               font,
                               fontScale,
                               fontColor,
                               thickness,
                               lineType)
            cells.append(cell)
        if len(cells) < 20:
            for _ in range(20 - len(cells)):
                cells.append(np.ones((100, 100, 3), dtype=np.uint8) * 255)

        cells = np.hstack(cells)
        cells = np.hstack((combination, cells))
        rows.append(cells)
    rows = np.vstack(rows)
    return rows


def create_ion_image(img, mz_index):
    ion = img[:, :, mz_index]
    ion = ion / np.percentile(ion[:], 99)
    ion[ion > 1] = 1
    ion = np.uint8(255 * ion)
    ion = cv2.applyColorMap(ion, cmapy.cmap('viridis'))[:, :, ::-1]
    return ion


def get_mask(visualization, segmentation_mask, x, y):
    label = segmentation_mask[y, x]

    result = visualization.copy()
    mask = np.uint8(255 * (segmentation_mask == label))
    result[mask == 0] = 0
    return result


class ObjectsComparison:
    def __init__(
            self,
            mz_image,
            segmentation_mask,
            visualization,
            start_mz=300,
            bins_per_mz=5):
        self.mz_image = norm_funtion(mz_image)
        self.segmentation_mask = segmentation_mask
        self.visualization = visualization
        self.start_mz = start_mz
        self.bins_per_mz = bins_per_mz

    def compare_point(self, point, size=3):
        x, y = point
        mask = np.zeros(
            (self.mz_image.shape[0],
             self.mz_image.shape[1]),
            dtype=np.uint8)
        mask = cv2.circle((mask, (x, y), size, 255, -1))
        mask[y, x] = 0

        values_a = self.mz_image[y, x]
        values_b = self.mz_image[mask > 0]


class RegionComparison:
    def __init__(
            self,
            mz_image,
            segmentation_mask,
            visualization,
            mzs,
            bins_per_mz=5):
        self.mz_image = mz_image
        self.segmentation_mask = segmentation_mask
        self.visualization = visualization
        self.mzs = mzs
        self.bins_per_mz = bins_per_mz

    def ranking_comparison(self, mask_a, mask_b, peak_minimum=0):
        values_a = self.mz_image[mask_a > 0]
        values_b = self.mz_image[mask_b > 0]
        t0 = time.time()

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

        result = scipy.stats.mannwhitneyu(
            values_b[:, peaks], values_a[:, peaks], keepdims=True)
        us = result.statistic[0, :]
        p = result.pvalue[0, :]
        print("u test took", time.time() - t0)

        result = us / (values_a.shape[0] * values_b.shape[0])
        result = dict(zip(range(len(result)), result))

        result = {peaks[index]: u for index,
                  u in result.items() if p[index] < 0.05}
        return result

    def compare_one_point(self, point, size=3):
        x, y = point
        x, y = int(x), int(y)
        mask_b = np.zeros(
            (self.mz_image.shape[0],
             self.mz_image.shape[1]),
            dtype=np.uint8)
        mask_b = cv2.circle(mask_b, (x, y), size, 255, -1)
        mask_b[y, x] = 0

        mask_a = np.zeros(
            (self.mz_image.shape[0],
             self.mz_image.shape[1]),
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

        values_a = self.mz_image[y1, x1, :]
        values_b = self.mz_image[y2, x2, :]

        indices_a = list(np.argsort(values_a)[-top_mzs:])
        indices_b = list(np.argsort(values_b)[-top_mzs:])
        indices = sorted(indices_a + indices_b)

        x = self.mz_image[:, :, indices]
        x = x.reshape((x.shape[0] * x.shape[1], -1))
        ranks = x.argsort(
            axis=0).argsort(
            axis=0).reshape(
            (self.mz_image.shape[0], self.mz_image.shape[1], -1))

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

    @lru_cache()
    def compare_two_labels(self, label_a, label_b):
        mask_a = np.uint8(255 * (self.segmentation_mask == label_a))
        mask_b = np.uint8(255 * (self.segmentation_mask == label_b))
        aucs = self.ranking_comparison(mask_a, mask_b)
        return aucs, mask_a, mask_b

    def visualize_comparison_between_regions(self, point_a, point_b):
        (aucs, mask_a, mask_b), label_a, label_b = self.compare_two_regions(
            point_a, point_b)
        color_a = self.visualization[mask_a > 0]
        color_a = color_a.mean(axis=0)
        color_b = self.visualization[mask_b > 0].mean(axis=0)

        aucs = {self.mzs[mz_index]: aucs[mz_index] for mz_index in aucs}

        image = self._create_auc_visualization(aucs, color_a, color_b)

        return image, label_a, label_b, (color_a, color_b, aucs)

    def _create_auc_visualization(self, aucs, color_a, color_b):
        combination = np.hstack((color_a *
                                 np.ones((100, 100, 3), dtype=np.uint8), color_b *
                                 np.ones((100, 100, 3), dtype=np.uint8)))
        combination = np.uint8(combination)
        cells = []
        sorted_indices = sorted(
            list(
                aucs.keys()),
            key=lambda mz: aucs[mz],
            reverse=True)

        for mz in sorted_indices:
            auc = aucs[mz]
            cell = np.ones((100, 100, 3), dtype=np.uint8) * 255

            if auc > 0.5:
                icon = np.ones((8, 8, 3)) * color_b
            else:
                icon = np.ones((8, 8, 3)) * color_a

            cell[20: 20 + icon.shape[0], 10: 10 + icon.shape[1], :] = icon

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 50)
            fontScale = 0.5
            fontColor = (0, 0, 00)
            thickness = 1
            lineType = 1
            cell = cv2.putText(cell, f"{mz:.3f}",
                               bottomLeftCornerOfText,
                               font,
                               fontScale,
                               fontColor,
                               thickness,
                               lineType)
            cells.append(cell)
        if len(cells) <= 20:
            for _ in range(20 - len(cells)):
                cells.append(np.ones((100, 100, 3), dtype=np.uint8) * 255)
            result = np.hstack(cells)
            # result = np.hstack((combination, result))
        elif len(cells) > 20:
            result = []
            for i in range(math.ceil(len(cells) / 20)):
                row = cells[i * 20: i * 20 + 20]
                if len(row) < 20:
                    for _ in range(20 - len(row)):
                        row.append(
                            np.ones(
                                (100, 100, 3), dtype=np.uint8) * 255)
                row = np.hstack(row)
                # row = np.hstack((combination, row))
                result.append(row)
            print(len(result))
            result = np.vstack(result)
            print(result.shape)

        return result
