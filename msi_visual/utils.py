import numpy as np
from sklearn.decomposition import non_negative_factorization
from matplotlib import pyplot as plt
from collections import defaultdict
from scipy.stats import entropy


def segment_visualization(visualization: np.ndarray, number_of_bins_for_comparison: int = 5):
    bins = np.linspace(0, 255, number_of_bins_for_comparison)

    digitized_a = np.digitize(visualization[:, :, 0], bins)
    digitized_b = np.digitize(visualization[:, :, 1], bins)
    digitized_c = np.digitize(visualization[:, :, 2], bins)

    digitized = digitized_a * number_of_bins_for_comparison * \
        number_of_bins_for_comparison + digitized_b * number_of_bins_for_comparison + digitized_c

    return digitized


def normalize(visualiation, low=0.001, high=99.999, mask=None):
    """
    Normalize each channel to [0,1] using percentile scaling.
    If mask is provided (2D bool), percentiles are computed only over masked pixels.
    This avoids skewing the color scale when the model was trained on a subset (e.g. ROI).
    """
    result = visualiation.copy()
    for i in range(result.shape[-1]):
        channel = result[:, :, i]
        if mask is not None and np.any(mask):
            vals = channel[mask]
            if len(vals) > 0:
                p_low = np.percentile(vals, low)
                p_high = np.percentile(vals, high)
            else:
                p_low, p_high = np.percentile(channel, low), np.percentile(channel, high)
        else:
            p_low = np.percentile(channel, low)
            p_high = np.percentile(channel, high)
        channel = channel - p_low
        channel[channel < 0] = 0
        channel = channel / (p_high - p_low + 1e-8)
        channel[channel > 1] = 1
        result[:, :, i] = channel
    return result


def normalize_image_grayscale(
        grayscale,
        low_percentile: int = 0.1,
        high_percentile: int = 99.9):
    a = grayscale.copy()
    low = np.percentile(a[:], low_percentile)
    a = (a - low)
    a[a < 0] = 0
    high = np.percentile(a[:], high_percentile)
    a = a / (1e-7 + high)
    a[a < 0] = 0
    a[a > 1] = 1
    return a


def image_histogram_equalization(image, mask, number_bins=256):
    # from
    # http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(
        image[mask > 0].flatten(), number_bins, density=True)

    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (number_bins - 1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def get_certainty(segmentation):
    normalized = segmentation / (1e-6 + segmentation.sum(axis=0))
    e = entropy(normalized, axis=0, base=normalized.shape[0])
    e[np.isnan(e)] = 0
    e = 1 - e
    return e


def set_region_importance(segmentation_mask, factors):
    for label, factor in factors.items():
        segmentation_mask[label, :,
                          :] = segmentation_mask[label, :, :] * factor
    return segmentation_mask