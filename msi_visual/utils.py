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


def cluster_id_labels_to_rgb_u8(
    labels_hw: np.ndarray,
    *,
    fill_value: int = -1,
    background: tuple[int, int, int] = (0, 0, 0),
    color_scheme: str = "gist_rainbow",
) -> np.ndarray:
    """
    Color integer cluster ids (0..K-1) with a matplotlib colormap; ``fill_value`` pixels
    (e.g. MiCS non-sampled sites) are ``background``.
    """
    labels_hw = np.asarray(labels_hw)
    h, w = labels_hw.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for bi, c in enumerate(background):
        rgb[:, :, bi] = int(c)
    m = labels_hw != fill_value
    if not np.any(m):
        return rgb
    kmax = int(labels_hw[m].max()) + 1
    cmap = plt.cm.get_cmap(color_scheme)
    for lid in range(kmax):
        sel = (labels_hw == lid) & m
        if not np.any(sel):
            continue
        r, g, b, _a = cmap(lid / max(kmax, 1))
        rgb[sel, 0] = int(255 * r)
        rgb[sel, 1] = int(255 * g)
        rgb[sel, 2] = int(255 * b)
    return rgb


def cluster_palette_rgb_u8(n_clusters: int, color_scheme: str = "gist_rainbow") -> np.ndarray:
    """RGB palette (K, 3) uint8 for cluster ids 0..K-1 (matches ``cluster_id_labels_to_rgb_u8``)."""
    k = max(1, int(n_clusters))
    cmap = plt.cm.get_cmap(color_scheme)
    rgba = cmap(np.arange(k, dtype=np.float64) / k)
    return (255.0 * rgba[:, :3]).astype(np.uint8)


def soft_cluster_probs_to_rgb_u8(
    probs_hwk: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    palette: np.ndarray | None = None,
    color_scheme: str = "gist_rainbow",
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Blend cluster colors by softmax probabilities: rgb = sum_k p_k * palette[k].
    ``probs_hwk`` is H×W×K with rows summing to ~1 on valid tissue pixels.
    """
    p = np.asarray(probs_hwk, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError(f"probs_hwk must be HxWxK, got {p.shape}")
    h, w, k = p.shape
    pal = (
        np.asarray(palette, dtype=np.float64)
        if palette is not None
        else cluster_palette_rgb_u8(k, color_scheme).astype(np.float64)
    )
    if pal.shape[0] != k:
        raise ValueError(f"palette rows {pal.shape[0]} != K={k}")
    rgb = (p.reshape(-1, k) @ pal).reshape(h, w, 3)
    out = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        out[~m] = np.asarray(background, dtype=np.uint8)
    return out