import cv2
import time
import glob
import numpy as np
import cmapy
import time
import cv2
from PIL import Image
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


def core_sets(data, Np):
    """Reduces (NxD) data matrix from N to Np data points.

    Args:
        data: ndarray of shape [N, D]
        Np: number of data points in the coreset
    Returns:
        coreset: ndarray of shape [Np, D]
        weights: 1darray of shape [Np, 1]
    """
    N = data.shape[0]
    D = data.shape[1]

    # compute mean
    u = np.mean(data, axis=0)

    # compute proposal distribution
    q = np.linalg.norm(data - u, axis=1)**2
    sum = np.sum(q)
    q = 0.5 * (q / sum + 1.0 / N)

    # get sample and fill coreset
    samples = np.random.choice(N, Np, p=q)
    coreset = data[samples]
    weights = 1.0 / (q[samples] * Np)

    return coreset, weights, samples


def get_outlier_image(img):
    reshaped = img.reshape(img.shape[0] * img.shape[1], -1)
    # nonzero = reshaped[reshaped.max(axis=-1) > 0]

    coreset_indices = np.random.choice(
        np.arange(len(reshaped)), size=100, replace=False)
    coreset_indices = [
        i for i in coreset_indices if reshaped[i, :].max(axis=-1) > 0]
    coreset = reshaped[coreset_indices]

    chebyshev = pairwise_distances(reshaped, coreset, metric='chebyshev')
    for i, coreset_index in enumerate(coreset_indices):
        chebyshev[coreset_index, i] = 10000

    chebyshev = chebyshev.min(axis=-1)

    chebyshev = chebyshev.reshape((img.shape[0], img.shape[1]))
    chebyshev = chebyshev / np.percentile(chebyshev, 99.9)
    chebyshev[chebyshev > 1] = 1

    chebyshev = chebyshev - np.percentile(chebyshev, 0.01)
    chebyshev[chebyshev < 0] = 0

    visualization = cv2.merge([(np.uint8(255 * chebyshev)),
                               (np.uint8(255 * chebyshev)),
                               (np.uint8(255 * chebyshev))])
    visualization[img.max(axis=-1) == 0] = 0
    visualization = cv2.applyColorMap(
        visualization, cmapy.cmap('viridis'))[:, :, ::-1]

    return visualization
