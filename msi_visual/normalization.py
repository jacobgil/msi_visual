
import numpy as np
import time


def total_ion_count(img):
    return img / (1e-6 + np.sum(img, axis=-1)[:, :, None])


def median_ion(img):
    return img / (1e-6 + np.median(img, axis=-1)[:, :, None])


def spatial_total_ion_count(img):
    t0 = time.time()
    processed = total_ion_count(img)
    processed = processed / \
        (1e-6 + np.percentile(processed, 99, axis=(0, 1))[None, None, :])
    processed[processed > 1] = 1
    print("norm took", time.time() - t0)
    return img
