import numpy as np
import scipy


class ObjectDetector:
    def __init__(self, size=3, peak_multiply=1000):
        self.size = size
        self.peak_multiply = peak_multiply

        print(self.size, self.peak_multiply)

    def get_mask(self, img):
        img = np.float32(img)
        normalized = img / (1e-6 + np.sum(img, axis=-1)[:, :, None])
        normalized = normalized / \
            (1e-6 + np.percentile(normalized, 99, axis=(0, 1))[None, None, :])
        normalized[normalized > 1] = 1
        normalized = np.float32(normalized)
        print("normalized", normalized.shape)
        local_max = scipy.ndimage.maximum_filter(normalized, size=self.size)
        local_median = scipy.ndimage.median_filter(normalized, size=self.size)

        mask = np.float32(
            (local_max == normalized) & (
                local_max > local_median *
                self.peak_multiply))

        norm = np.percentile(mask, 99, axis=-1)
        norm = norm / np.percentile(norm, 99)
        norm[norm > 1] = 1
        norm = np.uint8(255 * norm)

        result = norm > scipy.ndimage.minimum_filter(norm, size=self.size)
        result = np.uint8(result) * 255
        print(result.shape)

        return result
