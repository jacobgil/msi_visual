import numpy as np


class AutoColorizeRandom:
    def __init__(self, color_schemes):
        self.color_schemes = color_schemes

    def colorize(self, ratio, k):
        num_low = int(ratio * k)
        colors_low = np.random.choice(self.color_schemes["low"], num_low)
        colors_high = np.random.choice(self.color_schemes["high"], k - num_low)
        colors = list(colors_low) + list(colors_high)
        return colors


class AutoColorizeArea:
    def __init__(self, color_schemes):
        self.color_schemes = color_schemes

    def colorize(self, img, segmentation, heatmap, ratio, k):
        areas = np.zeros(k)
        for i in range(k):
            mask = segmentation == k
            areas[i] = np.sum(mask)

        num_low = int(ratio * k)
        colors_low = np.random.choice(self.color_schemes["low"], num_low)
        colors_high = np.random.choice(self.color_schemes["high"], k - num_low)
        colors = list(colors_low) + list(colors_high)

        indices = list(np.argsort(areas))
        return [colors[indices.index(i)] for i in range(k)]
