import scipy
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.stats import entropy
import random
import time
import sys
import numpy as np
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def smoothness_saliency_metrics(cosine, maxabs, outputs):
    max_rank = np.maximum(
        cosine.argsort().argsort(),
        maxabs.argsort().argsort())
    cosine_rank = cosine.argsort().argsort()
    result_rank = outputs.argsort().argsort()
    result = {}

    corr_cosine = scipy.stats.pearsonr(cosine, outputs).statistic
    corr_maxabs = scipy.stats.pearsonr(maxabs, outputs).statistic

    N = len(outputs)
    saliency_30 = (((result_rank - max_rank + N * 0.3) > 0)
                   * max_rank**2).sum() / sum(max_rank**2)
    saliency_20 = (((result_rank - max_rank + N * 0.2) > 0)
                   * max_rank**2).sum() / sum(max_rank**2)
    saliency_10 = (((result_rank - max_rank + N * 0.1) > 0)
                   * max_rank**2).sum() / sum(max_rank**2)
    smoothness_30 = (((cosine_rank - result_rank - N * 0.3) < 0)
                     * (N - cosine_rank)**2).sum() / sum((N - cosine_rank)**2)
    smoothness_20 = (((cosine_rank - result_rank - N * 0.2) < 0)
                     * (N - cosine_rank)**2).sum() / sum((N - cosine_rank)**2)
    smoothness_10 = (((cosine_rank - result_rank - N * 0.1) < 0)
                     * (N - cosine_rank)**2).sum() / sum((N - cosine_rank)**2)

    result["Saliency Δ=20%"] = saliency_20
    result["Smoothness Δ=20%"] = smoothness_20
    result["Saliency Δ=30%"] = saliency_30
    result["Smoothness Δ=30%"] = smoothness_30
    result["Saliency Δ=10%"] = saliency_10
    result["Smoothness Δ=10%"] = smoothness_10
    result["Correlation Cosine"] = corr_cosine
    result["Correlation L-∞"] = corr_maxabs

    return result


class RandomPairSampler:
    def __init__(self, img, num_samples):
        self.pairs = self.generate_pairs(img, num_samples)

    def generate_pairs(self, img, num_samples):
        nrow, ncol = img.shape[0], img.shape[1]
        points = np.mgrid[:nrow, :ncol].reshape(2, -1).T
        indices = list(points)
        indices = [(a, b) for a, b in indices if img[a, b, :].max() > 0]
        pairs = []
        for _ in range(min(len(indices), num_samples)):
            point_a = random.choice(indices)
            point_b = random.choice(indices)
            pairs.append((point_a, point_b))
        return pairs

    def get_input_distances(self, img):
        data = {}
        distances = ["L-2", "Cosine", "L-∞", "Output"]
        for distance in distances:
            data[distance] = []

        for index, (point_a, point_b) in enumerate(self.pairs):
            a, b = img[point_a[0], point_a[1]], img[point_b[0], point_b[1]]
            data["L-2"].append(np.linalg.norm(a - b, 2))
            data['Cosine'].append(
                1 -
                cosine_similarity(
                    np.array(
                        [a]),
                    np.array(
                        [b]))[
                    0,
                    0])
            data['L-∞'].append(np.abs(a - b).max())

        for key in data:
            data[key] = np.float32(data[key])

        return data

    def get_output_distance(self, visualization):
        result = []
        for point_a, point_b in self.pairs:
            a, b = visualization[point_a[0], point_a[1]
                                 ], visualization[point_b[0], point_b[1]]
            a = np.float32(a)
            b = np.float32(b)

            d = np.linalg.norm(a - b, 2)
            # d = (int(d) // 10) * 10

            result.append(d)
        return np.float32(result)


class MSIVisualizationMetrics:
    def __init__(self, normalized, visualization, num_samples=3000):

        sampler = RandomPairSampler(normalized, num_samples)

        self.data = self.generate_input_output_samples(
            normalized, cv2.cvtColor(
                visualization, cv2.COLOR_RGB2LAB), sampler)

    def generate_input_output_samples(
            self, normalized, visualization, sampler):
        data = sampler.get_input_distances(normalized)
        data["Output"] = sampler.get_output_distance(visualization)
        return data

    def get_metrics(self):
        cosine = self.data["Cosine"]
        maxabs = self.data["L-∞"]
        outputs = self.data["Output"]
        metrics = smoothness_saliency_metrics(cosine, maxabs, outputs)
        return metrics

    def get_correlation_scatter_plot(self, title=None):
        cosine = self.data["Cosine"]
        maxabs = self.data["L-∞"]
        outputs = self.data["Output"]
        metrics = self.get_metrics()
        fig = plt.figure()
        ax = fig.add_subplot()
        max_rank = np.maximum(
            cosine.argsort().argsort(),
            maxabs.argsort().argsort())

        # corr = scipy.stats.pearsonr(max_rank, outputs).statistic

        expected_rank = max_rank
        result_rank = outputs.argsort().argsort()
        diff_rank = np.abs(result_rank - expected_rank)
        cm = matplotlib.pyplot.get_cmap("RdYlGn_r")
        colors = cm(np.linspace(0, 1, len(outputs)))
        N = len(diff_rank) // 2
        cs = [colors[diff_rank[i]]
              for i in range(len(outputs))]  # could be done with numpy's repmat
        matplotlib.pyplot.scatter(cosine, maxabs, color=cs, s=10, alpha=0.5)
        sm = plt.cm.ScalarMappable(cmap=cm)
        sm.set_clim(vmin=0, vmax=1)
        plt.colorbar(sm, ax=plt.gca())
        ax.set_xlabel('m/z values Cosine distance')
        ax.set_ylabel('m/z values L-∞ distance')
        if title:
            ax.set_title(title)

        for name, value in metrics.items():
            plt.plot([], [], ' ', label=f"{name} {value:.3f}")
        plt.legend()
        return fig

    def get_correlation_plot(self, title=None):
        cosine = self.data["Cosine"]
        maxabs = self.data["L-∞"]
        outputs = self.data["Output"]

        fig, ax = plt.subplots()

        for inputs, name in zip([cosine, maxabs], ["Cosine", "L-∞"]):
            sorted_distances = sorted(inputs)
            indices = list(
                range(
                    len(sorted_distances) // 10,
                    len(sorted_distances),
                    len(sorted_distances) // 10))
            indices.append(len(sorted_distances) - 1)
            graph_distances, graph_corrs = [], []
            for index in indices:
                d = sorted_distances[index]
                relevant_indices = [
                    i for i in range(
                        len(inputs)) if inputs[i] < d]
                corr = scipy.stats.pearsonr(
                    inputs[relevant_indices],
                    outputs[relevant_indices]).statistic
                graph_distances.append(
                    int(100 * index / len(sorted_distances)))
                graph_corrs.append(corr)

            plt.plot(graph_distances, graph_corrs, label=name)
        plt.legend()
        if title:
            plt.title(
                'Visualization distance correlation with spectra\n' +
                title)
        else:
            plt.title('Visualization distance correlation with spectra')

        plt.xlabel('Maximum input distance percentile')
        plt.ylabel('Pearson rank correlation')
        ax.set_yticks(list(np.arange(0, np.max(corr) + 0.05, 0.05)))
        return fig


if __name__ == "__main__":
    img = np.load(sys.argv[1])
    visualization = np.array(Image.open(sys.argv[2]))
    normalized = total_ion_count(img)
    random.seed(0)
    metrics = MSIVisualizationMetrics(normalized, visualization)
    print(metrics.get_metrics())
    plt.show()
