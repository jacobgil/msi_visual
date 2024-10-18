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
from zadu import zadu
from sklearn.manifold import trustworthiness

def smoothness_saliency_metrics(cosine, maxabs, outputs):
    max_rank = np.maximum(
        cosine.argsort().argsort(),
        maxabs.argsort().argsort())

    cosine_rank = cosine.argsort().argsort()
    result_rank = outputs.argsort().argsort()

    result = {}

    corr_cosine = scipy.stats.pearsonr(cosine, outputs).statistic
    corr_maxabs = scipy.stats.pearsonr(maxabs, outputs).statistic

    spearman_cosine = scipy.stats.spearmanr(cosine, outputs).statistic
    spearman_maxabs = scipy.stats.spearmanr(maxabs, outputs).statistic

    N = len(outputs)
    for gamma in [0, 2.0]:

        #max_rank = max_rank + 1e-6
        saliency_30 = (((result_rank - max_rank + N * 0.3) > 0)
                    * max_rank**gamma).sum() / sum(max_rank**gamma)
        saliency_20 = (((result_rank - max_rank + N * 0.2) > 0)
                    * max_rank**gamma).sum() / sum(max_rank**gamma)
        saliency_10 = (((result_rank - max_rank + N * 0.1) > 0)
                    * max_rank**gamma).sum() / sum(max_rank**gamma)
        smoothness_30 = (((cosine_rank - result_rank - N * 0.3) < 0)
                        * (N - cosine_rank)**gamma).sum() / sum((N - cosine_rank)**gamma)
        smoothness_20 = (((cosine_rank - result_rank - N * 0.2) < 0)
                        * (N - cosine_rank)**gamma).sum() / sum((N - cosine_rank)**gamma)
        smoothness_10 = (((cosine_rank - result_rank - N * 0.1) < 0)
                        * (N - cosine_rank)**gamma).sum() / sum((N - cosine_rank)**gamma)        
        saliency_20 = (((result_rank - max_rank + N * 0.2) > 0)
                    * max_rank**gamma).sum() / sum(max_rank**gamma)
        result[f"Saliency Δ=20% gamma={gamma}"] = saliency_20
        result[f"Smoothness Δ=20% gamma={gamma}"] = smoothness_20
        result[f"Saliency Δ=30% gamma={gamma}"] = saliency_30
        result[f"Smoothness Δ=30% gamma={gamma}"] = smoothness_30
        result[f"Saliency Δ=10% gamma={gamma}"] = saliency_10
        result[f"Smoothness Δ=10% gamma={gamma}"] = smoothness_10
        result[f"Avg. Saliency gamma={gamma}"] = (saliency_10 + saliency_20 + saliency_30) / 3
        result[f"Avg. Smoothness gamma={gamma}"] = (smoothness_10 + smoothness_20 + smoothness_30) / 3
    result["Correlation Cosine"] = corr_cosine
    result["Correlation L-∞"] = corr_maxabs
    result["Spearman Cosine"] = spearman_cosine
    result["Spearman L-∞"] = spearman_maxabs



    return result


class RandomPairSampler:
    def __init__(self, img, num_samples, mask):
    
        self.pairs = self.generate_pairs(img, num_samples, mask)

    def generate_pairs(self, img, num_samples, mask):
        nrow, ncol = img.shape[0], img.shape[1]
        points = np.mgrid[:nrow, :ncol].reshape(2, -1).T
        points = list(points)
        img_mask = np.uint8(img.max(axis=-1) > 0) * 255
        if mask is not None:
            img_mask[mask == 0] = 0

        points = [(a, b) for a, b in points if img_mask[a, b] > 0]

        pairs = []        
        for _ in range(min(len(points), num_samples)):
            point_a = random.choice(points)
            point_b = random.choice(points)
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

            result.append(d)
        return np.float32(result)


class MSIVisualizationMetrics:
    def __init__(self, normalized, visualization, mask=None, num_samples=30000):
        self.img_mask = np.uint8(normalized.max(axis=-1) > 0) * 255
        self.img_mask_reshaped = self.img_mask.reshape(self.img_mask.shape[0] * self.img_mask.shape[1])
        indices = [i for i in range(len(self.img_mask_reshaped)) if self.img_mask_reshaped[i] > 0]
        if len(indices) <= num_samples:
            self.random_indices = indices
        else:
            self.random_indices = random.sample(indices, num_samples // 5)

        self.data_subset = normalized.reshape(-1, normalized.shape[-1])
        self.visualization_subset = visualization.reshape((self.data_subset.shape[0], -1))
        self.data_subset = self.data_subset[self.random_indices]
        self.visualization_subset = self.visualization_subset[self.random_indices]

        self.sampler = RandomPairSampler(normalized, num_samples, mask)

        self.data = self.generate_input_output_samples(
            normalized, cv2.cvtColor(
                visualization, cv2.COLOR_RGB2LAB), self.sampler)

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


        spec = [{"id": "mrre", "params": { "k": 100 },}, {"id": "lcmc", "params": { "k": 100 }}]
        t0 = time.time()
        scores = zadu.ZADU(spec, self.data_subset).measure(self.visualization_subset)
        for zadu_metric in scores:
            for m in zadu_metric:
                metrics[m] = zadu_metric[m]
        print("zadu took", time.time() - t0)
        t0 = time.time()
        trustworthiness_score = trustworthiness(self.data_subset, self.visualization_subset, metric='euclidean', n_neighbors=100)
        trustworthiness_score_chevbyshev = trustworthiness(self.data_subset, self.visualization_subset, metric='chebyshev', n_neighbors=100)
        metrics["Trustworthiness"] = trustworthiness_score
        metrics["Trustworthiness L-∞"] = trustworthiness_score_chevbyshev

        print(time.time() - t0, "trustwo")

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
    metrics.get_correlation_scatter_plot()
    plt.show()
