from msi_visual.saliency_opt import SaliencyOptimization
from msi_visual.nmf_3d import NMF3D
from msi_visual.nonparametric_umap import MSINonParametricUMAP
from msi_visual.percentile_ratio import top3, percentile_ratio_rgb
from msi_visual.metrics import MSIVisualizationMetrics
from msi_visual.normalization import total_ion_count
from PIL import Image
import tqdm
import time
import argparse
import numpy as np
import glob
from pathlib import Path
import random
import time
from collections import defaultdict
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    paths = glob.glob(str(Path(args.dir) / "*.npy"))

    methods = [
        ("SALIENCYOPT-1", SaliencyOptimization(num_epochs=200, regularization_strength=0.005, sampling="coreset", number_of_points=200)),
        ("SALIENCYOPT-2", SaliencyOptimization(num_epochs=200, regularization_strength=0.005, sampling="coreset", number_of_points=1000)),
        ("SALIENCYOPT-3", SaliencyOptimization(num_epochs=200, regularization_strength=0.05, sampling="coreset", number_of_points=200)),
        ("SALIENCYOPT-4", SaliencyOptimization(num_epochs=200, regularization_strength=0.05, sampling="coreset", number_of_points=1000)),
        ("SALIENCYOPT-5", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=200)),
        ("SALIENCYOPT-6", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=1000)),
        ("SALIENCYOPT-7", SaliencyOptimization(num_epochs=200, regularization_strength=0.005, sampling="random", number_of_points=200)),
        ("SALIENCYOPT-8", SaliencyOptimization(num_epochs=200, regularization_strength=0.005, sampling="random", number_of_points=1000)),
        ("SALIENCYOPT-9", SaliencyOptimization(num_epochs=200, regularization_strength=0.05, sampling="random", number_of_points=200)),
        ("SALIENCYOPT-10", SaliencyOptimization(num_epochs=200, regularization_strength=0.05, sampling="random", number_of_points=1000)),
        ("SALIENCYOPT-11", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=200)),
        ("SALIENCYOPT-12", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=1000)),
        ("TOP3", top3), ("PR", percentile_ratio_rgb),
        ("NMF", NMF3D()),
        ("UMAP1", MSINonParametricUMAP(min_dist=0.1, n_neighbors=15, metric='euclidean')),
        ("UMAP2", MSINonParametricUMAP(min_dist=0.1, n_neighbors=50, metric='euclidean')),
        ("UMAP3", MSINonParametricUMAP(min_dist=0.5, n_neighbors=15, metric='euclidean')),
        ("UMAP4", MSINonParametricUMAP(min_dist=0.5, n_neighbors=50, metric='euclidean')),
        ("UMAP5", MSINonParametricUMAP(min_dist=0.9, n_neighbors=100, metric='euclidean')),
        ("UMAP6", MSINonParametricUMAP(min_dist=0.9, n_neighbors=100, metric='euclidean')),
        ("UMAP7", MSINonParametricUMAP(min_dist=0.9, n_neighbors=100, metric='euclidean')),
        ("UMAP8", MSINonParametricUMAP(min_dist=0.1, n_neighbors=15, metric='chebyshev')),
        ("UMAP9", MSINonParametricUMAP(min_dist=0.1, n_neighbors=50, metric='chebyshev')),
        ("UMAP10", MSINonParametricUMAP(min_dist=0.5, n_neighbors=15, metric='chebyshev')),
        ("UMAP11", MSINonParametricUMAP(min_dist=0.5, n_neighbors=50, metric='chebyshev')),
        ("UMAP12", MSINonParametricUMAP(min_dist=0.9, n_neighbors=100, metric='chebyshev')),
        ("UMAP13", MSINonParametricUMAP(min_dist=0.9, n_neighbors=100, metric='chebyshev')),
        ("UMAP14", MSINonParametricUMAP(min_dist=0.9, n_neighbors=100, metric='chebyshev'))
    ]

    result = defaultdict(list)
    for index, path in enumerate(paths):
        img = np.load(path)
        t0 = time.time()
        img = total_ion_count(img)
        visualizations = {}
        metrics = {}
        times = {}
        for name, method in tqdm.tqdm(methods):
            t0 = time.time()
            visualizations[name] = method(img)
            t = time.time() - t0
            random.seed(0)
            metrics[name] = MSIVisualizationMetrics(
                img, visualizations[name]).get_metrics()
            times[name] = t

            result["method"].append(name)
            result["path"].append(path)
            result["time"].append(t)
            for m in metrics[name]:
                result[m].append(metrics[name][m])

            Image.fromarray(visualizations[name]).save(
                str(Path(args.dst) / f"{index}_{name}.png"))

            print(index, name, metrics[name])

    result = pd.DataFrame.from_dict(result)
    print(result)
    result.to_csv(str(Path(args.dst) / "benchmark.csv"), index=False)
