from msi_visual.saliency_opt import SaliencyOptimization
from msi_visual.nmf_3d import NMF3D

from msi_visual.nonparametric_umap  import MSINonParametricUMAP 
from msi_visual.percentile_ratio import top3, percentile_ratio_rgb
from msi_visual.metrics import MSIVisualizationMetrics
from msi_visual.normalization import total_ion_count
from PIL import Image
import tqdm
import os
import time
import argparse
import numpy as np
import glob
from pathlib import Path
import random
import time
from collections import defaultdict
import pandas as pd
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    paths = glob.glob(str(Path(args.dir) / "*.npy"))
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


    methods = [("SALIENCY-OPTIMIZATION_coreset_100", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=100, init="random")),
        ("SALIENCY-OPTIMIZATION_random_100", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=100, init="random")),
        ("SALIENCY-OPTIMIZATION_coreset_200", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=200, init="random")),
        ("SALIENCY-OPTIMIZATION_random_200", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=200, init="random")),
        ("SALIENCY-OPTIMIZATION_coreset_400", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=400, init="random")),
        ("SALIENCY-OPTIMIZATION_random_400", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=400, init="random")),
        ("SALIENCY-OPTIMIZATION_coreset_600", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=600, init="random")),
        ("SALIENCY-OPTIMIZATION_random_600", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=600, init="random")),
        ("SALIENCY-OPTIMIZATION_coreset_800", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=800, init="random")),
        ("SALIENCY-OPTIMIZATION_random_800", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=800, init="random")),
        ("SALIENCY-OPTIMIZATION_coreset_1000", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=1000, init="random")),
        ("SALIENCY-OPTIMIZATION_random_1000", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=1000, init="random"))]
        # ("SALIENCY-OPTIMIZATION_coreset_2000", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=2000, init="random")),
        # ("SALIENCY-OPTIMIZATION_random_2000", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=2000, init="random"))]
        # ("SALIENCY-OPTIMIZATION_coreset_4000", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=4000, init="random")),
        # ("SALIENCY-OPTIMIZATION_random_4000", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="random", number_of_points=4000, init="random"))]




        # ("SALIENCY-OPTIMIZATION1", SaliencyOptimization(num_epochs=200, regularization_strength=0.001, sampling="coreset", number_of_points=1000, init="random")),
        # ("SALIENCY-OPTIMIZATION2", SaliencyOptimization(num_epochs=500, regularization_strength=0.001, sampling="coreset", number_of_points=1000, init="random")),
        # ("TOP3", top3), ("Percentile Ratio", percentile_ratio_rgb),
        # ("NMF", NMF3D()),
        # ("UMAP min_dist=0.1, n=15, metric=L2", MSINonParametricUMAP (min_dist=0.1, n_neighbors=15, metric='euclidean')),
        # ("UMAP min_dist=0.1, n=50, metric=L2", MSINonParametricUMAP (min_dist=0.1, n_neighbors=50, metric='euclidean')),
        # ("UMAP min_dist=0.5, n=15, metric=L2", MSINonParametricUMAP (min_dist=0.5, n_neighbors=15, metric='euclidean')),
        # ("UMAP min_dist=0.5, n=50, metric=L2", MSINonParametricUMAP (min_dist=0.5, n_neighbors=50, metric='euclidean')),
        # ("UMAP min_dist=0.9, n=100, metric=L2", MSINonParametricUMAP (min_dist=0.9, n_neighbors=100, metric='euclidean')),
        # ("UMAP min_dist=0.9, n=300, metric=L2", MSINonParametricUMAP (min_dist=0.9, n_neighbors=300, metric='euclidean')),
        # ("UMAP min_dist=0.5, n=300, metric=L2", MSINonParametricUMAP (min_dist=0.5, n_neighbors=300, metric='euclidean')),

        # ("UMAP min_dist=0.1, n=15, metric=L∞", MSINonParametricUMAP (min_dist=0.1, n_neighbors=15, metric='chebyshev')),
        # ("UMAP min_dist=0.1, n=50, metric=L∞", MSINonParametricUMAP (min_dist=0.1, n_neighbors=50, metric='chebyshev')),
        # ("UMAP min_dist=0.5, n=15, metric=L∞", MSINonParametricUMAP (min_dist=0.5, n_neighbors=15, metric='chebyshev')),
        # ("UMAP min_dist=0.5, n=50, metric=L∞", MSINonParametricUMAP (min_dist=0.5, n_neighbors=50, metric='chebyshev')),
        # ("UMAP min_dist=0.9, n=100, metric=L∞", MSINonParametricUMAP (min_dist=0.9, n_neighbors=100, metric='chebyshev')),
        # ("UMAP min_dist=0.9, n=300, metric=L∞", MSINonParametricUMAP (min_dist=0.9, n_neighbors=300, metric='chebyshev')),
        # ("UMAP min_dist=0.5, n=300, metric=L∞", MSINonParametricUMAP (min_dist=0.5, n_neighbors=300, metric='chebyshev'))]

    result = defaultdict(list)
    for index, path in enumerate(paths):
        img = np.load(path)
        t0 = time.time()
        img = total_ion_count(img)
        visualizations = {}
        metrics = {}
        for name, method in tqdm.tqdm(methods):

            exists = False            
            dst_path = str(Path(args.dst) / f"{index}_{name}.png")
            if os.path.exists(dst_path):
                exists = True
                visualizations[name] = np.array(Image.open(dst_path))
                print(f'Existing {dst_path}')
            else:
                t0 = time.time()
                visualizations[name] = method(img)
                t = time.time() - t0
            random.seed(0)
            metrics[name] = MSIVisualizationMetrics(
                img, visualizations[name]).get_metrics()
            print(index, name, metrics[name])


            result["method"].append(name)
            result["path"].append(path)
            #if not exists:
            #    result["time"].append(t)
            for m in metrics[name]:
                result[m].append(metrics[name][m])


            if not exists:
                Image.fromarray(visualizations[name]).save(dst_path)

        result_csv = pd.DataFrame.from_dict(result)
        result_csv.to_csv(str(Path(args.dst) / "benchmark_new.csv"), index=False)
