from msi_visual.pca_3d import PCA3D
from msi_visual.saliency_opt import SaliencyOptimization
from msi_visual.spearman_opt import SpearmanOptimization
from msi_visual.nmf_3d import NMF3D
from msi_visual.pca_3d import PCA3D
from msi_visual.nonparametric_umap  import MSINonParametricUMAP 
from msi_visual.percentile_ratio import TOP3, PercentileRatio
from msi_visual.pacmac_3d import PACMAC3D
# from msi_visual.lle_3d import LLE3D
from msi_visual.tsne_3d import TSNE3D
#from msi_visual.mds_3d import MDS3D
from msi_visual.phate3d import PHATE3D
from msi_visual.isomap_3d import Isomap3D
from msi_visual.trimap_3d import Trimap3D
from msi_visual.spectral_3d import Spectral3D
from msi_visual.fastica_3d import FastICA3D
from msi_visual.nmf_segmentation import NMFSegmentation
from msi_visual.kmeans_segmentation import KmeansSegmentation
from msi_visual.saliency_clustering_opt import SaliencyClusteringOptimization

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
import cv2
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
    print(paths)
    print(len(paths))
    paths += glob.glob(str(Path(args.dir) / "*" / "*.npy"))
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    methods = [
        "FastICA3D",
        "Spectral3D",
        "PCA3D",
        "SaliencyOptimization",
        "SpearmanOptimization",
        "NMF3D",
        "PACMAC3D",
        "TSNE3D",
        "PHATE3D",
        "Isomap3D",
        "Trimap3D",
        "UMAP1",
        "UMAP2",
        "TOP3",
        "PercentileRatio"
    ]

    methods = ["PCA3D"]

    methods =["SaliencyClusteringOptimization1", "SaliencyClusteringOptimization2", "SaliencyClusteringOptimization3"]


    result = defaultdict(list)
    for index, path in enumerate(paths):
        img = np.load(path)
        t0 = time.time()
        img = total_ion_count(img)
        visualizations = {}
        metrics = {}
        for name in tqdm.tqdm(methods):
            if name == "FastICA3D":
                method = FastICA3D()
            elif name == "Spectral3D":
                method = Spectral3D()
            elif name == "PCA3D":
                method = PCA3D()
            elif name == "SaliencyOptimization":
                method = SaliencyOptimization(num_epochs=500, regularization_strength=0.001, sampling="coreset", number_of_points=1000, init="random")
            elif name == "SpearmanOptimization":
                method = SpearmanOptimization(num_epochs=500, regularization_strength=0.001, sampling="coreset", number_of_points=1000, init="random")
            elif name == "NMF3D":
                method = NMF3D()
            elif name == "PACMAC3D":
                method = PACMAC3D()
            elif name == "TSNE3D":
                method = TSNE3D()
            elif name == "PHATE3D":
                method = PHATE3D()
            elif name == "Isomap3D":
                method = Isomap3D()
            elif name == "Trimap3D":
                method = Trimap3D()
            elif name == "UMAP1":
                method = MSINonParametricUMAP()
            elif name == "UMAP2":
                method = MSINonParametricUMAP(metric='chebyshev')
            elif name == "TOP3":
                method = TOP3()
            elif name == "PercentileRatio":
                method = PercentileRatio()
            elif name == "SaliencyClusteringOptimization1":
                method = SaliencyClusteringOptimization(num_epochs=500, regularization_strength=0.001, cluster_fraction=0.1,
                                                        number_of_points=1000, clusters=[8], sampling="coreset", lab_to_rgb=True)
            elif name == "SaliencyClusteringOptimization2":
                method = SaliencyClusteringOptimization(num_epochs=500, regularization_strength=0.001, cluster_fraction=0.1,
                                                        number_of_points=1000, clusters=[16], sampling="coreset", lab_to_rgb=True)
            elif name == "SaliencyClusteringOptimization3":
                method = SaliencyClusteringOptimization(num_epochs=500, regularization_strength=0.001, cluster_fraction=0.1,
                                                        number_of_points=1000, clusters=[8, 16, 32, 64], sampling="coreset", lab_to_rgb=True)

            exists = False            
            dst_path = str(Path(args.dst) / f"{index}_{name}.png")
            if os.path.exists(dst_path):
                exists = True
                visualizations[name] = np.array(Image.open(dst_path))
                print(f'Existing {dst_path}')
            else:
                t0 = time.time()
                visualizations[name] = method(img)
                if isinstance(visualizations[name], list):
                    visualizations[name] = visualizations[name][0]

                t = time.time() - t0
            random.seed(0)



            metrics[name] = MSIVisualizationMetrics(
                img, visualizations[name], num_samples=8000).get_metrics()
            print(index, name, metrics[name])
            result["method"].append(name)
            result["path"].append(path)
            if not exists:
               result["time"].append(t)
            for m in metrics[name]:
                result[m].append(metrics[name][m])

            # # with equalization
            # visualizations[name + "eq"] = cv2.merge([cv2.equalizeHist(visualizations[name][:, :, i]) for i in range(3)])
            # metrics[name + "eq"] = MSIVisualizationMetrics(
            #     img, visualizations[name + "eq"]).get_metrics()
            # print(index, name + "eq", metrics[name + "eq"])
            # result["method"].append(name + "eq")
            # result["path"].append(path)
            # if not exists:
            #    result["time"].append(t)
            # for m in metrics[name + "eq"]:
            #     result[m].append(metrics[name + "eq"][m])


            if not exists:
                Image.fromarray(visualizations[name]).save(dst_path)
                eq = cv2.merge([cv2.equalizeHist(visualizations[name][:, :, i]) for i in range(3)])
                dst_path_eq = str(Path(args.dst) / f"{index}_{name}_eq.png")
                Image.fromarray(eq).save(dst_path_eq)

        result_csv = pd.DataFrame.from_dict(result)
        result_csv.to_csv(str(Path(args.dst) / "benchmark_new.csv"), index=False)
