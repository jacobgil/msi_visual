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
from msi_visual.lda_3d import LDA3D
from msi_visual.nmf_segmentation import NMFSegmentation
from msi_visual.kmeans_segmentation import KmeansSegmentation


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
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


    methods = [("FastICA3D", FastICA3D()),
               ("Spectral3D", Spectral3D()), \
               ("PCA3D", PCA3D()), 
               ("SaliencyOptimization", SaliencyOptimization(num_epochs=500, regularization_strength=0.001, sampling="coreset", number_of_points=1000, init="random")), 
               ("SpearmanOptimization", SpearmanOptimization(num_epochs=500, regularization_strength=0.001, sampling="coreset", number_of_points=1000, init="random")),
               ("NMF3D", NMF3D()), ("PACMAC3D", PACMAC3D()),
               ("TSNE3D", TSNE3D()), 
               ("PHATE3D", PHATE3D()),
               ("Isomap3D", Isomap3D()),
               ("Trimap3D", Trimap3D()), 
               ("UMAP1", MSINonParametricUMAP()),
               ("UMAP2", MSINonParametricUMAP(metric='chebyshev')),
               ("TOP3", TOP3()),
               ("PercentileRatio", PercentileRatio())]

    result = defaultdict(list)
    for index, path in enumerate(paths):
        img = np.load(path)[::4, ::4, :]
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
                if isinstance(visualizations[name], list):
                    visualizations[name] = visualizations[name][0]

                t = time.time() - t0
            random.seed(0)



            metrics[name] = MSIVisualizationMetrics(
                img, visualizations[name], num_samples=6000).get_metrics()
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
