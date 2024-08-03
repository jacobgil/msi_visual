from sklearn.decomposition import NMF, non_negative_factorization
import matplotlib
from matplotlib import pyplot as plt
import cv2
import numpy as np
from typing import List, Dict, Optional
import math
import glob
from PIL import Image
from matplotlib import pyplot as plt
import math
import tqdm
import warnings
import pathlib
import sys
import os
import joblib
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from component_UMAP  import get_UMAP 
from visualizations import get_colors, show_factorization_on_image
warnings.filterwarnings('ignore')

def get_unsupervised_decomposition(path: str, 
                                   NUM_COMPONENTS: int,
                                   colors: list,
                                   start_mz: int = 300,
                                   consistent_coloring: bool = True,
                                   mask: Optional[np.ndarray] = None,
                                   do_subsegmentation: bool = False):
    name = pathlib.Path(path).stem
    img = np.load(path)
    img = np.float32(img)
    img = img[:, :, 600 : ]

    if mask is not None:
        img[mask == 0] = 0

    vector = img.reshape((-1, img.shape[-1]))
    vector = vector / (1e-6 + np.median(vector, axis=-1)[:, None])

    if do_subsegmentation:
        model = NMF(n_components=NUM_COMPONENTS, init='random', random_state=0)
        W = model.fit_transform(vector)
        H = model.components_
    else:
        H = np.load("h_cosegmentation.npy")
        W, H, n_iter = non_negative_factorization(vector, H=H, W=None, n_components=NUM_COMPONENTS, update_H=False, random_state=0)

    order, diffs = None, None
    if consistent_coloring:
        centers = np.load("h_cosegmentation.npy")
        order = []
        diffs = []
        for i in range(len(H)):
            h = H[i, :]
            h = h[None, :]

            sim = cosine_similarity(h, centers)[0, :]
            diffs.append(sim.max())
            order.append(sim.argmax())
        colors_ordered = [colors[i] for i in order]
    else:
        colors_ordered = colors
    
    explanations = W.transpose().reshape(NUM_COMPONENTS, img.shape[0], img.shape[1])
    explanations = explanations / explanations.sum(axis=(1, 2))[:, None, None]
    explanations = explanations / np.percentile(explanations, 99, axis=(1, 2))[:, None, None]
    # Remove the noise component
    if consistent_coloring:
        explanations[order.index(4), :] = 0
    
    masks_per_component = []
    if do_subsegmentation:
        for i in range(NUM_COMPONENTS):
            mask = np.uint8(explanations.argmax(axis=0) == i)
            mask = cv2.medianBlur(mask, 3)
            dilatation_size = 2
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                            (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                            (dilatation_size, dilatation_size))
            mask = cv2.erode(mask, element)
            masks_per_component.append(mask)

    img = img / img.max(axis=(0, 1))[None, None, :]
    visualization = show_factorization_on_image(np.zeros(shape=((img.shape[0], img.shape[1], 3))),
                                                explanations,
                                                image_weight=0.0,
                                                colors=colors)
    visualization2 = show_factorization_on_image(np.zeros(shape=((img.shape[0], img.shape[1], 3))),
                                                explanations,
                                                image_weight=0.0,
                                                colors=colors_ordered)
    visualization = \
        cv2.putText(visualization,f"{chr(int(name)+ord('A'))}", 
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

    visualization2 = \
        cv2.putText(visualization2,f"{chr(int(name) + ord('A'))}", 
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

    #visualization = np.hstack((visualization, visualization2))
    visualization = visualization2

    images = []
    for i in range(NUM_COMPONENTS):
        fig = plt.figure()
        low = np.min(H[i, :])
        high = np.max(H[i, :])
        plt.ylim([0, math.ceil(high+2)])
        plt.tight_layout()
        plt.bar(range(900, 1301), H[i, :], color = colors[i])
        plt.title(f'Region: {chr(int(name)+ord("A") )} Component: {i}', y=0.9)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig=fig)
        images.append(data)
    images = np.vstack(images)
    diff_table = None
    if consistent_coloring:        
        table = np.zeros((NUM_COMPONENTS, 3))
        table[:, 0] = [f"{o:2d}" for o in list(range(NUM_COMPONENTS))]
        table[:, 1] = [f"{o:2d}" for o in np.int32(order)]
        table[:, 2] = [f"{d:.3f}" for d in diffs]
        fig = plt.figure()
        plt.table(table, colLabels=["Component", "Reference Comp.", "Similarity"],
                loc='upper left')
        plt.title(f'Similarity to reference components\n{chr(int(name)+ord("A") )}')
        fig.tight_layout()
        fig.patch.set_visible(False)
        plt.axis('off')
        plt.axis('tight')    
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig=fig)
        diff_table = Image.fromarray(data)

    result = {"visualization": Image.fromarray(visualization),
              "mz_plots": Image.fromarray(images), 
              "components": (W, H),
              "path": path,
              "name": name,
              "order": order,
              "diffs": diffs,
              "diff_table": diff_table,
              "masks_per_component": masks_per_component}
    return result

if __name__ == '__main__':
    paths = glob.glob(os.path.join(sys.argv[1], "*.npy"))
    NUM_COMPONENTS = int(sys.argv[3])
    do_subsegmentation = int(sys.argv[4])
    colors = get_colors(NUM_COMPONENTS)
    colors = colors + colors

    results = []
    sub_segmentations = []
    for path in tqdm.tqdm(paths):
        result = get_unsupervised_decomposition(path,
                                                NUM_COMPONENTS, 
                                                colors=colors,
                                                do_subsegmentation=do_subsegmentation)
        masks = result["masks_per_component"]
        for mask in masks:
            sub_seg = get_unsupervised_decomposition(path,
                                                    NUM_COMPONENTS, 
                                                    colors=colors,
                                                    consistent_coloring=False,
                                                    mask=mask)
            sub_segmentations.append(sub_seg["visualization"])


        results.append(result)

    images_for_pdf = []
    visualizations = [result["visualization"] for result in results]
    mz_plots = [result["mz_plots"] for result in results]
    diff_tables = [result["diff_table"] for result in results]

    UMAP _plot = get_UMAP (results, colors)

    images_for_pdf.extend(visualizations)
    images_for_pdf.extend(sub_segmentations)
    images_for_pdf.append(UMAP _plot)
    images_for_pdf.extend(mz_plots)
    images_for_pdf.extend(diff_tables)

    output_path = sys.argv[2]

    images_for_pdf[0].save(os.path.join(output_path, "report.pdf"),
                        save_all=True,
                        append_images=images_for_pdf[1:])

    for result in results:
        name = result["name"]
        w, h = result["components"]

        np.save(os.path.join(output_path, name + "_w.npy"), w)
        np.save(os.path.join(output_path, name + "_h.npy"), h)