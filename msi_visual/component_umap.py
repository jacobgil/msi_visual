import numpy as np
import matplotlib.pyplot as plt
import UMAP 
from PIL import Image


def get_UMAP (results, colors_for_components):
    fig = plt.figure()
    all_embeddings = []
    all_labels = []
    all_colors = []
    all_markers = []
    marker_list = ['x', 'o', "D", "<", "*", "p", "v", 'h', '+', '|']
    colors = colors_for_components
    regions = [result["name"] for result in results]
    for region_index, result in enumerate(results):
        name = result["name"]
        W, H = result["components"]
        order = result["order"]
        for i in range(len(H)):
            all_embeddings.append(H[i, :])
            all_labels.append(f"Region: {name} Component: {i}")
            all_colors.append(colors[order[i]])
            marker = marker_list[region_index]
            all_markers.append(marker)
    all_embeddings = np.float32(all_embeddings)
    normalized = all_embeddings
    # normalized = all_embeddings - np.mean(all_embeddings, axis = 0)
    # normalized = normalized / (1e-5 + np.std(normalized, axis = 0))
    # all_embeddings = normalized
    plt.gca().set_aspect("equal", "datalim")
    UMAP _embeddings = UMAP .UMAP (
        n_neighbors=5, min_dist=0.95, verbose=True
    ).fit_transform(normalized)
    count = len(all_labels)
    for index, marker in enumerate(set(all_markers)):
        indices = [
            i for i in range(
                len(all_markers)) if all_markers[i] == marker]
        sc = plt.scatter(*UMAP _embeddings[indices, :].T,
                         s=30,
                         c=[all_colors[i] for i in indices],
                         marker=marker,
                         alpha=1.0,
                         label=f"Mouse model {chr(ord('A')+index)}")
        plt.legend(loc='lower left')

    ax = plt.gca()
    leg = ax.get_legend()
    for handle in leg.legendHandles:
        handle.set_color('black')

    plt.title(
        f"UMAP  m/z vectors for different animals and their segmented components")
    plt.xlabel("First UMAP  embedding dimension")
    plt.ylabel("Second UMAP  embedding dimension")
    # plt.axis('off')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close(fig=fig)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(data)
