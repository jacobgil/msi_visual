import streamlit as st

from msi_visual.saliency_opt import SaliencyOptimization
from msi_visual.nmf_3d import NMF3D
from msi_visual.nonparametric_umap  import MSINonParametricUMAP
from msi_visual.kmeans_segmentation import KmeansSegmentation
from msi_visual.percentile_ratio import TOP3, PercentileRatio
from msi_visual.nmf_segmentation import NMFSegmentation
from msi_visual.normalization import total_ion_count, spatial_total_ion_count

def create_pipeline():
    options = ["NMF3D", "Segmentation-NMF", "Segmentation-Kmeans", "Saliency Optimization", "UMAP-3D", "TOP-3", "Percentile-Ratio", "Existing model"]
    method = st.selectbox("Add visualization method", options, index=None)

    model = None
    if method == "Segmentation-Kmeans":
        k = st.number_input("Number of components", value=8, step=1)
        if k > 0:
            add = st.button("Add")
            if add:
                model = KmeansSegmentation(k)

    elif method == "Segmentation-NMF":
        k = st.number_input("Number of components", value=8, step=1)
        max_iter = st.number_input('Number of iterations', value=2000, step=1)
        if k > 0:
            add = st.button("Add")
            if add:
                model = NMFSegmentation(k, max_iter=max_iter)

    elif method == "Percentile-Ratio":
        if st.button("Add"):
            model = PercentileRatio()
    
    elif method == "NMF3D":
        max_iter = st.number_input('Number of iterations', value=200, step=1)
        if st.button("Add"):
            model = NMF3D(max_iter=max_iter)
    
    elif method == "TOP-3":
        if st.button("Add"):
            model = TOP3()

    elif method == "Existing model":
        path = st.text_input()
        if path:
            if st.button("Add"):
                model = joblib.load(path)

    elif method == "Saliency Optimization":
        number_of_points = st.number_input("Number of reference points", value=1000, min_value=50, step=1)
        sampling = st.selectbox("Reference point sampling method", ["coreset", "random", "kmeans++", "kmeans"])
        regularization_strength=st.number_input("Rank loss Regularization", value=0.001, min_value=0.0, step=1e-4, format="%.5f")
        epochs = st.number_input("Number of epochs", value=200, min_value=1, step=1)
        
        if st.button("Add"):
            model = SaliencyOptimization(number_of_points=number_of_points, regularization_strength=regularization_strength,
                sampling=sampling, num_epochs=epochs)
    elif method == "UMAP-3D":
        min_dist = st.number_input("min_dist", value=0.5, min_value=0.0)
        distance = st.selectbox('Distance', ['euclidean', 'chebyshev'])
        n_neighbors = st.number_input('n_neighbors', value=100, min_value=5)

        if st.button("Add"):
            model = MSINonParametricUMAP(min_dist=min_dist, n_neighbors=n_neighbors, metric=distance)

    normalization_method = st.selectbox('Normalization', ['total_ion_count', 'spatial_total_ion_count'])
    normalization = {'total_ion_count': total_ion_count, 'spatial_total_ion_count': spatial_total_ion_count}[normalization_method]
    output_path = st.text_input("Output Folder", 'visualizations')


    return model, normalization, output_path
        