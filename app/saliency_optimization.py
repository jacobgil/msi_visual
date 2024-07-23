import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import time
import joblib
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from PIL import Image
from st_pages import show_pages_from_config, add_page_title
import importlib
from msi_visual.normalization import total_ion_count, spatial_total_ion_count
import msi_visual.percentile_ratio
import msi_visual.saliency_opt
importlib.reload(msi_visual.percentile_ratio)
importlib.reload(msi_visual.saliency_opt)
from msi_visual.saliency_opt import SaliencyOptimization
from msi_visual.metrics import MSIVisualizationMetrics
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder
import contextlib
import cv2

def save_gif(images, path):
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    pil_images = [Image.fromarray(img) for img in images]
    pil_images[0].save(fp=path, format='GIF', append_images=pil_images[1 : ],
            save_all=True, duration=200, loop=0)

def save_data(path=None):
    folder = "saliency_optimization"
    os.makedirs(folder, exist_ok=True)
    if path is None:
        paths = list(st.session_state.saliency_opt.keys())
    else:
        paths = [path]
    for path in paths:
        image_name = path.replace("/", "_").replace("\\", "_").replace(".", "_").replace(":", "_")
        Image.fromarray(st.session_state.saliency_opt[path]).save(Path(folder) /
                                f"{image_name}_so.png")

        save_gif(st.session_state.epoch_images[path], Path(folder) /
                                f"{image_name}_so_gif.gif")

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

show_pages_from_config()

if 'saliency_opt' not in st.session_state:
    st.session_state.saliency_opt = {}
    st.session_state.metrics = {}
    st.session_state.epoch_images = {}


regions = []
with st.sidebar:
    extraction_root_folder = st.text_input("Extraction Root Folder")

    if extraction_root_folder:
        extraction_folders = display_paths_to_extraction_paths(extraction_root_folder)

        selected_extraction = st.selectbox('Extraction folder', extraction_folders.keys())
        if selected_extraction:
            extraction_folder = extraction_folders[selected_extraction]
            paths=get_files_from_folder(extraction_folder)
            regions = st.multiselect('Regions to include', paths, paths)


epochs = st.number_input("Number of epochs", min_value=1, value=200, step=1)
number_of_reference_points = st.number_input("Number of reference points", min_value=50, value=500, step=1)
regularization = float(st.text_input("Regularization strength", value="0.01"))
input_normalization = st.radio(
    'Select Input Normalization', [
        'tic', 'spatial_tic'], index=0, key="norm", horizontal=1, captions=[
        "Total ION Count", "Total ION Count + Spatial"])

settings_str = str(epochs) + str(number_of_reference_points) + str(regularization) + str(input_normalization)

if st.button("Run"):
    for path in regions:
        key = path+settings_str
        if key in st.session_state.saliency_opt:
            result = st.session_state.saliency_opt[key]
            metrics = st.session_state.metrics
            st.image(result)
            st.write(metrics)
        else:
            with st.spinner(text=f"Generating saliency optimization {path}.."):
                st.write(path)
                img = np.float32(np.load(path))

                if input_normalization == 'tic':
                    img = total_ion_count(img)
                else:
                    img = spatial_total_ion_count(img)

                with st.spinner(text=f"Initializing ranking dataset.."):
                    opt = SaliencyOptimization(img, number_of_reference_points, regularization)
                placeholder = st.empty()
                epoch_images = []
                for epoch in range(epochs):
                    with st.spinner(text=f"Epoch {epoch}.."):
                        result = opt.compute_epoch()

                        result_for_gif = result.copy()
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (5, 20)
                        fontScale = 0.5
                        fontColor = (255, 255, 255)
                        thickness = 1
                        lineType = 1
                        result_for_gif = cv2.putText(result_for_gif, f"Epoch: {epoch}",
                                            bottomLeftCornerOfText,
                                            font,
                                            fontScale,
                                            fontColor,
                                            thickness,
                                            lineType)
                        epoch_images.append(result_for_gif)

                        placeholder.image(result_for_gif)
                metrics = MSIVisualizationMetrics(img, result, num_samples=3000).get_metrics()
                st.write(metrics)
                st.session_state.metrics = metrics
                st.session_state.saliency_opt[key] = result
                st.session_state.epoch_images[key] = epoch_images

            save_data(path=path+settings_str)
            