import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import joblib
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from PIL import Image
from st_pages import show_pages_from_config, add_page_title
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from msi_visual import nmf_3d
from msi_visual import parametric_UMAP 
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder
from keras.callbacks import Callback

trainprogress_bar = st.progress(0, text="Training..")
trainprogress_bar.empty()

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global train_progress
        train_progress = train_progress + 0.01
        trainprogress_bar.progress(int(train_progress * 100), "Training..")


# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

show_pages_from_config()

def get_settings():
    return {
        "extraction_root_folder": extraction_root_folder,
        "method": method,
        "output_path": output_path
    }


def save_to_cache(cache_path='dim_reduc_train.cache'):
    cache = get_settings()
    
    joblib.dump(cache, cache_path)

if os.path.exists("dim_reduc_train.cache"):
    state = joblib.load("dim_reduc_train.cache")
    cached_state = defaultdict(str)
    for k, v in state.items():
        cached_state[k] = v
else:
    cached_state = defaultdict(str)



if 'bins' not in st.session_state:
    st.session_state.bins = 5

with st.sidebar:
    output_path = st.text_input('Output path for segmentation model', value=cached_state["output_path"])    
    extraction_root_folder = st.text_input("Extraction Root Folder", value=cached_state['extraction_root_folder'])

    if extraction_root_folder:
        extraction_folders = display_paths_to_extraction_paths(extraction_root_folder)

        selected_extraction = st.selectbox('Extraction folder', extraction_folders.keys())
        if selected_extraction:
            extraction_folder = extraction_folders[selected_extraction]
            regions = st.multiselect('Regions to include', get_files_from_folder(extraction_folder))

            extraction_args = eval(open(Path(extraction_folder) / "args.txt").read())
            st.session_state.bins = extraction_args.bins
            st.session_state.extraction_start_mz = extraction_args.start_mz
            st.session_state.extraction_end_mz = extraction_args.end_mz

        start_mz = st.number_input('Start m/z', st.session_state.extraction_start_mz, step=50)
        end_mz = st.number_input('End m/z', 
            min_value=st.session_state.extraction_start_mz+50, 
            max_value=st.session_state.extraction_end_mz, 
            value=None, 
            step=50)
        
        sub_sample = st.number_input('Subsample pixels', value=None, step=1)


    method = st.selectbox('Dimensionality Reduction Method', ['1D Parametric UMAP ', '3D Parametric UMAP ', 'NMF 3D'])
    normalization = st.radio('Normalization', ['tic', 'spatial_tic'], index=0, key="norm", horizontal=1, captions=["total ion count", "spatial"])

save_to_cache()

if output_path:
    start = st.button(f"Train {method}")
    if start:
        if start_mz is not None:
            start_bin = int(st.session_state.bins * (start_mz - st.session_state.extraction_start_mz))
        else:
            start_bin = 0
        
        if end_mz is not None:
            end_bin = int(st.session_state.bins * (end_mz - st.session_state.extraction_start_mz))
        else:
            end_bin = None

        if method == '1D Parametric UMAP ':
            model = parametric_UMAP .UMAP VirtualStain(n_components=1, start_bin=start_bin, end_bin=end_bin)
        elif method == '3D Parametric UMAP ':
            model = parametric_UMAP .UMAP VirtualStain(n_components=3, start_bin=start_bin, end_bin=end_bin)
        elif method == 'NMF 3D':
            model = nmf_3d.NMF3D(start_bin=start_bin, end_bin=end_bin)

        norm_funtion = {'tic': total_ion_count, 'median': median_ion, 'spatial_tic': spatial_total_ion_count}[normalization]

        if sub_sample:
            images = [norm_funtion(np.load(p)[::int(sub_sample), ::int(sub_sample), :]) for p in regions]
        else:
            images = [norm_funtion(np.load(p)) for p in regions]
        
        with st.spinner(text=f"Training {method}.."):
            train_progress = 0.0

            trainprogress_bar = st.progress(0, text=f"Training {method}..")

            keras_fit_kwargs = {"callbacks": [MyCallback()]}

            if 'UMAP ' in method:
                model.fit(images, keras_fit_kwargs=keras_fit_kwargs)
            else:
                model.fit(images)
            trainprogress_bar.empty()

            model.save(output_path)