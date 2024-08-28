import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import joblib
import datetime
from pathlib import Path
from argparse import Namespace
from PIL import Image
from collections import defaultdict
from msi_visual import nmf_segmentation, kmeans_segmentation
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder

if 'bins' not in st.session_state:
    st.session_state.bins = 5

def get_settings():
    return {
        "extraction_root_folder": extraction_root_folder,
        "extraction_folders": extraction_folders,
        "selected_extraction": selected_extraction,
        "start_mz": start_mz,
        "end_mz": end_mz,
        'number_of_components': number_of_components,
        "normalization": normalization,
        "sub_sample": sub_sample,
        "sample_name": sample_name,
        "output_file": output_file,
        "model_root_folder": model_root_folder,
        "output_path": output_path
    }

def save_to_cache(cache_path='train.cache'):    
    cache = get_settings()
    
    joblib.dump(cache, cache_path)

extraction_folders = None
selected_extraction = None
start_mz = None
end_mz = None
output_path = None

if os.path.exists("train.cache"):
    state = joblib.load("train.cache")
    cached_state = defaultdict(str)
    for k, v in state.items():
        cached_state[k] = v
else:
    cached_state = defaultdict(str)

with st.sidebar:    
    model_type = st.radio("Segmentation method", key="model",options=["NMF", "Kmeans"],)
        
    extraction_root_folder_default = ""
    extraction_root_folder = st.text_input("Extraction Root Folder", value=cached_state['extraction_root_folder'])
    
    if extraction_root_folder:
        extraction_folders = display_paths_to_extraction_paths(extraction_root_folder)

        extraction_folders_list = list(extraction_folders.keys())
        default = None
        if cached_state['selected_extraction'] in extraction_folders_list:
            default = cached_state['selected_extraction'].index(cached_state['selected_extraction'])

        selected_extraction = st.selectbox('Extraction folder', extraction_folders_list, index=default)
        if selected_extraction:
            extraction_folder = extraction_folders[selected_extraction]
            regions = st.multiselect('Regions to include', get_files_from_folder(extraction_folder))

            extraction_args = eval(open(Path(extraction_folder) / "args.txt").read())
            st.session_state.bins = extraction_args.bins
            st.session_state.extraction_start_mz = extraction_args.start_mz
            st.session_state.extraction_end_mz = extraction_args.end_mz

        if 'extraction_start_mz' in st.session_state:
            min_value = st.session_state.extraction_start_mz
            max_value = st.session_state.extraction_end_mz
            step = 50
        else:
            min_value = None
            max_value = None
            step = None

        start_mz = st.number_input('Start m/z', min_value=min_value, max_value=max_value, step=step, value=cached_state['start_mz'])
        end_mz = st.number_input('End m/z', min_value=min_value, max_value=max_value, value=cached_state['end_mz'], step=step)
        number_of_components = st.number_input('Number of components (k)', min_value=2, max_value=100, value=5, step=5)
        sub_sample = st.number_input('Subsample pixels', value=1, step=1)
        normalization = st.radio('Normalization', ['tic', 'spatial_tic'], index=0, key="norm", horizontal=1, captions=["total ion current", "spatial"])
        
        if normalization == "spatial_tic":normalization_short = 'sptic'
        else: normalization_short = normalization
        
        save_model_sel = st.radio(label='Save model to path', key="path", options=['generated', 'custom'], horizontal=1)
        
        if save_model_sel == "custom":
            output_path = st.text_input('Custom output path', value=f"..\{model_type}-model.joblib")    
            sample_name = cached_state['sample_name']
            output_file = cached_state['output_file']
            model_root_folder = cached_state['model_root_folder']
        else:
            sample_name = st.text_input('Add sample name / identifier 👇', value=cached_state['sample_name'])
            output_file_suggestion  = f"{sample_name}_{normalization_short}_subs{sub_sample}_b{extraction_args.bins}_k{number_of_components}_startmz{start_mz}_endmz{end_mz}_{model_type}.joblib" 
            output_file  = st.text_input('Suggested output file name', value=output_file_suggestion)
            model_root_folder = st.text_input("Add Model Root Folder 👇", value=cached_state['model_root_folder'])
            output_path_default = f"{model_root_folder}\{model_type}-models\{sample_name}\\"
            output_path = st.text_input('Suggested output path for segmentation model', value=output_path_default + output_file)
        
        save_to_cache()

start = st.button("Train " + model_type + " segmentation")
if start:

    extraction_args = eval(open(Path(extraction_folder) / "args.txt").read())
    st.session_state.bins = extraction_args.bins
    st.session_state.extraction_start_mz = extraction_args.start_mz
    st.session_state.extraction_end_mz = extraction_args.end_mz


    if start_mz is not None:
        start_bin = int(st.session_state.bins * (start_mz - st.session_state.extraction_start_mz))
    else:
        start_bin = 0
    
    if end_mz is not None:
        end_bin = int(st.session_state.bins * (end_mz - st.session_state.extraction_start_mz))
    else:
        end_bin = None
    
    st.write(model_type)
    if model_type == "NMF":
        seg = nmf_segmentation.NMFSegmentation(k=int(number_of_components), normalization=normalization, start_bin=start_bin, end_bin=end_bin)
    if model_type == "Kmeans":
        seg = kmeans_segmentation.KmeansSegmentation(k=int(number_of_components), normalization=normalization, start_bin=start_bin, end_bin=end_bin)
    else:
        seg = kmeans_segmentation.KmeansSegmentation(k=int(number_of_components), normalization=normalization, start_bin=start_bin, end_bin=end_bin)

    if sub_sample == 1:
        images = [np.load(p) for p in regions]
    else:
        images = [np.load(p)[::int(sub_sample), ::int(sub_sample), :] for p in regions]
    
    with st.spinner(text="Training segmentation.."):
        seg.fit(images)

    output_path_folder = str(Path(output_path).parent)
    os.makedirs(output_path_folder, exist_ok=True)
    joblib.dump(seg, output_path)

    for index, img in enumerate(seg.visualize_training_components(images)):
        st.image(img)
        
        folder = datetime.datetime.today().strftime('%Y-%m-%d')
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img).save(Path(folder) / f"{model_type}_{number_of_components}_{index}.png")