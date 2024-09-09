import streamlit as st
import os
import joblib
from collections import defaultdict
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder

def get_extraction(cache_path='extraction.cache'):
    selected_extraction, regions = None, None
    if os.path.exists(cache_path):
        cache = joblib.load(cache_path)
    else:
        cache = defaultdict(str)

    extraction_root_folder = st.text_input(
        'Extraction Root Folder',
        value=cache["Extraction Root Folder"])
    if extraction_root_folder:
        extraction_folders = display_paths_to_extraction_paths(
            extraction_root_folder)
        extraction_folders_keys = list(extraction_folders.keys())
        if cache['Extraction folder'] == '':
            cache['Extraction folder'] = extraction_folders_keys[0]
        if cache['Extraction folder'] in extraction_folders_keys:
            extraction_folders_keys_index = extraction_folders_keys.index(
                cache['Extraction folder'])
        else:
            extraction_folders_keys_index = None
        selected_extraction = st.multiselect(
            'Extraction folder',
            extraction_folders_keys,
            [])
        
        if selected_extraction:
            region_list = []
            if cache['Regions to include'] == '':
                default = None
            else:
                default = cache['Regions to include']

            for folder in selected_extraction:
                extraction_folder = extraction_folders[folder]
                region_list.extend(get_files_from_folder(extraction_folder))

            regions = st.multiselect(
                'Regions to include', region_list, default=region_list)

        cache['Regions to include'] = regions
        cache['Extraction folder'] = selected_extraction
        cache['Extraction Root Folder'] = extraction_root_folder

        joblib.dump(cache, cache_path)
        return regions
