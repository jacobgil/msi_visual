import streamlit as st
import numpy as np
from msi_visual import visualizations
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder
import cv2
from pathlib import Path
from argparse import Namespace

def load_image(path):
    if path in st.session_state.images:
        img = st.session_state.images[path]
    else:
        img = np.load(path)
        
        st.session_state.images[path] = img

    return img

def create_ion_image(img, mz_orig):
    mz_index = np.abs(np.float32(st.session_state.extraction_mzs) - \
                      float(mz_orig)).argmin()
    mz = st.session_state.extraction_mzs[mz_index]
    if mz_orig != mz:
        st.text(f'Showing ION image for closest mz {mz}')
    mz_img = visualizations.create_ion_image(img, mz_index) * 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5, 20)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 1
    mz_img = cv2.putText(mz_img, f"{mz:.3f}",
                         bottomLeftCornerOfText,
                         font,
                         fontScale,
                         fontColor,
                         thickness,
                         lineType)

    return mz_img

if 'images' not in st.session_state:
    st.session_state.images = {}

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

            extraction_args = eval(
                open(
                    Path(extraction_folder) /
                    "args.txt").read())
            st.session_state.bins = extraction_args.bins
            st.session_state.extraction_start_mz = extraction_args.start_mz
            st.session_state.extraction_end_mz = extraction_args.end_mz
            try:
                st.session_state.extraction_mzs = extraction_args.mzs
            except Exception as e:
                st.session_state.extraction_mzs = list(
                    np.arange(
                        st.session_state.extraction_start_mz,
                        st.session_state.extraction_end_mz + 1,
                        1.0 / st.session_state.bins))
                st.session_state.extraction_mzs = [
                    float(f"{mz:.3f}") for mz in st.session_state.extraction_mzs]


mz = st.text_input('Create ION image for m/z:')
if mz:
    for path in regions:
        img = load_image(path)
        st.write(path)
        st.image(create_ion_image(img, mz))
