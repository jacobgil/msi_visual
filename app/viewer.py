import streamlit as st
import numpy as np
import cv2
from collections import defaultdict
import cmapy
import importlib
import time
import utils.viewer
importlib.reload(utils.viewer)
from utils.viewer import display_comparison, get_stats, get_raw_ion_image, create_ion_image, get_data, viewer, ClickData, region_with_all_comparisons, region_with_region_comparisons, point_with_point_comparisons, show_ion_images
import msi_visual.visualizations
importlib.reload(msi_visual.visualizations)

def reset_clicks():
    st.session_state["clicks"] = defaultdict(list)
    st.session_state['id'] = st.session_state['id'] + 1

if "clicks" not in st.session_state:
    st.session_state["clicks"] = defaultdict(list)
    st.session_state["data"] = {}
    st.session_state["extraction_mzs"] = {}
    st.session_state['id'] = 0


viz_tab, region_tab, ion_tab, settings_tab = st.tabs(["Visualizations", "Comparisons", "M/Z", "Settings"])

with settings_tab:
    folder = st.text_input("Visualization folder")
    bins = st.number_input("Number of segmentation bins", min_value=5, step=1, value=5)
    equalize = st.checkbox('Equalize')
    comparison = st.selectbox("Comparison type", ["Region with Region", "Region with all", "Point with Point"], on_change=reset_clicks)
    selection_type = st.selectbox("Selection Type", ["Click", "Polygon"], on_change=reset_clicks)
    stats_method = st.selectbox("Compairson method", ["U-Test", "Difference in ROI-MEAN"], on_change=reset_clicks)


with viz_tab:
    viewer(folder, bins, equalize, comparison, selection_type)


with region_tab:
    if comparison == "Region with all":
        region_with_all_comparisons(stats_method)
    elif comparison == "Region with Region":
        region_with_region_comparisons(stats_method)
    elif comparison == "Point with Point":
        point_with_point_comparisons(stats_method)
        
with ion_tab:
    mzs_scores = st.text_input("mz values (comma separated)")
    if mzs_scores:
        try:
            mzs = [float(mz) for mz in list(mzs_scores.replace(' ', '').split(','))]
            show_ion_images(mzs)
        except Exception as e:
            st.write(e)
            pass