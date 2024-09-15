import streamlit as st
import numpy as np
import cv2
import os
import joblib
from collections import defaultdict
import cmapy
import importlib
import time
import msi_visual.app.utils.viewer
importlib.reload(msi_visual.app.utils.viewer)
from msi_visual.app.utils.viewer import display_comparison, get_stats, get_raw_ion_image, create_ion_image, get_data, viewer, ClickData, region_with_all_comparisons, region_with_region_comparisons, point_with_point_comparisons, show_ion_images
import msi_visual.visualizations
importlib.reload(msi_visual.visualizations)

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000


def reset():
    st.session_state["clicks"] = defaultdict(list)
    st.session_state["data"] = {}
    st.session_state["extraction_mzs"] = {}
    st.session_state['id'] = 0

def reset_clicks():
    st.session_state["clicks"] = defaultdict(list)
    st.session_state['id'] = st.session_state['id'] + 1

def save_cache():
    cache["Visualization folder"] = folder
    cache["bins"] = bins
    cache["Equalize Visualizations"] = equalize
    cache["Comparison type"] = comparison
    cache["Selection type"] = selection_type
    joblib.dump(cache, "viewer.cache")    

if 'rotate' not in st.session_state:
    st.session_state['rotate'] = False

if "clicks" not in st.session_state:
    reset()

if os.path.exists("viewer.cache"):
    cache = joblib.load("viewer.cache")
else:
    cache = {}

viz_tab, region_tab, ion_tab, settings_tab = st.tabs(["Settings", "Visualizations", "Comparisons", "M/Z"])

with settings_tab:
    folder = st.text_input("Visualization folder", value=cache.get("Visualization folder", ""), on_change=reset)
    bins = st.number_input("Number of RGB bins", min_value=5, step=1, value=cache.get("bins", 5))
    equalize = st.checkbox('Equalize Visualizations', value=cache.get("Equalize", False))
    comparison_types = ["Region with Region", "Region with all", "Point with Point"]
    comparison = st.selectbox("Comparison type", comparison_types, on_change=reset_clicks, index=comparison_types.index(cache.get("Comparison type", "Region with all")))
    selection_types = ["Click", "Polygon"]
    selection_type = st.selectbox("Selection type", selection_types, on_change=reset_clicks, index=selection_types.index(cache.get("Selection type", "Polygon")))

    stats_method = st.selectbox("Compairson method", ["U-Test", "Difference in ROI-MEAN"], on_change=reset_clicks)

    rotate = st.checkbox('Rotate', value=st.session_state['rotate'])
    st.session_state['rotate'] = rotate


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

save_cache()