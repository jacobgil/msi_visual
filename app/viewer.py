import streamlit as st
import numpy as np
import cv2
from collections import defaultdict
import cmapy
import importlib
import time
import utils.viewer
importlib.reload(utils.viewer)
from utils.viewer import display_comparison, get_stats, get_raw_ion_image, create_ion_image, get_data, viewer, ClickData
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
    comparison = st.selectbox("Comparison type", ["Region with Region", "Region with all"], on_change=reset_clicks)
    selection_type = st.selectbox("Selection Type", ["Click", "Polygon"], on_change=reset_clicks)

with viz_tab:
    viewer(folder, bins, equalize, comparison, selection_type)

if comparison == "Region with all":
    with region_tab:
        if 'ion' in st.session_state:
            with st.sidebar:
                st.image(st.session_state.ion)

        for data_path in st.session_state["clicks"]:
            st.session_state["clicks"][data_path] = sorted(st.session_state["clicks"][data_path], key = lambda x: time.time() - x.timestamp)[:1]
            clicks = st.session_state["clicks"][data_path]
            if len(clicks) == 0:
                break
            cols = st.columns(2)
            data = get_data(data_path)
            mask_a = clicks[0].mask
            cols[0].image(mask_a)
            mask_b = 255 - mask_a
            mask_b[data.max(axis=-1) == 0] = 0
            mask_b[mask_a.max(axis=-1) > 0] = 0

            mask_b_reduced = mask_b * 0
            mask_b_reduced[::3, ::3, :] = mask_b[::3, ::3, :]
            mask_b = mask_b_reduced.copy()


            cols[1].image(mask_b)
            
            extraction_mzs = st.session_state["extraction_mzs"][data_path]
            next_click = ClickData(clicks[0].visualiation_path, clicks[0].point, mask_b, clicks[0].visualization, time.time())
            stats = get_stats(data, extraction_mzs, [clicks[0], next_click])
            
            mask_a = np.uint8(mask_a.max(axis=-1)) * 255
            mask_b = np.uint8(mask_b.max(axis=-1)) * 255
            with st.spinner(text=f"Comparing regions.."):
                display_comparison(data_path,
                    stats,
                    data,
                    clicks[0].visualization,
                    next_click.visualization, mask_a, mask_b, extraction_mzs, threshold=None)
        

else:
    with region_tab:
        if 'ion' in st.session_state:
            with st.sidebar:
                st.image(st.session_state.ion)

        for data_path in st.session_state["clicks"]:
            st.session_state["clicks"][data_path] = sorted(st.session_state["clicks"][data_path], key = lambda x: time.time() - x.timestamp)[:2]
            clicks = st.session_state["clicks"][data_path]
            if len(clicks) > 0:
                cols = st.columns(2)
                mask_a = clicks[0].mask
                cols[0].image(mask_a)
                if len(clicks) == 2:
                    mask_b = clicks[1].mask
                    cols[1].image(mask_b)
                    data = get_data(data_path)
                    extraction_mzs = st.session_state["extraction_mzs"][data_path]
                    stats = get_stats(data, extraction_mzs, clicks)
                    mask_a = np.uint8(clicks[0].mask.max(axis=-1)) * 255
                    mask_b = np.uint8(clicks[1].mask.max(axis=-1)) * 255
                    with st.spinner(text=f"Comparing regions.."):
                        display_comparison(data_path, stats, data, clicks[0].visualization, clicks[1].visualization, mask_a, mask_b, extraction_mzs, threshold=None)
        

with ion_tab:
    mzs_scores = st.text_input("mz values (comma separated)")
    if mzs_scores:
        try:
            mzs = [float(mz) for mz in list(mzs_scores.replace(' ', '').split(','))]
            for path in st.session_state["data"]:
                img = st.session_state["data"][path]
                aggregated = []
                for mz in mzs:
                    extraction_mzs = st.session_state["extraction_mzs"][path]

                    ion, mz = create_ion_image(img, mz, extraction_mzs)
                    st.image(ion)
                    ion, mz = get_raw_ion_image(img, mz, extraction_mzs)                    
                    aggregated.append(ion)
                
                if len(aggregated) > 1:
                    st.write('mean-aggregated')

                    aggregated = np.float32(aggregated)
                    aggregated = np.mean(aggregated, axis=0)
                    aggregated = aggregated / np.max(aggregated)
                    aggregated = np.uint8(aggregated * 255)
                    aggregated = cv2.applyColorMap(aggregated, cmapy.cmap('viridis'))[:, :, ::-1].copy()
                    st.image(aggregated)
            
        except Exception as e:
            st.write(e)
            pass