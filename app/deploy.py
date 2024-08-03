import base64
import hashlib
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from msi_visual.percentile_ratio import percentile_ratio_rgb
from msi_visual.percentile_ratio_segmentation import PercentileRatioSegmentation
from msi_visual.metrics import MSIVisualizationMetrics
from msi_visual.auto_colorization import AutoColorizeRandom, AutoColorizeArea
from msi_visual.avgmz import AvgMZVisualization
from msi_visual.rare_nmf_segmentation import SegmentationPercentileRatio
from msi_visual.avgmz_nmf_segmentation import SegmentationAvgMZVisualization
from msi_visual.UMAP _nmf_segmentation import SegmentationUMAP Visualization
from msi_visual import parametric_UMAP 
from msi_visual.utils import get_certainty, set_region_importance
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder
from msi_visual import visualizations
from msi_visual import nmf_segmentation
import streamlit as st
import glob
import os
import traceback
import datetime
import json
import math
import sys
import time
from PIL import Image
import numpy as np
import joblib
from matplotlib import colormaps
from matplotlib import pyplot as plt
from collections import defaultdict
from argparse import Namespace
from st_pages import show_pages_from_config, add_page_title
from streamlit_image_coordinates import streamlit_image_coordinates

from pathlib import Path
import importlib
import cmapy
import wx
import cv2
import random
import msi_visual
importlib.reload(msi_visual)
importlib.reload(visualizations)

add_page_title()

show_pages_from_config()


def make_hash_sha256(o):
    hasher = hashlib.md5()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.urlsafe_b64encode(hasher.digest()).decode()


def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o


def get_settings():
    return {'Extraction Root Folder': extraction_root_folder,
            'Extraction folder': selected_extraction,
            'Slide ROI to include': regions,
            'Model folder': nmf_model_folder,
            'confidence_thresholds': st.session_state.confidence_thresholds,
            'UMAP  Model folder (optional)': UMAP _model_folder,
            'combination_method': combination_method,
            'Segmentation model': nmf_model_name,
            'Image to show': image_to_show,
            'output_normalization': output_normalization,
            'input_normalization': input_normalization,
            'color_schemes': st.session_state.color_schemes,
            'region_importance': st.session_state.region_importance,
            'current_color_scheme': current_color_scheme}


def get_settings_hash():
    settings = get_settings()
    return make_hash_sha256(settings)


def model_hash():
    settings = {'UMAP _model_folder': st.session_state.UMAP _model_folder,
                'nmf_model_folder': st.session_state.nmf_model_folder}
    return make_hash_sha256(settings)


def display_mzs(aucs_to_display, color_a, color_b, threshold=0.5, title=None):

    pos_aucs = {mz: -aucs_to_display[mz]
                for mz in aucs_to_display if aucs_to_display[mz] < threshold}
    neg_aucs = {
        mz: aucs_to_display[mz] for mz in aucs_to_display if aucs_to_display[mz] > (
            1 - threshold)}

    pos_aucs_keys = sorted(list(pos_aucs.keys()),
                           key=lambda x: pos_aucs[x])[-num_mzs:]
    neg_aucs_keys = sorted(list(neg_aucs.keys()),
                           key=lambda x: neg_aucs[x])[-num_mzs:]

    pos_aucs = {mz: pos_aucs[mz] for mz in pos_aucs_keys}
    neg_aucs = {mz: neg_aucs[mz] for mz in neg_aucs_keys}

    print(f"display_mzs: {title} {len(neg_aucs)} {len(pos_aucs)}")
    for auc_group, color in zip([pos_aucs, neg_aucs], [color_a, color_b]):
        header_cols = st.columns(2)
        header_cols[0].text(f'{title} m/z values for ')
        header_cols[1].image(np.uint8(color * np.ones((64, 64, 3))))
        sorted_indices = sorted(
            list(
                auc_group.keys()),
            key=lambda mz: aucs_to_display[mz],
            reverse=True)
        N = 8
        num_rows = math.ceil(len(sorted_indices) / N)
        for row in range(num_rows):
            indices = sorted_indices[N * row: N * row + N]
            indices = indices + [None] * (N - len(indices))
            for i, (mz, col) in enumerate(
                    zip(indices, st.columns(len(indices)))):
                if mz is None:
                    col.button(
                        "",
                        key=f"{st.session_state.run_id}+{title}+default+{i}+{row}+{col}+{mz}",
                        disabled=True,
                        use_container_width=True)
                else:
                    col.button(
                        f"{mz}",
                        use_container_width=True,
                        key=f"{str(st.session_state.run_id)}+{title}+{mz}+{i}+{row}+{col}",
                        on_click=create_ion_image,
                        args=[
                            mz,
                            image_to_show])
                    # create_ion_image(mz, image_to_show)


def get_model():
    if nmf_model_name and (
            "NMF" in combination_method or "Seg" in combination_method):
        # if nmf_model_name:
        if ('segmentation_model_path' in st.session_state and st.session_state[
                'segmentation_model_path'] != nmf_model_name) or st.session_state['segmentation_model'] is None:
            nmf_model_path = [
                p for p in nmf_model_paths if Path(p).stem == nmf_model_name][0]
            model = joblib.load(open(nmf_model_path, 'rb'))
            st.session_state['segmentation_model'] = model
            st.session_state['segmentation_model_path'] = nmf_model_name
        else:
            model = st.session_state['segmentation_model']

        if model.k > len(st.session_state.color_schemes):
            print(
                "Resetting color scheme", len(
                    st.session_state.color_schemes), model.k)
            st.session_state.color_schemes = ["gist_yarg"] * model.k
            st.session_state.region_importance = {}

    if combination_method == "PercentileRatio":
        model = PercentileRatioSegmentation(equalize)

    if combination_method == "Seg+UMAP ":
        if UMAP _model_folder:
            if st.session_state['model'][combination_method] is None:
                UMAP  = parametric_UMAP .UMAP VirtualStain()
                UMAP .load(UMAP _model_folder)
                model = SegmentationUMAP Visualization(UMAP , model)
            else:
                model = st.session_state['model'][combination_method]
        else:
            model = None

    elif combination_method == "Seg+Rare":
        if st.session_state['model'][combination_method] is None:
            model = SegmentationPercentileRatio(model)
        else:
            model = st.session_state['model'][combination_method]

    elif combination_method == "Seg+SpectrumHeatmap":
        if st.session_state['model'][combination_method] is None:
            model = SegmentationAvgMZVisualization(model)
        else:
            model = st.session_state['model'][combination_method]
    elif combination_method == "SpectrumHeatmap":
        if st.session_state['model'][combination_method] is None:
            model = AvgMZVisualization()
        else:
            model = st.session_state['model'][combination_method]

    elif combination_method == "Dim. Reduction UMAP " and UMAP _model_folder:
        if ('UMAP _model_folder' in st.session_state and st.session_state['UMAP _model_folder'] != UMAP _model_folder) \
                or (st.session_state['model'][combination_method] is None) \
                or ('UMAP _model_folder' not in st.session_state):
            model = parametric_UMAP .UMAP VirtualStain()
            model.load(UMAP _model_folder)
            st.session_state['UMAP _model_folder'] = UMAP _model_folder
        else:
            print(f"loading {combination_method}")
            model = st.session_state['model'][combination_method]

    if model is not None:
        st.session_state['model'][combination_method] = model
    print(model, nmf_model_name, UMAP _model_folder)
    return model


def create_ion_image(mz_orig, image_to_show):
    mz_image = st.session_state.results[input_normalization +
                                        image_to_show]["mz_image"]
    mz_index = np.abs(np.float32(st.session_state.extraction_mzs) - \
                      float(mz_orig)).argmin()
    mz = st.session_state.extraction_mzs[mz_index]
    if mz_orig != mz:
        st.text(f'Showing ION image for closest mz {mz}')
    mz_img = visualizations.create_ion_image(mz_image, mz_index) * 1

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

    st.session_state.ion_image = (mz, mz_img)


def auto_colorization():
    result = st.session_state.results[input_normalization + image_to_show]
    if "heatmap" in result:
        st.title('Auto Colorization 🎨')
        low_var_ratio = st.slider('Fraction', 0.0, 1.0, step=0.05, value=0.5)
        if st.button('Random Colorization'):
            colorization = AutoColorizeRandom(
                json.load(open("auto_color_schemes.json")))
            k = st.session_state['segmentation_model'].k
            st.session_state.color_schemes = colorization.colorize(
                result["mz_image"],
                result["segmentation_mask_argmax"],
                result["heatmap"],
                low_var_ratio,
                k)
        if st.button('Area based colorization'):
            k = st.session_state['segmentation_model'].k
            colorization = AutoColorizeArea(
                json.load(open("auto_color_schemes.json")))
            st.session_state.color_schemes = colorization.colorize(
                result["mz_image"],
                result["segmentation_mask_argmax"],
                result["heatmap"],
                low_var_ratio,
                k)
        current_color_scheme = str([str(x)
                                   for x in st.session_state.color_schemes])


def save_data():
    if 'results' not in st.session_state:
        return
    if image_to_show:
        folder = datetime.datetime.today().strftime('%Y-%m-%d')
        os.makedirs(folder, exist_ok=True)
        image_name = Path(image_to_show).stem
        prefix = combination_method
        now = datetime.datetime.now()
        prefix = get_settings_hash() + "_" + prefix

        if 'latest_diff' in st.session_state:
            img, visualization, label_a, label_b = st.session_state.latest_diff
            Image.fromarray(img).save(Path(folder) /
                                      f"{image_name}_{label_a}_{label_b}_{prefix}.png")

            if 'latest_heatmaps' in st.session_state:
                for index, heatmap in enumerate(
                        st.session_state.latest_heatmaps):
                    Image.fromarray(heatmap).save(
                        Path(folder) / f"{image_name}_{[label_a, label_b][index]}_{prefix}_mask.png")
        else:
            visualization = st.session_state.results[input_normalization +
                                                     image_to_show]["visualization"]
        Image.fromarray(visualization).save(
            Path(folder) / f"{image_name}_{prefix}.png")

        if 'ion_image' in st.session_state:
            mz, ion_image = st.session_state.ion_image
            Image.fromarray(ion_image).save(
                Path(folder) / f"{image_name}_mz_{mz}.png")

        if 'k_seg' in st.session_state:
            k_index, k_seg = st.session_state.k_seg
            Image.fromarray(k_seg).save(Path(folder) /
                                        f"{image_name}_{prefix}_region_{k_index}.png")

        if st.session_state.results[input_normalization +
                                    image_to_show]["certainty_image"] is not None:
            certainty = st.session_state.results[input_normalization +
                                                 image_to_show]["certainty_image"]
            Image.fromarray(
                np.uint8(
                    255 *
                    certainty)).save(
                Path(folder) /
                f"{image_name}_{prefix}_certainty.png")


def save_to_cache(cache_path='deploy.cache'):
    cache = get_settings()

    joblib.dump(cache, cache_path)


if os.path.exists("deploy.cache"):
    state = joblib.load("deploy.cache")
    cached_state = defaultdict(str)
    for k, v in state.items():
        cached_state[k] = v
else:
    cached_state = defaultdict(str)

app = wx.App()
wx.DisableAsserts()

if 'UMAP _model_folder' not in st.session_state:
    st.session_state.UMAP _model_folder = None
if 'nmf_model_folder' not in st.session_state:
    st.session_state.nmf_model_folder = None

if 'color_schemes' not in st.session_state:
    st.session_state.color_schemes = cached_state['color_schemes']
if st.session_state.color_schemes == '':
    st.session_state.color_schemes = ["gist_yarg"] * 100
if 'region_importance' not in st.session_state:
    st.session_state.region_importance = {}

if 'ion_image' not in st.session_state:
    st.session_state['ion_image'] = None

if 'comparison_results' not in st.session_state:
    st.session_state['comparison_results'] = {}

if 'model' not in st.session_state:
    st.session_state.model = {"PercentileRatio": None,
                              "Dim. Reduction NMF": None,
                              "Dim. Reduction UMAP ": None,
                              'Seg+Rare': None,
                              'Seg+SpectrumHeatmap': None,
                              'Seg+UMAP ': None,
                              "Segmentation": None,
                              "SpectrumHeatmap": None}

    st.session_state.segmentation_model = None

if 'setting_hash' not in st.session_state:
    st.session_state.setting_hash = None

if 'model_hash' not in st.session_state:
    st.session_state.model_hash = None

if 'confidence_thresholds' not in st.session_state:
    st.session_state.confidence_thresholds = None

if 'images' not in st.session_state:
    st.session_state.images = {}

if 'pr' not in st.session_state:
    st.session_state.pr = {}

regions = []

results = {}
image_to_show = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = 0
if 'bins' not in st.session_state:
    st.session_state.bins = 5

if 'coordinates' not in st.session_state or st.session_state.coordinates is None:
    st.session_state.coordinates = defaultdict(list)

try:
    UMAP _model_folder, nmf_model_path, nmf_model_name = None, None, None
    region_colorscheme = None
    with st.sidebar:
        st.title('Data Source')
        extraction_root_folder = st.text_input(
            'Extraction Root Folder',
            value=cached_state["Extraction Root Folder"])
        if extraction_root_folder:
            extraction_folders = display_paths_to_extraction_paths(
                extraction_root_folder)
            extraction_folders_keys = list(extraction_folders.keys())
            if cached_state['Extraction folder'] == '':
                cached_state['Extraction folder'] = extraction_folders_keys[0]
            if cached_state['Extraction folder'] in extraction_folders_keys:
                extraction_folders_keys_index = extraction_folders_keys.index(
                    cached_state['Extraction folder'])
            else:
                extraction_folders_keys_index = None
            selected_extraction = st.selectbox(
                'Extraction folder',
                extraction_folders_keys,
                index=extraction_folders_keys_index)
            if selected_extraction:
                extraction_folder = extraction_folders[selected_extraction]
                region_list = get_files_from_folder(extraction_folder)
                if cached_state['Regions to include'] == '':
                    default = None
                else:
                    default = cached_state['Regions to include']
                    if False in [
                            r in region_list for r in cached_state['Regions to include']]:
                        default = None

                regions = st.multiselect(
                    'Regions to include', region_list, default=default)

        with st.sidebar:
            image_to_show = st.selectbox(
                'Image to show',
                list(regions),
                key=st.session_state.run_id)

        st.divider()
        st.title('Model Settings')
        nmf_model_folder = st.text_input(
            'Model folder', value=cached_state['Model folder'])
        if nmf_model_folder:
            nmf_model_paths = list(glob.glob(nmf_model_folder + "\\*.joblib")) \
                + list(glob.glob(nmf_model_folder + "\\*\\*.joblib"))
            nmf_model_display_paths = [Path(p).stem for p in nmf_model_paths]

            default = cached_state['Segmentation model']
            if default == '':
                default = None
            else:
                if default in nmf_model_display_paths:
                    default = nmf_model_display_paths.index(default)
                else:
                    default = None
            nmf_model_name = st.selectbox(
                'Segmentation model',
                nmf_model_display_paths,
                index=default)
            st.session_state.nmf_model_path = nmf_model_path

        combination_method = st.radio('Visualization Method',
                                      ["PercentileRatio",
                                       "Dim. Reduction UMAP ",
                                       "Dim. Reduction NMF",
                                       "Segmentation",
                                       "Seg+UMAP ",
                                       "Seg+SpectrumHeatmap",
                                       "Seg+Rare",
                                       "SpectrumHeatmap"],
                                      index=0)

        if "UMAP " in combination_method:
            UMAP _model_folder = st.text_input(
                'UMAP  Model folder (optional)',
                value=cached_state['UMAP  Model folder (optional)'])
        else:
            UMAP _model_folder = None
        st.session_state.UMAP _model_folder = UMAP _model_folder

        if combination_method == "PercentileRatio":
            equalize = st.checkbox("Equalize")

        st.divider()
        st.title('Normalization')
        output_normalization = st.radio(
            "Select segmentation normalization", [
                'spatial_norm', 'None'])
        input_normalization = st.radio(
            'Select Input Normalization', [
                'tic', 'spatial_tic'], index=0, key="norm", horizontal=1, captions=[
                "Total ION Count", "Total ION Count + Spatial"])
        st.divider()
        sub_sample = 1

        st.title('Statistics')
        num_samples = st.number_input(
            'Samples for metrics', min_value=300, step=1)
        num_mzs = st.number_input(
            'Number of top m/z values to show on clicks',
            min_value=30,
            step=1)

    if image_to_show is not None:
        if model_hash() != st.session_state.model_hash:
            st.session_state.model_hash = model_hash

        model = get_model()

        st.write(combination_method)

    colorschemes = list(colormaps)
    with st.sidebar:
        st.divider()
        st.title('Region Settings')
        if combination_method not in [
            "SpectrumHeatmap",
            "Dim. Reduction UMAP ",
            "Dim. Reduction NMF",
                "PercentileRatio"]:
            region_selectbox = 0
            if image_to_show:
                region_selectbox = st.selectbox(
                    f"Region to control", [
                        i for i in range(
                            model.k)], index=0)

            region_default = 1.0
            if region_selectbox in st.session_state.region_importance:
                region_default = st.session_state.region_importance[int(
                    region_selectbox)]
            region_factor = st.slider(
                label=f'Region weight ({region_selectbox})',
                min_value=0.0,
                max_value=4.0,
                value=region_default,
                step=0.1)

            if combination_method != "Segmentation":
                default = colorschemes.index(st.session_state.color_schemes[0])
                if st.session_state.color_schemes != '' and len(
                        st.session_state.color_schemes) > int(region_selectbox):
                    default = colorschemes.index(
                        st.session_state.color_schemes[int(region_selectbox)])
                region_colorscheme = st.selectbox(
                    f"Color Scheme for segmented region ({region_selectbox})",
                    colorschemes,
                    index=default)

                col1, col2 = st.columns(2)

                with col1:
                    exportbutton = st.button("Export Settings 💾")

                with col2:
                    importbutton = st.button("Import Settings 📂")

                if exportbutton: 
                    dialog = wx.FileDialog(
                        None, "Settings file", style=wx.DD_DEFAULT_STYLE)
                    if dialog.ShowModal() == wx.ID_OK:
                        # folder_path will contain the path of the folder you
                        # have selected as
                        export_path = dialog.GetPath()
                        with open(export_path, 'w') as f:
                            data_to_save = {
                                'color_schemes': st.session_state.color_schemes,
                                'region_weights': st.session_state.region_importance}
                            json.dump(data_to_save, f)

                if importbutton:
                    dialog = wx.FileDialog(
                        None, "Settings file", style=wx.DD_DEFAULT_STYLE)
                    if dialog.ShowModal() == wx.ID_OK:
                        # folder_path will contain the path of the folder you
                        # have selected as
                        export_path = dialog.GetPath()
                        with open(export_path, 'r') as f:
                            loaded_data = json.load(f)
                            if 'color_schemes' in loaded_data:
                                st.session_state.color_schemes = loaded_data['color_schemes']
                                st.session_state.region_importance = loaded_data['region_weights']
                            else:
                                st.session_state.color_schemes = loaded_data

                current_color_scheme = str(
                    [str(x) for x in st.session_state.color_schemes])

            else:
                region_colorscheme = st.selectbox(
                    f"Color Scheme", colorschemes, index=colorschemes.index("gist_rainbow"))
                current_color_scheme = region_colorscheme

            col1, col2 = st.columns(2)

            with col1:
                resetbutton = st.button('Re-set ↺')

            with col2:
                updatebutton = st.button('Update ►')

            if resetbutton:
                st.session_state.color_schemes = ["gist_yarg"] * 100
                st.session_state.region_importance = {}

            if updatebutton:
                st.session_state.color_schemes[int(
                    region_selectbox)] = region_colorscheme
                st.session_state.region_importance[int(
                    region_selectbox)] = region_factor

            st.divider()
            st.title('Segmentation Confidence')
            certainty_slider = st.slider(
                'Confidence threshold range', 0.0, 1.0, (0.0, 1.0), step=0.05,)
            st.session_state['confidence_thresholds'] = certainty_slider
        else:
            region_colorscheme = st.selectbox(
                f"Color Scheme",
                colorschemes,
                index=colorschemes.index("gist_rainbow"))
            current_color_scheme = region_colorscheme

    start = st.button("Run ►")

    for path in regions:
        if input_normalization + path not in st.session_state.images:
            if sub_sample:
                img = np.load(path)[
                    ::int(sub_sample), ::int(sub_sample), :]
            else:
                img = np.load(path)

            normalization = {
                'tic': total_ion_count,
                'median': median_ion,
                'spatial_tic': spatial_total_ion_count}[input_normalization]
            with st.spinner(text=f"Loading {path}.."):
                st.session_state.images[input_normalization +
                                        path] = normalization(img)
            print('Loaded image')

    col1, col2 = st.columns(2)
    if start:
        st.session_state.setting_hash = None
        save_to_cache()
        st.session_state.run_id = st.session_state.run_id + 1

        with st.spinner(text="Computing visualization.."):
            st.session_state.coordinates = None
            st.session_state.results = None
            st.session_state.difference_visualizations = None
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

            if 'results' in st.session_state:
                del st.session_state.results

            st.session_state.results = {}
            for path in regions:
                img = st.session_state.images[input_normalization + path]
                results = {}
                results["mz_image"] = img

                t0 = time.time()
                if "+" in combination_method:
                    results["segmentation"], results["heatmap"] = model.predict(
                        img)
                else:
                    results["segmentation"] = model.predict(img)

                print("Factorization took", time.time() - t0)

                st.session_state.results[input_normalization + path] = results

    if 'results' in st.session_state:
        for path, data in st.session_state.results.items():
            if path == input_normalization + image_to_show:
                img = data["mz_image"]
                segmentation_mask = data["segmentation"].copy()

                with st.sidebar:
                    st.divider()
                    if "+" in combination_method:
                        auto_colorization()

                if st.session_state.setting_hash != get_settings_hash():
                    st.session_state.setting_hash = get_settings_hash()

                    roi_mask = np.uint8(img.max(axis=-1) > 0)
                    if 'confidence_thresholds' in st.session_state and st.session_state[
                            'confidence_thresholds']:
                        low, high = st.session_state['confidence_thresholds']
                        if 'certainty_image' in st.session_state.results[path]:
                            if st.session_state.results[path]['certainty_image'] is not None:
                                certainty_image = st.session_state.results[path]["certainty_image"]
                                roi_mask[certainty_image < low] = 0
                                roi_mask[certainty_image > high] = 0

                    if combination_method in [
                        "Dim. Reduction UMAP ",
                        "Dim. Reduction NMF",
                            "PercentileRatio"]:
                        sub_segmentation_mask, visualization = model.segment_visualization(
                            img, segmentation_mask, method=output_normalization)
                        segmentation_mask_argmax = None
                        segmentation_mask_for_comparisons = sub_segmentation_mask

                    if "+" in combination_method:
                        segmentation_mask, sub_segmentation_mask, visualization = model.segment_visualization(img,
                                                                                                                segmentation_mask,
                                                                                                                data["heatmap"],
                                                                                                                roi_mask,
                                                                                                                color_scheme_per_region=st.session_state.color_schemes,
                                                                                                                method=output_normalization,
                                                                                                                region_factors=st.session_state.region_importance)
                        segmentation_mask_argmax = segmentation_mask.argmax(
                            axis=0)
                        segmentation_mask_for_comparisons = sub_segmentation_mask

                    elif combination_method == "Segmentation":
                        segmentation_mask, visualization = model.segment_visualization(
                            img, segmentation_mask, region_colorscheme, method=output_normalization, region_factors=st.session_state.region_importance)

                        visualization[roi_mask == 0] = 0

                        segmentation_mask_argmax = segmentation_mask.argmax(
                            axis=0)
                        segmentation_mask_argmax[roi_mask == 0] = 0
                        segmentation_mask_for_comparisons = segmentation_mask_argmax

                    elif combination_method == "SpectrumHeatmap":
                        segmentation_mask, visualization = model.segment_visualization(
                            img, segmentation_mask, roi_mask, region_colorscheme, method=output_normalization, region_factors=st.session_state.region_importance)

                        visualization[roi_mask == 0] = 0
                        segmentation_mask_for_comparisons = segmentation_mask
                        segmentation_mask_argmax = None

                    certainty_image = None
                    if combination_method != "SpectrumHeatmap" and combination_method not in [
                            "Dim. Reduction UMAP ", "Dim. Reduction NMF", "PercentileRatio"]:
                        certainty_image = get_certainty(segmentation_mask)
                        certainty_image[img.max(axis=-1) == 0] = 0
                        certainty_image = np.uint8(certainty_image * 255)
                        certainty_image = cv2.equalizeHist(certainty_image)
                        certainty_image = np.float32(certainty_image) / 255

                    t0 = time.time()
                    with st.spinner(text="Generating visualization evaluation metrics.."):
                        random.seed(10)
                        metrics = MSIVisualizationMetrics(img, visualization, num_samples=num_samples)
                        fig = metrics.get_correlation_plot()
                        fig2 = metrics.get_correlation_scatter_plot()
                        print("Metrics computation took", time.time() - t0)
                        fig.set_size_inches(5, 2.5)
                        fig2.set_size_inches(5, 2.5)

                    st.session_state.results[path]["certainty_image"] = certainty_image
                    st.session_state.results[path]["segmentation_mask"] = segmentation_mask
                    st.session_state.results[path]["visualization"] = visualization
                    st.session_state.results[path]["correlation_plot"] = (
                        fig, fig2)

                    st.session_state.results[path]["segmentation_mask_for_comparisons"] = segmentation_mask_for_comparisons
                    st.session_state.results[path]["segmentation_mask_argmax"] = segmentation_mask_argmax

                certainty_image = st.session_state.results[path]["certainty_image"]
                segmentation_mask = st.session_state.results[path]["segmentation_mask"]
                visualization = st.session_state.results[path]["visualization"]
                segmentation_mask_for_comparisons = st.session_state.results[
                    path]["segmentation_mask_for_comparisons"]
                segmentation_mask_argmax = st.session_state.results[path]["segmentation_mask_argmax"]
                figures = st.session_state.results[path]["correlation_plot"]

                st.text('Computed visualization')
                point = streamlit_image_coordinates(visualization)

                for fig in figures:
                    st.pyplot(fig)

                num_cols = 2
                with st.container():
                    cols = st.columns(num_cols)

                    if combination_method not in [
                        "SpectrumHeatmap",
                        "Dim. Reduction NMF",
                        "Dim. Reduction UMAP ",
                            "PercentileRatio"]:
                        region_visualization = st.session_state.results[path]["visualization"].copy(
                        )
                        mask = np.uint8(
                            segmentation_mask_argmax == int(region_selectbox)) * 255
                        region_visualization[mask == 0] = 0
                        st.session_state.k_seg = (
                            region_selectbox, region_visualization)

                        cols[0].text(f'Selected Region {region_selectbox}')
                        cols[0].image(region_visualization)

                    if point is not None:
                        if image_to_show in st.session_state.coordinates and point in [
                                p for p in st.session_state.coordinates[image_to_show]]:
                            pass
                        else:
                            st.session_state.coordinates[image_to_show].append(
                                point)
                            st.session_state.coordinates[image_to_show] = st.session_state.coordinates[image_to_show][-2:]

                    if st.session_state.results[path]["certainty_image"] is not None:
                        cols[1].text('Visualization of confidence')
                        certainty = np.uint8(
                            255 * st.session_state.results[path]["certainty_image"])
                        certainty = cv2.applyColorMap(
                            certainty, cmapy.cmap("viridis"))
                        certainty = certainty[:, :, ::-1]
                        certainty[img.max(axis=-1) == 0] = 0
                        cols[1].image(certainty)

    if image_to_show and st.session_state.coordinates and image_to_show in st.session_state.coordinates and input_normalization + \
            image_to_show in st.session_state.results:
        images = []
        visualization = st.session_state.results[input_normalization +
                                                 image_to_show]["visualization"]
        for point in st.session_state.coordinates[image_to_show]:
            x, y = int(point['x']), int(point['y'])
            heatmap = visualizations.get_mask(
                visualization, segmentation_mask_for_comparisons, x, y)
            images.append(heatmap)
        st.session_state.latest_heatmaps = images
        with st.container():
            for index, col in enumerate(st.columns(len(images))):
                col.image(images[index])
            if len(st.session_state.coordinates[image_to_show]) > 1:
                with st.spinner(text="Comparing regions.."):
                    mz_image = st.session_state.results[input_normalization +
                                                        image_to_show]["mz_image"]
                    visualization = st.session_state.results[input_normalization +
                                                             image_to_show]["visualization"]

                    if 'difference_visualizations' not in st.session_state or st.session_state.difference_visualizations is None:
                        st.session_state['difference_visualizations'] = {}

                    if 'current_color_scheme' not in st.session_state:
                        st.session_state['current_color_scheme'] = current_color_scheme

                    if image_to_show not in st.session_state.difference_visualizations or st.session_state[
                            'current_color_scheme'] != current_color_scheme:
                        save_to_cache()
                        diff = visualizations.RegionComparison(
                            mz_image,
                            segmentation_mask_for_comparisons,
                            visualization,
                            mzs=st.session_state.extraction_mzs,
                            bins_per_mz=st.session_state.bins)
                        st.session_state.difference_visualizations[image_to_show] = diff
                    else:
                        diff = st.session_state.difference_visualizations[image_to_show]

                    st.session_state['current_color_scheme'] = current_color_scheme

                    point_a = st.session_state.coordinates[image_to_show][0]
                    point_a = int(point_a['x']), int(point_a['y'])
                    point_b = st.session_state.coordinates[image_to_show][1]
                    point_b = int(point_b['x']), int(point_b['y'])

                    diff_result_key = str(
                        point_a) + str(point_b) + image_to_show + input_normalization
                    if diff_result_key in st.session_state.comparison_results:
                        image, label_a, label_b, (
                            color_a, color_b, aucs), point_aucs = st.session_state.comparison_results[diff_result_key]
                    else:
                        image, label_a, label_b, (color_a, color_b, aucs) = diff.visualize_comparison_between_regions(
                            point_a, point_b)

                        point_aucs = diff.compare_two_points(point_a, point_b)

                        st.session_state.comparison_results[diff_result_key] = image, label_a, label_b, (
                            color_a, color_b, aucs), point_aucs

                    display_mzs(
                        aucs,
                        color_a,
                        color_b,
                        threshold=0.5,
                        title='Comparing Regions')
                    if st.session_state.ion_image:
                        st.image(st.session_state.ion_image[1])
                    display_mzs(
                        point_aucs,
                        color_a,
                        color_b,
                        threshold=0.5,
                        title='Comparing Clicked points')
                    if st.session_state.ion_image:
                        st.image(st.session_state.ion_image[1])

                    st.session_state.latest_diff = (
                        image, visualization, label_a, label_b)
    if image_to_show:
        mz = st.text_input('Create ION image for m/z:')
        if mz:
            create_ion_image(mz, image_to_show)
            st.image(st.session_state.ion_image[1])

    save_data()


except Exception as e:
    print(traceback.format_exc())
    print(e)
