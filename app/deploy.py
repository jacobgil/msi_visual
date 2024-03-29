import streamlit as st
import glob
import os
import traceback
import datetime
import json
import sys
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
import wx
import cv2
import msi_visual
importlib.reload(msi_visual)
from msi_visual import nmf_segmentation
from msi_visual import visualizations
from msi_visual import objects
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder
from msi_visual.utils import get_certainty, set_region_importance
from msi_visual import parametric_umap
from msi_visual.umap_nmf_segmentation import SegmentationUMAPVisualization
from msi_visual.avgmz_nmf_segmentation import SegmentationAvgMZVisualization
from msi_visual.avgmz import AvgMZVisualization
importlib.reload(visualizations)
import hashlib
import base64

def make_hash_sha256(o):
    hasher = hashlib.md5()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.urlsafe_b64encode(hasher.digest()).decode()

def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k,make_hashable(v)) for k,v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o


def get_settings():
    return {'Extraction Root Folder': extraction_root_folder,
     'Extraction folder': selected_extraction,
     'Slide ROI to include': regions,
     'Model folder': nmf_model_folder,
     'confidence_thresholds': st.session_state.confidence_thresholds,
     'UMAP Model folder (optional)': umap_model_folder,
     'combination_method': combination_method,
     'Segmentation model': nmf_model_name,
     'Image to show': image_to_show,
     'output_normalization': output_normalization,
     'color_schemes': st.session_state.color_schemes,
     'region_importance': st.session_state.region_importance,
     'current_color_scheme': current_color_scheme}

def get_settings_hash():
    settings = get_settings()
    return make_hash_sha256(settings)

def model_hash():
    settings = {'umap_model_folder': st.session_state.umap_model_folder, 
                'nmf_model_folder': st.session_state.nmf_model_folder}
    return make_hash_sha256(settings)

def get_model():
    if nmf_model_name:
        if st.session_state['segmentation_model'] is None:
            nmf_model_path = [p for p in nmf_model_paths if Path(p).stem == nmf_model_name][0]
            model = joblib.load(open(nmf_model_path, 'rb'))
            st.session_state['segmentation_model'] = model
        else:
            model = st.session_state['segmentation_model']
    
    if combination_method == "Seg+UMAP":
        if umap_model_folder:
            if st.session_state['model'][combination_method] is None:
                umap = parametric_umap.UMAPVirtualStain()
                umap.load(umap_model_folder)        
                model = SegmentationUMAPVisualization(umap, model)
            else:
                model = st.session_state['model'][combination_method]
        else:
            model = None
    
    elif combination_method == "Seg+SpectrumHeatmap":
        if st.session_state['model'][combination_method] is None:
            model = SegmentationAvgMZVisualization(model)
        else:
            model = st.session_state['model'][combination_method]
    elif combination_method == "SpectrumHeatmap":
        if st.session_state['model'][combination_method] is not None:
            model = AvgMZVisualization()
        else:
            model = st.session_state['model'][combination_method]
    
    if model is not None:
        st.session_state['model'][combination_method] = model
    return model

def save_data():
    if image_to_show:
        folder = datetime.datetime.today().strftime('%Y-%m-%d')
        os.makedirs(folder, exist_ok=True)
        image_name = Path(image_to_show).stem
        prefix = combination_method
        now = datetime.datetime.now()
        prefix = get_settings_hash() + "_" + prefix

        if 'latest_diff' in st.session_state:
            img, visualization, label_a, label_b = st.session_state.latest_diff
            Image.fromarray(img).save(Path(folder) / f"{image_name}_{label_a}_{label_b}_{prefix}.png")

            if 'latest_heatmaps' in st.session_state:
                for index, heatmap in enumerate(st.session_state.latest_heatmaps):
                    Image.fromarray(heatmap).save(Path(folder) / f"{image_name}_{[label_a, label_b][index]}_{prefix}_mask.png")
        else:
            visualization = st.session_state.results[image_to_show]["visualization"]
        Image.fromarray(visualization).save(Path(folder) / f"{image_name}_{prefix}.png")    

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

if 'umap_model_folder' not in st.session_state:
    st.session_state.umap_model_folder = None
if 'nmf_model_folder' not in st.session_state:
    st.session_state.nmf_model_folder = None

if 'color_schemes' not in st.session_state:
    st.session_state.color_schemes = cached_state['color_schemes']
if st.session_state.color_schemes == '':
    st.session_state.color_schemes = ["gist_yarg"] * 100
if 'region_importance' not in st.session_state:
    st.session_state.region_importance = {}
if 'model' not in st.session_state:
    st.session_state.model = {'Seg+SpectrumHeatmap': None,
                              'Seg+UMAP': None,
                              "Segmentation": None,
                              "SpectrumHeatmap": None}
    
    st.session_state.segmentation_model = None

if 'setting_hash' not in st.session_state:
    st.session_state.setting_hash = None

if 'model_hash' not in st.session_state:
    st.session_state.model_hash = None


results = {}
image_to_show = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = 0
if 'bins' not in st.session_state:
    st.session_state.bins = 5    

if 'coordinates' not in st.session_state or st.session_state.coordinates is None:
    st.session_state.coordinates = defaultdict(list)

try:
    umap_model_folder, nmf_model_path, nmf_model_name = None, None, None
    region_colorscheme = None
    with st.sidebar:
        extraction_root_folder = st.text_input('Extraction Root Folder', value=cached_state["Extraction Root Folder"])
        if extraction_root_folder:
            extraction_folders = display_paths_to_extraction_paths(extraction_root_folder)
            extraction_folders_keys = list(extraction_folders.keys())
            if cached_state['Extraction folder'] == '':
                cached_state['Extraction folder'] = extraction_folders_keys[0]
            selected_extraction = st.selectbox('Extraction folder', extraction_folders_keys, index=extraction_folders_keys.index(cached_state['Extraction folder']))
            if selected_extraction:
                extraction_folder = extraction_folders[selected_extraction]
                region_list = get_files_from_folder(extraction_folder)
                if cached_state['Regions to include'] == '':
                    default=None
                else:
                    default = cached_state['Regions to include']
                    if False in [r in region_list for r in cached_state['Regions to include'] ]:
                        default = None

                regions = st.multiselect('Regions to include', region_list,  default=default)
        
        st.divider()
        
        nmf_model_folder = st.text_input('Model folder', value=cached_state['Model folder'])
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
            nmf_model_name = st.selectbox('Segmentation model', nmf_model_display_paths, index=default)
            st.session_state.nmf_model_path = nmf_model_path

        combination_method = st.radio('Color Coding Method',
                                    ["Segmentation", "Seg+UMAP", "Seg+SpectrumHeatmap", "SpectrumHeatmap"], index=0)
        
        #if combination_method == "Seg+UMAP":
        umap_model_folder = st.text_input('UMAP Model folder (optional)', value=cached_state['UMAP Model folder (optional)'])
        st.session_state.umap_model_folder = umap_model_folder
        output_normalization = st.selectbox('Segmentation Output Normalization', ['spatial_norm', 'None'])
        sub_sample = st.number_input('Subsample pixels', value=None)
        
        #objects_mode = st.checkbox('Objects mode')
        #objects_window = st.selectbox('Objects window size', [3, 6, 9, 12])
        objects_mode = False

    with st.sidebar:
        image_to_show = st.selectbox('Image to show', list(regions), key = st.session_state.run_id)

    if image_to_show is not None:
        if model_hash() != st.session_state.model_hash:
            st.session_state.model_hash = model_hash

        model = get_model()
        st.write(combination_method)

    colorschemes = list(colormaps)
    with st.sidebar:
        st.divider()
        if combination_method != "Spectrum-Heatmap":
                region_selectbox = 0
                if image_to_show:
                    region_selectbox = st.selectbox(f"k-segment to control", [i for i in range(model.k)], index=0)
                
                
                region_default = 1.0
                if region_selectbox in st.session_state.region_importance:
                    region_default = st.session_state.region_importance[int(region_selectbox)]
                region_factor = st.slider(label=f'Weight of k-segment ({region_selectbox})',
                                        min_value=0.0,
                                        max_value=4.0,
                                        value=region_default,
                                        step=0.1)


                if combination_method != "Segmentation":
                    default = colorschemes.index(st.session_state.color_schemes[0])
                    if st.session_state.color_schemes != '' and len(st.session_state.color_schemes) > int(region_selectbox):
                        default = colorschemes.index(st.session_state.color_schemes[int(region_selectbox)])
                    region_colorscheme = st.selectbox(f"Color Scheme k-segment ({region_selectbox})", colorschemes, index=default)

                    if st.button("Export color scheme"):
                        dialog = wx.FileDialog(None, "Color scheme file location", style=wx.DD_DEFAULT_STYLE)
                        if dialog.ShowModal() == wx.ID_OK:
                            export_path = dialog.GetPath() # folder_path will contain the path of the folder you have selected as
                            with open(export_path, 'w') as f:
                                data_to_save = {'color_schemes': st.session_state.color_schemes,
                                                'region_weights': st.session_state.region_importance}
                                json.dump(data_to_save, f)

                    if st.button("Import color scheme"):
                        dialog = wx.FileDialog(None, "Color scheme file location", style=wx.DD_DEFAULT_STYLE)
                        if dialog.ShowModal() == wx.ID_OK:
                            export_path = dialog.GetPath() # folder_path will contain the path of the folder you have selected as
                            with open(export_path, 'r') as f:
                                loaded_data = json.load(f)
                                if 'color_schemes' in loaded_data:
                                    st.session_state.color_schemes = loaded_data['color_schemes']
                                    st.session_state.region_importance = loaded_data['region_weights']
                                else:
                                    st.session_state.color_schemes = loaded_data
                    
                    current_color_scheme = str([str(x) for x in st.session_state.color_schemes])

                else:
                    region_colorscheme = st.selectbox(f"Color Scheme", colorschemes, index = colorschemes.index("gist_rainbow"))
                    current_color_scheme = region_colorscheme

                col1, col2 = st.columns(2)

                with col1:
                    button1=st.button('Re-set')

                with col2:
                    button2=st.button('Update Region Settings')                    
                    
                if button1:
                    st.session_state.color_schemes = ["gist_yarg"] * 100
                    st.session_state.region_importance = {}

                if button2:
                    st.session_state.color_schemes[int(region_selectbox)] = region_colorscheme
                    st.session_state.region_importance[int(region_selectbox)] = region_factor

                st.divider()
                certainty_slider = st.slider('Confidence threshold range', 0.0, 1.0, (0.0, 1.0))
                st.session_state['confidence_thresholds'] = certainty_slider
        else:
            region_colorscheme = st.selectbox(f"Color Scheme", colorschemes, index = colorschemes.index("gist_rainbow"))
            current_color_scheme = region_colorscheme


    start = st.button("Run")

    if start:
        st.session_state.setting_hash = None
        save_to_cache()
        st.session_state.run_id = st.session_state.run_id + 1

        with st.spinner(text="Running segmentation.."):
            st.session_state.coordinates = None
            st.session_state.results = None
            st.session_state.difference_visualizations = None
            extraction_args = eval(open(Path(extraction_folder) / "args.txt").read())
            st.session_state.bins = extraction_args.bins
            st.session_state.extraction_start_mz = extraction_args.start_mz
            st.session_state.extraction_end_mz = extraction_args.end_mz

            if 'results' in st.session_state:
                del st.session_state.results

            st.session_state.results = {}
            for path in regions:
                if sub_sample:
                    img = np.load(path)[::int(sub_sample), ::int(sub_sample), :]
                else:
                    img = np.load(path)
                img = np.float32(img)
                
                results = {}
                results["mz_image"] = img

                if combination_method in ["Seg+UMAP", "Seg+SpectrumHeatmap"]:
                    results["segmentation"], results["heatmap"] = model.factorize(img)
                else:
                    results["segmentation"] = model.factorize(img)
                
                st.session_state.results[path] = results

    if 'results' in st.session_state:
        for path, data in st.session_state.results.items():
            if path == image_to_show:
                img = data["mz_image"]
                segmentation_mask = data["segmentation"].copy()     
                
                if st.session_state.setting_hash != get_settings_hash():
                    st.session_state.setting_hash = get_settings_hash()

                    roi_mask = np.uint8(img.max(axis=-1) > 0)
                    if 'confidence_thresholds' in st.session_state:
                        low, high = st.session_state['confidence_thresholds']
                        if 'certainty_image' in st.session_state.results[path]:
                            if st.session_state.results[path]['certainty_image'] is not None:
                                certainty_image = st.session_state.results[path]["certainty_image"]
                                roi_mask[certainty_image < low] = 0
                                roi_mask[certainty_image > high] = 0

                    if combination_method in ["Seg+UMAP", "Seg+SpectrumHeatmap"]:
                        segmentation_mask, sub_segmentation_mask, visualization = model.visualize_factorization(img,
                                                                        segmentation_mask,
                                                                        data["heatmap"],
                                                                        roi_mask,
                                                                        color_scheme_per_region=st.session_state.color_schemes,
                                                                        method=output_normalization,
                                                                        region_factors=st.session_state.region_importance)
                        segmentation_mask_argmax = segmentation_mask.argmax(axis=0)
                        segmentation_mask_for_comparisons = sub_segmentation_mask
                    elif combination_method == "Segmentation":
                        segmentation_mask, visualization = model.visualize_factorization(img,
                                                                                        segmentation_mask,
                                                                                        region_colorscheme,
                                                                                        method=output_normalization,
                                                                                        region_factors=st.session_state.region_importance)
                        
                        visualization[roi_mask == 0] = 0
                        segmentation_mask_argmax = segmentation_mask.argmax(axis=0)
                        segmentation_mask_argmax[roi_mask == 0] = 0
                        segmentation_mask_for_comparisons = segmentation_mask_argmax
                    elif combination_method == "SpectrumHeatmap":
                        segmentation_mask, visualization = model.visualize_factorization(img,
                                                                                        segmentation_mask,
                                                                                        roi_mask,
                                                                                        region_colorscheme,
                                                                                        method=output_normalization,
                                                                                        region_factors=st.session_state.region_importance)
                        
                        visualization[roi_mask == 0] = 0
                        segmentation_mask_for_comparisons = segmentation_mask
                        segmentation_mask_argmax = None

                    certainty_image= None
                    if combination_method != "SpectrumHeatmap":
                        certainty_image = get_certainty(segmentation_mask)
                        #certainty_image = segmentation_mask.max(axis=0)
                        certainty_image[img.max(axis=-1) == 0] = 0
                        certainty_image = np.uint8(certainty_image * 255)
                        certainty_image = cv2.equalizeHist(certainty_image)
                        certainty_image = np.float32(certainty_image) / 255

                    st.session_state.results[path]["certainty_image"] = certainty_image
                    st.session_state.results[path]["segmentation_mask"] = segmentation_mask
                    st.session_state.results[path]["visualization"] = visualization
                    st.session_state.results[path]["segmentation_mask_for_comparisons"] = segmentation_mask_for_comparisons
                    st.session_state.results[path]["segmentation_mask_argmax"] = segmentation_mask_argmax

                certainty_image = st.session_state.results[path]["certainty_image"]
                segmentation_mask = st.session_state.results[path]["segmentation_mask"]
                visualization = st.session_state.results[path]["visualization"]
                segmentation_mask_for_comparisons = st.session_state.results[path]["segmentation_mask_for_comparisons"]
                segmentation_mask_argmax = st.session_state.results[path]["segmentation_mask_argmax"]

                if objects_mode:
                    with st.spinner(text="Detecting objects.."):

                        key = str(objects_window) + "_" + path
                        if 'object_mask' in st.session_state and key in st.session_state.object_mask:
                            object_mask = st.session_state.object_mask[key]
                        else:
                            if 'object_mask' not in st.session_state:
                                st.session_state.object_mask = {}
                                
                            detector = objects.ObjectDetector(size=int(objects_window))
                            object_mask = detector.get_mask(img)
                            st.session_state.object_mask[key] = object_mask

                        segmentation_mask[object_mask == 0] = -1
                        visualization[object_mask == 0] = 0
                point = streamlit_image_coordinates(st.session_state.results[path]["visualization"])
                num_cols = 2

                if combination_method != "SpectrumHeatmap":
                    with st.container():
                        cols = st.columns(num_cols)

                        region_visualization = st.session_state.results[path]["visualization"].copy()                    
                        mask = np.uint8(segmentation_mask_argmax == int(region_selectbox)) * 255
                        region_visualization[mask == 0] = 0
                        cols[0].text('Selected k-segment')
                        cols[0].image(region_visualization)

                        if point is not None:
                            if path in st.session_state.coordinates and point in [p for p in st.session_state.coordinates[path]]:
                                pass
                            else:
                                st.session_state.coordinates[path].append(point)
                                st.session_state.coordinates[path] = st.session_state.coordinates[path][-2 : ]

                        if st.session_state.results[path]["certainty_image"] is not None:
                            cols[1].text('Visualization of confidence')
                            cols[1].image(np.uint8(255*st.session_state.results[path]["certainty_image"]))

    if image_to_show and st.session_state.coordinates and image_to_show in st.session_state.coordinates:
        images = []
        visualization = st.session_state.results[image_to_show]["visualization"]
        for point in st.session_state.coordinates[image_to_show]:
            x, y = int(point['x']), int(point['y'])
            heatmap = visualizations.get_mask(visualization, segmentation_mask_for_comparisons, x, y)
            images.append(heatmap)
        st.session_state.latest_heatmaps = images
        with st.container():
            for index, col in enumerate(st.columns(len(images))):
                col.image(images[index])
            if (objects_mode and len(st.session_state.coordinates[image_to_show]) > 0) or \
                ((not objects_mode) and len(st.session_state.coordinates[image_to_show]) > 1):
                    with st.spinner(text="Comparing regions.."):
                        mz_image = st.session_state.results[image_to_show]["mz_image"]
                        visualization = st.session_state.results[image_to_show]["visualization"]

                        if 'difference_visualizations' not in st.session_state or st.session_state.difference_visualizations is None:
                            st.session_state['difference_visualizations'] = {}

                        if 'current_color_scheme' not in st.session_state:
                            st.session_state['current_color_scheme'] = current_color_scheme

                        if image_to_show not in st.session_state.difference_visualizations or st.session_state['current_color_scheme'] != current_color_scheme:
                            save_to_cache()
                            diff = visualizations.RegionComparison(mz_image,
                                                                segmentation_mask_for_comparisons,
                                                                visualization,
                                                                start_mz=st.session_state.extraction_start_mz,
                                                                bins_per_mz=st.session_state.bins)
                            st.session_state.difference_visualizations[image_to_show] = diff
                        else:
                            diff = st.session_state.difference_visualizations[image_to_show]

                        st.session_state['current_color_scheme'] = current_color_scheme
                        
                        if objects_mode:
                            point = st.session_state.coordinates[image_to_show][-1]
                            point = int(point['x']), int(point['y'])
                            image = diff.visualize_object_comparison(point, size=objects_window)
                            label_a, label_b = None, None
                            st.image(image)
                        else:
                            point_a = st.session_state.coordinates[image_to_show][0]
                            point_a = int(point_a['x']), int(point_a['y'])
                            point_b = st.session_state.coordinates[image_to_show][1]
                            point_b = int(point_b['x']), int(point_b['y'])
                            image, label_a, label_b = diff.visualize_comparison_between_points(point_a, point_b)
                            st.image(image)
                        st.session_state.latest_diff = (image, visualization, label_a, label_b)
        save_data()

    if image_to_show:
        mz = st.text_input('Create ION image for m/z:')
        if mz:
            mz_image = st.session_state.results[image_to_show]["mz_image"]
            val = int( (float(mz)-st.session_state.extraction_start_mz) * st.session_state.bins)
            st.image(visualizations.create_ion_image(mz_image, val))
except Exception as e:
    print(e)
    