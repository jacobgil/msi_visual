import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_extras.image_selector import image_selector 
from msi_visual import visualizations
import math
import cv2
import cmapy
import time
from PIL import Image
from pathlib import Path
import pandas as pd
from msi_visual.normalization import total_ion_count, spatial_total_ion_count
from msi_visual.extraction import get_extraction_mz_list
from msi_visual.visualizations import get_mask
from msi_visual.utils import segment_visualization
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

@dataclass
class ClickData:
    visualiation_path: str
    point: tuple[int, int]
    mask: np.ndarray
    visualization: np.ndarray
    timestamp: float


def get_image(path, bins=5, equalize=False):
    key = "cache" + str(bins) + str(equalize)
    if key not in st.session_state:
        st.session_state[key] = {}

    if path in st.session_state[key]:
        img, segmentation = st.session_state[key][path]
    else:
        img = np.array(Image.open(path))
        if equalize:
            img = cv2.merge([cv2.equalizeHist(img[:, :, 0]), cv2.equalizeHist(img[:, :, 1]), cv2.equalizeHist(img[:, :, 2])])
        segmentation = segment_visualization(img, bins)
        st.session_state[key][path] = img, segmentation
    
    return img, segmentation

def get_data(path):
    if path in st.session_state["data"]:
        return st.session_state["data"][path]
    else:
        img = total_ion_count(np.load(path))
        extraction_mzs = get_extraction_mz_list(Path(path).parent)
        st.session_state["data"][path] = img
        st.session_state["extraction_mzs"][path] = extraction_mzs
        return img


def viewer(folder, bins, equalize, comparison, selection_type):
    if folder:
        csv = pd.read_csv(Path(folder) / "visualization_details.csv")
        selected_visualization = None
        if selection_type == "Polygon":
            selected_visualization = st.selectbox("Visualization", list(csv.visualization.values))
        
        data_paths = set(csv.data.values)

        if len(st.session_state["data"]) == 0:
            with st.spinner('Loading images..'):
                # Fill the cache
                for data_path in data_paths:
                    _ = get_data(data_path)

        visualization_paths = csv.visualization

        if selection_type == "Click":
            N = 2
        else:
            N = 1
        cols = st.columns(N)

        rows = [visualization_paths[i : i + N] for i in range(0, len(visualization_paths), N)]
        for row in rows:        
            for col, path in zip(cols, row):
                if selected_visualization and path != selected_visualization:
                    continue
                visualization, seg = get_image(Path(folder) / path, bins, equalize)

                data_path = csv[csv.visualization==path].data.values[0]
                with col:
                    st.write(path)
                    if selection_type == "Click":
                        point = streamlit_image_coordinates(visualization, key=path+str(st.session_state['id']))
                        if point:
                            x, y = int(point['x']), int(point['y'])
                            last_click_for_visualizaton = None
                            if "last" not in st.session_state:
                                st.session_state["last"] = {}
                            if "last" in st.session_state and path in st.session_state["last"]:
                                last_click_for_visualizaton = st.session_state["last"][path]

                            
                            if  (not last_click_for_visualizaton) or (last_click_for_visualizaton and last_click_for_visualizaton != (x, y)):
                                mask = get_mask(visualization, seg, x, y)
                                st.session_state["last"][path] = (x,y)
                                st.session_state["clicks"][data_path].append(ClickData(path, (x, y), mask, visualization, time.time()))
                            else:
                                print(f"Point exists {x} {y}")
                            
                    elif selection_type == "Polygon":
                        polygon = image_selector(visualization, selection_type="lasso", key=str(st.session_state['id'])+path+'lasso')
                        if polygon and "selection" in polygon and len(polygon["selection"]["lasso"]) > 0:
                            xs = [int(x+0.5) for x in polygon["selection"]["lasso"][0]["x"]]
                            ys = [int(y+0.5) for y in polygon["selection"]["lasso"][0]["y"]]
                            polygon = np.int32(list(zip(xs, ys)))
                            name = str(polygon)
                            existing_clicks = [str(p.point) for p in st.session_state["clicks"][data_path]]
                            if str(polygon) not in existing_clicks:
                                mask = np.zeros(shape=visualization.shape[:2], dtype=np.uint8)
                                mask = cv2.drawContours(mask, [polygon], -1, 255, -1)
                                draw = visualization.copy()
                                draw[mask == 0] = 0
                                st.session_state["clicks"][data_path].append(ClickData(path, polygon, draw, visualization, time.time()))


def point_with_point_comparisons(stats_method):
    for data_path in st.session_state["clicks"]:
        st.session_state["clicks"][data_path] = sorted(st.session_state["clicks"][data_path], key = lambda x: time.time() - x.timestamp)[:2]
        clicks = st.session_state["clicks"][data_path]
        if len(clicks) > 0:
            cols = st.columns(2)
            if len(clicks) == 2:
                data = get_data(data_path)
                extraction_mzs = st.session_state["extraction_mzs"][data_path]
                stats = get_stats_point_to_point(data, extraction_mzs, clicks)
                mask = clicks[0].visualization * 0
                mask = cv2.circle(mask, clicks[0].point, 3, [255, 255, 255], -1)
                mask_a = clicks[0].visualization.copy()
                mask_a[mask == 0] = 0

                mask = clicks[1].visualization * 0
                mask = cv2.circle(mask, clicks[1].point, 3, [255, 255, 255], -1)
                mask_b = clicks[1].visualization.copy()
                mask_b[mask == 0] = 0
                
                cols[0].image(mask_a)
                cols[1].image(mask_b)
                with st.spinner(text=f"Comparing points.."):
                    display_comparison(data_path, stats, data, clicks[0].visualization, clicks[1].visualization, mask_a, mask_b, extraction_mzs, threshold=None)


def region_with_region_comparisons(stats_method):
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
                stats = get_stats(data, extraction_mzs, clicks, stats_method=stats_method)
                mask_a = np.uint8(clicks[0].mask.max(axis=-1)) * 255
                mask_b = np.uint8(clicks[1].mask.max(axis=-1)) * 255
                with st.spinner(text=f"Comparing regions.."):
                    display_comparison(data_path, stats, data, clicks[0].visualization, clicks[1].visualization, mask_a, mask_b, extraction_mzs, threshold=None)


def region_with_all_comparisons(stats_method):
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
        stats = get_stats(data, extraction_mzs, [clicks[0], next_click], stats_method=stats_method)
        mask_a = np.uint8(mask_a.max(axis=-1)) * 255
        mask_b = np.uint8(mask_b.max(axis=-1)) * 255
        with st.spinner(text=f"Comparing regions.."):
            display_comparison(data_path,
                stats,
                data,
                clicks[0].visualization,
                next_click.visualization, mask_a, mask_b, extraction_mzs, threshold=None)

def show_ion_images(mzs):
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
            aggregated = aggregated.transpose().transpose(1, 2, 0)[::-1, :, :]
            st.image(aggregated)


def show_mz_on_ion_image(ion, mz):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5, 20)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 1
    ion = cv2.putText(ion, f"{mz:.3f}",
                         bottomLeftCornerOfText,
                         font,
                         fontScale,
                         fontColor,
                         thickness,
                         lineType)
    return ion

def get_closest_mz(mz, extraction_mzs):
    mz_index = np.abs(np.float32(extraction_mzs) - float(mz)).argmin()
    return extraction_mzs[mz_index], mz_index

def get_raw_ion_image(img, mz, extraction_mzs):
    closest_mz, mz_index = get_closest_mz(mz, extraction_mzs)
    ion = visualizations.create_ion_img(img, mz_index)
    return ion, closest_mz

def create_ion_image(img, mz, extraction_mzs):
    closest_mz, mz_index = get_closest_mz(mz, extraction_mzs)
    ion = visualizations.create_ion_heatmap(img, mz_index)
    ion = show_mz_on_ion_image(ion, closest_mz)
    print("setting ion", mz, closest_mz)
    st.session_state.ion = ion
    return ion, closest_mz

def get_stats_point_to_point(data, extraction_mzs, clicks):
    visualization_path_a, visualization_path_b = clicks[0].visualiation_path, clicks[1].visualiation_path,
    point_a, point_b = clicks[0].point, clicks[1].point
    key = visualization_path_a + visualization_path_b + str(point_a) + str(point_b) + "p2p"
    if 'stats' not in st.session_state:
        st.session_state['stats'] = {}
        
    if key in st.session_state['stats']:
        stats = st.session_state['stats'][key]
    else:
        comparison = visualizations.RegionComparison(
            data,
            mzs=extraction_mzs)
        stats = comparison.compare_two_points(point_a, point_b)
        st.session_state['stats'][key] = stats
    return stats
        


def get_stats(data, extraction_mzs, clicks, stats_method="U-Test"):
    visualization_path_a, visualization_path_b = clicks[0].visualiation_path, clicks[1].visualiation_path,
    mask_a = np.uint8(clicks[0].mask.max(axis=-1)) * 255
    mask_b = np.uint8(clicks[1].mask.max(axis=-1)) * 255
    point_a, point_b = clicks[0].point, clicks[1].point
    key = visualization_path_a + visualization_path_b + str(point_a) + str(point_b)
    if 'stats' not in st.session_state:
        st.session_state['stats'] = {}
        
    if key in st.session_state['stats']:
        stats = st.session_state['stats'][key]
    else:
        comparison = visualizations.RegionComparison(
            data,
            mzs=extraction_mzs)
        stats = comparison.ranking_comparison(mask_a, mask_b, method=stats_method)
        st.session_state['stats'][key] = stats
    return stats

def get_2d_spectrum(
        scores,
        extraction_mzs,
        color_scheme="cividis"):
    qr_images = []
    qr_cols, qr_rows = int(1 + (len(extraction_mzs))** 0.5), int(1 + (len(extraction_mzs))** 0.5)
    qr_cell_size = 10
    color = cv2.applyColorMap(255*np.uint8(np.ones((qr_cell_size, qr_cell_size))), cmapy.cmap(color_scheme))

    qr_image = np.zeros(
        (qr_rows *
            qr_cell_size +
            qr_cell_size,
            qr_cols *
            qr_cell_size +
            qr_cell_size,
            3),
        dtype=np.uint8)
    color = cv2.applyColorMap(0*np.uint8(np.ones((1, 1))), cmapy.cmap(color_scheme))
    qr_image[:, :, :] = color
    for mz, score in scores.items():
        mz_index = extraction_mzs.index(mz)
        score_for_display = score**4
        lab = cv2.cvtColor(color, cv2.COLOR_RGB2Lab)
        l, a, b = cv2.split(lab)
        l = np.uint8(l * score_for_display)
        lab = cv2.merge([l, a, b])
        color = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

        color = cv2.applyColorMap(np.uint8(255*score_for_display*np.ones((qr_cell_size, qr_cell_size))), cmapy.cmap(color_scheme))
        row, col = int(mz_index / qr_rows), int(mz_index % qr_cols)
        qr_image[row *
                    qr_cell_size: row *
                    qr_cell_size +
                    qr_cell_size, col *
                    qr_cell_size: col *
                    qr_cell_size +
                    qr_cell_size, :] = color

    fig = plt.figure()
    plt.rcParams['text.color'] = 'white'
    fig.patch.set_facecolor("#01224d")
    qr_cols, qr_rows = int(1 + (len(extraction_mzs))** 0.5), int(1 + (len(extraction_mzs))** 0.5)
    qr_cell_size = 10
    spec = qr_image[:, :, ::-1]
    plt.imshow(spec)
    plt.annotate(f"m/z {min(extraction_mzs):.1f}", (0, 20))
    plt.annotate(f"m/z {max(extraction_mzs)}", (0.8 *spec.shape[1], 0.95*spec.shape[0]))
    plt.tight_layout()
    plt.axis('off')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[20:-55, 110:-110]
    return data

def get_aggregated_ion_image(stats, data, extraction_mzs):
    mzs = list(stats.keys())
    mz_indices = [extraction_mzs.index(mz) for mz in mzs]
    scores = np.float32([stats[mz] for mz in mzs])
    scores = np.float32([stats[mz] for mz in mzs])
    mz_indices = np.int32([extraction_mzs.index(mz) for mz in mzs])
    mz_indices = mz_indices[np.argsort(scores)[-5 : ]]

    top5 = data / np.max(data, axis=(0, 1))
    top5 = top5[:, :, mz_indices].mean(axis=-1)
    top5 = top5 / np.percentile(top5, 99)
    top5[top5 > 1] = 1
    top5 = np.uint8(255 * top5)
    top5 = cv2.applyColorMap(top5, cmapy.cmap("cividis"))[:, :, ::-1]
    return top5


def get_mz_value_img(stats, height, top=5):
    mzs = list(stats.keys())
    mzs = sorted(mzs, key = lambda x: stats[x])[-top :][::-1]    
    cell = np.zeros((1*128, 1*128, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    loc = (cell.shape[1]//2-50, cell.shape[0]//2)
    fontScale = 1.0
    fontColor = (255, 255, 255)
    lineType = 1
    thickness = 3
    cells = []
    for mz in mzs:
        cell = np.zeros((1*128, 1*128, 3), dtype=np.uint8)
        cell = cv2.putText(cell, f"{mz:.1f}",
                            loc,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
        cells.append(cell)
    result = np.vstack(cells)
    
    result = cv2.resize(result, (height//top, height))
    return result

def display_aggregated_ion_image(stats, data, extraction_mzs, color_scheme="cividis"):
    img = get_aggregated_ion_image(stats, data, extraction_mzs)
    img = img.transpose().transpose(1, 2, 0)[::-1, :, :]
    img = cv2.resize(img, (8*img.shape[1], 8*img.shape[0]))
    spectrum = get_2d_spectrum(
        stats,
        extraction_mzs,
        color_scheme="cividis")
    spectrum = cv2.resize(spectrum, (img.shape[1], img.shape[0]))
    mz_list = get_mz_value_img(stats, img.shape[0])
    mz_list = cv2.resize(mz_list, (mz_list.shape[1], img.shape[0]))
    img1 = np.hstack((img, spectrum, mz_list))

    return img1

def display_comparison(data_path, stats, data, visualization_a, visualization_b, mask_a, mask_b, extraction_mzs, threshold=1.0):
    key = "6"+data_path + ''.join([str(mz) + str(stats[mz]) for mz in stats])
    if key not in st.session_state:
        img1 = display_aggregated_ion_image(stats, data, extraction_mzs)
        reverse_stats = {mz: 1-stats[mz] for mz in stats}
        img2 = display_aggregated_ion_image(reverse_stats, data, extraction_mzs)
        st.session_state[key] = img1, img2
    else:
        img1, img2 = st.session_state[key]
    st.image(img1)
    st.image(img2)


    color_a = visualization_a[mask_a > 0].mean(axis=0)
    color_b = visualization_b[mask_b > 0].mean(axis=0)
    display_mzs(data, stats, color_a, color_b, extraction_mzs, threshold=threshold)

def display_mzs(img, stats, color_a, color_b, extraction_mzs, threshold=1.0, num_mzs=50):
    
    if threshold is not None:
        pos_aucs = {mz: stats[mz]
                    for mz in stats if stats[mz] > threshold}
        neg_aucs = {
            mz: stats[mz] for mz in stats if stats[mz] < threshold}
    else:
        pos_aucs = {mz: stats[mz]
                    for mz in stats}
        neg_aucs = {
            mz: stats[mz] for mz in stats}

    pos_aucs_keys = sorted(list(pos_aucs.keys()),
                           key=lambda x: pos_aucs[x])[-num_mzs:]
    neg_aucs_keys = sorted(list(neg_aucs.keys()),
                           key=lambda x: neg_aucs[x])[:num_mzs]

    pos_aucs = {mz: pos_aucs[mz] for mz in pos_aucs_keys}
    neg_aucs = {mz: neg_aucs[mz] for mz in neg_aucs_keys}

    for auc_group, color in zip([pos_aucs, neg_aucs], [color_a, color_b]):
        header_cols = st.columns(2)
        header_cols[0].text(f'm/z values for ')
        header_cols[1].image(np.uint8(color * np.ones((64, 64, 3))))
        sorted_indices = sorted(
            list(
                auc_group.keys()),
            key=lambda mz: stats[mz],
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
                        key=f"{time.time()}+default+{i}+{row}+{col}+{mz}",
                        disabled=True,
                        use_container_width=True)
                else:
                    col.button(
                        f"{mz}: {stats[mz]:.3f}",
                        use_container_width=True,
                        key=f"{time.time()}+{mz}+{i}+{row}+{col}",
                        on_click=create_ion_image,
                        args=[img,
                            mz,
                            extraction_mzs])
        
        mzs = ",".join([str(mz) for mz in sorted_indices])
        st.write(mzs)

            
    with st.sidebar:
        if "ion" in st.session_state:
            st.image(st.session_state.ion)
