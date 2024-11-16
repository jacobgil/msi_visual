import json
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from msi_visual.normalization import total_ion_count, spatial_total_ion_count
import numpy as np
from PIL import Image
import cv2
import xgboost as xgb
import cv2
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from functools import lru_cache


@lru_cache(maxsize=1000)
def parse_category(category):
    category = category.strip().replace("\n", "")
    if category == '':
        category = 'artefact'
    return category

@lru_cache(maxsize=1000)
def parse_parent_category(category):
    category = category.strip().replace("\n", "")
    if category == "HPC":
        category = "HPF"
    return category

@lru_cache(maxsize=1000)
def get_annotations(annotation_path, index, ignore=[], keep=None, category_key="Allen atlas anatomy"):
    annotation = json.load(open(annotation_path))
    
    if "_via_img_metadata" in annotation:
        annotation = annotation["_via_img_metadata"]

    categories = []
    polygons = []
    
    for key in annotation:
        
        if not key.startswith(f"{index}_"):
            continue
        

        regions = annotation[key]["regions"]        
        if len(regions) == 0:
            continue
        for r in regions:
            category = r["region_attributes"][category_key].upper()
            if category_key == "category":
                category = parse_parent_category(category)
            else:
                category = parse_category(category)

            if category == "PLAQUE" and "2_PercentileRatio_eq" not in key:
                continue

            if category in ignore:
                continue

            if category_key == "Allen atlas anatomy":
                parent_cagory = parse_parent_category(r["region_attributes"]["category"].upper())
                category = parent_cagory + "_" + category

            if keep is not None and category not in keep:
                continue

            categories.append(category)
            if r['shape_attributes']['name'] == 'ellipse':
                x, y, r = r['shape_attributes']['cx'], r['shape_attributes']['cy'], r['shape_attributes']['rx']
                t = np.linspace(0, 2*np.pi, 50)
                circle_x = x + r * np.cos(t)
                circle_y = y + r * np.sin(t)
                polygon = np.array(list(zip(circle_x, circle_y)))[:, None, :]
            else:
                xs = r['shape_attributes']['all_points_x']
                ys = r['shape_attributes']['all_points_y']
                polygon = np.array(list(zip(xs, ys)))[:, None, :]
            polygons.append(polygon)

    return polygons, categories


def get_dataset(annotation_path, paths, subsample=10, average=False, ignore=[], keep=None, category_key="Allen atlas anatomy"):
    all_polygons = []
    all_categories = []
    X, y = [], []

    for path, index in paths:
        annotation_index = path.split("\\")[-1].split(".")[0]    
        polygons, categories = get_annotations(annotation_path,
                                               annotation_index,
                                               ignore=ignore,
                                               keep=keep,
                                               category_key=category_key)
        all_categories.append(categories)

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate(all_categories))

    for path, index in paths:
        annotation_index = path.split("\\")[-1].split(".")[0]    
        polygons, categories = get_annotations(annotation_path,
                                               annotation_index,
                                               ignore=ignore,
                                               keep=keep,
                                               category_key=category_key)
        indices = [i for i in range(len(polygons))]
        indices = sorted(indices, key=lambda i: cv2.contourArea(np.int32(polygons[i])), reverse=True)
        polygons = [polygons[i] for i in indices]
        categories = [categories[i] for i in indices]
        data = get_img(path)
        rotated_data = data.transpose().transpose(1, 2, 0)[::-1, :, :]

        mask = np.zeros(rotated_data.shape[:2], dtype=np.uint8)
        labels = label_encoder.transform(categories)
        for polygon, label in zip(polygons, labels):
            polygon = np.int32(polygon)
            cv2.drawContours(mask, [polygon], -1, int(label) + 1, -1)
        
        for label in labels:
            
            if average:
                features = rotated_data[mask == label + 1].mean(axis=0)
                X.append(features)
                y.append(label)
            else:
                features = rotated_data[mask == label + 1][::subsample, :].copy()
                y.extend([label] * len(features))   
                X.append(features)
        del data, rotated_data

    if average:
        X = np.array(X)
    else:
        X = np.concatenate(X)
    return X, y, label_encoder


def get_img(path):
    img = np.load(path)
    img = total_ion_count(img)
    return img

def get_visualization(path=r"C:\Users\Jacob Gildenblat\Desktop\maldi\app\visualizations\0_SaliencyOptimization_num_epochs_300regularization_strength_0.01sampling_coresetnumber_of_points_1000.png"):
    viz = Image.open(path)
    viz = np.array(viz)
    viz = cv2.merge([cv2.equalizeHist(viz[:,:,i]) for i in range(3)])
    return viz
