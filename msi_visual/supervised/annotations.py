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

def get_annotations(annotation_path, index):
    annotation = json.load(open(annotation_path))
    categories = []
    polygons = []
    for key in annotation:
        if f"{index}_" not in key:
            continue
        regions = annotation[key]["regions"]        
        if len(regions) == 0:
            continue
        for r in regions:
            categories.append(r["region_attributes"]["anatomy"])
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

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(categories)
    return polygons, categories


def get_dataset(annotation_path, all_images, indices=[0, 2], subsample=10, average=False):
    all_polygons = []
    all_categories = []

    for index in indices:
        polygons, categories = get_annotations(annotation_path, index)
        all_polygons.append(polygons)
        all_categories.append(categories)


    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate(all_categories))


    X = []
    y = []
    for data, categories, polygons in zip(all_images, all_categories, all_polygons):
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
                features = rotated_data[mask == label + 1][::subsample, :]
                y.extend([label] * len(features))   
                X.append(features)

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
