import sys
import json
from msi_visual.supervised.annotations import get_img, get_visualization, get_dataset
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import shap
import numpy as np
from msi_visual.extraction import get_extraction_mz_list
from scipy.stats import mannwhitneyu
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import tqdm
from sklearn.utils.class_weight import compute_sample_weight


if __name__ == "__main__":
    extraction_mzs = get_extraction_mz_list(r"D:\maldi\slides\slide2_notol_5_bin")

    paths = [(r"D:\maldi\slides\slide2_notol_5_bin\0.npy", 1),
            (r"D:\maldi\slides\slide2_notol_5_bin\2.npy", 0),
            (r"D:\maldi\slides\slide2_notol_5_bin\1.npy", 2),
            (r"D:\maldi\slides\slide2_notol_5_bin\3.npy", 3)]



    similarity = {}
    separation_data = {}
    ks = {}

    X_all, y_all, label_encoder_all = get_dataset(r"expAI_NRL4489-s2_json_new6.json", paths, subsample=10, ignore=["HPF", "CTX"])
    categories = label_encoder_all.classes_
    for i in tqdm.tqdm(range(len(categories))):
        category1 = categories[i]
        for j in range(i+1, len(categories)):
            category2 = categories[j]
            a, b = label_encoder_all.transform([category1, category2])
            indices_a = [i for i in range(len(y_all)) if y_all[i] == a]
            indices_b = [i for i in range(len(y_all)) if y_all[i] == b]

            X_all_a = X_all[indices_a, :]
            X_all_b = X_all[indices_b, :]
            X = np.concatenate([X_all_a, X_all_b], axis=0)
            y = [0] * len(X_all_a) + [1] * len(X_all_b)    
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)
            sample_weights = compute_sample_weight(class_weight='balanced', y=y)
            
            xgb_model = XGBClassifier(
                n_estimators=40,
                learning_rate=0.1,
                max_depth=4,
                random_state=42)
            
            xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
            preds = xgb_model.predict(X_test)
            balanced_accuracy = balanced_accuracy_score(y_test, preds)
            print(category1, category2, X_train.shape, len(y_train), sample_weights.shape, np.mean(y_train), np.mean(y_test))
            print(category1, category2, balanced_accuracy, np.mean(y_test), np.mean(preds))
            print("*")

            similarity[(category1, category2)] = 1 - balanced_accuracy
            
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test[::5, :])
            meanabs = np.abs(shap_values).mean(axis=0)
            
            indices = list(np.argsort(meanabs)[-20 : ])
            
            separation_data[(category1, category2)] = []
            for index in indices:
                statistic, pvalue = mannwhitneyu(X_all_a[::5, index], X_all_b[::5, index])
                statistic = statistic / (len(a) * len(b))
                separation_data[(category1, category2)].append((extraction_mzs[index], statistic, pvalue))
                
        result = {"ks": ks, "separation_data": separation_data, "similarity": similarity, "categories": categories}

        with open("graph_data_coarse_model.joblib", "wb") as f:
            joblib.dump(result, f)