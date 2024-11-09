import sys
import json
from msi_visual.supervised.annotations import get_img, get_visualization, get_dataset
from xgboost import XGBClassifier
import joblib

if __name__ == "__main__":
    path1 = r"D:\maldi\slides\slide2_notol_5_bin\0.npy"
    path2 = r"D:\maldi\slides\slide2_notol_5_bin\2.npy"
    img = get_img(path1)
    img2 = get_img(path2)
    all_images = [img, img2]
    X, y, label_encoder = get_dataset(r"expAI_NRL4489-s2_json.json", all_images, subsample=10)



    # Initialize and train XGBoost classifier
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        max_delta_step=1,  # Helps with class imbalance
        min_child_weight=5  # Helps prevent overfitting on minority classes
    )


    xgb_model.fit(X, y)

    joblib.dump(xgb_model, "xgb_model.pkl")