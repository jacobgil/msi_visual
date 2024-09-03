import streamlit as st
import joblib
import os
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import pandas as pd
import time
from msi_visual.app.utils.pipeline import create_pipeline
from msi_visual.app.utils.extraction import get_extraction
import wx
app = wx.App()
wx.DisableAsserts()




extraction_tab, pipeline_tab = st.tabs(["Data", "Visualizations"])

with extraction_tab:
    regions = get_extraction()

with pipeline_tab:
    models = []

    if os.path.exists("pipeline.cache"):
        models = joblib.load("pipeline.cache")

    cols = st.columns(2)
    with cols[0]:
        if st.button("Load"):
            dialog = wx.FileDialog(
                None, "Settings file", style=wx.DD_DEFAULT_STYLE)
            if dialog.ShowModal() == wx.ID_OK:
                # folder_path will contain the path of the folder you
                # have selected as
                import_path = dialog.GetPath()
                models = joblib.load(import_path)

    with cols[1]:
        if st.button("Save"):
            dialog = wx.FileDialog(
                None, "Settings file", style=wx.DD_DEFAULT_STYLE)
            if dialog.ShowModal() == wx.ID_OK:
                # folder_path will contain the path of the folder you
                # have selected as
                export_path = dialog.GetPath()
                joblib.dump(models, export_path)

    model, normalization, output_path = create_pipeline()
    if model:
        models.append(model)

    names = [str(name) for name in models]
    names_to_models = {str(model): model for model in models}

    pipeline = st.multiselect(label="Pipeline", options=names, default=names, key="pipeline")
    st.write(pipeline)
    
    models = [names_to_models[name] for name in pipeline]
    joblib.dump(models, 'pipeline.cache')
    
    paths = defaultdict(list)

    if st.button('Run'):
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        if regions:
            for index, path in enumerate(regions):
                img = normalization(np.load(path))

                for method_index, method in enumerate(models):
                    progress = (index * len(models) + method_index) / (len(regions) * len(models))
                    with st.spinner(f'Computing {method} for {path}'):
                        result = method(img)
                        if not isinstance(result, list):
                            result = [result]
                        for visualization_index, visualization in enumerate(result):
                            visualization[img.max(axis=-1) == 0] = 0
                            if len(result) > 1:
                                name = str(index) + "_" + str(method).replace(' ', '').replace(':', '_') + str(visualization_index) + '.png'
                            else:
                                name = str(index) + "_" + str(method).replace(' ', '').replace(':', '_') + '.png'
                                
                            visualization_output_path = str(Path(output_path) / name)
                            Image.fromarray(visualization).save(visualization_output_path)

                            paths["visualization"].append(name)
                            paths["data"].append(path)
                            paths["method"].append(str(method))
                    st.progress(progress)


                pd.DataFrame.from_dict(paths).to_csv(str(Path(output_path) / "visualization_details.csv"), index=False)