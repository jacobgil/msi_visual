import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.image("logo.png")

main_page = st.Page("pages/help.py", title="Main")
extract_page = st.Page("pages/extract.py", title="Import MSI data", icon='🔍')
viewer_page = st.Page("pages/viewer.py", title="Viewer", icon='🔍')
pipeline_page = st.Page("pages/pipeline.py", title="Create visualizations", icon='🎨')
seg_page = st.Page("pages/seg_train.py", title="Train segmentation models", icon='💪🏻')
dim_reduc_page = st.Page("pages/dimensionality_reduction_train.py", title="Dimensionality reduction models", icon='🔦')

st.logo("logo.png")


pg = st.navigation(
    {
        "Main": [main_page],
        "Viewer": [viewer_page],
        "Create visualizations": [pipeline_page],
        "Import data": [extract_page],
        #"Train models": [seg_page, dim_reduc_page]
    }
)

pg.run()