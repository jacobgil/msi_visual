import streamlit as st
from msi_visual.extract.pymzml_to_numpy import PymzmlToNumpy

st.title('Convert a PyMZML file into an MSI-VISUAL numpy file')

pymzl_file = st.text_input('PyMZML file')
output_path = st.text_input('Output folder')

start_mz = st.number_input('Start m/z', value=None, help='If you specify the start and stop m/z, bins in these ranges will be created. Otherwise, the minimum and maximum m/z values in the data will be used')
end_mz = st.number_input('End M/Z', value=None, help='If you specify the start and stop m/z, bins in these ranges will be created. Otherwise, the minimum and maximum m/z values in the data will be used')

bins = st.number_input('Number of bins per m/z', value=5, min_value=2, step=1)

nonzero = st.checkbox('Extract only non zero values', help="This can be used to compress the file size. Only m/z values with non zero peaks will be extracted")

if st.button("Run"):
    if not start_mz or not end_mz:
        start_mz, end_mz = None, None

    extraction = PymzmlToNumpy(start_mz, end_mz, bins, nonzero)
    with st.spinner("Extracting.. "):
        extraction(pymzl_file, output_path)
