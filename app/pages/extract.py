import streamlit as st
import glob
from msi_visual.extract.pymzml_to_numpy import PymzmlToNumpy
from msi_visual.extract.bruker_tims_to_numpy import BrukerTimsToNumpy
from msi_visual.extract.bruker_tsf_to_numpy import BrukerTsfToNumpy

st.title('Convert a PyMZML or Bruker TSF/Tims files into an MSI-VISUAL numpy file')

input_path = st.text_input('Input (PyMZML file, or Bruker data folder)')
output_path = st.text_input('Output folder')

start_mz = st.number_input('Start m/z', value=None, help='If you specify the start and stop m/z, bins in these ranges will be created. Otherwise, the minimum and maximum m/z values in the data will be used')
end_mz = st.number_input('End M/Z', value=None, help='If you specify the start and stop m/z, bins in these ranges will be created. Otherwise, the minimum and maximum m/z values in the data will be used')

bins = st.number_input('Number of bins per m/z', value=5, min_value=2, step=1)

nonzero = st.checkbox('Extract only non zero values', help="This can be used to compress the file size. Only m/z values with non zero peaks will be extracted")

if st.button("Run"):
    if not start_mz or not end_mz:
        start_mz, end_mz = None, None

    if '.imzML' in input_path:
        extraction = PymzmlToNumpy(start_mz, end_mz, bins, nonzero)
    elif len(glob.glob(input + "/*.tdf")) > 0:
        extraction = BrukerTimsToNumpy(start_mz, end_mz, bins, nonzero)
    elif len(glob.glob(input + "/*.tsf")) > 0:
        extraction = BrukerTsfToNumpy(start_mz, end_mz, bins, nonzero)

    with st.spinner("Extracting.. "):
        extraction(input_path, output_path)
