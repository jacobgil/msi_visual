import glob
from pathlib import Path

import streamlit as st
from msi_visual.extract.bruker_tims_to_numpy import BrukerTimsToNumpy
from msi_visual.extract.bruker_tsf_to_numpy import BrukerTsfToNumpy
from msi_visual.extract.pymzml_to_numpy import PymzmlToNumpy

st.title("Convert a PyMZML or Bruker TSF/Tims files into an MSI-VISUAL numpy file")

input_path = st.text_input("Input (PyMZML file, or Bruker data folder)")
output_path = st.text_input("Output folder")

id = st.text_input("Identifier")

start_mz = st.number_input(
    "Start m/z",
    value=None,
    help="If you specify the start and stop m/z, bins in these ranges will be created. Otherwise, the minimum and maximum m/z values in the data will be used",
)
end_mz = st.number_input(
    "End M/Z",
    value=None,
    help="If you specify the start and stop m/z, bins in these ranges will be created. Otherwise, the minimum and maximum m/z values in the data will be used",
)

bins = st.number_input("Number of bins per m/z", value=5, min_value=2, step=1)

nonzero = st.checkbox(
    "Extract only non zero values",
    help="This can be used to compress the file size. Only m/z values with non zero peaks will be extracted",
)

use_fast = st.checkbox(
    "Fast extraction",
    value=True,
    help="Streaming + vectorized binning (+ workers). Default on. Turn off for the legacy extractor.",
)
num_workers = 1
if use_fast:
    num_workers = int(
        st.number_input(
            "Worker processes",
            value=8,
            min_value=1,
            step=1,
            help="Parallel processes for the fast backend (TIMS benefits most).",
        )
    )

if st.button("Run"):
    if not start_mz or not end_mz:
        start_mz, end_mz = None, None

    if not input_path or not output_path:
        st.error("Input path and output folder are required.")
        st.stop()

    is_imzml = ".imzML" in input_path or str(input_path).lower().endswith(".imzml")
    is_tims = len(glob.glob(str(Path(input_path) / "*.tdf"))) > 0 or len(
        glob.glob(input_path + "/*.tdf")
    ) > 0
    is_tsf = len(glob.glob(str(Path(input_path) / "*.tsf"))) > 0 or len(
        glob.glob(input_path + "/*.tsf")
    ) > 0

    fast = bool(use_fast)
    if fast and nonzero:
        st.warning("Fast extraction does not support nonzero mode; using legacy extractor.")
        fast = False

    extraction = None
    try:
        if is_imzml:
            if fast:
                from msi_visual.extract.imzml_fast import ImzmlFastExtractor

                extraction = ImzmlFastExtractor(
                    min_mz=start_mz,
                    max_mz=end_mz,
                    bins_per_mz=int(bins),
                    num_workers=num_workers,
                    id=id,
                )
            else:
                extraction = PymzmlToNumpy(
                    min_mz=start_mz,
                    max_mz=end_mz,
                    bins_per_mz=bins,
                    nonzero=nonzero,
                    id=id,
                )
        elif is_tims:
            if fast:
                if start_mz is None or end_mz is None:
                    st.error("Fast Bruker extraction requires Start m/z and End m/z.")
                    st.stop()
                from msi_visual.extract.bruker_fast import BrukerFastExtractor

                extraction = BrukerFastExtractor(
                    min_mz=start_mz,
                    max_mz=end_mz,
                    bins_per_mz=int(bins),
                    num_workers=num_workers,
                    id=id,
                    kind="tims",
                )
            else:
                extraction = BrukerTimsToNumpy(
                    min_mz=start_mz,
                    max_mz=end_mz,
                    bins_per_mz=bins,
                    nonzero=nonzero,
                    id=id,
                )
        elif is_tsf:
            if fast:
                if start_mz is None or end_mz is None:
                    st.error("Fast Bruker extraction requires Start m/z and End m/z.")
                    st.stop()
                from msi_visual.extract.bruker_fast import BrukerFastExtractor

                extraction = BrukerFastExtractor(
                    min_mz=start_mz,
                    max_mz=end_mz,
                    bins_per_mz=int(bins),
                    num_workers=num_workers,
                    id=id,
                    kind="tsf",
                )
            else:
                extraction = BrukerTsfToNumpy(
                    min_mz=start_mz,
                    max_mz=end_mz,
                    bins_per_mz=bins,
                    nonzero=nonzero,
                    id=id,
                )
        else:
            st.error("Could not detect input type (expected .imzML, analysis.tdf, or analysis.tsf).")
            st.stop()

        with st.spinner("Extracting.. "):
            extraction(input_path, output_path)
        st.success(f"Done ({'fast' if fast else 'legacy'} backend).")
    except Exception as exc:
        st.exception(exc)
