# MSI-VISUAL: high fidelity visualization, mapping and exploration of mass spectrometry images

`pip install msi-visual`

⭐ Unlock the potential of MASS mass spectrometry imaging (MSI) data with extremely rich visualizations.

⭐ Quick interactive exploration and mapping of MSI data based on statistical tests between different regions.

⭐ Improved quantitive methods for evaluating these visualizations.


We believe that Mass Spectometry Imaging is the future of biological and medical diagnostics and research. There is extremely rich information in this kind of data that is currently not being utilized. We are on a mission to solve this.

MSI-VISUAL is a python package for visualizing and interacting with MSI data.
It is intended for both life science researchers that can use it from an interactive web app, as well as developers that can use the underlying algorithms in their own software.
It includes:
- A python library with multiple novel state of the art visualization and evaluation algorithms for MSI data.
- A web app (currently based on [streamlit](https://streamlit.io/)) for interactively visualizing MSI data, and performing statistical comparison between different regions, to map significant m/z values.


# Using from the python library

## Creating visualizations

The input data is expected to be a .npy file with a tensor of shape rows x cols x number_of_mz_values.
See the data extraction section for more information on how to convert your data into this format.

### Dimensionality reduction methods

Supported visualizations are:
| Visualization Name | Import | Description |
|--------------------|--------|-------------|
| Saliency Optimization | `from msi_visual.saliency_opt import SaliencyOptimization` | A novel optimization-based approach for creating high-fidelity visualizations of MSI data. |
| NMF (Non-negative Matrix Factorization) | `from msi_visual.nmf_3d import NMF3D` | Applies NMF to decompose the MSI data into meaningful components for visualization. |
| Non-parametric UMAP | `from msi_visual.nonparametric_umap import MSINonParametricUMAP` | Uses UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction and visualization of MSI data. |
| Top 3 | `from msi_visual.percentile_ratio import top3` | Visualizes the top 3 most intense m/z values for each pixel. |
| Percentile Ratio RGB | `from msi_visual.percentile_ratio import percentile_ratio_rgb` | Creates an RGB image based on the ratio of high to low percentiles of m/z intensities. |

```python
method = SaliencyOptimization(num_epochs=200,
                              regularization_strength=0.001,
                              sampling="random",
                              number_of_points=600,
                              init="coreset")
visualization = method(data)
```

During the first call on an image, the visualization model is trained.
It can also be trained on several images by called `.fit([images])` directly.



### Clustering visualization methods

from msi_visual.kmeans_segmentation import KmeansSegmentation
from msi_visual.nmf_segmentation import NMFSegmentation
| Visualization Name | Import | Description |
|--------------------|--------|-------------|
| K-means Segmentation | `from msi_visual.kmeans_segmentation import KmeansSegmentation` | Kmeans clustering. |
| NMF Segmentation | `from msi_visual.nmf_segmentation import NMFSegmentation` | NMF based clustering. |

```python
method = KmeansSegmentation(num_clusters=10)
visualization = method(data)
```
### Combining clustering and dimensionality reduction
In this approach a single dimensional image is created with a method, and is then visualized with a different color scheme in every region segmented by a clustering method.

| Visualization Name | Import | Description |
|--------------------|--------|-------------|
| Rare NMF Segmentation | `from msi_visual.rare_nmf_segmentation import SegmentationPercentileRatio` | NMF based clustering. |
| Avg MZ NMF Segmentation | `from msi_visual.avgmz_nmf_segmentation import SegmentationAvgMZVisualization` | NMF based clustering. |
| UMAP NMF Segmentation | `from msi_visual.umap_nmf_segmentation import SegmentationUMAPVisualization` | NMF based clustering. |

```python
method = SegmentationPercentileRatio(joblib.load("nmf_model_k=16.joblib"))
visualization = method(data)
```




## Evaluating visualizations

```python
from msi_visual.metrics import MSIVisualizationMetrics
metrics = MSIVisualizationMetrics(data, visualization)
```

## Data format and data extraction

The input files a simply .npy files with tensors of shape rows x cols x number_of_mz_values.


To help you convert your raw MSI data into the required format, we provide several data extraction scripts. Here's a summary of the available data extraction methods:

| Data Format | Import | Script Name | Description |
|-------------|--------|-------------|-------------|
| Bruker TIMS | `from msi_visual.extract.bruker_tims_to_numpy import BrukerTimsToNumpy` | `scripts/extraction/extract_bruker_tims.py` | Converts Bruker TIMS data (.d folder) |
| Bruker TSF | `from msi_visual.extract.bruker_tsf_to_numpy import BrukerTSFToNumpy` | `scripts/extraction/extract_bruker_tsf.py` | Converts Bruker TSF data (.d folder) |
| pymzML | `from msi_visual.extract.pymzml_to_numpy import ImzMLToNumpy` | `scripts/extraction/extract_pymzml.py` | Converts the open source pymzML data format to numpy arrays (.npy files) |

Example running dataset extraction from python code:

```python
extractor = BrukerTimsToNumpy(identifier, start_mz, end_mz, bins, nonzero)
extractor(input_folder, output_folder)
```

if nonzero is set to true, the script will identify m/z values that have non zero intensities somewhere in the data, and will keep only those.
In case peak selection was used, this may reduce the data size substantially.

You can also run the extraction from scripts (and from the User Interface).

```bash
python extract_bruker_tims.py --input_path input_folder --output_path output_folder --bins 5 --num_workers 1 --id some_string_identifier
```



