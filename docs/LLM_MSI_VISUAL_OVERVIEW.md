# MSI-VISUAL — LLM context overview

Concise reference for how this repo extracts MSI data, builds visualizations, ranks ions on regions in the Streamlit app, and computes SoLaCE edges. Paths are relative to the repo root.

---

## 1. Extraction (raw MSI → dense `.npy` cubes)

### Goal

Convert imzML or Bruker `.d` / `.d.zip` into a dense **float32** cube `H × W × C` (spatial Y × X × m/z channels), plus `args.txt` with the m/z axis and extraction knobs.

**TIC normalization is not applied during extraction.** Cubes on disk are raw summed intensities. TIC is applied later when loading for visualization / Viewer (`scripts/msi_viz_io.py`).

### Main modules

| Path | Role |
|------|------|
| `msi_visual/extract/base_msi_to_numpy.py` | Abstract `BaseMSIToNumpy` (legacy dense/discrete binning + save) |
| `msi_visual/extract/imzml_fast.py` | `ImzmlFastExtractor` — default fast imzML path |
| `msi_visual/extract/bruker_fast.py` | `BrukerFastExtractor` — default fast Bruker TIMS/TSF |
| `msi_visual/extract/bruker_input.py` | Resolve `.d` / unzip `.d.zip`, read acquisition metadata |
| `msi_visual/extract/pymzml_to_numpy.py` | Legacy imzML (`PymzmlToNumpy`) |
| `msi_visual/extract/bruker_tims_to_numpy.py` / `bruker_tsf_to_numpy.py` | Legacy Bruker |

**Binning (fast paths):**  
`bin = round((mz - min_mz) * bins_per_mz)` → accumulate intensity into channel `bin` (out-of-range dropped). Coordinates shifted to origin.

### CLI / batch entrypoints

- `scripts/extraction/extract_pymzml.py` — imzML (default `--fast`)
- `scripts/extraction/extract_bruker_tims.py` / `extract_bruker_tsf.py` — Bruker
- `scripts/batch_extract_pymzml.py` — Hydra batch over a folder
- Typical config: `scripts/configs/batch_extract_pymzml.yaml`

### Output layout

```
{output_root}/[{rel}/]{slide_stem}/{N}_bins/{region}.npy
{output_root}/.../args.txt
```

Examples: `.../5_bins/0.npy`, `.../10_bins/0.npy`. Bruker may write multiple regions (`0.npy`, `1.npy`, …).

| Artifact | Content |
|----------|---------|
| `{region}.npy` | `float32` `(H, W, C)` |
| `args.txt` | `start_mz`, `end_mz`, `bins`, `mzs`, `id`, etc. |

**Channel count:** roughly `C ≈ bins_per_mz × (max_mz − min_mz + 1)`.

### Useful defaults

| Knob | Typical |
|------|---------|
| `bins` / bins-per-m/z | CLI often `1`; batch configs often **`5`** → folder `5_bins` |
| Backend | **`fast`** |
| `num_workers` | **8** |
| imzML region | always `0` |
| TIC on disk | off; on for viz/Viewer loads |

### Alternate path: IsoMobile

`msi_visual/extract/isomobile.py` — unbinned TIMS peaks + mobility (CSR-style files), **not** an `H×W×C` cube.

---

## 2. Streamlit app flow (Import → Pipeline → Viewer)

### Entrypoint

`app/main.py` — navigation:

1. **Help** — `app/pages/help.py`
2. **Import data** — `app/pages/extract.py` (run extraction → `.npy`)
3. **Create visualizations** — `app/pages/pipeline.py` (fit viz methods → PNGs + `visualization_details.csv`)
4. **Viewer** — `app/pages/viewer.py` (**ROI selection + m/z ranking**)

Helpers: `msi_visual/app/utils/extraction.py`, `pipeline.py`, `viewer.py`.

Intended workflow: **Import → Pipeline → Viewer**.

---

## 3. Region selection and m/z ranking (Viewer)

### Where it lives

- UI: `app/pages/viewer.py`
- Logic / display: `msi_visual/app/utils/viewer.py`
- Ranking math: `RegionComparison` in `msi_visual/visualizations.py`

### Viewer UX (defaults)

| Setting | Default |
|---------|---------|
| Comparison type | **Region with all** |
| Selection | **Polygon** (lasso) |
| Ranking method | **U-Test** |
| RGB segmentation bins (click mode) | 5 |

**Tabs:** Settings | Visualizations | Comparisons | M/Z

**Selections**

1. Load a pipeline folder via `visualization_details.csv` (maps PNG viz → source `.npy`).
2. Spectra are **TIC-normalized** in the Viewer.
3. **Polygon:** lasso → filled mask → stored as `ClickData`.
4. **Click:** pick a color segment under the cursor (`segment_visualization` / `get_mask`).

Stored in `st.session_state["clicks"][data_path]`.

**Comparison modes**

| Mode | What is compared |
|------|------------------|
| Region with all | Newest 1 ROI vs tissue \ ROI (background subsampled `::3`) |
| Region with Region | Newest 2 ROIs vs each other |
| Point with Point | Newest 2 clicks (separate rank path; ignores U-Test / ROI-MEAN) |

### Ranking algorithms (`RegionComparison`)

Shared prep:

1. Spectra inside mask A / B.
2. Candidate peaks = union of `find_peaks` on the **30th-percentile** spectrum of each ROI (`peak_minimum=0`).
3. Rank **only** those peak channels (not every m/z).

#### Default: `"U-Test"`

- Mann–Whitney U per peak (`scipy.stats.mannwhitneyu`).
- Score ≈ `U / (n_a * n_b)` (AUC-like).
- Keep peaks with **p &lt; 0.05**.
- Higher score ⇒ elevated in region A vs B.

#### `"Difference in ROI-MEAN"`

- Max-normalize each ion image; score = `mean_A / (eps + mean_B)`, then rescale to [0, 1].
- No p-value filter.

#### Point vs point

- Top-200 intensities at each point; spatial-rank difference mapped to [0, 1].

**Not used for Viewer ion ranking:** SHAP, XGBoost feature importance, or saliency as ion scorers. (XGBoost / saliency appear elsewhere as *visualization* tools.)

### Ranking → UI

```
Settings → draw ROI → Comparisons → ranking_comparison()
  → top/bottom ~50 ions as buttons → click → ion image in sidebar
```

Also: aggregate overlay of top scored ions; **Spec QR** mosaic of ranked m/z (below).

---

## 4. Spec QR visualization

There is **no** class named `SpecQR`. “Spec QR” is a **2-D mosaic** of m/z cells (QR-code-like), used in the Streamlit Viewer with region rankings.

| Piece | Path |
|-------|------|
| Helpers | `get_2d_spectrum()` / `GetOverlay.get_2d_spectrum()` in `msi_visual/app/utils/viewer.py` |

**Behavior**

- Grid ≈ square; each cell is one m/z (cell size ~10×10 px).
- Color encodes ranking **score** (often `score**4` with a colormap such as `cividis`).
- Dark blue figure background (`#01224d`); min/max m/z annotated.
- Used alongside aggregated ion images for region/point comparisons.

**Visually:** a compact “QR” of the extraction m/z list — bright cells = important ions for the current comparison — **not** a tissue RGB map.

---

## 5. Parametric visualizations (MiCS+LMC, UMAP)

### 5.1 Parametric MiCS + LMC (`MSIParametricMiCSLMC`)

| | |
|--|--|
| Class | `MSIParametricMiCSLMC` |
| File | `msi_visual/parametric_mics_lmc.py` |
| Output | Dense RGB “virtual stain”: pixel MLP maps spectra → 3-D embedding (LAB), percentile stretch, optional LAB→RGB |

Multiscale **cluster** heads encourage anatomically coherent structure. Optional SoLaCE-guided bilateral smooth before RGB (`predict_edge_smooth` + edge map).

**Training config (production-style):** `scripts/configs/train_parametric_mics_lmc.yaml`  
**Annotator interactive defaults:** `scripts/annotator_server/configs/default.yaml`  
**Fast preset:** `scripts/annotator_server/configs/fast.yaml`  
**Panel:** `scripts/configs/generate_visualization_panel.yaml` (`mics_lmc` / `parametric_mics_lmc`)

#### Default settings cheat-sheet

| Knob | Annotator `default.yaml` | Typical train / panel (mics2-like) |
|------|--------------------------|-----------------------------------|
| `model_type` | `mics_lmc` | `mics_lmc` |
| `number_of_points` / sampling | 500 / coreset | 1000 / coreset |
| `num_epochs` | 30 | 30 |
| `num_layers` | 3 | 7 (train) / ~3 (panel merge) |
| `clusters` | `4-8` | e.g. `16-32-64-1024` or `128-256-512-1024` |
| `cluster` | true | true |
| `num_samples` | 1000 | 5000 |
| `beta` | 1.0 | ~20 |
| `lr` / `batch_size` | 1.0 / (config) | 1.0 / 1024 |
| `pixel_sampling` | random | superpixel (train) / often random in benchmarks |
| `pca_fit_step` | 16 | 16 |
| `lab_to_rgb` | true | true |
| Predict percentiles | (code/config) | often ~1–99 in train yaml |

**Looks like:** smooth, colorful tissue-type maps with coherent regions (stronger structure than TOP3 when clusters/β are set).

### 5.2 Parametric UMAP

| | |
|--|--|
| Classes | `MSIParametricUMAP`, `UMAPVirtualStain` |
| File | `msi_visual/parametric_umap.py` |
| Output | Sampled-pixel ParametricUMAP encoder → 3-channel RGB |

Typical knobs: `n_components=3`, `n_neighbors=50`, `n_training_epochs=2`, `num_samples=5000`, `pixel_sampling=superpixel`, `pca_fit_step=4`.

### 5.3 PCA3D

`msi_visual/pca_3d.py` — linear 3-PC → percentile RGB. Often included in panels alongside parametric methods.

---

## 6. Classical / other visualizations

Wired in panel (`generate_visualization_panel.py`) and/or annotator (`model_type`):

| Method | Class / key | File | Defaults / notes | Visual character |
|--------|-------------|------|------------------|------------------|
| **TOP3** | `TOP3` / `top3` | `msi_visual/percentile_ratio.py` | `low=99.9`, `norm_percentile=99.9`; panel often `to_lab=true` | Per-pixel top-3 intensities → R/G/B (dominant-ion RGB) |
| **Percentile ratio (PR3D)** | `PercentileRatio` / `percentile_ratio` / `pr3d` | same | percentiles e.g. `[99.99, 99.9, 99.9, 99, 98, 85]` | Intensity-ratio channels → LAB→RGB |
| **NMF3D** | `NMF3D` / `nmf_3d` | `msi_visual/nmf_3d.py` | `n_components=3`, `max_iter=200` | Soft additive 3-factor RGB |
| **TOP3 on NMF** | `top3_nmf` | panel pipeline | `k_list` e.g. 3…15 | TOP3 aesthetic on NMF coefficient cubes |
| **K-means seg** | `KmeansSegmentation` | `msi_visual/kmeans_segmentation.py` | `simple_k` (annotator default 8) | Discrete rainbow regions + certainty |
| **NMF seg** | `NMFSegmentation` | `msi_visual/nmf_segmentation.py` | `simple_k`, `max_iter` ~2000 | Soft factor blend |
| **Nonparametric UMAP** | `MSINonParametricUMAP` | `msi_visual/nonparametric_umap.py` | panel/annotator: `min_dist=0.5`, `n_neighbors=100` | Full-image umap-learn RGB (slow) |

**Annotator classical knobs** (`default.yaml`): `top3_low/norm=99.9`, `nmf_3d_*=3/200`, `umap_np_*=0.5/100/euclidean`, `simple_k=8`.

Many other DR helpers exist under `msi_visual/` (t-SNE, PHATE, etc.) but are not the primary panel/annotator set.

---

## 7. SoLaCE edge detection (“blue” edges)

### What SoLaCE is

**SoLaCE** = **Soft Landmark Contrast Edges**  
Code name: `hd_method: soft_landmark_contrast` (soft evolution of older landmark-kNN edges).

| Piece | Path |
|-------|------|
| Core edge computation | `scripts/debug_edge_maps.py` (`_landmark_knn_*`, soft patch contrast) |
| Gallery / ranking configs | `scripts/configs/hd_methods_gallery.yaml`, `rank_visualizations_by_f1.yaml` |
| Panel soft edges | `soft_edges` block in `generate_visualization_panel.yaml` |

### Mechanism (defaults from HD configs)

Typical `soft_landmark_contrast` knobs:

| Knob | Default (approx.) |
|------|-------------------|
| `score_mode` | `soft_patch_contrast` |
| `n_landmarks` | 1000 |
| `soft_topk` / `soft_temperature` | 80 / 0.06 |
| `soft_patch_radius` | 1 |
| PCA on spectra | on, ~64 components |
| Spatial radii / connectivity | `[1]`, 4-connected |

**Idea:** soft membership of each pixel to spectral landmarks → local soft-patch signatures → contrast across neighbors → `H×W` edge strength in [0, 1].

### Visual appearance (“blue visualization”)

- Primary product is an **edge map** (spectral boundaries), not a multi-channel tissue stain.
- Publication / gallery panels often color it with **`PuBu`** (blue-on-white) — this is the familiar “blue SoLaCE edges” look. Label in batches is often overridden to `SoLaCE`.
- Companion **L∞** maps (`hd_edge_linf_norm.npy`) exist for metrics / agreement, not as the main “pretty” blue panel.

### Use beyond display

- Benchmarking visualization quality (edge F1 / continuous Dice / agreement vs RGB edges of TOP3, MiCS, etc.).
- Optional **SoLaCE-guided bilateral smooth** inside MiCS predict: smooth embeddings within regions but not across SoLaCE boundaries (`solace_guided_bilateral_smooth` in `parametric_mics_lmc.py`).

---

## 8. Related UIs (quick pointers)

| UI | Role |
|----|------|
| Streamlit `app/` | Extract → pipeline PNG gallery → Viewer ranking + Spec QR |
| Annotator FastAPI `scripts/annotator_server/` | Interactive ROI recompute, parametric/classical viz, parameter sweeps |
| Visualization ranker | Offline ranking / edge-quality analysis of saved viz |

---

## 9. Mental model (one paragraph)

Raw MSI is **extracted** into dense binned cubes (`*_bins/*.npy`). The Streamlit **pipeline** turns cubes into RGB visualizations (TOP3, PR3D, NMF, MiCS+LMC, UMAPs, …). The **Viewer** lets you draw regions; **Mann–Whitney U-test** (default) ranks which m/z differ vs background/other ROI, shown as ion lists and a **Spec QR** score mosaic. **SoLaCE** independently estimates high-dimensional spectral edges (often shown in **PuBu blue**) for quality metrics and optional MiCS smoothing—not for ion ranking.
