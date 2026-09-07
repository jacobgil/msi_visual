#!/usr/bin/env python3
"""
Interactive MiCS interpretability-panel builder (Streamlit).

Workflow
--------
1. Pick a config + slide in the sidebar and press "Build model".
   This trains the frozen MiCS teacher, fits the sparse NMF concept explainer,
   and renders the MiCS RGB visualization (same pipeline as
   ``scripts/explain_mics_concepts.py``).
2. Click points directly on the visualization to choose pixels.
3. Press "Generate panel" to render the interpretability panel
   (central tissue map + per-pixel concept panels) from the clicked points,
   then download the high-DPI PNG.

Run (repo root):
    streamlit run scripts/interpretability_panel_app.py

The heavy model build is cached per (config, slide, key hyperparameters), so
re-clicking / re-generating panels is fast.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_CONFIG_DIR = _SCRIPTS / "configs"
_OUTPUT_ROOT = _REPO / "outputs" / "interpretability_panel_app"
_DISPLAY_WIDTH = 720  # px width of the clickable visualization

MARKER_COLOR = "#ffe066"


# --------------------------------------------------------------------------- #
# Model / explainer pipeline (mirrors scripts/explain_mics_concepts.py main())
# --------------------------------------------------------------------------- #
def _excel_label(i: int) -> str:
    """0->A, 25->Z, 26->AA (matches _pick_example_pixels labeling)."""
    i = int(i)
    if i < 26:
        return chr(ord("A") + i)
    chars: list[str] = []
    n = i
    while True:
        chars.append(chr(ord("A") + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(chars))


def _compose_cfg(
    *,
    config_name: str,
    npy_path: str,
    mics_group: str,
    seed: int,
    num_samples: int,
    sampling_mode: str,
    normalization: str,
    rotate_deg: int,
    cache_teacher: bool,
    n_concepts: int,
    top_k: int,
    fit_method: str,
    weight_mode: str,
    softmax_temperature: float,
    train_epochs: int,
    train_lr: float,
):
    """Compose the explain-mics config and overwrite the interactive fields."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name=config_name)

    OmegaConf.set_struct(cfg, False)
    # Overwrite ${hydra:runtime.cwd} interpolations with absolute paths (no Hydra runtime here).
    OmegaConf.update(cfg, "mics.defaults_yaml", str((_CONFIG_DIR / "train_parametric_mics_lmc.yaml").resolve()))
    OmegaConf.update(cfg, "mics.parameter_groups_config", str((_CONFIG_DIR / "batch_extract_mics_groups.yaml").resolve()))

    OmegaConf.update(cfg, "seed", int(seed))
    OmegaConf.update(cfg, "gallery.npy_path", str(npy_path))
    OmegaConf.update(cfg, "gallery.num_samples", int(num_samples))
    OmegaConf.update(cfg, "gallery.sampling_mode", str(sampling_mode))
    OmegaConf.update(cfg, "gallery.cache_teacher", bool(cache_teacher))
    OmegaConf.update(cfg, "mics.group", str(mics_group))
    OmegaConf.update(cfg, "data.normalization", str(normalization))
    OmegaConf.update(cfg, "data.rotate_clockwise_deg", int(rotate_deg))

    OmegaConf.update(cfg, "concepts.n_concepts", int(n_concepts))
    OmegaConf.update(cfg, "concepts.top_k", int(top_k))
    OmegaConf.update(cfg, "concepts.fit_method", str(fit_method))
    OmegaConf.update(cfg, "concepts.weight_mode", str(weight_mode))
    OmegaConf.update(cfg, "concepts.softmax_temperature", float(softmax_temperature))
    OmegaConf.update(cfg, "concepts.train_epochs", int(train_epochs))
    OmegaConf.update(cfg, "concepts.train_lr", float(train_lr))
    # Overview gallery needs an HD-edge rank config + SoLaCE compute; not used in the app.
    OmegaConf.update(cfg, "viz.overview_gallery.enabled", False)
    return cfg


def _build_bundle(cfg) -> dict[str, Any]:
    """Train MiCS teacher, fit the concept explainer, return arrays for the panel."""
    import gc

    import torch
    from omegaconf import OmegaConf

    from benchmark_parametric_methods import _seed_everything
    from debug_edge_maps import _load_msi, _resolve_normalization_mode, _to_uint8_rgb
    from explain_mics_concepts import _abs_path, _concepts_cfg
    from gallery_mics_concept_distill import (
        _load_teacher_cache,
        _mics_kwargs,
        _pixel_sampling_cache_path,
        _save_teacher_cache,
        _teacher_cache_path,
    )
    from gallery_mics_sampling_cache import (
        build_pixel_sampling_cache,
        load_pixel_sampling_cache,
        save_pixel_sampling_cache,
    )
    from msi_visual.mics_concept_explain import (
        FrozenConceptExplainer,
        concept_colors_precomputed,
    )
    from msi_visual.mics_concept_explain_viz import _equalize_map_display
    from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC

    seed = int(cfg.seed)
    _seed_everything(seed)

    npy_path = _abs_path(str(cfg.gallery.npy_path))
    if not npy_path.is_file():
        raise FileNotFoundError(f"gallery.npy_path not found: {npy_path}")

    norm_mode = _resolve_normalization_mode(str(cfg.data.normalization))
    msi = _load_msi(npy_path, transpose_msi=bool(cfg.data.transpose_msi), normalization=norm_mode)
    rot_cw = int(OmegaConf.select(cfg, "data.rotate_clockwise_deg", default=0)) % 360
    if rot_cw:
        if rot_cw not in (90, 180, 270):
            raise ValueError(f"data.rotate_clockwise_deg must be 0/90/180/270; got {rot_cw}")
        k_ccw = {90: 3, 180: 2, 270: 1}[rot_cw]
        msi = np.ascontiguousarray(np.rot90(msi, k=k_ccw, axes=(0, 1)))

    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    model_kw = _mics_kwargs(cfg)
    mics_group = str(OmegaConf.select(cfg, "mics.group", default="custom"))
    cache_teacher = bool(OmegaConf.select(cfg, "gallery.cache_teacher", default=True))
    teacher_cache_path = _teacher_cache_path(cfg, npy_path, model_kw, mics_group)

    concepts_cfg = _concepts_cfg(cfg, model_kw)
    needs_teacher_model = str(concepts_cfg.fit_method).strip().lower() in {
        "mics_input",
        "through_mics",
        "input_backprop",
    }

    teacher_emb: np.ndarray | None = None
    mics_rgb: np.ndarray | None = None
    teacher: MSIParametricMiCSLMC | None = None

    if cache_teacher and not needs_teacher_model:
        cached = _load_teacher_cache(teacher_cache_path)
        if cached is not None:
            teacher_emb, mics_rgb = cached

    if teacher_emb is None or needs_teacher_model:
        share_sampling = bool(OmegaConf.select(cfg, "gallery.share_pixel_sampling", default=True))
        n_pix = int(msi.shape[0]) * int(msi.shape[1])

        def _valid_cache(path: Path):
            """Load a per-slide sampling cache only if its indices fit this image."""
            if path is None or not path.is_file():
                return None
            try:
                c = load_pixel_sampling_cache(path)
            except Exception:
                return None
            flat = np.asarray(c.get("sampled_flat_indices", []), dtype=np.int64).ravel()
            if flat.size == 0 or int(flat.max()) >= n_pix or int(flat.min()) < 0:
                return None
            return c

        cache = None
        if share_sampling:
            # Per-slide cache path (digest keyed on slide + sampling params); build fresh if
            # missing or incompatible. Avoids the slide-agnostic fallback in resolve_pixel_sampling_cache.
            sampling_cache_path = _pixel_sampling_cache_path(cfg, npy_path, model_kw, mics_group)
            cache = _valid_cache(sampling_cache_path)
            if cache is None:
                if sampling_cache_path.is_file():
                    sampling_cache_path.unlink()
                cache = build_pixel_sampling_cache(msi, model_kw)
                save_pixel_sampling_cache(sampling_cache_path, cache)

        teacher = MSIParametricMiCSLMC(**model_kw)
        teacher.fit(msi, pixel_sampling_cache=cache)
        teacher_emb = np.asarray(teacher.predict_embedding(msi), dtype=np.float32)
        mics_rgb = _to_uint8_rgb(teacher.predict(msi))
        if cache_teacher and not needs_teacher_model:
            _save_teacher_cache(teacher_cache_path, teacher_emb, mics_rgb)

    assert teacher_emb is not None and mics_rgb is not None
    mics_rgb = np.asarray(mics_rgb, dtype=np.uint8)
    mics_rgb[~valid_mask] = 0
    teacher_emb = np.asarray(teacher_emb, dtype=np.float32)
    teacher_emb[~valid_mask] = 0.0

    explainer = FrozenConceptExplainer.from_mics(
        msi,
        teacher_emb,
        valid_mask,
        concepts_cfg,
        mics_model=teacher.model if teacher is not None else None,
    )
    if teacher is not None:
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    alpha_hwk = np.asarray(explainer.alpha_hwk, dtype=np.float32)
    concept_rgbs = concept_colors_precomputed(explainer, valid_mask=valid_mask)
    tissue_mean_alpha = (
        alpha_hwk[valid_mask].mean(axis=0) if np.any(valid_mask) else alpha_hwk.mean(axis=(0, 1))
    )
    display_rgb = _equalize_map_display(mics_rgb, valid_mask)

    return {
        "npy_name": npy_path.name,
        "mics_rgb": mics_rgb,
        "display_rgb": np.asarray(display_rgb, dtype=np.uint8),
        "alpha_hwk": alpha_hwk,
        "concept_rgbs": np.asarray(concept_rgbs, dtype=np.uint8),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
        "tissue_mean_alpha": np.asarray(tissue_mean_alpha, dtype=np.float32),
        "shape": (int(mics_rgb.shape[0]), int(mics_rgb.shape[1])),
    }


@st.cache_resource(show_spinner=False)
def get_bundle(
    config_name: str,
    npy_path: str,
    mics_group: str,
    seed: int,
    num_samples: int,
    sampling_mode: str,
    normalization: str,
    rotate_deg: int,
    cache_teacher: bool,
    n_concepts: int,
    top_k: int,
    fit_method: str,
    weight_mode: str,
    softmax_temperature: float,
    train_epochs: int,
    train_lr: float,
) -> dict[str, Any]:
    cfg = _compose_cfg(
        config_name=config_name,
        npy_path=npy_path,
        mics_group=mics_group,
        seed=seed,
        num_samples=num_samples,
        sampling_mode=sampling_mode,
        normalization=normalization,
        rotate_deg=rotate_deg,
        cache_teacher=cache_teacher,
        n_concepts=n_concepts,
        top_k=top_k,
        fit_method=fit_method,
        weight_mode=weight_mode,
        softmax_temperature=softmax_temperature,
        train_epochs=train_epochs,
        train_lr=train_lr,
    )
    return _build_bundle(cfg)


# --------------------------------------------------------------------------- #
# Panel rendering
# --------------------------------------------------------------------------- #
def render_panel(bundle: dict[str, Any], points: list[tuple[int, int]], out_path: Path, viz: dict[str, Any]) -> str:
    from msi_visual.mics_concept_explain_viz import save_map_concept_overlay

    example_pixels = [(int(y), int(x), _excel_label(i)) for i, (y, x) in enumerate(points)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return save_map_concept_overlay(
        bundle["mics_rgb"],
        bundle["alpha_hwk"],
        bundle["concept_rgbs"],
        bundle["valid_mask"],
        out_path,
        example_pixels=example_pixels,
        tissue_mean_alpha=bundle["tissue_mean_alpha"],
        concept_rank_mode=str(viz["concept_rank_mode"]),
        top_terms=int(viz["top_terms"]),
        min_weight=float(viz["min_concept_weight"]),
        concept_node_style="heatmap",
        concept_thumb_px=int(viz["concept_thumb_px"]),
        activation_percentile=float(viz["activation_percentile"]),
        activation_gamma=float(viz["activation_gamma"]),
        activation_colormap=str(viz["activation_colormap"]),
        map_fig_width=float(viz["map_fig_width"]),
        map_fig_height=float(viz["map_fig_height"]),
        map_marker_fontsize=float(viz["map_marker_fontsize"]),
        map_marker_size=float(viz["map_marker_size"]),
        equalize_map_display=bool(viz["equalize_map_display"]),
        equalize_thumb_activation=bool(viz["equalize_thumb_activation"]),
        dpi=int(viz["dpi"]),
    )


def export_full_folder(
    bundle: dict[str, Any],
    points: list[tuple[int, int]],
    run_dir: Path,
    viz: dict[str, Any],
) -> dict[str, Any]:
    """Write the full per-pixel export folder (same structure as explain_mics_concepts)."""
    from msi_visual.mics_concept_explain_viz import save_pixel_interpretability_exports

    run_dir.mkdir(parents=True, exist_ok=True)
    example_pixels = [(int(y), int(x), _excel_label(i)) for i, (y, x) in enumerate(points)]

    # Top-level MiCS visualization (raw + equalized), mirroring explain_mics_concepts outputs.
    Image.fromarray(bundle["mics_rgb"]).save(run_dir / "mics_teacher.png")
    Image.fromarray(bundle["display_rgb"]).save(run_dir / "mics_teacher_equalized.png")
    np.save(run_dir / "alpha_hwk.npy", bundle["alpha_hwk"])
    np.save(run_dir / "concept_colors.npy", bundle["concept_rgbs"])
    Image.fromarray(bundle["concept_rgbs"].reshape(1, -1, 3)).save(run_dir / "concept_colors_row.png")

    return save_pixel_interpretability_exports(
        bundle["mics_rgb"],
        bundle["alpha_hwk"],
        bundle["concept_rgbs"],
        bundle["valid_mask"],
        run_dir,
        example_pixels=example_pixels,
        tissue_mean_alpha=bundle["tissue_mean_alpha"],
        concept_rank_mode=str(viz["concept_rank_mode"]),
        top_terms=int(viz["top_terms"]),
        min_weight=float(viz["min_concept_weight"]),
        concept_node_style="heatmap",
        export_thumb_px=int(viz["export_thumb_px"]),
        export_dpi=int(viz["dpi"]),
        activation_percentile=float(viz["activation_percentile"]),
        activation_gamma=float(viz["activation_gamma"]),
        activation_colormap=str(viz["activation_colormap"]),
        equalize_map_display=bool(viz["equalize_map_display"]),
        equalize_thumb_activation=bool(viz["equalize_thumb_activation"]),
        save_both_display_variants=True,
        map_fig_width=float(viz["map_fig_width"]),
        map_fig_height=float(viz["map_fig_height"]),
        map_marker_fontsize=float(viz["map_marker_fontsize"]),
        map_marker_size=float(viz["map_marker_size"]),
        subdir="interpretability",
    )


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def _display_image_with_markers(bundle: dict[str, Any], points: list[tuple[int, int]]) -> tuple[Image.Image, float]:
    """Resize the MiCS RGB to display width and draw numbered markers for picked points."""
    h, w = bundle["shape"]
    scale = w / float(_DISPLAY_WIDTH)
    disp_w = int(_DISPLAY_WIDTH)
    disp_h = max(1, int(round(h / scale)))
    img = Image.fromarray(bundle["display_rgb"]).resize((disp_w, disp_h), Image.NEAREST).convert("RGB")

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    r = 9
    for i, (y, x) in enumerate(points):
        dx, dy = x / scale, y / scale
        draw.ellipse([dx - r, dy - r, dx + r, dy + r], outline=MARKER_COLOR, width=3)
        draw.text((dx + r + 2, dy - r - 2), _excel_label(i), fill=MARKER_COLOR, font=font)
    return img, scale


@st.cache_data(show_spinner=False)
def _available_mics_groups() -> list[str]:
    """MiCS group preset names from batch_extract_mics_groups.yaml."""
    from omegaconf import OmegaConf

    try:
        cfg = OmegaConf.load(_CONFIG_DIR / "batch_extract_mics_groups.yaml")
        names = [str(g["name"]) for g in cfg.mics_lmc.parameter_groups]
        return names or ["mics_multiscale"]
    except Exception:
        return ["mics_multiscale"]


def main() -> None:
    st.set_page_config(page_title="MiCS Interpretability Panel", layout="wide", page_icon="🔬")
    st.title("🔬 MiCS Interpretability Panel Builder")
    st.caption(
        "Build a MiCS visualization from a config, click pixels on the image, "
        "and render the concept interpretability panel from your selection."
    )

    ss = st.session_state
    ss.setdefault("points", [])
    ss.setdefault("last_click", None)
    ss.setdefault("panel_path", None)

    # ------------------------------ Sidebar ------------------------------ #
    with st.sidebar:
        st.header("1 · Model")
        configs = sorted(p.stem for p in _CONFIG_DIR.glob("explain_mics_concepts*.yaml"))
        config_name = st.selectbox("Config", configs or ["explain_mics_concepts"], index=0)
        npy_path = st.text_input(
            "Slide (.npy)",
            value="E:/MSImaging-data/_msi_visual/Extractions/NRL4485-s2/5_bins/1.npy",
        )
        _groups = _available_mics_groups()
        mics_group = st.selectbox(
            "MiCS group",
            _groups,
            index=_groups.index("mics_multiscale") if "mics_multiscale" in _groups else 0,
        )
        col_a, col_b = st.columns(2)
        seed = col_a.number_input("Seed", value=17, step=1)
        num_samples = col_b.number_input("Num samples", value=5000, step=1000)
        sampling_mode = col_a.selectbox("Pixel sampling", ["superpixel", "random", "edge_stratified"], index=0)
        normalization = col_b.selectbox("Normalization", ["tic", "median", "none"], index=0)
        rotate_deg = col_a.selectbox("Rotate CW", [0, 90, 180, 270], index=0)
        cache_teacher = col_b.checkbox("Cache teacher", value=True)

        with st.expander("Concept explainer"):
            n_concepts = st.number_input("n_concepts", value=128, step=8)
            top_k = st.number_input("top_k (active/pixel)", value=8, step=1)
            fit_method = st.selectbox("fit_method", ["mics_input", "backprop", "through_mics"], index=0)
            weight_mode = st.selectbox("weight_mode", ["sparse_softmax", "softmax", "relu"], index=0)
            softmax_temperature = st.number_input("softmax_temperature", value=0.3, step=0.05, format="%.2f")
            train_epochs = st.number_input("train_epochs", value=300, step=50)
            train_lr = st.number_input("train_lr", value=0.01, step=0.005, format="%.3f")

        build = st.button("Build model", type="primary", use_container_width=True)
        if build:
            try:
                with st.spinner("Training MiCS teacher + fitting concept explainer… (first build is slow)"):
                    ss["bundle"] = get_bundle(
                        config_name,
                        npy_path,
                        mics_group,
                        int(seed),
                        int(num_samples),
                        sampling_mode,
                        normalization,
                        int(rotate_deg),
                        bool(cache_teacher),
                        int(n_concepts),
                        int(top_k),
                        fit_method,
                        weight_mode,
                        float(softmax_temperature),
                        int(train_epochs),
                        float(train_lr),
                    )
                ss["points"] = []
                ss["last_click"] = None
                ss["panel_path"] = None
                st.success("Model ready.")
            except Exception as exc:  # surfaced to UI instead of crashing the app
                st.exception(exc)

        st.header("3 · Panel style")
        top_terms = st.slider("Concepts per pixel (2×2 grid)", 2, 4, 4)
        concept_thumb_px = st.select_slider("Concept thumb px", [160, 240, 360, 480], value=360)
        dpi = st.select_slider("Export DPI", [150, 200, 300, 400], value=300)
        equalize_map = st.checkbox("Equalize map display", value=True)
        equalize_thumb = st.checkbox("Equalize concept thumbs", value=False)
        activation_colormap = st.selectbox("Activation colormap", ["magma", "assigned"], index=0)
        export_full = st.checkbox("Export full raw folder", value=True, help="Per-pixel folders + raw/annotated thumbnails + panels, like explain_mics_concepts.")
        export_thumb_px = st.select_slider("Export thumb px", [240, 360, 480, 640], value=480)

    if "bundle" not in ss:
        st.info("Configure a slide in the sidebar and press **Build model** to begin.")
        return

    bundle = ss["bundle"]

    # ------------------------------ Picker ------------------------------ #
    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.subheader("2 · Click pixels")
        st.caption(f"Slide: `{bundle['npy_name']}` · {bundle['shape'][0]}×{bundle['shape'][1]} px")
        disp_img, scale = _display_image_with_markers(bundle, ss["points"])
        click = streamlit_image_coordinates(disp_img, key="picker")
        if click is not None and click != ss["last_click"]:
            ss["last_click"] = click
            x = int(round(float(click["x"]) * scale))
            y = int(round(float(click["y"]) * scale))
            h, w = bundle["shape"]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            if not bundle["valid_mask"][y, x]:
                st.warning(f"({y}, {x}) is outside tissue — point added anyway.")
            ss["points"].append((y, x))
            st.rerun()

        c1, c2, c3 = st.columns(3)
        if c1.button("Undo last", use_container_width=True, disabled=not ss["points"]):
            ss["points"].pop()
            st.rerun()
        if c2.button("Clear all", use_container_width=True, disabled=not ss["points"]):
            ss["points"] = []
            st.rerun()
        gen = c3.button(
            "Generate panel",
            type="primary",
            use_container_width=True,
            disabled=not ss["points"],
        )

        if ss["points"]:
            st.write("**Selected pixels**")
            st.dataframe(
                {
                    "label": [_excel_label(i) for i in range(len(ss["points"]))],
                    "y": [p[0] for p in ss["points"]],
                    "x": [p[1] for p in ss["points"]],
                    "on tissue": [bool(bundle["valid_mask"][p[0], p[1]]) for p in ss["points"]],
                },
                use_container_width=True,
                hide_index=True,
            )
        if len(ss["points"]) > 5:
            st.info("Only the first 5 points get concept sub-panels; the rest are marked on the map.")

    with right:
        st.subheader("Interpretability panel")
        if gen:
            viz = {
                "concept_rank_mode": "relative",
                "top_terms": int(top_terms),
                "min_concept_weight": 0.04,
                "concept_thumb_px": int(concept_thumb_px),
                "export_thumb_px": int(export_thumb_px),
                "activation_percentile": 98.0,
                "activation_gamma": 0.85,
                "activation_colormap": str(activation_colormap),
                "map_fig_width": 26.0,
                "map_fig_height": 24.0,
                "map_marker_fontsize": 18.0,
                "map_marker_size": 14.0,
                "equalize_map_display": bool(equalize_map),
                "equalize_thumb_activation": bool(equalize_thumb),
                "dpi": int(dpi),
            }
            stem = Path(bundle["npy_name"]).stem
            run_dir = _OUTPUT_ROOT / f"{stem}_{datetime.now():%Y%m%d-%H%M%S}_{len(ss['points'])}pts"
            try:
                with st.spinner("Rendering panel…"):
                    ss["panel_path"] = render_panel(
                        bundle, ss["points"], run_dir / "concept_arrows_on_map.png", viz
                    )
                ss["run_dir"] = str(run_dir)
                ss["export_summary"] = None
                if bool(export_full):
                    with st.spinner("Exporting full raw folder (per-pixel thumbnails + panels)…"):
                        ss["export_summary"] = export_full_folder(bundle, ss["points"], run_dir, viz)
            except Exception as exc:
                st.exception(exc)

        if ss.get("panel_path") and Path(ss["panel_path"]).is_file():
            st.image(ss["panel_path"], use_container_width=True)
            with open(ss["panel_path"], "rb") as fh:
                st.download_button(
                    "Download panel PNG",
                    fh.read(),
                    file_name=Path(ss["panel_path"]).name,
                    mime="image/png",
                    use_container_width=True,
                )
            if ss.get("run_dir"):
                st.success("Saved output folder:")
                st.code(str(Path(ss["run_dir"]).resolve()), language="text")
                summary = ss.get("export_summary")
                if summary:
                    interp_dir = summary.get("interpretability_dir", "")
                    n_pix = len(summary.get("pixels", []))
                    st.caption(
                        f"Full raw export: {n_pix} pixel folder(s) under "
                        f"`{Path(interp_dir).name}/` (raw + annotated thumbnails, per-pixel panels, "
                        "and `pixels_manifest.json`)."
                    )
        else:
            st.caption("Pick points and press **Generate panel**.")


if __name__ == "__main__":
    main()
