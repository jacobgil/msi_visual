#!/usr/bin/env python3
"""One-shot: MiCS preset gallery + explain_mics_concepts for NRL4509 s16-r4-5um (5/10/20 bins)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
BASE = Path(r"E:/MSImaging-data/_msi_visual/Extractions/NRL4509/s16-r4-5um")
OUT_ROOT = ROOT / "outputs" / "interpretability"
MICS_GALLERY_ROOT = ROOT / "outputs" / "mics_param_gallery" / "NRL4509_s16-r4-5um"

# Named presets from batch_extract_mics_groups.yaml (same family as ID9 / mics_multiscale).
MICS_GROUPS = [
    "mics_multiscale",
    "mics_std",
    "mics_beta0",
    "mics_beta1",
    "mics_shallow",
    "mics_deep",
    "mics_coarse",
    "mics_pca",
    "mics_fine",
]
BINS = ["5_bins", "10_bins", "20_bins"]
N_POINTS = 100

CONCEPT_OVERRIDES = [
    "concepts.weight_mode=sparse_softmax",
    "concepts.top_k=8",
    "concepts.sparsity_weight=1000",
    "concepts.softmax_temperature=0.3",
    "concepts.diagnostic_sparse_top_k=4",
]


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_mics_gallery(bins_name: str) -> Path:
    npy = BASE / bins_name / "0.npy"
    if not npy.is_file():
        raise FileNotFoundError(npy)
    out_dir = MICS_GALLERY_ROOT / bins_name
    out_dir.mkdir(parents=True, exist_ok=True)
    groups_csv = ",".join(MICS_GROUPS)
    cmd = [
        PY,
        str(ROOT / "scripts" / "generate_visualization_panel.py"),
        f"data.npy_path={npy.as_posix()}",
        "methods=[]",
        f"mics_groups=[{groups_csv}]",
        "mics_benchmark_config=scripts/configs/batch_extract_mics_groups.yaml",
        "mics_model={}",
        "mics_parameter_group=null",
        "benchmark.sampling_mode=superpixel",
        "benchmark.num_samples=5000",
        f"output.dir={out_dir.as_posix()}",
        f"output.stem=NRL4509_s16-r4-5um_{bins_name}",
        "output.save_panel=true",
        "output.save_equalized=true",
        "output.columns=3",
        f"hydra.run.dir={out_dir.as_posix()}/_hydra",
    ]
    _run(cmd)
    return out_dir


def run_explain(bins_name: str) -> Path:
    npy = BASE / bins_name / "0.npy"
    run_dir = OUT_ROOT / f"NRL4509_s16-r4-5um_{bins_name}"
    cmd = [
        PY,
        str(ROOT / "scripts" / "explain_mics_concepts.py"),
        f"gallery.npy_path={npy.as_posix()}",
        "mics.group=mics_multiscale",
        "gallery.cache_teacher=false",
        "gallery.sampling_mode=superpixel",
        *CONCEPT_OVERRIDES,
        f"viz.n_example_pixels={N_POINTS}",
        f"hydra.run.dir={run_dir.as_posix()}",
    ]
    _run(cmd)
    return run_dir


def regen_100_if_needed(run_dir: Path) -> None:
    """Ensure 100 pixel folders + per-pixel maps (in case explain used fewer)."""
    sys.path.insert(0, str(ROOT))
    from msi_visual.mics_concept_explain_viz import save_explanation_visualizations

    if not (run_dir / "alpha_hwk.npy").is_file():
        print("SKIP regen missing alpha", run_dir, flush=True)
        return
    pixels = list((run_dir / "interpretability").glob("pixel_*")) if (run_dir / "interpretability").is_dir() else []
    n_maps = sum(1 for p in pixels if (p / "concept_arrows_on_map.png").is_file())
    if len(pixels) >= N_POINTS and n_maps >= N_POINTS:
        print(f"OK {run_dir.name}: {len(pixels)} pixels, {n_maps} maps", flush=True)
        return

    print(f"=== regen {run_dir.name} to {N_POINTS} ===", flush=True)
    cfg = OmegaConf.load(run_dir / "config_resolved.yaml")
    vz = cfg.viz if hasattr(cfg, "viz") else {}
    alpha = np.load(run_dir / "alpha_hwk.npy")
    concept_rgbs = np.load(run_dir / "concept_colors.npy")
    mics_rgb = np.asarray(Image.open(run_dir / "mics_teacher.png").convert("RGB"), dtype=np.uint8)
    recon_rgb = np.asarray(Image.open(run_dir / "mics_reconstruction.png").convert("RGB"), dtype=np.uint8)
    valid_mask = np.any(alpha > 1e-8, axis=-1) | np.any(mics_rgb > 0, axis=-1)
    interp = run_dir / "interpretability"
    if interp.is_dir():
        for d in interp.glob("pixel_*"):
            if d.is_dir():
                shutil.rmtree(d)
    seed = int(OmegaConf.select(cfg, "seed", default=17))
    top_k = int(OmegaConf.select(cfg, "concepts.top_k", default=8))
    viz = save_explanation_visualizations(
        mics_rgb,
        recon_rgb,
        alpha,
        concept_rgbs,
        valid_mask,
        run_dir,
        teacher_emb=None,
        n_concepts=int(alpha.shape[-1]),
        top_k=top_k,
        pixel_rgb_mse=0.0,
        embedding_mse=0.0,
        n_example_pixels=N_POINTS,
        dpi=int(OmegaConf.select(vz, "dpi", default=200)),
        seed=seed,
        concept_node_style=str(OmegaConf.select(vz, "concept_node_style", default="heatmap")),
        concept_thumb_px=int(OmegaConf.select(vz, "concept_thumb_px", default=360)),
        map_fig_width=float(OmegaConf.select(vz, "map_fig_width", default=26.0)),
        map_fig_height=float(OmegaConf.select(vz, "map_fig_height", default=24.0)),
        map_marker_fontsize=float(OmegaConf.select(vz, "map_marker_fontsize", default=14.0)),
        map_marker_size=float(OmegaConf.select(vz, "map_marker_size", default=10.0)),
        equalize_map_display=bool(OmegaConf.select(vz, "equalize_map_display", default=True)),
        equalize_thumb_activation=bool(OmegaConf.select(vz, "equalize_thumb_activation", default=False)),
        save_both_display_variants=bool(OmegaConf.select(vz, "save_both_display_variants", default=True)),
        example_pixel_spatial_weight=float(OmegaConf.select(vz, "example_pixel_spatial_weight", default=2.5)),
        example_pixel_interior_min_px=float(OmegaConf.select(vz, "example_pixel_interior_min_px", default=28.0)),
        example_pixel_interior_weight=float(OmegaConf.select(vz, "example_pixel_interior_weight", default=5.0)),
        activation_percentile=float(OmegaConf.select(vz, "activation_percentile", default=98.0)),
        activation_gamma=float(OmegaConf.select(vz, "activation_gamma", default=0.85)),
        activation_colormap=str(OmegaConf.select(vz, "activation_colormap", default="magma")),
        concept_rank_mode=str(OmegaConf.select(vz, "concept_rank_mode", default="relative")),
        min_concept_weight=float(OmegaConf.select(vz, "min_concept_weight", default=0.04)),
        top_terms=int(OmegaConf.select(vz, "top_terms", default=4)),
        export_interpretability=True,
        interpretability_subdir="interpretability",
        export_dpi=int(OmegaConf.select(vz, "export_dpi", default=300)),
        export_thumb_px=int(OmegaConf.select(vz, "export_thumb_px", default=480)),
    )
    if hasattr(cfg, "viz"):
        cfg.viz.n_example_pixels = N_POINTS
        (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    man_path = run_dir / "explain_mics_concepts_manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man.setdefault("explanation_viz", {})["example_pixels"] = viz.get("example_pixels", [])
        man["explanation_viz"]["n_example_pixels"] = N_POINTS
        if viz.get("interpretability") is not None:
            man["interpretability"] = viz["interpretability"]
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    pixels = list((run_dir / "interpretability").glob("pixel_*"))
    n_maps = sum(1 for p in pixels if (p / "concept_arrows_on_map.png").is_file())
    print(f"done regen {run_dir.name}: pixels={len(pixels)} maps={n_maps}", flush=True)


def main() -> None:
    skip_gallery = "--skip-gallery" in sys.argv
    skip_explain = "--skip-explain" in sys.argv
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = a.split("=", 1)[1]

    bins_list = [only] if only else list(BINS)
    for bins_name in bins_list:
        print(f"\n========== {bins_name} ==========", flush=True)
        if not skip_gallery:
            gdir = run_mics_gallery(bins_name)
            print("MiCS gallery:", gdir, flush=True)
        if not skip_explain:
            rdir = run_explain(bins_name)
            regen_100_if_needed(rdir)
            print("Interpretability:", rdir, flush=True)
    print("\nALL NRL4509 JOBS DONE", flush=True)


if __name__ == "__main__":
    main()
