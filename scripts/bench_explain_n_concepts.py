#!/usr/bin/env python3
"""Compare sparse concept explain quality for different n_concepts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from debug_edge_maps import _load_msi  # noqa: E402
from msi_visual.mics_concept_distill import embedding_channel_bounds, embedding_to_display_rgb  # noqa: E402
from msi_visual.mics_concept_explain import ConceptExplainConfig, FrozenConceptExplainer  # noqa: E402


def _masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    d = np.asarray(a, dtype=np.float64)[m] - np.asarray(b, dtype=np.float64)[m]
    return float(np.mean(d * d))


def main() -> None:
    cache = np.load(
        _REPO / "outputs/gallery_mics_concept_distill/_teacher_cache/1__mics_multiscale__2e13baf69a30df30.npz"
    )
    teacher_emb = cache["teacher_emb"]
    teacher_rgb = cache["teacher_rgb"]
    msi = _load_msi(
        "E:/MSImaging-data/_msi_visual/Extractions/NRL4485-s2/5_bins/1.npy",
        transpose_msi=False,
        normalization="tic",
    )
    valid = msi.sum(axis=-1) > 0

    base = dict(
        top_k=8,
        fit_method="backprop",
        weight_mode="sparse_softmax",
        softmax_temperature=0.3,
        train_epochs=300,
        train_lr=0.01,
        seed=42,
        device="cpu",
        nmf_init_pixels=12000,
    )
    rows: list[dict] = []
    for n_concepts in (32, 64, 128):
        nmf_iter = 400 if n_concepts <= 64 else 500
        cfg = ConceptExplainConfig(**base, n_concepts=int(n_concepts), nmf_max_iter=int(nmf_iter))
        print(f"Fitting n_concepts={n_concepts} ...", flush=True)
        ex = FrozenConceptExplainer.from_mics(msi, teacher_emb, valid, cfg)
        recon_emb = ex.predict_embedding()
        recon_rgb = ex.predict_rgb(valid)
        emb_mse = _masked_mse(teacher_emb, recon_emb, valid)
        rgb_mse = _masked_mse(teacher_rgb, recon_rgb, valid)
        alpha = ex.alpha_hwk[valid]
        row = {
            "n_concepts": int(n_concepts),
            "embedding_mse": emb_mse,
            "pixel_rgb_mse": rgb_mse,
            "ridge_readout_mse": float(ex.fit_stats.get("ridge_readout_mse", float("nan"))),
            "backprop_emb_mse": float(ex.fit_stats.get("backprop_emb_mse", float("nan"))),
            "mean_top1_weight": float(alpha.max(axis=1).mean()),
            "mean_nnz_above_1e-4": float((alpha > 1e-4).sum(axis=1).mean()),
        }
        rows.append(row)
        print(
            f"  K={n_concepts:3d}  rgb_mse={rgb_mse:8.1f}  emb_mse={emb_mse:12.0f}  "
            f"ridge={row['ridge_readout_mse']:.0f}  top1={row['mean_top1_weight']:.1%}",
            flush=True,
        )

    best = min(rows, key=lambda r: r["pixel_rgb_mse"])
    out = {
        "sparse_settings": base,
        "results": rows,
        "best_n_concepts_by_rgb_mse": int(best["n_concepts"]),
    }
    out_path = _REPO / "outputs/explain_mics_concepts/bench_n_concepts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nBest by RGB MSE: K={best['n_concepts']} ({best['pixel_rgb_mse']:.1f})", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
