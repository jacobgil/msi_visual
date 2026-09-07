#!/usr/bin/env python3
"""
MiCS m/z mask ladder: constant-depth PCC (default 3 layers) with learned per-m/z sigmoid gates.

Each stage trains a fresh model on MSI with x' = x ⊙ σ(logits) ⊙ fixed_mask, plus L1 on the
effective mask. L1 weight increases per stage; fixed_mask accumulates the product of prior
stage masks when ``gallery.cumulative_mask=true``.

Optional ``gallery.use_residuals=true``: stage k>1 trains a delta on top of the cumulative
embedding from prior stages (same mechanism as the residual depth ladder).

``gallery.baseline_stage_plain_pcc=true`` (default): stage 1 is unmasked 1-layer PCC to match
the residual ladder base; stages 2+ use the masked PCC at ``gallery.num_layers``.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from gallery_mics_sampling_cache import resolve_pixel_sampling_cache  # noqa: E402

logger = logging.getLogger(__name__)

_MICS_KEYS: tuple[str, ...] = (
    "number_of_points",
    "sampling",
    "num_epochs",
    "number_of_components",
    "lab_to_rgb",
    "cluster",
    "clusters",
    "num_samples",
    "beta",
    "k_epoch",
    "temperature",
    "lr",
    "batch_size",
    "warmup_epochs",
    "factor",
    "random_state",
    "verbose",
    "cluster_loss_weight",
    "category_loss_weight",
    "pixel_sampling",
    "pca_fit_step",
    "cluster_on_pca",
    "cluster_pca_dims",
    "predict_percentile_low",
    "predict_percentile_high",
    "predict_spatial_smooth_sigma",
    "clusters_auto_tune",
)


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    s = msi.sum(axis=-1, keepdims=True)
    return msi / (s + 1e-8)


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(str(path), mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI HxWxC, got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI must be nonnegative.")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def _parse_clusters(val: Any) -> list[int]:
    if val is None:
        return [1024, 2048, 4096]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip()]
    return [int(s)]


def _mics_kwargs_from_cfg(
    cfg: DictConfig,
    num_layers: int,
    seed: int,
    ga: Any | None = None,
    *,
    plain_pcc: bool = False,
) -> dict[str, Any]:
    merged = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    kw: dict[str, Any] = {}
    for name in _MICS_KEYS:
        if name in ("num_layers", "clusters"):
            continue
        if name in merged:
            kw[name] = merged[name]
    kw.setdefault("number_of_points", int(merged.get("number_of_points", 1000)))
    kw.setdefault("sampling", str(merged.get("sampling", "coreset")))
    kw.setdefault("num_epochs", int(merged.get("num_epochs", 30)))
    kw.setdefault("number_of_components", int(merged.get("number_of_components", 3)))
    kw.setdefault("lab_to_rgb", bool(merged.get("lab_to_rgb", True)))
    kw.setdefault("cluster", bool(merged.get("cluster", True)))
    kw.setdefault("clusters", _parse_clusters(merged.get("clusters", [1024, 2048, 4096])))
    kw.setdefault("num_samples", int(merged.get("num_samples", 5000)))
    kw.setdefault("beta", float(merged.get("beta", 20.0)))
    kw.setdefault("k_epoch", int(merged.get("k_epoch", 20)))
    kw.setdefault("temperature", float(merged.get("temperature", 100.0)))
    kw.setdefault("lr", float(merged.get("lr", 1.0)))
    kw.setdefault("batch_size", int(merged.get("batch_size", 1024)))
    kw.setdefault("warmup_epochs", int(merged.get("warmup_epochs", 10)))
    kw.setdefault("factor", float(merged.get("factor", 1.0)))
    kw.setdefault("verbose", bool(merged.get("verbose", False)))
    kw.setdefault("pixel_sampling", str(merged.get("pixel_sampling", "superpixel")))
    kw.setdefault("predict_spatial_smooth_sigma", float(merged.get("predict_spatial_smooth_sigma", 0.8)))
    kw["clusters_auto_tune"] = None
    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw["num_layers"] = max(1, int(num_layers))
    kw["random_state"] = int(seed)
    if plain_pcc:
        kw["learn_input_mask"] = False
        kw["input_mask_l1_weight"] = 0.0
    else:
        kw["learn_input_mask"] = True
        ga_init = (
            OmegaConf.select(ga, "input_mask_logits_init", default=None) if ga is not None else None
        )
        if ga_init is not None:
            kw["input_mask_logits_init"] = float(ga_init)
        else:
            kw.setdefault(
                "input_mask_logits_init", float(merged.get("input_mask_logits_init", 4.0))
            )
    return kw


def _l1_schedule(ga: Any, n_stages: int) -> list[float]:
    raw_sel = OmegaConf.select(ga, "l1_ladder", default=None)
    if raw_sel is not None and not (
        isinstance(raw_sel, str) and raw_sel.strip().lower() in ("", "~", "null", "none")
    ):
        if OmegaConf.is_config(raw_sel):
            raw = OmegaConf.to_container(raw_sel, resolve=True)
        else:
            raw = raw_sel
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            return [max(0.0, float(x)) for x in raw]

    lo = float(OmegaConf.select(ga, "l1_start", default=0.0))
    hi = float(OmegaConf.select(ga, "l1_end", default=0.01))
    mode = str(OmegaConf.select(ga, "l1_schedule", default="linear")).strip().lower()
    if n_stages <= 1:
        return [lo]
    if mode == "geometric" and hi > 0 and lo > 0:
        return [float(x) for x in np.geomspace(lo, hi, n_stages)]
    return [float(x) for x in np.linspace(lo, hi, n_stages)]


def _compute_norm_anchor(
    emb: np.ndarray, mask: np.ndarray, low: float, high: float
) -> list[tuple[float, float]]:
    anchor: list[tuple[float, float]] = []
    m = np.asarray(mask, dtype=bool)
    for c in range(int(emb.shape[-1])):
        vals = np.asarray(emb[..., c], dtype=np.float64)[m]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            anchor.append((0.0, 1.0))
        else:
            anchor.append((float(np.percentile(vals, low)), float(np.percentile(vals, high))))
    return anchor


def _match_embedding_intensity(
    emb: np.ndarray,
    ref: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gain: float = 1.0,
) -> np.ndarray:
    """Per-channel std match so shared percentile stretch keeps similar brightness."""
    out = np.asarray(emb, dtype=np.float32).copy()
    m = np.asarray(valid_mask, dtype=bool)
    if not np.any(m):
        return out
    g = float(gain)
    for c in range(int(out.shape[-1])):
        e = np.asarray(out[..., c], dtype=np.float64)[m]
        r = np.asarray(ref[..., c], dtype=np.float64)[m]
        e = e[np.isfinite(e)]
        r = r[np.isfinite(r)]
        if e.size == 0 or r.size == 0:
            continue
        std_e = float(np.std(e))
        std_r = float(np.std(r))
        if std_e < 1e-12:
            continue
        out[..., c] *= g * (std_r / std_e)
    return out


def _embedding_to_rgb_with_anchor(
    emb: np.ndarray,
    valid_mask: np.ndarray,
    anchor: list[tuple[float, float]],
    *,
    lab_to_rgb: bool,
    n_components: int,
    smooth_sigma: float,
) -> np.ndarray:
    import cv2

    from msi_visual.parametric_mics_lmc import _spatial_gaussian_smooth_channels

    result = np.asarray(emb, dtype=np.float32).copy()
    m = np.asarray(valid_mask, dtype=bool)
    for c, (p_lo, p_hi) in enumerate(anchor):
        ch = result[..., c]
        ch = (ch - p_lo) / max(p_hi - p_lo, 1e-8)
        ch = np.clip(ch, 0.0, 1.0)
        result[..., c] = ch
    result[~m] = 0.0
    result = _spatial_gaussian_smooth_channels(result, smooth_sigma)
    out = np.uint8(255.0 * result)
    out[~m] = 0
    if lab_to_rgb and n_components == 3:
        out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    out[~m] = 0
    return out


def _hist_equalize_rgb(viz_rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2

        out = np.asarray(viz_rgb, dtype=np.uint8).copy()
        for ch in range(min(3, out.shape[-1])):
            out[:, :, ch] = cv2.equalizeHist(out[:, :, ch])
        return out
    except Exception:
        return viz_rgb


def _save_mask_diagnostic(
    path: Path,
    mz_mask: np.ndarray,
    *,
    l1_w: float,
    stage_idx: int,
    top_k: int,
) -> None:
    m = np.asarray(mz_mask, dtype=np.float64).ravel()
    n_active = int(np.sum(m > 0.5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(m, lw=0.6)
    axes[0].set_title(f"stage {stage_idx + 1} effective m/z mask (L1={l1_w:g})")
    axes[0].set_xlabel("m/z index")
    axes[0].set_ylabel("σ(logit)×fixed")
    axes[0].set_ylim(-0.02, 1.02)

    k = min(int(top_k), m.size)
    if k > 0:
        order = np.argsort(-m)[:k]
        axes[1].bar(np.arange(k), m[order], width=0.85)
        axes[1].set_title(f"top {k} m/z (active>{n_active} @0.5)")
        axes[1].set_xlabel("rank")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_mask_ladder")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ga = getattr(cfg, "gallery", None)
    if ga is None:
        raise ValueError("Config must define gallery:")

    npy_path = Path(to_absolute_path(str(ga.npy_path))).expanduser().resolve()
    if not npy_path.is_file():
        raise FileNotFoundError(f"gallery.npy_path not found: {npy_path}")

    msi = _load_msi(npy_path, bool(ga.transpose_msi), bool(ga.tic_normalize))
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("Empty MSI valid mask.")

    n_stages = max(1, int(OmegaConf.select(ga, "max_stages", default=10)))
    num_layers = max(1, int(OmegaConf.select(ga, "num_layers", default=3)))
    l1_weights = _l1_schedule(ga, n_stages)
    cumulative = bool(OmegaConf.select(ga, "cumulative_mask", default=True))
    baseline_plain = bool(OmegaConf.select(ga, "baseline_stage_plain_pcc", default=True))
    baseline_layers = max(1, int(OmegaConf.select(ga, "baseline_num_layers", default=1)))
    baseline_temp_raw = OmegaConf.select(ga, "baseline_temperature", default=None)
    baseline_temperature: float | None = None
    if baseline_temp_raw is not None and str(baseline_temp_raw).strip().lower() not in (
        "",
        "~",
        "null",
        "none",
    ):
        baseline_temperature = float(baseline_temp_raw)
    use_residuals = bool(OmegaConf.select(ga, "use_residuals", default=True))
    residual_lr_mul = float(OmegaConf.select(ga, "residual_lr_multiplier", default=1.0))
    base_seed = int(OmegaConf.select(ga, "seed", default=42))
    eq = bool(OmegaConf.select(ga, "equalize_visualization", default=False))
    shared_norm = bool(OmegaConf.select(ga, "shared_normalization", default=True))
    intensity_match = bool(OmegaConf.select(ga, "display_intensity_match", default=True))
    intensity_gain = float(OmegaConf.select(ga, "display_intensity_gain", default=1.0))
    top_k = int(OmegaConf.select(ga, "mask_plot_top_k", default=48))
    pred_lo = float(OmegaConf.select(cfg.model, "predict_percentile_low", default=1.0))
    pred_hi = float(OmegaConf.select(cfg.model, "predict_percentile_high", default=99.0))
    ncols = max(1, int(OmegaConf.select(ga, "ncols", default=3)))
    dpi = float(OmegaConf.select(ga, "dpi", default=160))
    fig_w = float(OmegaConf.select(ga, "figure_size_inches", default=14))
    title_fs = float(OmegaConf.select(ga, "panel_title_fontsize", default=10))

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    pan_dir = out_dir / "panels"
    pan_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )

    share_sampling = bool(OmegaConf.select(ga, "share_pixel_sampling", default=True))
    cache_path_raw = OmegaConf.select(ga, "pixel_sampling_cache_path", default=None)
    cache_path: Path | None = None
    if cache_path_raw is not None and str(cache_path_raw).strip().lower() not in (
        "",
        "~",
        "null",
        "none",
    ):
        cache_path = Path(to_absolute_path(str(cache_path_raw))).expanduser().resolve()

    cache_layers = baseline_layers if baseline_plain else num_layers
    sampling_kw = _mics_kwargs_from_cfg(
        cfg, cache_layers, base_seed, ga=ga, plain_pcc=baseline_plain
    )
    pixel_cache = resolve_pixel_sampling_cache(
        msi,
        sampling_kw,
        out_dir=out_dir,
        share=share_sampling,
        cache_path=cache_path,
    )

    n_mz = int(msi.shape[-1])
    fixed_mask = np.ones(n_mz, dtype=np.float32)
    cumulative_emb: np.ndarray | None = None
    trained: list[dict[str, Any]] = []

    for idx, l1_w in enumerate(l1_weights):
        is_baseline = baseline_plain and idx == 0
        stage_layers = baseline_layers if is_baseline else num_layers
        kw = _mics_kwargs_from_cfg(
            cfg,
            stage_layers,
            seed=base_seed + idx,
            ga=ga,
            plain_pcc=is_baseline,
        )
        if is_baseline:
            if baseline_temperature is not None:
                kw["temperature"] = baseline_temperature
        else:
            kw["input_mask_l1_weight"] = float(l1_w)
        if use_residuals and idx > 0 and residual_lr_mul != 1.0:
            kw["lr"] = float(kw["lr"]) * residual_lr_mul
        logger.info(
            "Stage %d/%d | layers=%d | plain_pcc=%s | L1=%g | residual=%s | fixed_mask mean=%.3f",
            idx + 1,
            len(l1_weights),
            stage_layers,
            "yes" if is_baseline else "no",
            0.0 if is_baseline else l1_w,
            "yes" if use_residuals and cumulative_emb is not None else "no",
            float(np.mean(fixed_mask)),
        )
        model: MSIParametricMiCSLMC | None = None
        try:
            model = MSIParametricMiCSLMC(**kw)
            if not is_baseline:
                model.set_fixed_input_mask(fixed_mask)
            if use_residuals:
                if cumulative_emb is not None:
                    model.set_residual_base(cumulative_emb)
                else:
                    model.clear_residual_base()
            model.fit(msi, pixel_sampling_cache=pixel_cache)
            masks = model.get_input_mask_arrays()
            viz = model.predict(msi)
            emb = model.predict_embedding(msi).astype(np.float32, copy=False)
            if use_residuals:
                delta_emb = model.predict_delta(msi).astype(np.float32, copy=False)
                prev_emb = cumulative_emb
                inc_emb = emb - prev_emb if prev_emb is not None else delta_emb
                vm = valid_mask
                rms_delta = (
                    float(np.sqrt(np.mean(np.sum(delta_emb[vm] ** 2, axis=-1))))
                    if np.any(vm)
                    else 0.0
                )
                rms_inc = (
                    float(np.sqrt(np.mean(np.sum(inc_emb[vm] ** 2, axis=-1))))
                    if np.any(vm)
                    else 0.0
                )
                rms_cum = (
                    float(np.sqrt(np.mean(np.sum(emb[vm] ** 2, axis=-1))))
                    if np.any(vm)
                    else 0.0
                )
                rel = rms_inc / max(rms_cum, 1e-8)
                logger.info(
                    "Stage %d embed RMS: |delta|=%.4g |increment|=%.4g |cum|=%.4g rel_inc=%.3f%%",
                    idx + 1,
                    rms_delta,
                    rms_inc,
                    rms_cum,
                    100.0 * rel,
                )
                cumulative_emb = emb
            eff = masks["effective"]
            stg = masks["stage"]
            n_active = int(np.sum(eff > 0.5))
            mean_mask = float(np.mean(eff))
            logger.info(
                "Stage %d mask: mean=%.4f active@0.5=%d/%d",
                idx + 1,
                mean_mask,
                n_active,
                n_mz,
            )
            trained.append(
                {
                    "idx": idx,
                    "l1_w": l1_w,
                    "plain_pcc": is_baseline,
                    "kw": kw,
                    "viz": viz,
                    "emb": emb,
                    "effective_mask": eff,
                    "stage_mask": stg,
                    "n_active": n_active,
                    "mean_mask": mean_mask,
                }
            )
            if not is_baseline:
                if cumulative:
                    fixed_mask = (fixed_mask * stg).astype(np.float32)
                else:
                    fixed_mask = stg.astype(np.float32)
        finally:
            if model is not None:
                try:
                    model.release_resources()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ref_emb = trained[0]["emb"]
    norm_anchor = _compute_norm_anchor(ref_emb, valid_mask, pred_lo, pred_hi)
    panel_rgbs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for rec in trained:
        idx = int(rec["idx"])
        kw = rec["kw"]
        emb = np.asarray(rec["emb"], dtype=np.float32)
        if shared_norm and intensity_match and idx > 0:
            emb = _match_embedding_intensity(
                emb, ref_emb, valid_mask, gain=intensity_gain
            )
        if shared_norm:
            rgb = _embedding_to_rgb_with_anchor(
                emb,
                valid_mask,
                norm_anchor,
                lab_to_rgb=bool(kw.get("lab_to_rgb", True)),
                n_components=int(kw.get("number_of_components", 3)),
                smooth_sigma=float(kw.get("predict_spatial_smooth_sigma", 0.0)),
            )
        else:
            rgb = np.asarray(rec["viz"], dtype=np.uint8)[..., :3]
        if eq:
            rgb = _hist_equalize_rgb(rgb)

        stage_l = int(kw.get("num_layers", num_layers))
        l1_lbl = "base" if rec.get("plain_pcc") else f"l1_{rec['l1_w']:g}"
        viz_path = pan_dir / f"panel_{idx:02d}__L{stage_l}__{l1_lbl}__viz.png"
        Image.fromarray(rgb, mode="RGB").save(viz_path)
        mask_path = pan_dir / f"panel_{idx:02d}__mz_mask.png"
        _save_mask_diagnostic(mask_path, rec["effective_mask"], l1_w=rec["l1_w"], stage_idx=idx, top_k=top_k)
        np.save(pan_dir / f"panel_{idx:02d}__mz_mask.npy", rec["effective_mask"])

        panel_rgbs.append(rgb)
        rows.append(
            {
                "stage": idx,
                "l1": rec["l1_w"],
                "mean_mask": rec["mean_mask"],
                "n_active": rec["n_active"],
                "viz": str(viz_path),
                "mask_plot": str(mask_path),
            }
        )

    n_p = len(panel_rgbs)
    nrows = int(np.ceil(n_p / ncols))
    mh = float(max(fig_w * 0.35, fig_w * 0.28 * max(1, nrows)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, mh))
    if nrows == 1 and ncols == 1:
        ax_grid = np.array([[axes]])
    elif nrows == 1:
        ax_grid = np.asarray(axes).reshape(1, -1)
    elif ncols == 1:
        ax_grid = np.asarray(axes).reshape(-1, 1)
    else:
        ax_grid = np.asarray(axes)

    for idx, rgb in enumerate(panel_rgbs):
        r, c = divmod(idx, ncols)
        ax = ax_grid[r, c]
        ax.imshow(rgb)
        row = rows[idx]
        rec_plain = bool(trained[idx].get("plain_pcc"))
        if rec_plain:
            tit = f"stage {idx + 1}: L{trained[idx]['kw']['num_layers']} base (plain PCC)"
        else:
            res_lbl = " +res" if use_residuals and idx > 0 else ""
            tit = (
                f"stage {idx + 1}: L1={row['l1']:g}{res_lbl}\n"
                f"mask mean={row['mean_mask']:.3f} active={row['n_active']}"
            )
        ax.set_title(tit, fontsize=title_fs)
        ax.axis("off")
    for idx in range(n_p, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax_grid[r, c].axis("off")

    res_title = ", residual↑" if use_residuals else ""
    fig.suptitle(
        f"MiCS mask ladder ({num_layers} layers, L1↑{res_title})",
        fontsize=title_fs + 2,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    mosaic = out_dir / "gallery_mics_mask_ladder.png"
    fig.savefig(mosaic, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary = out_dir / "mask_ladder_summary.txt"
    with summary.open("w", encoding="utf-8") as fh:
        fh.write(f"npy_path: {npy_path}\n")
        fh.write(f"num_layers: {num_layers}\n")
        fh.write(f"l1_weights: {l1_weights}\n")
        fh.write(f"cumulative_mask: {cumulative}\n")
        fh.write(f"baseline_stage_plain_pcc: {baseline_plain}\n")
        if baseline_plain:
            fh.write(f"baseline_num_layers: {baseline_layers}\n")
            fh.write(f"baseline_temperature: {baseline_temperature}\n")
        fh.write(f"shared_normalization: {shared_norm}\n")
        fh.write(f"display_intensity_match: {intensity_match}\n")
        fh.write(f"display_intensity_gain: {intensity_gain}\n")
        fh.write(f"use_residuals: {use_residuals}\n")
        if use_residuals:
            fh.write(f"residual_lr_multiplier: {residual_lr_mul}\n")
        fh.write("\n")
        for i, row in enumerate(rows):
            rec = trained[i]
            if rec.get("plain_pcc"):
                fh.write(
                    f"stage {row['stage']}: plain_pcc L{rec['kw']['num_layers']} "
                    f"T={rec['kw'].get('temperature')}\n"
                )
            else:
                fh.write(
                    f"stage {row['stage']}: L1={row['l1']:g} "
                    f"mask_mean={row['mean_mask']:.6f} active@0.5={row['n_active']}\n"
                )
            fh.write(f"  viz: {row['viz']}\n")
            fh.write(f"  mask: {row['mask_plot']}\n")
        fh.write(f"\nmosaic: {mosaic}\n")

    logger.info("Gallery: %s", mosaic)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
