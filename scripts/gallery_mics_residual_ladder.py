#!/usr/bin/env python3
"""
Progressive MiCS residual ladder gallery.

**capacity** (legacy): L1, L2, … — each stage adds a depth residual to frozen cumulative emb.

**shared_local**: cross-region **shared** base (PCA or shallow clustered MiCS) plus **local**
residual stages ``E = E_shared + Σ Δ_local`` with frozen shared base throughout.

``gallery.pca_mics_morph=true``: display blends PCA → cumulative MiCS (capacity mode).

Outputs ``gallery_shared_local_decomposition.png`` (shared | local | combined) in shared_local mode.
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
    "num_layers",
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


def _pca_embedding_hwc(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    from sklearn.decomposition import PCA

    x = np.asarray(msi[valid_mask], dtype=np.float32)
    if x.size == 0:
        raise ValueError("No valid pixels for PCA coarse embedding.")
    n_comp = max(1, min(int(n_components), int(x.shape[0]), int(x.shape[1])))
    pca = PCA(n_components=n_comp, random_state=int(random_state), svd_solver="randomized")
    z = pca.fit_transform(x).astype(np.float32, copy=False)
    out = np.zeros((msi.shape[0], msi.shape[1], int(n_components)), dtype=np.float32)
    out[valid_mask, :n_comp] = z
    return out


def _match_channels_to_ref(
    ref: np.ndarray, src: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Per-channel mean/std match so PCA↔MiCS morph does not wash out contrast."""
    out = np.asarray(src, dtype=np.float32).copy()
    m = np.asarray(valid_mask, dtype=bool)
    if not np.any(m):
        return out
    ref_a = np.asarray(ref, dtype=np.float64)
    for c in range(int(out.shape[-1])):
        r = ref_a[..., c][m]
        s = np.asarray(out[..., c], dtype=np.float64)[m]
        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]
        if r.size == 0 or s.size == 0:
            continue
        std_s = float(np.std(s))
        std_r = float(np.std(r))
        if std_s < 1e-12:
            continue
        mean_s = float(np.mean(s))
        mean_r = float(np.mean(r))
        out[..., c] = (out[..., c] - mean_s) * (std_r / std_s) + mean_r
    return out


def _blend_pca_mics(
    pca_emb: np.ndarray,
    mics_emb: np.ndarray,
    alpha: float,
    valid_mask: np.ndarray,
    *,
    scale_align: bool,
) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0:
        return np.asarray(pca_emb, dtype=np.float32).copy()
    if a >= 1.0:
        return np.asarray(mics_emb, dtype=np.float32).copy()
    mics_use = (
        _match_channels_to_ref(pca_emb, mics_emb, valid_mask) if scale_align else mics_emb
    )
    return ((1.0 - a) * pca_emb + a * mics_use).astype(np.float32, copy=False)


def _resolve_morph_alpha_ladder(ga: Any, n_stages: int) -> list[float]:
    raw_sel = OmegaConf.select(ga, "morph_alpha_ladder", default=None)
    if raw_sel is not None and not (
        isinstance(raw_sel, str) and raw_sel.strip().lower() in ("", "~", "null", "none")
    ):
        alphas = [
            float(x)
            for x in _resolve_stage_value_ladder(
                ga, "morph_alpha_ladder", n_stages, 0.0, cast=float, min_value=0.0
            )
        ]
        return [float(np.clip(a, 0.0, 1.0)) for a in alphas]

    lo = float(OmegaConf.select(ga, "morph_alpha_start", default=0.0))
    hi = float(OmegaConf.select(ga, "morph_alpha_end", default=1.0))
    lo, hi = float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))
    if n_stages <= 1:
        return [lo]
    mode = str(OmegaConf.select(ga, "morph_schedule", default="linear")).strip().lower()
    if mode == "geometric" and lo > 0 and hi > 0:
        return [float(x) for x in np.geomspace(lo, hi, n_stages)]
    return [float(x) for x in np.linspace(lo, hi, n_stages)]


def _tic_normalize(msi: np.ndarray) -> np.ndarray:
    s = msi.sum(axis=-1, keepdims=True)
    return msi / (s + 1e-8)


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(str(path), mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI HxWxC, got shape {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if np.nanmin(img) < 0:
        raise ValueError("MSI must be nonnegative.")
    if tic_normalize:
        img = _tic_normalize(img)
    return img


def _to_uint8_rgb(viz: Any) -> np.ndarray:
    arr = np.asarray(viz)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr
    out = arr.astype(np.float32)
    mx = float(np.nanmax(out)) if out.size else 0.0
    if mx <= 1.0 + 1e-6:
        out = (out / mx * 255.0) if mx > 1e-12 else np.zeros_like(out)
    return np.clip(out, 0, 255).astype(np.uint8)


def _hist_equalize_rgb(viz_rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2

        out = np.asarray(viz_rgb, dtype=np.uint8).copy()
        for ch in range(min(3, out.shape[-1])):
            out[:, :, ch] = cv2.equalizeHist(out[:, :, ch])
        return out
    except Exception:
        return viz_rgb


def _compute_norm_anchor(
    emb: np.ndarray,
    mask: np.ndarray,
    low: float,
    high: float,
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


def _embedding_to_rgb_with_anchor(
    emb: np.ndarray,
    mask: np.ndarray,
    anchor: list[tuple[float, float]],
    *,
    lab_to_rgb: bool,
    n_components: int,
    smooth_sigma: float,
) -> np.ndarray:
    """Display embedding with fixed per-channel percentiles (fair cross-stage comparison)."""
    import cv2

    from msi_visual.parametric_mics_lmc import _spatial_gaussian_smooth_channels

    result = np.asarray(emb, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=bool)
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


def _embedding_to_rgb_from_kw(
    emb: np.ndarray,
    mask: np.ndarray,
    kw: dict[str, Any],
) -> np.ndarray:
    import cv2

    from msi_visual.parametric_mics_lmc import _spatial_gaussian_smooth_channels
    from msi_visual.utils import normalize

    m = np.asarray(mask, dtype=bool)
    result = normalize(
        np.asarray(emb, dtype=np.float32),
        low=float(kw.get("predict_percentile_low", 1.0)),
        high=float(kw.get("predict_percentile_high", 99.0)),
        mask=m,
    )
    result = _spatial_gaussian_smooth_channels(
        result, float(kw.get("predict_spatial_smooth_sigma", 0.0))
    )
    out = np.uint8(255.0 * result)
    out[~m] = 0
    if bool(kw.get("lab_to_rgb", True)) and int(kw.get("number_of_components", 3)) == 3:
        out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    out[~m] = 0
    return out


def _increment_magnitude_map(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d = np.asarray(cur - prev, dtype=np.float32)
    m = np.linalg.norm(d, axis=-1)
    m[~np.asarray(mask, dtype=bool)] = 0.0
    return m


def _magnitude_to_heatmap(mag: np.ndarray, mask: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """RGB heatmap of |Δembed| with percentile scaling on valid tissue."""
    import cv2

    m = np.asarray(mask, dtype=bool)
    v = mag[m]
    v = v[np.isfinite(v)]
    if v.size == 0:
        u8 = np.zeros(mag.shape, dtype=np.uint8)
    else:
        p99 = float(np.percentile(v, 99.0))
        scaled = np.clip(mag * float(gain) / max(p99, 1e-8), 0.0, 1.0)
        u8 = (255.0 * scaled).astype(np.uint8)
    u8[~m] = 0
    rgb = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb[~m] = 0
    return rgb


def _parse_clusters(val: Any) -> list[int]:
    if val is None:
        return [1024, 2048, 4096]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip()]
    return [int(s)]


def _mics_kwargs_from_cfg(cfg: DictConfig, num_layers: int, seed: int) -> dict[str, Any]:
    mt = str(OmegaConf.select(cfg.model, "model_type", default="mics_lmc")).strip().lower()
    if mt != "mics_lmc":
        raise ValueError(f"gallery_mics_residual_ladder requires model_type=mics_lmc (got {mt!r})")

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

    for sk in list(kw):
        if str(sk).startswith("clusters_auto_tune_"):
            kw.pop(sk, None)
    kw["clusters_auto_tune"] = None
    kw["num_layers"] = max(1, int(num_layers))
    kw["random_state"] = int(seed)
    return kw


def _resolve_stage_value_ladder(
    ga: Any,
    key: str,
    n_stages: int,
    default: float | int,
    *,
    cast: type[float] | type[int],
    min_value: float | int | None = None,
) -> list[float] | list[int]:
    """Resolve per-stage list from ``gallery.<key>``; null → repeat model default."""
    raw_sel = OmegaConf.select(ga, key, default=None)
    if raw_sel is None or (
        isinstance(raw_sel, str) and raw_sel.strip().lower() in ("", "~", "null", "none")
    ):
        return [cast(default)] * n_stages

    if OmegaConf.is_config(raw_sel):
        raw = OmegaConf.to_container(raw_sel, resolve=True)
    else:
        raw = raw_sel
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise ValueError(f"gallery.{key} must be a non-empty list")

    values: list[float] | list[int] = []
    for x in raw:
        v = cast(x)
        if min_value is not None:
            v = cast(max(min_value, v))
        values.append(v)

    if len(values) < n_stages:
        pad = values[-1]
        logger.warning(
            "%s has %d values for %d stages; padding with last value %s",
            key,
            len(values),
            n_stages,
            pad,
        )
        values.extend([pad] * (n_stages - len(values)))
    elif len(values) > n_stages:
        logger.warning(
            "%s has %d values for %d stages; truncating",
            key,
            len(values),
            n_stages,
        )
        values = values[:n_stages]
    return values


def _stage_train_kwargs(
    cfg: DictConfig,
    ga: Any,
    n_layers: int,
    stage_idx: int,
    base_seed: int,
    *,
    temperature: float | None = None,
    k_epoch: int | None = None,
    beta: float | None = None,
    cluster_loss_weight: float | None = None,
) -> dict[str, Any]:
    kw = _mics_kwargs_from_cfg(cfg, n_layers, seed=base_seed + stage_idx)
    if temperature is not None:
        kw["temperature"] = max(1e-6, float(temperature))
    if k_epoch is not None:
        kw["k_epoch"] = max(1, int(k_epoch))
    if beta is not None:
        kw["beta"] = float(beta)
    if cluster_loss_weight is not None:
        kw["cluster_loss_weight"] = max(0.0, float(cluster_loss_weight))
    ep_mul = float(OmegaConf.select(ga, "epochs_multiplier_per_stage", default=1.0))
    if stage_idx > 0 and ep_mul > 1.0:
        kw["num_epochs"] = max(1, int(round(int(kw["num_epochs"]) * ep_mul)))
    lr_mul = float(OmegaConf.select(ga, "residual_lr_multiplier", default=1.0))
    if stage_idx > 0 and lr_mul != 1.0:
        kw["lr"] = float(kw["lr"]) * lr_mul
    return kw


def _local_layer_stages(ga: Any) -> list[int]:
    """Layer counts for local residual stages (``shared_local`` mode)."""
    raw_sel = OmegaConf.select(ga, "local_layer_ladder", default=None)
    if raw_sel is None or (
        isinstance(raw_sel, str) and raw_sel.strip().lower() in ("", "~", "null", "none")
    ):
        raw = None
    elif OmegaConf.is_config(raw_sel):
        raw = OmegaConf.to_container(raw_sel, resolve=True)
    else:
        raw = raw_sel
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        return [max(1, int(x)) for x in raw]
    return [max(1, int(OmegaConf.select(ga, "local_num_layers", default=5)))]


def _layer_stages(ga: Any) -> list[int]:
    raw_sel = OmegaConf.select(ga, "layer_ladder", default=None)
    if raw_sel is None or (
        isinstance(raw_sel, str) and raw_sel.strip().lower() in ("", "~", "null", "none")
    ):
        raw = None
    elif OmegaConf.is_config(raw_sel):
        raw = OmegaConf.to_container(raw_sel, resolve=True)
    else:
        raw = raw_sel
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        stages = [max(1, int(x)) for x in raw]
    else:
        n = max(1, int(OmegaConf.select(ga, "max_layers", default=5)))
        stages = list(range(1, n + 1))
    return stages


def _build_shared_embedding(
    *,
    msi: np.ndarray,
    valid_mask: np.ndarray,
    cfg: DictConfig,
    ga: Any,
    pixel_cache: dict[str, Any],
    base_seed: int,
    n_components: int,
) -> tuple[np.ndarray, str, dict[str, Any] | None]:
    """Cross-region shared base: PCA spectral factors or shallow clustered MiCS."""
    mode = str(OmegaConf.select(ga, "shared_base_mode", default="mics")).strip().lower()
    if mode in ("pca", "spectral_pca"):
        emb = _pca_embedding_hwc(msi, valid_mask, n_components, base_seed)
        logger.info("Shared base: PCA (%d ch, tissue pixels only)", n_components)
        return emb, "pca", None

    if mode not in ("mics", "mics_lmc", "cluster_mics"):
        raise ValueError(f"shared_base_mode must be pca or mics, got {mode!r}")

    n_layers = max(1, int(OmegaConf.select(ga, "shared_mics_layers", default=1)))
    kw = _mics_kwargs_from_cfg(cfg, n_layers, seed=base_seed)
    epochs_raw = OmegaConf.select(ga, "shared_num_epochs", default=None)
    if epochs_raw is not None and str(epochs_raw).strip().lower() not in ("", "~", "null", "none"):
        kw["num_epochs"] = max(1, int(epochs_raw))
    kw["cluster"] = True
    kw["cluster_loss_weight"] = float(
        OmegaConf.select(ga, "shared_cluster_loss_weight", default=1.0)
    )
    logger.info(
        "Shared base: MiCS L%d (epochs=%s, cluster_loss_weight=%s)",
        n_layers,
        kw["num_epochs"],
        kw["cluster_loss_weight"],
    )
    model: MSIParametricMiCSLMC | None = None
    try:
        model = MSIParametricMiCSLMC(**kw)
        model.clear_residual_base()
        model.fit(msi, pixel_sampling_cache=pixel_cache)
        emb = model.predict_embedding(msi).astype(np.float32, copy=False)
        return emb, f"mics_L{n_layers}", kw
    finally:
        if model is not None:
            try:
                model.release_resources()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_shared_local_decomposition_figure(
    *,
    out_dir: Path,
    shared_emb: np.ndarray,
    local_emb: np.ndarray,
    combined_emb: np.ndarray,
    valid_mask: np.ndarray,
    norm_anchor: list[tuple[float, float]],
    lab_to_rgb: bool,
    n_components: int,
    smooth_sigma: float,
    increment_gain: float,
    shared_label: str,
    local_label: str,
    dpi: float,
    fig_w: float,
    title_fs: float,
) -> Path:
    """Three-panel figure: shared | local | shared+local."""
    shared_rgb = _embedding_to_rgb_with_anchor(
        shared_emb,
        valid_mask,
        norm_anchor,
        lab_to_rgb=lab_to_rgb,
        n_components=n_components,
        smooth_sigma=smooth_sigma,
    )
    combined_rgb = _embedding_to_rgb_with_anchor(
        combined_emb,
        valid_mask,
        norm_anchor,
        lab_to_rgb=lab_to_rgb,
        n_components=n_components,
        smooth_sigma=smooth_sigma,
    )
    local_mag = np.linalg.norm(local_emb, axis=-1).astype(np.float32)
    local_mag[~valid_mask] = 0.0
    local_rgb = _magnitude_to_heatmap(local_mag, valid_mask, gain=increment_gain)

    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_w * 0.32), dpi=int(dpi))
    axes[0].imshow(shared_rgb)
    axes[0].set_title(f"Shared ({shared_label})", fontsize=title_fs)
    axes[0].axis("off")
    axes[1].imshow(local_rgb)
    axes[1].set_title(f"Local ({local_label})\n|emb| heatmap", fontsize=title_fs)
    axes[1].axis("off")
    axes[2].imshow(combined_rgb)
    axes[2].set_title("Shared + local", fontsize=title_fs)
    axes[2].axis("off")
    fig.suptitle("MiCS shared / local decomposition", fontsize=title_fs + 2, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "gallery_shared_local_decomposition.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


@hydra.main(version_base=None, config_path="configs", config_name="gallery_mics_residual_ladder")
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

    ladder_mode = str(OmegaConf.select(ga, "ladder_mode", default="capacity")).strip().lower()
    if ladder_mode in ("shared_local", "shared-local", "shared"):
        stages = _local_layer_stages(ga)
    else:
        stages = _layer_stages(ga)
    default_temp = float(OmegaConf.select(cfg.model, "temperature", default=100.0))
    default_k_epoch = int(OmegaConf.select(cfg.model, "k_epoch", default=20))
    default_beta = float(OmegaConf.select(cfg.model, "beta", default=20.0))
    default_cluster_lw = float(OmegaConf.select(cfg.model, "cluster_loss_weight", default=1.0))
    n_stage = len(stages)
    stage_temperatures = _resolve_stage_value_ladder(
        ga, "temperature_ladder", n_stage, default_temp, cast=float, min_value=1e-6
    )
    stage_k_epochs = _resolve_stage_value_ladder(
        ga, "k_epoch_ladder", n_stage, default_k_epoch, cast=int, min_value=1
    )
    stage_betas = _resolve_stage_value_ladder(
        ga, "beta_ladder", n_stage, default_beta, cast=float
    )
    stage_cluster_lws = _resolve_stage_value_ladder(
        ga, "cluster_loss_weight_ladder", n_stage, default_cluster_lw, cast=float, min_value=0.0
    )
    default_local_clw = float(OmegaConf.select(ga, "local_cluster_loss_weight", default=0.0))
    pca_mics_morph = bool(OmegaConf.select(ga, "pca_mics_morph", default=False))
    if ladder_mode in ("shared_local", "shared-local", "shared"):
        pca_mics_morph = False
    morph_scale_align = bool(OmegaConf.select(ga, "morph_scale_align", default=True))
    morph_alphas = _resolve_morph_alpha_ladder(ga, n_stage)
    base_seed = int(getattr(ga, "seed", 42))
    n_components = int(OmegaConf.select(cfg.model, "number_of_components", default=3))
    eq = bool(getattr(ga, "equalize_visualization", False))
    save_delta = bool(getattr(ga, "save_residual_panels", True))
    shared_norm = bool(OmegaConf.select(ga, "shared_normalization", default=True))
    save_increment = bool(OmegaConf.select(ga, "save_increment_panels", default=True))
    increment_gain = float(OmegaConf.select(ga, "increment_gain", default=8.0))
    pred_lo = float(OmegaConf.select(cfg.model, "predict_percentile_low", default=1.0))
    pred_hi = float(OmegaConf.select(cfg.model, "predict_percentile_high", default=99.0))
    ncols = max(1, int(getattr(ga, "ncols", 3)))
    dpi = float(getattr(ga, "dpi", 160))
    fig_w = float(getattr(ga, "figure_size_inches", 14))
    title_fs = float(getattr(ga, "panel_title_fontsize", 11))

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

    sampling_kw = _mics_kwargs_from_cfg(cfg, stages[0], base_seed)
    pixel_cache = resolve_pixel_sampling_cache(
        msi,
        sampling_kw,
        out_dir=out_dir,
        share=share_sampling,
        cache_path=cache_path,
    )

    pca_emb: np.ndarray | None = None
    if pca_mics_morph:
        pca_emb = _pca_embedding_hwc(msi, valid_mask, n_components, base_seed)
        logger.info("PCA→MiCS morph: alphas=%s", morph_alphas)

    shared_emb: np.ndarray | None = None
    shared_label = ""
    if ladder_mode in ("shared_local", "shared-local", "shared"):
        shared_emb, shared_label, _ = _build_shared_embedding(
            msi=msi,
            valid_mask=valid_mask,
            cfg=cfg,
            ga=ga,
            pixel_cache=pixel_cache,
            base_seed=base_seed,
            n_components=n_components,
        )
        np.save(out_dir / "shared_embedding.npy", shared_emb)
        vm = valid_mask
        rms_s = (
            float(np.sqrt(np.mean(np.sum(shared_emb[vm] ** 2, axis=-1))))
            if np.any(vm)
            else 0.0
        )
        logger.info("Shared embedding RMS: %.4g (%s)", rms_s, shared_label)

    cumulative_emb: np.ndarray | None = shared_emb.copy() if shared_emb is not None else None
    stage_rows: list[dict[str, Any]] = []
    trained: list[dict[str, Any]] = []
    if shared_emb is not None:
        trained.append(
            {
                "idx": -1,
                "lbl": "shared",
                "role": "shared",
                "n_layers": 0,
                "kw": {},
                "cum_emb": shared_emb,
                "display_emb": shared_emb,
                "morph_alpha": 1.0,
                "delta_emb": np.zeros_like(shared_emb),
                "inc_emb": np.zeros_like(shared_emb),
                "rms_delta": 0.0,
                "rms_inc": 0.0,
                "rel_inc": 0.0,
            }
        )

    if ladder_mode in ("shared_local", "shared-local", "shared"):
        stage_cluster_lws = [default_local_clw] * n_stage

    for idx, n_layers in enumerate(stages):
        lbl = f"local_L{n_layers}" if shared_emb is not None else f"L{n_layers}"
        stage_temp = stage_temperatures[idx]
        stage_k = stage_k_epochs[idx]
        stage_beta = stage_betas[idx]
        stage_clw = stage_cluster_lws[idx]
        kw = _stage_train_kwargs(
            cfg,
            ga,
            n_layers,
            idx,
            base_seed,
            temperature=stage_temp,
            k_epoch=stage_k,
            beta=stage_beta,
            cluster_loss_weight=stage_clw,
        )
        if shared_emb is not None:
            residual_base_lbl = "shared+local_cum"
        elif pca_mics_morph and idx == 0 and pca_emb is not None:
            residual_base_lbl = "pca"
        elif cumulative_emb is not None:
            residual_base_lbl = "yes"
        else:
            residual_base_lbl = "no"

        stage_disp = idx + 1
        stage_total = len(stages)
        role = "local" if shared_emb is not None else "capacity"
        logger.info(
            "%s stage %d/%d | num_layers=%d | residual=%s | morph_α=%s | epochs=%s lr=%s "
            "temp=%s k_epoch=%s beta=%s cluster_loss_weight=%s",
            role.capitalize(),
            stage_disp,
            stage_total,
            n_layers,
            residual_base_lbl,
            morph_alphas[idx] if pca_mics_morph else "—",
            kw.get("num_epochs"),
            kw.get("lr"),
            kw.get("temperature"),
            kw.get("k_epoch"),
            kw.get("beta"),
            kw.get("cluster_loss_weight"),
        )

        model: MSIParametricMiCSLMC | None = None
        try:
            model = MSIParametricMiCSLMC(**kw)
            if pca_mics_morph and idx == 0 and pca_emb is not None:
                model.set_residual_base(pca_emb)
            elif cumulative_emb is not None:
                model.set_residual_base(cumulative_emb)
            else:
                model.clear_residual_base()

            model.fit(msi, pixel_sampling_cache=pixel_cache)
            cum_emb = model.predict_embedding(msi).astype(np.float32, copy=False)
            delta_emb = model.predict_delta(msi).astype(np.float32, copy=False)

            prev_emb = cumulative_emb
            inc_emb = (
                cum_emb - prev_emb
                if prev_emb is not None
                else delta_emb
            )
            vm = valid_mask
            rms_delta = float(np.sqrt(np.mean(np.sum(delta_emb[vm] ** 2, axis=-1)))) if np.any(vm) else 0.0
            rms_inc = float(np.sqrt(np.mean(np.sum(inc_emb[vm] ** 2, axis=-1)))) if np.any(vm) else 0.0
            rms_cum = float(np.sqrt(np.mean(np.sum(cum_emb[vm] ** 2, axis=-1)))) if np.any(vm) else 0.0
            rel = rms_inc / max(rms_cum, 1e-8)
            logger.info(
                "Stage %d embed RMS: |delta|=%.4g |increment|=%.4g |cum|=%.4g rel_inc=%.3f%%",
                idx + 1,
                rms_delta,
                rms_inc,
                rms_cum,
                100.0 * rel,
            )

            if pca_mics_morph and pca_emb is not None:
                display_emb = _blend_pca_mics(
                    pca_emb,
                    cum_emb,
                    morph_alphas[idx],
                    valid_mask,
                    scale_align=morph_scale_align,
                )
            else:
                display_emb = cum_emb

            local_total_emb = (
                cum_emb - shared_emb
                if shared_emb is not None
                else inc_emb
            )
            trained.append(
                {
                    "idx": idx,
                    "lbl": lbl,
                    "role": role,
                    "n_layers": n_layers,
                    "kw": kw,
                    "cum_emb": cum_emb,
                    "display_emb": display_emb,
                    "local_total_emb": local_total_emb,
                    "morph_alpha": float(morph_alphas[idx]) if pca_mics_morph else 1.0,
                    "delta_emb": delta_emb,
                    "inc_emb": inc_emb,
                    "rms_delta": rms_delta,
                    "rms_inc": rms_inc,
                    "rel_inc": rel,
                }
            )
            cumulative_emb = cum_emb.copy()
        finally:
            if model is not None:
                try:
                    model.release_resources()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not trained:
        raise RuntimeError("No stages trained.")

    if shared_emb is not None:
        anchor_emb = shared_emb
    else:
        anchor_emb = trained[0].get("display_emb", trained[0]["cum_emb"])
    norm_anchor = _compute_norm_anchor(anchor_emb, valid_mask, pred_lo, pred_hi)
    panel_rgbs: list[np.ndarray] = []
    increment_rgbs: list[np.ndarray] = []
    panel_meta: list[dict[str, Any]] = []

    for rec in trained:
        idx = int(rec["idx"])
        lbl = str(rec["lbl"])
        role = str(rec.get("role", "capacity"))
        n_layers = int(rec["n_layers"])
        kw = rec.get("kw") or {}
        cum_emb = rec["cum_emb"]
        viz_emb = rec.get("display_emb", cum_emb)
        viz_smooth = float(kw.get("predict_spatial_smooth_sigma", 0.0))
        lab2rgb = bool(kw.get("lab_to_rgb", OmegaConf.select(cfg.model, "lab_to_rgb", default=True)))
        n_ch = int(kw.get("number_of_components", n_components))

        if shared_norm:
            cum_rgb = _embedding_to_rgb_with_anchor(
                viz_emb,
                valid_mask,
                norm_anchor,
                lab_to_rgb=lab2rgb,
                n_components=n_ch,
                smooth_sigma=viz_smooth,
            )
        else:
            cum_rgb = _embedding_to_rgb_from_kw(viz_emb, valid_mask, kw if kw else _mics_kwargs_from_cfg(cfg, 1, base_seed))
        if eq:
            cum_rgb = _hist_equalize_rgb(cum_rgb)

        cum_path = pan_dir / f"panel__{lbl}__cumulative_viz.png"
        Image.fromarray(cum_rgb, mode="RGB").save(cum_path)
        np.save(pan_dir / f"panel__{lbl}__cumulative_embedding.npy", cum_emb)

        delta_path = ""
        if save_delta and role != "shared":
            delta_emb = rec["delta_emb"]
            mag = np.linalg.norm(delta_emb, axis=-1).astype(np.float32)
            mag[~valid_mask] = 0.0
            delta_rgb = _magnitude_to_heatmap(mag, valid_mask, gain=increment_gain)
            delta_path = str(pan_dir / f"panel__{lbl}__residual_viz.png")
            Image.fromarray(delta_rgb, mode="RGB").save(delta_path)
            np.save(pan_dir / f"panel__{lbl}__residual_embedding.npy", delta_emb)

        inc_path = ""
        if save_increment and role == "local":
            prev_rec = next((t for t in trained if int(t["idx"]) == idx - 1), None)
            if prev_rec is not None:
                prev_viz = prev_rec.get("display_emb", prev_rec["cum_emb"])
                inc_mag = _increment_magnitude_map(prev_viz, viz_emb, valid_mask)
                inc_rgb = _magnitude_to_heatmap(inc_mag, valid_mask, gain=increment_gain)
                inc_path = str(pan_dir / f"panel__{lbl}__increment_viz.png")
                Image.fromarray(inc_rgb, mode="RGB").save(inc_path)
                increment_rgbs.append(inc_rgb)

        stage_rows.append(
            {
                "stage": idx,
                "role": role,
                "num_layers": n_layers,
                "temperature": float(kw.get("temperature", default_temp)),
                "k_epoch": int(kw.get("k_epoch", default_k_epoch)),
                "beta": float(kw.get("beta", default_beta)),
                "cluster_loss_weight": float(
                    kw.get("cluster_loss_weight", default_cluster_lw if role != "local" else default_local_clw)
                ),
                "morph_alpha": float(rec.get("morph_alpha", 1.0)),
                "rms_delta": rec["rms_delta"],
                "rms_inc": rec["rms_inc"],
                "rel_inc_pct": 100.0 * rec["rel_inc"],
                "cumulative_viz": str(cum_path),
                "residual_viz": delta_path,
                "increment_viz": inc_path,
            }
        )
        panel_rgbs.append(cum_rgb)
        panel_meta.append({"lbl": lbl, "role": role, "n_layers": n_layers, "idx": idx})

    n_p = len(panel_rgbs)
    nrows = int(np.ceil(n_p / ncols))
    mh = float(max(fig_w * 0.35, fig_w * 0.26 * max(1, nrows)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, mh))
    if nrows == 1 and ncols == 1:
        ax_grid = np.array([[axes]])
    elif nrows == 1:
        ax_grid = np.asarray(axes).reshape(1, -1)
    elif ncols == 1:
        ax_grid = np.asarray(axes).reshape(-1, 1)
    else:
        ax_grid = np.asarray(axes)

    for pidx, (rgb, meta) in enumerate(zip(panel_rgbs, panel_meta)):
        r, c = divmod(pidx, ncols)
        ax = ax_grid[r, c]
        ax.imshow(rgb)
        lbl = str(meta["lbl"])
        role = str(meta["role"])
        n_layers = int(meta["n_layers"])
        rec_idx = int(meta["idx"])
        if role == "shared":
            tit = f"shared ({shared_label})\ncross-region base"
        elif role == "local":
            li = max(0, rec_idx)
            temp = stage_temperatures[li]
            k_ep = stage_k_epochs[li]
            b = stage_betas[li]
            clw = stage_cluster_lws[li]
            tit = (
                f"local {li + 1}: {lbl}\n"
                f"T={temp:g} k={k_ep} β={b:g} clw={clw:g}"
            )
        else:
            temp = stage_temperatures[pidx]
            k_ep = stage_k_epochs[pidx]
            b = stage_betas[pidx]
            clw = stage_cluster_lws[pidx]
            rec = trained[pidx]
            ma = float(rec.get("morph_alpha", 1.0))
            if pca_mics_morph and ma < 1.0:
                tit = (
                    f"stage {pidx + 1}: L{n_layers} PCA→MiCS α={ma:g}\n"
                    f"T={temp:g} k={k_ep} β={b:g} clw={clw:g}"
                )
            elif pidx == 0:
                tit = f"stage {pidx + 1}: {n_layers} layer(s) (base)\nT={temp:g} k={k_ep} β={b:g} clw={clw:g}"
            else:
                tit = (
                    f"stage {pidx + 1}: {n_layers} layers (+ residual)\n"
                    f"T={temp:g} k={k_ep} β={b:g} clw={clw:g}"
                )
        ax.set_title(tit, fontsize=title_fs, pad=8)
        ax.axis("off")

    for idx in range(n_p, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax_grid[r, c].axis("off")

    norm_note = "shared norm (stage-1 percentiles)" if shared_norm else "per-stage norm"
    if shared_emb is not None:
        mode_note = f"shared/local ({shared_label})"
    elif pca_mics_morph:
        mode_note = "PCA → MiCS morph"
    else:
        mode_note = "capacity ladder"
    fig.suptitle(
        f"MiCS residual ladder — {mode_note} ({norm_note})",
        fontsize=title_fs + 2,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    mosaic_path = out_dir / "gallery_mics_residual_ladder.png"
    fig.savefig(mosaic_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    inc_mosaic_path = out_dir / "gallery_mics_residual_increments.png"
    if increment_rgbs:
        n_inc = len(increment_rgbs)
        nrows_i = int(np.ceil(n_inc / ncols))
        fig_i, axes_i = plt.subplots(nrows_i, ncols, figsize=(fig_w, mh * 0.85))
        if nrows_i == 1 and ncols == 1:
            ax_i_grid = np.array([[axes_i]])
        elif nrows_i == 1:
            ax_i_grid = np.asarray(axes_i).reshape(1, -1)
        elif ncols == 1:
            ax_i_grid = np.asarray(axes_i).reshape(-1, 1)
        else:
            ax_i_grid = np.asarray(axes_i)
        local_trained = [t for t in trained if t.get("role") == "local"]
        for j, rgb in enumerate(increment_rgbs):
            r, c = divmod(j, ncols)
            ax_i_grid[r, c].imshow(rgb)
            if j + 1 < len(local_trained):
                cur_lbl = str(local_trained[j + 1]["lbl"])
                prev_lbl = str(local_trained[j]["lbl"]) if j < len(local_trained) else "shared"
                ax_i_grid[r, c].set_title(
                    f"Δ {cur_lbl} − {prev_lbl} (gain×{increment_gain:g})",
                    fontsize=title_fs,
                )
            else:
                st = stages[min(j + 1, len(stages) - 1)]
                prev_st = stages[j] if j < len(stages) else stages[0]
                ax_i_grid[r, c].set_title(
                    f"Δ L{st} − L{prev_st} (gain×{increment_gain:g})",
                    fontsize=title_fs,
                )
            ax_i_grid[r, c].axis("off")
        for j in range(n_inc, nrows_i * ncols):
            r, c = divmod(j, ncols)
            ax_i_grid[r, c].axis("off")
        fig_i.suptitle("|embedding increment| heatmaps", fontsize=title_fs + 2, y=0.98)
        fig_i.tight_layout(rect=[0, 0, 1, 0.96])
        fig_i.savefig(inc_mosaic_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_i)

    decomp_path: Path | None = None
    if shared_emb is not None and trained:
        final_rec = trained[-1]
        final_kw = final_rec.get("kw") or {}
        local_total = final_rec.get("local_total_emb", final_rec["cum_emb"] - shared_emb)
        np.save(out_dir / "local_total_embedding.npy", local_total.astype(np.float32))
        decomp_path = _save_shared_local_decomposition_figure(
            out_dir=out_dir,
            shared_emb=shared_emb,
            local_emb=local_total,
            combined_emb=final_rec["cum_emb"],
            valid_mask=valid_mask,
            norm_anchor=norm_anchor,
            lab_to_rgb=bool(final_kw.get("lab_to_rgb", OmegaConf.select(cfg.model, "lab_to_rgb", default=True))),
            n_components=n_components,
            smooth_sigma=float(final_kw.get("predict_spatial_smooth_sigma", 0.0)),
            increment_gain=increment_gain,
            shared_label=shared_label,
            local_label=str(final_rec["lbl"]),
            dpi=dpi,
            fig_w=fig_w,
            title_fs=title_fs,
        )
        vm = valid_mask
        rms_loc = float(np.sqrt(np.mean(np.sum(local_total[vm] ** 2, axis=-1)))) if np.any(vm) else 0.0
        rms_com = float(np.sqrt(np.mean(np.sum(final_rec["cum_emb"][vm] ** 2, axis=-1)))) if np.any(vm) else 0.0
        logger.info(
            "Decomposition RMS: shared=%.4g local_total=%.4g combined=%.4g",
            float(np.sqrt(np.mean(np.sum(shared_emb[vm] ** 2, axis=-1)))) if np.any(vm) else 0.0,
            rms_loc,
            rms_com,
        )

    summary_path = out_dir / "residual_ladder_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"npy_path: {npy_path}\n")
        fh.write(f"ladder_mode: {ladder_mode}\n")
        if shared_emb is not None:
            fh.write(f"shared_base_mode: {shared_label}\n")
            fh.write(f"local_stages: {stages}\n")
        else:
            fh.write(f"stages: {stages}\n")
        fh.write(f"temperatures: {stage_temperatures}\n")
        fh.write(f"k_epochs: {stage_k_epochs}\n")
        fh.write(f"betas: {stage_betas}\n")
        fh.write(f"cluster_loss_weights: {stage_cluster_lws}\n")
        fh.write(f"pca_mics_morph: {pca_mics_morph}\n")
        if pca_mics_morph:
            fh.write(f"morph_alphas: {morph_alphas}\n")
            fh.write(f"morph_scale_align: {morph_scale_align}\n")
        fh.write(f"shared_normalization: {shared_norm}\n")
        fh.write(f"increment_gain: {increment_gain}\n\n")
        for row in stage_rows:
            fh.write(
                f"{row.get('role', 'stage')} {row['stage']}: layers={row['num_layers']} "
                f"T={row['temperature']:g} k_epoch={row['k_epoch']} beta={row['beta']:g} "
                f"cluster_loss_weight={row['cluster_loss_weight']:g} "
                f"morph_α={row.get('morph_alpha', 1.0):g} "
                f"|delta|_rms={row['rms_delta']:.6g} "
                f"|inc|_rms={row['rms_inc']:.6g} "
                f"rel_inc={row['rel_inc_pct']:.3f}%\n"
            )
            fh.write(f"  cumulative: {row['cumulative_viz']}\n")
            if row["residual_viz"]:
                fh.write(f"  residual:   {row['residual_viz']}\n")
            if row.get("increment_viz"):
                fh.write(f"  increment:  {row['increment_viz']}\n")
        fh.write(f"\nmosaic: {mosaic_path}\n")
        if decomp_path is not None:
            fh.write(f"decomposition: {decomp_path}\n")
        if increment_rgbs:
            fh.write(f"increments_mosaic: {inc_mosaic_path}\n")

    logger.info("Gallery saved: %s", mosaic_path)
    if decomp_path is not None:
        logger.info("Shared/local decomposition: %s", decomp_path)
    if increment_rgbs:
        logger.info("Increment gallery: %s", inc_mosaic_path)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
