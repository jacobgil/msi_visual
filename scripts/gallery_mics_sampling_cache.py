"""Shared superpixel + coreset sampling cache for MiCS gallery ladders."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np

from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC

logger = logging.getLogger(__name__)

_MICS_LMC_INIT_PARAMS = frozenset(
    p for p in inspect.signature(MSIParametricMiCSLMC.__init__).parameters if p != "self"
)


def _kwargs_for_sampling_cache(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop subclass-only keys (e.g. residual_factor) before base MiCS cache build."""
    return {k: v for k, v in model_kwargs.items() if k in _MICS_LMC_INIT_PARAMS}


def export_pixel_sampling_cache(model: MSIParametricMiCSLMC) -> dict[str, Any]:
    return model.export_pixel_sampling_cache()


def save_pixel_sampling_cache(path: Path, cache: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = cache.get("cluster_labels")
    payload: dict[str, Any] = {
        "sampled_flat_indices": np.asarray(cache["sampled_flat_indices"], dtype=np.int64),
        "indices": np.asarray(cache["indices"], dtype=np.int64),
        "euclidean": np.asarray(cache["euclidean"], dtype=np.float32),
        "number_of_points": np.asarray([int(cache["number_of_points"])], dtype=np.int64),
    }
    if labels:
        payload["cluster_labels"] = np.asarray(labels, dtype=object)
    np.savez_compressed(path, **payload)
    logger.info("Saved pixel sampling cache: %s", path)


def load_pixel_sampling_cache(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"pixel sampling cache not found: {path}")
    z = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {
        "sampled_flat_indices": np.asarray(z["sampled_flat_indices"], dtype=np.int64),
        "indices": np.asarray(z["indices"], dtype=np.int64),
        "euclidean": np.asarray(z["euclidean"], dtype=np.float32),
        "number_of_points": int(np.asarray(z["number_of_points"]).reshape(-1)[0]),
    }
    if "cluster_labels" in z.files:
        raw = z["cluster_labels"]
        if raw.size:
            out["cluster_labels"] = [np.asarray(x, dtype=np.int32) for x in raw.tolist()]
        else:
            out["cluster_labels"] = None
    else:
        out["cluster_labels"] = None
    logger.info(
        "Loaded pixel sampling cache: %s (n_samples=%d, n_ref=%d)",
        path,
        out["sampled_flat_indices"].size,
        out["indices"].size,
    )
    return out


def build_pixel_sampling_cache(msi: np.ndarray, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    """One-time superpixel + coreset (+ KMeans labels); no training epochs."""
    cache_kw = _kwargs_for_sampling_cache(model_kwargs)
    model = MSIParametricMiCSLMC(**cache_kw)
    try:
        model.set_image(msi)
        return model.export_pixel_sampling_cache()
    finally:
        try:
            model.release_resources()
        except Exception:
            pass


def resolve_pixel_sampling_cache(
    msi: np.ndarray,
    model_kwargs: dict[str, Any],
    *,
    out_dir: Path,
    share: bool = True,
    cache_path: Path | None = None,
    strip_cluster_labels: bool = False,
) -> dict[str, Any] | None:
    """Build, load, or skip shared sampling cache for gallery stage loops."""
    if not share:
        return None

    if cache_path is not None and Path(cache_path).is_file():
        cache = load_pixel_sampling_cache(Path(cache_path))
    elif (local := Path(out_dir) / "pixel_sampling_cache.npz").is_file():
        cache = load_pixel_sampling_cache(local)
    else:
        logger.info("Building shared pixel sampling cache (superpixel + coreset, no training)...")
        cache = build_pixel_sampling_cache(msi, model_kwargs)
        save_pixel_sampling_cache(local, cache)

    if strip_cluster_labels and cache is not None:
        cache = dict(cache)
        cache.pop("cluster_labels", None)
    return cache
