"""Scatter tile embeddings onto a 2-D grid and prepare MiCS-friendly feature cubes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from msi_visual.extract.he_tiles_atd import ExtractedTile, TilePlan, grid_shape_from_plan


def scatter_embeddings(
    tiles: list[ExtractedTile],
    features: np.ndarray,
    plan: TilePlan,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (H, W, C) embedding map and boolean tissue mask from sparse tiles.

    Empty grid cells remain zero; mask marks cells with embeddings.
    """
    if len(tiles) != features.shape[0]:
        raise ValueError(f"Tile count {len(tiles)} != feature rows {features.shape[0]}")
    gh, gw = grid_shape_from_plan(plan)
    c = int(features.shape[1])
    embed = np.zeros((gh, gw, c), dtype=np.float32)
    mask = np.zeros((gh, gw), dtype=bool)
    for tile, vec in zip(tiles, features):
        embed[tile.row, tile.col] = vec
        mask[tile.row, tile.col] = True
    return embed, mask


def prepare_mics_input(
    embed_map: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    pca_dims: int | None = 128,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Reduce high-D embeddings and min–max scale to non-negative values for MiCS.

    MiCS treats H×W×C like an MSI cube; background stays zero.
    """
    embed = np.asarray(embed_map, dtype=np.float32)
    mask = np.asarray(tissue_mask, dtype=bool)
    h, w, c = embed.shape
    flat = embed.reshape(-1, c)
    valid = mask.reshape(-1)
    meta: dict[str, Any] = {"input_channels": c, "pca_dims": pca_dims}

    if not np.any(valid):
        raise ValueError("No valid embedding cells for MiCS.")

    x = flat[valid]
    if pca_dims is not None and int(pca_dims) > 0 and int(pca_dims) < c:
        pca = PCA(n_components=int(pca_dims), random_state=int(random_state))
        x = pca.fit_transform(x).astype(np.float32)
        meta["pca_explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
        meta["input_channels"] = int(x.shape[1])

    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    x_norm = (x - lo) / (hi - lo + 1e-8)

    out = np.zeros((h * w, x_norm.shape[1]), dtype=np.float32)
    out[valid] = x_norm
    return out.reshape(h, w, -1), meta


def save_embedding_artifacts(
    out_dir: Path,
    *,
    embed_map: np.ndarray,
    tissue_mask: np.ndarray,
    features: np.ndarray,
    coords: np.ndarray,
    meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embedding_map.npy", embed_map.astype(np.float32))
    np.save(out_dir / "tissue_mask.npy", tissue_mask.astype(np.uint8))
    np.save(out_dir / "tile_features.npy", features.astype(np.float32))
    np.save(out_dir / "tile_coords_l0.npy", coords.astype(np.int32))
    with (out_dir / "embedding_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def tile_coords_array(tiles: list[ExtractedTile]) -> np.ndarray:
    return np.asarray([[t.x_l0, t.y_l0] for t in tiles], dtype=np.int32)
