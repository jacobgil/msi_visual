#!/usr/bin/env python3
"""
Load VIA-style annotation JSON (same format as msi_visual.supervised.annotations.get_annotations):
optional top-level ``_via_img_metadata``, keys like ``{file_stem}_...``, regions with
``shape_attributes`` (polygon / circle / ellipse) and ``region_attributes``.

Renders category-colored fills and boundary outlines. Canvas size should match the MSI cube
(``H×W`` from ``np.load(...).shape[:2]``), as in supervised training: see ``get_dataset`` /
``get_annotations`` in ``msi_visual/supervised/annotations.py`` and notebooks such as
``notebooks/variational.ipynb`` / ``notebooks/umap_annotations.ipynb`` (``paths`` as
``[(path/to/0.npy, label), ...]`` — VIA prefix is always ``Path(npy).stem``, same as
``get_dataset`` in ``annotations.py``; the second tuple value is not used for keys) or
``notebooks/compare_between_slides.ipynb`` / ``notebooks/noise_analysis.ipynb``.

Set ``npy`` or ``background`` in the YAML (or CLI) to the same ``.npy`` as your slide so
width/height come from the array (memory-mapped ``shape``, not polygon bounds).

Configuration: ``scripts/configs/view_annotation_boundaries.yaml`` (Hydra). Override any key on the CLI.

Examples:
  python scripts/view_annotation_boundaries.py annotation_json=path/to/ann.json list_keys=true
  python scripts/view_annotation_boundaries.py annotation_json=path/to/ann.json npy=path/to/0.npy show=true
  python scripts/view_annotation_boundaries.py annotation_json=path/to/ann.json background=path/to/viz.png index=0_MyStem out=boundaries.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Match get_annotations defaults in msi_visual/supervised/annotations.py
_DEFAULT_CATEGORY_KEYS = ("main_cat1", "Allen_name", "sub_cat2")


def _parse_category(category: str) -> str:
    category = str(category).strip().replace("\n", "")
    if category == "":
        category = "artefact"
    return category


def _region_to_polygon(r: dict) -> np.ndarray | None:
    """Return Nx1x2 float32 polygon for OpenCV, or None."""
    sa = r.get("shape_attributes") or {}
    name = str(sa.get("name", "polygon")).lower()
    if name == "ellipse":
        x = float(sa["cx"])
        y = float(sa["cy"])
        rx = float(sa["rx"])
        t = np.linspace(0, 2 * np.pi, 50)
        xs = x + rx * np.cos(t)
        ys = y + float(sa["ry"]) * np.sin(t)
        poly = np.stack([xs, ys], axis=1).astype(np.float32)
        return poly[:, None, :]
    if name == "circle":
        x = float(sa["cx"])
        y = float(sa["cy"])
        r0 = float(sa["r"])
        t = np.linspace(0, 2 * np.pi, 50)
        xs = x + r0 * np.cos(t)
        ys = y + r0 * np.sin(t)
        poly = np.stack([xs, ys], axis=1).astype(np.float32)
        return poly[:, None, :]
    xs = sa.get("all_points_x")
    ys = sa.get("all_points_y")
    if not xs or not ys or len(xs) != len(ys):
        return None
    pts = np.array(list(zip(xs, ys)), dtype=np.float32)
    return pts[:, None, :]


def _category_string(r: dict, category_keys: tuple[str, ...]) -> str:
    ra = r.get("region_attributes") or {}
    parts = []
    for k in category_keys:
        v = ra.get(k, "")
        parts.append(_parse_category(str(v)).upper())
    return "_".join(parts) if parts else "UNKNOWN"


def _region_main_cat1(r: dict) -> str:
    ra = r.get("region_attributes") or {}
    return _parse_category(str(ra.get("main_cat1", ""))).upper()


def _category_ignored(cat: str, ignore: list[str] | None, *, main_cat1: str | None = None) -> bool:
    """
    True if category should be dropped.

    ``ignore`` entries match (case-insensitive):
      - full category string (e.g. ``BG_ARTEFACT_ARTEFACT``)
      - ``main_cat1`` token (e.g. ``BG``)
      - prefix of the full string (``BG`` matches ``BG_...``)
    """
    if not ignore:
        return False
    tokens = {str(x).strip().upper() for x in ignore if str(x).strip()}
    if not tokens:
        return False
    cat_u = str(cat).strip().upper()
    main_u = str(main_cat1).strip().upper() if main_cat1 else ""
    if cat_u in tokens or (main_u and main_u in tokens):
        return True
    for tok in tokens:
        if cat_u.startswith(tok + "_"):
            return True
    return False


def load_annotation_polygons(
    annotation_path: Path,
    index_prefix: str,
    *,
    ignore: list[str] | None = None,
    keep: set[str] | None = None,
    category_keys: tuple[str, ...] = _DEFAULT_CATEGORY_KEYS,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Same selection rules as ``get_annotations`` in msi_visual/supervised/annotations.py.
    ``index_prefix`` must match ``get_annotations``: only keys where ``key.startswith(index_prefix + "_")`` are used
    (e.g. index ``0_MyStem`` for key ``0_MyStem_rest``).
    """
    ignore = ignore or []
    with annotation_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if "_via_img_metadata" in raw:
        meta = raw["_via_img_metadata"]
    else:
        meta = raw

    polygons: list[np.ndarray] = []
    categories: list[str] = []

    prefix = f"{index_prefix}_"
    for key in meta:
        if not key.startswith(prefix):
            continue
        regions = meta[key].get("regions") or []
        for r in regions:
            cat = _category_string(r, category_keys)
            main = _region_main_cat1(r)
            if _category_ignored(cat, ignore, main_cat1=main):
                continue
            if keep is not None and cat not in keep:
                continue
            poly = _region_to_polygon(r)
            if poly is None or len(poly) < 3:
                continue
            polygons.append(poly.astype(np.float32))
            categories.append(cat)

    return polygons, categories


def list_annotation_keys(annotation_path: Path) -> list[str]:
    """Return VIA image keys; pass ``--index`` such that the desired key starts with ``index + '_'``."""
    with annotation_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if "_via_img_metadata" in raw:
        meta = raw["_via_img_metadata"]
    else:
        meta = raw
    return sorted(meta.keys())


def _infer_hw_from_polygons(
    polygons: list[np.ndarray], margin: int = 2
) -> tuple[int, int]:
    if not polygons:
        return 256, 256
    xs: list[float] = []
    ys: list[float] = []
    for p in polygons:
        p2 = np.asarray(p).reshape(-1, 2)
        xs.extend(p2[:, 0].tolist())
        ys.extend(p2[:, 1].tolist())
    h = int(np.ceil(max(ys)) + margin)
    w = int(np.ceil(max(xs)) + margin)
    return max(h, 1), max(w, 1)


def _random_colors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((max(n, 1), 3)) * 255).astype(np.uint8)


def _background_rgb_to_uint8(background: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """
    H×W×3 uint8 image for ``render_overlay``. Float inputs in ~[0, 1] are scaled to 0–255
    (``astype(uint8)`` alone would truncate them to black). Dim floats map max→255, not *255 only.
    """
    h, w = shape_hw
    arr = np.asarray(background)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if arr.shape[:2] != (h, w):
        if arr.dtype == np.uint8:
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            arr = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if arr.dtype == np.uint8:
        return arr
    x = np.asarray(arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros((h, w, 3), dtype=np.uint8)
    mx = float(np.nanmax(x[finite]))
    if mx <= 1.0 + 1e-3:
        if mx > 1e-12:
            x = (x / mx) * 255.0
        else:
            x = np.zeros_like(x)
    x = np.clip(x, 0.0, 255.0)
    return np.round(x).astype(np.uint8)


def render_overlay(
    background: np.ndarray | None,
    polygons: list[np.ndarray],
    categories: list[str],
    shape_hw: tuple[int, int],
    boundary_width: int = 2,
    *,
    fill_alpha: float = 0.45,
    boundary_halo: int = 2,
    uniform_fill_rgb: tuple[int, int, int] | None = None,
    boundary_rgb: tuple[int, int, int] = (255, 255, 255),
    halo_rgb: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    h, w = shape_hw
    if background is None:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        rgb = _background_rgb_to_uint8(background, (h, w))

    if uniform_fill_rgb is not None:
        colormap = {c: tuple(int(x) for x in uniform_fill_rgb) for c in set(categories)}
    else:
        uniq = sorted(set(categories))
        palette = _random_colors(len(uniq), seed=42)
        colormap = {c: tuple(int(x) for x in palette[i]) for i, c in enumerate(uniq)}

    # Blend fill only inside each polygon. Global addWeighted per polygon would multiply the
    # entire image by (1-alpha) each time, erasing the background when there are many regions.
    overlay = rgb.astype(np.float32).copy()
    alpha = float(np.clip(fill_alpha, 0.0, 1.0))
    if alpha > 1e-6:
        for poly, cat in zip(polygons, categories):
            color = colormap.get(cat, uniform_fill_rgb or (200, 200, 200))
            p = np.int32(np.round(poly))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [p], -1, 255, -1)
            m = mask.astype(bool)
            c = np.array(color, dtype=np.float32)
            overlay[m] = overlay[m] * (1.0 - alpha) + c * alpha
    overlay = np.clip(np.round(overlay), 0, 255).astype(np.uint8)

    thick = max(1, int(boundary_width))
    halo = max(0, int(boundary_halo))
    for poly in polygons:
        p = np.int32(np.round(poly))
        if halo > 0:
            cv2.polylines(
                overlay,
                [p],
                isClosed=True,
                color=tuple(int(x) for x in halo_rgb),
                thickness=thick + 2 * halo,
                lineType=cv2.LINE_AA,
            )
        cv2.polylines(
            overlay,
            [p],
            isClosed=True,
            color=tuple(int(x) for x in boundary_rgb),
            thickness=thick,
            lineType=cv2.LINE_AA,
        )

    return overlay


def _normalize_transpose_spatial_mode(mode: str) -> str:
    t = str(mode).strip().lower().replace("-", "_")
    if not t:
        return "none"
    aliases = {
        "swap": "swap_hw",
        "transpose": "swap_hw",
        "transpose_hw": "swap_hw",
        "1_0_2": "swap_hw",
        "chain": "chain_flip",
        "notebook": "chain_flip",
        "umap": "chain_flip",
        "transpose_chain": "chain_flip",
    }
    return aliases.get(t, t)


def resolve_transpose_spatial_mode(ann: Any) -> str:
    """
    ``transpose_spatial_mode`` wins if set. Else ``transpose_spatial: true`` → ``chain_flip``
    (same as ``data.transpose().transpose(1, 2, 0)[::-1, :, :]`` on H×W×C).
    """
    raw = OmegaConf.select(ann, "transpose_spatial_mode")
    if raw is not None and str(raw).strip() not in ("", "~", "null", "None"):
        return _normalize_transpose_spatial_mode(str(raw))
    if bool(OmegaConf.select(ann, "transpose_spatial") is True):
        return "chain_flip"
    return "none"


def spatial_hw_after_transpose_mode(file_hw: tuple[int, int], mode: str) -> tuple[int, int]:
    """Spatial (H, W) of the on-disk array → (H', W') after ``apply_annotation_spatial_transform``."""
    h, w = file_hw
    m = _normalize_transpose_spatial_mode(mode)
    if m in ("none", "off", "identity", ""):
        return (h, w)
    if m in ("swap_hw", "chain_flip"):
        return (w, h)
    raise ValueError(f"Unknown transpose_spatial_mode for shape mapping: {mode!r}")


def apply_annotation_spatial_transform(cube: np.ndarray, mode: str) -> np.ndarray:
    """
    Align MSI array with VIA coordinates.

    - ``swap_hw``: ``np.transpose(x, (1, 0, 2))`` (or 2D ``.T``).
    - ``chain_flip``: ``x.transpose().transpose(1, 2, 0)[::-1, :, :]`` for H×W×C (matches common
      notebook pipelines); 2D uses ``x.transpose()[::-1, :]``.
    """
    x = np.asarray(cube)
    m = _normalize_transpose_spatial_mode(mode)
    if m in ("none", "off", "identity", ""):
        return x
    if m == "swap_hw":
        if x.ndim == 3:
            y = np.transpose(x, (1, 0, 2))
        elif x.ndim == 2:
            y = x.T
        else:
            raise ValueError(f"swap_hw expects 2D or HxWxC, got shape {x.shape}")
        return np.ascontiguousarray(y)
    if m == "chain_flip":
        if x.ndim == 3:
            y = x.transpose().transpose(1, 2, 0)[::-1, :, :]
        elif x.ndim == 2:
            y = x.transpose()[::-1, :]
        else:
            raise ValueError(f"chain_flip expects 2D or HxWxC, got shape {x.shape}")
        return np.ascontiguousarray(y)
    raise ValueError(
        f"Unknown transpose_spatial_mode {mode!r}; use none, swap_hw, or chain_flip"
    )


def transpose_spatial_hwc(cube: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for ``apply_annotation_spatial_transform(..., 'swap_hw')``."""
    return apply_annotation_spatial_transform(cube, "swap_hw")


def spatial_hw_from_npy(path: Path) -> tuple[int, int] | None:
    """First two dimensions of an MSI array (H, W) using mmap; no full load."""
    try:
        arr = np.load(path, mmap_mode="r")
    except OSError:
        return None
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        return int(arr.shape[0]), int(arr.shape[1])
    return None


def load_background(
    path: Path,
    shape_hw: tuple[int, int] | None,
    *,
    spatial_mode: str = "none",
) -> np.ndarray | None:
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
        if _normalize_transpose_spatial_mode(spatial_mode) != "none":
            arr = apply_annotation_spatial_transform(arr, spatial_mode)
        if arr.ndim == 2:
            g = _to_u8_gray(arr)
            return np.stack([g, g, g], axis=-1)
        if arr.ndim == 3:
            x = np.asarray(arr, dtype=np.float32)
            if x.shape[2] >= 3:
                r = x[:, :, :3]
            else:
                g = x[:, :, 0]
                r = np.stack([g, g, g], axis=-1)
            r = r - np.nanmin(r)
            den = np.nanmax(r) - np.nanmin(r) + 1e-8
            r = (r / den * 255.0).clip(0, 255).astype(np.uint8)
            return r
        return None
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _to_u8_gray(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmin(x)
    x = x / (np.nanmax(x) + 1e-8) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def _opt_path(value: Any) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None
    return Path(s)


def _keep_from_cfg(cfg: Any) -> set[str] | None:
    if cfg is None:
        return None
    items = OmegaConf.to_container(cfg, resolve=True)
    if items is None:
        return None
    if isinstance(items, list):
        return {str(x) for x in items}
    return None


def _category_keys_from_cfg(cfg: DictConfig) -> tuple[str, ...]:
    raw = OmegaConf.select(cfg, "category_keys")
    if raw is None:
        return _DEFAULT_CATEGORY_KEYS
    items = OmegaConf.to_container(raw, resolve=True)
    if not items:
        return _DEFAULT_CATEGORY_KEYS
    return tuple(str(x) for x in items)


def annotation_index_from_npy_path(npy_path: Path) -> str:
    """
    Same rule as ``get_dataset`` / ``get_annotations`` pairing in
    ``msi_visual.supervised.annotations``: ``annotation_index`` is the ``.npy`` filename
    stem (``path`` → ``0`` for ``0.npy``). VIA keys must start with ``f"{stem}_"``.
    """
    return npy_path.stem


def _npy_from_paths_list(paths_rows: Any, slide: int) -> Path | None:
    """First column of ``paths[slide]``: ``[npy_path, label]`` or ``{npy: ...}``."""
    if not isinstance(paths_rows, list) or not paths_rows:
        return None
    if slide < 0 or slide >= len(paths_rows):
        return None
    row = paths_rows[slide]
    if isinstance(row, (list, tuple)) and len(row) >= 1:
        return _opt_path(row[0])
    if isinstance(row, dict):
        return _opt_path(row.get("npy") or row.get("path"))
    return None


@hydra.main(version_base=None, config_path="configs", config_name="view_annotation_boundaries")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    annotation_json = _opt_path(cfg.annotation_json)
    if annotation_json is None or not annotation_json.is_file():
        print(f"Not found: {annotation_json}", file=sys.stderr)
        sys.exit(1)

    list_keys = bool(OmegaConf.select(cfg, "list_keys") or False)
    if list_keys:
        for s in list_annotation_keys(annotation_json):
            print(s)
        return

    paths_raw = OmegaConf.select(cfg, "paths")
    slide = int(OmegaConf.select(cfg, "slide") or 0)
    paths_rows: list[Any] | None = None
    if paths_raw is not None:
        paths_rows = OmegaConf.to_container(paths_raw, resolve=True)
        if not isinstance(paths_rows, list):
            paths_rows = None

    npy = _opt_path(OmegaConf.select(cfg, "npy"))
    if npy is None and paths_rows:
        if slide < 0 or slide >= len(paths_rows):
            print(
                f"slide={slide} out of range for paths (len={len(paths_rows)})",
                file=sys.stderr,
            )
            sys.exit(1)
        npy = _npy_from_paths_list(paths_rows, slide)
        if npy is None:
            print(
                f"paths[{slide}] must be [npy_path, optional_label] or {{npy: path}}",
                file=sys.stderr,
            )
            sys.exit(1)

    index_raw = OmegaConf.select(cfg, "index")
    if index_raw is not None and str(index_raw).strip() != "":
        index_str = str(index_raw).strip()
    elif npy is not None:
        index_str = annotation_index_from_npy_path(npy)
    else:
        print(
            "Set index=, or npy= / paths=+slide= (index defaults to stem of .npy, as in notebooks)",
            file=sys.stderr,
        )
        sys.exit(1)
    background = _opt_path(OmegaConf.select(cfg, "background"))
    height = OmegaConf.select(cfg, "height")
    width = OmegaConf.select(cfg, "width")
    spatial_mode = resolve_transpose_spatial_mode(cfg)
    ignore = list(OmegaConf.to_container(cfg.ignore, resolve=True) or [])
    out_path = _opt_path(OmegaConf.select(cfg, "out"))
    show = bool(cfg.show)
    boundary_width = int(cfg.boundary_width)

    category_keys = _category_keys_from_cfg(cfg)
    keep = _keep_from_cfg(OmegaConf.select(cfg, "keep"))

    polys, cats = load_annotation_polygons(
        annotation_json,
        index_str,
        ignore=ignore,
        keep=keep,
        category_keys=category_keys,
    )
    if not polys:
        print(f"No regions for index prefix {index_str!r}", file=sys.stderr)
        sys.exit(2)

    h0, w0 = _infer_hw_from_polygons(polys)
    shape_hw = (height or h0, width or w0)
    if height is None and width is None:
        if npy is not None and npy.suffix.lower() == ".npy":
            sh = spatial_hw_from_npy(npy)
            if sh is not None:
                shape_hw = spatial_hw_after_transpose_mode(sh, spatial_mode)
        elif background is not None:
            if background.suffix.lower() == ".npy":
                sh = spatial_hw_from_npy(background)
                if sh is not None:
                    shape_hw = spatial_hw_after_transpose_mode(sh, spatial_mode)
            else:
                bg_try = load_background(background, None, spatial_mode="none")
                if bg_try is not None:
                    shape_hw = (bg_try.shape[0], bg_try.shape[1])

    bg_path = background or npy
    bg = load_background(bg_path, shape_hw, spatial_mode=spatial_mode) if bg_path else None
    out = render_overlay(bg, polys, cats, shape_hw, boundary_width=boundary_width)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"Wrote {out_path}")

    if show or not out_path:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 12))
            plt.imshow(out)
            plt.axis("off")
            plt.title(f"{index_str} | {len(polys)} regions")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            if show:
                raise
            if not out_path:
                print("Set out=path.png or show=true; matplotlib not available.", file=sys.stderr)
                print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
