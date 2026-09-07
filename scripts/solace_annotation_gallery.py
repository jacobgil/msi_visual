#!/usr/bin/env python3
"""
Overlay VIA brain-region annotations on SoLaCE edge maps and compose a gallery.

Uses the same ``chain_flip`` spatial alignment as ``view_annotations_on_methods``.
SoLaCE can be loaded from a prior ``hd_methods_gallery_solace_baselines`` raw grayscale
export, or recomputed.

Run (repo root):
  python scripts/solace_annotation_gallery.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from debug_edge_maps import (  # noqa: E402
    _gray_u8_to_rgb_colormap,
    _load_msi,
)
from hd_methods_gallery import _human_npy_slug  # noqa: E402
from view_annotation_boundaries import (  # noqa: E402
    apply_annotation_spatial_transform,
    load_annotation_polygons,
    render_overlay,
    resolve_transpose_spatial_mode,
    spatial_hw_after_transpose_mode,
    spatial_hw_from_npy,
)

logger = logging.getLogger(__name__)


def _opt_path(value: Any) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "~", "null", "None"):
        return None
    return Path(to_absolute_path(s)).expanduser().resolve()


def _load_ui_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    size = max(8, int(size))
    candidates: list[str] = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\arialbd.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _annotation_index_from_npy(npy: Path) -> str:
    return npy.stem


def _resolve_slide_rows(ann: Any) -> list[tuple[Path, str, str | None]]:
    """Return list of (npy_path, via_index_prefix, optional_label)."""
    paths_raw = OmegaConf.to_container(getattr(ann, "paths", None), resolve=True)
    if not isinstance(paths_raw, (list, tuple)) or not paths_raw:
        raise ValueError("annotation.paths must be a non-empty list of [npy, label] rows")

    rows: list[tuple[Path, Any]] = []
    for row in paths_raw:
        if not isinstance(row, (list, tuple)) or len(row) < 1:
            raise ValueError(f"Bad annotation.paths row: {row!r}")
        rows.append((Path(to_absolute_path(str(row[0]))).expanduser().resolve(), row[1] if len(row) > 1 else None))

    slide = getattr(ann, "slide", "all")
    if isinstance(slide, str) and slide.strip().lower() in ("all", "*"):
        selected = list(range(len(rows)))
    elif isinstance(slide, (list, tuple)):
        selected = [int(x) for x in slide]
    else:
        selected = [int(slide)]

    fixed_index = getattr(ann, "index", None)
    fixed = str(fixed_index).strip() if fixed_index is not None and str(fixed_index).strip() not in ("", "null", "None") else None

    out: list[tuple[Path, str, str | None]] = []
    for i in selected:
        if i < 0 or i >= len(rows):
            raise IndexError(f"annotation.slide index {i} out of range for {len(rows)} paths")
        npy, lab = rows[i]
        if not npy.is_file():
            raise FileNotFoundError(f"Missing MSI: {npy}")
        idx = fixed if fixed is not None else _annotation_index_from_npy(npy)
        out.append((npy, idx, None if lab is None else str(lab)))
    return out


def _find_cached_solace(raw_root: Path, npy: Path, method_key: str) -> Path | None:
    """Match hd_methods_gallery slug folders under raw_grayscale/."""
    if not raw_root.is_dir():
        return None
    slug = _human_npy_slug(npy)
    exact = raw_root / slug / f"{method_key}.png"
    if exact.is_file():
        return exact
    # Prefix match (disambiguated slugs like slug__deadbeef)
    hits: list[Path] = []
    for d in raw_root.iterdir():
        if d.is_dir() and (d.name == slug or d.name.startswith(slug + "__")):
            png = d / f"{method_key}.png"
            if png.is_file():
                hits.append(png)
    if hits:
        hits.sort(key=lambda p: len(p.parent.name))
        return hits[0]
    return None


def _compute_solace_gray(
    npy: Path,
    *,
    transpose_msi: bool,
    normalization: str,
    spatial_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray_u8 H×W, valid_mask) in annotation spatial frame."""
    from hd_methods_gallery import _compute_merged_hd_edges, _spatial_norm_edge
    from hydra import compose, initialize_config_dir

    msi = _load_msi(npy, transpose_msi=transpose_msi, normalization=normalization)
    if spatial_mode not in ("none", "off", "identity", ""):
        msi = apply_annotation_spatial_transform(msi, spatial_mode)
    valid_mask = msi.sum(axis=-1) > 0

    with initialize_config_dir(version_base=None, config_dir=str(_SCRIPTS / "configs")):
        cfg = compose(config_name="hd_methods_solace_baselines")
    merged, _ = _compute_merged_hd_edges(cfg, msi, valid_mask, "soft_landmark_contrast", msi_raw=None)
    disp = _spatial_norm_edge(cfg, merged, valid_mask)
    from debug_edge_maps import _to_uint8_gray01

    return _to_uint8_gray01(disp), np.asarray(valid_mask, dtype=bool)


def _load_solace_gray_aligned(
    npy: Path,
    *,
    source: str,
    raw_root: Path | None,
    method_key: str,
    spatial_mode: str,
    fallback_compute: bool,
    transpose_msi: bool,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Gray SoLaCE + valid mask in annotation-aligned (H', W') frame."""
    src = str(source).strip().lower()
    file_hw = spatial_hw_from_npy(npy)
    if file_hw is None:
        raise ValueError(f"Cannot read shape from {npy}")
    target_hw = spatial_hw_after_transpose_mode(file_hw, spatial_mode)

    if src in ("cached", "cache", "png"):
        if raw_root is None:
            raise ValueError("solace.solace_raw_root required when source=cached")
        png = _find_cached_solace(raw_root, npy, method_key)
        if png is not None:
            gray = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
            # Cached SoLaCE was computed without chain_flip → apply same spatial transform.
            if spatial_mode not in ("none", "off", "identity", ""):
                gray = apply_annotation_spatial_transform(gray, spatial_mode)
            if gray.shape[:2] != target_hw:
                logger.warning(
                    "SoLaCE shape %s != target %s for %s — resizing",
                    gray.shape[:2],
                    target_hw,
                    npy.name,
                )
                gray = cv2.resize(gray, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
            valid = gray > 0
            return gray, valid
        if not fallback_compute:
            raise FileNotFoundError(f"No cached SoLaCE for {npy} under {raw_root}")
        logger.warning("Cached SoLaCE missing for %s — computing", npy.name)
        src = "compute"

    if src in ("compute", "recompute", "run"):
        return _compute_solace_gray(
            npy,
            transpose_msi=transpose_msi,
            normalization=normalization,
            spatial_mode=spatial_mode,
        )
    raise ValueError(f"Unknown solace.source: {source!r}")


def _colorize_solace(gray: np.ndarray, valid: np.ndarray, cmap: str) -> np.ndarray:
    try:
        return _gray_u8_to_rgb_colormap(gray, cmap, valid_mask=valid)
    except Exception as e:
        logger.warning("Colormap %r failed (%s); using gray", cmap, e)
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[~valid] = 0
        return rgb


def _compose_gallery(
    panels: list[np.ndarray],
    titles: list[str],
    *,
    columns: int,
    cell_gap: int,
    margin: int,
    show_titles: bool,
    title_font_size: int,
) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to compose")
    n = len(panels)
    cols = max(1, min(int(columns), n))
    rows = int(np.ceil(n / cols))
    title_h = int(title_font_size) + 16 if show_titles else 0
    cell_h = max(int(p.shape[0]) for p in panels) + title_h
    cell_w = max(int(p.shape[1]) for p in panels)
    gap = max(0, int(cell_gap))
    m = max(0, int(margin))
    total_w = m * 2 + cols * cell_w + gap * max(0, cols - 1)
    total_h = m * 2 + rows * cell_h + gap * max(0, rows - 1)
    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_ui_font(title_font_size, bold=True)

    for i, (panel, title) in enumerate(zip(panels, titles)):
        r, c = divmod(i, cols)
        x0 = m + c * (cell_w + gap)
        y0 = m + r * (cell_h + gap)
        if show_titles and title:
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x0 + max(0, (cell_w - tw) // 2), y0 + 4), title, fill=(230, 230, 230), font=font)
        im = Image.fromarray(np.asarray(panel, dtype=np.uint8), mode="RGB")
        dx = max(0, (cell_w - im.size[0]) // 2)
        dy = title_h + max(0, (cell_h - title_h - im.size[1]) // 2)
        canvas.paste(im, (x0 + dx, y0 + dy))
    return np.asarray(canvas, dtype=np.uint8)


def _legend_image(categories: list[str], colors: dict[str, tuple[int, int, int]]) -> np.ndarray:
    cats = sorted(set(categories))
    font = _load_ui_font(16)
    row_h = 26
    swatch = 18
    # Measure widest label
    probe = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(probe)
    max_tw = 0
    for c in cats:
        bbox = draw.textbbox((0, 0), c, font=font)
        max_tw = max(max_tw, bbox[2] - bbox[0])
    w = 16 + swatch + 10 + max_tw + 16
    h = 12 + row_h * max(1, len(cats)) + 12
    im = Image.new("RGB", (w, h), (18, 18, 20))
    draw = ImageDraw.Draw(im)
    y = 12
    for c in cats:
        rgb = colors.get(c, (180, 180, 180))
        draw.rectangle((12, y + 3, 12 + swatch, y + 3 + swatch), fill=rgb, outline=(240, 240, 240))
        draw.text((12 + swatch + 10, y + 2), c, fill=(235, 235, 235), font=font)
        y += row_h
    return np.asarray(im, dtype=np.uint8)


def _category_colors(categories: list[str], seed: int = 42) -> dict[str, tuple[int, int, int]]:
    uniq = sorted(set(categories))
    rng = np.random.default_rng(seed)
    # Match render_overlay palette seed
    pal = (rng.random((max(len(uniq), 1), 3)) * 255).astype(np.uint8)
    return {c: (int(pal[i, 0]), int(pal[i, 1]), int(pal[i, 2])) for i, c in enumerate(uniq)}


@hydra.main(version_base=None, config_path="configs", config_name="solace_annotation_gallery")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ann = cfg.annotation
    sol = cfg.solace
    gal = cfg.gallery

    annotation_json = _opt_path(ann.annotation_json)
    if annotation_json is None or not annotation_json.is_file():
        raise FileNotFoundError(f"annotation_json not found: {annotation_json}")

    spatial_mode = resolve_transpose_spatial_mode(ann)
    slides = _resolve_slide_rows(ann)
    raw_root = _opt_path(getattr(sol, "solace_raw_root", None))
    method_key = str(getattr(sol, "method_key", "soft_landmark_contrast"))
    cmap = str(getattr(sol, "colormap", "PuBu"))
    source = str(getattr(sol, "source", "cached"))
    fallback = bool(getattr(sol, "fallback_compute", True))
    transpose_msi = bool(getattr(sol, "transpose_msi", False))
    normalization = str(getattr(sol, "normalization", "tic"))

    ignore = list(OmegaConf.to_container(getattr(ann, "ignore", []), resolve=True) or [])
    keep_raw = getattr(ann, "keep", None)
    keep = set(OmegaConf.to_container(keep_raw, resolve=True)) if keep_raw is not None else None
    cat_keys = tuple(
        str(x) for x in (OmegaConf.to_container(getattr(ann, "category_keys", None), resolve=True) or [])
    ) or ("main_cat1", "Allen_name", "sub_cat2")
    fill_alpha = float(getattr(ann, "fill_alpha", 0.22))
    boundary_width = int(getattr(ann, "boundary_width", 1))
    boundary_halo = int(getattr(ann, "boundary_halo", 0))
    fill_rgb_raw = OmegaConf.to_container(getattr(ann, "fill_rgb", [255, 180, 40]), resolve=True)
    boundary_rgb_raw = OmegaConf.to_container(getattr(ann, "boundary_rgb", [255, 245, 220]), resolve=True)
    fill_rgb = tuple(int(x) for x in fill_rgb_raw) if fill_rgb_raw is not None else (255, 180, 40)
    boundary_rgb = tuple(int(x) for x in boundary_rgb_raw) if boundary_rgb_raw is not None else (255, 245, 220)

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    pan_dir = out_dir / "panels"
    pan_dir.mkdir(parents=True, exist_ok=True)

    overlay_panels: list[np.ndarray] = []
    titles: list[str] = []
    all_cats: list[str] = []

    for npy, index_str, path_label in slides:
        logger.info("Slide %s (VIA index %s)", npy.name, index_str)
        polys, cats = load_annotation_polygons(
            annotation_json,
            index_str,
            ignore=ignore,
            keep=keep,
            category_keys=cat_keys,
        )
        if not polys:
            logger.warning("No regions for index %s — skip", index_str)
            continue
        all_cats.extend(cats)

        gray, valid = _load_solace_gray_aligned(
            npy,
            source=source,
            raw_root=raw_root,
            method_key=method_key,
            spatial_mode=spatial_mode,
            fallback_compute=fallback,
            transpose_msi=transpose_msi,
            normalization=normalization,
        )
        solace_rgb = _colorize_solace(gray, valid, cmap)
        h, w = solace_rgb.shape[:2]
        overlay = render_overlay(
            solace_rgb,
            polys,
            cats,
            (h, w),
            boundary_width=boundary_width,
            fill_alpha=fill_alpha,
            boundary_halo=boundary_halo,
            uniform_fill_rgb=fill_rgb,
            boundary_rgb=boundary_rgb,
        )

        stem = npy.stem
        title = f"Hemisphere {len(titles) + 1}"
        titles.append(title)
        overlay_panels.append(overlay)

        Image.fromarray(overlay, mode="RGB").save(pan_dir / f"{stem}__solace_regions.png")
        if bool(getattr(gal, "save_solace_only", True)):
            Image.fromarray(solace_rgb, mode="RGB").save(pan_dir / f"{stem}__solace_only.png")
        logger.info("  regions=%d  shape=%dx%d", len(polys), h, w)

    if not overlay_panels:
        raise SystemExit("No panels produced — check annotation JSON / SoLaCE cache.")

    mosaic = _compose_gallery(
        overlay_panels,
        titles,
        columns=int(getattr(gal, "columns", 4)),
        cell_gap=int(getattr(gal, "cell_gap", 12)),
        margin=int(getattr(gal, "margin", 8)),
        show_titles=bool(getattr(gal, "show_titles", True)),
        title_font_size=int(getattr(gal, "title_font_size", 22)),
    )
    dpi = int(getattr(gal, "dpi", 200))
    mosaic_path = out_dir / "solace_regions_gallery.png"
    Image.fromarray(mosaic, mode="RGB").save(mosaic_path, dpi=(dpi, dpi))
    logger.info("Wrote %s (%dx%d)", mosaic_path, mosaic.shape[1], mosaic.shape[0])

    if bool(getattr(gal, "save_legend", False)) and all_cats:
        colors = _category_colors(all_cats, seed=42)
        legend = _legend_image(all_cats, colors)
        legend_path = out_dir / "region_legend.png"
        Image.fromarray(legend, mode="RGB").save(legend_path)
        logger.info("Wrote %s (%d categories)", legend_path, len(set(all_cats)))

    readme = out_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "SoLaCE + VIA region overlay gallery",
                f"annotation_json: {annotation_json}",
                f"spatial_mode: {spatial_mode}",
                f"solace.source: {source}",
                f"colormap: {cmap}",
                f"gallery: {mosaic_path.name}",
                f"panels: {pan_dir.name}/",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[solace_annotation_gallery] Saved {mosaic_path}", flush=True)


if __name__ == "__main__":
    main()
