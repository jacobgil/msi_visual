"""Extract downsampled RGB thumbnails from H&E whole-slide images."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_WSI_EXTENSIONS = (
    ".ndpi",
    ".svs",
    ".tif",
    ".tiff",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
)

_SVS_AUX_KEYWORDS = ("label", "macro", "thumbnail", "overview")


def _save_rgb(arr: np.ndarray, out_path: Path, *, fmt: str, jpeg_quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    fmt_l = fmt.strip().lower()
    if fmt_l in ("jpg", "jpeg"):
        img.save(out_path, format="JPEG", quality=int(jpeg_quality), optimize=True)
    else:
        img.save(out_path, format="PNG", optimize=True)


def _resize_longest_side(rgb: np.ndarray, max_size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return rgb
    scale = max_size / float(longest)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = Image.fromarray(rgb)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _series_is_auxiliary(series: Any) -> bool:
    desc = str(getattr(series, "description", "") or "").lower()
    return any(k in desc for k in _SVS_AUX_KEYWORDS)


def _series_level_shapes(series: Any) -> list[tuple[int, int]]:
    levels = list(getattr(series, "levels", None) or [series])
    shapes: list[tuple[int, int]] = []
    for lv in levels:
        shape = getattr(lv, "shape", None)
        if shape is None or len(shape) < 2:
            continue
        shapes.append((int(shape[-2]), int(shape[-1])))
    return shapes


def _pick_pyramid_level(series: Any, max_size: int) -> Any:
    levels = list(getattr(series, "levels", None) or [series])
    best_lv = levels[0]
    best_score: tuple[int, int] | None = None
    for lv in levels:
        shape = getattr(lv, "shape", None)
        if shape is None or len(shape) < 2:
            continue
        h, w = int(shape[-2]), int(shape[-1])
        longest = max(h, w)
        if longest <= max_size:
            score = (longest, -levels.index(lv))
        else:
            score = (max_size - longest, -levels.index(lv))
        if best_score is None or score > best_score:
            best_score = score
            best_lv = lv
    return best_lv


def _array_to_rgb_u8(arr: np.ndarray, slide_path: Path) -> np.ndarray:
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        raise ValueError(f"Unsupported array shape {arr.shape} in {slide_path}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


MICRONS_PER_INCH = 25400.0


def mpp_from_dpi(dpi: float) -> float:
    """Microns per pixel for a given print/display DPI."""
    return MICRONS_PER_INCH / max(1.0, float(dpi))


def target_dpi_for_long_side(
    *,
    level0_width: int,
    level0_height: int,
    slide_mpp: float,
    min_long_side: int,
) -> float:
    """DPI so the whole-slide read has at least ``min_long_side`` pixels on its long edge."""
    min_long_side = max(1, int(min_long_side))
    l0_long = max(int(level0_width), int(level0_height))
    downsample = l0_long / float(min_long_side)
    target_mpp = max(1e-6, float(slide_mpp) * downsample)
    return MICRONS_PER_INCH / target_mpp


def _resolve_slide_mpp(slide: Any, *, fallback_mpp: float) -> float:
    import openslide

    vals: list[float] = []
    for key in (openslide.PROPERTY_NAME_MPP_X, openslide.PROPERTY_NAME_MPP_Y):
        raw = slide.properties.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v > 0:
            vals.append(v)
    if vals:
        return float(np.mean(vals))
    return float(fallback_mpp)


def extract_he_from_wsi_at_dpi(
    slide_path: Path,
    *,
    target_dpi: float = 400.0,
    max_size: int | None = None,
    min_long_side: int | None = None,
    max_dpi: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a full-slide RGB overview from OpenSlide at approximately ``target_dpi``.

    Picks the nearest pyramid level via ``get_best_level_for_downsample``, then
    resizes to the pixel size implied by slide MPP and ``target_dpi``
    (``target_mpp / slide_mpp`` downsample from level 0).

    When ``min_long_side`` is set, ``target_dpi`` is raised so the output long edge
    is at least that many pixels (subject to ``max_size`` / ``max_dpi`` caps).
    """
    _bootstrap_openslide()
    import openslide

    slide_path = Path(slide_path).expanduser().resolve()
    if not slide_path.is_file():
        raise FileNotFoundError(slide_path)

    base_dpi = max(1.0, float(target_dpi))
    slide = openslide.OpenSlide(str(slide_path))
    try:
        mpp = _resolve_slide_mpp(slide, fallback_mpp=mpp_from_dpi(base_dpi))
        w0, h0 = slide.dimensions
        target_dpi = base_dpi
        if min_long_side is not None:
            need_dpi = target_dpi_for_long_side(
                level0_width=w0,
                level0_height=h0,
                slide_mpp=mpp,
                min_long_side=int(min_long_side),
            )
            target_dpi = max(target_dpi, need_dpi)
        if max_dpi is not None:
            target_dpi = min(float(target_dpi), float(max_dpi))
        target_mpp = mpp_from_dpi(target_dpi)
        # slide mpp < target_mpp => slide is higher-res than target => downsample > 1
        downsample = max(1.0, float(target_mpp) / float(mpp))
        tw = max(1, int(round(w0 / downsample)))
        th = max(1, int(round(h0 / downsample)))
        capped = False
        if max_size is not None:
            cap = max(64, int(max_size))
            longest = max(tw, th)
            if longest > cap:
                scale = cap / float(longest)
                tw = max(1, int(round(tw * scale)))
                th = max(1, int(round(th * scale)))
                capped = True
                downsample = max(1.0, float(w0) / float(tw))

        level = int(slide.get_best_level_for_downsample(downsample))
        level = max(0, min(level, slide.level_count - 1))
        level_ds = float(slide.level_downsamples[level])
        lw, lh = slide.level_dimensions[level]
        region = slide.read_region((0, 0), level, (lw, lh)).convert("RGB")
        img = Image.fromarray(np.asarray(region, dtype=np.uint8))
        if img.size != (tw, th):
            img = img.resize((tw, th), Image.Resampling.LANCZOS)
        rgb = np.asarray(img, dtype=np.uint8)

        effective_mpp = float(mpp) * (float(w0) / max(float(tw), 1.0))
        meta: dict[str, Any] = {
            "backend": "openslide",
            "source": str(slide_path),
            "base_target_dpi": base_dpi,
            "target_dpi": target_dpi,
            "target_mpp": target_mpp,
            "requested_min_long_side": int(min_long_side) if min_long_side is not None else None,
            "max_dpi": float(max_dpi) if max_dpi is not None else None,
            "slide_mpp": mpp,
            "downsample": downsample,
            "openslide_level": level,
            "openslide_level_downsample": level_ds,
            "level_dimensions": [int(lh), int(lw)],
            "effective_mpp": effective_mpp,
            "max_size_capped": capped,
            "level0_dimensions": [int(h0), int(w0)],
            "output_shape": [int(rgb.shape[0]), int(rgb.shape[1])],
        }
        return rgb, meta
    finally:
        slide.close()


def _bootstrap_openslide() -> None:
    """Load OpenSlide native library on Windows (conda DLL dir or openslide-bin wheel)."""
    try:
        import openslide_bin  # noqa: F401 — pip package bundles libopenslide on Windows
    except ImportError:
        pass
    if sys.platform != "win32":
        return
    for prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if not prefix:
            continue
        dll_dir = Path(prefix) / "Library" / "bin"
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))


def _extract_with_openslide(
    slide_path: Path,
    *,
    max_size: int,
    level: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    _bootstrap_openslide()
    import openslide

    slide = openslide.OpenSlide(str(slide_path))
    try:
        meta: dict[str, Any] = {
            "backend": "openslide",
            "dimensions_level0": list(slide.dimensions),
            "level_count": int(slide.level_count),
            "level_dimensions": [list(d) for d in slide.level_dimensions],
            "level_downsamples": [float(x) for x in slide.level_downsamples],
            "mpp_x": slide.properties.get(openslide.PROPERTY_NAME_MPP_X),
            "mpp_y": slide.properties.get(openslide.PROPERTY_NAME_MPP_Y),
            "vendor": slide.properties.get(openslide.PROPERTY_NAME_VENDOR),
        }
        if level is not None:
            lv = int(level)
            if lv < 0 or lv >= slide.level_count:
                raise ValueError(f"level={lv} out of range [0, {slide.level_count - 1}]")
            w, h = slide.level_dimensions[lv]
            region = slide.read_region((0, 0), lv, (w, h)).convert("RGB")
            rgb = np.asarray(region, dtype=np.uint8)
            meta["level_used"] = lv
            rgb = _resize_longest_side(rgb, max_size)
        else:
            thumb = slide.get_thumbnail((int(max_size), int(max_size)))
            rgb = np.asarray(thumb.convert("RGB"), dtype=np.uint8)
            meta["level_used"] = "thumbnail"
        meta["output_shape"] = [int(rgb.shape[0]), int(rgb.shape[1])]
        return rgb, meta
    finally:
        slide.close()


def _extract_with_tifffile(slide_path: Path, *, max_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    import imagecodecs  # noqa: F401 — required for JPEG-compressed SVS tiles
    import tifffile

    with tifffile.TiffFile(str(slide_path)) as tif:
        series_list = list(tif.series)
        if not series_list:
            raise ValueError(f"No TIFF series in {slide_path}")

        candidates = [s for s in series_list if not _series_is_auxiliary(s)]
        if not candidates:
            candidates = series_list

        def _area(s: Any) -> int:
            shapes = _series_level_shapes(s)
            if not shapes:
                return 0
            h, w = shapes[0]
            return h * w

        main_series = max(candidates, key=_area)
        level_series = _pick_pyramid_level(main_series, max_size)
        arr = level_series.asarray()
        rgb = _array_to_rgb_u8(arr, slide_path)
        rgb = _resize_longest_side(rgb, max_size)
        meta = {
            "backend": "tifffile",
            "series_index": int(series_list.index(main_series)),
            "series_count": len(series_list),
            "series_description": str(getattr(main_series, "description", "") or "")[:240],
            "level_shape": [int(x) for x in arr.shape],
            "output_shape": [int(rgb.shape[0]), int(rgb.shape[1])],
        }
        return rgb, meta


def _extract_with_pil(slide_path: Path, *, max_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(slide_path) as img:
        img = img.convert("RGB")
        rgb = np.asarray(img, dtype=np.uint8)
    rgb = _resize_longest_side(rgb, max_size)
    meta = {
        "backend": "pil",
        "output_shape": [int(rgb.shape[0]), int(rgb.shape[1])],
    }
    return rgb, meta


def _format_read_error(slide_path: Path, errors: list[str]) -> str:
    suffix = slide_path.suffix.lower()
    hint = (
        "For .svs / .ndpi on Windows install: "
        "conda install -c conda-forge imagecodecs && "
        "pip install openslide-bin openslide-python"
    )
    if suffix in (".svs", ".ndpi", ".scn", ".mrxs"):
        return f"Could not read {slide_path}. Attempts: {'; '.join(errors)}. {hint}"
    return f"Could not read {slide_path}. Attempts: {'; '.join(errors)}"


def extract_he_thumbnail(
    slide_path: Path,
    out_path: Path,
    *,
    max_size: int = 4096,
    level: int | None = None,
    image_format: str = "png",
    jpeg_quality: int = 92,
    save_metadata: bool = True,
) -> dict[str, Any]:
    """
    Write an RGB thumbnail for one whole-slide H&E image.

    Uses OpenSlide when available (NDPI, SVS, …), else tifffile+imagecodecs / PIL.
    """
    slide_path = Path(slide_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    if not slide_path.is_file():
        raise FileNotFoundError(slide_path)

    max_size = max(64, int(max_size))
    errors: list[str] = []

    rgb: np.ndarray | None = None
    meta: dict[str, Any] = {"source": str(slide_path), "max_size": max_size}

    try:
        rgb, backend_meta = _extract_with_openslide(slide_path, max_size=max_size, level=level)
        meta.update(backend_meta)
    except ImportError:
        errors.append("openslide not installed (pip install openslide-bin openslide-python)")
    except Exception as exc:
        errors.append(f"openslide: {exc}")

    if rgb is None:
        try:
            rgb, backend_meta = _extract_with_tifffile(slide_path, max_size=max_size)
            meta.update(backend_meta)
        except ImportError as exc:
            errors.append(f"tifffile/imagecodecs: {exc}")
        except Exception as exc:
            errors.append(f"tifffile: {exc}")

    if rgb is None:
        try:
            rgb, backend_meta = _extract_with_pil(slide_path, max_size=max_size)
            meta.update(backend_meta)
        except Exception as exc:
            errors.append(f"pil: {exc}")

    if rgb is None:
        raise RuntimeError(_format_read_error(slide_path, errors))

    _save_rgb(rgb, out_path, fmt=image_format, jpeg_quality=jpeg_quality)
    meta["output_path"] = str(out_path)
    logger.info(
        "Saved H&E thumbnail %s (%dx%d, backend=%s)",
        out_path.name,
        rgb.shape[1],
        rgb.shape[0],
        meta.get("backend"),
    )

    if save_metadata:
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    return meta


def find_wsi_files(
    root: Path,
    *,
    recursive: bool = True,
    extensions: tuple[str, ...] = DEFAULT_WSI_EXTENSIONS,
    include_glob: str | None = None,
) -> list[Path]:
    root = root.expanduser().resolve()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        if include_glob and not path.match(str(include_glob)):
            continue
        files.append(path)
    return files
