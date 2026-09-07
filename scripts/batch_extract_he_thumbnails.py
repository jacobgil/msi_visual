"""
Batch H&E whole-slide thumbnail extraction.

Scans ``extract.input_root`` (e.g. CarstenHopf HEStaining folder) for WSI files
(NDPI, SVS, TIFF, …) and writes large RGB thumbnails under ``extract.output_root``.

Example:
  python scripts/batch_extract_he_thumbnails.py
  python scripts/batch_extract_he_thumbnails.py extract.dry_run=true
  python scripts/batch_extract_he_thumbnails.py extract.max_size=8192
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msi_visual.extract.he_thumbnail import DEFAULT_WSI_EXTENSIONS, extract_he_thumbnail, find_wsi_files

logger = logging.getLogger(__name__)


def _resolve_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _output_path_for_slide(
    input_root: Path,
    output_root: Path,
    slide_path: Path,
    *,
    preserve_relative_dirs: bool,
    output_filename: str,
    flat_output: bool,
) -> Path:
    stem = slide_path.stem
    if flat_output:
        return output_root / f"{stem}__{output_filename}"
    if preserve_relative_dirs:
        rel_parent = slide_path.parent.relative_to(input_root)
        if str(rel_parent) in (".", ""):
            return output_root / stem / output_filename
        return output_root / rel_parent / stem / output_filename
    return output_root / stem / output_filename


def _parse_extensions(raw) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_WSI_EXTENSIONS
    items = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(items, (list, tuple)) or not items:
        return DEFAULT_WSI_EXTENSIONS
    out: list[str] = []
    for item in items:
        s = str(item).strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = f".{s}"
        out.append(s)
    return tuple(out) if out else DEFAULT_WSI_EXTENSIONS


@hydra.main(version_base=None, config_path="configs", config_name="batch_extract_he_thumbnails")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ex = cfg.extract

    input_root = _resolve_path(ex.input_root)
    output_root = _resolve_path(ex.output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"extract.input_root not found: {input_root}")

    max_size = int(ex.max_size)
    level_raw = ex.level
    level = None if level_raw is None or str(level_raw).strip().lower() in ("", "null", "none", "~") else int(level_raw)
    image_format = str(ex.format).strip().lower()
    jpeg_quality = int(ex.jpeg_quality)
    recursive = bool(ex.recursive)
    skip_existing = bool(ex.skip_existing)
    dry_run = bool(ex.dry_run)
    flat_output = bool(ex.flat_output)
    preserve_relative_dirs = bool(ex.preserve_relative_dirs)
    output_filename = str(ex.output_filename).strip() or "he_thumbnail.png"
    save_metadata = bool(ex.save_metadata)

    include_glob = ex.include_glob
    include_glob = None if include_glob is None or str(include_glob).strip().lower() in ("", "null", "none") else str(include_glob)

    extensions = _parse_extensions(ex.extensions)
    slide_files = find_wsi_files(
        input_root,
        recursive=recursive,
        extensions=extensions,
        include_glob=include_glob,
    )
    if not slide_files:
        raise FileNotFoundError(f"No WSI files ({', '.join(extensions)}) under {input_root}")

    logger.info("Found %d slide file(s) under %s", len(slide_files), input_root)
    logger.info("Output root: %s | max_size=%d | format=%s", output_root, max_size, image_format)

    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        run_dir = Path.cwd()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    n_ok = n_skip = n_fail = 0

    for idx, slide_path in enumerate(slide_files, start=1):
        out_path = _output_path_for_slide(
            input_root,
            output_root,
            slide_path,
            preserve_relative_dirs=preserve_relative_dirs,
            output_filename=output_filename,
            flat_output=flat_output,
        )
        row = {
            "index": str(idx),
            "slide": str(slide_path),
            "output_path": str(out_path),
            "status": "",
            "message": "",
        }

        if dry_run:
            row["status"] = "dry_run"
            manifest_rows.append(row)
            logger.info("[%d/%d] dry-run: %s -> %s", idx, len(slide_files), slide_path.name, out_path)
            continue

        if skip_existing and out_path.is_file():
            row["status"] = "skipped"
            row["message"] = "thumbnail exists"
            n_skip += 1
            logger.info("[%d/%d] skip (exists): %s", idx, len(slide_files), out_path)
            manifest_rows.append(row)
            continue

        logger.info("[%d/%d] extracting: %s -> %s", idx, len(slide_files), slide_path.name, out_path)
        try:
            meta = extract_he_thumbnail(
                slide_path,
                out_path,
                max_size=max_size,
                level=level,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
                save_metadata=save_metadata,
            )
            row["status"] = "ok"
            row["message"] = f"{meta.get('backend')} {meta.get('output_shape')}"
            n_ok += 1
        except Exception as exc:
            row["status"] = "failed"
            row["message"] = str(exc)
            n_fail += 1
            logger.exception("Thumbnail failed: %s", slide_path)

        manifest_rows.append(row)

    if bool(ex.write_manifest):
        manifest_path = run_dir / "batch_he_thumbnails_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["index", "slide", "output_path", "status", "message"],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        logger.info("Manifest: %s", manifest_path)

    resolved_cfg = run_dir / "config_resolved.yaml"
    with resolved_cfg.open("w", encoding="utf-8") as fh:
        fh.write(OmegaConf.to_yaml(cfg, resolve=True))

    logger.info("Done: ok=%d skipped=%d failed=%d | total=%d", n_ok, n_skip, n_fail, len(slide_files))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
