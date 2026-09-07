"""Path helpers for pairing MSI extractions with H&E thumbnails."""

from __future__ import annotations

import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


from msi_visual.extract.he_thumbnail import DEFAULT_WSI_EXTENSIONS, find_wsi_files


def slide_key_from_npy(npy_path: Path) -> str:
    if npy_path.parent.name in ("5_bins", "5bins"):
        return npy_path.parent.parent.name
    return npy_path.stem


def msi_sample_id(slide_key: str) -> str | None:
    m = re.search(r"(ID\d+[a-z]?)", slide_key, flags=re.IGNORECASE)
    return m.group(1) if m else None


def resolve_he_thumbnail(slide_key: str, he_cfg: DictConfig, *, resolve_path) -> Path | None:
    """Resolve H&E thumbnail path; ``resolve_path`` maps config strings to absolute Path."""
    if not bool(he_cfg.enabled):
        return None
    root = resolve_path(str(he_cfg.root))
    sample_id = msi_sample_id(slide_key)
    if sample_id is None:
        return None
    suffix = str(he_cfg.folder_suffix)
    fname = str(he_cfg.filename)
    direct = root / f"{sample_id}{suffix}" / fname
    if direct.is_file():
        return direct
    nested = root / slide_key / fname
    if nested.is_file():
        return nested
    for hit in root.rglob(f"{sample_id}*{fname}"):
        if hit.is_file():
            return hit
    return None


def resolve_he_wsi(slide_key: str, he_cfg: DictConfig, *, resolve_path) -> Path | None:
    """Resolve whole-slide H&E path (``.svs`` / ``.ndpi`` / …) for OpenSlide reads."""
    raw_root = OmegaConf.select(he_cfg, "wsi_root")
    if raw_root is None or not str(raw_root).strip():
        return None
    root = resolve_path(str(raw_root))
    if not root.is_dir():
        return None
    sample_id = msi_sample_id(slide_key)
    if sample_id is None:
        return None
    exts_raw = OmegaConf.select(he_cfg, "wsi_extensions")
    if exts_raw is None:
        extensions = DEFAULT_WSI_EXTENSIONS
    else:
        extensions = tuple(str(x) for x in OmegaConf.to_container(exts_raw, resolve=True))
    recursive = bool(OmegaConf.select(he_cfg, "wsi_recursive", default=True))
    sid = sample_id.lower()
    hits: list[Path] = []
    for path in find_wsi_files(root, recursive=recursive, extensions=extensions):
        stem = path.stem.lower()
        if sid in stem or stem.startswith(sid):
            hits.append(path)
    if not hits:
        return None
    return sorted(hits, key=lambda p: (len(p.name), str(p)))[0]
