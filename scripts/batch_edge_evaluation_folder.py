#!/usr/bin/env python3
"""
Batch edge_evaluation metrics on existing visualization PNGs using benchmark_parametric_methods config.

Efficiency: groups images by MSI using the same file_id prefix as the benchmark
(`_path_identifier`), loads each .npy once, builds HD edges once per MSI, then scores each viz.

Filenames must match: {file_id}__r{repeat}__s{seed}__{sampling}__{method...}.png
where file_id is computed from `data.npy_paths` entries (see benchmark_parametric_methods.py).
"""

from __future__ import annotations

import csv
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from scripts.benchmark_parametric_methods import (
    _build_metric_mask_from_tic,
    _compute_visualization_quality_metrics,
    _load_existing_viz_rgb,
    _load_msi,
    _load_raw_msi,
    _maybe_lab_to_rgb,
    _path_identifier,
)
from scripts.debug_visualization_metrics import DebugNMFEvaluator

logger = logging.getLogger(__name__)

_VIZ_METHOD_RE = re.compile(r"^r\d+__s\d+__[^_]+__(.+)$")


def _iter_viz_paths(folder: Path, recursive: bool, extensions: List[str]) -> List[Path]:
    exts = {e.lower() if str(e).startswith(".") else f".{str(e).lower()}" for e in extensions}
    out: List[Path] = []
    if recursive:
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    return sorted(out)


def _parse_method_token(stem: str, file_id: str) -> Optional[str]:
    """Return method segment after r*__s*__sampling__ (e.g. parametric_umap or parametric_mics_lmc__mics1)."""
    prefix = file_id + "__"
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix) :]
    m = _VIZ_METHOD_RE.match(rest)
    return m.group(1) if m else None


def _method_name_for_lab_rgb(method_token: str) -> str:
    """Map filename token to benchmark.lab_to_rgb_methods entries (e.g. ...__mics1 -> parametric_mics_lmc)."""
    t = method_token.strip()
    if t.startswith("parametric_mics_lmc"):
        return "parametric_mics_lmc"
    return t.split("__")[0] if "__" in t else t


def _build_npy_to_file_id(cfg: DictConfig) -> Dict[Path, str]:
    """Resolved absolute path -> file_id (same string embedded in benchmark PNG names)."""
    paths = getattr(cfg.data, "npy_paths", None) or []
    out: Dict[Path, str] = {}
    for p in paths:
        path = Path(str(p)).expanduser().resolve()
        if not path.exists():
            logger.warning("data.npy_paths entry not found (skipped for matching): %s", path)
            continue
        if path.suffix.lower() != ".npy":
            logger.warning("Not a .npy file, skipped: %s", path)
            continue
        fid = _path_identifier(path)
        out[path] = fid
    return out


def _group_viz_by_msi(
    viz_paths: List[Path],
    file_ids_longest_first: List[str],
    npy_by_fid: Dict[str, Path],
) -> Tuple[DefaultDict[Path, List[Path]], List[Path]]:
    """
    Match each viz stem to the longest file_id prefix; map to npy path.
    Unmatched paths are returned separately.
    """
    groups: DefaultDict[Path, List[Path]] = defaultdict(list)
    unmatched: List[Path] = []

    for vp in viz_paths:
        stem = vp.stem
        fid: Optional[str] = None
        for candidate in file_ids_longest_first:
            if stem.startswith(candidate + "__"):
                fid = candidate
                break
        if fid is None or fid not in npy_by_fid:
            unmatched.append(vp)
            continue
        groups[npy_by_fid[fid]].append(vp)

    for k in groups:
        groups[k].sort(key=lambda p: p.name)
    return groups, unmatched


@hydra.main(version_base=None, config_path="configs", config_name="batch_edge_evaluation_folder")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    batch = getattr(cfg, "batch", None)
    if batch is None:
        raise ValueError("Config must define a `batch` section (see batch_edge_evaluation_folder.yaml)")

    viz_folder = Path(str(batch.visualization_folder)).expanduser().resolve()
    if not viz_folder.is_dir():
        raise FileNotFoundError(f"batch.visualization_folder is not a directory: {viz_folder}")

    npy_to_fid = _build_npy_to_file_id(cfg)
    if not npy_to_fid:
        raise ValueError(
            "No valid entries in data.npy_paths — need at least one existing .npy to match filenames."
        )

    fid_to_npy = {fid: p for p, fid in npy_to_fid.items()}
    file_ids_longest_first = sorted(fid_to_npy.keys(), key=len, reverse=True)

    recursive = bool(getattr(batch, "recursive", False))
    extensions = [str(x) for x in list(getattr(batch, "extensions", [".png"]))]
    viz_paths = _iter_viz_paths(viz_folder, recursive=recursive, extensions=extensions)
    if not viz_paths:
        raise ValueError(f"No images found in {viz_folder} with extensions {extensions}")

    groups, unmatched = _group_viz_by_msi(viz_paths, file_ids_longest_first, fid_to_npy)

    print(
        f"[batch_edge] folder={viz_folder} | images={len(viz_paths)} | "
        f"MSI groups={len(groups)} | unmatched={len(unmatched)}",
        flush=True,
    )
    if unmatched:
        for u in unmatched[:20]:
            print(f"[batch_edge] unmatched (no data.npy_paths file_id prefix): {u.name}", flush=True)
        if len(unmatched) > 20:
            print(f"[batch_edge] ... and {len(unmatched) - 20} more unmatched", flush=True)

    fail_unmatched = bool(getattr(batch, "fail_on_unmatched", False))
    if fail_unmatched and unmatched:
        raise RuntimeError(f"{len(unmatched)} file(s) did not match any configured MSI file_id")

    try:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_name = str(getattr(batch, "output_csv", "edge_agreement_folder.csv"))
    csv_path = out_dir / csv_name

    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    apply_lab = bool(getattr(batch, "apply_lab_to_rgb_from_filename", True))
    rows_out: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for npy_path in sorted(groups.keys(), key=lambda p: str(p)):
        vps = groups[npy_path]
        fid = npy_to_fid[npy_path]
        print(f"[batch_edge] MSI {npy_path.name} | file_id={fid} | {len(vps)} viz", flush=True)
        try:
            raw_msi_img = _load_raw_msi(npy_path, bool(cfg.data.transpose_msi))
            msi_img = _load_msi(npy_path, bool(cfg.data.transpose_msi), bool(cfg.data.tic_normalize))
            metric_mask = _build_metric_mask_from_tic(raw_msi_img, cfg)
            metric_mask_effective = metric_mask & (msi_img.sum(axis=-1) > 0)
            edge_eval = DebugNMFEvaluator(msi_img, cfg, edge_only=True)
        except Exception as e:
            logger.exception("Failed to build evaluator for %s", npy_path)
            for vp in vps:
                rows_out.append(
                    {
                        "npy_path": str(npy_path),
                        "file_id": fid,
                        "viz_path": str(vp),
                        "viz_filename": vp.name,
                        "status": "evaluator_failed",
                        "error": str(e),
                    }
                )
            continue

        for vp in vps:
            method_token = _parse_method_token(vp.stem, fid)
            row_base: Dict[str, Any] = {
                "npy_path": str(npy_path),
                "file_id": fid,
                "viz_path": str(vp),
                "viz_filename": vp.name,
                "parsed_method": method_token or "",
                "status": "ok",
                "error": "",
            }
            try:
                viz_rgb = _load_existing_viz_rgb(vp, expected_hw=msi_img.shape[:2])
                if apply_lab and method_token:
                    viz_rgb = _maybe_lab_to_rgb(viz_rgb, _method_name_for_lab_rgb(method_token), cfg)
                updates = edge_eval.evaluate_edge_evaluation_only(viz_rgb)
                updates.update(_compute_visualization_quality_metrics(viz_rgb, metric_mask_effective))
                for k, v in updates.items():
                    if isinstance(v, str):
                        row_base[k] = v
                    elif v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        row_base[k] = v
                    else:
                        row_base[k] = float(v) if isinstance(v, (float, np.floating)) else v
                rows_out.append(row_base)
            except Exception as e:
                row_base["status"] = "viz_failed"
                row_base["error"] = str(e)
                rows_out.append(row_base)
                logger.warning("viz failed %s: %s", vp, e)

        del edge_eval
        del msi_img
        del raw_msi_img

    # Stable column order: base keys first, then remaining keys from first data row
    base_cols = [
        "npy_path",
        "file_id",
        "viz_path",
        "viz_filename",
        "parsed_method",
        "status",
        "error",
    ]
    extra: List[str] = []
    for r in rows_out:
        for k in r:
            if k not in base_cols and k not in extra:
                extra.append(k)
    fieldnames = base_cols + sorted(extra)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    dt = time.perf_counter() - t0
    print(f"[batch_edge] wrote {len(rows_out)} row(s) -> {csv_path} | {dt:.2f}s", flush=True)


if __name__ == "__main__":
    main()
