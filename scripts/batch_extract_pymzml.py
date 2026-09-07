"""
Batch imzML extraction to .npy using PymzmlToNumpy (extract_pymzml.py backend).

Output layout (per slide):
  {output_root}/[{relative_subdirs}/]{imzml_stem}/{bins_subdir}/{region_filename}
  optional: .../{viz_subdir}/{stem}__TOP3.png, panel.png, etc.

Example (CarstenHopf Mannheim):
  positive: .../MSimaging/positive/ID10_.../5_bins/0.npy
  negative: .../MSimaging/negative/ID10_.../5_bins/0.npy
  Run negative: python scripts/batch_extract_pymzml.py --config-name=batch_extract_pymzml_negative
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from msi_visual.extract.pymzml_to_numpy import PymzmlToNumpy  # noqa: E402

logger = logging.getLogger(__name__)
VIZ_DEFAULTS_YAML = SCRIPTS / "configs" / "generate_visualization_panel.yaml"


def _resolve_path(raw: str) -> Path:
    return Path(to_absolute_path(str(raw))).expanduser().resolve()


def _find_imzml_files(input_root: Path, *, recursive: bool, include_glob: str | None) -> list[Path]:
    pattern = "**/*.imzML" if recursive else "*.imzML"
    files = sorted(input_root.glob(pattern))
    files = [p for p in files if p.is_file()]
    if include_glob:
        glob_pat = str(include_glob).strip()
        files = [p for p in files if p.match(glob_pat)]
    return files


def _output_dir_for_imzml(
    input_root: Path,
    output_root: Path,
    imzml_path: Path,
    *,
    bins_subdir: str,
    preserve_relative_dirs: bool,
) -> Path:
    stem = imzml_path.stem
    if preserve_relative_dirs:
        rel_parent = imzml_path.parent.relative_to(input_root)
        if str(rel_parent) in (".", ""):
            return output_root / stem / bins_subdir
        return output_root / rel_parent / stem / bins_subdir
    return output_root / stem / bins_subdir


def _optional_int(val) -> int | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "null", "none", "~"):
        return None
    return int(val)


def _build_viz_cfg(cfg: DictConfig) -> DictConfig | None:
    viz = getattr(cfg, "visualizations", None)
    if viz is None or not bool(viz.get("enabled", False)):
        return None
    base = OmegaConf.load(VIZ_DEFAULTS_YAML) if VIZ_DEFAULTS_YAML.is_file() else OmegaConf.create({})
    merged = OmegaConf.merge(base, viz)
    if OmegaConf.select(merged, "mics_benchmark_config"):
        merged.mics_benchmark_config = to_absolute_path(str(merged.mics_benchmark_config))
    rank_sel = OmegaConf.select(merged, "soft_edges.rank_config")
    if rank_sel:
        merged.soft_edges.rank_config = to_absolute_path(str(rank_sel))
    # ``mics_model: {}`` in batch yaml does not clear defaults from generate_visualization_panel.yaml
    # (OmegaConf merge keeps base keys). When running multiple mics_groups, per-group overrides must win.
    raw_groups = OmegaConf.select(merged, "mics_groups")
    if raw_groups:
        groups = OmegaConf.to_container(raw_groups, resolve=True)
        if isinstance(groups, list) and len(groups) > 0:
            merged.mics_model = OmegaConf.create({})
            merged.mics_parameter_group = None
    return merged


def _run_visualizations_for_npy(
    npy_path: Path,
    stem: str,
    viz_dir: Path,
    viz_cfg: DictConfig,
) -> str:
    # Windows: faiss (benchmark → metrics → zadu) must not load before torch (c10.dll).
    import torch  # noqa: F401

    from generate_visualization_panel import (
        resolve_method_specs,
        run_visualization_specs,
        save_visualization_outputs,
        visualizations_complete,
    )
    from msi_viz_io import load_msi, seed_everything

    if bool(viz_cfg.get("skip_existing", True)) and visualizations_complete(viz_dir, stem, viz_cfg):
        logger.info("Visualizations exist, skipping: %s", viz_dir)
        return "viz_skipped"

    seed = int(OmegaConf.select(viz_cfg, "seed", default=42))
    seed_everything(seed)
    msi = load_msi(
        npy_path,
        transpose_msi=bool(OmegaConf.select(viz_cfg, "data.transpose_msi", default=False)),
        tic_normalize=bool(OmegaConf.select(viz_cfg, "data.tic_normalize", default=True)),
    )
    specs = resolve_method_specs(viz_cfg)
    pairs = run_visualization_specs(msi, viz_cfg, specs, seed)
    save_visualization_outputs(pairs, viz_dir, stem, viz_cfg)
    return "viz_ok"


def _run_visualizations_subprocess(
    npy_path: Path,
    stem: str,
    viz_dir: Path,
    viz_cfg_path: Path,
    viz_cfg: DictConfig,
) -> str:
    """Run viz in a fresh Python process (isolates native DLL state from imzML extract)."""
    py = OmegaConf.select(viz_cfg, "python")
    executable = str(py).strip() if py is not None and str(py).strip() else sys.executable
    cmd = [
        executable,
        str(Path(__file__).resolve()),
        "--viz-worker",
        str(viz_cfg_path),
        str(npy_path.resolve()),
        stem,
        str(viz_dir.resolve()),
    ]
    logger.info("Spawning viz worker (%s): %s", executable, npy_path.name)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.stdout:
        for line in proc.stdout.rstrip().splitlines():
            logger.info("[viz-worker] %s", line)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(err or f"viz worker exited with code {proc.returncode}")
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "viz_ok"
    return last if last in ("viz_ok", "viz_skipped") else "viz_ok"


def _viz_use_subprocess(viz_cfg: DictConfig) -> bool:
    raw = OmegaConf.select(viz_cfg, "subprocess")
    if raw is None:
        return os.name == "nt"
    return bool(raw)


def _dispatch_visualizations(
    npy_path: Path,
    stem: str,
    viz_dir: Path,
    viz_cfg: DictConfig,
    viz_cfg_path: Path,
) -> str:
    if _viz_use_subprocess(viz_cfg):
        return _run_visualizations_subprocess(npy_path, stem, viz_dir, viz_cfg_path, viz_cfg)
    return _run_visualizations_for_npy(npy_path, stem, viz_dir, viz_cfg)


def _viz_worker_cli(argv: list[str]) -> None:
    if len(argv) != 4:
        raise SystemExit("usage: --viz-worker <viz_cfg.yaml> <npy_path> <stem> <viz_dir>")
    cfg_path, npy_path, stem, viz_dir = Path(argv[0]), Path(argv[1]), argv[2], Path(argv[3])
    if not cfg_path.is_file():
        raise FileNotFoundError(f"viz config not found: {cfg_path}")
    if not npy_path.is_file():
        raise FileNotFoundError(f"npy not found: {npy_path}")
    viz_cfg = OmegaConf.load(cfg_path)
    status = _run_visualizations_for_npy(npy_path, stem, viz_dir, viz_cfg)
    print(status, flush=True)


@hydra.main(version_base=None, config_path="configs", config_name="batch_extract_pymzml")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ex = cfg.extract
    viz_cfg = _build_viz_cfg(cfg)
    viz_subdir = str(OmegaConf.select(viz_cfg, "subdir", default="viz")) if viz_cfg else "viz"

    input_root = _resolve_path(ex.input_root)
    output_root = _resolve_path(ex.output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"extract.input_root not found: {input_root}")

    bins = int(ex.bins)
    bins_subdir = str(ex.bins_subdir)
    region_filename = str(ex.region_filename)
    extraction_id = str(ex.id)
    start_mz = _optional_int(ex.start_mz)
    end_mz = _optional_int(ex.end_mz)
    nonzero = bool(ex.nonzero)
    skip_existing = bool(ex.skip_existing)
    dry_run = bool(ex.dry_run)
    include_glob = ex.include_glob
    include_glob = None if include_glob is None or str(include_glob).strip().lower() in ("", "null", "none") else str(include_glob)

    imzml_files = _find_imzml_files(
        input_root,
        recursive=bool(ex.recursive),
        include_glob=include_glob,
    )
    if not imzml_files:
        raise FileNotFoundError(f"No .imzML files under {input_root}")

    logger.info("Found %d imzML file(s) under %s", len(imzml_files), input_root)
    logger.info("Output root: %s | bins=%d | subdir=%s", output_root, bins, bins_subdir)
    if viz_cfg is not None:
        logger.info("Post-extract visualizations enabled → %s/", viz_subdir)

    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        run_dir = Path.cwd()
    run_dir.mkdir(parents=True, exist_ok=True)

    viz_cfg_path: Path | None = None
    if viz_cfg is not None:
        viz_cfg_path = run_dir / "viz_config_resolved.yaml"
        with viz_cfg_path.open("w", encoding="utf-8") as fh:
            fh.write(OmegaConf.to_yaml(viz_cfg, resolve=True))
        if _viz_use_subprocess(viz_cfg):
            logger.info("Visualizations run in subprocess per slide (visualizations.subprocess=true)")

    manifest_rows: list[dict[str, str]] = []
    n_ok = n_skip = n_fail = 0
    n_viz_ok = n_viz_fail = 0

    for idx, imzml_path in enumerate(imzml_files, start=1):
        out_dir = _output_dir_for_imzml(
            input_root,
            output_root,
            imzml_path,
            bins_subdir=bins_subdir,
            preserve_relative_dirs=bool(ex.recursive),
        )
        out_npy = out_dir / region_filename
        viz_dir = out_dir / viz_subdir
        stem = imzml_path.stem
        row = {
            "index": str(idx),
            "imzml": str(imzml_path),
            "output_dir": str(out_dir),
            "output_npy": str(out_npy),
            "viz_dir": str(viz_dir) if viz_cfg else "",
            "status": "",
            "viz_status": "",
            "message": "",
        }

        need_extract = not (skip_existing and out_npy.is_file())

        if dry_run:
            row["status"] = "dry_run"
            if viz_cfg is not None:
                row["viz_status"] = "dry_run"
            manifest_rows.append(row)
            logger.info("[%d/%d] dry-run: %s -> %s", idx, len(imzml_files), imzml_path.name, out_dir)
            continue

        if need_extract:
            logger.info("[%d/%d] extracting: %s -> %s", idx, len(imzml_files), imzml_path.name, out_dir)
            try:
                backend = str(OmegaConf.select(ex, "backend", default="legacy")).strip().lower()
                if backend in ("fast", "imzml_fast"):
                    if nonzero:
                        raise ValueError("extract.backend=fast does not support nonzero=true yet")
                    from msi_visual.extract.imzml_fast import ImzmlFastExtractor

                    extractor = ImzmlFastExtractor(
                        id=extraction_id,
                        min_mz=start_mz,
                        max_mz=end_mz,
                        bins_per_mz=bins,
                        num_workers=int(OmegaConf.select(ex, "num_workers", default=8)),
                        max_rows=OmegaConf.select(ex, "max_rows", default=None),
                    )
                    extractor(str(imzml_path), str(out_dir))
                else:
                    extractor = PymzmlToNumpy(
                        id=extraction_id,
                        min_mz=start_mz,
                        max_mz=end_mz,
                        bins_per_mz=bins,
                        nonzero=nonzero,
                    )
                    extractor(str(imzml_path), str(out_dir))
                if not out_npy.is_file():
                    raise FileNotFoundError(f"Expected output not written: {out_npy}")
                row["status"] = "ok"
                n_ok += 1
            except Exception as exc:
                row["status"] = "failed"
                row["message"] = str(exc)
                n_fail += 1
                logger.exception("Extract failed: %s", imzml_path)
                manifest_rows.append(row)
                continue
        else:
            row["status"] = "skipped"
            row["message"] = "npy exists"
            n_skip += 1
            logger.info("[%d/%d] skip extract (exists): %s", idx, len(imzml_files), out_npy)

        if viz_cfg is not None and out_npy.is_file():
            try:
                logger.info("[%d/%d] visualizations: %s", idx, len(imzml_files), viz_dir)
                assert viz_cfg_path is not None
                viz_status = _dispatch_visualizations(
                    out_npy, stem, viz_dir, viz_cfg, viz_cfg_path
                )
                row["viz_status"] = viz_status
                if viz_status in ("viz_ok", "viz_skipped"):
                    n_viz_ok += 1
            except Exception as exc:
                row["viz_status"] = "viz_failed"
                row["message"] = f"{row['message']}; viz: {exc}".strip("; ")
                n_viz_fail += 1
                logger.exception("Visualization failed: %s", imzml_path)

        manifest_rows.append(row)

    if bool(ex.write_manifest):
        manifest_path = run_dir / "batch_extract_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "index",
                    "imzml",
                    "output_dir",
                    "output_npy",
                    "viz_dir",
                    "status",
                    "viz_status",
                    "message",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        logger.info("Manifest: %s", manifest_path)

    resolved_cfg = run_dir / "config_resolved.yaml"
    with resolved_cfg.open("w", encoding="utf-8") as fh:
        fh.write(OmegaConf.to_yaml(cfg, resolve=True))

    logger.info(
        "Done: extract ok=%d skipped=%d failed=%d | viz ok=%d failed=%d | total=%d",
        n_ok,
        n_skip,
        n_fail,
        n_viz_ok,
        n_viz_fail,
        len(imzml_files),
    )
    if n_fail or n_viz_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--viz-worker":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
        _viz_worker_cli(sys.argv[2:])
    else:
        main()
