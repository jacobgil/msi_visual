#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from msi_visual.parametric_dino_nmf import DINONMFVirtualStain


def _load_msi(path: Path, transpose_msi: bool, tic_normalize: bool) -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected 3D MSI array, got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32, copy=False)
    if tic_normalize:
        sums = img.sum(axis=-1, keepdims=True) + 1e-8
        img = img / sums
    return img


def _to_uint8_rgb(viz: np.ndarray) -> np.ndarray:
    arr = np.asarray(viz, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if np.nanmax(arr) <= 1.0:
        arr = arr * 255.0
    return np.uint8(np.clip(arr, 0, 255))


def _histogram_equalize_rgb(viz_rgb: np.ndarray) -> np.ndarray:
    if viz_rgb.ndim != 3 or viz_rgb.shape[-1] < 3:
        return viz_rgb
    out = viz_rgb.copy()
    for ch in range(3):
        out[:, :, ch] = cv2.equalizeHist(out[:, :, ch])
    return out


def _as_list(x):
    if isinstance(x, (list, tuple, ListConfig)):
        return list(x)
    return [x]


@hydra.main(version_base=None, config_path="configs", config_name="sweep_parametric_dino_nmf")
def main(cfg: DictConfig) -> None:
    input_npy = Path(str(cfg.data.input_npy))
    if not input_npy.exists():
        raise FileNotFoundError(f"input_npy not found: {input_npy}")
    out_dir = Path(str(cfg.output.out_dir)) if cfg.output.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    img = _load_msi(
        input_npy,
        transpose_msi=bool(cfg.data.transpose_msi),
        tic_normalize=bool(cfg.data.tic_normalize),
    )
    (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    sweep_space = list(
        itertools.product(
            _as_list(cfg.sweep.pixel_sampling),
            _as_list(cfg.sweep.num_samples),
            _as_list(cfg.sweep.train_epochs),
            _as_list(cfg.sweep.learning_rate),
            _as_list(cfg.sweep.token_dropout),
            _as_list(cfg.sweep.pca_fit_step),
        )
    )
    rows = []
    paths_txt = out_dir / "image_paths.txt"
    for i, (sampling, num_samples, epochs, lr, token_dropout, pca_fit_step) in enumerate(sweep_space, start=1):
        run_cfg = {
            "pixel_sampling": str(sampling),
            "num_samples": int(num_samples),
            "train_epochs": int(epochs),
            "lr": float(lr),
            "token_dropout": float(token_dropout),
            "pca_fit_step": int(pca_fit_step),
            "random_state": int(cfg.model.random_state),
            "nmf_k": int(cfg.model.nmf_k),
        }
        tag = (
            f"s{i:03d}"
            f"__{run_cfg['pixel_sampling']}"
            f"__n{run_cfg['num_samples']}"
            f"__e{run_cfg['train_epochs']}"
            f"__lr{run_cfg['lr']}"
            f"__td{run_cfg['token_dropout']}"
            f"__pca{run_cfg['pca_fit_step']}"
        )
        print(f"[{i}/{len(sweep_space)}] {tag}")
        t0 = time.perf_counter()
        model = DINONMFVirtualStain(
            nmf_k=cfg.model.nmf_k,
            pixel_sampling=run_cfg["pixel_sampling"],
            num_samples=run_cfg["num_samples"],
            pca_fit_step=run_cfg["pca_fit_step"],
            random_state=run_cfg["random_state"],
            train_epochs=run_cfg["train_epochs"],
            lr=run_cfg["lr"],
            token_dropout=run_cfg["token_dropout"],
            batch_size=int(cfg.model.batch_size),
            d_model=int(cfg.model.d_model),
            n_layers=int(cfg.model.n_layers),
            dino_dim=int(cfg.model.dino_dim),
            ema_momentum=float(cfg.model.ema_momentum),
            student_temp=float(cfg.model.student_temp),
            teacher_temp=float(cfg.model.teacher_temp),
            max_nmf_fit_samples=int(cfg.model.max_nmf_fit_samples),
            koleo_weight=float(cfg.model.koleo_weight),
            view_noise_std=float(OmegaConf.select(cfg, "model.view_noise_std", default=0.02)),
            spectral_dropout_p=float(OmegaConf.select(cfg, "model.spectral_dropout_p", default=0.0)),
            teacher_center_logits=bool(OmegaConf.select(cfg, "model.teacher_center_logits", default=True)),
            pre_mlp_pca_dim=int(OmegaConf.select(cfg, "model.pre_mlp_pca_dim", default=0)),
            cls_decorrelate_weight=float(OmegaConf.select(cfg, "model.cls_decorrelate_weight", default=0.0)),
        )
        model.fit([img], keras_fit_kwargs={}, roi_mask=None)
        viz = model.predict(img)
        elapsed = time.perf_counter() - t0

        png_path = out_dir / f"{tag}.png"
        viz_rgb = _to_uint8_rgb(viz)
        if bool(getattr(cfg.output, "equalize_visualizations", True)):
            viz_rgb = _histogram_equalize_rgb(viz_rgb)
        Image.fromarray(viz_rgb, mode="RGB").save(png_path)
        png_abs = png_path.resolve()
        print(f"  saved PNG: {png_abs}")
        meta_path = out_dir / f"{tag}.json"
        meta = {
            "sweep_cfg": OmegaConf.to_container(cfg.sweep, resolve=True),
            "model_cfg": OmegaConf.to_container(cfg.model, resolve=True),
            "run_cfg": run_cfg,
        }
        meta["input_npy"] = str(input_npy.resolve())
        meta["elapsed_seconds"] = float(elapsed)
        meta["png_path"] = str(png_abs)
        meta["json_path"] = str(meta_path.resolve())
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append(meta)
        paths_txt.write_text("\n".join(r["png_path"] for r in rows) + "\n", encoding="utf-8")

    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} sweep runs to {out_dir}")


if __name__ == "__main__":
    main()
