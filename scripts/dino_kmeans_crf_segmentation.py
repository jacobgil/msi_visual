#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.cluster import KMeans
from skimage.segmentation import felzenszwalb

logger = logging.getLogger(__name__)


def _load_msi(path: Path, transpose_msi: bool) -> np.ndarray:
    img = np.load(path, mmap_mode=None)
    if img.ndim != 3:
        raise ValueError(f"Expected MSI shape HxWxC (or CxHxW with transpose), got {img.shape}")
    if transpose_msi:
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.float32, copy=False)


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] > 3:
        x = x[:, :, :3]
    if x.dtype == np.uint8:
        return x
    x = x.astype(np.float32)
    if np.nanmax(x) <= 1.0:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def _load_visualization(path: Path, target_hw: Tuple[int, int], allow_resize: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        viz = np.load(path, mmap_mode=None)
        rgb = _to_uint8_rgb(viz)
    else:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = target_hw
    if rgb.shape[:2] != (h, w):
        if not allow_resize:
            raise ValueError(f"Visualization shape mismatch for {path}: got {rgb.shape[:2]}, expected {(h, w)}")
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb


def _load_segformer_model(model_name: str, device: torch.device) -> torch.nn.Module:
    try:
        from transformers import SegformerModel
    except Exception as exc:
        raise SystemExit(
            "transformers is required for SegFormer features. Install with: pip install transformers"
        ) from exc
    try:
        model = SegformerModel.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SegFormer model '{model_name}'. "
            "Check internet access and model id (e.g. nvidia/mit-b0, nvidia/mit-b1)."
        ) from exc
    model = model.to(device)
    model.eval()
    return model


def _image_to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    t = torch.from_numpy(x).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (t - mean) / std


def _extract_patch_embeddings(
    model: torch.nn.Module,
    rgb: np.ndarray,
    device: torch.device,
    pre_upscale: float,
    pre_upscale_interpolation: str,
    align_to: int,
    feature_stage: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    h, w = rgb.shape[:2]
    scale = max(1.0, float(pre_upscale))
    h_u = int(round(h * scale))
    w_u = int(round(w * scale))
    if str(pre_upscale_interpolation).strip().lower() == "nearest":
        interp = cv2.INTER_NEAREST
    elif str(pre_upscale_interpolation).strip().lower() == "cubic":
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR
    rgb_u = cv2.resize(rgb, (w_u, h_u), interpolation=interp) if (h_u != h or w_u != w) else rgb

    align = max(1, int(align_to))
    # Align to model stride without shrinking to preserve effective resolution.
    h_m = max(align, int(np.ceil(h_u / align) * align))
    w_m = max(align, int(np.ceil(w_u / align) * align))
    rgb_m = cv2.resize(rgb_u, (w_m, h_m), interpolation=interp) if (h_m != h_u or w_m != w_u) else rgb_u
    x = _image_to_tensor(rgb_m, device)

    with torch.no_grad():
        out = model(pixel_values=x, output_hidden_states=True, return_dict=True)
    hidden_states = out.hidden_states
    if hidden_states is None or len(hidden_states) == 0:
        raise RuntimeError("SegFormer did not return hidden states.")

    stage_idx = int(feature_stage)
    if stage_idx < 0:
        stage_idx = len(hidden_states) + stage_idx
    stage_idx = max(0, min(stage_idx, len(hidden_states) - 1))
    feat = hidden_states[stage_idx]
    if feat.ndim != 4:
        raise RuntimeError(f"Unexpected SegFormer feature shape: {tuple(feat.shape)}")
    feat = feat[0].detach().cpu().numpy().astype(np.float32, copy=False)  # C,H,W
    _, gh, gw = feat.shape
    embeddings = np.transpose(feat, (1, 2, 0)).reshape(gh * gw, -1).astype(np.float32, copy=False)
    return embeddings, (gh, gw), (h_m, w_m)


def _run_felzenszwalb_segments(
    rgb: np.ndarray,
    pre_upscale: float,
    pre_upscale_interpolation: str,
    scale: float,
    sigma: float,
    min_size: int,
    out_hw: tuple[int, int],
) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale_up = max(1.0, float(pre_upscale))
    h_u = int(round(h * scale_up))
    w_u = int(round(w * scale_up))
    interp_name = str(pre_upscale_interpolation).strip().lower()
    if interp_name == "nearest":
        interp = cv2.INTER_NEAREST
    elif interp_name == "cubic":
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR
    rgb_u = cv2.resize(rgb, (w_u, h_u), interpolation=interp) if (h_u != h or w_u != w) else rgb
    seg_u = felzenszwalb(
        rgb_u.astype(np.float32) / 255.0,
        scale=float(scale),
        sigma=float(sigma),
        min_size=int(min_size),
    ).astype(np.int32, copy=False)
    oh, ow = out_hw
    seg = cv2.resize(seg_u, (ow, oh), interpolation=cv2.INTER_NEAREST).astype(np.int32, copy=False)
    return seg


def _segments_to_category_probs(
    rgb: np.ndarray,
    segments: np.ndarray,
    n_clusters: int,
    random_state: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = segments.shape
    seg_ids = np.unique(segments)
    n_seg = int(seg_ids.size)
    seg_index = {int(s): i for i, s in enumerate(seg_ids.tolist())}

    rgb_f = rgb.astype(np.float32)
    means = np.zeros((n_seg, 3), dtype=np.float32)
    counts = np.zeros((n_seg,), dtype=np.float32)
    flat_seg = segments.ravel()
    flat_rgb = rgb_f.reshape(-1, 3)
    for idx in range(flat_seg.size):
        sid = int(flat_seg[idx])
        i = seg_index[sid]
        means[i] += flat_rgb[idx]
        counts[i] += 1.0
    counts = np.maximum(counts, 1.0)[:, None]
    means = means / counts

    k = max(2, min(int(n_clusters), n_seg))
    km = KMeans(n_clusters=k, random_state=int(random_state), n_init="auto")
    km.fit(means)
    dists = km.transform(means).astype(np.float32, copy=False)
    probs_seg = _softmax(-dists / max(float(temperature), 1e-6), axis=1).astype(np.float32, copy=False)

    probs_chw = np.zeros((k, h, w), dtype=np.float32)
    for sid in seg_ids.tolist():
        i = seg_index[int(sid)]
        mask = segments == int(sid)
        for c in range(k):
            probs_chw[c, mask] = probs_seg[i, c]
    labels = np.argmax(probs_chw, axis=0).astype(np.int32, copy=False)
    return labels, probs_chw


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(np.sum(ex, axis=axis, keepdims=True), 1e-8)


def _kmeans_patch_probs(
    embeddings: np.ndarray,
    grid_hw: tuple[int, int],
    n_clusters: int,
    random_state: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(embeddings)
    labels = km.labels_.astype(np.int32, copy=False)
    dists = km.transform(embeddings).astype(np.float32, copy=False)
    probs = _softmax(-dists / max(float(temperature), 1e-6), axis=1).astype(np.float32, copy=False)
    gh, gw = grid_hw
    label_grid = labels.reshape(gh, gw)
    prob_grid = probs.reshape(gh, gw, n_clusters)
    return label_grid, prob_grid


def _upsample_probs(prob_grid: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    gh, gw, k = prob_grid.shape
    h, w = out_hw
    out = np.zeros((k, h, w), dtype=np.float32)
    for c in range(k):
        out[c] = cv2.resize(prob_grid[:, :, c], (w, h), interpolation=cv2.INTER_LINEAR)
    denom = np.maximum(np.sum(out, axis=0, keepdims=True), 1e-8)
    out = out / denom
    return out.astype(np.float32, copy=False)


def _run_dense_crf(
    rgb: np.ndarray,
    probs_chw: np.ndarray,
    iterations: int,
    sxy_gaussian: int,
    compat_gaussian: int,
    sxy_bilateral: int,
    srgb_bilateral: int,
    compat_bilateral: int,
) -> np.ndarray:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    k, h, w = probs_chw.shape
    unary = unary_from_softmax(np.clip(probs_chw, 1e-8, 1.0))
    unary = np.ascontiguousarray(unary)
    rgb_u8 = np.ascontiguousarray(rgb.astype(np.uint8))

    crf = dcrf.DenseCRF2D(w, h, k)
    crf.setUnaryEnergy(unary)
    crf.addPairwiseGaussian(sxy=int(sxy_gaussian), compat=int(compat_gaussian))
    crf.addPairwiseBilateral(
        sxy=int(sxy_bilateral),
        srgb=int(srgb_bilateral),
        rgbim=rgb_u8,
        compat=int(compat_bilateral),
    )

    q = np.array(crf.inference(int(iterations)), dtype=np.float32).reshape((k, h, w))
    return np.argmax(q, axis=0).astype(np.int32, copy=False)


def _run_fallback_refinement(
    probs_chw: np.ndarray,
    iterations: int,
    sigma_space: float,
    sigma_color: float,
) -> np.ndarray:
    """Dependency-free fallback when DenseCRF is unavailable (Windows-friendly)."""
    k, h, w = probs_chw.shape
    probs = np.asarray(probs_chw, dtype=np.float32).copy()
    iters = max(1, int(iterations))
    sig_s = max(0.1, float(sigma_space))
    sig_c = max(0.1, float(sigma_color))
    for _ in range(iters):
        for c in range(k):
            ch = probs[c]
            ch = cv2.GaussianBlur(ch, (0, 0), sigmaX=sig_s, sigmaY=sig_s)
            ch = cv2.bilateralFilter(ch, d=0, sigmaColor=sig_c, sigmaSpace=sig_s)
            probs[c] = ch
        denom = np.maximum(np.sum(probs, axis=0, keepdims=True), 1e-8)
        probs = probs / denom
    return np.argmax(probs, axis=0).astype(np.int32, copy=False)


def _make_palette(n_clusters: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 255, size=(n_clusters, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def _colorize_labels(labels: np.ndarray, palette: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = palette[np.clip(labels, 0, len(palette) - 1)]
    out[~valid_mask] = 0
    return out


@hydra.main(version_base=None, config_path="configs", config_name="dino_kmeans_crf_segmentation")
def main(cfg: DictConfig) -> None:
    log_level = str(getattr(getattr(cfg, "logging", None), "level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    npy_path = Path(str(cfg.input.npy_path))
    viz_paths = [Path(str(p)) for p in cfg.input.visualization_paths]
    if not npy_path.exists():
        raise FileNotFoundError(f"MSI file not found: {npy_path}")
    for p in viz_paths:
        if not p.exists():
            raise FileNotFoundError(f"Visualization file not found: {p}")

    out_dir = Path(str(cfg.output.out_dir)) if cfg.output.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    msi = _load_msi(npy_path, transpose_msi=bool(cfg.data.transpose_msi))
    valid_mask = msi.sum(axis=-1) > 0
    if not np.any(valid_mask):
        raise ValueError("MSI valid mask is empty.")
    logger.info("Loaded MSI | shape=%s | valid_pixels=%d", tuple(msi.shape), int(valid_mask.sum()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = str(getattr(cfg.model, "type", "segformer")).strip().lower()
    pre_upscale = float(getattr(cfg.model, "pre_upscale", 1.0))
    pre_upscale_interpolation = str(getattr(cfg.model, "pre_upscale_interpolation", "linear"))
    align_to = int(getattr(cfg.model, "align_to", 32))
    feature_stage = int(getattr(cfg.model, "feature_stage", 1))

    model = None
    model_name = str(getattr(cfg.model, "name", "nvidia/mit-b0"))
    if model_type == "segformer":
        model = _load_segformer_model(model_name, device=device)
        logger.info("Loaded SegFormer model=%s on %s", model_name, str(device))
    elif model_type == "felzenszwalb":
        logger.info("Using Felzenszwalb segmentation backbone.")
    else:
        raise ValueError(f"Unsupported model.type: {model_type}. Use 'segformer' or 'felzenszwalb'.")

    n_clusters = int(cfg.kmeans.n_clusters)
    palette = _make_palette(n_clusters)
    temp = float(cfg.kmeans.temperature)
    seed = int(cfg.kmeans.random_state)

    for i, viz_path in enumerate(viz_paths):
        rgb = _load_visualization(viz_path, target_hw=(msi.shape[0], msi.shape[1]), allow_resize=bool(cfg.input.allow_resize))
        if model_type == "segformer":
            assert model is not None
            embeds, grid_hw, resized_hw = _extract_patch_embeddings(
                model,
                rgb,
                device=device,
                pre_upscale=pre_upscale,
                pre_upscale_interpolation=pre_upscale_interpolation,
                align_to=align_to,
                feature_stage=feature_stage,
            )
            label_grid, prob_grid = _kmeans_patch_probs(
                embeds,
                grid_hw=grid_hw,
                n_clusters=n_clusters,
                random_state=seed + i,
                temperature=temp,
            )
            prob_up = _upsample_probs(prob_grid, out_hw=(msi.shape[0], msi.shape[1]))
            patch_meta = f"patch_grid={int(grid_hw[0])}x{int(grid_hw[1])} resized={int(resized_hw[0])}x{int(resized_hw[1])} stage={int(feature_stage)}"
        else:
            seg = _run_felzenszwalb_segments(
                rgb=rgb,
                pre_upscale=pre_upscale,
                pre_upscale_interpolation=pre_upscale_interpolation,
                scale=float(getattr(cfg.felzenszwalb, "scale", 100.0)),
                sigma=float(getattr(cfg.felzenszwalb, "sigma", 0.8)),
                min_size=int(getattr(cfg.felzenszwalb, "min_size", 50)),
                out_hw=(msi.shape[0], msi.shape[1]),
            )
            label_grid, prob_up = _segments_to_category_probs(
                rgb=rgb,
                segments=seg,
                n_clusters=n_clusters,
                random_state=seed + i,
                temperature=temp,
            )
            patch_meta = f"segments={int(np.unique(seg).size)}"
        crf_method = str(getattr(cfg.crf, "method", "auto")).strip().lower()
        if crf_method not in {"auto", "densecrf", "fallback", "none"}:
            crf_method = "auto"

        if crf_method == "none":
            labels_crf = np.argmax(prob_up, axis=0).astype(np.int32, copy=False)
            refinement_used = "none"
        elif crf_method in {"auto", "densecrf"}:
            try:
                labels_crf = _run_dense_crf(
                    rgb=rgb,
                    probs_chw=prob_up,
                    iterations=int(cfg.crf.iterations),
                    sxy_gaussian=int(cfg.crf.sxy_gaussian),
                    compat_gaussian=int(cfg.crf.compat_gaussian),
                    sxy_bilateral=int(cfg.crf.sxy_bilateral),
                    srgb_bilateral=int(cfg.crf.srgb_bilateral),
                    compat_bilateral=int(cfg.crf.compat_bilateral),
                )
                refinement_used = "densecrf"
            except Exception:
                if crf_method == "densecrf":
                    raise
                labels_crf = _run_fallback_refinement(
                    probs_chw=prob_up,
                    iterations=int(getattr(cfg.crf, "fallback_iterations", 3)),
                    sigma_space=float(getattr(cfg.crf, "fallback_sigma_space", 2.0)),
                    sigma_color=float(getattr(cfg.crf, "fallback_sigma_color", 0.1)),
                )
                refinement_used = "fallback"
                logger.warning("DenseCRF unavailable; used fallback refinement.")
        else:
            labels_crf = _run_fallback_refinement(
                probs_chw=prob_up,
                iterations=int(getattr(cfg.crf, "fallback_iterations", 3)),
                sigma_space=float(getattr(cfg.crf, "fallback_sigma_space", 2.0)),
                sigma_color=float(getattr(cfg.crf, "fallback_sigma_color", 0.1)),
            )
            refinement_used = "fallback"

        color = _colorize_labels(labels_crf, palette=palette, valid_mask=valid_mask)
        stem = viz_path.stem
        Image.fromarray(color).save(out_dir / f"{stem}__{model_type}_kmeans_crf.png")
        np.save(out_dir / f"{stem}__{model_type}_kmeans_crf_labels.npy", labels_crf.astype(np.int32))
        np.save(out_dir / f"{stem}__{model_type}_patch_labels.npy", label_grid.astype(np.int32))

        logger.info(
            "Saved %s | mode=%s | %s | pre_upscale=%.2f | refinement=%s",
            stem,
            model_type,
            patch_meta,
            float(pre_upscale),
            refinement_used,
        )

    print(f"Saved segmentation outputs to: {out_dir}")


if __name__ == "__main__":
    main()

