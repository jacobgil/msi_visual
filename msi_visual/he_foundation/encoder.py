"""Load pathology foundation models and encode H&E tile batches."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

EncoderBundle = tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor], int, str]

_DINOV2_HUB_ENTRY = "dinov2_vitg14_reg"


def _device_or_cpu(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tile_to_model_input(transform: Callable[[Image.Image], torch.Tensor], rgb: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    return transform(pil)


def _hf_auth_help(repo_id: str) -> str:
    return (
        f"Cannot download gated model {repo_id}.\n"
        "  1. Open https://huggingface.co/{repo} and click 'Agree and access repository'.\n"
        "  2. Log in: huggingface-cli login   (or set HF_TOKEN in the environment).\n"
        "  3. Re-run the script.\n"
        "OpenMidnight may require an institutional email on your HF account.\n"
        "Alternatively use foundation.model=uni2-h or foundation.local_checkpoint=/path/to/teacher_checkpoint_load.pt"
    ).format(repo=repo_id)


def _resolve_openmidnight_checkpoint(local_checkpoint: str | None) -> str:
    if local_checkpoint:
        path = Path(local_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"foundation.local_checkpoint not found: {path}")
        return str(path)

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError as exc:
        raise ImportError("openmidnight requires huggingface_hub: pip install huggingface_hub") from exc

    repo_id = "SophontAI/OpenMidnight"
    try:
        return hf_hub_download(repo_id=repo_id, filename="teacher_checkpoint_load.pt")
    except GatedRepoError as exc:
        raise PermissionError(_hf_auth_help(repo_id)) from exc


def _dinov2_repo_help() -> str:
    return (
        "OpenMidnight needs the DINOv2 Python repo (ViT-g/14+reg architecture).\n"
        "torch.hub normally downloads it from GitHub, which failed here (often SSL on Windows).\n"
        "One-time setup — clone locally, then point the script at it:\n"
        "  git clone https://github.com/facebookresearch/dinov2.git C:/models/dinov2\n"
        "  python scripts/batch_he_embed_mics_viz.py foundation.dinov2_repo=C:/models/dinov2 ...\n"
        "Or set env DINOV2_REPO=C:/models/dinov2\n"
        "Alternative: foundation.model=uni2-h (no DINOv2 repo required)."
    )


def _resolve_dinov2_repo(explicit: str | None) -> Path | None:
    """Return a local dinov2 checkout (hubconf.py present), or None."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("DINOV2_REPO", "").strip()
    if env:
        candidates.append(Path(env))
    hub_dir = Path(torch.hub.get_dir())
    candidates.extend(
        [
            hub_dir / "facebookresearch_dinov2_main",
            hub_dir / "dinov2_main",
        ]
    )
    for raw in candidates:
        path = raw.expanduser().resolve()
        if (path / "hubconf.py").is_file():
            logger.info("Using local DINOv2 repo: %s", path)
            return path
    return None


def _load_dinov2_vitg14_reg(*, dinov2_repo: str | None) -> torch.nn.Module:
    """Build ViT-g/14+reg architecture only; OpenMidnight weights loaded separately."""
    hub_kw = dict(pretrained=False, trust_repo=True)
    local = _resolve_dinov2_repo(dinov2_repo)
    if local is not None:
        return torch.hub.load(str(local), _DINOV2_HUB_ENTRY, source="local", **hub_kw)

    try:
        return torch.hub.load("facebookresearch/dinov2", _DINOV2_HUB_ENTRY, **hub_kw)
    except OSError as exc:
        if "ssl" in str(exc).lower() or "ASN1" in str(exc):
            raise RuntimeError(_dinov2_repo_help()) from exc
        raise
    except Exception as exc:
        err = str(exc).lower()
        if "ssl" in err or "certificate" in err or "github" in err:
            raise RuntimeError(_dinov2_repo_help()) from exc
        raise


def load_he_encoder(
    name: str,
    *,
    device: str | None = None,
    local_checkpoint: str | None = None,
    dinov2_repo: str | None = None,
) -> EncoderBundle:
    """
    Load a foundation-model encoder.

    Supported ``name`` values:
      - ``uni2-h`` / ``uni-2`` — MahmoodLab UNI2-h via timm (224×224 input, 1536-D)
      - ``openmidnight`` / ``open-midnight`` — SophontAI OpenMidnight ViT-G/14 (224×224, 1536-D)
    """
    key = str(name).strip().lower().replace("_", "-")
    dev = _device_or_cpu(device)
    if key in ("uni2-h", "uni-2", "uni2h"):
        return _load_uni2_h(dev)
    if key in ("openmidnight", "open-midnight", "midnight"):
        return _load_openmidnight(dev, local_checkpoint=local_checkpoint, dinov2_repo=dinov2_repo)
    raise ValueError(f"Unknown encoder {name!r}. Use uni2-h or openmidnight.")


def _load_uni2_h(device: torch.device) -> EncoderBundle:
    try:
        import timm
        from timm.data import create_transform, resolve_data_config
        from huggingface_hub.errors import GatedRepoError
    except ImportError as exc:
        raise ImportError("uni2-h requires timm: pip install timm") from exc

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    repo_id = "MahmoodLab/UNI2-h"
    try:
        model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True, **timm_kwargs)
    except GatedRepoError as exc:
        raise PermissionError(_hf_auth_help(repo_id)) from exc
    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**cfg)
    model.eval().to(device)
    logger.info("Loaded UNI2-h on %s", device)
    return model, transform, 1536, "uni2-h"


def _load_openmidnight(
    device: torch.device,
    *,
    local_checkpoint: str | None = None,
    dinov2_repo: str | None = None,
) -> EncoderBundle:
    try:
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("openmidnight requires torchvision") from exc

    ckpt_path = _resolve_openmidnight_checkpoint(local_checkpoint)
    model = _load_dinov2_vitg14_reg(dinov2_repo=dinov2_repo)
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.pos_embed = torch.nn.Parameter(checkpoint["pos_embed"])
    model.load_state_dict(checkpoint)
    model.eval().to(device)
    transform = T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    logger.info("Loaded OpenMidnight on %s", device)
    return model, transform, 1536, "openmidnight"


def encode_tiles(
    model: torch.nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    tiles_rgb: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 16,
    model_input_px: int = 224,
    tile_px: int = 256,
) -> np.ndarray:
    """Encode N tiles to (N, D) float32 embeddings."""
    if not tiles_rgb:
        return np.zeros((0, 0), dtype=np.float32)

    feats: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tiles_rgb), batch_size):
            batch = tiles_rgb[start : start + batch_size]
            tensors: list[torch.Tensor] = []
            for rgb in batch:
                arr = np.asarray(rgb, dtype=np.uint8)
                if tile_px != model_input_px:
                    arr = np.asarray(
                        Image.fromarray(arr).resize((model_input_px, model_input_px), Image.Resampling.BICUBIC),
                        dtype=np.uint8,
                    )
                tensors.append(_tile_to_model_input(transform, arr))
            x = torch.stack(tensors, dim=0).to(device)
            y = model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            feats.append(y.detach().float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)
