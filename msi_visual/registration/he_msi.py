"""Affine H&E → MSI registration (multi-stage coarse-to-fine ECC)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ReferenceMode = Literal["tic", "solace", "valid_mask", "mics", "mics_edges"]
MicsChannel = Literal["l", "edges", "chroma"]
MotionMode = Literal["euclidean", "affine", "homography"]
RegistrationStrategy = Literal["mask_only", "multistage"]


@dataclass
class RegistrationStageResult:
    name: str
    reference: str
    motion: str
    max_side: int | None
    scale: float
    success: bool
    correlation: float | None
    message: str


@dataclass
class RegistrationResult:
    sample_id: str
    slide_key: str
    msi_path: str
    he_path: str
    target_space: str
    msi_shape_hw: list[int]
    he_shape_hw: list[int]
    warp_matrix_2x3: list[list[float]]
    method: str
    reference: str
    success: bool
    correlation: float | None
    message: str
    rotation_deg: int = 0
    init_method: str = "letterbox"
    mics_source: str | None = None
    mics_channel: str = "l"
    stages: list[dict[str, Any]] = field(default_factory=list)

    def matrix(self) -> np.ndarray:
        return np.asarray(self.warp_matrix_2x3, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EccStageConfig:
    name: str
    reference: ReferenceMode = "tic"
    max_side: int | None = 256
    motion: MotionMode = "affine"
    ecc_iterations: int = 3000
    ecc_epsilon: float = 1e-6
    gauss_filt_size: int = 5


def _normalize_uint8(x: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if mask is not None:
        vals = arr[mask]
        if vals.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        lo, hi = float(np.percentile(vals, 2.0)), float(np.percentile(vals, 98.0))
    else:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    span = max(hi - lo, 1e-6)
    out = np.clip((arr - lo) / span, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def letterbox_matrix(he_h: int, he_w: int, msi_h: int, msi_w: int) -> np.ndarray:
    scale = min(msi_w / max(he_w, 1), msi_h / max(he_h, 1))
    tx = (msi_w - he_w * scale) * 0.5
    ty = (msi_h - he_h * scale) * 0.5
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)


def _compose_affine(m_left: np.ndarray, m_right: np.ndarray) -> np.ndarray:
    a = np.vstack([m_right, [0.0, 0.0, 1.0]])
    b = np.vstack([m_left, [0.0, 0.0, 1.0]])
    return (b @ a)[:2, :].astype(np.float32)


def _rotate_he_rgb(he_rgb: np.ndarray, rotation_deg: int) -> np.ndarray:
    rot = int(rotation_deg) % 360
    if rot == 0:
        return np.asarray(he_rgb, dtype=np.uint8)
    if rot == 90:
        return np.rot90(he_rgb, k=1).copy()
    if rot == 180:
        return np.rot90(he_rgb, k=2).copy()
    if rot == 270:
        return np.rot90(he_rgb, k=3).copy()
    raise ValueError(f"rotation_deg must be one of 0, 90, 180, 270; got {rotation_deg}")


def warp_he_to_msi(
    he_rgb: np.ndarray,
    warp_matrix: np.ndarray,
    out_hw: tuple[int, int],
    *,
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    h, w = out_hw
    return cv2.warpAffine(
        np.asarray(he_rgb, dtype=np.uint8),
        np.asarray(warp_matrix, dtype=np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _he_gray_l(he_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(he_rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB)[:, :, 0]


def _he_tissue_mask(he_rgb: np.ndarray, *, min_l: int = 8) -> np.ndarray:
    gray = _he_gray_l(he_rgb)
    return gray > int(min_l)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    return inter / union if union > 0 else 0.0


def _similarity_matrix(
    angle_deg: float,
    scale: float,
    cx: float,
    cy: float,
    tx: float,
    ty: float,
) -> np.ndarray:
    """Uniform scale + rotation about (cx, cy), then translation."""
    a = np.deg2rad(float(angle_deg))
    cos_a, sin_a = float(np.cos(a)), float(np.sin(a))
    s = float(scale)
    tx_p = tx + cx - s * (cos_a * cx - sin_a * cy)
    ty_p = ty + cy - s * (sin_a * cx + cos_a * cy)
    return np.array(
        [[s * cos_a, -s * sin_a, tx_p], [s * sin_a, s * cos_a, ty_p]],
        dtype=np.float32,
    )


def _mask_alignment_iou(
    msi_valid: np.ndarray,
    he_rgb: np.ndarray,
    warp: np.ndarray,
    out_hw: tuple[int, int],
) -> float:
    preview = warp_he_to_msi(he_rgb, warp, out_hw)
    return _mask_iou(msi_valid, _he_tissue_mask(preview))


def _scale_affine_matrix(m: np.ndarray, factor: float) -> np.ndarray:
    """Scale translation components of a 2×3 affine for a downscaled coordinate frame."""
    out = np.asarray(m, dtype=np.float32).copy()
    out[:, 2] *= float(factor)
    return out


def _refine_mask_similarity(
    msi_valid: np.ndarray,
    he_rgb: np.ndarray,
    m_init: np.ndarray,
    *,
    msi_h: int,
    msi_w: int,
    max_side: int = 384,
    angle_deg_range: tuple[float, float] = (-35.0, 35.0),
    angle_deg_step: float = 5.0,
    scale_range: tuple[float, float] = (0.82, 1.18),
    scale_step: float = 0.04,
    shift_frac: float = 0.12,
    shift_step: float = 0.04,
) -> tuple[np.ndarray, float, str]:
    """
    Grid-search similarity deltas (rotation, uniform scale, shift) on top of m_init.

    Optimizes IoU between MSI outer tissue mask and warped H&E tissue mask.
    """
    m_box = _bbox(msi_valid)
    if m_box is None:
        return m_init, _mask_alignment_iou(msi_valid, he_rgb, m_init, (msi_h, msi_w)), "no_msi_tissue"

    my0, mx0, my1, mx1 = m_box
    cx = (mx0 + mx1) * 0.5
    cy = (my0 + my1) * 0.5
    shift_px = max(mx1 - mx0 + 1, my1 - my0 + 1) * float(shift_frac)

    work_h, work_w = msi_h, msi_w
    valid_work = msi_valid
    m_work = np.asarray(m_init, dtype=np.float32)
    img_scale = 1.0
    longest = max(msi_h, msi_w)
    if longest > max_side > 0:
        img_scale = max_side / float(longest)
        work_h = max(1, int(round(msi_h * img_scale)))
        work_w = max(1, int(round(msi_w * img_scale)))
        valid_u8 = (msi_valid.astype(np.uint8) * 255)
        valid_work = cv2.resize(valid_u8, (work_w, work_h), interpolation=cv2.INTER_NEAREST) > 127
        m_work = _scale_affine_matrix(m_init, img_scale)
        cx *= img_scale
        cy *= img_scale
        shift_px *= img_scale

    def score(m: np.ndarray) -> float:
        return _mask_alignment_iou(valid_work, he_rgb, m, (work_h, work_w))

    best_m = m_work
    best_score = score(m_work)

    angles = np.arange(angle_deg_range[0], angle_deg_range[1] + 1e-6, angle_deg_step)
    scales = np.arange(scale_range[0], scale_range[1] + 1e-6, scale_step)
    shifts = np.arange(-shift_px, shift_px + 1e-6, shift_px * shift_step / max(shift_frac, 1e-6))
    if shifts.size == 0:
        shifts = np.array([0.0], dtype=np.float64)

    for angle in angles:
        for scale in scales:
            for dx in shifts:
                for dy in shifts:
                    delta = _similarity_matrix(float(angle), float(scale), cx, cy, float(dx), float(dy))
                    m_try = _compose_affine(delta, m_work)
                    iou = score(m_try)
                    if iou > best_score:
                        best_score = iou
                        best_m = m_try

    fine_angles = np.arange(-4.0, 4.0 + 1e-6, 1.0)
    fine_scales = np.arange(-0.06, 0.061, 0.015)
    fine_shifts = np.arange(-shift_px * 0.06, shift_px * 0.061, shift_px * 0.02)
    if fine_shifts.size == 0:
        fine_shifts = np.array([0.0], dtype=np.float64)

    for angle in fine_angles:
        for ds in fine_scales:
            scale_mult = max(1.0 + float(ds), 1e-3)
            for dx in fine_shifts:
                for dy in fine_shifts:
                    delta = _similarity_matrix(float(angle), scale_mult, cx, cy, float(dx), float(dy))
                    m_try = _compose_affine(delta, best_m)
                    iou = score(m_try)
                    if iou > best_score:
                        best_score = iou
                        best_m = m_try

    if img_scale < 0.999:
        inv = 1.0 / img_scale
        best_m = _scale_affine_matrix(best_m, inv)

    label = f"refine_iou{best_score:.3f}"
    return best_m, best_score, label


def mask_similarity_matrix(
    msi_valid: np.ndarray,
    he_rgb: np.ndarray,
    *,
    msi_h: int,
    msi_w: int,
) -> np.ndarray:
    """Similarity init: scale + translate HE tissue bbox onto MSI tissue bbox."""
    he_h, he_w = he_rgb.shape[:2]
    he_mask = _he_tissue_mask(he_rgb)
    m_box = _bbox(msi_valid)
    h_box = _bbox(he_mask)
    if m_box is None or h_box is None:
        return letterbox_matrix(he_h, he_w, msi_h, msi_w)

    my0, mx0, my1, mx1 = m_box
    hy0, hx0, hy1, hx1 = h_box
    mh, mw = my1 - my0 + 1, mx1 - mx0 + 1
    hh, hw = hy1 - hy0 + 1, hx1 - hx0 + 1
    scale = min(mw / max(hw, 1), mh / max(hh, 1))
    scale = float(np.clip(scale, 1e-4, 1e4))

    mcx, mcy = (mx0 + mx1) * 0.5, (my0 + my1) * 0.5
    hcx, hcy = (hx0 + hx1) * 0.5, (hy0 + hy1) * 0.5
    tx = mcx - scale * hcx
    ty = mcy - scale * hcy
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)


def _resize_gray(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    gray = np.asarray(gray, dtype=np.uint8)
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return gray, 1.0
    scale = max_side / float(longest)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA), scale


def _resize_mask(mask: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    mask_u8 = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    resized, scale = _resize_gray(mask_u8, max_side)
    return resized > 127, scale


def _motion_constant(motion: MotionMode) -> int:
    key = str(motion).strip().lower()
    if key == "euclidean":
        return cv2.MOTION_EUCLIDEAN
    if key == "homography":
        return cv2.MOTION_HOMOGRAPHY
    return cv2.MOTION_AFFINE


def _ecc_delta_to_full(msi_h: int, msi_w: int, m_delta: np.ndarray, image_scale: float) -> np.ndarray:
    """Lift an ECC delta (estimated on downscaled images) to full MSI pixel units."""
    if image_scale >= 0.999:
        return np.asarray(m_delta, dtype=np.float32)
    inv = 1.0 / float(image_scale)
    s = np.array([[inv, 0.0, 0.0], [0.0, inv, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    s_inv = np.array([[image_scale, 0.0, 0.0], [0.0, image_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    m3 = np.vstack([m_delta, [0.0, 0.0, 1.0]]).astype(np.float32)
    lifted = s_inv @ m3 @ s
    return lifted[:2, :].astype(np.float32)


def _run_ecc_stage(
    ref_gray: np.ndarray,
    mov_gray: np.ndarray,
    *,
    ref_mask: np.ndarray | None,
    mov_mask: np.ndarray | None,
    motion: MotionMode,
    ecc_iterations: int,
    ecc_epsilon: float,
    gauss_filt_size: int,
) -> tuple[float | None, np.ndarray, bool, str]:
    motion_mode = _motion_constant(motion)
    if motion_mode == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(ecc_iterations), float(ecc_epsilon))
    ref_m = (ref_mask.astype(np.uint8) * 255) if ref_mask is not None else None
    mov_m = (mov_mask.astype(np.uint8) * 255) if mov_mask is not None else None
    try:
        corr, warp_out = cv2.findTransformECC(
            ref_gray,
            mov_gray,
            warp,
            motion_mode,
            criteria,
            inputMask=mov_m,
            gaussFiltSize=int(gauss_filt_size),
        )
        if motion_mode == cv2.MOTION_HOMOGRAPHY:
            m_delta = warp_out[:2, :].astype(np.float32)
        else:
            m_delta = np.asarray(warp_out, dtype=np.float32)
        return float(corr), m_delta, True, "ok"
    except cv2.error as exc:
        return None, np.eye(2, 3, dtype=np.float32), False, str(exc)


def _load_msi_volume(
    npy_path: Path,
    *,
    transpose_msi: bool,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    from debug_edge_maps import _load_msi

    vol = _load_msi(npy_path, transpose_msi=transpose_msi, normalization=normalization)
    valid = vol.sum(axis=-1) > 0
    return vol, valid


def _msi_reference_uint8(
    vol: np.ndarray,
    valid: np.ndarray,
    *,
    mode: ReferenceMode,
    rank_config_path: Path | None,
    mics_rgb: np.ndarray | None = None,
    mics_channel: MicsChannel = "l",
) -> np.ndarray | None:
    if mode in ("mics", "mics_edges"):
        if mics_rgb is None:
            return None
        from msi_visual.registration.mics_reference import mics_reference_gray

        ch: MicsChannel = "edges" if mode == "mics_edges" else mics_channel
        return mics_reference_gray(mics_rgb, valid, channel=ch)

    if mode == "valid_mask":
        return valid.astype(np.uint8) * 255

    if mode == "tic":
        tic = vol.sum(axis=-1)
        tic = np.where(valid, tic, 0.0)
        return _normalize_uint8(tic, mask=valid)

    if mode == "solace":
        if rank_config_path is None or not rank_config_path.is_file():
            raise FileNotFoundError(f"rank_config required for solace reference: {rank_config_path}")
        from debug_edge_maps import _to_uint8_gray01
        from omegaconf import OmegaConf
        from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method

        rank_cfg = OmegaConf.load(str(rank_config_path))
        ms_cfg = getattr(rank_cfg, "edge_maps", None)
        multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
        sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
        if multi is None or not bool(getattr(multi, "enabled", False)):
            sigmas = [0.0]
        aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
        spatial_enabled = bool(getattr(rank_cfg.spatial_normalization, "enabled", True))
        edge_n = _compute_hd_edge_n_for_method(
            rank_cfg,
            vol,
            valid,
            getattr(rank_cfg, "hd_edges", None),
            "soft_landmark_contrast",
            sigmas,
            aggregation,
            spatial_enabled,
        )
        return _to_uint8_gray01(edge_n)

    raise ValueError(f"Unknown reference mode: {mode!r}")


def default_registration_stages() -> list[EccStageConfig]:
    return [
        EccStageConfig(name="coarse_euclidean", reference="tic", max_side=128, motion="euclidean", ecc_iterations=2500),
        EccStageConfig(name="medium_affine", reference="tic", max_side=256, motion="affine", ecc_iterations=3500),
        EccStageConfig(name="fine_affine", reference="tic", max_side=512, motion="affine", ecc_iterations=4000),
        EccStageConfig(name="full_affine", reference="tic", max_side=None, motion="affine", ecc_iterations=5000),
        EccStageConfig(name="mics_medium_affine", reference="mics", max_side=256, motion="affine", ecc_iterations=3500),
        EccStageConfig(name="mics_full_affine", reference="mics", max_side=None, motion="affine", ecc_iterations=4000),
        EccStageConfig(name="edge_refine", reference="solace", max_side=None, motion="affine", ecc_iterations=3500),
        EccStageConfig(name="mics_edge_refine", reference="mics_edges", max_side=None, motion="affine", ecc_iterations=3000),
    ]


def _pick_rotation_and_init(
    he_rgb: np.ndarray,
    msi_valid: np.ndarray,
    *,
    msi_h: int,
    msi_w: int,
    rotation_search: Sequence[int],
    init_method: str,
    mics_rgb: np.ndarray | None = None,
    mics_channel: MicsChannel = "l",
    mask_only: bool = False,
) -> tuple[np.ndarray, int, str, np.ndarray]:
    mics_mask = None
    if not mask_only and mics_rgb is not None:
        from msi_visual.registration.mics_reference import mics_structure_mask

        mics_mask = mics_structure_mask(mics_rgb, msi_valid, channel=mics_channel)

    rotations = [int(r) % 360 for r in rotation_search] or [0]
    best_score = -1.0
    best_rot = 0
    best_m = letterbox_matrix(he_rgb.shape[0], he_rgb.shape[1], msi_h, msi_w)
    best_he = he_rgb

    for rot in rotations:
        he_rot = _rotate_he_rgb(he_rgb, rot)
        if init_method == "mask_similarity":
            m0 = mask_similarity_matrix(msi_valid, he_rot, msi_h=msi_h, msi_w=msi_w)
        else:
            m0 = letterbox_matrix(he_rot.shape[0], he_rot.shape[1], msi_h, msi_w)
        score = _mask_alignment_iou(msi_valid, he_rot, m0, (msi_h, msi_w))
        if mics_mask is not None:
            preview = warp_he_to_msi(he_rot, m0, (msi_h, msi_w))
            score = 0.5 * score + 0.5 * _mask_iou(mics_mask, _he_tissue_mask(preview))
        if score > best_score:
            best_score = score
            best_rot = rot
            best_m = m0
            best_he = he_rot

    init_label = f"{init_method}_rot{best_rot}_iou{best_score:.3f}"
    return best_he, best_rot, init_label, best_m


def _register_mask_only(
    *,
    he_rgb: np.ndarray,
    he_h: int,
    he_w: int,
    valid: np.ndarray,
    msi_h: int,
    msi_w: int,
    slide_key: str,
    sample_id: str,
    npy_path: Path,
    he_path: Path,
    rotation_search: Sequence[int],
    init_method: str,
    refine_enabled: bool,
    refine_max_side: int,
    refine_angle_deg_range: tuple[float, float],
    refine_angle_deg_step: float,
    refine_scale_range: tuple[float, float],
    refine_scale_step: float,
    refine_shift_frac: float,
    refine_shift_step: float,
) -> RegistrationResult:
    """Align outer MSI valid mask to H&E tissue mask (rotation + uniform scale + shift)."""
    he_work, rotation_deg, init_label, m_total = _pick_rotation_and_init(
        he_rgb,
        valid,
        msi_h=msi_h,
        msi_w=msi_w,
        rotation_search=rotation_search,
        init_method=init_method,
        mask_only=True,
    )

    init_iou = _mask_alignment_iou(valid, he_work, m_total, (msi_h, msi_w))
    refine_label = "refine_skipped"
    final_iou = init_iou

    if refine_enabled:
        m_total, final_iou, refine_label = _refine_mask_similarity(
            valid,
            he_work,
            m_total,
            msi_h=msi_h,
            msi_w=msi_w,
            max_side=refine_max_side,
            angle_deg_range=refine_angle_deg_range,
            angle_deg_step=refine_angle_deg_step,
            scale_range=refine_scale_range,
            scale_step=refine_scale_step,
            shift_frac=refine_shift_frac,
            shift_step=refine_shift_step,
        )

    stages = [
        {
            "name": "coarse_rotation",
            "reference": "valid_mask",
            "motion": "similarity",
            "max_side": None,
            "scale": 1.0,
            "success": True,
            "correlation": init_iou,
            "message": init_label,
        }
    ]
    if refine_enabled:
        stages.append(
            {
                "name": "mask_refine",
                "reference": "valid_mask",
                "motion": "similarity",
                "max_side": refine_max_side,
                "scale": 1.0,
                "success": True,
                "correlation": final_iou,
                "message": refine_label,
            }
        )

    return RegistrationResult(
        sample_id=sample_id,
        slide_key=slide_key,
        msi_path=str(npy_path),
        he_path=str(he_path),
        target_space="msi_hw",
        msi_shape_hw=[msi_h, msi_w],
        he_shape_hw=[he_h, he_w],
        warp_matrix_2x3=m_total.tolist(),
        method="mask_only",
        reference="valid_mask",
        success=True,
        correlation=final_iou,
        message=f"{init_label};{refine_label}",
        rotation_deg=int(rotation_deg),
        init_method=str(init_method),
        mics_source=None,
        mics_channel="l",
        stages=stages,
    )


def register_he_to_msi(
    *,
    npy_path: Path,
    he_path: Path,
    slide_key: str,
    sample_id: str,
    he_rgb: np.ndarray | None = None,
    transpose_msi: bool = False,
    normalization: str = "tic",
    reference: ReferenceMode = "tic",
    rank_config_path: Path | None = None,
    motion: str = "affine",
    ecc_iterations: int = 5000,
    ecc_epsilon: float = 1e-6,
    gauss_filt_size: int = 5,
    stages: Sequence[EccStageConfig] | None = None,
    rotation_search: Sequence[int] | None = None,
    init_method: str = "mask_similarity",
    mics_rgb: np.ndarray | None = None,
    mics_channel: MicsChannel = "l",
    mics_source: str | None = None,
    strategy: RegistrationStrategy = "mask_only",
    refine_enabled: bool = True,
    refine_max_side: int = 384,
    refine_angle_deg_range: tuple[float, float] = (-35.0, 35.0),
    refine_angle_deg_step: float = 5.0,
    refine_scale_range: tuple[float, float] = (0.82, 1.18),
    refine_scale_step: float = 0.04,
    refine_shift_frac: float = 0.12,
    refine_shift_step: float = 0.04,
) -> RegistrationResult:
    """
    Register H&E thumbnail to MSI pixel grid.

    ``mask_only`` (default): outer MSI valid mask vs H&E tissue mask,
    coarse 90° rotation + bbox scale/translate, then grid refinement for
    rotation, uniform scale, and shift.

    ``multistage``: legacy coarse-to-fine ECC on TIC / MiCS / SoLACE references.
    """
    if he_rgb is not None:
        he_rgb = np.asarray(he_rgb, dtype=np.uint8)
    else:
        he_rgb = np.asarray(Image.open(he_path).convert("RGB"), dtype=np.uint8)
    he_h, he_w = he_rgb.shape[:2]

    vol, valid = _load_msi_volume(npy_path, transpose_msi=transpose_msi, normalization=normalization)
    msi_h, msi_w = int(valid.shape[0]), int(valid.shape[1])

    rot_search = list(rotation_search) if rotation_search is not None else [0, 90, 180, 270]

    if strategy == "mask_only":
        return _register_mask_only(
            he_rgb=he_rgb,
            he_h=he_h,
            he_w=he_w,
            valid=valid,
            msi_h=msi_h,
            msi_w=msi_w,
            slide_key=slide_key,
            sample_id=sample_id,
            npy_path=npy_path,
            he_path=he_path,
            rotation_search=rot_search,
            init_method=init_method,
            refine_enabled=refine_enabled,
            refine_max_side=refine_max_side,
            refine_angle_deg_range=refine_angle_deg_range,
            refine_angle_deg_step=refine_angle_deg_step,
            refine_scale_range=refine_scale_range,
            refine_scale_step=refine_scale_step,
            refine_shift_frac=refine_shift_frac,
            refine_shift_step=refine_shift_step,
        )

    if mics_rgb is not None:
        mics_rgb = np.asarray(mics_rgb, dtype=np.uint8)
        if mics_rgb.shape[:2] != (msi_h, msi_w):
            mics_rgb = cv2.resize(mics_rgb, (msi_w, msi_h), interpolation=cv2.INTER_AREA)

    stage_cfgs = list(stages) if stages is not None else default_registration_stages()
    if not stage_cfgs:
        stage_cfgs = [
            EccStageConfig(
                name="single",
                reference=reference,
                max_side=None,
                motion=motion,  # type: ignore[arg-type]
                ecc_iterations=ecc_iterations,
                ecc_epsilon=ecc_epsilon,
                gauss_filt_size=gauss_filt_size,
            )
        ]

    he_work, rotation_deg, init_label, m_total = _pick_rotation_and_init(
        he_rgb,
        valid,
        msi_h=msi_h,
        msi_w=msi_w,
        rotation_search=rot_search,
        init_method=init_method,
        mics_rgb=mics_rgb,
        mics_channel=mics_channel,
        mask_only=False,
    )

    stage_results: list[RegistrationStageResult] = []
    last_corr: float | None = None
    any_success = False
    refs_used: list[str] = []

    for sc in stage_cfgs:
        ref_mode: ReferenceMode = sc.reference  # type: ignore[assignment]
        if ref_mode in ("mics", "mics_edges") and mics_rgb is None:
            logger.warning("%s: skipping stage %s (no MiCS reference)", slide_key, sc.name)
            stage_results.append(
                RegistrationStageResult(
                    name=sc.name,
                    reference=sc.reference,
                    motion=sc.motion,
                    max_side=sc.max_side,
                    scale=1.0,
                    success=False,
                    correlation=None,
                    message="no_mics_reference",
                )
            )
            continue

        msi_ref = _msi_reference_uint8(
            vol,
            valid,
            mode=ref_mode,
            rank_config_path=rank_config_path,
            mics_rgb=mics_rgb,
            mics_channel=mics_channel,
        )
        if msi_ref is None:
            logger.warning("%s: skipping stage %s (reference unavailable)", slide_key, sc.name)
            continue
        ref_gray = msi_ref if msi_ref.ndim == 2 else cv2.cvtColor(msi_ref, cv2.COLOR_RGB2GRAY)
        he_warped = warp_he_to_msi(he_work, m_total, (msi_h, msi_w))
        mov_gray = _he_gray_l(he_warped)

        max_side = sc.max_side
        img_scale = 1.0
        if max_side is not None and max_side > 0:
            ref_gray, img_scale = _resize_gray(ref_gray, int(max_side))
            mov_gray, _ = _resize_gray(mov_gray, int(max_side))
            ref_mask, _ = _resize_mask(valid, int(max_side))
            mov_mask, _ = _resize_mask(_he_tissue_mask(he_warped), int(max_side))
        else:
            ref_mask = valid
            mov_mask = _he_tissue_mask(he_warped)

        corr, m_delta, ok, msg = _run_ecc_stage(
            ref_gray,
            mov_gray,
            ref_mask=ref_mask,
            mov_mask=mov_mask,
            motion=sc.motion,
            ecc_iterations=sc.ecc_iterations,
            ecc_epsilon=sc.ecc_epsilon,
            gauss_filt_size=sc.gauss_filt_size,
        )
        if ok:
            m_delta_full = _ecc_delta_to_full(msi_h, msi_w, m_delta, img_scale)
            m_total = _compose_affine(m_delta_full, m_total)
            last_corr = corr
            any_success = True
            refs_used.append(sc.reference)
        stage_results.append(
            RegistrationStageResult(
                name=sc.name,
                reference=sc.reference,
                motion=sc.motion,
                max_side=sc.max_side,
                scale=img_scale,
                success=ok,
                correlation=corr,
                message=msg,
            )
        )
        if not ok:
            logger.warning("%s stage %s failed: %s", slide_key, sc.name, msg)

    if rotation_deg != 0:
        # Final matrix applies to rotated HE; store note in message. Caller must rotate first.
        pass

    method = "multistage" if len(stage_cfgs) > 1 else "single_ecc"
    if any_success:
        method += "+ecc"
    else:
        method += "+init_only"

    return RegistrationResult(
        sample_id=sample_id,
        slide_key=slide_key,
        msi_path=str(npy_path),
        he_path=str(he_path),
        target_space="msi_hw",
        msi_shape_hw=[msi_h, msi_w],
        he_shape_hw=[he_h, he_w],
        warp_matrix_2x3=m_total.tolist(),
        method=method,
        reference=",".join(dict.fromkeys(refs_used)) if refs_used else str(reference),
        success=any_success,
        correlation=last_corr,
        message=init_label,
        rotation_deg=int(rotation_deg),
        init_method=str(init_method),
        mics_source=mics_source,
        mics_channel=str(mics_channel),
        stages=[asdict(s) for s in stage_results],
    )


def _he_shape_after_rotation(shape_hw: tuple[int, int], rotation_deg: int) -> tuple[int, int]:
    h, w = (int(shape_hw[0]), int(shape_hw[1]))
    if int(rotation_deg) % 180 == 90:
        return w, h
    return h, w


def scale_warp_matrix_for_he_resize(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Scale an MSI←HE warp when the H&E source image is uniformly resized."""
    s = float(scale)
    if not np.isfinite(s) or abs(s - 1.0) < 1e-6:
        return np.asarray(matrix, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32).copy()
    m[0, :] *= s
    m[1, :] *= s
    return m


def registration_matches_he_msi(
    reg: RegistrationResult,
    *,
    slide_key: str,
    he_hw: tuple[int, int],
    msi_hw: tuple[int, int],
) -> bool:
    """True when ``reg`` was computed for this slide, MSI grid, and H&E raster."""
    if not reg.success:
        return False
    if str(reg.method) != "mask_only":
        return False
    if str(reg.slide_key) != str(slide_key):
        return False
    if tuple(int(x) for x in reg.msi_shape_hw) != tuple(int(x) for x in msi_hw):
        return False
    if tuple(int(x) for x in reg.he_shape_hw) != (int(he_hw[0]), int(he_hw[1])):
        return False
    return True


def apply_registration_to_he(
    he_rgb: np.ndarray,
    result: RegistrationResult,
    out_hw: tuple[int, int],
    *,
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Apply stored rotation + affine warp to an H&E RGB image."""
    he_rot = _rotate_he_rgb(he_rgb, result.rotation_deg)
    ref_rot_hw = _he_shape_after_rotation(tuple(result.he_shape_hw), result.rotation_deg)
    matrix = result.matrix()
    if he_rot.shape[:2] != ref_rot_hw:
        sy = he_rot.shape[0] / max(ref_rot_hw[0], 1)
        sx = he_rot.shape[1] / max(ref_rot_hw[1], 1)
        matrix = scale_warp_matrix_for_he_resize(matrix, 0.5 * (sx + sy))
    return warp_he_to_msi(he_rot, matrix, out_hw, border_value=border_value)


def resolve_registration_path(
    he_path: Path,
    json_filename: str = "registration.json",
    *,
    slide_key: str | None = None,
) -> Path:
    """Registration JSON beside the H&E folder (per ``slide_key`` when given)."""
    if slide_key:
        safe = str(slide_key).replace("/", "_").replace("\\", "_")
        return he_path.parent / f"registration_{safe}.json"
    return he_path.parent / json_filename


def save_registration(result: RegistrationResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def load_registration(path: Path) -> RegistrationResult | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(RegistrationResult)}
    filtered = {k: v for k, v in data.items() if k in known}
    if "stages" not in filtered:
        filtered["stages"] = []
    if "rotation_deg" not in filtered:
        filtered["rotation_deg"] = 0
    if "mics_source" not in filtered:
        filtered["mics_source"] = None
    if "mics_channel" not in filtered:
        filtered["mics_channel"] = "l"
    return RegistrationResult(**filtered)
