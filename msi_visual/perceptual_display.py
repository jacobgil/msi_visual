"""Method-agnostic perceptual display remapping for MSI RGB visualizations."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

RGB_PERMUTATIONS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)

LUT_DELTA_MIN = -45.0
LUT_DELTA_MAX = 45.0


def _lut_grid_sizes(
    lut_ab_grid: int,
    lut_l_grid: int,
) -> tuple[int, int]:
    g = max(2, int(lut_ab_grid))
    lg = max(1, int(lut_l_grid))
    return lg, g


def _zero_lut_delta(lut_l_grid: int, lut_ab_grid: int) -> np.ndarray:
    lg, g = _lut_grid_sizes(lut_ab_grid, lut_l_grid)
    return np.zeros((lg, g, g, 2), dtype=np.float64)


@dataclass
class PerceptualDisplayParams:
    """Display remap knobs."""

    strategy: str = "perceptual_lut"
    # --- perceptual_lut: global 1-to-1 (L,a,b) -> (L,a',b') trilinear LUT ---
    lut_ab_grid: int = 5
    lut_l_grid: int = 1
    lut_delta_ab: np.ndarray = field(default_factory=lambda: _zero_lut_delta(1, 5))
    # --- perceptual matrix (L fixed; 2x2 a,b) ---
    hue_rotation_deg: float = 0.0
    chroma_scale: float = 1.0
    ab_shear_deg: float = 0.0
    chroma_aniso_b: float = 1.0
    rgb_perm_index: int = 0
    ab_offset_a: float = 0.0
    ab_offset_b: float = 0.0
    rgb_channel_order: tuple[int, int, int] = (0, 1, 2)
    # --- contrast (legacy) ---
    gamma: float = 1.0
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    l_contrast: float = 1.0
    rgb_diag: tuple[float, float, float] = (1.0, 1.0, 1.0)
    clahe_clip: float = 0.0
    clahe_tile: int = 8

    def normalized_strategy(self) -> str:
        s = str(self.strategy).strip().lower().replace("-", "_")
        if s in ("perceptual_lut", "lut", "lut3d", "3d_lut"):
            return "perceptual_lut"
        if s in ("perceptual", "perceptual_learned", "learned", "lab", "perceptual_matrix"):
            return "perceptual"
        if s in ("shuffle", "color", "hue", "recolor", "color_shuffle"):
            return "color_shuffle"
        if s in ("legacy", "legacy_contrast", "contrast_boost"):
            return "contrast"
        return s

    def effective_rgb_order(self) -> tuple[int, int, int]:
        order = tuple(int(x) for x in self.rgb_channel_order)
        if order != (0, 1, 2):
            return order
        if self.normalized_strategy() == "color_shuffle":
            idx = int(self.rgb_perm_index) % len(RGB_PERMUTATIONS)
            return RGB_PERMUTATIONS[idx]
        return order

    def _lut_shape_ok(self) -> bool:
        lg, g = _lut_grid_sizes(self.lut_ab_grid, self.lut_l_grid)
        d = np.asarray(self.lut_delta_ab)
        return d.shape == (lg, g, g, 2)

    def to_vector(self) -> np.ndarray:
        st = self.normalized_strategy()
        if st == "perceptual_lut":
            if not self._lut_shape_ok():
                raise ValueError(
                    f"lut_delta_ab shape {np.asarray(self.lut_delta_ab).shape} "
                    f"!= ({self.lut_l_grid}, {self.lut_ab_grid}, {self.lut_ab_grid}, 2)"
                )
            return np.asarray(self.lut_delta_ab, dtype=np.float64).ravel()
        if st == "perceptual":
            return np.array(
                [
                    self.hue_rotation_deg,
                    self.chroma_scale,
                    self.ab_shear_deg,
                    self.chroma_aniso_b,
                    self.ab_offset_a,
                    self.ab_offset_b,
                ],
                dtype=np.float64,
            )
        if st == "color_shuffle":
            return np.array(
                [
                    self.hue_rotation_deg,
                    self.chroma_scale,
                    float(self.rgb_perm_index),
                    self.ab_offset_a,
                    self.ab_offset_b,
                ],
                dtype=np.float64,
            )
        return np.array(
            [
                self.gamma,
                self.percentile_low,
                self.percentile_high,
                self.chroma_scale,
                self.l_contrast,
                self.rgb_diag[0],
                self.rgb_diag[1],
                self.rgb_diag[2],
                self.clahe_clip,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(
        cls,
        x: np.ndarray,
        *,
        strategy: str = "perceptual_lut",
        lut_ab_grid: int = 5,
        lut_l_grid: int = 1,
    ) -> PerceptualDisplayParams:
        v = np.asarray(x, dtype=np.float64).ravel()
        st = str(strategy).strip().lower().replace("-", "_")
        if st in ("perceptual_lut", "lut", "lut3d", "3d_lut"):
            st = "perceptual_lut"
        if st in ("perceptual", "perceptual_learned", "learned", "lab", "perceptual_matrix"):
            st = "perceptual"
        if st in ("shuffle", "color", "hue", "recolor", "color_shuffle"):
            st = "color_shuffle"
        if st == "perceptual_lut":
            lg, g = _lut_grid_sizes(lut_ab_grid, lut_l_grid)
            n = lg * g * g * 2
            if v.size < n:
                raise ValueError(f"Expected {n} LUT parameters, got {v.size}")
            delta = v[:n].reshape(lg, g, g, 2)
            return cls(
                strategy="perceptual_lut",
                lut_ab_grid=g,
                lut_l_grid=lg,
                lut_delta_ab=delta,
            )
        if st == "perceptual":
            if v.size < 6:
                raise ValueError(f"Expected 6 perceptual parameters, got {v.size}")
            return cls(
                strategy="perceptual",
                hue_rotation_deg=float(v[0]),
                chroma_scale=float(v[1]),
                ab_shear_deg=float(v[2]),
                chroma_aniso_b=float(v[3]),
                ab_offset_a=float(v[4]),
                ab_offset_b=float(v[5]),
            )
        if st == "color_shuffle":
            if v.size < 5:
                raise ValueError(f"Expected 5 color_shuffle parameters, got {v.size}")
            return cls(
                strategy="color_shuffle",
                hue_rotation_deg=float(v[0]),
                chroma_scale=float(v[1]),
                rgb_perm_index=int(round(v[2])) % len(RGB_PERMUTATIONS),
                ab_offset_a=float(v[3]),
                ab_offset_b=float(v[4]),
            )
        if v.size < 9:
            raise ValueError(f"Expected 9 contrast parameters, got {v.size}")
        return cls(
            strategy="contrast",
            gamma=float(v[0]),
            percentile_low=float(v[1]),
            percentile_high=float(v[2]),
            chroma_scale=float(v[3]),
            l_contrast=float(v[4]),
            rgb_diag=(float(v[5]), float(v[6]), float(v[7])),
            clahe_clip=float(v[8]),
        )

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rgb_diag"] = list(self.rgb_diag)
        d["rgb_channel_order"] = list(self.effective_rgb_order())
        d["strategy"] = self.normalized_strategy()
        d["lut_delta_ab"] = np.asarray(self.lut_delta_ab, dtype=np.float64).tolist()
        if self.normalized_strategy() == "perceptual":
            d["perceptual_ab_matrix"] = perceptual_ab_matrix(self).tolist()
        return d

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> PerceptualDisplayParams:
        if not raw:
            return cls()
        diag = raw.get("rgb_diag", (1.0, 1.0, 1.0))
        diag_t = (
            (float(diag[0]), float(diag[1]), float(diag[2]))
            if isinstance(diag, (list, tuple))
            else (1.0, 1.0, 1.0)
        )
        order = raw.get("rgb_channel_order", (0, 1, 2))
        order_t = (
            (int(order[0]), int(order[1]), int(order[2]))
            if isinstance(order, (list, tuple)) and len(order) == 3
            else (0, 1, 2)
        )
        ab_g = int(raw.get("lut_ab_grid", 5))
        l_g = int(raw.get("lut_l_grid", 1))
        delta_raw = raw.get("lut_delta_ab")
        if delta_raw is not None:
            delta = np.asarray(delta_raw, dtype=np.float64)
        else:
            delta = _zero_lut_delta(l_g, ab_g)
        return cls(
            strategy=str(raw.get("strategy", "perceptual_lut")),
            lut_ab_grid=ab_g,
            lut_l_grid=l_g,
            lut_delta_ab=delta,
            hue_rotation_deg=float(raw.get("hue_rotation_deg", 0.0)),
            chroma_scale=float(raw.get("chroma_scale", 1.0)),
            ab_shear_deg=float(raw.get("ab_shear_deg", 0.0)),
            chroma_aniso_b=float(raw.get("chroma_aniso_b", 1.0)),
            rgb_perm_index=int(raw.get("rgb_perm_index", 0)),
            ab_offset_a=float(raw.get("ab_offset_a", 0.0)),
            ab_offset_b=float(raw.get("ab_offset_b", 0.0)),
            rgb_channel_order=order_t,
            gamma=float(raw.get("gamma", 1.0)),
            percentile_low=float(raw.get("percentile_low", 1.0)),
            percentile_high=float(raw.get("percentile_high", 99.0)),
            l_contrast=float(raw.get("l_contrast", 1.0)),
            rgb_diag=diag_t,
            clahe_clip=float(raw.get("clahe_clip", 0.0)),
            clahe_tile=int(raw.get("clahe_tile", 8)),
        )


PARAM_BOUNDS_CONTRAST: list[tuple[float, float]] = [
    (0.55, 1.85),
    (0.0, 12.0),
    (88.0, 100.0),
    (0.5, 2.5),
    (0.7, 2.0),
    (0.65, 1.45),
    (0.65, 1.45),
    (0.65, 1.45),
    (0.0, 4.0),
]

PARAM_BOUNDS_COLOR_SHUFFLE: list[tuple[float, float]] = [
    (0.0, 360.0),
    (0.5, 1.5),
    (0.0, 5.99),
    (-35.0, 35.0),
    (-35.0, 35.0),
]

PARAM_BOUNDS_PERCEPTUAL: list[tuple[float, float]] = [
    (0.0, 360.0),
    (0.4, 2.2),
    (-40.0, 40.0),
    (0.4, 2.2),
    (-40.0, 40.0),
    (-40.0, 40.0),
]

PARAM_BOUNDS = PARAM_BOUNDS_CONTRAST

PERCEPTUAL_COLOR_METRIC_KEYS: tuple[str, ...] = (
    "lab_chroma_entropy",
    "lab_mean_chroma",
    "luminance_gradient_mean",
    "luminance_rms_contrast",
)


def param_bounds_for(
    strategy: str,
    *,
    lut_ab_grid: int = 5,
    lut_l_grid: int = 1,
) -> list[tuple[float, float]]:
    st = str(strategy).strip().lower().replace("-", "_")
    if st in ("perceptual_lut", "lut", "lut3d", "3d_lut"):
        lg, g = _lut_grid_sizes(lut_ab_grid, lut_l_grid)
        n = lg * g * g * 2
        return [(LUT_DELTA_MIN, LUT_DELTA_MAX)] * n
    if st in ("perceptual", "perceptual_learned", "learned", "lab", "perceptual_matrix"):
        return PARAM_BOUNDS_PERCEPTUAL
    if st in ("shuffle", "color", "hue", "recolor", "color_shuffle"):
        return PARAM_BOUNDS_COLOR_SHUFFLE
    return PARAM_BOUNDS_CONTRAST


def perceptual_ab_matrix(params: PerceptualDisplayParams) -> np.ndarray:
    th = math.radians(float(params.hue_rotation_deg))
    sh = math.radians(float(params.ab_shear_deg))
    c, sn = math.cos(th), math.sin(th)
    t = math.tan(sh)
    rot = np.array([[c, -sn], [sn, c]], dtype=np.float64)
    shear = np.array([[1.0, t], [0.0, 1.0]], dtype=np.float64)
    sc = float(params.chroma_scale)
    an = float(params.chroma_aniso_b)
    scale = np.diag([sc, sc * an])
    return rot @ shear @ scale


def _coord_to_frac(values: np.ndarray, n_nodes: int) -> np.ndarray:
    """Map [0,255] OpenCV LAB channel to fractional grid index [0, n-1]."""
    if n_nodes <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values.astype(np.float64) / 255.0, 0.0, 1.0) * float(n_nodes - 1)


def _trilinear_sample_lut(
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """Trilinear sample ``lut`` (l_g, g, g) at each pixel (L,a,b)."""
    lg, g, g2 = lut.shape
    assert g == g2
    lf = _coord_to_frac(L, lg)
    af = _coord_to_frac(a, g)
    bf = _coord_to_frac(b, g)

    l0 = np.floor(lf).astype(np.int32)
    a0 = np.floor(af).astype(np.int32)
    b0 = np.floor(bf).astype(np.int32)
    l1 = np.clip(l0 + 1, 0, max(0, lg - 1))
    a1 = np.clip(a0 + 1, 0, g - 1)
    b1 = np.clip(b0 + 1, 0, g - 1)
    l0 = np.clip(l0, 0, max(0, lg - 1))
    a0 = np.clip(a0, 0, g - 1)
    b0 = np.clip(b0, 0, g - 1)

    ld = lf - l0
    ad = af - a0
    bd = bf - b0

    c000 = lut[l0, a0, b0]
    c001 = lut[l0, a0, b1]
    c010 = lut[l0, a1, b0]
    c011 = lut[l0, a1, b1]
    c100 = lut[l1, a0, b0]
    c101 = lut[l1, a0, b1]
    c110 = lut[l1, a1, b0]
    c111 = lut[l1, a1, b1]

    c00 = c000 * (1.0 - ad) + c010 * ad
    c01 = c001 * (1.0 - ad) + c011 * ad
    c10 = c100 * (1.0 - ad) + c110 * ad
    c11 = c101 * (1.0 - ad) + c111 * ad
    c0 = c00 * (1.0 - bd) + c01 * bd
    c1 = c10 * (1.0 - bd) + c11 * bd
    return c0 * (1.0 - ld) + c1 * ld


def tissue_mask_from_rgb(rgb: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    if valid_mask is not None:
        return np.asarray(valid_mask, dtype=bool)
    x = np.asarray(rgb)
    if x.ndim == 2:
        return x > 0
    return np.sum(x[..., :3], axis=-1) > 0


def _to_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    src = np.asarray(rgb)
    if src.ndim == 2:
        src = np.stack([src, src, src], axis=-1)
    if np.issubdtype(src.dtype, np.integer):
        return np.clip(src[..., :3], 0, 255).astype(np.uint8)
    return (np.clip(src[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def _luminance_from_rgb01(rgb01: np.ndarray) -> np.ndarray:
    r = rgb01[..., 0].astype(np.float32)
    g = rgb01[..., 1].astype(np.float32)
    b = rgb01[..., 2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32, copy=False)


def compute_ab_shuffle_distance(
    rgb_in: np.ndarray,
    rgb_out: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    vm = np.asarray(valid_mask, dtype=bool)
    if not np.any(vm):
        return float("nan")
    lab_in = cv2.cvtColor(_to_rgb_u8(rgb_in), cv2.COLOR_RGB2LAB)
    lab_out = cv2.cvtColor(_to_rgb_u8(rgb_out), cv2.COLOR_RGB2LAB)
    da = lab_out[..., 1].astype(np.float32) - lab_in[..., 1].astype(np.float32)
    db = lab_out[..., 2].astype(np.float32) - lab_in[..., 2].astype(np.float32)
    dist = np.sqrt(da * da + db * db)
    return float(np.mean(dist[vm]))


def compute_color_metrics(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lab_entropy_bins_a: int = 16,
    lab_entropy_bins_b: int = 16,
) -> dict[str, float]:
    vm = np.asarray(valid_mask, dtype=bool)
    nan = {
        "luminance_rms_contrast": float("nan"),
        "luminance_gradient_mean": float("nan"),
        "lab_chroma_entropy": float("nan"),
        "lab_mean_chroma": float("nan"),
    }
    if not np.any(vm):
        return nan

    x = np.asarray(rgb)
    if x.ndim != 3 or x.shape[-1] < 3:
        return nan

    rgb_u8 = _to_rgb_u8(x)
    rgb01 = rgb_u8.astype(np.float32) / 255.0

    y = _luminance_from_rgb01(rgb01)
    lum_rms = float(np.std(y[vm]))

    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    grad_mean = float(np.mean(np.sqrt(gx * gx + gy * gy)[vm]))

    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    a = lab[..., 1].astype(np.float32)
    bch = lab[..., 2].astype(np.float32)
    chroma = np.sqrt((a - 128.0) ** 2 + (bch - 128.0) ** 2)
    mean_chroma = float(np.mean(chroma[vm]))

    ba = max(2, int(lab_entropy_bins_a))
    bb = max(2, int(lab_entropy_bins_b))
    a_idx = np.clip((a * ba / 256.0).astype(np.int32), 0, ba - 1)
    b_idx = np.clip((bch * bb / 256.0).astype(np.int32), 0, bb - 1)
    joint = a_idx * bb + b_idx
    vals = joint[vm].astype(np.int64, copy=False)
    counts = np.bincount(vals, minlength=ba * bb).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        entropy = float("nan")
    else:
        p = counts[counts > 0] / total
        entropy = float(-np.sum(p * np.log(p + 1e-15)))

    return {
        "luminance_rms_contrast": lum_rms,
        "luminance_gradient_mean": grad_mean,
        "lab_chroma_entropy": entropy,
        "lab_mean_chroma": mean_chroma,
    }


def _apply_perceptual_lut(
    rgb_u8: np.ndarray,
    params: PerceptualDisplayParams,
    mask: np.ndarray,
) -> np.ndarray:
    """Global 1-to-1 LUT: (L,a,b) in -> (L,a+da,b+db) out; L unchanged (monotone)."""
    if not params._lut_shape_ok():
        params = PerceptualDisplayParams.from_mapping(
            {
                "strategy": "perceptual_lut",
                "lut_ab_grid": params.lut_ab_grid,
                "lut_l_grid": params.lut_l_grid,
                "lut_delta_ab": params.lut_delta_ab,
            }
        )
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1]
    bch = lab[..., 2]
    delta = np.asarray(params.lut_delta_ab, dtype=np.float64)
    da = _trilinear_sample_lut(L, a, bch, delta[..., 0])
    db = _trilinear_sample_lut(L, a, bch, delta[..., 1])
    a_out = np.clip(a + da, 0.0, 255.0)
    b_out = np.clip(bch + db, 0.0, 255.0)
    lab_out = np.stack([L, a_out, b_out], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    out[~mask] = 0
    return out


def _apply_lab_chroma_map(
    rgb_u8: np.ndarray,
    params: PerceptualDisplayParams,
    mask: np.ndarray,
    *,
    use_matrix: bool,
) -> np.ndarray:
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0].copy()
    a = lab[..., 1] - 128.0
    bch = lab[..., 2] - 128.0

    if use_matrix:
        M = perceptual_ab_matrix(params)
        ab = np.stack([a, bch], axis=-1)
        ab_t = np.einsum("ij,...j->...i", M, ab)
        a_out = 128.0 + ab_t[..., 0] + float(params.ab_offset_a)
        b_out = 128.0 + ab_t[..., 1] + float(params.ab_offset_b)
    else:
        theta = math.radians(float(params.hue_rotation_deg))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        a_rot = cos_t * a - sin_t * bch
        b_rot = sin_t * a + cos_t * bch
        cs = float(params.chroma_scale)
        a_out = 128.0 + a_rot * cs + float(params.ab_offset_a)
        b_out = 128.0 + b_rot * cs + float(params.ab_offset_b)

    a_out = np.clip(a_out, 0.0, 255.0)
    b_out = np.clip(b_out, 0.0, 255.0)
    lab_out = np.stack([L, a_out, b_out], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    if not use_matrix:
        p0, p1, p2 = params.effective_rgb_order()
        if (p0, p1, p2) != (0, 1, 2):
            out = out[..., [p0, p1, p2]]

    out[~mask] = 0
    return out


def _apply_color_shuffle(
    rgb_u8: np.ndarray,
    params: PerceptualDisplayParams,
    mask: np.ndarray,
) -> np.ndarray:
    return _apply_lab_chroma_map(rgb_u8, params, mask, use_matrix=False)


def _apply_perceptual_learned(
    rgb_u8: np.ndarray,
    params: PerceptualDisplayParams,
    mask: np.ndarray,
) -> np.ndarray:
    return _apply_lab_chroma_map(rgb_u8, params, mask, use_matrix=True)


def _apply_contrast_remap(
    rgb_u8: np.ndarray,
    params: PerceptualDisplayParams,
    mask: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0].copy()
    a = lab[..., 1].copy()
    bch = lab[..., 2].copy()

    lo = float(params.percentile_low)
    hi = float(params.percentile_high)
    if hi <= lo + 0.5:
        hi = lo + 0.5
    lv = L[mask]
    if lv.size >= 2:
        p_lo = float(np.percentile(lv, lo))
        p_hi = float(np.percentile(lv, hi))
        L = (L - p_lo) / (p_hi - p_lo + 1e-6) * 255.0
    L = np.clip(L, 0.0, 255.0)

    lc = float(params.l_contrast)
    if abs(lc - 1.0) > 1e-4:
        L = 128.0 + (L - 128.0) * lc
        L = np.clip(L, 0.0, 255.0)

    clip = float(params.clahe_clip)
    if clip > 0.05:
        tgs = max(2, int(params.clahe_tile))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tgs, tgs))
        L_u8 = np.clip(L, 0, 255).astype(np.uint8)
        L = clahe.apply(L_u8).astype(np.float32)

    gamma = float(params.gamma)
    if abs(gamma - 1.0) > 1e-4:
        L01 = np.clip(L / 255.0, 0.0, 1.0)
        L = np.power(L01, gamma) * 255.0

    cs = float(params.chroma_scale)
    if abs(cs - 1.0) > 1e-4:
        a = 128.0 + (a - 128.0) * cs
        bch = 128.0 + (bch - 128.0) * cs
    a = np.clip(a, 0.0, 255.0)
    bch = np.clip(bch, 0.0, 255.0)

    lab_out = np.stack([np.clip(L, 0, 255), a, bch], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    dr, dg, db = params.rgb_diag
    if any(abs(x - 1.0) > 1e-4 for x in (dr, dg, db)):
        out_f = out.astype(np.float32)
        out_f[..., 0] *= dr
        out_f[..., 1] *= dg
        out_f[..., 2] *= db
        out = np.clip(out_f, 0, 255).astype(np.uint8)

    out[~mask] = 0
    return out


def apply_perceptual_display(
    rgb: np.ndarray,
    params: PerceptualDisplayParams,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    rgb_u8 = _to_rgb_u8(rgb)
    mask = tissue_mask_from_rgb(rgb_u8, valid_mask)
    st = params.normalized_strategy()
    if st == "perceptual_lut":
        return _apply_perceptual_lut(rgb_u8, params, mask)
    if st == "perceptual":
        return _apply_perceptual_learned(rgb_u8, params, mask)
    if st == "color_shuffle":
        return _apply_color_shuffle(rgb_u8, params, mask)
    return _apply_contrast_remap(rgb_u8, params, mask)


def save_params_json(path: str | Any, params: PerceptualDisplayParams) -> None:
    from pathlib import Path

    Path(path).write_text(json.dumps(params.to_json_dict(), indent=2), encoding="utf-8")


def save_lut_npy(path: str | Any, params: PerceptualDisplayParams) -> None:
    from pathlib import Path

    if params.normalized_strategy() != "perceptual_lut":
        return
    np.save(Path(path), np.asarray(params.lut_delta_ab, dtype=np.float64))
