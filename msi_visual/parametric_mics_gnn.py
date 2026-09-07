"""
Two-stage MiCS + GNN spatial refinement.

Stage 1 (caller): train ``MSIParametricMiCSLMC`` per-pixel PCC.
Stage 2 (this module): frozen MiCS embedding + trainable pixel-neighbor GNN residual
``e = e_mics + Δ_gnn``, optimizing patch continuous Dice vs HD edges + full MiCS batch loss
(cluster + correlation with β/α, same as ``MSIParametricMiCSLMC.step``).

Optional ``gnn_input: embedding_plus_pca`` concatenates tissue PCA of raw MSI spectra
alongside the MiCS embedding so the GNN sees both visualization coordinates and
compressed high-dimensional mass-spec features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from msi_visual.parametric_mics_lmc import (
    MSIParametricMiCSLMC,
    _spatial_gaussian_smooth_channels,
    correlation,
)
from msi_visual.parametric_mics_unet import (
    _luminance_rgb01,
    _sobel_mag_gray,
    soft_dice_continuous,
)
from msi_visual.utils import normalize

logger = logging.getLogger(__name__)


def _lab01_opencv_to_rgb01(lab: torch.Tensor) -> torch.Tensor:
    """Differentiable OpenCV-style uint8 LAB/255 -> RGB/255 (D65)."""
    L = lab[..., 0] * 255.0
    a = lab[..., 1] * 255.0 - 128.0
    b = lab[..., 2] * 255.0 - 128.0
    L = L * (100.0 / 255.0)

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f_inv(t: torch.Tensor) -> torch.Tensor:
        t3 = t ** 3
        return torch.where(t3 > eps, t3, (116.0 * t - 16.0) / kappa)

    X = 0.95047 * f_inv(fx)
    Y = 1.00000 * f_inv(fy)
    Z = 1.08883 * f_inv(fz)

    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_rgb = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    rgb = torch.stack([r, g, b_rgb], dim=-1)

    def gamma(c: torch.Tensor) -> torch.Tensor:
        return torch.where(
            c > 0.0031308,
            1.055 * (c.clamp(min=0.0) ** (1.0 / 2.4)) - 0.055,
            12.92 * c,
        )

    return gamma(rgb).clamp(0.0, 1.0)


def _percentile_norm_map(
    mag: torch.Tensor,
    mask: torch.Tensor,
    *,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> torch.Tensor:
    """Percentile stretch of (1,1,H,W) edge map on tissue mask (matches eval spatial_norm)."""
    m = mask.squeeze() > 0.5
    vals = mag.squeeze()[m]
    if vals.numel() < 2:
        return mag
    q_lo = torch.quantile(vals.float(), low_pct / 100.0)
    q_hi = torch.quantile(vals.float(), high_pct / 100.0)
    return ((mag - q_lo) / (q_hi - q_lo + 1e-8)).clamp(0.0, 1.0) * mask.float()


def _max_channel_sobel(rgb_nchw: torch.Tensor) -> torch.Tensor:
    """Max Sobel magnitude over RGB channels (closer to eval LAB-max gradient edges)."""
    mags = [_sobel_mag_gray(rgb_nchw[:, i : i + 1]) for i in range(min(3, rgb_nchw.shape[1]))]
    return torch.max(torch.stack(mags, dim=0), dim=0).values


def _grid_4neighbor_mean(h_hwc: torch.Tensor, valid_hw: torch.Tensor) -> torch.Tensor:
    m = valid_hw.to(dtype=h_hwc.dtype).unsqueeze(-1)
    xv = h_hwc * m
    acc = torch.zeros_like(xv)
    cnt = torch.zeros(
        h_hwc.shape[0], h_hwc.shape[1], 1, device=h_hwc.device, dtype=h_hwc.dtype
    )
    h, w = int(h_hwc.shape[0]), int(h_hwc.shape[1])
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        sy0, sy1 = max(0, -dy), h - max(0, dy)
        sx0, sx1 = max(0, -dx), w - max(0, dx)
        dy0, dy1 = max(0, dy), h - max(0, -dy)
        dx0, dx1 = max(0, dx), w - max(0, -dx)
        acc[sy0:sy1, sx0:sx1] += xv[dy0:dy1, dx0:dx1]
        cnt[sy0:sy1, sx0:sx1] += m[dy0:dy1, dx0:dx1]
    return acc / cnt.clamp(min=1.0)


def _grid_radius_neighbor_mean(
    h_hwc: torch.Tensor,
    valid_hw: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Masked mean of valid neighbors within Chebyshev radius ``r`` (window ``(2r+1)²``).

    ``radius=1`` keeps legacy 4-neighbor aggregation. ``radius>=2`` uses a square
    window excluding the center pixel (GraphSAGE-style neighbor pool).
    """
    r = max(1, int(radius))
    if r == 1:
        return _grid_4neighbor_mean(h_hwc, valid_hw)

    device, dtype = h_hwc.device, h_hwc.dtype
    k = 2 * r + 1
    n_ch = int(h_hwc.shape[-1])
    x = h_hwc.permute(2, 0, 1).unsqueeze(0)
    m = valid_hw.to(dtype=dtype).view(1, 1, int(h_hwc.shape[0]), int(h_hwc.shape[1]))
    kernel = torch.ones(1, 1, k, k, device=device, dtype=dtype)
    kernel[0, 0, r, r] = 0.0
    ch_kernel = kernel.expand(n_ch, 1, k, k)
    num = F.conv2d(x * m, ch_kernel, padding=r, groups=n_ch)
    den = F.conv2d(m, kernel, padding=r)
    out = num / den.clamp(min=1e-8)
    return out.squeeze(0).permute(1, 2, 0)


class GridResidualGNN(nn.Module):
    """GraphSAGE on H×W grid: refines a frozen MiCS embedding with a residual Δ."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 1,
        neighbor_radius: int = 1,
    ):
        super().__init__()
        self.n_layers = max(1, int(n_layers))
        self.neighbor_radius = max(1, int(neighbor_radius))
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(int(in_dim), self.hidden_dim)
        self.sage_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim * 2, self.hidden_dim) for _ in range(self.n_layers)]
        )
        self.delta_proj = nn.Linear(self.hidden_dim, int(out_dim))
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward_delta(self, base_hwc: torch.Tensor, valid_hw: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.input_proj(base_hwc))
        for layer in self.sage_layers:
            neigh = _grid_radius_neighbor_mean(h, valid_hw, self.neighbor_radius)
            h = F.gelu(layer(torch.cat([h, neigh], dim=-1)))
        return self.delta_proj(h)


class MSIMiCSGNNRefine:
    """
    Stage-2 fine-tuner: ``e = e_mics(frozen) + GNN(e_mics, neighbors)``.
    """

    def __init__(
        self,
        mics: MSIParametricMiCSLMC,
        *,
        num_epochs: int = 40,
        lr: float = 0.05,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 1,
        gnn_neighbor_radius: int = 1,
        gnn_input: str = "embedding",
        gnn_pca_dims: int = 32,
        gnn_pca_fit_step: int = 16,
        dice_loss_weight: float = 0.01,
        dice_patch_size: int = 16,
        dice_patches_per_step: int = 32,
        dice_patch_sampling: str = "uniform",
        dice_edge_uniform_mix: float = 0.1,
        dice_edge_power: float = 2.0,
        dice_global_weight: float = 0.5,
        dice_patch_weight: float = 0.5,
        dice_min_valid_fraction: float = 0.25,
        mics_loss_weight: float = 1.0,
        delta_scale: float = 1.0,
        delta_l2_weight: float = 0.0,
        delta_highpass: bool = True,
        delta_max_rel: float = 0.0,
        delta_space: str = "embedding",
        contrast_loss_weight: float = 0.0,
        edge_mag_loss_weight: float = 0.0,
        edge_mag_hd_power: float = 2.0,
        edge_corr_loss_weight: float = 0.0,
        gnn_hd_edge_channel: bool = False,
        edge_delta_gate_strength: float = 0.0,
        edge_delta_gate_power: float = 1.5,
        edge_loss_ramp_epochs: int = 0,
        dice_edge_color_space: str = "rgb_max",
        dice_edge_percentile_norm: bool = True,
        non_edge_l1_weight: float = 0.0,
        batch_size: int = 1024,
        verbose: bool = False,
        predict_percentile_low: float = 1.0,
        predict_percentile_high: float = 99.0,
        predict_spatial_smooth_sigma: float = 0.0,
        predict_anchor_base_percentiles: bool = True,
        lab_to_rgb: bool = True,
    ):
        self.mics = mics
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.gnn_hidden_dim = int(gnn_hidden_dim)
        self.gnn_layers = max(1, int(gnn_layers))
        self.gnn_neighbor_radius = max(1, int(gnn_neighbor_radius))
        self.gnn_input = str(gnn_input).lower().strip() or "embedding"
        self.gnn_pca_dims = max(1, int(gnn_pca_dims))
        self.gnn_pca_fit_step = max(1, int(gnn_pca_fit_step))
        self.dice_loss_weight = float(dice_loss_weight)
        self.dice_patch_size = max(4, int(dice_patch_size))
        self.dice_patches_per_step = max(1, int(dice_patches_per_step))
        self.dice_patch_sampling = str(dice_patch_sampling).lower().strip() or "uniform"
        self.dice_edge_uniform_mix = float(np.clip(dice_edge_uniform_mix, 0.0, 1.0))
        self.dice_edge_power = max(0.1, float(dice_edge_power))
        gw = max(0.0, float(dice_global_weight))
        pw = max(0.0, float(dice_patch_weight))
        norm = gw + pw
        self.dice_global_weight = gw / norm if norm > 0.0 else 0.5
        self.dice_patch_weight = pw / norm if norm > 0.0 else 0.5
        self.dice_min_valid_fraction = float(dice_min_valid_fraction)
        self.mics_loss_weight = float(mics_loss_weight)
        self.delta_scale = float(delta_scale)
        self.delta_l2_weight = max(0.0, float(delta_l2_weight))
        self.delta_highpass = bool(delta_highpass)
        self.delta_max_rel = max(0.0, float(delta_max_rel))
        self.delta_space = str(delta_space).lower().strip() or "embedding"
        self.contrast_loss_weight = max(0.0, float(contrast_loss_weight))
        self.edge_mag_loss_weight = max(0.0, float(edge_mag_loss_weight))
        self.edge_mag_hd_power = max(0.1, float(edge_mag_hd_power))
        self.edge_corr_loss_weight = max(0.0, float(edge_corr_loss_weight))
        self.gnn_hd_edge_channel = bool(gnn_hd_edge_channel)
        self.edge_delta_gate_strength = max(0.0, float(edge_delta_gate_strength))
        self.edge_delta_gate_power = max(0.1, float(edge_delta_gate_power))
        self.edge_loss_ramp_epochs = max(0, int(edge_loss_ramp_epochs))
        self.dice_edge_color_space = str(dice_edge_color_space).lower().strip() or "rgb_max"
        self.dice_edge_percentile_norm = bool(dice_edge_percentile_norm)
        self.non_edge_l1_weight = max(0.0, float(non_edge_l1_weight))
        self.batch_size = int(batch_size)
        self.step_counter = 0
        self.verbose = bool(verbose)
        self.predict_percentile_low = float(predict_percentile_low)
        self.predict_percentile_high = float(predict_percentile_high)
        self.predict_spatial_smooth_sigma = max(0.0, float(predict_spatial_smooth_sigma))
        self.predict_anchor_base_percentiles = bool(predict_anchor_base_percentiles)
        self.lab_to_rgb = bool(lab_to_rgb)

        self.gnn: GridResidualGNN | None = None
        self.optim: torch.optim.Optimizer | None = None
        self.img: np.ndarray | None = None
        self.valid_mask: np.ndarray | None = None
        self.mics_base_emb: np.ndarray | None = None
        self._base_t: torch.Tensor | None = None
        self._gnn_base_t: torch.Tensor | None = None
        self._gnn_input_t: torch.Tensor | None = None
        self._pca_spectrum_t: torch.Tensor | None = None
        self._hd_edges_t: torch.Tensor | None = None
        self._hd_edges_np: np.ndarray | None = None
        self._mask_hw_t: torch.Tensor | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._patch_rng = np.random.default_rng(int(getattr(mics, "random_state", 42)))
        self._last_loss_total = 0.0
        self._last_loss_dice = 0.0
        self._last_loss_mics = 0.0
        self._last_loss_delta_l2 = 0.0
        self._last_loss_contrast = 0.0
        self._last_loss_edge_mag = 0.0
        self._last_loss_edge_corr = 0.0
        self._last_loss_non_edge = 0.0
        self._gnn_in_dim: int | None = None
        self._current_epoch = 0
        self._hd_edge_feat_t: torch.Tensor | None = None
        self._viz_p_low: torch.Tensor | None = None
        self._viz_p_high: torch.Tensor | None = None

    def _freeze_mics(self) -> None:
        if self.mics.model is None:
            raise ValueError("MiCS model must be trained before GNN refinement.")
        self.mics.model.eval()
        for p in self.mics.model.parameters():
            p.requires_grad = False
        for layer in getattr(self.mics, "visualiation_to_cluster", []) or []:
            layer.eval()
            for p in layer.parameters():
                p.requires_grad = False
        cat_head = getattr(self.mics, "category_head", None)
        if cat_head is not None:
            cat_head.eval()
            for p in cat_head.parameters():
                p.requires_grad = False

    def _set_base_embedding(self, emb: np.ndarray) -> None:
        """Pin the frozen MiCS embedding used as GNN residual base (H,W,C)."""
        if self.img is None:
            raise ValueError("set_image before set_base_embedding")
        base = np.asarray(emb, dtype=np.float32)
        h, w = self.img.shape[:2]
        c = int(self.mics.number_of_components)
        if base.shape != (h, w, c):
            raise ValueError(f"base_embedding shape {base.shape} != expected {(h, w, c)}")
        self.mics_base_emb = base.copy()
        self._base_t = torch.from_numpy(self.mics_base_emb).float().to(self._device)
        self._cache_display_anchors(base)

    def _cache_display_anchors(self, base: np.ndarray) -> None:
        """Freeze percentile anchors from stage-1 embedding (matches predict() display)."""
        if self.valid_mask is None:
            raise ValueError("valid_mask required before display anchors")
        mask = np.asarray(self.valid_mask, dtype=bool)
        lows: list[float] = []
        highs: list[float] = []
        for c in range(int(base.shape[-1])):
            vals = np.asarray(base[:, :, c], dtype=np.float64)[mask]
            if vals.size == 0:
                vals = np.asarray(base[:, :, c], dtype=np.float64).ravel()
            lows.append(float(np.percentile(vals, self.predict_percentile_low)))
            highs.append(float(np.percentile(vals, self.predict_percentile_high)))
        self._viz_p_low = torch.tensor(lows, device=self._device, dtype=torch.float32)
        self._viz_p_high = torch.tensor(highs, device=self._device, dtype=torch.float32)

    def _uses_pca_input(self) -> bool:
        return self.gnn_input in ("embedding_plus_pca", "embedding_pca", "dual")

    def _refresh_gnn_base(self) -> None:
        """Residual base: MiCS embedding or frozen display-stretched [0,1] channels."""
        if self._base_t is None:
            self._gnn_base_t = None
            return
        if self.delta_space == "display":
            self._gnn_base_t = self._stretch_display_hwc(self._base_t).detach()
        else:
            self._gnn_base_t = self._base_t

    def _refresh_gnn_input(self) -> None:
        """Full GNN node features (embedding [+ optional PCA spectra])."""
        self._refresh_gnn_base()
        if self._gnn_base_t is None:
            self._gnn_input_t = None
            return
        parts: list[torch.Tensor] = [self._gnn_base_t]
        if self._uses_pca_input() and self._pca_spectrum_t is not None:
            parts.append(self._pca_spectrum_t)
        if self.gnn_hd_edge_channel and self._hd_edge_feat_t is not None:
            parts.append(self._hd_edge_feat_t)
        self._gnn_input_t = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _ensure_gnn(self) -> None:
        """Create or rebuild GNN when input feature dimension changes (e.g. HD channel added)."""
        if self._gnn_input_t is None:
            return
        in_dim = int(self._gnn_input_t.shape[-1])
        c = int(self.mics.number_of_components)
        if self.gnn is not None and self._gnn_in_dim == in_dim:
            return
        self.gnn = GridResidualGNN(
            in_dim=in_dim,
            hidden_dim=self.gnn_hidden_dim,
            out_dim=c,
            n_layers=self.gnn_layers,
            neighbor_radius=self.gnn_neighbor_radius,
        ).to(self._device)
        self.optim = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, weight_decay=0)
        self._gnn_in_dim = in_dim
        logger.info(
            "GNN input: %s | in_dim=%d (emb=%d, pca=%s, hd_ch=%s)",
            self.gnn_input,
            in_dim,
            c,
            self._pca_spectrum_t is not None,
            self.gnn_hd_edge_channel,
        )

    def _cache_pca_spectrum_features(self) -> None:
        """PCA of raw MSI on tissue; z-scored components stored as (H,W,k) node features."""
        from sklearn.decomposition import PCA

        if self.img is None or self.valid_mask is None:
            raise ValueError("set_image before PCA spectrum cache")
        img = np.asarray(self.img, dtype=np.float32)
        mask = np.asarray(self.valid_mask, dtype=bool)
        h, w, spec_dim = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
        n_valid = int(mask.sum())
        if n_valid < 2:
            raise ValueError("Need at least 2 valid pixels for PCA spectrum features.")
        k = min(self.gnn_pca_dims, spec_dim, n_valid)
        x_valid = img[mask].astype(np.float64, copy=False)
        step = self.gnn_pca_fit_step
        x_fit = x_valid[::step] if step > 1 else x_valid
        if x_fit.shape[0] < k:
            x_fit = x_valid
        rs = int(getattr(self.mics, "random_state", 42))
        pca = PCA(
            n_components=k,
            random_state=rs,
            svd_solver="randomized",
        )
        pca.fit(x_fit)
        z_valid = pca.transform(x_valid).astype(np.float32, copy=False)
        mu = z_valid.mean(axis=0)
        sd = z_valid.std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
        z_valid = (z_valid - mu) / sd
        hwk = np.zeros((h, w, k), dtype=np.float32)
        hwk[mask] = z_valid
        self._pca_spectrum_t = torch.from_numpy(hwk).float().to(self._device)
        if self.verbose:
            logger.info(
                "GNN PCA spectrum features: %d dims (fit n=%d, tissue n=%d, fit_step=%d)",
                k,
                int(x_fit.shape[0]),
                n_valid,
                step,
            )

    def _cache_mics_embedding(self, img: np.ndarray) -> None:
        self.mics.img = np.asarray(img, dtype=np.float32)
        with torch.no_grad():
            emb = self.mics.predict_embedding(img).astype(np.float32, copy=False)
        self._set_base_embedding(emb)

    def set_hd_edges(self, hd_edges: np.ndarray) -> None:
        he = np.squeeze(np.asarray(hd_edges, dtype=np.float32))
        if self.img is None:
            raise ValueError("set_image before set_hd_edges")
        h, w = self.img.shape[:2]
        if he.shape != (h, w):
            raise ValueError(f"hd_edges shape {he.shape} != MSI {(h, w)}")
        he = np.clip(he, 0.0, 1.0)
        self._hd_edges_np = np.asarray(he, dtype=np.float32)
        self._hd_edges_t = (
            torch.from_numpy(self._hd_edges_np).to(self._device).unsqueeze(0).unsqueeze(0)
        )
        self._hd_edge_feat_t = (
            torch.from_numpy(self._hd_edges_np).to(self._device).unsqueeze(-1)
        )
        mask = self.valid_mask.astype(np.float32)
        self._mask_hw_t = (
            torch.from_numpy(mask).to(self._device).unsqueeze(0).unsqueeze(0)
        )
        self._refresh_gnn_input()
        self._ensure_gnn()

    def set_image(
        self,
        img: np.ndarray,
        *,
        pixel_sampling_cache: dict[str, Any] | None = None,
        base_embedding: np.ndarray | None = None,
    ) -> None:
        self.img = np.asarray(img, dtype=np.float32)
        self.valid_mask = self.img.sum(axis=-1) > 0
        if not np.any(self.valid_mask):
            raise ValueError("Empty MSI valid mask.")
        self._freeze_mics()
        if getattr(self.mics, "sampled_flat_indices", None) is None:
            self.mics.set_image(img, pixel_sampling_cache=pixel_sampling_cache)
        if base_embedding is not None:
            self._set_base_embedding(base_embedding)
        else:
            self._cache_mics_embedding(img)
        if self._uses_pca_input():
            self._cache_pca_spectrum_features()
        else:
            self._pca_spectrum_t = None
        self._refresh_gnn_input()
        if not self.gnn_hd_edge_channel:
            self._ensure_gnn()

    def _forward_delta_tensor(self) -> torch.Tensor:
        """Scaled (optionally high-pass, capped) residual on embedding or display space."""
        assert self.gnn is not None and self._gnn_input_t is not None and self.valid_mask is not None
        valid_t = torch.from_numpy(self.valid_mask).bool().to(self._device)
        delta = self.gnn.forward_delta(self._gnn_input_t, valid_t) * self.delta_scale
        if self.delta_highpass:
            delta = delta - _grid_radius_neighbor_mean(
                delta, valid_t, self.gnn_neighbor_radius
            )
        if self.delta_max_rel > 0.0:
            if self.delta_space == "display":
                cap = self.delta_max_rel
            else:
                base_std = self._base_t.detach().std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
                cap = self.delta_max_rel * base_std
            delta = delta.clamp(-cap, cap)
        if self.edge_delta_gate_strength > 0.0 and self._hd_edge_feat_t is not None:
            gate = 1.0 + self.edge_delta_gate_strength * self._hd_edge_feat_t.pow(
                self.edge_delta_gate_power
            )
            delta = delta * gate
        return delta * valid_t.unsqueeze(-1).to(dtype=delta.dtype)

    def _refined_tensor_from_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Composed refined map in the space used for edge losses (display or embedding)."""
        if self.delta_space == "display":
            assert self._gnn_base_t is not None
            return self._gnn_base_t + delta
        assert self._base_t is not None
        return self._base_t + delta

    def _edge_loss_ramp(self) -> float:
        if self.edge_loss_ramp_epochs <= 0:
            return 1.0
        return min(1.0, float(self._current_epoch + 1) / float(self.edge_loss_ramp_epochs))

    def _forward_embedding_tensor(self) -> torch.Tensor:
        assert self._gnn_base_t is not None
        return self._gnn_base_t + self._forward_delta_tensor()

    def _delta_to_embedding(self, delta: torch.Tensor) -> torch.Tensor:
        """Map display-space Δ back to embedding units (for saving / MiCS loss)."""
        if self.delta_space != "display":
            return delta
        if self._viz_p_low is None or self._viz_p_high is None:
            raise ValueError("display anchors missing")
        span = (self._viz_p_high - self._viz_p_low).view(1, 1, -1)
        return delta * span

    def _stretch_display_hwc(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        if self._viz_p_low is None or self._viz_p_high is None:
            raise ValueError("display anchors not cached; call set_base_embedding first")
        pl = self._viz_p_low.view(1, 1, -1)
        ph = self._viz_p_high.view(1, 1, -1)
        return ((emb_hwc - pl) / (ph - pl + 1e-8)).clamp(0.0, 1.0)

    def _viz_edge_mag(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        """Display-aligned viz edges (percentile stretch + LAB->RGB like predict)."""
        if self.delta_space == "display":
            stretched = emb_hwc.clamp(0.0, 1.0)
        else:
            stretched = self._stretch_display_hwc(emb_hwc)
        if self.lab_to_rgb and stretched.shape[-1] >= 3:
            rgb = _lab01_opencv_to_rgb01(stretched[..., :3])
        else:
            rgb = stretched[..., :3]
        feat = rgb.permute(2, 0, 1).unsqueeze(0)
        if self.dice_edge_color_space in ("luminance", "gray", "lum"):
            mag = _sobel_mag_gray(_luminance_rgb01(feat))
        else:
            mag = _max_channel_sobel(feat)
        if self.dice_edge_percentile_norm and self._mask_hw_t is not None:
            mag = _percentile_norm_map(
                mag,
                self._mask_hw_t,
                low_pct=self.predict_percentile_low,
                high_pct=self.predict_percentile_high,
            )
        return mag

    def _display_hwc_for_viz(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        if self.delta_space == "display":
            return emb_hwc.clamp(0.0, 1.0)
        return self._stretch_display_hwc(emb_hwc)

    def _display_to_rgb_nchw(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        stretched = self._display_hwc_for_viz(emb_hwc)
        if self.lab_to_rgb and stretched.shape[-1] >= 3:
            rgb = _lab01_opencv_to_rgb01(stretched[..., :3])
        else:
            rgb = stretched[..., :3]
        return rgb.permute(2, 0, 1).unsqueeze(0)

    def _base_display_hwc(self) -> torch.Tensor:
        if self.delta_space == "display":
            assert self._gnn_base_t is not None
            return self._gnn_base_t
        assert self._base_t is not None
        return self._stretch_display_hwc(self._base_t)

    def _non_edge_fidelity_loss(
        self, refined_hwc: torch.Tensor, base_hwc: torch.Tensor
    ) -> torch.Tensor:
        """Keep RGB close to MiCS off SoLaCE edges."""
        if self.non_edge_l1_weight <= 0.0 or self._hd_edge_feat_t is None:
            return refined_hwc.sum() * 0.0
        refined_rgb = self._display_to_rgb_nchw(refined_hwc)
        base_rgb = self._display_to_rgb_nchw(base_hwc).detach()
        w = (1.0 - self._hd_edge_feat_t.permute(2, 0, 1).unsqueeze(0)).clamp(0.0, 1.0)
        if self._mask_hw_t is not None:
            w = w * self._mask_hw_t
        diff = (refined_rgb - base_rgb).abs() * w
        return diff.sum() / w.sum().clamp(min=1.0)

    def _contrast_loss(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        """Push luminance contrast up on tissue (sharper boundaries in RGB)."""
        if self._mask_hw_t is None:
            return emb_hwc.sum() * 0.0
        if self.delta_space == "display":
            stretched = emb_hwc.clamp(0.0, 1.0)
        else:
            stretched = self._stretch_display_hwc(emb_hwc)
        if self.lab_to_rgb and stretched.shape[-1] >= 3:
            rgb = _lab01_opencv_to_rgb01(stretched[..., :3])
        else:
            rgb = stretched[..., :3]
        y_hw = _luminance_rgb01(rgb.permute(2, 0, 1).unsqueeze(0)).squeeze()
        m_hw = self._mask_hw_t.squeeze() > 0.5
        vals = y_hw[m_hw]
        if vals.numel() < 2:
            return emb_hwc.sum() * 0.0
        return -torch.std(vals)

    def _edge_mag_loss(self, mag_v: torch.Tensor) -> torch.Tensor:
        """HD-weighted MSE between percentile-normalized viz edges and HD reference."""
        if self._hd_edges_t is None or self._mask_hw_t is None:
            return mag_v.sum() * 0.0
        m = self._mask_hw_t.float()
        hd = self._hd_edges_t * m
        w = hd.pow(self.edge_mag_hd_power)
        diff = (mag_v - hd).pow(2) * w
        return diff.sum() / (w.sum() + 1e-8)

    def _edge_corr_loss(self, mag_v: torch.Tensor) -> torch.Tensor:
        """HD-weighted Pearson correlation (maximize alignment on boundary pixels)."""
        if self._hd_edges_t is None or self._mask_hw_t is None:
            return mag_v.sum() * 0.0
        m = self._mask_hw_t.squeeze() > 0.5
        p = mag_v.squeeze()[m].float()
        t = self._hd_edges_t.squeeze()[m].float()
        if p.numel() < 8:
            return mag_v.sum() * 0.0
        w = t.clamp(min=0.0).pow(self.edge_mag_hd_power)
        w = w / (w.sum() + 1e-8)
        pm = (p * w).sum()
        tm = (t * w).sum()
        pv = p - pm
        tv = t - tm
        num = (w * pv * tv).sum()
        den = torch.sqrt((w * pv * pv).sum() * (w * tv * tv).sum()).clamp(min=1e-8)
        return -(num / den)

    def _sample_patch_origins(self, ps: int, n_patches: int) -> list[tuple[int, int]]:
        """Top-left (y0, x0) for ``ps×ps`` patches; edge-stratified or uniform."""
        assert self.valid_mask is not None
        vm = np.asarray(self.valid_mask, dtype=bool)
        h, w = vm.shape[:2]
        y_max = max(0, h - ps)
        x_max = max(0, w - ps)
        min_frac = self.dice_min_valid_fraction
        max_tries = max(n_patches * 8, n_patches)
        origins: list[tuple[int, int]] = []

        use_edge = self.dice_patch_sampling in (
            "edge",
            "edge_stratified",
            "hd",
            "hd_edge",
        )
        hd = self._hd_edges_np
        if use_edge and hd is not None and hd.shape == (h, w):
            flat_vm = vm.ravel()
            valid_flat = np.flatnonzero(flat_vm)
            if valid_flat.size > 0:
                wts = np.power(
                    np.clip(hd.ravel()[valid_flat], 0.0, None),
                    self.dice_edge_power,
                    dtype=np.float64,
                )
                if not np.any(np.isfinite(wts)) or float(wts.sum()) <= 1e-12:
                    wts = np.ones(valid_flat.size, dtype=np.float64)
                mix = self.dice_edge_uniform_mix
                if mix > 0.0:
                    wts = (1.0 - mix) * wts + mix * (wts.mean() if wts.size else 1.0)
                wts = wts / (wts.sum() + 1e-12)
                for _ in range(max_tries):
                    center = int(self._patch_rng.choice(valid_flat, p=wts))
                    row, col = np.unravel_index(center, (h, w))
                    y0 = int(np.clip(int(row) - ps // 2, 0, y_max))
                    x0 = int(np.clip(int(col) - ps // 2, 0, x_max))
                    if float(vm[y0 : y0 + ps, x0 : x0 + ps].mean()) >= min_frac:
                        origins.append((y0, x0))
                    if len(origins) >= n_patches:
                        break

        while len(origins) < n_patches:
            for _ in range(max_tries):
                y0 = int(self._patch_rng.integers(0, y_max + 1))
                x0 = int(self._patch_rng.integers(0, x_max + 1))
                if float(vm[y0 : y0 + ps, x0 : x0 + ps].mean()) >= min_frac:
                    origins.append((y0, x0))
                    break
            else:
                break
            if len(origins) >= n_patches:
                break
        return origins[:n_patches]

    def _dice_loss(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        """Global + edge-focused patch soft Dice vs HD edges (negative dice to minimize)."""
        if self._hd_edges_t is None or self._mask_hw_t is None:
            return emb_hwc.sum() * 0.0
        return self._dice_loss_from_mag(self._viz_edge_mag(emb_hwc))

    def _dice_loss_from_mag(self, mag_v: torch.Tensor) -> torch.Tensor:
        """Global + edge-focused patch soft Dice from precomputed viz edge magnitude."""
        if self._hd_edges_t is None or self._mask_hw_t is None:
            return mag_v.sum() * 0.0

        global_dice = soft_dice_continuous(mag_v, self._hd_edges_t, self._mask_hw_t)
        loss = -global_dice

        h, w = int(mag_v.shape[2]), int(mag_v.shape[3])
        ps = min(self.dice_patch_size, h, w)
        patch_losses: list[torch.Tensor] = []
        for y0, x0 in self._sample_patch_origins(ps, self.dice_patches_per_step):
            pv = mag_v[:, :, y0 : y0 + ps, x0 : x0 + ps]
            pt = self._hd_edges_t[:, :, y0 : y0 + ps, x0 : x0 + ps]
            pm = self._mask_hw_t[:, :, y0 : y0 + ps, x0 : x0 + ps]
            patch_losses.append(-soft_dice_continuous(pv, pt, pm))

        if patch_losses:
            patch_term = torch.stack(patch_losses).mean()
            loss = self.dice_global_weight * loss + self.dice_patch_weight * patch_term
        return loss

    def _mics_training_loss(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        """Same cluster + correlation logic as ``MSIParametricMiCSLMC.step`` on refined embeddings."""
        if self.mics_loss_weight <= 0.0:
            return emb_hwc.sum() * 0.0
        mics = self.mics
        assert mics.sampled_flat_indices is not None
        assert mics.euclidean is not None
        assert mics.indices is not None

        flat_emb = emb_hwc.reshape(-1, emb_hwc.shape[-1])
        ref_flat = torch.from_numpy(mics.sampled_flat_indices[mics.indices]).long().to(self._device)
        ref_emb = flat_emb[ref_flat]

        n_samples = int(mics.sampled_data.shape[0])
        batch_size = self.batch_size
        order = np.random.permutation(n_samples)
        n_batches = max(1, (n_samples + batch_size - 1) // batch_size)
        batch_losses: list[torch.Tensor] = []

        for bi in range(n_batches):
            sl = order[bi * batch_size : (bi + 1) * batch_size]
            if sl.size == 0:
                continue
            batch_flat = torch.from_numpy(mics.sampled_flat_indices[sl]).long().to(self._device)
            batch_outputs = flat_emb[batch_flat]
            low_d = torch.cdist(batch_outputs, ref_emb)
            high_d = mics.euclidean[sl]

            loss = torch.zeros((), device=self._device, dtype=torch.float32)

            if mics.cluster and len(mics.visualiation_to_cluster) > 0:
                cluster_loss_sum = torch.zeros((), device=self._device, dtype=torch.float32)
                for i, layer in enumerate(mics.visualiation_to_cluster):
                    labels = mics.cluster_labels[i]
                    batch_clusters = torch.from_numpy(labels[sl]).long().to(self._device)
                    layer_output = layer(batch_outputs)
                    cluster_loss_sum = cluster_loss_sum + F.cross_entropy(
                        layer_output / mics.temperature,
                        batch_clusters,
                        ignore_index=-1,
                    )
                loss = loss + float(mics.cluster_loss_weight) * (
                    cluster_loss_sum / len(mics.visualiation_to_cluster)
                )

            if mics.category_head is not None and mics.category_labels is not None:
                batch_cats = torch.from_numpy(mics.category_labels[sl]).long().to(self._device)
                cat_output = mics.category_head(batch_outputs)
                loss = loss + float(mics.category_loss_weight) * F.cross_entropy(
                    cat_output / mics.temperature, batch_cats, ignore_index=-1
                )

            if self.step_counter % int(mics.k_epoch) == int(mics.k_epoch) - 1:
                correlation_loss = -correlation(low_d, high_d).mean()
                alpha = max(-float(correlation_loss.detach().cpu().numpy()), 1e-6)
                if mics.cluster and len(mics.visualiation_to_cluster) > 0:
                    loss = loss + correlation_loss * float(mics.beta) / alpha
                elif mics.category_head is not None:
                    loss = loss + correlation_loss * float(mics.beta) / alpha
                else:
                    loss = loss + correlation_loss

            if not getattr(loss, "requires_grad", False):
                loss = loss + batch_outputs.sum() * 0.0
            batch_losses.append(loss)

        if not batch_losses:
            return emb_hwc.sum() * 0.0
        return torch.stack(batch_losses).mean()

    def step(self) -> None:
        assert self.gnn is not None and self.optim is not None
        self.gnn.train()
        self.optim.zero_grad()
        delta = self._forward_delta_tensor()
        emb = self._refined_tensor_from_delta(delta)
        base_hwc = self._base_display_hwc()
        loss = torch.zeros((), device=self._device, dtype=torch.float32)
        edge_ramp = self._edge_loss_ramp()
        dice_term = emb.sum() * 0.0
        mics_term = emb.sum() * 0.0
        delta_l2_term = delta.sum() * 0.0
        contrast_term = emb.sum() * 0.0
        edge_mag_term = emb.sum() * 0.0
        edge_corr_term = emb.sum() * 0.0
        non_edge_term = emb.sum() * 0.0
        mag_v = None
        if (
            self.dice_loss_weight > 0.0
            or self.edge_mag_loss_weight > 0.0
            or self.edge_corr_loss_weight > 0.0
        ):
            mag_v = self._viz_edge_mag(emb)
        if self.dice_loss_weight > 0.0:
            dice_term = self._dice_loss_from_mag(mag_v)
            loss = loss + edge_ramp * self.dice_loss_weight * dice_term
        if self.edge_mag_loss_weight > 0.0 and mag_v is not None:
            edge_mag_term = self._edge_mag_loss(mag_v)
            loss = loss + edge_ramp * self.edge_mag_loss_weight * edge_mag_term
        if self.edge_corr_loss_weight > 0.0 and mag_v is not None:
            edge_corr_term = self._edge_corr_loss(mag_v)
            loss = loss + edge_ramp * self.edge_corr_loss_weight * edge_corr_term
        if self.contrast_loss_weight > 0.0:
            contrast_term = self._contrast_loss(emb)
            loss = loss + edge_ramp * self.contrast_loss_weight * contrast_term
        if self.mics_loss_weight > 0.0:
            mics_emb = emb
            if self.delta_space == "display" and self._base_t is not None:
                mics_emb = self._base_t + self._delta_to_embedding(delta)
            mics_term = self._mics_training_loss(mics_emb)
            # Scale MiCS to O(1) so it cannot drown SoLaCE after grad clip.
            mics_scale = mics_term.detach().abs().clamp(min=1e-3)
            loss = loss + self.mics_loss_weight * (mics_term / mics_scale)
        if self.delta_l2_weight > 0.0:
            if self.delta_space == "display":
                base_var = self._gnn_base_t.detach().pow(2).mean().clamp(min=1e-8)
            else:
                base_var = self._base_t.detach().pow(2).mean().clamp(min=1e-8)
            delta_l2_term = delta.pow(2).mean() / base_var
            loss = loss + self.delta_l2_weight * delta_l2_term
        if self.non_edge_l1_weight > 0.0:
            non_edge_term = self._non_edge_fidelity_loss(emb, base_hwc)
            loss = loss + self.non_edge_l1_weight * non_edge_term
        if not loss.requires_grad:
            loss = loss + emb.sum() * 0.0
        self._last_loss_dice = float(dice_term.detach().cpu().item())
        self._last_loss_mics = float(mics_term.detach().cpu().item())
        self._last_loss_delta_l2 = float(delta_l2_term.detach().cpu().item())
        self._last_loss_contrast = float(contrast_term.detach().cpu().item())
        self._last_loss_edge_mag = float(edge_mag_term.detach().cpu().item())
        self._last_loss_edge_corr = float(edge_corr_term.detach().cpu().item())
        self._last_loss_non_edge = float(non_edge_term.detach().cpu().item())
        self._last_loss_total = float(loss.detach().cpu().item())
        loss.backward()
        nn.utils.clip_grad_norm_(self.gnn.parameters(), max_norm=1.0)
        self.optim.step()

        if self.step_counter % int(self.mics.k_epoch) == int(self.mics.k_epoch) - 1:
            self.mics.k_epoch = max(1, int(self.mics.k_epoch * 0.92))
        self.step_counter += 1

    def fit(
        self,
        img: np.ndarray,
        *,
        hd_edges: np.ndarray,
        pixel_sampling_cache: dict[str, Any] | None = None,
        base_embedding: np.ndarray | None = None,
    ) -> MSIMiCSGNNRefine:
        self.set_image(
            img,
            pixel_sampling_cache=pixel_sampling_cache,
            base_embedding=base_embedding,
        )
        self.set_hd_edges(hd_edges)
        for epoch in tqdm.tqdm(range(self.num_epochs), desc="MiCS→GNN refine"):
            self._current_epoch = epoch
            self.step()
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "GNN refine epoch %d/%d | loss=%.4f dice=%.4f mics=%.4f",
                    epoch + 1,
                    self.num_epochs,
                    self._last_loss_total,
                    self._last_loss_dice,
                    self._last_loss_mics,
                )
        logger.info(
            "GNN refine final loss=%.4f (dice=%.4f, edge_mag=%.4f, edge_corr=%.4f, "
            "contrast=%.4f, mics=%.4f, delta_l2=%.4f | input=%s delta_space=%s "
            "layers=%d radius=%d delta_scale=%.4g hd_ch=%s gate=%.2f)",
            self._last_loss_total,
            self._last_loss_dice,
            self._last_loss_edge_mag,
            self._last_loss_edge_corr,
            self._last_loss_contrast,
            self._last_loss_mics,
            self._last_loss_delta_l2,
            self.gnn_input,
            self.delta_space,
            self.gnn_layers,
            self.gnn_neighbor_radius,
            self.delta_scale,
            self.gnn_hd_edge_channel,
            self.edge_delta_gate_strength,
        )
        return self

    def predict_delta(self) -> np.ndarray:
        assert self.gnn is not None and self._gnn_base_t is not None and self.valid_mask is not None
        self.gnn.eval()
        with torch.no_grad():
            delta_t = self._forward_delta_tensor()
            if self.delta_space == "display":
                delta_t = self._delta_to_embedding(delta_t)
            delta = delta_t.cpu().numpy().astype(np.float32)
        return delta

    def predict_embedding(self) -> np.ndarray:
        assert self.mics_base_emb is not None
        delta = self.predict_delta()
        out = self.mics_base_emb + delta
        if self.valid_mask is not None:
            out[~self.valid_mask] = 0.0
        return out.astype(np.float32, copy=False)

    def _stretch_embedding_with_base_anchors(self, emb_hwc: np.ndarray) -> np.ndarray:
        """Stretch channels with frozen stage-1 percentile anchors (Δ visible in RGB)."""
        if self._viz_p_low is None or self._viz_p_high is None:
            raise ValueError("display anchors missing; call set_base_embedding first")
        out = np.asarray(emb_hwc, dtype=np.float32).copy()
        pl = self._viz_p_low.detach().cpu().numpy()
        ph = self._viz_p_high.detach().cpu().numpy()
        for c in range(out.shape[-1]):
            ch = out[:, :, c]
            ch = (ch - float(pl[c])) / (float(ph[c]) - float(pl[c]) + 1e-8)
            out[:, :, c] = np.clip(ch, 0.0, 1.0)
        return out

    def predict(self, img: np.ndarray | None = None, *, viz_delta_gain: float = 1.0) -> np.ndarray:
        """RGB uint8 viz. ``viz_delta_gain`` scales Δ for preview only (1.0 = trained residual)."""
        if img is not None and (self.img is None or img.shape != self.img.shape):
            raise ValueError("GNN refine expects the same MSI slice used in fit().")
        gain = max(0.0, float(viz_delta_gain))
        mask = self.valid_mask
        assert self.gnn is not None and self._base_t is not None
        self.gnn.eval()
        with torch.no_grad():
            delta = self._forward_delta_tensor()
            if gain != 1.0:
                delta = delta * gain
            # Smooth Δ only — never blur the frozen MiCS base.
            if self.predict_spatial_smooth_sigma > 0.0:
                delta_np = delta.detach().cpu().numpy().astype(np.float32)
                delta_np = _spatial_gaussian_smooth_channels(
                    delta_np, self.predict_spatial_smooth_sigma
                )
                delta = torch.from_numpy(delta_np).to(device=delta.device, dtype=delta.dtype)
            if self.delta_space == "display":
                assert self._gnn_base_t is not None
                refined = self._gnn_base_t + delta
                result = refined.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
                n_ch = int(result.shape[-1])
            else:
                emb = (self._base_t + delta).cpu().numpy().astype(np.float32)
                n_ch = int(emb.shape[-1])
                mask = self.valid_mask if self.valid_mask is not None else emb[..., 0] != 0
                if self.predict_anchor_base_percentiles:
                    result = self._stretch_embedding_with_base_anchors(emb)
                else:
                    result = normalize(
                        emb,
                        low=self.predict_percentile_low,
                        high=self.predict_percentile_high,
                        mask=mask,
                    )
        if self.delta_space == "display":
            result = np.clip(result, 0.0, 1.0)
        if mask is None:
            mask = result[..., 0] != 0
        out = np.uint8(255.0 * result)
        out[~mask] = 0
        if self.lab_to_rgb and n_ch == 3:
            import cv2

            out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
        out[~mask] = 0
        return out

    def release_resources(self) -> None:
        self.gnn = None
        self.optim = None
        self._base_t = None
        self._gnn_base_t = None
        self._gnn_input_t = None
        self._pca_spectrum_t = None
        self._hd_edge_feat_t = None
        self._gnn_in_dim = None
        self._hd_edges_t = None
        self._hd_edges_np = None
        self._mask_hw_t = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
