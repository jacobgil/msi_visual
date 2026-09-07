"""
Two-stage MiCS + U-Net spatial edge residual (style-preserving).

Stage 1 (caller): train ``MSIParametricMiCSLMC`` — cluster + correlation, no edge loss.
Stage 2: frozen MiCS embedding + trainable U-Net residual
``e = e_mics + gate(hd_edge) * Δ_unet([base, hd_edge, optional PCA])`` with edge Dice/corr,
non-edge fidelity, and optional patch adversary so edits stay MiCS-like.

Compared to ``MSIMiCSGNNRefine``: U-Net sees full-image context (better for long SoLaCE
contours); patch discriminator explicitly preserves visualization style.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from msi_visual.parametric_mics_gnn import (
    MSIMiCSGNNRefine,
    _grid_radius_neighbor_mean,
    _lab01_opencv_to_rgb01,
)
from msi_visual.parametric_mics_unet import SmallUNet, forward_unet_full_resolution

logger = logging.getLogger(__name__)


class PatchDiscriminator(nn.Module):
    """Patch-GAN style discriminator on RGB patches (MiCS vs refined)."""

    def __init__(self, in_channels: int = 3, base: int = 32) -> None:
        super().__init__()
        b = max(8, int(base))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b, b * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b * 2, b * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b * 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MSIMiCSUnetEdgeRefine(MSIMiCSGNNRefine):
    """
    ``MSIMiCSGNNRefine`` with U-Net delta field + optional patch adversary.

    Reuses edge Dice/corr/mag, delta gating, and display anchoring from the GNN refine path.
    """

    def __init__(
        self,
        mics,
        *,
        unet_base_channels: int = 32,
        adv_loss_weight: float = 0.08,
        adv_patch_size: int = 32,
        adv_patches_per_step: int = 12,
        adv_d_lr: float | None = None,
        non_edge_l1_weight: float = 0.35,
        edge_delta_mask_only: bool = True,
        **kwargs: Any,
    ):
        kwargs.setdefault("gnn_hd_edge_channel", True)
        kwargs.setdefault("gnn_input", "embedding_plus_pca")
        kwargs.setdefault("delta_highpass", False)
        kwargs.setdefault("delta_space", "display")
        kwargs.setdefault("mics_loss_weight", 0.0)
        super().__init__(mics, **kwargs)
        self.unet_base_channels = max(8, int(unet_base_channels))
        self.adv_loss_weight = max(0.0, float(adv_loss_weight))
        self.adv_patch_size = max(8, int(adv_patch_size))
        self.adv_patches_per_step = max(1, int(adv_patches_per_step))
        self.adv_d_lr = float(adv_d_lr if adv_d_lr is not None else self.lr * 0.5)
        self.non_edge_l1_weight = max(0.0, float(non_edge_l1_weight))
        self.edge_delta_mask_only = bool(edge_delta_mask_only)
        self.unet: nn.Module | None = None
        self.discriminator: PatchDiscriminator | None = None
        self.d_optim: torch.optim.Optimizer | None = None
        self._unet_in_dim: int | None = None
        self.gnn = None
        self._last_loss_adv = 0.0
        self._last_loss_non_edge = 0.0

    def _ensure_gnn(self) -> None:
        self._ensure_unet()

    def _ensure_unet(self) -> None:
        if self._gnn_input_t is None:
            return
        in_dim = int(self._gnn_input_t.shape[-1])
        out_dim = int(self.mics.number_of_components)
        if self.unet is not None and self._unet_in_dim == in_dim:
            return
        device = self._device
        self.unet = SmallUNet(in_dim, out_dim, base=self.unet_base_channels).to(device)
        with torch.no_grad():
            self.unet.out_conv.weight.zero_()
            self.unet.out_conv.bias.zero_()
        self.optim = torch.optim.AdamW(self.unet.parameters(), lr=self.lr, weight_decay=0)
        if self.adv_loss_weight > 0.0:
            self.discriminator = PatchDiscriminator(3, base=self.unet_base_channels).to(device)
            self.d_optim = torch.optim.AdamW(
                self.discriminator.parameters(), lr=self.adv_d_lr, weight_decay=0
            )
        else:
            self.discriminator = None
            self.d_optim = None
        self._unet_in_dim = in_dim
        logger.info(
            "U-Net refine: in_dim=%d out_dim=%d base=%d adv=%s pca=%s hd_ch=%s",
            in_dim,
            out_dim,
            self.unet_base_channels,
            self.adv_loss_weight > 0.0,
            self._pca_spectrum_t is not None,
            self.gnn_hd_edge_channel,
        )

    def _unet_input_nchw(self) -> torch.Tensor:
        assert self._gnn_input_t is not None
        return self._gnn_input_t.permute(2, 0, 1).unsqueeze(0)

    def _forward_delta_tensor(self) -> torch.Tensor:
        assert self.unet is not None and self._gnn_input_t is not None and self.valid_mask is not None
        valid_t = torch.from_numpy(self.valid_mask).bool().to(self._device)
        x = self._unet_input_nchw()
        delta_hwc = forward_unet_full_resolution(self.unet, x).squeeze(0).permute(1, 2, 0)
        delta = delta_hwc * self.delta_scale
        if self.delta_highpass:
            radius = max(1, int(getattr(self, "gnn_neighbor_radius", 3)))
            delta = delta - _grid_radius_neighbor_mean(delta, valid_t, radius)
        if self.delta_max_rel > 0.0:
            if self.delta_space == "display":
                cap = self.delta_max_rel
            else:
                base_std = self._base_t.detach().std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
                cap = self.delta_max_rel * base_std
            delta = delta.clamp(-cap, cap)
        if self.edge_delta_gate_strength > 0.0 and self._hd_edge_feat_t is not None:
            hd = self._hd_edge_feat_t.clamp(0.0, 1.0)
            if self.edge_delta_mask_only:
                # Zero Δ off edges; only strong HD pixels are edited.
                mask = (hd.pow(self.edge_delta_gate_power) * self.edge_delta_gate_strength).clamp(
                    0.0, 1.0
                )
                delta = delta * mask
            else:
                # Soft suppress off edges (keeps MiCS interiors); old 1+gate amplified everywhere.
                floor = float(np.clip(1.0 / (1.0 + self.edge_delta_gate_strength), 0.0, 0.5))
                gate = floor + (1.0 - floor) * hd.pow(self.edge_delta_gate_power)
                delta = delta * gate
        # Light blur on Δ only (never the MiCS base) to reduce checkerboard.
        d = delta.permute(2, 0, 1).unsqueeze(0)
        d = F.avg_pool2d(d, kernel_size=3, stride=1, padding=1)
        delta = d.squeeze(0).permute(1, 2, 0)
        return delta * valid_t.unsqueeze(-1).to(dtype=delta.dtype)

    def _refined_tensor_from_delta(self, delta: torch.Tensor) -> torch.Tensor:
        emb = super()._refined_tensor_from_delta(delta)
        if self.delta_space == "display":
            return emb.clamp(0.0, 1.0)
        return emb

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
        if self.non_edge_l1_weight <= 0.0 or self._hd_edge_feat_t is None:
            return refined_hwc.sum() * 0.0
        refined_rgb = self._display_to_rgb_nchw(refined_hwc)
        base_rgb = self._display_to_rgb_nchw(base_hwc).detach()
        w = (1.0 - self._hd_edge_feat_t.permute(2, 0, 1).unsqueeze(0)).clamp(0.0, 1.0)
        if self._mask_hw_t is not None:
            w = w * self._mask_hw_t
        diff = (refined_rgb - base_rgb).abs() * w
        return diff.sum() / w.sum().clamp(min=1.0)

    def _sample_adv_patch_origins(self, ps: int, n_patches: int) -> list[tuple[int, int]]:
        old = self.dice_patch_sampling
        try:
            self.dice_patch_sampling = "uniform"
            return self._sample_patch_origins(ps, n_patches)
        finally:
            self.dice_patch_sampling = old

    def _adversarial_loss(
        self, refined_hwc: torch.Tensor, base_hwc: torch.Tensor
    ) -> torch.Tensor:
        if self.adv_loss_weight <= 0.0 or self.discriminator is None:
            return refined_hwc.sum() * 0.0
        ps = min(self.adv_patch_size, int(refined_hwc.shape[0]), int(refined_hwc.shape[1]))
        origins = self._sample_adv_patch_origins(ps, self.adv_patches_per_step)
        if not origins:
            return refined_hwc.sum() * 0.0
        base_rgb = self._display_to_rgb_nchw(base_hwc)
        refined_rgb = self._display_to_rgb_nchw(refined_hwc)
        real_list: list[torch.Tensor] = []
        fake_list: list[torch.Tensor] = []
        for y0, x0 in origins:
            real_list.append(base_rgb[:, :, y0 : y0 + ps, x0 : x0 + ps])
            fake_list.append(refined_rgb[:, :, y0 : y0 + ps, x0 : x0 + ps])
        real = torch.cat(real_list, dim=0).detach()
        fake = torch.cat(fake_list, dim=0)
        d_real = self.discriminator(real)
        d_fake_for_d = self.discriminator(fake.detach())
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake_for_d).mean()
        assert self.d_optim is not None
        self.d_optim.zero_grad(set_to_none=True)
        d_loss.backward()
        self.d_optim.step()
        d_fake_for_g = self.discriminator(fake)
        return -d_fake_for_g.mean()

    def step(self) -> None:
        assert self.unet is not None and self.optim is not None
        self.unet.train()
        self.optim.zero_grad(set_to_none=True)
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
        adv_term = emb.sum() * 0.0
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
        if self.adv_loss_weight > 0.0:
            adv_term = self._adversarial_loss(emb, base_hwc)
            loss = loss + self.adv_loss_weight * adv_term
        if not loss.requires_grad:
            loss = loss + emb.sum() * 0.0
        self._last_loss_dice = float(dice_term.detach().cpu().item())
        self._last_loss_mics = float(mics_term.detach().cpu().item())
        self._last_loss_delta_l2 = float(delta_l2_term.detach().cpu().item())
        self._last_loss_contrast = float(contrast_term.detach().cpu().item())
        self._last_loss_edge_mag = float(edge_mag_term.detach().cpu().item())
        self._last_loss_edge_corr = float(edge_corr_term.detach().cpu().item())
        self._last_loss_non_edge = float(non_edge_term.detach().cpu().item())
        self._last_loss_adv = float(adv_term.detach().cpu().item())
        self._last_loss_total = float(loss.detach().cpu().item())
        if not torch.isfinite(loss):
            logger.warning(
                "U-Net refine step %d: non-finite loss (%.4g); skipping update",
                self.step_counter,
                self._last_loss_total,
            )
            self.optim.zero_grad(set_to_none=True)
            self.step_counter += 1
            return
        loss.backward()
        nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
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
    ) -> MSIMiCSUnetEdgeRefine:
        self.set_image(
            img,
            pixel_sampling_cache=pixel_sampling_cache,
            base_embedding=base_embedding,
        )
        self.set_hd_edges(hd_edges)
        for epoch in tqdm.tqdm(range(self.num_epochs), desc="MiCS→U-Net refine"):
            self._current_epoch = epoch
            self.step()
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "U-Net refine epoch %d/%d | loss=%.4f dice=%.4f adv=%.4f non_edge=%.4f",
                    epoch + 1,
                    self.num_epochs,
                    self._last_loss_total,
                    self._last_loss_dice,
                    self._last_loss_adv,
                    self._last_loss_non_edge,
                )
        logger.info(
            "U-Net refine final loss=%.4f (dice=%.4f, edge_corr=%.4f, adv=%.4f, "
            "non_edge=%.4f, delta_l2=%.4f | delta_space=%s adv_w=%.3g)",
            self._last_loss_total,
            self._last_loss_dice,
            self._last_loss_edge_corr,
            self._last_loss_adv,
            self._last_loss_non_edge,
            self._last_loss_delta_l2,
            self.delta_space,
            self.adv_loss_weight,
        )
        return self

    def predict_delta(self) -> np.ndarray:
        assert self.unet is not None and self._gnn_base_t is not None and self.valid_mask is not None
        self.unet.eval()
        with torch.no_grad():
            delta_t = self._forward_delta_tensor()
            if self.delta_space == "display":
                delta_t = self._delta_to_embedding(delta_t)
            delta = delta_t.cpu().numpy().astype(np.float32)
        return delta

    def predict(self, img: np.ndarray | None = None, *, viz_delta_gain: float = 1.0) -> np.ndarray:
        if img is not None and (self.img is None or img.shape != self.img.shape):
            raise ValueError("U-Net refine expects the same MSI slice used in fit().")
        gain = max(0.0, float(viz_delta_gain))
        mask = self.valid_mask
        assert self.unet is not None and self._base_t is not None
        self.unet.eval()
        with torch.no_grad():
            delta = self._forward_delta_tensor()
            if gain != 1.0:
                delta = delta * gain
            # Optional extra Gaussian on Δ only — MiCS base stays sharp.
            if self.predict_spatial_smooth_sigma > 0.0:
                from msi_visual.parametric_mics_lmc import _spatial_gaussian_smooth_channels

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
                from msi_visual.parametric_mics_lmc import normalize

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
        self.unet = None
        self.discriminator = None
        self.d_optim = None
        self._unet_in_dim = None
        super().release_resources()
