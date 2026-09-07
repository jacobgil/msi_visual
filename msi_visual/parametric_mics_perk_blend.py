"""
Locally weighted blend of per-k MiCS embeddings (separate models per cluster scale).

Each pixel gets softmax weights over level embeddings; only the mix head is trained
to maximize soft continuous Dice vs HD edge target (optional preserve on coarse display).
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
    _lab01_opencv_to_rgb01,
    _max_channel_sobel,
    _percentile_norm_map,
)
from msi_visual.parametric_mics_lmc import (
    MSIParametricMiCSLMC,
    _spatial_gaussian_smooth_channels,
)
from msi_visual.parametric_mics_unet import (
    _luminance_rgb01,
    _sobel_mag_gray,
    soft_dice_continuous,
)

logger = logging.getLogger(__name__)


class MSIMiCSPerKBlend:
    """Frozen per-k embeddings + learned spatial mix weights → blended MiCS RGB."""

    def __init__(
        self,
        display_ref: MSIParametricMiCSLMC,
        level_ks: list[int],
        *,
        num_epochs: int = 40,
        lr: float = 0.02,
        mix_temperature: float = 1.0,
        mix_spatial_kernel: int = 1,
        blend_mode: str = "conv",
        local_edge_window: int = 15,
        local_edge_temperature: float = 0.08,
        hd_edge_power: float = 1.0,
        dice_loss_weight: float = 1.0,
        dice_global_weight: float = 1.0,
        dice_patch_weight: float = 0.5,
        dice_patches_per_step: int = 8,
        dice_patch_size: int = 48,
        mix_input: str = "emb_edge",
        mix_init_level_idx: int = -1,
        preserve_weight: float = 0.0,
        preserve_level_idx: int = 0,
        edge_loss_ramp_epochs: int = 8,
        dice_edge_color_space: str = "rgb_max",
        dice_edge_percentile_norm: bool = True,
        verbose: bool = False,
        predict_percentile_low: float | None = None,
        predict_percentile_high: float | None = None,
        predict_spatial_smooth_sigma: float = 0.0,
        lab_to_rgb: bool | None = None,
    ):
        self.display_ref = display_ref
        self.level_ks = [int(k) for k in level_ks]
        self.n_levels = len(self.level_ks)
        if self.n_levels < 2:
            raise ValueError("Per-k blend requires at least two level embeddings.")
        self.num_epochs = max(0, int(num_epochs))
        self.lr = float(lr)
        self.mix_temperature = max(1e-3, float(mix_temperature))
        self.blend_mode = str(blend_mode).lower().strip() or "conv"
        self.local_edge_window = max(1, int(local_edge_window))
        self.local_edge_temperature = max(1e-4, float(local_edge_temperature))
        self.hd_edge_power = max(0.1, float(hd_edge_power))
        self.dice_loss_weight = max(0.0, float(dice_loss_weight))
        self.dice_global_weight = max(0.0, float(dice_global_weight))
        self.dice_patch_weight = max(0.0, float(dice_patch_weight))
        self.dice_patches_per_step = max(0, int(dice_patches_per_step))
        self.dice_patch_size = max(8, int(dice_patch_size))
        self.mix_input = str(mix_input).lower().strip() or "emb_edge"
        self.mix_init_level_idx = int(mix_init_level_idx)
        self.preserve_weight = max(0.0, float(preserve_weight))
        self.preserve_level_idx = int(preserve_level_idx)
        self.edge_loss_ramp_epochs = max(0, int(edge_loss_ramp_epochs))
        self.dice_edge_color_space = str(dice_edge_color_space).lower().strip() or "rgb_max"
        self.dice_edge_percentile_norm = bool(dice_edge_percentile_norm)
        self.verbose = bool(verbose)
        self.predict_percentile_low = float(
            predict_percentile_low
            if predict_percentile_low is not None
            else getattr(display_ref, "predict_percentile_low", 1.0)
        )
        self.predict_percentile_high = float(
            predict_percentile_high
            if predict_percentile_high is not None
            else getattr(display_ref, "predict_percentile_high", 99.0)
        )
        self.predict_spatial_smooth_sigma = max(0.0, float(predict_spatial_smooth_sigma))
        self.lab_to_rgb = (
            bool(lab_to_rgb)
            if lab_to_rgb is not None
            else bool(getattr(display_ref, "lab_to_rgb", True))
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_comp = int(display_ref.number_of_components)
        edge_ch = self.n_levels if self.mix_input in ("emb_edge", "edge", "emb+edge") else 0
        in_ch = n_comp * self.n_levels + edge_ch
        ksz = max(1, int(mix_spatial_kernel))
        if ksz == 1:
            self.mix_head: nn.Module = nn.Conv2d(in_ch, self.n_levels, kernel_size=1)
        else:
            pad = ksz // 2
            self.mix_head = nn.Sequential(
                nn.Conv2d(in_ch, max(8, self.n_levels * 4), kernel_size=ksz, padding=pad),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(8, self.n_levels * 4), self.n_levels, kernel_size=1),
            )
        with torch.no_grad():
            for p in self.mix_head.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.05)
                else:
                    p.zero_()
            self._init_mix_level_bias()
        self.mix_head.to(self._device)
        self.optim = torch.optim.AdamW(self.mix_head.parameters(), lr=self.lr, weight_decay=0)

        self.img: np.ndarray | None = None
        self.valid_mask: np.ndarray | None = None
        self._level_emb_t: torch.Tensor | None = None
        self._level_display_t: torch.Tensor | None = None
        self._level_edge_feat_t: torch.Tensor | None = None
        self._level_edge_mags: list[np.ndarray] | None = None
        self._weights_hw_l: torch.Tensor | None = None
        self._hd_edges_np: np.ndarray | None = None
        self._hd_edges_t: torch.Tensor | None = None
        self._mask_hw_t: torch.Tensor | None = None
        self._valid_mask_t: torch.Tensor | None = None
        self.step_counter = 0
        self._current_epoch = 0
        self._last_loss_total = 0.0
        self._last_dice = float("nan")
        self._rng = np.random.default_rng(0)

    def _init_mix_level_bias(self) -> None:
        li = int(self.mix_init_level_idx)
        if li < 0:
            li = self.n_levels + li
        li = int(np.clip(li, 0, self.n_levels - 1))
        bias_mod: nn.Module | None = self.mix_head
        if isinstance(self.mix_head, nn.Sequential):
            bias_mod = self.mix_head[-1]
        if isinstance(bias_mod, nn.Conv2d) and bias_mod.bias is not None:
            bias_mod.bias.fill_(-1.0)
            bias_mod.bias[li] = 2.0

    def _cache_level_edge_features(self) -> None:
        assert self._level_emb_t is not None
        mags: list[torch.Tensor] = []
        for li in range(self._level_emb_t.shape[0]):
            rgb01 = self._emb_hwc_to_rgb01_tensor(self._level_emb_t[li])
            mags.append(self._viz_edge_mag_rgb(rgb01.permute(2, 0, 1).unsqueeze(0)))
        self._level_edge_feat_t = torch.cat(mags, dim=1)

    def set_hd_edges(self, hd_edges: np.ndarray | None) -> None:
        if hd_edges is None:
            self._hd_edges_np = None
            self._hd_edges_t = None
            return
        he = np.squeeze(np.asarray(hd_edges, dtype=np.float32))
        if self.img is None:
            raise ValueError("set_image before set_hd_edges")
        h, w = self.img.shape[:2]
        if he.shape != (h, w):
            raise ValueError(f"hd_edges shape {he.shape} != {(h, w)}")
        self._hd_edges_np = np.clip(he, 0.0, 1.0)
        self._hd_edges_t = (
            torch.from_numpy(self._hd_edges_np).to(self._device).unsqueeze(0).unsqueeze(0)
        )

    def set_level_embeddings(
        self,
        img: np.ndarray,
        level_embeddings: list[np.ndarray],
    ) -> None:
        self.img = np.asarray(img, dtype=np.float32)
        self.valid_mask = self.img.sum(axis=-1) > 0
        if not np.any(self.valid_mask):
            raise ValueError("Empty MSI valid mask.")
        if len(level_embeddings) != self.n_levels:
            raise ValueError(
                f"Expected {self.n_levels} embeddings, got {len(level_embeddings)}"
            )
        h, w = self.img.shape[:2]
        c = int(self.display_ref.number_of_components)
        stacked: list[torch.Tensor] = []
        for i, emb in enumerate(level_embeddings):
            arr = np.asarray(emb, dtype=np.float32)
            if arr.shape != (h, w, c):
                raise ValueError(f"level {i} embedding shape {arr.shape} != {(h, w, c)}")
            stacked.append(torch.from_numpy(arr).float().to(self._device))
        self._level_emb_t = torch.stack(stacked, dim=0)
        mask = self.valid_mask.astype(np.float32)
        self._mask_hw_t = torch.from_numpy(mask).to(self._device).unsqueeze(0).unsqueeze(0)
        self._valid_mask_t = torch.from_numpy(self.valid_mask).to(self._device)
        displays: list[torch.Tensor] = []
        for emb in stacked:
            displays.append(self._embedding_to_display_tensor(emb))
        self._level_display_t = torch.stack(displays, dim=0)
        if self.mix_input in ("emb_edge", "edge", "emb+edge"):
            self._cache_level_edge_features()

    @staticmethod
    def _local_continuous_dice_map(
        hd_edge: np.ndarray,
        viz_edge: np.ndarray,
        valid_mask: np.ndarray,
        window: int,
    ) -> np.ndarray:
        import cv2

        k = max(1, int(window))
        m = np.asarray(valid_mask, dtype=bool)
        h = np.where(m, np.asarray(hd_edge, dtype=np.float32), 0.0)
        v = np.where(m, np.asarray(viz_edge, dtype=np.float32), 0.0)
        ksize = (k, k)
        sh = cv2.boxFilter(h, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
        sv = cv2.boxFilter(v, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
        shv = cv2.boxFilter(h * v, ddepth=cv2.CV_32F, ksize=ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
        denom = sh + sv + 1e-8
        out = (2.0 * shv / denom).astype(np.float32)
        out = np.where((sh + sv) > 1e-8, out, 0.0)
        out = np.where(m, out, 0.0)
        return np.clip(out, 0.0, 1.0)

    def _compute_level_edge_mags(self) -> list[np.ndarray]:
        assert self._level_display_t is not None and self.valid_mask is not None
        mags: list[np.ndarray] = []
        for li in range(self._level_display_t.shape[0]):
            disp = self._level_display_t[li]
            rgb01 = self._emb_hwc_to_rgb01_tensor(
                self._level_emb_t[li] if self._level_emb_t is not None else disp
            )
            mag_t = self._viz_edge_mag_rgb(rgb01.permute(2, 0, 1).unsqueeze(0))
            mags.append(mag_t.squeeze().detach().cpu().numpy().astype(np.float32))
        return mags

    def _fit_edge_local_weights(self) -> None:
        assert self._hd_edges_np is not None and self.valid_mask is not None
        if self._level_edge_mags is None:
            self._level_edge_mags = self._compute_level_edge_mags()
        hd = np.power(self._hd_edges_np, self.hd_edge_power).astype(np.float32)
        mask = np.asarray(self.valid_mask, dtype=bool)
        scores = np.stack(
            [
                self._local_continuous_dice_map(hd, e, mask, self.local_edge_window)
                for e in self._level_edge_mags
            ],
            axis=-1,
        )
        scores = np.where(mask[..., None], scores, -np.inf)
        temp = self.local_edge_temperature
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores / temp)
        exp_s = np.where(mask[..., None], exp_s, 0.0)
        w = exp_s / np.maximum(exp_s.sum(axis=-1, keepdims=True), 1e-8)
        self._weights_hw_l = torch.from_numpy(w.astype(np.float32)).to(self._device)
        logger.info(
            "Edge-local blend: window=%d temp=%.3g (no conv training)",
            self.local_edge_window,
            self.local_edge_temperature,
        )

    def _mix_features(self) -> torch.Tensor:
        assert self._level_emb_t is not None
        n_levels, h, w, c = self._level_emb_t.shape
        emb_feat = self._level_emb_t.permute(0, 3, 1, 2).reshape(1, n_levels * c, h, w)
        if self.mix_input in ("emb_edge", "edge", "emb+edge") and self._level_edge_feat_t is not None:
            return torch.cat([emb_feat, self._level_edge_feat_t], dim=1)
        return emb_feat

    def _mix_weights(self) -> torch.Tensor:
        if self._weights_hw_l is not None:
            return self._weights_hw_l.permute(2, 0, 1).unsqueeze(0)
        logits = self.mix_head(self._mix_features())
        return F.softmax(logits / self.mix_temperature, dim=1)

    def _mixed_display(self, weights: torch.Tensor) -> torch.Tensor:
        assert self._level_display_t is not None
        w = weights.squeeze(0).permute(1, 2, 0)
        disp = self._level_display_t.permute(1, 2, 0, 3)
        return (w.unsqueeze(-1) * disp).sum(dim=2)

    def _mixed_embedding(self, weights: torch.Tensor) -> torch.Tensor:
        assert self._level_emb_t is not None
        w = weights.squeeze(0).permute(1, 2, 0)
        emb = self._level_emb_t.permute(1, 2, 0, 3)
        return (w.unsqueeze(-1) * emb).sum(dim=2)

    def _embedding_to_display_tensor(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        assert self.valid_mask is not None
        emb = emb_hwc.detach().cpu().numpy().astype(np.float32)
        mask = np.asarray(self.valid_mask, dtype=bool)
        out = np.zeros_like(emb)
        for ch in range(emb.shape[-1]):
            vals = emb[:, :, ch][mask]
            if vals.size == 0:
                vals = emb[:, :, ch].ravel()
            lo = float(np.percentile(vals, self.predict_percentile_low))
            hi = float(np.percentile(vals, self.predict_percentile_high))
            out[:, :, ch] = np.clip((emb[:, :, ch] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        return torch.from_numpy(out).float().to(emb_hwc.device)

    def _percentile_stretch_channels(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        assert self._valid_mask_t is not None
        m = self._valid_mask_t
        out = emb_hwc.clone()
        lo_p = self.predict_percentile_low / 100.0
        hi_p = self.predict_percentile_high / 100.0
        for c in range(int(emb_hwc.shape[-1])):
            vals = emb_hwc[..., c][m]
            if vals.numel() == 0:
                vals = emb_hwc[..., c].reshape(-1)
            lo = torch.quantile(vals, lo_p)
            hi = torch.quantile(vals, hi_p)
            out[..., c] = (emb_hwc[..., c] - lo) / (hi - lo + 1e-8)
        return out.clamp(0.0, 1.0)

    def _emb_hwc_to_rgb01_tensor(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        stretched = self._percentile_stretch_channels(emb_hwc)
        if self.lab_to_rgb and stretched.shape[-1] >= 3:
            return _lab01_opencv_to_rgb01(stretched[..., :3])
        return stretched[..., :3]

    def _rgb_u8_from_weights(self, weights: torch.Tensor) -> np.ndarray:
        mixed_emb = self._mixed_embedding(weights).detach().cpu().numpy().astype(np.float32)
        return self._emb_hwc_to_rgb_u8(mixed_emb)

    def _viz_edge_mag_rgb(self, rgb01_bchw: torch.Tensor) -> torch.Tensor:
        feat = rgb01_bchw.clamp(0.0, 1.0)
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

    def _edge_loss_ramp(self) -> float:
        if self.edge_loss_ramp_epochs <= 0:
            return 1.0
        return min(1.0, float(self._current_epoch + 1) / float(self.edge_loss_ramp_epochs))

    def _sample_patch_origins(self, patch_size: int, n_patches: int) -> list[tuple[int, int]]:
        assert self.valid_mask is not None and self._hd_edges_np is not None
        h, w = self.valid_mask.shape
        ps = min(int(patch_size), h, w)
        vm = np.asarray(self.valid_mask, dtype=bool)
        hd = np.asarray(self._hd_edges_np, dtype=np.float32)
        y_max = max(0, h - ps)
        x_max = max(0, w - ps)
        origins: list[tuple[int, int]] = []
        if y_max == 0 and x_max == 0:
            return [(0, 0)]
        edge_thr = float(np.percentile(hd[vm], 70.0)) if np.any(vm) else 0.5
        for _ in range(max(1, n_patches) * 4):
            if len(origins) >= n_patches:
                break
            y0 = int(self._rng.integers(0, y_max + 1))
            x0 = int(self._rng.integers(0, x_max + 1))
            patch_m = vm[y0 : y0 + ps, x0 : x0 + ps]
            patch_e = hd[y0 : y0 + ps, x0 : x0 + ps]
            if float(patch_m.mean()) < 0.35:
                continue
            if float(patch_e[patch_m].mean()) < edge_thr:
                continue
            origins.append((y0, x0))
        while len(origins) < n_patches:
            origins.append(
                (int(self._rng.integers(0, y_max + 1)), int(self._rng.integers(0, x_max + 1)))
            )
        return origins[:n_patches]

    def _dice_loss_from_mag(self, mag_v: torch.Tensor) -> torch.Tensor:
        assert self._hd_edges_t is not None and self._mask_hw_t is not None
        global_dice = soft_dice_continuous(mag_v, self._hd_edges_t, self._mask_hw_t)
        self._last_dice = float(global_dice.detach().cpu().item())
        loss = -self.dice_global_weight * global_dice
        if self.dice_patch_weight > 0.0 and self.dice_patches_per_step > 0:
            h, w = int(mag_v.shape[2]), int(mag_v.shape[3])
            ps = min(self.dice_patch_size, h, w)
            patch_losses: list[torch.Tensor] = []
            for y0, x0 in self._sample_patch_origins(ps, self.dice_patches_per_step):
                pv = mag_v[:, :, y0 : y0 + ps, x0 : x0 + ps]
                pt = self._hd_edges_t[:, :, y0 : y0 + ps, x0 : x0 + ps]
                pm = self._mask_hw_t[:, :, y0 : y0 + ps, x0 : x0 + ps]
                patch_losses.append(-soft_dice_continuous(pv, pt, pm))
            if patch_losses:
                loss = loss + self.dice_patch_weight * torch.stack(patch_losses).mean()
        return loss

    def step(self) -> None:
        assert self._level_emb_t is not None
        self.optim.zero_grad(set_to_none=True)
        weights = self._mix_weights()
        mixed_emb = self._mixed_embedding(weights)
        loss = torch.zeros((), device=self._device, dtype=torch.float32)

        if self.preserve_weight > 0.0:
            anchor_emb = self._level_emb_t[int(self.preserve_level_idx)]
            loss = loss + self.preserve_weight * F.mse_loss(mixed_emb, anchor_emb)

        if self.dice_loss_weight > 0.0 and self._hd_edges_t is not None:
            rgb01 = self._emb_hwc_to_rgb01_tensor(mixed_emb)
            mag_v = self._viz_edge_mag_rgb(rgb01.permute(2, 0, 1).unsqueeze(0))
            loss = loss + self._edge_loss_ramp() * self.dice_loss_weight * self._dice_loss_from_mag(
                mag_v
            )

        if not loss.requires_grad:
            loss = loss + weights.sum() * 0.0
        self._last_loss_total = float(loss.detach().cpu().item())
        if not torch.isfinite(loss):
            logger.warning("Per-k blend step %d: non-finite loss", self.step_counter)
            self.step_counter += 1
            return
        loss.backward()
        nn.utils.clip_grad_norm_(self.mix_head.parameters(), max_norm=1.0)
        self.optim.step()
        self.step_counter += 1

    def fit(
        self,
        img: np.ndarray,
        level_embeddings: list[np.ndarray],
        *,
        hd_edges: np.ndarray | None = None,
        eval_dice_fn: Any | None = None,
        eval_refine: bool = True,
        eval_grid_steps: int = 41,
        eval_logit_steps: int = 120,
    ) -> MSIMiCSPerKBlend:
        self.set_level_embeddings(img, level_embeddings)
        self.set_hd_edges(hd_edges)
        if self.blend_mode in ("edge_local", "local_edge", "edge", "soft_edge"):
            if hd_edges is None:
                raise ValueError("edge_local blend requires hd_edges")
            self._fit_edge_local_weights()
            self._log_mix_stats()
            return self
        if self.num_epochs <= 0:
            logger.info("Per-k blend: uniform mix (levels=%s)", self.level_ks)
            self._log_mix_stats()
            return self
        logger.info(
            "Per-k blend: levels=%s epochs=%d dice_w=%.2g preserve_w=%.2g",
            self.level_ks,
            self.num_epochs,
            self.dice_loss_weight,
            self.preserve_weight,
        )
        for epoch in tqdm.tqdm(range(self.num_epochs), desc="MiCS per-k blend"):
            self._current_epoch = epoch
            self.step()
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "epoch %d/%d loss=%.4f dice=%.4f",
                    epoch + 1,
                    self.num_epochs,
                    self._last_loss_total,
                    self._last_dice,
                )
        logger.info("Per-k blend final loss=%.4f dice=%.4f", self._last_loss_total, self._last_dice)
        if eval_dice_fn is not None and eval_refine and self.blend_mode not in (
            "edge_local",
            "local_edge",
            "edge",
            "soft_edge",
        ):
            self.refine_with_eval_dice(
                eval_dice_fn,
                grid_steps=int(eval_grid_steps),
                logit_steps=int(eval_logit_steps),
            )
        self._log_mix_stats()
        return self

    def refine_with_eval_dice(
        self,
        eval_dice_fn: Any,
        *,
        grid_steps: int = 41,
        logit_steps: int = 120,
    ) -> float:
        """Post-train search aligned with gallery eval Dice (embedding blend + display)."""
        assert self._level_emb_t is not None and self.valid_mask is not None
        h, w = self.valid_mask.shape
        n = max(2, int(grid_steps))
        best_dice = float("-inf")
        best_w = np.full(self.n_levels, 1.0 / self.n_levels, dtype=np.float32)

        def _dice_for_w(w_vec: np.ndarray) -> float:
            wt = torch.from_numpy(w_vec.astype(np.float32)).to(self._device)
            whw = wt.view(1, 1, self.n_levels).expand(h, w, self.n_levels)
            weights = whw.permute(2, 0, 1).unsqueeze(0)
            rgb = self._rgb_u8_from_weights(weights)
            return float(eval_dice_fn(rgb))

        for li in range(self.n_levels):
            one = np.zeros(self.n_levels, dtype=np.float32)
            one[li] = 1.0
            d = _dice_for_w(one)
            if d > best_dice:
                best_dice, best_w = d, one.copy()
                logger.info("Eval refine baseline k=%d dice=%.4f", self.level_ks[li], d)

        for i in range(n):
            for j in range(n - i):
                if self.n_levels != 3:
                    break
                w_vec = np.array(
                    [i / (n - 1), j / (n - 1), (n - 1 - i - j) / (n - 1)],
                    dtype=np.float32,
                )
                d = _dice_for_w(w_vec)
                if d > best_dice:
                    best_dice, best_w = d, w_vec.copy()
        if self.n_levels == 2:
            for i in range(n):
                w_vec = np.array([i / (n - 1), (n - 1 - i) / (n - 1)], dtype=np.float32)
                d = _dice_for_w(w_vec)
                if d > best_dice:
                    best_dice, best_w = d, w_vec.copy()

        logits = np.log(np.maximum(best_w, 1e-6)).astype(np.float32)
        logits = logits - logits.mean()
        step_deltas = (0.25, 0.5, 1.0, -0.25, -0.5, -1.0)
        for _ in range(max(0, int(logit_steps))):
            improved = False
            for li in range(self.n_levels):
                for delta in step_deltas:
                    trial = logits.copy()
                    trial[li] += float(delta)
                    w_vec = np.exp(trial - np.max(trial))
                    w_vec = w_vec / np.maximum(w_vec.sum(), 1e-8)
                    d = _dice_for_w(w_vec)
                    if d > best_dice + 1e-6:
                        best_dice, best_w, logits = d, w_vec.copy(), trial
                        improved = True
            if not improved:
                break

        self.mix_head.eval()
        with torch.no_grad():
            conv_weights = self._mix_weights()
        conv_rgb = self._rgb_u8_from_weights(conv_weights)
        conv_dice = float(eval_dice_fn(conv_rgb))
        if conv_dice > best_dice:
            logger.info(
                "Eval refine keeping conv spatial dice=%.4f (global best=%.4f)",
                conv_dice,
                best_dice,
            )
            return conv_dice

        self._weights_hw_l = (
            torch.from_numpy(best_w.astype(np.float32))
            .to(self._device)
            .view(1, 1, self.n_levels)
            .expand(h, w, self.n_levels)
        )
        logger.info(
            "Eval refine best dice=%.4f weights (%s): %s",
            best_dice,
            ", ".join(f"k={k}" for k in self.level_ks),
            ", ".join(f"{x:.3f}" for x in best_w),
        )
        return best_dice

    def _log_mix_stats(self) -> None:
        if self.valid_mask is None:
            return
        w = self.predict_mix_weights()
        mask = np.asarray(self.valid_mask, dtype=bool)
        mean_w = w[mask].mean(axis=0) if w.ndim == 3 else w.mean(axis=0)
        logger.info(
            "Mean mix weights (%s): %s",
            ", ".join(f"k={k}" for k in self.level_ks),
            ", ".join(f"{x:.3f}" for x in mean_w),
        )

    def _emb_hwc_to_rgb_u8(self, emb_hwc: np.ndarray) -> np.ndarray:
        assert self.img is not None
        emb = _spatial_gaussian_smooth_channels(
            np.asarray(emb_hwc, dtype=np.float32),
            self.predict_spatial_smooth_sigma,
        )
        rgb = self.display_ref._embedding_to_display_rgb(emb, self.img)
        if self.valid_mask is not None:
            rgb = np.asarray(rgb, dtype=np.uint8)
            rgb[~self.valid_mask] = 0
        return np.asarray(rgb, dtype=np.uint8)[..., :3]

    def predict_mixed_rgb(self) -> np.ndarray:
        self.mix_head.eval()
        with torch.no_grad():
            weights = self._mix_weights()
        return self._rgb_u8_from_weights(weights)

    def predict_mix_weights(self) -> np.ndarray:
        self.mix_head.eval()
        with torch.no_grad():
            w = self._mix_weights().squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        if self.valid_mask is not None:
            w[~self.valid_mask] = 0.0
        return w

    def release_resources(self) -> None:
        self.mix_head = nn.Identity()
        self.optim = None
        self._level_emb_t = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
