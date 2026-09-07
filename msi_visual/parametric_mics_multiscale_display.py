"""
Stage-2 multi-scale **continuous MiCS** displays.

Per cluster level L (k=8, 32, 128, …):
    field_L(x) = sum_k p_k(x) * centroid_k^L     (centroids in MiCS embedding space)
    emb_L = (1 - blend_L) * base_emb + blend_L * field_L
    RGB_L = MiCS percentile stretch + LAB→RGB on emb_L

Finer k uses a larger blend (more cluster geometry, sharper detail).
This is **not** palette segmentation (``predict_cluster_soft_rgb_u8``).
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


def _pca_init_linear(probs_valid: np.ndarray, n_out: int) -> np.ndarray:
    """Return (n_out, K) weight matrix mapping cluster probs → display dims."""
    x = np.asarray(probs_valid, dtype=np.float64)
    if x.shape[0] < max(n_out + 1, 8):
        return np.zeros((n_out, x.shape[1]), dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:n_out]
    # Scale so typical delta is O(1) in embedding units.
    std = np.std(x @ comps.T, axis=0)
    scale = np.where(std > 1e-8, 1.0 / std, 1.0)
    return (comps * scale[:, None]).astype(np.float32)


class MSIMiCSMultiDisplayRefine:
    """Frozen MiCS + cluster-conditioned continuous displays + optional mix refine."""

    def __init__(
        self,
        mics: MSIParametricMiCSLMC,
        *,
        num_epochs: int = 40,
        lr: float = 0.02,
        cluster_scale: float = 0.35,
        display_mode: str = "centroid_blend",
        level_blend: list[float] | None = None,
        init_display: str = "pca",
        mix_temperature: float = 1.0,
        mix_coarse_bias: float = 0.0,
        dice_loss_weight: float = 0.0,
        edge_corr_loss_weight: float = 0.0,
        edge_mag_loss_weight: float = 0.0,
        edge_mag_hd_power: float = 2.0,
        mics_preserve_weight: float = 0.85,
        level_delta_l2_weight: float = 0.02,
        non_edge_l1_weight: float = 0.0,
        edge_loss_ramp_epochs: int = 8,
        dice_edge_color_space: str = "rgb_max",
        dice_edge_percentile_norm: bool = True,
        edge_delta_gate_power: float = 2.0,
        verbose: bool = False,
        predict_percentile_low: float = 1.0,
        predict_percentile_high: float = 99.0,
        predict_spatial_smooth_sigma: float = 0.0,
        lab_to_rgb: bool = True,
    ):
        self.mics = mics
        self.num_epochs = max(0, int(num_epochs))
        self.lr = float(lr)
        self.cluster_scale = float(cluster_scale)
        self.display_mode = str(display_mode).lower().strip() or "centroid_blend"
        self.level_blend = [float(x) for x in level_blend] if level_blend else None
        self.init_display = str(init_display).lower().strip() or "pca"
        self.mix_temperature = max(1e-3, float(mix_temperature))
        self.mix_coarse_bias = float(mix_coarse_bias)
        self.dice_loss_weight = max(0.0, float(dice_loss_weight))
        self.edge_corr_loss_weight = max(0.0, float(edge_corr_loss_weight))
        self.edge_mag_loss_weight = max(0.0, float(edge_mag_loss_weight))
        self.edge_mag_hd_power = max(0.1, float(edge_mag_hd_power))
        self.mics_preserve_weight = max(0.0, float(mics_preserve_weight))
        self.level_delta_l2_weight = max(0.0, float(level_delta_l2_weight))
        self.non_edge_l1_weight = max(0.0, float(non_edge_l1_weight))
        self.edge_loss_ramp_epochs = max(0, int(edge_loss_ramp_epochs))
        self.dice_edge_color_space = str(dice_edge_color_space).lower().strip() or "rgb_max"
        self.dice_edge_percentile_norm = bool(dice_edge_percentile_norm)
        self.edge_delta_gate_power = max(0.1, float(edge_delta_gate_power))
        self.verbose = bool(verbose)
        self.predict_percentile_low = float(predict_percentile_low)
        self.predict_percentile_high = float(predict_percentile_high)
        self.predict_spatial_smooth_sigma = max(0.0, float(predict_spatial_smooth_sigma))
        self.lab_to_rgb = bool(lab_to_rgb)

        heads = getattr(mics, "visualiation_to_cluster", None) or []
        if not heads:
            raise ValueError("MiCS must have cluster heads (model.cluster=true).")
        self.n_levels = len(heads)
        clusters = getattr(mics, "clusters", None) or []
        self.level_ks = [int(clusters[i]) if i < len(clusters) else i for i in range(self.n_levels)]
        self.level_n_classes = [int(h[-1].out_features) for h in heads]

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_comp = int(mics.number_of_components)
        self.level_projs = nn.ModuleList(
            [nn.Linear(k, n_comp, bias=False) for k in self.level_n_classes]
        )
        self.mix_head = nn.Linear(n_comp, self.n_levels)
        with torch.no_grad():
            for proj in self.level_projs:
                proj.weight.zero_()
            self.mix_head.weight.zero_()
            if self.mix_head.bias is not None:
                self.mix_head.bias.zero_()
                if self.mix_coarse_bias != 0.0:
                    self.mix_head.bias[0] = self.mix_coarse_bias
        self.level_projs.to(self._device)
        self.mix_head.to(self._device)
        self.optim = torch.optim.AdamW(
            list(self.level_projs.parameters()) + list(self.mix_head.parameters()),
            lr=self.lr,
            weight_decay=0,
        )

        self.img: np.ndarray | None = None
        self.valid_mask: np.ndarray | None = None
        self.mics_base_emb: np.ndarray | None = None
        self._base_t: torch.Tensor | None = None
        self._level_probs_t: list[torch.Tensor] | None = None
        self._hd_edges_np: np.ndarray | None = None
        self._hd_edges_t: torch.Tensor | None = None
        self._hd_edge_feat_t: torch.Tensor | None = None
        self._mask_hw_t: torch.Tensor | None = None
        self.step_counter = 0
        self._current_epoch = 0
        self._last_loss_total = 0.0

    def _freeze_mics(self) -> None:
        if self.mics.model is None:
            raise ValueError("MiCS must be trained before multi-display refine.")
        self.mics.model.eval()
        for p in self.mics.model.parameters():
            p.requires_grad = False
        for layer in getattr(self.mics, "visualiation_to_cluster", []) or []:
            layer.eval()
            for p in layer.parameters():
                p.requires_grad = False

    def _set_base_embedding(self, emb: np.ndarray) -> None:
        base = np.asarray(emb, dtype=np.float32)
        h, w = self.img.shape[:2]
        c = int(self.mics.number_of_components)
        if base.shape != (h, w, c):
            raise ValueError(f"base_embedding shape {base.shape} != {(h, w, c)}")
        self.mics_base_emb = base.copy()
        self._base_t = torch.from_numpy(self.mics_base_emb).float().to(self._device)

    def _cache_mics_embedding(self, img: np.ndarray) -> None:
        self.mics.img = np.asarray(img, dtype=np.float32)
        with torch.no_grad():
            emb = self.mics.predict_embedding(img).astype(np.float32, copy=False)
        self._set_base_embedding(emb)

    def _cache_level_probs(self, img: np.ndarray) -> None:
        assert self._base_t is not None
        h, w, c = self._base_t.shape
        probs_list: list[torch.Tensor] = []
        for level_idx in range(self.n_levels):
            probs_np = self.mics.predict_cluster_soft_probs(img, level_idx=level_idx)
            probs_t = torch.from_numpy(probs_np).float().to(self._device)
            probs_list.append(probs_t)
        self._level_probs_t = probs_list

    def _level_blend_weights(self) -> list[float]:
        if self.level_blend is not None:
            if len(self.level_blend) != self.n_levels:
                raise ValueError(
                    f"level_blend length {len(self.level_blend)} != n_levels {self.n_levels}"
                )
            return [float(np.clip(b, 0.0, 1.0)) for b in self.level_blend]
        # Coarse → fine: more cluster-field geometry at higher k.
        if self.n_levels == 1:
            return [0.5]
        out: list[float] = []
        for i in range(self.n_levels):
            t = i / float(self.n_levels - 1)
            out.append(float(0.22 + 0.68 * t))
        return out

    def _cluster_field_embedding(
        self, base: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        """Soft centroid map in embedding space: sum_k p_k(x) * c_k."""
        h, w, c = base.shape
        flat_e = base.reshape(-1, c)
        flat_p = probs.reshape(-1, probs.shape[-1])
        mass = flat_p.sum(dim=0).clamp(min=1e-8)
        centroids = (flat_p.T @ flat_e) / mass.unsqueeze(-1)
        field = (flat_p @ centroids).reshape(h, w, c)
        return field

    def _init_level_projs(self) -> None:
        assert self._level_probs_t is not None and self.valid_mask is not None
        if self.display_mode != "pca_delta":
            return
        if self.init_display not in ("pca", "pca_probs"):
            return
        mask = np.asarray(self.valid_mask, dtype=bool)
        n_comp = int(self.mics.number_of_components)
        with torch.no_grad():
            for level_idx, probs_t in enumerate(self._level_probs_t):
                probs_np = probs_t.detach().cpu().numpy().reshape(-1, probs_t.shape[-1])
                valid = probs_np[mask.ravel()]
                w = _pca_init_linear(valid, n_comp)
                self.level_projs[level_idx].weight.copy_(
                    torch.from_numpy(w).to(self._device)
                )

    def set_hd_edges(self, hd_edges: np.ndarray | None) -> None:
        if hd_edges is None:
            self._hd_edges_np = None
            self._hd_edges_t = None
            self._hd_edge_feat_t = None
            return
        he = np.squeeze(np.asarray(hd_edges, dtype=np.float32))
        if self.img is None:
            raise ValueError("set_image before set_hd_edges")
        h, w = self.img.shape[:2]
        if he.shape != (h, w):
            raise ValueError(f"hd_edges shape {he.shape} != {(h, w)}")
        he = np.clip(he, 0.0, 1.0)
        self._hd_edges_np = he
        self._hd_edges_t = torch.from_numpy(he).to(self._device).unsqueeze(0).unsqueeze(0)
        self._hd_edge_feat_t = torch.from_numpy(he).to(self._device).unsqueeze(-1)

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
        self._cache_level_probs(img)
        self._init_level_projs()
        mask = self.valid_mask.astype(np.float32)
        self._mask_hw_t = torch.from_numpy(mask).to(self._device).unsqueeze(0).unsqueeze(0)

    def _forward_embeddings(
        self, base: torch.Tensor, level_probs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        h, w, c = base.shape
        flat_base = base.reshape(-1, c)
        blends = self._level_blend_weights()
        level_embs: list[torch.Tensor] = []
        for level_idx, (probs, proj) in enumerate(zip(level_probs, self.level_projs, strict=True)):
            if self.display_mode == "centroid_blend":
                field = self._cluster_field_embedding(base, probs)
                alpha = blends[level_idx]
                level_embs.append((1.0 - alpha) * base + alpha * field)
            elif self.display_mode == "centroid_residual":
                field = self._cluster_field_embedding(base, probs)
                residual = field - field.mean(dim=(0, 1), keepdim=True)
                level_embs.append(base + blends[level_idx] * residual)
            else:
                flat_p = probs.reshape(-1, probs.shape[-1])
                delta = proj(flat_p).reshape(h, w, c) * self.cluster_scale
                level_embs.append(base + delta)
        mix_logits = self.mix_head(flat_base).reshape(h, w, self.n_levels)
        weights = F.softmax(mix_logits / self.mix_temperature, dim=-1)
        stacked = torch.stack(level_embs, dim=-1)
        mixed = (stacked * weights.unsqueeze(-2)).sum(dim=-1)
        return mixed, level_embs, weights

    def _embedding_to_display_tensor(self, emb_hwc: torch.Tensor) -> torch.Tensor:
        """MiCS-style percentile stretch on embedding (LAB 0–1), no LAB→RGB."""
        assert self.img is not None and self.valid_mask is not None
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

    def _edge_loss_ramp(self) -> float:
        if self.edge_loss_ramp_epochs <= 0:
            return 1.0
        return min(1.0, float(self._current_epoch + 1) / float(self.edge_loss_ramp_epochs))

    def _viz_edge_mag(self, display_hwc: torch.Tensor) -> torch.Tensor:
        stretched = display_hwc.clamp(0.0, 1.0)
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

    def _edge_corr_loss(self, mag_v: torch.Tensor) -> torch.Tensor:
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

    def _edge_mag_loss(self, mag_v: torch.Tensor) -> torch.Tensor:
        if self._hd_edges_t is None or self._mask_hw_t is None:
            return mag_v.sum() * 0.0
        m = self._mask_hw_t.float()
        hd = self._hd_edges_t * m
        w = hd.pow(self.edge_mag_hd_power)
        diff = (mag_v - hd).pow(2) * w
        return diff.sum() / (w.sum() + 1e-8)

    def step(self) -> None:
        assert self._base_t is not None and self._level_probs_t is not None
        self.optim.zero_grad(set_to_none=True)
        mixed_emb, level_embs, _weights = self._forward_embeddings(self._base_t, self._level_probs_t)
        base_display = self._embedding_to_display_tensor(self._base_t)
        mixed_display = self._embedding_to_display_tensor(mixed_emb)
        edge_ramp = self._edge_loss_ramp()
        loss = torch.zeros((), device=self._device, dtype=torch.float32)

        if self.mics_preserve_weight > 0.0:
            loss = loss + self.mics_preserve_weight * F.mse_loss(mixed_display, base_display)

        if self.level_delta_l2_weight > 0.0:
            pen = torch.zeros((), device=self._device, dtype=torch.float32)
            for le in level_embs:
                pen = pen + F.mse_loss(le, self._base_t)
            loss = loss + self.level_delta_l2_weight * (pen / len(level_embs))

        use_edge = (
            self.dice_loss_weight > 0.0
            or self.edge_corr_loss_weight > 0.0
            or self.edge_mag_loss_weight > 0.0
        )
        if use_edge and self._hd_edges_t is not None:
            mag_v = self._viz_edge_mag(mixed_display)
            if self.dice_loss_weight > 0.0:
                dice = soft_dice_continuous(mag_v, self._hd_edges_t, self._mask_hw_t)
                loss = loss + edge_ramp * self.dice_loss_weight * (-dice)
            if self.edge_corr_loss_weight > 0.0:
                loss = loss + edge_ramp * self.edge_corr_loss_weight * self._edge_corr_loss(mag_v)
            if self.edge_mag_loss_weight > 0.0:
                loss = loss + edge_ramp * self.edge_mag_loss_weight * self._edge_mag_loss(mag_v)

        if self.non_edge_l1_weight > 0.0 and self._hd_edge_feat_t is not None:
            w = (1.0 - self._hd_edge_feat_t.pow(self.edge_delta_gate_power)).clamp(0.0, 1.0)
            diff = (mixed_display - base_display).abs() * w
            loss = loss + self.non_edge_l1_weight * diff.mean()

        if not loss.requires_grad:
            loss = loss + mixed_emb.sum() * 0.0
        self._last_loss_total = float(loss.detach().cpu().item())
        if not torch.isfinite(loss):
            logger.warning("Multi-display step %d: non-finite loss", self.step_counter)
            self.step_counter += 1
            return
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.level_projs.parameters()) + list(self.mix_head.parameters()),
            max_norm=1.0,
        )
        self.optim.step()
        self.step_counter += 1

    def fit(
        self,
        img: np.ndarray,
        *,
        hd_edges: np.ndarray | None = None,
        pixel_sampling_cache: dict[str, Any] | None = None,
        base_embedding: np.ndarray | None = None,
    ) -> MSIMiCSMultiDisplayRefine:
        self.set_image(img, pixel_sampling_cache=pixel_sampling_cache, base_embedding=base_embedding)
        self.set_hd_edges(hd_edges)
        if self.num_epochs <= 0:
            blends = self._level_blend_weights()
            logger.info(
                "Multi-display: %s (levels=%s, blends=%s)",
                self.display_mode,
                self.level_ks,
                [f"{b:.2f}" for b in blends],
            )
            self._log_level_diversity()
            return self
        logger.info(
            "Multi-display refine: levels=%s epochs=%d preserve_w=%.2g cluster_scale=%.2g",
            self.level_ks,
            self.num_epochs,
            self.mics_preserve_weight,
            self.cluster_scale,
        )
        for epoch in tqdm.tqdm(range(self.num_epochs), desc="MiCS multi-display"):
            self._current_epoch = epoch
            self.step()
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info("epoch %d/%d loss=%.4f", epoch + 1, self.num_epochs, self._last_loss_total)
        logger.info("Multi-display final loss=%.4f", self._last_loss_total)
        self._log_level_diversity()
        return self

    def _log_level_diversity(self) -> None:
        assert self._base_t is not None and self._level_probs_t is not None
        with torch.no_grad():
            _, level_embs, _ = self._forward_embeddings(self._base_t, self._level_probs_t)
        if self.valid_mask is None or len(level_embs) < 2:
            return
        mask = np.asarray(self.valid_mask, dtype=bool)
        base_np = self._base_t.detach().cpu().numpy()
        base_rms = float(np.sqrt(np.mean(base_np[mask] ** 2)))
        for i, le in enumerate(level_embs):
            arr = le.detach().cpu().numpy()
            diff_rms = float(np.sqrt(np.mean((arr - base_np)[mask] ** 2)))
            logger.info(
                "Level k=%s vs base emb RMS diff: %.4f (%.1f%% of base RMS)",
                self.level_ks[i],
                diff_rms,
                100.0 * diff_rms / (base_rms + 1e-8),
            )
        for i in range(len(level_embs)):
            for j in range(i + 1, len(level_embs)):
                a = level_embs[i].detach().cpu().numpy()
                b = level_embs[j].detach().cpu().numpy()
                d = float(np.sqrt(np.mean((a - b)[mask] ** 2)))
                logger.info(
                    "Level k=%s vs k=%s emb RMS diff: %.4f",
                    self.level_ks[i],
                    self.level_ks[j],
                    d,
                )

    def _emb_hwc_to_rgb_u8(self, emb_hwc: np.ndarray) -> np.ndarray:
        assert self.img is not None
        emb = _spatial_gaussian_smooth_channels(
            np.asarray(emb_hwc, dtype=np.float32),
            self.predict_spatial_smooth_sigma,
        )
        rgb = self.mics._embedding_to_display_rgb(emb, self.img)
        if self.valid_mask is not None:
            rgb[~self.valid_mask] = 0
        return np.asarray(rgb, dtype=np.uint8)[..., :3]

    def predict_level_display(self, level_idx: int) -> np.ndarray:
        assert self._base_t is not None and self._level_probs_t is not None
        self.level_projs.eval()
        self.mix_head.eval()
        with torch.no_grad():
            _, level_embs, _ = self._forward_embeddings(self._base_t, self._level_probs_t)
            emb = level_embs[int(level_idx)].cpu().numpy().astype(np.float32)
        return self._emb_hwc_to_rgb_u8(emb)

    def predict_mixed(self) -> np.ndarray:
        assert self._base_t is not None and self._level_probs_t is not None
        self.level_projs.eval()
        self.mix_head.eval()
        with torch.no_grad():
            mixed, _, _ = self._forward_embeddings(self._base_t, self._level_probs_t)
            emb = mixed.cpu().numpy().astype(np.float32)
        return self._emb_hwc_to_rgb_u8(emb)

    def predict_mix_weights(self) -> np.ndarray:
        assert self._base_t is not None and self._level_probs_t is not None
        self.level_projs.eval()
        self.mix_head.eval()
        with torch.no_grad():
            _, _, w = self._forward_embeddings(self._base_t, self._level_probs_t)
            weights = w.cpu().numpy().astype(np.float32)
        if self.valid_mask is not None:
            weights[~self.valid_mask] = 0.0
        return weights

    def release_resources(self) -> None:
        self.level_projs = nn.ModuleList()
        self.mix_head = nn.Linear(1, 1)
        self.optim = None
        self._base_t = None
        self._level_probs_t = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
