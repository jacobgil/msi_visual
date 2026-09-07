"""
Parallel MiCS display heads on a frozen shared encoder.

Stage 1 (caller): train ``MSIParametricMiCSLMC`` (one encoder + multi-level cluster heads).
Stage 2: freeze encoder + cluster heads; learn per-level affine display maps::

    d_L(x) from level-specific cluster centroid fields (continuous, not palette):

    - ``field_residual`` (default): d_0 = blend(base, field_0);
      d_L = base + scale_L * (field_L - field_0) for L > 0
    - ``head_blend``: learned head with (1-α)*base + α*head(e), trained toward fields

Each RGB_L uses MiCS percentile stretch + LAB→RGB on d_L.
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
)

logger = logging.getLogger(__name__)


class MSIMiCSParallelDisplay:
    """Frozen MiCS encoder + one learned 3D display head per cluster level."""

    def __init__(
        self,
        mics: MSIParametricMiCSLMC,
        *,
        num_epochs: int = 80,
        lr: float = 0.02,
        batch_size: int | None = None,
        anchor_level_idx: int = 0,
        anchor_weight: float = 0.55,
        level_cluster_weights: list[float] | None = None,
        level_residual_scales: list[float] | None = None,
        level_blend_alphas: list[float] | None = None,
        field_coarse_scales: list[float] | None = None,
        display_composition: str = "field_residual",
        embed_clamp: float = 80.0,
        diversity_weight: float = 0.06,
        diversity_margin: float = 0.08,
        separation_weight: float = 0.25,
        temperature: float | None = None,
        verbose: bool = False,
        predict_percentile_low: float = 1.0,
        predict_percentile_high: float = 99.0,
        predict_spatial_smooth_sigma: float = 0.0,
        lab_to_rgb: bool = True,
    ):
        self.mics = mics
        self.num_epochs = max(0, int(num_epochs))
        self.lr = float(lr)
        self.batch_size = int(batch_size) if batch_size is not None else int(mics.batch_size)
        self.anchor_level_idx = int(anchor_level_idx)
        self.anchor_weight = max(0.0, float(anchor_weight))
        self.diversity_weight = max(0.0, float(diversity_weight))
        self.diversity_margin = max(0.0, float(diversity_margin))
        self.separation_weight = max(0.0, float(separation_weight))
        self.display_composition = str(display_composition).lower().strip() or "field_residual"
        self.temperature = float(temperature if temperature is not None else mics.temperature)
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

        if level_cluster_weights is not None:
            if len(level_cluster_weights) != self.n_levels:
                raise ValueError(
                    f"level_cluster_weights length {len(level_cluster_weights)} != {self.n_levels}"
                )
            self.level_cluster_weights = [float(w) for w in level_cluster_weights]
        else:
            self.level_cluster_weights = [
                float(0.6 + 0.7 * i / max(1, self.n_levels - 1)) for i in range(self.n_levels)
            ]
        if level_residual_scales is not None:
            if len(level_residual_scales) != self.n_levels:
                raise ValueError(
                    f"level_residual_scales length {len(level_residual_scales)} != {self.n_levels}"
                )
            self.level_residual_scales = [float(s) for s in level_residual_scales]
        else:
            self.level_residual_scales = [
                float(0.2 + 0.5 * i / max(1, self.n_levels - 1)) for i in range(self.n_levels)
            ]
        if level_blend_alphas is not None:
            if len(level_blend_alphas) != self.n_levels:
                raise ValueError(
                    f"level_blend_alphas length {len(level_blend_alphas)} != {self.n_levels}"
                )
            self.level_blend_alphas = [float(a) for a in level_blend_alphas]
        else:
            self.level_blend_alphas = [
                float(0.35 + 0.55 * i / max(1, self.n_levels - 1)) for i in range(self.n_levels)
            ]
        if field_coarse_scales is not None:
            if len(field_coarse_scales) != self.n_levels:
                raise ValueError(
                    f"field_coarse_scales length {len(field_coarse_scales)} != {self.n_levels}"
                )
            self.field_coarse_scales = [float(s) for s in field_coarse_scales]
        else:
            # L=0 unused; L>=1 amplifies (field_L - field_0) relative to base MiCS.
            self.field_coarse_scales = [0.0] + [
                float(0.85 + 0.95 * i / max(1, self.n_levels - 2))
                for i in range(max(0, self.n_levels - 1))
            ]
        self.embed_clamp = max(1.0, float(embed_clamp))

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_comp = int(mics.number_of_components)
        self.display_heads = nn.ModuleList(
            [nn.Linear(n_comp, n_comp, bias=True) for _ in range(self.n_levels)]
        )
        with torch.no_grad():
            for head in self.display_heads:
                head.weight.copy_(torch.eye(n_comp))
                head.bias.zero_()
        self.display_heads.to(self._device)

        self.optim = torch.optim.AdamW(self.display_heads.parameters(), lr=self.lr, weight_decay=0)

        self.img: np.ndarray | None = None
        self.valid_mask: np.ndarray | None = None
        self._base_emb: np.ndarray | None = None
        self._level_emb_cache: list[np.ndarray] | None = None
        self.step_counter = 0
        self._last_loss_total = 0.0

    def _freeze_mics(self) -> None:
        if self.mics.model is None:
            raise ValueError("MiCS must be trained before parallel display training.")
        self.mics.model.eval()
        for p in self.mics.model.parameters():
            p.requires_grad = False
        for layer in getattr(self.mics, "visualiation_to_cluster", []) or []:
            layer.eval()
            for p in layer.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.mics.model(x.to(self._device))
        emb = torch.nan_to_num(emb, nan=0.0, posinf=self.embed_clamp, neginf=-self.embed_clamp)
        return torch.clamp(emb, -self.embed_clamp, self.embed_clamp)

    def _level_probs(self, emb: torch.Tensor, level_idx: int) -> torch.Tensor:
        layer = self.mics.visualiation_to_cluster[int(level_idx)].to(self._device)
        logits = layer(emb)
        return F.softmax(logits / self.temperature, dim=-1)

    def _cluster_fields(self, emb: torch.Tensor) -> list[torch.Tensor]:
        return [
            self._cluster_field(emb, self._level_probs(emb, level_idx))
            for level_idx in range(self.n_levels)
        ]

    def _field_residual_compose(
        self, emb: torch.Tensor, fields: list[torch.Tensor], level_idx: int
    ) -> torch.Tensor:
        li = int(level_idx)
        if li == 0:
            alpha = self.level_blend_alphas[0]
            return (1.0 - alpha) * emb + alpha * fields[0]
        scale = self.field_coarse_scales[li] if li < len(self.field_coarse_scales) else 1.0
        return emb + scale * (fields[li] - fields[0])

    def _compose_from_fields(
        self, emb: torch.Tensor, fields: list[torch.Tensor], level_idx: int
    ) -> torch.Tensor:
        li = int(level_idx)
        if self.display_composition in ("field", "fields"):
            return fields[li]
        if self.display_composition in ("field_residual", "residual", "residual_field"):
            return self._field_residual_compose(emb, fields, li)
        mapped = self.display_heads[li](emb)
        alpha = self.level_blend_alphas[li]
        head_mix = (1.0 - alpha) * emb + alpha * mapped
        if self.display_composition in ("head", "head_only"):
            return head_mix
        base = self._field_residual_compose(emb, fields, li)
        beta = self.level_residual_scales[li]
        return base + beta * (head_mix - base)

    def _uses_display_heads(self) -> bool:
        return self.display_composition not in (
            "field",
            "fields",
            "field_residual",
            "residual",
            "residual_field",
        )

    def _level_display(self, emb: torch.Tensor, level_idx: int) -> torch.Tensor:
        fields = self._cluster_fields(emb)
        return self._compose_from_fields(emb, fields, level_idx)

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
        if getattr(self.mics, "sampled_data", None) is None:
            self.mics.set_image(img, pixel_sampling_cache=pixel_sampling_cache)
        if base_embedding is not None:
            self._base_emb = np.asarray(base_embedding, dtype=np.float32).copy()
        else:
            self.mics.img = self.img
            self._base_emb = self.mics.predict_embedding(img).astype(np.float32, copy=False)
        self._level_emb_cache = None

    def _diversity_loss(self, displays: list[torch.Tensor]) -> torch.Tensor:
        if self.diversity_weight <= 0.0 or len(displays) < 2:
            return torch.zeros((), device=self._device, dtype=torch.float32)
        pen = torch.zeros((), device=self._device, dtype=torch.float32)
        n_pairs = 0
        for i in range(len(displays)):
            for j in range(i + 1, len(displays)):
                diff = (displays[i] - displays[j]).pow(2).mean().sqrt()
                pen = pen + F.relu(self.diversity_margin - diff)
                n_pairs += 1
        return pen / max(1, n_pairs)

    def _separation_loss(self, displays: list[torch.Tensor]) -> torch.Tensor:
        if self.separation_weight <= 0.0 or len(displays) < 2:
            return torch.zeros((), device=self._device, dtype=torch.float32)
        sep = torch.zeros((), device=self._device, dtype=torch.float32)
        n_pairs = 0
        for i in range(len(displays)):
            for j in range(i + 1, len(displays)):
                sep = sep + (displays[i] - displays[j]).pow(2).mean().sqrt()
                n_pairs += 1
        return -(sep / max(1, n_pairs))

    def _cluster_field(self, emb: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Soft centroid map: sum_k p_k(x) * c_k in embedding space."""
        if emb.ndim == 2:
            mass = probs.sum(dim=0).clamp(min=1e-4)
            centroids = (probs.mT @ emb) / mass.unsqueeze(-1)
            field = probs @ centroids
        else:
            h, w, c = emb.shape
            flat_e = emb.reshape(-1, c)
            flat_p = probs.reshape(-1, probs.shape[-1])
            mass = flat_p.sum(dim=0).clamp(min=1e-4)
            centroids = (flat_p.mT @ flat_e) / mass.unsqueeze(-1)
            field = (flat_p @ centroids).reshape(h, w, c)
        return torch.nan_to_num(field, nan=0.0, posinf=self.embed_clamp, neginf=-self.embed_clamp)

    def step(self) -> None:
        if self.mics.sampled_data is None:
            raise ValueError("MiCS sampled_data missing; call set_image first.")
        n_samples = int(self.mics.sampled_data.shape[0])
        batch_size = max(1, min(self.batch_size, n_samples))
        order = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_steps = 0

        for start in range(0, n_samples, batch_size):
            batch_mask = order[start : start + batch_size]
            x = torch.from_numpy(self.mics.sampled_data[batch_mask]).float().to(self._device)
            emb = self._embed_batch(x)
            with torch.no_grad():
                fields = [
                    self._cluster_field(emb, self._level_probs(emb, level_idx))
                    for level_idx in range(self.n_levels)
                ]
            level_displays: list[torch.Tensor] = []
            loss = torch.zeros((), device=self._device, dtype=torch.float32)

            for level_idx in range(self.n_levels):
                d_l = self._level_display(emb, level_idx)
                level_displays.append(d_l)
                loss = loss + self.level_cluster_weights[level_idx] * F.mse_loss(
                    d_l, fields[level_idx]
                )

            if self.anchor_weight > 0.0:
                loss = loss + self.anchor_weight * F.mse_loss(
                    level_displays[int(self.anchor_level_idx)], emb
                )

            loss = loss + self.diversity_weight * self._diversity_loss(level_displays)
            loss = loss + self.separation_weight * self._separation_loss(level_displays)

            if not torch.isfinite(loss):
                logger.warning("Parallel display step %d batch: non-finite loss", self.step_counter)
                continue

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.display_heads.parameters(), max_norm=1.0)
            self.optim.step()

            epoch_loss += float(loss.detach().cpu().item())
            n_steps += 1

        self._last_loss_total = epoch_loss / max(1, n_steps)
        self.step_counter += 1
        self._level_emb_cache = None

    def fit(
        self,
        img: np.ndarray,
        *,
        pixel_sampling_cache: dict[str, Any] | None = None,
        base_embedding: np.ndarray | None = None,
    ) -> MSIMiCSParallelDisplay:
        self.set_image(img, pixel_sampling_cache=pixel_sampling_cache, base_embedding=base_embedding)
        if not self._uses_display_heads() or self.num_epochs <= 0:
            logger.info(
                "Parallel display: %s (levels=%s, no head training)",
                self.display_composition,
                self.level_ks,
            )
            self._log_level_diversity()
            return self
        logger.info(
            "Parallel display heads: mode=%s levels=%s epochs=%d anchor_w=%.2g",
            self.display_composition,
            self.level_ks,
            self.num_epochs,
            self.anchor_weight,
        )
        for epoch in tqdm.tqdm(range(self.num_epochs), desc="MiCS parallel display"):
            self.step()
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info("epoch %d/%d loss=%.4f", epoch + 1, self.num_epochs, self._last_loss_total)
        logger.info("Parallel display final loss=%.4f", self._last_loss_total)
        self._level_emb_cache = None
        self._log_level_diversity()
        return self

    def _compute_level_embeddings(self) -> list[np.ndarray]:
        assert self._base_emb is not None
        self.display_heads.eval()
        base_t = torch.nan_to_num(
            torch.from_numpy(self._base_emb).float().to(self._device),
            nan=0.0,
            posinf=self.embed_clamp,
            neginf=-self.embed_clamp,
        )
        base_t = torch.clamp(base_t, -self.embed_clamp, self.embed_clamp)
        with torch.no_grad():
            out: list[np.ndarray] = []
            for level_idx in range(self.n_levels):
                d = self._level_display(base_t, level_idx).cpu().numpy().astype(np.float32)
                out.append(d)
        return out

    def _log_level_diversity(self) -> None:
        if self._base_emb is None or self.valid_mask is None:
            return
        levels = self._compute_level_embeddings()
        mask = np.asarray(self.valid_mask, dtype=bool)
        base = np.clip(self._base_emb, -self.embed_clamp, self.embed_clamp)
        base_rms = float(np.sqrt(np.mean(base[mask] ** 2)))
        for i, arr in enumerate(levels):
            diff_rms = float(np.sqrt(np.mean((arr - base)[mask] ** 2)))
            logger.info(
                "Parallel k=%s vs base emb RMS diff: %.4f (%.1f%% of base)",
                self.level_ks[i],
                diff_rms,
                100.0 * diff_rms / (base_rms + 1e-8),
            )
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                d = float(np.sqrt(np.mean((levels[i] - levels[j])[mask] ** 2)))
                logger.info(
                    "Parallel k=%s vs k=%s emb RMS diff: %.4f",
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
            rgb = np.asarray(rgb, dtype=np.uint8)
            rgb[~self.valid_mask] = 0
        return np.asarray(rgb, dtype=np.uint8)[..., :3]

    def predict_level_display(self, level_idx: int) -> np.ndarray:
        if self._level_emb_cache is None:
            self._level_emb_cache = self._compute_level_embeddings()
        return self._emb_hwc_to_rgb_u8(self._level_emb_cache[int(level_idx)])

    def predict_base_mics(self) -> np.ndarray:
        assert self._base_emb is not None
        return self._emb_hwc_to_rgb_u8(self._base_emb)

    def release_resources(self) -> None:
        self.display_heads = nn.ModuleList()
        self.optim = None
        self._base_emb = None
        self._level_emb_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
