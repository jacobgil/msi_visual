"""
MiCS + edge loss on sampled superpixels.

Default ``edge_loss_mode=sampled``: a small head on the embedding predicts
edge strength (or edge vs non-edge) at each **already-sampled** MiCS pixel —
no extra full-image forward pass during training.

Optional ``edge_loss_mode=spatial`` keeps the slower Sobel-on-viz map loss.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC, correlation
from msi_visual.parametric_mics_unet import (
    _luminance_rgb01,
    _rgb_from_feat,
    _sobel_mag_gray,
    soft_dice_continuous,
)

logger = logging.getLogger(__name__)


def _compute_edge_target_from_msi(msi: np.ndarray) -> np.ndarray:
    """Ion-sum magnitude → Sobel edge strength in [0, 1] (tissue only)."""
    import cv2

    h = np.linalg.norm(np.asarray(msi, dtype=np.float32), axis=-1)
    mask = h > 0
    if not np.any(mask):
        return np.zeros_like(h, dtype=np.float32)
    lo, hi = np.percentile(h[mask], [1.0, 99.0])
    z = np.clip((h - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
    z[~mask] = 0.0
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    mag[~mask] = 0.0
    mx = float(mag[mask].max())
    if mx > 1e-8:
        mag[mask] /= mx
    return mag


class MSIParametricMiCSEdge(MSIParametricMiCSLMC):
    """
    Standard per-pixel MiCS with edge alignment.

    ``edge_loss_mode``:
    - ``sampled`` (default): ``edge_head(embedding)`` vs edge target on MiCS samples
    - ``spatial``: differentiable Sobel on full viz map (slow)

    Sampled edge targets use continuous edge strength in [0, 1] by default.
    Optional ``edge_binary_threshold`` restores hard 0/1 labels.
    ``edge_label_smoothing`` blends targets toward 0.5; ``edge_sample_weight_mode``
    reweights per-sample BCE/MSE (``strength`` or ``emphasis``).
    """

    def __init__(
        self,
        *args: Any,
        edge_loss_weight: float = 1.0,
        edge_mse_weight: float = 0.0,
        edge_loss_mode: str = "sampled",
        edge_binary_threshold: float | None = None,
        edge_label_smoothing: float = 0.0,
        edge_sample_weight_mode: str = "none",
        edge_sample_weight_alpha: float = 1.0,
        edge_sample_weight_floor: float = 0.25,
        edge_loss_every_n_steps: int = 1,
        edge_spatial_stride: int = 4,
        hd_edges: Optional[np.ndarray] = None,
        **kwargs: Any,
    ):
        self.edge_loss_weight = max(0.0, float(edge_loss_weight))
        self.edge_mse_weight = max(0.0, float(edge_mse_weight))
        mode = str(edge_loss_mode).strip().lower()
        if mode not in ("sampled", "spatial"):
            raise ValueError("edge_loss_mode must be 'sampled' or 'spatial'")
        self.edge_loss_mode = mode
        self.edge_binary_threshold = edge_binary_threshold
        self.edge_label_smoothing = max(0.0, min(1.0, float(edge_label_smoothing)))
        weight_mode = str(edge_sample_weight_mode).strip().lower()
        if weight_mode not in ("none", "strength", "emphasis"):
            raise ValueError(
                "edge_sample_weight_mode must be 'none', 'strength', or 'emphasis'"
            )
        self.edge_sample_weight_mode = weight_mode
        self.edge_sample_weight_alpha = max(0.0, float(edge_sample_weight_alpha))
        self.edge_sample_weight_floor = max(0.0, min(1.0, float(edge_sample_weight_floor)))
        self.edge_loss_every_n_steps = max(1, int(edge_loss_every_n_steps))
        self.edge_spatial_stride = max(1, int(edge_spatial_stride))
        self.hd_edges_np: Optional[np.ndarray] = (
            None if hd_edges is None else np.asarray(hd_edges, dtype=np.float32)
        )
        self._hd_edges_t: Optional[torch.Tensor] = None
        self._mask_hw_t: Optional[torch.Tensor] = None
        self.edge_head: nn.Module | None = None
        self.edge_sample_strength: np.ndarray | None = None
        self.edge_sample_targets: np.ndarray | None = None
        super().__init__(*args, **kwargs)

    def _targets_from_strength(self, strength: np.ndarray) -> np.ndarray:
        t = np.clip(strength, 0.0, 1.0).astype(np.float32, copy=True)
        thr = self.edge_binary_threshold
        if thr is not None:
            t = (t >= float(thr)).astype(np.float32)
        eps = self.edge_label_smoothing
        if eps > 0.0:
            t = t * (1.0 - eps) + 0.5 * eps
        return t.astype(np.float32)

    def _sample_weights_from_strength(self, strength: torch.Tensor) -> torch.Tensor | None:
        mode = self.edge_sample_weight_mode
        if mode == "none":
            return None
        if mode == "strength":
            floor = self.edge_sample_weight_floor
            return floor + (1.0 - floor) * strength
        return 1.0 + self.edge_sample_weight_alpha * strength

    def set_hd_edges(self, hd_edges: np.ndarray) -> None:
        self.hd_edges_np = np.asarray(hd_edges, dtype=np.float32)
        if self.img is not None:
            self._sync_edge_targets()

    def _sync_edge_targets(self) -> None:
        if self.img is None or self.model is None:
            return
        H, W = int(self.img.shape[0]), int(self.img.shape[1])
        device = next(self.model.parameters()).device
        mask_np = self.img.sum(axis=-1) > 0
        self._mask_hw_t = (
            torch.from_numpy(mask_np.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
        )
        if self.hd_edges_np is None:
            self.hd_edges_np = _compute_edge_target_from_msi(self.img)
        he = self.hd_edges_np
        if he.shape[:2] != (H, W):
            raise ValueError(f"hd_edges shape {he.shape[:2]} != MSI {(H, W)}")
        self._hd_edges_t = (
            torch.from_numpy(np.clip(he, 0.0, 1.0)).to(device).unsqueeze(0).unsqueeze(0)
        )
        if getattr(self, "sampled_flat_indices", None) is None:
            self.edge_sample_targets = None
            return
        flat = np.asarray(self.sampled_flat_indices, dtype=np.int64).ravel()
        strength = np.clip(he.reshape(-1)[flat], 0.0, 1.0).astype(np.float32)
        self.edge_sample_strength = strength
        self.edge_sample_targets = self._targets_from_strength(strength)

    def _init_edge_head(self) -> None:
        if self.edge_loss_mode != "sampled":
            self.edge_head = None
            return
        if self.edge_loss_weight <= 0.0 and self.edge_mse_weight <= 0.0:
            self.edge_head = None
            return
        device = next(self.model.parameters()).device
        self.edge_head = nn.Linear(int(self.number_of_components), 1)
        if torch.cuda.is_available():
            self.edge_head = self.edge_head.cuda()
        else:
            self.edge_head = self.edge_head.to(device)

    def _rebuild_optimizer(self) -> None:
        params: list[dict[str, Any]] = [{"params": self.model.parameters(), "weight_decay": 0}]
        if getattr(self, "visualiation_to_cluster", None):
            for layer in self.visualiation_to_cluster:
                params.append({"params": layer.parameters(), "weight_decay": 0})
        if getattr(self, "category_head", None) is not None:
            params.append({"params": self.category_head.parameters(), "weight_decay": 0})
        if self.edge_head is not None:
            params.append({"params": self.edge_head.parameters(), "weight_decay": 0})
        self.optim = torch.optim.AdamW(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=self._lr_lambda
        )

    def set_image(
        self,
        img,
        roi_mask=None,
        category_annotations=None,
        pixel_sampling_cache=None,
    ):
        super().set_image(
            img,
            roi_mask=roi_mask,
            category_annotations=category_annotations,
            pixel_sampling_cache=pixel_sampling_cache,
        )
        self._sync_edge_targets()
        self._init_edge_head()
        self._rebuild_optimizer()

    def release_resources(self) -> None:
        self._hd_edges_t = None
        self._mask_hw_t = None
        self.edge_head = None
        self.edge_sample_strength = None
        self.edge_sample_targets = None
        super().release_resources()

    def _weighted_mean(
        self, per_sample: torch.Tensor, weights: torch.Tensor | None
    ) -> torch.Tensor:
        if weights is None:
            return per_sample.mean()
        denom = weights.sum().clamp(min=1e-8)
        return (per_sample * weights).sum() / denom

    def _sampled_edge_loss(self, batch_outputs: torch.Tensor, batch_mask: np.ndarray) -> torch.Tensor:
        if self.edge_head is None or self.edge_sample_targets is None:
            return batch_outputs.sum() * 0.0
        device = batch_outputs.device
        strength = torch.from_numpy(self.edge_sample_strength[batch_mask]).float().to(device)
        targets = torch.from_numpy(self.edge_sample_targets[batch_mask]).float().to(device)
        weights = self._sample_weights_from_strength(strength)
        logits = self.edge_head(batch_outputs).squeeze(-1)
        loss = torch.zeros((), device=device, dtype=batch_outputs.dtype)
        if self.edge_loss_weight > 0.0:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = loss + self.edge_loss_weight * self._weighted_mean(bce, weights)
        if self.edge_mse_weight > 0.0:
            pred = torch.sigmoid(logits)
            mse = F.mse_loss(pred, targets, reduction="none")
            loss = loss + self.edge_mse_weight * self._weighted_mean(mse, weights)
        return loss

    def _embedding_map_bchw(self, stride: int | None = None) -> torch.Tensor:
        if self.model is None or self.img is None:
            raise RuntimeError("Model and image must be set.")
        H, W = int(self.img.shape[0]), int(self.img.shape[1])
        stride = max(1, int(self.edge_spatial_stride if stride is None else stride))
        device = next(self.model.parameters()).device
        if stride <= 1:
            x_all = torch.from_numpy(np.asarray(self.reshaped, dtype=np.float32)).to(device)
            chunks: list[torch.Tensor] = []
            for x_batch in torch.split(x_all, 1024 * 16):
                chunks.append(self.model(x_batch))
            emb = torch.cat(chunks, dim=0)
            return emb.reshape(H, W, -1).permute(2, 0, 1).unsqueeze(0)

        rows = torch.arange(0, H, stride, device=device)
        cols = torch.arange(0, W, stride, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        flat = (grid_r * W + grid_c).reshape(-1)
        x_sub = torch.from_numpy(np.asarray(self.reshaped, dtype=np.float32)).to(device)[flat]
        chunks = []
        for x_batch in torch.split(x_sub, 1024 * 16):
            chunks.append(self.model(x_batch))
        emb = torch.cat(chunks, dim=0)
        return emb.reshape(int(rows.numel()), int(cols.numel()), -1).permute(2, 0, 1).unsqueeze(0)

    def _spatial_edge_loss(self) -> torch.Tensor:
        if self._hd_edges_t is None or self._mask_hw_t is None:
            return torch.zeros((), device=next(self.model.parameters()).device)
        stride = self.edge_spatial_stride
        feat = self._embedding_map_bchw(stride=stride)
        mask = self._mask_hw_t
        mag_t = self._hd_edges_t
        if stride > 1:
            size = feat.shape[-2:]
            mask = F.interpolate(mask, size=size, mode="area")
            mag_t = F.interpolate(mag_t, size=size, mode="area")
        rgb = _rgb_from_feat(feat, mask=mask)
        y = _luminance_rgb01(rgb)
        mag_v = _sobel_mag_gray(y)
        loss = torch.zeros((), device=feat.device, dtype=feat.dtype)
        if self.edge_loss_weight > 0.0:
            dice = soft_dice_continuous(mag_v, mag_t, mask)
            loss = loss + self.edge_loss_weight * (-dice)
        if self.edge_mse_weight > 0.0:
            diff = (mag_v - mag_t) * mask
            loss = loss + self.edge_mse_weight * (diff.pow(2).sum() / mask.sum().clamp(min=1.0))
        return loss

    def predict_edge_probability_map(self, img: np.ndarray | None = None) -> np.ndarray:
        """H×W edge probability from ``edge_head`` (inference only, full image)."""
        if self.edge_head is None:
            raise ValueError("edge_head missing; use edge_loss_mode=sampled with edge_loss_weight>0")
        if img is None:
            img = self.img
        if img is None:
            raise ValueError("Pass img or call fit() first.")
        H, W = int(img.shape[0]), int(img.shape[1])
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        self.edge_head.eval()
        try:
            with torch.no_grad():
                reshaped = np.asarray(img, dtype=np.float32).reshape(-1, img.shape[-1])
                chunks: list[np.ndarray] = []
                for x_batch in torch.split(torch.from_numpy(reshaped).float(), 1024 * 16):
                    x_batch = x_batch.to(device)
                    logits = self.edge_head(self.model(x_batch)).squeeze(-1)
                    chunks.append(torch.sigmoid(logits).cpu().numpy())
                flat = np.concatenate(chunks, axis=0)
            out = flat.reshape(H, W).astype(np.float32)
            out[img.sum(axis=-1) <= 0] = 0.0
            return out
        finally:
            if was_training:
                self.model.train()
                self.edge_head.train()

    def predict_viz_edge_magnitude(self, img: np.ndarray | None = None) -> np.ndarray:
        """Sobel magnitude of visualization luminance (for gallery comparison)."""
        if img is None:
            img = self.img
        if img is None:
            raise ValueError("Pass img or call fit() first.")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                feat = self._embedding_map_bchw(stride=1)
                rgb = _rgb_from_feat(feat, mask=self._mask_hw_t)
                y = _luminance_rgb01(rgb)
                mag = _sobel_mag_gray(y)
            out = mag.detach().cpu().numpy()[0, 0].astype(np.float32)
            m = img.sum(axis=-1) > 0
            out = out.copy()
            out[~m] = 0.0
            mx = float(out[m].max()) if np.any(m) else 0.0
            if mx > 1e-8:
                out[m] /= mx
            return out
        finally:
            if was_training:
                self.model.train()

    def step(self) -> None:
        batch_size = self.batch_size
        n_samples = self.sampled_data.shape[0]
        num_batches = (n_samples + batch_size - 1) // batch_size
        if self.step_counter == 0:
            if self.edge_loss_mode == "sampled":
                mean_strength = 0.0
                if self.edge_sample_strength is not None:
                    mean_strength = float(np.mean(self.edge_sample_strength))
                logger.info(
                    "MiCS edge (sampled): %d samples, %d batches, mean strength %.3f "
                    "(threshold=%s, smoothing=%.3f, weight=%s)",
                    n_samples,
                    num_batches,
                    mean_strength,
                    self.edge_binary_threshold,
                    self.edge_label_smoothing,
                    self.edge_sample_weight_mode,
                )
            else:
                H, W = int(self.img.shape[0]), int(self.img.shape[1])
                n_edge = int(
                    ((H + self.edge_spatial_stride - 1) // self.edge_spatial_stride)
                    * ((W + self.edge_spatial_stride - 1) // self.edge_spatial_stride)
                )
                logger.info(
                    "MiCS edge (spatial): %d samples; edge grid %d px (stride=%d)",
                    n_samples,
                    n_edge,
                    self.edge_spatial_stride,
                )

        sbw = getattr(self, "_sample_batch_weights", None)
        use_weighted = (
            sbw is not None
            and getattr(self, "finetune_miss_weighted_batches", False)
            and len(sbw) == n_samples
        )
        if use_weighted:
            p = np.asarray(sbw, dtype=np.float64)
            p = p / (p.sum() + 1e-12)
            epoch_order = np.random.default_rng(
                int(self.random_state) + int(self.step_counter)
            ).choice(n_samples, size=n_samples, replace=True, p=p)
        else:
            epoch_order = np.random.permutation(n_samples)

        device = next(self.model.parameters()).device
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        use_spatial_edge = (
            self.edge_loss_mode == "spatial"
            and (self.edge_loss_weight > 0.0 or self.edge_mse_weight > 0.0)
            and self._hd_edges_t is not None
            and (self.step_counter % self.edge_loss_every_n_steps)
            == (self.edge_loss_every_n_steps - 1)
        )
        use_sampled_edge = (
            self.edge_loss_mode == "sampled"
            and self.edge_head is not None
            and self.edge_sample_targets is not None
            and (self.edge_loss_weight > 0.0 or self.edge_mse_weight > 0.0)
        )

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_mask = epoch_order[start_idx:end_idx]

            x = torch.from_numpy(self.sampled_data[batch_mask])
            if torch.cuda.is_available():
                x = x.cuda()

            loss = torch.zeros((), device=x.device, dtype=torch.float32)
            batch_outputs = self._forward_training_embed(x, batch_mask)
            batch_reference_points = self._reference_embeddings()
            low_d_distances = torch.cdist(batch_outputs, batch_reference_points)
            high_d_distances = self.euclidean[batch_mask]

            if self.cluster and len(self.visualiation_to_cluster) > 0:
                cluster_loss_sum = 0.0
                for i in range(len(self.visualiation_to_cluster)):
                    layer = self.visualiation_to_cluster[i]
                    clusters = self.cluster_labels[i]
                    batch_clusters = torch.from_numpy(clusters[batch_mask]).long()
                    if torch.cuda.is_available():
                        batch_clusters = batch_clusters.cuda()
                    emb_i = self._embed_for_cluster_head(x, i)
                    layer_output = layer(emb_i)
                    cluster_loss = nn.CrossEntropyLoss(ignore_index=-1)(
                        layer_output / self.temperature, batch_clusters
                    )
                    cluster_loss_sum = cluster_loss_sum + cluster_loss
                loss = loss + self.cluster_loss_weight * (
                    cluster_loss_sum / len(self.visualiation_to_cluster)
                )

            if use_sampled_edge:
                loss = loss + self._sampled_edge_loss(batch_outputs, batch_mask)

            if self.category_head is not None and self.category_labels is not None:
                batch_cats = torch.from_numpy(self.category_labels[batch_mask]).long()
                if torch.cuda.is_available():
                    batch_cats = batch_cats.cuda()
                cat_output = self.category_head(batch_outputs)
                cat_loss = F.cross_entropy(
                    cat_output / self.temperature, batch_cats, ignore_index=-1
                )
                loss = loss + self.category_loss_weight * cat_loss

            if self.step_counter % self.k_epoch == self.k_epoch - 1:
                correlation_loss = -correlation(low_d_distances, high_d_distances).mean()
                alpha = -float(correlation_loss.detach().cpu().numpy())
                alpha = max(alpha, 1e-6)
                if self.cluster and len(self.visualiation_to_cluster) > 0:
                    loss = loss + correlation_loss * self.beta / alpha
                elif self.category_head is not None:
                    loss = loss + correlation_loss * self.beta / alpha
                else:
                    loss = correlation_loss

            if not getattr(loss, "requires_grad", False):
                loss = loss + (batch_outputs.sum() * 0.0)

            loss_sum = loss_sum + loss / float(max(num_batches, 1))

        if use_spatial_edge:
            loss_sum = loss_sum + self._spatial_edge_loss()

        if not getattr(loss_sum, "requires_grad", False):
            loss_sum = loss_sum + (next(self.model.parameters()).sum() * 0.0)

        self.optim.zero_grad()
        loss_sum.backward()
        trainable = list(self.model.parameters())
        if self.visualiation_to_cluster:
            for layer in self.visualiation_to_cluster:
                trainable.extend(layer.parameters())
        if self.edge_head is not None:
            trainable.extend(self.edge_head.parameters())
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        self.optim.step()

        if self.step_counter % self.k_epoch == self.k_epoch - 1:
            self.k_epoch = max(1, int(self.k_epoch * 0.92))

        self.step_counter += 1
