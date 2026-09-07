"""
Experimental MiCS: shared 3D embedding + per-cluster-scale residual heads.

Architecture (one model, multiple cluster levels K):

- ``shared(x)`` → 3D embedding (RGB visualization base)
- ``residual_heads[i](x)`` → 3D increment for cluster level *i*
- ``fused_i = shared + Σ_{j≤i} residual_j``
- ``cluster_classifier_i(fused_i)`` → K_i classes

Correlation loss uses the deepest fused embedding (or the active level when
``level_wise_freeze``). Optional ``residual_l2_weight`` penalizes large Δ.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import tqdm

from msi_visual.parametric_mics_lmc import (
    MSIParametricMiCSLMC,
    PCCWithLearnedInputMask,
    _build_pcc_sequential,
    _spatial_gaussian_smooth_channels,
    correlation,
)

logger = logging.getLogger(__name__)


class MiCSResidualClusterNet(torch.nn.Module):
    """Shared PCC backbone + small per-level residual PCC heads (same input x)."""

    def __init__(
        self,
        input_dim: int,
        n_components: int = 3,
        factor: float = 1.0,
        num_layers: int = 2,
        *,
        residual_factor: float = 0.25,
        residual_num_layers: int = 2,
    ):
        super().__init__()
        self.n_components = int(n_components)
        self._input_dim = int(input_dim)
        self._factor = float(factor)
        self._residual_factor = float(residual_factor)
        self._residual_num_layers = max(1, int(residual_num_layers))
        self.shared = _build_pcc_sequential(
            self._input_dim, self.n_components, self._factor, max(1, int(num_layers))
        )
        self.residual_heads = torch.nn.ModuleList()

    def ensure_residual_heads(self, n_levels: int) -> None:
        n_levels = max(0, int(n_levels))
        if len(self.residual_heads) == n_levels:
            return
        device = next(self.parameters()).device
        heads = torch.nn.ModuleList()
        width_factor = self._factor * self._residual_factor
        for _ in range(n_levels):
            head = _build_pcc_sequential(
                self._input_dim,
                self.n_components,
                width_factor,
                self._residual_num_layers,
            )
            heads.append(head.to(device))
        self.residual_heads = heads

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x)

    def forward_residual(self, x: torch.Tensor, level: int) -> torch.Tensor:
        return self.residual_heads[int(level)](x)

    def forward_cumulative(self, x: torch.Tensor, level: int) -> torch.Tensor:
        level = int(level)
        z = self.forward_shared(x)
        for j in range(level + 1):
            z = z + self.forward_residual(x, j)
        return z

    def forward_final(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.residual_heads) == 0:
            return self.forward_shared(x)
        return self.forward_cumulative(x, len(self.residual_heads) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_final(x)


class MSIParametricMiCSResidualCluster(MSIParametricMiCSLMC):
    """
    MiCS with shared RGB embedding and cumulative per-level residual heads.

    Extra kwargs:
        residual_factor: width scale for residual heads vs shared backbone (default 0.25)
        residual_num_layers: depth of each residual head (default 2)
        residual_l2_weight: L2 penalty on per-level Δ magnitudes (default 0.05; 0 = off)
        residual_l2_relative: scale penalty by mean shared embedding energy (default True)
        level_wise_freeze: train level 0, freeze, train level 1, … (default False)
        epochs_per_level: epochs per cluster level when level_wise_freeze (default: num_epochs/n_levels)
    """

    def __init__(
        self,
        *args: Any,
        residual_factor: float = 0.25,
        residual_num_layers: int = 2,
        residual_l2_weight: float = 0.05,
        residual_l2_relative: bool = True,
        level_wise_freeze: bool = False,
        epochs_per_level: int | None = None,
        **kwargs: Any,
    ):
        self.residual_factor = float(residual_factor)
        self.residual_num_layers = max(1, int(residual_num_layers))
        self.residual_l2_weight = max(0.0, float(residual_l2_weight))
        self.residual_l2_relative = bool(residual_l2_relative)
        self.level_wise_freeze = bool(level_wise_freeze)
        self.epochs_per_level = (
            None if epochs_per_level is None else max(1, int(epochs_per_level))
        )
        self._active_train_level: int | None = None
        self._k_epoch_initial: int | None = None
        super().__init__(*args, **kwargs)

    def _create_default_model(self, input_dim: int) -> torch.nn.Module:
        if self.learn_input_mask:
            raise ValueError(
                "MSIParametricMiCSResidualCluster does not support learn_input_mask yet."
            )
        model = MiCSResidualClusterNet(
            int(input_dim),
            self.number_of_components,
            self.factor,
            self.num_layers,
            residual_factor=self.residual_factor,
            residual_num_layers=self.residual_num_layers,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def _init_clustering_heads(self) -> None:
        super()._init_clustering_heads()
        if isinstance(self.model, MiCSResidualClusterNet):
            n_levels = len(self.visualiation_to_cluster) if self.cluster else 0
            self.model.ensure_residual_heads(n_levels)

    def _embed_for_cluster_head(self, x: torch.Tensor, head_idx: int) -> torch.Tensor:
        if isinstance(self.model, MiCSResidualClusterNet):
            return self.model.forward_cumulative(x, head_idx)
        return super()._embed_for_cluster_head(x, head_idx)

    def _cumulative_embed(self, x: torch.Tensor, level: int) -> torch.Tensor:
        if isinstance(self.model, MiCSResidualClusterNet):
            return self.model.forward_cumulative(x, level)
        return self.model(x)

    def _reference_embeddings_at_level(self, level: int) -> torch.Tensor:
        if not isinstance(self.model, MiCSResidualClusterNet):
            return self._reference_embeddings()
        ref_x = self.reference_points
        return self._cumulative_embed(ref_x, level)

    def _set_requires_grad_module(self, module: torch.nn.Module | None, flag: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = bool(flag)

    def _freeze_all_residual_cluster_params(self) -> None:
        if not isinstance(self.model, MiCSResidualClusterNet):
            return
        self._set_requires_grad_module(self.model.shared, False)
        for head in self.model.residual_heads:
            self._set_requires_grad_module(head, False)
        if getattr(self, "visualiation_to_cluster", None):
            for layer in self.visualiation_to_cluster:
                self._set_requires_grad_module(layer, False)

    def _configure_trainable_level(self, level: int) -> None:
        """Level 0: shared + Δ₀ + head₀; later levels: only Δ_i + head_i."""
        if not isinstance(self.model, MiCSResidualClusterNet):
            raise TypeError("Expected MiCSResidualClusterNet.")
        level = int(level)
        n_heads = len(self.visualiation_to_cluster)
        if level < 0 or level >= n_heads:
            raise IndexError(f"train level must be in [0, {n_heads - 1}], got {level}")
        self._freeze_all_residual_cluster_params()
        if level == 0:
            self._set_requires_grad_module(self.model.shared, True)
        if level < len(self.model.residual_heads):
            self._set_requires_grad_module(self.model.residual_heads[level], True)
        self._set_requires_grad_module(self.visualiation_to_cluster[level], True)

    def _rebuild_optimizer_trainable(self) -> None:
        params: list[dict[str, Any]] = []
        if isinstance(self.model, MiCSResidualClusterNet):
            for p in self.model.parameters():
                if p.requires_grad:
                    params.append({"params": [p], "weight_decay": 0})
        if getattr(self, "visualiation_to_cluster", None):
            for layer in self.visualiation_to_cluster:
                for p in layer.parameters():
                    if p.requires_grad:
                        params.append({"params": [p], "weight_decay": 0})
        if self.category_head is not None:
            for p in self.category_head.parameters():
                if p.requires_grad:
                    params.append({"params": [p], "weight_decay": 0})
        if not params:
            raise RuntimeError("No trainable parameters for current level.")
        self.optim = torch.optim.AdamW(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=self._lr_lambda
        )

    def fit(self, X, pixel_sampling_cache: dict[str, Any] | None = None):
        if not self.level_wise_freeze:
            return super().fit(X, pixel_sampling_cache=pixel_sampling_cache)

        if logger.isEnabledFor(logging.INFO):
            t0 = time.perf_counter()
            logger.info("[fit] set_image starting: shape=%s", getattr(X, "shape", None))
        self.set_image(X, pixel_sampling_cache=pixel_sampling_cache)
        if logger.isEnabledFor(logging.INFO):
            logger.info("[fit] set_image done in %.2fs", time.perf_counter() - t0)

        n_levels = len(self.visualiation_to_cluster) if self.cluster else 0
        if n_levels == 0:
            return super().fit(X, pixel_sampling_cache=pixel_sampling_cache)

        if self._k_epoch_initial is None:
            self._k_epoch_initial = int(self.k_epoch)
        epochs_each = self.epochs_per_level
        if epochs_each is None:
            epochs_each = max(1, int(self.num_epochs) // n_levels)

        if self.verbose:
            print(
                f"[fit] level-wise freeze: {n_levels} levels × {epochs_each} epochs "
                f"(clusters={self.clusters})"
            )
        logger.info(
            "[fit] level-wise freeze: levels=%s epochs_per_level=%d",
            self.clusters,
            epochs_each,
        )

        t_train = time.perf_counter()
        for level in range(n_levels):
            k_label = self.clusters[level] if level < len(self.clusters) else level
            self._active_train_level = level
            self._configure_trainable_level(level)
            self._rebuild_optimizer_trainable()
            self.step_counter = 0
            self.k_epoch = int(self._k_epoch_initial)
            logger.info(
                "[fit] training level %d/%d (k=%s) for %d epochs",
                level + 1,
                n_levels,
                k_label,
                epochs_each,
            )
            if self.verbose:
                print(f"[fit] level {level + 1}/{n_levels} k={k_label} ({epochs_each} epochs)")

            for _epoch in tqdm.tqdm(range(epochs_each), desc=f"level {level} k={k_label}"):
                self.step()

            self._freeze_all_residual_cluster_params()
            logger.info("[fit] frozen through level %d (k=%s)", level, k_label)

        self._active_train_level = None
        if logger.isEnabledFor(logging.INFO):
            logger.info("[fit] level-wise training finished in %.2fs", time.perf_counter() - t_train)
        return self

    def _forward_training_embed(
        self, x: torch.Tensor, sample_row_indices: np.ndarray
    ) -> torch.Tensor:
        del sample_row_indices
        active = getattr(self, "_active_train_level", None)
        if active is not None and isinstance(self.model, MiCSResidualClusterNet):
            out = self.model.forward_cumulative(x, int(active))
        else:
            out = self.model(x)
        if getattr(self, "residual_base_sampled", None) is not None:
            raise ValueError(
                "MSIParametricMiCSResidualCluster does not support external residual_base; "
                "residuals are learned per cluster level inside the model."
            )
        return out

    def _reference_embeddings(self) -> torch.Tensor:
        active = getattr(self, "_active_train_level", None)
        if active is not None and isinstance(self.model, MiCSResidualClusterNet):
            return self._reference_embeddings_at_level(int(active))
        ref = self.model(self.reference_points)
        if getattr(self, "residual_base_sampled", None) is not None:
            raise ValueError("External residual_base is not supported.")
        return ref

    def _infer_embedding_flat(
        self, reshaped: np.ndarray, batch_size: int | None = None
    ) -> np.ndarray:
        if batch_size is None:
            batch_size = 1024 * 32
        self.model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch in torch.split(torch.from_numpy(reshaped).float(), batch_size):
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                emb = self.model(x_batch).cpu().numpy()
                chunks.append(emb.astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0)

    def _run_spatial_embed_fn(self, img: np.ndarray, fn) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be initialized. Call fit() first.")
        img = np.asarray(img, dtype=np.float32)
        reshaped = img.reshape(-1, img.shape[-1])
        self.model.eval()
        batch_size = 1024 * 32
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch in torch.split(torch.from_numpy(reshaped).float(), batch_size):
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                chunks.append(fn(x_batch).cpu().numpy().astype(np.float32, copy=False))
        flat = np.concatenate(chunks, axis=0)
        return flat.reshape(img.shape[0], img.shape[1], -1)

    def predict_shared_embedding(self, img: np.ndarray) -> np.ndarray:
        """Shared backbone only, H×W×3 float32 (before display stretch)."""
        if not isinstance(self.model, MiCSResidualClusterNet):
            raise TypeError("Expected MiCSResidualClusterNet.")
        return self._run_spatial_embed_fn(img, self.model.forward_shared)

    def predict_level_residual_embedding(self, img: np.ndarray, level_idx: int) -> np.ndarray:
        """Single-level residual Δ_i, H×W×3 float32."""
        if not isinstance(self.model, MiCSResidualClusterNet):
            raise TypeError("Expected MiCSResidualClusterNet.")
        level_idx = int(level_idx)
        if level_idx < 0 or level_idx >= len(self.model.residual_heads):
            raise IndexError(
                f"level_idx must be in [0, {len(self.model.residual_heads) - 1}], got {level_idx}"
            )
        return self._run_spatial_embed_fn(
            img, lambda x, li=level_idx: self.model.forward_residual(x, li)
        )

    def predict_level_fused_embedding(self, img: np.ndarray, level_idx: int) -> np.ndarray:
        """Cumulative fused embedding at cluster level i, H×W×3 float32."""
        if not isinstance(self.model, MiCSResidualClusterNet):
            raise TypeError("Expected MiCSResidualClusterNet.")
        level_idx = int(level_idx)
        if level_idx < 0 or level_idx >= len(self.model.residual_heads):
            raise IndexError(
                f"level_idx must be in [0, {len(self.model.residual_heads) - 1}], got {level_idx}"
            )
        return self._run_spatial_embed_fn(
            img, lambda x, li=level_idx: self.model.forward_cumulative(x, li)
        )

    def embedding_to_rgb_u8(self, embedding_hwc: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Percentile stretch (+ optional LAB→RGB) for arbitrary embedding maps."""
        emb = _spatial_gaussian_smooth_channels(
            np.asarray(embedding_hwc, dtype=np.float32),
            self.predict_spatial_smooth_sigma,
        )
        return self._embedding_to_display_rgb(emb, img)

    def _residual_l2_penalty(
        self, x: torch.Tensor, *, active_level: int | None = None
    ) -> torch.Tensor:
        """Mean squared L2 of each Δ level (optionally relative to shared energy)."""
        if (
            self.residual_l2_weight <= 0.0
            or not isinstance(self.model, MiCSResidualClusterNet)
            or len(self.model.residual_heads) == 0
        ):
            return torch.zeros((), device=x.device, dtype=x.dtype)

        shared = self.model.forward_shared(x)
        shared_energy = shared.pow(2).mean().clamp(min=1e-8)
        if active_level is not None:
            level_range = [int(active_level)]
        else:
            level_range = list(range(len(self.model.residual_heads)))
        level_penalties: list[torch.Tensor] = []
        for j in level_range:
            delta = self.model.forward_residual(x, j)
            delta_mse = delta.pow(2).mean()
            if self.residual_l2_relative:
                delta_mse = delta_mse / shared_energy
            level_penalties.append(delta_mse)
        return torch.stack(level_penalties).mean()

    def compute_residual_delta_stats(
        self, img: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> dict[str, Any]:
        """RMS embedding norms on valid tissue (for gallery / diagnostics)."""
        img = np.asarray(img, dtype=np.float32)
        if valid_mask is None:
            valid_mask = img.sum(axis=-1) > 0
        vm = np.asarray(valid_mask, dtype=bool)
        shared = self.predict_shared_embedding(img)
        final = self.predict_embedding(img)

        def _rms(emb: np.ndarray) -> float:
            v = np.asarray(emb[vm], dtype=np.float64)
            if v.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.sum(v * v, axis=-1))))

        rms_shared = _rms(shared)
        rms_final = _rms(final)
        levels: list[dict[str, float | int]] = []
        n_levels = (
            len(self.model.residual_heads)
            if isinstance(self.model, MiCSResidualClusterNet)
            else 0
        )
        for li in range(n_levels):
            delta = self.predict_level_residual_embedding(img, li)
            fused = self.predict_level_fused_embedding(img, li)
            rms_delta = _rms(delta)
            rms_fused = _rms(fused)
            levels.append(
                {
                    "level": li,
                    "rms_delta": rms_delta,
                    "rms_fused": rms_fused,
                    "delta_over_shared": rms_delta / max(rms_shared, 1e-8),
                    "fused_minus_shared_over_shared": _rms(fused - shared)
                    / max(rms_shared, 1e-8),
                }
            )
        return {
            "rms_shared": rms_shared,
            "rms_final": rms_final,
            "final_minus_shared_over_shared": _rms(final - shared) / max(rms_shared, 1e-8),
            "levels": levels,
        }

    def step(self) -> None:
        """Same as base MiCS but cluster heads use per-level fused embeddings."""
        batch_size = self.batch_size
        n_samples = self.sampled_data.shape[0]
        num_batches = (n_samples + batch_size - 1) // batch_size
        if self.step_counter == 0 and self.verbose:
            print(
                f"[step] Starting optimization: {num_batches} batches, "
                f"batch_size={batch_size}, n_samples={n_samples}"
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
                active = getattr(self, "_active_train_level", None)
                level_range = (
                    [int(active)]
                    if active is not None
                    else range(len(self.visualiation_to_cluster))
                )
                n_cluster_terms = len(level_range)
                for i in level_range:
                    layer = self.visualiation_to_cluster[i]
                    clusters = self.cluster_labels[i]
                    batch_clusters = torch.from_numpy(clusters[batch_mask]).long()
                    if torch.cuda.is_available():
                        batch_clusters = batch_clusters.cuda()
                    emb_i = self._embed_for_cluster_head(x, i)
                    layer_output = layer(emb_i)
                    cluster_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(
                        layer_output / self.temperature, batch_clusters
                    )
                    cluster_loss_sum = cluster_loss_sum + cluster_loss
                loss = loss + self.cluster_loss_weight * (
                    cluster_loss_sum / max(1, n_cluster_terms)
                )

            if self.category_head is not None and self.category_labels is not None:
                batch_cats = torch.from_numpy(self.category_labels[batch_mask]).long()
                if torch.cuda.is_available():
                    batch_cats = batch_cats.cuda()
                cat_output = self.category_head(batch_outputs)
                cat_loss = torch.nn.functional.cross_entropy(
                    cat_output / self.temperature, batch_cats, ignore_index=-1
                )
                loss = loss + self.category_loss_weight * cat_loss

            if self.step_counter % self.k_epoch == self.k_epoch - 1:
                correlation_loss = -correlation(low_d_distances, high_d_distances).mean()
                if correlation_loss > 0 and self.verbose:
                    print(f"Correlation loss is positive: {correlation_loss}")
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

            l1w = float(getattr(self, "input_mask_l1_weight", 0.0))
            if l1w > 0.0 and isinstance(self.model, PCCWithLearnedInputMask):
                loss = loss + l1w * self.model.l1_mask_penalty()

            rl2w = float(getattr(self, "residual_l2_weight", 0.0))
            if rl2w > 0.0:
                active = getattr(self, "_active_train_level", None)
                loss = loss + rl2w * self._residual_l2_penalty(
                    x, active_level=active
                )

            self.optim.zero_grad()
            loss.backward()
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if getattr(self, "visualiation_to_cluster", None):
                for layer in self.visualiation_to_cluster:
                    trainable.extend(p for p in layer.parameters() if p.requires_grad)
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            self.optim.step()

        if self.step_counter % self.k_epoch == self.k_epoch - 1:
            old_k_epoch = self.k_epoch
            self.k_epoch = max(1, int(self.k_epoch * 0.92))
            if old_k_epoch != self.k_epoch and self.verbose:
                print(f"[step] Updated k_epoch: {old_k_epoch} -> {self.k_epoch}")

        self.step_counter += 1
