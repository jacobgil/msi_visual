"""Sparse concept distillation of MiCS RGB visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances

from msi_visual.parametric_mics_lmc import _spatial_gaussian_smooth_channels, correlation
from msi_visual.utils import normalize


DisplayBoundsMode = Literal["self", "teacher"]
RouterMode = Literal["linear", "spatial_conv"]
TrainObjective = Literal["distill", "mics", "hybrid"]


@dataclass
class ConceptDistillConfig:
    n_concepts: int = 128
    top_k: int = 12
    epochs: int = 120
    steps_per_epoch: int = 4
    lr: float = 1e-2
    embed_w_lr: float = 0.05
    sparsity_weight: float = 0.005
    recon_weight: float = 0.01
    concept_l1_weight: float = 1e-5
    batch_size: int = 16384
    nmf_max_iter: int = 200
    router_mode: RouterMode = "spatial_conv"
    router_conv_kernel: int = 3
    spatial_crop_size: int = 192
    router_temperature: float = 1.0
    predict_embedding_smooth_sigma: float = 0.35
    lab_to_rgb: bool = True
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    display_bounds: DisplayBoundsMode = "teacher"
    readout_ridge_fit: bool = True
    readout_ridge_lambda: float = 1e-3
    readout_ridge_pixels: int = 40000
    readout_epochs: int = 60
    readout_steps_per_epoch: int = 4
    readout_refine_pixels: int = 40000
    readout_lr: float = 0.05
    train_objective: TrainObjective = "distill"
    mics_number_of_points: int = 1000
    mics_sampling: str = "coreset"
    mics_num_samples: int = 5000
    mics_beta: float = 20.0
    mics_k_epoch: int = 1
    mics_teacher_blend: float = 0.0
    embed_diversity_weight: float = 1e-3
    concept_diversity_weight: float = 1e-4
    seed: int = 42
    device: str = "cpu"


def _sparse_topk_softmax(scores: torch.Tensor, top_k: int, temperature: float) -> torch.Tensor:
    k = max(1, min(int(top_k), int(scores.shape[-1])))
    vals, idx = torch.topk(scores, k, dim=-1)
    masked = torch.full_like(scores, float("-inf"))
    masked.scatter_(-1, idx, vals)
    temp = max(float(temperature), 1e-3)
    return F.softmax(masked / temp, dim=-1)


def embedding_channel_bounds(
    embedding_hwc: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Per-channel ``[p_low, p_high]`` for fixed display stretch, shape ``(C, 2)``."""
    mask = np.asarray(tissue_mask, dtype=bool)
    emb = np.asarray(embedding_hwc, dtype=np.float64)
    bounds = np.zeros((emb.shape[-1], 2), dtype=np.float64)
    for ch in range(emb.shape[-1]):
        vals = emb[:, :, ch][mask] if np.any(mask) else emb[:, :, ch].ravel()
        bounds[ch, 0] = float(np.percentile(vals, float(percentile_low)))
        bounds[ch, 1] = float(np.percentile(vals, float(percentile_high)))
    return bounds


def embedding_to_display_rgb(
    embedding_hwc: np.ndarray,
    *,
    tissue_mask: np.ndarray,
    lab_to_rgb: bool = True,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    channel_bounds: np.ndarray | None = None,
) -> np.ndarray:
    """Percentile stretch (+ optional LAB→RGB) as MiCS ``predict``."""
    mask = np.asarray(tissue_mask, dtype=bool)
    emb = np.asarray(embedding_hwc, dtype=np.float32)
    if channel_bounds is not None:
        bounds = np.asarray(channel_bounds, dtype=np.float64)
        result = emb.copy()
        for ch in range(result.shape[-1]):
            p_low, p_high = float(bounds[ch, 0]), float(bounds[ch, 1])
            channel = result[:, :, ch]
            channel = channel - p_low
            channel = np.clip(channel, 0.0, None)
            channel = channel / (p_high - p_low + 1e-8)
            result[:, :, ch] = np.clip(channel, 0.0, 1.0)
        stretched = np.float32(result)
    else:
        stretched = np.float32(
            normalize(
                emb,
                low=float(percentile_low),
                high=float(percentile_high),
                mask=mask,
            )
        )
    rgb = np.uint8(np.clip(stretched * 255.0, 0, 255))
    rgb[~mask] = 0
    if lab_to_rgb and emb.shape[-1] == 3:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_LAB2RGB)
        rgb[~mask] = 0
    return rgb


class SparseConceptDistiller:
    """K sparse m/z concepts + router + linear MiCS embedding readout."""

    def __init__(
        self,
        n_mz: int,
        cfg: ConceptDistillConfig,
        *,
        concepts_init: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.n_mz = int(n_mz)
        self.device = torch.device(cfg.device)
        k = int(cfg.n_concepts)
        if concepts_init is not None:
            c0 = np.asarray(concepts_init, dtype=np.float32)
            if c0.shape != (k, self.n_mz):
                raise ValueError(f"concepts_init shape {c0.shape} != ({k}, {self.n_mz})")
        else:
            c0 = np.random.default_rng(cfg.seed).random((k, self.n_mz), dtype=np.float32)
        self.concepts = nn.Parameter(
            torch.from_numpy(np.maximum(c0, 0.0)).to(self.device),
            requires_grad=True,
        )
        self.embed_w = nn.Parameter(torch.randn(3, k, device=self.device) * 0.05)
        self.router_conv: nn.Conv2d | None = None
        if str(cfg.router_mode) == "spatial_conv":
            kernel = max(1, int(cfg.router_conv_kernel))
            pad = kernel // 2
            self.router_conv = nn.Conv2d(k, k, kernel_size=kernel, padding=pad, bias=True).to(self.device)
            nn.init.eye_(self.router_conv.weight[:, :, kernel // 2, kernel // 2])
            if self.router_conv.bias is not None:
                nn.init.zeros_(self.router_conv.bias)

        self._display_channel_bounds: np.ndarray | None = None
        self._readout_alpha: torch.Tensor | None = None
        self._readout_target: torch.Tensor | None = None
        self._spec_hwm: torch.Tensor | None = None
        self._teacher_hwc: torch.Tensor | None = None
        self._mask_hw: torch.Tensor | None = None
        self._mics_sampled_x: torch.Tensor | None = None
        self._mics_ref_x: torch.Tensor | None = None
        self._mics_euclidean: torch.Tensor | None = None
        self._mics_step_counter: int = 0

    def set_display_bounds_from_embedding(
        self,
        embedding_hwc: np.ndarray,
        tissue_mask: np.ndarray,
    ) -> None:
        self._display_channel_bounds = embedding_channel_bounds(
            embedding_hwc,
            tissue_mask,
            percentile_low=float(self.cfg.percentile_low),
            percentile_high=float(self.cfg.percentile_high),
        )

    @staticmethod
    def init_concepts_nmf(spectra: np.ndarray, cfg: ConceptDistillConfig) -> np.ndarray:
        """Nonnegative sparse concept dictionary from tissue spectra (N, M)."""
        x = np.asarray(spectra, dtype=np.float64)
        x = np.maximum(x, 0.0)
        nmf = NMF(
            n_components=int(cfg.n_concepts),
            init="nndsvda",
            max_iter=int(cfg.nmf_max_iter),
            random_state=int(cfg.seed),
        )
        w = nmf.fit_transform(x)
        h = np.asarray(nmf.components_, dtype=np.float32)
        del w
        row_max = h.max(axis=1, keepdims=True)
        h = h / np.maximum(row_max, 1e-8)
        return h

    def _concept_scores_flat(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(F.linear(x, self.concepts))

    def _spatial_mix_scores(self, scores_khw: torch.Tensor) -> torch.Tensor:
        if self.router_conv is None:
            return scores_khw
        return torch.relu(self.router_conv(scores_khw.unsqueeze(0))).squeeze(0)

    def _router_alpha_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        return _sparse_topk_softmax(scores, self.cfg.top_k, self.cfg.router_temperature)

    def _router_alpha(self, x: torch.Tensor) -> torch.Tensor:
        scores = self._concept_scores_flat(x)
        return self._router_alpha_from_scores(scores)

    def _embedding_from_scores_hwk(self, scores_hwk: torch.Tensor) -> torch.Tensor:
        alpha = self._router_alpha_from_scores(scores_hwk)
        return alpha @ self.embed_w.t()

    def _embedding_from_spec(self, spec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """spec (H,W,M), mask (H,W) bool -> embedding (H,W,3)."""
        h, w, m = spec.shape
        flat = spec.reshape(-1, m)
        scores = self._concept_scores_flat(flat).reshape(h, w, -1)
        if str(self.cfg.router_mode) == "spatial_conv":
            scores_khw = scores.permute(2, 0, 1)
            scores_khw = self._spatial_mix_scores(scores_khw)
            scores = scores_khw.permute(1, 2, 0)
        return self._embedding_from_scores_hwk(scores)

    def _raw_embedding_from_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        return alpha @ self.embed_w.t()

    def _raw_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self._raw_embedding_from_alpha(self._router_alpha(x))

    def _trainable_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.concepts, self.embed_w]
        if self.router_conv is not None:
            params.extend(list(self.router_conv.parameters()))
        return params

    def _concept_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.concepts]
        if self.router_conv is not None:
            params.extend(list(self.router_conv.parameters()))
        return params

    def _subsample_rows(
        self,
        x_np: np.ndarray,
        t_np: np.ndarray,
        *,
        max_pixels: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = x_np.shape[0]
        cap = int(max_pixels)
        if cap <= 0 or n <= cap:
            return x_np, t_np
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=cap, replace=False)
        return x_np[idx], t_np[idx]

    def _collect_alpha_target_spatial(
        self,
        *,
        max_pixels: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._spec_hwm is None or self._teacher_hwc is None or self._mask_hw is None:
            raise RuntimeError("Spatial readout cache requires image tensors from fit().")
        spec = self._spec_hwm
        teacher = self._teacher_hwc
        mask = self._mask_hw
        with torch.inference_mode():
            scores = self._concept_scores_flat(spec.reshape(-1, spec.shape[-1])).reshape(
                spec.shape[0], spec.shape[1], -1
            )
            if str(self.cfg.router_mode) == "spatial_conv":
                scores = self._spatial_mix_scores(scores.permute(2, 0, 1)).permute(1, 2, 0)
            alpha_full = self._router_alpha_from_scores(scores).reshape(-1, scores.shape[-1])
        flat_mask = mask.reshape(-1)
        target = teacher.reshape(-1, 3)
        idx = torch.from_numpy(np.flatnonzero(flat_mask.detach().cpu().numpy())).to(self.device)
        if idx.numel() > int(max_pixels):
            rng = np.random.default_rng(seed)
            pick = rng.choice(int(idx.numel()), size=int(max_pixels), replace=False)
            idx = idx[pick]
        alpha = alpha_full.index_select(0, idx)
        target = target.index_select(0, idx)
        return alpha, target

    def _collect_alpha_target(
        self,
        x_np: np.ndarray,
        t_np: np.ndarray,
        *,
        max_pixels: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if str(self.cfg.router_mode) == "spatial_conv" and self._spec_hwm is not None:
            return self._collect_alpha_target_spatial(max_pixels=max_pixels, seed=seed)
        x_sub, t_sub = self._subsample_rows(x_np, t_np, max_pixels=max_pixels, seed=seed)
        bs = max(8192, int(self.cfg.batch_size))
        alphas: list[torch.Tensor] = []
        self.eval()
        with torch.inference_mode():
            for start in range(0, x_sub.shape[0], bs):
                x = torch.from_numpy(x_sub[start : start + bs]).to(self.device, non_blocking=True)
                alphas.append(self._router_alpha(x))
        alpha = torch.cat(alphas, dim=0)
        target = torch.from_numpy(np.asarray(t_sub, dtype=np.float32)).to(self.device, non_blocking=True)
        return alpha, target

    def _prepare_readout_cache(self, x_np: np.ndarray, t_np: np.ndarray, *, seed: int) -> None:
        cfg = self.cfg
        alpha, target = self._collect_alpha_target(
            x_np,
            t_np,
            max_pixels=int(cfg.readout_refine_pixels),
            seed=seed + 17,
        )
        self._readout_alpha = alpha
        self._readout_target = target

    def fit_readout_ridge(
        self,
        x_np: np.ndarray,
        t_np: np.ndarray,
        *,
        seed: int,
    ) -> dict[str, float]:
        """Closed-form ridge fit for visualization weights ``embed_w`` at fixed router."""
        cfg = self.cfg
        if self._readout_alpha is None or self._readout_target is None:
            alpha, target = self._collect_alpha_target(
                x_np,
                t_np,
                max_pixels=int(cfg.readout_ridge_pixels),
                seed=seed,
            )
        else:
            alpha = self._readout_alpha
            target = self._readout_target
        k = alpha.shape[1]
        lam = float(cfg.readout_ridge_lambda)
        ata = alpha.t() @ alpha + lam * torch.eye(k, device=alpha.device, dtype=alpha.dtype)
        wt = torch.linalg.solve(ata, alpha.t() @ target)
        with torch.no_grad():
            self.embed_w.copy_(wt.t())
        pred = alpha @ wt
        mse = float(F.mse_loss(pred, target).item())
        return {"readout_ridge_mse": mse, "readout_ridge_pixels": float(alpha.shape[0])}

    def refine_readout(
        self,
        x_np: np.ndarray,
        t_np: np.ndarray,
        *,
        seed: int,
    ) -> dict[str, float]:
        """Freeze concepts/router; tune visualization weights only."""
        cfg = self.cfg
        epochs = int(cfg.readout_epochs)
        if epochs <= 0:
            return {"readout_refine_mse": float("nan"), "readout_refine_epochs": 0.0}

        if self._readout_alpha is None or self._readout_target is None:
            self._prepare_readout_cache(x_np, t_np, seed=seed)

        alpha = self._readout_alpha
        target = self._readout_target
        assert alpha is not None and target is not None

        for p in self._concept_params():
            p.requires_grad_(False)
        self.embed_w.requires_grad_(True)
        opt = torch.optim.Adam([self.embed_w], lr=float(cfg.readout_lr))
        n = alpha.shape[0]
        bs = max(512, int(cfg.batch_size))
        steps = max(1, int(cfg.readout_steps_per_epoch))
        rng = np.random.default_rng(seed + 17)
        last_mse = float("nan")

        for _ep in range(epochs):
            for _step in range(steps):
                idx = rng.choice(n, size=min(bs, n), replace=False)
                idx_t = torch.from_numpy(idx).to(self.device)
                pred = self._raw_embedding_from_alpha(alpha.index_select(0, idx_t))
                loss = F.mse_loss(pred, target.index_select(0, idx_t))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                last_mse = float(loss.detach().cpu())

        for p in self._concept_params():
            p.requires_grad_(True)
        return {
            "readout_refine_mse": last_mse,
            "readout_refine_epochs": float(epochs),
            "readout_refine_steps_per_epoch": float(steps),
            "readout_refine_pixels": float(n),
        }

    def _sample_crop(
        self,
        spec: torch.Tensor,
        teacher: torch.Tensor,
        mask: torch.Tensor,
        *,
        crop: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = spec.shape[:2]
        crop = min(int(crop), h, w)
        y0 = int(rng.integers(0, max(1, h - crop + 1)))
        x0 = int(rng.integers(0, max(1, w - crop + 1)))
        sl_y = slice(y0, y0 + crop)
        sl_x = slice(x0, x0 + crop)
        return spec[sl_y, sl_x], teacher[sl_y, sl_x], mask[sl_y, sl_x]

    @staticmethod
    def _select_reference_indices(data: np.ndarray, n_points: int, *, sampling: str, seed: int) -> np.ndarray:
        n = int(data.shape[0])
        n_points = max(1, min(int(n_points), n))
        method = str(sampling).lower()
        rng = np.random.RandomState(int(seed))
        if method == "random":
            return rng.choice(n, n_points, replace=False)
        if method == "kmeans++":
            _, indices = kmeans_plusplus(data, n_clusters=n_points, random_state=int(seed))
            return np.asarray(indices, dtype=np.int64)
        if method == "kmeans":
            kmeans = KMeans(n_clusters=n_points, random_state=int(seed), n_init="auto").fit(data)
            return pairwise_distances(data, kmeans.cluster_centers_).argmin(axis=0).astype(np.int64)
        if method == "coreset":
            u = np.mean(data, axis=0)
            q = np.linalg.norm(data - u, axis=1) ** 2
            q_sum = float(np.sum(q))
            d = q / q_sum
            q_mix = 0.5 * (d + 1.0 / n)
            return rng.choice(n, n_points, replace=False, p=q_mix)
        raise ValueError(f"Unknown mics sampling method: {sampling!r}")

    @staticmethod
    def _rank_distance_matrix(sampled_data: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
        chebyshev = pairwise_distances(sampled_data, reference_points, metric="chebyshev")
        chebyshev = chebyshev.argsort().argsort()
        euclidean = pairwise_distances(sampled_data, reference_points, metric="cosine")
        euclidean = euclidean.argsort().argsort()
        return ((euclidean + chebyshev) / 2.0).astype(np.float32)

    def prepare_mics_training(
        self,
        spec_image: np.ndarray,
        *,
        valid_mask: np.ndarray,
        cache: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Build MiCS-style sampled pixels, coreset refs, and rank distance matrix."""
        cfg = self.cfg
        spec = np.asarray(spec_image, dtype=np.float32)
        mask = np.asarray(valid_mask, dtype=bool)
        flat = spec.reshape(-1, spec.shape[-1])
        h, w = mask.shape

        if cache is not None:
            sample_flat = np.asarray(cache["sampled_flat_indices"], dtype=np.int64).ravel()
            ref_idx = np.asarray(cache["indices"], dtype=np.int64).ravel()
            euc = np.asarray(cache["euclidean"], dtype=np.float32)
            if sample_flat.max() >= flat.shape[0] or sample_flat.min() < 0:
                raise ValueError("mics_cache sampled_flat_indices out of range for spec_image")
            sampled = flat[sample_flat]
            if euc.shape[0] != sampled.shape[0] or euc.shape[1] != ref_idx.size:
                raise ValueError(
                    f"mics_cache euclidean shape {euc.shape} != ({sampled.shape[0]}, {ref_idx.size})"
                )
        else:
            valid_flat = np.flatnonzero(mask.ravel())
            rng = np.random.default_rng(int(cfg.seed))
            n_sample = min(int(cfg.mics_num_samples), valid_flat.size)
            if valid_flat.size > n_sample:
                pick = rng.choice(valid_flat.size, size=n_sample, replace=False)
                sample_flat = valid_flat[pick]
            else:
                sample_flat = valid_flat
            sampled = flat[sample_flat]
            ref_idx = self._select_reference_indices(
                sampled,
                int(cfg.mics_number_of_points),
                sampling=str(cfg.mics_sampling),
                seed=int(cfg.seed),
            )
            euc = self._rank_distance_matrix(sampled, sampled[ref_idx])

        self._mics_sampled_x = torch.from_numpy(sampled).to(self.device)
        self._mics_ref_x = self._mics_sampled_x.index_select(
            0, torch.from_numpy(ref_idx).long().to(self.device)
        )
        self._mics_euclidean = torch.from_numpy(euc).to(self.device)
        self._mics_step_counter = 0
        return {
            "mics_n_samples": float(sampled.shape[0]),
            "mics_n_ref": float(ref_idx.size),
        }

    def _reference_embeddings(self) -> torch.Tensor:
        if self._mics_ref_x is None:
            raise RuntimeError("prepare_mics_training must run before MiCS objective steps")
        return self._raw_embedding(self._mics_ref_x)

    def _diversity_loss(self) -> torch.Tensor:
        cfg = self.cfg
        loss = torch.zeros((), device=self.device, dtype=torch.float32)
        k = int(self.concepts.shape[0])
        eye = torch.eye(k, device=self.device, dtype=torch.float32)
        ew = float(cfg.embed_diversity_weight)
        if ew > 0.0:
            w = F.normalize(self.embed_w, dim=0, eps=1e-8)
            gram = w.transpose(0, 1) @ w
            loss = loss + ew * (gram - eye).pow(2).mean()
        cw = float(cfg.concept_diversity_weight)
        if cw > 0.0:
            c = F.normalize(self.concepts, dim=1, eps=1e-8)
            gram = c @ c.transpose(0, 1)
            loss = loss + cw * (gram - eye).pow(2).mean()
        return loss

    def _regularization_loss(self, alpha: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        loss = torch.zeros((), device=self.device, dtype=torch.float32)
        if float(cfg.sparsity_weight) > 0.0:
            loss = loss + float(cfg.sparsity_weight) * alpha.abs().mean()
        if float(cfg.recon_weight) > 0.0:
            loss = loss + float(cfg.recon_weight) * F.mse_loss(alpha @ self.concepts, x)
        if float(cfg.concept_l1_weight) > 0.0:
            loss = loss + float(cfg.concept_l1_weight) * self.concepts.abs().mean()
        loss = loss + self._diversity_loss()
        return loss

    def _mics_correlation_loss(self, batch_outputs: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        ref_emb = self._reference_embeddings()
        low_d = torch.cdist(batch_outputs, ref_emb)
        high_d = self._mics_euclidean.index_select(0, batch_mask)
        corr_loss = -correlation(low_d, high_d).mean()
        alpha_scale = -float(corr_loss.detach().cpu().item())
        alpha_scale = max(alpha_scale, 1e-6)
        return corr_loss * float(cfg.mics_beta) / alpha_scale

    def _fit_mics_objective(
        self,
        *,
        teacher_embedding: np.ndarray | None = None,
        spectra: np.ndarray | None = None,
    ) -> dict[str, float]:
        cfg = self.cfg
        if self._mics_sampled_x is None:
            raise RuntimeError("prepare_mics_training required for MiCS objective")
        assert self._mics_euclidean is not None

        sampled_x = self._mics_sampled_x
        n = int(sampled_x.shape[0])
        bs = max(512, min(int(cfg.batch_size), n))
        steps = max(1, int(cfg.steps_per_epoch))
        rng = np.random.default_rng(int(cfg.seed))
        opt = torch.optim.Adam(
            [
                {"params": self._concept_params(), "lr": float(cfg.lr)},
                {"params": [self.embed_w], "lr": float(cfg.embed_w_lr)},
            ]
        )
        use_teacher = (
            teacher_embedding is not None
            and spectra is not None
            and float(cfg.mics_teacher_blend) > 0.0
        )
        teacher_t: torch.Tensor | None = None
        fit_x: torch.Tensor | None = None
        if use_teacher:
            fit_x = torch.from_numpy(np.asarray(spectra, dtype=np.float32)).to(self.device)
            teacher_t = torch.from_numpy(np.asarray(teacher_embedding, dtype=np.float32)).to(self.device)

        last_corr = 0.0
        last_emb = 0.0
        k_epoch = max(1, int(cfg.mics_k_epoch))

        for _ep in range(int(cfg.epochs)):
            order = rng.permutation(n)
            for _step in range(steps):
                start = (_step * bs) % n
                batch_idx = order[start : start + bs]
                if batch_idx.size == 0:
                    continue
                idx_t = torch.from_numpy(batch_idx).long().to(self.device)
                x = sampled_x.index_select(0, idx_t)
                pred = self._raw_embedding(x)
                alpha = self._router_alpha(x)
                loss = torch.zeros((), device=self.device, dtype=torch.float32)

                if self._mics_step_counter % k_epoch == k_epoch - 1:
                    mics_term = self._mics_correlation_loss(pred, idx_t)
                    loss = loss + mics_term
                    last_corr = float(mics_term.detach().cpu())

                if use_teacher and fit_x is not None and teacher_t is not None:
                    t_idx = rng.choice(fit_x.shape[0], size=min(bs, fit_x.shape[0]), replace=False)
                    t_idx_t = torch.from_numpy(t_idx).to(self.device)
                    x_t = fit_x.index_select(0, t_idx_t)
                    target = teacher_t.index_select(0, t_idx_t)
                    pred_t = self._raw_embedding(x_t)
                    emb_loss = F.mse_loss(pred_t, target)
                    loss = loss + float(cfg.mics_teacher_blend) * emb_loss
                    last_emb = float(emb_loss.detach().cpu())

                reg = self._regularization_loss(alpha, x)
                if reg.requires_grad or loss.requires_grad:
                    loss = loss + reg
                elif not loss.requires_grad:
                    loss = loss + pred.sum() * 0.0 + reg

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    self.concepts.clamp_(min=0.0)
                self._mics_step_counter += 1

        return {
            "mics_correlation_loss": last_corr,
            "teacher_blend_mse": last_emb,
            "mics_steps": float(self._mics_step_counter),
        }

    def fit(
        self,
        spectra: np.ndarray,
        teacher_embedding: np.ndarray | None = None,
        *,
        valid_mask: np.ndarray | None = None,
        teacher_display_embedding: np.ndarray | None = None,
        spec_image: np.ndarray | None = None,
        teacher_image: np.ndarray | None = None,
        mask_image: np.ndarray | None = None,
        mics_cache: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Train sparse concepts with MiCS distillation or MiCS correlation objective."""
        cfg = self.cfg
        objective = str(cfg.train_objective).lower()
        use_mics = objective in {"mics", "hybrid"}
        use_distill = objective in {"distill", "hybrid"}

        if use_distill and teacher_embedding is None:
            raise ValueError("teacher_embedding required for distill/hybrid objectives")
        if use_mics and spec_image is None:
            raise ValueError("spec_image required for mics/hybrid objectives")

        spec = np.asarray(spectra, dtype=np.float32)
        if spec.ndim == 3:
            h, w, m = spec.shape
            mask = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else spec.sum(axis=-1) > 0
            x_np = spec[mask].reshape(-1, m)
            t_np = (
                np.asarray(teacher_embedding, dtype=np.float32)[mask].reshape(-1, 3)
                if teacher_embedding is not None
                else None
            )
        else:
            x_np = spec.reshape(-1, spec.shape[-1])
            t_np = (
                np.asarray(teacher_embedding, dtype=np.float32).reshape(-1, 3)
                if teacher_embedding is not None
                else None
            )
            mask = None

        if x_np.shape[0] < cfg.n_concepts:
            raise ValueError(f"Need at least {cfg.n_concepts} tissue pixels, got {x_np.shape[0]}")

        if spec_image is not None and mask_image is not None:
            self._spec_hwm = torch.from_numpy(np.asarray(spec_image, dtype=np.float32)).to(self.device)
            self._mask_hw = torch.from_numpy(np.asarray(mask_image, dtype=bool)).to(self.device)
            if teacher_image is not None:
                self._teacher_hwc = torch.from_numpy(np.asarray(teacher_image, dtype=np.float32)).to(self.device)

        if str(cfg.display_bounds) == "teacher" and teacher_display_embedding is not None:
            disp_emb = np.asarray(teacher_display_embedding, dtype=np.float32)
            if disp_emb.ndim == 3 and mask is not None:
                disp_mask = mask
            elif disp_emb.ndim == 3:
                disp_mask = disp_emb.sum(axis=-1) > 0
            else:
                disp_mask = np.ones(disp_emb.shape[:2], dtype=bool)
            self.set_display_bounds_from_embedding(disp_emb, disp_mask)

        stats: dict[str, float] = {"n_pixels": float(x_np.shape[0])}

        if use_mics:
            assert spec_image is not None and mask_image is not None
            stats.update(
                self.prepare_mics_training(
                    spec_image,
                    valid_mask=mask_image,
                    cache=mics_cache,
                )
            )
            stats.update(
                self._fit_mics_objective(
                    teacher_embedding=t_np,
                    spectra=x_np if use_distill or float(cfg.mics_teacher_blend) > 0.0 else None,
                )
            )
            stats["train_objective"] = 2.0 if objective == "hybrid" else 1.0

        if use_distill and not use_mics:
            assert t_np is not None
            stats.update(
                self._fit_distill_objective(
                    x_np,
                    t_np,
                    mask=mask,
                )
            )
            stats["train_objective"] = 0.0

        if use_distill and not use_mics and t_np is not None and bool(cfg.readout_ridge_fit):
            self._prepare_readout_cache(x_np, t_np, seed=int(cfg.seed))
            stats.update(self.fit_readout_ridge(x_np, t_np, seed=int(cfg.seed)))
            if int(cfg.readout_epochs) > 0:
                stats.update(self.refine_readout(x_np, t_np, seed=int(cfg.seed)))
        elif use_mics and str(cfg.display_bounds) == "self" and spec_image is not None and mask_image is not None:
            emb = self.predict_embedding(spec_image, valid_mask=mask_image)
            self.set_display_bounds_from_embedding(emb, mask_image)

        return stats

    def _fit_distill_objective(
        self,
        x_np: np.ndarray,
        t_np: np.ndarray,
        *,
        mask: np.ndarray | None,
        blend_weight: float = 1.0,
    ) -> dict[str, float]:
        """Distill teacher cumulative embedding (pre percentile / LAB→RGB)."""
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        opt = torch.optim.Adam(
            [
                {"params": self._concept_params(), "lr": float(cfg.lr)},
                {"params": [self.embed_w], "lr": float(cfg.embed_w_lr)},
            ]
        )
        n = x_np.shape[0]
        bs = max(512, int(cfg.batch_size))
        steps = max(1, int(cfg.steps_per_epoch))
        last_emb_loss = 0.0
        use_recon = float(cfg.recon_weight) > 0.0
        spatial = str(cfg.router_mode) == "spatial_conv" and self._spec_hwm is not None

        if spatial:
            spec_t = self._spec_hwm
            teacher_t = self._teacher_hwc
            mask_t = self._mask_hw
            assert spec_t is not None and teacher_t is not None and mask_t is not None
            crop = int(cfg.spatial_crop_size)
            for _ep in range(int(cfg.epochs)):
                for _step in range(steps):
                    crop_spec, crop_teacher, crop_mask = self._sample_crop(
                        spec_t, teacher_t, mask_t, crop=crop, rng=rng
                    )
                    if not torch.any(crop_mask):
                        continue
                    pred = self._embedding_from_spec(crop_spec, crop_mask)
                    target = crop_teacher
                    m = crop_mask
                    emb_loss = F.mse_loss(pred[m], target[m])
                    ch, cw, _ = crop_spec.shape
                    scores = self._concept_scores_flat(crop_spec.reshape(-1, crop_spec.shape[-1])).reshape(
                        ch, cw, -1
                    )
                    if self.router_conv is not None:
                        scores = self._spatial_mix_scores(scores.permute(2, 0, 1)).permute(1, 2, 0)
                    alpha = self._router_alpha_from_scores(scores).reshape(-1, scores.shape[-1])
                    loss = float(blend_weight) * emb_loss + self._regularization_loss(alpha, crop_spec.reshape(-1, crop_spec.shape[-1]))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        self.concepts.clamp_(min=0.0)
                    last_emb_loss = float(emb_loss.detach().cpu())
        else:
            fit_x = torch.from_numpy(x_np).to(self.device)
            fit_t = torch.from_numpy(t_np).to(self.device)
            for _ep in range(int(cfg.epochs)):
                for _step in range(steps):
                    idx = rng.choice(n, size=min(bs, n), replace=False)
                    idx_t = torch.from_numpy(idx).to(self.device)
                    x = fit_x.index_select(0, idx_t)
                    target = fit_t.index_select(0, idx_t)
                    pred = self._raw_embedding(x)
                    emb_loss = F.mse_loss(pred, target)
                    alpha = self._router_alpha(x)
                    loss = float(blend_weight) * emb_loss + self._regularization_loss(alpha, x)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        self.concepts.clamp_(min=0.0)
                    last_emb_loss = float(emb_loss.detach().cpu())

        return {
            "embedding_mse": last_emb_loss,
            "epochs": float(cfg.epochs),
            "steps_per_epoch": float(steps),
            "router_mode": 1.0 if spatial else 0.0,
        }

    def predict_embedding(self, msi: np.ndarray, *, valid_mask: np.ndarray | None = None) -> np.ndarray:
        spec = np.asarray(msi, dtype=np.float32)
        h, w, m = spec.shape
        mask = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else spec.sum(axis=-1) > 0

        self.eval()
        with torch.inference_mode():
            if str(self.cfg.router_mode) == "spatial_conv":
                spec_t = torch.from_numpy(spec).to(self.device)
                mask_t = torch.from_numpy(mask).to(self.device)
                emb = self._embedding_from_spec(spec_t, mask_t).cpu().numpy()
            else:
                flat = spec.reshape(-1, m)
                out = np.zeros((h * w, 3), dtype=np.float32)
                idx = np.flatnonzero(mask.ravel())
                bs = max(8192, int(self.cfg.batch_size))
                for start in range(0, idx.size, bs):
                    sel = idx[start : start + bs]
                    x = torch.from_numpy(flat[sel]).to(self.device, non_blocking=True)
                    out[sel] = self._raw_embedding(x).cpu().numpy()
                emb = out.reshape(h, w, 3)

        sigma = float(self.cfg.predict_embedding_smooth_sigma)
        if sigma > 0.0:
            emb = _spatial_gaussian_smooth_channels(emb, sigma)
            emb[~mask] = 0.0
        return emb

    def predict_rgb(self, msi: np.ndarray, *, valid_mask: np.ndarray | None = None) -> np.ndarray:
        cfg = self.cfg
        mask = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else msi.sum(axis=-1) > 0
        emb = self.predict_embedding(msi, valid_mask=mask)
        bounds = self._display_channel_bounds if str(cfg.display_bounds) == "teacher" else None
        return embedding_to_display_rgb(
            emb,
            tissue_mask=mask,
            lab_to_rgb=bool(cfg.lab_to_rgb),
            percentile_low=float(cfg.percentile_low),
            percentile_high=float(cfg.percentile_high),
            channel_bounds=bounds,
        )

    def concept_mz_summary(self, top_peaks: int = 5) -> list[dict[str, Any]]:
        """Top m/z indices per concept (for inspection / manifest)."""
        c = np.asarray(self.concepts.detach().cpu().numpy(), dtype=np.float64)
        out: list[dict[str, Any]] = []
        for ki in range(c.shape[0]):
            row = c[ki]
            idx = np.argsort(-row)[: max(1, int(top_peaks))]
            out.append(
                {
                    "concept": int(ki),
                    "top_mz_indices": [int(i) for i in idx.tolist()],
                    "top_weights": [float(row[i]) for i in idx],
                }
            )
        return out

    def eval(self) -> None:
        for p in self._trainable_params():
            p.requires_grad_(False)

    def train(self) -> None:
        for p in self._trainable_params():
            p.requires_grad_(True)
