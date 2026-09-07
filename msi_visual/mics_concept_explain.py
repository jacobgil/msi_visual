"""Homogeneous-region color explanations from sparse MiCS concept weights."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

from msi_visual.mics_concept_distill import (
    _sparse_topk_softmax,
    embedding_channel_bounds,
    embedding_to_display_rgb,
)
from msi_visual.mics_concept_viz import _all_concept_rgbs

logger = logging.getLogger(__name__)

FitMethod = Literal["backprop", "nnls", "mics_input"]
WeightMode = Literal["sparse_softmax", "softmax", "entmax15", "hard_concrete"]


@dataclass
class ConceptExplainConfig:
    n_concepts: int = 32
    top_k: int = 4
    nmf_max_iter: int = 200
    nmf_init_pixels: int = 12000
    ridge_lambda: float = 1e-2
    normalize_alpha: bool = False
    lab_to_rgb: bool = True
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    nnls_batch_size: int = 8192
    fit_method: FitMethod = "backprop"
    weight_mode: WeightMode = "sparse_softmax"
    softmax_temperature: float = 1.0
    train_epochs: int = 300
    train_lr: float = 0.05
    sparsity_weight: float = 0.0
    log_every: int = 50
    mics_input_batch_size: int = 2048
    mics_input_tic_normalize: bool = True
    hard_concrete_l0_weight: float = 0.0
    hard_concrete_temperature: float = 2.0 / 3.0
    hard_concrete_gamma: float = -0.1
    hard_concrete_zeta: float = 1.1
    hard_concrete_init_prob: float = 0.1
    device: str = "cpu"
    seed: int = 42


@dataclass
class FrozenConceptExplainer:
    """Fixed NMF concepts + sparse per-pixel concept weights for MiCS explanations."""

    concepts: np.ndarray
    embed_w: np.ndarray
    alpha_hwk: np.ndarray
    channel_bounds: np.ndarray
    cfg: ConceptExplainConfig
    fit_stats: dict[str, float] = field(default_factory=dict)
    reconstructed_embedding: np.ndarray | None = None

    @staticmethod
    def init_concepts_nmf(spectra: np.ndarray, cfg: ConceptExplainConfig) -> np.ndarray:
        x = np.maximum(np.asarray(spectra, dtype=np.float64), 0.0)
        nmf = NMF(
            n_components=int(cfg.n_concepts),
            init="nndsvda",
            max_iter=int(cfg.nmf_max_iter),
            random_state=int(cfg.seed),
        )
        nmf.fit(x)
        h = np.asarray(nmf.components_, dtype=np.float32)
        row_max = h.max(axis=1, keepdims=True)
        return h / np.maximum(row_max, 1e-8)

    @staticmethod
    def concept_scores(spectra: np.ndarray, concepts: np.ndarray) -> np.ndarray:
        x = np.asarray(spectra, dtype=np.float32)
        c = np.asarray(concepts, dtype=np.float32)
        return np.maximum(x @ c.T, 0.0)

    @classmethod
    def fit_ridge_readout(
        cls,
        scores: np.ndarray,
        teacher_emb: np.ndarray,
        *,
        ridge_lambda: float,
    ) -> tuple[np.ndarray, float]:
        s = np.asarray(scores, dtype=np.float64)
        t = np.asarray(teacher_emb, dtype=np.float64)
        k = s.shape[1]
        lam = float(ridge_lambda)
        ata = s.T @ s + lam * np.eye(k, dtype=np.float64)
        w_e = np.linalg.solve(ata, s.T @ t).astype(np.float32)
        pred = s @ w_e
        mse = float(np.mean((pred - t) ** 2))
        return w_e, mse

    @classmethod
    def fit_pixel_alphas(
        cls,
        scores: np.ndarray,
        teacher_emb: np.ndarray,
        embed_w: np.ndarray,
        *,
        top_k: int,
        normalize: bool,
        batch_size: int,
    ) -> tuple[np.ndarray, float]:
        s = np.asarray(scores, dtype=np.float32)
        t = np.asarray(teacher_emb, dtype=np.float32)
        w_e = np.asarray(embed_w, dtype=np.float32)
        n, k = s.shape
        top_k = max(1, min(int(top_k), k))
        alpha = np.zeros((n, k), dtype=np.float32)
        bs = max(256, int(batch_size))
        sq_err = 0.0
        for start in range(0, n, bs):
            end = min(n, start + bs)
            for i in range(start, end):
                idx = np.argpartition(-s[i], top_k - 1)[:top_k]
                a, _ = nnls(w_e[idx].T, t[i])
                alpha[i, idx] = a.astype(np.float32)
                if normalize:
                    sm = float(alpha[i].sum())
                    if sm > 1e-8:
                        alpha[i] /= sm
                pred = alpha[i] @ w_e
                sq_err += float(np.sum((pred - t[i]) ** 2))
        mse = sq_err / max(1, n * t.shape[1])
        return alpha, mse

    @staticmethod
    def _entmax15(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Differentiable entmax-1.5 transform with exact zeros and sum-to-one rows."""
        temp = max(float(temperature), 1e-3)
        x = logits / temp
        # Standard entmax-1.5 closed-form threshold algorithm.
        x = x / 2.0
        x = x - x.max(dim=-1, keepdim=True).values
        xsrt = torch.sort(x, descending=True, dim=-1).values
        d = int(x.shape[-1])
        rho_shape = [1] * x.ndim
        rho_shape[-1] = d
        rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).reshape(rho_shape)
        mean = xsrt.cumsum(dim=-1) / rho
        mean_sq = (xsrt * xsrt).cumsum(dim=-1) / rho
        ss = rho * (mean_sq - mean * mean)
        delta = torch.clamp((1.0 - ss) / rho, min=0.0)
        tau = mean - torch.sqrt(delta)
        support_size = (tau <= xsrt).sum(dim=-1, keepdim=True).clamp(min=1)
        tau_star = tau.gather(-1, support_size - 1)
        out = torch.clamp(x - tau_star, min=0.0) ** 2
        return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _alpha_from_logits(logits: torch.Tensor, cfg: ConceptExplainConfig) -> torch.Tensor:
        mode = str(cfg.weight_mode).strip().lower()
        if mode == "sparse_softmax":
            return _sparse_topk_softmax(logits, int(cfg.top_k), float(cfg.softmax_temperature))
        if mode == "softmax":
            temp = max(float(cfg.softmax_temperature), 1e-3)
            return F.softmax(logits / temp, dim=-1)
        if mode in {"entmax15", "entmax", "entmax_15"}:
            return FrozenConceptExplainer._entmax15(logits, float(cfg.softmax_temperature))
        if mode in {"hard_concrete", "l0", "l0_gates"}:
            # Fallback for old readout paths: deterministic gates from one logit tensor.
            gates = FrozenConceptExplainer._hard_concrete_gate(logits, cfg, sample=False)
            base = F.softmax(logits / max(float(cfg.softmax_temperature), 1e-3), dim=-1)
            alpha = gates * base
            return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        raise ValueError(
            "concepts.weight_mode must be sparse_softmax|softmax|entmax15|hard_concrete; "
            f"got {mode!r}"
        )

    @staticmethod
    def _hard_concrete_gate(
        gate_logits: torch.Tensor,
        cfg: ConceptExplainConfig,
        *,
        sample: bool,
    ) -> torch.Tensor:
        beta = max(float(cfg.hard_concrete_temperature), 1e-3)
        gamma = float(cfg.hard_concrete_gamma)
        zeta = float(cfg.hard_concrete_zeta)
        if sample:
            eps = torch.finfo(gate_logits.dtype).eps
            u = torch.rand_like(gate_logits).clamp(min=eps, max=1.0 - eps)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + gate_logits) / beta)
        else:
            s = torch.sigmoid(gate_logits)
        stretched = s * (zeta - gamma) + gamma
        return torch.clamp(stretched, 0.0, 1.0)

    @staticmethod
    def _hard_concrete_l0_probability(
        gate_logits: torch.Tensor,
        cfg: ConceptExplainConfig,
    ) -> torch.Tensor:
        beta = max(float(cfg.hard_concrete_temperature), 1e-3)
        gamma = float(cfg.hard_concrete_gamma)
        zeta = float(cfg.hard_concrete_zeta)
        return torch.sigmoid(gate_logits - beta * np.log(-gamma / zeta))

    @staticmethod
    def _hard_concrete_alpha(
        weight_logits: torch.Tensor,
        gate_logits: torch.Tensor,
        cfg: ConceptExplainConfig,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = F.softmax(weight_logits / max(float(cfg.softmax_temperature), 1e-3), dim=-1)
        if sample:
            gates = FrozenConceptExplainer._hard_concrete_gate(gate_logits, cfg, sample=True)
        else:
            # Use the expected nonzero probability for stable exported explanations.
            gates = FrozenConceptExplainer._hard_concrete_l0_probability(gate_logits, cfg)
        alpha = gates * base
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return alpha, gates

    @classmethod
    def fit_pixel_alphas_backprop(
        cls,
        teacher_emb: np.ndarray,
        embed_w: np.ndarray,
        valid_mask: np.ndarray,
        cfg: ConceptExplainConfig,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Learn concept logits (init zero) -> softmax weights -> match frozen MiCS embedding."""
        device = torch.device(cfg.device)
        emb_np = np.asarray(teacher_emb, dtype=np.float32)
        w_np = np.asarray(embed_w, dtype=np.float32)
        mask = np.asarray(valid_mask, dtype=bool)
        h, w, _ = emb_np.shape
        k = w_np.shape[0]

        logits = nn.Parameter(torch.zeros(h, w, k, device=device, dtype=torch.float32))
        embed_w_t = torch.from_numpy(w_np).to(device)
        teacher_t = torch.from_numpy(emb_np).to(device)
        mask_t = torch.from_numpy(mask).to(device)

        opt = torch.optim.Adam([logits], lr=float(cfg.train_lr))
        epochs = max(1, int(cfg.train_epochs))
        sparsity = float(cfg.sparsity_weight)
        log_every = max(1, int(cfg.log_every))
        last_mse = float("nan")
        last_entropy = float("nan")

        for epoch in range(epochs):
            opt.zero_grad(set_to_none=True)
            alpha = cls._alpha_from_logits(logits, cfg)
            pred = torch.einsum("hwk,kc->hwc", alpha, embed_w_t)
            diff = pred[mask_t] - teacher_t[mask_t]
            mse = (diff * diff).mean()
            loss = mse
            if sparsity > 0.0 and str(cfg.weight_mode).strip().lower() == "softmax":
                a = alpha[mask_t].clamp(min=1e-8)
                entropy = -(a * a.log()).sum(dim=-1).mean()
                loss = loss + sparsity * entropy
                last_entropy = float(entropy.detach().cpu())
            else:
                last_entropy = 0.0
            loss.backward()
            opt.step()
            last_mse = float(mse.detach().cpu())
            if epoch == 0 or (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
                logger.info(
                    "alpha backprop epoch %d/%d - emb MSE %.6f (%s, top_k=%d)",
                    epoch + 1,
                    epochs,
                    last_mse,
                    cfg.weight_mode,
                    int(cfg.top_k),
                )

        with torch.no_grad():
            alpha_np = cls._alpha_from_logits(logits, cfg).detach().cpu().numpy().astype(np.float32)
        alpha_np[~mask] = 0.0
        stats = {
            "backprop_emb_mse": last_mse,
            "backprop_entropy_reg": last_entropy,
            "train_epochs": float(epochs),
            "train_lr": float(cfg.train_lr),
            "weight_mode": str(cfg.weight_mode),
            "softmax_temperature": float(cfg.softmax_temperature),
            "top_k": float(cfg.top_k),
        }
        return alpha_np, stats

    @staticmethod
    def _tic_normalize_torch(x: torch.Tensor) -> torch.Tensor:
        return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @classmethod
    def concept_embeddings_through_mics(
        cls,
        concepts: np.ndarray,
        mics_model: nn.Module,
        cfg: ConceptExplainConfig,
    ) -> np.ndarray:
        """Embed each NMF concept spectrum by passing it through frozen MiCS."""
        device = torch.device(cfg.device)
        model = mics_model.to(device)
        model.eval()
        c = torch.from_numpy(np.asarray(concepts, dtype=np.float32)).to(device)
        if bool(cfg.mics_input_tic_normalize):
            c = cls._tic_normalize_torch(c)
        outs: list[np.ndarray] = []
        bs = max(1, min(int(cfg.mics_input_batch_size), int(c.shape[0])))
        with torch.no_grad():
            for start in range(0, int(c.shape[0]), bs):
                pred = model(c[start : start + bs])
                outs.append(pred.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    @classmethod
    def fit_pixel_alphas_through_mics(
        cls,
        msi: np.ndarray,
        teacher_emb: np.ndarray,
        valid_mask: np.ndarray,
        concepts: np.ndarray,
        mics_model: nn.Module,
        cfg: ConceptExplainConfig,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Learn sparse concept mixtures whose reconstructed spectra reproduce frozen MiCS.

        The optimized path is alpha -> reconstructed m/z spectrum -> frozen MiCS model,
        rather than alpha -> linear concept readout.
        """
        device = torch.device(cfg.device)
        model = mics_model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        mask = np.asarray(valid_mask, dtype=bool)
        h, w, _ = np.asarray(msi).shape
        tissue = np.asarray(msi[mask], dtype=np.float32)
        target = np.asarray(teacher_emb, dtype=np.float32)[mask]
        concepts_np = np.asarray(concepts, dtype=np.float32)
        scores = cls.concept_scores(tissue, concepts_np)
        init_logits = np.log1p(scores).astype(np.float32, copy=False)

        mode = str(cfg.weight_mode).strip().lower()
        logits = nn.Parameter(torch.from_numpy(init_logits).to(device))
        gate_logits: nn.Parameter | None = None
        if mode in {"hard_concrete", "l0", "l0_gates"}:
            p0 = float(np.clip(float(cfg.hard_concrete_init_prob), 1e-4, 1.0 - 1e-4))
            beta = max(float(cfg.hard_concrete_temperature), 1e-3)
            gate_offset = beta * np.log(-float(cfg.hard_concrete_gamma) / float(cfg.hard_concrete_zeta))
            # Initialize the expected nonzero probability, not the unstretched sigmoid gate.
            gate_init = np.full_like(init_logits, np.log(p0 / (1.0 - p0)) + gate_offset, dtype=np.float32)
            # Give abundant NMF concepts a small head start without making all gates active.
            gate_init += 0.25 * (init_logits - init_logits.mean(axis=1, keepdims=True))
            gate_logits = nn.Parameter(torch.from_numpy(gate_init).to(device))
        concepts_t = torch.from_numpy(concepts_np).to(device)
        target_t = torch.from_numpy(target).to(device)

        params: list[nn.Parameter] = [logits]
        if gate_logits is not None:
            params.append(gate_logits)
        opt = torch.optim.Adam(params, lr=float(cfg.train_lr))
        epochs = max(1, int(cfg.train_epochs))
        bs = max(1, int(cfg.mics_input_batch_size))
        n = int(target.shape[0])
        log_every = max(1, int(cfg.log_every))
        rng = np.random.default_rng(int(cfg.seed) + 3)
        last_mse = float("nan")
        last_l0 = float("nan")
        last_entropy = float("nan")
        entropy_modes = {"softmax", "sparse_softmax"}
        entropy_weight = float(cfg.sparsity_weight)

        for epoch in range(epochs):
            order = rng.permutation(n)
            epoch_sse = 0.0
            epoch_count = 0
            epoch_l0 = 0.0
            epoch_l0_count = 0
            epoch_entropy = 0.0
            epoch_entropy_count = 0
            for start in range(0, n, bs):
                idx_np = order[start : start + bs]
                idx = torch.from_numpy(idx_np).to(device=device, dtype=torch.long)
                opt.zero_grad(set_to_none=True)
                if gate_logits is not None:
                    gate_b = gate_logits.index_select(0, idx)
                    alpha, _gates = cls._hard_concrete_alpha(
                        logits.index_select(0, idx),
                        gate_b,
                        cfg,
                        sample=True,
                    )
                    l0_prob = cls._hard_concrete_l0_probability(gate_b, cfg)
                    l0_loss = l0_prob.sum(dim=-1).mean()
                else:
                    alpha = cls._alpha_from_logits(logits.index_select(0, idx), cfg)
                    l0_loss = None
                recon_x = alpha @ concepts_t
                if bool(cfg.mics_input_tic_normalize):
                    recon_x = cls._tic_normalize_torch(recon_x)
                pred = model(recon_x)
                tgt = target_t.index_select(0, idx)
                diff = pred - tgt
                loss = (diff * diff).mean()
                if l0_loss is not None and float(cfg.hard_concrete_l0_weight) > 0.0:
                    loss = loss + float(cfg.hard_concrete_l0_weight) * l0_loss
                if entropy_weight > 0.0 and mode in entropy_modes:
                    a = alpha.clamp_min(1e-8)
                    entropy = -(a * a.log()).sum(dim=-1).mean()
                    loss = loss + entropy_weight * entropy
                else:
                    entropy = None
                loss.backward()
                opt.step()
                epoch_sse += float((diff.detach() * diff.detach()).sum().cpu())
                epoch_count += int(diff.numel())
                if l0_loss is not None:
                    epoch_l0 += float(l0_loss.detach().cpu()) * int(idx_np.size)
                    epoch_l0_count += int(idx_np.size)
                if entropy is not None:
                    epoch_entropy += float(entropy.detach().cpu()) * int(idx_np.size)
                    epoch_entropy_count += int(idx_np.size)
            last_mse = epoch_sse / max(1, epoch_count)
            last_l0 = epoch_l0 / max(1, epoch_l0_count) if epoch_l0_count else 0.0
            last_entropy = (
                epoch_entropy / max(1, epoch_entropy_count) if epoch_entropy_count else 0.0
            )
            if epoch == 0 or (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
                logger.info(
                    (
                        "MiCS-input alpha epoch %d/%d - emb MSE %.6f "
                        "(%s, top_k=%d, bs=%d, L0 %.2f, H %.3f)"
                    ),
                    epoch + 1,
                    epochs,
                    last_mse,
                    cfg.weight_mode,
                    int(cfg.top_k),
                    bs,
                    last_l0,
                    last_entropy,
                )

        alpha_np = np.zeros((h, w, concepts_np.shape[0]), dtype=np.float32)
        recon_emb = np.zeros_like(np.asarray(teacher_emb, dtype=np.float32))
        flat_mask = np.flatnonzero(mask.ravel())
        alpha_flat = alpha_np.reshape(-1, concepts_np.shape[0])
        emb_flat = recon_emb.reshape(-1, recon_emb.shape[-1])
        with torch.no_grad():
            for start in range(0, n, bs):
                end = min(n, start + bs)
                if gate_logits is not None:
                    alpha_b, _gates = cls._hard_concrete_alpha(
                        logits[start:end],
                        gate_logits[start:end],
                        cfg,
                        sample=False,
                    )
                else:
                    alpha_b = cls._alpha_from_logits(logits[start:end], cfg)
                recon_x = alpha_b @ concepts_t
                if bool(cfg.mics_input_tic_normalize):
                    recon_x = cls._tic_normalize_torch(recon_x)
                pred = model(recon_x)
                idx_flat = flat_mask[start:end]
                alpha_flat[idx_flat] = alpha_b.detach().cpu().numpy().astype(np.float32)
                emb_flat[idx_flat] = pred.detach().cpu().numpy().astype(np.float32)

        stats = {
            "mics_input_emb_mse": last_mse,
            "train_epochs": float(epochs),
            "train_lr": float(cfg.train_lr),
            "weight_mode": str(cfg.weight_mode),
            "softmax_temperature": float(cfg.softmax_temperature),
            "sparsity_weight": float(cfg.sparsity_weight),
            "entropy": float(last_entropy),
            "top_k": float(cfg.top_k),
            "mics_input_batch_size": float(bs),
            "mics_input_tic_normalize": float(bool(cfg.mics_input_tic_normalize)),
            "hard_concrete_l0_weight": float(cfg.hard_concrete_l0_weight),
            "hard_concrete_expected_l0": float(last_l0),
            "hard_concrete_temperature": float(cfg.hard_concrete_temperature),
            "hard_concrete_gamma": float(cfg.hard_concrete_gamma),
            "hard_concrete_zeta": float(cfg.hard_concrete_zeta),
        }
        return alpha_np, recon_emb, stats

    @classmethod
    def from_mics(
        cls,
        msi: np.ndarray,
        teacher_emb: np.ndarray,
        valid_mask: np.ndarray,
        cfg: ConceptExplainConfig,
        *,
        mics_model: nn.Module | None = None,
    ) -> FrozenConceptExplainer:
        mask = np.asarray(valid_mask, dtype=bool)
        h, w, _m = msi.shape
        tissue = msi[mask]
        teacher_tissue = teacher_emb[mask]
        if tissue.shape[0] < cfg.n_concepts:
            raise ValueError(f"Need >= {cfg.n_concepts} tissue pixels, got {tissue.shape[0]}")

        nmf_n = int(cfg.nmf_init_pixels)
        if nmf_n > 0 and tissue.shape[0] > nmf_n:
            rng = np.random.default_rng(int(cfg.seed) + 1)
            pick = rng.choice(tissue.shape[0], size=nmf_n, replace=False)
            tissue_nmf = tissue[pick]
        else:
            tissue_nmf = tissue

        concepts = cls.init_concepts_nmf(tissue_nmf, cfg)
        scores_tissue = cls.concept_scores(tissue, concepts)

        fit_method = str(cfg.fit_method).strip().lower()
        reconstructed_embedding: np.ndarray | None = None
        if fit_method in {"mics_input", "through_mics", "input_backprop"}:
            if mics_model is None:
                raise ValueError("fit_method=mics_input requires a trained frozen MiCS model.")
            embed_w = cls.concept_embeddings_through_mics(concepts, mics_model, cfg)
            alpha_hwk, reconstructed_embedding, fit_extra = cls.fit_pixel_alphas_through_mics(
                msi,
                teacher_emb,
                mask,
                concepts,
                mics_model,
                cfg,
            )
            ridge_mse = float("nan")
            fit_method = "mics_input"
        else:
            embed_w, ridge_mse = cls.fit_ridge_readout(
                scores_tissue,
                teacher_tissue,
                ridge_lambda=float(cfg.ridge_lambda),
            )

        if fit_method == "backprop":
            alpha_hwk, fit_extra = cls.fit_pixel_alphas_backprop(
                teacher_emb,
                embed_w,
                mask,
                cfg,
            )
        elif fit_method == "nnls":
            alpha_tissue, nnls_mse = cls.fit_pixel_alphas(
                scores_tissue,
                teacher_tissue,
                embed_w,
                top_k=int(cfg.top_k),
                normalize=bool(cfg.normalize_alpha),
                batch_size=int(cfg.nnls_batch_size),
            )
            alpha_hwk = np.zeros((h, w, concepts.shape[0]), dtype=np.float32)
            alpha_hwk[mask] = alpha_tissue
            fit_extra = {"nnls_pixel_mse": nnls_mse}
        elif fit_method != "mics_input":
            raise ValueError(f"concepts.fit_method must be backprop|nnls; got {fit_method!r}")

        channel_bounds = embedding_channel_bounds(
            teacher_emb,
            mask,
            percentile_low=float(cfg.percentile_low),
            percentile_high=float(cfg.percentile_high),
        )
        stats = {
            "fit_method": fit_method,
            "ridge_readout_mse": ridge_mse,
            "n_concepts": float(concepts.shape[0]),
            "top_k": float(cfg.top_k),
            "n_tissue_pixels": float(tissue.shape[0]),
            **fit_extra,
        }
        return cls(
            concepts=concepts,
            embed_w=embed_w,
            alpha_hwk=alpha_hwk,
            channel_bounds=channel_bounds,
            cfg=cfg,
            fit_stats=stats,
            reconstructed_embedding=reconstructed_embedding,
        )

    def predict_embedding(self) -> np.ndarray:
        if self.reconstructed_embedding is not None:
            return np.asarray(self.reconstructed_embedding, dtype=np.float32).copy()
        return np.einsum("hwk,kc->hwc", self.alpha_hwk, self.embed_w).astype(np.float32)

    def predict_rgb(self, valid_mask: np.ndarray) -> np.ndarray:
        emb = self.predict_embedding()
        return embedding_to_display_rgb(
            emb,
            tissue_mask=valid_mask,
            lab_to_rgb=bool(self.cfg.lab_to_rgb),
            percentile_low=float(self.cfg.percentile_low),
            percentile_high=float(self.cfg.percentile_high),
            channel_bounds=self.channel_bounds,
        )

    def concept_colors(self) -> np.ndarray:
        """Per-concept display RGB using the same stretch bounds as the frozen MiCS map."""
        k = self.embed_w.shape[0]
        bounds = np.asarray(self.channel_bounds, dtype=np.float64)
        rgbs = np.zeros((k, 3), dtype=np.uint8)
        for ki in range(k):
            lab = self.embed_w[ki].reshape(1, 1, 3)
            rgbs[ki] = embedding_to_display_rgb(
                lab,
                tissue_mask=np.ones((1, 1), dtype=bool),
                lab_to_rgb=bool(self.cfg.lab_to_rgb),
                percentile_low=float(self.cfg.percentile_low),
                percentile_high=float(self.cfg.percentile_high),
                channel_bounds=bounds,
            )[0, 0]
        return rgbs

    def concept_mz_summary(self, top_peaks: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        c = np.asarray(self.concepts, dtype=np.float64)
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

    def as_gallery_student(self) -> _GalleryStudentAdapter:
        return _GalleryStudentAdapter(self)


@dataclass
class _GalleryStudentAdapter:
    """Duck-type wrapper so ``save_concept_gallery`` can reuse frozen explainer."""

    explainer: FrozenConceptExplainer

    @property
    def concepts(self):
        import torch

        return torch.from_numpy(self.explainer.concepts)

    @property
    def embed_w(self):
        import torch

        return torch.from_numpy(self.explainer.embed_w.T.copy())

    @property
    def cfg(self):
        return self.explainer.cfg

    @property
    def device(self):
        import torch

        return torch.device("cpu")


@dataclass
class RegionSpec:
    region_id: int
    label: str
    cluster_id: int
    n_pixels: int
    flat_indices: np.ndarray
    mean_emb: np.ndarray
    mean_rgb: np.ndarray
    mean_alpha: np.ndarray
    centroid_yx: tuple[int, int]
    crop_bbox: tuple[int, int, int, int]
    emb_std: float


def _rgb_color_name(rgb: np.ndarray) -> str:
    """Rough color label for explanation text."""
    r, g, b = [int(x) for x in np.asarray(rgb, dtype=np.int64).reshape(3)]
    mx = max(r, g, b)
    if mx < 40:
        return "black"
    if r > 180 and g < 80 and b < 80:
        return "red"
    if g > 180 and r < 100 and b < 100:
        return "green"
    if b > 180 and r < 100 and g < 120:
        return "blue"
    if r > 180 and g > 150 and b < 80:
        return "yellow"
    if r > 150 and g < 120 and b > 150:
        return "magenta"
    if r < 120 and g > 150 and b > 150:
        return "cyan"
    if r > 200 and g > 120 and b > 120:
        return "pink"
    if r > 120 and g > 100 and b < 80:
        return "orange"
    if abs(r - g) < 30 and abs(g - b) < 30:
        return "gray" if mx < 160 else "white"
    return f"RGB({r},{g},{b})"


def concept_colors_precomputed(
    explainer: FrozenConceptExplainer | Any,
    *,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """``(K, 3)`` uint8 RGB for each concept under shared display bounds."""
    if isinstance(explainer, FrozenConceptExplainer):
        return np.asarray(explainer.concept_colors(), dtype=np.uint8)
    return np.asarray(_all_concept_rgbs(explainer, tissue_mask=valid_mask), dtype=np.uint8)


def select_homogeneous_regions(
    teacher_emb: np.ndarray,
    mics_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_regions: int = 6,
    n_clusters: int = 12,
    min_pixels: int = 600,
    max_emb_std: float = 0.12,
    crop_size: int = 64,
    seed: int = 42,
) -> list[RegionSpec]:
    """Pick low-variance MiCS embedding clusters as homogeneous explanation regions."""
    emb = np.asarray(teacher_emb, dtype=np.float64)
    rgb = np.asarray(mics_rgb, dtype=np.uint8)
    mask = np.asarray(valid_mask, dtype=bool)
    h, w = mask.shape
    flat_idx = np.flatnonzero(mask.ravel())
    if flat_idx.size < max(min_pixels, n_clusters):
        raise ValueError(f"Not enough tissue pixels ({flat_idx.size}) for region selection.")

    tissue_emb = emb.reshape(-1, 3)[flat_idx]
    k = min(int(n_clusters), tissue_emb.shape[0])
    km = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
    labels = km.fit_predict(tissue_emb)

    candidates: list[RegionSpec] = []
    for cid in range(k):
        sel = flat_idx[labels == cid]
        if sel.size < int(min_pixels):
            continue
        pts = emb.reshape(-1, 3)[sel]
        std = float(np.mean(np.std(pts, axis=0)))
        if std > float(max_emb_std):
            continue
        mean_emb = pts.mean(axis=0)
        mean_rgb = rgb.reshape(-1, 3)[sel].mean(axis=0).astype(np.uint8)
        ys, xs = np.divmod(sel, w)
        cy = int(np.median(ys))
        cx = int(np.median(xs))
        half = max(8, int(crop_size) // 2)
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)
        candidates.append(
            RegionSpec(
                region_id=len(candidates),
                label="",
                cluster_id=int(cid),
                n_pixels=int(sel.size),
                flat_indices=sel.copy(),
                mean_emb=mean_emb.astype(np.float32),
                mean_rgb=mean_rgb,
                mean_alpha=np.zeros(1, dtype=np.float32),
                centroid_yx=(cy, cx),
                crop_bbox=(y0, y1, x0, x1),
                emb_std=std,
            )
        )

    candidates.sort(key=lambda r: (-r.n_pixels, r.emb_std))
    chosen = candidates[: max(1, int(n_regions))]
    for i, reg in enumerate(chosen):
        reg.region_id = i
        reg.label = f"Region {chr(ord('A') + i)}"
    return chosen


def attach_region_alphas(
    regions: list[RegionSpec],
    alpha_hwk: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Fill ``mean_alpha`` for each region from pixel concept weights."""
    alpha = np.asarray(alpha_hwk, dtype=np.float64)
    h, w = alpha.shape[:2]
    for reg in regions:
        sel = np.asarray(reg.flat_indices, dtype=np.int64).ravel()
        if sel.size == 0:
            continue
        ys, xs = np.divmod(sel, w)
        reg.mean_alpha = alpha[ys, xs, :].mean(axis=0).astype(np.float32)


def build_color_explanation(
    mean_alpha: np.ndarray,
    concept_rgbs: np.ndarray,
    *,
    top_n: int = 5,
    min_weight: float = 0.04,
    color_label: str | None = None,
    tissue_mean_alpha: np.ndarray | None = None,
    rank_mode: str = "absolute",
    renormalize_terms: bool | None = None,
) -> dict[str, Any]:
    """Build human-readable mix formula and approximate RGB blend.

    With dense softmax (many tiny weights), use ``rank_mode='relative'`` and
    ``renormalize_terms=False`` so labels show true mixture fractions, not
    misleading renormalized top-k shares.
    """
    alpha = np.asarray(mean_alpha, dtype=np.float64).ravel()
    rgbs = np.asarray(concept_rgbs, dtype=np.float64)
    k = alpha.shape[0]
    mode = str(rank_mode).strip().lower()
    if mode == "relative" and tissue_mean_alpha is not None:
        mu = np.asarray(tissue_mean_alpha, dtype=np.float64).ravel()
        if mu.shape[0] != k:
            raise ValueError(f"tissue_mean_alpha length {mu.shape[0]} != {k}")
        score = alpha - mu
        min_score = 0.0
    else:
        score = alpha
        min_score = float(min_weight)

    order = np.argsort(-score)
    terms: list[dict[str, Any]] = []
    for ki in order[: max(1, int(top_n))]:
        ki = int(ki)
        w = float(alpha[ki])
        if mode == "relative" and tissue_mean_alpha is not None:
            if float(score[ki]) <= min_score or w < 1e-6:
                continue
        elif w < float(min_weight):
            continue
        rgb = rgbs[ki].astype(np.uint8)
        terms.append(
            {
                "concept": ki,
                "weight": w,
                "score": float(score[ki]),
                "color_rgb": [int(x) for x in rgb.tolist()],
                "color_name": _rgb_color_name(rgb),
            }
        )
    if not terms:
        ki = int(order[0])
        terms = [
            {
                "concept": ki,
                "weight": float(alpha[ki]),
                "score": float(score[ki]),
                "color_rgb": [int(x) for x in rgbs[ki].astype(np.uint8).tolist()],
                "color_name": _rgb_color_name(rgbs[ki]),
            }
        ]

    if renormalize_terms is None:
        renormalize_terms = float(alpha.max()) >= 0.18 and int((alpha > 1e-5).sum()) <= max(6, int(top_n) + 2)

    if renormalize_terms:
        total = sum(t["weight"] for t in terms) or 1.0
        for t in terms:
            t["weight_norm"] = float(t["weight"] / total)
        blend_weights = [t["weight_norm"] for t in terms]
        formula_weights = [(t["weight_norm"], t["concept"], t["color_name"]) for t in terms]
    else:
        for t in terms:
            t["weight_norm"] = float(t["weight"])
        blend_weights = [t["weight"] for t in terms]
        formula_weights = [(t["weight"], t["concept"], t["color_name"]) for t in terms]

    label = color_label or "MiCS color"
    parts = [f"{w:.2f} × concept_{ki} ({name})" for w, ki, name in formula_weights]
    formula = f"{label} ≈ " + " + ".join(parts)
    rgb_blend = np.clip(
        sum(w * np.asarray(t["color_rgb"], dtype=np.float64) for w, t in zip(blend_weights, terms)),
        0.0,
        255.0,
    ).astype(np.uint8)
    return {
        "terms": terms,
        "formula": formula,
        "rank_mode": mode,
        "renormalize_terms": bool(renormalize_terms),
        "rgb_blend_approx": [int(x) for x in rgb_blend.tolist()],
        "rgb_blend_name": _rgb_color_name(rgb_blend),
    }


def embedding_from_alpha(
    mean_alpha: np.ndarray,
    embed_w: np.ndarray,
) -> np.ndarray:
    """Concept mix → 3D embedding (LAB space before display)."""
    a = np.asarray(mean_alpha, dtype=np.float64).ravel()
    w = np.asarray(embed_w, dtype=np.float64)
    return (a @ w.T).astype(np.float32)


def save_reconstruction_mosaic(
    mics_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    *,
    gap_px: int = 2,
) -> None:
    """Side-by-side MiCS | concept reconstruction | abs diff."""
    mask = np.asarray(valid_mask, dtype=bool)
    a = np.asarray(mics_rgb, dtype=np.float64)
    b = np.asarray(recon_rgb, dtype=np.float64)
    diff = np.clip(np.abs(a - b), 0.0, 255.0).astype(np.uint8)
    diff[~mask] = 0
    gap = max(0, int(gap_px))
    h, w = mics_rgb.shape[:2]
    canvas = np.zeros((h, 3 * w + 2 * gap, 3), dtype=np.uint8)
    for i, img in enumerate([mics_rgb, recon_rgb, diff]):
        x0 = i * (w + gap)
        canvas[:, x0 : x0 + w] = np.asarray(img, dtype=np.uint8)
    Image.fromarray(canvas).save(out_path)


def _draw_region_card(
    mics_crop: np.ndarray,
    recon_crop: np.ndarray,
    explanation: dict[str, Any],
    concept_rgbs: np.ndarray,
    *,
    card_w: int = 520,
    card_h: int = 200,
) -> np.ndarray:
    """One explanation card: crops + formula + concept swatches."""
    crop_h, crop_w = mics_crop.shape[:2]
    sw = 28
    pad = 8
    img_h = max(card_h, crop_h + 2 * pad)
    canvas = Image.new("RGB", (card_w, img_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    x = pad
    for label, crop in [("MiCS", mics_crop), ("Recon", recon_crop)]:
        draw.text((x, 2), label, fill=(200, 200, 200), font=font_sm)
        canvas.paste(Image.fromarray(crop), (x, pad + 14))
        x += crop_w + pad

    tx = x + 4
    ty = pad
    draw.text((tx, ty), explanation.get("formula", ""), fill=(240, 240, 240), font=font_sm)
    ty += 36
    for term in explanation.get("terms", [])[:5]:
        ki = int(term["concept"])
        wn = float(term["weight_norm"])
        rgb = tuple(int(x) for x in term["color_rgb"])
        draw.rectangle([tx, ty, tx + sw, ty + sw], fill=rgb, outline=(80, 80, 80))
        draw.text(
            (tx + sw + 6, ty + 4),
            f"c{ki}  {wn:.0%}  {term['color_name']}",
            fill=(220, 220, 220),
            font=font_sm,
        )
        ty += sw + 4
    return np.asarray(canvas, dtype=np.uint8)


def save_region_explanations(
    regions: list[RegionSpec],
    explanations: list[dict[str, Any]],
    mics_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    concept_rgbs: np.ndarray,
    out_dir: Path,
    *,
    overview_rgb: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write per-region cards, overview figure, and JSON manifest."""
    out_dir = Path(out_dir)
    regions_dir = out_dir / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)

    cards: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    h, w = mics_rgb.shape[:2]

    for reg, expl in zip(regions, explanations):
        y0, y1, x0, x1 = reg.crop_bbox
        mics_crop = np.asarray(mics_rgb[y0:y1, x0:x1], dtype=np.uint8)
        recon_crop = np.asarray(recon_rgb[y0:y1, x0:x1], dtype=np.uint8)
        card = _draw_region_card(mics_crop, recon_crop, expl, concept_rgbs)
        card_path = regions_dir / f"{reg.label.lower().replace(' ', '_')}_explanation.png"
        Image.fromarray(card).save(card_path)
        cards.append(card)
        records.append(
            {
                "region": reg.label,
                "cluster_id": reg.cluster_id,
                "n_pixels": reg.n_pixels,
                "centroid_yx": list(reg.centroid_yx),
                "crop_bbox": list(reg.crop_bbox),
                "mean_rgb": [int(x) for x in reg.mean_rgb.tolist()],
                "mean_rgb_name": _rgb_color_name(reg.mean_rgb),
                "emb_std": float(reg.emb_std),
                "explanation": expl,
                "card_png": str(card_path),
            }
        )

    if overview_rgb is not None:
        overview_path = out_dir / "regions_overview.png"
        Image.fromarray(np.asarray(overview_rgb, dtype=np.uint8)).save(overview_path)
    else:
        overview_path = None
        ov = mics_rgb.copy()
        draw_ov = ImageDraw.Draw(Image.fromarray(ov))
        for reg in regions:
            cy, cx = reg.centroid_yx
            r = 10
            draw_ov.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 0), width=2)
            draw_ov.text((cx + 12, cy - 8), reg.label, fill=(255, 255, 0))
        overview_path = out_dir / "regions_overview.png"
        Image.fromarray(ov).save(overview_path)

    if cards:
        cw = max(c.shape[1] for c in cards)
        ch = max(c.shape[0] for c in cards)
        n = len(cards)
        gallery = np.zeros((n * ch, cw, 3), dtype=np.uint8)
        for i, card in enumerate(cards):
            gallery[i * ch : i * ch + card.shape[0], : card.shape[1]] = card
        gallery_path = out_dir / "region_explanations.png"
        Image.fromarray(gallery).save(gallery_path)
    else:
        gallery_path = None

    summary = {
        "regions_overview_png": str(overview_path) if overview_path else None,
        "region_explanations_png": str(gallery_path) if gallery_path else None,
        "regions": records,
    }
    json_path = out_dir / "region_explanations.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote region explanations: %s (%d regions)", json_path, len(records))
    return summary


def save_concept_color_palette(
    concept_rgbs: np.ndarray,
    out_path: Path,
    *,
    cols: int = 8,
    swatch: int = 48,
) -> str:
    """Grid of precomputed concept colors."""
    rgbs = np.asarray(concept_rgbs, dtype=np.uint8)
    k = rgbs.shape[0]
    ncols = max(1, int(cols))
    nrows = int(np.ceil(k / ncols))
    gap = 2
    canvas = np.zeros(
        (nrows * (swatch + gap) - gap, ncols * (swatch + gap) - gap, 3),
        dtype=np.uint8,
    )
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    for ki in range(k):
        r, c = divmod(ki, ncols)
        x0 = c * (swatch + gap)
        y0 = r * (swatch + gap)
        tile = np.broadcast_to(rgbs[ki], (swatch, swatch, 3)).astype(np.uint8)
        img.paste(Image.fromarray(tile), (x0, y0))
        draw.text((x0 + 2, y0 + 2), str(ki), fill=(255, 255, 255), font=font)
    out_path = Path(out_path)
    img.save(out_path)
    return str(out_path)


def build_region_overview(
    mics_rgb: np.ndarray,
    regions: list[RegionSpec],
) -> np.ndarray:
    """MiCS map with region labels at centroids."""
    out = np.asarray(mics_rgb, dtype=np.uint8).copy()
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for reg in regions:
        cy, cx = reg.centroid_yx
        r = 12
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 0), width=2)
        draw.text((cx + 14, cy - 10), reg.label, fill=(255, 255, 0), font=font)
    return np.asarray(img, dtype=np.uint8)
