"""Peak-token MiCS: small transformer on top-K peaks with hybrid m/z encoding."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from msi_visual.extraction import resolve_mass_axis
from msi_visual.mics_concept_distill import embedding_to_display_rgb
from msi_visual.parametric_mics_lmc import _spatial_gaussian_smooth_channels, correlation
from msi_visual.utils import normalize

logger = logging.getLogger(__name__)


def mass_axis_is_dalton(mass_axis: np.ndarray) -> bool:
    """True when axis looks like real m/z (Da), not consecutive bin indices."""
    axis = np.asarray(mass_axis, dtype=np.float64).reshape(-1)
    if axis.size < 2:
        return False
    if np.allclose(axis, np.arange(axis.size, dtype=np.float64)):
        return False
    diffs = np.diff(axis)
    if (
        float(axis.min()) >= 0.0
        and float(axis.max()) < float(axis.size)
        and np.allclose(diffs, diffs[0])
        and abs(float(diffs[0]) - 1.0) < 1e-6
    ):
        return False
    return float(axis.max()) > 150.0 and float(axis.min()) >= 50.0


def resolve_mass_encoding_params(
    mass_min: float,
    mass_max: float,
    mass_axes: list[np.ndarray],
    *,
    window_size: float,
    mass_scale: float,
    max_windows: int = 512,
) -> tuple[float, float, bool]:
    """Pick window width / Fourier scale for Dalton vs bin-index axes."""
    span = max(float(mass_max) - float(mass_min), 1.0)
    dalton = any(mass_axis_is_dalton(ax) for ax in mass_axes)
    if dalton:
        win = float(window_size) if float(window_size) > 0.0 else 0.05
        scale = float(mass_scale) if float(mass_scale) > 0.0 else 1000.0
    else:
        win = float(window_size) if float(window_size) > 0.0 else max(8.0, span / 256.0)
        scale = float(mass_scale) if float(mass_scale) > 0.0 else span
    n_windows = int(np.ceil(span / max(win, 1e-6)))
    if n_windows > int(max_windows):
        win = span / float(max_windows)
    return win, scale, dalton


def batch_topk_peaks(
    spectra: np.ndarray,
    mass_axis: np.ndarray,
    top_k: int,
    *,
    sort_by_mass: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-K peaks per spectrum. Returns ``values (N,K)``, ``masses (N,K)``."""
    x = np.asarray(spectra, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n, m = x.shape
    k = max(1, min(int(top_k), m))
    mass = np.asarray(mass_axis, dtype=np.float32).reshape(-1)
    if mass.size != m:
        raise ValueError(f"mass_axis size {mass.size} != spectrum width {m}")
    part = np.argpartition(-x, kth=min(k - 1, m - 1), axis=1)[:, :k]
    values = np.take_along_axis(x, part, axis=1)
    masses = mass[part]
    if sort_by_mass:
        order = np.argsort(masses, axis=1)
    else:
        order = np.argsort(-values, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    masses = np.take_along_axis(masses, order, axis=1)
    return values, masses


class HybridMassEmbedding(nn.Module):
    """Continuous Fourier base f(m) + linearly interpolated window residuals r(m)."""

    def __init__(
        self,
        d_model: int,
        *,
        mass_min: float,
        mass_max: float,
        window_size: float = 0.05,
        n_fourier: int | None = None,
        mass_scale: float = 1000.0,
        max_windows: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.mass_min = float(mass_min)
        self.mass_max = float(max(mass_max, mass_min + 1e-6))
        span = self.mass_max - self.mass_min
        self.window_size = float(max(window_size, 1e-6))
        self.mass_scale = float(mass_scale)
        n_windows = max(1, int(np.ceil(span / self.window_size)))
        if n_windows > int(max_windows):
            self.window_size = span / float(max_windows)
            n_windows = int(max_windows)
        self.n_windows = n_windows
        n_fourier = int(n_fourier or max(8, d_model // 4))
        self.n_fourier = n_fourier
        self.register_buffer(
            "fourier_freqs",
            torch.exp(torch.linspace(-2.0, 2.0, n_fourier, dtype=torch.float32)),
            persistent=False,
        )
        self.fourier_proj = nn.Linear(2 * n_fourier, d_model)
        self.window_residual = nn.Embedding(self.n_windows, d_model)
        nn.init.normal_(self.window_residual.weight, std=0.02)

    def _fourier(self, mass: torch.Tensor) -> torch.Tensor:
        m = mass.unsqueeze(-1) / self.mass_scale
        angles = m * self.fourier_freqs.unsqueeze(0) * (2.0 * torch.pi)
        feat = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.fourier_proj(feat)

    def _window_blend(self, mass: torch.Tensor) -> torch.Tensor:
        pos = (mass - self.mass_min) / self.window_size
        pos = torch.clamp(pos, 0.0, float(self.n_windows - 1) - 1e-6)
        i0 = torch.floor(pos).long()
        i1 = torch.clamp(i0 + 1, max=self.n_windows - 1)
        w1 = (pos - i0.float()).unsqueeze(-1)
        r0 = self.window_residual(i0)
        r1 = self.window_residual(i1)
        return r0 * (1.0 - w1) + r1 * w1

    def forward(self, mass: torch.Tensor) -> torch.Tensor:
        return self._fourier(mass) + self._window_blend(mass)


class PeakTokenTransformer(nn.Module):
    """Top-K peak tokens -> 3D MiCS embedding."""

    def __init__(
        self,
        *,
        embed_dim: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        mass_min: float = 0.0,
        mass_max: float = 5000.0,
        window_size: float = 0.05,
        mass_scale: float = 1000.0,
        max_windows: int = 512,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.d_model = int(d_model)
        self.mass_embed = HybridMassEmbedding(
            d_model,
            mass_min=mass_min,
            mass_max=mass_max,
            window_size=window_size,
            mass_scale=mass_scale,
            max_windows=max_windows,
        )
        self.value_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.token_norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.embed_dim),
        )

    def forward(
        self,
        peak_values: torch.Tensor,
        peak_masses: torch.Tensor,
        *,
        peak_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``peak_*`` shape ``(B, K)``; returns ``(B, embed_dim)``."""
        if peak_mask is None:
            peak_mask = peak_values > 0.0
        logv = torch.log1p(torch.clamp(peak_values, min=0.0)).unsqueeze(-1)
        mass_tok = self.mass_embed(peak_masses)
        val_gate = 1.0 + torch.tanh(self.value_proj(logv))
        tok = self.token_norm(mass_tok * val_gate)
        tok = tok * peak_mask.unsqueeze(-1).to(tok.dtype)
        b = tok.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tok], dim=1)
        pad = torch.cat(
            [torch.ones(b, 1, dtype=torch.bool, device=tok.device), peak_mask],
            dim=1,
        )
        x = self.encoder(x, src_key_padding_mask=~pad)
        return self.head(x[:, 0])


def resample_spectrum_to_canonical(
    spectrum: np.ndarray,
    mass_axis: np.ndarray,
    canonical_mz: np.ndarray,
) -> np.ndarray:
    """Linear resample one spectrum onto ``canonical_mz`` (zero outside native range)."""
    src = np.asarray(mass_axis, dtype=np.float64).reshape(-1)
    canon = np.asarray(canonical_mz, dtype=np.float64).reshape(-1)
    y = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    if y.size != src.size:
        raise ValueError(f"Spectrum width {y.size} != mass axis {src.size}")
    return np.interp(canon, src, y, left=0.0, right=0.0).astype(np.float32)


def build_canonical_mz_axis(
    mass_min: float,
    mass_max: float,
    *,
    step_da: float = 5.0,
    padding_da: float = 20.0,
) -> np.ndarray:
    """Shared m/z grid for cross-dataset MiCS distances."""
    step = max(float(step_da), 1e-6)
    lo = max(0.0, float(mass_min) - float(padding_da))
    hi = float(mass_max) + float(padding_da)
    if hi <= lo:
        hi = lo + step
    n = int(np.ceil((hi - lo) / step)) + 1
    return np.linspace(lo, hi, max(2, n), dtype=np.float64)


def resample_spectra_to_canonical(
    spectra: np.ndarray,
    mass_axis: np.ndarray,
    canonical_mz: np.ndarray,
) -> np.ndarray:
    """Resample ``(N, M)`` spectra onto a common m/z axis (zero outside native range)."""
    src = np.asarray(mass_axis, dtype=np.float64).reshape(-1)
    canon = np.asarray(canonical_mz, dtype=np.float64).reshape(-1)
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != src.size:
        raise ValueError(f"Spectrum width {x.shape[1]} != mass axis {src.size}")
    out = np.empty((x.shape[0], canon.size), dtype=np.float32)
    for i in range(x.shape[0]):
        out[i] = resample_spectrum_to_canonical(x[i], src, canon)
    return out


@dataclass
class PeakTokenMiCSConfig:
    top_k: int = 128
    embed_dim: int = 3
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 192
    dropout: float = 0.05
    mass_window_size: float = 0.0
    mass_scale: float = 0.0
    max_mass_windows: int = 512
    sort_peaks_by_mass: bool = True
    num_epochs: int = 60
    batch_size: int = 512
    lr: float = 8e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    num_samples: int = 5000
    number_of_points: int = 1000
    sampling: str = "coreset"
    beta: float = 20.0
    k_epoch: int = 20
    temperature: float = 100.0
    cluster: bool = True
    clusters: str | list[int] = "16-32-64-1024"
    cluster_loss_weight: float = 1.0
    cluster_on_pca: bool = False
    cluster_pca_dims: int = 64
    lab_to_rgb: bool = True
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    predict_spatial_smooth_sigma: float = 0.0
    seed: int = 42
    device: str = "cpu"
    foundation_mode: bool = False
    canonical_mz_step: float = 5.0
    mass_range_padding_da: float = 20.0


@dataclass
class PeakTokenFitOptions:
    """Long-run / multi-slide training controls."""

    max_epochs: int | None = None
    max_hours: float | None = None
    pool_refresh_epochs: int = 0
    adaptive_sampling: str = "proportional"  # proportional | equal | sqrt
    cosine_lr: bool = True
    min_lr_ratio: float = 0.05
    release_slide_memory: bool = True
    refresh_clusters_on_pool_refresh: bool = True
    checkpoint_every_epochs: int = 0
    checkpoint_dir: Path | None = None
    on_epoch_end: Callable[[int, dict[str, float], "PeakTokenMiCSTrainer"], None] | None = None


@dataclass
class SlideRecord:
    name: str
    npy_path: str
    spectra: np.ndarray | None
    valid_mask: np.ndarray
    mass_axis: np.ndarray
    tissue_pixels: int = 0
    sampled_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


class PeakTokenMiCSTrainer:
    """Train peak-token transformer with full MiCS loss (correlation + cluster CE)."""

    def __init__(self, cfg: PeakTokenMiCSConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model: PeakTokenTransformer | None = None
        self.cluster_heads: nn.ModuleList = nn.ModuleList()
        self.cluster_labels: list[np.ndarray] = []
        self.slides: list[SlideRecord] = []
        self._sampled_spectra: np.ndarray | None = None
        self._sampled_values: np.ndarray | None = None
        self._sampled_masses: np.ndarray | None = None
        self._sampled_slide_id: np.ndarray | None = None
        self._euclidean: torch.Tensor | None = None
        self._ref_indices: np.ndarray | None = None
        self._mass_min = float("inf")
        self._mass_max = float("-inf")
        self._canonical_mz: np.ndarray | None = None
        self._mass_window_size = 0.05
        self._mass_scale = 1000.0
        self._mass_is_dalton = False
        self._step_counter = 0
        self._best_correlation = -1.0
        self._best_state: dict[str, Any] | None = None
        self._best_epoch = 0
        self._training_history: list[dict[str, float]] = []
        self._fit_start_time: float | None = None
        self._slide_loader: Callable[[SlideRecord], tuple[np.ndarray, np.ndarray]] | None = None

    def set_slide_loader(
        self,
        loader: Callable[[SlideRecord], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Optional callback to reload ``(spectra, valid_mask)`` from disk on pool refresh."""
        self._slide_loader = loader

    def _ensure_spectra_loaded(self) -> None:
        if self._slide_loader is None:
            missing = [s.name for s in self.slides if s.spectra is None]
            if missing:
                raise ValueError(
                    "Training pool refresh requires slide spectra; set_slide_loader() or "
                    f"disable release_slide_memory. Missing: {missing[:3]}"
                )
            return
        for slide in self.slides:
            if slide.spectra is None:
                spec, mask = self._slide_loader(slide)
                slide.spectra = np.asarray(spec, dtype=np.float32)
                slide.valid_mask = np.asarray(mask, dtype=bool)
                slide.tissue_pixels = int(np.count_nonzero(slide.valid_mask))

    @staticmethod
    def _normalize_clusters(clusters: str | list[int] | None) -> list[int]:
        if clusters is None:
            return []
        if isinstance(clusters, (list, tuple)):
            return [int(x) for x in clusters]
        s = str(clusters).strip()
        if not s or s.lower() in ("", "null", "none", "~"):
            return []
        if "-" in s:
            return [int(x) for x in s.split("-") if x.strip()]
        return [int(s)]

    @staticmethod
    def _clamp_n_clusters(n_clusters: int, n_samples: int) -> int:
        k = int(n_clusters)
        n = int(n_samples)
        if n <= 0:
            return max(1, k)
        return max(1, min(k, n))

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
        raise ValueError(f"Unknown sampling method: {sampling!r}")

    @staticmethod
    def _peak_feature_matrix(values: np.ndarray, masses: np.ndarray, *, mass_min: float, mass_max: float) -> np.ndarray:
        """Legacy helper; MiCS high-D distances use full spectra instead."""
        logv = np.log1p(np.maximum(np.asarray(values, dtype=np.float32), 0.0))
        span = max(float(mass_max) - float(mass_min), 1.0)
        mass_norm = (np.asarray(masses, dtype=np.float32) - float(mass_min)) / span
        return np.concatenate([logv, mass_norm], axis=1).astype(np.float32)

    @staticmethod
    def _rank_distance_matrix(sampled: np.ndarray, reference: np.ndarray) -> np.ndarray:
        cheb = pairwise_distances(sampled, reference, metric="chebyshev").argsort().argsort()
        euc = pairwise_distances(sampled, reference, metric="cosine").argsort().argsort()
        return ((cheb + euc) / 2.0).astype(np.float32)

    def add_slide(
        self,
        spectra_hwm: np.ndarray | None,
        *,
        valid_mask: np.ndarray,
        mass_axis: np.ndarray,
        name: str,
        npy_path: str | None = None,
    ) -> None:
        mask = np.asarray(valid_mask, dtype=bool)
        mass = np.asarray(mass_axis, dtype=np.float64).reshape(-1)
        if bool(self.cfg.foundation_mode) and not mass_axis_is_dalton(mass):
            raise ValueError(
                f"foundation_mode requires Dalton m/z axis for slide {name!r}; "
                "provide args.txt or mass sidecar beside the .npy"
            )
        if spectra_hwm is None:
            if self._slide_loader is None:
                raise ValueError(
                    f"Slide {name!r} registered without spectra; set_slide_loader() first"
                )
            valid_flat = np.flatnonzero(mask.ravel())
            shape = tuple(mask.shape) + (int(mass.size),)
            self.slides.append(
                SlideRecord(
                    name=str(name),
                    npy_path=str(npy_path or name),
                    spectra=None,
                    valid_mask=mask,
                    mass_axis=mass,
                    tissue_pixels=int(valid_flat.size),
                )
            )
            logger.info("Registered slide %s (lazy): shape=%s valid=%d", name, shape, valid_flat.size)
        else:
            spec = np.asarray(spectra_hwm, dtype=np.float32)
            valid_flat = np.flatnonzero(mask.ravel())
            self.slides.append(
                SlideRecord(
                    name=str(name),
                    npy_path=str(npy_path or name),
                    spectra=spec,
                    valid_mask=mask,
                    mass_axis=mass,
                    tissue_pixels=int(valid_flat.size),
                )
            )
            logger.info("Registered slide %s: shape=%s valid=%d", name, spec.shape, valid_flat.size)
        self._mass_min = min(self._mass_min, float(mass.min()))
        self._mass_max = max(self._mass_max, float(mass.max()))

    @staticmethod
    def _adaptive_slide_counts(
        tissue_counts: np.ndarray,
        total_samples: int,
        *,
        mode: str,
    ) -> np.ndarray:
        """Allocate ``total_samples`` across slides (at least 1 per slide when possible)."""
        n = int(tissue_counts.size)
        total = max(n, int(total_samples))
        counts = np.asarray(tissue_counts, dtype=np.float64)
        if str(mode).strip().lower() == "equal":
            weights = np.ones(n, dtype=np.float64)
        elif str(mode).strip().lower() == "sqrt":
            weights = np.sqrt(np.maximum(counts, 1.0))
        else:
            weights = np.maximum(counts, 1.0)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(n, dtype=np.float64)
        raw = weights / float(weights.sum()) * float(total)
        alloc = np.floor(raw).astype(np.int64)
        alloc = np.maximum(alloc, 1)
        while int(alloc.sum()) > total:
            j = int(np.argmax(alloc))
            if alloc[j] <= 1:
                break
            alloc[j] -= 1
        while int(alloc.sum()) < total:
            order = np.argsort(-weights)
            for j in order:
                cap = int(max(1, counts[j]))
                if alloc[j] < cap:
                    alloc[j] += 1
                    if int(alloc.sum()) >= total:
                        break
            else:
                alloc[int(order[0])] += 1
                if int(alloc.sum()) >= total:
                    break
        for j in range(n):
            alloc[j] = min(int(alloc[j]), int(max(1, counts[j])))
        return alloc

    def release_slide_spectra(self) -> None:
        """Drop full HxWxM arrays after the training pool is built."""
        released = 0
        for slide in self.slides:
            if slide.spectra is not None:
                slide.spectra = None
                released += 1
        if released:
            logger.info("Released in-memory spectra for %d slide(s)", released)

    def _build_training_pool(
        self,
        *,
        pool_seed: int | None = None,
        adaptive_sampling: str = "proportional",
    ) -> None:
        cfg = self.cfg
        if not self.slides:
            raise ValueError("No slides registered")
        seed = int(cfg.seed if pool_seed is None else pool_seed)
        rng = np.random.default_rng(seed)
        lazy_slides = self._slide_loader is not None
        if bool(cfg.foundation_mode):
            self._canonical_mz = build_canonical_mz_axis(
                self._mass_min,
                self._mass_max,
                step_da=float(cfg.canonical_mz_step),
                padding_da=float(cfg.mass_range_padding_da),
            )
            logger.info(
                "Foundation canonical m/z: %d bins, %.1f..%.1f Da (step=%.2g)",
                int(self._canonical_mz.size),
                float(self._canonical_mz[0]),
                float(self._canonical_mz[-1]),
                float(cfg.canonical_mz_step),
            )
        tissue_counts = np.array(
            [max(1, int(s.tissue_pixels or np.count_nonzero(s.valid_mask))) for s in self.slides],
            dtype=np.int64,
        )
        per_slide = self._adaptive_slide_counts(
            tissue_counts,
            int(cfg.num_samples),
            mode=str(adaptive_sampling),
        )
        values_all: list[np.ndarray] = []
        masses_all: list[np.ndarray] = []
        spectra_all: list[np.ndarray] = []
        slide_ids: list[np.ndarray] = []
        for sid, slide in enumerate(self.slides):
            if slide.spectra is None:
                if self._slide_loader is None:
                    raise ValueError(
                        f"Slide {slide.name} has no spectra loaded; reload before resampling pool"
                    )
                spec, mask = self._slide_loader(slide)
                slide.spectra = np.asarray(spec, dtype=np.float32)
                slide.valid_mask = np.asarray(mask, dtype=bool)
            flat = slide.spectra.reshape(-1, slide.spectra.shape[-1])
            valid = np.flatnonzero(slide.valid_mask.ravel())
            n_pick = min(int(per_slide[sid]), valid.size)
            pick = valid if valid.size <= n_pick else rng.choice(valid, size=n_pick, replace=False)
            slide.sampled_indices = pick.astype(np.int64)
            tissue = flat[pick]
            vals, masses = batch_topk_peaks(
                tissue,
                slide.mass_axis,
                cfg.top_k,
                sort_by_mass=bool(cfg.sort_peaks_by_mass),
            )
            values_all.append(vals)
            masses_all.append(masses)
            if bool(cfg.foundation_mode):
                assert self._canonical_mz is not None
                tissue = resample_spectra_to_canonical(tissue, slide.mass_axis, self._canonical_mz)
            else:
                tissue = tissue.astype(np.float32, copy=False)
            spectra_all.append(tissue)
            slide_ids.append(np.full(vals.shape[0], sid, dtype=np.int64))
            if lazy_slides:
                slide.spectra = None
        self._sampled_values = np.concatenate(values_all, axis=0)
        self._sampled_masses = np.concatenate(masses_all, axis=0)
        self._sampled_spectra = np.concatenate(spectra_all, axis=0)
        self._sampled_slide_id = np.concatenate(slide_ids, axis=0)
        ref_idx = self._select_reference_indices(
            self._sampled_spectra,
            int(cfg.number_of_points),
            sampling=str(cfg.sampling),
            seed=int(cfg.seed),
        )
        self._ref_indices = ref_idx
        self._euclidean = torch.from_numpy(
            self._rank_distance_matrix(self._sampled_spectra, self._sampled_spectra[ref_idx])
        ).to(self.device)
        logger.info(
            "Training pool: %d sampled pixels across %d slides (%s), %d reference points, seed=%d",
            self._sampled_values.shape[0],
            len(self.slides),
            adaptive_sampling,
            ref_idx.size,
            seed,
        )

    def _cluster_feature_matrix(self) -> np.ndarray:
        cfg = self.cfg
        assert self._sampled_spectra is not None
        x = np.asarray(self._sampled_spectra, dtype=np.float64)
        if not bool(cfg.cluster_on_pca):
            return x
        n_comp = min(
            int(cfg.cluster_pca_dims),
            int(x.shape[1]),
            max(1, int(x.shape[0]) - 1),
        )
        return PCA(n_components=n_comp, random_state=int(cfg.seed)).fit_transform(x)

    def _init_cluster_heads(self) -> None:
        cfg = self.cfg
        self.cluster_heads = nn.ModuleList()
        self.cluster_labels = []
        if not bool(cfg.cluster):
            return
        cluster_levels = self._normalize_clusters(cfg.clusters)
        if not cluster_levels:
            return
        assert self._sampled_spectra is not None
        data_for_cluster = self._cluster_feature_matrix()
        n_samples_dc = int(data_for_cluster.shape[0])
        embed_dim = int(cfg.embed_dim)
        for k in cluster_levels:
            k_eff = self._clamp_n_clusters(int(k), n_samples_dc)
            if k_eff != int(k):
                logger.warning("n_clusters reduced from %d to %d (n_samples=%d)", k, k_eff, n_samples_dc)
            head = nn.Linear(embed_dim, k_eff)
            kmeans = KMeans(n_clusters=k_eff, random_state=int(cfg.seed), n_init="auto")
            labels = kmeans.fit_predict(data_for_cluster)
            self.cluster_heads.append(head)
            self.cluster_labels.append(labels.astype(np.int64))
        if self.cluster_heads:
            self.cluster_heads.to(self.device)
            logger.info(
                "Cluster heads: %d levels %s",
                len(self.cluster_heads),
                [int(np.max(lab)) + 1 for lab in self.cluster_labels],
            )

    def _init_model(self) -> None:
        cfg = self.cfg
        win, scale, dalton = resolve_mass_encoding_params(
            self._mass_min,
            self._mass_max,
            [s.mass_axis for s in self.slides],
            window_size=float(cfg.mass_window_size),
            mass_scale=float(cfg.mass_scale),
            max_windows=int(cfg.max_mass_windows),
        )
        self._mass_window_size = win
        self._mass_scale = scale
        self._mass_is_dalton = dalton
        logger.info(
            "Mass encoding: dalton=%s window=%.4g scale=%.4g (range %.1f..%.1f)",
            dalton,
            win,
            scale,
            self._mass_min,
            self._mass_max,
        )
        self.model = PeakTokenTransformer(
            embed_dim=int(cfg.embed_dim),
            d_model=int(cfg.d_model),
            nhead=int(cfg.nhead),
            num_layers=int(cfg.num_layers),
            dim_feedforward=int(cfg.dim_feedforward),
            dropout=float(cfg.dropout),
            mass_min=float(self._mass_min),
            mass_max=float(self._mass_max),
            window_size=float(win),
            mass_scale=float(scale),
            max_windows=int(cfg.max_mass_windows),
        ).to(self.device)

    def _reference_embeddings(self) -> torch.Tensor:
        assert (
            self.model is not None
            and self._ref_indices is not None
            and self._sampled_values is not None
            and self._sampled_masses is not None
        )
        ref_vals = torch.from_numpy(self._sampled_values[self._ref_indices]).to(self.device)
        ref_masses = torch.from_numpy(self._sampled_masses[self._ref_indices]).to(self.device)
        return self._embed_batch(ref_vals, ref_masses)

    def _embed_batch(self, values: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        mask = values > 0.0
        return self.model(values, masses, peak_mask=mask)

    def _training_step(self, batch_idx: torch.Tensor) -> tuple[torch.Tensor, float | None, float | None]:
        """One MiCS step: cluster CE every batch; correlation every ``k_epoch`` (same as parametric MiCS)."""
        cfg = self.cfg
        assert self.model is not None and self._euclidean is not None
        assert self._sampled_values is not None and self._sampled_masses is not None
        idx = batch_idx.detach().cpu().numpy()
        vals = torch.from_numpy(self._sampled_values[idx]).to(self.device)
        masses = torch.from_numpy(self._sampled_masses[idx]).to(self.device)
        pred = self._embed_batch(vals, masses)
        loss = torch.zeros((), device=self.device, dtype=torch.float32)
        cluster_loss_value: float | None = None

        if bool(cfg.cluster) and len(self.cluster_heads) > 0:
            cluster_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            temp = float(cfg.temperature)
            for head_idx, head in enumerate(self.cluster_heads):
                batch_clusters = torch.from_numpy(self.cluster_labels[head_idx][idx]).long().to(self.device)
                logits = head(pred) / temp
                cluster_loss_sum = cluster_loss_sum + F.cross_entropy(logits, batch_clusters)
            cluster_term = float(cfg.cluster_loss_weight) * (cluster_loss_sum / len(self.cluster_heads))
            loss = loss + cluster_term
            cluster_loss_value = float(cluster_term.detach().cpu().item())

        corr_value: float | None = None
        k_epoch = max(1, int(cfg.k_epoch))
        if self._step_counter % k_epoch == k_epoch - 1:
            assert self._ref_indices is not None
            ref_emb = self._reference_embeddings()
            low_d = torch.cdist(pred, ref_emb)
            high_d = self._euclidean.index_select(0, batch_idx)
            correlation_loss = -correlation(low_d, high_d).mean()
            alpha = max(-float(correlation_loss.detach().cpu().item()), 1e-6)
            if bool(cfg.cluster) and len(self.cluster_heads) > 0:
                loss = loss + correlation_loss * float(cfg.beta) / alpha
            else:
                loss = correlation_loss
            corr_value = -float(correlation_loss.detach().cpu().item())

        if not loss.requires_grad:
            loss = loss + pred.sum() * 0.0
        return loss, corr_value, cluster_loss_value

    def _epoch_learning_rate(self, epoch: int, global_step: int, *, opts: PeakTokenFitOptions, warmup_steps: int) -> float:
        base = float(self.cfg.lr)
        if global_step < warmup_steps:
            return base * float(global_step + 1) / float(max(1, warmup_steps))
        max_epochs = int(opts.max_epochs if opts.max_epochs is not None else self.cfg.num_epochs)
        if bool(opts.cosine_lr) and max_epochs > 1:
            progress = float(epoch) / float(max(1, max_epochs - 1))
            ratio = float(opts.min_lr_ratio) + (1.0 - float(opts.min_lr_ratio)) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            return base * ratio
        return base

    def fit(self, fit_opts: PeakTokenFitOptions | None = None) -> dict[str, float]:
        cfg = self.cfg
        opts = fit_opts or PeakTokenFitOptions()
        max_epochs = int(opts.max_epochs if opts.max_epochs is not None else cfg.num_epochs)
        adaptive = str(opts.adaptive_sampling)
        self._build_training_pool(pool_seed=int(cfg.seed), adaptive_sampling=adaptive)
        self._init_model()
        self._init_cluster_heads()
        if bool(opts.release_slide_memory):
            self.release_slide_spectra()
        assert self.model is not None
        params: list[torch.nn.Parameter] = list(self.model.parameters())
        for head in self.cluster_heads:
            params.extend(head.parameters())
        opt = torch.optim.AdamW(
            params,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
        n = int(self._sampled_values.shape[0])
        bs = max(32, int(cfg.batch_size))
        rng = np.random.default_rng(int(cfg.seed))
        last_corr = 0.0
        last_cluster = 0.0
        steps_per_epoch = max(1, int(np.ceil(n / bs)))
        warmup_steps = max(1, int(cfg.warmup_epochs) * steps_per_epoch)
        self._training_history = []
        self._fit_start_time = time.monotonic()
        stopped_reason = "max_epochs"

        for epoch in range(max_epochs):
            if opts.max_hours is not None:
                elapsed_h = (time.monotonic() - self._fit_start_time) / 3600.0
                if elapsed_h >= float(opts.max_hours):
                    stopped_reason = "max_hours"
                    logger.info("Stopping early: max_hours=%.2f reached (epoch %d)", opts.max_hours, epoch)
                    break

            if (
                epoch > 0
                and int(opts.pool_refresh_epochs) > 0
                and epoch % int(opts.pool_refresh_epochs) == 0
            ):
                pool_seed = int(cfg.seed) + epoch
                logger.info("Refreshing training pool (epoch %d, seed=%d)", epoch + 1, pool_seed)
                self._build_training_pool(pool_seed=pool_seed, adaptive_sampling=adaptive)
                if bool(opts.refresh_clusters_on_pool_refresh):
                    self._init_cluster_heads()
                n = int(self._sampled_values.shape[0])
                steps_per_epoch = max(1, int(np.ceil(n / bs)))

            epoch_start_step = epoch * steps_per_epoch
            order = rng.permutation(n)
            epoch_corr: list[float] = []
            epoch_cluster: list[float] = []
            for batch_i, start in enumerate(range(0, n, bs)):
                global_step = epoch_start_step + batch_i
                lr = self._epoch_learning_rate(epoch, global_step, opts=opts, warmup_steps=warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = lr
                batch_idx = torch.from_numpy(order[start : start + bs]).long().to(self.device)
                if batch_idx.numel() == 0:
                    continue
                loss, batch_corr, batch_cluster = self._training_step(batch_idx)
                if batch_corr is not None:
                    last_corr = batch_corr
                    epoch_corr.append(batch_corr)
                if batch_cluster is not None:
                    last_cluster = batch_cluster
                    epoch_cluster.append(batch_cluster)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                self._step_counter += 1

            epoch_mean = float(np.mean(epoch_corr)) if epoch_corr else last_corr
            epoch_cluster_mean = float(np.mean(epoch_cluster)) if epoch_cluster else last_cluster
            elapsed_h = (time.monotonic() - self._fit_start_time) / 3600.0
            if epoch_mean > self._best_correlation:
                self._best_correlation = epoch_mean
                self._best_epoch = int(epoch + 1)
                assert self.model is not None
                self._best_state = {
                    "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                    "cluster_heads": [
                        {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                        for head in self.cluster_heads
                    ],
                }

            epoch_rec = {
                "epoch": float(epoch + 1),
                "correlation": epoch_mean,
                "best_correlation": float(self._best_correlation),
                "cluster_loss": epoch_cluster_mean,
                "lr": float(opt.param_groups[0]["lr"]),
                "n_samples": float(n),
                "elapsed_hours": float(elapsed_h),
            }
            self._training_history.append(epoch_rec)

            cluster_note = f"  cluster_loss={epoch_cluster_mean:.4f}" if bool(cfg.cluster) else ""
            logger.info(
                "epoch %d/%d  correlation=%.4f  best=%.4f (ep %d)  lr=%.2e%s  [%.2fh]",
                epoch + 1,
                max_epochs,
                epoch_mean,
                self._best_correlation,
                self._best_epoch,
                float(opt.param_groups[0]["lr"]),
                cluster_note,
                elapsed_h,
            )

            if opts.on_epoch_end is not None:
                opts.on_epoch_end(epoch, epoch_rec, self)

            ckpt_every = int(opts.checkpoint_every_epochs)
            if ckpt_every > 0 and opts.checkpoint_dir is not None and (epoch + 1) % ckpt_every == 0:
                ckpt_path = Path(opts.checkpoint_dir) / f"epoch_{epoch + 1:04d}.pt"
                self.save(ckpt_path, epoch=epoch + 1, tag="checkpoint")

        if self._best_state is not None:
            assert self.model is not None
            self.model.load_state_dict(self._best_state["model"])
            for head, state in zip(self.cluster_heads, self._best_state["cluster_heads"], strict=True):
                head.load_state_dict(state)
            last_corr = self._best_correlation

        return {
            "mean_correlation": last_corr,
            "best_correlation": self._best_correlation,
            "best_epoch": float(self._best_epoch),
            "mean_cluster_loss": last_cluster,
            "mics_correlation_loss": -last_corr,
            "n_samples": float(n),
            "n_slides": float(len(self.slides)),
            "n_cluster_levels": float(len(self.cluster_heads)),
            "steps": float(self._step_counter),
            "epochs_completed": float(len(self._training_history)),
            "stopped_reason": stopped_reason,
            "elapsed_hours": float((time.monotonic() - self._fit_start_time) / 3600.0)
            if self._fit_start_time is not None
            else 0.0,
        }

    @torch.inference_mode()
    def predict_embedding(
        self,
        spectra_hwm: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
        mass_axis: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.cfg
        assert self.model is not None
        self.model.eval()
        spec = np.asarray(spectra_hwm, dtype=np.float32)
        h, w, m = spec.shape
        mask = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else spec.sum(axis=-1) > 0
        if mass_axis is None:
            mass_axis = self.slides[0].mass_axis if self.slides else np.arange(m, dtype=np.float64)
        mass_axis = np.asarray(mass_axis, dtype=np.float64).reshape(-1)
        if mass_axis.size != m:
            raise ValueError(f"mass_axis size {mass_axis.size} != spectrum width {m}")
        flat = spec.reshape(-1, m)
        idx = np.flatnonzero(mask.ravel())
        out = np.zeros((flat.shape[0], int(cfg.embed_dim)), dtype=np.float32)
        bs = max(256, int(cfg.batch_size))
        for start in range(0, idx.size, bs):
            sel = idx[start : start + bs]
            vals, masses = batch_topk_peaks(
                flat[sel],
                mass_axis,
                cfg.top_k,
                sort_by_mass=bool(cfg.sort_peaks_by_mass),
            )
            pred = self._embed_batch(
                torch.from_numpy(vals).to(self.device),
                torch.from_numpy(masses).to(self.device),
            ).cpu().numpy()
            out[sel] = pred
        emb = out.reshape(h, w, -1)
        sigma = float(cfg.predict_spatial_smooth_sigma)
        if sigma > 0.0:
            emb = _spatial_gaussian_smooth_channels(emb, sigma)
            emb[~mask] = 0.0
        return emb

    def predict_rgb(
        self,
        spectra_hwm: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
        mass_axis: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.cfg
        mask = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else spectra_hwm.sum(axis=-1) > 0
        emb = self.predict_embedding(spectra_hwm, valid_mask=mask, mass_axis=mass_axis)
        return embedding_to_display_rgb(
            emb,
            tissue_mask=mask,
            lab_to_rgb=bool(cfg.lab_to_rgb),
            percentile_low=float(cfg.percentile_low),
            percentile_high=float(cfg.percentile_high),
        )

    def save(self, path: Path, *, epoch: int | None = None, tag: str | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        payload = {
            "cfg": asdict(self.cfg),
            "mass_min": float(self._mass_min),
            "mass_max": float(self._mass_max),
            "mass_window_size": float(self._mass_window_size),
            "mass_scale": float(self._mass_scale),
            "mass_is_dalton": bool(self._mass_is_dalton),
            "foundation_mode": bool(cfg.foundation_mode),
            "canonical_mz": self._canonical_mz.tolist() if self._canonical_mz is not None else None,
            "epoch": int(epoch) if epoch is not None else int(len(self._training_history)),
            "best_epoch": int(self._best_epoch),
            "best_correlation": float(self._best_correlation),
            "tag": str(tag) if tag is not None else None,
            "training_history": list(self._training_history),
            "slides": [
                {
                    "name": s.name,
                    "npy_path": s.npy_path,
                    "shape": list(s.valid_mask.shape) + [int(s.mass_axis.size)],
                    "tissue_pixels": int(s.tissue_pixels),
                }
                for s in self.slides
            ],
            "state_dict": self.model.state_dict(),
            "cluster_head_state_dicts": [head.state_dict() for head in self.cluster_heads],
            "cluster_levels": [int(np.max(lab)) + 1 for lab in self.cluster_labels],
        }
        torch.save(payload, path)
        meta = {k: v for k, v in payload.items() if k not in ("state_dict", "cluster_head_state_dicts")}
        (path.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, device: str | None = None) -> PeakTokenMiCSTrainer:
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg_raw = dict(payload["cfg"])
        if device is not None:
            cfg_raw["device"] = device
        cfg = PeakTokenMiCSConfig(**cfg_raw)
        obj = cls(cfg)
        obj._mass_min = float(payload["mass_min"])
        obj._mass_max = float(payload["mass_max"])
        canon_raw = payload.get("canonical_mz")
        if canon_raw is not None:
            obj._canonical_mz = np.asarray(canon_raw, dtype=np.float64)
        obj._best_correlation = float(payload.get("best_correlation", -1.0))
        obj._best_epoch = int(payload.get("best_epoch", 0))
        obj._training_history = list(payload.get("training_history", []))
        obj._init_model()
        assert obj.model is not None
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.to(obj.device)
        head_states = payload.get("cluster_head_state_dicts")
        cluster_levels = payload.get("cluster_levels")
        if head_states and cluster_levels and bool(cfg.cluster):
            obj.cluster_heads = nn.ModuleList()
            for k_eff, state in zip(cluster_levels, head_states, strict=True):
                head = nn.Linear(int(cfg.embed_dim), int(k_eff))
                head.load_state_dict(state)
                head.to(obj.device)
                obj.cluster_heads.append(head)
        for s in payload.get("slides", []):
            shape_raw = s.get("shape")
            if isinstance(shape_raw, (list, tuple)) and len(shape_raw) >= 3:
                h, w, m = int(shape_raw[0]), int(shape_raw[1]), int(shape_raw[2])
            else:
                legacy = s.get("shape", [0, 0, 0])
                h, w, m = int(legacy[0]), int(legacy[1]), int(legacy[2])
            obj.slides.append(
                SlideRecord(
                    name=str(s["name"]),
                    npy_path=str(s["npy_path"]),
                    spectra=None,
                    valid_mask=np.zeros((h, w), dtype=bool),
                    mass_axis=np.arange(m, dtype=np.float64),
                    tissue_pixels=int(s.get("tissue_pixels", 0)),
                )
            )
        return obj
