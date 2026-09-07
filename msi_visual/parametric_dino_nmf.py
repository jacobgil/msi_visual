from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from msi_visual.parametric_mics_lmc import random_pixel_sampling, uniform_superpixel_sampling
from msi_visual.utils import normalize


@dataclass
class _TrainBatch:
    tokens: torch.Tensor


class _DINOMLPModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        dino_dim: int,
        token_dropout: float,
    ) -> None:
        super().__init__()
        self.token_dropout = float(token_dropout)
        layers = [torch.nn.Linear(input_dim, d_model), torch.nn.GELU()]
        for _ in range(max(1, int(n_layers)) - 1):
            layers.extend([torch.nn.Linear(d_model, d_model), torch.nn.GELU()])
        self.backbone = torch.nn.Sequential(*layers)
        self.cls_head = torch.nn.Linear(d_model, 3)
        self.dino_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, dino_dim),
        )

    def forward(self, tokens: torch.Tensor, apply_token_dropout: bool) -> tuple[torch.Tensor, torch.Tensor]:
        x = tokens
        if apply_token_dropout and self.token_dropout > 0:
            keep = (torch.rand_like(x) > self.token_dropout).float()
            x = x * keep
        feat = self.backbone(x)
        return self.cls_head(feat), self.dino_head(feat)


def _orthogonal_init_cls_head(model: _DINOMLPModel) -> None:
    """Break symmetry so RGB logits start (and tend to stay) less collinear."""
    torch.nn.init.orthogonal_(model.cls_head.weight, gain=1.0)
    if model.cls_head.bias is not None:
        torch.nn.init.zeros_(model.cls_head.bias)


class DINONMFVirtualStain:
    """
    Parametric visualization with:
    - raw spectral features as tokens (optional linear PCA before the MLP)
    - MLP backbone
    - DINO-style student/teacher training with student token dropout on MLP inputs
      (i.e. on PCA coordinates when ``pre_mlp_pca_dim > 0``)

    Note: ``predict()`` maps RGB from the 3-way ``cls_head`` only; the high-dimensional
    ``dino_head`` shapes the shared backbone via the distillation loss but is not used
    directly for the saved image unless you change ``predict``.
    """

    def __init__(
        self,
        nmf_k: int = 128,
        pixel_sampling: str = "superpixel",
        num_samples: int = 5000,
        pca_fit_step: int = 4,
        random_state: int = 42,
        train_epochs: int = 20,
        batch_size: int = 2048,
        lr: float = 1e-3,
        d_model: int = 192,
        n_layers: int = 4,
        dino_dim: int = 256,
        token_dropout: float = 0.15,
        ema_momentum: float = 0.996,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        max_nmf_fit_samples: int = 40000,
        koleo_weight: float = 0.02,
        view_noise_std: float = 0.02,
        spectral_dropout_p: float = 0.0,
        teacher_center_logits: bool = True,
        pre_mlp_pca_dim: int = 0,
        cls_decorrelate_weight: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        self.nmf_k = int(nmf_k)
        self.pixel_sampling = str(pixel_sampling).lower()
        self.num_samples = int(num_samples)
        self.pca_fit_step = int(max(1, pca_fit_step))
        self.random_state = int(random_state)
        self.train_epochs = int(train_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.dino_dim = int(dino_dim)
        self.token_dropout = float(token_dropout)
        self.ema_momentum = float(ema_momentum)
        self.student_temp = float(student_temp)
        self.teacher_temp = float(teacher_temp)
        self.max_nmf_fit_samples = int(max_nmf_fit_samples)  # backward-compat (unused in MLP mode)
        self.koleo_weight = float(koleo_weight)
        self.view_noise_std = float(view_noise_std)
        self.spectral_dropout_p = float(max(0.0, min(0.999, spectral_dropout_p)))
        self.teacher_center_logits = bool(teacher_center_logits)
        self.pre_mlp_pca_dim = int(pre_mlp_pca_dim)
        self.cls_decorrelate_weight = float(cls_decorrelate_weight)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.student: Optional[_DINOMLPModel] = None
        self.teacher: Optional[_DINOMLPModel] = None
        self._pre_mlp_pca: Optional[Any] = None

    def _sample_indices(self, img: np.ndarray, roi_mask: Optional[np.ndarray]) -> np.ndarray:
        h, w = img.shape[:2]
        reshaped = img.reshape(h * w, -1)
        if self.pixel_sampling == "random":
            _, sampled_flat_indices, _ = random_pixel_sampling(
                reshaped,
                self.num_samples,
                h,
                w,
                roi_mask=roi_mask,
                rng_seed=self.random_state,
            )
        else:
            _, sampled_flat_indices, _ = uniform_superpixel_sampling(
                img,
                reshaped,
                approx_budget=self.num_samples,
                init_superpixels=512,
                init_points_per_superpixel=8,
                compactness=10,
                max_iters=8,
                verbose=False,
                visualize=False,
                rng_seed=self.random_state,
                roi_mask=roi_mask,
                pca_fit_step=self.pca_fit_step,
            )
        if sampled_flat_indices is None or len(sampled_flat_indices) == 0:
            raise ValueError("Sampling produced no points for DINO-NMF")
        return np.asarray(sampled_flat_indices, dtype=np.int64)

    def _build_training_batch(self, img: np.ndarray, sampled_flat_indices: np.ndarray) -> _TrainBatch:
        h, w = img.shape[:2]
        flat = img.reshape(h * w, -1)
        valid_mask = flat.max(axis=1) > 0
        sampled_flat_indices = sampled_flat_indices[valid_mask[sampled_flat_indices]]
        if sampled_flat_indices.size == 0:
            raise ValueError("No valid sampled pixels for MLP fit.")
        sampled_tokens = flat[sampled_flat_indices].astype(np.float32, copy=False)
        return _TrainBatch(
            tokens=torch.from_numpy(sampled_tokens),
        )

    def _two_augmented_views(self, tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian noise + optional per-channel dropout (different masks per view)."""
        std = self.view_noise_std
        noise1 = torch.randn_like(tok) * std
        noise2 = torch.randn_like(tok) * std
        if self._pre_mlp_pca is not None:
            v1 = tok + noise1
            v2 = tok + noise2
        else:
            v1 = torch.clamp(tok + noise1, min=0.0)
            v2 = torch.clamp(tok + noise2, min=0.0)
        p = self.spectral_dropout_p
        if p > 0.0:
            m1 = (torch.rand(tok.shape[0], tok.shape[1], device=tok.device, dtype=tok.dtype) > p).float()
            m2 = (torch.rand(tok.shape[0], tok.shape[1], device=tok.device, dtype=tok.dtype) > p).float()
            v1 = v1 * m1
            v2 = v2 * m2
        return v1, v2

    def _dino_loss(self, s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
        t = t_logits
        if self.teacher_center_logits:
            t = t - t.mean(dim=0, keepdim=True)
        t_probs = torch.softmax(t / self.teacher_temp, dim=-1).detach()
        s_log_probs = torch.log_softmax(s_logits / self.student_temp, dim=-1)
        return -(t_probs * s_log_probs).sum(dim=-1).mean()

    def _koleo_loss(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[0] <= 1:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        z = F.normalize(z, dim=-1)
        sim = torch.matmul(z, z.T)
        sim.fill_diagonal_(-1e9)
        nn_sim = torch.max(sim, dim=1).values
        # Lower nearest-neighbor similarity => more spread-out embeddings.
        return torch.mean(nn_sim)

    def _cls_channel_decorr_loss(self, c: torch.Tensor) -> torch.Tensor:
        """Penalize correlation between the 3 CLS channels across batch (reduces gray / achromatic maps)."""
        if c.shape[0] <= 2:
            return torch.zeros((), device=c.device, dtype=c.dtype)
        z = c - c.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, unbiased=False, keepdim=True) + 1e-4)
        corr = (z.T @ z) / float(c.shape[0])
        eye = torch.eye(3, device=c.device, dtype=c.dtype)
        off = (corr * (1.0 - eye)).pow(2).sum() / 6.0
        return off

    def fit(self, images, keras_fit_kwargs=None, roi_mask=None):
        if images is None or len(images) == 0:
            raise ValueError("images must be non-empty")
        images_np = [np.asarray(img, dtype=np.float32) for img in images]
        sampled_by_image = [self._sample_indices(img, roi_mask=roi_mask) for img in images_np]
        batches = [self._build_training_batch(img, sampled_flat_indices)
                   for img, sampled_flat_indices in zip(images_np, sampled_by_image)]
        tokens = torch.cat([b.tokens for b in batches], dim=0)
        batch = _TrainBatch(tokens=tokens)
        if self.pre_mlp_pca_dim <= 0:
            self._pre_mlp_pca = None
        else:
            X = batch.tokens.numpy().astype(np.float64, copy=False)
            n_req = min(int(self.pre_mlp_pca_dim), X.shape[0], X.shape[1])
            if n_req < 1:
                raise ValueError(
                    f"pre_mlp_pca_dim: need >=1 component; got n_samples={X.shape[0]}, n_features={X.shape[1]}"
                )
            self._pre_mlp_pca = PCA(n_components=n_req, random_state=self.random_state)
            self._pre_mlp_pca.fit(X)
            Xp = self._pre_mlp_pca.transform(X).astype(np.float32, copy=False)
            batch = _TrainBatch(tokens=torch.from_numpy(Xp))
            print(
                f"[DINO-NMF] pre-MLP PCA: spectrum_dim={X.shape[1]} -> n_components={n_req} "
                f"(requested {self.pre_mlp_pca_dim})"
            )
        input_dim = int(batch.tokens.shape[1])

        self.student = _DINOMLPModel(
            input_dim=input_dim,
            d_model=self.d_model,
            n_layers=self.n_layers,
            dino_dim=self.dino_dim,
            token_dropout=self.token_dropout,
        ).to(self.device)
        _orthogonal_init_cls_head(self.student)
        self.teacher = _DINOMLPModel(
            input_dim=input_dim,
            d_model=self.d_model,
            n_layers=self.n_layers,
            dino_dim=self.dino_dim,
            token_dropout=0.0,
        ).to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.student.train()

        optim = torch.optim.AdamW(self.student.parameters(), lr=self.lr, weight_decay=1e-4)
        rng = np.random.default_rng(self.random_state)
        n = batch.tokens.shape[0]

        for epoch in range(1, self.train_epochs + 1):
            perm = rng.permutation(n)
            sum_total = sum_dino = sum_cls = sum_var = sum_koleo = sum_decorr = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                tok = batch.tokens[idx].to(self.device).float()
                v1, v2 = self._two_augmented_views(tok)

                c_s1, s1 = self.student(v1, apply_token_dropout=True)
                c_s2, s2 = self.student(v2, apply_token_dropout=True)
                with torch.no_grad():
                    c_t1, t1 = self.teacher(v1, apply_token_dropout=False)
                    c_t2, t2 = self.teacher(v2, apply_token_dropout=False)

                # Train both the high-D DINO head and the 3D CLS output head.
                loss_dino = 0.5 * (self._dino_loss(s1, t2) + self._dino_loss(s2, t1))
                loss_cls = 0.5 * (self._dino_loss(c_s1, c_t2) + self._dino_loss(c_s2, c_t1))
                # Light anti-collapse regularization on student outputs.
                std_s = torch.sqrt(torch.var(s1, dim=0, unbiased=False) + 1e-4)
                std_c = torch.sqrt(torch.var(c_s1, dim=0, unbiased=False) + 1e-4)
                loss_var = torch.mean(torch.relu(1.0 - std_s)) + torch.mean(torch.relu(1.0 - std_c))
                loss_koleo = self._koleo_loss(s1) + self._koleo_loss(c_s1)
                loss_decorr = torch.zeros((), device=self.device)
                if self.cls_decorrelate_weight > 0.0:
                    loss_decorr = 0.5 * (
                        self._cls_channel_decorr_loss(c_s1) + self._cls_channel_decorr_loss(c_s2)
                    )
                loss = (
                    loss_dino
                    + loss_cls
                    + 0.1 * loss_var
                    + self.koleo_weight * loss_koleo
                    + self.cls_decorrelate_weight * loss_decorr
                )
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                optim.step()

                with torch.no_grad():
                    for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                        pt.data.mul_(self.ema_momentum).add_(ps.data * (1.0 - self.ema_momentum))

                sum_total += loss.detach().item()
                sum_dino += loss_dino.detach().item()
                sum_cls += loss_cls.detach().item()
                sum_var += loss_var.detach().item()
                sum_koleo += loss_koleo.detach().item()
                sum_decorr += float(loss_decorr.detach().item())
                n_batches += 1

            inv_b = 1.0 / max(1, n_batches)
            msg = (
                f"[DINO-NMF] epoch {epoch}/{self.train_epochs} "
                f"loss={sum_total * inv_b:.6f} "
                f"(dino={sum_dino * inv_b:.6f} cls={sum_cls * inv_b:.6f} "
                f"var={sum_var * inv_b:.6f} koleo={sum_koleo * inv_b:.6f}"
            )
            if self.cls_decorrelate_weight > 0.0:
                msg += f" cls_decorr={sum_decorr * inv_b:.6f}"
            msg += ")"
            print(msg)
        return self

    def predict(self, img: np.ndarray) -> np.ndarray:
        if self.student is None:
            raise ValueError("Call fit() before predict().")
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape[:2]
        flat = arr.reshape(h * w, -1)
        valid_mask = flat.max(axis=1) > 0
        x_valid = flat[valid_mask].astype(np.float32, copy=False)
        if self._pre_mlp_pca is not None:
            x_valid = self._pre_mlp_pca.transform(x_valid.astype(np.float64, copy=False)).astype(
                np.float32, copy=False
            )
        tok_t = torch.from_numpy(x_valid).float().to(self.device)
        preds = []
        self.student.eval()
        with torch.no_grad():
            for start in range(0, tok_t.shape[0], 8192):
                end = start + 8192
                cls_out, _ = self.student(tok_t[start:end], apply_token_dropout=False)
                preds.append(cls_out.detach().cpu().numpy())
        pred_valid = np.concatenate(preds, axis=0).astype(np.float32, copy=False)
        out = np.zeros((h * w, 3), dtype=np.float32)
        out[valid_mask] = pred_valid
        out = out.reshape(h, w, 3)
        mask_2d = valid_mask.reshape(h, w)
        out = normalize(out, mask=mask_2d)
        out[~mask_2d] = 0.0
        return out


class ParametricDINONMF:
    """Thin wrapper to match other visualization class style."""

    def __init__(self, **kwargs):
        self.model = DINONMFVirtualStain(**kwargs)

    def fit(self, images, keras_fit_kwargs=None, roi_mask=None):
        self.model.fit(images, keras_fit_kwargs=keras_fit_kwargs, roi_mask=roi_mask)
        return self

    def predict(self, img: np.ndarray) -> np.ndarray:
        return self.model.predict(img)
