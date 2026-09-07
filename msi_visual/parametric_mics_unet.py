"""
Spatial MiCS+LMC: small U-Net maps H×W×C spectra to an embedding map (same as PCC output dim),
then the same MiCS cluster losses and LMC correlation loss on sampled pixels vs reference points.

Adds optional **image-level** terms (defaults 0): maximize soft Dice between differentiable Sobel
edge magnitude on the visualization and a fixed HD edge target map, and maximize luminance RMS contrast.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC, correlation

logger = logging.getLogger(__name__)

# PyTorch rejects tensors whose numel does not fit in int32 indexing for many ops (pad, conv, …).
_MAX_UNET_SPEC_ELEMENTS = min((2**31 - 2) // 4, 400_000_000)


def forward_unet_full_resolution(
    model: nn.Module,
    spectral: torch.Tensor,
    *,
    max_spec_elements: int = _MAX_UNET_SPEC_ELEMENTS,
    on_downsample: Optional[Callable[[int, int, int, int], None]] = None,
) -> torch.Tensor:
    """Run ``SmallUNet`` so output aligns with full ``(H,W)``.

    If ``C*H*W`` exceeds ``max_spec_elements``, spatially downsample ``spectral`` before the U-Net
    and bilinearly upsample features back (MiCS indices stay full-resolution).
    """
    if spectral.dim() != 4 or spectral.shape[0] != 1:
        raise ValueError(f"Expected spectral (1,C,H,W), got {tuple(spectral.shape)}")
    _, C, H, W = spectral.shape
    C = int(C)
    H = int(H)
    W = int(W)
    if C * H * W <= max_spec_elements:
        out = model(spectral)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

    spatial_budget = max(16, max_spec_elements // max(C, 1))
    scale = math.sqrt(spatial_budget / float(H * W))
    Hs = max(4, int(H * scale))
    Ws = max(4, int(W * scale))
    Hs = max(4, (Hs // 4) * 4)
    Ws = max(4, (Ws // 4) * 4)
    while C * Hs * Ws > max_spec_elements and Hs > 4 and Ws > 4:
        if Hs >= Ws:
            Hs = max(4, Hs - 4)
        else:
            Ws = max(4, Ws - 4)

    if C * Hs * Ws > max_spec_elements:
        raise RuntimeError(
            f"U-Net input still too large after downsampling: C={C}, spatial {Hs}x{Ws} "
            f"(budget {max_spec_elements} elements). Reduce MSI channels (bins) or spatial size."
        )

    small = F.interpolate(spectral, size=(Hs, Ws), mode="bilinear", align_corners=False)
    if on_downsample is not None:
        on_downsample(Hs, Ws, H, W)
    feat_s = model(small)
    return F.interpolate(feat_s, size=(H, W), mode="bilinear", align_corners=False)


def _pad_to_multiple(x: torch.Tensor, m: int = 4) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return x, 0, 0
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), ph, pw


def _unpad(x: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    if ph == 0 and pw == 0:
        return x
    return x[..., : x.shape[-2] - ph, : x.shape[-1] - pw]


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpsampleConv(nn.Module):
    """Bilinear upsample + 3×3 conv (avoids ConvTranspose checkerboard)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.proj(x)


class SmallUNet(nn.Module):
    """Lightweight 2-level U-Net for H×W×C_in → H×W×C_out (spectral cube in, embedding channels out)."""

    def __init__(self, in_channels: int, out_channels: int, base: int = 32) -> None:
        super().__init__()
        b = max(8, int(base))
        self.down1 = _DoubleConv(in_channels, b)
        # AvgPool reduces checkerboard/aliasing vs MaxPool before bilinear ups.
        self.pool1 = nn.AvgPool2d(2)
        self.down2 = _DoubleConv(b, b * 2)
        self.pool2 = nn.AvgPool2d(2)
        self.mid = _DoubleConv(b * 2, b * 2)
        self.up2 = _UpsampleConv(b * 2, b)
        # concat(skip x2: b*2 ch, upsampled: b ch) → b*3 input channels
        self.dec2 = _DoubleConv(b + b * 2, b)
        self.up1 = _UpsampleConv(b, b)
        self.dec1 = _DoubleConv(b * 2, b)
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, ph, pw = _pad_to_multiple(x, 4)
        x1 = self.down1(x)
        x = self.pool1(x1)
        x2 = self.down2(x)
        x = self.pool2(x2)
        x = self.mid(x)
        x = self.up2(x, size_hw=(int(x2.shape[-2]), int(x2.shape[-1])))
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = self.up1(x, size_hw=(int(x1.shape[-2]), int(x1.shape[-1])))
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        x = self.out_conv(x)
        return _unpad(x, ph, pw)


def _sobel_mag_gray(y: torch.Tensor) -> torch.Tensor:
    """y: (N,1,H,W) — Sobel magnitude."""
    device, dtype = y.device, y.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def _rgb_from_feat(feat: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """(N,K,H,W) → (N,3,H,W) in [0,1] via masked per-channel min–max (matches predict stretch)."""
    n, k, h, w = feat.shape
    if k >= 3:
        rgb = feat[:, :3]
    else:
        rgb = feat[:, :1].repeat(1, 3, 1, 1)
    if mask is None:
        lo = rgb.amin(dim=(2, 3), keepdim=True)
        hi = rgb.amax(dim=(2, 3), keepdim=True)
    else:
        m = mask > 0.5
        if not bool(m.any()):
            lo = rgb.amin(dim=(2, 3), keepdim=True)
            hi = rgb.amax(dim=(2, 3), keepdim=True)
        else:
            m2d = m.reshape(h, w)
            lo = torch.full((n, 3, 1, 1), float("inf"), device=feat.device, dtype=feat.dtype)
            hi = torch.full((n, 3, 1, 1), float("-inf"), device=feat.device, dtype=feat.dtype)
            for c in range(3):
                vals = rgb[0, c][m2d]
                if vals.numel() > 0:
                    lo[:, c] = vals.min()
                    hi[:, c] = vals.max()
                else:
                    lo[:, c] = rgb[:, c].min()
                    hi[:, c] = rgb[:, c].max()
    return ((rgb - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _luminance_rgb01(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def soft_dice_continuous(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred, target, mask: (1,1,H,W) finite; soft Dice coefficient in [0,1]."""
    m = mask.float()
    p = pred * m
    t = target * m
    eps = 1e-6
    inter = (p * t).sum()
    denom = p.sum() + t.sum() + eps
    return (2.0 * inter + eps) / denom


class MSIParametricMiCSUnet(MSIParametricMiCSLMC):
    """
    MiCS+LMC with a spatial encoder (U-Net) instead of a per-pixel MLP.

    Auxiliary losses (weighted, added every mini-batch with 1/num_batches scaling):
    - ``dice_loss_weight`` × (− soft Dice) between Sobel magnitude of viz luminance vs ``hd_edges``.
    - ``contrast_loss_weight`` × (− luminance standard deviation on tissue).

    ``hd_edges``: optional float32 (H,W) in [0,1], same shape as the MSI slice; background ignored via mask.
    """

    def __init__(
        self,
        model=None,
        number_of_points=500,
        sampling="coreset",
        num_epochs=200,
        number_of_components=3,
        lab_to_rgb=True,
        cluster=False,
        clusters=None,
        num_samples=5000,
        beta=5.0,
        k_epoch=1,
        temperature=10.0,
        lr=1.0,
        batch_size=4096 * 2,
        warmup_epochs=10,
        factor=1.0,
        num_layers=2,
        random_state=42,
        verbose=False,
        cluster_loss_weight=1.0,
        category_loss_weight=1.0,
        pixel_sampling="superpixel",
        pca_fit_step=4,
        cluster_on_pca=False,
        cluster_pca_dims=50,
        *,
        unet_base_channels: int = 32,
        dice_loss_weight: float = 0.0,
        contrast_loss_weight: float = 0.0,
        hd_edges: Optional[np.ndarray] = None,
        predict_percentile_low: float = 0.001,
        predict_percentile_high: float = 99.999,
        predict_spatial_smooth_sigma: float = 0.0,
        clusters_auto_tune: Optional[str] = None,
        clusters_auto_tune_k_min: int = 2,
        clusters_auto_tune_k_max: int = 256,
        clusters_auto_tune_n_candidates: int = 20,
        clusters_auto_tune_k_step: Optional[int] = None,
    ) -> None:
        self.unet_base_channels = max(8, int(unet_base_channels))
        self.dice_loss_weight = float(dice_loss_weight)
        self.contrast_loss_weight = float(contrast_loss_weight)
        self.hd_edges_np: Optional[np.ndarray] = (
            None if hd_edges is None else np.asarray(hd_edges, dtype=np.float32)
        )
        self._hd_edges_t: Optional[torch.Tensor] = None
        self._mask_hw_t: Optional[torch.Tensor] = None
        self._sampled_flat_torch: Optional[torch.Tensor] = None
        super().__init__(
            model=model,
            number_of_points=number_of_points,
            sampling=sampling,
            num_epochs=num_epochs,
            number_of_components=number_of_components,
            lab_to_rgb=lab_to_rgb,
            cluster=cluster,
            clusters=clusters,
            num_samples=num_samples,
            beta=beta,
            k_epoch=k_epoch,
            temperature=temperature,
            lr=lr,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            factor=factor,
            num_layers=num_layers,
            random_state=random_state,
            verbose=verbose,
            cluster_loss_weight=cluster_loss_weight,
            category_loss_weight=category_loss_weight,
            pixel_sampling=pixel_sampling,
            pca_fit_step=pca_fit_step,
            cluster_on_pca=cluster_on_pca,
            cluster_pca_dims=cluster_pca_dims,
            predict_percentile_low=predict_percentile_low,
            predict_percentile_high=predict_percentile_high,
            predict_spatial_smooth_sigma=predict_spatial_smooth_sigma,
            clusters_auto_tune=clusters_auto_tune,
            clusters_auto_tune_k_min=clusters_auto_tune_k_min,
            clusters_auto_tune_k_max=clusters_auto_tune_k_max,
            clusters_auto_tune_n_candidates=clusters_auto_tune_n_candidates,
            clusters_auto_tune_k_step=clusters_auto_tune_k_step,
        )
        self._unet_ds_warned = False

    def _on_unet_downsample(self, hs: int, ws: int, h: int, w: int) -> None:
        if self._unet_ds_warned:
            return
        self._unet_ds_warned = True
        logger.warning(
            "U-Net: MSI size %d×%d×C exceeds PyTorch int32 indexing limit on flat tensors; "
            "running the encoder at %d×%d and bilinearly upsampling features to full resolution.",
            h,
            w,
            hs,
            ws,
        )

    def _create_default_model(self, input_dim: int) -> torch.nn.Module:
        _ = input_dim
        if self.img is None:
            raise RuntimeError("U-Net init requires self.img before model creation.")
        c_in = int(self.img.shape[-1])
        net = SmallUNet(c_in, self.number_of_components, base=self.unet_base_channels)
        if torch.cuda.is_available():
            net = net.cuda()
        return net

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
        H, W = int(self.img.shape[0]), int(self.img.shape[1])
        device = next(self.model.parameters()).device
        mask_np = self.img.sum(axis=-1) > 0
        self._mask_hw_t = torch.from_numpy(mask_np.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)

        if self.hd_edges_np is not None:
            he = self.hd_edges_np
            if he.shape[:2] != (H, W):
                raise ValueError(f"hd_edges shape {he.shape[:2]} != MSI {(H, W)}")
            self._hd_edges_t = torch.from_numpy(np.clip(he, 0.0, 1.0)).to(device).unsqueeze(0).unsqueeze(0)
        else:
            self._hd_edges_t = None

        self._sampled_flat_torch = torch.from_numpy(self.sampled_flat_indices.astype(np.int64)).to(device)

    def release_resources(self) -> None:
        self._hd_edges_t = None
        self._mask_hw_t = None
        self._sampled_flat_torch = None
        super().release_resources()

    def step(self):
        batch_size = self.batch_size
        n_samples = self.sampled_data.shape[0]
        H, W = int(self.img.shape[0]), int(self.img.shape[1])
        device = next(self.model.parameters()).device
        batch_indices = torch.randperm(n_samples, device=device)
        num_batches = (n_samples + batch_size - 1) // batch_size
        spectral = (
            torch.from_numpy(np.asarray(self.img, dtype=np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        if self.step_counter == 0 and self.verbose:
            print(
                f"[step/unet] {num_batches} batches, batch_size={batch_size}, "
                f"dice_w={self.dice_loss_weight}, contrast_w={self.contrast_loss_weight}"
            )

        sf = self._sampled_flat_torch
        assert sf is not None
        ref_idx_t = torch.from_numpy(np.asarray(self.indices, dtype=np.int64)).long().to(device)
        ref_flat = sf[ref_idx_t]
        ref_rows = (ref_flat // W).long()
        ref_cols = (ref_flat % W).long()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_mask = batch_indices[start_idx:end_idx]

            feat_map = forward_unet_full_resolution(
                self.model,
                spectral,
                on_downsample=self._on_unet_downsample,
            )

            batch_flat = sf[batch_mask]
            rows = (batch_flat // W).long()
            cols = (batch_flat % W).long()
            batch_outputs = feat_map[0, :, rows, cols].transpose(0, 1)

            ref_embeddings = feat_map[0, :, ref_rows, ref_cols].transpose(0, 1)

            loss = torch.zeros((), device=device, dtype=torch.float32)

            low_d_distances = torch.cdist(batch_outputs, ref_embeddings)
            high_d_distances = self.euclidean[batch_mask.to(self.euclidean.device)]

            bm_np = batch_mask.detach().cpu().numpy()
            if self.cluster and len(self.visualiation_to_cluster) > 0:
                cluster_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                for i in range(len(self.visualiation_to_cluster)):
                    layer = self.visualiation_to_cluster[i]
                    clusters = self.cluster_labels[i]
                    batch_clusters = torch.from_numpy(clusters[bm_np]).long().to(device)

                    layer_output = layer(batch_outputs)
                    cluster_loss = nn.CrossEntropyLoss(ignore_index=-1)(
                        layer_output / self.temperature, batch_clusters
                    )
                    cluster_loss_sum = cluster_loss_sum + cluster_loss
                loss = loss + self.cluster_loss_weight * (cluster_loss_sum / len(self.visualiation_to_cluster))

            if self.category_head is not None and self.category_labels is not None:
                batch_cats = torch.from_numpy(self.category_labels[bm_np]).long().to(device)
                cat_output = self.category_head(batch_outputs)
                cat_loss = F.cross_entropy(cat_output / self.temperature, batch_cats, ignore_index=-1)
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

            aux_scale = 1.0 / float(max(num_batches, 1))
            if self.dice_loss_weight > 1e-12 and self._hd_edges_t is not None and self._mask_hw_t is not None:
                rgb = _rgb_from_feat(feat_map, mask=self._mask_hw_t)
                y = _luminance_rgb01(rgb)
                mag_v = _sobel_mag_gray(y)
                mag_t = self._hd_edges_t
                m = self._mask_hw_t
                dice = soft_dice_continuous(mag_v, mag_t, m)
                loss = loss + aux_scale * self.dice_loss_weight * (-dice)

            if self.contrast_loss_weight > 1e-12 and self._mask_hw_t is not None:
                rgb = _rgb_from_feat(feat_map, mask=self._mask_hw_t)
                y_hw = _luminance_rgb01(rgb).squeeze()
                m_hw = self._mask_hw_t.squeeze() > 0.5
                vals = y_hw[m_hw]
                if vals.numel() > 1:
                    contrast = torch.std(vals)
                    loss = loss + aux_scale * self.contrast_loss_weight * (-contrast)

            if not getattr(loss, "requires_grad", False):
                loss = loss + (batch_outputs.sum() * 0.0)

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optim.step()

        if self.step_counter % self.k_epoch == self.k_epoch - 1:
            old_k_epoch = self.k_epoch
            self.k_epoch = max(1, int(self.k_epoch * 0.92))
            if old_k_epoch != self.k_epoch and self.verbose:
                print(f"[step] Updated k_epoch: {old_k_epoch} -> {self.k_epoch}")

        self.step_counter += 1

    def predict_embedding(self, img: np.ndarray) -> np.ndarray:
        """Raw H×W×C embedding from the U-Net (before percentile stretch / LAB→RGB)."""
        if self.model is None:
            raise ValueError("Model must be initialized before predict_embedding.")
        img = np.asarray(img, dtype=np.float32)
        device = next(self.model.parameters()).device
        spectral = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            feat = forward_unet_full_resolution(
                self.model,
                spectral,
                on_downsample=self._on_unet_downsample,
            )[0].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
        mask = img.sum(axis=-1) > 0
        feat[mask == 0] = 0.0
        return feat

    def predict(self, img):
        if self.model is None:
            raise ValueError("Model must be initialized.")
        self.img = img
        feat = self.predict_embedding(img)
        return self._embedding_to_display_rgb(feat, img)

    def predict_cluster_argmax_labels(self, img=None, level_idx: int = 0) -> np.ndarray:
        if self.model is None:
            raise ValueError("Call fit() first.")
        if not getattr(self, "visualiation_to_cluster", None) or len(self.visualiation_to_cluster) == 0:
            raise ValueError("No cluster heads.")
        if level_idx < 0 or level_idx >= len(self.visualiation_to_cluster):
            raise IndexError(f"level_idx out of range: {level_idx}")
        if img is None:
            img = self.img
        if img is None:
            raise ValueError("Pass img or call fit() first.")
        img = np.asarray(img, dtype=np.float32)
        H, W = int(img.shape[0]), int(img.shape[1])
        mask = img.sum(axis=-1) > 0
        device = next(self.model.parameters()).device
        classifier = self.visualiation_to_cluster[level_idx].to(device)
        spectral = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        self.model.eval()
        classifier.eval()
        with torch.no_grad():
            feat = forward_unet_full_resolution(
                self.model,
                spectral,
                on_downsample=self._on_unet_downsample,
            )[0]
            logits = classifier(feat.permute(1, 2, 0).reshape(H * W, -1))
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)
        out = pred.reshape(H, W)
        out = np.where(mask, out, -1).astype(np.int32)
        return out

    def uncertainty2(self, img):
        if self.model is None:
            raise ValueError("Model must be initialized.")
        if not getattr(self, "visualiation_to_cluster", None) or len(self.visualiation_to_cluster) == 0:
            raise ValueError("No classifiers available.")
        self.img = img
        img = np.asarray(img, dtype=np.float32)
        H, W = int(img.shape[0]), int(img.shape[1])
        device = next(self.model.parameters()).device
        spectral = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        self.model.eval()
        batch_size = 1024 * 32
        with torch.no_grad():
            feat = forward_unet_full_resolution(
                self.model,
                spectral,
                on_downsample=self._on_unet_downsample,
            )[0].permute(1, 2, 0).reshape(H * W, -1)
            y_chunks = torch.split(feat, batch_size, dim=0)
            entropies = []
            for classifier in self.visualiation_to_cluster:
                classifier.eval()
                classifier = classifier.to(device)
                layer_entropy = []
                for y_batch in y_chunks:
                    logits = classifier(y_batch)
                    probs = torch.softmax(logits, dim=1)
                    entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
                    layer_entropy.append(entropy.cpu().numpy())
                entropies.append(np.concatenate(layer_entropy, axis=0))
        mean_entropy = np.mean(np.stack(entropies, axis=1), axis=1).reshape(H, W)
        mask = img.sum(axis=-1) > 0
        mean_entropy[~mask] = 0.0
        min_val = np.min(mean_entropy)
        max_val = np.max(mean_entropy)
        if max_val > min_val:
            mean_entropy = (mean_entropy - min_val) / (max_val - min_val)
        else:
            mean_entropy = np.zeros_like(mean_entropy)
        return mean_entropy
