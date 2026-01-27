import numpy as np
import cv2
import torch
import tqdm
from sklearn.decomposition import PCA


def _normalize_map(values, mask):
    if mask.any():
        min_val = values[mask].min()
        max_val = values[mask].max()
        if max_val > min_val:
            values = (values - min_val) / (max_val - min_val)
        else:
            values = np.zeros_like(values)
    else:
        values = np.zeros_like(values)
    values[~mask] = 0.0
    return values


def _to_grayscale(viz):
    if viz.ndim == 2:
        gray = viz.astype(np.float32)
    else:
        viz_f = viz.astype(np.float32)
        if viz_f.max() > 1.0:
            viz_f = viz_f / 255.0
        gray = 0.2989 * viz_f[..., 0] + 0.5870 * viz_f[..., 1] + 0.1140 * viz_f[..., 2]
    return gray.astype(np.float32)


def _apply_equalization(image_uint8):
    equalized = cv2.equalizeHist(image_uint8)
    equalized[image_uint8 == 0] = 0
    return equalized


def _local_correlation(x, y, window_size, mask, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    ksize = (window_size, window_size)
    mean_x = cv2.boxFilter(x, ddepth=-1, ksize=ksize, normalize=True)
    mean_y = cv2.boxFilter(y, ddepth=-1, ksize=ksize, normalize=True)
    mean_xy = cv2.boxFilter(x * y, ddepth=-1, ksize=ksize, normalize=True)
    mean_x2 = cv2.boxFilter(x * x, ddepth=-1, ksize=ksize, normalize=True)
    mean_y2 = cv2.boxFilter(y * y, ddepth=-1, ksize=ksize, normalize=True)
    cov = mean_xy - mean_x * mean_y
    var_x = mean_x2 - mean_x * mean_x
    var_y = mean_y2 - mean_y * mean_y
    denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0)) + eps
    corr = cov / denom
    corr = np.clip((corr + 1.0) * 0.5, 0.0, 1.0)
    corr[~mask] = 0.0
    return corr


def compute_highd_summary(img, method="pca1", random_state=42, max_samples=20000):
    mask = img.sum(axis=-1) > 0
    flat = img.reshape(-1, img.shape[-1]).astype(np.float32, copy=False)
    mask_flat = mask.reshape(-1)
    if not mask_flat.any():
        return np.zeros(img.shape[:2], dtype=np.float32)

    if method == "pca1":
        X = flat[mask_flat]
        if X.shape[0] > max_samples:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(X.shape[0], size=max_samples, replace=False)
            X_fit = X[idx]
        else:
            X_fit = X
        pca = PCA(n_components=1, random_state=random_state)
        pca.fit(X_fit)
        proj = (X - pca.mean_) @ pca.components_[0]
        summary = np.zeros(mask_flat.shape[0], dtype=np.float32)
        summary[mask_flat] = proj.astype(np.float32)
    elif method == "tic":
        summary = flat.sum(axis=1)
    elif method == "maxproj":
        summary = flat.max(axis=1)
    elif method == "meanproj":
        summary = flat.mean(axis=1)
    else:
        raise ValueError(f"Unknown high-d summary method: {method}")

    summary = summary.reshape(img.shape[:2])
    summary = _normalize_map(summary, mask)
    return summary


def compute_quality_score(
    model,
    img,
    zero_top_n,
    num_rand=5,
    batch_pixels=4096,
    seed=None,
    eps=1e-6,
    clip_negative=True,
    norm_percentiles=(1.0, 99.0),
    tissue_threshold=0.0,
    base_viz=None,
    show_progress=True,
):
    """Top-N ablation importance map, baseline-corrected by intensity-matched random dropout."""
    if zero_top_n < 0:
        raise ValueError("zero_top_n must be non-negative")

    H, W, F = img.shape
    N = min(int(zero_top_n), F)

    if base_viz is None:
        base_viz = model.predict(img)
    base_viz = base_viz.astype(np.float32)
    if base_viz.ndim != 3 or base_viz.shape[0] != H or base_viz.shape[1] != W:
        raise ValueError(f"Expected base_viz shape (H,W,C) = ({H},{W},C). Got {base_viz.shape}")

    mask = (img.sum(axis=-1) > tissue_threshold)
    if not mask.any() or N == 0:
        return np.zeros((H, W), dtype=np.float32)

    flat = img.reshape(-1, F).astype(np.float32, copy=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flat_t = torch.from_numpy(flat).to(device)

    top_vals, top_idx = torch.topk(flat_t, k=N, dim=1)
    top_sum = top_vals.sum(dim=1)

    masked_top_t = flat_t.clone()
    masked_top_t.scatter_(1, top_idx, 0.0)
    masked_top_img = masked_top_t.detach().cpu().numpy().reshape(H, W, F)
    masked_top_viz = model.predict(masked_top_img).astype(np.float32)

    if seed is None:
        seed = int(getattr(model, "random_state", 42))
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))

    def make_mass_matched_random_masked_flat():
        masked_rand = np.empty_like(flat)
        total_pixels = flat_t.shape[0]
        total_batches = (total_pixels + batch_pixels - 1) // batch_pixels
        iterator = range(0, total_pixels, batch_pixels)
        if show_progress:
            iterator = tqdm.tqdm(iterator, total=total_batches, desc="Random baseline", leave=False)
        for start in iterator:
            end = min(start + batch_pixels, total_pixels)
            values = flat_t[start:end]
            top_idx_b = top_idx[start:end]
            top_sum_b = top_sum[start:end]

            non_top_vals = values.clone()
            non_top_vals.scatter_(1, top_idx_b, 0.0)

            scores = torch.rand(non_top_vals.shape, generator=rng, device=device)
            scores.scatter_(1, top_idx_b, 2.0)
            order = torch.argsort(scores, dim=1)

            ordered_vals = torch.gather(non_top_vals, 1, order)
            cumsum = ordered_vals.cumsum(dim=1)

            target = top_sum_b.unsqueeze(1)
            has_target = (top_sum_b > 0)
            select_mask = (cumsum < target) & has_target.unsqueeze(1)
            reached = (cumsum[:, -1] >= top_sum_b) & has_target
            if reached.any():
                idx_first = torch.argmax((cumsum >= target).int(), dim=1)
                rows = torch.arange(values.shape[0], device=device)
                select_mask[rows, idx_first] |= reached

            drop_mask = torch.zeros_like(select_mask, dtype=torch.bool)
            drop_mask.scatter_(1, order, select_mask)

            masked_batch = values.clone()
            masked_batch[drop_mask] = 0.0
            masked_rand[start:end] = masked_batch.detach().cpu().numpy()

        return masked_rand

    diff_top = base_viz - masked_top_viz
    delta_top = np.linalg.norm(diff_top, axis=-1).astype(np.float32)

    delta_rand_acc = np.zeros((H, W), dtype=np.float32)
    for _ in range(int(num_rand)):
        masked_rand_flat = make_mass_matched_random_masked_flat()
        masked_rand_img = masked_rand_flat.reshape(H, W, F)
        masked_rand_viz = model.predict(masked_rand_img).astype(np.float32)
        diff_rand = base_viz - masked_rand_viz
        delta_rand = np.linalg.norm(diff_rand, axis=-1).astype(np.float32)
        delta_rand_acc += delta_rand

    delta_rand_mean = delta_rand_acc / max(int(num_rand), 1)
    raw = (delta_top - delta_rand_mean).astype(np.float32)
    raw[~mask] = 0.0

    if clip_negative:
        raw = np.maximum(raw, 0.0)

    lo_p, hi_p = norm_percentiles
    tissue_vals = raw[mask]
    if tissue_vals.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    q_lo = float(np.percentile(tissue_vals, lo_p))
    q_hi = float(np.percentile(tissue_vals, hi_p))
    if q_hi <= q_lo + 1e-12:
        rel01 = np.zeros((H, W), dtype=np.float32)
    else:
        rel01 = (raw - q_lo) / (q_hi - q_lo + eps)
        rel01 = np.clip(rel01, 0.0, 1.0).astype(np.float32)

    rel01[~mask] = 0.0
    return rel01


def save_quality_score_image(
    model,
    img,
    output_path,
    filename,
    zero_top_n,
    base_viz=None,
    show_progress=True,
    num_rand=5,
):
    quality = compute_quality_score(
        model,
        img,
        zero_top_n,
        num_rand=num_rand,
        base_viz=base_viz,
        show_progress=show_progress,
    )
    quality_uint8 = (quality * 255).clip(0, 255).astype(np.uint8)
    equalized = _apply_equalization(quality_uint8)
    cv2.imwrite(str(output_path / f"{filename}_quality.png"), equalized)
    return equalized


def compute_edge_quality(model, img, window_size=9, highd_summary="pca1", base_viz=None, summary=None):
    mask = img.sum(axis=-1) > 0
    if summary is None:
        summary = compute_highd_summary(img, method=highd_summary, random_state=getattr(model, "random_state", 42))
    if base_viz is None:
        base_viz = model.predict(img)
    viz_gray = _to_grayscale(base_viz)

    gx_s = cv2.Sobel(summary, cv2.CV_32F, 1, 0, ksize=3)
    gy_s = cv2.Sobel(summary, cv2.CV_32F, 0, 1, ksize=3)
    gx_v = cv2.Sobel(viz_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_v = cv2.Sobel(viz_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_s = np.sqrt(gx_s * gx_s + gy_s * gy_s)
    edge_v = np.sqrt(gx_v * gx_v + gy_v * gy_v)

    return _local_correlation(edge_s, edge_v, window_size, mask)


def save_edge_quality_image(model, img, output_path, filename, window_size=9, highd_summary="pca1", base_viz=None, summary=None):
    quality = compute_edge_quality(model, img, window_size, highd_summary, base_viz=base_viz, summary=summary)
    quality_uint8 = (quality * 255).clip(0, 255).astype(np.uint8)
    equalized = _apply_equalization(quality_uint8)
    cv2.imwrite(str(output_path / f"{filename}_edge_quality.png"), equalized)
    return equalized


def compute_structure_quality(model, img, window_size=5, highd_summary="pca1", base_viz=None, summary=None):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    mask = img.sum(axis=-1) > 0
    if summary is None:
        summary = compute_highd_summary(img, method=highd_summary, random_state=getattr(model, "random_state", 42))
    if base_viz is None:
        base_viz = model.predict(img)
    viz_gray = _to_grayscale(base_viz)

    pad = window_size // 2
    summary_p = cv2.copyMakeBorder(summary, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
    viz_p = cv2.copyMakeBorder(viz_gray, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)

    num_offsets = window_size * window_size - 1
    sum_h = np.zeros_like(summary, dtype=np.float32)
    sum_v = np.zeros_like(summary, dtype=np.float32)
    sum_h2 = np.zeros_like(summary, dtype=np.float32)
    sum_v2 = np.zeros_like(summary, dtype=np.float32)
    sum_hv = np.zeros_like(summary, dtype=np.float32)

    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            if dy == 0 and dx == 0:
                continue
            shifted_h = summary_p[pad + dy:pad + dy + summary.shape[0], pad + dx:pad + dx + summary.shape[1]]
            shifted_v = viz_p[pad + dy:pad + dy + summary.shape[0], pad + dx:pad + dx + summary.shape[1]]
            diff_h = np.abs(summary - shifted_h)
            diff_v = np.abs(viz_gray - shifted_v)
            sum_h += diff_h
            sum_v += diff_v
            sum_h2 += diff_h * diff_h
            sum_v2 += diff_v * diff_v
            sum_hv += diff_h * diff_v

    mean_h = sum_h / num_offsets
    mean_v = sum_v / num_offsets
    cov = (sum_hv / num_offsets) - mean_h * mean_v
    var_h = (sum_h2 / num_offsets) - mean_h * mean_h
    var_v = (sum_v2 / num_offsets) - mean_v * mean_v
    denom = np.sqrt(np.maximum(var_h, 0.0) * np.maximum(var_v, 0.0)) + 1e-6
    corr = cov / denom
    corr = np.clip((corr + 1.0) * 0.5, 0.0, 1.0)
    corr[~mask] = 0.0
    return corr


def save_structure_quality_image(model, img, output_path, filename, window_size=5, highd_summary="pca1", base_viz=None, summary=None):
    quality = compute_structure_quality(model, img, window_size, highd_summary, base_viz=base_viz, summary=summary)
    quality_uint8 = (quality * 255).clip(0, 255).astype(np.uint8)
    equalized = _apply_equalization(quality_uint8)
    cv2.imwrite(str(output_path / f"{filename}_structure_quality.png"), equalized)
    return equalized


def compute_contrast_quality(model, img, window_size=9, highd_summary="pca1", base_viz=None, summary=None):
    mask = img.sum(axis=-1) > 0
    if summary is None:
        summary = compute_highd_summary(img, method=highd_summary, random_state=getattr(model, "random_state", 42))
    if base_viz is None:
        base_viz = model.predict(img)
    viz_gray = _to_grayscale(base_viz)

    ksize = (window_size, window_size)
    mean_s = cv2.boxFilter(summary, ddepth=-1, ksize=ksize, normalize=True)
    mean_s2 = cv2.boxFilter(summary * summary, ddepth=-1, ksize=ksize, normalize=True)
    std_s = np.sqrt(np.maximum(mean_s2 - mean_s * mean_s, 0.0))

    mean_v = cv2.boxFilter(viz_gray, ddepth=-1, ksize=ksize, normalize=True)
    mean_v2 = cv2.boxFilter(viz_gray * viz_gray, ddepth=-1, ksize=ksize, normalize=True)
    std_v = np.sqrt(np.maximum(mean_v2 - mean_v * mean_v, 0.0))

    diff = np.abs(std_v - std_s)
    diff = _normalize_map(diff, mask)
    quality = 1.0 - diff
    quality[~mask] = 0.0
    return np.clip(quality, 0.0, 1.0)


def save_contrast_quality_image(model, img, output_path, filename, window_size=9, highd_summary="pca1", base_viz=None, summary=None):
    quality = compute_contrast_quality(model, img, window_size, highd_summary, base_viz=base_viz, summary=summary)
    quality_uint8 = (quality * 255).clip(0, 255).astype(np.uint8)
    equalized = _apply_equalization(quality_uint8)
    cv2.imwrite(str(output_path / f"{filename}_contrast_quality.png"), equalized)
    return equalized
