"""
Path-Edge Spectral Consistency (PESC) for soft MSI edge maps (no ground truth).

Sample random tissue pixel pairs (p, q) at controlled spatial distance. For each pair:
  s(p,q) = spectral L2 between endpoints (PCA space)
  e(p,q) = max edge response along the straight line from p to q
           (edge map is rank-normalized within tissue so methods are scale-fair)

PESC = mean(e | s in top quantile) - mean(e | s in bottom quantile)
Also report Spearman(s, e). Higher = edges appear between spectrally different points
and not between similar ones (penalizes clutter / false ridges).
"""
from __future__ import annotations

import numpy as np


def _as_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _prepare_spectral_channels(
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    pca_components: int = 64,
    random_state: int = 42,
) -> np.ndarray:
    """Per-channel z-score then optional PCA for fast spectral distances."""
    h, w, c = msi.shape
    m = np.asarray(valid_mask, dtype=bool)
    channels = np.zeros((h, w, c), dtype=np.float32)
    for i in range(c):
        ch = np.asarray(msi[:, :, i], dtype=np.float32)
        vals = ch[m]
        if vals.size == 0:
            continue
        mu = float(vals.mean())
        std = float(vals.std())
        if std < 1e-8:
            continue
        channels[:, :, i] = (ch - mu) / (std + 1e-8)
    channels[~m] = 0.0

    k = int(pca_components)
    if k <= 0 or k >= c:
        return channels

    from sklearn.decomposition import PCA

    X = channels[m]
    n_comp = min(k, X.shape[0], X.shape[1])
    if n_comp < 1:
        return channels
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=int(random_state))
    Z = pca.fit_transform(X).astype(np.float32, copy=False)
    out = np.zeros((h, w, n_comp), dtype=np.float32)
    out[m] = Z
    return out


def _rank_normalize_edge(edge: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map valid edge values to ranks in (0, 1]; invalid pixels stay 0."""
    out = np.zeros(mask.shape, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    vals = _as_f32(edge)[m]
    n = int(vals.size)
    if n == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = (np.arange(1, n + 1, dtype=np.float32) / float(n))
    out[m] = ranks
    return out


def _sample_pairs(
    mask: np.ndarray,
    *,
    n_pairs: int,
    dist_min: float,
    dist_max: float,
    rng: np.random.Generator,
    max_tries_factor: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample random valid pixel pairs with Euclidean distance in [dist_min, dist_max]."""
    ys, xs = np.nonzero(mask)
    n = int(ys.size)
    if n < 2:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty, empty, empty

    d0 = float(dist_min)
    d1 = float(dist_max)
    y0_list: list[int] = []
    x0_list: list[int] = []
    y1_list: list[int] = []
    x1_list: list[int] = []
    target = int(n_pairs)
    tries = 0
    max_tries = max(target * int(max_tries_factor), target)
    while len(y0_list) < target and tries < max_tries:
        need = target - len(y0_list)
        batch = min(max(need * 4, 256), 8192)
        i0 = rng.integers(0, n, size=batch)
        i1 = rng.integers(0, n, size=batch)
        same = i0 == i1
        if np.any(same):
            i1[same] = (i1[same] + 1) % n
        yy0 = ys[i0].astype(np.float64)
        xx0 = xs[i0].astype(np.float64)
        yy1 = ys[i1].astype(np.float64)
        xx1 = xs[i1].astype(np.float64)
        dist = np.sqrt((yy0 - yy1) ** 2 + (xx0 - xx1) ** 2)
        ok = (dist >= d0) & (dist <= d1)
        if np.any(ok):
            y0_list.extend(ys[i0[ok]].tolist())
            x0_list.extend(xs[i0[ok]].tolist())
            y1_list.extend(ys[i1[ok]].tolist())
            x1_list.extend(xs[i1[ok]].tolist())
        tries += batch
    n_keep = min(len(y0_list), target)
    return (
        np.asarray(y0_list[:n_keep], dtype=np.int32),
        np.asarray(x0_list[:n_keep], dtype=np.int32),
        np.asarray(y1_list[:n_keep], dtype=np.int32),
        np.asarray(x1_list[:n_keep], dtype=np.int32),
    )


def _line_max_edge_fast(
    edge_rank: np.ndarray,
    mask: np.ndarray,
    y0: np.ndarray,
    x0: np.ndarray,
    y1: np.ndarray,
    x1: np.ndarray,
) -> np.ndarray:
    """
    Max rank-normalized edge along each segment. Groups pairs by step count and uses
    numpy broadcasting within each group.
    """
    n = int(y0.size)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return out

    steps = np.maximum(
        np.abs(y1.astype(np.int32) - y0.astype(np.int32)),
        np.abs(x1.astype(np.int32) - x0.astype(np.int32)),
    )
    steps = np.maximum(steps, 1)
    h, w = edge_rank.shape

    for t in np.unique(steps):
        idx = np.flatnonzero(steps == t)
        if idx.size == 0:
            continue
        tt = int(t)
        alphas = (np.arange(tt + 1, dtype=np.float64) / float(tt))[None, :]
        yy = np.rint(y0[idx, None].astype(np.float64) + (y1[idx] - y0[idx])[:, None].astype(np.float64) * alphas)
        xx = np.rint(x0[idx, None].astype(np.float64) + (x1[idx] - x0[idx])[:, None].astype(np.float64) * alphas)
        yi = np.clip(yy.astype(np.int32), 0, h - 1)
        xi = np.clip(xx.astype(np.int32), 0, w - 1)
        on = mask[yi, xi]
        vals = edge_rank[yi, xi]
        # Exclude endpoints: score the path *between* the pair, not the pixels themselves.
        if tt >= 2:
            on = on[:, 1:-1]
            vals = vals[:, 1:-1]
        vals = np.where(on, vals, -np.inf)
        mx = vals.max(axis=1)
        good = np.isfinite(mx) & (mx > -np.inf)
        out[idx[good]] = mx[good].astype(np.float32)
    return out


def pesc_for_edge(
    edge: np.ndarray,
    channels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    y0: np.ndarray,
    x0: np.ndarray,
    y1: np.ndarray,
    x1: np.ndarray,
    high_frac: float = 0.20,
    low_frac: float = 0.20,
) -> dict[str, float]:
    """Compute PESC (+ Spearman) for one edge map given fixed pair endpoints."""
    m = np.asarray(valid_mask, dtype=bool)
    if y0.size == 0:
        return {
            "pesc": float("nan"),
            "spearman": float("nan"),
            "e_high": float("nan"),
            "e_low": float("nan"),
            "n_pairs": 0.0,
        }

    edge_rank = _rank_normalize_edge(edge, m)
    e_line = _line_max_edge_fast(edge_rank, m, y0, x0, y1, x1)
    s = np.linalg.norm(channels[y0, x0, :] - channels[y1, x1, :], axis=-1).astype(np.float64)

    ok = np.isfinite(e_line) & np.isfinite(s)
    e_line = e_line[ok].astype(np.float64)
    s = s[ok]
    n = int(s.size)
    if n < 20:
        return {
            "pesc": float("nan"),
            "spearman": float("nan"),
            "e_high": float("nan"),
            "e_low": float("nan"),
            "n_pairs": float(n),
        }

    # Quantile split on spectral distance
    hi = float(np.quantile(s, 1.0 - float(high_frac)))
    lo = float(np.quantile(s, float(low_frac)))
    high_mask = s >= hi
    low_mask = s <= lo
    e_high = float(np.mean(e_line[high_mask])) if np.any(high_mask) else float("nan")
    e_low = float(np.mean(e_line[low_mask])) if np.any(low_mask) else float("nan")
    pesc = e_high - e_low if np.isfinite(e_high) and np.isfinite(e_low) else float("nan")

    # Spearman via rank correlation
    rs = s.argsort(kind="mergesort").argsort().astype(np.float64)
    re = e_line.argsort(kind="mergesort").argsort().astype(np.float64)
    rs = (rs - rs.mean()) / (rs.std() + 1e-12)
    re = (re - re.mean()) / (re.std() + 1e-12)
    spearman = float(np.mean(rs * re))

    return {
        "pesc": float(pesc),
        "spearman": spearman,
        "e_high": e_high,
        "e_low": e_low,
        "n_pairs": float(n),
    }


def evaluate_edge_methods(
    edge_by_method: dict[str, np.ndarray],
    msi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    pca_components: int = 64,
    n_pairs: int = 12000,
    dist_min: float = 5.0,
    dist_max: float = 40.0,
    high_frac: float = 0.20,
    low_frac: float = 0.20,
    random_state: int = 42,
    **_ignored,
) -> dict[str, dict]:
    """
    Per-method PESC only (same random pairs for all methods on a slide).

    Returns dict with: pesc, spearman, e_high, e_low, n_pairs.
    """
    m = np.asarray(valid_mask, dtype=bool)
    channels = _prepare_spectral_channels(msi, m, pca_components=pca_components, random_state=random_state)
    rng = np.random.default_rng(int(random_state))
    y0, x0, y1, x1 = _sample_pairs(
        m,
        n_pairs=int(n_pairs),
        dist_min=float(dist_min),
        dist_max=float(dist_max),
        rng=rng,
    )

    out: dict[str, dict] = {}
    for name, edge in edge_by_method.items():
        met = pesc_for_edge(
            edge,
            channels,
            m,
            y0=y0,
            x0=x0,
            y1=y1,
            x1=x1,
            high_frac=high_frac,
            low_frac=low_frac,
        )
        out[name] = met
    return out


def format_panel_metric_lines(method_metrics: dict) -> list[str]:
    """Short lines for gallery title boxes."""
    pesc = method_metrics.get("pesc", float("nan"))
    sp = method_metrics.get("spearman", float("nan"))
    e_hi = method_metrics.get("e_high", float("nan"))
    e_lo = method_metrics.get("e_low", float("nan"))
    return [
        f"PESC {pesc:.3f}",
        f"Spearman {sp:.3f}",
        f"e|s↑ {e_hi:.3f}  e|s↓ {e_lo:.3f}",
    ]
