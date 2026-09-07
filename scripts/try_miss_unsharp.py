"""Miss-driven / luminance-only unsharp sweep vs SoLaCE HD edges (ID9)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bayes_tune_mics_lmc import _compute_viz_edge_n, _to_uint8_rgb  # noqa: E402
from debug_edge_maps import _gray_u8_to_rgb_colormap, _to_uint8_gray01  # noqa: E402
from experiment_mics_gnn import _global_continuous_dice, _load_rank_cfg  # noqa: E402
from hydra.utils import to_absolute_path  # noqa: E402
from msi_visual.parametric_mics_lmc import (  # noqa: E402
    _spatial_gaussian_smooth_channels,
    solace_guided_bilateral_smooth,
)


def _valid_mask_from_msi(npy_path: Path) -> np.ndarray:
    msi = np.load(str(npy_path), mmap_mode="r")
    return np.asarray(msi).sum(axis=-1) > 0


def _norm01(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[mask]
    if vals.size == 0:
        return out
    hi = float(np.percentile(vals, 99.0))
    if hi <= 1e-8:
        return out
    out[mask] = np.clip(vals / hi, 0.0, 1.0)
    return out


def _miss_map(solace: np.ndarray, viz_edge: np.ndarray, mask: np.ndarray) -> np.ndarray:
    e = _norm01(np.asarray(solace, dtype=np.float32), mask)
    v = _norm01(np.asarray(viz_edge, dtype=np.float32), mask)
    m = e * (1.0 - v)
    m[~mask] = 0.0
    return m


def _unsharp_rgb(
    rgb: np.ndarray,
    weight: np.ndarray,
    *,
    amount: float,
    sigma: float,
    mask: np.ndarray,
    luma_only: bool,
) -> np.ndarray:
    """x <- x + amount * weight * (x - G_sigma(x)), optional Lab L only."""
    x = rgb.astype(np.float32)
    w = np.clip(np.asarray(weight, dtype=np.float32), 0.0, 1.0)
    if luma_only:
        lab = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0:1]
        blur = _spatial_gaussian_smooth_channels(L, float(sigma))
        high = L - blur
        lab[:, :, 0:1] = np.clip(L + float(amount) * w[..., None] * high, 0.0, 255.0)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    else:
        blur = _spatial_gaussian_smooth_channels(x, float(sigma))
        high = x - blur
        out_f = np.clip(x + float(amount) * w[..., None] * high, 0.0, 255.0)
        out = out_f.astype(np.uint8)
    out[~mask] = 0
    return out


def main() -> None:
    run_dir = _ROOT / "outputs" / "experiment_mics_gnn" / "ID9_pos_unet_sharp"
    out_dir = _ROOT / "outputs" / "experiment_mics_gnn" / "ID9_pos_miss_unsharp"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(run_dir / "config_resolved.yaml")
    cfg.rank_config = to_absolute_path(str(cfg.rank_config))
    rank_cfg = _load_rank_cfg(cfg)

    mics = np.asarray(Image.open(run_dir / "stage1_mics_viz.png").convert("RGB"), dtype=np.uint8)
    hd = np.load(run_dir / "hd_edges.npy").astype(np.float32)
    mask = _valid_mask_from_msi(Path(str(cfg.data.npy_path)))

    viz_edge = _compute_viz_edge_n(mics, mask, rank_cfg).astype(np.float32)
    miss = _miss_map(hd, viz_edge, mask)
    solace_n = _norm01(hd, mask)

    Image.fromarray(_gray_u8_to_rgb_colormap(_to_uint8_gray01(miss), "magma")).save(
        out_dir / "miss_map.png"
    )
    Image.fromarray(_gray_u8_to_rgb_colormap(_to_uint8_gray01(viz_edge), "magma")).save(
        out_dir / "mics_viz_edge.png"
    )

    rows: list[dict] = []

    def score(name: str, rgb: np.ndarray) -> float:
        d = _global_continuous_dice(cfg, rgb, hd, mask)
        rows.append({"method": name, "dice": float(d)})
        print(f"{name:48s}  Dice={d:.4f}")
        return float(d)

    base = score("mics", mics)

    # Prior best from blend sweep
    bil_sharp = solace_guided_bilateral_smooth(
        mics.astype(np.float32),
        hd,
        mask=mask,
        spatial_sigma=1.5,
        edge_scale=0.12,
        iterations=4,
        sharpen_amount=0.8,
        sharpen_sigma=1.0,
    )
    bil_sharp_u8 = _to_uint8_rgb(bil_sharp)
    bil_sharp_u8[~mask] = 0
    score("bilateral_sharpen_ref", bil_sharp_u8)

    # Grid: weight kind × luma_only × amount × sigma × optional light bilateral first
    amounts = [0.8, 1.5, 2.5, 3.5]
    sigmas = [0.6, 1.0, 1.5]
    gallery: list[tuple[str, np.ndarray, float]] = [("MiCS", mics, base)]

    configs: list[tuple[str, np.ndarray, bool, bool]] = [
        ("solace", solace_n, False, False),
        ("miss", miss, False, False),
        ("miss_L", miss, True, False),
        ("miss_L_prebil", miss, True, True),
        ("solace_L", solace_n, True, False),
    ]

    best_name, best_dice, best_rgb = "mics", base, mics

    for wname, weight, luma_only, prebil in configs:
        for amount in amounts:
            for sigma in sigmas:
                src = mics
                if prebil:
                    sm = solace_guided_bilateral_smooth(
                        mics.astype(np.float32),
                        hd,
                        mask=mask,
                        spatial_sigma=1.5,
                        edge_scale=0.12,
                        iterations=2,
                        sharpen_amount=0.0,
                    )
                    src = _to_uint8_rgb(sm)
                    src[~mask] = 0
                rgb = _unsharp_rgb(
                    src, weight, amount=amount, sigma=sigma, mask=mask, luma_only=luma_only
                )
                tag = f"{wname}_a{amount:g}_s{sigma:g}"
                d = score(tag, rgb)
                if d > best_dice:
                    best_dice, best_name, best_rgb = d, tag, rgb
                # keep a few mid/high settings for the panel
                if (wname, amount, sigma) in {
                    ("miss_L", 1.5, 1.0),
                    ("miss_L", 2.5, 1.0),
                    ("miss_L", 3.5, 0.6),
                    ("solace_L", 2.5, 1.0),
                    ("miss", 2.5, 1.0),
                    ("miss_L_prebil", 2.5, 1.0),
                }:
                    Image.fromarray(rgb).save(out_dir / f"{tag}.png")
                    gallery.append((tag.replace("_", " "), rgb, d))

    Image.fromarray(best_rgb).save(out_dir / f"best__{best_name}.png")

    csv_path = out_dir / "miss_unsharp_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "dice"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["dice"]))

    # Top-8 panel by Dice among saved + refs
    ranked = sorted(rows, key=lambda r: -r["dice"])[:8]
    # rebuild panel from best methods: load from saved or recompute known
    panel_items: list[tuple[str, np.ndarray, float]] = []
    # always include mics + bil_sharp + best
    panel_items.append(("MiCS", mics, base))
    panel_items.append(("bil+sharp", bil_sharp_u8, next(r["dice"] for r in rows if r["method"] == "bilateral_sharpen_ref")))
    panel_items.append((f"BEST\n{best_name}", best_rgb, best_dice))

    # add a few distinct high scorers from gallery (dedupe)
    seen = {id(mics), id(bil_sharp_u8), id(best_rgb)}
    for title, rgb, d in sorted(gallery, key=lambda t: -t[2]):
        if id(rgb) in seen:
            continue
        panel_items.append((title, rgb, d))
        seen.add(id(rgb))
        if len(panel_items) >= 8:
            break

    n = len(panel_items)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.4), dpi=140)
    if n == 1:
        axes = [axes]
    for ax, (title, rgb, d) in zip(axes, panel_items):
        ax.imshow(rgb)
        ax.set_title(f"{title}\n{d:.4f}", fontsize=7)
        ax.axis("off")
    fig.suptitle(
        f"ID9 miss-unsharp sweep  best={best_dice:.4f} ({best_dice - base:+.4f})",
        fontsize=11,
        fontweight="600",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "miss_unsharp_panel.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {out_dir}")
    print(f"MiCS={base:.4f}  best={best_name}  Dice={best_dice:.4f} ({best_dice - base:+.4f})")
    print("Top 10:")
    for r in ranked[:10]:
        print(f"  {r['dice']:.4f}  {r['method']}")


if __name__ == "__main__":
    main()
