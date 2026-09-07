"""Alpha-sweep: blend MiCS RGB with SoLaCE (edge RGB / edge-masked / bilateral).

Uses existing ID9 experiment artifacts by default. Scores continuous Dice vs HD edges
with the same path as experiment_mics_gnn.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

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

from bayes_tune_mics_lmc import _to_uint8_rgb  # noqa: E402
from debug_edge_maps import _gray_u8_to_rgb_colormap, _to_uint8_gray01  # noqa: E402
from experiment_mics_gnn import _global_continuous_dice  # noqa: E402
from hydra.utils import to_absolute_path  # noqa: E402
from msi_visual.parametric_mics_lmc import solace_guided_bilateral_smooth  # noqa: E402


def _valid_mask_from_msi(npy_path: Path) -> np.ndarray:
    msi = np.load(str(npy_path), mmap_mode="r")
    return np.asarray(msi).sum(axis=-1) > 0


def _solace_edge_rgb(hd: np.ndarray, mask: np.ndarray, cmap: str = "PuBu") -> np.ndarray:
    e = np.clip(np.asarray(hd, dtype=np.float32), 0.0, 1.0)
    e[~mask] = 0.0
    rgb = _gray_u8_to_rgb_colormap(_to_uint8_gray01(e), cmap)
    rgb[~mask] = 0
    return rgb


def _solace_gray_rgb(hd: np.ndarray, mask: np.ndarray) -> np.ndarray:
    e = np.clip(np.asarray(hd, dtype=np.float32), 0.0, 1.0)
    g = _to_uint8_gray01(e)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[~mask] = 0
    return rgb


def _blend(mics: np.ndarray, other: np.ndarray, alpha: float, mask: np.ndarray) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - a) * mics.astype(np.float32) + a * other.astype(np.float32)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    out[~mask] = 0
    return out


def _edge_masked_blend(
    mics: np.ndarray, other: np.ndarray, alpha: float, edge: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    w = np.clip(np.asarray(edge, dtype=np.float32), 0.0, 1.0)[..., None] * a
    out = (1.0 - w) * mics.astype(np.float32) + w * other.astype(np.float32)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    out[~mask] = 0
    return out


def _edge_luminance_boost(
    mics: np.ndarray, edge: np.ndarray, alpha: float, mask: np.ndarray
) -> np.ndarray:
    """Push MiCS toward white along SoLaCE edges (keeps hue better than PuBu wash)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    w = np.clip(np.asarray(edge, dtype=np.float32), 0.0, 1.0)[..., None] * a
    white = np.full_like(mics, 255, dtype=np.float32)
    out = (1.0 - w) * mics.astype(np.float32) + w * white
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    out[~mask] = 0
    return out


def main() -> None:
    run_dir = _ROOT / "outputs" / "experiment_mics_gnn" / "ID9_pos_unet_sharp"
    out_dir = _ROOT / "outputs" / "experiment_mics_gnn" / "ID9_pos_solace_blend"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(run_dir / "config_resolved.yaml")
    cfg.rank_config = to_absolute_path(str(cfg.rank_config))

    mics = np.asarray(Image.open(run_dir / "stage1_mics_viz.png").convert("RGB"), dtype=np.uint8)
    hd = np.load(run_dir / "hd_edges.npy").astype(np.float32)
    npy_path = Path(str(cfg.data.npy_path))
    mask = _valid_mask_from_msi(npy_path)
    if mask.shape != hd.shape:
        raise ValueError(f"mask {mask.shape} != hd {hd.shape}")

    solace_cmap = _solace_edge_rgb(hd, mask, "PuBu")
    solace_gray = _solace_gray_rgb(hd, mask)
    Image.fromarray(solace_cmap).save(out_dir / "solace_edges_pubu.png")
    Image.fromarray(solace_gray).save(out_dir / "solace_edges_gray.png")

    alphas = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    rows: list[dict] = []

    def score(name: str, rgb: np.ndarray) -> float:
        d = _global_continuous_dice(cfg, rgb, hd, mask)
        rows.append({"method": name, "dice": d})
        print(f"{name:40s}  Dice={d:.4f}")
        return d

    score("mics", mics)
    score("solace_pubu_alone", solace_cmap)
    score("solace_gray_alone", solace_gray)

    gallery: list[tuple[str, np.ndarray, float]] = [("MiCS", mics, rows[-3]["dice"])]

    for a in alphas[1:]:
        rgb = _blend(mics, solace_cmap, a, mask)
        d = score(f"full_blend_pubu_a{a:g}", rgb)
        Image.fromarray(rgb).save(out_dir / f"full_blend_pubu_a{a:g}.png")
        if a in (0.2, 0.35, 0.5):
            gallery.append((f"full α={a:g}", rgb, d))

    for a in (0.35, 0.5, 0.75, 1.0):
        rgb = _blend(mics, solace_gray, a, mask)
        d = score(f"full_blend_gray_a{a:g}", rgb)
        Image.fromarray(rgb).save(out_dir / f"full_blend_gray_a{a:g}.png")
        if a == 0.5:
            gallery.append((f"gray α={a:g}", rgb, d))

    for a in (0.35, 0.5, 0.75, 1.0):
        rgb = _edge_masked_blend(mics, solace_cmap, a, hd, mask)
        d = score(f"edge_mask_pubu_a{a:g}", rgb)
        Image.fromarray(rgb).save(out_dir / f"edge_mask_pubu_a{a:g}.png")
        if a == 0.75:
            gallery.append((f"edge-mask α={a:g}", rgb, d))

    for a in (0.2, 0.35, 0.5, 0.75):
        rgb = _edge_luminance_boost(mics, hd, a, mask)
        d = score(f"edge_white_boost_a{a:g}", rgb)
        Image.fromarray(rgb).save(out_dir / f"edge_white_boost_a{a:g}.png")
        if a == 0.5:
            gallery.append((f"white-edge α={a:g}", rgb, d))

    # SoLaCE-guided bilateral on float MiCS RGB (+ optional edge sharpen)
    base_f = mics.astype(np.float32)
    for label, kwargs in (
        ("bilateral", dict(spatial_sigma=1.5, edge_scale=0.15, iterations=4, sharpen_amount=0.0)),
        (
            "bilateral_sharpen",
            dict(spatial_sigma=1.5, edge_scale=0.12, iterations=4, sharpen_amount=0.8, sharpen_sigma=1.0),
        ),
    ):
        sm = solace_guided_bilateral_smooth(base_f, hd, mask=mask, **kwargs)
        rgb = _to_uint8_rgb(sm)
        rgb[~mask] = 0
        d = score(label, rgb)
        Image.fromarray(rgb).save(out_dir / f"{label}.png")
        gallery.append((label.replace("_", " "), rgb, d))

    csv_path = out_dir / "blend_dice_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "dice"])
        w.writeheader()
        w.writerows(rows)

    # Compact visual panel of representative blends
    n = len(gallery)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.2), dpi=140)
    if n == 1:
        axes = [axes]
    for ax, (title, rgb, d) in zip(axes, gallery):
        ax.imshow(rgb)
        ax.set_title(f"{title}\nDice {d:.4f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("ID9 MiCS × SoLaCE blend sweep", fontsize=11, fontweight="600")
    fig.tight_layout()
    fig.savefig(out_dir / "blend_panel.png", bbox_inches="tight")
    plt.close(fig)

    best = max(rows, key=lambda r: r["dice"])
    base = next(r for r in rows if r["method"] == "mics")
    print(f"\nWrote {out_dir}")
    print(f"MiCS Dice={base['dice']:.4f}  best={best['method']} Dice={best['dice']:.4f} "
          f"({best['dice'] - base['dice']:+.4f})")


if __name__ == "__main__":
    main()
