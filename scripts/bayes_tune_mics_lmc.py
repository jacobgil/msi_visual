#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (_REPO, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from debug_edge_maps import (  # noqa: E402
    _compute_continuous_dice,
    _compute_viz_contrast_color_metrics,
    _edge_maps_multiscale,
    _edges_for_continuous_metrics,
    _gray_u8_to_rgb_colormap,
    _load_msi,
    _normalize_map_percentile,
    _normalize_on_mask,
    _parse_edge_detection_cfg,
    _resolve_normalization_mode,
    _to_uint8_gray01,
)
from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC  # noqa: E402
from rank_visualizations_by_f1 import _compute_hd_edge_n_for_method  # noqa: E402
from train_parametric_mics_lmc import _clusters_auto_tune_kw_from_cfg  # noqa: E402

logger = logging.getLogger(__name__)

_METRIC_KEYS = ("continuous_dice", "luminance_rms_contrast", "lab_chroma_entropy")


def _omega_subtree_to_plain_dict(sel: Any) -> dict[str, Any]:
    """Resolve DictConfig subtree to plain dict. OmegaConf.to_container rejects a raw dict (e.g. select default {})."""
    if sel is None:
        return {}
    if OmegaConf.is_config(sel):
        oc = OmegaConf.to_container(sel, resolve=True)
        return dict(oc) if isinstance(oc, dict) else {}
    if isinstance(sel, dict):
        return dict(sel)
    return {}


def _safe_token(value: object, max_len: int = 160) -> str:
    s = str(value)
    out = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "-" for ch in s)
    out = out.strip("-_")
    return (out[:max_len] if out else "x") or "x"


_PARAM_NAME_ALIASES = {
    "beta": "b",
    "cluster_features": "clf",
    "cluster_ladder": "clad",
    "cluster_pca_dims": "cpd",
    "clusters": "cl",
    "factor": "fac",
    "k_epoch": "ke",
    "lr": "lr",
    "num_epochs": "ep",
    "num_layers": "nl",
    "num_samples": "ns",
    "number_of_points": "pts",
    "pca_fit_step": "pfs",
    "pixel_sampling": "ps",
    "sampling": "s",
    "temperature": "temp",
}


def _param_tag(params: dict[str, Any], max_len: int = 145) -> str:
    parts = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, float):
            v_s = f"{v:.5g}"
        elif isinstance(v, (list, tuple)):
            v_s = "-".join(str(x) for x in v)
        else:
            v_s = str(v)
        key = _PARAM_NAME_ALIASES.get(str(k), _safe_token(k, max_len=24))
        parts.append(f"{key}-{_safe_token(v_s, max_len=36)}")
    full = "__".join(parts)
    digest = hashlib.sha1(repr(sorted(params.items())).encode("utf-8")).hexdigest()[:8]
    if len(full) <= max_len:
        return f"{full}__h-{digest}"
    return f"{full[:max_len].rstrip('-_')}__h-{digest}"


def _to_uint8_rgb(viz: np.ndarray) -> np.ndarray:
    arr = np.asarray(viz)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        out = arr
    else:
        x = arr.astype(np.float32)
        mx = float(np.nanmax(x)) if x.size else 0.0
        if mx <= 1.0 + 1e-6:
            x = x * 255.0
        out = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return out


def _hist_equalize_rgb(rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2

        out = np.asarray(rgb, dtype=np.uint8).copy()
        for ch in range(min(3, out.shape[-1])):
            out[:, :, ch] = cv2.equalizeHist(out[:, :, ch])
        return out
    except Exception:
        return rgb


def _search_space_items(cfg: DictConfig) -> list[tuple[str, dict[str, Any]]]:
    raw = OmegaConf.to_container(cfg.search_space, resolve=True)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("search_space must be a non-empty mapping.")
    items: list[tuple[str, dict[str, Any]]] = []
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"search_space.{name} must be a mapping")
        typ = str(spec.get("type", "float")).strip().lower()
        spec = dict(spec)
        spec["type"] = typ
        items.append((str(name), spec))
    return items


def _load_fixed_params(
    cfg: DictConfig,
    space: list[tuple[str, dict[str, Any]]],
    extra_fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hyperparameters held constant during OFAT / hill (must not overlap ``search_space``)."""
    raw = OmegaConf.select(cfg, "tuning.fixed_params", default={})
    fixed = _omega_subtree_to_plain_dict(raw) if raw is not None else {}
    if not isinstance(fixed, dict):
        fixed = {}
    if extra_fixed:
        fixed = {**fixed, **dict(extra_fixed)}
    space_names = {name for name, _ in space}
    overlap = space_names & set(fixed.keys())
    if overlap:
        raise ValueError(
            "tuning.fixed_params must not overlap search_space keys: "
            + ", ".join(sorted(overlap))
        )
    return dict(fixed)


def _merge_tuned_and_fixed(fixed: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    return {**fixed, **tuned}


def _resolve_cluster_search_params(merged: dict[str, Any]) -> None:
    """Expand ``cluster_ladder`` search tokens into MiCS cluster fields (in-place)."""
    token = merged.pop("cluster_ladder", None)
    if token is None:
        return
    s = str(token).strip()
    if s.lower() in ("", "null", "none", "~"):
        return
    if s.startswith(("silhouette:", "sil:")):
        tpl = s.split(":", 1)[1].strip()
        merged["clusters_auto_tune"] = "silhouette"
        merged["clusters_auto_tune_ladder"] = tpl
        merged["clusters"] = None
        return
    if s.startswith("manual:"):
        ladder = s.split(":", 1)[1].strip()
        merged["clusters_auto_tune"] = None
        merged["clusters"] = ladder
        return
    merged["clusters_auto_tune"] = None
    merged["clusters"] = s


def _resolve_cluster_features_param(merged: dict[str, Any]) -> None:
    """Expand ``cluster_features`` token (``raw`` | ``pca_<n>``) into on/off + dims."""
    token = merged.pop("cluster_features", None)
    if token is None:
        return
    s = str(token).strip().lower()
    if s in ("raw", "native", "spectrum", "none"):
        merged["cluster_on_pca"] = False
        return
    if s.startswith("pca_"):
        merged["cluster_on_pca"] = True
        merged["cluster_pca_dims"] = int(s.split("_", 1)[1])
        return
    raise ValueError(
        f"cluster_features={token!r} invalid; use 'raw' or 'pca_<dims>' (e.g. pca_8)."
    )


def _sample_param(spec: dict[str, Any], rng: np.random.Generator) -> Any:
    typ = str(spec["type"]).lower()
    if typ in {"categorical", "choice", "enum"}:
        choices = list(spec.get("choices", []))
        if not choices:
            raise ValueError("categorical search parameter needs choices")
        return choices[int(rng.integers(0, len(choices)))]
    low = float(spec.get("low"))
    high = float(spec.get("high"))
    log = bool(spec.get("log", False))
    if log:
        value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    else:
        value = float(rng.uniform(low, high))
    if typ in {"int", "integer"}:
        return int(np.clip(round(value), int(round(low)), int(round(high))))
    return value


def _sample_params(space: list[tuple[str, dict[str, Any]]], rng: np.random.Generator) -> dict[str, Any]:
    return {name: _sample_param(spec, rng) for name, spec in space}


def _coerce_param_for_spec(spec: dict[str, Any], raw_val: Any) -> Any:
    """Map optional YAML overrides into typed search_space values."""
    typ = str(spec["type"]).lower()
    if OmegaConf.is_config(raw_val):
        raw_val = OmegaConf.to_container(raw_val, resolve=True)
    if typ in {"categorical", "choice", "enum"}:
        choices = list(spec.get("choices", []))
        if raw_val not in choices:
            if len(choices) == 2 and isinstance(raw_val, (str, bool, int)):
                for cand in choices:
                    if str(cand) == str(raw_val):
                        raw_val = cand
                        break
                else:
                    raise ValueError(f"OFAT baseline value {raw_val!r} not in choices={choices}")
            else:
                raise ValueError(f"OFAT baseline value {raw_val!r} must be one of choices={choices}")
        return raw_val
    if typ in {"int", "integer"}:
        return int(raw_val)
    return float(raw_val)


def _baseline_params(
    space: list[tuple[str, dict[str, Any]]],
    rng: np.random.Generator,
    yaml_overrides: dict[str, Any] | DictConfig | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if yaml_overrides is None:
        ovs: dict[str, Any] = {}
    elif OmegaConf.is_config(yaml_overrides):
        raw = OmegaConf.to_container(yaml_overrides, resolve=True)
        ovs = dict(raw) if isinstance(raw, dict) else {}
    else:
        ovs = dict(yaml_overrides)
    if not isinstance(ovs, dict):
        ovs = {}
    for name, spec in space:
        if name in ovs:
            out[name] = _coerce_param_for_spec(spec, ovs[name])
        else:
            out[name] = _sample_param(spec, rng)
    return out


def _ofat_values_for_spec(
    spec: dict[str, Any],
    *,
    rng: np.random.Generator,
    skip_numeric: bool,
    float_steps: int,
    category_max_levels: int | None,
) -> list[Any]:
    typ = str(spec["type"]).lower()
    if typ in {"categorical", "choice", "enum"}:
        choices = list(spec.get("choices", []))
        if not choices:
            raise ValueError("categorical search parameter needs choices")
        idx = list(range(len(choices)))
        rng.shuffle(idx)
        if category_max_levels is not None:
            idx = idx[: max(1, min(int(category_max_levels), len(choices)))]
        idx.sort()
        return [choices[j] for j in idx]

    if skip_numeric:
        return []

    low = float(spec.get("low"))
    high = float(spec.get("high"))
    steps = max(2, int(float_steps))

    if typ in {"int", "integer"}:
        lo_i, hi_i = int(round(low)), int(round(high))
        if hi_i <= lo_i:
            return [lo_i]
        arr = np.unique(
            np.round(np.linspace(float(lo_i), float(hi_i), num=min(steps, hi_i - lo_i + 1))).astype(int)
        )
        return sorted({int(np.clip(a, lo_i, hi_i)) for a in arr})

    log_s = bool(spec.get("log", False))
    if abs(high - low) < 1e-15:
        return [low]
    if log_s:
        t = np.linspace(np.log(low), np.log(high), num=steps)
        vals = np.exp(t).astype(np.float64)
        return sorted({float(v) for v in vals})
    return sorted({float(v) for v in np.linspace(low, high, num=steps)})


def _encode_params(params: dict[str, Any], space: list[tuple[str, dict[str, Any]]]) -> np.ndarray:
    xs: list[float] = []
    for name, spec in space:
        typ = str(spec["type"]).lower()
        val = params[name]
        if typ in {"categorical", "choice", "enum"}:
            choices = list(spec.get("choices", []))
            xs.extend([1.0 if val == c else 0.0 for c in choices])
            continue
        low = float(spec.get("low"))
        high = float(spec.get("high"))
        if bool(spec.get("log", False)):
            low_t, high_t = np.log(low), np.log(high)
            val_t = np.log(float(val))
        else:
            low_t, high_t = low, high
            val_t = float(val)
        denom = high_t - low_t
        xs.append(0.0 if abs(denom) < 1e-12 else float((val_t - low_t) / denom))
    return np.asarray(xs, dtype=np.float64)


class _OfatBayesRunner:
    """One-factor-at-a-time coordinator; then exhaustion hands off to Bayesian proposal."""

    def __init__(
        self,
        space: list[tuple[str, dict[str, Any]]],
        rng: np.random.Generator,
        ofat_cfg_raw: dict[str, Any],
        fixed_params: dict[str, Any] | None = None,
    ) -> None:
        self.space = space
        self.fixed_params = dict(fixed_params or {})
        self.iterated = bool(ofat_cfg_raw.get("update_working_between_dims", True))
        skip_numeric = bool(ofat_cfg_raw.get("skip_numeric_ofat", True))
        float_steps = int(ofat_cfg_raw.get("float_linspace_steps", 3))
        cap_raw = ofat_cfg_raw.get("category_max_levels", None)
        category_cap: int | None = None if cap_raw is None else int(cap_raw)

        ovs = dict(ofat_cfg_raw.get("baseline") or {})
        self.baseline = _baseline_params(space, rng, ovs)
        self.working = dict(self.baseline)
        self.anchor = dict(self.baseline) if not self.iterated else None
        self.dims_meta: list[tuple[str, list[Any]]] = []
        for name, spec in space:
            vals = _ofat_values_for_spec(
                spec,
                rng=rng,
                skip_numeric=skip_numeric,
                float_steps=float_steps,
                category_max_levels=category_cap,
            )
            self.dims_meta.append((name, vals))
        self.dim_ix = 0
        self.val_ix = 0
        self.best_dim_score = float("-inf")
        self.best_dim_val: Any = None
        self.emit_dim: str | None = None
        self.emit_val: Any = None
        self.exhausted = False

    def plan_summary(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, vals in self.dims_meta:
            out.append({"parameter": name, "n_candidates": len(vals)})
        return out

    def _close_dimension(self, name_at_close: str) -> None:
        if self.iterated and self.best_dim_val is not None:
            self.working[name_at_close] = self.best_dim_val
        self.dim_ix += 1
        self.val_ix = 0
        self.best_dim_score = float("-inf")
        self.best_dim_val = None

    def next_params(self, tried_keys: set[tuple[tuple[str, str], ...]]) -> dict[str, Any] | None:
        """Next OFAT experiment or None → switch to Bayesian phase."""
        safety = 0
        max_iter = sum(max(len(v), 1) for _, v in self.dims_meta) * 10 + 50

        while not self.exhausted:
            if self.dim_ix >= len(self.dims_meta):
                self.exhausted = True
                return None

            safety += 1
            if safety > max_iter:
                logger.warning("OFAT aborted: too many skipped duplicates.")
                self.exhausted = True
                return None

            dim_name, vals = self.dims_meta[self.dim_ix]
            base = self.working if self.iterated else (self.anchor or self.baseline)

            if not vals:
                self._close_dimension(dim_name)
                continue

            if self.val_ix >= len(vals):
                self._close_dimension(dim_name)
                continue

            v = vals[self.val_ix]
            self.val_ix += 1
            cand = dict(base)
            cand[dim_name] = v
            cand = _merge_tuned_and_fixed(self.fixed_params, cand)

            pk = _params_key(cand)
            if pk in tried_keys:
                continue

            self.emit_dim = dim_name
            self.emit_val = v
            return cand

        return None

    def register_score(self, params: dict[str, Any], score: float) -> None:
        if self.exhausted or self.emit_dim is None:
            return
        if not np.isfinite(score):
            return
        pv = params.get(self.emit_dim)
        ev = self.emit_val
        matched = False
        try:
            if isinstance(pv, (bool, np.bool_)) or isinstance(ev, (bool, np.bool_)):
                matched = bool(pv) == bool(ev)
            elif isinstance(pv, str) or isinstance(ev, str):
                matched = str(pv) == str(ev)
            else:
                matched = math.isclose(float(pv), float(ev), rel_tol=0.0, abs_tol=1e-6)
        except Exception:
            matched = pv == ev
        if matched and float(score) >= self.best_dim_score:
            self.best_dim_score = float(score)
            self.best_dim_val = self.emit_val


def _params_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in params.items()))


def _choice_vals_distinct(a: Any, b: Any) -> bool:
    try:
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            return bool(a) != bool(b)
        if isinstance(a, str) or isinstance(b, str):
            return str(a) != str(b)
        return not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
    except Exception:
        return a != b


def _hill_neighbor_candidate(
    center: dict[str, Any],
    space: list[tuple[str, dict[str, Any]]],
    rng: np.random.Generator,
    relative_float_step: float,
) -> dict[str, Any] | None:
    """Single-coordinate stochastic neighbor inside search_space."""
    items = list(space)
    rng.shuffle(items)
    rel = float(max(1e-9, relative_float_step))

    for name, spec in items:
        typ = str(spec["type"]).lower()
        cand = dict(center)
        cur = center[name]

        if typ in {"categorical", "choice", "enum"}:
            choices = list(spec.get("choices", []))
            if len(choices) < 2:
                continue
            alts = [c for c in choices if _choice_vals_distinct(c, cur)]
            if not alts:
                continue
            cand[name] = alts[int(rng.integers(0, len(alts)))]
            return cand

        if typ in {"int", "integer"}:
            lo_i, hi_i = int(round(float(spec.get("low")))), int(round(float(spec.get("high"))))
            if hi_i <= lo_i:
                continue
            ci = int(cur)
            picks = [ci + d for d in (-2, -1, 1, 2) if lo_i <= ci + d <= hi_i]
            picks = [p for p in picks if p != ci]
            if not picks:
                nv = int(rng.integers(lo_i, hi_i + 1))
                if nv == ci:
                    continue
                cand[name] = nv
            else:
                cand[name] = int(rng.choice(picks))
            return cand

        low = float(spec.get("low"))
        high = float(spec.get("high"))
        if abs(high - low) < 1e-15:
            continue
        fv = float(cur)
        log_s = bool(spec.get("log", False))
        if log_s:
            log_hi = math.log(high)
            log_lo = math.log(low)
            lv = math.log(max(low * (1 + 1e-9), fv))
            span = max(log_hi - log_lo, 1e-12)
            delta = rng.uniform(-rel, rel) * span
            cand[name] = float(np.clip(math.exp(lv + delta), low, high))
        else:
            span = high - low
            cand[name] = float(np.clip(fv + rng.uniform(-rel, rel) * span, low, high))
        return cand

    return None


def _hill_unique_params(
    center: dict[str, Any],
    space: list[tuple[str, dict[str, Any]]],
    rng: np.random.Generator,
    relative_float_step: float,
    tried_keys: set[tuple[tuple[str, str], ...]],
    fixed_params: dict[str, Any] | None = None,
    *,
    max_attempts: int = 160,
    random_fallback_attempts: int = 500,
) -> dict[str, Any] | None:
    """One-coordinate hill step, then uniform random over untried tuned configs."""
    fixed = dict(fixed_params or {})
    space_names = [name for name, _ in space]

    for _ in range(max_attempts):
        cand_full = _hill_neighbor_candidate(center, space, rng, relative_float_step)
        if cand_full is None:
            break
        if _params_key(cand_full) not in tried_keys:
            return {name: cand_full[name] for name in space_names}

    for _ in range(random_fallback_attempts):
        tuned = _sample_params(space, rng)
        full = _merge_tuned_and_fixed(fixed, tuned)
        if _params_key(full) not in tried_keys:
            return tuned

    return None


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def _propose_params(
    observations: list[dict[str, Any]],
    space: list[tuple[str, dict[str, Any]]],
    rng: np.random.Generator,
    n_initial_random: int,
    n_candidates: int,
    *,
    ei_xi: float = 0.04,
    gp_alpha: float = 2e-4,
) -> dict[str, Any]:
    tried = {_params_key(o["params"]) for o in observations}
    if len(observations) < n_initial_random:
        for _ in range(10000):
            p = _sample_params(space, rng)
            if _params_key(p) not in tried:
                return p
        return _sample_params(space, rng)

    x_all = np.stack([_encode_params(o["params"], space) for o in observations], axis=0)
    y_all = np.asarray([float(o["score"]) for o in observations], dtype=np.float64)
    finite = np.isfinite(y_all)
    if finite.sum() < 2:
        for _ in range(10000):
            p = _sample_params(space, rng)
            if _params_key(p) not in tried:
                return p
        return _sample_params(space, rng)

    x_train = x_all[finite]
    y_train = y_all[finite]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=max(float(gp_alpha), 1e-12),
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    gp.fit(x_train, y_train)

    candidates: list[dict[str, Any]] = []
    candidate_keys = set()
    for _ in range(max(10, int(n_candidates)) * 3):
        p = _sample_params(space, rng)
        key = _params_key(p)
        if key in tried or key in candidate_keys:
            continue
        candidates.append(p)
        candidate_keys.add(key)
        if len(candidates) >= int(n_candidates):
            break
    if not candidates:
        return _sample_params(space, rng)

    x_cand = np.stack([_encode_params(p, space) for p in candidates], axis=0)
    mu, sigma = gp.predict(x_cand, return_std=True)
    best = float(np.max(y_train))
    sigma = np.maximum(sigma, 1e-9)
    xi = float(max(ei_xi, 0.0))
    imp = mu - best - xi
    z = imp / sigma
    expected_improvement = imp * _norm_cdf(z) + sigma * _norm_pdf(z)
    return candidates[int(np.argmax(expected_improvement))]


def _clusters_effective_repr(model: MSIParametricMiCSLMC) -> str:
    levels = getattr(model, "clusters", None) or []
    return "-".join(str(int(k)) for k in levels)


def _cluster_outcome_from_model(model: MSIParametricMiCSLMC) -> dict[str, Any]:
    """Per-slice MiCS ladder after fit (silhouette k₀ only when auto-tune ran)."""
    bk = getattr(model, "clusters_auto_tune_base_k", None)
    return {
        "clusters_effective": _clusters_effective_repr(model),
        "clusters_auto_tune_base_k": int(bk) if bk is not None else None,
    }


def _base_mics_kwargs(train_cfg: DictConfig, params: dict[str, Any], seed: int) -> dict[str, Any]:
    m = train_cfg.model
    merged = OmegaConf.to_container(m, resolve=True)
    if not isinstance(merged, dict):
        merged = {}
    merged.update(params)
    _resolve_cluster_search_params(merged)
    _resolve_cluster_features_param(merged)
    kwargs = {
        "number_of_points": int(merged.get("number_of_points", 1000)),
        "sampling": str(merged.get("sampling", "coreset")),
        "num_epochs": int(merged.get("num_epochs", 100)),
        "number_of_components": int(merged.get("number_of_components", 3)),
        "num_layers": int(merged.get("num_layers", 2)),
        "lab_to_rgb": bool(merged.get("lab_to_rgb", True)),
        "cluster": bool(merged.get("cluster", True)),
        "clusters": merged.get("clusters", None),
        "num_samples": int(merged.get("num_samples", 5000)),
        "beta": float(merged.get("beta", 1.0)),
        "k_epoch": int(merged.get("k_epoch", 20)),
        "temperature": float(merged.get("temperature", 100.0)),
        "lr": float(merged.get("lr", 1.0)),
        "batch_size": int(merged.get("batch_size", 1024)),
        "warmup_epochs": int(merged.get("warmup_epochs", 10)),
        "factor": float(merged.get("factor", 1.0)),
        "random_state": int(seed),
        "verbose": bool(merged.get("verbose", False)),
        "cluster_loss_weight": float(merged.get("cluster_loss_weight", 1.0)),
        "category_loss_weight": float(merged.get("category_loss_weight", 0.0)),
        "pixel_sampling": str(merged.get("pixel_sampling", "superpixel")),
        "pca_fit_step": int(merged.get("pca_fit_step", 4)),
        "cluster_on_pca": bool(merged.get("cluster_on_pca", False)),
        "cluster_pca_dims": int(merged.get("cluster_pca_dims", 50)),
    }
    if kwargs["cluster"] and isinstance(kwargs["clusters"], str):
        kwargs["clusters"] = [int(x) for x in kwargs["clusters"].split("-") if x]
    cat = kwargs.get("clusters_auto_tune")
    if cat is not None and str(cat).strip().lower() in ("", "null", "none", "~"):
        kwargs["clusters_auto_tune"] = None
    elif cat is not None and str(cat).strip().lower() == "bic":
        raise ValueError("clusters_auto_tune='bic' is no longer supported; use 'silhouette'.")
    kwargs.update(_clusters_auto_tune_kw_from_cfg(OmegaConf.create(merged)))
    kwargs["edge_stratified_grid_cells"] = int(merged.get("edge_stratified_grid_cells", 16))
    kwargs["edge_stratified_edge_floor"] = float(merged.get("edge_stratified_edge_floor", 0.05))
    kwargs["edge_stratified_uniform_mix"] = float(merged.get("edge_stratified_uniform_mix", 0.15))
    return kwargs


def _compute_viz_edge_n(viz_rgb: np.ndarray, valid_mask: np.ndarray, rank_cfg: DictConfig) -> np.ndarray:
    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    ms_enabled = bool(getattr(multi, "enabled", False)) if multi is not None else False
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    if (not ms_enabled) or not sigmas:
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    edge_cfg = dict(_parse_edge_detection_cfg(rank_cfg))
    v_edge, _v_edge_linf = _edge_maps_multiscale(
        "rgb",
        viz_rgb,
        valid_mask,
        sigmas=sigmas,
        aggregation=aggregation,
        hd_cfg=None,
        edge_method_cfg=edge_cfg,
    )
    sn = getattr(rank_cfg, "spatial_normalization", None)
    if sn is not None and bool(getattr(sn, "enabled", True)):
        return _normalize_map_percentile(
            v_edge,
            valid_mask,
            low_pct=float(getattr(sn, "low_percentile", 1.0)),
            high_pct=float(getattr(sn, "high_percentile", 99.0)),
        )
    return _normalize_on_mask(v_edge, valid_mask)


def _save_score_plot(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    ylabel: str = "Combined objective",
    title: str | None = None,
    show_best_so_far: bool = True,
) -> None:
    it = [int(r["iteration"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    sc_arr = np.asarray(scores, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    ax.plot(it, scores, marker="o", linewidth=1.5, label="trial objective")
    best_arr = None
    if show_best_so_far:
        running = float("-inf")
        best_list: list[float] = []
        for v in scores:
            fv = float(v)
            if np.isfinite(fv):
                running = max(running, fv)
            best_list.append(running if running > float("-inf") else float("nan"))
        best_arr = np.asarray(best_list, dtype=np.float64)
        ax.plot(it, best_list, marker=".", linewidth=2.0, label="best objective so far")
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    fin_sc = sc_arr[np.isfinite(sc_arr)]
    if fin_sc.size:
        lo = float(np.min(fin_sc))
        if show_best_so_far and best_arr is not None:
            fin_best = best_arr[np.isfinite(best_arr)]
            hi = float(np.max(fin_best)) if fin_best.size else float(np.max(fin_sc))
        else:
            hi = float(np.max(fin_sc))
        pad = max((hi - lo) * 0.12, 0.02)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _save_mean_metrics_plot(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Per-trial mean metrics in original units (CSV *_mean_over_images columns)."""
    it = np.asarray([int(r["iteration"]) for r in rows], dtype=np.float64)
    series = [
        ("continuous_dice_mean_over_images", "Continuous Dice (mean over images)", True),
        ("luminance_rms_contrast_mean_over_images", "Luminance RMS contrast", False),
        ("lab_chroma_entropy_mean_over_images", "LAB chroma entropy", False),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.2), dpi=160, sharex=True)
    for ax, (col, ylab, dice_like) in zip(axes, series):
        y = np.asarray([float(r.get(col, np.nan)) for r in rows], dtype=np.float64)
        ax.plot(it, y, marker="o", linewidth=1.2, color="#1f77b4")
        ax.set_ylabel(ylab, fontsize=9)
        fin = y[np.isfinite(y)]
        if fin.size:
            lo, hi = float(np.min(fin)), float(np.max(fin))
            pad = max((hi - lo) * 0.12, 1e-6)
            if dice_like:
                ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
            else:
                ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Optimization iteration")
    fig.suptitle("Trial-wise metric means (actual units, not cumulative best)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _resolve_npy_paths(cfg: DictConfig) -> list[Path]:
    paths_sel = OmegaConf.select(cfg, "input.npy_paths")
    if paths_sel is not None:
        raw = OmegaConf.to_container(paths_sel, resolve=True)
        if isinstance(raw, (list, tuple)) and raw:
            return sorted(Path(to_absolute_path(str(p))).resolve() for p in raw)
    npy_dir = OmegaConf.select(cfg, "input.npy_dir")
    if npy_dir is not None and str(npy_dir).strip():
        glob_pat = str(OmegaConf.select(cfg, "input.npy_glob", default="**/5_bins/0.npy"))
        root = Path(to_absolute_path(str(npy_dir))).resolve()
        if not root.is_dir():
            raise ValueError(f"input.npy_dir is not a directory: {root}")
        paths = sorted(root.glob(glob_pat))
        if not paths:
            raise ValueError(f"No .npy files matched {glob_pat!r} under {root}")
        return paths
    p = OmegaConf.select(cfg, "input.npy_path")
    if p is None or str(p).strip() == "":
        raise ValueError("Provide input.npy_path, non-empty input.npy_paths, or input.npy_dir.")
    return [Path(to_absolute_path(str(p))).resolve()]


def _normalized_rank_last_in_series(col: np.ndarray) -> float:
    """Higher values rank higher; latest trial normalized to [0, 1]. NaN last -> 0."""
    col = np.asarray(col, dtype=np.float64)
    if col.size == 0:
        return 0.0
    if not np.isfinite(col[-1]):
        return 0.0
    fin = col[np.isfinite(col)]
    if fin.size <= 1:
        return 0.5
    ranks = rankdata(fin, method="average")
    last_r = float(ranks[-1])
    return (last_r - 1.0) / float(max(fin.size - 1, 1))


def _parse_score_mode(cfg: DictConfig) -> str:
    raw = str(OmegaConf.select(cfg, "objective.score_mode", default="rank_composite")).strip().lower()
    if raw in {"rank_composite", "rank", "cumulative_rank"}:
        return "rank_composite"
    if raw in {"weighted_metrics", "weighted", "linear", "combined_raw"}:
        return "weighted_metrics"
    raise ValueError(
        f"objective.score_mode={raw!r} is unknown; use 'rank_composite' or 'weighted_metrics'."
    )


def _parse_metric_bounds(cfg: DictConfig) -> dict[str, tuple[float, float]]:
    """Per-metric [low, high] for min–max normalization in weighted_metrics mode (higher is better)."""
    defaults: dict[str, tuple[float, float]] = {
        "continuous_dice": (0.0, 1.0),
        "luminance_rms_contrast": (0.0, 0.5),
        "lab_chroma_entropy": (0.0, 8.0),
    }
    raw = OmegaConf.select(cfg, "objective.metric_bounds", default=None)
    if raw is None:
        return defaults
    oc = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(oc, dict):
        return defaults
    out = dict(defaults)
    for m in _METRIC_KEYS:
        if m not in oc:
            continue
        v = oc[m]
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[m] = (float(v[0]), float(v[1]))
        elif isinstance(v, dict) and "min" in v and "max" in v:
            out[m] = (float(v["min"]), float(v["max"]))
    return out


def _trial_means_from_vectors(row_vec: dict[str, np.ndarray]) -> dict[str, float]:
    return {m: float(np.nanmean(row_vec[m])) for m in _METRIC_KEYS}


def _weighted_linear_objective(
    means: dict[str, float],
    weights: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> tuple[float, dict[str, float]]:
    detail: dict[str, float] = {}
    num = 0.0
    den = 0.0
    for m in _METRIC_KEYS:
        w = float(weights.get(m, 0.0))
        if w <= 1e-15:
            detail[f"objective_norm__{m}"] = float("nan")
            continue
        raw = means.get(m)
        lo, hi = bounds.get(m, (0.0, 1.0))
        span = max(float(hi) - float(lo), 1e-12)
        if not np.isfinite(raw):
            nm = 0.0
        else:
            nm = float(np.clip((float(raw) - float(lo)) / span, 0.0, 1.0))
        detail[f"objective_norm__{m}"] = nm
        num += w * nm
        den += w
    if den <= 1e-15:
        return float("nan"), detail
    return float(num / den), detail


def _parse_objective_weights(cfg: DictConfig) -> dict[str, float]:
    raw_w = OmegaConf.select(cfg, "objective.weights")
    legacy_metric = str(OmegaConf.select(cfg, "objective.metric", default="continuous_dice")).strip()
    out = {k: 0.0 for k in _METRIC_KEYS}
    if legacy_metric in _METRIC_KEYS:
        out[legacy_metric] = 1.0
    elif legacy_metric:
        out["continuous_dice"] = 1.0
    if raw_w is not None:
        oc = OmegaConf.to_container(raw_w, resolve=True)
        if isinstance(oc, dict) and oc:
            for k in _METRIC_KEYS:
                out[k] = float(oc.get(k, out[k]))
            if sum(abs(out[k]) for k in _METRIC_KEYS) <= 0:
                out = {kk: (1.0 if kk == "continuous_dice" else 0.0) for kk in _METRIC_KEYS}
        elif not isinstance(oc, dict):
            pass
    pos = sum(max(0.0, out[k]) for k in _METRIC_KEYS)
    if pos <= 0:
        return {kk: (1.0 if kk == "continuous_dice" else 0.0) for kk in _METRIC_KEYS}
    return {k: max(0.0, out[k]) / pos for k in _METRIC_KEYS}


def _append_metric_row(metric_histories: dict[str, list[np.ndarray]], row_vec: dict[str, np.ndarray]) -> None:
    for k in _METRIC_KEYS:
        metric_histories[k].append(np.asarray(row_vec[k], dtype=np.float64).reshape(-1).copy())


def _combined_rank_objective(metric_histories: dict[str, list[np.ndarray]], weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    detail: dict[str, float] = {}
    contrib: list[float] = []
    active_w: list[float] = []
    for m in _METRIC_KEYS:
        w = float(weights.get(m, 0.0))
        if w <= 1e-15:
            continue
        hist = metric_histories.get(m)
        if not hist:
            detail[f"rank_mean__{m}"] = float("nan")
            continue
        stack = np.stack(hist, axis=0)
        per_k = [_normalized_rank_last_in_series(stack[:, k]) for k in range(stack.shape[1])]
        mean_rank = float(np.mean(per_k)) if per_k else 0.0
        detail[f"rank_mean__{m}"] = mean_rank
        contrib.append(mean_rank)
        active_w.append(w)
    ws = float(sum(active_w))
    if ws <= 0:
        return float("nan"), detail
    score = float(sum(c * ww for c, ww in zip(contrib, active_w)) / ws)
    return score, detail


def _save_metric_histories_npy(metric_histories: dict[str, list[np.ndarray]], out_dir: Path) -> None:
    for m in _METRIC_KEYS:
        hist = metric_histories.get(m) or []
        if not hist:
            continue
        arr = np.stack(hist, axis=0).astype(np.float64)
        np.save(out_dir / f"metric_history__{m}.npy", arr)


def _build_datasets_from_cfg(
    cfg: DictConfig,
    run_dir: Path,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    npy_paths = _resolve_npy_paths(cfg)
    rank_cfg = OmegaConf.load(to_absolute_path(str(cfg.rank_config)))
    normalization = _resolve_normalization_mode(OmegaConf.select(cfg, "data.normalization", default="tic"))
    transpose_msi = bool(OmegaConf.select(cfg, "data.transpose_msi", default=False))
    ms_cfg = getattr(rank_cfg, "edge_maps", None)
    multi = getattr(ms_cfg, "multi_scale", None) if ms_cfg is not None else None
    sigmas = [float(v) for v in list(getattr(multi, "sigmas", [0.0]))] if multi is not None else [0.0]
    ms_enabled = bool(getattr(multi, "enabled", False)) if multi is not None else False
    if not ms_enabled:
        sigmas = [0.0]
    aggregation = str(getattr(multi, "aggregation", "mean")) if multi is not None else "mean"
    spatial_enabled = bool(getattr(rank_cfg.spatial_normalization, "enabled", True))
    hd_method = str(OmegaConf.select(rank_cfg, "edge_detection.hd_method", default="soft_landmark_contrast"))
    if hd_method != "soft_landmark_contrast":
        logger.warning("rank_config edge_detection.hd_method is %s; using soft_landmark_contrast anyway", hd_method)
    rank_cfg.edge_detection.hd_method = "soft_landmark_contrast"
    edge_cmap = str(OmegaConf.select(cfg, "output.edge_colormap", default="PuBu"))
    datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for npy_path in npy_paths:
        msi_vol = _load_msi(npy_path, transpose_msi=transpose_msi, normalization=normalization)
        valid_mask = msi_vol.sum(axis=-1) > 0
        if not np.any(valid_mask):
            raise ValueError(f"MSI valid mask is empty for {npy_path}")
        hd_edge_n = _compute_hd_edge_n_for_method(
            rank_cfg,
            msi_vol,
            valid_mask,
            getattr(rank_cfg, "hd_edges", None),
            "soft_landmark_contrast",
            sigmas,
            aggregation,
            spatial_enabled,
        )
        stem = Path(npy_path).stem
        hd_edge_u8 = _to_uint8_gray01(hd_edge_n)
        hd_edge_rgb = _gray_u8_to_rgb_colormap(hd_edge_u8, edge_cmap)
        Image.fromarray(hd_edge_rgb, mode="RGB").save(
            run_dir / f"soft_landmark_contrast_reference__{stem}__{_safe_token(edge_cmap)}.png"
        )
        datasets.append((msi_vol, valid_mask, hd_edge_n, stem))
    return datasets


def run_mics_hyperparameter_tune(
    cfg: DictConfig,
    run_dir: Path,
    datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    *,
    study_arm_id: str | None = None,
    arm_fixed_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """OFAT + hill (or Bayesian) loop; objective = mean over ``datasets`` slices."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.tuning.random_state))
    random.seed(int(cfg.tuning.random_state))
    np.random.seed(int(cfg.tuning.random_state))

    rank_cfg = OmegaConf.load(to_absolute_path(str(cfg.rank_config)))
    train_cfg = OmegaConf.load(to_absolute_path(str(cfg.train_config)))
    weights = _parse_objective_weights(cfg)
    score_mode = _parse_score_mode(cfg)
    metric_bounds = _parse_metric_bounds(cfg)
    logger.info(
        "Tuning start | arm=%s | images=%d | objective=%s | weights=%s",
        study_arm_id or "(default)",
        len(datasets),
        score_mode,
        weights,
    )
    if score_mode == "weighted_metrics":
        logger.info("Weighted objective uses metric_bounds=%s", metric_bounds)

    space = _search_space_items(cfg)
    fixed_params = _load_fixed_params(cfg, space, extra_fixed=arm_fixed_overrides)
    if fixed_params:
        logger.info("Fixed hyperparameters (excluded from OFAT/hill sweeps): %s", fixed_params)
    rows: list[dict[str, Any]] = []
    csv_path = run_dir / "bayes_mics_trials.csv"
    images_dir = run_dir / "trial_images"
    edges_dir = run_dir / "trial_viz_edges"
    hist_dir = run_dir / "metric_histories"
    images_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)
    plot_path = run_dir / "objective_rank_composite_by_iteration.png"
    metrics_plot_path = run_dir / "metrics_mean_by_iteration.png"
    score_plot_show_best = bool(OmegaConf.select(cfg, "output.score_plot.show_best_so_far", default=True))
    score_plot_metrics_panel = bool(OmegaConf.select(cfg, "output.score_plot.plot_mean_metrics_panel", default=True))

    n_iter = int(cfg.tuning.n_iterations)
    n_initial = int(cfg.tuning.n_initial_random)
    n_candidates = int(cfg.tuning.n_candidates)
    bay_any = OmegaConf.select(cfg, "tuning.bayesian", default={})
    bay_oc = _omega_subtree_to_plain_dict(bay_any)
    ei_xi = float(bay_oc.get("ei_xi", 0.04))
    gp_alpha = float(bay_oc.get("gp_alpha", 2e-4))
    random_explore_probability = float(np.clip(float(bay_oc.get("random_explore_probability", 0.12)), 0.0, 1.0))
    logger.info(
        "Bayesian acquisition: ei_xi=%s gp_alpha=%s random_explore_probability=%s",
        ei_xi,
        gp_alpha,
        random_explore_probability,
    )
    square = bool(OmegaConf.select(cfg, "objective.square_before_continuous_dice", default=False))
    save_equalized = bool(OmegaConf.select(cfg, "output.equalize_mics_rgb", default=False))
    edge_cmap = str(OmegaConf.select(cfg, "output.edge_colormap", default="PuBu"))

    metric_histories: dict[str, list[np.ndarray]] = {k: [] for k in _METRIC_KEYS}
    k_images = len(datasets)

    metric_mean_cols = [f"{k}_mean_over_images" for k in _METRIC_KEYS]
    rank_detail_cols = [f"rank_mean__{m}" for m in _METRIC_KEYS]
    norm_detail_cols = [f"objective_norm__{m}" for m in _METRIC_KEYS]
    fieldnames = (
        ["iteration", "phase", "study_arm_id", "objective_score_mode", "score", "best_score", "elapsed_sec", "image_paths", "viz_edge_paths", "tag_leader"]
        + rank_detail_cols
        + norm_detail_cols
        + metric_mean_cols
        + [
            "metrics_by_image_json",
            "cluster_outcome_by_image_json",
            "clusters_auto_tune_base_k_mean_over_images",
            "clusters_auto_tune_base_k_std_over_images",
            "clusters_effective_sample",
        ]
        + sorted(set(fixed_params) | {name for name, _ in space})
    )
    observations: list[dict[str, Any]] = []

    strategy = str(OmegaConf.select(cfg, "tuning.strategy", default="bayesian")).strip().lower()
    ofat_any = OmegaConf.select(cfg, "tuning.ofat_then_bayesian", default={})
    ofat_oc = _omega_subtree_to_plain_dict(ofat_any)
    runner: _OfatBayesRunner | None = None
    ofat_watch: _OfatBayesRunner | None = None
    if strategy in ("ofat_then_bayesian", "ofat_then_hill"):
        runner = _OfatBayesRunner(space, rng, ofat_oc, fixed_params=fixed_params)
        ofat_watch = runner
        (run_dir / "ofat_plan.json").write_text(
            json.dumps(
                {
                    "strategy": strategy,
                    "dims": runner.plan_summary(),
                    "baseline": runner.baseline,
                    "iterated": runner.iterated,
                    "fixed_params": fixed_params,
                },
                default=str,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "OFAT phase: sweeping %s parameter axes (iterated=%s)",
            len(runner.dims_meta),
            runner.iterated,
        )
        if strategy == "ofat_then_hill":
            logger.info("Strategy ofat_then_hill: hill climbing after OFAT; Bayesian phase disabled.")
    elif strategy != "bayesian":
        raise ValueError(
            f"Unknown tuning.strategy {strategy!r}; use 'bayesian', 'ofat_then_bayesian', or 'ofat_then_hill'."
        )

    hill_any = OmegaConf.select(cfg, "tuning.hill_climbing", default={})
    hill_oc = _omega_subtree_to_plain_dict(hill_any)
    hill_enabled_cfg = bool(hill_oc.get("enabled", False))
    hill_after_ofat_only = bool(hill_oc.get("after_ofat_only", True))
    hill_max_trials = int(hill_oc.get("max_trials", 0))
    hill_rel_step = float(hill_oc.get("relative_float_step", 0.2))
    hill_random_fallback = bool(hill_oc.get("random_fallback", True))
    hill_random_fallback_attempts = int(hill_oc.get("random_fallback_attempts", 500))
    hill_trials_left = hill_max_trials if hill_enabled_cfg else 0
    hill_started = False
    hill_center: dict[str, Any] | None = None
    hill_best_score = float("-inf")

    if hill_trials_left > 0:
        if hill_after_ofat_only and strategy not in ("ofat_then_bayesian", "ofat_then_hill"):
            logger.warning(
                "hill_climbing disabled: after_ofat_only=true requires tuning.strategy=ofat_then_bayesian "
                "or ofat_then_hill (got %s). Set strategy accordingly or hill_climbing.after_ofat_only=false "
                "(latter enables hill with strategy=bayesian).",
                strategy,
            )
            hill_trials_left = 0
        elif hill_max_trials <= 0:
            hill_trials_left = 0
        elif strategy == "ofat_then_hill" and not hill_enabled_cfg:
            logger.warning("ofat_then_hill requires hill_climbing.enabled=true; no post-OFAT phase will run.")
        else:
            hc_mode = (
                "after OFAT completes (Bayesian disabled)"
                if strategy == "ofat_then_hill" and hill_after_ofat_only
                else "after OFAT completes"
                if strategy == "ofat_then_bayesian" and hill_after_ofat_only
                else "interleaved with Bayesian draws"
                if strategy == "bayesian"
                else "after OFAT / alongside Bayesian"
            )
            logger.info(
                "Hill climbing queued: max_trials=%s relative_float_step=%s (%s)",
                hill_max_trials,
                hill_rel_step,
                hc_mode,
            )

    tried_keys: set[tuple[tuple[str, str], ...]] = set()

    for iteration in range(1, n_iter + 1):
        trial_phase = "bayes"
        params = None

        if runner is not None and not runner.exhausted:
            trial_phase = "ofat"
            params = runner.next_params(tried_keys)
            if params is None:
                if strategy == "ofat_then_hill":
                    logger.info(
                        "OFAT phase complete (%d trials queued from coordinate sweeps). Switching to hill climbing.",
                        iteration - 1,
                    )
                else:
                    logger.info(
                        "OFAT phase complete (%d trials queued from coordinate sweeps). Switching to Bayesian.",
                        iteration - 1,
                    )
                runner = None

        hill_eligible = hill_trials_left > 0 and (
            strategy in ("ofat_then_bayesian", "ofat_then_hill")
            or (strategy == "bayesian" and not hill_after_ofat_only)
        )
        if params is None and hill_eligible and observations:
            if not hill_started:
                finite_obs = [o for o in observations if np.isfinite(float(o["score"]))]
                pool = finite_obs if finite_obs else observations
                best_obs = max(pool, key=lambda o: float(o["score"]))
                hill_center = dict(best_obs["params"])
                hill_best_score = float(best_obs["score"])
                hill_started = True
                logger.info(
                    "Hill climbing phase | starting from best objective=%.6f",
                    hill_best_score,
                )
            cand = _hill_unique_params(
                hill_center if hill_center is not None else {},
                space,
                rng,
                hill_rel_step,
                tried_keys,
                fixed_params,
                random_fallback_attempts=hill_random_fallback_attempts if hill_random_fallback else 0,
            )
            if cand is None:
                logger.warning(
                    "Hill climbing: no untried configs left in search space (%d keys tried).",
                    len(tried_keys),
                )
                hill_trials_left = 0
            else:
                params = cand
                trial_phase = "hill"
                hill_trials_left -= 1

        if params is None:
            if strategy == "ofat_then_hill":
                if hill_trials_left > 0:
                    logger.warning(
                        "Stopping with %d hill_climbing trial(s) remaining (no neighbor or iteration cap).",
                        hill_trials_left,
                    )
                else:
                    logger.info("OFAT + hill climbing complete. Stopping (Bayesian disabled).")
                break

            tried_now = {_params_key(o["params"]) for o in observations}
            if (
                random_explore_probability > 0.0
                and len(observations) >= n_initial
                and rng.random() < random_explore_probability
            ):
                for _ in range(12000):
                    p = _sample_params(space, rng)
                    if _params_key(p) not in tried_now:
                        params = p
                        break
            if params is None:
                params = _propose_params(
                    observations,
                    space,
                    rng,
                    n_initial_random=n_initial,
                    n_candidates=n_candidates,
                    ei_xi=ei_xi,
                    gp_alpha=gp_alpha,
                )
            trial_phase = "bayes"

        if params is None:
            continue
        params = _merge_tuned_and_fixed(fixed_params, params)

        trial_seed = int(cfg.tuning.random_state) + iteration - 1
        kwargs = _base_mics_kwargs(train_cfg, params, seed=trial_seed)
        ps = str(kwargs.get("pixel_sampling", "superpixel")).lower()
        logger.info(
            "Trial %d/%d | phase=%s | params=%s | pixel_sampling=%s",
            iteration,
            n_iter,
            trial_phase,
            params,
            ps,
        )
        t0 = time.perf_counter()

        image_paths_row: list[str] = []
        edge_paths_row: list[str] = []
        dice_vec = np.full(k_images, np.nan, dtype=np.float64)
        lum_vec = np.full(k_images, np.nan, dtype=np.float64)
        ent_vec = np.full(k_images, np.nan, dtype=np.float64)
        base_k_vec = np.full(k_images, np.nan, dtype=np.float64)
        clusters_eff_by_image: list[str] = []
        image_stems: list[str] = []

        for img_idx, (msi_vol, valid_mask, hd_edge_n, stem) in enumerate(datasets):
            image_stems.append(stem)
            eff_ladder = ""
            model: MSIParametricMiCSLMC | None = None
            try:
                trial_kwargs = dict(kwargs)
                if ps in ("edge_stratified", "edge_stratified_hd"):
                    trial_kwargs["hd_edge_map_for_sampling"] = hd_edge_n
                model = MSIParametricMiCSLMC(**trial_kwargs)
                model.fit(msi_vol)
                cluster_out = _cluster_outcome_from_model(model)
                eff_ladder = str(cluster_out["clusters_effective"])
                bk = cluster_out["clusters_auto_tune_base_k"]
                if bk is not None:
                    base_k_vec[img_idx] = float(bk)
                viz = model.predict(msi_vol)
                viz_rgb = _to_uint8_rgb(viz)
                if save_equalized:
                    viz_rgb = _hist_equalize_rgb(viz_rgb)
                viz_edge_n = _compute_viz_edge_n(viz_rgb, valid_mask, rank_cfg)
                hd_d, viz_d = _edges_for_continuous_metrics(hd_edge_n, viz_edge_n, square=square)
                dice_i = float(_compute_continuous_dice(hd_d, viz_d, valid_mask))
                cq = _compute_viz_contrast_color_metrics(viz_rgb, valid_mask)
                lum_i = float(cq["luminance_rms_contrast"])
                ent_i = float(cq["lab_chroma_entropy"])
                dice_vec[img_idx] = dice_i
                lum_vec[img_idx] = lum_i
                ent_vec[img_idx] = ent_i

                param_hash = hashlib.sha1(repr(sorted(params.items())).encode("utf-8")).hexdigest()[:10]
                dice_tag = dice_i if np.isfinite(dice_i) else 0.0
                tag = f"i{iteration:03d}__img{img_idx}__dice-{float(dice_tag):.5f}__h-{param_hash}"

                image_path = images_dir / f"mics__{stem}__{tag}.png"
                edge_path = edges_dir / f"viz_edge__{stem}__{tag}__{_safe_token(edge_cmap)}.png"
                Image.fromarray(viz_rgb, mode="RGB").save(image_path)
                viz_edge_rgb = _gray_u8_to_rgb_colormap(_to_uint8_gray01(viz_edge_n), edge_cmap)
                Image.fromarray(viz_edge_rgb, mode="RGB").save(edge_path)
                image_paths_row.append(str(image_path))
                edge_paths_row.append(str(edge_path))
            finally:
                clusters_eff_by_image.append(eff_ladder)
                if model is not None:
                    try:
                        model.release_resources()
                    except Exception:
                        pass
                del model
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        row_vec = {
            "continuous_dice": dice_vec,
            "luminance_rms_contrast": lum_vec,
            "lab_chroma_entropy": ent_vec,
        }
        _append_metric_row(metric_histories, row_vec)
        trial_means = _trial_means_from_vectors(row_vec)
        if score_mode == "rank_composite":
            score, rank_detail = _combined_rank_objective(metric_histories, weights)
            _, norm_detail = _weighted_linear_objective(trial_means, weights, metric_bounds)
        else:
            score, norm_detail = _weighted_linear_objective(trial_means, weights, metric_bounds)
            rank_detail = {f"rank_mean__{m}": float("nan") for m in _METRIC_KEYS}
        _save_metric_histories_npy(metric_histories, hist_dir)

        elapsed = time.perf_counter() - t0
        param_hash = hashlib.sha1(repr(sorted(params.items())).encode("utf-8")).hexdigest()[:10]
        dice_mean = float(np.nanmean(dice_vec))
        tag_leader = f"i{iteration:03d}__score-{score:.5f}__dice_mean-{dice_mean:.5f}__h-{param_hash}"

        observations.append({"params": params, "score": score})
        tried_keys.add(_params_key(params))
        if trial_phase == "ofat" and runner is not None:
            runner.register_score(params, score)
        if trial_phase == "hill" and hill_started and np.isfinite(score) and float(score) > hill_best_score:
            hill_best_score = float(score)
            hill_center = dict(params)

        finite_scores_iter = [
            float(o["score"]) for o in observations if np.isfinite(float(o["score"]))
        ]
        best_score = max(finite_scores_iter) if finite_scores_iter else float("nan")

        metrics_json = json.dumps(
            {
                "continuous_dice": dice_vec.tolist(),
                "luminance_rms_contrast": lum_vec.tolist(),
                "lab_chroma_entropy": ent_vec.tolist(),
            }
        )
        cluster_outcome_json = json.dumps(
            {
                "image_stems": image_stems,
                "clusters_effective": clusters_eff_by_image,
                "clusters_auto_tune_base_k": [
                    None if not np.isfinite(base_k_vec[i]) else int(base_k_vec[i])
                    for i in range(k_images)
                ],
            }
        )
        bk_mean = float(np.nanmean(base_k_vec))
        bk_std = float(np.nanstd(base_k_vec))
        row = {
            "iteration": iteration,
            "phase": trial_phase,
            "study_arm_id": study_arm_id or "",
            "objective_score_mode": score_mode,
            "score": score,
            "best_score": best_score,
            "elapsed_sec": elapsed,
            "image_paths": "|".join(image_paths_row),
            "viz_edge_paths": "|".join(edge_paths_row),
            **{f"rank_mean__{m}": float(rank_detail.get(f"rank_mean__{m}", float("nan"))) for m in _METRIC_KEYS},
            **{
                f"objective_norm__{m}": float(norm_detail.get(f"objective_norm__{m}", float("nan")))
                for m in _METRIC_KEYS
            },
            "continuous_dice_mean_over_images": float(np.nanmean(dice_vec)),
            "luminance_rms_contrast_mean_over_images": float(np.nanmean(lum_vec)),
            "lab_chroma_entropy_mean_over_images": float(np.nanmean(ent_vec)),
            "metrics_by_image_json": metrics_json,
            "cluster_outcome_by_image_json": cluster_outcome_json,
            "clusters_auto_tune_base_k_mean_over_images": bk_mean,
            "clusters_auto_tune_base_k_std_over_images": bk_std,
            "clusters_effective_sample": clusters_eff_by_image[0] if clusters_eff_by_image else "",
            "tag_leader": tag_leader,
            **params,
        }
        rows.append(row)
        _append_csv(csv_path, row, fieldnames)
        w_parts = [f"{m}:{weights[m]:.2f}" for m in _METRIC_KEYS if weights.get(m, 0) > 1e-8]
        if score_mode == "rank_composite":
            plot_title = "Rank composite — " + (", ".join(w_parts) if w_parts else "see objective.weights")
            plot_ylabel = "Composite (weighted mean normalized rank)"
        else:
            plot_title = "Weighted metrics (min–max) — " + (", ".join(w_parts) if w_parts else "see objective.metric_bounds")
            plot_ylabel = "Composite (weighted min–max metrics)"
        _save_score_plot(
            rows,
            plot_path,
            ylabel=plot_ylabel,
            title=plot_title,
            show_best_so_far=score_plot_show_best,
        )
        if score_plot_metrics_panel:
            _save_mean_metrics_plot(rows, metrics_plot_path)
        logger.info(
            "Trial %d done | phase=%s | objective=%.6f | dice_mean=%.6f | best=%.6f | %.1fs",
            iteration,
            trial_phase,
            score,
            dice_mean,
            best_score,
            elapsed,
        )

    if ofat_watch is not None and not ofat_watch.exhausted:
        logger.warning(
            "tuning.n_iterations=%d exhausted before OFAT finished (stopped around dimension index %d of %d). "
            "Increase n_iterations to allow the full OFAT sweep plus post-OFAT trials.",
            n_iter,
            int(ofat_watch.dim_ix),
            len(ofat_watch.dims_meta),
        )
    if hill_enabled_cfg and hill_max_trials > 0 and hill_trials_left > 0:
        logger.warning(
            "tuning.n_iterations=%d exhausted with %d hill_climbing trial(s) unused. Increase n_iterations.",
            n_iter,
            hill_trials_left,
        )
    finite_rows = [r for r in rows if np.isfinite(float(r["score"]))]
    best_row = max(finite_rows, key=lambda r: float(r["score"])) if finite_rows else rows[-1]
    (run_dir / "best_trial.txt").write_text(
        "\n".join(
            [
                f"score={best_row['score']}",
                f"image_paths={best_row['image_paths']}",
                f"viz_edge_paths={best_row['viz_edge_paths']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Best trial: %s", best_row)
    return {"best_row": best_row, "rows": rows, "csv_path": str(csv_path)}


@hydra.main(version_base=None, config_path="configs", config_name="bayes_tune_mics_lmc")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(getattr(cfg.logging, "level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    datasets = _build_datasets_from_cfg(cfg, run_dir)
    run_mics_hyperparameter_tune(cfg, run_dir, datasets)


if __name__ == "__main__":
    main()
