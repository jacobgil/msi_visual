"""
Mixture-of-experts wrapper over multiple PCC (pixel MLP) branches for MiCS+LMC training.

Fusion modes:
    - mean: calibrated per-expert outputs averaged (recommended default baseline).
    - softmax_gate: per-pixel softmax weights over experts from spectral input x.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC, create_pcc_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _resolve_cluster_levels(raw: Any, default_levels: list[int]) -> list[int]:
    """Expert ``clusters`` field: omit/None → use ``default_levels``; str 'a-b-c'; seq of ints; empty → no clustering."""
    dl = [int(k) for k in default_levels]
    if raw is None:
        return dl
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        return [int(x) for x in s.split("-") if str(x).strip()]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    if isinstance(raw, int) and not isinstance(raw, bool):
        return [int(raw)]
    raise TypeError(f"clusters must be str, sequence, or null (got {type(raw)})")


def _kmeans_label_cache_key(data_for_cluster: Any, k_eff: int) -> tuple[Any, ...]:
    """Stable key when ``data_for_cluster`` is reused across experts / levels (same buffer → same clusters)."""
    if not isinstance(data_for_cluster, np.ndarray):
        return (id(data_for_cluster), int(k_eff))
    try:
        ptr = int(data_for_cluster.__array_interface__["data"][0])
    except Exception:
        ptr = id(data_for_cluster)
    return (ptr, tuple(int(x) for x in data_for_cluster.shape), str(data_for_cluster.dtype), int(k_eff))


class MixturePCCExperts(nn.Module):
    """Runs K PCC experts on identical inputs; optionally LayerNorm embeddings; fused output."""

    def __init__(
        self,
        input_dim: int,
        n_components: int,
        experts: nn.ModuleList,
        *,
        fusion: str = "mean",
        layer_norm_before_fuse: bool = False,
        gate_temperature: float = 1.0,
        gate_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(experts, nn.ModuleList) or len(experts) < 1:
            raise ValueError("experts must be a non-empty nn.ModuleList")

        fusion = str(fusion).strip().lower().replace("-", "_")
        valid = {"mean", "softmax_gate"}
        if fusion not in valid:
            raise ValueError(f"fusions must be one of {sorted(valid)} (got {fusion!r})")

        self.input_dim = int(input_dim)
        self.n_components = int(n_components)
        self.experts = experts
        self.k = len(experts)
        self.fusion = fusion
        self.gate_temperature = max(float(gate_temperature), 1e-8)

        if layer_norm_before_fuse:
            self.per_expert_ln = nn.ModuleList(
                nn.LayerNorm(self.n_components) for _ in range(self.k)
            )
        else:
            self.per_expert_ln = None

        self.gate_net: nn.Module | None = None
        if fusion == "softmax_gate":
            hid = int(gate_hidden_dim) if gate_hidden_dim > 0 else min(512, max(128, self.input_dim))
            self.gate_net = nn.Sequential(
                nn.Linear(self.input_dim, hid),
                nn.GELU(),
                nn.Linear(hid, self.k),
            )

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys: list[torch.Tensor] = []
        for i, expert in enumerate(self.experts):
            y = expert(x)
            if self.per_expert_ln is not None:
                y = self.per_expert_ln[i](y)
            ys.append(y)

        if self.fusion == "mean":
            return torch.mean(torch.stack(ys, dim=0), dim=0)

        # softmax_gate — weights (batch, K) applied per channel
        if self.gate_net is None:
            raise RuntimeError("gate fusion requires gate_net")
        logits = self.gate_net(x) / self.gate_temperature
        w = torch.softmax(logits, dim=-1)  # (B,K)
        stack = torch.stack(ys, dim=-1)  # (B, C, K)
        wc = w.unsqueeze(1).expand_as(stack)  # (B, C, K)
        return (stack * wc).sum(dim=-1)


class MSIParametricMiCSMoE(MSIParametricMiCSLMC):
    """
    MiCS+LMC where ``self.model`` is a ``MixturePCCExperts`` over K PCC backbones.

    Expert overrides:

    - ``num_layers``, ``factor`` – PCC trunk per expert.

    Optional **per-expert clustering** for MiCS pseudo-labels:

    - ``clusters`` – ``None``/omit inherits ``model.clusters``. String ``\"8-16-32\"``, list ``[8, 16]``,
      or ``[]`` / empty string disables clustering on that expert.
    - ``cluster_on_pca`` – ``None`` inherits ``model.cluster_on_pca``.
    - ``cluster_pca_dims`` – ``None`` inherits ``model.cluster_pca_dims``.

    Heads are flattened in expert order (all levels expert 0, then expert 1, …). Routing uses the
    **same calibrated expert embedding as fusion** (LayerNorm branch when enabled).
    Speed / reuse (initialization only):
    - **Shared PCA** (optional): ``moe_share_cluster_pca=True`` fits one PCA on ``sampled_data`` with
      dimension ``moe_shared_cluster_pca_dims`` or, if omitted, ``max`` of per-expert ``cluster_pca_dims``
      among experts using ``cluster_on_pca``. Faster than fitting PCA per expert; subspace differs from
      per-expert fits.
    - **KMeans label cache**: identical ``(feature matrix pointer, shape, dtype, K)`` skips refitting.
    """

    def __init__(
        self,
        moe_expert_specs: list[dict[str, Any]],
        moe_fusion: str = "mean",
        moe_layer_norm_before_fuse: bool = True,
        moe_gate_temperature: float = 1.0,
        moe_gate_hidden_dim: int = 0,
        moe_share_cluster_pca: bool = False,
        moe_shared_cluster_pca_dims: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not moe_expert_specs:
            raise ValueError("moe_expert_specs must be a non-empty list")
        self._moe_expert_specs = copy.deepcopy([dict(x) for x in moe_expert_specs])
        self._moe_fusion = str(moe_fusion).strip().lower().replace("-", "_")
        self._moe_layer_norm = bool(moe_layer_norm_before_fuse)
        self._moe_gate_t = float(moe_gate_temperature)
        self._moe_gate_h = int(moe_gate_hidden_dim)
        self._moe_share_cluster_pca = bool(moe_share_cluster_pca)
        self._moe_shared_cluster_pca_dims = (
            None if moe_shared_cluster_pca_dims is None else max(1, int(moe_shared_cluster_pca_dims))
        )
        super().__init__(model=None, **kwargs)

    def _create_default_model(self, input_dim: int) -> nn.Module:
        experts = nn.ModuleList()
        for spec in self._moe_expert_specs:
            nl = int(spec.get("num_layers", self.num_layers))
            fac = float(spec.get("factor", self.factor))
            experts.append(
                create_pcc_model(
                    int(input_dim),
                    self.number_of_components,
                    fac,
                    nl,
                )
            )
        return MixturePCCExperts(
            int(input_dim),
            self.number_of_components,
            experts,
            fusion=self._moe_fusion,
            layer_norm_before_fuse=self._moe_layer_norm,
            gate_temperature=self._moe_gate_t,
            gate_hidden_dim=self._moe_gate_h,
        )

    def _init_clustering_heads(self) -> None:
        self.visualiation_to_cluster = []
        self.cluster_labels = []
        self._cluster_head_expert_index = []

        if not getattr(self, "cluster", False):
            self._cluster_head_expert_index = None
            return

        if self._clusters_auto_tune_active():
            tune_X = self._prepare_cluster_feature_matrix_for_clustering()
            self._maybe_apply_clusters_auto_tune(tune_X)

        default_levels = [int(k) for k in (self.clusters or [])]
        sd = self.sampled_data
        n_feat = int(sd.shape[1])
        n_samples = int(sd.shape[0])
        rank_cap = max(1, n_samples - 1)

        shared_pca_proj: np.ndarray | None = None
        if self._moe_share_cluster_pca:
            max_dims = 0
            any_pca_expert = False
            for spec in self._moe_expert_specs:
                lev = _resolve_cluster_levels(spec.get("clusters", None), default_levels)
                if not lev:
                    continue
                if spec.get("cluster_on_pca", None) is not None:
                    use_e = bool(spec["cluster_on_pca"])
                else:
                    use_e = bool(getattr(self, "cluster_on_pca", False))
                if not use_e:
                    continue
                any_pca_expert = True
                if spec.get("cluster_pca_dims", None) is not None:
                    max_dims = max(max_dims, int(spec["cluster_pca_dims"]))
                else:
                    max_dims = max(max_dims, int(getattr(self, "cluster_pca_dims", 50)))

            if any_pca_expert:
                tgt = self._moe_shared_cluster_pca_dims
                if tgt is not None:
                    n_shared = min(int(tgt), n_feat, rank_cap)
                else:
                    n_shared = min(max_dims, n_feat, rank_cap) if max_dims > 0 else 0
                if n_shared > 0:
                    pca_once = PCA(n_components=n_shared, random_state=self.random_state)
                    shared_pca_proj = pca_once.fit_transform(sd)
                    if self.verbose:
                        print(
                            f"[MiCS MoE] shared cluster PCA once: spectra {n_feat} -> "
                            f"{shared_pca_proj.shape[1]} dims for all PCA experts"
                        )

        kmeans_labels_cache: dict[tuple[Any, ...], Any] = {}

        def _maybe_cached_kmeans(data_for_cluster: np.ndarray, k_eff: int, ctx: str) -> Any:
            ck = _kmeans_label_cache_key(data_for_cluster, k_eff)
            if ck not in kmeans_labels_cache:
                km = KMeans(
                    n_clusters=k_eff,
                    random_state=self.random_state,
                    n_init="auto",
                )
                kmeans_labels_cache[ck] = km.fit_predict(data_for_cluster)
            elif self.verbose:
                print(f"[MiCS MoE] KMeans reuse {ctx} K={k_eff} (same feature matrix + K)")
            return kmeans_labels_cache[ck]

        for ei, spec in enumerate(self._moe_expert_specs):
            lev = _resolve_cluster_levels(spec.get("clusters", None), default_levels)
            if not lev:
                continue

            if spec.get("cluster_on_pca", None) is not None:
                use_pca = bool(spec["cluster_on_pca"])
            else:
                use_pca = bool(getattr(self, "cluster_on_pca", False))

            if spec.get("cluster_pca_dims", None) is not None:
                pca_dims = int(spec["cluster_pca_dims"])
            else:
                pca_dims = int(getattr(self, "cluster_pca_dims", 50))

            if use_pca:
                if shared_pca_proj is not None:
                    data_for_cluster = shared_pca_proj
                else:
                    n_comp = min(pca_dims, n_feat, rank_cap)
                    pca_cluster = PCA(n_components=n_comp, random_state=self.random_state)
                    data_for_cluster = pca_cluster.fit_transform(sd)
                    if self.verbose:
                        print(
                            f"[MiCS MoE] expert[{ei}] KMeans targets via PCA "
                            f"{n_feat} -> {n_comp} dims | levels={lev}"
                        )
            else:
                data_for_cluster = sd

            n_samples_dc = int(data_for_cluster.shape[0])
            for k in lev:
                k_eff = self._clamp_n_clusters(k, n_samples_dc)
                layer = torch.nn.Sequential(
                    torch.nn.Linear(self.number_of_components, k_eff)
                )
                if torch.cuda.is_available():
                    layer = layer.cuda()
                self.visualiation_to_cluster.append(layer)
                labels = _maybe_cached_kmeans(
                    data_for_cluster,
                    k_eff,
                    ctx=f"expert[{ei}]",
                )
                self.cluster_labels.append(labels)
                self._cluster_head_expert_index.append(ei)

        if not self.visualiation_to_cluster:
            self._cluster_head_expert_index = None

    def _embed_for_cluster_head(self, x: torch.Tensor, head_idx: int) -> torch.Tensor:
        routing = getattr(self, "_cluster_head_expert_index", None)
        if routing is None or head_idx < 0 or head_idx >= len(routing):
            return self.model(x)

        ek = routing[head_idx]
        mo = self.model
        if not isinstance(mo, MixturePCCExperts):
            return mo(x)

        y = mo.experts[ek](x)
        if mo.per_expert_ln is not None:
            y = mo.per_expert_ln[ek](y)
        return y

