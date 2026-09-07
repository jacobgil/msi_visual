"""
Differentiable soft ranking for optimization.

Uses `torchsort` when installed (OT-regularized, closest to the original paper).
Otherwise falls back to a pairwise sigmoid soft count — same tensor shape and
keyword arguments, not bit-identical to torchsort.
"""

from __future__ import annotations

import torch

try:
    import torchsort

    _HAS_TORCHSORT = True
except ImportError:
    _HAS_TORCHSORT = False


def _soft_rank_pairwise(
    x: torch.Tensor,
    *,
    dim: int = -1,
    regularization_strength: float = 0.05,
) -> torch.Tensor:
    """Soft rank along `dim` via pairwise sigmoids (ascending: smaller -> smaller rank)."""
    tau = max(float(regularization_strength), 1e-8)
    if dim != -1:
        x = x.movedim(dim, -1)
    xi = x.unsqueeze(-1)
    xj = x.unsqueeze(-2)
    # Sum_j sigmoid((x_i - x_j) / tau) - 0.5 removes the diagonal contribution.
    out = torch.sigmoid((xi - xj) / tau).sum(dim=-1) - 0.5
    if dim != -1:
        out = out.movedim(-1, dim)
    return out


def soft_rank(
    x: torch.Tensor,
    *,
    dim: int = -1,
    regularization_strength: float = 1.0,
) -> torch.Tensor:
    if _HAS_TORCHSORT:
        return torchsort.soft_rank(
            x,
            dim=dim,
            regularization_strength=regularization_strength,
        )
    return _soft_rank_pairwise(
        x,
        dim=dim,
        regularization_strength=regularization_strength,
    )
