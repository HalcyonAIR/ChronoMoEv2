# chronomoe/lens.py
"""
ChronoMoE Lens Module

Phase 1: IdentityLens (no-op)
Phase 2: ChronoLens with low-rank residual warp, pressure-gated

The lens transforms router input geometry x → x' to influence routing
decisions without modifying the router itself.

Design constraints:
- Cheap: O(B*T*D) with minimal overhead
- Per-layer: layer_id provided for layer-specific behavior
- Gated: warp strength controlled by external scalar s
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LensState:
    """
    State object passed into the lens.

    Phase 1: mostly placeholders.
    Phase 2+: includes pressure/heat/forgetting from controller.
    """
    step: int
    mode: str  # "TRAIN" or "INFER"

    # Control signals (from controller)
    pressure: float = 0.0
    heat: float = 0.0
    forgetting: float = 0.0

    # Optional per-layer metrics
    layer_metrics: Optional[Dict[int, Dict[str, float]]] = None

    # Freeform metadata
    meta: Optional[Dict[str, Any]] = None


class ChronoLens(nn.Module):
    """
    Anti-dominance steering for router input geometry.

    Phase 1: Identity (no warp)
    Phase 2: Deterministic anti-dominance steering using router weights
    Phase 3+: Learnable refinement around deterministic baseline

    The steering direction is derived from router weights:
    - Identify dominant expert e* from routing statistics
    - Compute d = normalize(W_others - W_e*)
    - Apply: x' = x + s * d (pushes away from dominant expert)

    Args:
        d_model: Hidden dimension of input
        rank: Rank of low-rank warp (for learnable refinement)
        layer_id: Layer index for logging/debugging

    Usage:
        lens = ChronoLens(d_model=512, rank=8, layer_id=0)
        lens.set_router_weights(router.w_g.weight)  # Set once at init
        lens.set_dominant_expert(2)  # Updated by controller
        lens.set_scale(0.1)  # Gated by pressure
        x_steered = lens(x)
    """

    def __init__(self, d_model: int, rank: int = 8, layer_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.layer_id = layer_id

        # Low-rank parameters for learnable refinement (Phase 3+)
        # Initialize tiny so lens starts near-identity while learning direction
        self.V = nn.Parameter(torch.randn(d_model, rank) * 1e-3)
        self.U = nn.Parameter(torch.randn(rank, d_model) * 1e-3)

        # Gating scalar (set by controller, not a learned parameter)
        self.register_buffer('_scale', torch.tensor(0.0))

        # Anti-dominance steering (Phase 2)
        # These are set from router weights, not learned
        self.register_buffer('_steering_direction', torch.zeros(d_model))
        self.register_buffer('_router_weights', torch.zeros(1, d_model))  # Placeholder
        self._n_experts = 0
        self._dominant_expert = -1
        self._use_deterministic_steering = True  # Phase 2: use W-based steering

    @property
    def s(self) -> float:
        """Current gating scale."""
        return self._scale.item()

    def set_router_weights(self, w_g: torch.Tensor):
        """
        Set router weights for computing anti-dominance direction.

        Args:
            w_g: Router weight matrix [n_exp, d_model]
        """
        self._n_experts = w_g.shape[0]
        # Store on same device as lens
        self._router_weights = w_g.detach().clone().to(self._steering_direction.device)

    def set_dominant_expert(self, expert_id: int, utilization_shares: list = None):
        """
        Update dominant expert and recompute steering direction.

        The steering direction pushes toward uniform distribution
        by reducing over-utilized experts and boosting under-utilized ones.

        Args:
            expert_id: Index of dominant expert (0 to n_exp-1)
            utilization_shares: Current utilization per expert (if available)
        """
        if expert_id < 0 or self._n_experts == 0:
            self._dominant_expert = -1
            self._steering_direction.zero_()
            return

        self._dominant_expert = expert_id
        W = self._router_weights  # [n_exp, d_model]

        if utilization_shares is not None and len(utilization_shares) == self._n_experts:
            # Use utilization shares to compute steering toward uniform
            # Direction = weighted sum of W_i, where weight = (1/n - share_i)
            # This pushes toward under-utilized experts, away from over-utilized
            shares = torch.tensor(utilization_shares, device=W.device, dtype=W.dtype)
            uniform = 1.0 / self._n_experts
            weights = uniform - shares  # Positive for under-utilized, negative for over
            # Weight each expert's direction by how much we want to boost/reduce it
            d = torch.einsum('e,ed->d', weights, W)  # [d_model]
        else:
            # Fallback: simple push away from dominant toward others
            W_dominant = W[expert_id]
            mask = torch.ones(self._n_experts, dtype=torch.bool, device=W.device)
            mask[expert_id] = False
            W_others = W[mask].mean(dim=0)
            d = W_others - W_dominant

        d_norm = d.norm()
        if d_norm > 1e-6:
            d = d / d_norm  # Normalize

        self._steering_direction = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply anti-dominance steering and/or learnable warp.

        Args:
            x: Router input tensor [B, T, D]

        Returns:
            Steered tensor with same shape as x
        """
        if self._scale.item() == 0.0:
            return x  # No intervention when disabled

        # Phase 2: Deterministic anti-dominance steering
        if self._use_deterministic_steering and self._dominant_expert >= 0:
            # Add steering direction scaled by pressure
            # This directly reduces logit for dominant expert
            steering = self._steering_direction.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            x = x + self._scale * steering

        # Phase 3+: Learnable low-rank refinement (optional, additive)
        if not self._use_deterministic_steering:
            warp = torch.matmul(torch.matmul(x, self.V), self.U)
            x = x + self._scale * warp

        return x

    def set_scale(self, s: float):
        """
        Update gating scalar from controller.

        Args:
            s: New scale value (typically 0 to lens_scale_max)
        """
        self._scale.fill_(s)

    def get_norms(self) -> Dict[str, float]:
        """Get parameter norms for logging."""
        return {
            'u_norm': self.U.norm().item(),
            'v_norm': self.V.norm().item(),
            'steering_norm': self._steering_direction.norm().item(),
        }

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, rank={self.rank}, layer_id={self.layer_id}, dominant={self._dominant_expert}'


def compute_lens_aux_loss(
    router_probs: torch.Tensor,
    target_entropy: float = 1.386,  # log(4) for 4 experts
    top2_target: float = 0.5,       # Ideal top2 share for 4 experts with top_k=2
) -> torch.Tensor:
    """
    Auxiliary loss to train lens toward anti-collapse directions.

    This gives the lens a direct training signal rather than hoping
    the main task loss teaches it indirectly.

    The loss penalizes:
    1. Low entropy (concentration toward few experts)
    2. High top2 share (dominant experts)

    Args:
        router_probs: Softmax routing probabilities [B, T, n_exp]
        target_entropy: Target entropy (log(n_exp) for uniform)
        top2_target: Target top2 share (0.5 for 4 experts with k=2)

    Returns:
        Scalar loss tensor (backpropagates to lens U, V)
    """
    # Mean probabilities across batch and sequence
    mean_probs = router_probs.mean(dim=(0, 1))  # [n_exp]

    # Entropy loss: penalize low entropy
    eps = 1e-8
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps))
    entropy_loss = torch.relu(target_entropy - entropy)  # Only penalize if below target

    # Top2 loss: penalize high concentration
    sorted_probs, _ = torch.sort(mean_probs, descending=True)
    top2_share = sorted_probs[:2].sum()
    top2_loss = torch.relu(top2_share - top2_target)  # Only penalize if above target

    return entropy_loss + top2_loss


class IdentityLens(nn.Module):
    """
    Explicit identity lens (always no-op).

    Use for Phase 1 or as a drop-in replacement when you want to
    disable lens warping entirely.
    """

    def __init__(self, layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        self.rank = 0
        self._scale = 0.0

    @property
    def s(self) -> float:
        return 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def set_scale(self, s: float):
        """No-op for identity lens."""
        pass

    def get_norms(self) -> Dict[str, float]:
        """Identity lens has no parameters."""
        return {'u_norm': 0.0, 'v_norm': 0.0}
