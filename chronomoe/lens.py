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
from enum import Enum
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn


class SteeringMode(Enum):
    """
    Phase 2.5: Multiple steering modes for Clock 2 policy selection.

    The harm guard selects which mode to use based on what helps each layer.
    """
    ANTI_DOMINANCE = "anti_dominance"  # Push away from dominant toward others
    ENTROPY_MAX = "entropy_max"        # Push toward uniform distribution
    LIFT_TAIL = "lift_tail"            # Boost under-utilized only (no suppression)
    ABSTAIN = "abstain"                # Explicit no intervention


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
    Multi-mode steering for router input geometry.

    Phase 1: Identity (no warp)
    Phase 2: Single anti-dominance steering
    Phase 2.5: Multiple steering modes with harm-guard selection
    Phase 3+: Learnable refinement around deterministic baseline

    Steering modes:
    - ANTI_DOMINANCE: Push away from dominant toward others
    - ENTROPY_MAX: Push toward uniform distribution
    - LIFT_TAIL: Boost under-utilized only (no suppression)
    - ABSTAIN: No intervention

    The harm guard tracks which mode helps each layer and selects accordingly.

    Args:
        d_model: Hidden dimension of input
        rank: Rank of low-rank warp (for learnable refinement)
        layer_id: Layer index for logging/debugging

    Usage:
        lens = ChronoLens(d_model=512, rank=8, layer_id=0)
        lens.set_router_weights(router.w_g.weight)  # Set once at init
        lens.set_steering_mode(SteeringMode.ANTI_DOMINANCE)
        lens.update_steering(utilization_shares)  # Updated by controller
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

        # Multi-mode steering (Phase 2.5)
        self.register_buffer('_steering_direction', torch.zeros(d_model))
        self.register_buffer('_router_weights', torch.zeros(1, d_model))  # Placeholder
        self._n_experts = 0
        self._dominant_expert = -1
        self._steering_mode = SteeringMode.ANTI_DOMINANCE
        self._use_deterministic_steering = True  # Phase 2+: use W-based steering
        self._utilization_shares: Optional[List[float]] = None

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

    def set_steering_mode(self, mode: SteeringMode):
        """Set the active steering mode."""
        self._steering_mode = mode

    def get_steering_mode(self) -> SteeringMode:
        """Get the current steering mode."""
        return self._steering_mode

    def _compute_anti_dominance_direction(self, W: torch.Tensor, shares: torch.Tensor) -> torch.Tensor:
        """
        ANTI_DOMINANCE: Push away from dominant toward others.
        Original Phase 2 steering.
        """
        uniform = 1.0 / self._n_experts
        weights = uniform - shares  # Positive for under-utilized, negative for over
        d = torch.einsum('e,ed->d', weights, W)
        return d

    def _compute_entropy_max_direction(self, W: torch.Tensor, shares: torch.Tensor) -> torch.Tensor:
        """
        ENTROPY_MAX: Push toward uniform distribution.
        Uses softmax temperature-style scaling.
        """
        # Direction toward the mean of all expert directions
        W_mean = W.mean(dim=0)
        # Weighted deviation from mean
        deviations = W - W_mean  # [n_exp, d_model]
        uniform = 1.0 / self._n_experts
        # Push toward experts that are below uniform, away from those above
        weights = uniform - shares
        d = torch.einsum('e,ed->d', weights, deviations)
        return d

    def _compute_lift_tail_direction(self, W: torch.Tensor, shares: torch.Tensor) -> torch.Tensor:
        """
        LIFT_TAIL: Boost under-utilized experts only (no suppression).
        Positive-only weights - never pushes against dominant.
        """
        uniform = 1.0 / self._n_experts
        weights = torch.relu(uniform - shares)  # Only positive (under-utilized)
        if weights.sum() < 1e-6:
            return torch.zeros_like(W[0])
        weights = weights / weights.sum()  # Normalize
        d = torch.einsum('e,ed->d', weights, W)
        return d

    def update_steering(self, utilization_shares: list = None):
        """
        Recompute steering direction based on current mode and utilization.

        Args:
            utilization_shares: Current utilization per expert
        """
        if self._n_experts == 0:
            self._steering_direction.zero_()
            return

        W = self._router_weights  # [n_exp, d_model]

        # Store for reference
        self._utilization_shares = utilization_shares

        # ABSTAIN mode: zero direction
        if self._steering_mode == SteeringMode.ABSTAIN:
            self._steering_direction.zero_()
            return

        # Need utilization shares for other modes
        if utilization_shares is None or len(utilization_shares) != self._n_experts:
            # Fallback to uniform assumption
            utilization_shares = [1.0 / self._n_experts] * self._n_experts

        shares = torch.tensor(utilization_shares, device=W.device, dtype=W.dtype)

        # Compute direction based on mode
        if self._steering_mode == SteeringMode.ANTI_DOMINANCE:
            d = self._compute_anti_dominance_direction(W, shares)
        elif self._steering_mode == SteeringMode.ENTROPY_MAX:
            d = self._compute_entropy_max_direction(W, shares)
        elif self._steering_mode == SteeringMode.LIFT_TAIL:
            d = self._compute_lift_tail_direction(W, shares)
        else:
            d = torch.zeros_like(W[0])

        # Normalize
        d_norm = d.norm()
        if d_norm > 1e-6:
            d = d / d_norm

        self._steering_direction = d

        # Update dominant expert tracking (for compatibility)
        dominant_idx = shares.argmax().item()
        self._dominant_expert = dominant_idx

    def set_dominant_expert(self, expert_id: int, utilization_shares: list = None):
        """
        Legacy method - calls update_steering for backward compatibility.

        Args:
            expert_id: Index of dominant expert (ignored, computed from shares)
            utilization_shares: Current utilization per expert
        """
        self.update_steering(utilization_shares)

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
