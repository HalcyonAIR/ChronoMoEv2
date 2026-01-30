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
    Low-rank residual warp for router input geometry.

    x' = x + s * (x @ V) @ U

    Where s is gated by pressure/heat from controller.

    Args:
        d_model: Hidden dimension of input
        rank: Rank of low-rank warp (default 8)
        layer_id: Layer index for logging/debugging

    Usage:
        lens = ChronoLens(d_model=512, rank=8, layer_id=0)
        lens.set_scale(0.02)  # Set by controller
        x_warped = lens(x)
    """

    def __init__(self, d_model: int, rank: int = 8, layer_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.layer_id = layer_id

        # Low-rank parameters
        # Initialize very small so it's near-identity even with s>0
        self.V = nn.Parameter(torch.randn(d_model, rank) * 1e-3)
        self.U = nn.Parameter(torch.randn(rank, d_model) * 1e-3)

        # Gating scalar (set by controller, not a learned parameter)
        self.register_buffer('_scale', torch.tensor(0.0))

    @property
    def s(self) -> float:
        """Current gating scale."""
        return self._scale.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply lens warp.

        Args:
            x: Router input tensor [B, T, D] or [T, B, D]

        Returns:
            Warped tensor with same shape as x
        """
        if self._scale.item() == 0.0:
            return x  # No warp when disabled

        # Low-rank residual: x + s * (x @ V) @ U
        warp = torch.matmul(torch.matmul(x, self.V), self.U)
        return x + self._scale * warp

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
        }

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, rank={self.rank}, layer_id={self.layer_id}'


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
