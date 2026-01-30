# chronomoe/lens.py
"""
ChronoMoE Phase 1 Lens Skeleton

Purpose:
- Provide a stable interface for a "lens" that warps router input geometry.
- Phase 1 behavior is strictly identity (no-op) unless explicitly enabled.
- Phase 2+ will parameterize this lens using pressure/heat signals and basin drift.

Design constraints:
- Must not change model outputs in Phase 1 by default.
- Must be cheap: O(B*T*D) with minimal overhead.
- Must be callable per-layer (layer_id provided).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LensState:
    """
    Minimal state object passed into the lens.

    Phase 1: mostly placeholders.
    Phase 2+: will include pressure/heat/forgetting + per-layer health metrics.
    """
    step: int
    mode: str  # "TRAIN" or "INFER"

    # Optional scalar controls (Phase 2+)
    pressure: float = 0.0
    heat: float = 0.0
    forgetting: float = 0.0

    # Optional per-layer metrics (Phase 2+)
    layer_metrics: Optional[Dict[int, Dict[str, float]]] = None

    # Freeform metadata (avoid logging huge payloads here)
    meta: Optional[Dict[str, Any]] = None


class ChronoLens(nn.Module):
    """
    Identity-by-default lens module.

    Usage:
        lens = ChronoLens(enabled=False)
        x_lensed = lens(x, state, layer_id)

    Parameters:
        enabled: if False, returns x unchanged.
    """

    def __init__(self, enabled: bool = False):
        super().__init__()
        self.enabled = enabled

        # Phase 1: no parameters.
        # Phase 2+: add low-rank warp params, per-layer adapters, etc.

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LensState],
        layer_id: int
    ) -> torch.Tensor:
        """
        Args:
            x: router input tensor, typically shape [B, T, D]
               (or [T, B, D] depending on codebase).
            state: LensState or None. Phase 1 can pass None safely.
            layer_id: which transformer block / MoE layer this routing belongs to.

        Returns:
            x' with same shape as x.
        """
        if not self.enabled:
            return x

        # Phase 1 "enabled" still defaults to identity unless you explicitly
        # implement something here. Keeping it identity avoids accidental drift.
        return x


class IdentityLens(ChronoLens):
    """
    Explicit identity lens (always no-op). Useful as a clear default.
    """

    def __init__(self):
        super().__init__(enabled=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LensState],
        layer_id: int
    ) -> torch.Tensor:
        return x
