# chronomoe/events.py
"""
Core data structures for ChronoMoE Phase 1 telemetry.

RoutingEvent: Atomic log record per (step, layer) capturing actual dispatch.
ExpertState: Rolling state per (layer, expert) for health tracking.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Literal
import time


@dataclass
class RoutingEvent:
    """
    Atomic log record emitted per (training_step, moe_layer).
    Records actual dispatch, not just router preferences.
    """

    # Identification
    timestamp: float
    run_id: str
    step: int
    mode: Literal["TRAIN", "INFER"]
    layer_id: int

    # Configuration
    n_experts: int
    top_k: int

    # Ground truth dispatch (REQUIRED)
    # token_count is derived from sum(expert_token_counts) for consistency
    expert_token_counts: List[int]  # Length n_experts
    active_experts: List[int]       # Indices where count > 0

    # Router internals (OPTIONAL)
    router_prob_sums: Optional[List[float]] = None

    # Correlation signals (OPTIONAL)
    loss: Optional[float] = None
    aux_loss: Optional[float] = None

    @property
    def token_count(self) -> int:
        """
        Total tokens dispatched. Derived from expert_token_counts
        to ensure consistency with actual dispatch (not B*T which
        may include padding or masked positions).
        """
        return sum(self.expert_token_counts)

    @classmethod
    def from_router_output(
        cls,
        run_id: str,
        step: int,
        mode: str,
        layer_id: int,
        used_capacity: 'torch.Tensor',  # [n_experts]
        n_experts: int,
        top_k: int,
    ) -> 'RoutingEvent':
        """
        Factory method to create from Router.forward() output.

        Args:
            run_id: Unique identifier for this training run
            step: Current training/inference step
            mode: "TRAIN" or "INFER"
            layer_id: Index of the MoE layer
            used_capacity: Tensor of shape [n_experts] with token counts
            n_experts: Number of experts in the layer
            top_k: Number of experts selected per token

        Returns:
            RoutingEvent with dispatch information
        """
        counts = used_capacity.detach().cpu().tolist()
        # Ensure integer counts
        counts = [int(c) for c in counts]
        active = [i for i, c in enumerate(counts) if c > 0]

        return cls(
            timestamp=time.time(),
            run_id=run_id,
            step=step,
            mode=mode,
            layer_id=layer_id,
            n_experts=n_experts,
            top_k=top_k,
            expert_token_counts=counts,
            active_experts=active,
        )

    def to_dict(self) -> dict:
        """Serialize for JSONL output, including derived token_count."""
        d = asdict(self)
        d['token_count'] = self.token_count
        return d


@dataclass
class ExpertState:
    """
    Rolling state per (layer, expert) - in-memory only.
    Updated from RoutingEvents. Used for health tracking.
    """
    layer_id: int
    expert_id: int

    # Rolling metrics
    utilization_ema: float = 0.0
    share_ema: float = 0.0

    # Health tracking
    dead_steps: int = 0
    is_dead: bool = False

    # Last observation
    last_token_count: int = 0
    last_step: int = 0

    # Configuration
    dead_threshold: int = 50  # Consecutive zero-traffic steps to mark dead
    ema_alpha: float = 0.1    # EMA decay for smoothing

    def update(self, token_count: int, step: int) -> None:
        """
        Update state from new observation.

        Args:
            token_count: Tokens received by this expert in current step
            step: Current training step
        """
        self.last_token_count = token_count
        self.last_step = step

        # Update dead tracking
        if token_count == 0:
            self.dead_steps += 1
            if self.dead_steps > self.dead_threshold:
                self.is_dead = True
        else:
            self.dead_steps = 0
            self.is_dead = False
