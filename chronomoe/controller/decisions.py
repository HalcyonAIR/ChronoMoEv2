# chronomoe/controller/decisions.py
"""
ControlDecision logging for Phase 2 controller.

Every control action is logged for reproducibility and analysis.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..snapshots import LayerSnapshot
    from .state import ControlState


@dataclass
class ControlDecision:
    """
    Logged record of controller action - the paper trail.
    Written at each eval checkpoint.
    """
    # Identity
    run_id: str
    step: int
    layer_id: int

    # Observed metrics (from snapshot)
    observed: Dict[str, float]  # entropy, Neff, Top2, dead

    # Computed control
    computed: Dict[str, float]  # debt, debt_components, pressure, heat, forgetting

    # Actuator state
    actuator: Dict[str, float]  # lens_scale, lens_rank, lens_norms

    # Emergency state
    emergency: Dict[str, Any]  # quota_applied, quota_values

    # Alerts (copied from snapshot if relevant)
    notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_layer_and_state(
        cls,
        run_id: str,
        step: int,
        layer: 'LayerSnapshot',
        state: 'ControlState',
        lens_scale: float,
        lens_rank: int,
        lens_norms: Optional[Dict[str, float]] = None,
        alerts: Optional[List[str]] = None
    ) -> 'ControlDecision':
        """Factory method to create decision log from layer snapshot and control state."""

        observed = {
            'entropy': layer.entropy,
            'n_effective': layer.n_effective,
            'top2_share': layer.top2_share,
            'dead_expert_count': float(layer.dead_expert_count),
        }

        computed = {
            'debt': state.collapse_score,
            'pressure': state.pressure,
            'heat': state.heat,
            'forgetting': state.forgetting,
            'harm_backoff': getattr(state, 'harm_backoff', 1.0),
            'abstain': getattr(state, 'abstain', False),
            'abstain_reason': getattr(state, 'abstain_reason', ''),
            'active_mode': getattr(state, 'active_mode', 'anti_dominance'),
            'mode_scores': getattr(state, 'mode_scores', None),
        }

        actuator = {
            'lens_scale': lens_scale,
            'lens_rank': float(lens_rank),
            'lens_u_norm': lens_norms.get('u_norm', 0.0) if lens_norms else 0.0,
            'lens_v_norm': lens_norms.get('v_norm', 0.0) if lens_norms else 0.0,
        }

        emergency = {
            'quota_applied': state.quota is not None,
            'quota_summary': f"Cap {state.dominant}" if state.quota else "None",
        }

        return cls(
            run_id=run_id,
            step=step,
            layer_id=layer.layer_id,
            observed=observed,
            computed=computed,
            actuator=actuator,
            emergency=emergency,
            notes=alerts or [],
        )
