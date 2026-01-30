# chronomoe/controller/state.py
"""
Control state and configuration for Phase 2 governance.

ControlState: Per-layer control memory (Clock 2 state)
ControlConfig: Controller hyperparameters
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ControlState:
    """Per-layer control memory - this is Clock 2 state"""

    # Identity
    layer_id: int
    n_experts: int

    # Control parameters (thermodynamics)
    pressure: float = 0.0      # Topology debt EMA
    heat: float = 0.0          # Exploration parameter
    forgetting: float = 0.02   # Baseline decay

    # Internal tracking
    debt_ema: float = 0.0      # Can be same as pressure
    collapse_score: float = 0.0

    # Emergency mechanisms (Phase 2: compute but don't enforce)
    quota: Optional[List[float]] = None  # Per-expert max share cap
    dominant: Optional[List[int]] = None  # Top experts in window

    # Metadata
    last_update_step: int = 0

    def to_dict(self):
        return {
            'layer_id': self.layer_id,
            'n_experts': self.n_experts,
            'pressure': self.pressure,
            'heat': self.heat,
            'forgetting': self.forgetting,
            'collapse_score': self.collapse_score,
            'last_update_step': self.last_update_step,
            'quota': self.quota,
            'dominant': self.dominant,
        }


@dataclass
class ControlConfig:
    """Controller hyperparameters - make these tunable"""

    # Debt thresholds
    neff_threshold_ratio: float = 0.6    # Neff < 0.6*n triggers debt
    top2_warning: float = 0.75           # Top2 > 0.75 triggers debt
    top2_emergency: float = 0.90         # Top2 > 0.90 triggers quota

    # Debt weights
    debt_weight_neff: float = 0.4
    debt_weight_top2: float = 0.4
    debt_weight_dead: float = 0.2

    # Control dynamics
    pressure_ema_alpha: float = 0.2      # EMA smoothing
    heat_gain: float = 1.0               # heat = k * pressure
    forgetting_baseline: float = 0.02
    forgetting_gain: float = 0.10

    # Lens gating
    lens_scale_max: float = 0.05         # s_max - start small!
    lens_pressure_coeff: float = 1.0     # c1
    lens_heat_coeff: float = 0.0         # c2 - keep simple initially
