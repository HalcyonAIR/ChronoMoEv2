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

    # Closed-loop "do no harm" guard
    prev_top2: float = 0.0        # Top2 at previous checkpoint
    prev_scale: float = 0.0       # Scale applied at previous checkpoint
    harm_backoff: float = 1.0     # Multiplier for scale (1.0 = normal, <1.0 = backing off)

    # Explicit abstention mode
    # When True, controller deliberately chooses "no intervention" as policy
    # This is different from low backoff - it's a first-class decision
    abstain: bool = False
    abstain_reason: str = ""      # Why abstaining: "harm_backoff", "no_pressure", etc.

    # Phase 2.5: Multi-mode steering selection
    # The harm guard tracks which modes help this layer
    active_mode: str = "anti_dominance"  # Current steering mode
    mode_scores: Optional[dict] = None   # Per-mode success scores

    # Clock 3: Intervention outcome tracking
    # EMA of whether intervention helps (+1) or hurts (-1)
    intervention_helped_ema: float = 0.0

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
            'harm_backoff': self.harm_backoff,
            'prev_top2': self.prev_top2,
            'abstain': self.abstain,
            'abstain_reason': self.abstain_reason,
            'active_mode': self.active_mode,
            'mode_scores': self.mode_scores,
            'intervention_helped_ema': self.intervention_helped_ema,
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
    lens_scale_max: float = 0.5          # s_max - allow strong intervention
    lens_pressure_coeff: float = 1.0     # c1
    lens_heat_coeff: float = 0.0         # c2 - keep simple initially

    # Pressure cap: prevent over-steering at extreme severity
    # Empirical finding: linear response fails above ~0.5 pressure
    pressure_cap: float = 0.5            # Max effective pressure for lens scale

    # Warmup: cap scale during early training while lens learns direction
    lens_warmup_steps: int = 100         # Steps before full scale allowed
    lens_warmup_scale: float = 0.02      # Max scale during warmup

    # Closed-loop "do no harm" guard
    # If Top2 increases after intervention, back off scale for that layer
    harm_top2_threshold: float = 0.02    # Min Top2 increase to trigger backoff
    harm_backoff_factor: float = 0.5     # Multiply scale by this on harm detection
    harm_recovery_rate: float = 0.2      # Rate at which backoff recovers toward 1.0

    # Explicit abstention
    # When backoff drops below this, switch to explicit abstain mode
    # Abstain = deliberate "no intervention" policy, not just near-zero scale
    abstain_backoff_threshold: float = 0.15  # Below this, abstain entirely

    # Phase 2.5: Multi-mode steering selection
    # Harm guard selects which mode helps each layer
    mode_harm_penalty: float = 0.3       # Reduce mode score on harm detection
    mode_success_bonus: float = 0.1      # Increase mode score on success
    mode_min_score: float = 0.2          # Below this, mode is disabled
    mode_switch_threshold: float = 0.15  # Switch modes if current score below this
