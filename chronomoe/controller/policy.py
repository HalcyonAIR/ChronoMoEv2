# chronomoe/controller/policy.py
"""
Policy functions for Phase 2 controller.

Computes topology debt from metrics and updates control state.
All functions are pure where possible.
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..snapshots import LayerSnapshot
    from .state import ControlState, ControlConfig


def compute_topology_debt(
    layer: 'LayerSnapshot',
    config: 'ControlConfig'
) -> Tuple[float, dict]:
    """
    Compute topology debt from layer metrics.

    Debt measures how far the layer is from healthy topology.
    Higher debt = more intervention needed.

    Returns: (total_debt, debt_components)
    """
    n = len(layer.utilization_shares)
    neff = layer.n_effective
    top2 = layer.top2_share
    dead = layer.dead_expert_count

    # Component debts (all clamped to [0,1])

    # Neff debt: how far below threshold
    # If Neff = 0.6*n, debt = 0. If Neff = 0, debt = 1.
    debt_neff = np.clip(
        (config.neff_threshold_ratio * n - neff) / (config.neff_threshold_ratio * n),
        0, 1
    )

    # Top2 debt: how far above warning threshold
    # If Top2 = 0.75, debt = 0. If Top2 = 1.0, debt = 1.
    debt_top2 = np.clip(
        (top2 - config.top2_warning) / (1.0 - config.top2_warning),
        0, 1
    )

    # Dead debt: fraction of experts that are dead
    # Normalize by n-1 (at least 1 expert should be alive)
    debt_dead = np.clip(
        dead / max(1, n - 1),
        0, 1
    )

    # Weighted sum
    total_debt = (
        config.debt_weight_neff * debt_neff +
        config.debt_weight_top2 * debt_top2 +
        config.debt_weight_dead * debt_dead
    )

    components = {
        'debt_neff': float(debt_neff),
        'debt_top2': float(debt_top2),
        'debt_dead': float(debt_dead),
    }

    return float(total_debt), components


def update_control_state(
    state: 'ControlState',
    layer: 'LayerSnapshot',
    config: 'ControlConfig',
    step: int
) -> 'ControlState':
    """
    Update control state from layer snapshot.
    This is the core control loop.

    Phase 2.5: Includes multi-mode steering selection.
    The harm guard tracks which modes help each layer and selects the best.

    Mutates state in place and returns it.
    """
    # Initialize mode scores if not set
    MODES = ['anti_dominance', 'entropy_max', 'lift_tail', 'abstain']
    if state.mode_scores is None:
        state.mode_scores = {m: 1.0 for m in MODES}

    # Closed-loop "do no harm" guard
    # Check if previous intervention made things worse
    current_top2 = layer.top2_share
    top2_delta = current_top2 - state.prev_top2

    harm_threshold = getattr(config, 'harm_top2_threshold', 0.02)
    backoff_factor = getattr(config, 'harm_backoff_factor', 0.5)
    recovery_rate = getattr(config, 'harm_recovery_rate', 0.2)

    # Mode selection config
    mode_harm_penalty = getattr(config, 'mode_harm_penalty', 0.3)
    mode_success_bonus = getattr(config, 'mode_success_bonus', 0.1)
    mode_min_score = getattr(config, 'mode_min_score', 0.2)
    mode_switch_threshold = getattr(config, 'mode_switch_threshold', 0.15)

    harm_detected = state.prev_scale > 0.01 and top2_delta > harm_threshold

    if harm_detected:
        # Intervention increased Top2 - back off and penalize current mode
        state.harm_backoff = state.harm_backoff * backoff_factor

        # Penalize the active mode that caused harm
        active = state.active_mode
        if active in state.mode_scores:
            state.mode_scores[active] = max(
                mode_min_score,
                state.mode_scores[active] - mode_harm_penalty
            )

        # If current mode score dropped below threshold, switch to best alternative
        if state.mode_scores.get(active, 0) < mode_switch_threshold:
            # Find best mode that isn't the current one
            best_mode = active
            best_score = 0
            for mode, score in state.mode_scores.items():
                if mode != active and score > best_score:
                    best_mode = mode
                    best_score = score
            state.active_mode = best_mode
    else:
        # No harm detected - gradually recover backoff and reward current mode
        state.harm_backoff = state.harm_backoff + recovery_rate * (1.0 - state.harm_backoff)

        # Small bonus to current mode for not causing harm
        active = state.active_mode
        if active in state.mode_scores and state.prev_scale > 0.01:
            state.mode_scores[active] = min(
                1.0,
                state.mode_scores[active] + mode_success_bonus
            )

    # Clamp backoff to reasonable range
    state.harm_backoff = np.clip(state.harm_backoff, 0.1, 1.0)

    # Store current top2 for next comparison
    state.prev_top2 = current_top2

    # Compute debt
    debt, components = compute_topology_debt(layer, config)

    # Update pressure (EMA of debt - this gives hysteresis)
    alpha = config.pressure_ema_alpha
    state.pressure = (1 - alpha) * state.pressure + alpha * debt
    state.debt_ema = state.pressure  # Can be same variable
    state.collapse_score = debt

    # Compute heat (complementary to pressure)
    state.heat = np.clip(config.heat_gain * state.pressure, 0, 1)

    # Compute forgetting (increases with pressure)
    state.forgetting = config.forgetting_baseline + config.forgetting_gain * state.pressure

    # Emergency quota computation (Phase 2: compute but don't enforce)
    top2 = layer.top2_share
    dead = layer.dead_expert_count

    if top2 > config.top2_emergency or dead > 0:
        # Identify dominant experts (top 2)
        shares = np.array(layer.utilization_shares)
        dominant_indices = np.argsort(shares)[::-1][:2].tolist()
        state.dominant = dominant_indices

        # Compute recommended quota (cap dominant to average)
        avg_share = 1.0 / len(shares)
        state.quota = [avg_share if i in dominant_indices else 1.0
                      for i in range(len(shares))]
    else:
        state.quota = None
        state.dominant = None

    state.last_update_step = step

    return state


def compute_lens_scale(
    state: 'ControlState',
    config: 'ControlConfig',
    step: int = 0
) -> float:
    """
    Compute lens gating scalar from control state.
    s = clamp(c1*pressure_eff + c2*heat, 0, s_max) * harm_backoff

    Uses capped pressure to prevent over-steering at high severity.
    Empirically, linear response causes over-correction above pressure=0.5.

    Includes warmup: during early training, cap scale to allow lens
    to learn useful warp direction before applying strong intervention.

    Applies harm_backoff: if previous intervention made Top2 worse,
    reduce scale to avoid repeating the harm. This is the closed-loop
    "do no harm" guard.

    EXPLICIT ABSTENTION: When backoff drops below threshold, or when
    there's no meaningful pressure, the controller deliberately chooses
    "no intervention" as a first-class policy decision. This is logged
    separately from low-scale intervention.

    This determines how much the lens warps router input geometry.
    """
    harm_backoff = getattr(state, 'harm_backoff', 1.0)
    abstain_threshold = getattr(config, 'abstain_backoff_threshold', 0.15)

    # EXPLICIT ABSTENTION CHECK
    # Abstain is a deliberate policy choice, not just near-zero scale

    # Reason 1: Harm backoff too low - we've learned intervention hurts this layer
    if harm_backoff < abstain_threshold:
        state.abstain = True
        state.abstain_reason = "harm_backoff"
        state.prev_scale = 0.0
        return 0.0

    # Reason 2: No meaningful pressure - nothing to fix
    if state.pressure < 0.01:
        state.abstain = True
        state.abstain_reason = "no_pressure"
        state.prev_scale = 0.0
        return 0.0

    # Not abstaining - compute scale normally
    state.abstain = False
    state.abstain_reason = ""

    # Cap pressure to prevent over-steering at extreme severity
    # Empirical finding: linear response fails above pressure ~0.5
    pressure_cap = getattr(config, 'pressure_cap', 0.5)
    pressure_effective = min(state.pressure, pressure_cap)

    s = (
        config.lens_pressure_coeff * pressure_effective +
        config.lens_heat_coeff * state.heat
    )

    # Apply warmup cap during early training
    if step < config.lens_warmup_steps:
        s_max = config.lens_warmup_scale
    else:
        s_max = config.lens_scale_max

    # Apply base clamp
    s = float(np.clip(s, 0, s_max))

    # Apply closed-loop "do no harm" backoff
    # If previous intervention increased Top2, reduce this layer's scale
    s = s * harm_backoff

    # Store scale for next harm check
    state.prev_scale = s

    return s
