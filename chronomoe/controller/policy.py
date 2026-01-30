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

    Mutates state in place and returns it.
    """
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


def compute_lens_scale(state: 'ControlState', config: 'ControlConfig') -> float:
    """
    Compute lens gating scalar from control state.
    s = clamp(c1*pressure + c2*heat, 0, s_max)

    This determines how much the lens warps router input geometry.
    """
    s = (
        config.lens_pressure_coeff * state.pressure +
        config.lens_heat_coeff * state.heat
    )
    return float(np.clip(s, 0, config.lens_scale_max))
