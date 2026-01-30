# chronomoe/controller/__init__.py
"""
Phase 2 Controller Module

Provides pressure-gated lens control for MoE topology governance.
Training-only in Phase 2.

Components:
- ControlState: Per-layer control memory
- ControlConfig: Controller hyperparameters
- Controller: Main integration class
- ControlDecision: Decision logging
- Policy functions: Debt computation, state updates

Usage:
    from chronomoe.controller import Controller, ControlConfig

    controller = Controller(
        n_layers=4,
        n_experts_per_layer=[8, 8, 8, 8],
        config=ControlConfig()
    )
    controller.initialize(run_id="my_run")

    # At each eval checkpoint:
    decisions = controller.update(snapshot, lenses)
"""

from .state import ControlState, ControlConfig
from .policy import compute_topology_debt, update_control_state, compute_lens_scale
from .decisions import ControlDecision
from .hooks import Controller

__all__ = [
    # State
    "ControlState",
    "ControlConfig",
    # Policy
    "compute_topology_debt",
    "update_control_state",
    "compute_lens_scale",
    # Decisions
    "ControlDecision",
    # Integration
    "Controller",
]
