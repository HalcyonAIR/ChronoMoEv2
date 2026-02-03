# chronomoe/__init__.py
"""
ChronoMoE: Temporal Governance and Topology Telemetry for MoE

Phase 1: Telemetry - observe and alert
Phase 2: Governance - pressure controller, lens parameterization

Key Components:
- RoutingEvent: Atomic log record per (step, layer) with actual dispatch
- ExpertState: Rolling health tracking per expert
- SystemSnapshot: Periodic summary with alerts (the regression contract)
- TelemetryWriter: JSONL output handlers
- ChronoLens: Low-rank geometry warp for router inputs (Phase 2)
- Controller: Pressure-gated lens control (Phase 2)

Usage (Phase 1):
    from chronomoe import RoutingEvent, SystemSnapshot, TelemetryWriter
    from chronomoe.metrics import compute_entropy, compute_n_effective

Usage (Phase 2):
    from chronomoe import ChronoLens
    from chronomoe.controller import Controller, ControlConfig
"""

from .events import RoutingEvent, ExpertState
from .snapshots import SystemSnapshot, LayerSnapshot, check_immediate_alerts
from .io import TelemetryWriter, create_run_manifest, get_git_commit, load_events, load_snapshots, load_manifest
from .lens import ChronoLens, IdentityLens, LensState, compute_lens_aux_loss
from .metrics import (
    compute_utilization_shares,
    compute_entropy,
    compute_n_effective,
    compute_top2_share,
    count_dead_experts,
    count_nearly_dead_experts,
    compute_layer_metrics,
    compute_gini_coefficient,
)
from .probes import (
    PathologyProbe,
    PathologyConfig,
    SeverityLevel,
    ResilienceSurface,
    compare_surfaces,
)

__all__ = [
    # Data structures
    "RoutingEvent",
    "ExpertState",
    "SystemSnapshot",
    "LayerSnapshot",
    # Lens (Phase 1: identity, Phase 2+: warp)
    "ChronoLens",
    "IdentityLens",
    "LensState",
    "compute_lens_aux_loss",
    # I/O
    "TelemetryWriter",
    "create_run_manifest",
    "get_git_commit",
    "load_events",
    "load_snapshots",
    "load_manifest",
    # Metrics
    "compute_utilization_shares",
    "compute_entropy",
    "compute_n_effective",
    "compute_top2_share",
    "count_dead_experts",
    "count_nearly_dead_experts",
    "compute_layer_metrics",
    "compute_gini_coefficient",
    # Alerts
    "check_immediate_alerts",
    # Probes (forced pathology testing)
    "PathologyProbe",
    "PathologyConfig",
    "SeverityLevel",
    "ResilienceSurface",
    "compare_surfaces",
]

__version__ = "0.2.0"
