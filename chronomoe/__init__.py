# chronomoe/__init__.py
"""
ChronoMoE: Temporal Governance and Topology Telemetry for MoE

This module provides observability infrastructure for Mixture-of-Experts
routing topology. Phase 1 is pure instrumentation: log everything,
compute metrics, raise alerts, but don't intervene.

Key Components:
- RoutingEvent: Atomic log record per (step, layer) with actual dispatch
- ExpertState: Rolling health tracking per expert
- SystemSnapshot: Periodic summary with alerts (the regression contract)
- TelemetryWriter: JSONL output handlers

Usage:
    from chronomoe import RoutingEvent, SystemSnapshot, TelemetryWriter
    from chronomoe.metrics import compute_entropy, compute_n_effective
"""

from .events import RoutingEvent, ExpertState
from .snapshots import SystemSnapshot, LayerSnapshot, check_immediate_alerts
from .io import TelemetryWriter, create_run_manifest, get_git_commit, load_events, load_snapshots, load_manifest
from .lens import ChronoLens, IdentityLens, LensState
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
]

__version__ = "0.1.0"
