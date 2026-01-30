# chronomoe/snapshots.py
"""
SystemSnapshot and LayerSnapshot for ChronoMoE Phase 1.

These are the regression contract - automated tests verify these metrics.
Alerts are embedded in snapshot creation for early collapse detection.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, TYPE_CHECKING
import time

from .metrics import compute_layer_metrics

if TYPE_CHECKING:
    from .events import RoutingEvent


@dataclass
class LayerSnapshot:
    """Per-layer topology metrics at a point in time."""

    layer_id: int

    # Required metrics
    utilization_shares: List[float]  # Length n_experts, sums to 1.0
    entropy: float
    dead_expert_count: int           # STRICT: share == 0

    # Derived metrics
    n_effective: float               # exp(entropy)
    top2_share: float                # Sum of two largest shares

    # Additional tracking
    nearly_dead_count: int           # 0 < share < 0.01 (early warning)
    active_experts: List[int]        # Indices where share > 0

    def to_dict(self) -> dict:
        """Serialize for JSONL output."""
        return asdict(self)


@dataclass
class SystemSnapshot:
    """
    Periodic summary at evaluation cadence.
    This is the REGRESSION CONTRACT - automated tests verify these metrics.
    """

    step: int
    timestamp: float

    # Global metrics
    train_loss: float
    val_loss: Optional[float]

    # Per-layer topology
    layers: List[LayerSnapshot]

    # Alerts (human-readable strings)
    alerts: List[str]

    # Metadata
    run_id: str
    model_name: str

    def to_dict(self) -> dict:
        """Serialize for JSONL output."""
        d = asdict(self)
        d['layers'] = [layer.to_dict() for layer in self.layers]
        return d

    @classmethod
    def from_events(
        cls,
        events: List['RoutingEvent'],
        step: int,
        train_loss: float,
        val_loss: Optional[float],
        run_id: str,
        model_name: str,
        n_layers: int,
        alert_history: Optional[dict] = None,
    ) -> 'SystemSnapshot':
        """
        Factory method to create snapshot from accumulated events.

        Args:
            events: RoutingEvents accumulated since last snapshot
            step: Current training step
            train_loss: Training loss at this step
            val_loss: Validation loss (optional)
            run_id: Unique run identifier
            model_name: Model name for metadata
            n_layers: Number of MoE layers to expect
            alert_history: Optional dict tracking alert persistence
                           {layer_id: {"pre_collapse_count": N, "collapse_count": M}}

        Returns:
            SystemSnapshot with computed metrics and alerts
        """
        layers = []
        all_alerts = []

        # Initialize alert history if not provided
        if alert_history is None:
            alert_history = {}

        for layer_id in range(n_layers):
            shares, entropy, n_eff, top2, dead, nearly_dead = compute_layer_metrics(
                events, layer_id
            )

            if len(shares) == 0:
                continue  # Skip layers with no events

            n_experts = len(shares)

            # Create layer snapshot
            layer = LayerSnapshot(
                layer_id=layer_id,
                utilization_shares=shares.tolist(),
                entropy=float(entropy),
                dead_expert_count=dead,
                n_effective=float(n_eff),
                top2_share=float(top2),
                nearly_dead_count=nearly_dead,
                active_experts=[i for i, s in enumerate(shares) if s > 0],
            )
            layers.append(layer)

            # Initialize layer alert tracking
            if layer_id not in alert_history:
                alert_history[layer_id] = {
                    "pre_collapse_count": 0,
                    "collapse_count": 0,
                }

            # Check alert conditions
            layer_alerts = alert_history[layer_id]

            # Pre-collapse warning: Neff < 60% of n_experts OR Top2 > 0.75
            is_pre_collapse = (n_eff < 0.6 * n_experts) or (top2 > 0.75)
            if is_pre_collapse:
                layer_alerts["pre_collapse_count"] += 1
            else:
                layer_alerts["pre_collapse_count"] = 0

            # Alert after N=3 consecutive snapshots
            if layer_alerts["pre_collapse_count"] >= 3:
                all_alerts.append(
                    f"PRE_COLLAPSE: Layer {layer_id} showing concentration "
                    f"(Neff={n_eff:.2f}/{n_experts}, Top2={top2:.2f}) "
                    f"for {layer_alerts['pre_collapse_count']} snapshots"
                )

            # Collapse confirmed: dead_experts >= 1 AND Top2 > 0.85
            is_collapsed = (dead >= 1) and (top2 > 0.85)
            if is_collapsed:
                layer_alerts["collapse_count"] += 1
            else:
                layer_alerts["collapse_count"] = 0

            # Alert after M=5 consecutive snapshots
            if layer_alerts["collapse_count"] >= 5:
                all_alerts.append(
                    f"COLLAPSE_CONFIRMED: Layer {layer_id} has {dead} dead experts, "
                    f"Top2={top2:.2f} for {layer_alerts['collapse_count']} snapshots"
                )

        return cls(
            step=step,
            timestamp=time.time(),
            train_loss=train_loss,
            val_loss=val_loss,
            layers=layers,
            alerts=all_alerts,
            run_id=run_id,
            model_name=model_name,
        )


def check_immediate_alerts(
    layers: List[LayerSnapshot],
) -> List[str]:
    """
    Check for immediate (non-persistent) alert conditions.
    Use for dashboards that want instant feedback without history tracking.

    Returns:
        List of alert strings for current state only
    """
    alerts = []

    for layer in layers:
        n_experts = len(layer.utilization_shares)

        # Immediate pre-collapse signal
        if layer.n_effective < 0.6 * n_experts or layer.top2_share > 0.75:
            alerts.append(
                f"WARNING: Layer {layer.layer_id} concentration "
                f"(Neff={layer.n_effective:.2f}, Top2={layer.top2_share:.2f})"
            )

        # Immediate collapse signal
        if layer.dead_expert_count >= 1:
            alerts.append(
                f"ALERT: Layer {layer.layer_id} has {layer.dead_expert_count} "
                f"dead expert(s)"
            )

    return alerts
