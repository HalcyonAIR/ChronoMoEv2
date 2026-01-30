#!/usr/bin/env python3
"""
ChronoMoE Phase 1 Regression Tests

Validates:
1. Collapse metrics computation (Neff, Top2, dead experts)
2. Healthy metrics (no false positives)
3. Alert persistence logic (PRE_COLLAPSE at 3, COLLAPSE_CONFIRMED at 5)
4. Full telemetry flow (healthy -> collapsed transition)

Run with: pytest tests/test_phase1_regression.py -v
"""

import shutil
import tempfile

import numpy as np

from chronomoe.events import RoutingEvent
from chronomoe.snapshots import SystemSnapshot
from chronomoe.io import TelemetryWriter, create_run_manifest, load_snapshots
from chronomoe.metrics import (
    compute_entropy, compute_n_effective, compute_top2_share, count_dead_experts
)


def create_collapsed_events(run_id: str, step: int, n_layers: int = 2,
                            n_experts: int = 4, active_experts: tuple = (0, 1),
                            tokens_per_active: int = 200) -> list:
    """Create mock routing events showing binary collapse."""
    events = []
    for layer_id in range(n_layers):
        counts = [0] * n_experts
        for exp in active_experts:
            counts[exp] = tokens_per_active
        event = RoutingEvent(
            timestamp=1000.0 + step,
            run_id=run_id,
            step=step,
            mode="TRAIN",
            layer_id=layer_id,
            n_experts=n_experts,
            top_k=2,
            expert_token_counts=counts,
            active_experts=list(active_experts),
        )
        events.append(event)
    return events


def create_healthy_events(run_id: str, step: int, n_layers: int = 2,
                          n_experts: int = 4, tokens_per_expert: int = 100) -> list:
    """Create mock routing events showing healthy balanced routing."""
    events = []
    for layer_id in range(n_layers):
        counts = [tokens_per_expert + (i % 5) for i in range(n_experts)]
        event = RoutingEvent(
            timestamp=1000.0 + step,
            run_id=run_id,
            step=step,
            mode="TRAIN",
            layer_id=layer_id,
            n_experts=n_experts,
            top_k=2,
            expert_token_counts=counts,
            active_experts=list(range(n_experts)),
        )
        events.append(event)
    return events


def test_collapse_metrics():
    """Test that metrics correctly identify collapse."""
    events = create_collapsed_events("test", step=0)
    for event in events:
        counts = np.array(event.expert_token_counts)
        shares = counts / counts.sum()
        n_eff = compute_n_effective(compute_entropy(shares))
        top2 = compute_top2_share(shares)
        dead = count_dead_experts(shares)

        assert abs(n_eff - 2.0) < 0.01, f"Neff should be ~2.0, got {n_eff}"
        assert abs(top2 - 1.0) < 0.01, f"Top2 should be ~1.0, got {top2}"
        assert dead == 2, f"Dead should be 2, got {dead}"


def test_healthy_metrics():
    """Test that healthy routing doesn't show collapse."""
    events = create_healthy_events("test", step=0)
    for event in events:
        counts = np.array(event.expert_token_counts)
        shares = counts / counts.sum()
        n_eff = compute_n_effective(compute_entropy(shares))
        dead = count_dead_experts(shares)

        assert n_eff > 3.5, f"Healthy Neff should be >3.5, got {n_eff}"
        assert dead == 0, f"Healthy should have no dead experts, got {dead}"


def test_alert_persistence():
    """Test that alerts fire after persistence threshold."""
    tmp_dir = tempfile.mkdtemp(prefix="chrono_test_")
    try:
        run_id = "test_alerts"
        writer = TelemetryWriter(run_id, tmp_dir)
        writer.write_manifest(create_run_manifest(run_id, "TestModel", {}))

        alert_history = {}
        pre_collapse_fired = False
        collapse_confirmed_fired = False

        for snap_idx in range(10):
            events = create_collapsed_events(run_id, step=snap_idx * 100)
            snapshot = SystemSnapshot.from_events(
                events=events, step=snap_idx * 100, train_loss=3.5, val_loss=4.0,
                run_id=run_id, model_name="TestModel", n_layers=2,
                alert_history=alert_history,
            )
            for alert in snapshot.alerts:
                if "PRE_COLLAPSE" in alert:
                    pre_collapse_fired = True
                if "COLLAPSE_CONFIRMED" in alert:
                    collapse_confirmed_fired = True

        assert pre_collapse_fired, "PRE_COLLAPSE should fire after 3 consecutive snapshots"
        assert collapse_confirmed_fired, "COLLAPSE_CONFIRMED should fire after 5 consecutive snapshots"
    finally:
        shutil.rmtree(tmp_dir)


def test_full_telemetry_flow():
    """Test complete telemetry flow with collapse detection."""
    tmp_dir = tempfile.mkdtemp(prefix="chrono_full_")
    try:
        run_id = "test_flow"
        writer = TelemetryWriter(run_id, tmp_dir)
        writer.write_manifest(create_run_manifest(run_id, "TestModel", {}))

        alert_history = {}

        # Phase 1: Healthy (2 snapshots)
        for i in range(2):
            events = create_healthy_events(run_id, step=i * 100)
            snapshot = SystemSnapshot.from_events(
                events=events, step=i * 100, train_loss=2.0, val_loss=2.2,
                run_id=run_id, model_name="TestModel", n_layers=2,
                alert_history=alert_history,
            )
            writer.write_snapshot(snapshot)
            assert len(snapshot.alerts) == 0, "No alerts in healthy phase"

        # Phase 2: Collapse (8 snapshots)
        final_snapshot = None
        for i in range(2, 10):
            events = create_collapsed_events(run_id, step=i * 100)
            snapshot = SystemSnapshot.from_events(
                events=events, step=i * 100, train_loss=4.0, val_loss=4.5,
                run_id=run_id, model_name="TestModel", n_layers=2,
                alert_history=alert_history,
            )
            writer.write_snapshot(snapshot)
            final_snapshot = snapshot

        # Verify final state
        saved = load_snapshots(str(writer.snapshots_path))
        assert len(saved) == 10, f"Expected 10 snapshots, got {len(saved)}"
        assert final_snapshot.alerts, "Final snapshot should have alerts"
        assert any("COLLAPSE_CONFIRMED" in a for a in final_snapshot.alerts)

        for layer in final_snapshot.layers:
            assert layer.n_effective < 2.5
            assert layer.dead_expert_count >= 1

    finally:
        shutil.rmtree(tmp_dir)


def test_token_count_derived():
    """Test that token_count is derived from expert_token_counts."""
    event = RoutingEvent(
        timestamp=1000.0,
        run_id="test",
        step=0,
        mode="TRAIN",
        layer_id=0,
        n_experts=4,
        top_k=2,
        expert_token_counts=[100, 150, 50, 200],
        active_experts=[0, 1, 2, 3],
    )
    assert event.token_count == 500, f"token_count should be sum of expert counts"


def test_strict_dead_definition():
    """Test that dead experts require exactly zero share."""
    # All experts have some traffic
    shares_healthy = np.array([0.4, 0.3, 0.2, 0.1])
    assert count_dead_experts(shares_healthy) == 0

    # One expert has tiny but non-zero traffic
    shares_nearly_dead = np.array([0.5, 0.49, 0.009, 0.001])
    assert count_dead_experts(shares_nearly_dead) == 0

    # One expert has exactly zero
    shares_one_dead = np.array([0.5, 0.5, 0.0, 0.0])
    assert count_dead_experts(shares_one_dead) == 2


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ChronoMoE Phase 1 Regression Tests")
    print("=" * 60)

    tests = [
        ("Collapse Metrics", test_collapse_metrics),
        ("Healthy Metrics", test_healthy_metrics),
        ("Alert Persistence", test_alert_persistence),
        ("Full Telemetry Flow", test_full_telemetry_flow),
        ("Token Count Derived", test_token_count_derived),
        ("Strict Dead Definition", test_strict_dead_definition),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            print(f"\n{name}...", end=" ")
            fn()
            print("OK")
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
