#!/usr/bin/env python3
"""
Phase 2 Controller Tests

Verifies:
1. Lens doesn't perturb baseline (s=0 matches Phase 1)
2. Controller detects collapse and reacts
3. Debt computation is deterministic
4. Control state updates correctly
"""

import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np

print("Testing Phase 2 imports...")
try:
    import torch
    from chronomoe import ChronoLens, IdentityLens, LensState
    from chronomoe.controller import (
        Controller,
        ControlConfig,
        ControlState,
        compute_topology_debt,
        update_control_state,
        compute_lens_scale,
        ControlDecision,
    )
    from chronomoe import (
        RoutingEvent,
        SystemSnapshot,
        LayerSnapshot,
    )
    print("  All Phase 2 imports successful")
except ImportError as e:
    print(f"  FAILED: {e}")
    sys.exit(1)


def create_layer_snapshot(
    layer_id: int,
    shares: list,
    entropy: float = None,
    n_effective: float = None,
) -> LayerSnapshot:
    """Create a LayerSnapshot for testing."""
    shares_arr = np.array(shares)
    if entropy is None:
        nonzero = shares_arr[shares_arr > 1e-10]
        entropy = float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) > 0 else 0.0
    if n_effective is None:
        n_effective = float(np.exp(entropy))

    top2 = float(np.sort(shares_arr)[::-1][:2].sum()) if len(shares_arr) >= 2 else 1.0
    dead = int(np.sum(shares_arr == 0))
    nearly_dead = int(np.sum((shares_arr > 0) & (shares_arr < 0.01)))
    active = [i for i, s in enumerate(shares_arr) if s > 0]

    return LayerSnapshot(
        layer_id=layer_id,
        utilization_shares=list(shares),
        entropy=entropy,
        dead_expert_count=dead,
        n_effective=n_effective,
        top2_share=top2,
        nearly_dead_count=nearly_dead,
        active_experts=active,
    )


def test_lens_identity_at_zero_scale():
    """Lens with s=0 should be identity."""
    print("\nTesting lens identity at zero scale...")

    d_model = 64
    lens = ChronoLens(d_model=d_model, rank=8, layer_id=0)

    # Scale is 0 by default
    assert lens.s == 0.0, f"Default scale should be 0, got {lens.s}"

    # Forward should be identity
    x = torch.randn(2, 10, d_model)
    y = lens(x)

    assert torch.allclose(x, y), "Lens with s=0 should return input unchanged"
    print("  Lens identity at s=0: PASSED")


def test_lens_warp_with_scale():
    """Lens with s>0 should warp input."""
    print("\nTesting lens warp with scale...")

    d_model = 64
    lens = ChronoLens(d_model=d_model, rank=8, layer_id=0)

    x = torch.randn(2, 10, d_model)

    # Set scale
    lens.set_scale(0.05)
    assert abs(lens.s - 0.05) < 1e-6, f"Scale should be ~0.05, got {lens.s}"

    y = lens(x)

    # Should be different from input
    diff = (y - x).abs().sum()
    assert diff > 0, "Lens with s>0 should warp input"

    # But not too different (residual warp is small)
    relative_diff = diff / x.abs().sum()
    assert relative_diff < 0.1, f"Warp should be small, got {relative_diff:.4f}"

    print(f"  Lens warp magnitude: {diff:.4f} (relative: {relative_diff:.4f})")
    print("  Lens warp with scale: PASSED")


def test_identity_lens():
    """IdentityLens should always be no-op."""
    print("\nTesting IdentityLens...")

    lens = IdentityLens(layer_id=0)

    x = torch.randn(2, 10, 64)
    y = lens(x)

    assert torch.equal(x, y), "IdentityLens should return input unchanged"

    # set_scale should be no-op
    lens.set_scale(999.0)
    y2 = lens(x)
    assert torch.equal(x, y2), "IdentityLens should ignore scale"

    print("  IdentityLens: PASSED")


def test_debt_computation_healthy():
    """Healthy layer should have zero debt."""
    print("\nTesting debt computation (healthy)...")

    config = ControlConfig()

    # Balanced 4-expert distribution
    layer = create_layer_snapshot(
        layer_id=0,
        shares=[0.25, 0.25, 0.25, 0.25],
    )

    debt, components = compute_topology_debt(layer, config)

    # Should be near zero
    assert debt < 0.01, f"Healthy layer should have near-zero debt, got {debt}"

    print(f"  Healthy debt: {debt:.4f}")
    print(f"  Components: {components}")
    print("  Debt computation (healthy): PASSED")


def test_debt_computation_collapsed():
    """Collapsed layer should have high debt."""
    print("\nTesting debt computation (collapsed)...")

    config = ControlConfig()

    # Collapsed: 2 experts dominate, 2 dead
    layer = create_layer_snapshot(
        layer_id=0,
        shares=[0.5, 0.5, 0.0, 0.0],
    )

    debt, components = compute_topology_debt(layer, config)

    # Should be high
    assert debt > 0.5, f"Collapsed layer should have high debt, got {debt}"
    assert components['debt_dead'] > 0, "Should detect dead experts"
    assert components['debt_top2'] > 0, "Should detect top2 concentration"

    print(f"  Collapsed debt: {debt:.4f}")
    print(f"  Components: {components}")
    print("  Debt computation (collapsed): PASSED")


def test_control_state_update():
    """Control state should update from layer metrics."""
    print("\nTesting control state update...")

    config = ControlConfig()
    state = ControlState(layer_id=0, n_experts=4)

    # Initial state
    assert state.pressure == 0.0
    assert state.heat == 0.0

    # Update with collapsed layer
    layer = create_layer_snapshot(
        layer_id=0,
        shares=[0.5, 0.5, 0.0, 0.0],
    )

    update_control_state(state, layer, config, step=100)

    # Pressure should increase
    assert state.pressure > 0, "Pressure should increase with collapse"
    assert state.heat > 0, "Heat should increase with pressure"
    assert state.collapse_score > 0, "Collapse score should be set"
    assert state.last_update_step == 100

    # Emergency should be triggered (top2 > 0.85)
    assert state.dominant is not None, "Should detect dominant experts"
    assert state.quota is not None, "Should compute quota"

    print(f"  Pressure: {state.pressure:.4f}")
    print(f"  Heat: {state.heat:.4f}")
    print(f"  Dominant: {state.dominant}")
    print("  Control state update: PASSED")


def test_lens_scale_gating():
    """Lens scale should be gated by pressure."""
    print("\nTesting lens scale gating...")

    config = ControlConfig(
        lens_scale_max=0.05,
        lens_pressure_coeff=1.0,
    )

    # Low pressure -> low scale
    state_low = ControlState(layer_id=0, n_experts=4, pressure=0.01)
    scale_low = compute_lens_scale(state_low, config)

    # High pressure -> higher scale (but capped)
    state_high = ControlState(layer_id=0, n_experts=4, pressure=0.5)
    scale_high = compute_lens_scale(state_high, config)

    assert scale_low < scale_high, "Higher pressure should give higher scale"
    assert scale_high <= config.lens_scale_max, "Scale should be capped"

    print(f"  Low pressure ({state_low.pressure}) -> scale {scale_low:.4f}")
    print(f"  High pressure ({state_high.pressure}) -> scale {scale_high:.4f}")
    print("  Lens scale gating: PASSED")


def test_controller_integration():
    """Controller should update state and set lens scales."""
    print("\nTesting controller integration...")

    tmpdir = tempfile.mkdtemp(prefix="chrono_phase2_test_")

    try:
        config = ControlConfig()
        controller = Controller(
            n_layers=2,
            n_experts_per_layer=[4, 4],
            config=config,
            output_dir=tmpdir,
        )
        controller.initialize(run_id="test_run")

        # Create lenses
        d_model = 64
        lenses = {
            0: ChronoLens(d_model=d_model, rank=8, layer_id=0),
            1: ChronoLens(d_model=d_model, rank=8, layer_id=1),
        }

        # Create snapshot with one healthy, one collapsed layer
        snapshot = SystemSnapshot(
            step=100,
            timestamp=1000.0,
            train_loss=2.5,
            val_loss=2.8,
            layers=[
                create_layer_snapshot(0, [0.25, 0.25, 0.25, 0.25]),  # Healthy
                create_layer_snapshot(1, [0.5, 0.5, 0.0, 0.0]),     # Collapsed
            ],
            alerts=[],
            run_id="test_run",
            model_name="TestModel",
        )

        # Update controller
        decisions = controller.update(snapshot, lenses)

        # Check decisions
        assert len(decisions) == 2

        # Layer 0 should have low pressure
        d0 = decisions[0]
        assert d0.layer_id == 0
        assert d0.computed['pressure'] < 0.1, f"Healthy layer pressure too high: {d0.computed['pressure']}"

        # Layer 1 should have high pressure
        d1 = decisions[1]
        assert d1.layer_id == 1
        assert d1.computed['pressure'] > 0.1, f"Collapsed layer pressure too low: {d1.computed['pressure']}"

        # Lens scales should be set
        assert lenses[1].s > lenses[0].s, "Collapsed layer should have higher lens scale"

        # Check decisions file was written
        decisions_path = Path(tmpdir) / "test_run" / "control_decisions.jsonl"
        assert decisions_path.exists(), "Decisions file should be written"

        print(f"  Layer 0 pressure: {d0.computed['pressure']:.4f}, scale: {d0.actuator['lens_scale']:.4f}")
        print(f"  Layer 1 pressure: {d1.computed['pressure']:.4f}, scale: {d1.actuator['lens_scale']:.4f}")
        print("  Controller integration: PASSED")

    finally:
        shutil.rmtree(tmpdir)


def test_pressure_ema_hysteresis():
    """Pressure EMA should provide hysteresis."""
    print("\nTesting pressure EMA hysteresis...")

    config = ControlConfig(pressure_ema_alpha=0.2)
    state = ControlState(layer_id=0, n_experts=4)

    # Simulate sequence: healthy -> collapsed -> healthy
    healthy = create_layer_snapshot(0, [0.25, 0.25, 0.25, 0.25])
    collapsed = create_layer_snapshot(0, [0.5, 0.5, 0.0, 0.0])

    pressures = []

    # Start healthy
    for step in range(5):
        update_control_state(state, healthy, config, step)
        pressures.append(state.pressure)

    # Collapse
    for step in range(5, 15):
        update_control_state(state, collapsed, config, step)
        pressures.append(state.pressure)

    # Recover
    for step in range(15, 25):
        update_control_state(state, healthy, config, step)
        pressures.append(state.pressure)

    # Pressure should rise gradually during collapse
    assert pressures[10] > pressures[5], "Pressure should rise during collapse"

    # Pressure should fall gradually during recovery (hysteresis)
    assert pressures[20] < pressures[14], "Pressure should fall during recovery"
    assert pressures[20] > 0, "Pressure should have hysteresis (not instant drop)"

    print(f"  Healthy (step 4): {pressures[4]:.4f}")
    print(f"  Collapsed (step 14): {pressures[14]:.4f}")
    print(f"  Recovering (step 20): {pressures[20]:.4f}")
    print("  Pressure EMA hysteresis: PASSED")


def main():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("ChronoMoE Phase 2 - Controller Tests")
    print("=" * 60)

    tests = [
        test_lens_identity_at_zero_scale,
        test_lens_warp_with_scale,
        test_identity_lens,
        test_debt_computation_healthy,
        test_debt_computation_collapsed,
        test_control_state_update,
        test_lens_scale_gating,
        test_controller_integration,
        test_pressure_ema_hysteresis,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
