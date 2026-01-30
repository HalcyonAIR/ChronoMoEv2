#!/usr/bin/env python3
"""
Pathology probe tests.

Verifies:
1. Probe phases transition correctly
2. Routing overrides generate correct distributions
3. Trajectory recording works
4. Surface comparison produces meaningful metrics
"""

import sys
import tempfile
import shutil

print("Testing pathology probe imports...")
try:
    from chronomoe.probes import (
        PathologyProbe,
        PathologyConfig,
        SeverityLevel,
        RecoveryTrajectory,
        ResilienceSurface,
        compare_surfaces,
    )
    print("  All probe imports successful")
except ImportError as e:
    print(f"  FAILED: {e}")
    sys.exit(1)


def test_severity_configs():
    """Test predefined severity configurations."""
    print("\nTesting severity configs...")

    for level in SeverityLevel:
        config = PathologyConfig.from_severity(level)
        assert config.severity == level.value
        assert 0 < config.concentration_ratio <= 1.0
        assert config.injection_duration > 0
        assert config.recovery_window > 0

    # Severity should increase
    mild = PathologyConfig.from_severity(SeverityLevel.MILD)
    catastrophic = PathologyConfig.from_severity(SeverityLevel.CATASTROPHIC)

    assert catastrophic.concentration_ratio >= mild.concentration_ratio
    assert catastrophic.injection_duration >= mild.injection_duration

    print("  Severity configs: PASSED")


def test_probe_phases():
    """Test probe phase transitions."""
    print("\nTesting probe phases...")

    probe = PathologyProbe(model_name="test", run_id="test_run")
    config = PathologyConfig(
        severity="test",
        concentration_ratio=0.8,
        injection_duration=10,
        recovery_window=20,
    )

    probe.start_probe(layer_id=0, config=config, start_step=100)

    # Before start
    assert probe.get_phase(99) == "baseline"

    # During injection
    assert probe.get_phase(100) == "injection"
    assert probe.get_phase(105) == "injection"
    assert probe.get_phase(109) == "injection"

    # During recovery
    assert probe.get_phase(110) == "recovery"
    assert probe.get_phase(120) == "recovery"
    assert probe.get_phase(129) == "recovery"

    # Complete
    assert probe.get_phase(130) == "complete"

    print("  Probe phases: PASSED")


def test_routing_override():
    """Test routing override generation."""
    print("\nTesting routing override...")

    probe = PathologyProbe(model_name="test", run_id="test_run")
    config = PathologyConfig(
        severity="test",
        concentration_ratio=0.8,
        n_dominant=2,
        injection_duration=10,
        recovery_window=20,
    )

    probe.start_probe(layer_id=0, config=config, start_step=0)

    # During injection, should get override
    override = probe.get_routing_override(step=5, n_experts=4)
    assert override is not None
    assert len(override) == 4

    # Dominant experts should have high weight
    assert override[0] == 0.4  # 0.8 / 2
    assert override[1] == 0.4

    # Remaining experts share the rest
    assert abs(override[2] - 0.1) < 0.01  # 0.2 / 2
    assert abs(override[3] - 0.1) < 0.01

    # Sum should be 1.0
    assert abs(sum(override) - 1.0) < 0.001

    # Before injection, no override
    assert probe.get_routing_override(step=-1, n_experts=4) is None

    # After injection, no override
    assert probe.get_routing_override(step=15, n_experts=4) is None

    print("  Routing override: PASSED")


def test_trajectory_recording():
    """Test trajectory point recording."""
    print("\nTesting trajectory recording...")

    probe = PathologyProbe(model_name="test", run_id="test_run")
    config = PathologyConfig(
        severity="test",
        concentration_ratio=0.8,
        injection_duration=5,
        recovery_window=10,
        recovery_threshold=0.1,
    )

    probe.start_probe(layer_id=0, config=config, start_step=0)

    # Simulate injection phase
    for step in range(5):
        probe.record_point(
            step=step,
            pressure=0.5 + step * 0.1,  # Rising pressure
            lens_scale=0.02 + step * 0.005,
            n_effective=2.0,
            top2_share=0.9,
            dead_count=2,
        )

    # Simulate recovery phase
    for step in range(5, 15):
        recovery_progress = (step - 5) / 10
        probe.record_point(
            step=step,
            pressure=0.9 * (1 - recovery_progress),  # Falling pressure
            lens_scale=0.04 * (1 - recovery_progress),
            n_effective=2.0 + recovery_progress * 2,
            top2_share=0.9 - recovery_progress * 0.4,
            dead_count=max(0, 2 - int(recovery_progress * 2)),
        )

    probe.end_probe()

    # Check trajectory
    surface = probe.get_surface(layer_id=0)
    assert surface is not None

    trajectory = surface.trajectories["test"]
    assert len(trajectory.points) == 15

    # Check summary
    assert trajectory.peak_pressure > 0.8
    assert trajectory.peak_lens_scale > 0.03
    assert trajectory.min_neff_during_injection < 2.5
    assert trajectory.recovery_achieved

    print(f"  Peak pressure: {trajectory.peak_pressure:.2f}")
    print(f"  Peak scale: {trajectory.peak_lens_scale:.3f}")
    print(f"  Steps to recovery: {trajectory.steps_to_recovery}")
    print("  Trajectory recording: PASSED")


def test_surface_comparison():
    """Test comparing teacher vs student surfaces."""
    print("\nTesting surface comparison...")

    # Create teacher surface (higher intervention needed)
    teacher_surface = ResilienceSurface(
        model_name="teacher",
        run_id="teacher_run",
        layer_id=0,
    )

    for severity, peak_p, peak_s, recovery in [
        ("mild", 0.3, 0.02, 10),
        ("moderate", 0.5, 0.04, 20),
        ("severe", 0.8, 0.05, 40),
    ]:
        config = PathologyConfig(severity=severity, injection_duration=10, recovery_window=50)
        traj = RecoveryTrajectory(
            model_name="teacher",
            run_id="teacher_run",
            layer_id=0,
            severity=severity,
            config=config,
        )
        traj.peak_pressure = peak_p
        traj.peak_lens_scale = peak_s
        traj.steps_to_recovery = recovery
        traj.recovery_achieved = True
        teacher_surface.add_trajectory(traj)

    # Create student surface (lower intervention = better)
    student_surface = ResilienceSurface(
        model_name="student",
        run_id="student_run",
        layer_id=0,
    )

    for severity, peak_p, peak_s, recovery in [
        ("mild", 0.25, 0.015, 8),      # Better than teacher
        ("moderate", 0.45, 0.035, 18),  # Better than teacher
        ("severe", 0.75, 0.045, 35),    # Better than teacher
    ]:
        config = PathologyConfig(severity=severity, injection_duration=10, recovery_window=50)
        traj = RecoveryTrajectory(
            model_name="student",
            run_id="student_run",
            layer_id=0,
            severity=severity,
            config=config,
        )
        traj.peak_pressure = peak_p
        traj.peak_lens_scale = peak_s
        traj.steps_to_recovery = recovery
        traj.recovery_achieved = True
        student_surface.add_trajectory(traj)

    # Compare
    comparison = compare_surfaces(teacher_surface, student_surface)

    assert comparison['student_wins'] == True
    assert comparison['governance_efficiency'] > 1.0  # Student more efficient

    print(f"  Scale ratios: {comparison['scale_ratio']}")
    print(f"  Governance efficiency: {comparison['governance_efficiency']:.2f}")
    print(f"  Student wins: {comparison['student_wins']}")
    print("  Surface comparison: PASSED")


def test_save_results():
    """Test saving results to JSON."""
    print("\nTesting save results...")

    tmpdir = tempfile.mkdtemp(prefix="probe_test_")

    try:
        probe = PathologyProbe(
            model_name="test",
            run_id="test_save",
            output_dir=tmpdir,
        )

        config = PathologyConfig.from_severity(SeverityLevel.MILD)
        probe.start_probe(layer_id=0, config=config, start_step=0)

        # Add a few points
        for step in range(15):
            probe.record_point(
                step=step,
                pressure=0.5,
                lens_scale=0.02,
                n_effective=3.0,
                top2_share=0.6,
                dead_count=0,
            )

        probe.end_probe()

        # Save
        output_path = probe.save()
        assert output_path.exists()

        # Check file is valid JSON
        import json
        with open(output_path) as f:
            data = json.load(f)

        assert data['model_name'] == "test"
        assert '0' in data['surfaces']

        print(f"  Saved to: {output_path}")
        print("  Save results: PASSED")

    finally:
        shutil.rmtree(tmpdir)


def main():
    """Run all probe tests."""
    print("=" * 60)
    print("ChronoMoE Pathology Probe Tests")
    print("=" * 60)

    tests = [
        test_severity_configs,
        test_probe_phases,
        test_routing_override,
        test_trajectory_recording,
        test_surface_comparison,
        test_save_results,
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
