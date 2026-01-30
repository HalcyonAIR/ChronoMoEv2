# chronomoe/probes.py
"""
Pathology probes for testing topology resilience.

Forces routing pathologies and records recovery trajectories.
Used to compare governance dynamics between models (e.g., teacher vs student).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
from pathlib import Path

import numpy as np


class SeverityLevel(Enum):
    """Predefined pathology severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


@dataclass
class PathologyConfig:
    """Configuration for a single pathology injection."""

    severity: str

    # What fraction of tokens to force to dominant experts
    concentration_ratio: float = 0.8

    # How many steps to maintain forced pathology
    injection_duration: int = 20

    # How many experts to force traffic to (typically 2)
    n_dominant: int = 2

    # Recovery tracking: steps to monitor after release
    recovery_window: int = 100

    # Pressure threshold to consider "recovered"
    recovery_threshold: float = 0.1

    @classmethod
    def from_severity(cls, level: SeverityLevel) -> 'PathologyConfig':
        """Create config from predefined severity level."""
        configs = {
            SeverityLevel.MILD: cls(
                severity="mild",
                concentration_ratio=0.6,
                injection_duration=10,
                recovery_window=50,
            ),
            SeverityLevel.MODERATE: cls(
                severity="moderate",
                concentration_ratio=0.8,
                injection_duration=20,
                recovery_window=100,
            ),
            SeverityLevel.SEVERE: cls(
                severity="severe",
                concentration_ratio=0.95,
                injection_duration=50,
                recovery_window=150,
            ),
            SeverityLevel.CATASTROPHIC: cls(
                severity="catastrophic",
                concentration_ratio=1.0,
                injection_duration=100,
                recovery_window=200,
            ),
        }
        return configs[level]


@dataclass
class TrajectoryPoint:
    """Single point in a recovery trajectory."""
    step: int
    pressure: float
    lens_scale: float
    n_effective: float
    top2_share: float
    dead_count: int

    # Phase of the probe
    phase: str  # "baseline", "injection", "recovery"


@dataclass
class RecoveryTrajectory:
    """Full trajectory from a single pathology probe."""

    # Identification
    model_name: str
    run_id: str
    layer_id: int
    severity: str

    # Config used
    config: PathologyConfig

    # The trajectory data
    points: List[TrajectoryPoint] = field(default_factory=list)

    # Summary metrics (computed after trajectory complete)
    peak_pressure: float = 0.0
    peak_lens_scale: float = 0.0
    min_neff_during_injection: float = 0.0
    steps_to_recovery: int = -1  # -1 if never recovered
    recovery_achieved: bool = False

    def add_point(self, point: TrajectoryPoint):
        """Add a trajectory point."""
        self.points.append(point)

    def compute_summary(self):
        """Compute summary metrics from trajectory."""
        if not self.points:
            return

        # Peak values during injection and recovery
        injection_recovery = [p for p in self.points if p.phase in ("injection", "recovery")]
        if injection_recovery:
            self.peak_pressure = max(p.pressure for p in injection_recovery)
            self.peak_lens_scale = max(p.lens_scale for p in injection_recovery)

        # Min Neff during injection
        injection = [p for p in self.points if p.phase == "injection"]
        if injection:
            self.min_neff_during_injection = min(p.n_effective for p in injection)

        # Steps to recovery
        recovery_start = None
        for i, p in enumerate(self.points):
            if p.phase == "recovery" and recovery_start is None:
                recovery_start = i
            if recovery_start is not None and p.pressure < self.config.recovery_threshold:
                self.steps_to_recovery = i - recovery_start
                self.recovery_achieved = True
                break

        if not self.recovery_achieved:
            self.steps_to_recovery = -1

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            'model_name': self.model_name,
            'run_id': self.run_id,
            'layer_id': self.layer_id,
            'severity': self.severity,
            'config': asdict(self.config),
            'summary': {
                'peak_pressure': self.peak_pressure,
                'peak_lens_scale': self.peak_lens_scale,
                'min_neff_during_injection': self.min_neff_during_injection,
                'steps_to_recovery': self.steps_to_recovery,
                'recovery_achieved': self.recovery_achieved,
            },
            'points': [asdict(p) for p in self.points],
        }


@dataclass
class ResilienceSurface:
    """Collection of trajectories across severity levels for one model."""

    model_name: str
    run_id: str
    layer_id: int

    trajectories: Dict[str, RecoveryTrajectory] = field(default_factory=dict)

    def add_trajectory(self, trajectory: RecoveryTrajectory):
        """Add a trajectory for a severity level."""
        self.trajectories[trajectory.severity] = trajectory

    def get_response_curve(self) -> List[Tuple[str, float, float, int]]:
        """
        Get response curve: (severity, peak_pressure, peak_s, steps_to_recovery).
        Sorted by severity.
        """
        severity_order = ["mild", "moderate", "severe", "catastrophic"]
        curve = []
        for sev in severity_order:
            if sev in self.trajectories:
                t = self.trajectories[sev]
                curve.append((sev, t.peak_pressure, t.peak_lens_scale, t.steps_to_recovery))
        return curve

    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'run_id': self.run_id,
            'layer_id': self.layer_id,
            'response_curve': self.get_response_curve(),
            'trajectories': {k: v.to_dict() for k, v in self.trajectories.items()},
        }


class PathologyProbe:
    """
    Manages pathology injection and trajectory recording.

    Usage:
        probe = PathologyProbe(model_name="teacher", run_id="run1")

        # Run sweep across severity levels
        for severity in SeverityLevel:
            config = PathologyConfig.from_severity(severity)
            probe.start_probe(layer_id=0, config=config)

            for step in range(total_steps):
                # Get routing override (if any)
                override = probe.get_routing_override(step)
                if override:
                    # Apply override to routing
                    pass

                # After controller update, record point
                probe.record_point(step, state, layer_snapshot)

            probe.end_probe()

        # Get resilience surface
        surface = probe.get_surface(layer_id=0)
    """

    def __init__(self, model_name: str, run_id: str, output_dir: str = "outputs"):
        self.model_name = model_name
        self.run_id = run_id
        self.output_dir = Path(output_dir)

        # Current probe state
        self.active_probe: Optional[Tuple[int, PathologyConfig]] = None  # (layer_id, config)
        self.current_trajectory: Optional[RecoveryTrajectory] = None
        self.probe_start_step: int = 0

        # Collected surfaces per layer
        self.surfaces: Dict[int, ResilienceSurface] = {}

    def start_probe(self, layer_id: int, config: PathologyConfig, start_step: int = 0):
        """Start a new pathology probe."""
        self.active_probe = (layer_id, config)
        self.probe_start_step = start_step
        self.current_trajectory = RecoveryTrajectory(
            model_name=self.model_name,
            run_id=self.run_id,
            layer_id=layer_id,
            severity=config.severity,
            config=config,
        )

    def get_phase(self, step: int) -> str:
        """Get current phase of the probe."""
        if self.active_probe is None:
            return "inactive"

        _, config = self.active_probe
        relative_step = step - self.probe_start_step

        if relative_step < 0:
            return "baseline"
        elif relative_step < config.injection_duration:
            return "injection"
        elif relative_step < config.injection_duration + config.recovery_window:
            return "recovery"
        else:
            return "complete"

    def should_override_routing(self, step: int) -> bool:
        """Check if routing should be overridden at this step."""
        return self.get_phase(step) == "injection"

    def get_routing_override(self, step: int, n_experts: int) -> Optional[List[float]]:
        """
        Get routing weight override for forced pathology.

        Returns: List of routing weights [n_experts] or None if no override.
        """
        if not self.should_override_routing(step):
            return None

        _, config = self.active_probe

        # Create concentrated distribution
        weights = np.zeros(n_experts)

        # Dominant experts get the concentration
        dominant_share = config.concentration_ratio / config.n_dominant
        for i in range(min(config.n_dominant, n_experts)):
            weights[i] = dominant_share

        # Remaining experts share the rest
        remaining = 1.0 - config.concentration_ratio
        n_remaining = n_experts - config.n_dominant
        if n_remaining > 0:
            for i in range(config.n_dominant, n_experts):
                weights[i] = remaining / n_remaining

        return weights.tolist()

    def record_point(
        self,
        step: int,
        pressure: float,
        lens_scale: float,
        n_effective: float,
        top2_share: float,
        dead_count: int,
    ):
        """Record a trajectory point."""
        if self.current_trajectory is None:
            return

        phase = self.get_phase(step)
        if phase in ("inactive", "complete"):
            return

        point = TrajectoryPoint(
            step=step,
            pressure=pressure,
            lens_scale=lens_scale,
            n_effective=n_effective,
            top2_share=top2_share,
            dead_count=dead_count,
            phase=phase,
        )
        self.current_trajectory.add_point(point)

    def end_probe(self):
        """End current probe and store trajectory."""
        if self.current_trajectory is None:
            return

        # Compute summary metrics
        self.current_trajectory.compute_summary()

        # Store in surface
        layer_id = self.active_probe[0]
        if layer_id not in self.surfaces:
            self.surfaces[layer_id] = ResilienceSurface(
                model_name=self.model_name,
                run_id=self.run_id,
                layer_id=layer_id,
            )

        self.surfaces[layer_id].add_trajectory(self.current_trajectory)

        # Reset state
        self.active_probe = None
        self.current_trajectory = None

    def get_surface(self, layer_id: int) -> Optional[ResilienceSurface]:
        """Get resilience surface for a layer."""
        return self.surfaces.get(layer_id)

    def save(self, filename: str = "pathology_results.json"):
        """Save all results to JSON."""
        run_dir = self.output_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        output_path = run_dir / filename

        results = {
            'model_name': self.model_name,
            'run_id': self.run_id,
            'surfaces': {str(k): v.to_dict() for k, v in self.surfaces.items()},
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        return output_path


def compare_surfaces(
    teacher_surface: ResilienceSurface,
    student_surface: ResilienceSurface,
) -> Dict[str, any]:
    """
    Compare teacher and student resilience surfaces.

    Returns comparison metrics indicating if student inherited
    control dynamics with lower governance cost.
    """
    teacher_curve = teacher_surface.get_response_curve()
    student_curve = student_surface.get_response_curve()

    # Match by severity
    comparison = {
        'severities': [],
        'pressure_ratio': [],  # student/teacher peak pressure
        'scale_ratio': [],     # student/teacher peak lens scale
        'recovery_ratio': [],  # student/teacher recovery time
    }

    teacher_dict = {t[0]: t for t in teacher_curve}
    student_dict = {s[0]: s for s in student_curve}

    for severity in ["mild", "moderate", "severe", "catastrophic"]:
        if severity in teacher_dict and severity in student_dict:
            t = teacher_dict[severity]
            s = student_dict[severity]

            comparison['severities'].append(severity)

            # Pressure ratio (lower is better for student)
            if t[1] > 0:
                comparison['pressure_ratio'].append(s[1] / t[1])

            # Scale ratio (lower is better for student)
            if t[2] > 0:
                comparison['scale_ratio'].append(s[2] / t[2])

            # Recovery ratio (lower is better for student)
            if t[3] > 0 and s[3] > 0:
                comparison['recovery_ratio'].append(s[3] / t[3])

    # Summary
    if comparison['scale_ratio']:
        avg_scale_ratio = np.mean(comparison['scale_ratio'])
        comparison['governance_efficiency'] = 1.0 / avg_scale_ratio if avg_scale_ratio > 0 else 0
        comparison['student_wins'] = avg_scale_ratio < 1.0
    else:
        comparison['governance_efficiency'] = None
        comparison['student_wins'] = None

    return comparison
