# chronomoe/io.py
"""
I/O utilities for ChronoMoE Phase 1 telemetry.

Handles JSONL output for events and snapshots, plus run manifest.
All outputs are append-only for crash safety and auditability.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import RoutingEvent
    from .snapshots import SystemSnapshot


class TelemetryWriter:
    """
    Handles JSONL output for Phase 1 telemetry.

    Creates run-specific directory under output_dir with:
    - routing_events.jsonl (append-only event stream)
    - snapshots.jsonl (append-only snapshot stream)
    - run_manifest.json (single file with run metadata)
    """

    def __init__(self, run_id: str, output_dir: str = "outputs"):
        """
        Initialize writer for a specific run.

        Args:
            run_id: Unique identifier for this run
            output_dir: Base directory for all runs (default: "outputs")
        """
        self.run_id = run_id
        self.run_dir = Path(output_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.run_dir / "routing_events.jsonl"
        self.snapshots_path = self.run_dir / "snapshots.jsonl"
        self.manifest_path = self.run_dir / "run_manifest.json"

    def write_event(self, event: 'RoutingEvent') -> None:
        """
        Append single RoutingEvent to JSONL.

        Args:
            event: RoutingEvent to write
        """
        with open(self.events_path, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')

    def write_events_batch(self, events: List['RoutingEvent']) -> None:
        """
        Append batch of RoutingEvents efficiently.

        Args:
            events: List of RoutingEvents to write
        """
        if not events:
            return

        with open(self.events_path, 'a') as f:
            for event in events:
                f.write(json.dumps(event.to_dict()) + '\n')

    def write_snapshot(self, snapshot: 'SystemSnapshot') -> None:
        """
        Append SystemSnapshot to JSONL.

        Args:
            snapshot: SystemSnapshot to write
        """
        with open(self.snapshots_path, 'a') as f:
            f.write(json.dumps(snapshot.to_dict()) + '\n')

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        """
        Write run manifest (once at start).

        Args:
            manifest: Run metadata dictionary
        """
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def flush_events(self, events: List['RoutingEvent']) -> int:
        """
        Batch write and return count. Caller should clear buffer.

        Args:
            events: Events to flush

        Returns:
            Number of events written
        """
        if events:
            self.write_events_batch(events)
        return len(events)


def get_git_commit() -> str:
    """
    Get current git commit hash, or 'unknown' if not in a git repo.

    Returns:
        Short git commit hash or 'unknown'
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def create_run_manifest(
    run_id: str,
    model_name: str,
    config: Dict[str, Any],
    git_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate run_manifest.json structure with complete run metadata.

    Args:
        run_id: Unique run identifier
        model_name: Name of the model
        config: Configuration dictionary
        git_commit: Optional git commit hash (auto-detected if not provided)

    Returns:
        Manifest dictionary ready for JSON serialization
    """
    if git_commit is None:
        git_commit = get_git_commit()

    return {
        "run_id": run_id,
        "model_name": model_name,
        "dataset": config.get("dataset", "unknown"),
        "seed": config.get("seed", 42),
        "architecture": {
            "n_layers": config.get("n_layer", 0),
            "n_experts_per_layer": config.get("n_experts_per_layer", []),
            "top_k": config.get("top_k", 2),
            "capacity_factor": config.get("capacity_factor", 1.25),
        },
        "training": {
            "aux_loss_weight": config.get("aux_loss_weight", 0.01),
            "use_aux_loss": config.get("use_aux_loss", True),
            "router_z_loss_weight": config.get("router_z_loss_weight", 0.001),
            "use_router_z_loss": config.get("use_router_z_loss", True),
            "learning_rate": config.get("learning_rate", 3e-4),
            "batch_size": config.get("batch_size", 64),
            "max_iters": config.get("max_iters", 0),
            "eval_interval": config.get("eval_interval", 0),
        },
        "git_commit": git_commit,
    }


def load_events(events_path: str) -> List[Dict[str, Any]]:
    """
    Load routing events from JSONL file.

    Args:
        events_path: Path to routing_events.jsonl

    Returns:
        List of event dictionaries
    """
    events = []
    with open(events_path, 'r') as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def load_snapshots(snapshots_path: str) -> List[Dict[str, Any]]:
    """
    Load snapshots from JSONL file.

    Args:
        snapshots_path: Path to snapshots.jsonl

    Returns:
        List of snapshot dictionaries
    """
    snapshots = []
    with open(snapshots_path, 'r') as f:
        for line in f:
            if line.strip():
                snapshots.append(json.loads(line))
    return snapshots


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Load run manifest from JSON file.

    Args:
        manifest_path: Path to run_manifest.json

    Returns:
        Manifest dictionary
    """
    with open(manifest_path, 'r') as f:
        return json.load(f)
