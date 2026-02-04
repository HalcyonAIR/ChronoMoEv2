# chronomoe/controller/hooks.py
"""
Controller integration for Phase 2 governance.

The Controller class reads Phase 1 snapshots, updates control state,
and gates lens parameters. Training-only in Phase 2.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING

from .state import ControlState, ControlConfig
from .policy import update_control_state, compute_lens_scale
from .decisions import ControlDecision
from .priors import LayerPriors, load_priors, save_priors, apply_priors_to_state

if TYPE_CHECKING:
    from ..snapshots import SystemSnapshot
    from ..lens import ChronoLens


class Controller:
    """
    Phase 2 controller - reads snapshots, updates control state, gates lens.
    Training-only in Phase 2.

    Usage:
        controller = Controller(n_layers=4, n_experts_per_layer=[8, 8, 8, 8])
        controller.initialize(run_id="my_run")

        # At each eval checkpoint:
        decisions = controller.update(snapshot, lenses)
    """

    def __init__(
        self,
        n_layers: int,
        n_experts_per_layer: List[int],
        config: Optional[ControlConfig] = None,
        output_dir: str = "outputs"
    ):
        """
        Initialize controller.

        Args:
            n_layers: Number of MoE layers
            n_experts_per_layer: List of expert counts per layer
            config: Controller hyperparameters (uses defaults if None)
            output_dir: Base directory for output files
        """
        self.config = config or ControlConfig()
        self.n_layers = n_layers

        # Initialize control state per layer
        self.states: Dict[int, ControlState] = {}
        for layer_id in range(n_layers):
            n_experts = n_experts_per_layer[layer_id] if layer_id < len(n_experts_per_layer) else n_experts_per_layer[-1]
            self.states[layer_id] = ControlState(
                layer_id=layer_id,
                n_experts=n_experts
            )

        # Output path
        self.output_dir = Path(output_dir)
        self.decisions_path: Optional[Path] = None
        self.run_id: Optional[str] = None

        # Clock 3: Constitutional priors
        self._priors: Dict[int, LayerPriors] = {}

    def initialize(self, run_id: str):
        """
        Initialize for a training run.

        Args:
            run_id: Unique identifier for this run
        """
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_path = run_dir / "control_decisions.jsonl"
        self.run_id = run_id

    def update(
        self,
        snapshot: 'SystemSnapshot',
        lenses: Optional[Dict[int, 'ChronoLens']] = None
    ) -> List[ControlDecision]:
        """
        Update control state from snapshot and set lens scales.
        Call this at each eval checkpoint.

        Args:
            snapshot: SystemSnapshot with layer metrics
            lenses: Dict mapping layer_id to ChronoLens (optional)

        Returns:
            List of ControlDecision logs
        """
        if lenses is None:
            lenses = {}

        decisions = []

        for layer_snap in snapshot.layers:
            layer_id = layer_snap.layer_id

            # Get or create control state for this layer
            if layer_id not in self.states:
                self.states[layer_id] = ControlState(
                    layer_id=layer_id,
                    n_experts=len(layer_snap.utilization_shares)
                )

            state = self.states[layer_id]

            # Get prior for this layer (if any)
            prior = self.get_layer_prior(layer_id)

            # Update control state from metrics
            update_control_state(state, layer_snap, self.config, snapshot.step, prior=prior)

            # Compute lens scale (with warmup)
            s = compute_lens_scale(state, self.config, step=snapshot.step, prior=prior)

            # Set lens scale, mode, and steering direction (this is the actuator)
            lens_norms = None
            lens_rank = 0
            if layer_id in lenses:
                lens = lenses[layer_id]
                lens.set_scale(s)

                # Phase 2.5: Set steering mode from control state
                from ..lens import SteeringMode
                mode_map = {
                    'anti_dominance': SteeringMode.ANTI_DOMINANCE,
                    'entropy_max': SteeringMode.ENTROPY_MAX,
                    'lift_tail': SteeringMode.LIFT_TAIL,
                    'abstain': SteeringMode.ABSTAIN,
                }
                active_mode = getattr(state, 'active_mode', 'anti_dominance')
                lens.set_steering_mode(mode_map.get(active_mode, SteeringMode.ANTI_DOMINANCE))

                # Update steering direction with current utilization shares
                shares = layer_snap.utilization_shares
                if shares and len(shares) > 0:
                    lens.update_steering(utilization_shares=shares)

                lens_norms = lens.get_norms()
                lens_rank = lens.rank

            # Create decision log
            layer_alerts = [a for a in snapshot.alerts if f"Layer {layer_id}" in a]
            decision = ControlDecision.from_layer_and_state(
                run_id=self.run_id or "unknown",
                step=snapshot.step,
                layer=layer_snap,
                state=state,
                lens_scale=s,
                lens_rank=lens_rank,
                lens_norms=lens_norms,
                alerts=layer_alerts
            )
            decisions.append(decision)

        # Write decisions to JSONL
        self._write_decisions(decisions)

        return decisions

    def _write_decisions(self, decisions: List[ControlDecision]):
        """Append decisions to JSONL file."""
        if self.decisions_path:
            with open(self.decisions_path, 'a') as f:
                for decision in decisions:
                    f.write(json.dumps(decision.to_dict()) + '\n')

    def get_state(self, layer_id: int) -> Optional[ControlState]:
        """Get current control state for a layer."""
        return self.states.get(layer_id)

    def get_all_states(self) -> Dict[int, ControlState]:
        """Get all control states."""
        return self.states.copy()

    def get_lens_scales(self, step: int = 0) -> Dict[int, float]:
        """Get current lens scales for all layers."""
        return {
            layer_id: compute_lens_scale(state, self.config, step=step, prior=self.get_layer_prior(layer_id))
            for layer_id, state in self.states.items()
        }

    # -------------------------------------------------------------------------
    # Clock 3: Constitutional Priors
    # -------------------------------------------------------------------------

    def get_layer_prior(self, layer_id: int) -> Optional[LayerPriors]:
        """
        Get loaded prior for a layer.

        Args:
            layer_id: Layer index

        Returns:
            LayerPriors if loaded, None otherwise
        """
        return self._priors.get(layer_id)

    def load_priors_from_file(self, path: Union[Path, str]) -> bool:
        """
        Load constitutional priors from JSON file and apply to states.

        This enables warm-start: the controller begins with learned
        characteristics from a previous run.

        Args:
            path: Path to priors JSON file

        Returns:
            True if priors were loaded successfully
        """
        path = Path(path)
        if not path.exists():
            print(f"Clock 3: Priors file not found: {path}")
            return False

        try:
            self._priors = load_priors(path)

            # Apply priors to existing states (warm start)
            for layer_id, prior in self._priors.items():
                if layer_id in self.states:
                    apply_priors_to_state(self.states[layer_id], prior)

            print(f"Clock 3: Loaded priors for {len(self._priors)} layers from {path}")
            return True

        except Exception as e:
            print(f"Clock 3: Failed to load priors: {e}")
            return False

    def save_priors_to_file(self, path: Union[Path, str], compress: bool = True) -> Optional[Path]:
        """
        Extract priors from current states and save to JSON file.

        Call this at the end of a run to persist learned characteristics
        for future warm-starts.

        Args:
            path: Path to save priors file
            compress: If True (default), gzip the output

        Returns:
            Path to saved file, or None on failure
        """
        try:
            saved_path = save_priors(self.states, path, self.config, compress=compress)
            print(f"Clock 3: Saved priors for {len(self.states)} layers to {saved_path}")
            return saved_path
        except Exception as e:
            print(f"Clock 3: Failed to save priors: {e}")
            return None

    def has_priors(self) -> bool:
        """Check if priors are loaded."""
        return len(self._priors) > 0

    def get_all_priors(self) -> Dict[int, LayerPriors]:
        """Get all loaded priors."""
        return self._priors.copy()
