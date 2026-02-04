# chronomoe/controller/priors.py
"""
Clock 3 Constitutional Priors Module

Constitutional priors capture geometry-agnostic learned characteristics that
persist across model resets. Unlike raw control state (which is entangled with
specific weight trajectories), priors represent "what kind of system this is":

- fragility: How much intervention this layer typically needs
- backoff_bias: Persistent memory of past harm (slow decay)
- scale_cap: Learned maximum safe intervention intensity
- abstain_threshold: Learned threshold for "don't intervene"
- intervention_helped_ema: Track record of whether intervention helps

These priors survive weight resets and inform the controller about layer
characteristics without requiring relearning from scratch.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any, List, Tuple, Union, TYPE_CHECKING
import json
import gzip
from pathlib import Path

if TYPE_CHECKING:
    from .state import ControlState, ControlConfig

# Fixed mode enum - never grows
STEERING_MODES = ('anti_dominance', 'entropy_max', 'lift_tail', 'abstain')
MODE_TO_ID = {m: i for i, m in enumerate(STEERING_MODES)}
ID_TO_MODE = {i: m for i, m in enumerate(STEERING_MODES)}


def _round(v: float, decimals: int = 3) -> float:
    """Round float to reduce JSON bloat."""
    return round(v, decimals)


@dataclass
class LayerPriors:
    """
    Constitutional prior for a single layer.

    These are geometry-agnostic characteristics learned from experience
    that survive model weight resets.

    Design: Fixed width, bounded storage. No growing arrays.
    - All floats rounded to 3 decimals
    - Mode stored as ID (0-3) not string
    - Top-3 mode scores only (fixed array, not dict)
    """
    layer_id: int

    # Intervention intensity multiplier (1.0 = neutral)
    # > 1.0: layer needs more help (fragile)
    # < 1.0: layer is stable, back off
    fragility: float = 1.0

    # Persistent harm memory (slow decay)
    # Carries forward learned backoff from previous runs
    backoff_bias: float = 0.0

    # Per-layer maximum scale (None = use config default)
    scale_cap: Optional[float] = None

    # Per-layer abstain threshold (None = use config default)
    abstain_threshold: Optional[float] = None

    # EMA of intervention outcomes
    # +1 = intervention helped (Top2 decreased)
    # -1 = intervention hurt (Top2 increased)
    # Tracks whether intervention generally helps this layer
    intervention_helped_ema: float = 0.0

    # Preferred steering mode (learned from mode selection history)
    preferred_mode: Optional[str] = None

    # Per-mode scores from previous run (can warm-start mode selection)
    # Now stored as fixed-size array [anti_dom, entropy_max, lift_tail, abstain]
    mode_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to compact dict for JSON storage.
        - Omits None values
        - Rounds floats to 3 decimals
        - Converts mode_scores dict to fixed array
        """
        d = {'l': self.layer_id}  # Compact key

        # Only include non-default values
        if self.fragility != 1.0:
            d['f'] = _round(self.fragility)
        if self.backoff_bias != 0.0:
            d['b'] = _round(self.backoff_bias)
        if self.scale_cap is not None:
            d['sc'] = _round(self.scale_cap)
        if self.abstain_threshold is not None:
            d['at'] = _round(self.abstain_threshold)
        if self.intervention_helped_ema != 0.0:
            d['ih'] = _round(self.intervention_helped_ema)
        if self.preferred_mode is not None:
            d['pm'] = MODE_TO_ID.get(self.preferred_mode, 0)
        if self.mode_scores is not None:
            # Store as fixed array [anti_dom, entropy_max, lift_tail, abstain]
            d['ms'] = [_round(self.mode_scores.get(m, 1.0)) for m in STEERING_MODES]

        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerPriors':
        """Deserialize from compact dict."""
        # Support both compact and verbose keys for backwards compat
        layer_id = data.get('l', data.get('layer_id', 0))
        fragility = data.get('f', data.get('fragility', 1.0))
        backoff_bias = data.get('b', data.get('backoff_bias', 0.0))
        scale_cap = data.get('sc', data.get('scale_cap'))
        abstain_threshold = data.get('at', data.get('abstain_threshold'))
        intervention_helped_ema = data.get('ih', data.get('intervention_helped_ema', 0.0))

        # Mode: support both ID and string
        pm = data.get('pm', data.get('preferred_mode'))
        if isinstance(pm, int):
            preferred_mode = ID_TO_MODE.get(pm, 'anti_dominance')
        else:
            preferred_mode = pm

        # Mode scores: support both array and dict
        ms = data.get('ms', data.get('mode_scores'))
        if isinstance(ms, list):
            mode_scores = {m: ms[i] for i, m in enumerate(STEERING_MODES) if i < len(ms)}
        else:
            mode_scores = ms

        return cls(
            layer_id=layer_id,
            fragility=fragility,
            backoff_bias=backoff_bias,
            scale_cap=scale_cap,
            abstain_threshold=abstain_threshold,
            intervention_helped_ema=intervention_helped_ema,
            preferred_mode=preferred_mode,
            mode_scores=mode_scores,
        )


def extract_priors_from_states(
    states: Dict[int, 'ControlState'],
    config: 'ControlConfig',
    global_mean_scale: Optional[float] = None,
) -> Dict[int, LayerPriors]:
    """
    Extract constitutional priors from control states.

    This is called at the end of a run to capture learned characteristics
    for persistence.

    Args:
        states: Dict of layer_id -> ControlState
        config: ControlConfig for reference values
        global_mean_scale: Mean scale across all layers (for fragility calc)

    Returns:
        Dict of layer_id -> LayerPriors
    """
    # Compute global mean scale if not provided
    if global_mean_scale is None:
        scales = [s.prev_scale for s in states.values()]
        global_mean_scale = sum(scales) / max(1, len(scales))

    priors = {}
    for layer_id, state in states.items():
        # Fragility = relative intervention need
        # If this layer's typical scale > global mean, it's fragile
        layer_scale = state.prev_scale
        if global_mean_scale > 0.001:
            fragility = layer_scale / global_mean_scale
        else:
            fragility = 1.0
        # Clamp to reasonable range
        fragility = max(0.5, min(2.0, fragility))

        # Backoff bias: carry forward harm_backoff deficit
        # 1.0 - harm_backoff = how much we've backed off
        backoff_bias = 1.0 - state.harm_backoff

        # Learn scale cap from observed maximum if we hit issues
        # If harm_backoff dropped, current scale was too high
        scale_cap = None
        if state.harm_backoff < 0.5 and state.prev_scale > 0:
            # We're backing off significantly - remember this as a cap
            scale_cap = state.prev_scale * 0.8  # 20% safety margin

        # Learn abstain threshold from behavior
        abstain_threshold = None
        if state.abstain and state.abstain_reason == "harm_backoff":
            # We learned to abstain - remember the threshold that caused it
            abstain_threshold = config.abstain_backoff_threshold * 1.5  # Higher = abstain earlier

        # Intervention outcome tracking is updated during training
        # Just copy the current EMA value
        intervention_helped_ema = getattr(state, 'intervention_helped_ema', 0.0)

        priors[layer_id] = LayerPriors(
            layer_id=layer_id,
            fragility=fragility,
            backoff_bias=backoff_bias,
            scale_cap=scale_cap,
            abstain_threshold=abstain_threshold,
            intervention_helped_ema=intervention_helped_ema,
            preferred_mode=state.active_mode,
            mode_scores=state.mode_scores.copy() if state.mode_scores else None,
        )

    return priors


def write_priors(
    priors: Dict[int, LayerPriors],
    path: Union[Path, str],
    compress: bool = True,
) -> Path:
    """
    Write priors dict to JSON file (optionally gzipped).

    This is the low-level serialization function. Use this when you
    already have LayerPriors objects (e.g., from telemetry parsing).

    Args:
        priors: Dict of layer_id -> LayerPriors
        path: Path to save file (will add .gz if compress=True)
        compress: If True, gzip the output (highly recommended)

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to compact serializable format
    data = {
        'v': 2,  # Version 2 = compact format
        'p': {str(k): v.to_dict() for k, v in priors.items()}
    }

    json_str = json.dumps(data, separators=(',', ':'))  # Compact JSON

    if compress:
        # Add .gz extension if not present
        if not path.suffix == '.gz':
            path = path.with_suffix(path.suffix + '.gz')
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(json_str)
    else:
        with open(path, 'w') as f:
            f.write(json_str)

    return path


def save_priors(
    states: Dict[int, 'ControlState'],
    path: Union[Path, str],
    config: 'ControlConfig',
    compress: bool = True,
) -> Path:
    """
    Extract priors from states and save to JSON file (optionally gzipped).

    This is the high-level function for use with a running controller.
    It extracts priors from ControlState objects and writes them.

    Args:
        states: Dict of layer_id -> ControlState
        path: Path to save file (will add .gz if compress=True)
        config: ControlConfig for reference values
        compress: If True, gzip the output (highly recommended)

    Returns:
        Path to saved file
    """
    priors = extract_priors_from_states(states, config)
    return write_priors(priors, path, compress=compress)


def load_priors(path: Union[Path, str]) -> Dict[int, LayerPriors]:
    """
    Load priors from JSON file (supports both plain and gzipped).

    Args:
        path: Path to JSON or JSON.gz file

    Returns:
        Dict of layer_id -> LayerPriors

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(path)

    # Try gzipped first if .gz extension, otherwise try plain
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    elif path.with_suffix(path.suffix + '.gz').exists():
        # Auto-detect gzipped version
        with gzip.open(path.with_suffix(path.suffix + '.gz'), 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    # Version check - support both v1 and v2
    version = data.get('v', data.get('version', 0))
    if version < 1:
        raise ValueError(f"Unsupported priors version: {version}")

    # Support both compact ('p') and verbose ('priors') keys
    priors_data = data.get('p', data.get('priors', {}))

    priors = {}
    for layer_id_str, prior_dict in priors_data.items():
        layer_id = int(layer_id_str)
        priors[layer_id] = LayerPriors.from_dict(prior_dict)

    return priors


def apply_priors_to_state(
    state: 'ControlState',
    prior: LayerPriors,
) -> None:
    """
    Apply loaded priors to a control state (warm start).

    This modifies the state in place to incorporate learned priors.

    Args:
        state: ControlState to modify
        prior: LayerPriors to apply
    """
    # Apply backoff bias (reduce initial harm_backoff by learned amount)
    state.harm_backoff = max(0.1, 1.0 - prior.backoff_bias)

    # Apply preferred mode (warm-start mode selection)
    if prior.preferred_mode:
        state.active_mode = prior.preferred_mode

    # Apply mode scores (warm-start mode selection)
    if prior.mode_scores:
        state.mode_scores = prior.mode_scores.copy()

    # Store prior reference for policy functions
    # (fragility, scale_cap, abstain_threshold applied in compute_lens_scale)
