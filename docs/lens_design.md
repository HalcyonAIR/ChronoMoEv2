# Lens Design

The lens is the intervention surface for ChronoMoE governance.

## Phase 1: Identity

In Phase 1, the lens is a **no-op**. It exists only to establish the interface.

```python
class IdentityLens(ChronoLens):
    def forward(self, x, state, layer_id):
        return x  # No transformation
```

This ensures:
- Model outputs are unchanged
- Interface is stable before Phase 2
- Integration points are tested

## Phase 2: Geometry Warping

Phase 2 will parameterize the lens using pressure/heat signals.

The lens transforms router input geometry `x` → `x'` to influence routing decisions without modifying the router itself.

Proposed interface:

```python
@dataclass(frozen=True)
class LensState:
    step: int
    mode: str  # "TRAIN" or "INFER"

    # Control signals
    pressure: float = 0.0      # Global pressure (0-1)
    heat: float = 0.0          # Temperature for exploration
    forgetting: float = 0.0    # Basin forgetting rate

    # Per-layer metrics from Phase 1
    layer_metrics: Optional[Dict[int, Dict[str, float]]] = None
```

## Design Constraints

1. **Cheap**: O(B*T*D) with minimal overhead
2. **Per-layer**: `layer_id` provided for layer-specific behavior
3. **Stateless**: All state in `LensState`, lens itself is pure function
4. **Reversible**: Warp should be invertible for analysis

## Future: Pressure Controller

The pressure controller will:
1. Read Phase 1 metrics (Neff, Top2, dead counts)
2. Compute global pressure signal
3. Pass to lens via `LensState`
4. Lens applies geometry warp to redistribute routing

This creates a closed loop: observe → compute → intervene → observe.
