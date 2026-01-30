# ChronoMoE

Temporal Governance and Topology Telemetry for Mixture-of-Experts models.

## Overview

ChronoMoE provides observability and governance infrastructure for MoE routing topology. It makes routing topology **observable, measurable, and controllable** - enabling detection and prevention of expert collapse during training.

**Phase 1**: Telemetry - observe and alert
**Phase 2**: Governance - pressure controller, lens parameterization (current)

## Installation

```bash
pip install git+https://github.com/HalcyonAIR/ChronoMoEv2.git
```

Or install from source:

```bash
git clone https://github.com/HalcyonAIR/ChronoMoEv2.git
cd ChronoMoEv2
pip install -e .
```

## Quick Start (Phase 1 - Telemetry)

```python
from chronomoe import RoutingEvent, SystemSnapshot, TelemetryWriter
from chronomoe.metrics import compute_entropy, compute_n_effective

# Create telemetry writer
writer = TelemetryWriter(run_id="my_run", output_dir="outputs")

# Log routing events from your MoE forward pass
event = RoutingEvent.from_router_output(
    run_id="my_run",
    step=100,
    mode="TRAIN",
    layer_id=0,
    used_capacity=used_capacity_tensor,  # [n_experts]
    n_experts=8,
    top_k=2,
)

# Create periodic snapshots with alert detection
snapshot = SystemSnapshot.from_events(
    events=events,
    step=100,
    train_loss=2.5,
    val_loss=2.8,
    run_id="my_run",
    model_name="MyMoE",
    n_layers=4,
    alert_history=alert_history,
)

# Check for collapse alerts
if snapshot.alerts:
    print("WARNING:", snapshot.alerts)
```

## Quick Start (Phase 2 - Governance)

```python
from chronomoe import ChronoLens
from chronomoe.controller import Controller, ControlConfig

# Initialize controller
controller = Controller(
    n_layers=4,
    n_experts_per_layer=[8, 8, 8, 8],
    config=ControlConfig(),
)
controller.initialize(run_id="my_run")

# Create lenses for each MoE layer
lenses = {
    layer_id: ChronoLens(d_model=512, rank=8, layer_id=layer_id)
    for layer_id in range(4)
}

# At each eval checkpoint:
decisions = controller.update(snapshot, lenses)

# Controller automatically:
# 1. Computes topology debt from metrics
# 2. Updates pressure/heat/forgetting signals
# 3. Sets lens scale to gate geometry warp
# 4. Logs all decisions to control_decisions.jsonl
```

## Semantic Definitions

### token_count
**Derived from `sum(expert_token_counts)`**, NOT from `B*T`.

This ensures consistency with actual dispatch after capacity constraints. Tokens dropped by capacity limits are not counted.

### Dead Experts (strict)
An expert is **dead** if `share == 0` over the snapshot window.

This is a strict definition - we do NOT use a threshold like `share < 0.01`. For early warning about "nearly dead" experts, use the separate `nearly_dead_count` field.

### Alert Thresholds

| Alert | Condition | Persistence |
|-------|-----------|-------------|
| **PRE_COLLAPSE** | `Neff < 0.6 * n_experts` OR `Top2 > 0.75` | 3 consecutive snapshots |
| **COLLAPSE_CONFIRMED** | `dead_experts >= 1` AND `Top2 > 0.85` | 5 consecutive snapshots |

## Key Metrics

- **Neff** (effective experts): `exp(entropy)` - intuitive measure of "how many experts are really being used"
- **Top2Share**: Sum of two largest utilization shares - early warning for binary collapse
- **Entropy**: Shannon entropy over expert utilization shares
- **Pressure**: EMA of topology debt - drives lens intervention
- **Heat**: Exploration parameter, proportional to pressure

## Package Structure

```
chronomoe/
├── __init__.py      # Public API
├── events.py        # RoutingEvent, ExpertState dataclasses
├── metrics.py       # Pure metric functions (deterministic)
├── snapshots.py     # SystemSnapshot, LayerSnapshot, alert logic
├── io.py            # JSONL writers, manifest generation
├── lens.py          # ChronoLens (low-rank warp), IdentityLens
└── controller/
    ├── state.py     # ControlState, ControlConfig
    ├── policy.py    # Debt computation, state updates
    ├── decisions.py # ControlDecision logging
    └── hooks.py     # Controller integration class
```

## Phase 2 Control Loop

```
observe → compute debt → update pressure → gate lens → observe
   │                                           │
   └───────────── closed loop ────────────────┘
```

The lens applies a low-rank residual warp: `x' = x + s * (x @ V) @ U`

Where `s` (scale) is gated by pressure from the controller. When topology is healthy, `s ≈ 0` and the lens is near-identity. When collapse is detected, `s` increases to redistribute routing.

## Testing

```bash
# Run all tests
python tests/test_phase1_regression.py
python tests/test_phase2_controller.py

# Or with pytest
pytest tests/ -v
```

## Roadmap

- **Phase 1** ✅: Telemetry - observe and alert
- **Phase 2** ✅: Governance - pressure controller, lens parameterization
- **Phase 3**: Lifecycle - expert spawning/pruning, basin tracking

## License

MIT
