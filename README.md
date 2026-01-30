# ChronoMoE

Temporal Governance and Topology Telemetry for Mixture-of-Experts models.

## Overview

ChronoMoE provides observability infrastructure for MoE routing topology. It makes routing topology **observable, measurable, and regressible** - enabling detection of expert collapse before it causes irreversible training degradation.

## Installation

```bash
pip install chronomoe
```

Or install from source:

```bash
git clone https://github.com/HalcyonAIR/ChronoMoEv2.git
cd ChronoMoEv2
pip install -e .
```

## Quick Start

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

## Package Structure

```
chronomoe/
├── __init__.py      # Public API
├── events.py        # RoutingEvent, ExpertState dataclasses
├── metrics.py       # Pure metric functions (deterministic)
├── snapshots.py     # SystemSnapshot, LayerSnapshot, alert logic
├── io.py            # JSONL writers, manifest generation
├── lens.py          # Identity lens (Phase 2: geometry warping)
└── controller/      # Phase 2+: pressure controller, lifecycle manager
```

## Testing

```bash
pytest tests/ -v
```

Or run directly:

```bash
python tests/test_phase1_regression.py
```

## Roadmap

- **Phase 1** (current): Telemetry - observe and alert
- **Phase 2**: Governance - pressure controller, lens parameterization
- **Phase 3**: Lifecycle - expert spawning/pruning, basin tracking

## License

MIT
