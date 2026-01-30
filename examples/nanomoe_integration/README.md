# nanoMoE Integration Example

This directory shows how to integrate ChronoMoE with a training loop.

## Overview

ChronoMoE requires two integration points:

1. **Router**: Apply lens before routing, log events after
2. **Training Loop**: Create snapshots at eval intervals

## Router Integration

```python
# In your Router.forward()
def forward(self, x):
    # Apply lens transformation
    x = manager.apply_lens(x, self.layer_id)

    # Normal routing
    logits = self.w_g(x)
    # ... rest of routing logic ...

    # Log routing event
    manager.add_routing_event(
        layer_id=self.layer_id,
        used_capacity=used_capacity,
        n_experts=self.n_exp,
        top_k=self.top_k,
    )

    return output
```

## Training Loop Integration

```python
from chronomoe import SystemSnapshot, TelemetryWriter, create_run_manifest

# Setup
writer = TelemetryWriter(run_id, output_dir)
writer.write_manifest(create_run_manifest(run_id, model_name, config))
alert_history = {}

# Training loop
for step in range(max_iters):
    # ... forward/backward pass ...

    if step % eval_interval == 0:
        # Flush events
        writer.flush_events(manager.routing_events)

        # Create snapshot
        snapshot = SystemSnapshot.from_events(
            events=manager.routing_events,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            run_id=run_id,
            model_name=model_name,
            n_layers=n_moe_layers,
            alert_history=alert_history,
        )

        writer.write_snapshot(snapshot)

        # Handle alerts
        for alert in snapshot.alerts:
            print(f"[ALERT] {alert}")

        # Clear buffer
        manager.routing_events.clear()
```

## Full Example

See the nanoMoE repository for a complete working integration:
- `nanoMoE/manager.py`: MOEManager with telemetry hooks
- `nanoMoE/model.py`: Router with lens and event logging
- `nanoMoE/train.py`: Training loop with snapshot creation
