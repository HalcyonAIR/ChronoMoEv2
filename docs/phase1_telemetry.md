# Phase 1: Telemetry

Phase 1 is pure instrumentation: log everything, compute metrics, raise alerts, but don't intervene.

## Design Philosophy

ChronoMoE Phase 1 follows the principle of **observe before govern**. We cannot fix what we cannot see.

The system provides:
1. **Atomic logging** - Every routing decision recorded
2. **Derived metrics** - Entropy, Neff, Top2, dead counts
3. **Persistent alerts** - No false positives, only confirmed collapse
4. **Regression contract** - Tests verify metric semantics

## Components

### RoutingEvent

Atomic log record per (step, layer). Records **actual dispatch**, not router preferences.

Key fields:
- `expert_token_counts`: Ground truth dispatch (REQUIRED)
- `token_count`: Derived from `sum(expert_token_counts)` for consistency
- `active_experts`: Indices where count > 0

### SystemSnapshot

Periodic summary at evaluation cadence. This is the **regression contract**.

Contains:
- Per-layer topology metrics
- Alerts (with persistence tracking)
- Loss correlation signals

### Alert Logic

Alerts require **persistence** to avoid noise:

```
PRE_COLLAPSE: 3 consecutive snapshots
COLLAPSE_CONFIRMED: 5 consecutive snapshots
```

This prevents false positives from transient routing fluctuations.

## Integration

See `examples/nanomoe_integration/` for how to integrate ChronoMoE with a training loop.

Typical pattern:

```python
# In your training loop
if step % eval_interval == 0:
    # Flush accumulated events
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

    # Log alerts
    if snapshot.alerts:
        for alert in snapshot.alerts:
            logger.warning(alert)

    # Clear buffer
    manager.routing_events.clear()
```
