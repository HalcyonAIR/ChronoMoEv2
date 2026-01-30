# ChronoMoE Architecture

## Overview

ChronoMoE is a **governance framework** for Mixture-of-Experts routing.

```
┌─────────────────────────────────────────────────────────────┐
│                        Training Loop                        │
├─────────────────────────────────────────────────────────────┤
│  Input → [Lens] → Router → Experts → Output                │
│            │         │                                      │
│            │         └─── RoutingEvent ───┐                │
│            │                              ↓                 │
│            │                      ┌───────────────┐        │
│            │                      │   Telemetry   │        │
│            │                      │   (Phase 1)   │        │
│            │                      └───────┬───────┘        │
│            │                              ↓                 │
│            │                      ┌───────────────┐        │
│            │                      │  Snapshots    │        │
│            │                      │  + Alerts     │        │
│            │                      └───────┬───────┘        │
│            │                              ↓                 │
│            │                      ┌───────────────┐        │
│            └──── LensState ◄──────│  Controller   │        │
│                  (Phase 2+)       │  (Phase 2+)   │        │
│                                   └───────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Phases

### Phase 1: Telemetry (Current)
- **RoutingEvent**: Log actual dispatch per (step, layer)
- **Metrics**: Entropy, Neff, Top2, dead counts
- **Snapshots**: Periodic summaries at eval intervals
- **Alerts**: PRE_COLLAPSE, COLLAPSE_CONFIRMED

### Phase 2: Governance (Planned)
- **Pressure Controller**: Compute intervention signals from metrics
- **Lens Parameterization**: Warp geometry to redistribute routing
- **Basin Tracking**: Monitor attraction basins in routing space

### Phase 3: Lifecycle (Future)
- **Expert Spawning**: Add experts when needed
- **Expert Pruning**: Remove dead experts safely
- **Rules/Maniac**: Governance policy system

## Key Principles

1. **Observe Before Govern**: Phase 1 must be stable before Phase 2
2. **Strict Semantics**: Dead = zero share, no thresholds
3. **Persistence Required**: Alerts need N consecutive confirmations
4. **Derived Metrics**: token_count from actual dispatch, not B*T
5. **Regression Contract**: Tests verify metric semantics

## Integration Points

ChronoMoE integrates at two points:

1. **Router Input**: Lens transformation (x → x')
2. **Router Output**: Event logging (used_capacity → RoutingEvent)

This minimal surface keeps coupling low while enabling full observability.
