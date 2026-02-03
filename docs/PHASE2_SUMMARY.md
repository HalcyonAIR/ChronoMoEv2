# ChronoMoE Phase 2: Controller Validation Summary

**Status:** LOCKED
**Date:** 2026-02-03
**Validated by:** Severity sweep + Distillation experiment

---

## Architecture

Phase 2 implements a **damage-limitation controller** for MoE expert collapse:

```
Clock 1 (Fast):  Task MoE - normal forward pass, routing decisions
Clock 2 (Medium): Control - pressure sensing, lens warping, harm guard
Clock 3 (Slow):  [Deferred] - Policy memory, identity persistence
```

### Components

| Component | Purpose | Status |
|-----------|---------|--------|
| Topology Debt | Measure collapse severity (Neff, Top2, dead experts) | ✅ |
| Pressure | EMA of debt with hysteresis | ✅ |
| Pressure Cap | Prevent over-steering at high severity (cap=0.5) | ✅ |
| Lens Warp | Low-rank transformation of router input geometry | ✅ |
| Anti-dominance Steering | Push toward under-utilized experts | ✅ |
| Harm Guard | Closed-loop: back off if Top2 increases | ✅ |
| Explicit Abstain | First-class "no intervention" policy | ✅ |

---

## Validated Behavior

### Three Operating Regimes

| Regime | Bias Level | Controller Behavior | Outcome |
|--------|------------|---------------------|---------|
| **Mild** | 5.0 | Intervenes gently | Helps (Neff +0.09, recovers) |
| **Intermediate** | 10-15 | Intervenes but wrong direction | Hurts (more dead experts) |
| **Extreme** | 20.0 | Abstains (backoff → 0.1) | Neutral (stops hurting) |

### Distillation Test

| Metric | Controller ON | Controller OFF | Delta |
|--------|---------------|----------------|-------|
| Mean Neff | 3.53 | 3.34 | **+0.19** |
| Final Neff | 3.20 | 3.10 | **+0.09** |
| Dead Events | 0 | 0 | 0 |
| Abstain Rate | 0% | - | No harm detected |

**Verdict:** Distillation lives in **mild perturbation** regime. Controller helps.

---

## Key Findings

### What Works

1. **Magnitude control** - Pressure cap and harm guard prevent over-steering
2. **Closed-loop feedback** - Harm guard detects when intervention hurts
3. **Explicit abstain** - Controller can choose "do nothing" as policy
4. **Mild perturbation** - Controller helps during distillation and low-severity stress

### Known Gap

**Direction selection is missing.** The controller has one steering direction (anti-dominance). At intermediate severity (bias 10-15), this direction is wrong for some layers, causing harm even with reduced magnitude.

**This gap is isolated and deferred to Phase 2.5.**

---

## Configuration

```python
# chronomoe/controller/state.py

@dataclass
class ControlConfig:
    # Debt thresholds
    neff_threshold_ratio: float = 0.6
    top2_warning: float = 0.75
    top2_emergency: float = 0.90

    # Pressure dynamics
    pressure_ema_alpha: float = 0.2
    pressure_cap: float = 0.5  # Prevent over-steering

    # Lens gating
    lens_scale_max: float = 0.5
    lens_pressure_coeff: float = 1.0
    lens_warmup_steps: int = 100
    lens_warmup_scale: float = 0.02

    # Harm guard (closed-loop)
    harm_top2_threshold: float = 0.02
    harm_backoff_factor: float = 0.5
    harm_recovery_rate: float = 0.2

    # Explicit abstain
    abstain_backoff_threshold: float = 0.15
```

---

## Telemetry

Control decisions are logged to `telemetry/run_*/control_decisions.jsonl`:

```json
{
  "step": 100,
  "layer_id": 0,
  "observed": {"n_effective": 3.5, "top2_share": 0.65, "dead_expert_count": 0},
  "computed": {"pressure": 0.05, "harm_backoff": 1.0, "abstain": false},
  "actuator": {"lens_scale": 0.05}
}
```

---

## Phase 2.5: Directional Policy Selection (Deferred)

The next architectural step is giving Clock 2 multiple steering modes:

1. Anti-dominance push (current)
2. Entropy-increasing push
3. "Spread mass" vs "lift tail"
4. Explicit "do nothing"

The harm guard would then select which mode helps, rather than just reducing magnitude.

**This is only needed if future experiments land in the intermediate regime.**

---

## References

- Severity sweep: `nanoMoE/experiments/severity_sweep.py`
- Distillation test: `nanoMoE/experiments/distillation_test.py`
- Controller policy: `chronomoe/controller/policy.py`
- Control state: `chronomoe/controller/state.py`
