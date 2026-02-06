# ChronoMoE Experiments

Research tools that live outside the package import path. These can depend on
implementation details and aren't part of the public API.

## nanoMoE Graduation Summary

**Status:** nanoMoE has reached its usefulness ceiling. Moving to swiss-ai/MoE.

### What We Validated

Clock 3 constitutional priors work in nanoMoE:
- **Targeting correlation r=0.94-0.995** across 8 MoE layers
- Fragile layers get +50-160% more intervention in warm start
- Stable layers get 30-50% less intervention
- Topology preserved (Neff unchanged)

### Why We're Graduating

nanoMoE failures are "too polite" - clean and separable rather than ambiguous:

1. **DIFFERENTIATED_8L (4 experts, top-1):** Validated r=0.94 targeting, but edge
   cases are clean. Borderline layers (fragility ~0.65) trip binary rules without
   genuine ambiguity.

2. **NANOMOE_PLUS (16 experts, top-2):** Counterintuitively, more experts created
   MORE stability, not messiness:
   - Neff jumped from ~4 to ~13 (nearly all experts active)
   - Controller rarely/never triggered (no intervention needed)
   - Perfect bimodal fragility (all 0.5 or all 2.0, nothing between)
   - Full test aborted: "All priors are default" (no differentiation)

**Key insight:** More experts + top-2 routing = more routing options = more stability.
The "messy routing" we wanted requires genuine capacity pressure where tokens compete
for limited expert slots, not abundant capacity where everyone fits.

### Next Testbed: swiss-ai/MoE

Moving to swiss-ai/MoE because:
- GPT-2 style transformer with MoE swapped into FFN (the shape we want)
- Designed for controller/expert surgery (not fighting a framework)
- Can create genuine capacity pressure at realistic scale
- Same A/B harness and ShockProfile discipline for clean attribution

Reference: HuggingFace SwitchTransformers for routing semantics and standard
router losses, but not as primary codebase (ecosystems fight back).

---

## Clock 3 A/B Test

Validates that constitutional priors enable right-sized warm-start intervention.

```bash
# Run with nanoMoE trainer
python experiments/clock3_ab.py \
    --trainer-cmd "python train.py" \
    --trainer-cwd /path/to/nanoMoE \
    --profile validated

# Quick smoke test
python experiments/clock3_ab.py \
    --trainer-cwd /path/to/nanoMoE \
    --quick

# With custom profile
python experiments/clock3_ab.py \
    --trainer-cwd /path/to/nanoMoE \
    --profile harsh \
    --finetune-iters 500
```

### Success Criteria

- **Targeting**: `correlation(fragility, scale) > 0.3` AND stable layers behave correctly
- **Topology**: `warm.final_neff >= cold.final_neff - 0.05`

Fragility thresholds:
- **Stable**: fragility < 0.8 (should reduce, or increase by less than 10%)
- **Neutral**: 0.8 <= fragility <= 1.2 (exempt from checks)
- **Fragile**: fragility > 1.2 (must increase)

### Shock Profiles

| Profile | Description |
|---------|-------------|
| `gentle` | Minimal stress baseline |
| `moderate` | Some pressure, creates differentiation |
| `harsh` | Strong pressure, tests limits |
| `differentiated_8l` | 8 MoE layers, validated r=0.995 |
| `nanomoe_plus` | 16 experts, top-2 (too stable - graduation trigger) |
| `validated` | Alias for differentiated_8l |

### Output

- `summary.json`: Machine-readable results
- `priors.json.gz`: Extracted priors from cold run
- `run_A/`, `run_B/`: Full telemetry

## Design Principles

1. **Named shock profiles**: "Uniform brutality causes uniform collapse.
   Structured shocks create differentiated failure."

2. **Trainer-agnostic**: Plug in any trainer that reads a config file.

3. **Fail loudly**: If targeting regresses, the test fails.
