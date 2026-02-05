# ChronoMoE Experiments

Research tools that live outside the package import path. These can depend on
implementation details and aren't part of the public API.

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
| `nanomoe_plus` | 16 experts, top-2, messy routing (graduation testbed) |
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
