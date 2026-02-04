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

- **Targeting**: `correlation(fragility, scale) > 0.3` AND stable layers reduced
- **Topology**: `warm.final_neff >= cold.final_neff - 0.05`

### Shock Profiles

| Profile | Description |
|---------|-------------|
| `gentle` | Minimal stress baseline |
| `moderate` | Some pressure, creates differentiation |
| `harsh` | Strong pressure, tests limits |
| `differentiated_8l` | 8 MoE layers, validated r=0.995 |
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
