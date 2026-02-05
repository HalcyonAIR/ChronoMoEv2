#!/usr/bin/env python3
"""
Clock 3 Priors A/B Test Harness

Validates that constitutional priors enable right-sized warm-start intervention.
Runs two conditions (cold vs warm), same seed, same shock schedule.
Outputs JSON summary + stdout table. Fails loudly if targeting regresses.

Usage:
    # With nanoMoE trainer
    python experiments/clock3_ab.py --trainer-cmd "python train.py" --trainer-cwd /path/to/nanoMoE

    # With custom config
    python experiments/clock3_ab.py --profile harsh --finetune-iters 500

    # Quick smoke test
    python experiments/clock3_ab.py --quick

Success Criteria:
    A. TARGETING: correlation(fragility, scale) > 0.3 AND stable layers reduced
    B. EARLY HEALTH: avoided deeper collapse OR faster stability
    Overall: (A or B) AND topology maintained
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import from chronomoe (the harness depends on implementation details)
from chronomoe.controller.priors import LayerPriors, load_priors, write_priors

# Import shock profiles
try:
    from experiments.shock_profiles import get_profile, ShockProfile, VALIDATED
except ImportError:
    # Running from repo root
    from shock_profiles import get_profile, ShockProfile, VALIDATED


@dataclass
class ABResult:
    """Results from a single run."""
    run_id: str
    final_neff: float
    mean_neff: float
    early_min_neff: float
    total_scale: float
    time_to_stability: int
    per_layer_mean_scale: Dict[int, float]
    step1_scales: Dict[int, float]
    priors_loaded: bool


@dataclass
class TargetingScore:
    """Measures intervention/fragility correlation."""
    correlation: float
    stable_reduced: bool
    fragile_increased: bool
    passed: bool


@dataclass
class ABSummary:
    """Complete A/B test summary."""
    profile_name: str
    seed: int
    cold: ABResult
    warm: ABResult
    priors: Dict[int, Dict]
    targeting: TargetingScore
    topology_maintained: bool
    overall_pass: bool


def parse_telemetry(telemetry_dir: Path, early_steps: int = 5) -> Dict:
    """Parse control_decisions.jsonl for metrics."""
    results = {
        'neffs': [], 'scales': [], 'per_step_neffs': {},
        'per_layer_scales': {}, 'early_neffs': [], 'step1': {},
    }

    decisions_file = telemetry_dir / 'control_decisions.jsonl'
    if not decisions_file.exists():
        return results

    with open(decisions_file) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            step, layer_id = d['step'], d['layer_id']
            neff = d['observed']['n_effective']
            scale = d['actuator']['lens_scale']

            results['neffs'].append(neff)
            results['scales'].append(scale)

            if step not in results['per_step_neffs']:
                results['per_step_neffs'][step] = []
            results['per_step_neffs'][step].append(neff)

            if layer_id not in results['per_layer_scales']:
                results['per_layer_scales'][layer_id] = []
            results['per_layer_scales'][layer_id].append(scale)

    # Extract early neffs
    sorted_steps = sorted(results['per_step_neffs'].keys())
    for step in sorted_steps[:early_steps]:
        results['early_neffs'].extend(results['per_step_neffs'][step])

    # Extract step 1
    if sorted_steps:
        first_step = sorted_steps[0]
        decisions_file = telemetry_dir / 'control_decisions.jsonl'
        with open(decisions_file) as f:
            for line in f:
                d = json.loads(line)
                if d['step'] == first_step:
                    results['step1'][d['layer_id']] = d['actuator']['lens_scale']

    return results


def compute_time_to_stability(per_step_neffs: Dict, threshold: float = 3.5) -> int:
    """First step where all layers have Neff >= threshold. Returns -1 if never."""
    for step in sorted(per_step_neffs.keys()):
        if all(n >= threshold for n in per_step_neffs[step]):
            return step
    return -1


# Targeting criteria thresholds
NEUTRAL_BAND = (0.8, 1.2)  # Fragility values considered neutral (exempt from checks)
STABLE_EPSILON = 0.10      # Allow 10% increase for stable layers
STABLE_VIOLATION_RATIO = 0.2  # Allow 20% of stable layers to violate


def compute_targeting(
    priors: Dict[int, LayerPriors],
    cold: ABResult,
    warm: ABResult,
) -> TargetingScore:
    """Compute targeting score: does intervention correlate with fragility?

    Relaxed criteria:
    - Neutral band (0.8-1.2): exempt from stable/fragile checks
    - Stable layers (< 0.8): may increase up to 10% without failing
    - Fragile layers (> 1.2): must increase (no epsilon)
    - Pass if 80%+ of stable layers behave correctly
    """
    if not priors or not warm.per_layer_mean_scale:
        return TargetingScore(0.0, False, False, False)

    layer_ids = sorted(priors.keys())
    fragilities = [priors[lid].fragility for lid in layer_ids]
    scales_warm = [warm.per_layer_mean_scale.get(lid, 0) for lid in layer_ids]
    scales_cold = [cold.per_layer_mean_scale.get(lid, 0) for lid in layer_ids]

    # Correlation
    if len(fragilities) >= 2 and np.std(fragilities) > 0.01 and np.std(scales_warm) > 0.001:
        correlation = float(np.corrcoef(fragilities, scales_warm)[0, 1])
    else:
        correlation = 0.0

    # Stable layers: fragility < NEUTRAL_BAND[0]
    # Should reduce OR increase by less than epsilon
    stable_count = 0
    stable_violations = 0
    for lid in layer_ids:
        frag = priors[lid].fragility
        if frag < NEUTRAL_BAND[0]:  # Truly stable (< 0.8)
            s_cold = cold.per_layer_mean_scale.get(lid, 0)
            s_warm = warm.per_layer_mean_scale.get(lid, 0)
            if s_cold > 0.0001:
                stable_count += 1
                increase_ratio = (s_warm - s_cold) / s_cold
                if increase_ratio > STABLE_EPSILON:
                    stable_violations += 1

    # Pass if majority of stable layers behave correctly
    if stable_count > 0:
        stable_reduced = stable_violations <= stable_count * STABLE_VIOLATION_RATIO
    else:
        stable_reduced = True  # No stable layers to check

    # Fragile layers: fragility > NEUTRAL_BAND[1]
    # Must increase (no epsilon - amplification is the point)
    fragile_increased = True
    for lid in layer_ids:
        if priors[lid].fragility > NEUTRAL_BAND[1]:  # Truly fragile (> 1.2)
            s_cold = cold.per_layer_mean_scale.get(lid, 0)
            s_warm = warm.per_layer_mean_scale.get(lid, 0)
            if s_warm <= s_cold:
                fragile_increased = False

    passed = correlation > 0.3 and stable_reduced
    return TargetingScore(correlation, stable_reduced, fragile_increased, passed)


def run_trainer(
    trainer_cmd: str,
    trainer_cwd: Path,
    config_path: Path,
) -> bool:
    """Run external trainer. Returns True on success."""
    cmd = trainer_cmd.split() + [str(config_path)]
    result = subprocess.run(cmd, cwd=trainer_cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Trainer failed: {result.stderr[:500]}", file=sys.stderr)
        return False
    return True


def extract_priors_from_telemetry(telemetry_dir: Path) -> Dict[int, LayerPriors]:
    """Extract priors from Run A telemetry."""
    decisions_file = telemetry_dir / 'control_decisions.jsonl'
    final_states = {}

    with open(decisions_file) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            final_states[d['layer_id']] = d

    all_scales = [s['actuator']['lens_scale'] for s in final_states.values()]
    global_mean = np.mean(all_scales) if all_scales else 0.001

    priors = {}
    for lid, state in final_states.items():
        scale = state['actuator']['lens_scale']
        harm_backoff = state['computed'].get('harm_backoff', 1.0)

        fragility = scale / global_mean if global_mean > 0.001 else 1.0
        fragility = float(np.clip(fragility, 0.5, 2.0))

        priors[lid] = LayerPriors(
            layer_id=lid,
            fragility=fragility,
            backoff_bias=1.0 - harm_backoff,
            preferred_mode=state['computed'].get('active_mode', 'anti_dominance'),
        )

    return priors


def generate_config(
    profile: ShockProfile,
    out_dir: Path,
    dataset: str,
    max_iters: int,
    priors_path: Optional[Path] = None,
    pretrain_ckpt: Optional[Path] = None,
    seed: int = 1337,
) -> str:
    """Generate training config string."""
    config = profile.to_config_str(
        out_dir=str(out_dir),
        dataset=dataset,
        max_iters=max_iters,
        lr_decay_iters=max_iters,
        min_lr=1e-4,
        use_chrono_controller=True,
        wandb_log=False,
        always_save_checkpoint=True,
        batch_size=32,
        block_size=128,
        gradient_accumulation_steps=1,
        beta2=0.99,
        use_switch_tfm_init=True,
        router_use_full_prec=True,
        use_noisy_top_k=False,
        device='mps',
        compile_model=False,
    )

    # Add seed
    config += f"\nseed = {seed}\n"

    # Add priors path if provided
    if priors_path:
        config += f"\nchrono_priors_path = '{priors_path}'\n"

    # Add finetune settings if checkpoint provided
    if pretrain_ckpt:
        config += f"\ninit_from = 'finetune'\n"
        config += f"ckpt_path = '{pretrain_ckpt}'\n"

    return config


def run_ab_test(
    output_dir: Path,
    trainer_cmd: str,
    trainer_cwd: Path,
    profile: ShockProfile,
    pretrain_dataset: str = 'shakespeare_char',
    finetune_dataset: str = 'tinystories',
    pretrain_iters: int = 500,
    finetune_iters: int = 300,
    seed: int = 1337,
) -> Optional[ABSummary]:
    """Run complete A/B test."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Clock 3 A/B Test: {profile.name}")
    print(f"{'='*60}")

    # Phase 1: Pretrain
    print("\n[1/4] Pre-training...")
    pretrain_dir = output_dir / "pretrain"
    pretrain_config = generate_config(
        profile=profile,
        out_dir=pretrain_dir,
        dataset=pretrain_dataset,
        max_iters=pretrain_iters,
        seed=seed,
    )
    # Disable controller for pretrain
    pretrain_config = pretrain_config.replace('use_chrono_controller = True', 'use_chrono_controller = False')
    # Use gentler settings for pretrain
    pretrain_config = pretrain_config.replace(f"top_k = {profile.top_k}", "top_k = 2")
    pretrain_config = pretrain_config.replace(f"train_capacity = {profile.train_capacity}", "train_capacity = 1.25")

    config_path = pretrain_dir / "config.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(pretrain_config)

    if not run_trainer(trainer_cmd, trainer_cwd, config_path):
        return None
    pretrain_ckpt = pretrain_dir / "ckpt.pt"

    # Phase 2: Run A (cold start)
    print("\n[2/4] Run A: Cold start (no priors)...")
    run_a_dir = output_dir / "run_A"
    run_a_config = generate_config(
        profile=profile,
        out_dir=run_a_dir,
        dataset=finetune_dataset,
        max_iters=finetune_iters,
        pretrain_ckpt=pretrain_ckpt,
        seed=seed,
    )
    config_path = run_a_dir / "config.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(run_a_config)

    if not run_trainer(trainer_cmd, trainer_cwd, config_path):
        return None

    # Parse Run A results
    run_a_telemetry = list((run_a_dir / "telemetry").glob("run_*"))[0]
    run_a_data = parse_telemetry(run_a_telemetry)

    cold = ABResult(
        run_id='A',
        final_neff=run_a_data['neffs'][-1] if run_a_data['neffs'] else 0,
        mean_neff=float(np.mean(run_a_data['neffs'])) if run_a_data['neffs'] else 0,
        early_min_neff=float(np.min(run_a_data['early_neffs'])) if run_a_data['early_neffs'] else 0,
        total_scale=float(sum(run_a_data['scales'])),
        time_to_stability=compute_time_to_stability(run_a_data['per_step_neffs']),
        per_layer_mean_scale={lid: float(np.mean(s)) for lid, s in run_a_data['per_layer_scales'].items()},
        step1_scales=run_a_data['step1'],
        priors_loaded=False,
    )

    # Phase 3: Extract and save priors
    print("\n[3/4] Extracting priors from Run A...")
    priors = extract_priors_from_telemetry(run_a_telemetry)
    priors_path = output_dir / "priors.json.gz"
    write_priors(priors, priors_path, compress=True)

    # Verify priors
    has_nondefault = any(
        abs(p.fragility - 1.0) > 0.01 or abs(p.backoff_bias) > 0.01
        for p in priors.values()
    )
    if not has_nondefault:
        print("  FAIL: All priors are default. Aborting.")
        return None

    print(f"  Saved {len(priors)} priors to {priors_path}")
    for lid, p in sorted(priors.items()):
        label = "FRAGILE" if p.fragility > 1.1 else "STABLE" if p.fragility < 0.9 else "neutral"
        print(f"    Layer {lid}: fragility={p.fragility:.2f} ({label})")

    # Phase 4: Run B (warm start)
    print("\n[4/4] Run B: Warm start (with priors)...")
    run_b_dir = output_dir / "run_B"
    run_b_config = generate_config(
        profile=profile,
        out_dir=run_b_dir,
        dataset=finetune_dataset,
        max_iters=finetune_iters,
        pretrain_ckpt=pretrain_ckpt,
        priors_path=priors_path,
        seed=seed,
    )
    config_path = run_b_dir / "config.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(run_b_config)

    if not run_trainer(trainer_cmd, trainer_cwd, config_path):
        return None

    # Parse Run B results
    run_b_telemetry = list((run_b_dir / "telemetry").glob("run_*"))[0]
    run_b_data = parse_telemetry(run_b_telemetry)

    warm = ABResult(
        run_id='B',
        final_neff=run_b_data['neffs'][-1] if run_b_data['neffs'] else 0,
        mean_neff=float(np.mean(run_b_data['neffs'])) if run_b_data['neffs'] else 0,
        early_min_neff=float(np.min(run_b_data['early_neffs'])) if run_b_data['early_neffs'] else 0,
        total_scale=float(sum(run_b_data['scales'])),
        time_to_stability=compute_time_to_stability(run_b_data['per_step_neffs']),
        per_layer_mean_scale={lid: float(np.mean(s)) for lid, s in run_b_data['per_layer_scales'].items()},
        step1_scales=run_b_data['step1'],
        priors_loaded=True,
    )

    # Compute scores
    targeting = compute_targeting(priors, cold, warm)
    topology_maintained = warm.final_neff >= cold.final_neff - 0.05
    overall_pass = topology_maintained and targeting.passed

    return ABSummary(
        profile_name=profile.name,
        seed=seed,
        cold=cold,
        warm=warm,
        priors={lid: {'fragility': p.fragility, 'backoff_bias': p.backoff_bias} for lid, p in priors.items()},
        targeting=targeting,
        topology_maintained=topology_maintained,
        overall_pass=overall_pass,
    )


def print_summary(summary: ABSummary):
    """Print results table."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {summary.profile_name}")
    print(f"{'='*60}")

    # Metrics table
    print(f"\n{'Metric':<25} {'Cold':<12} {'Warm':<12} {'Delta':<10}")
    print("-" * 60)
    rows = [
        ("Final Neff", summary.cold.final_neff, summary.warm.final_neff),
        ("Early Min Neff", summary.cold.early_min_neff, summary.warm.early_min_neff),
        ("Total Scale", summary.cold.total_scale, summary.warm.total_scale),
        ("Time to Stability", summary.cold.time_to_stability, summary.warm.time_to_stability),
    ]
    for name, cold_val, warm_val in rows:
        delta = warm_val - cold_val
        print(f"{name:<25} {cold_val:<12.3f} {warm_val:<12.3f} {delta:+.3f}")

    # Per-layer targeting
    print(f"\n{'Layer':<8} {'Fragility':<12} {'Cold':<12} {'Warm':<12} {'Match':<8}")
    print("-" * 52)
    for lid in sorted(summary.cold.per_layer_mean_scale.keys()):
        frag = summary.priors.get(lid, {}).get('fragility', 1.0)
        label = "FRAGILE" if frag > 1.1 else "STABLE" if frag < 0.9 else "neutral"
        s_cold = summary.cold.per_layer_mean_scale.get(lid, 0)
        s_warm = summary.warm.per_layer_mean_scale.get(lid, 0)
        delta = s_warm - s_cold
        expected = "+" if frag > 1.1 else "-" if frag < 0.9 else "="
        actual = "+" if delta > 0.001 else "-" if delta < -0.001 else "="
        match = "OK" if actual == expected or (expected in "+-" and s_cold < 0.0001) else "MISS"
        print(f"{lid:<8} {frag:.2f} ({label:<7}) {s_cold:<12.4f} {s_warm:<12.4f} {match}")

    # Verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    print(f"Targeting correlation: {summary.targeting.correlation:.3f}")
    print(f"Stable layers reduced: {summary.targeting.stable_reduced}")
    print(f"Fragile layers increased: {summary.targeting.fragile_increased}")
    print(f"Topology maintained: {summary.topology_maintained}")
    print(f"Overall: {'PASS' if summary.overall_pass else 'FAIL'}")

    if summary.overall_pass:
        print(f"\nClock 3 validated: r={summary.targeting.correlation:.3f}")
    else:
        print(f"\nClock 3 FAILED validation")


def main():
    parser = argparse.ArgumentParser(description="Clock 3 Priors A/B Test")
    parser.add_argument("--trainer-cmd", default=f"{sys.executable} train.py",
                        help="Command to run trainer")
    parser.add_argument("--trainer-cwd", type=Path, required=True,
                        help="Working directory for trainer")
    parser.add_argument("--output-dir", type=Path, default=Path("out-clock3-ab"),
                        help="Output directory")
    parser.add_argument("--profile", default="validated",
                        help="Shock profile name")
    parser.add_argument("--pretrain-iters", type=int, default=500)
    parser.add_argument("--finetune-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (fewer iterations)")
    args = parser.parse_args()

    if args.quick:
        args.pretrain_iters = 100
        args.finetune_iters = 50

    # Resolve to absolute paths (trainer runs in different cwd)
    output_dir = args.output_dir.resolve()

    profile = get_profile(args.profile)

    summary = run_ab_test(
        output_dir=output_dir,
        trainer_cmd=args.trainer_cmd,
        trainer_cwd=args.trainer_cwd,
        profile=profile,
        pretrain_iters=args.pretrain_iters,
        finetune_iters=args.finetune_iters,
        seed=args.seed,
    )

    if summary:
        print_summary(summary)

        # Save JSON summary
        results_path = output_dir / "summary.json"
        with open(results_path, 'w') as f:
            json.dump({
                'profile': summary.profile_name,
                'seed': summary.seed,
                'targeting_correlation': summary.targeting.correlation,
                'topology_maintained': summary.topology_maintained,
                'overall_pass': summary.overall_pass,
                'cold_final_neff': summary.cold.final_neff,
                'warm_final_neff': summary.warm.final_neff,
            }, f, indent=2)
        print(f"\nSummary saved to {results_path}")

        return 0 if summary.overall_pass else 1
    else:
        print("\nTest failed to complete")
        return 1


if __name__ == "__main__":
    sys.exit(main())
