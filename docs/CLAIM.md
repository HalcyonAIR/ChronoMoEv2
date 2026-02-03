# ChronoMoE: Validated Claim

**Version:** 0.2.6
**Date:** 2026-02-03
**Status:** Validated under replication

---

## The Claim

ChronoMoE provides a **protective envelope** for Mixture of Experts routing topology during training.

Specifically:

1. **Engages under measurable stress.** When routing pressure is detected (Neff below threshold, Top2 concentration rising), the controller activates. Validated at 73-81% engagement under capacity stress.

2. **Abstains when no stress.** When topology is healthy (pressure ≈ 0), the controller explicitly abstains. Validated at 100% abstention during unstressed domain shift.

3. **Improves mean and final Neff.** Under capacity stress, controller ON produces higher effective expert utilization than OFF. Validated at +0.07 to +0.13 Final Neff improvement across stress configurations.

4. **Does not introduce dead experts.** Zero dead expert events in all validated runs, both ON and OFF conditions.

5. **Validated under capacity stress without synthetic bias.** Stress induced by top_k reduction (2→1) and capacity factor tightening (1.25→1.0) during domain-shift fine-tuning. No artificial router bias injection.

6. **Replicated across stress configurations.** Same controller, same logging, different stressor intensity. Both show improvement.

---

## What This Is Not

- Not a claim about optimal steering direction (Phase 2.5 identified W-basis limitation)
- Not a claim about extreme severity (controller correctly abstains)
- Not a claim about production readiness (research validation only)

---

## Evidence Summary

| Experiment | Stress | Engagement | Final Neff Δ | Dead Events |
|------------|--------|------------|--------------|-------------|
| Domain shift (unstressed) | None | 0% | 0 | 0 |
| Domain shift + top_k=1, cap=1.25 | Moderate | 73% | +0.13 | 0 |
| Domain shift + top_k=1, cap=1.0 | Higher | 81% | +0.07 | 0 |

---

## Reproduction

```bash
cd nanoMoE
uv pip install ../ChronoMoEv2

# Original stressed run
python experiments/domain_shift_stressed.py

# Replication with tighter capacity
python experiments/domain_shift_replication.py
```

---

## Architecture

```
Clock 1 (Fast):   MoE forward pass, routing decisions
Clock 2 (Medium): Pressure sensing, lens warping, harm guard
Clock 3 (Slow):   [Deferred] Policy memory, identity persistence
```

The protective envelope operates at Clock 2: it observes routing topology, computes pressure, and gates a low-rank lens transformation on router inputs. The harm guard backs off if intervention increases Top2 concentration.

---

## Limitations & Future Work

**Known gap:** Steering direction selection. All current modes are W-basis variants. When the router weight basis is wrong for a layer, steering fails regardless of magnitude control. This is documented, not fixed.

**Phase 3 path:** Orthogonal bases (activation PCA, gradient-based directions, learned adapters) would address the steering gap. With the envelope validated, Phase 3 is optional enhancement.

---

*HalcyonAI Research, 2026*
