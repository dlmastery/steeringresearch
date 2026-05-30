---
name: steering-experiment
description: >
  Use when running a single steering experiment on the autoresearch ladder.
  Enforces the 7-step ritual (Diagnose, Cite, Hypothesize, Predict, Execute,
  Analyse, Checkpoint), the one-config-change rule, the pre-registration gate,
  and EXPERIMENT_LEDGER.md row discipline. This is the steering instantiation
  of the meta autoresearch-experiment process.
---

# Skill — steering-experiment

This is the steering instantiation of [meta-skills/autoresearch-experiment/SKILL.md](../../meta-skills/autoresearch-experiment/SKILL.md).

Read that file first for the general process. This file adds steering-specific
constraints and checkpoints that override or extend the meta-skill.

---

## Steering-specific notes

### The one-config-change rule (steering version)

A single "config change" in steering means changing exactly ONE of the 12 axes:
- A1 (site/layer): change the injection layer only
- A2 (direction): change which vector or extraction method only
- A3 (coefficient): change alpha only
- A4 (operation): change add vs rotate vs project only
- A5 (condition): change the gating condition only
- A6 (span): change which tokens receive the intervention only
- A7 (how derived): change the source dataset or estimator only
- A8 (geometry): change the path (chord vs geodesic) only
- A9 (metric): change the space (Euclidean vs spherical vs hyperbolic) only
- A10 (identifiability): change the gauge representative only
- A11 (dynamics): change one-shot vs trajectory only
- A12 (basis): change dense vs sparse basis only

Changing two axes simultaneously is FORBIDDEN — it makes the marginal effect
unattributable. If two axes must change together (e.g., spherical metric requires
rotation operation), pre-register the joint change as a single named intervention
and document the coupling.

### Pre-run gate (steering-specific fields)

The pre-run reasoning entry in `autoresearch_results/reasoning_annotations.json`
MUST include:
- Which of the 12 axes is being perturbed (axis number + name)
- The baseline configuration (champion config from `autoresearch_results/best_config.json`)
- The cited paper in full format: `Author1, ..., YEAR VENUE 'Title' (arXiv:XXXX.XXXXX)`
- The predicted composite range AND at least one sub-metric with threshold
- The composite formula fingerprint (SHA-256 of `src/steering/eval.py:COMPOSITE_FORMULA`)

### Geometry leading indicators (always log)

Every experiment row in EXPERIMENT_LEDGER.md must include:
- `delta_norm`: off-shell displacement delta(||h||) averaged over tokens/prompts
- `eff_rank_drop`: effective-rank drop at injection layer vs unsteered
- `norm_budget`: cumulative ||delta h||/||h|| (N5 budget fraction)
- `part_ratio`: participation ratio at injection layer (N3)

These are not optional. A missing geometry column is a protocol violation.

### Safety gate (mandatory for stacking/guard experiments)

Any experiment that stacks vectors or modifies a safety-relevant behavior MUST:
1. Measure JailbreakBench Compliance Rate (CR_jailbreak).
2. A CR_jailbreak > 0% is an automatic DISCARD regardless of behavior score.
3. Log the 20-vector universal attack probe result at Rung 4.

### Activation cache discipline

Never recompute activations inside a sweep loop. Validate the cache SHA-256
against `cache/activations/<model>/<dataset>/metadata.json` at the start of
every run. If the SHA does not match, stop and recompute before proceeding.

### Commit discipline

Commit the pre-run reasoning entry BEFORE launching the experiment.
Commit the post-run entry + EXPERIMENT_LEDGER.md row BEFORE launching the next.
Never let git status be dirty from experiment N when experiment N+1 launches.

---

## Quick checklist (copy into pre-run entry)

- [ ] Reading from `autoresearch_results/best_config.json` (not from memory)
- [ ] Exactly ONE of the 12 axes changes
- [ ] Cite real arXiv ID; mark [UNVERIFIED] if unsure
- [ ] Predicted composite range pre-registered BEFORE run
- [ ] Composite fingerprint matches `src/steering/eval.py`
- [ ] Geometry leading indicators configured to log
- [ ] Safety gate configured (if stacking/guard experiment)
- [ ] Activation cache SHA validated
- [ ] Pre-run entry committed to git BEFORE launch

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-experiment/SKILL.md`
- Vector extraction: `../steering-vector-extraction/SKILL.md`
- Eval bundle: `../steering-eval-bundle/SKILL.md`
- Ladder gates: `../steering-tiered-ladder/SKILL.md`
- Ledger schema: `../../EXPERIMENT_LEDGER.md`
- Hypothesis registry: `../../IDEA_TABLE.md`
