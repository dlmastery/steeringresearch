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

### L6 — Mandatory controls in every sweep

Every experiment that claims a directional effect must beat **two controls**,
not just the do-nothing (alpha=0) baseline:

**Control A — Matched-norm random-direction baseline:**
Extract a random unit vector of the same norm as the steering vector and apply
it with the same alpha, layer, and span as the experiment config. Report the
behavior score delta over this control (not over alpha=0). A real directional
effect must beat a random direction of the same magnitude.

> *Why:* a large alpha perturbs the residual stream regardless of direction,
> which can change output statistics in ways that inflate behavior scores
> measured by pattern-matching metrics. If the random-direction control scores
> nearly as high as the intended vector, the effect is not directional.

**Control B — Shuffled-label baseline:**
Extract the steering vector from the SAME contrast pairs but with positive and
negative labels swapped (shuffled). Run through the identical pipeline. The
behavior score on the shuffled-label vector must be significantly lower than
on the true vector (ideally near 0 or below chance). If it is not, the metric
is not measuring a real directional effect.

> *Why:* this is the activation-steering analogue of the shuffle test from the
> meta-process ([[meta-skills/autoresearch-shuffle-test]]). A metric that gives
> a high score for a semantically reversed vector is measuring an artifact
> (e.g., output length, fluency change, or lexical frequency) rather than
> concept presence.

**Reporting rule:** every EXPERIMENT_LEDGER.md row must include two delta
columns:
- `delta_vs_random_direction`: behavior efficacy − behavior efficacy of
  matched-norm random-direction control
- `delta_vs_shuffled_label`: behavior efficacy − behavior efficacy of
  shuffled-label control

A claim of a directional effect requires both deltas to be positive and
outside the seed noise band. A row where `delta_vs_random_direction ≤ 0` is
a DISCARD regardless of raw behavior score.

**New harness support:** `src/steering/controls.py:matched_norm_random` and
`src/steering/controls.py:shuffled_label_vector`.

---

### L10 — Extraction stability gate (before any downstream use)

A steering vector extracted from a small contrast set may be unstable: a
different random draw of the same N pairs can produce a substantially
different direction. Unstable directions make everything downstream noisy and
make the entire hill-climb arbitrary.

**Bootstrap-cosine stability protocol:**

1. Bootstrap the contrast set K=100 times (sample N pairs with replacement).
2. For each bootstrap, recompute the steering vector (DiffMean or PCA-top1).
3. Compute pairwise cosines between all bootstrap directions and the original
   direction. Report `bootstrap_cosine_mean` and `bootstrap_cosine_p5`
   (5th-percentile cosine, the stability floor).
4. **Gate:** if `bootstrap_cosine_p5 < 0.85`, the direction is UNSTABLE.
   An unstable direction is NOT used downstream without first increasing N
   (more contrast pairs) until the stability gate clears.

Log `bootstrap_cosine_mean`, `bootstrap_cosine_p5`, and `n_pairs` in every
EXPERIMENT_LEDGER.md row that involves a new or modified steering vector.

**Near-tautological estimator caveat:** when DiffMean and PCA-top1 are
computed on a tiny contrast set (N < 20) and their cosine agreement is
cos ≈ 0.99, this is almost always a near-tautology (both estimators converge
on the same small-sample direction, not because both are well-estimated but
because the sample is too small to separate them). Tag such rows as
`NEAR_TAUTOLOGICAL` and do not interpret the high cosine agreement as evidence
that the direction is well-estimated.

**New harness support:** `src/steering/controls.py:extraction_stability`.

---

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

### Provenance tracing at Checkpoint (steering instantiation)

Every Checkpoint (Step 7) must also:

1. **Update `ideas/<NN>/PROVENANCE.md`**: add this experiment's tag, the exact
   command used, the result artifact path
   (`autoresearch_results/` + run folder), and a 2–3 sentence interpretation of
   what the result means for hypothesis NN specifically.

2. **Update EXPERIMENT_LEDGER.md campaign-arc narrative** if a phase boundary
   has been crossed (e.g., switching from layer-sweep phase to alpha-sweep phase).
   "Phase boundary" = a deliberate shift in which of the 12 axes is being
   explored. The narrative groups rows by axis-phase, not just chronologically.

3. **Verify FINDINGS.md preamble** if this experiment's result graduates to a
   finding: confirm all five axis abbreviations, the hypothesis ID, and the
   result ID are defined in the preamble before committing.

See `../../meta-skills/autoresearch-findings-ledger/SKILL.md` for the full
ledger and findings discipline.

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
- [ ] Extraction stability gate cleared: bootstrap_cosine_p5 ≥ 0.85 for
      any new or modified steering vector (log in EXPERIMENT_LEDGER.md row)
- [ ] Matched-norm random-direction control configured to run (L6)
- [ ] Shuffled-label control configured to run (L6)
- [ ] EXPERIMENT_LEDGER.md row will include delta_vs_random_direction and
      delta_vs_shuffled_label columns

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-experiment/SKILL.md`
- Vector extraction + stability gate: `../steering-vector-extraction/SKILL.md`
- Eval bundle (metric calibration, off-family judge): `../steering-eval-bundle/SKILL.md`
- Ladder gates: `../steering-tiered-ladder/SKILL.md`
- Findings and ledger discipline: `../../meta-skills/autoresearch-findings-ledger/SKILL.md`
- Paper rigor (findings gate, power, multiple comparisons): `../steering-paper-rigor/SKILL.md`
- Hill-climb (adaptive seeds, joint surface, Pareto): `../steering-hillclimb/SKILL.md`
- Ledger schema: `../../EXPERIMENT_LEDGER.md`
- Hypothesis registry: `../../IDEA_TABLE.md`
- Harness modules: `src/steering/controls.py` (matched_norm_random,
  shuffled_label_vector, extraction_stability), `src/steering/stats.py`
