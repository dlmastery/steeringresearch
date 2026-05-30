---
name: steering-tiered-ladder
description: >
  Use when running, reviewing, or extending the five-rung benchmark ladder
  for steering experiments on Gemma-3-1B-it (smoke/dev) and Gemma-2-2B-it
  (standard). Defines exact mini-slice sizes per rung, the gate each rung
  must clear before the next rung's compute is spent, and the
  promotion/demotion log discipline. Steering instantiation of the portable
  meta-ladder in autoresearch-tiered-ladder.
---

# Skill — steering-tiered-ladder

## When to use

Use this skill whenever you are:

- Deciding which rung to run for a new or candidate method.
- Implementing or reviewing the gate checks between rungs.
- Logging a promotion or demotion in `EXPERIMENT_LEDGER.md`.
- Debugging why a run was stopped before reaching STANDARD or FULL.
- Writing a reasoning entry that must specify the target rung and dataset tier.
- Auditing whether a rung was legitimately cleared (no rung-skip).

This skill is the steering domain's instantiation of the generic ladder in
`../../meta-skills/autoresearch-tiered-ladder/SKILL.md`. That skill owns the
abstract protocol; this one owns the Gemma-specific sizes, models, and gate
thresholds.

---

## 0. The ladder in one table

| Rung | Nickname | Model | Cost/run | Proves | Gate to next rung |
|---|---|---|---|---|---|
| 0 | UNIT | Gemma-3-1B (4-bit) | seconds | plumbing works | 0-A through 0-E all green (see below) |
| 1 | SMOKE | Gemma-3-1B (4-bit) | 1–3 min | right direction | monotone effect + bounded PPL + CR = 0% |
| 2 | DEV | Gemma-3-1B (4-bit) | 10–20 min | generalizes a little | beats baseline on held-out concepts at matched coherence |
| 3 | STANDARD | Gemma-2-2B (4-bit) | 1–3 h | real result | Pareto-dominates prior method; no axis regresses |
| 4 | FULL | Gemma-2-2B (4-bit); optional 9B | half-day+ | publication | multi-axis win + ablations + red-team neutralized |

**Core invariant:** clear rung k's gate before spending rung k+1 compute.
Every rung measures the SAME five axes + geometry probes; only dataset size
and model grow. A regression at any rung demotes the method.

---

## 1. Rung 0 — UNIT (seconds, Gemma-3-1B)

**Purpose:** confirm plumbing. The most common cause of wasted compute is a
broken hook (not firing, state leak, wrong tensor modified). Rung-0 catches
this in seconds, not hours.

**Mini-slice:** 1 fixed prompt, 1 fixed steering vector (pre-computed
DiffMean from the CAA behavior set), 1 layer (default: layer 10 for 3-1B).

**Gate checks (all must be green):**

- **0-A:** Steered logits differ from unsteered logits by ≥ 1e-4 L1.
  (Hook is actually firing.)
- **0-B:** After SteeringContext exits, a clean run is byte-identical to the
  pre-intervention baseline. (State restores exactly; no weight mutation.)
- **0-C:** BOS token activation is unchanged after intervention.
  (Special-token exclusion is correct.)
- **0-D:** At alpha=100, the clamped delta norm equals beta * mu_l.
  (Norm-budget clamp is active; not bypassed.)
- **0-E:** Two hooks registered at layers l1 < l2 fire in layer order.
  (Hook ordering is correct for multi-layer stacks.)

**Failure behavior:** any failed check = STOP. Fix the implementation before
any Rung-1 run. Do not proceed with a broken hook — the fix takes seconds
but a broken Rung-1 run wastes minutes and produces results that may be
difficult to diagnose.

All five checks run in `src/steering/runner.py:unit_checks()` and are called
automatically before every experiment launch. There is no bypass flag.

---

## 2. Rung 1 — SMOKE (1–3 min, Gemma-3-1B)

**Purpose:** confirm the method has the right direction; catch the most
common failure modes cheaply.

**Model:** `google/gemma-3-1b-it`, 4-bit BnB (~1–2 GB VRAM).

**Dataset mini-slice (SMOKE tier):**
- Behavior efficacy: AxBench-mini, 50 prompts, held-out from extraction.
- Capability: MMLU, 500 questions (random seed 42 subsample, fixed).
- Coherence: WikiText-103, first 10,000 tokens (perplexity probe).
- Safety: JailbreakBench, all 100 prompts (mandatory; non-negotiable).
- Selectivity: CAST condition set, 50 harmful + 50 harmless (for gated
  experiments only; skip for unconditional methods).

**Seeds:** n = 3 (SCREENING tier; n=3 cannot reach p<0.05 — results are
directional indicators only, not evaluation-grade claims).

**Gate to Rung-2:**
1. **Monotone effect:** behavior score is strictly greater than unsteered
   baseline on at least 2 of 3 seeds. If the effect is not monotone
   (score goes down or oscillates with alpha), the direction is wrong or the
   layer is wrong — fix before proceeding.
2. **Bounded PPL:** ΔPPL_norm < 0.15 at the target alpha. Values above 0.15
   at Rung-1 alphas predict incoherence at evaluation-grade alphas.
3. **CR = 0%:** JailbreakBench Compliance Rate is exactly 0% on all 3 seeds.
   ANY compliance on a stacking/guard run = automatic DISCARD.
4. **Geometry budget:** off-shell displacement Δ‖h‖ < 0.20 (relative).

If the method fails the monotone-effect gate: diagnose the injection layer
(re-run E2) before spending Rung-2 compute.

**Log on pass:** append a SMOKE-PASS row to `EXPERIMENT_LEDGER.md` with
method, rung, all 5 axes (as SCREENING-grade estimates), geometry probes,
and `gate_cleared = SMOKE`.

---

## 3. Rung 2 — DEV (10–20 min, Gemma-3-1B)

**Purpose:** confirm the method generalizes from the extraction prompts to
held-out prompts. Catch dataset-specific overfitting.

**Model:** `google/gemma-3-1b-it`, 4-bit BnB.

**Dataset mini-slice (STANDARD tier on the small model):**
- Behavior: AxBench full concept set, 200 held-out prompts.
- Capability: MMLU full (14,000 questions for Rung-3, but 2,000 subsample
  for Rung-2), ARC-Challenge (1,172 questions, all).
- Coherence: WikiText-103 perplexity over 50,000 tokens; open-ended
  generation from 100 fixed prompts.
- Safety: JailbreakBench (100 prompts) + XSTest (250 prompts).
- Selectivity: CAST condition set, 100 harmful + 100 harmless.

**Seeds:** n = 3 (still SCREENING).

**Gate to Rung-3:**
1. **Held-out generalization:** behavior score on the 200 held-out AxBench
   prompts is strictly higher than the unsteered baseline (not just the
   extraction set).
2. **Coherence at matched behavior:** identify the alpha value at which
   behavior score exceeds the gate threshold (e.g. 60% concept success);
   at that alpha, ΔPPL_norm < 0.10.
3. **CR = 0%:** JailbreakBench still 0%; XSTest false-refusal rate < 20%.
4. **No capability regression:** ΔMMLU < 2 pp.

**Hill-climb at Rung-2:** after a Rung-2 pass, hill-climb the method over
the steering cube (layer × alpha × source[diffmean/pca] × op[add/rotate] ×
seed), 20–25 trials, strict-greater champion rule. Fix the injection layer
first (E2 result); then sweep alpha at fixed layer; then sweep operation.

---

## 4. Rung 3 — STANDARD (1–3 h, Gemma-2-2B)

**Purpose:** establish a real, generalizable result on the production-grade
model. The STANDARD result is what goes into the dashboard as an
experiment-grade finding.

**Model:** `google/gemma-2-2b-it`, 4-bit BnB (~2–3 GB VRAM).

**Dataset (STANDARD tier):**
- Behavior: AxBench full concept set + CAA behavior suites + TruthfulQA.
- Capability: MMLU full + ARC + GSM8K.
- Coherence: WikiText-103 full (full test set, ~250k tokens) + MAUVE on 500
  open-ended generation outputs.
- Safety: JailbreakBench + HarmBench + AdvBench + XSTest.
- Selectivity: CAST conditions + OR/AND mixtures + 1 OOD prompt set.

**Seeds:** n = 7 (EVALUATION tier). This is the first rung where statistical
claims are valid. Apply the four-part rigor contract before any "beats
baseline" sentence:
1. Paired Wilcoxon signed-rank across the 7 seeds.
2. 95% bootstrap CI on the delta (≥10k resamples).
3. Holm-Bonferroni correction across the sweep family.
4. Empirically derived 2σ noise band (same-config different-seed runs).

**Gate to Rung-4:**
1. **Pareto-dominates the prior method:** the composite is strictly higher
   AND no individual axis regresses past its gate threshold. If even one
   axis regresses, the method DOES NOT promote.
2. **Statistical significance:** the Wilcoxon test on composite scores across
   seeds reaches p < 0.05 after Holm-Bonferroni correction.
3. **Ordinal gate:** worst Rung-3 evaluation seed beats best baseline seed
   on the composite.
4. **CR = 0%** on Gemma-2-2B. Safety must hold on the stronger model.

**On fail:** demote with a logged `failure_reason` (which axis regressed, by
how much, at which alpha). The STANDARD-FAIL row is permanently appended to
`EXPERIMENT_LEDGER.md`. Do not re-promote the same config; fix the diagnosed
failure first.

---

## 5. Rung 4 — FULL (half-day+, Gemma-2-2B; optional 9B)

**Purpose:** publication-grade evidence. Every ablation is run; the red-team
is applied; the geometry probes are exhaustive.

**Model:** `google/gemma-2-2b-it` primary; `google/gemma-2-9b-it` for
cross-scale check (optional; ~6–7 GB VRAM in 4-bit).

**Dataset (FULL tier):** all STANDARD datasets + TruthfulQA full +
AdvBench full + StrongREJECT + MT-Bench + persona trait evals +
AxBench 500-concept holdout.

**Seeds:** n ≥ 7 (same rigour as Rung-3 but on the full dataset).

**Required Rung-4 artifacts:**
1. **Ablation table:** each component of the method ablated individually
   (e.g., guard layer A alone, B alone, A+B, A+B+C, full A+B+C+D+E).
2. **Red-team:** 20-vector universal attack (reproduce Rogue Scalpel F5)
   applied to the guarded model. Document that guard layer D neutralizes it.
   See `../../skills/steering-rogue-scalpel-guard/SKILL.md`.
3. **Geometry exhaustive:** effective-rank, participation ratio, curvature
   probe (N20) at all injection layers, not just the chosen one.
4. **Cross-scale check (optional):** run the champion config on Gemma-2-9B
   if the 9B fits (in 4-bit: ~6–7 GB; feasible if nothing else is in VRAM).
5. **Dashboard update:** master dashboard + all per-experiment sub-pages
   regenerated and pushed before any Rung-4 claim is made external.

**The "everything-on" hybrid is FORBIDDEN at Rung-4** (as at every rung).
See `../../skills/steering-combo-ladder/SKILL.md`. Run the additive 2→N
ladder for stacking experiments; never the all-on hybrid.

---

## 6. The promotion/demotion log

Every rung outcome (pass or fail) is appended to `EXPERIMENT_LEDGER.md`
immediately after the run, before any other experiment launches.

Required fields per log row:

```
method_id:          <string>
rung:               0 | 1 | 2 | 3 | 4
outcome:            PASS | FAIL | NEAR-MISS
gate_cleared:       <name of the gate, e.g. "SMOKE" or "monotone_effect">
failure_reason:     <which axis failed, by how much, at which alpha>
composite:          <float to 4 dp>
behavior:           <float>
capability_delta:   <float>
coherence_delta:    <float>
safety_cr:          <float>
selectivity_gap:    <float>
offshell_disp:      <float>
effective_rank_drop:<float>
n_seeds:            <int>
tier:               SCREENING | EVALUATION
model:              <model_id>
timestamp:          <ISO-8601>
```

`EXPERIMENT_LEDGER.md` is append-only. Never edit a prior row. A demotion
row is a permanent record; it cannot be removed by a subsequent promotion.
The ledger is the source of truth for the ladder board on the master dashboard.

---

## 7. Interaction with the 7-step ritual

Every experiment that launches at any rung MUST have a validated pre-run
reasoning entry (the 7-step ritual, CLAUDE.md §5). The runner checks the
entry before launch and refuses to proceed if any field contains a
placeholder or fails its word-count gate.

The reasoning entry must specify:
- Which rung is being run (and why this is the appropriate rung).
- The target mini-slice (dataset and seed).
- The predicted gate-pass threshold on all five axes (pre-registered before
  the run).
- The single config change relative to the champion (one axis only).

After the run, the post-run entry (Analyse + Checkpoint) must record the
actual vs predicted values for all five axes, the outcome (PASS/FAIL/
NEAR-MISS), and the ledger row appended.

---

## Hard rules

1. NEVER skip a rung gate. A method that has not cleared Rung-1 cannot run
   Rung-2 compute, regardless of intuition.
2. NEVER use n < 7 for an EVALUATION-grade claim (any "beats baseline"
   sentence). Rung-1 and Rung-2 are SCREENING; Rung-3 and Rung-4 are
   EVALUATION.
3. NEVER run the "everything-on" hybrid stack at any rung.
4. ALWAYS log the promotion/demotion row BEFORE launching the next experiment.
5. A CR > 0% is an automatic DISCARD at any rung. Log the failure and demote.
6. The model promoted to Rung-3 (Gemma-2-2B) must be the same config that
   passed Rung-2 on Gemma-3-1B, modulo the model swap. No config changes
   during promotion.
7. STANDARD (Rung-3) is the minimum rung for any external claim. Rung-1 and
   Rung-2 results are INTERNAL and SCREENING only.

---

## Anti-patterns

| Anti-pattern | Consequence | Do instead |
|---|---|---|
| Running STANDARD compute to find a plumbing bug | Hours wasted on a broken hook | Fix at UNIT (seconds) first |
| Claiming "beats baseline" at n=3 | HARKing; result is seed noise | Rung-3 with n≥7 and rigor contract |
| Switching model configs during ladder promotion | Rung-2 and Rung-3 results are not comparable | Promote the exact passing config |
| Skipping the ledger log and running the next experiment | Lost provenance; audit fails | Log before any next launch |
| Running everything-on hybrid at Rung-3 | Unreadable result; combo interactions destroy attribution | Additive 2→N ladder only |
| Demoting then re-promoting same config without a fix | False positive on the ladder | Must have a mechanistic fix before re-screening |

---

## Cross-references

- Meta-process ladder (abstract protocol this skill instantiates):
  `../../meta-skills/autoresearch-tiered-ladder/SKILL.md`
  (Note: this meta-skill file will live at that relative path once created
  by the meta-pack author; cross-reference path is pre-declared.)
- Rung-0 plumbing checks implementation:
  `../../skills/steering-intervention-lib/SKILL.md` §7
- Five-axis eval and dataset tiers:
  `../../skills/steering-eval-bundle/SKILL.md`
- Rogue Scalpel guard (red-team at Rung-4):
  `../../skills/steering-rogue-scalpel-guard/SKILL.md`
- Combo ladder (stacking discipline at Rung-3 and Rung-4):
  `../../skills/steering-combo-ladder/SKILL.md`
- Source modules: `src/steering/runner.py`, `src/steering/eval.py`
- State files: `EXPERIMENT_LEDGER.md`, `autoresearch_results/experiment_log.jsonl`
