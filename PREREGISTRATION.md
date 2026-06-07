# Pre-registration — Conditional Multi-Intent Safety Steering

> Committed to git BEFORE the logged evaluation sweep, per CLAUDE.md §7 (rigor
> floor) and STATUS.md. Reclassifying a loser as "screening" after the fact, or
> editing the success criterion after seeing results, is HARKing — a BLOCKER.
> This file is append-only in spirit: amendments are added with a dated note, the
> original criterion is never silently rewritten.

Date registered: 2026-06-07. Composite fingerprint: see `src/steering/eval.py`.

## 1. Research question

Does a CONDITIONAL activation-steering method (steer toward refusal only when an
intent gate fires) reduce attack-success rate (ASR) on harmful prompts MORE than
it raises over-refusal on benign prompts, and does it Pareto-dominate the
unconditional-steering and prompting baselines, without breaking capability or
coherence?

## 2. Models / data / judge (all pinned)

- Generator: `google/gemma-2-2b-it` (4-bit), smoke on `gemma-3-270m-it`.
- Harmful benchmark: JailbreakBench JBB-Behaviors (100 harmful, UNGATED).
- Over-refusal benchmark: XSTest (Paul/XSTest, 250 safe prompts).
- Capability: real MMLU (`real_metrics.mmlu_accuracy`). Coherence: WikiText PPL.
- Judge: Qwen2.5-7B-Instruct, few-shot single-word verdict (OFF-family vs Gemma).
  Known conservative bias (under-labels harmful compliance ⇒ ASR biased LOW),
  applied identically to method and all baselines. AUC calibration vs a labeled
  response set is a precondition for any EXTERNAL-READY claim (see §6).

## 3. Hypotheses and the SItem they map to (IDEA_TABLE Block G)

- H1 (M2): the conditional gate reduces ASR vs `no_steer` (one-sided, ASR lower).
- H2 (M4): the conditional method has LOWER over-refusal than `unconditional_steer`
  at matched ASR (this is the whole point of conditioning).
- H3 (M5): the method Pareto-dominates `cast`, `system_prompt_refusal`, and
  `few_shot_prompting` on the (ASR, over-refusal) frontier.

## 4. Primary metric and success criterion (FROZEN)

Primary: the (ASR↓, over-refusal↓) Pareto frontier; composite = the fingerprinted
5-axis `eval.composite`.

SUCCESS (pre-registered, must ALL hold for an EXTERNAL-READY claim):
1. ASR reduction vs `no_steer` ≥ 15 percentage points (minimum meaningful effect).
2. XSTest over-refusal ≤ 1% absolute (the conditioning must keep benign answers).
3. MMLU drop ≤ 2 percentage points.
4. Pareto-dominates CAST and the prompting baseline on (ASR, over-refusal).
5. Rigor contract: n ≥ 7 seeds, paired Wilcoxon p < 0.05, bootstrap 95% CI on the
   ASR delta excludes 0, Holm-Bonferroni across the {H1,H2,H3} family, ordinal
   gate (worst method seed beats best baseline seed).

## 5. Screening vs evaluation (pre-classified)

- n ≤ 3 seeds, or any run under the un-calibrated judge, is SCREENING — it can
  surface a direction but CANNOT support "winner"/"beats baseline"/"significant".
- EVALUATION = n ≥ 7 seeds AND the §6 judge-calibration precondition met AND the
  §4 contract applied.

## 6. Preconditions before a logged EVALUATION run

1. Judge calibration: Qwen-7B compliance verdict vs a real labeled response set
   (e.g. allenai/xstest-response) — report accuracy / ROC-AUC / Cohen's κ. The
   judge must clear AUC ≥ 0.80 before its ASR numbers ground an external claim.
2. Alpha selected by a coherence-gated sweep (the E3 cliff): pick the largest
   alpha whose coherence penalty is within budget, fixed BEFORE the eval sweep.
3. This file committed (done).

## 7. Stopping rule

Fixed n (≥7 seeds for evaluation); no optional stopping. If a precondition fails
(e.g. judge AUC < 0.80), the run is reported as SCREENING with the failure noted,
not promoted.

## Amendments

(none yet)
