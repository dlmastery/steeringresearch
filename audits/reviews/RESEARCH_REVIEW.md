# Deep Scientific Review — Conditional Multi-Intent Safety Steering

**Reviewer role:** elite ML researcher / area chair (steering + interpretability).
**Scope:** experimental design, validity, inference quality. NOT a code review.
**Date:** 2026-06-09. **Repo:** C:\Users\evija\steeringresearch.
**Files read:** FINDINGS.md (S-1..S-26), PREREGISTRATION.md, STATUS.md, README.md,
src/steering/DESIGN.md, cast.py, controls.py, intent_gate.py, safety_judge.py,
local_judge.py, judge.py, docs/METHOD_LADDER.md, IDEA_TABLE.md Block G (M1-M7),
scripts/run_axbench_e7.py, scripts/run_axbench_conditional.py.

---

## 1. Summary verdict and score

**Score: 5.5 / 10** (as a *research program in progress*; as a *paper*, it is a
clear reject — but the program itself is unusually honest and the negative results
are real, which is worth more than most positive-result drafts).

The single most important sentence in the whole repo is correct and well-earned:
**"plain steering direction is generic at small scale; the displacement norm +
coherence carry the effect, not the specific direction"** (S-16, S-19, S-24). This
is a genuine, falsification-grade finding produced by a properly-built control
(`shuffled_label_vector`) and a paired population test over concepts. The project's
rigor discipline (rigor contract, screening/evaluation split, the fingerprinted
composite, the unanimous-reject self-report) is **better than the median accepted
paper** in this subfield. That candor is the program's primary asset.

However, the *pivot* that this finding motivated — "therefore the contribution is
the conditional COSINE gate at AUC 0.74" — is **not yet supported**, is partly
**confounded**, and on the current dev config (1b + AxBench + Qwen-3B judge) is
**on a path that cannot produce a top-tier conditional-SAFETY result**. The program
has correctly diagnosed that its old contribution is dead, but has not yet
established that its new one is alive. It is at an inflection point, not a result.

---

## 2. What is valid (defensible today)

- **E7 NULL on AxBench (S-16/S-19/S-24).** The control is the right *kind* of null
  (matched-displacement, label-destroyed, applied via the identical `relative_add`
  path), the replication unit is the concept (genuinely independent replicates, not
  bootstrap redraws — the script is explicit about this), and the verdict logic is
  sign-aware (a negative delta is reported as NEGATIVE, not spun as a win). The
  conclusion "direction is largely generic at this scale" is **valid as a screening
  claim**, with the one caveat in §3.
- **E3 / N17 geometry (S-17, N17 rung-3).** "How far you push off-shell predicts
  incoherence" is the most defensible result and is correctly *down-graded* (N5
  universal-law FALSIFIED at held-out R²=-1.6; the within-pool R²=0.81 honestly
  retracted as a model-mixing artifact). The non-iid caveat is disclosed. This is
  model-of-good-practice negative-result handling.
- **The honest map of wash results** (E2 layer flat, E36 source wash, E27 rotate~add)
  — all correctly reported as "alpha is the only knob that matters," which is itself
  a real (if deflationary) scientific statement about small-model steering.
- **S-22 reframing.** The observation that gemma-2-2b-it has **ASR=0 on raw JBB**
  (no headroom without attacks) is a sharp, correct, and consequential finding —
  it invalidates the naive "reduce ASR vs baseline" framing and forces the project
  toward either under-attack or over-refusal regimes. This is good science.

## 3. What is over-claimed or confounded

**(a) "real == shuffled ⇒ direction is generic" is partly judge-limited, not purely
a property of the direction.** This is the review's central technical concern. At
270m, BOTH arms sit at ~0.05 on a 0–1 scale (S-16); at 1b both sit at ~0.43–0.46
(S-24). When both arms are near a floor (or compressed), a judge with **AUC 0.68**
(barely above chance; the validation is itself only `--per 10` per concept) **cannot
resolve a small true direction effect even if one exists.** "real == shuffled" is
therefore confounded by (i) near-floor behavior on small models and (ii) a judge
whose discrimination is too weak to separate the arms. The repo states the generic
conclusion more strongly than the instrument licenses. The correct claim is: *"under
an AUC-0.68 judge at small scale, any direction-specific effect is below the
detection floor"* — which is weaker and more honest than "the direction is generic."
The 2B +0.004 result (significant, ordinal-fail) is actually *consistent with a real
but tiny direction effect that the judge mostly washes out*, not with zero effect.

**(b) The shuffled-label control may be too strong on AxBench's structure.** AxBench
contrast data is unpaired and concept-dominated; S-19 itself shows PCA picks up
text-variation variance. A random re-partition of a concept-dominated pool can
recover a substantial fraction of the dominant axis of variation, so the shuffled
control is **not a pure null — it is a partially-concept-loaded null.** "97% captured
by shuffled" may overstate genericity because the denominator (the control) is
inflated. A `matched_norm_random` control (which exists in controls.py but is NOT
the one used in the headline E7 runs) would be a *cleaner* upper-bound null; the
right experiment reports the effect against BOTH and brackets it.

**(c) AUC 0.74 gate-as-detector is over-promoted to "the contribution."** Three
problems: (i) **No committed driver.** S-25(a)/S-26 (the 0.747 / 0.738-vs-0.434
numbers) have no script in `scripts/` that I can find — they appear to be ad-hoc
runs (cf. the uncommitted `*.partial.json` in git status). A headline contribution
must have a re-runnable, logged driver. (ii) **0.74 is "moderate = weak," not
publishable-useful for a *safety* gate.** A safety gate that fires the steer must
operate at a chosen FPR; at AUC 0.74 the recall at FPR≤1% (the pre-registered
over-refusal bound) is poor — the ROC is too shallow to give both high harmful-recall
AND ≤1% benign-firing. The selectivity headroom the method needs lives precisely in
the region a 0.74 detector cannot serve. (iii) **The detector AUC is measured on
AxBench concept-presence in *inputs*, but the safety method gates on harm-intent in
*inputs* — different distribution.** The 0.74 does not transfer to the safety task
by assumption.

**(d) The conditional GENERATION result (S-25b, "+0.125 fluency saved") is
confounded by the harness, as the repo half-admits.** Worse than admitted: in
`run_axbench_conditional.py` the gate threshold `tau` is set from the **same**
off-target cosine vector it is then evaluated on (lines 128–131: `tau = quantile(cos_off, 1-fpr)`
then `off_fire = cos_off > tau`). This is an **in-sample threshold** — the off-target
FPR is mechanically pinned to `target_fpr` by construction, so "conditional preserves
off-target fluency" is partly a tautology of how tau was chosen, not an out-of-sample
property. The +0.125 cannot be cited even as screening without a held-out tau.

**(e) AxBench is the wrong testbed for the actual contribution, and the repo says so
but keeps using it.** S-25 explicitly notes the trigger in AxBench is in the OUTPUT
not the INPUT, and clean induction ≥ steered, so the gate rarely fires on-target
(0.25). A conditional INPUT-gated method has **no clean signal to optimize** on
AxBench. Developing the conditional gate on a benchmark where the gate can't fire
on-target is a category error that no amount of seeds fixes.

## 4. Judge-validity verdict

**The off-family choice is correct; the discrimination is not adequate for any
direction-level or safety-level claim.**

- *Circularity:* Qwen judging Gemma is genuinely off-family — the same-family
  rubber-stamp risk is broken. This is handled correctly and disclosed everywhere.
- *Behavior judge (AxBench, AUC 0.68):* below the project's own 0.80/0.85 bar.
  AUC 0.68 is fine for *gross* effects (the E3 cliff: behavior swings 0.16→0.05, a
  3× change a coarse judge can see) but **inadequate for the *small* contrasts** the
  E7 direction question hinges on (deltas of 0.004–0.03). Using the same coarse judge
  for both a 3× effect and a 1% effect, and trusting both, is the core validity gap.
  The N17/E3 conclusions survive a weak judge; the E7 "generic" conclusion does not
  cleanly, per §3(a).
- *Safety judge (Qwen few-shot, 9–10/12 dev, conservative bias):* a 12-item dev set
  is **not a calibration** — it is an anecdote. The conservative bias is applied
  identically to method and baselines (good, it cancels in *differences*), but a
  conservative judge **compresses ASR toward 0 for everyone**, which destroys the
  very headroom the method needs to show an ASR win. The project's own
  PREREGISTRATION §6 sets AUC≥0.80 as a precondition and the method ladder Rung 2
  demands a calibrated classifier (Llama-Guard-3 / ShieldGemma, ≥0.90 on ≥100 items).
  **Until that precondition is met, no compliance number can ground an external
  claim — full stop.** The repo states this; it must hold the line.

**Bottom line on judging:** AUC 0.68–0.74 is good enough to *falsify* (kill a
hypothesis, as it did for E7-direction and the trained gate) but NOT good enough to
*establish* anything positive — least of all a small-margin conditional-safety win.
Falsification under a weak instrument is sound (a weak judge that still kills your
hypothesis is conservative evidence the hypothesis is dead). Confirmation under a
weak instrument is not.

## 5. Is the conditional-gate contribution real?

**Not established, and the current evidence is closer to "weak/confounded" than
"real."** The logic chain is: E7 direction NULL ⇒ pivot to gate. But:

- The pivot is *motivated* correctly (if direction is generic, the value must be in
  *when* to steer / collateral avoidance — this is sound reasoning and matches the
  CAST/Rogue-Scalpel literature).
- But the gate's empirical basis is AUC 0.74 (moderate, confounded per §3c), its
  generation result is in-sample (§3d), and **the cosine gate is unsupervised and
  un-learnable at this data budget** (S-26: 12 positives in 1152 dims ⇒ the logistic
  overfits, AUC 0.43 < chance). So the contribution collapses to: *"an unsupervised
  cosine threshold at AUC ~0.74 caps the method's selectivity."* That is a **ceiling
  result, not a method** — it says how *little* the small-model gate can do, not that
  it does something publishable.

A conditional gate IS the right *shape* of contribution for this North Star (CAST is
the correct lineage, and conditioning is the meta-layer that stacks). But the project
has shown the gate is *possible and moderate*, not that it *Pareto-dominates a
prompting baseline on real safety benchmarks* — which is the only claim that matters.

## 6. Pre-registration / HARKing assessment

PREREGISTRATION.md is **sound, falsifiable, and well-matched to the *intended*
experiment**: numeric, frozen, pre-classified screening-vs-evaluation, with explicit
judge-AUC and alpha-selection preconditions and a no-optional-stopping rule. This is
exemplary.

**HARKing risk is LOW but not zero, in two places:**
1. The success criterion (ASR↓≥15pp) was registered for gemma-2-2b-it, then S-22
   discovered ASR=0 (no headroom). The program is now drifting toward "over-refusal /
   under-attack" regimes. This is *legitimate* (the world changed the question), but
   it MUST be handled by a **dated amendment** to PREREGISTRATION.md with a NEW frozen
   criterion BEFORE the next eval — not by silently retargeting. The Amendments
   section is still "(none yet)"; this is the single biggest process risk right now.
2. The target pivot (2b→270m→1b) and dataset (AxBench) were user-mandated dev
   choices, but the *success criterion* still references 2b/JBB/XSTest. The dev config
   and the eval config are now **disjoint**, and there is no pre-registered bridge
   showing a 1b/AxBench result predicts a 2b/JBB result. Re-registration needed.

## 7. Can the current dev config produce a top-tier conditional-SAFETY result?

**No — not as currently composed.** The dev config (gemma-3-1b-it + AxBench +
Qwen-3B judge) has three disqualifying mismatches to the North Star:
1. **AxBench tests output-concept induction, not input-intent gating** (§3e). The
   contribution can't even fire on-target here.
2. **The judge is below its own precondition bar** and is conservative on exactly
   the axis (compliance) the headline needs.
3. **1b is near-floor / coherence-limited** (S-24: no rising window past the gentlest
   push), so any margin is small and judge-unresolvable.

It is a reasonable *plumbing/dev* config (cheap iteration, real benchmark, off-family
judge) but it is a **dead end for the publishable claim** unless the eval moves to a
real safety, input-gated regime.

## 8. What a NeurIPS/ICLR reviewer still rejects on

- **Zero external-ready results** (the repo's own headline). No paper without one.
- **Judge below bar** (AUC 0.68/0.74; safety judge calibrated on n=12). Reviewer 2
  ends the discussion here.
- **No real safety benchmark wired end-to-end** (CR is still a regex per STATUS).
- **The one rigorous result is negative** and the positive pivot is in-sample /
  un-driven. "Moderate AUC 0.74 gate" is not a contribution; it is a limitation.
- **Aligned-model ASR=0** means the headline ASR-reduction framing is dead on
  un-attacked prompts; no adaptive-attack result exists yet.
- **Dev≠eval config** with no transfer argument.
- **Prompting baseline never confronted** on real safety (AxBench's own thesis is
  "simple baselines beat SAEs" — a steering method must beat prompting, untested).

## 9. The single highest-value next experiment

**Run the real INPUT-gated safety selectivity experiment on gemma-2-2b-it, UNDER
ATTACK, with a held-out gate threshold and a calibrated safety classifier — measuring
the over-refusal axis where headroom provably exists.**

Concretely, ONE experiment:
- Generator gemma-2-2b-it (the aligned model with ASR=0 baseline → headroom only
  under attack and on over-refusal).
- Extract a refusal/harm-intent direction from real harmful-vs-benign *prompts*
  (M1, never yet run on real Gemma — this is the actual missing primitive).
- Conditional gate on harm-intent in the INPUT (the regime where the gate is supposed
  to fire), threshold calibrated on a TRAIN split, evaluated on a HELD-OUT split.
- Three arms (no_steer / unconditional / conditional) on **XSTest (over-refusal,
  headroom confirmed at 33% baseline over-refusal) + JBB-under-attack (ASR headroom)**.
- Judge: ShieldGemma or Llama-Guard-3 (clears the AUC≥0.80 precondition). If
  unavailable, this run is SCREENING and labeled so.
- Pre-register (amendment) the over-refusal-reduction criterion BEFORE running.

This is high-value because it simultaneously (a) tests the *actual* contribution in
its *correct* regime, (b) attacks the axis with real headroom, (c) replaces the
washed-out judge, and (d) exercises M1, the one un-run primitive everything depends on.

## 10. Shortest scientifically-valid path to ONE external-ready result

The fastest defensible win is **NOT** an ASR-reduction SOTA claim (no headroom on
aligned models without heavy adaptive-attack engineering). It is an
**over-refusal-selectivity claim**, which the data already hints is real (S-22:
conditional 0.25 vs unconditional 0.67 over-refusal, n=12):

1. **Amend PREREGISTRATION.md** (dated): new frozen primary = "conditional gate
   reduces XSTest over-refusal by ≥X% vs unconditional steering at matched
   harmful-refusal, on gemma-2-2b-it, n≥7 seeds, full rigor contract." Freeze X and
   the matched-refusal definition NOW.
2. **Wire ONE real safety classifier** (ShieldGemma/Llama-Guard-3) and calibrate on
   ≥100 labeled items to clear AUC≥0.80. This is the gating precondition; nothing
   counts until it passes.
3. **Run M1** (real refusal/intent direction from real prompt contrasts) — the
   un-run foundation.
4. **Held-out gate calibration** (train τ on one split, evaluate on another) to kill
   the in-sample confound of §3d.
5. **n≥7 seeds, the four-part contract, ordinal gate**, on XSTest over-refusal +
   matched harmful-refusal. Confront the prompting baseline (system-prompt refusal)
   explicitly.

This yields an external-ready result of the form *"conditioning buys selectivity:
the gate preserves benign helpfulness that unconditional steering and prompting
destroy, at matched safety, with statistical rigor."* That is a **real, modest,
publishable-as-a-finding** contribution that matches the North Star's "without
over-refusing" clause, sidesteps the dead ASR-headroom problem, and uses the axis
where the screening signal already points positive. It does NOT require winning the
ASR race — which is the right strategic call given S-22.

**One caution:** even this win needs the gate to operate above the AUC-0.74 ceiling
on *harm-intent inputs* at the chosen FPR. If real-prompt intent detection also caps
at ~0.74, the over-refusal-vs-harmful-refusal tradeoff may be too shallow to clear a
≥X% bar at n≥7. Test the detector ROC on real harmful-vs-benign prompts FIRST (cheap,
judge-free) and abort early if it is < ~0.85 — do not spend generation compute on a
gate that cannot separate the inputs.

---

## Appendix: claim-by-claim hold/over-claim ledger

| Claim | Holds? | Note |
|---|---|---|
| S-23: 270m unsteerable | HOLDS | Near-floor at all alpha; correct to abandon. |
| S-24: 1b steers modestly; E7 direction NULL | HOLDS AS SCREENING, with §3a caveat | NULL is judge-floor-limited, not proven-generic. Report as "below detection floor." |
| S-25a: gate-as-detector AUC 0.74 | WEAK / NO DRIVER | Moderate; no committed re-runnable script; wrong distribution for safety. |
| S-25b: conditional saves +0.125 fluency | CONFOUNDED | In-sample tau (lines 128–131); AxBench output-trigger mismatch; cannot cite. |
| S-26: trained gate overfits, cosine wins | HOLDS | Robust replication of E15; correctly concludes "ceiling," not "method." |
| "Conditional gate is THE contribution" | NOT ESTABLISHED | Right shape, wrong testbed, ceiling-not-method, sub-bar judge. |

> Internal QA note: this review was produced by a model in the same lineage as the
> repo's other critics; per CLAUDE.md §14 it carries the "Internal QA pass —
> independent external review pending" qualifier. No git operations were performed.
> Composite fingerprint referenced by the repo: a9001e87087e.
