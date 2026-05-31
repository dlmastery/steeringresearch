# FINDINGS — External-Ready Results

> **Contract:** A claim may appear in this file ONLY when ALL of the following hold:
>
> 1. **n >= 7 seeds** for the winning configuration.
> 2. **Paired Wilcoxon signed-rank test** on the delta (p < 0.05 after correction).
> 3. **95% bootstrap CI** (>= 10,000 resamples) on the delta excludes zero.
> 4. **Holm-Bonferroni correction** applied across the sweep family.
> 5. The **worst evaluation seed beats the best baseline seed** (ordinal gate).
> 6. The result is logged in `EXPERIMENT_LEDGER.md` with verdict `KEEP` at rung >= 3.
>
> Everything that does not meet this full contract lives in `EXPERIMENT_LEDGER.md`
> as a screening observation. Moving a result from ledger to here without meeting
> the contract is a HARKing violation -- a BLOCKER.
>
> All claims price ALL five axes via the fingerprinted composite. A method that
> wins on behavior but loses on coherence or safety is NOT an external-ready finding.
>
> Composite formula fingerprint: `[TO BE FILLED AFTER eval.py IS WRITTEN -- placeholder a9001e87087e]`

---

## Status

**No external-ready findings yet -- program initialized 2026-05-30.**

The research program is in the pre-experiment backbone phase. Hypothesis
registration (IDEA_TABLE.md) and experiment infrastructure (EXPERIMENT_LEDGER.md)
are complete. The first experiments (E1-E3 infrastructure block) have not yet run.

---

## Screening observations (NOT external claims)

> This section records directional observations from n <= 3 seed runs that are
> INTERESTING but do NOT meet the rigor contract above. They motivate hill-climbing
> and full evaluation but cannot be cited or presented as results. Label each with
> the experiment tag, n, and "SCREENING ONLY -- not an external claim."

**S-1 (E4, SCREENING ONLY — not an external claim).** On Qwen2.5-0.5B-Instruct,
`cos(DiffMean, PCA-top1) = 0.996` at the max-Fisher layer (L21) — consistent with
E4's pre-registered ">0.95" threshold. tag=bring-up, n=1. *Gate to external:
n≥7, multiple behaviors, Gemma reproduction.*

**S-2 (E3 + N17, SCREENING ONLY — not an external claim).** On Qwen2.5-0.5B @L21,
additive steering shows a **super-linear coherence cliff**: PPL +20%→+82%→6×→77×
across α=1→2→4→8, knee at α≈1–2; the fingerprinted composite peaks at α≈1.
**Off-shell Δ‖h‖ rises monotonically in lockstep with PPL** (N17: geometry probe
predicts the cliff). tags=`E3-cliff-a*`, exp#2–9, n=1. See
`ideas/30_alpha_coherence_cliff/results.md`. *Gate to external: real
generation-based behavior + real safety + n≥7 + prompting baseline + Gemma.*

> **Both observations are SCREENING on a non-Gemma 0.5B model with synthetic
> mini-data and (for S-2) a circular behavior proxy.** They establish mechanism
> direction, not magnitude. The instrument fixes and the full required-experiment
> list are tracked below.

## Required experiments before ANY external claim (from `audits/ICML_REVIEW.md`)

The ICML-area-chair review (Reject 3/10 as a claims-source today; Borderline 6/10
methodology) requires, before promotion off SCREENING:

1. **Real behavior efficacy** — replace the projection proxy with an independent
   LLM-judge / AxBench scorer on **real generated text**, human-calibrated. (In
   progress: generation-based concept-incorporation scorer.)
2. **Real safety + selectivity** — generate on real JailbreakBench + XSTest, judge
   SAFE/UNSAFE, prove baseline CR≈0% and that auto-DISCARD can fire. Delete the
   `_fake_responses` stub. (In progress.)
3. **n≥7 seeds on real Gemma-2-2B** with real AxBench/MMLU/WikiText + a **prompting
   baseline**, the full rigor contract (Wilcoxon + bootstrap CI + Holm-Bonferroni
   + ordinal gate), a held-out-concept split, and a shuffle-test negative control.

---

## Finding template (for when findings exist)

The following template shows the required structure for each finding entry:

### F-<N>: <One-line claim>

**Source experiments:** exp-NNN, exp-MMM (tags: ...)
**Hypothesis:** E1 / N5 / etc. from IDEA_TABLE.md
**Model:** Gemma-2-2B-it (4-bit), Gemma-3-1B-it (smoke confirmation)
**Rung:** STANDARD (rung 3) / FULL (rung 4)

**Result table:**

| Metric | Value | 95% CI | n |
|--------|-------|--------|---|
| Behavior efficacy | X.XX | [A, B] | 7 |
| MMLU delta (pp) | X.XX | [A, B] | 7 |
| PPL | X.XX | [A, B] | 7 |
| JailbreakBench CR | X.X% | [A, B] | 7 |
| Over-refusal (XSTest) | X.X% | [A, B] | 7 |
| Composite | X.XXXX | [A, B] | 7 |
| Delta vs baseline composite | +X.XXXX | [A, B] | 7 |

**Statistics:**
- Wilcoxon signed-rank: W=X, p=Y (Holm-Bonferroni corrected)
- Bootstrap CI (10,000 resamples): [A, B]
- Ordinal gate: worst eval seed (X.XXXX) > best baseline seed (X.XXXX) -- PASS

**Falsifier status:** Pre-registered falsifier on IDEA.md: [quoted]. Result: NOT FIRED.

**Qualifier:** Internal QA pass -- independent external review pending.

---

> *This file will be updated each time an experiment clears rung 3 (STANDARD)
> or rung 4 (FULL) with the full rigor contract satisfied. Until then, all
> quantitative observations live in EXPERIMENT_LEDGER.md.*
