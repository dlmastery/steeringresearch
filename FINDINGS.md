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

*(No screening observations yet -- no experiments run.)*

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
