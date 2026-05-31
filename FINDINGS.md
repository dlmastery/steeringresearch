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

**S-3 (E4 cross-model, SCREENING ONLY).** On **real Gemma-3-270m-it** @L12,
`cos(DiffMean, PCA-top1) = 0.994` (vs 0.996 on Qwen-0.5B @L21) — E4's ">0.95"
holds across both architectures. exp#15–19, n=1.

**S-4 (E27 scale-fragility + E3/N17 on Gemma, SCREENING ONLY).** On real
Gemma-3-270m-it the coherence cliff is **sharper and earlier** than on Qwen-0.5B:
cliff at α≈1 (PPL +65%) vs α≈2, and **behavior never improves** with steering
(0.50→0.22 monotone) — the smaller model exits the manifold before clean
injection (supports **E27**: smaller models leave the manifold more easily).
Off-shell Δ‖h‖ tracks PPL super-linearly (N17). Baseline CR=0.80 rising to 1.00
under steering (Rogue-Scalpel direction). exp#15–19, n=1. See
`ideas/30_alpha_coherence_cliff/results.md`.

**S-5 (E2 FALSIFIED, SCREENING ONLY).** Gemma-3-270m layer sweep @α=2 (exp#20–27):
**Spearman(Fisher ratio, behavior efficacy) = +0.14 (p=0.74)** — far below E2's
predicted ≥0.7. Max-Fisher (L12) is NOT the best steering layer; L16 gives more
behavior at lower PPL. Linear separability ≠ steering efficacy (cf. N8/E37). The
earlier E3 ran at a suboptimal layer. n=1/cell. See `ideas/_campaigns/C1_C2_results.md`.

**S-6 (N17 + N5 SUPPORTED, SCREENING ONLY).** Pooled over 23 real steered rows
(2 models, 8 layers, α∈{1..24}): **Spearman(off-shell Δ‖h‖, log PPL) = +0.705**
(Pearson 0.899); a single law **log PPL = 5.40 + 2.87·Δ‖h‖ fits with R²=0.81**
across architectures and layers. Off-shell displacement (cheap, behavior-free)
governs the coherence cliff — supports N17 and the N5 norm-budget collapse. The
strongest screening result; gate to external = n≥7 + real WikiText PPL.

**S-7 (E27 FALSIFIED + N16/CRH SUPPORTED, SCREENING ONLY).** Small-angle rotate-vs-add
at L16 on Gemma-3-270m (exp#37–46): additive steering is *gentler* than full-vector
rotation (add holds PPL 90–99; rotate degrades 100→11211; +42% PPL at matched
behavior) — **E27 falsified** for full-vector rotation (caveat: corpus Angular/Selective
use selective 2D-plane rotation). The added **angular (1−cos) metric predicts rotation's
log-PPL at R²=0.997** while radial Δ‖h‖ governs additive (R²=0.81, C2) — the
**Cylindrical Representation Hypothesis (N16) confirmed**: coherence cost = radial×angular
displacement. exp#37–46, n=1. See `ideas/_campaigns/C3_results.md`.

> **All seven observations are SCREENING on ≤0.5B models with synthetic
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
