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
> Composite formula fingerprint: `a9001e87087e`

---

## Status

**No EXTERNAL-READY findings yet — 83 experiments run, all SCREENING (n=1).**

Experiments exp#1–83 have run on real Gemma-3-270m / Gemma-3-1b and Qwen-2.5-0.5b
(see EXPERIMENT_LEDGER.md). They produced **nine screening observations (S-1…S-9
below)** spanning E2/E3/E4/E7/E27/E36/N5/N16/N17/N20. **None has cleared the rung-3
evaluation gate** (n≥7 + paired Wilcoxon + bootstrap CI + Holm-Bonferroni +
prompting baseline + real AxBench/judge), so by the program's own rigor floor
there are **zero external-ready findings**. The screening observations below
motivate the required experiments (enumerated at the bottom) — they are NOT
citable claims.

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

**S-8 (E27 cross-scale + E4 third-model, SCREENING ONLY).** E3 cliff on
**gemma-3-1b @L18** (exp#55–59): behavior PEAKS at α=1 (0.646) then declines — a
**clean steering window** the gemma-3-270m lacked — and the cliff is gentler
(+41%/+181% PPL at α=1/2 vs 270m's +65%/+257%). Cross-scale: 270m (no window, most
fragile) < Qwen-0.5b (window) < 1b (window, least fragile) — **E27 "the steering
window emerges with scale" supported across 3 models / 2 architectures**. E4
confirmed on the 3rd model (cos=0.9945). CR 0.80→1.00 (Rogue-Scalpel). n=1. See
`ideas/_campaigns/C6_results.md`.

**S-9 (E7 + E36 SUPPORTED via relative steering, SCREENING ONLY).** Adding `relative_add` (Δh=α·‖h‖·v̂; α=fractional displacement) gives a clean, ‖h‖-independent cliff on Gemma-270m @L16 (exp C9b): behavior peaks at **α≈0.1 (10% displacement)**, knee consistent with N17/C2. At matched fractional α, **DiffMean and PCA-top1 steer near-identically** (behavior ±0.02, PPL ±8%) — E4's 0.99 cosine ⇒ equivalent steering (E36); the earlier raw-α gaps were norm artifacts. n=1. See `ideas/_campaigns/C9_results.md`.

> **All nine observations are SCREENING on ≤1B models with synthetic
> mini-data and (for S-2) a circular behavior proxy.** They establish mechanism
> direction, not magnitude. The instrument fixes and the full required-experiment
> list are tracked below.

**S-10 (E4 cross-BEHAVIOR SUPPORTED, SCREENING).** Added 4 distinct concepts
(ocean/happiness/anger/formality). cos(DiffMean, PCA-top1) at each concept's
max-Fisher layer on gemma-3-270m: anger 0.995, formality 0.996, happiness 0.999,
ocean 0.997 — **E4's >0.95 holds across all 4 behaviors** (and 3 models). The
single-behavior limitation is substantially addressed: DiffMean≈PCA is behavior-
and architecture-robust. n=1/concept.

---

**S-11 (E3 cliff cross-behavior SUPPORTED; N17 is an α-sweep law, SCREENING).**
Relative cliff on 4 concepts @L16 gemma-270m (C11): PPL rises with α for EVERY
behavior (90→293 anger, →602 formality, →246 happiness, →370 ocean) — the
coherence cliff (E3) is general, not ocean-specific. Behavior efficacy is
concept-dependent (anger 0.77 > happiness 0.63 > ocean/formality 0.53 @α=0.1:
concrete emotions steer stronger). Pooled cross-behavior-at-fixed-α N17 is weak
(Spearman +0.17, n=8) — confirming N17 is primarily an **α-sweep** relationship
(off-shell predicts PPL as you scale α within a config), not a cross-behavior law.

---

**S-12 (E17 stacking SUPPORTED; E10 partial, SCREENING).** Gemma-270m @L16,
4 concepts. E10: pairwise concept-vector cosines mostly near-orthogonal
(|cos|<0.3) — anger↔formality −0.18, formality↔ocean −0.10 — EXCEPT anger↔happiness
+0.48 (both high-arousal emotions). E17: stacking anger+happiness (summed, relative
0.1 each) RETAINS each behavior fully (anger 101%, happiness 110% of solo) — NO
interference even at +0.48 correlation; the shared emotional direction helps rather
than hurts. Two behavior vectors compose additively. n=1. See `ideas/_campaigns/E10_E17.json`.

---

## Rung-3 evaluation attempt (real WikiText, held-out)

**R3-1 (N17 monotone SUPPORTED on real data; N5 universal-law FALSIFIED across scale).**
On REAL WikiText-2 (n=50 pooled model×layer×α points, gemma-3-270m + gemma-3-1b):
Spearman(off-shell Δ‖h‖, log real-PPL) = **+0.585, 95% bootstrap CI [+0.353, +0.758]**
(excludes 0), p=8e-6 — N17's monotone claim holds on held-out text across two scales.
BUT the N5 universal collapse-law coefficients do NOT transfer: held-out R² (fit 270m
→ predict 1b) = **−1.6**. So the relationship is directionally robust but
quantitatively model-specific; the screening R²=0.81 was a within-pool artifact.
**Caveat:** the 50 points are (layer×α) configs, not iid seeds — the CI is a
within-grid estimate; cross-behavior replication is the next rigor step. See
`ideas/_campaigns/RUNG3_results.md`. This is the program's first rung-3 EVALUATION.

---

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
