# ICML-Style Adversarial Review — Steering Research Program

**Reviewer stance:** top-tier, rigor-obsessed program-committee member.
**Judged against the stated goal:** *a state-of-the-art **conditional** activation-steering
method for **multi-intent safety** — steer an LLM, only when warranted, toward safer
responses than baseline, across multiple intents, without breaking capability/coherence
or over-refusing; method dev on AxBench, FINAL eval on SOTA safety benchmarks
(JailbreakBench, StrongREJECT, HarmBench, XSTest, AdvBench); deliverable = a paper a top
venue would accept.*
**Date:** 2026-06-06. **Commit context:** master @ 9e1652c. **Composite fingerprint:** a9001e87087e.

---

## 1. Verdict, score, and the bar to clear

**Verdict: REJECT (strong reject against the stated publication goal).**
**Score: 2 / 10** (ICML scale: 2 = "Strong reject"; the *infrastructure and honesty* are an
8, but the *scientific deliverable against the stated goal* is a 1).

**Why this is not close.** The stated goal is a **conditional safety-steering method**.
After 124 experiments, **the method does not exist and has never been measured.** Every
conditional/gated hypothesis (E9–E16) and every robustness/safety hypothesis (E41–E50) is
`PENDING/UNTESTED`. There is **no CAST pipeline** (only a `gate.py` cosine primitive), **no
real safety benchmark wired in** (no JailbreakBench, StrongREJECT, HarmBench, XSTest, or
AdvBench loaders — `grep` finds them only in prose and dashboards, never in code or data),
and the only "safety" instrument that exists is a 22-substring refusal matcher run against
**10 hand-authored harmful prompts** (`src/steering/data/jailbreak_mini.json`, n=10) and
**8 XSTest-style prompts** (n=8). The headline empirical result the program *did* obtain is
**negative**: on real AxBench, the concept *direction* barely beats a shuffled-label control
(+0.004 at 2B; a shuffled vector captures ~97% of the effect), and "alpha is the only knob
that matters." That is an honest and even publishable *negative finding about steering in
general* — but it is the opposite of the stated goal, and it is not a safety result at all.

**The bar to clear for a top venue (none currently met):**
1. A **named conditional method** implemented end-to-end (condition read → gated behavior write in one forward pass), not a design doc.
2. **Final evaluation on ≥3 of the named SOTA safety benchmarks** with a **high-precision harm judge** (Llama-Guard-3 / the JailbreakBench classifier), human-validated at ≥0.9 agreement on ≥100 items.
3. **The right baselines** — system-prompt refusal, few-shot prompting (AxBench's own headline: prompting beats steering), SFT/DPO-lite, unconditional steering, and a published safety-steering method (CAST, Arditi refusal-direction, circuit-breakers/RepE).
4. A **multi-intent** evaluation (≥5 harm categories) showing the gate generalizes across intents and does not over-refuse on XSTest.
5. The full rigor contract **applied across the garden of forking paths** (Holm across all 70 hypotheses, not vacuously skipped), with **effect sizes vs a pre-registered minimum meaningful effect**, not just p-values from n=500.

---

## 2. Honest summary of what the repo is

This is an unusually **well-engineered, scrupulously honest autoresearch harness** that has
screened ~19 of 70 pre-registered steering hypotheses on small Gemma/Qwen models, mostly at
n=1, with a recent and commendable "real-benchmark redo" of 6 hypotheses on AxBench using an
off-family local judge. The authors **repeatedly and correctly refuse to over-claim**: the
README, FINDINGS, and ledger all state "zero external-ready findings," disclose the judge's
weak AUC (0.68), and document how a validated judge reversed a proxy artifact (the exp#114→116
story is genuinely good scientific hygiene). The statistical toolkit (`stats.py`) implements
paired Wilcoxon, percentile bootstrap, Holm–Bonferroni, an empirical seed band, and a strict
ordinal gate — all clean, offline, and tested.

But measured against the **safety-method goal**, the project is at the *infrastructure +
exploratory-negative-result* stage. It has discovered that generic small-model concept
steering is weak; it has not begun the safety method it set out to build.

---

## 3. Real strengths (credit where due)

- **Intellectual honesty is the standout.** Negative results are reported as negative, the
  sign-blind auto-label bug is disclosed and corrected, proxy-vs-judge artifacts are
  documented, and "external-ready = 0" is stated everywhere. This is rarer than it should be.
- **The composite is at least *attempting* Goodhart resistance** by pricing five axes, and it
  is fingerprinted to deter silent edits.
- **Confound controls are real and correct in spirit.** `controls.py`
  (`matched_norm_random`, `shuffled_label_vector`, `extraction_stability`) is exactly the
  matched-displacement control most steering papers omit; using it is what exposed the weak
  E7 effect.
- **The off-family-judge discipline** (Gemma generator, Qwen/Gemini judge) correctly attacks
  same-family circularity, and the judge cache makes runs deterministic and reproducible.
- **Hooks/extract are clean and correct**: position masking is rebuilt per forward (handles
  KV-cache decode steps), special tokens are protected, state restoration is asserted.
- **Pre-registration scaffolding** (IDEA_TABLE thresholds frozen, 7-step ritual, append-only
  log) is the right anti-HARKing architecture.

---

## 4. Weaknesses, grouped

### 4.1 Statistical validity
- **The Holm leg is silently skipped on the real-benchmark runs.** `rigor_report` treats
  `holm_rejected = True` *vacuously* when `family_pvalues=None`, and every AxBench driver
  (`run_axbench_e7.py` etc.) calls `rigor_report(real, shuf)` **with no family**. So the
  "garden of forking paths" across 70 hypotheses / 6 redos / many alphas-layers-sources is
  **never corrected**. The contract's leg 4 is decorative in practice.
- **n=500 makes the Wilcoxon a triviality detector.** At 2B, p=0.011 for a +0.004 effect on a
  0–1 scale where the shuffled control already captures 97%. The pipeline still emits
  `verdict="DIRECTIONAL"`. **Significance is not effect size.** There is no pre-registered
  **minimum meaningful effect (MME)** anywhere; without it, every large-n run will "pass"
  legs 1–2 on noise-sized effects.
- **The ordinal gate is mis-specified for the concept-as-replicate design.** `ordinal_gate`
  computes `worst(eval) > best(baseline)` across the **500 heterogeneous concepts**. Concept
  difficulty dominates that comparison (easy concept's shuffled score >> hard concept's real
  score), so the gate is testing concept variance, not direction effect. The honest paired
  unit is the per-concept delta; the ordinal gate as written can essentially never pass and
  isn't measuring what the prose claims.
- **The bootstrap CI is a within-grid / within-concept-pool estimate** (the authors admit
  this for N17). It does not capture prompt-set or behavior-population variance, so the CIs
  are narrower than honest.
- **The numpy Wilcoxon fallback uses a normal approximation** even for tiny n; fine for n=20+,
  but combined with no power analysis it obscures that **n=3 screening can never reach
  p<0.05** (the CLAUDE.md rule is stated but the drivers don't enforce a power floor).

### 4.2 Sample size, power, seeds
- **Most verdicts are n=1.** 11 "SUPPORTED" hypotheses rest on single runs. The few multi-N
  results use **concepts as replicates (n=20–500), not seeds** — there is essentially **no
  seed-variance estimate** for the real-model behavior pipeline (greedy decoding hides it,
  but extraction, judge, and batching are stochastic across configs). `seed_noise_band`
  exists but is not exercised on the AxBench runs.
- **The "30-concept curve" results (E3 cliff) are not population tests** (the authors say so).
  20–30 concepts at a single model/single layer is a screening curve.
- **No cross-model or cross-seed replication for any positive claim** except N17 (which then
  failed the cross-scale law). The strongest result rests on 50 grid points across 2 models.

### 4.3 Baselines and comparisons (the most damaging gap for a safety paper)
- **AxBench's own headline finding — prompting beats steering — is not run as a baseline.**
  For a steering paper this is disqualifying: the field's current answer is "just prompt." You
  must include few-shot prompting and a system-prompt refusal baseline and beat them.
- **No published safety-steering baseline is implemented.** CAST (the method E9 claims to
  reproduce), Arditi et al. refusal-direction ablation, RepE/circuit-breakers — none exist in
  code. The only comparator is "unconditional steering vs shuffled control."
- **No SFT/DPO or guard-model baseline** (e.g., Llama-Guard as a router). A safety method that
  isn't compared to a cheap classifier-router gate has no claim to SOTA.
- **The "method" is DiffMean relative-add**, the simplest possible primitive; the program has
  not yet tested anything that could plausibly *be* SOTA.

### 4.4 Judge / measurement validity for SAFETY (acute)
- **A 0.68-AUC judge cannot back a safety claim. Full stop.** For concept expression it merely
  widens CIs; for a **refusal/harm decision** a 0.68 classifier is barely above chance and
  will mislabel a large fraction of harmful completions as safe (and vice versa). Safety
  decisions need ≥0.9 agreement.
- **There is no harm judge at all.** Safety is scored by `is_refusal()` — a 22-string substring
  matcher ("i can't", "as an ai", …). This (a) counts any non-refusal as **full harmful
  compliance** even if the output is useless/garbled, (b) misses paraphrased or partial
  refusals, (c) cannot grade *degree* of harm (StrongREJECT's entire point), and (d) is
  trivially gamed by steering toward gibberish (which the composite then rewards on the safety
  axis while penalizing on coherence — but the two are not jointly validated on real harmful
  prompts).
- **The "safety baseline is non-zero at alpha=0" caveat** in the ledger means the safety axis
  is **uncalibrated**: the unsteered model already "complies" 30–80% by the substring matcher,
  so every composite is dominated by an instrument artifact. The authors know this; it makes
  every composite "informative only within-campaign," i.e., not a real metric.

### 4.5 Ablations
- **No ablations of the proposed method, because there is no method.** When CAST is built, the
  paper will need: condition-layer L_c sweep, threshold θ_c sweep with PR curves, gate-on vs
  gate-off (the core ablation), single-vector vs OR-gated multi-intent, and norm-budget caps.
- **The composite weights (λ) are unablated.** Six hand-set weights (λ_safe=2.0, λ_coh=0.5,
  …) with **no sensitivity analysis**; conclusions could flip under reasonable reweighting.
- **No ablation isolating "generic displacement" from "concept direction"** beyond the
  shuffled control at one alpha — which is the single most important ablation and is run at
  exactly one operating point.

### 4.6 Metric design (composite is Goodhart-bait in its current form)
- **The composite is not on a comparable scale across regimes.** `dppl_norm = (PPL−base)/base`
  is unbounded; a PPL blow-up to 5M yields composite ≈ −53,000. So the composite is, in the
  unsafe regime, *just a coherence-explosion detector*; in the safe regime it is
  `behavior − tiny penalties`. It does not meaningfully trade off axes — it is dominated by
  whichever axis is currently exploding.
- **The fingerprint invariant is violated in practice.** The AxBench drivers log
  `composite = round(mean_delta, 4)` — overloading the `composite` field with a *completely
  different quantity* (a real-minus-shuffled delta) than `eval.composite()` computes, while
  still stamping fingerprint `a9001e87087e`. Two different numbers wear the same fingerprint;
  this defeats the integrity guarantee the project advertises.
- **Behavior efficacy uses a logistic squash with a hand-set scale** (`_CONCEPT_LOGIT_SCALE=8`),
  another unjustified constant that shapes the headline axis.
- **Safety is priced linearly (λ_safe·CR)** but harm is not linear — one successful jailbreak
  on a dangerous intent is worse than the average suggests; a safety metric should be
  worst-case / per-category, not a mean compliance rate.

### 4.7 Reproducibility
- **Strong on harness, weak on results.** Offline tests, fingerprinting, caching, and pinned
  data are excellent. But: the real-model results depend on a **local judge whose weights and
  exact decode settings gate every number**, the AxBench split (`2b/l20`) is hard-coded, and
  the headline "alpha is the only knob" rests on single-model/single-layer slices.
- **The judge cache is keyed by text+rubric**, so a judge-model swap silently changes results
  unless the rubric version is bumped — easy to mis-handle across a 70-hypothesis program.
- **No human-agreement study** anywhere — the entire behavior axis is machine-judged with no
  human anchor.

---

## 5. Twenty-three specific, actionable improvements (P0 = blocker, P1 = major, P2 = polish)

**P0-1.** Replace the substring `is_refusal()` safety instrument with a **high-precision harm
classifier** for all safety claims: **Llama-Guard-3-8B** (4-bit on the 4090) and/or the
**JailbreakBench `Llama-3-70B`/GPT-4 judge protocol**. Report **human agreement on ≥100
held-out items** (target Cohen's κ ≥ 0.8, agreement ≥ 0.9). *AUC 0.68 cannot back a safety
decision.*

**P0-2.** **Wire the real safety benchmarks** the goal names: JailbreakBench (behaviors +
its judge), **StrongREJECT** (its graded 0–1 rubric — this is the right primary metric, not
binary compliance), **HarmBench**, **AdvBench** (harmful behaviors), and **XSTest** (250
benign + 200 unsafe, for over-refusal). Ship loaders in `src/steering/` with pinned versions;
delete the n=10 `jailbreak_mini.json` as a *claim* instrument (keep only for smoke).

**P0-3.** **Implement the actual conditional method** end-to-end: a `cast_steer` dispatch that
reads a condition vector at L_c and applies the gated behavior write at L_b in **one forward
pass** (the design in E9 §5.2). Without this, there is no paper. Target: Gemma-2-2B-it.

**P0-4.** **Pre-register a minimum meaningful effect (MME)** for every axis before the next
sweep (e.g., "StrongREJECT score drop ≥ 0.15 absolute vs the strongest baseline at ≤2pp MMLU
loss and ≤5% XSTest over-refusal increase"). Gate verdicts on **effect ≥ MME AND p<α**, never
p alone. This neutralizes the n=500 triviality problem.

**P0-5.** **Apply Holm–Bonferroni across the whole forking-paths family.** Pass
`family_pvalues` into `rigor_report` for every confirmation run, where the family enumerates
all hypotheses × operating points actually tried. Stop letting leg 4 pass vacuously. Log the
family size in each report.

**P0-6.** **Add the baselines that decide the paper:** (a) few-shot **prompting** and (b)
**system-prompt refusal** (AxBench shows these beat steering — you must beat them), (c)
**unconditional steering**, (d) a **Llama-Guard input-classifier router**, (e) a published
safety-steering method — **CAST** (arXiv:2409.05907) and **Arditi refusal-direction**
(arXiv:2406.11717), ideally **RepE/circuit-breakers**. Report all on the same axes.

**P0-7.** **Fix the composite-field integrity violation.** The AxBench drivers must NOT write
a real-minus-shuffled delta into the `composite` field under fingerprint `a9001e87087e`. Use a
distinct field (`direction_delta`) and only stamp the fingerprint on genuine
`eval.composite()` outputs. One fingerprint must map to one formula.

**P1-8.** **Re-specify the ordinal gate** for the concept-as-replicate design: it should test
the **per-concept paired delta** (e.g., fraction of concepts with real>shuffled, or a sign
test), not `worst(real_concept) > best(shuffled_concept)` across heterogeneous concepts. The
current gate measures concept difficulty variance.

**P1-9.** **Make the safety metric worst-case and per-intent.** Report compliance/StrongREJECT
**per harm category** (≥5 intents: illegal, weapons, self-harm, cyber, harassment, …) and the
**max** (worst category), not just the mean. Multi-intent is in the goal; a mean hides the
dangerous tail.

**P1-10.** **Build the XSTest over-refusal arm as a first-class axis** with CIs, and report the
**joint operating point** (harmful-compliance ↓ AND benign-refusal flat) as a Pareto curve —
the core deliverable of a *conditional* method. One number on each axis is not enough.

**P1-11.** **Run seed variance for the real pipeline.** Even with greedy decoding, vary
extraction subsample, judge order/batching, and contrast-pair sampling across **≥7 seeds** for
any confirmation; report `seed_noise_band` and require the effect to exceed 2σ_seed (the
contract's own rule, currently unenforced on AxBench).

**P1-12.** **Calibrate the safety axis to a true alpha=0 baseline.** The non-zero baseline
compliance is an instrument artifact; report **Δcompliance vs the unsteered model under the
same judge**, and validate that the unsteered Gemma-2-2B baseline is near-0 on JailbreakBench
with the real judge before any steering claim.

**P1-13.** **Ablate the composite weights.** Provide a sensitivity table: do verdicts survive
λ_safe ∈ {1,2,4}, λ_coh ∈ {0.25,0.5,1.0}? If conclusions flip, the composite is not
defensible and must be replaced with explicit per-axis Pareto reporting.

**P1-14.** **Bound the coherence tax** so the composite is comparable across regimes (e.g.,
`dppl_norm := tanh((PPL−base)/base)` or cap at 1.0), OR drop the single composite for the
paper and report the **5-axis Pareto frontier** directly. A −53,000 composite is not a metric.

**P1-15.** **Run the gate-on vs gate-off ablation** as the central experiment: identical
behavior vector, identical alpha, only the condition gate toggled. This is the one result that
isolates "conditional" from "steering" and must dominate the paper.

**P1-16.** **Validate the behavior judge against humans**, not just against keyword-soup spot
checks. Collect ~150 human ratings, report Pearson/Spearman/κ via the existing
`calibration_agreement`, and only use the judge above a pre-set agreement floor.

**P1-17.** **Power the screening tier honestly.** Enforce the project's own rule in code:
n=3 → SCREENING only (no p-value claims), n≥7 → EVALUATION. Have the runner refuse to emit
"SUPPORTED/DIRECTIONAL" at n<7.

**P1-18.** **Multi-intent OR-gating (E11) must be tested**, not just single-intent. Show
coverage scales across ≥5 condition vectors while XSTest over-refusal stays flat — the actual
"multi-intent" claim in the goal.

**P1-19.** **Add a red-team / robustness arm** (the goal mentions HarmBench-style adversarial
suffixes): reproduce GCG/AutoDAN or the corpus's "20-vector universal attack" against the
guarded method and show it is neutralized at the final rung. A safety claim without an attacker
is not a safety claim.

**P2-20.** **Scale past 2B for the final claim.** Small-model fragility (270M can't express
concepts at all) confounds everything; run the final method on **Gemma-2-9B-it** (4-bit fits
16GB) so reviewers don't dismiss it as a toy-model artifact.

**P2-21.** **Report effect sizes with practical anchors** everywhere (Cliff's δ or % of
control captured, which you already computed — "97% generic" — promote that to the primary
reported number alongside p).

**P2-22.** **Pin and version every external dependency for results** (AxBench split, judge
model revision, transformers version) in a per-experiment manifest, and bump the judge cache
namespace on any judge change to prevent silent staleness across the 70-hypothesis program.

**P2-23.** **Separate "negative result about steering" from "safety method" in the write-up.**
The honest AxBench negative ("alpha is the only knob; direction is ~97% generic; prompting
likely wins") is a **legitimate short paper / workshop contribution on its own** — frame it as
such rather than as a way-station to a method that isn't built. It is the strongest scientific
asset the repo currently has.

---

## 6. One-paragraph guidance to the authors

You have built the harness and the honesty culture that most steering papers lack — and your
own rigor caught your headline effect being 97% generic. Believe that result. Right now you do
not have a safety method (P0-3), a safety benchmark (P0-2), or a safety judge (P0-1); you have
a substring matcher on 10 prompts and a composite field that sometimes holds a different
number than its fingerprint promises (P0-7). The path to acceptance is narrow but real:
implement one conditional method, evaluate it on StrongREJECT + JailbreakBench + XSTest with
Llama-Guard-3 and human-validated agreement, beat the prompting and CAST baselines on a joint
harmful-compliance/over-refusal Pareto frontier with a pre-registered MME and Holm correction
across the whole search, and red-team it. Until then this is a strong-reject against the stated
goal — and, paradoxically, a perfectly good negative-results workshop paper if you choose to
write the one you actually proved.

*Internal QA pass — independent external review pending (the reviewer and the codebase author
share a model family; this verdict carries the same-family circularity disclosure mandated by
CLAUDE.md §14).*
