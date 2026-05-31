# A Pre-Registered Autoresearch Harness for Activation Steering of Small Language Models: Methods, a Goodhart-Resistant Composite, and First Screening Results

*Anonymous submission — methods / reproducible-harness / screening-results contribution.*
*Composite-metric fingerprint: `a9001e87087e` (SHA-256[:12] of the frozen formula string).*

---

## Abstract

We present an autonomous, pre-registered **research harness** for the activation /
conditional steering of small language models, together with the **first screening
observations** it has produced. The contribution is explicitly a *methods and
reproducible-infrastructure* contribution, **not** a steering-efficacy claim. The
harness encodes (i) a single-axis Karpathy-style keep/discard loop over a 12-axis
intervention taxonomy, (ii) a five-rung cost-ordered benchmark ladder
(UNIT→SMOKE→DEV→STANDARD→FULL) in which the same five measurement axes are scored
at every rung, (iii) a SHA-256-**fingerprinted, Goodhart-resistant composite** that
prices behavior, capability, coherence, safety, and selectivity simultaneously so a
method cannot win by sacrificing one axis, and (iv) a statistical rigor floor that
hard-classifies n≤3 as *screening* and n≥7 (with paired Wilcoxon, bootstrap CI,
Holm-Bonferroni, an empirical seed-noise band, and an ordinal gate) as *evaluation*.
A hostile-but-fair internal ICML-style review of an earlier build found that the
behavior axis was a **circular projection-of-the-edit-onto-itself proxy** and that
the safety axis was a **hardcoded constant**; we report the non-circular
replacements (generation-based concept-incorporation scoring; real steered-generation
refusal scoring) and re-run the affected screen.

**Limitations are central, not incidental, and bound every number below.** All
quantitative results are **screening only (n=1)**, on **synthetic mini-datasets**
(10–20 hand-written items, *not* real AxBench/MMLU/WikiText), on a **single behavior**
("ocean"), with **no human-calibrated judge**, on **sub-0.5B models**
(Gemma-3-270m-it and Qwen2.5-0.5B-Instruct). They establish mechanism *direction*,
never magnitude, and no claim in this paper is external-ready. Within that scope we
report three directional screening observations: a **super-linear coherence cliff**
in steering coefficient α with an identifiable knee, predicted by a cheap
behavior-free off-shell-displacement geometry probe (N17); **cross-model robustness**
of DiffMean≈PCA-top-1 alignment (E4: cos≈0.994–0.996); and **scale-dependent
fragility** (E27: the 270m model leaves the manifold *earlier and harder* than the
0.5B model, with behavior never improving under steering). We enumerate the exact
experiments required before any of these may be promoted to an external claim.

---

## 1. Introduction

Activation steering — adding or rotating a direction in a transformer's residual
stream to control a behavior — is attractive because it is training-free, cheap, and
interpretable. It is also **inherently multi-objective**: a steering edit that raises
a target behavior can simultaneously cost capability, wreck coherence, break safety
refusals (the "Rogue Scalpel" failure mode), or induce over-refusal on benign inputs.
A single scalar "efficacy" number therefore systematically over-states a method, and
the autoresearch literature is littered with pipelines that "win" a metric by quietly
sacrificing an unmeasured axis.

This paper does **not** claim a new steering method or a new steering result. It
contributes:

1. **A pre-registered autoresearch harness** (Section 3) that turns each experiment
   into a falsifiable, citation-gated, single-axis perturbation of a champion config,
   authored through a 7-step ritual whose pre-run fields are mechanically gated
   against placeholders, and promoted along a cost-ordered ladder.
2. **A Goodhart-resistant, fingerprinted composite** (Sections 3, 6 of CLAUDE.md)
   that prices all five axes with one-sided penalties, so a gibberish-but-"safe" run
   cannot win because the coherence tax dominates. The formula string is frozen and
   its SHA-256[:12] fingerprint (`a9001e87087e`) appears in every reasoning entry,
   ledger, and dashboard footer; editing it to crown a favored row is a defined
   protocol violation.
3. **An honest measurement-validity analysis** (Section 4): an internal
   ICML-area-chair review found the behavior axis was circular and the safety axis a
   constant; we document the circularity, the non-circular fixes, and what they
   revealed that the proxy had hidden.
4. **First screening results** (Section 5), correctly scoped as n=1 directional
   observations on synthetic data, with confidence intervals deliberately *withheld*
   because n=1 cannot support them.

We frame the work as an infrastructure-and-screening artifact precisely because, by
its own rigor floor, it cannot yet be anything more. The value is the *process* and
the *honest scoping*, not the (deliberately gated) numbers.

---

## 2. Related work

**Contrastive activation steering.** Contrastive Activation Addition (CAA; Rimsky et
al., 2023, *Steering Llama 2 via Contrastive Activation Addition*, arXiv:2312.06681)
establishes the DiffMean-of-contrast-pairs recipe we use as the default direction
source. Inference-Time Intervention (ITI; Li et al., 2023, arXiv:2306.03341) steers
along probe-identified directions at selected heads and motivates our
Fisher/separability-based layer selection (E2). Our harness treats both as baselines
to be priced on all five axes, not as targets to reproduce uncritically.

**Conditional / gated steering.** CAST (Conditional Activation Steering; Lee et al.,
2024, arXiv:2409.05907) gates the steering edit on an activation-space condition,
which we adopt as a *meta-layer* that stacks on other methods rather than a peer
method (the basis for hypotheses E9–E16, N2). The DiffMean≈PCA-top-1 alignment we
screen (E4) is motivated by CAST's need for a cheap condition vector.

**Safety of steering.** Rogue Scalpel (2025, arXiv:2509.22067) shows that steering
edits can break refusal behavior and exhibits a universal-attack construction; this
makes safety a *first-class, dominantly-weighted* axis in our composite (λ_safe=2.0)
and an automatic-DISCARD gate, and motivates the off-manifold framing of safety leaks
in Sections 4–5.

**Evaluation honesty.** AxBench (Wu et al., 2025, arXiv:2501.17148) provides the
apples-to-apples protocol comparing steering against prompting/probing/SAE baselines
and reports that prompting and finetuning often beat steering — which is precisely why
our review flags the *absent prompting baseline* as a blocker (Section 7) and why we
make no efficacy claim.

**Geometry of steering.** Manifold Steering (2026, arXiv:2605.05115) and the
non-identifiability of steering vectors (2026, arXiv:2602.06801) motivate our
geometry leading-indicators (off-shell displacement Δ‖h‖, effective-rank, norm
budget) and our caution that behaviorally-equivalent vectors can differ in collateral
damage. We use off-shell displacement (N17) as a cheap, behavior-free cliff predictor
and report its empirical lockstep with perplexity in Section 5; its *predictive
validity is itself a hypothesis under test*, a circularity we disclose in Section 4.

---

## 3. The autoresearch harness

The harness operationalizes a single invariant: *always start from the current best
config; change exactly one thing; keep iff the composite improves at matched
coherence with no axis regressing past its gate; revert otherwise.*

### 3.1 The 12-axis intervention taxonomy

Every experiment perturbs exactly one of twelve orthogonalized axes: A1 WHERE (site/
layer), A2 WHAT (direction), A3 HOW-MUCH (coefficient), A4 HOW (operation: add /
norm-preserving rotate / project-out), A5 WHEN (condition/gate), A6 WHICH-TOKENS
(span), A7 HOW-DERIVED (source: DiffMean / PCA / SAE), A8 GEOMETRY (path), A9 METRIC
(space), A10 IDENTIFIABILITY (gauge), A11 DYNAMICS (trajectory), A12
BASIS/SUPERPOSITION. The taxonomy tells the loop which axes are safe to perturb
independently and which combinations stack versus compete.

### 3.2 The five-rung ladder

| Rung | Nickname | Cost/run | Proves | Gate to next rung |
|---|---|---|---|---|
| 0 | UNIT | seconds | plumbing works | vector changes logits; state restores exactly |
| 1 | SMOKE | 1–3 min | right direction | monotone effect + bounded PPL + no safety leak |
| 2 | DEV | 10–20 min | generalizes a little | beats baseline on held-out concepts at matched coherence |
| 3 | STANDARD | 1–3 h | real result | Pareto-dominates prior method (no axis regresses) |
| 4 | FULL | half-day+ | publication | full multi-axis win + ablations + red-team neutralized |

The **same five axes** are scored at every rung; only size and realism grow. A method
may not consume rung *k+1* compute until it clears rung *k*'s gate, and a regression
at any rung demotes it with a logged `failure_reason`. This is experimental economics:
never run an expensive benchmark to find a bug a cheap one would catch.

### 3.3 The Goodhart-resistant fingerprinted composite

The composite (frozen string, fingerprint `a9001e87087e`) is

```
composite = behavior_efficacy
          − λ_cap     · max(0, mmlu_drop_pp)            (capability tax)
          − λ_coh     · max(0, dppl_norm)               (coherence tax, PPL)
          − λ_coh_rep · max(0, repetition_rate)         (coherence tax, repetition)
          − λ_safe    · compliance_rate                 (safety leak — Rogue Scalpel)
          − λ_sel     · max(0, harmless_refusal_rate)   (over-refusal / selectivity)
          − λ_geo     · max(0, offshell_displacement)   (off-manifold leading indicator)
```

with pinned weights λ_cap=1.0, λ_coh=0.5, λ_coh_rep=0.5, **λ_safe=2.0** (dominant),
λ_sel=1.0, λ_geo=0.25. The one-sided `max(0,·)` penalties mean an axis can only hurt,
never inflate, the composite. A run that emits gibberish scores "safe" on harm but
fails coherence; the coherence tax then dominates and it cannot win — a property the
unit tests assert directly. Fingerprinting prevents *mid-project* tampering; it does
**not** make the *initial* λ choice defensible, an honest limitation we restate in
Section 6 (no λ-sensitivity analysis has yet been run).

### 3.4 The 7-step ritual

Each experiment authors a pre-run reasoning entry — **Diagnose** (≥60 words, naming
the specific failure mode and referencing a prior experiment), **Cite** (a real arXiv
paper in full format motivating the change), **Hypothesize** (the residual-stream
mechanism), **Predict** (a numeric range on the composite and ≥1 sub-metric, stored
*before* the run) — and, after execution, a post-run entry — **Analyse** (actual vs
predicted, verdict KEEP/DISCARD/NEAR-MISS) and **Checkpoint**. The runner refuses to
fabricate pre-run fields; a missing diagnosis/citation/hypothesis/prediction is a
protocol violation, not a placeholder.

### 3.5 The screening → hill-climb → evaluation funnel

(1) **Screen** one config per hypothesis at a documented baseline (cheap, n≤3).
(2) **Hill-climb** a surfaced candidate by coordinate descent over the steering cube
(layer × α × source × operation × span) × seed, 20–25 trials, strict-`>` champion
rule. (3) **Confirm** at n≥7 with the full rigor contract before any external claim.
n≤3 is *screening, full stop*; n=3 cannot reach p<0.05 under a paired Wilcoxon
signed-rank test, so calling an n=3 result a "winner" is forbidden. Reclassifying a
loser as "screening" after the fact is named HARKing and is a blocker.

---

## 4. Measurement validity

The most important methodological lesson came from an internal, hostile-but-fair
ICML-area-chair review (`audits/ICML_REVIEW.md`) of an earlier build. It found two
crux defects that would have silently validated several hypotheses.

### 4.1 The projection-proxy circularity (the crux)

The original behavior scorer measured the change in the mean projection of layer-L
activations onto the unit target vector v̂, between a steered and an unsteered run. But
the steered run *is* h ← h + α·v with v̂ = v/‖v‖, so the measured quantity reduces to

```
Δproj = mean[(h + α·v)·v̂] − mean[h·v̂] = α·(v·v̂) = α·‖v‖,
```

a **deterministic, monotone function of the very edit being applied**. It cannot fail
for any non-degenerate v and positive α: it measures *that addition adds*, not that
behavior changed, and saturates to 1.0 by α=1 under the logistic squash. The SMOKE
"monotone effect" gate was therefore trivially passed by a tautology, and E1/E3/E7/E27
— all defined in terms of "behavior" — would have been satisfied by the proxy. No
efficacy, monotonicity, or Pareto claim can rest on this scorer.

### 4.2 The non-circular generation-based fix

The replacement, `generation_behavior_scorer`, scores **concept incorporation in
real generated text**: it counts concept-lexicon word stems in the model's *output*,
steered versus unsteered, never the injected vector. Re-running the affected screen
with this instrument exposed two findings the proxy had completely hidden: behavior
is **non-monotonic** — it peaks near α≈1 and then *declines* as over-steering destroys
the text that would express the concept (the proxy reported a flat saturated 1.0) —
and the composite optimum coincides with the behavior peak. A residual circularity
remains and is disclosed: the concept lexicon is derived from the same contrast pairs
that built the vector, so a model that parrots pair vocabulary can still inflate the
score; a held-out lexicon or an independent cross-family judge is required and is
listed in Section 7.

### 4.3 Real safety generation

The original safety axis returned the constant string `"I can't help with that."` for
every harmful prompt, so `compliance_rate ≡ 0` and `harmless_refusal_rate ≡ 0` *by
construction* — the dominant λ_safe penalty and the auto-DISCARD gate could never fire.
The replacement generates with the real model on the (synthetic) harmful prompts and
scores SAFE/UNSAFE with a rule-based refusal detector on the model's own text. On the
re-run this made the Rogue-Scalpel direction *measurable*: compliance rises with α
(Section 5). The offline FakeLM path retains a single documented refusal placeholder
that is **tagged** (`safety_real=False`) so a stubbed value can never be mistaken for a
measurement; real rows carry `safety_real=True`. The refusal detector is itself a
same-family rule-based judge on the model's own generations — a circularity disclosed
in Section 6 and gated in Section 7.

---

## 5. Screening results

> **Every number in this section is SCREENING (n=1), on synthetic mini-data, single
> behavior ("ocean"), no calibrated judge, on sub-0.5B models. Confidence intervals
> are withheld because n=1 cannot support them. These are directions, not
> magnitudes, and none is an external claim.** Sources: `EXPERIMENT_LEDGER.md`
> (exp#2–19), `ideas/30_alpha_coherence_cliff/results.md`, `FINDINGS.md` (S-1..S-4).

### 5.1 The α coherence cliff and its geometry predictor (E3, N17)

On Qwen2.5-0.5B-Instruct @L21 (max-Fisher), with the **non-circular** generation
behavior scorer and **real** safety generation (exp#10–14):

| α | behavior (generation) | PPL | compliance_rate | composite |
|---|---|---|---|---|
| 0 | 0.500 | 48.9 | 0.30 | −0.107 |
| 1 | **0.694** (peak) | 58.7 | 0.30 | **−0.073** (best) |
| 2 | 0.526 | 89.0 | 0.30 | −0.717 |
| 4 | 0.494 | 293.6 | 0.60 | −3.597 |
| 8 | 0.346 | 3 787 | 1.00 | −40.602 |

Two screening observations the circular proxy had hidden: (1) behavior is
**non-monotonic**, peaking at α≈1 then declining (0.69→0.53→0.49→0.35) — over-steering
destroys the behavior itself, not just coherence; and (2) steering **compromises
safety** — real compliance rises 0.30→0.60→1.00 as α goes 0→4→8 (the Rogue-Scalpel
effect, now measured). The composite peaks at α≈1, the behavior/coherence sweet spot.
PPL grows super-linearly (+20%→+82%→6×→77× across α=1→2→4→8), with a knee at α≈1–2.

The earlier superseded projection-proxy run (exp#2–9, retained for provenance)
exhibited the same PPL cliff shape and showed that **off-shell displacement Δ‖h‖ rises
monotonically in lockstep with PPL** (0.033→0.102→0.318→0.92→…→3.88 across α=1→24) —
i.e. the cheap, behavior-free geometry probe is a valid *leading indicator* of the
cliff (N17 supported as a screening observation). On the synthetic `mmlu_tiny`
tripwire the capability drop stayed <0.5pp even at α=24; we read this as a **limitation
of the 20-item synthetic tripwire**, not as evidence of capability robustness.

### 5.2 Cross-model robustness of DiffMean≈PCA alignment (E4)

E4 predicts cos(DiffMean, PCA-top-1) ≥ 0.95, which would let the cheaper DiffMean
substitute for PCA in CAST-style gating. At the max-Fisher layer we measure
**cos = 0.996** on Qwen2.5-0.5B @L21 and **cos = 0.994** on the real Gemma-3-270m-it
@L12 (exp#15–19). The alignment holds across two architectures — the most robust of
the four screening observations, though still n=1, single behavior, synthetic pairs.

### 5.3 Scale-dependent fragility on real Gemma (E27, E3/N17)

On the real target-family model Gemma-3-270m-it @L12 (Fisher=30.6), generation
behavior + real safety (exp#15–19):

| α | behavior | PPL | ΔPPL_norm | CR | off-shell Δ‖h‖ | composite |
|---|---|---|---|---|---|---|
| 0 | 0.500 | 90.2 | 0.00 | 0.80 | 0.000 | −1.107 |
| 1 | 0.438 | 149.2 | +0.65 | 0.80 | 0.021 | −1.602 |
| 2 | 0.319 | 322.3 | +2.57 | 0.90 | 0.057 | −2.889 |
| 4 | 0.217 | 2 775 | +29.8 | 1.00 | 0.175 | −16.91 |
| 8 | 0.211 | 141 578 | +1568 | 1.00 | 0.535 | −786.4 |

The 270m model is **more fragile** than the 0.5B model (supporting E27, that smaller
models exit the manifold more easily): its cliff is at **α≈1** (PPL already +65% at
α=1) versus Qwen's α≈2, and its behavior **never improves** under steering
(0.50→0.44→0.32→0.22, monotone decline) — there is no clean steering window at L12; the
model leaves the manifold *before* the concept can be cleanly injected, whereas
Qwen-0.5B had a behavior peak (0.69) at α=1. Off-shell Δ‖h‖ again tracks PPL
super-linearly (N17 confirmed cross-model). Safety: baseline CR=0.80 rising to 1.00
under steering — but this high baseline reflects a barely-safety-tuned 270m model
complying with synthetic harmful prompts even unsteered; the **direction** (steering↑
⇒ CR↑) is the signal, the **magnitude is not transferable**.

### 5.4 What the screen does and does not establish

The screen establishes a consistent *mechanistic direction* across two architectures:
a super-linear coherence cliff with an identifiable knee, predicted by a behavior-free
geometry probe (N17); robust DiffMean≈PCA alignment (E4); and smaller-is-more-fragile
scaling (E27). It establishes **no magnitude**, no efficacy, and no safety claim:
n=1, synthetic mini-data, one behavior, no calibrated judge, sub-0.5B models. Per the
program's own rigor floor these live in `EXPERIMENT_LEDGER.md` as screening
observations and are **forbidden from `FINDINGS.md`** until the Section-7 contract is met.

### 5.5 Autonomous campaign findings (C1–C3)

An overnight autonomous run (`scripts/campaign_sweep.py`, load-once grid sweeps on
Gemma-3-270m) produced three further screening results, each pre-registered and logged:

- **C1 — E2 is falsified on this setup.** A layer sweep (exp#20–27, α=2) gives
  Spearman(Fisher ratio, behavior) = **+0.14 (p=0.74)**, far below E2's pre-registered
  ≥0.7. The max-Fisher layer (L12) is *not* the best steering layer; L16 yields higher
  behavior at lower perplexity. Linear separability does not predict steering efficacy —
  a screening-level corroboration of the controllability≠interpretability theme (N8/E37).
- **C2 — the geometry predictor is architecture- and layer-independent (N17, N5).**
  Pooling 23 real steered rows across two models, eight layers, and the full α range,
  off-shell Δ‖h‖ predicts log-perplexity at Spearman **+0.71** (Pearson 0.90), and a
  single law `log PPL = 5.40 + 2.87·Δ‖h‖` fits with **R²=0.81**. This is the program's
  strongest screening result: a cheap, behavior-free predictor of the coherence cliff
  that generalizes across architecture and depth.
- **C3 — a productive negative result that refined the instrument.** An add-vs-rotate
  comparison surfaced that (a) the coefficient α is incommensurable across operations
  (rotation reads α as radians, so α≥1 is catastrophic), and (b) Δ‖h‖ is a *radial-only*
  predictor — a norm-preserving rotation registers Δ‖h‖≈0 while perplexity explodes.
  This motivated adding an **angular-displacement** metric `1−cos(h,h')` (the
  Cylindrical-Representation radial×angular split, N16), now logged per run, so N17 can
  be re-tested with the complete radial+angular displacement.

These illustrate the harness working as intended: pre-registered hypotheses are
*falsified* (E2), *supported* (N17/N5), or turned into *instrument refinements* (C3) —
without any overclaiming.

---

## 6. Limitations

These are not caveats appended to results; they *are* the current state of the
evidence.

1. **n=1 throughout.** Every result is a single seed. No confidence interval, no
   significance test, no noise band is reported because none is supportable. By the
   program's rigor floor this is *screening*, never *evaluation*.
2. **Synthetic mini-data.** The wired slices are 10 hand-written contrast pairs
   (`axbench_mini`), a 20-item synthetic MCQ tripwire (`mmlu_tiny`, *not* MMLU), 10
   synthetic passages (`wikitext_ppl_mini`, *not* WikiText), and 8–10 placeholder
   safety prompts. Construct validity fails: these measure toy items, not the
   constructs the axes name.
3. **Single behavior.** All steering targets one concept ("ocean"). The DEV-rung
   "generalizes on held-out concepts" gate cannot even be attempted — there are no
   held-out concepts.
4. **No calibrated judge.** Behavior and safety are scored by same-family rule-based
   instruments on the model's own text; no human-calibrated LLM judge exists yet.
5. **Capability tripwire, not MMLU.** The offline capability axis is a deterministic
   forward-pass corruption sensor, not an accuracy measurement; its λ_cap term is
   currently driven by a tripwire, not capability.
6. **Sub-0.5B models.** Real runs are on Gemma-3-270m-it and Qwen2.5-0.5B; the
   constitution's standard model (Gemma-2-2B-it) has not been exercised end-to-end.
7. **Unjustified λ-weights.** The composite weights are pinned and fingerprinted but
   *not derived*; the axes are not commensurate (an unbounded ΔPPL term can dominate),
   and no λ-sensitivity / champion-ordering-robustness analysis has been run.
8. **Geometry-probe circularity.** Off-shell displacement is *priced in the composite*
   while its predictive validity (N17/N20) is itself a hypothesis under test — the
   axis being tested is partly baked into the judge.
9. **Same-model-family circularity.** Vectors are extracted from and injected into the
   same model family that also judges the output; the concept lexicon derives from the
   extraction pairs. External validity requires a cross-family judge and a held-out
   lexicon.
10. **4-bit transfer untested.** Quantization↔fp16 invariance (E5) is a precondition
    for the "4090 results transfer" story and has not been verified.

---

## 7. Required experiments before external claims

These are taken directly from the internal ICML review and are *ordered*, each gating
the claim it supports:

1. **Independent behavior judge** — replace the projection proxy with an LLM-as-judge
   or AxBench scorer on real generated text, validated against a human-annotated slice
   (target ≈94% judge precision). No efficacy / monotonicity / Pareto claim ships
   otherwise. (Generation-based scorer landed; calibration pending.)
2. **Real safety + selectivity** — generate on real JailbreakBench (100 prompts, 10
   categories) and XSTest, judge SAFE/UNSAFE with a calibrated judge, confirm baseline
   CR≈0%, and demonstrate the λ_safe penalty and auto-DISCARD can fire. (Real
   generation landed; real benchmarks + calibrated judge pending.)
3. **Real datasets** — real AxBench concept set with a held-out-concept split, real
   MMLU (≥500), real WikiText-103 perplexity; retire the synthetic slices to
   UNIT/SMOKE only.
4. **n≥7 with the full rigor contract** — paired Wilcoxon + 10k-bootstrap CI +
   Holm-Bonferroni + empirical 2σ_seed band + ordinal gate, on the pre-registered
   evaluation split.
5. **Prompting baseline (and random-direction / mean-ablation controls)** — the
   AxBench apples-to-apples comparison; a steering "win" is only a contribution
   relative to a tuned prompt.
6. **Gemma-2-2B reproduction + λ-sensitivity** — reproduce on the standard model (not
   FakeLM, not 270m-only), and show the champion ordering survives a λ-perturbation
   and that E5 (4-bit↔fp16) holds.
7. **Held-out-concept generalization + shuffle-test negative control** — confirm the
   method works on concepts not used in extraction and *fails* under label/condition
   shuffling.

Until items 1–5 are real, the program is — by its own area-chair review — a
**Reject (3/10) as a source of steering claims** and a **Borderline (6/10) as a
methodology-and-harness contribution**. This paper claims only the latter.

---

## 8. Reproducibility

**Code & rigor gates.** `ruff check src/steering tests` (clean), `mypy src/steering
--ignore-missing-imports` (clean), `pytest tests/` (46 passing) — the Rung-0 plumbing,
mechanism-asserting hook/extract/geometry tests, the Goodhart composite tests, and the
markdown-leak dashboard tests. The composite fingerprint `a9001e87087e` is asserted
stable.

**Model & license note.** Gemma is a gated model. Reproduction requires accepting the
Gemma license on Hugging Face and authenticating (`huggingface-cli login`); the token
is read from the environment and is never committed (`.gitignore`, `.hf_token`). Real
runs used `models/google/gemma-3-270m-it`; the standard rung targets
`google/gemma-2-2b-it`. Qwen2.5-0.5B-Instruct is the non-gated bring-up surrogate.

**Determinism & commands.** Greedy decoding for all gates; seeds threaded through
`random`/`numpy`/`torch`/CUDA via `_set_seed`. A representative real-Gemma screening
run:

```powershell
$env:PYTHONPATH = "src"
python scripts/hf_fetch.py google/gemma-3-270m-it          # after accepting the license
$env:E3_MODEL = "models/google/gemma-3-270m-it"
python ideas/30_alpha_coherence_cliff/run_e3.py            # the α-sweep screen
```

The offline FakeLM path (`--model fake`) runs the full reasoning→runner→ledger→
dashboard loop without any model download for plumbing verification. Every artifact —
`autoresearch_results/experiment_log.jsonl` (append-only), `best_config.json`, the
three-tier dashboard (`dashboard/` + `docs/dashboard/` mirror) — is regenerated and
checkpointed each milestone. All inherited corpus numbers are tagged `[NEEDS
VERIFICATION]` until reproduced here.

---

## 9. Conclusion

We contribute a pre-registered autoresearch harness for activation steering whose
value is its *discipline*: a single-axis champion loop, a cost-ordered ladder, a
fingerprinted Goodhart-resistant composite that prices all five axes, a statistical
rigor floor that hard-classifies screening versus evaluation, and — crucially — an
internal review that caught and corrected a circular behavior proxy and a stubbed
safety axis before they could manufacture false results. The first screening run, on
synthetic data at n=1 on sub-0.5B models, yields three *directional* observations
(a geometry-predicted super-linear coherence cliff; cross-model DiffMean≈PCA
alignment; smaller-is-more-fragile scaling) that motivate, but do not constitute,
findings. We deliberately ship **zero external claims**: each is gated on the
enumerated required experiments — real AxBench with a calibrated judge, real safety
benchmarks, n≥7 with the full rigor contract, a prompting baseline, and Gemma-2-2B
reproduction. The contribution is the harness and the honesty, not the numbers.

---

*Internal QA pass — independent external review pending. This document and the program
it describes were authored, implemented, and audited largely by agents sharing one
model family; per the program's own circularity-disclosure rule, all internal
verdicts are a useful filter, not an external seal of approval.*
