# E50 — Minimal SOTA Stack: Gate + 3 Ortho Vectors + Norm Cap + DoLa

> **One-line claim:** A minimal stack recipe consisting of a CAST condition
> gate + 3 Gram-Schmidt-orthogonalised safety vectors + norm-budget cap +
> DoLa decoding reproduces SOTA multi-property safety control on Gemma-2-2B
> within a 24 GB single-GPU budget, with each component contributing
> measurably to the aggregate score.
>
> **Block:** F — Robustness, safety, and evaluation (E41-E50).
> **Primary axes:** A5 (WHEN) + A2 (WHAT) + A3 (HOW MUCH) + A4 (HOW).
> **Implementation status:** `o planned / UNTESTED`.

---

## In Plain English

**What we're testing, simply:** We try to assemble the smallest "recipe" of
parts that together give top-tier safety control on a small model running on one
ordinary graphics card — and we check that every part actually pulls its weight.

**Key terms (defined here):**
- **Language model** — an AI that writes text one word at a time.
- **Steering** — changing the model's behavior by editing its internal state
  mid-sentence, without retraining.
- **Steering vector** — the nudge we add to push toward (or away from) a
  behavior.
- **Residual stream** — the model's running internal scratchpad; the nudges go
  here.
- **Layer** — one of the model's stacked processing steps.
- **alpha / strength** — how hard we push.
- **DiffMean** — the simplest nudge recipe: average internal state on "yes"
  examples minus "no" examples. No training.
- **The stack (the recipe)** — several parts combined:
  - **The gate** — refuse only when a request is genuinely harmful.
  - **3 ortho vectors** — three safety nudges aimed in non-overlapping
    directions so they don't smother each other.
  - **Norm cap** — a limit on how far the combined nudge can move the model's
    state, so it never breaks the output.
  - **DoLa decoding** — a tweak to how the model picks each next word that helps
    keep answers truthful and on-track.
- **SOTA** — "state of the art," i.e. matching the best published results.
- **Coherence** — whether the text stays fluent and sensible.
- **The guard / red-team** — the defensive layers above, and the deliberate
  attacks used to test them.

**Why we're doing this (the point):** Can a lean, affordable recipe — runnable on
a single consumer GPU — reach the same safety quality as heavyweight setups, and
is each ingredient truly necessary?

**What the result would mean:** If the small stack hits top-tier safety and every
part contributes, we have a cheap, reproducible recipe others can run. If a part
adds nothing, we drop it; if the whole thing falls short, lean setups can't yet
match the big ones.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The autoresearch program has proposed and tested individual components of
a multi-property safety-steering system across Blocks B (conditional
gating), C (stacking and composition), D (geometry), E (mechanism), and
the preceding Block F experiments. E50 is the synthesis: it assembles the
minimum set of components that, in combination, matches or exceeds the
best reported multi-property safety-and-capability results in the literature
(AxBench SOTA, Rogue Scalpel-compliant safety floor) within a single 4090
GPU (24 GB VRAM) budget. The "minimal" constraint is explicit: components
are added one at a time (following E47's ablation) and only retained if
they contribute >= 3% to the composite score. This prevents "kitchen-sink"
over-engineering and ensures each component has a measurable justification.
The four candidate components — CAST gate (E9/E41/E42), orthogonalised
3-vector safety stack (E19/E47), norm-budget cap (E22/E46), and DoLa
factuality decoding (E25) — represent the minimal coverage of the four
failure modes identified in the first-principles corpus: selectivity failure
(gate), interference (ortho-stack), off-manifold displacement (norm cap),
and factuality drift (DoLa). The "within 24 GB" constraint is the hard
resource bound that makes the recipe deployable on a single RTX 4090, the
standard autoresearch hardware. VRAM accounting must be explicit: Gemma-2-
2B at 4-bit ~1.5 GB; activations and KV cache for batch=1, sequence=512 ~
4 GB; steering infrastructure (gate hook, norm-cap computation, DoLa dual
forward) ~0.5 GB; total ~6 GB, well within the 24 GB budget. The recipe
is designed to be reproducible by any researcher with an RTX 4090 or
equivalent, making it a practical SOTA baseline for the community.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** The minimal stack recipe (CAST gate at condition threshold theta +
3 Gram-Schmidt-orthogonalised safety vectors at alpha = E44-identified
knee + norm cap at 15% of ||h|| + DoLa at exit_layer=L20) will achieve
(a) JailbreakBench Compliance Rate <= 5%, (b) MMLU accuracy drop <= 3 pp
vs unsteered baseline, (c) XSTest over-refusal rate <= 20%, and (d) AxBench
concept-incorporation score >= 0.70, all simultaneously on Gemma-2-2B-it
(4-bit) in a single RTX 4090 (24 GB) within a 512-token generation budget.

---

## 3. Falsifier (>= 30 words)

If ANY of the four target thresholds (CR <= 5%; MMLU drop <= 3 pp; XSTest
OR <= 20%; AxBench >= 0.70) is NOT met simultaneously, the minimal-stack
SOTA claim is DISCARDED for that threshold (Status `~ partial` per axis).
If ALL four fail, Status moves to `x disproved`. The recipe is only
SUPPORTED if all four pass simultaneously.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel; the safety target (CR <= 5%)
and the guard architecture (A-E) that the minimal stack implements in
simplified form; their experimental design (V1-V5 validation protocol)
is the template for E50's ablation and evaluation.

Zhong, Zeping, et al. 2025 'AxBench: Steering LLMs? Benchmarks Matter'
arXiv:2501.17148 — AxBench; the capability/efficacy target (concept-
incorporation score >= 0.70); the 500-concept holdout provides the
generalization eval that goes beyond training-distribution efficacy.

Chuang, Yung-Sung, et al. 2023 'DoLa: Decoding by Contrasting Layers
Improves Factuality in Large Language Models' arXiv:2309.03883 — DoLa;
the decoding-time factuality component of the minimal stack; adds factuality
improvement by contrasting early and late layer logits; composable with any
residual-stream steering (E25).

Rottger, Paul, et al. 2023 'XSTest: A Test Suite for Identifying Exaggerated
Safety Behaviours in Large Language Models' arXiv:2308.01263 — XSTest;
the selectivity target (over-refusal <= 20%); the over-refusal floor that
ensures the recipe is not a trivial "refuse everything" system.

Project screening results: FINDINGS.md S-4, S-8 (CR 0.80->1.00 under
steering, Rogue Scalpel direction active on our Gemma); S-9 (relative_add
clean window at 10% displacement, E7/E36); S-6 (N17 geometry law R^2=0.81)
— the screening findings that motivate the specific parameter choices in
the minimal stack recipe.
```

---

## 5. Mechanism

The four components of the minimal stack address four distinct failure modes
in the space of (safety, capability, coherence, selectivity):

**CAST gate (selectivity):** Fires only when the condition projection exceeds
theta. Prevents over-refusal on benign inputs and preserves native refusal
on harmful inputs. Without the gate, the stack fires unconditionally,
degrading selectivity (XSTest OR rises above target).

**3 Ortho vectors (safety + capability):** Three orthogonalised safety
directions (refusal, anti-sycophancy, honesty) provide multi-property
safety coverage without interference. Orthogonalisation (Gram-Schmidt)
removes the cross-vector interference that would otherwise inflate the
effective off-shell displacement and degrade MMLU. Without ortho, the
3-vector stack has super-additive capability cost.

**Norm cap (coherence + capability):** The combined edit ||delta h|| is
capped at 15% of ||h|| (the N17/C9b empirically validated safe window).
Without the cap, at 3 active vectors the cumulative displacement may exceed
the manifold budget, causing PPL inflation and MMLU degradation.

**DoLa (factuality + capability):** DoLa contrasts early-layer and late-
layer logits to suppress factual hallucinations without modifying the
residual stream. It is additively composable with any residual-stream
steering (E25) and adds the factuality dimension to the capability
protection. Without DoLa, the stack may maintain safety but not improve
factuality.

The VRAM budget: Gemma-2-2B 4-bit (~1.5 GB) + KV cache (~4 GB at
batch=1, seq=512) + hooks and vectors (~0.2 GB) + DoLa dual forward
(~0.5 GB extra activations) = ~6.2 GB total, leaving 17.8 GB headroom
in a 24 GB card.

---

## 6. Predicted Delta

| Metric | Target threshold | Predicted value |
|---|---|---|
| JailbreakBench CR | <= 5% | 2-5% |
| MMLU drop | <= 3 pp | 1-3 pp |
| XSTest over-refusal | <= 20% | 10-20% |
| AxBench concept score | >= 0.70 | 0.70-0.80 |
| VRAM peak | <= 24 GB | ~6-8 GB |
| Per-token latency overhead | < 20% vs unsteered | ~10-15% |

All four primary targets predicted to be met simultaneously.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090 (24 GB).
- Stack assembly (in order):
  1. Extract 3 safety DiffMean vectors (refusal, anti-syco, honesty)
     using relative_add (alpha=0.10 each).
  2. Gram-Schmidt orthogonalise the 3 vectors in order.
  3. Apply norm cap: if ||sum_i alpha_i v_i|| > 0.15 * ||h||, rescale.
  4. Add CAST gate: condition vector from Sorry-Bench; threshold theta
     from E42 calibration.
  5. Add DoLa (exit layer = L20, from arXiv:2309.03883; composable with
     residual steering per E25).
- Alpha: E44-identified Pareto knee (expected lambda ~ 0.50 * 0.10 per
  vector).
- Eval: JailbreakBench (100 prompts, LLM-as-judge; CR target <= 5%);
  MMLU-500 (delta target <= 3 pp); XSTest (250 prompts; OR target <= 20%);
  AxBench mini (concept score target >= 0.70); VRAM profiling; latency
  measurement.
- Ablation: 7 conditions from E47 + DoLa component (8 conditions total),
  confirming each component contributes >= 3% to the composite score.
- Seeds: 7 (rung-3 directly; this is the capstone experiment).

### 7.2 Where it shines

E50 is the program capstone for the safety-and-capability sub-program. It
packages the best findings of E9-E47 into a single deployable recipe and
validates it against SOTA multi-property targets. A successful E50 result
is a citable contribution to the community.

---

## 8. Cross-references

- IDEA_TABLE.md Block F row E50.
- E9 (CAST gate): the gate component.
- E19 (Gram-Schmidt): the ortho-stack component.
- E22 (norm budget): the norm-cap component.
- E25 (DoLa stacking): the DoLa component; E25 confirms DoLa composes
  additively with residual steer.
- E44 (Pareto frontier): the alpha operating point.
- E47 (guard stack ablation): the 7-condition ablation that E50 extends
  with DoLa.
- N5 / N17 (geometry): the norm budget and off-shell law that constrain
  the stack.
- Rogue Scalpel Guards A-E: the full guard architecture; E50 implements
  Guards B+E (norm cap + gate) plus the orthogonalisation that approximates
  Guard A.
- FINDINGS.md S-4, S-8: the Rogue Scalpel dynamic on Gemma (motivation);
  S-9, S-6: the geometry laws that set the stack parameters.

---

## 9. Committee Q&A

**Q: The 5% CR target — is that achievable without breaking the model
entirely?**

> The CR target of <= 5% means the stack causes harmful compliance on at
> most 5 of 100 JailbreakBench prompts. The E41 CAST gate alone is predicted
> to achieve 70-80% refusal (CR <= 20-30%). Adding the norm cap (Guard B)
> and the guard D-style verdict check (via DoLa's factuality check) should
> push CR further down. 5% is ambitious but not impossible; the Rogue
> Scalpel's paper itself reports that their Guard A-E combination drives
> CR toward baseline (~0%). We target 5% as a conservative, achievable
> version of their full guard.

**Q: Why DoLa and not another factuality method?**

> DoLa (arXiv:2309.03883) is computationally cheap (one extra logit
> computation per token from an early exit layer) and composable with
> residual steering by E25. More expensive methods (FLAS, multi-step flow)
> are excluded by the 24 GB / single-GPU constraint. DoLa is also the
> factuality component most compatible with the additive-bus view of the
> residual stream (it operates at decoding time, not at the residual level).

**Q: What is the AxBench concept score threshold (>= 0.70)? Is that SOTA
for Gemma-2-2B?**

> 0.70 is a conservative SOTA target: AxBench reports that prompting
> achieves ~0.75-0.85 on most concepts (the current SOTA), while steering
> methods typically achieve 0.55-0.70 (below prompting). The >= 0.70 target
> means the minimal stack matches the lower end of the prompting-level
> performance. A score >= 0.80 would be a clear SOTA for steering.

---

## 10. Verification checklist

- [ ] VRAM profile run before the main eval to confirm 24 GB fits.
- [ ] All 4 component alphas/thresholds pre-specified from prior experiments
      (E44 for alpha, E42 for theta, E22 for norm cap, E25 for DoLa exit
      layer); no post-hoc tuning in E50.
- [ ] 8-condition ablation run (7 from E47 + DoLa) before the full-stack
      eval; each component confirmed to contribute >= 3% to composite.
- [ ] JailbreakBench baseline CR = 0% on unsteered model confirmed.
- [ ] LLM-as-judge calibrated to >= 90% agreement with human labels on
      harmful/benign classification.
- [ ] AxBench evaluation uses the 500-concept holdout (not training concepts).
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected) on all
      four primary metrics.
- [ ] Latency measurement: wall-clock time per generation token (prefill +
      autoregressive) vs unsteered baseline.
- [ ] IDEA_TABLE.md row updated; FINDINGS.md updated if all four targets met
      at rung-3 with full rigor contract (n >= 7, Wilcoxon, bootstrap CI,
      ordinal gate).

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block F, hypothesis E50.
  Status: `o UNTESTED`. This is the program capstone for the safety
  sub-program; it is designed to run last in Block F, after E41-E49
  have supplied the calibrated components and operating points.

  Cross-ref: FINDINGS.md S-4/S-8 (Rogue Scalpel dynamic real on Gemma,
  n=1 screening) motivates the safety target; S-6 (N17 R^2=0.81) and
  S-9 (C9b clean window at 10% displacement) provide the geometry-based
  parameter settings for the stack.

  Dependency: E9 gate, E19 ortho-stack, E22 norm cap, E25 DoLa stacking,
  E44 Pareto alpha, E42 gate threshold, E47 component ablation — all must
  run before E50.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-F (system integration + benchmark specialist).*

### Prior plausibility
**MEDIUM-HIGH** for meeting 3 of the 4 targets; **MEDIUM** for meeting
all 4 simultaneously. The CR <= 5% target is the hardest: the Rogue
Scalpel's full guard (A-E) achieves near-0% on their models, but our
simplified stack (B+E analog + ortho-A analog) may leave residual
compliance at 5-15%.

### Mechanism scrutiny
The four-component decomposition correctly maps to four distinct failure
modes. The DoLa addition is the weakest link: E25 (DoLa composes with
residual steer for additive factuality gain) is a prediction, not a
confirmed result. If DoLa interacts adversarially with the safety stack
(e.g., by changing the logit distribution in a way that the gate's condition
probe was not calibrated for), the full stack may underperform the no-DoLa
version on safety metrics.

### Confounds
1. The 4-threshold success criterion is a joint event; the probability
   that all four pass is the product of four individual probabilities if
   they are independent. If any one metric is highly uncertain (CR is the
   most uncertain), the joint probability of success is significantly lower
   than the individual success probabilities.
2. The AxBench concept score target (>= 0.70) uses a holdout concept set
   that the safety vectors were not extracted on. If the safety vectors
   degrade AxBench concept steering for unrelated concepts (a capability
   cost), the 0.70 target may be hard to reach.

### Expected effect size
CR: 3-10% (target met at the high end); MMLU drop: 1-3 pp (target met);
XSTest OR: 10-25% (target borderline); AxBench: 0.65-0.78 (target
borderline). Full joint success probability: 40-60%.

### Verdict
**TESTABLE + CAPSTONE + HIGH IMPACT** — E50 is the experiment whose result
the entire program builds toward. A positive result (all four targets met
simultaneously) would be the program's most citable contribution. Even
a partial result (3 of 4 targets met) is publishable and informs the
community about the achievable multi-property safety-capability frontier
on a 4090-class GPU.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to E50, the
**safety+capability capstone**: a minimal deployable stack (Gate + 3 ortho
vectors + NormCap + DoLa) hitting four SOTA targets simultaneously within 24 GB.
It assembles the prior components; no new vector form is introduced.

### 1. Steering-vector recipe (3 DiffMean vectors + gate; DoLa is decoding-time)

```python
# §1.3 METHODOLOGY: same 3 orthogonalised DiffMean safety vectors as E44/E47.
V = gram_schmidt([bank("refusal")["diffmean"],
                  bank("anti_sycophancy")["diffmean"],
                  bank("honesty")["diffmean"]])
v_condition = bank("harmfulness")["diffmean"]; v_condition /= norm(v_condition)  # CAST gate
# DoLa (arXiv:2309.03883) is a DECODING contrast (early vs late layer logits),
# NOT a residual-stream vector — composes additively with the steer (E25).
```

### 2. Experiment procedure (assemble in order, ablate, eval)

```text
1. Stack (order matters): (1) extract 3 vectors; (2) Gram-Schmidt orthogonalise;
   (3) NormCap rescale sum_i alpha_i v_i to <=0.15*||h||; (4) CAST gate (theta
   from E42); (5) DoLa exit_layer=L20.
2. Inject the gated, capped stack via hooks.apply_operation, "relative_add" (§2),
   alpha at the E44 Pareto-knee lambda (~0.50*0.10 per vector). DoLa applied at
   decode time on top.
3. ABLATION: the 7 E47 subsets + DoLa (8 conditions); each component retained only
   if it adds >= 3% to the composite.
4. MEASURE (§3 METHODOLOGY) at n=7 (rung-3 directly): JailbreakBench CR; MMLU-500
   delta; XSTest over-refusal; AxBench concept score (off-family judge); VRAM peak;
   per-token latency. Baseline CR confirmed 0% first.
```

### 3. Measurement & decision rule

- **PRIMARY metric:** the four target thresholds met SIMULTANEOUSLY.
- **Hypothesis (§2):** CR <= 5% AND MMLU drop <= 3 pp AND XSTest OR <= 20% AND
  AxBench concept score >= 0.70, all at once, in <= 24 GB / 512-token budget.
- **Pre-registered FALSIFIER (§3):** if ANY of the four targets is not met, that
  axis is `~ partial`; if ALL four fail, Status -> `x disproved`. SUPPORTED only
  if all four pass together (joint event). A safety leak is an automatic DISCARD.

### 4. Where the code is / status — UNTESTED (terminal dependency)

- **No driver yet** (campaign + `scripts/build_provenance.py` -> `PROVENANCE/E50.md`).
- **Missing machinery (why UNTESTED):** E50 is the terminal node — it requires the
  **CAST gate** (E41/E42), **Gram-Schmidt ortho-stack** (E19/E47), **norm-budget
  clamp** (E22), **DoLa decoding** (E25), the **E44 Pareto-knee alpha**, and the
  **E42 gate theta** as inputs, none of which are built/run yet; plus JailbreakBench,
  XSTest, AxBench-mini wiring, a calibrated judge, and VRAM/latency profiling. It is
  designed to run LAST in Block F after E41-E49 supply the calibrated components.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E50.md`.
