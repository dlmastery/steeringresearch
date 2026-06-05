# E6 — Over-Steering Linear Probe

> **One-line claim:** A linear probe trained on layer-L hidden states predicts,
> before generation begins, whether a given input-alpha combination will produce
> over-steered (incoherent) output — with AUC >= 0.80 on a held-out test set.
>
> **Source design space:** Block A — Foundations and measurement tooling (E6).
> **Primary axis:** A5 (WHEN — condition / gate at pre-generation time).
> **Implementation status:** `o UNTESTED`.

---

## In Plain English

**What we're testing, simply:** Pushing the nudge too hard breaks the text. Right
now we only find out *after* the model has written the broken text. This asks
whether a cheap predictor can warn us *before* generating — "this push will produce
gibberish" — so we can stop it in time.

**Key terms (defined here so you don't have to look anything up):**
- **Language model (LLM):** an AI that predicts the next word; here, small Gemma
  models.
- **Steering:** nudging the model's behavior by adding a direction to its internal
  state while it writes.
- **Steering vector:** the specific direction of the nudge.
- **Residual stream / hidden state:** the model's running internal "thought" state,
  where the nudge is added and which we can read.
- **Layer:** one of the model's stacked processing steps.
- **alpha (strength):** how hard we push the nudge.
- **Over-steering:** pushing so hard the output turns incoherent — and, dangerously,
  sometimes more willing to comply with harmful requests.
- **Coherence / perplexity:** whether text stays fluent (perplexity higher = more
  broken).
- **Linear probe:** a tiny, simple predictor trained to read the model's internal
  state and output a yes/no guess — here, "will this push break the text?"
- **AUC:** a 0.5-to-1.0 score for how good a yes/no predictor is. 0.5 = coin-flip
  useless; 1.0 = perfect. We want at least 0.80.
- **Pre-generation:** before the model writes any words — the probe runs early, so
  it can veto a bad push cheaply.

**Why we're doing this (the point):** A reliable early warning lets a safe steering
system skip a bad push instead of generating broken or unsafe text and redoing it.
That's faster and safer than checking after the fact.

**What the result would mean:** A positive result gives a cheap, built-in safety
brake for steering. A negative result means we can't predict breakage early and must
keep checking the finished text the slow way.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>=100 words)

Over-steering is the failure mode that makes deployment of activation steering
genuinely dangerous: a user-beneficial steering vector, applied at slightly too
large an alpha, causes the model to produce incoherent, repetitive, or off-manifold
output that not only fails to achieve the intended behavior but also raises the
Rogue Scalpel compliance rate (arXiv:2509.22067). Currently, the only way to detect
over-steering is post-hoc — by measuring perplexity or running a coherence judge
on the generated text. This is expensive (requires a full forward pass through the
generation) and too slow for real-time gating. A pre-generation over-steering probe
would transform the architecture of safe steering systems: instead of generating,
checking for incoherence, and regenerating at a lower alpha, a probe could predict
incoherence from the prefill hidden state alone and prevent the over-steered generation
from occurring. The key insight motivating E6 is the geometric characterization
established in C2/S-6: the coherence cliff correlates with off-shell displacement
Δ‖h‖ at R^2=0.81. Off-shell displacement is computable at prefill from the
pre-generation layer-L hidden state without any generation step. If a linear probe
on that state (possibly augmented with the planned alpha and vector norm) achieves
AUC >= 0.80 on predicting incoherence, the probe serves as a cheap pre-generation
safety check that can veto over-steered interventions before any tokens are produced.
This connects directly to CAST (arXiv:2409.05907), which gates the behavior injection
on a cosine condition; E6 gates not on whether to steer, but on whether the planned
steering magnitude will cause incoherence.

---

## 2. Formal hypothesis (>=50 words, falsifiable)

A logistic regression (linear probe) trained on the concatenation of: (i) the layer-L
hidden state of the last input token under planned steering (h + alpha*v), (ii) the
planned off-shell displacement Δ‖h‖ = |alpha| * ||v_perp||, and (iii) the planned
relative displacement alpha / ||h||, achieves AUC >= 0.80 on a held-out set of
(input, alpha) pairs labeled "coherent" (generation PPL within 30% of baseline) or
"incoherent" (generation PPL > 30% of baseline). The probe is trained on Gemma-2-2B-it
@L16 and evaluated on held-out prompts from the same behavior. A secondary generalization
test checks AUC on a different behavior unseen during training.

---

## 3. Falsifier (>=30 words)

**Fired if:** the logistic probe's AUC on the held-out set is < 0.80 AND a baseline
that uses only the planned Δ‖h‖ (without hidden-state features) achieves AUC < 0.75.
The second condition rules out the trivial case where Δ‖h‖ alone is sufficient
(i.e., a threshold rule, not a probe). If Δ‖h‖ alone achieves AUC >= 0.80, E6 is
satisfied without the probe (a simpler and more practically useful finding).

---

## 4. Citations (>=80 words, Citation Rigor format)

```
Korznikov, Mikhail, et al. 2025 ICML 'The Rogue Scalpel: Activation
Steering Compromises LLM Safety' (arXiv:2509.22067) — the threat model
E6 addresses: over-steering causes safety failure simultaneously with
coherence failure; a pre-generation probe that detects over-steering
prevents both failure modes.

Park, Junsoo, et al. 2025 ICML 'CAST: Conditional Activation Steering'
(arXiv:2409.05907) — CAST gates on whether to steer (condition check);
E6 gates on whether the planned MAGNITUDE of steering is safe — orthogonal
gating levels that can compose (E26: gate-before-behavior injection order).

Li, Kenneth, et al. 2023 NeurIPS 'Inference-Time Intervention: Eliciting
Truthful Answers from a Language Model' (arXiv:2306.03341) — ITI uses
linear probes on activations for behavioral detection; E6 extends this
to the incoherence-prediction task, which is a different target but uses
the same probe architecture.

Wurgaft, Ben, et al. 2026 arXiv 'Manifold Steering' (arXiv:2605.05115)
— manifold-aware steering directly avoids off-manifold departure; E6's
probe provides the same protection more cheaply by predicting manifold
departure rather than steering along the manifold. The two approaches
are complementary (probe for cheap screening; manifold steering for high-
value interventions).
```

---

## 5. Mechanism (deep technical)

The mechanism is the geometric off-manifold prediction established in C2:

    log PPL ~= 5.40 + 2.87 * Δ‖h‖    (R^2=0.81, pooled over 23 cells)

This law means that, for additive steering, the PPL of the generation is predictable
from the off-shell displacement at the injection layer — which is computable WITHOUT
GENERATION. The off-shell displacement for additive steering is:

    Δ‖h‖ = ‖(h + alpha*v) - h‖ / ‖h‖ = |alpha| * ‖v‖ / ‖h‖

This is a function of alpha, v, and h — all known at prefill. The N5 law therefore
provides a closed-form over-steering predictor:

    PPL_predicted = exp(5.40 + 2.87 * |alpha| * ‖v‖ / ‖h‖)

If PPL_predicted > threshold (e.g., baseline_PPL * 1.30), the probe predicts
incoherence. No generation required.

The linear probe adds value over the pure N5 formula when:
1. The concept direction v is not unit-normalized (raw DiffMean), so ‖v‖/‖h‖ varies
   by prompt and prompt-specific information is needed.
2. The off-manifold threshold is layer- or input-specific (e.g., "this prompt is
   already near the manifold boundary even without steering").
3. The model's LayerNorm statistics at L differ significantly from the average used
   to fit the N5 law (concept-direction-specific curvature).

The linear probe on h (the actual pre-steered hidden state) captures this input-
specific manifold distance: prompts that are already near the manifold boundary (low
effective-rank local neighborhood, high curvature at h) will be predicted as over-
steerability-prone at lower absolute alpha. The probe combines the geometric law
(Δ‖h‖ as a feature) with the local manifold geometry (h as a feature vector).

The AUC target of 0.80 is achievable if the geometric law accounts for 80% of the
incoherence variance (R^2=0.81 in C2 suggests the off-shell term already explains
most variance; residuals from a linear probe on h should further improve it).

Training data: run the layer-L alpha sweep (E3 protocol) on a diverse set of 200+
prompts at multiple alpha values; label each (prompt, alpha) pair as coherent/incoherent
based on the 30% PPL threshold. Extract h+alpha*v at L16 for each; train logistic
regression. Train/test split: 80% train, 20% held-out; additionally held-out concept
for generalization test.

---

## 6. Predicted Delta (pre-registered numbers)

| Metric | Predicted value | Observed |
|--------|----------------|---------|
| AUC of linear probe on held-out set | >= 0.80 | **UNTESTED** |
| AUC of Δ‖h‖-only baseline | >= 0.75 (from R^2=0.81 law) | **UNTESTED** |
| Generalization AUC on held-out behavior | >= 0.70 | **UNTESTED** |
| False positive rate at AUC-optimal threshold | < 15% | **UNTESTED** |
| Latency overhead of probe vs generation | < 5% | **UNTESTED** |

---

## 7. Experimental protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it (4-bit if E5 confirmed; bf16 otherwise), L16.
- **Probe inputs:** {h + alpha*v at L16, Δ‖h‖, alpha/‖h‖, ‖h‖} — 4 features plus
  the 2304-dim hidden state (if using the full linear probe).
- **Labels:** coherent (generation PPL <= 1.3 * baseline PPL) vs incoherent.
- **Training data:** 200 prompts × {0.5, 1, 2, 3, 4, 6, 8} alpha values = 1400 labeled
  (prompt, alpha) pairs. Balanced subsample if needed.
- **Probe architecture:** logistic regression with L2 regularization; no nonlinearity
  (the "linear" in linear probe must be respected for interpretability).
- **Metrics:** AUC, precision-recall curve, F1 at operating threshold.
- **Control:** Δ‖h‖-only baseline (single-feature logistic); random baseline.

### 7.2 Where this should SHINE

Inputs that are already near the manifold boundary for independent reasons (e.g., unusual
token sequences, domain shifts, very long prompts). For these, the hidden state h
carries information about local curvature that the global Δ‖h‖ law misses. If the
probe's AUC substantially exceeds the Δ‖h‖-only baseline for these inputs, the
probe's extra features are justified.

### 7.3 System integration

If E6 succeeds, the probe slot into the steering pipeline as a pre-generation safety
gate: after computing the planned steering (alpha, v), evaluate the probe on the
prefill hidden state; if the probe fires (predicted incoherent), reduce alpha to 50%
and re-evaluate, or pass to the CAST gate (E9) for a decision. This is the cheapest
possible safety gate: one linear operation on a 2304-dim vector.

---

## 8. Cross-references

- **N5** (norm-budget law): the C2 R^2=0.81 law is the theoretical backbone of the
  Δ‖h‖-only probe.
- **N17** (off-shell displacement governs coherence): the mechanistic predictor E6 uses.
- **E3** (alpha cliff): E6 is a probe for the cliff; E3 characterizes the cliff globally;
  E6 predicts it per-input.
- **E7** (norm-relative alpha): the relative alpha (alpha/‖h‖) is one of E6's probe features.
- **E9** (CAST harmless-refusal gate): CAST gates on whether to steer at all; E6 gates
  on whether the planned magnitude is safe — two orthogonal gates composable as E26 suggests.
- **Rogue Scalpel** (arXiv:2509.22067): the compliance-rate spike at the cliff motivates
  the probe as a safety check, not just a coherence check.
- **IDEA_TABLE.md** Block A row E6.

---

## 9. Committee Q&A

**Q: Why not just use the N5 law directly as the over-steering detector without training
a probe?**

> The N5 law is a global average (R^2=0.81 over 23 rows from 2 models, 8 layers).
> Per-prompt variance around this law is the residual 19%. A prompt-specific linear
> probe on h should reduce this variance by capturing local manifold curvature. If the
> Δ‖h‖-only baseline achieves AUC >= 0.80, the N5 law IS sufficient — the probe
> training is not needed. E6's falsifier is designed to capture this: the experiment
> provides value whether the probe or the formula wins.

**Q: Why logistic regression and not a small MLP?**

> "Linear probe" is the standard for mechanistic interpretability (ITI, RepE, etc.).
> Nonlinear probes can achieve higher AUC but are harder to interpret and may overfit.
> If a linear probe achieves AUC 0.80, the claim is that off-manifold departure is
> linearly predictable from the pre-steered hidden state — a clean mechanistic claim.
> A nonlinear probe achieving 0.80 would be a weaker claim about the information
> content of h.

**Q: 200 prompts × 7 alpha values = 1400 training examples — is that enough for a
2304-dim logistic regression?**

> Logistic regression with L2 regularization is sample-efficient in high dimensions
> (l2 regularization is equivalent to a Gaussian prior that prevents overfitting).
> With 1400 examples and 4 hand-crafted features (Δ‖h‖, alpha/‖h‖, ‖h‖, and h's
> projection on v), the feature space is effectively 4-dimensional for the main probe,
> which is well-conditioned at 1400 examples. The full-h probe (2304 dims) is the
> riskier variant — use aggressive regularization (CV over lambda).

---

## 10. Verification artifacts checklist

- [ ] 1400+ labeled (prompt, alpha) pairs with real PPL labels stored in
      `ideas/E6_oversteering_probe/data/labeled_pairs.jsonl`.
- [ ] Logistic probe trained and AUC reported on hold-out set.
- [ ] Δ‖h‖-only baseline AUC for comparison.
- [ ] Generalization AUC on held-out behavior.
- [ ] Precision-recall curve stored in `results/`.
- [ ] Latency benchmark: probe forward pass vs generation step.
- [ ] Result row in `EXPERIMENT_LEDGER.md`.

---

## 11. Status journal

- 2026-05-27 — hypothesis inherited from corpus E6; AUC >= 0.80 threshold pre-registered.
- 2026-05-29 — all screening (C1–C9) ran without over-steering probe infrastructure.
  The N5 law (C2, R^2=0.81) provides theoretical motivation for E6 but does not test
  the probe directly. **E6 UNTESTED.**
- 2026-05-31 — Design doc written. The N5/N17 geometric laws strengthen the prior that
  E6 will succeed (the off-manifold signal is there). Priority: after E3/E4 rung-3
  experiments generate labeled (prompt, alpha, PPL) data, use that data to train the probe.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-A. This is an untested hypothesis with strong theoretical motivation.*

### Prior plausibility

**HIGH for the Δ‖h‖-only baseline; MEDIUM for the full linear probe adding value.**
The N5 geometric law (R^2=0.81) already provides an 81% of variance explanation of
incoherence. A linear probe on h can at best add the remaining 19% of variance that
Δ‖h‖ misses. Whether that residual is linearly predictable from h is an open question.
The answer likely depends on whether the activation manifold's curvature is smooth
(linear probe works) or highly variable (needs nonlinear probe or per-layer calibration).

### Mechanism scrutiny

The off-manifold geometry argument is well-grounded (C2, N5, N17). The extension to
a pre-generation probe is the natural next step. The key mechanistic question is:
does h (the pre-steered hidden state of the last input token) carry information about
the local curvature of the manifold at that point? Theory says yes (local curvature
is a function of the neighborhood of h), but the practical range at which a linear
approximation to this curvature is reliable is unknown.

### Confounds

1. **Training-test contamination**: if training and test prompts are from the same
   concept, the probe may memorize concept-specific over-steering patterns rather than
   learning a universal manifold-boundary detector. The generalization test (held-out
   concept) is the essential anti-contamination check.
2. **PPL threshold choice**: the 30% PPL threshold for "incoherent" is arbitrary. At
   the E3 cliff, PPL rises super-linearly — so 30% is conservative (the real cliff is
   steeper). The probe performance will be sensitive to this threshold.
3. **Layer specificity**: the probe is trained at L16. It may not generalize to other
   injection layers (where the manifold curvature is different). A multi-layer probe
   or the E13 protocol (early condition check) may be needed.

### Does the AUC 0.80 threshold specifically matter?

**Yes, for system integration.** A probe with AUC 0.75 is not useful as a safety
gate (too many false negatives). A probe with AUC 0.90 changes the deployment story
significantly. The 0.80 threshold is the minimum for the probe to be useful in a
real-time safety system.

### Literature precedent

No prior work trains a linear probe specifically to predict incoherence from pre-
generation activations in the context of activation steering. The closest is ITI
(arXiv:2306.03341) which uses probes for behavioral detection, not coherence
prediction. This makes E6 genuinely novel if it succeeds. The N5 law (our own
screening result) provides the motivating observation.

### Skeptical effect-size re-prediction

Δ‖h‖-only AUC: in [0.75, 0.88], 80% CI (based on R^2=0.81 → AUC ~= 0.81 for
balanced binary classification via a linear threshold on one predictor). Linear probe
on h (adding hidden-state features): AUC in [0.78, 0.92], 80% CI. The probe's extra
features are expected to add 2–8 AUC points over the Δ‖h‖-only baseline.

### Minimum-distinguishing experiment

E3 sweep data: 200 prompts × {0, 1, 2, 4, 8} alpha × 1 behavior = 1000 labeled pairs.
Train Δ‖h‖-only logistic (1 feature), 4-feature logistic, and full-h logistic (2304
features with strong L2). Compare AUC. ~30 minutes of probe training after E3 data
collection. If Δ‖h‖-only AUC >= 0.80, the hypothesis is satisfied immediately.

### Verdict

**NOVEL+TESTABLE. High practical value if it succeeds.** The N5 geometric law
provides very strong motivation. The Δ‖h‖-only baseline (testable from E3 data)
is the fastest path to a useful finding. The linear probe on h adds potential
marginal value. Recommend: run the Δ‖h‖-only baseline first (trivial, from E3 data);
then decide whether the full probe training is worthwhile. If the baseline achieves
AUC >= 0.80, report it as the simpler and more interpretable solution.

---

## Pseudocode & Methodology

This hypothesis trains a **pre-generation linear probe** that predicts incoherence
(over-steering) from the prefill hidden state. The compared object is the probe vs the
N5 geometric law; the source vector is DiffMean @ L16, swept over alpha to *generate
labels*.

### 1. Steering-vector recipe

```python
# The steering vector itself is the standard DiffMean (METHODOLOGY §1.3); the
# experiment's novelty is a PROBE on the steered hidden state, not a new vector.
v = diffmean_vector(*collect_activations(model, tok, load_concept(behavior), 16))
```

### 2. Experiment procedure

```text
1. Build a labeled dataset by sweeping alpha (reusing the E3 sweep):
2.   for prompt in 200_prompts:
3.     for alpha in {0,1,2,3,4,6,8}:
4.        h        = prefill hidden state @ L16 (last input token)   # geometry.probe / ProbeHook
5.        h_steer  = apply_operation(h, v, "add", alpha)             # h + alpha*v
6.        dH       = geometry.offshell_displacement(h, h_steer)      # = |alpha|*||v||/||h||
7.        rel      = alpha/||h||
8.        PPL_gen  = eval.perplexity(generate under steering)
9.        label    = (PPL_gen > 1.3*baseline_PPL)  ? "incoherent" : "coherent"
10.       features = concat(h_steer, dH, rel, ||h||)
11. Train logistic regression (L2, CV over lambda) on 80% / hold out 20%.
12. BASELINE: logistic on dH alone (single feature).
13. Generalization: evaluate AUC on a held-out UNSEEN behavior.
```

### 3. Measurement & decision rule

PRIMARY metric: held-out **AUC** of the linear probe. FALSIFIER (§3): fires if probe
AUC `< 0.80` AND the `Δ‖h‖`-only baseline AUC `< 0.75` (this rules out the trivial
threshold-rule case; if `Δ‖h‖`-only already hits ≥0.80, E6 is satisfied by the simpler
rule). Pre-registered (§6): probe AUC `>= 0.80`; `Δ‖h‖`-only `>= 0.75`; held-out-behavior
AUC `>= 0.70`. Verdict: SUPPORTED iff probe (or the `Δ‖h‖` rule) reaches 0.80 on held-out
data. Mechanism: the N5 law `log PPL ≈ 5.40 + 2.87·Δ‖h‖` (R²=0.81) already explains most
incoherence variance from a prefill-computable quantity, so `Δ‖h‖` is a strong feature.

### 4. Where the code is / status

This needs **new machinery**: a probe-training script (logistic regression on prefill
features) that does not yet exist as a dedicated driver. The label-generating alpha sweep
is `scripts/campaign_sweep.py` (the E3 data); prefill states come from
`hooks.ProbeHook` / `hooks.probe_activations` and `geometry.offshell_displacement`. The
probe trainer itself (sklearn-style logistic + AUC/PR-curve) is the missing piece — this
is why E6 is **UNTESTED**. Fastest path: run the `Δ‖h‖`-only logistic on existing E3 sweep
data first.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E6.md`.
