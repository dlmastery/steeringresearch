# N2 — Conditioning = Curvature, Behavior = Direction (Algebraic Factorization)

> **One-line claim:** Any steering operation factorizes into a where-to-act scalar
> field g(h) and a what-to-do direction v; CAST, FineSteer-SCS, and discriminative-layer
> selection are the same gate operator parameterized differently, and one learned g(h)
> reproduces all three.
>
> **Primary axes:** A5 (condition/gate), A8 (geometry/curvature)
> **Status:** UNTESTED

---

## In Plain English

**What we're testing, simply:** Several different steering methods each answer the
same two questions in their own way: *should I nudge right now?* and *which way?*
This doc says all of them are secretly the same recipe — a "when" dial times a
"which-way" arrow — and that learning the "when" dial directly should match or beat
every individual method.

**Key terms (defined here):**
- **Steering / steering vector** — changing behavior by adding a chosen direction to
  the model's internal "thought" mid-sentence, instead of retraining. The vector is
  the "which-way" arrow.
- **Residual stream** — the model's running internal thought; what we edit.
- **Layer** — one of the model's processing steps; a knob.
- **Gate / conditioning** — the "when" decision: reading the current thought to decide
  whether (and how much) to nudge — e.g. only refuse when a request actually looks harmful.
- **Curvature** — how sharply the "healthy thought" surface bends near the current
  point; this doc treats the "when" dial as following that bending.
- **Coherence** — whether the text stays fluent and sensible.

**Why we're doing this (the point):** If three "different" gating methods are really
one recipe with different settings, we can stop arguing about which to use and just
fit the one best "when" dial — simpler and likely better.

**What the result would mean:** A win means a single learned gate reproduces all three
named methods and beats them. A loss means they really are different tools, not one.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The steering literature has produced a proliferation of gating and conditioning methods:
CAST uses a cosine threshold on the input against a condition vector; FineSteer-SCS
uses an energy ratio between the dangerous and total subspace projections; discriminative-
layer selection identifies layers where the contrast-set mean difference is large and
steers only there. Each method is motivated independently, with separate theoretical
stories. But all three answer the same question: given a current activation h, how much
should I steer right now? They differ only in what function of h they evaluate to
answer that question. This suggests a unifying algebraic factorization: the steering
update is g(h) * v, where g is a learned or structured scalar field on activation
space and v is the fixed behavior direction. The curvature interpretation is natural:
g(h) should be small when h is already near the target manifold region (low steering
needed) and large when h is in the "dangerous" zone (high steering needed). Curvature
of the decision boundary in activation space is precisely the quantity that varies
across input types and drives the conditional gate. If CAST, SCS, and discriminative-
layer selection are all approximating the same underlying g(h), then fitting g directly
should outperform any of them individually, because it optimizes over the full function
class rather than a restricted parametric family. Moreover, the factorized form g(h)*v
decouples the estimation of direction (from contrast pairs) and magnitude (from
distributional properties of h), enabling independent improvement of each.

## 2. Formal Hypothesis (>= 50 words)

There exists a scalar function g: R^d -> [0,1] such that:

  (i) g(h) >= theta (CAST threshold) iff cos(h, c) >= theta_CAST for all h in a
      held-out harmful/harmless set (CAST special case)
  (ii) g(h) >= 0.5 iff E_harm(h) / E_total(h) >= 0.5 (SCS special case)
  (iii) argmax_L g(h_L) = argmax_L Fisher_ratio(L) (discriminative-layer special case)

A logistic regression or 2-hidden-layer MLP fit to g(h) on a balanced harmful/harmless
training set should achieve gate PR-AUC >= 0.90 and simultaneously recover all three
special cases within their respective threshold ranges on a held-out test set.

## 3. Falsifier (>= 30 words)

If the learned g(h) fails to recover CAST, SCS, or discriminative-layer masks at
accuracy > 85% (measured by agreement rate on held-out harmful prompts across 3 categories),
or if its gate PR-AUC is <= max(PR-AUC_CAST, PR-AUC_SCS), the unification claim is
FALSIFIED and the three methods are genuinely distinct operators.

## 4. Citations (Citation Rigor >= 80 words)

```
Turner et al. 2023. 'Activation Addition: Steering Language Models Without
Optimization' arXiv:2312.06681 (CAA). The foundational additive method;
the constant-alpha formulation is the g(h)=alpha special case of our factorization.

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. Establishes that
the where-to-act problem (which region of the manifold requires steering) is
the primary geometric question; their M_h manifold parameterization implicitly
defines g via manifold membership probability.

Venkatesh & Kurapath 2026. 'On the Non-Identifiability of Steering Vectors'
arXiv:2602.06801 (ICLR 2026 workshop). Proves that v is non-identifiable;
equivalently, the direction component of our factorization lives in a coset.
The factorization g(h)*v clarifies: g is identifiable (it's a behavioral outcome
predictor), v is a gauge choice. The factorization separates an identifiable scalar
from a gauge-ambiguous direction.

Gao et al. 2026. 'Cylindrical Representation Hypothesis' arXiv:2605.01844. CRH
decomposes activation differences into radius and angle; g(h) in our factorization
corresponds to the radius component (how far to move) while v/||v|| is the angle.
N2's factorization is thus the CRH decomposition applied to the gating function.
```

## 5. Mechanism

The factorization g(h) * v is motivated by three observations. First, behavior
direction v is approximately constant across inputs within a behavior class (this is
the DiffMean assumption and is supported by S-3: cos=0.994 across models). Second,
the amount of steering needed varies with the input: a clearly harmful prompt needs
full steering; a clearly benign prompt needs none. Third, the existing gate functions
are all monotone in "harmfulness score" of h, which is a scalar function.

Formal construction: let c be the condition vector (DiffMean of harmful vs harmless).
Define three candidate gate functions:
  g_CAST(h) = sigmoid((cos(h,c) - theta) / tau)
  g_SCS(h) = sigmoid((E_c(h) / E_total(h) - 0.5) / tau)
  g_disc(L, h) = I[L = argmax Fisher(L)]

All three are functions of h. The claim is: there exists g*(h) (a logistic regression
on [h, h^2 terms, cos(h,c), E_c(h)] features) such that g*(h) agrees with each of
{g_CAST, g_SCS, g_disc} on their respective support sets. The learning objective is
binary cross-entropy on {should_steer(prompt)} labels derived from human annotations.

Curvature connection: g(h) is large when h is near the decision boundary between
harmful and benign behavior manifold regions. High local curvature of that boundary
corresponds to high sensitivity of behavior to small perturbations, which is exactly
the regime where active steering is most valuable. So g(h) = (local curvature of
M_harm at h)^(1/2) is a plausible mechanistic interpretation.

## 6. Predicted Delta

| Metric | Predicted Delta (vs best single gate) | Rationale |
|---|---|---|
| Gate PR-AUC (harmful vs harmless) | +0.03 to +0.08 | g* optimizes joint criterion |
| Over-refusal on benign (XSTest-style) | -15% to -30% relative | g* is softer than hard CAST threshold |
| True-positive refusal rate | +/- 2% of CAST | unified g* recovers CAST on in-distribution harmful |
| Agreement with CAST threshold | >= 85% | CAST is the simplest special case |
| Agreement with SCS energy ratio | >= 80% | SCS is more complex, slight slack |
| Agreement with discriminative-layer | >= 75% | layer selection is a different input space |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it (demonstrated steering window in C6, optimal for gate experiments)
- Behaviors: safety (harmful vs harmless), 3 categories (hate, self-harm, illegal)
- Training data for g: 100 harmful + 100 harmless prompts from JailbreakBench + XSTest,
  80/20 train/test split
- Features for g*: [h_L, cos(h_L, c), E_c(h_L)/E_total(h_L)] for L = layer 16
- Models for g*: logistic regression (baseline), 2-layer MLP (full)
- Evaluation: PR-AUC on held-out test set; agreement matrices against each gate method;
  XSTest over-refusal; JailbreakBench true-positive rate
- Seeds: 3 train/test splits x 3 extraction seeds
- Wall-clock: ~3 hours on RTX 4090

### 7.2 Where it shines

On out-of-distribution prompts (domain shift, paraphrased jailbreaks), where the
parametric CAST/SCS gates are most likely to fail. The learned g* adapts to the
actual geometric structure of h rather than a fixed threshold.

## 8. Cross-References

- N6 (gate in read not write): N6 enforces cos(condition, behavior)=0; N2 shows the
  gate IS a function of condition projection -- separating g(h) from v is equivalent
  to N6's separation principle
- N5 (norm-budget): g(h) should be multiplied by a norm-budget factor to yield the
  full N5 conservation prediction
- N11 (curvature-aware alpha): N11 is the per-prompt alpha version of N2's g(h) field
- N12 (capstone): g(h) is the conditioning factor in the unified operator
- CAST paper (implicit citation via corpus), FineSteer-SCS (corpus)
- IDEA_TABLE.md: N2 row, axes A5+A8

## 9. Committee Q&A

**Q: Isn't this just learning a better safety classifier and using it as a gate?**

> The contribution is the ALGEBRAIC CLAIM that existing methods are special cases of
> one parameterization, not a novel architecture. If the claim holds, it justifies
> replacing all three methods with one jointly-optimized gate, which has immediate
> engineering value. The classifier-gate itself is not new; the unification is.

**Q: How do you prevent g(h) from just memorizing the training distribution?**

> The agreement-with-existing-gates evaluation is done on held-out prompts, not the
> g* training set. This tests generalization jointly with unification. A g* that
> memorizes training data will fail the OOD agreement test (E15's distribution shift
> test is directly applicable here).

**Q: The three gate methods operate at different levels (token, layer, scalar) — can
they all be g(h) of the same form?**

> g_disc is indexed by layer L, not just h. The claim is more carefully:
> g*(h_L, L) recovers all three. This requires that g* also take L as input,
> making it a (h, L) -> [0,1] function. The implementation uses layer embedding
> as an additional feature.

## 10. Verification Checklist

- [ ] g* logistic regression implementation with [cos, E_c/E_total, layer] features
- [ ] PR-AUC measurement against CAST, SCS, discriminative-layer on held-out test set
- [ ] Agreement matrix g* vs each gate method (85% / 80% / 75% targets logged)
- [ ] XSTest over-refusal measured with g* gate vs CAST gate at matched behavior
- [ ] JailbreakBench true-positive rate logged
- [ ] 3 train/test splits x 3 seeds = 9 runs per configuration
- [ ] Result in EXPERIMENT_LEDGER.md with status update to IDEA_TABLE.md N2 row
- [ ] Cross-reference added to N6, N11, N12 docs

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. The factorization claim is a
  clean theoretical prediction that motivates a concrete learning experiment.
  No data exists. S-5 (E2 FALSIFIED: Fisher ratio != steering efficacy) is
  tangentially relevant: it suggests that discriminative-layer selection based on
  Fisher ratio alone is not the full g(h), supporting the need for a richer model.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM-HIGH. The factorization g(h)*v is natural and well-motivated. The literature
precedent is strong: conditional computation (Bengio 2013, "Estimating or Propagating
Gradients Through Stochastic Neurons") uses exactly this gating-times-direction
structure. The question is whether the learned g* genuinely unifies the three
methods or simply learns the majority-vote of their disagreements.

### Mechanism scrutiny

The curvature interpretation of g is suggestive but not proven. High local curvature
at h is the right informal story, but estimating curvature is expensive; what the
experiment actually measures is whether a low-complexity function of h (logistic or
small MLP) can approximate the three gate functions simultaneously. This is a weaker
claim than the curvature story implies.

### Confounds

1. Training-set leakage: if the harmful/harmless split is not carefully stratified,
   g* may memorize surface patterns rather than learn the geometric structure.
2. Gate agreement is a soft metric: "85% agreement" with CAST at some threshold
   tau is trivially achievable by setting g* = cos(h,c). The claim requires that
   the SAME g* simultaneously achieves 85/80/75 agreement with all three gates.
3. The discriminative-layer gate operates on a different input space (per-layer
   activations across the full forward pass) vs CAST/SCS which operate on a single
   layer's h. A joint g*(h, L) that unifies both is a significantly harder claim.

### Does the specific unification claim matter?

YES, but modestly. If the claim holds, it justifies a single gate instead of three,
reducing engineering complexity. But the practical improvement over "just use CAST"
may be small on in-distribution data. The OOD improvement (learned g* adapts better)
is the practically important sub-claim.

### Literature precedent

Conditional computation is decades old. The specific application to steering gating
is novel, but the machinery is standard. The N2 claim is more a "unification
observation" than a novel algorithm — which makes it scientifically cleaner (easier
to test) but less impactful if the unification is loose.

### Skeptical effect-size estimate

Gate PR-AUC improvement: 0 to +0.03 (vs claimed +0.03 to +0.08) on in-distribution
data. The main risk is that logistic regression on [cos, E_c/E_total] features is
effectively CAST+SCS combined, and the incremental gain over either alone is within
noise. OOD improvement may be larger: +0.05 to +0.12 PR-AUC. Recommend testing
OOD explicitly (paraphrased jailbreaks not seen during g* training).

### Minimum distinguishing experiment

One model, one behavior, held-out test: train g* on 50/50 harmful/harmless;
evaluate agreement with each of the three gates; report the 3x agreement numbers.
Total cost ~45 min including data prep. If all three agreements are >= 80%,
the unification claim is strongly supported.

### Verdict

TESTABLE-MEDIUM. A clean falsifiable claim with a 45-minute minimum experiment.
The theoretical story is slightly oversold (curvature is the suggestive framing,
but logistic regression is the actual mechanism). The unification is likely to
be approximate (not exact special cases) — the 85/80/75 thresholds are achievable
but will require careful threshold tuning. Recommend running minimum experiment
first; promote to full protocol only if the minimum finds agreement > 80% jointly.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N2 claims every gate is one operator `g(h)·v`: a scalar field `g(h)` (where-to-act) times a fixed direction `v` (what-to-do). **UNTESTED** — needs a learned-gate fitting harness.

### 1. Steering-vector recipe (factorized gate × direction)

The direction `v` is ordinary DiffMean; the novelty is the learned scalar `g(h)`:

```python
c = diffmean_vector(harmful, harmless, L)               # condition vector (extract.diffmean_vector)
b = diffmean_vector(refuse, comply, L)                  # behavior direction

# candidate gates, all functions of h:
g_CAST = sigmoid((cos(h, c) - theta) / tau)
g_SCS  = sigmoid((E_c(h)/E_total(h) - 0.5) / tau)
g_disc = 1.0 if L == argmax_L fisher_ratio(L) else 0.0  # extract.fisher_ratio

# learned unifier g*(h): logistic / 2-layer MLP on [h_L, cos(h,c), E_c/E_total, layer-embed]
delta = g_star(h) * b                                   # the factorized edit; inject via add (METHODOLOGY §2)
```

### 2. Experiment procedure

```text
1. Build c, b at L16; train g*(h) on 100 harmful + 100 harmless (80/20 split), BCE on should_steer labels.
2. On held-out test: compute agreement(g*, g_CAST), agreement(g*, g_SCS), agreement(g*, g_disc).
3. Measure gate PR-AUC (harmful vs harmless), XSTest over-refusal, JailbreakBench true-positive rate.
4. Repeat over 3 train/test splits x 3 extraction seeds; report agreement matrix + PR-AUC.
```

### 3. Measurement & decision rule

- **Primary metric:** gate PR-AUC of `g*` and its agreement with each named gate.
- **Pre-registered falsifier (§3):** recovery accuracy ≤ 85% for any of CAST/SCS/disc, OR PR-AUC(g*) ≤ max(PR-AUC_CAST, PR-AUC_SCS) ⇒ unification FALSIFIED (methods are genuinely distinct).
- **Verdict logic:** KEEP only if one `g*` simultaneously hits 85/80/75% agreement on held-out prompts.

### 4. Where the code is / status

UNTESTED. The five-axis bundle (`eval.evaluate_bundle`) and safety/over-refusal probes exist, but the **learned scalar-field gate `g*(h)` training loop** and the SCS energy-ratio feature are not implemented — that missing harness is why N2 is UNTESTED.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N2.md`.
