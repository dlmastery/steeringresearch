# N6 — Gating Belongs in the Read, Not the Write (Separation Principle)

> **One-line claim:** Forcing cos(condition_vector, behavior_vector) = 0 reduces
> XSTest-style over-refusal without lowering true-positive refusal rate, because
> detection and execution are more robust in orthogonal subspaces and entangling
> them is the root cause of over-refusal.
>
> **Primary axes:** A5 (condition/gate), A2 (direction/what)
> **Status:** UNTESTED

---

## 1. Motivation (>= 100 words)

Activation steering for safety uses two conceptually distinct operations: a CONDITION
step that detects whether steering is needed (reading the activation to classify
harm) and an EXECUTION step that performs the behavioral change (writing a refusal
or evasion direction into the residual stream). In most implementations these are
entangled: the same vector v serves both as the discriminating direction (is this
harmful?) and the behavioral direction (make the model refuse). When the detection
and execution vectors are the same, any ambiguous input that partially projects onto
v will trigger both detection AND execution simultaneously — producing over-refusal
on benign inputs that superficially resemble harmful ones. The information-theoretic
separation principle suggests that robust control systems separate sensing from
actuation. In LLM steering terms: the condition vector c should span directions
orthogonal to the behavior vector b, so that measuring h·c does not directly
predict whether h·b will create a behavioral shift. If c and b are orthogonal,
a high projection onto c (detecting a harmful input) is informative but does not
itself constitute the behavioral perturbation. Conversely, the behavioral shift b
does not interfere with the detection signal c. The experiment tests whether this
orthogonality constraint, imposed by Gram-Schmidt projection of b against c, reduces
over-refusal on XSTest-style benign-but-look-alike prompts while keeping true-positive
refusal intact.

## 2. Formal Hypothesis (>= 50 words)

Let c be the DiffMean condition vector (harmful vs harmless) and b the behavior
vector (refusal vs compliance). Define b_orth = b - (b·c/||c||^2)*c (Gram-Schmidt
projection). The claim is:

  Over-refusal(b_orth) < 0.7 * Over-refusal(b_raw)     [30%+ reduction]
  True-positive-refusal(b_orth) > 0.95 * True-positive-refusal(b_raw)   [< 5% loss]

measured on a balanced XSTest-style set (100 benign look-alike + 100 genuinely
harmful prompts) with matched gate activation threshold on Gemma-3-1B-it.

## 3. Falsifier (>= 30 words)

If over-refusal reduction is < 15% (vs claimed >= 30%), or if true-positive refusal
loss is > 10% (vs claimed < 5%), the separation principle for this specific
implementation is FALSIFIED. If over-refusal reduces but true-positive rate also
falls proportionally, the gain is from magnitude reduction not orthogonalization.

## 4. Citations (Citation Rigor >= 80 words)

```
Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.01844 — the bidirectional
M_h <-> M_y link; detection lives on the boundary of M_harm while execution
moves along M_harm's interior. N6's orthogonality claim follows: boundary-normal
(detection) and interior-tangential (execution) directions are orthogonal by
manifold geometry.

Venkatesh & Kurapath 2026. 'On the Non-Identifiability of Steering Vectors'
arXiv:2602.06801. The coset structure of behaviorally equivalent vectors means
there exist many b vectors that achieve the same refusal rate; b_orth is a specific
coset representative chosen for orthogonality with c. N6 is the safety-application
of N15's coset-min-collateral principle.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681. CAA's refusal vector
entangles detection and execution (same pair of contrastive prompts). N6 tests
whether separating the extraction of c and b improves calibration.

Gao et al. 2026. 'CRH' arXiv:2605.01844. CRH decomposes activation differences
into radial and angular; the c vector is primarily radial (intensity of harm signal)
while b is angular (direction of behavioral shift). Their empirical separation
motivates N6's orthogonality claim at a geometric level.
```

## 5. Mechanism

When cos(c, b) > 0, the gate activation h·c is partially predicting the behavioral
effect h·b, because they project onto overlapping subspaces. A benign prompt with
an ambiguous topic (e.g., "how to safely handle chemicals") may have a moderate
h·c projection (resembles harmful queries in topic space) and thus trigger the gate,
applying the b direction even though the true behavioral goal is not refusal. This
is the over-refusal mechanism.

Orthogonalization eliminates the predictor overlap: after Gram-Schmidt, b_orth has
zero projection onto c, so h·c provides no information about whether applying b_orth
will help or hurt. The gate fires based entirely on h·c (clean detection), and the
behavioral shift b_orth is applied independently. The two operations are now decoupled.

True-positive preservation: the projection b_orth = b - (b·c/||c||^2)*c removes the
c-component of b. For genuinely harmful inputs, h is in the harmful region and both
h·c and h·b are large — but the detection is via h·c, and b_orth's orthogonal component
still shifts the residual stream toward refusal. The claim is that the c-component of
b (removed by orthogonalization) is not the behaviorally active component for genuinely
harmful inputs — it is a "noise" component that only hurts benign inputs.

Mathematically: if b = b_c + b_orth where b_c is the c-parallel component, the
behavior of b on harmful inputs is achieved by both; b_orth alone is sufficient
if the behavioral direction in the harmful manifold is primarily orthogonal to c.
This will hold when c captures the "topic" signal (what the input is about) and
b captures the "response style" signal (how to reply) — which are geometrically
separated in most contrastive-pair extraction setups.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| Over-refusal reduction on XSTest look-alikes | 30% - 50% relative | Removing c-component eliminates false alarm |
| True-positive refusal loss | 0% - 5% relative | c-component of b is not the active signal |
| cos(c, b_orth) after projection | < 0.01 | Exact Gram-Schmidt projection |
| Behavior efficacy on clear harmful prompts | -3% to +2% | b_orth preserves the orthogonal component |
| ||b_orth|| / ||b|| | 0.80 - 0.97 | Depends on cos(c,b); typically small projection |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16 (established optimal layer)
- Condition vector c: DiffMean(harmful, harmless) from 50 contrast pairs per category
  (JailbreakBench harmful + XSTest harmless)
- Behavior vector b: DiffMean(refusal-format, compliance-format) from 50 output-style
  contrast pairs
- Projection: b_orth = b - (b·c/||c||^2)*c; verify cos(b_orth, c) < 0.01
- Evaluation: 100 XSTest benign look-alikes (over-refusal rate) + 100 JailbreakBench
  harmful (true-positive rate)
- Gate: same cosine threshold against c for both b and b_orth conditions
- alpha: matched to give same behavior efficacy on clearly harmful (no ambiguity) set
- Metrics: over-refusal rate, true-positive rate, behavior efficacy, ||b_orth||/||b||
- Seeds: 3 extraction seeds x 3 evaluation prompt seeds
- Wall-clock: ~3 hours on RTX 4090

### 7.2 Where it shines

Policies where safety and helpfulness must coexist: XSTest-style evaluations where
over-refusal on benign-but-similar inputs is penalized. The separation principle has
the most value when the condition vector c and behavior vector b are naturally
entangled by the extraction process (sharing the same harmful/harmless contrast pairs).

## 8. Cross-References

- N2 (conditioning = curvature factorization): N6 is the orthogonalization step that
  makes N2's g(h)*v factorization clean — g(h) is a function of c, v is b_orth,
  and they operate in orthogonal spaces
- N15 (coset min-collateral): b_orth is a specific coset rep (the c-orthogonal one);
  N15 chooses the coset rep differently (min projection onto fragile subspaces);
  these are complementary choices
- E32 (refusal vs detection direction separability): E32 measures the natural
  cos(c, b); if low, N6 is trivially satisfied; if high, N6's projection is needed
- E42 (gate cuts over-refusal): E42 tests gating as the remedy; N6 tests orthogonalization
  as an independent remedy that can stack with gating
- N12 (capstone): the separation of g(h) (detection) from v (execution) in the unified
  operator is the formal embodiment of N6
- IDEA_TABLE.md: N6 row, axes A5+A2

## 9. Committee Q&A

**Q: Gram-Schmidt projection changes the magnitude of b. How do you ensure the
comparison is fair (not just reduced magnitude reducing over-refusal)?**

> After projection, alpha is re-calibrated to restore the same behavior efficacy on
> clearly harmful prompts (where both c and b projections are unambiguous). This
> holds ||steered_effect|| constant on the positive class while measuring the
> over-refusal effect on the negative class. A separate magnitude-matched control
> (scale b to ||b_orth|| without orthogonalizing) is included to isolate the
> geometric effect from the magnitude reduction effect.

**Q: What if cos(c, b_raw) is already small (< 0.1)? Then orthogonalization has
no effect and the experiment tests nothing.**

> E32 is pre-run to measure cos(c, b). If E32 finds |cos| < 0.1 naturally, N6
> is trivially satisfied — detection and execution are already separated. In that
> case, N6's status becomes `VACUOUSLY SUPPORTED` and the experiment is not run.
> The interesting regime is |cos(c,b)| >= 0.2.

**Q: Why wouldn't you just train separate condition and behavior vectors from the
start, rather than orthogonalizing post-hoc?**

> Post-hoc orthogonalization is the minimal intervention and the cleanest test:
> it isolates the geometric effect without confounding training-set differences.
> If N6 holds, the follow-up is joint training with an orthogonality regularizer.

## 10. Verification Checklist

- [ ] E32 run first to measure natural cos(c, b); document if < 0.1 (trivially satisfied)
- [ ] Gram-Schmidt projection implemented; verify cos(b_orth, c) < 0.01
- [ ] Magnitude-matched control (scaled b_raw to ||b_orth||) implemented
- [ ] Over-refusal rate measured on 100 XSTest benign look-alikes for b_raw and b_orth
- [ ] True-positive rate measured on 100 JailbreakBench harmful for b_raw and b_orth
- [ ] Alpha re-calibration: same efficacy on clearly harmful (unambiguous) prompts
- [ ] 3 x 3 seed design recorded in EXPERIMENT_LEDGER.md
- [ ] IDEA_TABLE.md N6 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. The separation principle is a
  clean theoretical prediction. E32 (natural cos(c,b) measurement) is a prerequisite
  and is also UNTESTED. The experiment requires working safety evaluation infrastructure
  (JailbreakBench + XSTest), which per FINDINGS.md is "in progress" but not yet deployed.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. The control-theoretic separation of sensing and actuation is a well-established
engineering principle (Kalman filter, PID control). Its application to LLM steering
vectors is novel but the intuition is sound. The main uncertainty is whether DiffMean
extraction naturally produces high cos(c,b) — if it does not, the experiment is vacuous.

### Mechanism scrutiny

The mechanism assumes that the c-component of b is specifically responsible for
over-refusal (not other components of b that happen to match benign-look-alike
activations). This is testable: a random orthogonalization of b (projecting off a
random vector of same magnitude as c) should not reduce over-refusal, while the
c-projection should. Including this random-projection control is essential.

### Confounds

1. Magnitude confound: as noted, alpha re-calibration is essential.
2. Random-projection confound: must show c-orthogonalization specifically (not any
   dimensionality reduction) reduces over-refusal.
3. Threshold confound: the gate threshold may need re-tuning after orthogonalization;
   if the gate is loosened to match the same true-positive rate, over-refusal reduction
   may come from threshold change not orthogonalization.

### Does the specific orthogonality claim matter?

MODERATELY. The over-refusal problem is real and practically important (XSTest-style
failures are a known deployment issue). If orthogonalization achieves 30-50% reduction
at < 5% true-positive cost, it is a valuable, computationally free post-processing step
applicable to any steering system.

### Literature precedent

Disentangled representation learning (beta-VAE, Higgins et al. 2017) pursues exactly
this orthogonality in latent space; applying it to steering vector pairs is novel but
grounded. Conceptor theory (Jaeger 2014) uses orthogonal projection for multi-behavior
composition; N6 is the two-vector version applied to safety.

### Skeptical effect-size estimate

Over-refusal reduction: 10-25% (vs claimed 30-50%). The c-component of b may be
small if DiffMean extraction naturally produces near-orthogonal c and b (since they
are extracted from different types of pairs). True-positive preservation: likely 95%+
(high confidence this part holds). The practical case is strong regardless of whether
the 30% threshold is met.

### Minimum distinguishing experiment

E32 first (5 min): measure cos(c, b) on Gemma-3-1B @L16. If |cos| < 0.10, N6 is
vacuous. If |cos| in [0.15, 0.40], the orthogonalization has a measurable effect;
run the minimum over-refusal test on 20 XSTest look-alikes (20 min). Total: 25 min.

### Verdict

TESTABLE-MEDIUM. The separation principle is sound; the effect size is uncertain.
E32 is the critical pre-check (5 min). If cos(c,b) is naturally low, the hypothesis
is vacuously satisfied and effort should go elsewhere. If cos(c,b) >= 0.20, this
is a cheap and potentially high-value intervention.
