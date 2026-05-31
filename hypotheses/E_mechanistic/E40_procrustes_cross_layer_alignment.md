# E40 — Procrustes Cross-Layer Alignment Improves Multi-Layer Steering

> **One-line claim:** The same behavior direction at different transformer
> depths is the parallel transport of one underlying direction along the
> residual stream; Procrustes-aligning per-layer directions before multi-
> layer injection yields better efficacy and coherence than treating each
> layer's direction as independent.
>
> **Block:** E — Mechanistic and interpretability-guided (E34-E40).
> **Primary axes:** A1 (WHERE — site/layer) + A11 (DYNAMICS — trajectory).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

When a behavior direction v is extracted independently at each layer l of
a transformer (v_l = DiffMean at layer l activations), the set {v_l} is
not a single consistent direction — it is a family of directions that
evolve across layers. The residual stream is an additive highway (h_{l+1}
= h_l + delta_l), so a direction at layer l is related to a direction
at layer l+1 by the additive contribution of attention and MLP blocks.
In differential geometry, this evolution is described as parallel transport
along a curve (the residual stream trajectory) on the activation manifold.
Novel hypothesis N7 in our corpus makes this explicit: the behavior
direction {v_l} is not re-learned at each layer but is one direction
transported along the residual stream. If N7 is correct, then the standard
practice of extracting an independent v_l at each layer and injecting
each independently is wasteful and potentially incoherent — it is as if
one were treating the multiple measurements of a single moving object as
independent objects. The Procrustes-alignment approach (the orthogonal
rotation Q* that minimises sum_l ||v_{l+1} - Q v_l||^2) recovers the
"transport operator" connecting successive layer directions, and the
transported field (Q^{l-1} v_0) is a more principled multi-layer injection
than independent per-layer directions. The test in E40 is whether using
the Procrustes-aligned multi-layer injection improves either efficacy or
coherence relative to the standard practice. The connection to the Rogue
Scalpel guard framework is also notable: if steering vectors at different
layers are the same underlying direction, Guard Layer C (avoid fragile
layers) should be applied by excluding the fragile layer from the transport
path rather than simply skipping that layer's independent direction.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, the Procrustes alignment error between successive
layers' behavior directions (||v_{l+1} - Q* v_l||^2 / ||v_{l+1}||^2) is
significantly lower than chance alignment (i.e., random rotation), confirming
that a near-consistent transport operator exists. Furthermore, multi-layer
injection using Procrustes-transported directions (applying Q^{l-l_0} v_0
at each layer l instead of the independently extracted v_l) will achieve
behavior-success rate within 5% and perplexity within 5% of independent per-
layer injection while requiring the storage of only the base direction v_0
and the transport operator Q, instead of L separate direction vectors.

---

## 3. Falsifier (>= 30 words)

If the Procrustes alignment error does NOT significantly differ from chance
across layers (the null hypothesis that {v_l} are random rotations of each
other), OR if Procrustes-transported multi-layer injection achieves > 10%
LOWER behavior success than independent per-layer injection, the parallel-
transport hypothesis is DISCARDED (Status `x disproved`).

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Schonemann, Peter H. 1966 'A Generalized Solution of the Orthogonal
Procrustes Problem' Psychometrika 31:1-10 — the original Procrustes
algorithm; the mathematical tool for aligning successive layer directions
by an orthogonal rotation; provides the optimization problem (minimize
||A - BQ||^2 s.t. Q^T Q = I, solved by SVD of B^T A).

Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel; its multi-layer perturbation
findings (F3: damage peaks at early-middle layers) are directly relevant
to the question of how steering should be distributed across layers;
the transport-aware injection suggested by E40 provides a principled
layer-weight profile that concentrates injection away from the fragile band.

Zou, Andy, et al. 2023 'Representation Engineering: A Top-Down Approach
to AI Transparency' arXiv:2310.01405 — CAA / DiffMean; the per-layer
direction extraction methodology that E40 extends with cross-layer
Procrustes alignment.

Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive Activation
Addition' arXiv:2312.06681 — CAA; multi-layer injection variants are
discussed in this paper; E40 provides the theoretically motivated
alternative to ad-hoc multi-layer averaging.
```

---

## 5. Mechanism

For a transformer with L layers, let v_l = DiffMean extracted at layer l.
The parallel transport hypothesis (N7) predicts:

    v_{l+1} ≈ Q_l v_l + epsilon_l

where Q_l is a near-orthogonal rotation (the transport operator for step l)
and epsilon_l is noise. The transport operator Q_l can be estimated by
Procrustes alignment: given a matrix V = [v_0 | v_1 | ... | v_{L-1}] and
V' = [v_1 | v_2 | ... | v_L], solve:

    Q* = argmin_Q ||V' - V Q||^2  s.t. Q^T Q = I

via SVD of V^T V' = U Sigma W^T, giving Q* = W U^T.

Once Q* is found, the transported direction at layer l is:

    v_transported_l = (Q*)^{l - l_0} v_{l_0}

Multi-layer injection with transported directions:
    h_l <- h_l + alpha_l * v_transported_l  (for l in injection_range)

versus the baseline (independent per-layer):
    h_l <- h_l + alpha_l * v_l              (separate extraction per layer)

If the transport hypothesis holds, the Procrustes error ||v_{l+1} - Q* v_l||^2
will be small relative to ||v_{l+1}||^2, confirming a near-consistent
rotation exists. The functional test (behavior-success and PPL) then asks
whether using v_transported instead of v_l in injection changes outcomes.

Connection to N5/norm budget: transporting v_0 may produce different norms
at each layer than the independently extracted v_l; the relative_add
parameterization (alpha = fraction of ||h||) must be used to normalise
injection magnitude across layers.

---

## 6. Predicted Delta

| Metric | Procrustes-transported vs independent per-layer |
|---|---|
| Procrustes alignment error vs chance | significantly lower (< 0.3 relative Frobenius) |
| Behavior success rate | within 5% (no significant loss or gain) |
| PPL (WikiText-103) | within 5% |
| Storage cost | L x fewer direction vectors (only v_0 + Q stored) |

The primary prediction is that Procrustes-aligned injection is EQUIVALENT
to per-layer injection (within 5%), not that it dramatically outperforms it.
The theoretical gain is in interpretability (one direction + transport) and
in identifying the structure of the residual stream's transport geometry.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Behavior: refusal (primary); sycophancy (secondary).
- Step 1: extract v_l by DiffMean at EVERY layer l in {0, ..., L-1} from
  the same contrast set of >= 50 pairs; unit-normalize each.
- Step 2: fit the Procrustes transport operator Q* on the set {v_l, v_{l+1}}
  across all adjacent layers; compute alignment error relative to ||v_{l+1}||.
- Step 3: compare alignment error to a random-rotation baseline (shuffle
  layer assignments).
- Step 4: multi-layer injection with (A) independent per-layer v_l at each
  injection layer; (B) Procrustes-transported directions at each layer.
- Injection layers: the non-fragile band (exclude fragile mid-layer per Rogue
  Scalpel C guard; use the layer range identified in prior sweeps).
- Alpha: relative_add at alpha = 0.10 per layer (or alpha / num_injection_layers).
- Eval: behavior-success (LLM-judge), WikiText-103 PPL, MMLU-500.
- Seeds: 3 (screening), 7 for rung-3.

### 7.2 Where it shines

This experiment provides the mechanistic test of N7 (parallel transport
hypothesis) and establishes whether the residual stream geometry has a
consistent transport structure. If Q* is near-identity across all layers,
the behavior direction is literally unchanged by the transformer's processing
— a strong structural claim. If Q* is a non-trivial rotation, it reveals
how the transformer "rotates" the concept direction as it processes information.

---

## 8. Cross-references

- IDEA_TABLE.md Block E row E40.
- N7 (parallel transport across layers): E40 is the direct empirical test
  of N7's prediction; the two are companion hypotheses.
- N5 (norm budget): cross-layer injection must respect the total edit budget;
  use relative_add across layers.
- Rogue Scalpel Guard Layer C: the fragile-layer exclusion should be applied
  to the transport path; the guard-compliant injection range is the transport
  domain.
- E19 (Gram-Schmidt orthogonalization): if the transport operator introduces
  alignment between successive directions, orthogonalisation at each step
  may be necessary for clean stacking.
- arXiv:2312.06681 (CAA): multi-layer injection variants.

---

## 9. Committee Q&A

**Q: The residual stream is not a Riemannian manifold in the formal sense,
so "parallel transport" is a metaphor. Isn't Procrustes alignment just
fitting a rotation to data, without the geometric interpretation?**

> The "parallel transport" framing is a theoretical motivation, not a formal
> geometric claim. Procrustes alignment is a specific, well-defined matrix
> operation (SVD of V^T V'). Whether it deserves the "transport" interpretation
> depends on whether the fitted Q* is near-orthogonal (||Q* - Q||_F small)
> and consistent across adjacent layers. The alignment error check is the
> empirical test of whether the metaphor is apt. If Q* is far from the
> previous Q (high variance across layers), the transport interpretation
> is unwarranted.

**Q: What if the optimal multi-layer injection does not use uniform transport
but adapts Q at each layer?**

> The experiment includes a sub-condition (B2) where Q_l is estimated
> independently for each adjacent pair (l, l+1), yielding layer-specific
> transport operators. This tests whether a single global Q* (the simplest
> transport assumption) suffices, or whether layer-adaptive transport is
> needed.

**Q: How does this interact with the fragile-layer exclusion from Rogue
Scalpel Guard C?**

> The fragile layer is excluded from the injection range in all conditions.
> The transport path is computed over the non-fragile layers. The experiment
> also reports the alignment error at the fragile layer as a secondary
> analysis — if the fragile layer has anomalously high transport error, it
> may explain why it is the point of instability.

---

## 10. Verification checklist

- [ ] Per-layer DiffMean extraction at all L layers confirmed (vector per
      layer, not just injection layer).
- [ ] Procrustes alignment implemented correctly: SVD of V^T V' gives
      Q* = W U^T; verified on a synthetic rotation test case.
- [ ] Alignment error normalised by ||v_{l+1}||^2 (relative, not absolute).
- [ ] Random-rotation baseline computed (shuffle layer assignments) for
      significance test.
- [ ] Fragile layer excluded from injection range in all conditions.
- [ ] Relative_add used for matched fractional alpha across layers.
- [ ] Behavior eval on real generated text (LLM-judge).
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block E, hypothesis E40.
  Status: `o UNTESTED`. Theoretically motivated by N7 (parallel transport
  hypothesis) and the residual-stream additive-highway structure from the
  first-principles corpus. No prior screening run. Dependency: per-layer
  DiffMean extraction (trivial extension of existing extraction code) and
  Procrustes SVD computation (standard numpy/torch).

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-E (differential geometry + transformer specialist).*

### Prior plausibility
**MEDIUM.** The Procrustes-alignment step is mathematically well-defined.
Whether a single transport operator Q* fits all adjacent layers is an
empirical question. Transformers are known to specialise across layers
(early = syntactic, late = semantic), which may mean Q* varies substantially
across layers and a single global operator is a poor fit.

### Mechanism scrutiny
The mechanism is well-specified. The key risk is that the Procrustes error
comparison to "chance" (random rotation) may not be a meaningful null.
A better null would be: does the Procrustes-transported direction steer as
well as a randomly rotated direction? This tests whether the transport
operator is doing something meaningful vs. just being a fitted rotation.

### Confounds
1. The DiffMean vector norm varies across layers (the residual stream
   accumulates magnitude); without unit-normalisation, the Procrustes
   alignment is fitting both rotation AND scale, which is ill-defined for
   the transport interpretation. The protocol must unit-normalise before
   fitting Q*.
2. Procrustes assumes an orthogonal rotation, but the true "transport" on
   a non-flat manifold is not generally orthogonal. If the manifold
   curvature is high, the Procrustes approximation may be poor.

### Expected effect size
My prior: the Procrustes-transported injection is within 5-15% of
independent per-layer injection (not dramatically better or worse). The
theoretical gain in storage efficiency and interpretability is real, but
the functional gain in efficacy is likely small.

### Verdict
**TESTABLE + THEORETICALLY INTERESTING** — A clean test of the parallel
transport hypothesis. The primary value is in establishing the transport
geometry of the residual stream, not in achieving a large functional
improvement. Even a negative result (transported != independent) is
informative about the non-flat nature of the residual stream.
