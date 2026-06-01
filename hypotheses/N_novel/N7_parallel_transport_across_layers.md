# N7 — Behaviors are Parallel-Transported, Not Re-Learned, Across Layers

> **One-line claim:** The same behavior direction at different depths is one geometric
> object transported along the residual stream; transport-aligned multi-layer steering
> beats independent per-layer vectors with fewer parameters.
>
> **Primary axes:** A1 (where/site), A11 (dynamics/trajectory)
> **Status:** UNTESTED

---

## 1. Motivation (>= 100 words)

Multi-layer steering — injecting the same or related behavior vectors at multiple
layers simultaneously — is known to improve efficacy over single-layer injection
in some regimes. The standard implementation treats each layer independently: extract
a separate DiffMean vector at each layer, apply each independently. But this
ignores the fact that the residual stream is a dynamical system: the activation at
layer L+1 is a function of the activation at layer L. In a continuous dynamical
system, a vector "field" defined along a trajectory has a natural notion of
consistency — parallel transport — that ensures the field does not acquire spurious
components as it moves along the trajectory. In the residual stream, the transformer
blocks continuously transform the activation; the "same behavior direction" at each
layer should be the image of the behavior direction at the previous layer under this
transformation (the Jacobian of the residual block). If instead we treat each layer's
behavior vector as independently estimated, we are sampling from different points on
a single transported object and calling them independent, incurring unnecessary
estimation variance. The parallel transport hypothesis predicts: the multi-layer
behavior vector field is approximately consistent under block Jacobians, and using
this consistency constraint (transport alignment) reduces the effective parameter
count for multi-layer steering while improving coherence, because it removes the
degrees of freedom that are inconsistent with the dynamical constraints.

## 2. Formal Hypothesis (>= 50 words)

Let v_L be the DiffMean behavior vector at layer L. Let J_L = d(h_{L+1})/d(h_L)
be the Jacobian of the L-th residual block at the natural activation h_L. Define
the transport-aligned vector: v_L^T = J_{L-1} * v_{L-1}^T (initialized at the
first injection layer). The claim is:

  cos(v_L, v_L^T) >= 0.70 for all L in [L_start, L_start + 4]

and that multi-layer steering with {v_L^T} achieves >= the behavior efficacy of
{v_L} with 50% fewer degrees of freedom (one vector instead of L vectors).

## 3. Falsifier (>= 30 words)

If cos(v_L, v_L^T) < 0.50 for any layer in the tested range (indicating that the
true behavior direction deviates substantially from the transported prediction), the
parallel transport hypothesis is FALSIFIED for that range. If transport-aligned
multi-layer steering gives lower efficacy than independent per-layer vectors at
matched parameter count, the practical claim fails even if the cosine is high.

## 4. Citations (Citation Rigor >= 80 words)

```
Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. Establishes that
steering operates on M_h, which evolves across layers; the M_h at layer L+1 is
the image of M_h at layer L under the block transformation. Parallel transport of
tangent vectors on this evolving manifold is exactly the N7 construction.

Raval et al. 2026. 'Curveball Steering' arXiv:2603.09313. Multi-step steering
along the curved path is the trajectory-aware version of N7; N7 is the linearized
(tangent-parallel-transport) version of Curveball's curved flow.

GeoSteer 2026. arXiv:2601.10229 (Kazama et al.). GeoSteer steers reasoning across
CoT steps by following manifold gradients; N7 applies the same trajectory-consistency
principle to the layer-by-layer residual stream rather than CoT steps.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681. CAA applies the same
vector at all layers (a trivial form of transport alignment with J_L = I). N7
generalizes this to the actual Jacobian, predicting CAA is suboptimal because I
is a poor approximation to J_L.
```

## 5. Mechanism

The residual stream evolves as h_{L+1} = h_L + f_L(h_L), where f_L is the L-th
transformer block (MHA + MLP). The Jacobian J_L = I + df_L/dh_L captures how a
small perturbation at layer L propagates to layer L+1. For a DiffMean vector v_L
at layer L, the "natural" transported version at layer L+1 is J_L * v_L — this is
the first-order prediction of how the behavior direction at L transforms to L+1.

If the behavior is consistent (a stable latent direction), then v_{L+1} ≈ J_L * v_L,
i.e., the independently extracted vector at L+1 should be approximately the transported
version of the L-th vector. Testing this is the cosine measurement cos(v_L^T, v_{L+1}).

Practical transport-aligned steering: instead of extracting and storing N layer-specific
vectors, store only the first-layer vector v_{L_start} and compute subsequent layers'
vectors on-the-fly via J_L multiplication. This reduces memory and estimation variance
simultaneously, because each transported vector inherits the statistics of v_{L_start}
without the additional noise of a separate DiffMean estimate.

Jacobian computation: J_L does not need to be explicitly materialized (d_model x d_model
= ~4M for 2048-dim model). The matrix-vector product J_L * v_L is computed via a
backward pass: set h_L = natural activation, compute h_{L+1}, then use torch.autograd
to get (J_L^T * u) for arbitrary u, which equals (J_L * v_L) via the transpose trick.
Cost: one backward pass per layer = ~2x the forward pass cost (acceptable, done once).

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| cos(v_L, v_L^T) at L = L_start + 1 | 0.80 - 0.95 | Transport is first-order correct |
| cos(v_L, v_L^T) at L = L_start + 4 | 0.60 - 0.80 | Accumulated Jacobian approximation error |
| Efficacy of transport-aligned vs independent | -5% to +10% | Fewer parameters, less estimation noise |
| Parameter reduction | 80% (1 vector vs 5) | Only v_start needs storage |
| PPL difference transport-aligned vs independent | +/- 3% | Approximately equivalent coherence |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it
- Layers: L_start = 14, transport through L in {14, 16, 18, 20, 22}
- Vector extraction: DiffMean at each of the 5 layers independently (reference)
- Transport: compute J_L * v_{L-1} via backward pass at each layer; normalize
- Cosine measurement: cos(v_L_independent, v_L_transport) at each L
- Multi-layer steering: (a) independent vectors at all 5 layers,
  (b) transport-aligned vectors from v_14, (c) CAA (same vector at all layers)
- Behavior evaluation: behavior cosine shift at 5% off-shell displacement (E7 protocol)
- PPL: WikiText continuation log-PPL
- Seeds: 3 behavior-extraction seeds x 3 evaluation seeds
- Wall-clock: ~3 hours on RTX 4090 (Jacobian computations are the bottleneck)

### 7.2 Where it shines

Long-range multi-layer steering (5+ layers injected simultaneously) where independent
vector estimation compounds noise. Transport alignment provides a principled regularizer
that exploits the dynamical structure of the residual stream.

## 8. Cross-References

- N1 (tangent projection): N1 projects v onto the manifold tangent at each layer;
  N7 propagates v across layers via the Jacobian; combined = transport-projected
  multi-layer steering
- N9 (closed-loop control): N9 adjusts alpha at each step; N7 adjusts direction;
  combining gives a full transport-plus-feedback controller
- N19 (trajectory beats endpoint): N19's "distribute budget across layers" is most
  naturally implemented with transport-aligned vectors (N7)
- E40 (Procrustes cross-layer alignment): E40 uses Procrustes alignment of behavior
  directions across layers; N7 predicts the Procrustes result should approximate
  the Jacobian transport
- IDEA_TABLE.md: N7 row, axes A1+A11

## 9. Committee Q&A

**Q: Jacobian computation requires a backward pass — isn't this computationally
expensive compared to just extracting N independent vectors?**

> Yes, the Jacobian computation is ~2x the forward pass cost per layer, done
> once during vector extraction. This is comparable to the cost of running N
> additional DiffMean extractions. The advantage is that transport-aligned vectors
> have lower estimation variance (they share the statistics of v_start), which
> matters when the contrast-pair pool is small.

**Q: The Jacobian J_L depends on the specific activation h_L at which it is
evaluated. Which h_L do you use?**

> J_L is evaluated at the natural (unsteered) mean activation of the contrast
> set at layer L. This is the activation around which the linearization is done.
> For robustness, we also test J_L at 5 random natural activations and average
> the resulting transported vectors — this is the "mean-Jacobian transport."

**Q: If the blocks have approximately linear behavior (common for residual streams),
then J_L ≈ I + epsilon and transport is trivial. How is N7 different from CAA?**

> If J_L ≈ I, then transport-aligned vectors are approximately the same as using
> the same vector at all layers (CAA). In that case, N7 SUPPORTS CAA by providing
> a geometric explanation for why it works. The more interesting case is when J_L
> deviates significantly from I, which happens at MLP-dominated layers where
> df_L/dh_L has large eigenvalues.

## 10. Verification Checklist

- [ ] Jacobian-vector product implementation using torch.autograd.functional.jvp
- [ ] Correctness test: J_L * v matches finite-difference (h_{L+1}(h_L + eps*v) - h_{L+1}(h_L)) / eps
- [ ] cos(v_L, v_L^T) computed and plotted vs layer index for all 5 layers
- [ ] Three-way comparison: independent / transport / CAA at matched total norm
- [ ] Parameter count and efficacy per parameter recorded
- [ ] PPL difference measured and reported
- [ ] Result in EXPERIMENT_LEDGER.md, IDEA_TABLE.md N7 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. No data exists. The Procrustes
  alignment experiment (E40) is the closest planned test but has not been run.
  The transport hypothesis is motivated by differential geometry (parallel transport
  on a Riemannian manifold) applied to the discrete residual stream.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. The parallel transport idea is mathematically elegant. The practical question
is how large the deviation cos(v_L, v_L^T) is from 1.0 over multiple layers — if
the Jacobians are nearly identity, the transport is trivially accurate and the
claim reduces to "CAA works" (known). If Jacobians deviate significantly, transport
alignment is valuable but the accumulated approximation error over 4+ layers may
reduce it to noise.

### Mechanism scrutiny

The Jacobian J_L is evaluated at the natural mean activation; the steered activation
may be far from this point, making the linearization inaccurate precisely when
steering has the most effect. This "operating point" issue means transport alignment
is most accurate at small alpha and least accurate at the large alpha values needed
for strong behavior change.

### Confounds

1. The behavior direction may genuinely change across layers (not just be transported);
   different layers may encode the behavior at different levels of abstraction. In that
   case, cos(v_L, v_L^T) < 0.70 is the correct behavior, not a failure of the hypothesis.
2. The Jacobian approximation uses the same h_L for all steering intensities; for
   large alpha, the relevant operating point is the steered h_L, not the natural one.

### Does the specific transport claim matter?

MODESTLY. If transport-aligned and independent-per-layer vectors give similar efficacy
(within 5%), the claim is "true but unimportant" — the residual stream is robust
enough that direction choice at each layer has minimal impact. The practical value
only emerges if transport alignment provides a significant efficiency gain.

### Literature precedent

Parallel transport in neural networks has been studied in the context of weight-space
geometry (e.g., neural tangent kernel theory), but its application to inference-time
activation steering is novel. GeoSteer (arXiv:2601.10229) is the closest existing
work, using manifold gradients across CoT steps.

### Skeptical effect-size estimate

cos(v_L, v_L^T) at L+4: 0.45-0.65 (vs claimed 0.60-0.80). The residual stream's
Jacobians accumulate distortion over multiple layers. Efficacy: transport-aligned
within 10% of independent for 5-layer multi-steering (vs claimed -5% to +10%).
The parameter reduction benefit is real, but efficacy benefit is uncertain.

### Minimum distinguishing experiment

Single layer transition: L14 to L16. Compute J_14 * v_14; compare cos to v_16.
Cost ~30 min. If cos > 0.70, transport is accurate at one step; proceed to 4-step
protocol. If cos < 0.50, transport degrades rapidly and the multi-layer version
is not worth testing.

### Verdict

TESTABLE-LOW-to-MEDIUM confidence. The single-step cosine test (30 min) is the
essential pre-screen. If cos(v_L, J_{L-1}*v_{L-1}) < 0.50 even for one step,
the residual stream is not behaving like a smooth dynamical system with coherent
parallel transport, and the hypothesis fails immediately. The full 4-step protocol
is warranted only after the single-step test passes.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N7.md`.
