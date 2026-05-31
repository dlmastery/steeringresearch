# N19 — Trajectory Beats Endpoint

> **One-line claim:** Distributing a fixed total intervention budget across the forward
> trajectory (small nudges per layer/step) produces less collateral damage than one
> large endpoint shift at equal total behavioral effect.
>
> **Primary axes:** A11 (dynamics/trajectory), A1 (where/site)
> **Status:** UNTESTED

---

## 1. Motivation (>= 100 words)

Standard activation steering applies a single perturbation at one layer (CAA, DiffMean)
or at a small set of layers simultaneously. The total intervention is concentrated at
a few points in the forward computation. An alternative is to distribute the same total
budget across ALL layers of the forward pass: instead of one large alpha*v at layer L,
apply (alpha/N)*v at each of N layers. This "distributed steering" has two potential
advantages. First, each individual nudge is small, reducing the off-shell displacement
at each layer (N17 benefit: less LayerNorm saturation per step). Second, the effect is
applied throughout the computation, not just at one layer, allowing the transformer to
"process" each nudge through the downstream layers before the next nudge is applied.
This is analogous to the difference between integrating a differential equation with
large step sizes (endpoint steering) vs small step sizes (trajectory steering): small
steps follow the true trajectory more accurately. GeoSteer (arXiv:2601.10229) reports
+0.9 acc / +4.5 reasoning quality from trajectory-aware steering vs baseline; their
method is a specific form of trajectory steering along the manifold gradient. N19
tests the simplest version of the trajectory hypothesis: uniform distribution of
the steering budget across all layers, without manifold gradient computation.

## 2. Formal Hypothesis (>= 50 words)

Let endpoint-steering be: apply alpha*v at layer L only. Let trajectory-steering be:
apply (alpha/N)*v at each of N layers {L_1, ..., L_N} where L_N = L (same final layer).
Total budget: ||alpha*v|| = ||sum_i (alpha/N)*v|| (same aggregate magnitude). The claim is:

  At matched total behavior-cosine shift (within 5%):
  (A) mean off-shell displacement per-layer is lower for trajectory vs endpoint;
  (B) log-PPL(trajectory) <= log-PPL(endpoint) * 0.90 (at least 10% better);
  (C) rogue compliance rate CR(trajectory) <= CR(endpoint) * 0.85 (15% better).

Tested on Gemma-3-1B-it with trajectory spanning N=5 layers vs endpoint at L_5.

## 3. Falsifier (>= 30 words)

If log-PPL(trajectory) >= log-PPL(endpoint) at matched behavior for any N in {3, 5},
claim (B) is FALSIFIED. If the per-layer off-shell displacement is not lower for
trajectory vs endpoint (claim A fails), the mechanism claim fails. Full FALSIFIED
requires both A and B to fail.

## 4. Citations (Citation Rigor >= 80 words)

```
Kazama et al. 2026. 'GeoSteer: Faithful Chain-of-Thought Steering via Latent
Manifold Gradients' arXiv:2601.10229. GeoSteer demonstrates trajectory-aware
multi-step steering achieving +0.9 acc / +4.5 reasoning quality over baseline
[NEEDS REPLICATION]. N19 is the simplified version of GeoSteer: trajectory without
manifold gradient computation, testing whether trajectory alone (vs endpoint) explains
the gain.

Raval et al. 2026. 'Curveball Steering' arXiv:2603.09313. Curveball steers along
a curved path in one step; N19's trajectory is a discretization of the same idea
across layers rather than within one layer. The N19 prediction: trajectory across
layers provides similar benefits to Curveball's within-step curvature correction.

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. The M_h manifold evolves
across layers; trajectory steering allows each nudge to be "absorbed" by the manifold
dynamics before the next nudge, staying closer to M_h throughout.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681. The endpoint baseline;
N19 tests whether distributing the same budget outperforms concentration at one layer.
```

## 5. Mechanism

The mechanism has two components:

Component 1 — Per-step off-shell reduction: at each layer i, the off-shell displacement
is delta_r_i = (alpha/N) * cos(v, h_i) / ||h_i|| (first order). Since (alpha/N) < alpha,
each individual step causes less off-shell displacement than the full endpoint step.
The total off-shell displacement across N layers is sum_i delta_r_i, which may exceed
the single-step displacement (if they add coherently) or may be less (if they partially
cancel). The prediction depends on the sign structure of cos(v, h_i) across layers;
if cos(v, h_i) > 0 at all layers, the distributed steps sum to approximately the same
total displacement. The KEY advantage is not the TOTAL off-shell but the MAXIMUM
per-layer off-shell: capping at (alpha/N) per layer keeps each step in the linear
regime of the N17 law, avoiding the super-linear PPL increase that occurs at the cliff.

Component 2 — Distributed processing: after each nudge at layer L_i, the downstream
layers L_{i+1}, ..., L_{i+k} process the nudged activation through their MHA and MLP
circuits. This allows the model to "accommodate" the perturbation through its normal
computation before the next nudge. This is analogous to smooth feedback control: rather
than a step change in the input, small distributed changes allow the system to track
the target continuously.

The N17 law predicts: for each individual step at (alpha/N), log-PPL contribution ≈
a + b * (alpha/N) * cos(v, h)/||h|| (linear regime). For the full endpoint at alpha,
log-PPL ≈ a + b * alpha * cos(v, h)/||h||. If the distributed steps stay in the
linear regime (no super-linear cliff at any individual step), the trajectory version
achieves the same behavior at approximately the same PPL, which is better than the
endpoint if the endpoint's alpha is in the super-linear regime.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| PPL improvement trajectory vs endpoint | 10% - 25% reduction | Trajectory stays in linear N17 regime |
| Per-layer off-shell: max across trajectory | < 50% of endpoint's off-shell | (alpha/N) vs alpha per step |
| Total behavior-cosine shift | Within 5% of endpoint | Same total budget |
| Rogue compliance reduction | 15% - 30% relative | Less per-step off-shell excursion |
| Degradation at very large N (N=20) | Minimal additional gain | Diminishing returns from N >> 5 |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it
- Behaviors: 3 (refusal, politeness, factuality)
- Trajectory configurations: N in {1 (endpoint, baseline), 3, 5, 10}
  Layer sets: {L16} (endpoint), {L12, L14, L16} (N=3), {L10, L12, L14, L16, L18} (N=5)
- Total budget: alpha chosen so that N=5 trajectory gives delta_cos = 0.08
  Applied per step: alpha/N at each layer
- Metrics: log-PPL, per-layer off-shell at each nudge layer, total off-shell,
           behavior-cosine shift, rogue compliance (adversarial probe)
- Matched behavior: sweep endpoint alpha to match trajectory's delta_cos within 5%
- Seeds: 3 behavior-extraction x 3 evaluation = 9 cells per (N, behavior)
- Wall-clock: ~4 hours on RTX 4090

### 7.2 Where it shines

Large total steering budgets (strong behavior change required): when the required
behavior change would put the endpoint in the super-linear PPL regime (past the
cliff), distributing the budget across N layers keeps each step below the cliff,
potentially enabling stronger total steering without coherence collapse.

## 8. Cross-References

- N7 (parallel transport): transport-aligned vectors provide the correct v for each
  layer's nudge in the trajectory; N7 is the direction counterpart, N19 is the
  magnitude/schedule counterpart
- N9 (closed-loop control): N9 adjusts magnitude over the GENERATION trajectory (time);
  N19 distributes over the FORWARD PASS trajectory (layers); they are orthogonal time
  axes of the same trajectory-vs-endpoint idea
- N5 (norm-budget): the budget cap B applies to the TOTAL trajectory; N19 ensures no
  individual step exceeds B/N
- N17 (concentration penalty): each trajectory step's PPL is governed by N17's law;
  trajectory keeps all steps in the linear regime
- N13 (geodesic): N13 follows the geodesic within one layer; N19 follows the "layered
  geodesic" across the full forward pass
- GeoSteer (arXiv:2601.10229): the manifold-gradient version of N19
- IDEA_TABLE.md: N19 row, axes A11+A1

## 9. Committee Q&A

**Q: The trajectory applies v at multiple layers. But the behavior direction v is
extracted at one specific layer (L16). Is v the right direction at layers L10-L14?**

> This is the key mechanism question. If behavior directions are parallel-transported
> across layers (N7), then the same v (or its transported version) is appropriate at
> each layer. In the simplified N19 protocol, we use the same v at all trajectory
> layers; this is the control condition. An augmented protocol uses N7's transported
> vectors at each layer and tests whether this improves the trajectory further.
> The primary N19 test uses fixed v; the N7+N19 composition is a secondary test.

**Q: If v is applied at N=5 layers, each at (alpha/N), doesn't the TOTAL effect
compound multiplicatively (the activations at L16 have been nudged 5 times)? This
could produce a larger total effect than intended.**

> Yes, the total effect is NOT simply alpha but the cumulative effect of (alpha/N)
> nudges through N transformer blocks. The protocol matches behavior-cosine shift
> rather than matching alpha — so the endpoint alpha is tuned to achieve the same
> delta_cos as the N=5 trajectory. This ensures the behavioral comparison is fair
> regardless of the cumulative effect scaling.

**Q: What if trajectory steering has no advantage because the model's layers are
approximately identity maps (residual stream passes through nearly unchanged)?
Then each layer's nudge is equivalent and there's no "absorption" benefit.**

> If the model's layers are near-identity (small MLP/MHA contribution), then the
> trajectory reduces to N copies of (alpha/N)*v applied to the same activation —
> equivalent to alpha*v at that one activation. In this case, the trajectory provides
> no absorption benefit and the hypothesis reduces to a null result. The test of
> whether layers are near-identity is the Jacobian analysis from N7 (cos(J_L*v, v));
> if the Jacobians are near-identity, N19 is predicted to show no advantage.

## 10. Verification Checklist

- [ ] Multi-layer injection implemented for trajectory configurations
- [ ] Per-layer off-shell measurement at each nudge layer in the trajectory
- [ ] Matched behavior protocol: delta_cos within 5% for endpoint vs all N values
- [ ] log-PPL, behavior-cosine, rogue-compliance for all N in {1, 3, 5, 10} x 3 behaviors x 9 seeds
- [ ] N7 Jacobian analysis run as prerequisite: cos(J_L*v, v) at adjacent layers
- [ ] N7+N19 composition protocol (if N7 shows cos < 0.90 for adjacent layers)
- [ ] IDEA_TABLE.md N19 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. GeoSteer (arXiv:2601.10229)
  reports trajectory gains but has not been replicated on our Gemma models. The
  simplest version of N19 (uniform distribution across layers, no manifold gradient)
  is a more achievable first test. N7 (Jacobian transport) is a prerequisite to
  understand whether v is the right direction at each trajectory layer.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. The intuition (small steps are gentler than large steps) is sound, and the N17
law directly predicts that trajectory steps stay in the linear regime while the endpoint
may not. The main empirical question is whether the cumulative effect of N small steps
equals the total steering of one large step — if not, the matched-behavior comparison
requires alpha-tuning that may make the trajectory less interpretable.

### Mechanism scrutiny

The "absorption" mechanism (the model processes each nudge through downstream computation
before the next nudge is applied) is the novel claim that distinguishes N19 from simply
splitting the budget. This mechanism requires that the downstream computation CHANGES
the effective residual stream in a way that makes the next nudge more appropriate.
If the downstream computation is a near-identity (as in a very shallow model), there
is no absorption benefit and the trajectory reduces to the endpoint.

### Confounds

1. Layer-specific effects: applying v at layer L10 may not produce the same behavior
   shift as applying it at L16 (which is the established optimal layer). The matched-
   behavior protocol controls for this, but the MECHANISM of the trajectory benefit
   (absorption vs just smaller per-step excursion) requires disentangling.
2. The cumulative effect of N nudges is model-specific; for a 1B model with significant
   residual connections, earlier nudges may have stronger cumulative effects than later
   ones, making the trajectory unequal even when per-step alpha is constant.

### Does the trajectory distribution specifically matter?

MODERATELY. GeoSteer reports +0.9 acc / +4.5 reasoning points from trajectory-aware
steering, but this includes manifold gradient computation not present in N19's
simplified version. The simplified trajectory (uniform distribution, fixed v) may
provide 0.3 to 2.0 reasoning points — still practically useful if real.

### Skeptical effect-size estimate

PPL improvement trajectory vs endpoint: 5-15% (vs claimed 10-25%). The main risk
is that the per-step N17 advantage (smaller off-shell per step) is offset by the
cumulative effect of repeated nudges producing larger total off-shell than a single
endpoint step (if cos(v, h) is consistently positive across trajectory layers).

### Minimum distinguishing experiment

N=1 (endpoint at L16) vs N=5 (trajectory L10,L12,L14,L16,L18), one behavior,
matched delta_cos, 3 seeds. Measure PPL and per-layer off-shell. Cost ~1 hour.
If N=5 trajectory has lower PPL AND lower max per-layer off-shell, the mechanism
is supported. If PPL is equal or higher, N19 fails the minimum test.

### Verdict

TESTABLE-MEDIUM. The mechanism has a clean prediction (lower per-layer off-shell)
that is directly measurable. The minimum experiment (1 hour) is the essential first
step. GeoSteer replication (the manifold-gradient version) should be run as a parallel
track to provide a ceiling for what trajectory-aware steering can achieve.
