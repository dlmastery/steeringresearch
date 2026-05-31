# N15 — Coset-Min-Collateral Post-Processing

> **One-line claim:** Among all behaviorally-equivalent vectors (the Non-Identifiability
> coset), the representative with minimal projection onto fragile mid-layer subspaces
> has the least alignment damage at equal efficacy — a free convex post-processing
> step applicable to any steering vector.
>
> **Primary axes:** A10 (identifiability/gauge), A2 (direction/what)
> **Status:** UNTESTED

---

## 1. Motivation (>= 100 words)

The Non-Identifiability paper (arXiv:2602.06801) proved a remarkable and practically
important theorem: under single-layer activation steering, there exist large equivalence
classes of behaviorally-indistinguishable vectors. Any two vectors v and v' that differ
only in the null space of all downstream readouts produce identical behavioral outputs.
The null space is estimated to span 80-95% of d_model dimensions. This means the
standard DiffMean vector is just ONE representative of a huge equivalence class (a coset
of the null space). All coset members are behaviorally equivalent; they differ only in
their COLLATERAL EFFECTS — how much they disturb activations in downstream layers,
KV-caches, and capability-relevant subspaces. The key insight is that this is a FREE
optimization: we can choose any coset member and the behavior is unchanged. Choosing
the member with minimum projection onto "fragile" subspaces — directions that, when
perturbed, cause the most capability damage or the most rogue compliance — is a free
lunch. This post-processing step applies to ANY derived vector (DiffMean, SAE feature,
function vector, ICL-derived) and requires only a convex projection, costing O(d_model^2)
operations. The fragile subspaces are identified by the N20 curvature/fragility probe,
or alternatively by the known safe-to-ignore directions from the Non-ID paper's SVD
analysis. If this claim holds, it is one of the highest-leverage algorithmic contributions
in the program: a single-line code change that reduces alignment damage for every
steering method simultaneously.

## 2. Formal Hypothesis (>= 50 words)

Let v_raw be a DiffMean vector. Let N_null be the estimated null space of downstream
readouts (top-d_null eigenvectors of the activation covariance with zero behavioral
eigenvalue, estimated by behavioral testing). Let v_coset = v_raw - Proj_{N_null ∩ fragile}(v_raw)
be the min-collateral coset rep. The claim is:

  alignment_damage(v_coset) < 0.7 * alignment_damage(v_raw)

at equal behavioral efficacy (within 3%), where alignment_damage is measured as
the composite of MMLU capability drop and XSTest over-refusal rate, on Gemma-3-1B-it.

## 3. Falsifier (>= 30 words)

If alignment_damage(v_coset) >= 0.9 * alignment_damage(v_raw) at equal efficacy
(the min-collateral projection does not reduce damage by at least 10%), the
post-processing step provides no practical value and the claim is FALSIFIED.
Status moves to `FALSIFIED`.

## 4. Citations (Citation Rigor >= 80 words)

```
Venkatesh & Kurapath 2026. 'On the Non-Identifiability of Steering Vectors'
arXiv:2602.06801 (ICLR 2026 workshop). The primary theoretical foundation: proves
the coset structure, estimates the null space dimension as large (80-95% of d_model),
and demonstrates that orthogonal perturbations achieve near-equivalent behavioral
efficacy. N15 is the practical algorithm that exploits this theorem for collateral
reduction.

Schwinn et al. 2025. 'Rogue Scalpel' arXiv:2509.22067. Identifies "fragile mid-layer
subspaces" empirically: the directions most associated with rogue compliance in
adversarial feature probing. The fragile subspaces that N15 projects away from are
precisely those identified by the Rogue Scalpel analysis.

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. The manifold-tangent
subspace at h is the BEHAVIORALLY ACTIVE subspace; its complement (the normal
space) contains the null space. N15's projection is equivalent to projecting v_raw
onto the manifold tangent and then further reducing the fragile component.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681. CAA vectors are
the raw v_raw; N15 is a post-processing step that improves any CAA or DiffMean
vector without changing the extraction procedure.
```

## 5. Mechanism

The coset structure: for behavioral efficacy b(h + v), only the component of v
in the "effective subspace" E (orthogonal complement of the null space N_null) matters.
The component in N_null contributes zero behavioral effect. So v_raw = v_eff + v_null,
and all vectors v_raw + n for n in N_null are behaviorally equivalent.

Within this equivalence class, different coset representatives have different collateral.
The collateral is caused by the "wasted" component v_null: it pushes h into off-manifold
regions that may overlap with fragile subspaces. The min-collateral representative
v_coset removes the portion of v_null that projects onto fragile directions F:
v_coset = v_raw - Proj_{N_null ∩ F}(v_raw)

This is a convex projection because N_null ∩ F is a linear subspace (intersection
of two linear subspaces). Computation: identify the basis of N_null from SVD of the
activation covariance (top-d_null eigenvectors); identify fragile directions F from
the N20 fragility analysis or the Rogue Scalpel's harmful-feature directions;
compute the intersection basis; project v_raw off this intersection.

Null space estimation challenge: d_null ~ 0.80-0.95 * d_model (Non-ID estimate)
means N_null has dimension ~1600-1900 for d_model=2048. Estimating this from activation
covariance requires a large corpus (n >> d_model forward passes). Practical
approximation: use the top-80% variance eigenvectors as the "active" subspace and
project v_raw onto the complementary top-20% as an aggressive coset opt-out.
This is a conservative approximation that removes more than just N_null.

## 6. Predicted Delta

| Metric | Predicted Delta (coset rep vs raw) | Rationale |
|---|---|---|
| MMLU capability drop | -30% to -50% relative | Fragile subspaces cause most capability damage |
| XSTest over-refusal | -20% to -40% relative | Fragile directions drive benign misclassification |
| Behavioral efficacy | 0% difference (within 3%) | Coset equivalence theorem |
| ||v_coset|| vs ||v_raw|| | -10% to -30% | Fragile component removed |
| Computation overhead | < 2% of steering cost | Projection after vector extraction |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16
- Null space estimation: SVD of activation covariance from 2000 forward passes
  on diverse prompts; identify top-20% variance eigenvectors (active subspace)
- Fragile directions F: from N20's curvature-fragility analysis, or the max-Fisher
  direction L12 identified in C4 (most fragile layer)
- v_raw: DiffMean vector for 3 behaviors (refusal, politeness, factuality)
- v_coset: project v_raw off N_null ∩ F
- Behavioral efficacy check: confirm behavioral cosine shift v_coset within 3% of v_raw
- Collateral measurement: MMLU (5-shot, a 100-question subset), XSTest over-refusal (50 prompts)
- Comparison: v_raw, v_coset, control (random coset rep of same ||v_coset||)
- Seeds: 3 covariance estimation seeds x 3 evaluation seeds
- Wall-clock: ~4 hours on RTX 4090

### 7.2 Where it shines

High-stakes deployments where alignment damage is the primary concern (not just
behavior efficacy). Every existing steering system can benefit from v_coset post-
processing at negligible cost; it is a universal upgrade to any extraction method.

## 8. Cross-References

- N10 (Non-ID paper): the theoretical basis; N15 is the practical algorithm
- N3 (orthogonal capacity): the effective subspace dimension is PR; N15 projects
  within this same effective subspace
- N8 (controllability != interpretability): the null space is the inert subspace;
  N15 explicitly removes it
- N20 (curvature as fragility sensor): fragile directions F are identified by N20;
  N15 uses N20's output as the projection target
- N6 (separation principle): N6 projects b against c; N15 projects v against
  fragile subspaces — both are coset-selection operations at different levels
- N12 (capstone): N15's projection is part of the Proj_T term in the unified operator
- IDEA_TABLE.md: N15 row, axes A10+A2

## 9. Committee Q&A

**Q: The null space estimation requires SVD of a d_model x d_model covariance matrix.
For d_model=2048, this is a 4M-element matrix from 2000 forward passes. Is this
computationally and memory feasible?**

> 2000 forward passes x 2048-dimensional activations = a 2000 x 2048 matrix. The
> sample covariance is 2048 x 2048 (~16 MB), well within memory. The SVD is O(d^3) =
> O(8G) operations — about 8 seconds on a modern GPU. The practical approximation
> (top-20% variance eigenvectors) reduces this to a partial SVD requiring the top
> 410 eigenvalues, which is O(410 * 2048^2) ~ 2G operations: ~2 seconds. Feasible.

**Q: How do you identify "fragile" directions without running expensive behavioral
probing for every direction?**

> Two methods: (i) use N20's curvature-fragility correlation — high-curvature
> directions as identified by effective-rank analysis are fragile; (ii) use the
> Rogue Scalpel's published fragile-direction analysis (harmful SAE features) as a
> pre-identified fragile set. Method (ii) is faster and uses existing knowledge;
> method (i) is more principled but requires N20 to be run first.

**Q: The "free lunch" framing implies no cost. But the null space estimation and
projection DO have a one-time cost (2000 forward passes). Is this truly "free"?**

> "Free" relative to the extraction cost of the behavior vector (50-100 contrast-pair
> forward passes). The null space estimation (2000 passes, ~2 hours) is a one-time
> fixed cost per model and layer, not per behavior. Once estimated, the projection
> for ANY behavior vector costs O(d^2) = O(4M) operations — truly negligible.

## 10. Verification Checklist

- [ ] Activation covariance computed from 2000 forward passes; SVD verified
- [ ] Active subspace (top-20% variance): basis vectors saved and dimensionality logged
- [ ] Fragile directions F identified (N20 curvature or Rogue Scalpel published list)
- [ ] v_coset projection implemented; ||v_coset|| and cos(v_coset, v_raw) reported
- [ ] Behavioral equivalence: v_coset efficacy within 3% of v_raw (pre-check)
- [ ] MMLU 100-question subset measured for v_raw and v_coset
- [ ] XSTest 50 prompts over-refusal measured for v_raw and v_coset
- [ ] Random coset rep control measured (same ||v_coset||, different direction)
- [ ] 3 x 3 seed design, EXPERIMENT_LEDGER.md entry
- [ ] IDEA_TABLE.md N15 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. The Non-ID paper provides the
  theoretical guarantee of coset equivalence. The practical challenge is identifying
  fragile directions without running a full behavioral probing sweep. N20 (fragility
  sensor) is the prerequisite for the efficient version of N15; if N20 is also UNTESTED,
  N15 can proceed with the Rogue Scalpel published fragile-feature set as a proxy.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM-HIGH. The Non-ID theorem is a rigorous mathematical result; if it applies to
our model family (Gemma-3-1B), the coset structure guarantees the existence of a
better coset representative. The practical question is whether the fragile subspace
is well-characterized and whether the improvement is measurable against the noise
in MMLU and XSTest evaluations.

### Mechanism scrutiny

The "fragile" direction classification is circular if fragile directions are defined
as "directions where projection causes capability damage" — that's what we're trying
to reduce. The operational definition must use a BEHAVIOR-FREE criterion (curvature,
effective-rank, or Rogue Scalpel analysis) to avoid circular reasoning.

### Confounds

1. The null space dimension estimate from SVD may be wrong; if the "null space"
   is smaller than estimated, the projection removes some behaviorally active
   components and reduces efficacy.
2. MMLU is a coarse capability measure (n=100 questions); the expected delta is small
   (maybe 0.5-2 pp) and may be within MMLU's measurement noise.

### Does the coset-optimization claim specifically matter?

YES if the damage reduction is 30-50% relative (claimed). Even a 15% relative
reduction in capability damage is practically meaningful at negligible cost.
The free-lunch framing is accurate for the per-behavior cost; the one-time SVD
cost is real but amortized.

### Skeptical effect-size estimate

MMLU reduction: 10-25% relative (vs claimed 30-50%). The fragile subspace may be
small relative to the null space, limiting the projection's impact.
XSTest over-refusal reduction: 15-30% relative (vs claimed 20-40%).
These are still practically useful ranges.

### Minimum distinguishing experiment

Compute v_coset for one behavior; verify behavioral equivalence (3% tolerance on
efficacy); measure MMLU before/after for v_raw vs v_coset at 3 seeds. Cost ~1 hour.
If MMLU drop is 0.5+ pp less for v_coset (relative to v_raw), the signal is present;
proceed to full protocol. If MMLU noise is larger than the expected difference,
the evaluation instrument is insufficient and a more sensitive capability measure
(generation-based benchmark) is needed.

### Verdict

TESTABLE-HIGH-VALUE. The Non-ID theorem provides strong theoretical backing.
The practical challenge is evaluation sensitivity (MMLU noise) and fragile-direction
identification. Recommend: (1) use Rogue Scalpel published fragile-feature list
as F (no new analysis needed), (2) run minimum experiment with one behavior, (3)
commit to full protocol if MMLU difference signal is detectable at 3 seeds.
