# N11 — Curvature-Aware Conditioning Predicts the Coherence Cliff

> **One-line claim:** A per-prompt curvature estimate (local PCA spectrum decay)
> predicts that prompt's cliff alpha, enabling per-prompt adaptive alpha with lower
> cross-prompt variance than E7's norm-relative scheme.
>
> **Primary axes:** A3 (how-much/coefficient), A8 (geometry/curvature)
> **Status:** UNTESTED

---

## 1. Motivation (>= 100 words)

The coherence cliff — the sharp PPL increase above a threshold alpha (observed in
S-2 through S-8) — is not at the same alpha for every prompt. Some prompts tolerate
strong steering; others degrade at small alpha. The current best approach (E7,
supported in S-9) is norm-relative alpha: alpha_rel = alpha * ||h|| normalizes by
the activation norm, reducing cross-prompt variance. But norm-relative alpha treats
all prompts with the same ||h|| equally, even though they may have very different
local manifold structures. Two prompts with the same ||h|| but different local
curvature — one near a flat region of the manifold, one near a highly curved region —
will have very different cliff alphas. A prompt in a flat manifold region can tolerate
large steering without going off-manifold; a prompt in a highly curved region leaves
the manifold at much smaller alpha. This predicts that local curvature (not just
||h||) should be the normalization factor. The local curvature can be estimated
cheaply from the PCA spectrum of the k nearest neighbors of h in activation space:
rapid spectrum decay (few principal components explain most variance) indicates
a locally flat, low-dimensional neighborhood (high curvature tolerance); slow decay
(many components needed) indicates a locally flat high-dimensional neighborhood
(even higher tolerance) or a curved neighborhood. The specific signature of high
curvature is the ratio lambda_2 / lambda_1 of the top two eigenvalues: when this
ratio is close to 1, the neighborhood is isotropic (many equivalent directions,
less structured); when close to 0, the neighborhood is strongly directional (one
principal direction, which often aligns with the behavior direction).

## 2. Formal Hypothesis (>= 50 words)

Let kappa(h) = 1 / (||h|| * sum_i lambda_i^2 / (sum_i lambda_i)^2) = ||h||^{-1} / PR(h)
be the per-prompt curvature estimate, where PR(h) is the participation ratio of the
kNN neighborhood of h (k=50 nearest natural activations). The claim is:

  Spearman(kappa(h), alpha_cliff(h)) >= 0.50

where alpha_cliff(h) is the per-prompt cliff alpha (measured as the smallest alpha
at which log-PPL exceeds 1.5 * baseline), across 30 diverse prompts on Gemma-3-1B-it
@L16. Furthermore, alpha_curvature-adaptive = C / kappa(h) should reduce the
cross-prompt PPL variance by >= 20% vs alpha_norm-relative = C / ||h||.

## 3. Falsifier (>= 30 words)

If Spearman(kappa(h), alpha_cliff(h)) < 0.30 across 30 prompts, the local curvature
claim is FALSIFIED. If the variance reduction of curvature-adaptive vs norm-relative
alpha is < 10%, the practical benefit fails even if the correlation exists. Status
moves to `FALSIFIED` or `INCONCLUSIVE` accordingly.

## 4. Citations (Citation Rigor >= 80 words)

```
Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. The manifold curvature
governs how quickly the activation leaves M_h under linear steering; the cliff alpha
is exactly the point where curvature-induced off-manifold displacement reaches a
threshold. N11 operationalizes this as a per-prompt measurable quantity (kNN PCA spectrum).

Gao et al. 2026. 'CRH' arXiv:2605.01844 (ICML 2026). CRH's effective-rank analysis
measures the dimensionality of the activation neighborhood; the participation ratio
PR(h) used in N11's curvature estimate is directly the CRH effective-rank at the
local neighborhood level (per-prompt rather than per-layer-population).

Raval et al. 2026. 'Curveball Steering' arXiv:2603.09313. Curveball identifies that
curved paths outperform linear at high curvature; N11 provides a cheap curvature
measurement that could trigger Curveball routing without expensive full-manifold fitting.

Venkatesh & Kurapath 2026. 'Non-Identifiability' arXiv:2602.06801. The equivalence
class of vectors achieving a given behavior is larger in low-curvature regions
(more null space available); kappa(h) captures this indirectly as regions with
high participation ratio have more near-neutral directions.
```

## 5. Mechanism

Local manifold curvature at h is estimated by the PCA of the k=50 nearest natural
activations {h_1, ..., h_k} to h in activation space (L2 distance). The participation
ratio PR(h) = (sum lambda_i)^2 / (sum lambda_i^2) of the kNN covariance measures
the effective dimensionality of the local neighborhood.

Curvature interpretation: a small PR(h) indicates a locally low-dimensional
neighborhood — the k nearest points lie on a thin sub-manifold (high local curvature,
steep direction change). A large PR(h) indicates the neighborhood is spread across
many dimensions (lower curvature, flatter local geometry). The cliff alpha scales
with the radius of curvature: kappa^{-1} = radius_of_curvature. Steering by
alpha * v causes off-manifold displacement proportional to alpha^2 * kappa, so the
cliff occurs at alpha_cliff ~ kappa^{-1/2} * ||v||^{-1} * (displacement_threshold)^{1/2}.

Per-prompt adaptive alpha: alpha_adaptive(h) = C / sqrt(kappa(h)) where C is a
universal constant calibrated on a validation set. This is more expensive than
norm-relative (which uses 1/||h||) but requires only the kNN computation (50
distance evaluations in d_model dimensions) plus the PCA of the 50x50 covariance.
For d_model=2048, this is O(50 * 2048 + 50^2) = ~100,000 operations — negligible.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| Spearman(kappa, alpha_cliff) | 0.50 - 0.70 | Curvature is the geometric driver of cliff |
| Variance reduction vs norm-relative | 20% - 40% relative | Local curvature adds information beyond ||h|| |
| Additional compute | < 0.5% of forward pass | kNN + PCA at 50 points is cheap |
| alpha_cliff correlation with ||h|| alone | 0.20 - 0.40 | E7's norm-relative works but not optimally |
| Improvement on diverse prompt types | Most: technical > common prompts | Technical prompts tend to be higher curvature |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16
- Corpus: 30 diverse prompts (10 technical, 10 common-knowledge, 10 abstract/creative)
- kNN cache: extract activations for 5000 natural prompts from WikiText; build index
- Per-prompt kappa: for each of 30 prompts, find k=50 nearest neighbors, compute
  kNN PCA covariance, compute PR(h), compute kappa = 1 / (||h|| * PR(h))
- Per-prompt cliff alpha: for each prompt, run alpha sweep {0.5, 1, 2, 4, 8},
  measure log-PPL; find smallest alpha where PPL > 1.5 * baseline
- Spearman(kappa, alpha_cliff): computed over 30 prompts
- Adaptive alpha: C calibrated on 10 validation prompts; tested on 20 held-out
- Variance comparison: cross-prompt PPL variance at adaptive-alpha vs norm-relative
  at matched target PPL level
- Seeds: 3 kNN extraction seeds (vary the 5000-prompt corpus)
- Wall-clock: ~4 hours on RTX 4090

### 7.2 Where it shines

High-diversity prompt distributions (technical + casual + creative mixed) where
the cliff alpha varies most across prompt types. If E7's norm-relative is already
optimal for uniform prompt distributions, N11's additional gain comes from the
cross-type variance reduction.

## 8. Cross-References

- E7 (norm-relative alpha): N11 extends E7 by replacing ||h|| normalization with
  kappa-based normalization; the two are compared directly
- N3 (orthogonal capacity): local PR (per-prompt) is related to the global PR
  (per-layer) of N3; N11 predicts that per-prompt PR is more predictive than per-layer PR
- N5 (norm-budget): the cliff alpha predicted by N11 is the "budget B" of N5 on a
  per-prompt basis; N11 gives B = f(kappa), N5 gives B = q-quantile(||h||); N11 is tighter
- N2 (curvature = gating): N2's g(h) field should be high when kappa is high (more
  curvature = more need for adaptive gating); N11 provides the curvature input to N2
- N20 (curvature as fragility sensor): N20 is the per-layer version; N11 is the
  per-prompt version at a fixed layer
- IDEA_TABLE.md: N11 row, axes A3+A8

## 9. Committee Q&A

**Q: The kNN-based curvature estimate uses natural activations from WikiText.
If the prompt distribution is very different from WikiText, the kNN set may be
empty or misrepresentative. How is this handled?**

> If fewer than 20 of the 50 nearest neighbors are within a distance threshold
> (2.0 * mean pairwise distance in the corpus), the kappa estimate is flagged
> as unreliable and the experiment uses norm-relative alpha for that prompt.
> This is expected for fewer than 5% of prompts for technical domains; for
> highly OOD prompts, N11 degrades gracefully to E7.

**Q: The cliff alpha measurement requires running an alpha sweep for each of 30
prompts — 30 x 5 alpha values = 150 forward passes. Is this feasible?**

> Yes, 150 forward passes for 512-token generations on a 1B model takes ~2 hours.
> The kNN computation adds ~30 min. The total ~2.5 hours is within the session budget.

**Q: The participation ratio measures dimensionality, not curvature directly.
Why is 1/PR a curvature proxy?**

> PR = k (effective dimensions) means the local neighborhood spans k directions
> approximately equally. High k (high PR) means the neighborhood is locally flat
> (the manifold extends equally in many directions — low curvature). Low k (low PR)
> means the neighborhood is concentrated in 1-2 directions — the manifold has "bent"
> sharply. The reciprocal 1/PR is thus a curvature proxy. It is not the exact
> sectional curvature (which requires second-order geometry estimation), but it
> is a cheap first-order proxy.

## 10. Verification Checklist

- [ ] kNN search index built on 5000 WikiText activations @L16 (FAISS or BruteForce)
- [ ] PR(h) computation validated on synthetic data (uniform k-dim = PR close to k)
- [ ] kappa computation matches expected behavior on flat vs curved test cases
- [ ] 30-prompt cliff-alpha sweep: 150 forward passes logged in EXPERIMENT_LEDGER.md
- [ ] Spearman(kappa, alpha_cliff) reported with p-value and 90% CI
- [ ] Comparison: kappa-adaptive vs norm-relative vs fixed alpha, variance and PPL
- [ ] C calibration reported with validation set prompts documented
- [ ] IDEA_TABLE.md N11 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. E7 (norm-relative alpha) is
  supported (S-9), establishing that normalization matters. N11 predicts that local
  curvature normalization is better still. The kNN-PCA curvature estimation is
  computationally feasible but requires infrastructure (kNN index, PCA pipeline)
  that does not yet exist in the project.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. The connection between local manifold curvature and alpha tolerance is
geometrically sound. The practical question is whether the 50-point kNN PCA is
an accurate curvature estimate and whether curvature adds predictive power
beyond ||h||. If activations are quasi-isotropic in the local kNN neighborhood
(PR ≈ constant across prompts at the same layer), then kappa(h) ≈ constant * ||h||^{-1}
and N11 reduces to E7's norm-relative scheme with no additional gain.

### Mechanism scrutiny

The claim that "rapid PCA spectrum decay = high curvature tolerance" is the
OPPOSITE of the standard geometric interpretation: a high-curvature manifold
has a concentrated spectrum (the manifold bends strongly in one direction) but
this is a large-scale property. At the local kNN scale (50 nearest points), a
concentrated spectrum may indicate an INTRINSICALLY LOW-DIMENSIONAL neighborhood,
which has LESS local curvature (the neighborhood is flat in a low-dimensional
subspace). This potential inversion of the curvature interpretation is the key
mechanism concern and requires careful validation against the cliff alpha measurement.

### Confounds

1. The kNN cache from WikiText may not represent the prompt distribution for
   harmful/safety-relevant prompts (which have distinctive activation statistics).
2. The cliff alpha per prompt may be dominated by the prompt's base PPL
   (different prompts have different baseline perplexities): high-base-PPL prompts
   may appear to cliff earlier simply because PPL * 1.5 is reached at a smaller
   alpha increment. Control: measure cliff alpha relative to base PPL.
3. The 30-prompt sample may be too small to achieve Spearman = 0.50 with p < 0.05
   (requires ~25 samples for significance at 0.50 Spearman). At 30 samples,
   the test is borderline powered.

### Does the curvature estimate specifically matter?

UNCERTAIN. The key test is whether kappa(h) adds R2 above ||h|| in predicting
alpha_cliff. If R2 increase < 0.05, the curvature claim is not practically useful.
Recommend reporting both univariate (Spearman of kappa alone, ||h|| alone) and
multivariate (partial correlation of kappa controlling for ||h||) statistics.

### Literature precedent

Riemannian manifold learning (Isomap, LLE) uses local neighborhood structures
to estimate manifold curvature; the kNN-PCA approach is a simplified version.
No prior work applies this specifically to per-prompt cliff alpha prediction
in LLM steering.

### Skeptical effect-size estimate

Spearman(kappa, alpha_cliff): 0.25-0.45 (vs claimed 0.50-0.70). Variance reduction
vs norm-relative: 5-15% (vs claimed 20-40%). The main risk is that alpha_cliff
is primarily driven by ||h|| (which E7 already captures) and kappa adds only
marginal predictive power.

### Minimum distinguishing experiment

5 prompts (deliberately chosen to vary in curvature proxy: 2 technical, 2 casual,
1 creative): compute kappa and alpha_cliff for each; report whether rank order
matches. Cost ~30 min. If kappa rank and cliff rank match perfectly on 5 prompts,
proceed to full 30-prompt protocol.

### Verdict

TESTABLE-LOW-MEDIUM confidence. The curvature interpretation has a potential sign
inversion that must be resolved theoretically before the experiment is run. The
minimum experiment (5 prompts, 30 min) is the essential validation step. Recommend
against the full 4-hour protocol until the sign question is resolved and the
minimum experiment confirms positive correlation.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N11.md`.
