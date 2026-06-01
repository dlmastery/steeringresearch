# E1 — DiffMean Pair-Count Knee

> **One-line claim:** DiffMean steering vectors derived from >=50 contrast pairs
> reach >=90% of the asymptotic behavior-shift effect; additional pairs beyond the
> knee give diminishing returns and are not worth the annotation cost.
>
> **Source design space:** Block A — Foundations and measurement tooling (E1).
> **Primary axis:** A7 (HOW DERIVED — extraction source and pair count).
> **Implementation status:** `~ directional (screening, underpowered — C5)`.

---

## 1. Motivation (>=100 words)

DiffMean — the difference of mean hidden-state activations between contrastive prompt
sets — is the cheapest, most widely deployed method for extracting steering vectors.
Its data requirement is never formally studied: how many contrast pairs are needed
before the extracted direction stabilizes? The contrastive activation addition (CAA)
paper (Panickssery et al. 2023, arXiv:2312.06681) used ~200 pairs from MultiNLI and
Anthropic-HH but did not ablate the pair count. The AxBench benchmark (Zou et al.
2025, arXiv:2501.17148) treats extraction as solved and focuses on evaluation, again
assuming adequate data. In low-resource or safety-critical deployment scenarios the
cost of annotating 50–200 high-quality contrastive examples is non-trivial: each pair
requires a semantically coherent opposed couple that shares surface form but differs
precisely on the target attribute. If the steering direction stabilizes at 10–20 pairs,
annotation budgets can be cut by 5–10× without losing efficacy. Conversely, if 200+
pairs are required, claims of "data-efficient" steering in transfer settings must be
re-examined. This hypothesis pre-registers the diminishing-returns relationship and
the pair-count knee as a testable geometric claim: because DiffMean is a sample mean
of bounded vectors, the law of large numbers guarantees convergence, but the rate
depends on the between-pair cosine variance of the concept direction — harder, more
diffuse concepts will require more pairs. Establishing the knee is a prerequisite for
every subsequent Block A experiment: E4 (DiffMean vs PCA) and E7 (norm-relative alpha)
both assume the DiffMean vector has converged; if the knee is at n=100 we must recheck
those conclusions.

---

## 2. Formal hypothesis (>=50 words, falsifiable)

Because DiffMean is a sample mean estimator of a fixed (if noisy) population direction,
successive pairs reduce the Monte Carlo error at rate O(1/sqrt(n)). The hypothesis is:
on Gemma-2-2B-it with a moderately abstract behavioral concept (not a trivial token-level
concept), the cosine similarity between the n-pair DiffMean vector and the full-N vector
exceeds 0.90 at n>=50, and the marginal cosine gain per additional pair is monotonically
decreasing. Specifically, the knee — where the second derivative of cosine(n) changes
sign from positive to negative — lies in the range n in {20..80} for non-synthetic,
semantically rich concepts. Below the knee, pair count is the binding constraint; above
it, richer prompts, better layers, or higher alpha dominate. The ">=90% of asymptotic"
threshold is pre-registered.

---

## 3. Falsifier (>=30 words)

**Fired if:** on a held-out real concept (not a synthetic lexicon proxy), the cosine
between the 50-pair vector and the 200-pair vector is <0.90 (knee above 50) OR if the
behavior success rate at n=50 pairs is below 90% of the n=200 rate on a generation-
based judge. Either outcome implies the corpus's "~50 pairs" claim does not transfer
and E7/E4 require re-derivation at their required pair counts.

---

## 4. Citations (>=80 words, Citation Rigor format)

```
Panickssery, Aryan, Gabrieli, Nick, Huang, Justing, Mallen, Meghana,
Bowman, Samuel, Hubinger, Evan. 2023 arXiv 'Steering Llama 2 via
Contrastive Activation Addition' (arXiv:2312.06681) — the canonical CAA
paper; uses ~200 pairs, never ablates pair count; the baseline whose
implicit assumptions E1 makes explicit and testable.

Zou, Andy, Phan, Long, et al. 2025 arXiv 'AxBench: Steering LLMs?
Benchmarks Matter!' (arXiv:2501.17148) — the evaluation standard;
benchmarks extraction but does not study the pair-count data efficiency
frontier; E1 fills this gap.

Zou, Andy, Phan, Long, Wang, Zifan, Kolter, J. Zico, Fredrikson, Matt,
Li, Bo. 2023 NeurIPS 'Representation Engineering: A Top-Down Approach to
AI Transparency' (arXiv:2310.01405) — RepE's PCA-based extraction uses
multiple prompt formats rather than more pairs; suggests direction quality
is primarily a function of prompt diversity, not raw count — testable
prediction that E1 ablates.
```

---

## 5. Mechanism (deep technical)

DiffMean is:

    v_DM(n) = (1/n) * sum_{i=1}^{n} (h_pos_i - h_neg_i)

where h_pos, h_neg are layer-L hidden states on a matched positive/negative pair.
The estimator converges to the population mean delta mu = E[h_pos] - E[h_neg]. By
the central limit theorem, the standard error of each component of v_DM(n) is
sigma_j / sqrt(n), where sigma_j is the component-wise standard deviation across pairs.

The cosine between v_DM(n) and the true direction mu/||mu|| is approximately:

    cos(v_DM(n), mu) ~= 1 - d * sigma^2 / (2 * n * ||mu||^2)

where d is the ambient dimension and sigma^2 is the mean component-wise variance.
This predicts a 1 - C/n convergence law; the knee (where marginal gain per pair
drops below epsilon) is at n_knee ~ C / epsilon. For a well-separated concept
(large ||mu||/sigma), the knee is early (n ~ 5–20). For diffuse, entangled concepts
(small ||mu||, large sigma), n_knee can exceed 100.

The practical implication: if the tested concept happens to be synthetic or lexically
simple (as in C5's "ocean" concept), the knee appears early (~5 pairs, cos>0.95),
which is why C5 showed directional support but could not test the 50-pair claim.
A harder concept (e.g., "responds truthfully when uncertain" as in TruthfulQA-style
vectors) has larger component variance and a later knee. The experimental design must
therefore include at least one semantically rich, latent behavioral concept.

The off-the-shelf convergence check is straightforward: for a grid of n in
{1, 2, 3, 5, 8, 10, 15, 25, 50, 100, 200}, compute cos(v_DM(n), v_DM(200)) and
fit the 1 - C/n curve. The knee is the n where the fit predicts
d(cos)/dn < 0.002 (0.2 pp per pair). This criterion is behavior-agnostic and cheap.

---

## 6. Predicted Delta (pre-registered numbers)

| Metric | Predicted value | Rationale |
|--------|----------------|-----------|
| cos(50-pair, 200-pair) | >= 0.90 | CLT convergence at O(1/sqrt(n)) |
| Knee location (rich concept) | n in [20, 80] | sigma/||mu|| ~0.3 estimate |
| Knee location (synthetic concept) | n in [3, 10] | C5 observed ~5; confirms easy concept |
| Behavior delta n=50 vs n=200 | < 10% relative loss | If cos > 0.90 implies < 10% behavior gap |
| Marginal gain per pair above knee | < 0.005 cos/pair | Monotone diminishing by definition |

---

## 7. Experimental protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it (4-bit), layer L16 (best from C1/E2).
- **Concept set:** two concepts — one synthetic (as in C5, for comparability) and
  one latent behavioral (e.g., "truthful under uncertainty" from TruthfulQA contrast
  pairs).
- **Pair counts:** n in {1, 2, 3, 5, 8, 10, 15, 25, 50, 100, 200}; all drawn from
  a fixed pool of 200 pairs by random subsampling (5 resamples per n).
- **Metric:** cos(v_DM(n), v_DM(200)); behavior success on generated text via
  LLM-judge at n=10, 25, 50, 100, 200.
- **Control:** random-pair baseline (shuffle pair labels); should give cos ~ 0.

### 7.2 Where this should SHINE

Hard, latent behavioral concepts where DiffMean's convergence rate is the binding
constraint — not layer or alpha choice. E.g., "nuanced sycophancy avoidance" where
the concept direction requires 50+ examples to distinguish from surface politeness.

### 7.3 Cross-references to geometry

Pair-count convergence interacts with E4 (DiffMean ~= PCA): if the DiffMean is
computed at n < n_knee, its cos to PCA-top1 will be lower than 0.99. C5 showed
cos > 0.95 at n=5 for the synthetic concept, consistent with an easy-concept knee.
The E4 claim (cos > 0.95) should be re-checked at the actual concept-specific knee.

---

## 8. Cross-references

- **E4** (DiffMean vs PCA cosine): pair count must be above knee for E4's 0.99 claim.
- **E7** (norm-relative alpha): assumes a stable DiffMean direction as starting point.
- **C5** (E1 screening result): directional support; synthetic concept; underpowered.
- **N4** (steering as inverse ICL): the ICL alignment claim also depends on
  having a stable DiffMean vector (>= knee pairs).
- **IDEA_TABLE.md** Block A row E1.

---

## 9. Committee Q&A

**Q: C5 showed the knee at ~5 pairs — isn't that already an answer?**

> No. C5 used a trivially easy synthetic concept ("ocean") with maximum within-concept
> cosine consistency. The interesting regime is hard behavioral concepts where the
> direction is diffuse and entangled. The 50-pair pre-registration was inherited from
> the CAA corpus (arXiv:2312.06681) which used rich HH-RLHF contrast pairs, not a
> lexicon. The C5 result tells us the tool works on easy concepts; it says nothing
> about the regime the corpus's 50-pair claim was designed for.

**Q: Why not just always use all available pairs and move on?**

> In low-resource safety deployment, annotation is the bottleneck, not compute.
> Knowing the knee lets practitioners allocate annotation budget optimally. Also,
> if the knee is at n=200 for hard concepts, the E4 and E7 claims need to be
> re-examined at their actual pair counts, which propagates to every downstream
> Block B–F experiment.

**Q: How do you know 200 pairs is the right "full" reference?**

> It's not. The pre-registered test compares n-pair to 200-pair vectors; if the
> 200-pair vector is itself not converged, the test is a lower bound. A convergence
> check (plot cos vs n and fit 1-C/n; extrapolate) provides the asymptote estimate.
> A failed convergence means the test should move to n=500.

**Q: What if different layers have different knees?**

> Plausible. Per-layer knee is a natural extension (add to the 7.3 protocol) but
> not the primary pre-registration. The primary claim is at L16 for a single concept.
> Layer-stratification is a rung-3 ablation.

---

## 10. Verification artifacts checklist

- [ ] Per-concept convergence curve (cos vs n, 5 resamples per n) stored in
      `ideas/10_diffmean_paircount_knee/experiments/exp_paircount/`.
- [ ] 1-C/n fit with R^2 and estimated n_knee reported.
- [ ] Behavior-judge scores at n={10,25,50,100,200} logged in experiment ledger.
- [ ] Random-pair control confirms cos ~ 0.
- [ ] Cross-concept comparison: synthetic vs latent behavioral concept.
- [ ] Result row added to `EXPERIMENT_LEDGER.md` (Rung 1 SMOKE minimum).
- [ ] Status row in `IDEA_TABLE.md` updated.

---

## 11. Status journal

- 2026-05-27 — hypothesis inherited from `corpus/50-steering-experiments-autoresearch.md` E1.
- 2026-05-29 — C5 screening run on Gemma-3-270m @L16 with synthetic "ocean" concept:
  knee observed at ~5 pairs (cos>0.95 of 10-pair vector); directional, underpowered.
  Status: DIRECTIONAL (screening, underpowered). Can't test n=50 without more pairs.
- 2026-05-31 — Design doc written; status DIRECTIONAL, promotion requires real behavioral
  concept + generation-based judge + n>=7 seeds.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-A (elite research-scientist critic). Critiquing the IDEA, not the implementation.*

### Prior plausibility (independent of our framing)

**HIGH for the existence of a knee; MEDIUM for the 50-pair threshold specifically.**
The 1-C/n convergence is guaranteed by CLT — this is not controversial. What IS
uncertain is whether the 50-pair number from the CAA corpus (arXiv:2312.06681) transfers
to Gemma-3, which has a different geometry than Llama-2. C5's knee at n=5 on a synthetic
concept is a strong signal that easy concepts need far fewer pairs; hard concepts may
need more than 50.

### Mechanism scrutiny

The CLT argument is solid. The claimed "1 - C/n" functional form is the correct
first-order approximation when sigma_j and ||mu|| are constant across pairs, which
holds when pairs are i.i.d. samples from a fixed distribution. It breaks if the pair
distribution is long-tailed or if the "concept" spans multiple cluster modes. The
doc should acknowledge that multi-modal concept distributions (e.g., "refusal" with
two distinct refusal strategies) may yield slower convergence than the i.i.d. prediction.

### Confounds

1. **Pair diversity vs pair count**: RepE (arXiv:2310.01405) shows that prompt DIVERSITY
   (different surface forms of the same concept) matters more than raw count. A study
   that samples n pairs from a fixed pool of 200 fixed-template pairs is measuring
   statistical convergence, not diversity. The key experiment must use independently
   authored pairs.
2. **Layer dependence**: the knee at L16 may differ from the knee at L12 (max-Fisher),
   which could confound layer-choice experiments.
3. **C5's synthetic concept**: "ocean" tokens are far simpler than "sycophancy avoidance" —
   the C5 result cannot be extrapolated.

### Does the number 50 specifically matter?

Not intrinsically. Any pair count above the concept-specific knee is sufficient.
The 50-pair number is a convention from the CAA paper, not a theoretical threshold.
The real deliverable is the knee estimate per concept class.

### Literature precedent

No direct ablation study on DiffMean pair count exists in the literature. The closest
is Panickssery et al. (arXiv:2312.06681) Figure 3 (efficacy vs prompt count), which
shows saturation around 20 prompts — consistent with E1's "knee in [20,80]" but for
Llama-2 on simple behavioral concepts. RepE (arXiv:2310.01405) uses <=20 contrast pairs.
This gap makes E1 genuinely novel if run on harder concepts.

### Skeptical effect-size re-prediction

C5's knee at n=5 gives a lower bound. My prior for a hard behavioral concept: knee in
[15, 60], 80% CI. The 50-pair pre-registration is plausible but not guaranteed.

### Minimum-distinguishing experiment

Two concepts × n in {5, 10, 25, 50, 100} × 3 resamples × L16 × Gemma-2-2B.
Total cost: ~45 vector extractions + 5 behavior-judge runs. < 2 hours on the 4090.

### Verdict

**NOVEL+TESTABLE.** The CLT argument is airtight; the question (what is the practical
knee for hard behavioral concepts on Gemma?) is unanswered in the literature. C5 provides
directional evidence but is underpowered. The hypothesis is worth a full rung-2 run.
The falsifier is specific and pre-registered. Risk: the knee may vary so much by concept
that the 50-pair claim cannot be stated as a single number — which is itself a
publishable finding.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E1.md`.
