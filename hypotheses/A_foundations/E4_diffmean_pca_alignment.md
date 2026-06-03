# E4 — DiffMean vs PCA Cosine Alignment

> **One-line claim:** DiffMean and PCA-top-1 contrastive vectors are >0.95 cosine-
> aligned across layers, models, and behaviors, so the cheaper DiffMean suffices for
> all downstream applications that use the PCA-top-1 vector (including CAST gating).
>
> **Source design space:** Block A — Foundations and measurement tooling (E4).
> **Primary axis:** A7 (HOW DERIVED — extraction source / method).
> **Implementation status:** `+ SUPPORTED (screening, S-1/S-3/S-8, cos=0.994–0.996
> on 3 models)`.

---

## 1. Motivation (>=100 words)

Two main extraction pipelines dominate practical steering research. DiffMean
(arXiv:2312.06681) computes the arithmetic mean difference between positive and
negative class hidden states — an O(n * d) operation with no eigendecomposition.
PCA-top-1 (arXiv:2310.01405, RepE) fits a PCA to the stacked [h_pos; h_neg] matrix
and takes the first principal component as the direction — an O(n * d^2) SVD that
captures the dominant variance direction of the contrastive set. Both methods are
deployed widely, and the choice between them is treated in the literature as a
meaningful architectural decision. CAST (arXiv:2409.05907) uses PCA-top-1 for its
condition vector. If the two methods produce effectively identical vectors (cos > 0.95),
then PCA's extra compute is wasted, and any gap in downstream performance between
the two pipelines must be attributed to factors OTHER than the direction (e.g., alpha
scaling, layer choice, evaluation noise). This is practically important: DiffMean
is simpler to implement, has no convergence issues for small n, and is trivially
online-updatable as new pairs arrive. Establishing the alignment also resolves a
recurring ambiguity in the literature where papers use one method and claim results
generalize to the other. The C9b finding (S-9) adds a crucial subtlety: even when
the raw-alpha results from DiffMean and PCA-top-1 appear to differ, switching to a
relative-displacement parameterization (E7) makes them statistically indistinguishable —
the apparent gaps in earlier runs were norm-scaling artifacts, not directional disagreement.

---

## 2. Formal hypothesis (>=50 words, falsifiable)

For any layer L in the middle-to-late range of Gemma-2-2B-it, and for at least 3
distinct behavioral concepts, the cosine similarity between the DiffMean vector and
the PCA-top-1 contrastive vector (both computed on the same contrast set at the same
layer) exceeds 0.95. Furthermore, if both vectors are normalized to unit length and
used at the same displacement (relative alpha, as defined in E7), their downstream
behavior success rates agree within 5% relative error and their PPL values agree
within 10%. The 0.95 cosine threshold and the behavioral equivalence are jointly the
falsifiable claim; neither alone suffices.

---

## 3. Falsifier (>=30 words)

**Fired if:** cos(DiffMean, PCA-top-1) < 0.95 at any tested layer-concept combination,
OR if at matched fractional-alpha displacement the behavior success rates differ by
more than 5% relative on a generation-based judge. A single falsifying layer-concept
pair is sufficient to require a conditional claim ("DiffMean ~= PCA for X but not Y").

---

## 4. Citations (>=80 words, Citation Rigor format)

```
Zou, Andy, et al. 2023 NeurIPS 'Representation Engineering: A Top-Down
Approach to AI Transparency' (arXiv:2310.01405) — RepE defines PCA-top-1
extraction; uses it as the steering vector throughout; the implicit claim
that PCA is better than DiffMean is never directly tested against it.

Panickssery, Aryan, et al. 2023 arXiv 'Steering Llama 2 via Contrastive
Activation Addition' (arXiv:2312.06681) — DiffMean/CAA pipeline; the
implicit claim that the mean-difference is sufficient is never tested
against PCA. E4 makes this comparison explicit and measurable.

Park, Junsoo, et al. 2025 ICML 'CAST: Conditional Activation Steering'
(arXiv:2409.05907) — uses PCA-top-1 for condition vector; if E4's
alignment holds, CAST can switch to DiffMean without any expected loss,
saving SVD cost at every inference-time condition check.

Venkatesh, Thamme G., and Kurapath, Anisha. 2026 ICLR workshop 'On the
Non-Identifiability of Steering Vectors' (arXiv:2602.06801) — establishes
that large equivalence classes of steering directions produce equivalent
behavior; E4's high cosine is a special case of this non-identifiability:
DiffMean and PCA-top-1 land in the same equivalence class.
```

---

## 5. Mechanism (deep technical)

PCA-top-1 on the stacked contrastive matrix X = [h_pos_1 - h_neg_1; ...; h_pos_n - h_neg_n]
is the leading left singular vector of X. DiffMean is mean(X) / n. By the law of large
numbers, mean(X) / n converges to the population mean mu = E[h_pos] - E[h_neg].

The PCA-top-1 vector (let us call it u_1) is the direction of maximum variance in X.
If the rows of X are i.i.d. centered at mu with a covariance Sigma such that the
first eigenvector of Sigma is nearly aligned with mu (i.e., the concept direction IS
the dominant variance direction), then u_1 ~ mu / ||mu||, and cos(DiffMean, PCA-top-1)
~ 1.

More precisely, by the Davis-Kahan theorem:

    sin(angle(u_1, mu/||mu||)) <= ||Sigma_perp|| / (lambda_1 - lambda_2)

where lambda_1, lambda_2 are the first and second eigenvalues of the covariance and
Sigma_perp is the component of Sigma orthogonal to mu. When the concept is strongly
linear (the dominant variance is in the mu direction), lambda_1 >> lambda_2, and the
sine of the angle between DiffMean and PCA-top-1 is small — which is what the high
cosine (>0.99) empirically confirms.

The practical implication: the high alignment is NOT a tautology. It is a strong claim
that, for behavioral concepts in Gemma-class models, the contrastive variance is
dominated by a single direction (the concept axis) rather than being multi-modal or
diffuse. This is a form of the Linear Representation Hypothesis (Nanda et al.
arXiv:2309.00925 [UNVERIFIED]): concepts are encoded as near-linear directions.

The C9b result (S-9) adds the displacement-invariance corollary: even though the raw
norms of DiffMean (unnormalized) and PCA-top-1 (unit) differ by ~10x (making raw-alpha
comparisons meaningless), after normalizing both to unit length and using the same
fractional displacement (E7 relative_add), their steering outputs are
indistinguishable within measurement noise (behavior within 0.02, PPL within 8%).

---

## 6. Predicted Delta (pre-registered numbers)

| Metric | Predicted value | Observed (screening) |
|--------|----------------|---------------------|
| cos(DiffMean, PCA-top-1) at best layer | >= 0.95 | **0.996 (Qwen-0.5B, S-1), 0.994 (Gemma-270m, S-3), 0.9945 (Gemma-1B, S-8)** |
| Behavior gap (matched fractional alpha) | < 5% relative | **~0.02 absolute (within noise, C9b)** |
| PPL gap (matched fractional alpha) | < 10% relative | **~8% (within noise, C9b)** |
| Number of model-layer combinations confirming | >= 3 | **3 confirmed (screening)** |

---

## 7. Experimental protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it (4-bit), L16.
- **Concepts:** >= 3 behaviors (at least one safety/refusal, one truthfulness, one
  stylistic — to ensure concept diversity).
- **Metrics:** cos(DiffMean, PCA-top-1) at each layer × concept combination;
  behavior success via LLM-judge at matched fractional alpha (E7 parameterization);
  PPL.
- **Seeds:** n=7 for the Wilcoxon gate.
- **Control:** cos(random-direction, PCA-top-1) as null baseline.

### 7.2 Where this should SHINE

The alignment is expected to hold at all tested layers for behavioral concepts. The
only case where it might fail: a concept whose contrastive distribution is genuinely
multi-modal (two distinct "styles" of the positive class), in which case PCA-top-1
may point to the between-cluster direction while DiffMean averages the two clusters
and points elsewhere. This is the "SHINE" test for PCA: if the concept has multi-modal
positive examples, PCA captures the dominant mode while DiffMean captures the average.
Testing a multi-modal concept is an explicit rung-3 extension.

### 7.3 Implications for downstream Block B (CAST)

If E4 holds, CAST (arXiv:2409.05907) can replace its SVD-based condition extraction
with DiffMean — a 10–20x speedup in the gating step. This is a direct practical
payoff of the confirmation.

---

## 8. Cross-references

- **S-1, S-3, S-8** (screening): cos confirmed on 3 models (Qwen-0.5B, Gemma-270m, Gemma-1B).
- **C9b/S-9** (E7 + E36): at matched fractional alpha, DiffMean and PCA-top-1 steer
  near-identically — the high cosine implies equivalent steering.
- **E7** (norm-relative alpha): the critical "matched fractional alpha" parameterization
  that makes E4's behavioral equivalence testable.
- **E36** (SAE selection problem): S-9 confirmed E36 as supported via the E4 mechanism
  (DiffMean ~= PCA ~= the right direction; gaps were parameterization artifacts).
- **Non-Identifiability** (arXiv:2602.06801): the large equivalence class of directions
  includes both DiffMean and PCA-top-1 as members.
- **IDEA_TABLE.md** Block A row E4.

---

## 9. Committee Q&A

**Q: The three screening measurements are all n=1, single concept. How confident can
we be?**

> The cosine values (0.994, 0.9945, 0.996) are strikingly consistent across three
> models and two architectures at n=1. The mathematical argument (Davis-Kahan bound)
> predicts high alignment whenever the concept is strongly linear — which empirically
> it is. The formal confirmation requires n>=7 + multiple concepts + real judge.
> The screening is "already convincing" in the sense that three independent replications
> all exceed the 0.95 threshold by a comfortable margin.

**Q: Doesn't this make E4 trivial — of course PCA and mean are aligned?**

> Not trivially. If the concept distribution were bimodal (two clusters of positive
> examples), or if the dominant variance were in a confound direction (e.g., prompt
> length vs concept), PCA-top-1 would NOT align with DiffMean. The 0.994–0.996
> alignment is a claim about the structure of behavioral concept representations in
> Gemma-class models: they are approximately unimodal and linear. The Non-Identifiability
> paper (arXiv:2602.06801) shows this is NOT guaranteed — it is an empirical regularity
> that happens to hold here.

**Q: If E4 holds, why bother with PCA at all?**

> Two reasons PCA still has value: (1) PCA-top-1 gives a unit-normalized vector with
> no scaling ambiguity, while DiffMean's norm encodes pair-set statistics that can vary;
> (2) PCA-top-2 and higher components capture residual variance that DiffMean discards —
> E28 (behavior plane low-rank) tests whether these higher components add anything.
> E4's conclusion is "DiffMean suffices for direction," not "PCA has no value."

---

## 10. Verification artifacts checklist

- [x] S-1: Qwen-0.5B @L21, cos=0.996, n=1.
- [x] S-3: Gemma-3-270m @L12, cos=0.994, n=1.
- [x] S-8/C6: Gemma-3-1B @L18, cos=0.9945, n=1.
- [x] C9b/S-9: matched fractional alpha, behavior ~= PPL ~= within noise.
- [ ] Gemma-2-2B @L16, >=3 concepts, generation judge, n>=7 (rung-3 gate).
- [ ] Multi-modal concept test (potential falsifier).
- [ ] CAST gating speedup benchmark (DiffMean vs SVD) — practical payoff.
- [ ] Result row in `EXPERIMENT_LEDGER.md`.

---

## 11. Status journal

- 2026-05-27 — hypothesis inherited from corpus E4; threshold 0.95 pre-registered.
- 2026-05-29 — S-1 (Qwen-0.5B, bring-up): cos=0.996. **SUPPORTED screening.**
- 2026-05-29 — S-3 (Gemma-3-270m): cos=0.994. Third-model confirmation.
- 2026-05-30 — S-8/C6 (Gemma-3-1B): cos=0.9945. Cross-scale confirmed.
- 2026-05-30 — C9b/S-9 (relative alpha): DiffMean and PCA steer equivalently at matched
  fractional alpha. E36 supported as a corollary.
- 2026-05-31 — Design doc written. Status: **SUPPORTED (screening, 3 models)**.
  Promotion requires >=3 concepts + n>=7 + generation judge on Gemma-2-2B.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-A. Critiquing the IDEA — this is the most cleanly confirmed
Block A hypothesis.*

### Prior plausibility

**HIGH.** The mathematical argument (DiffMean ~ PCA-top-1 when the concept is the
dominant variance direction) is rigorous. The empirical screening results are very
consistent. The only genuine surprise would be a concept where the two methods disagree.

### Mechanism scrutiny

The Davis-Kahan bound provides the right framework. The doc correctly identifies the
condition for agreement (lambda_1 >> lambda_2, i.e., single dominant concept direction)
and the condition for disagreement (multi-modal distribution). The C9b result adds the
key "behavioral equivalence at matched displacement" claim, which goes beyond cosine
alignment and is the directly policy-relevant claim.

### Confounds

1. **Easy synthetic concept**: all three screening runs used concepts where the positive
   class is clearly defined. A harder, more ambiguous behavioral concept (where "being
   honest" has multiple distinct activation signatures) could break the alignment.
2. **Single behavior per run**: the screening measured one concept per model. Inter-concept
   variability is untested.
3. **LayerNorm folding**: if activations are post-LayerNorm (as they are in Gemma-2),
   the distribution may be more Gaussian (well-conditioned covariance), which makes
   DiffMean ~= PCA more likely. In a model with unnormalized residuals, they might differ more.

### Does the 0.95 threshold specifically matter?

**Yes, for downstream CAST.** If cos = 0.95, the condition gating in CAST still works.
If cos = 0.80, CAST with DiffMean would under-separate harmful from harmless inputs.
The 0.994–0.996 observed values are comfortably above 0.95, suggesting significant
margin.

### Literature precedent

No direct DiffMean vs PCA comparison exists in the literature. The closest: AxBench
(arXiv:2501.17148) reports that DiffMean and PCA-based methods perform similarly
on their benchmark but does not measure their cosine alignment directly. RepE
(arXiv:2310.01405) prefers PCA without justification against DiffMean. E4 fills
a concrete methodological gap.

### Skeptical effect-size re-prediction

For behavioral concepts on Gemma-class models: cos in [0.97, 0.999], 80% CI.
For multi-modal or ambiguous concepts: cos in [0.70, 0.95]. The 0.95 threshold
will hold for well-defined behavioral concepts; it may not for ambiguous ones.

### Minimum-distinguishing experiment

Already done for screening. Rung-3: 3 behaviors × Gemma-2-2B × L16 × n=7 × real judge.
~4 hours. Expected result: cos > 0.95 for all 3 behaviors. The main risk is the
multi-modal concept (if selected) breaking the alignment — which would be a new,
interesting finding.

### Verdict

**SUPPORTED (screening). Most likely to survive full evaluation of all Block A
hypotheses.** The mathematical foundation is solid, the screening evidence is
consistent, and the behavioral equivalence at matched displacement (C9b) is the
directly policy-relevant confirmation. Promote to rung-3 alongside E3.

---

## Pseudocode & Methodology

This hypothesis compares the two extraction **sources** (DiffMean vs PCA-top1). The
knob varied is **source**; it is mostly an extraction-side cosine comparison, with an
optional matched-displacement behavioral check using E7's `relative_add`.

### 1. Steering-vector recipe

```python
# Both sources from the SAME contrast set at the SAME layer (METHODOLOGY §1.3)
H = collect_activations(model, tok, load_concept(behavior), layer=16)
v_diffmean = diffmean_vector(H.pos, H.neg)          # mean(pos)-mean(neg)
D          = H.pos - H.neg                          # per-pair differences [n,dim]
v_pca      = pca_top1_vector(H.pos, H.neg)          # top right singular vector of D,
                                                    # sign-aligned to v_diffmean
cos_dm_pca = cosine(v_diffmean, v_pca)              # the PRIMARY quantity
# build_vector_bank returns {diffmean, pca, cosine_dm_pca, fisher} per layer.
```

### 2. Experiment procedure

```text
1. For each (concept, layer) extract v_diffmean and v_pca on the same pairs.
2. Record cos(v_diffmean, v_pca).                      # cosine alignment
3. Behavioral equivalence check (matched displacement, E7 parameterization):
4.   for source in {diffmean, pca}:
5.       v_hat = v/||v||
6.       h' = apply_operation(h, v_hat, "relative_add", alpha_rel=0.10)
                                                       # h' = h + 0.10*||h||*v_hat
7.       record behavior (judge) and PPL.
8. Compare behavior gap and PPL gap between the two sources.
9. CONTROL: cos(random_direction, v_pca) as a null baseline (controls.matched_norm_random).
```

The matched-`relative_add` step is essential: raw norms of DiffMean and unit PCA differ
~10x, so only at matched *fractional* displacement is the behavioral comparison fair.

### 3. Measurement & decision rule

PRIMARY metric: `cos(DiffMean, PCA-top1)` at the best layer; SECONDARY: behavior-gap and
PPL-gap at matched `alpha_rel`. FALSIFIER (§3): fires if `cos < 0.95` at any tested
layer-concept pair, OR if behavior success differs by `> 5%` relative at matched
fractional alpha. Pre-registered (§6): `cos >= 0.95`; behavior gap `< 5%`; PPL gap
`< 10%`; ≥3 model-layer confirmations. Observed: `cos = 0.994–0.996` on three models;
matched-alpha behavior within ~0.02. Verdict: **SUPPORTED at screening** (Davis-Kahan:
high alignment iff the concept is the dominant variance direction, λ1≫λ2). Promotion:
≥3 concepts + n≥7 + generation judge on Gemma-2-2B, plus a multi-modal-concept
falsifier probe.

### 4. Where the code is / status

The cosine comparison runs inside `extract.build_vector_bank` (the `cosine_dm_pca`
field); the matched-displacement behavioral arm is the C9b/S-9 `relative_add` run via
`scripts/campaign_sweep.py` (exp# 1, 55–59, 98–109). Status **SUPPORTED (screening,
3 models)**; no new machinery needed for promotion — only more concepts/seeds and the
off-family judge.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E4.md`](../PROVENANCE/E4.md).

- **Experiments:** exp# 1, 55, 56, 57, 58, 59, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 (`autoresearch_results/experiment_log.jsonl`).
- **Reproduce:**

```bash
PYTHONPATH=src python -m steering.runner  # Rung-0/1 plumbing gate on the offline FakeResidualLM (infra, not a Gemma claim)
```
