# E10 — Category Condition Orthogonality

> **One-line claim:** Condition vectors extracted for distinct safety categories
> (hate speech, self-harm, illegal advice) are near-orthogonal (|cos| < 0.3),
> enabling independent OR-gating without cross-category interference.
>
> **Source design space:** Block B — Conditional / Gated Steering (E9–E16).
>
> **Implementation status:** `PENDING — UNTESTED`. Condition vector extraction
> across multiple safety categories requires the CAST extraction pipeline
> (blocked on E9 code) plus a multi-category contrast dataset.

---

## 1. Motivation (>= 100 words)

A realistic deployment of conditional safety steering will not protect against
a single monolithic harm category — it must handle several simultaneously: hate
speech, self-harm facilitation, illegal-activity advice, and others. The key
architectural question is whether the condition vectors for these categories are
geometrically separable. If they are near-orthogonal in the activation space of
the model's early layers, they can be OR-composed (fire the behavior vector if
ANY condition is met) without mutual contamination: a hate-speech condition vector
that fires will not falsely trigger a self-harm gate. If they are aligned or
partially overlapping, the gates will either double-count (near-parallel) or
produce logical inconsistencies (partially anti-aligned). This geometric property
has a direct consequence for the architecture of Block B: near-orthogonality
licenses the independent OR-gating tested in E11, while overlap requires a
learned composite gate (E15). The result also constrains the norm budget: if
condition vectors are near-orthogonal and read-only, their composition costs
nothing in the N5 budget, making OR-gating essentially free as a meta-layer.

---

## 2. Formal Hypothesis (>= 50 words)

Because safety categories encode semantically distinct harm concepts (racial/ethnic
hatred, physical self-injury, law violation), and because DiffMean condition
vectors capture the mean activation direction that separates the category-relevant
from the category-irrelevant prompts, semantically orthogonal concepts should
map to geometrically near-orthogonal directions in the model's early-layer
representation space. This is the Linear Representation Hypothesis (LRH) applied
to condition vectors: distinct semantic concepts occupy distinct linear directions.
Formal claim: for all pairs from {hate-speech, self-harm, illegal-advice} condition
vectors extracted at the optimal condition layer L_c on Gemma-2-2B-it, the absolute
cosine similarity |cos(v_i, v_j)| < 0.3 at 3-seed median, with the within-category
cosine significantly higher (|cos| > 0.8) confirming the vectors are internally
consistent.

---

## 3. Falsifier (>= 30 words)

If any pair of category condition vectors exceeds |cos| = 0.3 at 3-seed median
across L_c in {6, 8, 10, 12}, this experiment is FALSIFIED for that pair, meaning
independent OR-gating is not safe for those categories without Gram-Schmidt
orthogonalization (E19 applies). If all three pairs satisfy |cos| < 0.3,
the hypothesis is SUPPORTED and E11 is licensed to proceed with OR-gating.
A within-category cosine below 0.8 (inconsistent vectors) would also indicate
that the extraction procedure is unreliable, demoting status to UNCLEAR.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering: Concept-Level
Control via Conditional Vectors' (arXiv:2409.05907) — introduces OR/AND logical
composition of condition vectors; the OR-gating architecture that E11 tests is
motivated by this work; orthogonality of condition vectors is an implicit
assumption in CAST's compositional design that this experiment makes explicit.

Park, Kiho, et al. 2024 arXiv 'The Geometry of Truth: Emergent Linear Structure
in Large Language Model Representations of True/False Datasets'
(arXiv:2310.06824) — demonstrates that semantically distinct concepts occupy
distinct linear directions in LLM representation space; provides the LRH backing
for why safety categories should be near-orthogonal condition vectors.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — the refusal direction is a specific case
of a "condition-like" direction; the finding that a single refusal direction
captures cross-category refusal behavior raises the question of whether per-
category condition vectors are truly independent or project onto this shared
direction (motivating the falsifier).

Tang, Haoran, et al. 2025 arXiv 'FineSteer: Fine-Grained Steering of Large
Language Models via Subspace-Guided Conditional Activation Steering'
(arXiv:2604.15488) — SCS energy-ratio gating is designed for fine-grained
category discrimination; whether its subspace decomposition produces more
orthogonal category representations than plain DiffMean is an implicit claim
this experiment can partially adjudicate.
```

---

## 5. Mechanism

### 5.1 Geometric argument for near-orthogonality

The DiffMean condition vector for category C is:

    v_C = mean(h_early | prompt in C) - mean(h_early | prompt not in C)

If two categories C1 and C2 are semantically unrelated, the set of prompts
activating C1 overlaps little with those activating C2. The distribution of
h_early on C1-prompts vs non-C1-prompts is shifted along a direction that
encodes C1's semantic content. Similarly for C2. If C1 and C2 occupy different
semantic subspaces in early-layer activation space (as LRH predicts), the two
difference vectors will be nearly orthogonal:

    cos(v_C1, v_C2) = (mean_diff_C1)·(mean_diff_C2) / ||...|| ||...||
                    ≈ 0  when C1, C2 semantically distinct

### 5.2 Failure mode

If C1 = hate-speech and C2 = illegal-advice share a common semantic ancestor
("malicious intent"), their condition vectors may both have a large component
along a shared "harm intent" direction, producing |cos| > 0.3. This is the
mechanistic prediction behind the falsifier.

### 5.3 Protocol sketch

```python
def extract_condition_vector(model, prompts_in, prompts_out, layer):
    """DiffMean at given layer."""
    h_in  = get_activations(model, prompts_in,  layer)  # (N, D)
    h_out = get_activations(model, prompts_out, layer)  # (N, D)
    return (h_in.mean(0) - h_out.mean(0))  # (D,)

categories = ["hate_speech", "self_harm", "illegal_advice"]
for layer in [6, 8, 10, 12]:
    vectors = {c: extract_condition_vector(model, pos[c], neg[c], layer)
               for c in categories}
    gram = {(a,b): cosine(vectors[a], vectors[b])
            for a,b in combinations(categories, 2)}
```

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| |cos(hate, self-harm)| | < 0.20 | Semantically very different; shared "harm" ancestor weak |
| |cos(hate, illegal)| | < 0.25 | Some thematic overlap via "malicious intent" |
| |cos(self-harm, illegal)| | < 0.25 | Moderate overlap; both deviant, not hateful |
| Within-category consistency |cos| | > 0.85 | Same extraction on same category should be stable |
| Worst-case cross-category |cos| | < 0.30 | Falsifier threshold |
| Layer-dependence of cos pattern | Mild (< 0.05 range across L_c grid) | LRH predicts layer-stable directions |

All values pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Categories:** hate-speech (HS), self-harm (SH), illegal-advice (IA)
- **Contrast pairs per category:** 50 positive / 50 negative (E1: 50 pairs
  sufficient); negative set = general benign prompts (same for all categories,
  ensuring comparability)
- **Layer grid:** L_c in {6, 8, 10, 12} — covers early-middle range
- **Metrics:** pairwise cosine matrix (3 pairs × 4 layers); within-category
  cosine (3 categories × 4 layers)
- **Control:** compute cosine of (random direction, v_C) — establishes null
  baseline for "accidentally orthogonal"
- **Seeds:** 3 (screening); promote to 7 if any |cos| > 0.25 (close to threshold)
- **Wall-clock:** ~30 min on 4090 (pure activation extraction, no generation)

### 7.2 Where it shines

Categories that are both semantically precise and corpus-balanced (equal numbers
of positives and negatives per category, not dominated by a single sub-type)
will produce the most reliable condition vectors. This experiment shines on
well-curated safety benchmark datasets with clear category labels.

---

## 8. Cross-References

- **E9** (CAST gate): this experiment's result determines whether E9's gate
  can be extended to multiple categories independently
- **E11** (OR-gate coverage vs N): directly depends on this orthogonality result;
  if |cos| > 0.3 for any pair, E11 must add Gram-Schmidt before OR-gating
- **E19** (Gram-Schmidt orthogonalization): the remedy if this experiment fails
- **N6** (gate in read not write): orthogonality of condition vectors supports
  the separation-principle prediction
- **N3** (orthogonal capacity theorem): the participation ratio at L_c governs
  how many orthogonal directions can coexist — relevant to multi-category gating
- **IDEA_TABLE.md** Block B row E10

---

## 9. Committee Q&A

**Q: Why use DiffMean for condition vectors rather than a trained classifier?**

> DiffMean requires no labels beyond the positive/negative split, is available
> from the same infrastructure as behavior vectors (E1–E4), and E4 shows
> DiffMean and PCA are cosine-aligned at 0.994. A trained classifier would be a
> different AXIS 7 (source) choice and is the subject of E15.

**Q: Does a 0.3 cosine threshold have theoretical justification?**

> It is a practical threshold: |cos| = 0.3 corresponds to ~72.5 degrees, meaning
> the vectors share about 9% of their squared norm. For a 2304-dim space this
> is a small but non-negligible component. The threshold is chosen so that
> OR-gating adds at most ~9% cross-category activation mass — empirically
> calibrated to prevent false-refusal leakage in E11.

**Q: Couldn't all safety categories share a common "harm" direction?**

> Arditi et al. (arXiv:2406.11717) shows a single refusal direction mediates
> much of model refusal behavior — this suggests the BEHAVIOR vector may be
> shared. But the CONDITION vector is a discriminative direction for category
> membership, not for the refusal response. These are distinct vectors (E32
> tests this separation at the behavior level). Nonetheless, if a shared
> "harm intent" direction dominates all category condition vectors, the
> experiment will reveal it via high cross-category cosine.

---

## 10. Verification Checklist

- [ ] 50 positive / 50 negative prompts per category assembled (fixed seed)
- [ ] Condition vectors extracted at L_c in {6, 8, 10, 12} for all 3 categories
- [ ] Pairwise cosine matrix (3×3, lower triangle) computed and logged per layer
- [ ] Within-category seed-stability checked (3 seeds; SD of cosine < 0.05)
- [ ] Null baseline: cosine(random_direction, v_C) logged as reference
- [ ] Worst-case |cos| vs 0.3 threshold reported
- [ ] E11 decision node: PROCEED with OR-gating (|cos| < 0.3) or ADD
      Gram-Schmidt (|cos| >= 0.3) flagged in EXPERIMENT_LEDGER.md
- [ ] IDEA_TABLE.md row E10 status updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Code needed:
  (a) multi-category contrast dataset (hate-speech, self-harm, illegal-advice,
  50+50 each, fixed seed); (b) condition vector extraction at multiple layers
  (reuses E9 CAST pipeline); (c) pairwise cosine computation and logging.
  No GPU compute needed beyond activation extraction (< 1 hr). Blocked on
  E9 CAST pipeline being built first.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-B.*

### Prior plausibility

MEDIUM. The LRH prediction of near-orthogonality is plausible for semantically
distant categories but uncertain for overlapping ones (hate/illegal both encode
"malicious intent" semantics). The 0.3 threshold is reasonable but not theoretically
derived.

### Mechanism scrutiny

The mechanism is clear and testable. The main weakness: "distinct categories"
is not formally defined — what counts as semantically distinct enough to be
orthogonal? The experiment sidesteps this by committing to three specific
categories and a numeric threshold, which is correct experimental practice.

### Confounds

1. **Negative-set design:** all categories share the same negative set (benign
   prompts). If benign prompts cluster tightly in activation space, all condition
   vectors will be pushed in the same direction (away from benign-cluster center),
   inducing artificial alignment. Fix: use category-specific negative sets
   (e.g., hate-negatives = politely critical but non-hateful prompts).
2. **Layer sensitivity:** cosines may vary substantially across layers; reporting
   only at optimal L_c is cherry-picking. The protocol correctly sweeps {6, 8, 10, 12}.

### Numerology check

The 0.3 threshold is a convention, not a theory. The experiment should also
report the continuous cosine values (not just pass/fail) to enable future
meta-analysis across models and categories.

### Verdict

**NOVEL+TESTABLE** in the Gemma-2-2B context. The result will directly gate E11
and inform whether multi-category CAST gating requires Gram-Schmidt
preprocessing (E19). Low compute cost makes this an early-priority experiment
in the Block B sequence.

---

## Pseudocode & Methodology

This hypothesis measures **pairwise orthogonality of per-category condition vectors**.
It is extraction-only (no injection): a DiffMean condition vector is built **per safety
category** and the Gram (cosine) matrix is computed. The knob varied is the **category**
(and the read layer `L_c`).

### 1. Steering-vector recipe

```python
# One DiffMean CONDITION vector per category (METHODOLOGY §1.3), at early layer L_c.
def condition_vector(category, L_c):
    H = collect_activations(model, tok, pairs[category], layer=L_c)   # 50 pos / 50 neg
    return diffmean_vector(H.pos, H.neg)        # mean(in-cat) - mean(out-of-cat)

categories = ["hate_speech", "self_harm", "illegal_advice"]
for L_c in [6, 8, 10, 12]:
    V = { c: condition_vector(c, L_c) for c in categories }
```

### 2. Experiment procedure

```text
1. For each L_c in {6,8,10,12}: extract a DiffMean condition vector per category.
2. Pairwise cross-category cosine matrix:
3.     for (a,b) in combinations(categories, 2):
4.         record |cosine(V[a], V[b])|              # extract.cosine
5. Within-category consistency: re-extract per category over 3 seeds; record |cos| (>0.8 expected).
6. CONTROL: cosine(random_direction, V[c]) as the "accidentally orthogonal" null
       (controls.matched_norm_random).
7. Decision node for E11: if worst |cos| < 0.3 -> OR-gating licensed; else add Gram-Schmidt (E19).
```

No generation and no steering — this is pure activation geometry, so it is cheap.

### 3. Measurement & decision rule

PRIMARY metric: worst-case cross-category `|cos(v_i, v_j)|` across the L_c grid (3-seed
median). FALSIFIER (§3): any pair exceeds `|cos| = 0.3` ⇒ FALSIFIED for that pair (OR-gating
unsafe without Gram-Schmidt, E19); a within-category `|cos| < 0.8` ⇒ UNCLEAR (extraction
unreliable). Pre-registered (§6): `|cos(hate,self-harm)| < 0.20`, `|cos(hate,illegal)| < 0.25`,
`|cos(self-harm,illegal)| < 0.25`; within-category `> 0.85`; worst-case `< 0.30`. Verdict:
SUPPORTED iff all pairs `< 0.3` (LRH: distinct concepts ⇒ near-orthogonal directions),
which licenses E11 to OR-compose without Gram-Schmidt.

### 4. Where the code is / status

The per-concept DiffMean banks come from `scripts/campaign_sweep.py` (the existing
multi-behavior extraction over anger/formality/happiness/ocean is the analog; safety
categories need a new harmful/harmless dataset), and the cosine-matrix / stacking analysis
is the E10_E17 cosine analysis over the extracted banks. Status **UNTESTED**: it needs
(a) the multi-category safety contrast dataset (hate/self-harm/illegal, 50+50 each), and
(b) the E9 CAST condition-extraction pipeline. The Gram-matrix computation itself is
trivial (`extract.cosine`); the blocker is the dataset + the upstream CAST pipeline.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E10.md`](../PROVENANCE/E10.md).

- **Experiments:** analysis campaign (computed quantities in the campaign JSON; see the provenance file).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/campaign_sweep.py --model models/google/gemma-3-270m-it --quant none --hyp E10 --tag-prefix E10-ortho --layers 16 --alphas 0.1 --ops relative_add --behaviors anger formality happiness ocean  # then the E10_E17 cosine/stacking analysis over the extracted banks
```
