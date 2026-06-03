# N10 — The Concept Algebra Closure Hypothesis

> **One-line claim:** Behavior directions form an approximately closed algebra under
> linear combination plus conceptor AND/OR; a novel behavior with no contrast data can
> be synthesized as a learned combination of primitive vectors and steered successfully
> at >= 60% of supervised efficacy.
>
> **Primary axes:** A2 (direction/what), A4 (operation)
> **Status:** UNTESTED

---

## 1. Motivation (>= 100 words)

Steering methods require extracting contrast pairs for every behavior of interest —
a significant data collection burden. If behavior directions form an approximately
closed algebra, this burden is reduced: novel behaviors can be synthesized algebraically
from a small basis of primitive directions, without collecting new contrast pairs.
The linear representation hypothesis (Park et al. 2023, arXiv:2311.03658) establishes
that many behavioral concepts ARE linearly represented in LLM activation space. If
the set of linear behavior directions forms an algebra closed under the operations
{add, subtract, scale, conceptor-AND, conceptor-OR}, then behavior "arithmetic" is
possible: (polite) - (formal) + (casual) might synthesize (casually polite), (safe)
AND (concise) might synthesize a safe-and-concise response style. This is the vector
arithmetic that King - Man + Woman = Queen exemplifies in word2vec — but for
behavioral directions rather than semantic concepts. The closure hypothesis is a
step-change claim: if true, it means N primitive behaviors (e.g., 10-20) are
sufficient to span the behavioral space of interest, and any new behavior is
expressible as an algebraic expression of primitives. This is the theoretical
foundation for the N12 unified operator's "composition" functionality. Conceptor
AND/OR (Jaeger 2014) provides the nonlinear composition operators that handle cases
where linear combination fails (e.g., behaviors that are incompatible cannot be
simply added).

## 2. Formal Hypothesis (>= 50 words)

Let P = {p_1, ..., p_K} be a set of K=10 primitive behavior vectors extracted by
DiffMean. Let B be a held-out behavior with DiffMean vector v_B (used as ground
truth). The claim is:

  There exist coefficients {c_i} and/or a conceptor-AND expression C such that:
  cos(v_B, sum_i c_i * p_i) >= 0.60 OR
  cos(v_B, Conceptor-AND({p_i : c_i > 0})) >= 0.60

and further that steering with the synthesized vector achieves >= 60% of the
efficacy (behavior-cosine shift) of steering with v_B at matched off-shell displacement,
on Gemma-3-1B-it, for at least 3 of 5 held-out behaviors.

## 3. Falsifier (>= 30 words)

If the maximum achievable cos(v_B, synthesized) < 0.40 for all algebraic
combinations of the primitive set, or if synthesized steering efficacy < 40% of
supervised for all 5 held-out behaviors, the closure hypothesis is FALSIFIED. A
random-combination baseline must be included: if random {c_i} achieve similar
cosine, the algebra is not informative.

## 4. Citations (Citation Rigor >= 80 words)

```
Park, Chowdhery, et al. 2023. 'The Linear Representation Hypothesis and the
Geometry of Large Language Models' arXiv:2311.03658. Establishes that many
behavioral/conceptual distinctions are linearly represented in LLM activations;
this is the foundation for N10's closure claim — if representations are linear,
their algebraic combinations should also be behavioral.

Mikolov, Chen, Corrado, Dean 2013. 'Efficient Estimation of Word Representations
in Vector Space' arXiv:1301.3781. Word2vec vector arithmetic (King - Man + Woman =
Queen) is the direct conceptual precedent for N10's behavior arithmetic claim.
N10 extends this from word embeddings to behavioral steering vectors.

Jaeger, Herbert 2014. 'Controlling Recurrent Neural Networks by Conceptors.'
arXiv:1403.3369. Conceptors are matrix-valued operators that implement fuzzy
AND/OR/NOT for neural reservoir patterns; N10's nonlinear composition is exactly
the Conceptor-AND of behavioral patterns. The closed-form update is computationally
cheap.

Venkatesh & Kurapath 2026. 'On the Non-Identifiability of Steering Vectors'
arXiv:2602.06801. The coset structure implies that multiple algebraic expressions
may achieve the same behavioral effect; N10 exploits this: if the synthesized
vector is in the same coset as v_B, it achieves the same behavior despite
being algebraically constructed.
```

## 5. Mechanism

The closure hypothesis has two levels:

Level 1 — Linear span: if v_B ≈ sum_i c_i * p_i for some {c_i}, then B is
in the linear span of the primitive set. This is testable by least-squares
regression: fit {c_i} to minimize ||v_B - sum_i c_i * p_i||^2. If R2 >= 0.36
(cos >= 0.60), the span claim holds.

Level 2 — Conceptor AND: conceptors C_i are matrix projections onto the "space of
patterns consistent with behavior i." Conceptor-AND C_i AND C_j = (C_i * C_j) with
normalization, producing the intersection of two behavior spaces. This handles
incompatible behaviors (e.g., "verbose" AND "concise" cancel; "factual" AND "safe"
compound). Conceptor-AND as an operation is O(d_model^2) but can be approximated
by the top-k SVD of the intersection for computational feasibility.

Primitive selection: the K=10 primitives should span diverse behavioral dimensions:
{refusal, politeness, formality, factuality, conciseness, verbosity, confidence,
hedging, emotional warmth, directness}. These are chosen to be approximately
orthogonal (|cos(p_i, p_j)| < 0.3 for all pairs — testable pre-experiment).

Synthesis procedure: for a new behavior B described by a few-sentence natural
language description, use a semantic embedding model to map the description to
a coefficient vector {c_i} (cosine similarity of description to each primitive's
name/description), then construct sum_i c_i * p_i as the synthesized vector.
This provides a zero-cost synthesis without any contrast pairs for B.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| cos(v_B, linear-span-synthesized) | 0.60 - 0.80 | Linear representation hypothesis |
| Steering efficacy of synthesized vs supervised | 60% - 85% | Some behavioral nuance lost |
| Random-combination baseline cos | 0.10 - 0.25 | Random coefficients are near-orthogonal to v_B |
| Fraction of 5 behaviors achieving >= 60% | 3 - 5 out of 5 | Some behaviors may be non-linear |
| Conceptor-AND advantage over linear for incompatible pairs | +5% to +20% efficacy | Handles composition non-linearity |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16
- Primitives: K=10 DiffMean vectors from 10 diverse behaviors, 50 pairs each
- Orthogonality check: confirm |cos(p_i, p_j)| < 0.40 for all pairs (if not, reduce K)
- Held-out behaviors: 5 behaviors from a different domain (not used in primitive extraction)
  e.g., {technical-explanation, empathetic-response, Socratic-questioning,
  assertive-correction, playful-humor}
- Synthesis: (a) least-squares regression of v_B onto primitives (oracle coefficients),
             (b) semantic-embedding-based coefficients (zero-shot coefficients)
- Conceptor-AND: compute AND of top-3 primitives by cosine with description embedding
- Metrics: cos(v_B, synthesized); behavior-cosine shift at 5% off-shell; PPL
- Baseline: random {c_i} synthesis
- Seeds: 3 primitive-extraction seeds x 3 evaluation seeds
- Wall-clock: ~4 hours on RTX 4090

### 7.2 Where it shines

New behavior deployment without contrast pair collection: in settings where labeling
contrast pairs is expensive (e.g., novel task types, cross-lingual behaviors, rare
personality traits), algebraic synthesis provides an immediate baseline that can
be iteratively refined.

## 8. Cross-References

- N4 (ICL as dual): N4 predicts that ICL demonstrations produce the same activation
  shift as DiffMean; if true, the primitive set could be learned from demos rather
  than contrast pairs, reducing the collection burden further
- N8 (causal SAE): the causal-SAE basis is the natural primitive set for N10's algebra
  (fewer, more causally potent atoms)
- N6 (separation principle): the behavior vector b in N6 is the "execution" component;
  N10's synthesized vectors should also be orthogonalized against condition vectors
- N18 (interference-budget additivity): algebraic combination of multiple primitives
  accumulates their off-diagonal Gram mass; N10's efficacy threshold of 60% must
  account for the N18 interference budget
- N12 (capstone unified operator): N10's synthesized vectors are the input to the
  N12 operator; concept algebra is the "how derived" axis (A7) of the capstone
- IDEA_TABLE.md: N10 row, axes A2+A4

## 9. Committee Q&A

**Q: "Closure" is a strong algebraic claim. You only need the synthesized vector to
be in the coset of v_B (Non-ID), not equal to v_B. Isn't the 60% efficacy threshold
just testing coset membership, not algebraic closure?**

> Exactly right. The claim is not that sum_i c_i * p_i = v_B exactly; it is that
> the synthesized vector is in the behavioral coset of v_B (achieves the same behavioral
> effect). This is the appropriate claim given the Non-ID coset structure. Calling it
> "closure" is shorthand for "the primitives' algebraic span covers the behavioral
> coset of most target behaviors."

**Q: The semantic-embedding-based coefficient estimation (zero-shot synthesis) may
produce poor {c_i} if the primitive behaviors are described differently from the
target behavior. How is this validated?**

> The oracle (least-squares) synthesis provides the ceiling: if oracle achieves 80%
> but zero-shot achieves only 40%, the bottleneck is the semantic encoding, not the
> algebra. In that case, the claim is "algebraic closure holds; semantic encoding
> is the limiting factor." A follow-up could optimize the encoding.

**Q: Word2vec arithmetic (King - Man + Woman = Queen) worked because word embeddings
are trained to be linear. Behavior vectors are NOT trained with algebraic constraints.
Why would they be algebraically consistent?**

> They are trained (fine-tuned) to represent behavior in a way that makes behavioral
> differences approximately linear (the DiffMean assumption). The linear representation
> hypothesis (Park et al. 2023) provides evidence that this IS the case for many
> behavioral concepts in instruction-tuned LLMs. N10 is a test of how broad this
> linearity extends — the scope of the "closure" in the algebra.

## 10. Verification Checklist

- [ ] K=10 primitive behaviors selected with documented diversity rationale
- [ ] Pairwise |cos(p_i, p_j)| < 0.40 verified for all 45 pairs
- [ ] 5 held-out behaviors selected from different domain, documented before extraction
- [ ] Oracle (least-squares) synthesis vs zero-shot (semantic) synthesis vs random
- [ ] Conceptor-AND implementation with top-3 primitive selection
- [ ] cos(v_B, synthesized) and behavior-efficacy measured for all 5 held-out behaviors
- [ ] Random-combination baseline included
- [ ] 3 x 3 seed design recorded in EXPERIMENT_LEDGER.md
- [ ] IDEA_TABLE.md N10 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. Word2vec arithmetic is the
  direct conceptual precedent and is well-established; the claim here is that behavior
  vectors (not word embeddings) form a similar algebra. No data exists for LLM behavior
  vector arithmetic at the 60% efficacy threshold. The linear representation hypothesis
  (Park et al. 2023) provides theoretical support.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. Word2vec arithmetic has well-known limitations even in its domain (it fails
on low-frequency words, on complex relational compositions, and on hierarchical
relationships). Behavior directions have been shown to be linearly separable for
simple binary contrasts; multi-dimensional algebraic composition is a significant
extrapolation. The 60% efficacy threshold is conservative enough to be achievable.

### Mechanism scrutiny

The closure mechanism assumes that the behavioral space is "approximately flat"
in the linear sense — that behavior vectors compose linearly without significant
nonlinear interactions. For behaviors that interact (e.g., "formal" and "empathetic"
may interact nonlinearly: formal-empathetic may not be a simple linear combination
of each), the linear algebra will fail. The Conceptor-AND extension handles some
nonlinearity but is still quadratic in the number of primitives.

### Confounds

1. Primitive set design: if the K=10 primitives are chosen to span the held-out
   behavior space (even implicitly), the synthesis will work by construction. The
   held-out behaviors must be selected BEFORE the primitive set is designed, and the
   primitive selection must use a different (non-behavioral) criterion (e.g., diversity
   in semantic embedding space).
2. The oracle synthesis uses v_B as the target — this is circular if v_B is used
   to design the primitives. The oracle provides a ceiling, not a prediction; the
   zero-shot synthesis is the actual prediction.
3. Efficacy vs cosine: high cos(synthesized, v_B) does not guarantee high steering
   efficacy if the synthesized vector's magnitude is wrong. The efficacy comparison
   must use matched off-shell displacement (not matched alpha).

### Does the algebraic closure claim specifically matter?

HIGHLY. If even the oracle synthesis achieves < 40% efficacy, the behavioral space
is fundamentally non-linear and word2vec-style arithmetic does not transfer to
behavior vectors. If oracle achieves > 80%, the behavioral space is approximately
linear and a small primitive set suffices for zero-shot behavior synthesis — a
practically transformative finding that reduces the contrast-pair collection burden.

### Literature precedent

Ilharco et al. 2023 (arXiv:2212.04089, "Editing Models with Task Arithmetic") shows
that fine-tuning weight deltas compose algebraically for task transfer — the weight-space
version of N10. If weight-space task vectors compose linearly, activation-space behavior
vectors might too (since fine-tuning weight deltas are designed to produce activation
differences). This is the closest precedent.

### Skeptical effect-size estimate

Oracle synthesis cos: 0.50-0.70 (vs claimed 0.60-0.80). The linear span of 10
primitives in a 2048-dimensional space can approximate many directions, but behavioral
nuance may require more dimensions than 10. Zero-shot synthesis: 0.35-0.55 (vs needed
> 0.60 for 60% efficacy threshold). Recommend starting with oracle to establish the
ceiling before testing zero-shot.

### Minimum distinguishing experiment

One held-out behavior, oracle synthesis only (no zero-shot): compute cos(v_B,
least-squares-synthesized) using K=10 primitives. Cost ~45 min (includes primitive
extraction for all 10). If cos > 0.60, proceed to zero-shot and full protocol.
If cos < 0.40, the linear span is insufficient and the hypothesis is likely FALSIFIED.

### Verdict

TESTABLE-MEDIUM. The oracle-synthesis pre-check (45 min) is the essential first step.
High oracle cos would confirm the linear representation hypothesis applies to behavior
vectors; zero-shot synthesis is the operationally important claim that follows.
Recommend running oracle first; commit to zero-shot and conceptor experiments
only if oracle cos > 0.55.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N10 tests whether behavior directions form a closed algebra, so a novel behavior can be synthesized from primitives with no contrast data. **UNTESTED** — needs least-squares + conceptor synthesis.

### 1. Steering-vector recipe (algebraic synthesis from primitives)

```python
P = [diffmean_vector(b_i, baseline, L) for b_i in 10_primitives]   # extract.diffmean_vector
v_B = diffmean_vector(held_out_behavior, baseline, L)              # ground-truth target

# (a) oracle synthesis: least-squares coefficients onto the primitive span
c = lstsq(stack(P), v_B)                ; v_syn = sum(c_i * P_i)
# (b) zero-shot synthesis: c_i = cos(embed(description_B), embed(name_i))
# (c) conceptor-AND of top-3 primitives (Jaeger 2014) for incompatible behaviors
# steer with v_syn additively (METHODOLOGY §2 add) at matched off-shell vs supervised v_B
```

### 2. Experiment procedure

```text
1. Pick 10 primitives with |cos(p_i,p_j)| < 0.40; pick 5 held-out behaviors (different domain), pre-registered.
2. Synthesize each held-out v_B by (a) oracle lstsq, (b) zero-shot embedding, (c) conceptor-AND.
3. Measure cos(v_B, v_syn) and steering efficacy of v_syn at matched off-shell (geometry.offshell_displacement).
4. RANDOM-coefficient baseline included (algebra must beat random).
```

### 3. Measurement & decision rule

- **Primary metric:** cos(v_B, v_syn) and synthesized efficacy as a fraction of supervised v_B.
- **Pre-registered falsifier (§3):** max achievable cos < 0.40 for all combinations, OR synth efficacy < 40% supervised on all 5 behaviors, OR random matches the algebra ⇒ FALSIFIED.
- **Verdict logic:** SUPPORTED needs cos ≥ 0.60 AND ≥ 60% efficacy on ≥3 of 5 behaviors, above the random baseline.

### 4. Where the code is / status

UNTESTED. DiffMean extraction and the efficacy/off-shell probes exist, but the **least-squares + semantic-embedding + conceptor-AND synthesis machinery** is not implemented — that missing code is why N10 is UNTESTED.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N10.md`.
