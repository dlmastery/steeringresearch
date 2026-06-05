# E19 — Gram-Schmidt Orthogonalization of New Behavior Vectors

> **One-line claim:** Gram-Schmidt orthogonalizing a new vector against the
> active set preserves the new behavior while removing its interference with
> existing ones.
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires multi-vector injection
> infrastructure from E17/E18 plus Gram-Schmidt preprocessing module.

---

## In Plain English

**What we're testing, simply:** Before adding a new nudge to ones already in use,
we mathematically strip out the part of it that overlaps with the existing nudges.
The question: does this "clean" nudge still do its job, while no longer interfering
with the others?

**Key terms (defined here):**
- **Steering / steering vector:** nudging the model by adding a direction to its
  internal "thoughts"; the direction is the steering vector.
- **Residual stream:** the model's running internal state that we edit mid-sentence.
- **Layer:** the processing step where we make the edit.
- **Alpha / strength:** how hard we push.
- **DiffMean:** the simple recipe for building a nudge.
- **Coherence:** whether the text stays fluent and sensible.
- **Stacking:** using several nudges at once.
- **Interference:** the nudges getting in each other's way.
- **Orthogonal:** directions that don't overlap. **Gram-Schmidt** is a standard
  procedure that takes a new direction and removes the parts that overlap with
  existing ones, leaving a purely non-overlapping remainder.
- **Norm budget:** the total push the text can take before it breaks; stripping
  overlap also shrinks how far we push, which is gentler on this budget.

**Why we're doing this (the point):** It's a clean recipe for adding behaviors one
at a time without each new one disturbing the ones already working — exactly what
a growing safety stack needs.

**What the result would mean:** If the cleaned nudge keeps almost all of its
effect, we have a safe, reusable way to grow a stack. If cleaning it guts its
effect, that means the overlapping part was actually carrying real behavior, and
this simple trick won't work for those behaviors.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

When a new behavior vector v_new must be added to an already-active stack of
vectors {v_1,...,v_k}, the raw DiffMean vector may have non-zero projections
onto the existing vectors — creating interference in both directions (v_new
partially activates the existing behaviors, and the existing vectors partially
suppress v_new's effect). Gram-Schmidt orthogonalization removes this problem
by projecting v_new onto the complement of the subspace spanned by {v_1,...,v_k}:

    v_new_orth = v_new - sum_i (v_new · v_i) * v_i   (simplified, unit vectors)

This produces a vector that has zero cosine with all existing vectors — guaranteed
independence by construction. The cost is a potential reduction in ||v_new_orth||
(the projection removes part of v_new), which may reduce the efficacy of the
new behavior if the removed component was behaviorally important (not just
interfering). This experiment tests whether Gram-Schmidt orthogonalization
preserves >= 95% of solo v_new efficacy (the removed component was primarily
interfering, not behavior-carrying) while delivering the zero-interference
property that E17 and E18 predict is necessary for clean stacking.

---

## 2. Formal Hypothesis (>= 50 words)

Because the DiffMean behavior vector encodes behavior in the direction of mean
activation difference (semantic content), and the Gram-Schmidt projection removes
the component of v_new that lies in the subspace of existing vectors, the
removed component encodes shared conceptual content between v_new and the existing
behaviors — a component that would produce cross-interference. The remaining
orthogonalized component v_new_orth should still encode the NEW behavior's
specific semantic direction (the part not shared with existing behaviors).
Formal claim: on Gemma-2-2B-it, the orthogonalized vector v_new_orth achieves
>= 95% of solo v_new efficacy on behavior B_new, while reducing interference
with existing behaviors by >= 80% relative to raw v_new addition, at 3-seed median.

---

## 3. Falsifier (>= 30 words)

If v_new_orth achieves less than 95% of solo v_new efficacy (the orthogonalization
removes too much behavior-carrying information), the hypothesis is FALSIFIED
for the efficacy-preservation claim. If interference is reduced by less than
80% (relative to raw add), the orthogonalization is not providing its geometric
guarantee, which would indicate the behaviors are not well-approximated by
their DiffMean vectors (a representation failure). Either failure falsifies
the hypothesis.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Templeton, Adly, et al. 2024 arXiv 'Scaling Monosemanticity: Extracting
Interpretable Features from Claude 3 Sonnet' via SAE-TS context: SAE feature
targeting and side-effect analysis as in arXiv:2411.02193 — SAE-TS uses a
related orthogonalization-like operation to remove side effects from steering
vectors; Gram-Schmidt is the explicit algebraic version of the same intuition.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — the refusal direction as v1 in the
active set; orthogonalizing a new safety behavior vector against it ensures
the new behavior does not disrupt the refusal mechanism.

[Steering-stackable-vs-competing-analysis.md §2.2, this project]: "Independent
behavior vectors stack additively only while they remain near-orthogonal and
the summed norm stays in-distribution. Conceptors and orthogonalized/targeted
vectors (SAE-TS) exist precisely to preserve non-interaction." — the design
rationale for Gram-Schmidt in the composition pipeline.

[N5 geometry result, C2, this project]: logPPL = 5.40 + 2.87 * offshell, R² =
0.81 — the orthogonalized vector has lower ||v_new_orth|| than ||v_new||;
by N5, the joint norm (and thus PPL overhead) should be lower post-orthogonalization.
```

---

## 5. Mechanism

### 5.1 Gram-Schmidt procedure

Given active set {v_1,...,v_k} (orthonormal by prior applications of GS or
by the initial unit-vector normalization):

    v_new_raw = DiffMean(new_behavior)
    for v_i in {v_1,...,v_k}:
        v_new_raw -= (v_new_raw · v_i) * v_i
    v_new_orth = v_new_raw / ||v_new_raw||

After GS: cos(v_new_orth, v_i) = 0 for all i in {1,...,k}.

### 5.2 Efficacy cost of orthogonalization

The removed component has magnitude:

    ||removed||^2 = sum_i (v_new · v_i)^2 = sum_i cos(v_new, v_i)^2

At |cos| = 0.3 for each of k active vectors: ||removed||^2 = k * 0.09

For k=3, |cos|=0.3: ||removed|| ~ 0.52 * ||v_new|| — ~52% of the norm is
removed. If the removed component is purely interfering (not behavior-carrying),
efficacy is preserved at 100%. If the removed component partially carries the
new behavior (shared semantic content with existing behaviors), efficacy degrades.

The empirical test determines the efficacy-preservation rate.

### 5.3 N5 norm impact

Post-GS: ||v_new_orth|| <= ||v_new||. The joint norm of {v_1,...,v_k, v_new_orth}
is exactly sqrt(k+1) * alpha (Pythagorean, since all vectors are now orthogonal).
This is optimal from the N5 perspective — the smallest possible joint norm for
N+1 behaviors. The N5 law predicts:

    logPPL_orth < logPPL_raw_add   (lower joint norm → lower PPL degradation)

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| v_new_orth efficacy (vs solo v_new) | >= 95% | Core claim |
| Interference reduction (|cos|=0.2 initial) | >= 80% relative | Core claim |
| ||v_new_orth|| / ||v_new|| | ~ 0.98 at |cos|=0.2 | Pythagorean removal |
| Interference at |cos|=0.3 (before GS) | ~30% | From E18 curve |
| Interference at |cos|~0 (after GS) | < 5% | Guaranteed by GS |
| Joint PPL overhead (post-GS vs raw) | [−0.3, −0.1] logPPL | N5 norm reduction |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B. N5 (SUPPORTED) grounds
the PPL prediction.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Active set:** v_1 = refusal vector (established baseline)
- **New vector:** v_new = sentiment-positive or formality vector
- **Conditions:** (1) solo v_new, (2) raw joint v1 + v_new, (3) GS-orthogonalized
  v1 + v_new_orth
- **Metrics:** B1 (refusal) efficacy and B_new efficacy for each condition;
  interference index; PPL; offshell ||delta h||
- **Cosine pre-check:** log cos(v_new, v_1) before and after GS (should be
  ~0 after GS within floating-point precision)
- **Seeds:** 3 (screening); 7 if efficacy preservation is between 90-97%
- **Wall-clock:** ~1 h on 4090

### 7.2 Where it shines

Gram-Schmidt is most beneficial when the initial |cos(v_new, v_i)| is in
[0.2, 0.5] — enough interference to matter, but not so much that the
orthogonalized vector loses most of its behavior-carrying information.
At |cos| > 0.6, the removed component may be so large that v_new_orth
has insufficient norm to achieve meaningful behavior, and SAE-TS (E20)
or Conceptors (E21) should be used instead.

---

## 8. Cross-References

- **E17** (near-orthogonal stacking): GS is the preprocessing that ensures
  the |cos| < 0.2 condition for E17
- **E18** (interference vs Gram mass): GS reduces M to near-zero; E18 predicts
  near-zero interference post-GS
- **E20** (SAE-TS orthogonalization): SAE-TS achieves similar orthogonalization
  in feature space rather than activation space; comparison is E20's focus
- **E21** (Conceptors): Conceptor AND achieves a different form of composition
- **E47** (gate + ortho-stack + norm-cap): E19 is the "ortho-stack" component
  of the recommended composite stack
- **N5** (norm budget): GS reduces joint norm (optimal for N5 budget)
- **IDEA_TABLE.md** Block C row E19

---

## 9. Committee Q&A

**Q: Does Gram-Schmidt guarantee zero interference in behavior space,
not just in activation space?**

> No — GS guarantees cos(v_new_orth, v_i) = 0 in activation space. Whether
> this translates to zero behavioral interference depends on whether behaviors
> respond linearly to their activation direction being perturbed. If the
> behavior evaluation is nonlinear (e.g., a judge that uses holistic generation
> quality), residual interference may persist even at cos = 0. The experiment
> tests this empirically — the 80% interference reduction claim is the
> empirical target, not a geometric guarantee.

**Q: Is GS order-dependent?**

> Yes — GS removes the component of v_new along the EXISTING vectors; the
> existing vectors are unchanged. E26 tests whether injection ORDER matters;
> GS is an order-dependent preprocessing step. For a growing active set,
> the protocol should specify the order (e.g., add vectors in decreasing
> efficacy order so the highest-efficacy vector is added first and preserved
> at full strength).

---

## 10. Verification Checklist

- [ ] cos(v_new, v_1) before and after GS logged (expect ~0 after GS)
- [ ] Solo v_new efficacy established (baseline)
- [ ] Raw joint (v1 + v_new) interference measured
- [ ] GS joint (v1 + v_new_orth) interference measured; reduction >= 80%
- [ ] v_new_orth efficacy compared to solo v_new; >= 95%
- [ ] Offshell ||delta h|| before and after GS logged (N5 reduction)
- [ ] PPL before and after GS logged
- [ ] IDEA_TABLE.md row E19 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Geometry
  grounding: N5 (SUPPORTED) predicts lower joint norm post-GS; the formula
  ||delta_h||^2_orth = (k+1)*alpha^2 (orthogonal case) vs ||delta_h||^2_raw =
  (k+1+off-diag-sum)*alpha^2 (raw case) gives an exact norm reduction.
  Code needed: GS orthogonalization module (trivial linear algebra);
  multi-vector injection (same as E17). Blocked on E17 multi-vector injection.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C.*

### Prior plausibility

HIGH for interference reduction (guaranteed by construction). MEDIUM for
efficacy preservation (depends on semantic orthogonality between behaviors).
If behaviors share a common dimension (e.g., both safety-related), GS removes
a real behavior component and efficacy will drop below 95%.

### Mechanism scrutiny

The GS guarantee is exact algebra. The efficacy claim is empirical. The
falsifier is appropriately calibrated: >= 95% efficacy is the test of whether
the removed component was "purely interfering."

### Confounds

1. **Norm rescaling:** after GS, ||v_new_orth|| < ||v_new||; if the experiment
   runs at the same alpha, the effective steering strength is reduced. The
   comparison should use the same effective norm (rescale alpha after GS).
2. **Existing vector integrity:** GS does not change v_1; if v_1 has been
   previously GS'd against earlier vectors, the order of operations matters.

### Verdict

**NOVEL+TESTABLE** with strong geometric grounding. The efficacy-preservation
claim (>= 95%) is the critical empirical test; the interference reduction
(>= 80%) is algebraically expected. Low compute cost.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to Gram-Schmidt
orthogonalization of a new behavior vector against an active stack. Status:
`UNTESTED` — GS is trivial linear algebra but rides on E17's multi-vector injection.

### 1. Steering-vector recipe (DiffMean + Gram-Schmidt projection)

```python
# Each vector is a closed-form DiffMean (METHODOLOGY §1.3). Active set assumed orthonormal.
v1 = normalize(extract.build_vector_bank(model, tok, load_concept("refusal"), L)[L]["diffmean"])
v_new_raw = normalize(extract.build_vector_bank(model, tok, load_concept(B_new), L)[L]["diffmean"])

# Gram-Schmidt: remove the component of v_new lying in span{v_1..v_k}  (§5.1)
v_new_orth = v_new_raw
for vi in active_set:                       # {v_1, ..., v_k}
    v_new_orth = v_new_orth - dot(v_new_orth, vi) * vi   # project out
v_new_orth = v_new_orth / norm(v_new_orth)  # renormalize -> cos(v_new_orth, vi)=0 for all i
# removed-mass = sqrt(Σ_i cos(v_new, vi)^2); efficacy survives iff removed mass was interfering
```

### 2. Experiment procedure

```text
conditions = { solo(v_new),
               raw_joint( inject v1 + v_new ),               # raw add, leaves cross-cosine
               gs_joint( inject v1 + v_new_orth ) }          # orthogonalized
for cond in conditions:
  for seed in 1..3:
    inject combined vector via hooks.apply_operation(h, v_sum, "add"/"relative_add", alpha)
    log: efficacy_B1(refusal), efficacy_Bnew, interference_index, PPL,
         cos(v_new, v1) before/after GS  (~0 after, to float precision),
         geometry.offshell_displacement   # joint norm = sqrt(k+1)*alpha post-GS (Pythagorean, §5.3)
```

### 3. Measurement & decision rule

- PRIMARY metrics: (a) `v_new_orth` efficacy vs solo `v_new`; (b) interference
  reduction vs raw add.
- Pre-registered FALSIFIER (§3): efficacy `< 95%` of solo (GS removed behavior-carrying
  information) OR interference reduction `< 80%` ⇒ FALSIFIED. Either failure rejects.
- Secondary (§6): joint PPL overhead `[−0.3,−0.1]` logPPL vs raw (N5: lower joint norm).

### 4. Where the code is / status

The GS projection is a few lines (`extract.py`-style closed form, no backprop) and
DiffMean exists. The blocker is the same **multi-vector injection** orchestration
as E17. `UNTESTED`.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E19.md`.
