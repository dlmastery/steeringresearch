# E32 — Refusal Direction vs Harmfulness-Detection Direction Are Separable

> **One-line claim:** The refusal behavior direction and the harmfulness-detection
> condition direction are distinct vectors (low cosine similarity across layers),
> establishing that detection and behavioral execution are separable subspaces
> within the residual stream.
>
> **Source design space:** Block D — Geometry and Rotational Methods (E27–E33).
>
> **Implementation status:** UNTESTED. No screening data as of 2026-05-31.

---

## In Plain English

**What we're testing, simply:** Refusing a harmful request really involves two
jobs: *noticing* that a request is harmful, and *deciding to say no*. The claim is
that these two jobs live in *different* directions inside the model — they're
separable, not one tangled thing.

**Key terms (defined here):**
- **Steering / steering vector:** nudging the model by adding a direction to its
  internal "thoughts"; the direction is the steering vector.
- **Residual stream:** the model's running internal "thoughts" that we read and
  edit.
- **Layer:** the model's stacked processing steps; we check this across many of
  them.
- **Alpha / strength:** how hard we push.
- **DiffMean:** the simple recipe for building a nudge.
- **Coherence:** whether the text stays fluent and sensible.
- **Refusal direction:** the internal direction tied to *saying no*.
- **Detection direction:** the internal direction tied to *noticing harm*.
- **Cosine similarity:** an overlap score between two directions, from 0 (no
  overlap, fully separate) to 1 (the same direction). A low score across layers
  would show the two jobs are separate.
- **Separable subspaces:** the idea that "notice harm" and "say no" occupy
  different regions of the thought space, so they can be studied and adjusted
  independently.

**Why we're doing this (the point):** If noticing harm and refusing are separate,
we could improve one without disturbing the other — for example, sharpen detection
without making the model refuse harmless things.

**What the result would mean:** A win (low overlap) means safety has two
independent knobs we can tune separately. A loss (high overlap) means they're
entangled and tweaking one inevitably drags the other along.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The CAST framework (arXiv:2409.05907) and its descendants operate by separating
two distinct operations: (1) detecting whether the input is harmful (the condition
check, which reads a condition vector from the prompt activations) and (2) injecting
the refusal behavior (the behavior injection, which adds a refusal direction to the
hidden state). This architectural separation is premised on an implicit assumption:
the direction that encodes "this input is harmful" is a DIFFERENT direction from
the direction that encodes "output a refusal response." If these two directions were
the same, there would be no need for a separate condition-vs-behavior distinction —
a single threshold on the refusal direction would suffice for both detection and
action. Conversely, if they are distinct, gating is both possible and necessary:
you can read the condition direction without triggering the behavioral direction,
and you can fire the behavioral direction independently of whether the condition
is met (which is the over-refusal failure mode). The Arditi et al. paper
(arXiv:2406.11717) identified a single "refusal direction" — but this is the
BEHAVIORAL direction (removing it suppresses refusal outputs). The DETECTION
direction is the direction in activation space that represents whether the model
has classified the input as harmful, which may reside in earlier layers and in a
different representational subspace. Our own data show (S-4) that the "Rogue-
Scalpel" direction — which we use as the refusal behavior vector — pushes
compliance rate from 0.80 to 1.00 (full rogue compliance) on Gemma-3-270M, while
the condition vectors (CAST-style) are computed from PCA of harmful vs. harmless
prompt activations. Whether these two vectors are cosine-similar or orthogonal is
the question E32 tests. The answer has direct safety implications: if they are
highly correlated (cos > 0.5), the model's detection machinery is entangled with
its execution machinery, making it harder to achieve independent calibration of
the two. If they are orthogonal (|cos| < 0.3), the two functions are cleanly
separable, the CAST architecture is well-motivated, and interventions can target
one without affecting the other. The Non-Identifiability paper (arXiv:2602.06801)
warns that behavioral equivalence classes are large — there exist many directions
that produce the same behavioral output, which means the "refusal direction" is
not unique. E32 must be careful to test the DETECTION-specific direction (derived
from harmful vs. harmless INPUT activations) against the BEHAVIOR-specific direction
(derived from refusal vs. compliance OUTPUT activations).

---

## 2. Formal hypothesis (>= 50 words)

The harmfulness-detection condition vector c (derived as DiffMean or PCA-top-1 of
harmful vs. harmless prompt activations at layer L_condition) and the refusal
behavior vector b (derived as DiffMean of refusal vs. compliance output activations
at layer L_behavior) satisfy |cos(c, b)| < 0.3 at all layers where both are
meaningfully defined. This low cosine must hold across at least 3 layers and at
least 2 model sizes (Gemma-3-270M and Gemma-3-1B). Furthermore, zeroing out the
condition direction from the behavior vector (b_ortho = b - dot(b, c_hat) * c_hat)
preserves behavioral efficacy within 10% while changing the detection signal.

---

## 3. Falsifier (>= 30 words)

If |cos(c, b)| >= 0.5 at the primary steering layer (L=16 for 270M, L=18 for 1B),
the detection and behavior directions are substantially aligned and the separability
claim is REJECTED. If the cosine is layer-dependent (low at some layers, high at
others), the claim is partially rejected: separability exists only at specific layers.
Status moves to x disproved if the cosine exceeds 0.5 at more than half the tested
layers.

---

## 4. Citations (Citation Rigor format, >= 80 words)

```
CAST: Cao et al. 2024 'Conditional Activation Steering: Steering Language Model
Behavior with Semantic Conditions' (arXiv:2409.05907) — the architecture that
E32's separability claim underlies; CAST assumes condition and behavior directions
are independently operable, which requires them to be non-collinear.

Arditi et al. 2024 'Refusal in Language Models is Mediated by a Single Direction'
(arXiv:2406.11717) — identifies a "refusal direction" in the residual stream;
E32 distinguishes between the behavioral refusal direction (Arditi) and the
harmfulness-detection direction (CAST condition), testing whether these are
the same or distinct directions.

Non-Identifiability of Steering Vectors: Venkatesh & Kurapath 2026 (arXiv:2602.06801)
— proves that large equivalence classes of behaviorally-indistinguishable vectors
exist; E32 must be interpreted carefully: "the" refusal direction is not unique,
and the detection direction is likewise not unique. The test is whether ANY
representative from each equivalence class has |cos| < 0.3.

FineSteer (SCS): Lyu et al. 2026 (arXiv:2604.15488) — energy-ratio gating as a
refinement of CAST; the SCS energy ratio is a more sophisticated version of the
condition check; E32's separability result predicts how much freedom SCS has in
choosing its gating subspace.

CAA: Turner et al. 2024 (arXiv:2308.10248) — baseline for extracting the behavior
direction (DiffMean on refusal vs compliance outputs); provides the b vector in E32.

Our own screening: S-4/S-7/S-8 — the Rogue-Scalpel direction (behavioral refusal
vector) has been used throughout; its cosine relationship to the CAST condition
vector has not yet been measured. E32 fills this gap.
```

---

## 5. Mechanism

### 5.1 Two-role decomposition of the refusal computation

The transformer processes a harmful prompt in (at least) two stages:
(a) DETECTION: at some early-to-mid layers, the model classifies the input as
    harmful. This is reflected in the activations of the prompt tokens — the
    condition direction c lives in the space of PROMPT-TOKEN activations.
(b) EXECUTION: at later layers, the model generates the refusal output. The
    behavior direction b lives in the space of RESPONSE-GENERATION activations
    (or in the transition from prompt encoding to generation start).

If (a) and (b) are implemented in distinct circuits — as would be the case if
the model uses an attention head to detect harm and a separate MLP to generate
refusal — the directions c and b should be orthogonal (they live in different
functional subspaces even at the same layer). This is the architecture that
motivates CAST's two-vector design.

If instead the model implements harm detection and refusal execution via the same
direction — e.g., the harm-detection representation IS the refusal trigger — then
c and b would be collinear. This would imply that CAST's two-vector framework is
redundant: a single direction suffices.

### 5.2 Layer-dependence prediction

Early layers (L < 8): the prompt is still being encoded; the condition direction
should be present (the model is building its harm representation) but the behavior
direction (which relates to generation) may not be. Expect |cos(c_L, b_L)| to
be low here simply because b is not yet meaningful at early layers.

Mid layers (L ≈ 10-16): both detection and behavior directions should be present.
This is the critical test layer.

Late layers (L > 20): the behavior direction dominates (generation is imminent);
the condition direction may have collapsed into the behavior direction. Expect
higher cosine at late layers if the model conflates detection and execution at
the generation stage.

---

## 6. Predicted Delta (pre-registered)

| Metric | Predicted value | Falsifier threshold |
|---|---|---|
| |cos(c, b)| at L=16 (270M) | < 0.3 | > 0.5 falsifies |
| |cos(c, b)| at L=18 (1B) | < 0.3 | > 0.5 falsifies |
| Layer profile of cosine | low early, possibly higher at late layers | monotone increase through all layers not predicted |
| Efficacy of b_ortho vs b | within 10% behavior success | > 20% drop falsifies separability utility |
| Condition direction alignment with harmful detection | measurable as AUC of cos(h_i, c) on harmful vs harmless prompts | AUC > 0.70 confirms c is meaningful |

---

## 7. Protocol

### 7.1 Primary experiment

Step 1 — Extract condition direction c:
  - Create 50+ pairs of (harmful prompt, harmless prompt) on the same topic
  - Extract activations h at layers L ∈ {8, 10, 12, 14, 16, 18, 20} for each prompt
  - Compute c_L = DiffMean(h_harmful - h_harmless) at each layer; also PCA-top-1

Step 2 — Extract behavior direction b:
  - Use existing Rogue-Scalpel DiffMean b from previous campaigns (C3/C9b)
  - Alternatively: extract from (refusal output, compliance output) pairs
    using DiffMean at the same layers

Step 3 — Compute cosine profile: cos(c_L, b_L) for L ∈ {8..20}

Step 4 — Compute b_ortho at L=16: b_ortho = b - dot(b, c_hat) * c_hat;
  run behavior sweep with b_ortho; compare efficacy to b.

Step 5 — Condition-direction validation: compute cos(h_i, c) for harmful vs
  harmless prompts; verify AUC > 0.70 (confirms c is actually encoding harm).

Wall-clock: ~3 hours on a 4090. n=3 seeds.

### 7.2 Where it shines

E32 is most informative as a safety architecture diagnostic: it tells us whether
the CAST-style two-vector design is necessary or whether a single direction suffices.
If |cos(c, b)| < 0.2 across all layers, the CAST design is well-grounded and the
condition + behavior pipeline is provably not redundant.

---

## 8. Cross-references

- E9 (CAST harmless refusal gate): E32's separability result explains WHY the gate
  can be calibrated independently of the behavior vector
- E10 (condition vector orthogonality across categories): extends E32 to multi-
  category case
- E6 (linear probe for over-steering): the condition direction c is the linear
  probe in E6
- N6 (gate in read not write): directly motivated by E32 — forcing cos(c, b) = 0
  is the "gate in read" operation; E32 tests whether this orthogonality is NATURAL
  or must be enforced
- Papers: CAST 2409.05907, Arditi 2406.11717, Non-ID 2602.06801

---

## 9. Committee Q&A

**Q: The Arditi paper already identified "the" refusal direction. Doesn't that
answer E32?**

> No. Arditi's refusal direction is the BEHAVIORAL direction (removal suppresses
> outputs). E32 asks whether the DETECTION direction (derived from harmful vs.
> harmless INPUTS) is the same or different. These are distinct extraction procedures.
> It is entirely possible that the behavioral direction and the detection direction
> are aligned (if the model implements detection and execution in the same subspace)
> or orthogonal (if they are separate circuits). E32 measures this empirically.

**Q: The Non-Identifiability paper says there are many equivalent directions.
How can E32 be meaningful?**

> E32 is meaningful because we are comparing TWO independently-extracted directions
> (one from input activations, one from output activations). Even if each direction
> has a large equivalence class, the cosine between the two canonical representatives
> (DiffMean in each case) is informative: if it is low, the two extraction processes
> are accessing different subspaces, which is the separability claim.

**Q: What if c and b are orthogonal but the CAST gate fires anyway on harmful
inputs — doesn't that require some overlap?**

> No. CAST gates on cos(h_prompt, c), not on cos(b, c). The gate fires when the
> prompt encoding is similar to the condition direction. Even if b ⊥ c, the gate
> can fire based on the prompt encoding. The two roles are: c = the recognition
> direction (applied to prompt tokens); b = the action direction (injected at
> generation start). They can be orthogonal and both work.

---

## 10. Verification checklist

- [ ] Harmful vs. harmless prompt pairs (>= 50) extracted and labeled
- [ ] Condition direction c_L extracted at layers L ∈ {8, 10, 12, 14, 16, 18, 20}
- [ ] Behavior direction b_L available (reuse from C3/C9b or re-extract)
- [ ] Cosine profile cos(c_L, b_L) computed and plotted across layers
- [ ] b_ortho computed at primary layer; efficacy sweep completed
- [ ] Condition direction validation: AUC > 0.70 on harmful vs harmless classification
- [ ] Results reported for >= 2 models (270M and 1B)
- [ ] Rows added to EXPERIMENT_LEDGER.md

---

## 11. Status journal

- 2026-05-31 — Created. UNTESTED. The separability assumption is implicit in our
  entire CAST-motivated experimental design but has never been measured. Priority:
  HIGH — this is a foundational assumption for the safety architecture. If c and b
  are collinear, several planned experiments (E9-E16, N6) need rethinking.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-D. Critiquing the idea, not the implementation.*

### Prior plausibility

**MEDIUM-HIGH.** The two-role (detection vs execution) architecture is a standard
assumption in the CAST literature and is mechanistically motivated by the attention-
vs-MLP specialization in transformers. Empirically, the fact that CAST works
(condition vectors can gate behavior vectors) is indirect evidence that they are
not exactly collinear — a perfectly collinear condition and behavior direction
would mean the gate always fires exactly when the behavior would have fired anyway,
providing no additional precision. The fact that CAST reduces over-refusal implies
the condition direction carries additional discriminative information not present
in the behavior direction.

### Mechanism scrutiny

The layer-dependence prediction (low cosine at early layers, possibly higher at
late layers) is the main mechanistic nuance. If the cosine is uniformly low across
all layers, the two-direction architecture is robustly supported. If the cosine is
high only at late layers (L > 20), this suggests the model conflates detection and
execution only at the decision point — which would mean the condition check should
be placed at early layers to ensure it captures detection, not execution. This
layer-dependence profile is practically actionable.

### Confounds

1. **Extraction confound:** DiffMean of harmful vs. harmless INPUTS gives the
   condition direction; DiffMean of refusal vs. compliance OUTPUTS gives the
   behavior direction. These use different contrast sets and may differ in
   quality/noisiness, making |cos| estimates noisy.
2. **Non-identifiability confound:** the equivalence classes of c and b may overlap
   even if the canonical representatives have low cosine. The test measures one
   pair of representatives, not the full equivalence class distance.
3. **Model-scale confound:** the 270M model may conflate detection and execution
   (limited capacity forces feature overlap), while larger models separate them.
   The hypothesis should predict higher |cos| for 270M than for 1B.

### Does it specifically matter?

**YES, for safety architecture.** If |cos(c, b)| > 0.5, the CAST two-vector design
is partially redundant and can be simplified to a single-vector threshold. If
|cos(c, b)| < 0.2, the two-vector design is necessary and the condition direction
carries unique discriminative information. This has direct implications for the
E47 "optimal combination" experiment.

### Literature precedent

No published paper directly compares the harmfulness-detection direction to the
refusal-behavior direction with cosine analysis. Arditi (2406.11717) and CAST
(2409.05907) both reference "a refusal direction" but from different extraction
procedures. E32 is filling a genuine empirical gap in the safety steering literature.

### Skeptical effect-size re-prediction

My prior: |cos(c, b)| ~ Uniform[0.10, 0.50] at the primary steering layer.
The claim |cos| < 0.3 is likely correct at early-to-mid layers (< L16) but
potentially violated at late layers (> L20) where detection and execution may
converge. A layer-conditional verdict is the most honest prediction.

### Minimum-distinguishing experiment

Extract c from 20 harmful vs harmless prompt pairs; use existing b from C3/C9b.
Compute cos(c, b) at L8, L12, L16, L20. If the profile is monotone increasing
(low early, high late), the separability claim holds for the condition layer but
not for the behavior layer. Cost: < 1 hour on existing infrastructure.

### Verdict

**HIGH PRIORITY AND TESTABLE. This is a foundational safety architecture
assumption that has not been directly measured. The experiment is cheap, uses
existing infrastructure, and the result is actionable regardless of which way it
goes.**

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to measuring whether
the harmfulness-DETECTION direction and the refusal-BEHAVIOR direction are separable.
Status: `UNTESTED`. GEOMETRY hypothesis: two independently-extracted DiffMean
directions and their cross-layer cosine — no injection of a combined vector needed for
the core test.

### 1. Steering-vector recipe (two DiffMean directions from different contrasts)

```python
# Condition direction c — DiffMean of harmful vs harmless INPUT activations (METHODOLOGY §1.3):
c_L = normalize(extract.diffmean_vector(h_harmful_prompts[L], h_harmless_prompts[L]))   # per layer L
# Behavior direction b — DiffMean of refusal vs compliance OUTPUT activations (reuse Rogue-Scalpel):
b_L = normalize(extract.build_vector_bank(model, tok, refusal_vs_compliance, L)[L]["diffmean"])
# Separability probe: cosine across layers, plus orthogonalized behavior vector:
cos_profile = { L: dot(c_L, b_L) for L in {8,10,12,14,16,18,20} }
b_ortho = normalize(b_L - dot(b_L, c_L) * c_L)     # project the condition direction OUT of b
```

### 2. Experiment procedure

```text
Step 1: extract c_L at L in {8,10,12,14,16,18,20} (50+ harmful/harmless pairs), DiffMean + PCA-top1.
Step 2: extract / reuse b_L (Rogue-Scalpel) at the same layers.
Step 3: cos_profile = cos(c_L, b_L) across layers; plot.
Step 4: at L=16 build b_ortho; inject via hooks.apply_operation(h, b_ortho, "add"/"relative_add", alpha);
        compare behavior efficacy of b_ortho vs b.
Step 5: validate c is meaningful: AUC of cos(h_i, c) separating harmful vs harmless prompts.
report for models {270M, 1B}, n=3 seeds.
```

### 3. Measurement & decision rule

- PRIMARY metric: |cos(c, b)| at the primary steering layer (L=16 for 270M, L=18 for 1B).
- Pre-registered FALSIFIER (§3): |cos(c,b)| `>= 0.5` at the primary layer ⇒ separability
  REJECTED; if `>0.5` at more than half the tested layers ⇒ disproved. Partial rejection
  if separability holds only at some layers.
- Utility check (§6): b_ortho efficacy within 10% of b (>20% drop falsifies the utility);
  condition-direction AUC `> 0.70` confirms c encodes harm.

### 4. Where the code is / status

`extract.diffmean_vector` / `pca_top1_vector` exist; the cosine profile, `b_ortho`
projection, and AUC computation are a few lines. Behavior efficacy of `b_ortho` uses
the standard `add`/`relative_add` path. MISSING only: the harmful/harmless prompt-pair
set and the layerwise extraction loop. `UNTESTED`.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E32.md`.
