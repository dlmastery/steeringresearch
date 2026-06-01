# E30 — Adaptive Rotation on Partially-Aligned Tokens

> **One-line claim:** Rotating only the subset of token hidden states that are
> partially aligned with the behavior direction (neither already aligned nor fully
> opposed) preserves coherence better than rotating all tokens uniformly, at matched
> aggregate behavior success.
>
> **Source design space:** Block D — Geometry and Rotational Methods (E27–E33).
>
> **Implementation status:** UNTESTED. No screening data as of 2026-05-31.

---

## 1. Motivation (>= 100 words)

Standard residual-stream steering applies the same intervention to every token
position in the forward pass: h'_i = h_i + alpha * v for all i ∈ {1 .. T}. This
is a blunt instrument. The behavior direction v is relevant at some token positions
— those where the model is actively computing behavior-relevant information — and
irrelevant or actively harmful at others — those where the model is computing
syntax, referencing factual knowledge, or tracking discourse structure. Applying
a large rotation to a token position that has nothing to do with the behavior in
question is purely destructive: it distorts the representation of that token
without any compensating behavioral benefit. The Selective Steering paper
(arXiv:2601.19375) already restricts which LAYERS to apply the rotation to
(discriminative layers, those where class means have opposite signs). E30 asks a
complementary question: within a single chosen layer, should we rotate ALL token
positions or only those where the hidden state h_i is already partially aligned
with the behavior direction? The "partially aligned" criterion is defined by the
cosine similarity between h_i and v: tokens with |cos(h_i, v)| < threshold are
not engaging the behavior direction at all (rotating them applies maximum angular
damage for zero behavioral return), while tokens with cos(h_i, v) > high_threshold
are already aligned (rotating them further may cause overshoot). The optimal
target is the "partially-aligned" window: tokens where cos(h_i, v) is in some
range [low, high] — engaged with the direction but not yet fully committed. This
is analogous to the CAST framework (arXiv:2409.05907), which gates an entire
intervention on whether the input is semantically similar to the target concept;
E30 applies the same gating principle at the level of individual token positions
within a single forward pass, using the current hidden state as the gate signal.
The CRH framework (arXiv:2605.01844) makes this precise: the angular displacement
of a rotation is the coherence cost; by skipping fully-aligned and fully-opposed
tokens, we minimize the average angular displacement across positions.

---

## 2. Formal hypothesis (>= 50 words)

For a fixed rotation operation applied at layer L, applying the rotation only to
token positions where cos(h_i / ||h_i||, v / ||v||) ∈ [-0.3, 0.7] (the
"partial alignment window") will achieve behavior success within 10% of all-token
rotation while producing strictly lower perplexity — predicted >= 15% lower PPL
at matched behavior. The optimal partial-alignment window boundaries [low, high]
are model-dependent but should be identifiable from the distribution of
cos(h_i, v) across a held-in training set. This claim must hold on at least two
models (Gemma-3-270M and Gemma-3-1B) and one behavioral trait.

---

## 3. Falsifier (>= 30 words)

If adaptive rotation (partial-alignment gate) produces PPL > 90% of all-token
rotation PPL at matched behavior success on both tested models, the coherence
advantage is negligible and the gating is unnecessary: E30 is DISCARDED, and the
recommendation is to apply all-token rotation without gating. Specifically: if
the PPL ratio (adaptive / all-token) > 0.90 at behavior success within 5%, the
claim is rejected.

---

## 4. Citations (Citation Rigor format, >= 80 words)

```
CAST: Cao et al. 2024 'Conditional Activation Steering: Steering Language Model
Behavior with Semantic Conditions' (arXiv:2409.05907) — the primary conceptual
ancestor: gates a behavior vector on input-level cosine similarity to a condition
direction. E30 applies the same gating logic to token positions within a forward
pass rather than to whole inputs.

Selective Steering: Bai et al. 2026 'Selective Steering: Norm-Preserving 2D-Plane
Rotation for Large Language Models' (arXiv:2601.19375) — restricts rotation to
discriminative layers (layer-level selection); E30 asks the same question at the
token position level within a layer.

Angular Steering: Liu et al. 2025 (arXiv:2510.26243) — the base rotation method
that E30 adapts; E30 adds a per-token gate to the Angular Steering framework.

Cylindrical Representation Hypothesis: Gao et al. 2026 (arXiv:2605.01844) —
angular displacement = coherence cost for rotation; gating tokens to minimize
average angular displacement is the mechanism behind E30's efficiency claim.

SpotLight: Voss et al. 2026 'SpotLight: Attention-Based Token Selection for
Targeted Language Model Steering' (arXiv:2505.12025) — uses token-level attention
mass to select which tokens to intervene on; E30 uses cos(h_i, v) for the same
purpose in the residual stream.

FineSteer (SCS): Lyu et al. 2026 'FineSteer: Subspace-Guided Conditional Steering
for Targeted Behavior Control' (arXiv:2604.15488) — energy-ratio gating as a
refinement of CAST; E30 borrows the energy-ratio concept for within-pass token
selection.
```

---

## 5. Mechanism

### 5.1 The partial-alignment gate

For each token position i at layer L, compute the cosine similarity:
c_i = dot(h_i / ||h_i||, v / ||v||)

Apply the rotation R(theta) to h_i only if c_i ∈ [low_threshold, high_threshold]:
- Tokens with c_i < low_threshold (anti-aligned): rotating these applies large
  angular displacement (theta passes through a large arc before reaching v) for
  near-zero behavior return (these tokens are suppressing the behavior).
  Skipping them avoids the collateral angular damage.
- Tokens with c_i > high_threshold (already aligned): rotating these further
  toward v constitutes overshoot. Small additional rotation has diminishing
  behavior return and may cause the model to "lose" the token's natural content.
  Skipping these avoids overshoot.
- Tokens in [low, high] (partially aligned): these are the tokens where applying
  the rotation moves them from partial to full alignment — maximum behavioral
  return per unit of angular displacement.

### 5.2 Expected window boundaries

Based on the angular-displacement law from C3b (log PPL = 4.57 + 43.1 * (1-cos),
R2=0.997), the coherence cost is a sensitive function of the rotation angle. For a
token at c_i = 0 (orthogonal to v), the rotation by theta = 0.1 rad produces
angular displacement (1-cos 0.1) = 0.005, a small cost. For a token at c_i =
-0.8 (anti-aligned, 143 degrees from v), rotating by 0.1 rad still leaves it at
~140 degrees — no behavioral benefit and a 0.005 cost. The marginal behavior
return is zero; all angular displacement is waste. The gating rule therefore has
a clear mechanism: eliminate the zero-return, non-zero-cost rotations.

A preliminary prior for the boundaries: low = -0.2 to 0.0 (skip anti-aligned),
high = 0.7 to 0.9 (skip already-aligned). These are to be confirmed empirically.

---

## 6. Predicted Delta (pre-registered)

| Metric | Adaptive rotation (partial gate) | All-token rotation |
|---|---|---|
| PPL at matched behavior ~0.55 | < 115 | ~ 131 (+42% at theta=0.1 rad, C3b) |
| Improvement over all-token | >= 15% PPL reduction | baseline |
| Behavior success | within 10% of all-token | baseline |
| Fraction of tokens rotated | 30-60% of positions | 100% |
| Angular displacement (mean) | lower than all-token | full theta * T positions |

---

## 7. Protocol

### 7.1 Primary experiment

Models: Gemma-3-270M @L16 and Gemma-3-1B @L18. Direction: Rogue-Scalpel
DiffMean. Operation: full-vector rotation (same as C3b, to keep comparisons
consistent; selective-plane rotation queued for E27-A follow-up).
Sweep: for each run, compute c_i = cos(h_i, v) for all token positions; apply
rotation R(theta=0.1 rad) only to positions where c_i ∈ [low, high]. Grid-search
low ∈ {-0.5, -0.2, 0.0} and high ∈ {0.5, 0.7, 0.9}. Record behavior, PPL, fraction
of positions rotated, mean angular displacement of rotated positions, composite.
Compare to all-token rotation at the same theta.
Wall-clock: ~2 hours per model per grid point; total ~12 hours for the grid.

### 7.2 Where it shines

E30 is expected to shine on long-context generations where many token positions
are irrelevant to the behavior (discourse connectors, factual references, syntax).
A generation of T=512 tokens with behavior-relevant tokens at positions 20-30%
of the sequence means 70-80% of rotations are wasted under all-token steering.
Adaptive rotation reclaims that wasted capacity.

---

## 8. Cross-references

- E13 (early condition layer): token-level gating within a layer vs layer-level
  gating in E13; the mechanisms are analogous
- E27-A (selective rotation): E30 assumes full-vector rotation initially; after
  E27-A confirms selective-plane rotation, the gate should be applied to selective
  rotation for a combined experiment
- E31 (norm preservation): adaptive rotation preserves norm on rotated tokens;
  skipped tokens are untouched (norm unchanged trivially)
- Corpus: CAST 2409.05907, SpotLight 2505.12025, Selective Steering 2601.19375
- N16 (CRH): the angular-displacement cost is the theoretical basis for the gate

---

## 9. Committee Q&A

**Q: Isn't CAST already doing this, just at the input level instead of token level?**

> CAST gates the ENTIRE intervention based on whether the INPUT is similar to the
> condition. E30 gates individual token POSITIONS based on their current hidden state
> alignment. These are complementary: CAST decides whether to steer; E30 decides
> which tokens to steer within a steered forward pass. Both can be active
> simultaneously (CAST fires, then E30 selects which positions to rotate).

**Q: How expensive is the per-token cosine similarity computation?**

> One matrix-vector dot product per layer: h_L @ v, O(T*d) operations. This is
> negligible compared to the attention computation O(T^2*d) and the MLP O(T*d^2).
> The gate adds < 1% overhead.

**Q: What if the optimal window boundaries vary by prompt?**

> Then the fixed window is a heuristic approximation. The correct version would use
> a per-prompt, per-position dynamic threshold — closer to the CAST energy-ratio
> gating (FineSteer). E30 tests the fixed-window version first as a lower bound on
> the adaptive version.

---

## 10. Verification checklist

- [ ] Token-level cosine gate implemented in steering_ops.py
- [ ] Unit test: gate correctly identifies partial-alignment positions on synthetic data
- [ ] C3b all-token rotation reproduced as baseline at n>=3
- [ ] Adaptive-rotation grid search on 270M @L16, n>=3
- [ ] Adaptive-rotation sweep on 1B @L18, n>=3
- [ ] Fraction of rotated tokens logged per run
- [ ] Mean angular displacement of rotated vs skipped tokens logged
- [ ] Rows added to EXPERIMENT_LEDGER.md

---

## 11. Status journal

- 2026-05-31 — Created. UNTESTED. The mechanism is grounded in the C3b angular-
  displacement law and the CAST/SpotLight token-selection literature. Priority:
  lower than E27-A and E29 (depends on rotation baseline); implement after C3c
  (selective rotation) confirms a rotation baseline to gate on.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-D. Critiquing the idea, not the implementation.*

### Prior plausibility

**MEDIUM.** The token-level gating logic is mechanistically sound: skipping tokens
with zero behavioral return avoids wasted angular displacement. The concern is that
the set of "behavior-relevant" tokens is not well-defined by cos(h_i, v) alone:
a token may have low cosine similarity to v but still be causally upstream of
behavior-relevant tokens. The gate misses second-order effects.

### Mechanism scrutiny

The gating rule (cos(h_i, v) ∈ [low, high]) is a zero-order approximation to the
optimal per-token intervention policy. The correct policy would weight tokens by
their causal contribution to the behavioral output — which requires attribution
(gradient-based or causal) information. Cosine similarity to v is a proxy for
direct alignment, not for causal contribution. A token that is orthogonal to v
today may become aligned after the next attention layer; rotating it now would
help despite the zero current cosine. The gate therefore has a one-step look-ahead
problem.

### Confounds

1. **Window-boundary confound:** the optimal [low, high] window is data- and
   model-dependent; grid-searching it on the same data used to evaluate creates
   a leakage confound. Must use a held-out evaluation set.
2. **Token-distribution shift confound:** the distribution of cos(h_i, v) changes
   with the prompt; a fixed window trained on one prompt distribution may not
   generalize.
3. **Causally-upstream confound:** skipping anti-aligned tokens may miss tokens
   that are anti-aligned but causally important (e.g. a pronoun that the behavior
   direction needs to affect downstream via attention).

### Does it specifically matter?

**LOW-to-MEDIUM.** The computational saving (skip 40-70% of token rotations) is
negligible since the rotation itself is cheap relative to attention. The coherence
saving is the real motivation, but the mechanism has the second-order causality
problem described above. SpotLight (2505.12025) does a more principled version of
this using attention mass rather than residual cosine; E30's approach is a simpler
and potentially weaker version.

### Literature precedent

SpotLight (2505.12025) is the closest published method. If SpotLight already shows
that attention-mass-based token selection improves coherence, E30 is a weaker
version of the same idea applied to the residual stream. The contribution is the
application of the cosine-similarity gate to rotation rather than to attention
reweighting.

### Skeptical effect-size re-prediction

PPL improvement from adaptive gating: expect 5-15% (vs claimed >= 15%). The
strongest effect is at large theta (aggressive rotation) where wasted rotations
dominate. At moderate theta (0.1 rad, our current test), the effect may be below
the noise floor at n=3.

### Minimum-distinguishing experiment

At theta=0.2 rad (where all-token rotation degrades PPL to 255 from 92 baseline,
C3b data), apply adaptive gating with low=-0.2, high=0.7. If PPL drops below 150
while behavior success stays above 0.45, the gating provides meaningful coherence
preservation at aggressive rotation. Cost: <1 hour on existing infrastructure.

### Verdict

**PLAUSIBLE BUT SECONDARY. The causal-upstream problem reduces the expected
effect size. SpotLight provides a more principled version. Test it as a
low-cost ablation after E27-A (selective rotation) is established; do not
prioritize over E29 (SLERP) or E31 (norm preservation).**

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E30.md`.
