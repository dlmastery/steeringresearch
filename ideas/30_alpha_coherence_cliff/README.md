# H30 — Alpha coherence cliff (E3)

> Back to master registry: [IDEA_TABLE.md](../../IDEA_TABLE.md)

## TL;DR

Steering coefficient alpha controls behavior strength but also coherence. This
idea characterizes the behavior-specific "coherence cliff": the alpha value
below which capability is preserved (MMLU drop < 2pp) and above which perplexity
rises super-linearly, indicating the activation has been pushed off the data
manifold. Locating this cliff per behavior establishes a safe alpha budget for
every downstream experiment and provides the empirical foundation for the N5
norm-budget conservation law.

## Citation

Panickssery, Aryan, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and
Alexander Turner. 2023. 'Steering Llama 2 via Contrastive Activation Addition'
(arXiv:2312.06681) — CAA paper; alpha sweep and coherence degradation are
discussed but the cliff shape is not formally characterized. The super-linearity
claim and MMLU < 2pp threshold are pre-registered predictions for this project.

Additional motivation: the high-dimensional geometry analysis in
`corpus/steering-missed-dimensions-and-highdim-algebra.md` (F-A, F-C) predicts
that off-manifold displacement (Axis 8: GEOMETRY) causes coherence collapse when
the chord displacement exceeds the manifold's radius of curvature. The cliff is
the empirical signature of this crossing. Hypotheses N5 (norm-budget conservation)
and N11 (curvature-aware per-prompt alpha) both depend on E3 locating the cliff.

## Hypothesis

Axis 3 (HOW MUCH / coefficient alpha) is the axis under test. The residual
stream of Gemma-2-2B-it has a characteristic scale set by its activation norm
||h||. When alpha*||v|| is small relative to ||h||, the intervention is a small
perturbation that the model can "absorb" while staying approximately on the
data manifold (Manifold Steering, arXiv:2605.05115). As alpha grows, the
cumulative off-manifold displacement (Axis 8: GEOMETRY; F-A: concentration of
measure) eventually crosses a threshold where the distribution of h_{l+1}
departs from the training distribution, causing perplexity to rise. This
departure is super-linear because once the model is off-manifold, each
additional layer amplifies the mismatch rather than correcting it (the residual
stream cannot restore manifold proximity in finite depth). Below the cliff,
capability (MMLU) is preserved because the representation is still semantically
coherent; above it, capability degrades because the model is generating from an
out-of-distribution hidden state. The cliff location is behavior-specific because
different behavior directions have different norms and different proximity to
coherence-critical subspaces. Locating the cliff per behavior sets the maximum
safe alpha for that behavior and informs the norm-budget for N5.

## Falsifier

If no alpha value in the sweep [0.1, 0.5, 1.0, 2.0, 4.0, 8.0] produces both
(a) MMLU drop < 2pp AND (b) any subsequent alpha produces super-linear PPL rise
on Gemma-2-2B-it (4-bit) refusal behavior at the Fisher-optimal layer (E2 output),
then the cliff hypothesis is DISCARDED for this behavior and alpha must be chosen
by cross-validation on PPL alone.

## Predicted delta range

All thresholds are [NEEDS VERIFICATION] pre-registered predictions.

| metric | value | rationale |
|--------|-------|-----------|
| MMLU drop below cliff | < 2 pp | pre-registered safety threshold [NEEDS VERIFICATION] |
| PPL below cliff | within 1.5x baseline | coherent generation expected [NEEDS VERIFICATION] |
| PPL above cliff | super-linear rise (d^2PPL/dalpha^2 > 0) | off-manifold amplification [NEEDS VERIFICATION] |
| Cliff alpha for refusal (Gemma-2-2B) | in [1.0, 4.0] | typical range from CAA observations [NEEDS VERIFICATION] |
| Control (zero-vector) | PPL unchanged across alpha sweep | no effect without a real vector |

## Status

| stage | done? |
|-------|-------|
| implementation written | [ ] |
| tests green | [ ] |
| AUDIT.md filed | [ ] |
| IMPROVEMENTS.md addressed | [ ] |
| VERIFY.md sealed | [ ] |
| First experiment archived | [ ] |
| Verdict authored | [ ] |

## How to test this idea (idea-specific experiment strategy)

1. Use the Fisher-optimal layer (output of E2) and default pair count (output of E1).
2. Sweep alpha in {0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0} on Gemma-3-1B-it (smoke).
3. At each alpha: measure MMLU (20 questions, pinned subset), PPL (WikiText-103
   first 500 tokens, pinned seed), behavior success rate (AxBench-mini held-out).
4. Also measure off-shell displacement delta_norm and effective-rank drop at
   each alpha (geometry leading indicators).
5. Fit a piecewise linear or quadratic model to PPL vs alpha; locate the
   inflection point (d^2PPL/dalpha^2 changes sign) as the cliff.
6. Verify super-linearity above the cliff: PPL(2*alpha_cliff) > 2*PPL(alpha_cliff).
7. Control arm: repeat with zero vector (alpha sweep on zero vector should be flat).
8. Confirm at Rung-1 SMOKE on Gemma-2-2B-it.

Pre-register the predicted cliff range [1.0, 4.0] for refusal BEFORE running.

## Cross-idea interactions

Composable with:
- E1 (pair-count knee) — provides the N; E3 uses fixed N from E1.
- E2 (Fisher layer) — provides the layer; E3 sweeps alpha at that layer.
- E7 (norm-relative alpha) — E3 locates the absolute cliff; E7 tests whether
  normalizing by ||h|| makes the cliff location more consistent across prompts.
- N5 (norm-budget conservation) — E3's cliff data is the primary input for
  fitting the master PPL vs ||delta h||/||h|| collapse curve.
- N11 (curvature-aware per-prompt alpha) — E3 provides the population-level cliff;
  N11 extends it to per-prompt prediction using local curvature.

Competes with: none (infrastructure measurement).

Combination suite: E1 + E2 + E3 = standing extraction + alpha calibration
infrastructure that every method must inherit.
