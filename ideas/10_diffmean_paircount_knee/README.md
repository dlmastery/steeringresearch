# H10 — DiffMean pair-count knee (E1)

> Back to master registry: [IDEA_TABLE.md](../../IDEA_TABLE.md)

## TL;DR

DiffMean steering vectors are mean estimators whose variance decreases as 1/N.
This idea locates the pair-count knee: the smallest N at which behavior-shift
efficacy reaches >=90% of its asymptotic value, beyond which additional pairs
give diminishing returns. Finding this knee sets the default extraction budget
for all downstream experiments, making the whole program faster without
sacrificing vector quality.

## Citation

Rimsky, Nick, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and
Alexander Turner. 2023. NeurIPS 'Steering Llama 2 via Contrastive Activation
Addition' (arXiv:2312.06681) — establishes the CAA/DiffMean framework and
reports empirical pair-count behavior on Llama 2; the knee is not explicitly
characterized but signal-to-noise intuition motivates the range [16, 64].

Note: the pair-count sweep as an explicit experiment is designed in the
autoresearch corpus (`corpus/50-steering-experiments-autoresearch.md`, E1).

## Hypothesis

DiffMean is the maximum-likelihood estimator of the class-mean difference under
isotropic Gaussian assumptions. Its estimation error decreases as 1/sqrt(N),
so the behavior-shift efficacy (measured on a fixed held-out eval set) should
show a characteristic knee: rapid improvement for small N and diminishing
returns past some threshold. On Gemma-2-2B-it with refusal/safety behaviors,
the 12-axis framework (Axis 7: HOW DERIVED) predicts this knee occurs at a
moderate pair count because the behavior direction is relatively high-SNR in
the residual stream. We predict the knee is at N in [16, 64] based on the
signal-to-noise intuition from the CAA paper, meaning N=50 is a conservative
but not extravagant default. More pairs give >=90% of asymptote. This claim
tests Axis 7 (HOW DERIVED) and informs the default extraction budget used by
every subsequent experiment (E2-E50, N1-N20).

## Falsifier

If behavior success rate at N=50 pairs is less than 90% of the N=256 asymptote
on Gemma-2-2B-it (4-bit) refusal behavior, evaluated on the AxBench-mini held-out
set (50 prompts, Rung-1 SMOKE), this idea is DISCARDED and the default extraction
budget must be set higher (>=100 pairs) with a logged justification.

## Predicted delta range

All thresholds are [NEEDS VERIFICATION] pre-registered predictions.

| metric | value | rationale |
|--------|-------|-----------|
| knee location | N in [16, 64] | CAA paper signal-to-noise intuition [NEEDS VERIFICATION] |
| asymptote fraction at knee | >=90% | by definition of "knee" |
| behavior score at N=256 (asymptote) | >0.70 on AxBench-mini | prior art baseline [NEEDS VERIFICATION] |
| MMLU drop at sweep alpha | <2 pp | extraction does not affect capability |
| PPL at test alpha | <15 | refusal steering is coherent at moderate alpha |

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

Sweep N_pairs in {4, 8, 16, 32, 64, 128, 256} on Gemma-3-1B-it (smoke default)
at a fixed injection layer (use layer 10 as proxy; refined by E2), alpha=0.8,
AxBench-mini held-out set (50 prompts, seed=42 pinned). Measure behavior success
rate at each N. Fit a saturation curve (Michaelis-Menten or logistic) to locate
the inflection / knee. Pre-register the predicted knee range [16, 64] BEFORE
running. Confirm at Rung-1 SMOKE on Gemma-2-2B-it with same params.

The "random-pair baseline" control draws N pairs randomly (not contrast-matched)
to verify that the knee is driven by contrast information, not by N alone.

## Cross-idea interactions

Composable with:
- E2 (20_fisher_layer_selection) — E2 fixes the optimal layer; E1 fixes the
  pair count; both inform the default extraction config for ALL downstream ideas.
- E4 (cosine alignment) — E4 uses the same extraction pipeline; knee from E1
  sets the N used there.

Competes with: none (E1 is pure tooling, not a behavioral method).

Combination suite: E1 + E2 + E4 form the standing extraction infrastructure
(Block A tooling) that every later idea depends on.

See also: `skills/steering-vector-extraction/SKILL.md` for the full protocol,
Fisher ratio computation, and activation caching discipline.
