# Formal claim for H30 (E3)

> Pre-registration contract. FROZEN once first experiment row is committed
> to EXPERIMENT_LEDGER.md.

## Claim

Steering coefficient alpha has a behavior-specific coherence cliff on
Gemma-2-2B-it (4-bit): below the cliff, MMLU drop is < 2 percentage points
(capability preserved); above the cliff, perplexity rises super-linearly
(d^2 PPL / d alpha^2 > 0), indicating displacement from the data manifold.
The cliff is identifiable from the alpha sweep {0.1, 0.25, 0.5, 1.0, 2.0,
4.0, 8.0} and is behavior-specific, not a universal constant.

## Falsifier

If no pair of consecutive alpha values in the sweep satisfies simultaneously:
  (a) lower alpha produces MMLU drop < 2 pp on Gemma-2-2B-it refusal, AND
  (b) higher alpha produces PPL that is super-linear (d^2PPL/dalpha^2 > 0,
      estimated by finite difference),
evaluated at the Fisher-optimal layer (E2 output), AxBench-mini held-out set
(seed=42), then this claim is DISCARDED for the refusal behavior. The program
must then determine safe alpha by PPL cross-validation without a cliff model.

## Pre-registered prediction

Primary metric: PPL vs alpha curve; cliff location and super-linearity above it
Predicted cliff alpha for refusal (Gemma-2-2B): in [1.0, 4.0] [NEEDS VERIFICATION]

Sub-predictions:
- MMLU drop at alpha <= cliff: < 2 pp [NEEDS VERIFICATION]
- PPL at alpha <= cliff: within 1.5x of unsteered baseline [NEEDS VERIFICATION]
- PPL at alpha = 2*cliff: > 2x PPL at cliff (super-linearity signature) [NEEDS VERIFICATION]
- Off-shell displacement delta_norm: monotone increasing in alpha [NEEDS VERIFICATION]
- Control (zero-vector sweep): PPL flat across all alpha values (within noise)

Geometry leading indicators (always log):
- delta_norm at each alpha
- eff_rank_drop at each alpha
- norm_budget fraction (||delta h||/||h||) at each alpha

All thresholds are [NEEDS VERIFICATION] pre-registered predictions.

## Composite fingerprint at time of registration

SHA-256 of `src/steering/eval.py:COMPOSITE_FORMULA`: a9001e87087e (placeholder)
Note: primary metric is the PPL vs alpha curve shape and cliff location.
Composite is reported at the cliff alpha as the recommended operating point.

## Signed off by

Author: autoresearch-registrar
Date: 2026-05-30
Git commit at registration: (not yet committed)
