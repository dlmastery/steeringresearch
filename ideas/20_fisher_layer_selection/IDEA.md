# Formal claim for H20 (E2)

> Pre-registration contract. FROZEN once first experiment row is committed
> to EXPERIMENT_LEDGER.md.

## Claim

On Gemma-2-2B-it (4-bit), the per-layer Fisher linear discriminant ratio
between positive and negative contrast activations is a strong predictor of
steering efficacy at that layer: Spearman rank correlation between Fisher ratio
and measured behavior efficacy across all layers is >= 0.7 for each of at least
three distinct behaviors tested, establishing Fisher-ratio maximization as the
principled automatic layer-selection procedure.

## Falsifier

If Spearman(Fisher_ratio, efficacy) < 0.7 for ANY of the three tested behaviors
(refusal, sentiment, factuality) on Gemma-2-2B-it (4-bit), Rung-1 SMOKE,
evaluated across all residual-stream layers with alpha=0.8 and pinned pair count
(output of E1), this claim is DISCARDED. Layer selection must then fall back to
held-out behavior-score cross-validation per behavior.

## Pre-registered prediction

Primary metric: Spearman(Fisher_ratio, behavior_efficacy) across layers
Predicted value: >= 0.7 for all 3 behaviors [NEEDS VERIFICATION]

Sub-predictions:
- Optimal layer for refusal on Gemma-2-2B: in [14, 20] [NEEDS VERIFICATION]
  (based on Rogue Scalpel arXiv:2509.22067 F3 fragility analysis — fragility
  peaks early-mid, implying cleaner concept representation in upper-mid layers)
- Optimal layer is NOT layer 0 or the final layer (degenerate cases)
- Fisher ratio profile shows a clear maximum, not a flat plateau

Ablation control: random-permutation baseline shuffles positive/negative labels;
Fisher ratio should collapse to near zero; efficacy should not correlate with
the permuted ratio.

All thresholds are [NEEDS VERIFICATION] pre-registered predictions.

## Composite fingerprint at time of registration

SHA-256 of `src/steering/eval.py:COMPOSITE_FORMULA`: a9001e87087e (placeholder)
Note: primary metric is Spearman correlation, not composite. Composite is
reported for the best-layer configuration as a secondary metric.

## Signed off by

Author: autoresearch-registrar
Date: 2026-05-30
Git commit at registration: (not yet committed)
