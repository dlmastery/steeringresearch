# Formal claim for H10 (E1)

> Pre-registration contract. FROZEN once first experiment row is committed
> to EXPERIMENT_LEDGER.md.

## Claim

On Gemma-2-2B-it (4-bit), DiffMean steering vectors extracted from >= 50
contrast pairs reach >= 90% of the asymptotic behavior-shift effect measured
on a fixed held-out set; extracting more pairs beyond the knee yields
diminishing returns and the pair-count knee is located at N in [16, 64].

## Falsifier

If behavior success rate at N=50 pairs is < 90% of the N=256 (asymptote)
value on Gemma-2-2B-it (4-bit) refusal behavior evaluated on the AxBench-mini
held-out set (50 prompts, seed=42, Rung-1 SMOKE), this claim is DISCARDED.
The program must then re-establish a higher default pair budget before
proceeding to E2.

## Pre-registered prediction

Primary metric: behavior success rate on AxBench-mini held-out set
Predicted knee: N in [16, 64] [NEEDS VERIFICATION]
Predicted asymptote fraction at knee: >= 90% [NEEDS VERIFICATION]
Predicted asymptote value (N=256): > 0.70 behavior success rate [NEEDS VERIFICATION]

Sub-metrics:
- MMLU drop: 0 pp (extraction sweep does not apply intervention; capability unchanged)
- PPL: not applicable at extraction stage (only measured at apply step)
- JailbreakBench CR: not applicable at extraction stage
- Over-refusal: not applicable at extraction stage

Geometry:
- Off-shell displacement: not measured at extraction stage
- Norm budget: not applicable

Control condition: random-pair baseline (N random unpaired prompts, no contrast
structure) should NOT show a knee — behavior success should stay near chance.

Citation anchor: CAA paper (arXiv:2312.06681) — knee intuition from the
signal-to-noise analysis of DiffMean estimation error. The specific range
[16, 64] is a pre-registered prediction, NOT a value from the paper.

All thresholds are [NEEDS VERIFICATION] pre-registered predictions.

## Composite fingerprint at time of registration

SHA-256 of `src/steering/eval.py:COMPOSITE_FORMULA`: a9001e87087e (placeholder)
Note: the composite is not the primary metric here (this is an extraction
calibration experiment). The primary metric is the raw behavior success rate
vs N_pairs curve.

## Signed off by

Author: autoresearch-registrar
Date: 2026-05-30
Git commit at registration: (not yet committed)
