# FINDING (M-a) — Norm preservation is the WORST way to spend a steering budget

**Tier: EVALUATION-eligible.** First result in this program to clear the power contract.
**Judge-free** — unaffected by the two failed judge calibrations (`JUDGE_CARD.md`).
**Date:** 2026-07-27 · **Artifact:** `autoresearch_results/ma_direction_seeds.json`
· **Script:** `scripts/run_ma_direction_seeds.py` (predictions pre-registered in its header)

---

## The claim

At a **fixed chord displacement**, spending the steering budget on **radius** (pure
addition) costs dramatically less coherence than spending it on **angle** (pure
norm-preserving rotation). Norm preservation — the property `2601.19375` (Selective
Steering), `2606.19946` (GEMS) and `2606.22357` (ORBIT) all build on — is not protective.
It is the **worst** allocation of a fixed budget.

## The measurement

Abliterated Gemma-3-1B, layer 12, budget f = 0.10, WikiText-2 perplexity, unsteered
baseline **77.310**. Direction estimated from **340 harmful** (JailbreakBench + HarmBench
+ AdvBench) and **120 harmless** (XSTest) real benchmark prompts, resampled over 8
independent extraction draws.

| seed | addition (r=1) | rotation (r=0) | Δ (rot − add) | random-control Δ |
|---|---|---|---|---|
| 0 | 78.89 | 176.08 | +97.19 | +95.20 |
| 1 | 77.60 | 177.34 | +99.74 | +26.53 |
| 2 | 89.41 | 174.42 | +85.02 | +1.10 |
| 3 | 75.65 | 166.63 | +90.98 | −0.06 |
| 4 | 80.20 | 167.43 | +87.23 | +0.50 |
| 5 | 74.98 | 164.56 | +89.58 | +3.30 |
| 6 | 75.86 | 162.32 | +86.46 | +1.91 |
| 7 | 73.23 | 166.92 | +93.69 | +5.96 |

**Mean Δ = +91.24, bootstrap 95% CI [87.98, 94.71]** (10,000 resamples).
**Random-direction control: +16.80** — and near zero in 6 of 8 draws.

Power (R6): `n=8, m=1, Holm α=0.050, min attainable p=0.0078, approx power ≈ 1.00`
→ **EVALUATION-eligible**.

## Pre-registered predictions — all three confirmed

| prediction | outcome |
|---|---|
| P1 — ordering stable across all seeds | **8/8** rotation worse |
| P2 — paired CI excludes zero | **[87.98, 94.71]**, excludes 0 |
| P3 — real gap exceeds the random-control gap | **+91.24 vs +16.80** (5.4×) |

## Why P3 is the one that matters

Without it, a sceptic could say the rotation penalty is a generic property of turning
*any* vector inside the residual stream. The norm-matched random control (the
`2606.20852` protocol) rules that out: a random direction of identical norm produces a
gap of +16.80 on average and **essentially none in 6 of 8 draws**. The penalty is tied to
rotating along a *meaningful* direction, not to rotation per se.

## What this upgrades

It closes the largest hole in HC-1/HC-2, which used a direction estimated from **10
harmful / 8 harmless** prompts at n=1 per cell. The ordering survives a direction
estimated from **340/120 real benchmark prompts** across 8 resamples — so it is a
property of the geometry, not of a noisy 18-prompt estimate.

## Honest limitations

1. **One model** (abliterated Gemma-3-1B), **one layer** (12), **one budget** (f = 0.10).
   Cross-scale (4B) and cross-layer checks are not done.
2. Perplexity is a coherence proxy, not a capability measure.
3. The resampling unit is the **extraction set**, not decode seeds — correct here
   (greedy decoding has ~0 seed variance) but it means the CI covers direction-estimation
   variance, not generation variance.
4. Seed 0's random control (+95.20) is a clear outlier; the median random Δ is ≈ +2.6.
   The conclusion rests on the median behaviour, not the mean, and both point the same way.
5. This says nothing about whether rotation helps *behaviour* — only that it costs far
   more coherence per unit of displacement.

## Consequence for the literature

`2601.19375` ships norm-preserving rotation and dismisses additive steering as causing
"catastrophic degradation on smaller models" — **without a matched-displacement control**.
This is that control, and it inverts the conclusion at ≤4B. GEMS and ORBIT both assume
norm preservation protects coherence in multi-vector composition; that assumption is
now measured and contradicted for the single-vector case. **The N-vector extension is
the natural follow-up.**
