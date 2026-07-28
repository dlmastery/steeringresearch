# M-b — Radial-beats-angular does NOT survive stacking. GEMS/ORBIT are vindicated.

**Tier: EVALUATION** — n=8 extraction resamples, m=4 pre-registered (four N values),
Holm α = 0.0125, min attainable p = 0.0078, power ≈ 1.00. **Judge-free** (WikiText-2
perplexity). Artifact: `autoresearch_results/nstack_matched_budget.json`.
Model: Gemma-3-1B abliterated, layer 12, 4-bit. Unsteered PPL **76.49**.

## What this tested and why

M-a established that at a *fixed chord displacement*, spending the budget on **radius**
(addition) beats spending it on **angle** (norm-preserving rotation): +91.24 PPL,
CI95 [87.98, 94.71], 8/8 resamples. That result stands. But it is a **single-vector**
result, and that is exactly where it is weakest as a challenge to the literature.

GEMS ([arXiv:2606.19946](https://arxiv.org/abs/2606.19946)) and ORBIT
([arXiv:2606.22357](https://arxiv.org/abs/2606.22357)) motivate norm preservation for
**stacking**: when several edits compose, uncontrolled norm growth compounds, so
rotation should pay off as N grows *even if it loses at N=1*. M-a cannot speak to that
regime. M-b was built to.

**Matched-budget construction.** N directions from *disjoint* harmful slices,
Gram-Schmidt orthonormalised, each carrying `f/√N` so the **total displacement is
exactly f for every N**. Without this the comparison would confound "more vectors" with
"more displacement".

## Result

| N | gap (rotation − addition) | 95% CI | additive wins | additive PPL / base |
|---|---|---|---|---|
| 1 | **+88.89** | [+84.33, +92.77] | **8/8** | 1.09× |
| 2 | +24.63 | [−6.12, +56.17] | 5/8 | 2.25× |
| 4 | **−45.99** | [−97.24, −6.53] | 2/8 | 3.32× |
| 8 | **−44.47** | [−75.30, −20.07] | 1/8 | 3.35× |

**The additive advantage collapses and inverts.** At N=1 addition wins by ~89 PPL in
8/8 seeds — M-a reproduced. By N=4 rotation wins with a CI excluding zero, and at N=8 it
wins in 7 of 8 seeds. The mechanism is visible in the last column: the additive arm
degrades from 1.09× to 3.35× base perplexity as vectors stack, while rotation holds. That
is precisely the compounding norm-growth argument GEMS and ORBIT make.

## Pre-registered predictions: all three fail, and one fails *deceptively*

| prediction | mechanical verdict | honest verdict |
|---|---|---|
| P1 — additive wins at every N | **FALSE** | FALSE |
| P2 — the gap does NOT shrink monotonically | *"TRUE"* | **MISLEADING — see below** |
| P3 — additive stays < 1.5× base at N=8 | **FALSE** | FALSE (3.35×) |

**P2's mechanical test returned the wrong answer, and reporting it as a pass would be
technically accurate and substantively dishonest.** I coded P2 as *"not strictly
monotonically decreasing"*. The gaps are `[+88.89, +24.63, −45.99, −44.47]` — a collapse
of 135 PPL and a sign inversion across N=1→4, followed by a **1.5 PPL uptick at N=8 that
sits deep inside its own CI**. Strict monotonicity fails only on that last, entirely
non-significant step. The substantive pattern — large, ordered decay and inversion — is
exactly what P2 was written to rule out.

The defect is mine: a discriminating prediction must be operationalised on the *effect*,
not on a brittle ordering property that a single noisy point can flip. Recorded rather
than quietly reinterpreted.

## Conclusion

**The norm-preservation premise is vindicated in the regime it was proposed for.**
M-a's finding is real but **strictly single-vector**, and every statement of it in this
repository must carry that scope. This program set out to challenge GEMS/ORBIT and
instead produced evidence *for* them at N ≥ 4 — a more useful outcome than confirmation
would have been.

## Limitations

- One model (Gemma-3-1B abliterated), one layer (12), one budget (f = 0.10).
- Directions are diff-of-means on refusal only; other concept families untested.
- N=2's CI includes zero, so the crossover point is bracketed between N=2 and N=4 rather
  than located.
- Rotations are applied sequentially, so for N > 1 the composed operation is
  order-dependent; a permutation control is not run.

> Internal QA pass — implementer and critic share a model family; independent external
> review pending.
