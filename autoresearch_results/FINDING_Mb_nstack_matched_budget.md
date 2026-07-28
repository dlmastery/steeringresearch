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

## THREE BASES, THREE ANSWERS — and only one of them controls both variables

The order control (above, in git history) showed the Gram-Schmidt basis was not
exchangeable. Fixing that exposed a deeper problem: **a stacking comparison needs two
things at once, and the obvious bases each supply only one.**

| basis | realised displacement | directions exchangeable? | measured trend in the gap |
|---|---|---|---|
| `gs` (original) | **1.000×f** at every N ✓ | **NO** — Gram-Schmidt makes direction *i*>1 a residual, quality decays with index | **shrinks**: +88.89 → −44.47 |
| `raw` | **1.00 → 1.41 → 1.99 → 2.81×f** ✗ | **YES** — every direction the same kind of estimate | **grows**: +88.89 → +621.95 |
| **`rawnorm`** | **1.000×f** at every N ✓ | **YES** ✓ | **FLAT** |

The `gs` "shrink" and the `raw` "growth" are **both artifacts**, in opposite directions,
of the variable each basis left free. Proof that `gs` is not exchangeable: at N=1, where
permuting the direction order *must* be a no-op, `add/base` moved **1.09× → 3.72×**.
Proof that `raw` does not hold the budget: displacement nearly **triples** by N=8,
because refusal directions are mutually correlated rather than orthogonal.

`rawnorm` keeps the exchangeable raw directions and rescales the **shared** per-vector
magnitude so the composite norm is exactly f (`per = f / ‖Σv̂ᵢ‖`) — satisfying both.

## Result on the controlled basis (n=8, EVALUATION)

| N | gap (rotation − addition) | sd | realised displacement | additive PPL / base |
|---|---|---|---|---|
| 1 | **+88.89** | 6.51 | 1.000×f | 1.085× |
| 2 | **+86.59** | 5.96 | 1.000×f | 1.092× |
| 4 | **+85.95** | 3.51 | 1.000×f | 1.055× |
| 8 | **+85.92** | 1.87 | 1.000×f | 1.016× |

**Additive wins in 32 of 32 cells.** The gap is **flat** — ~86–89 PPL from N=1 to N=8,
a 3 PPL drift against seed standard deviations of 2–7.

### Two conclusions

1. **M-a's radial-beats-angular result is INVARIANT to the number of stacked vectors**,
   once total displacement is genuinely held fixed. Rotation costs the same ~86 PPL at
   N=8 as at N=1.
2. **This restores the challenge to GEMS/ORBIT on solid ground, and sharpens it.** Their
   premise is that norm preservation becomes *more* valuable as edits compose. Measured
   with both variables controlled, it becomes **no more valuable at all**.

A third observation worth recording: `additive PPL / base` stays at **1.02–1.09×** at
every N, and *falls* slightly as N grows. Stacking eight vectors at a matched total
budget is essentially free in perplexity. The coherence collapse seen on the other two
bases (up to 3.38× base) was entirely the unmatched displacement, never the stacking.

## Pre-registered predictions, re-scored on the controlled basis

| prediction | verdict on `rawnorm` |
|---|---|
| P1 — additive wins at every N | **TRUE** (32/32 cells) |
| P2 — the gap does NOT shrink monotonically | **TRUE** — it is flat, not shrinking |
| P3 — additive stays < 1.5× base at N=8 | **TRUE** (1.016×) |

All three hold — but note they were scored **FALSE / misleading / FALSE** on the `gs`
basis. The predictions did not change; the experiment's control did. That is the whole
lesson of this finding, and it is why the earlier verdicts are left in git history rather
than deleted.

