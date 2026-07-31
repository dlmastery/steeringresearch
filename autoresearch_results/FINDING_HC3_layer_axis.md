# HC-3 — The LAYER axis: L11 beats the champion, and the random control is what proves it

**Tier: EVALUATION-eligible, NOT external-ready.** Confirm stage n=8 fresh seeds, m=1,
min attainable p=0.0078, power ≈1.00. **Judge-free** (WikiText-2 perplexity).
Held fixed at champion values: `f=0.05`, `r=1.0`, diffmean over real benchmark prompts.
Base PPL **77.310**. Artifacts: `hillclimb_layer.json`, `hillclimb_layer_confirm.json`.

## The result

| | PPL | ratio vs base |
|---|---|---|
| **L11 (candidate)** | **73.484** | **0.9505×** |
| L12 (champion) | 75.296 | 0.9740× |

Paired Δ = **−1.812**, bootstrap CI95 **[−2.383, −1.346]**, **8/8 seeds**,
2σ_paired = 1.639 → **beats L12 outside seed noise**.

**But the ordinal gate FAILS:** worst L11 seed (75.291) does not beat best L12 seed
(74.251). Under CLAUDE.md §7 a claim is EXTERNAL-READY only when the worst evaluation seed
beats the best baseline seed. So L11 is promoted as the internal champion and is **not**
an external claim.

## The screening sweep, and why the random control earns its cost

Every layer was also run with a **norm-matched random direction** (M-a's control protocol).
That column is what separates a real win from layer insensitivity:

| L | real PPL | random PPL | **real − random** | reading |
|---|---|---|---|---|
| **11** | **73.283** | 120.104 | **−46.8** | maximally direction-specific |
| 12 | 75.044 | 116.350 | −41.3 | the old champion |
| 17 | 73.939 | 78.537 | **−4.6** | **near-null** |
| 20 | 74.025 | 84.459 | −10.4 | near-null |
| 23 | 77.287 | 79.269 | −2.0 (CI crosses 0) | **NULL** |
| 25 | 81.896 | 78.984 | **+2.9** | **worse than random** |

**A naive read of the PPL column alone would have picked L17.** At 73.939 it looks
competitive with L11's 73.283. But L17's *random* control is also nearly free (78.537),
so its low perplexity is the layer being **insensitive to injection**, not the direction
doing work. L11's random control costs 120.104 — the gap is **−46.8 vs −4.6**.

**L23 is a true null** (real ≈ random, CI crosses zero) and **L25 is worse than random**.
Without the control column, all three would have entered the champion race on their
perplexity alone.

## Pre-registered predictions: two of four failed

| | verdict |
|---|---|
| P1 — the layer axis is live (spread > 10% of base) | **✘** spread 1.81 PPL, well under the 7.73 bar |
| **P2 — argmin layer ≥ 17** | **✘ argmin is 11 — my depth mechanism was backwards** |
| P3 — the best layer is null | ✘ (good: real−random = 42.58) |
| P4 — best ratio < 1.0 | ✔ |

**P2 is the informative failure.** I predicted the best injection site would be *deep*
(≥ L17), reasoning that later layers carry more abstract, more steerable representations.
The measured argmin is **L11 — just below the old champion — and L25, the deepest tested,
is the worst layer of all** (1.059× base, worse than its own random control). The
depth intuition is inverted for this model and budget.

P1 also failed on its own threshold: the axis moves the objective by only ~1.8 PPL across
26 layers, so **layer is a weak axis** — real, ordered, direction-specific, but small.

## Open cells

L17 and L20 are unresolved: low PPL but near-null controls. Whether they are genuinely
insensitive or merely need a larger budget to show an effect is untested.

> Internal QA pass — independent external review pending.
