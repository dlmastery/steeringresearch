# HC-S — The SOURCE axis: diffmean survives, and a 15° estimation error is expensive

**Tier: EVALUATION** — n=8 extraction resamples, m=4 pre-registered, Holm α=0.0125,
min attainable p=0.0078. **Judge-free** (WikiText-2 perplexity). Champion coordinates
held fixed: layer 12, `f=0.05`, `r=1.0`. Base PPL **77.310** (reproduces M-a's 77.310).
Artifact: `autoresearch_results/hillclimb_source.json`.

## What makes the arms comparable

Within a seed, all five estimators consume the **identical** cached pooled-activation
matrices, are **unit-normalised**, sign-oriented by the same rule, and injected at the
identical site/budget/operation. Pool, injection and norm fixed ⇒ **the estimator is the
only free variable**. Without that, this would be comparing estimators *and* magnitudes.

| estimator | PPL | ratio vs base | paired Δ vs diffmean [CI95] | seeds |
|---|---|---|---|---|
| **diffmean (champion)** | **75.341** | **0.9745×** | — | — |
| `pca_uncentered` | 78.675 | 1.0177× | +3.334 [+2.655, +4.008] | 8/8 worse |
| `pca_centered` | 79.817 | 1.0324× | +4.476 [+3.919, +5.076] | 8/8 worse |
| `lda_shrunk` (γ=0.10) | 99.543 | 1.2876× | +24.202 [+19.916, +28.998] | 8/8 worse |
| `random` (norm-matched floor) | 122.242 | 1.5812× | +46.900 [+36.452, +57.798] | 8/8 worse |

**diffmean is the only arm that goes BELOW unsteered base.** The ordinal gate clears:
worst diffmean seed (76.631) beats the best PCA seed (78.399).

## The interesting part: cos 0.966 is "the same direction" and still costs +3.33 PPL

| | dm | pca_u | pca_c | lda | rand |
|---|---|---|---|---|---|
| **dm** | 1 | **0.966** | **0.949** | 0.211 | 0.026 |
| pca_u | | 1 | **0.998** | 0.042 | 0.028 |

PCA sits **15.1°** from diffmean — above the 0.95 "effectively the same direction" bar
this program usually applies. Under that rule the +3.33 PPL gap should have been noise.
**It is not:** CI excludes zero, 8/8 seeds. Calibrating against the random floor, a
quadratic orthogonal-leakage model predicts the penalty almost exactly (pca_u 3.17
predicted vs 3.334 observed; pca_c 4.67 vs 4.476).

So PCA is best read as **diffmean plus a 15° estimation error**, and that small error is
**reproducibly expensive**. The practical lesson: a cosine bar of 0.95 is *not* sufficient
to call two steering directions interchangeable — at this budget, 15° of leakage costs
more than 4% of base perplexity.

LDA breaks the leakage model (44.8 predicted vs 24.2 observed), so it is a genuinely
different direction — but an unstable one: cross-seed self-cosine **0.759** against
diffmean's **0.993**.

## Pre-registered predictions

| | verdict |
|---|---|
| P1 — cos(diffmean, pca_uncentered) > 0.90 | ✔ |
| **P2 — cos(diffmean, pca_centered) < 0.50** | **✘ observed 0.949** |
| P3 — no estimator beats diffmean with a CI excluding zero | ✔ |
| P4 — random floor worse than diffmean | ✔ |

**P2 failed, and the failure is informative.** Centering did not decorrelate: the centered
PC carries **95.3%** explained variance and points at the mean-difference direction. The
paired-difference cloud is essentially **1-D**, so "take the top PC" and "take the mean
difference" recover the same axis by construction on this data.

## The caveat that limits the win

All three top arms have huge raw class separation (626 / 605 / 593) while LDA — which
whitens out the dominant high-variance residual axis — has **133**. So diffmean's
advantage **may partly reflect alignment with a high-variance residual-stream axis the
model is already robust to**, rather than refusal-specificity alone. Judge-free perplexity
cannot distinguish those two explanations, and this finding does not claim to.

> Internal QA pass — independent external review pending.
