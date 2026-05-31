# Campaigns C4 (N20) + C5 (E1) — results (SCREENING, underpowered)

## C4 / N20 — is per-layer geometry a behavior-free fragility sensor? (INCONCLUSIVE)

Layer sweep @α=4 (exp#47–54) + per-layer effective-rank/participation-ratio of
the unsteered contrast activations (behavior-free):

| layer | Fisher | eff_rank | part_ratio | logPPL@α=4 (fragility) |
|------:|------:|------:|------:|------:|
| 2 | 0.20 | 12.69 | 1.94 | 5.90 |
| 8 | 0.29 | 8.59 | 1.16 | 7.23 |
| 12 | **30.57** | 16.94 | 9.47 | **7.93 (most fragile)** |
| 14 | 17.23 | 16.99 | 8.44 | 7.00 |
| 16 | 18.25 | 17.21 | 10.32 | 7.04 |

- Spearman(effective_rank, logPPL) = **−0.21**; Spearman(participation_ratio,
  logPPL) = −0.10. **N20 INCONCLUSIVE (screening)**: per-layer effective-rank /
  participation-ratio do NOT predict layer fragility here. Underpowered: 8 layers,
  effective-rank from only 20 activations, single α. A real test needs more
  contrast data + a multi-α fragility measure.
- **Tie-in to C1/E2**: the max-Fisher layer L12 is also the **most fragile**
  (highest logPPL@α=4) — most-separable = most-fragile = worst steering layer,
  reinforcing why max-Fisher (E2) mis-selects the steering layer.

## C5 / E1 — DiffMean pair-count knee (DIRECTIONAL, underpowered)

DiffMean cosine-to-full(10-pair) vector vs n_pairs @L16 (Gemma-270m):

| n_pairs | cos to full |
|------:|------:|
| 1 | 0.688 |
| 2 | 0.768 |
| 3 | 0.860 |
| 5 | **0.951** |
| 8 | 0.983 |
| 10 | 1.000 |

- The **diminishing-returns shape (E1's core claim) is visible**: the vector
  stabilizes to cos>0.95 of the full vector by **n≈5 pairs**, with small gains
  after. But on this *easy synthetic* "ocean" concept the knee (~5) is far below
  the corpus's claimed ~50, and only 10 pairs are available. **E1 UNDERPOWERED**:
  the shape is confirmed, the magnitude/50-pair threshold cannot be tested without
  a real, harder contrast set.
