# MLP Head Hyperparameter Sweep — Safety Probe

Lesson 2 (`probe_tuning`): the model-selection search kept out of the minimal `hello_world` lesson.

Frozen-LLM activations `X` = **200x1152**, balanced ([100, 100]), read from lesson 1's cached `../hello_world/artifacts/features.npz`. CPU-only; the Gemma model is never loaded.

**23 configs** scored by StratifiedKFold(k=5, shuffle, random_state=0), StandardScaler fit per-fold on train only, deployed train recipe (Adam, BCE, early stop on a stratified val slice). **Selection is by cross-validation mean roc_auc — the held-out test set is never consulted (no test-set peeking).**

## Top configs by CV roc_auc

| rank | config | CV roc_auc | CV accuracy | note |
|---|---|---|---|---|
| 1 | `64->64 d0.3 lr0.001 wd0.001` | 0.9450 +/- 0.0150 | 0.8850 +/- 0.0406 | sweep-winner |
| 2 | `128->16 d0.3 lr0.001 wd0.001` | 0.9445 +/- 0.0157 | 0.9050 +/- 0.0292 |  |
| 3 | `64->32 d0.3 lr0.001 wd0.001` | 0.9435 +/- 0.0197 | 0.8850 +/- 0.0339 |  |
| 4 | `128->32 d0.5 lr0.001 wd0.0001` | 0.9435 +/- 0.0174 | 0.8800 +/- 0.0367 |  |
| 5 | `64->(2-layer) d0.3 lr0.001 wd0.001` | 0.9430 +/- 0.0148 | 0.8800 +/- 0.0187 |  |
| 6 | `128->32 d0.5 lr0.0003 wd0.0001` | 0.9430 +/- 0.0193 | 0.8950 +/- 0.0367 |  |
| 7 | `256->64 d0.3 lr0.001 wd0.001` | 0.9430 +/- 0.0164 | 0.8800 +/- 0.0400 |  |
| 8 | `128->32 d0.2 lr0.001 wd0.001` | 0.9430 +/- 0.0182 | 0.8750 +/- 0.0354 |  |

## Baseline — deployed default

`128->32 d0.3 lr0.001 wd0.001` (rank 15 of 23): **roc_auc 0.9415 +/- 0.0157**, accuracy 0.8700 +/- 0.0292.

## Verdict — does anything meaningfully beat the default?

The best swept head `64->64 d0.3 lr0.001 wd0.001` scores 0.9450 roc_auc — only **+0.0035** vs the default's 0.9415, which is **within the default's CV noise band** (1 std = 0.0157). No config beats the default by more than the fold-to-fold noise. On a 200-row set with 23 configs, the top of the leaderboard is dominated by CV noise. **Recommendation: keep the simple deployed default (128->32, dropout 0.30, lr 1e-3, wd 1e-3) — it is already near-optimal for this data.**
