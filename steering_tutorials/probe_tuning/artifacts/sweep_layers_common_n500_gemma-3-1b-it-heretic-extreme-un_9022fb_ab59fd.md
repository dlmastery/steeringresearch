# Layer x Pooling Sweep - where should the safety probe read?

Lesson 2 (`probe_tuning`). This is the sweep the README used to list as a KNOWN GAP: lesson 1's **layer 12, mean-pooled** is an inherited default, and until this ran, nothing in this repository had tested it.

Activations: `layer_features_common_n500_gemma-3-1b-it-heretic-extreme-un_9022fb.npz` - **1000 prompts** (500/class, balance [500, 500]) through frozen `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`, 26 layers x 3 poolings x windows [1, 3] = **150 cells**.

Each cell scored by StratifiedKFold(k=5, shuffle, random_state=0) with the StandardScaler fit per-fold on train only, using the DEPLOYED head and train recipe (`128->32 d0.3 lr0.001 wd0.001`). **Selection is by cross-validation mean roc_auc; the held-out test set is never consulted.**

## The benign / over-refusal arm

**PRESENT** — XSTest (Paul/XSTest CSV; Rottger et al. 2024, arXiv:2308.01263), n=300, balance [150, 150], loaded via `hello_world.eval_ood.load_xstest_balanced`. It is **reported-only**: every cell is scored on it zero-shot per CV fold, and it is **never** used to select a cell (that would be OOD test-set peeking).

This arm exists because of arXiv:2605.02958 Appendix D — *"sequence-level mean aggregation can achieve high recall on several attack sets but yields near-random XSTest AUROC"*. Without it, a mean-pooling cell could top this leaderboard while being broken on over-refusal, and the sweep would never know.

## Top cells by CV roc_auc

| rank | cell | in_dim | CV roc_auc | XSTest roc_auc | CV accuracy | note |
|---|---|---|---|---|---|---|
| 1 | `L10 mean` | 1152 | 0.9802 +/- 0.0065 | 0.9020 +/- 0.0067 | 0.9260 +/- 0.0073 | sweep-winner |
| 2 | `L12 mean` | 1152 | 0.9798 +/- 0.0058 | 0.8955 +/- 0.0084 | 0.9240 +/- 0.0107 | **deployed** |
| 3 | `L11 mean` | 1152 | 0.9798 +/- 0.0054 | 0.9112 +/- 0.0044 | 0.9230 +/- 0.0163 |  |
| 4 | `L13 mean` | 1152 | 0.9787 +/- 0.0064 | 0.8800 +/- 0.0056 | 0.9240 +/- 0.0146 |  |
| 5 | `L11-13 max (w3)` | 3456 | 0.9784 +/- 0.0078 | 0.9138 +/- 0.0040 | 0.9280 +/- 0.0108 |  |
| 6 | `L12-14 mean (w3)` | 3456 | 0.9781 +/- 0.0075 | 0.8871 +/- 0.0084 | 0.9270 +/- 0.0160 |  |
| 7 | `L09 mean` | 1152 | 0.9781 +/- 0.0067 | 0.9004 +/- 0.0058 | 0.9190 +/- 0.0193 |  |
| 8 | `L08-10 mean (w3)` | 3456 | 0.9780 +/- 0.0073 | 0.8979 +/- 0.0059 | 0.9270 +/- 0.0098 |  |
| 9 | `L10-12 max (w3)` | 3456 | 0.9779 +/- 0.0081 | 0.9195 +/- 0.0050 | 0.9270 +/- 0.0098 |  |
| 10 | `L14 mean` | 1152 | 0.9773 +/- 0.0081 | 0.8881 +/- 0.0081 | 0.9300 +/- 0.0182 |  |
| 11 | `L08-10 max (w3)` | 3456 | 0.9767 +/- 0.0098 | 0.9075 +/- 0.0060 | 0.9250 +/- 0.0130 |  |
| 12 | `L11 max` | 1152 | 0.9765 +/- 0.0091 | 0.9207 +/- 0.0030 | 0.9260 +/- 0.0156 |  |

## Appendix-D check — strong on attacks, near chance on over-refusal

Flag rule: main roc_auc >= 0.8 **and** XSTest roc_auc <= 0.6.

**No cell trips the flag** on this data — the Appendix-D failure mode did not reproduce here. That is a real negative result, and it is only sayable BECAUSE the benign arm was measured.

## Baseline - the deployed cell

`L12 mean` (rank 2 of 150): **roc_auc 0.9798 +/- 0.0058**, accuracy 0.9240 +/- 0.0107.

## Shuffle control

The winning cell re-scored with PERMUTED labels: roc_auc **0.4772 +/- 0.0469** (chance = 0.5). Passes - a pipeline that scores above chance on shuffled labels is leaking, and every number above would be void.

## Verdict - is layer 12 + mean pooling the right read?

The best cell `L10 mean` scores 0.9802 - only **+0.0004** vs the deployed 0.9798, **inside** the deployed cell's noise band (1 std = 0.0058). No layer and no pooling beats the inherited default by more than fold-to-fold noise. The honest reading is NOT 'layer 12 is optimal' but '**the probe is insensitive to this choice over the swept range**' - which is itself the answer to the question the README left open.
