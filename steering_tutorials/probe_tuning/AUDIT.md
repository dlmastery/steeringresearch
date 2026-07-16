# Paper Audit — `probe_tuning` (Layer/Head Model Selection)

Independent verification audit. Auditor did not modify lesson code or the README. This lesson cites **no external paper**; it is a cross-validation model-selection methodology, so checks 1/2/3 are recast as *methodology soundness + honest statement*, and check 4 is results honesty.

## Checks

| Check | Verdict | Evidence |
|---|---|---|
| 1. No external paper claimed (nothing to fabricate) | **PASS** | The README and `sweep_mlp.py` cite no arXiv id and make no attribution claim — correct; the content is a CV model-selection procedure. Nothing to verify externally, nothing overclaimed. |
| 2. Methodology fidelity (CV selection, no test peeking) | **PASS** | `sweep_mlp.cross_validate_config` uses `StratifiedKFold(n_splits=5, shuffle, random_state=0)`; the `StandardScaler` is `.fit(X[tr])` **per fold on train only**; the held-out fold is used only to score, and a stratified val slice carved from the fold's *train* split does early stopping. Selection is by CV mean ROC-AUC. Lesson-1's held-out test slice is never loaded (the script reads only cached features `X`,`y`). Claim of "CPU-only, model never loaded" is true — no `load_model` path exists here. |
| 3. Noise-band discipline honestly stated | **PASS** | `beats = (winner not default) and (margin > default_1std_noise_band)`. The winner (64→64) margin is **+0.0035** vs a default 1σ band of **0.0157**, so `beats_default_by_more_than_1std = false`. The README and the generated verdict both say to KEEP the default and require n≥7-seed confirmation before any swap — matching the project's rigor floor. |
| 4. Results honesty | **PASS** | `sweep_mlp.json` reconciles exactly with the README: default (128→32) ROC-AUC **0.9415 ± 0.0157** (rank 15/23), winner (64→64) **0.945 ± 0.015**, margin **+0.0035**, noise band **0.0157**, `beats=false`. The README frames the nominal "winner" as the leaderboard-noise trap this lesson exists to warn about — an honest read, not a spun one. |

## Note on the 1B self-judge flag
Not applicable — CPU-only, no model and no judge; labels come from cached JailbreakBench ground truth.

## Overall verdict
No paper to verify, and none is falsely claimed. The methodology is sound and strictly stated: 5-fold CV selection with per-fold train-only scaling, no test-set peeking, and a noise-band gate that correctly returns "keep the default." All reported numbers match `sweep_mlp.json`. No FAIL or CONCERN flags.

Internal QA pass — independent external review pending (auditor shares a model family with the author).
