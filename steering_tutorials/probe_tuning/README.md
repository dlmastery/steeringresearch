# probe_tuning — MLP-head hyperparameter search for the safety probe

> **Reference:** [Understanding intermediate layers using linear classifier probes (arXiv:1610.01644)](https://arxiv.org/abs/1610.01644) — the model-selection search behind lesson 1's probe.

> **Scope, stated up front.** This lesson currently implements **one** search: the
> **MLP-head hyperparameter sweep** (`sweep_mlp.py` →
> `artifacts/sweep_mlp.{json,md,png}`). A **layer sweep** — which residual-stream
> layer to read the activation from — is **planned but NOT implemented**: there is
> no code and no artifact for it in this directory, and nothing on this page is a
> layer-selection result. Lesson 1's layer 12 is an inherited default, not a
> validated choice. This page previously titled itself "layer & hyperparameter
> search"; that framing is **withdrawn** until the sweep exists.

This lesson is the **model-selection** work that was intentionally kept **out of
`hello_world`**. Lesson 1 (`hello_world`) trains and ships ONE fixed probe head so
it can be read top-to-bottom without distraction. This lesson asks the natural
follow-up question a practitioner asks next: *is the deployed default actually the
best head, or can we do better?* — and answers it honestly.

## The key idea in code

Score every candidate head by 5-fold cross-validation — never the held-out test
set — then keep the deployed default unless something beats it by more than the
fold-to-fold noise:

```python
# Each config: 5-fold CV, StandardScaler fit per-fold on TRAIN ONLY (no leakage).
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for tr, te in skf.split(X, y):
    scaler = StandardScaler().fit(X[tr])                  # never sees the held-out fold
    probe  = train_one(scaler.transform(X[tr]), y[tr], ..., cfg)
    aucs.append(roc_auc_score(y[te], proba(probe, scaler.transform(X[te]))))

# Selection rule: only a margin BIGGER than the default's own CV noise band counts.
margin = winner_auc - default_auc
beats  = (not is_default(winner)) and (margin > default_auc_std)   # 1 std = CV noise
```

On ~200 rows over ~23 configs the top of any leaderboard is mostly luck, so the
`> 1 std` gate is what stops a noise winner from displacing a good default. Full
file-by-file walkthrough below.

## Dataset

This lesson has **no dataset of its own**. Model selection needs the *features*,
not the raw text, so it **reads lesson-1's cached activations** at
`../hello_world/artifacts/features.npz` — the **200 JailbreakBench prompts**
(100 harmful + 100 benign, prompt-level `1 = harmful` / `0 = safe`) already run
once through the frozen Gemma-3-1B and stored as their layer-12 mean-pooled
vectors: an `X` matrix of shape `[200, 1152]` with the matching label vector `y`.
Because the expensive forward pass was done in lesson 1, this lesson is
**CPU-only — the Gemma model is never loaded**. It uses those features purely to
do **MLP-head model selection by 5-fold cross-validation**, never by peeking at
lesson-1's held-out test slice. (A *layer* sweep would need activations from
layers other than 12, which this cache does not contain — one reason it is still
unimplemented; see the gap note below.)

## What lives here

- `sweep_mlp.py` — an MLP head hyperparameter sweep (width, depth, dropout, lr,
  weight-decay). Run:

  ```
  python -m steering_tutorials.probe_tuning.sweep_mlp
  ```

  It **reads** lesson 1's cached frozen-LLM activations at
  `../hello_world/artifacts/features.npz` (resolved relative to the script) and
  reuses lesson 1's exact train recipe. It **writes** only into
  `probe_tuning/artifacts/` — nothing is written back into `hello_world`.

- **KNOWN GAP — the layer sweep does not exist.** A sweep over which
  residual-stream layer to read from belongs here (same kind of model selection,
  same reason it is kept out of the minimal lesson 1), but **no code and no
  artifact for it are present**: the only script in this directory is
  `sweep_mlp.py`, and `artifacts/` holds only `sweep_mlp.{json,md,png}`. It also
  cannot be run from the current cache, which stores layer-12 activations only —
  building it means re-running the frozen Gemma to dump every candidate layer,
  which is GPU work this lesson has never done. Treat "layer 12" throughout the
  course as **unvalidated by any sweep in this repository**. Nothing on this page
  should be read as evidence about layer choice.

## The discipline (why this is a separate lesson)

Model selection is where it is easiest to fool yourself, so the rules are strict:

- **Selection is by cross-validation, never by test-set peeking.** Every config is
  scored by StratifiedKFold(k=5), with the StandardScaler fit per-fold on the
  training split only. The held-out test slice from lesson 1 is never consulted
  to pick a winner.
- **A "winner" must clear the noise band.** On a ~200-row set with dozens of
  configs, the top of any leaderboard is mostly CV noise. A config only counts as
  a real improvement if it beats the deployed default by more than the default's
  own fold-to-fold std — and even then it must be confirmed at n>=7 seeds before
  it is deployed. The verdict says this plainly.

CPU-only: the Gemma model is never loaded here; we reuse lesson 1's cached
activations.

## Results — measured vs. the claim

| Claim | What we measured | Verdict |
|---|---|---|
| Model selection is by CV, never test-set peeking | 23 MLP-head configs, each scored by StratifiedKFold(k=5, `cv_seed=0`) mean ROC-AUC on `X` = [200, 1152], balanced [100, 100]; lesson-1's held-out slice never consulted | **Held** |
| The lesson also selects the *layer* | nothing — no layer sweep is implemented and no layer artifact exists | **NOT MEASURED** (see the gap note above) |
| A better head must clear the noise band, not just top the leaderboard | top config (64→64) CV ROC-AUC **0.945 ± 0.015** vs. the deployed default (128→32) **0.9415 ± 0.0157** — a margin of **+0.0035**, well inside the default's own 1-std band (**0.0157**) | **KEEP the default** |

**Honest read.** The 23-config sweep does surface a nominal "winner" (64→64),
but its edge over the shipped default is **+0.0035 CV ROC-AUC — about a fifth of
the noise band** (0.0157), so `beats_default_by_more_than_1std` is `false`. On a
200-row set scored by 5-fold CV this is exactly the leaderboard-noise trap this
lesson exists to warn about: the top row is not a real improvement, and even if
it were it would need confirmation at n≥7 seeds before deployment. The
disciplined call is to **keep the simple default** and not buy capacity the
cross-validation cannot justify.
