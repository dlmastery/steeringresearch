# probe_tuning — head, layer, and pooling search for the safety probe

> **References:**
> - [Understanding intermediate layers using linear classifier probes (arXiv:1610.01644)](https://arxiv.org/abs/1610.01644) — Guillaume Alain & Yoshua Bengio, ICLR 2017 workshop. The linear-probe-per-layer method that makes "which layer should we read?" a measurable question at all; the motivation for the layer axis of `sweep_layers.py`.
> - [Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection (arXiv:2605.02958)](https://arxiv.org/abs/2605.02958) — Xulin Hu, Che Wang, Wei Yang Bryan Lim, Jianbo Gao, Zhong Chen, 2 May 2026. Argues that "static directions extracted from terminal or pooled representations" miss how refusal is built across layer-token positions, and localises a **sparse** upstream pattern; their SALO detector reads a **layer window** of raw hidden states. Three lines drive this lesson's design, and **two of the three are body text, not in the abstract** — check the full text before deciding any of them is unsupported:
>   - **Sec. 5.4 (Ablation Study):** *"Mean pooling dilutes this strong 'needle' into the vast 'haystack' of irrelevant background noise."* → the pooling axis is mandatory, not decorative.
>   - **Table 2 caption:** *"Mean Pooling: Replaces the sparsity-aware Global Max-Pooling with Global Average Pooling."* → **Global max-pooling is the paper's own sparsity-aware default**, and mean is the ablation *of* it. So `max` is a first-class cell here and leads the default `PT_POOLINGS`.
>   - **Appendix D:** *"sequence-level mean aggregation can achieve high recall on several attack sets but yields near-random XSTest AUROC."* → an attack-only sweep would rank mean pooling well and never see it fail on over-refusal. Hence the **benign arm** below.
>
>   (Inspired-by, not a reproduction: we sweep a probe's read-out, we do not implement SALO.)
> - [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models (arXiv:2308.01263)](https://arxiv.org/abs/2308.01263) — Röttger et al., NAACL 2024. The over-refusal probe used as the sweep's benign arm: safe prompts that merely *sound* dangerous, plus their genuinely-harmful contrast twins.

> **Scope, stated up front.** This lesson ships **two** searches, at different
> stages of completeness:
> 1. The **MLP-head hyperparameter sweep** — `sweep_mlp.py` →
>    `artifacts/sweep_mlp.{json,md,png}`. **Run; results below.**
> 2. The **layer × pooling sweep** — `extract_layers.py` + `sweep_layers.py`.
>    **Code exists and is import-checked; it has NEVER BEEN RUN.** There is no
>    `sweep_layers_*.json` in `artifacts/`, so **nothing on this page is a
>    layer-selection result** and lesson 1's layer 12 + mean pooling remains an
>    **inherited default, not a validated choice**. The page title covers layer and
>    pooling because the apparatus is here; the *claim* stays withdrawn until the
>    artifact exists.

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
lesson-1's held-out test slice.

The **layer × pooling sweep cannot use that cache** — it stores layer 12,
mean-pooled, and nothing else, so the other layers simply are not in it. That
sweep therefore brings its own data: `extract_layers.py` re-runs the frozen Gemma
over the **≥500/class** length-matched harmful-vs-benign set from
`steering_tutorials.common.data` (the project-standard set; length-matched so a
length confound cannot masquerade as a layer effect) and dumps every layer under
every pooling. Lesson-1's 100+100 JailbreakBench set stays available as
`--dataset jbb`, but at 100/class it is **below the ≥500/class rubric** and is a
cross-check only, never a headline.

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

- `extract_layers.py` — **the GPU half of the layer sweep.** One forward pass per
  prompt with a hook on **every** decoder block, capturing **every** pooling
  (`max` / `mean` / `last`) at once, into
  `artifacts/layer_features_<tag>.npz`. This is the step lesson 1's cache cannot
  substitute for: `features.npz` holds layer 12, mean-pooled, and nothing else.
  Defaults to the project-standard **≥500/class** length-matched harmful-vs-benign
  set from `common.data`; resumable (checkpoints every `PT_CKPT_EVERY` prompts) so a
  reaped job costs a minute, not an hour.

  It also caches the **benign / over-refusal arm** — XSTest, via lesson 1's existing
  `eval_ood.load_xstest_balanced` against the locally cached `Paul/XSTest` CSV —
  into the same file under a `group` column (0 = main, 1 = XSTest). `--no-benign-arm`
  turns it off, and the run then warns that everything downstream is attack-only.

  ```
  python -m steering_tutorials.probe_tuning.extract_layers
  ```

- `sweep_layers.py` — **the CPU half**: scores every (layer × pooling × window)
  cell by the *same* 5-fold CV protocol as `sweep_mlp.py` (reusing that file's
  `train_one` / `stratified_val_split` / `DEFAULT`), using the deployed head, and
  prices the winner against the **deployed** cell (layer 12, mean) with the same
  1-std noise-band gate. Adds a shuffled-label control on the winning cell. The JSON
  is rewritten after **every** cell, so a crash still leaves data.

  ```
  python -m steering_tutorials.probe_tuning.sweep_layers
  python -m steering_tutorials.probe_tuning.sweep_layers --selftest
  ```

  **Two arms per cell.** The *main* arm (harmful vs benign) is what selection uses.
  The *XSTest* arm is scored **zero-shot per CV fold** and is **reported only** —
  selecting on it would be OOD test-set peeking. Any cell that is strong on attacks
  (`roc_auc ≥ 0.80`) yet near chance on over-refusal (`XSTest ≤ 0.60`) is stamped
  **APPENDIX-D FLAG** in the JSON, the markdown, and the console. If the cache has no
  XSTest rows, the JSON records `benign_arm: "ABSENT"` and the run prints a loud
  warning that its numbers are attack-only.

  `--selftest` proves `cross_validate_two_arm` reproduces
  `sweep_mlp.cross_validate_config` to 1e-12 when the benign arm is absent, so
  "identical protocol" is a checkable claim rather than a comment. It passes.

  Env caps to fit one foreground window: `PT_N`, `PT_LAYERS` (`all` / `every2` /
  `0-25` / `0,6,12`), `PT_POOLINGS`, `PT_WINDOWS`, `PT_FOLDS`, `PT_SEED`,
  `PT_BENIGN_N`.

- **KNOWN GAP — the layer sweep has CODE but NO RESULT.** The two scripts above are
  written and import-checked, and were smoke-tested end-to-end on synthetic caches:
  a planted signal at layer 12 was recovered and ranked first; a planted
  Appendix-D signature (a cell at main AUC 1.0000 with XSTest AUC 0.4208) was
  correctly flagged while the max-pooled cell that held up on **both** arms was not;
  and the `benign_arm: "ABSENT"` path was exercised on a cache with no `group`
  column. But the real
  extraction is GPU work that **this lesson has still never done**: `artifacts/`
  holds no `layer_features_*.npz` and no `sweep_layers_*.{json,md,png}`. Until it is
  run, treat "layer 12" and "mean pooling" throughout the course as **unvalidated by
  any sweep in this repository**, and read nothing on this page as evidence about
  layer or pooling choice.

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
| The lesson also selects the *layer* and the *pooling* | nothing — `extract_layers.py` + `sweep_layers.py` exist and import cleanly, but have never been run; no `layer_features_*.npz` and no `sweep_layers_*.json` exist | **NOT MEASURED** (code, not evidence — see the gap note above) |
| A better head must clear the noise band, not just top the leaderboard | top config (64→64) CV ROC-AUC **0.945 ± 0.015** vs. the deployed default (128→32) **0.9415 ± 0.0157** — a margin of **+0.0035**, well inside the default's own 1-std band (**0.0157**) | **KEEP the default** |

**Honest read.** The 23-config sweep does surface a nominal "winner" (64→64),
but its edge over the shipped default is **+0.0035 CV ROC-AUC — about a fifth of
the noise band** (0.0157), so `beats_default_by_more_than_1std` is `false`. On a
200-row set scored by 5-fold CV this is exactly the leaderboard-noise trap this
lesson exists to warn about: the top row is not a real improvement, and even if
it were it would need confirmation at n≥7 seeds before deployment. The
disciplined call is to **keep the simple default** and not buy capacity the
cross-validation cannot justify.
