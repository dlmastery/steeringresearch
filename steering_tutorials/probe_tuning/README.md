# probe_tuning — layer & hyperparameter search for the safety probe

This lesson is the **model-selection** work that was intentionally kept **out of
`hello_world`**. Lesson 1 (`hello_world`) trains and ships ONE fixed probe head so
it can be read top-to-bottom without distraction. This lesson asks the natural
follow-up question a practitioner asks next: *is the deployed default actually the
best head, or can we do better?* — and answers it honestly.

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

- (later) a **layer sweep** — which residual-stream layer to read the activation
  from — will join this lesson, since it is the same kind of model-selection
  search over a hyperparameter that does not belong in the minimal lesson 1.

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
