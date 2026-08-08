# SUPERSEDED — the 2026-07-27 run

These four files are the run the lesson shipped until 2026-08-08. They are kept, not
deleted, so the retracted claims remain auditable. **No number in them may be quoted as
a current result.** No code in this lesson writes to this directory.

## What they are

| file | what it is |
|---|---|
| `results.json` | the 2026-07-27 run: substrate `mixed` (toxic-chat `toxicity==1`, overt and disguised attacks pooled), **300 harmful + 300 benign**, layer 12, 40 generated tokens, 5-fold CV, bootstrap 2,000 |
| `roc_by_method.png` / `early_detection_auc.png` / `risk_drift_example.png` | its three plots |

## Why they were superseded

1. **The n was not what the config asked for.** `config.py` said 500/class, `README` §8
   said 120, the artifact shipped 300 — because `load_or_build()` checked no
   configuration knob and silently reused a cache built at a different size. The cache
   is now fingerprinted over the config *and the sampled prompt ids*, and the runner
   refuses a mismatch instead of absorbing it.
2. **The substrate could not test the paper's claim.** `mixed` pools overtly toxic
   inputs with disguised attacks; the abliterated model complies with the overt ones
   from token 1, so the drift a decoding-time monitor exists to read never happens. The
   lesson now runs `overt` and `disguised` as separate, labelled arms.
3. **The confound audit was missing two of four controls** — no content bar, no
   label-shuffle control. It now runs on the shared `common/confound.py` spine.
4. **The headline inference was wrong.** `prompt_charlen = 0.5032` was read as "you
   cannot call this from the prompt alone". That number is produced by the shared
   loader's length-matched benign sampler *by construction*, and prompt **content** —
   which it says nothing about — separates the classes at AUC 0.88–0.97 on these same
   pools. See README §10, "Retraction".
5. **Three F1 cells in the README disagreed with this file** (0.24 vs 0.172, 0.84 vs
   0.781, 0.86 vs 0.894), and §9's `threshold_freeform` margin was printed as −0.045
   where the arithmetic gives −0.098.

## What in them still stands

The **direction** of two findings, not their magnitudes: `threshold_freeform` (the
paper's own training-free method, 0.638) landed *below* the completion-character-length
bar (0.735), and the stateless per-token probe (0.931) essentially matched the sequence
models (0.944 / 0.945). Both are consistent with the current reading — on an
overt-toxicity substrate there is no *trajectory* signal, only a per-token one — but
they were measured on 300/class of a mixed substrate under a two-bar audit, so they are
corroboration, not evidence.
