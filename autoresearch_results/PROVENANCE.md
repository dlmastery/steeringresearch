# PROVENANCE WARNING — the 124-row experiment log is NOT one comparable scale

> Written 2026-07-25 during the post-mortem repair. Read this before comparing,
> sorting, or citing any `composite` value in `experiment_log.jsonl`.

## 1. The composite fingerprint no longer matches the code

| where | fingerprint |
|---|---|
| every one of the 124 rows in `experiment_log.jsonl` | `a9001e87087e` |
| `steering.eval.composite_fingerprint()` at HEAD | **`8509c229b58f`** |

`COMPOSITE_FORMULA` was deliberately edited after the log was written (the ΔPPL
coherence tax was bounded so it asymptotes to `DPPL_TAX_CAP` instead of exploding).
`src/steering/eval.py` documents this honestly in its header changelog:
*"COMPOSITE_FORMULA TEXT CHANGED ==> composite_fingerprint() CHANGES. This is
intended; prior-run fingerprints will no longer match."*

**The change was legitimate; the unhandled consequence is the defect.** No row was
re-scored and no fingerprint was updated, so the stored composites were produced by
a formula that no longer exists. Re-scoring the log under HEAD moves values
substantially (e.g. exp9: −53040.87 → −1.476). CLAUDE.md §6 treats the fingerprint
as an integrity seal precisely to catch this.

**Rule:** never sort, rank, or compare `composite` across rows without first
re-scoring them under a single formula version. The master dashboard's default
"sort by composite desc" is **not meaningful** across the full log.

## 2. Two of the five priced axes were inert

- `repetition_rate` is the constant `0.0145` in 109/124 rows — the runner scored the
  *fixed input* WikiText passages rather than generated text, so the coherence leg
  that is supposed to stop gibberish from winning contributed a constant offset.
- `harmless_refusal_rate` is `0.0` in 123/124 rows, so the selectivity term
  (`λ_sel`) contributed **exactly zero** across the entire program.

A composite that prices five axes but varies on three is not the Goodhart-resistant
metric CLAUDE.md §6 specifies.

## 3. The composite column is not always a composite

Rows 110–124 write method-specific metrics into the `composite` field (e.g. exp122's
`0.6529` is a mean cosine similarity). These rank against genuine composites in any
naive sort.

## 4. Instruments changed mid-log

`behavior_scorer` takes six different values across the log (generation, projection,
lexicon_proxy, gemini_judge, local_judge, geometry_cosine, gate_auc). The
`projection` scorer is **circular** by the code's own admission
(`eval.py:236`) — its delta is tautologically `alpha·‖v‖`. `safety_real=False` on
rows 110–124, and experiments 2–9 ran with safety stubbed.

## 5. The stored champion is not a valid champion

`best_config.json` is **exp3**: `Qwen2.5-0.5B-Instruct`, n=1, **stubbed safety**,
scored by the **circular projection proxy**. It was set 2026-05-30 and never
advanced across the following 121 experiments, yet every subsequent verdict was
written relative to it. It must not be treated as a global best, and it is retained
only as history.

## What is safe to cite

- **exps 118–119** (the AxBench E7 evaluation): real benchmark, off-family judge,
  full statistical contract computed. Judge weakness (AUC 0.68) is a stated caveat.
- **N17** (off-shell displacement ↔ incoherence, ρ=+0.585 on real WikiText-2):
  judge-independent geometry.
- **N5 falsified across scale** (held-out R² = −1.6).

Everything else in this log is single-seed screening on a substrate
(`gemma-3-270m`, 92/124 rows) that the program's own findings describe as
unsteerable, and should be treated as instrument development, not evidence.
