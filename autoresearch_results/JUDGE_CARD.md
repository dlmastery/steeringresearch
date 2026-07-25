# JUDGE CARD — instrument validity record (H0)

> Required by the program's own rule: **no claim may rest on an unvalidated
> instrument.** This card records every judge calibration attempted, passed or
> failed. A failed calibration is evidence and is never deleted.

---

## H0-1 — Qwen3-4B-Instruct-2507 — **FAILED**

| field | value |
|---|---|
| date | 2026-07-25 |
| judge model | `Qwen/Qwen3-4B-Instruct-2507` (7.6 GB, local cache, 4 safetensors) |
| harness | `scripts/validate_judge.py --judge local --dataset concept10 --concepts 10 --per 15` |
| interpreter | `C:\Users\evija\anaconda3\python.exe` (3.12.3, torch 2.6.0+cu124) |
| reference labels | AxBench `concept10`, 300 labelled items (150 positive) |
| **ROC-AUC** | **0.665** |
| positives mean concept score | 1.020 |
| negatives mean concept score | 0.440 |
| separation | +0.580 (on a 0–2 scale) |
| mean fluency | 1.830 / 2 |
| runtime | 1208.5 s (~20 min) on RTX 4090 |
| **gate** | **≥ 0.85 required → FAILED** (harness's own label: `WEAK`) |
| artifact | `tasks/b0yz55a9d.output` |

### Interpretation

**The judge upgrade did not fix the instrument.** The prior judge measured AUC 0.68;
this one measures 0.665. A newer, differently-trained 4B model is *no better* at
AxBench concept scoring, so the failure is **not** attributable to model vintage.

**Diagnosis — score quantization is a contributing cause.** The AxBench rubric emits an
integer concept score in `{0, 1, 2}` (verified from the judge cache: distribution
0→191, 1→68, 2→141 over the last 400 cached items). A large mean separation (+0.58)
combined with a poor AUC (0.665) is the signature of a **coarsely quantized score with
genuine class overlap**. Note that quantization alone does not cap AUC — a *perfect*
3-level judge would still reach AUC 1.0 — so part of this is real mis-scoring, and part
is resolution loss.

### Consequences (binding)

1. **No steering efficacy claim, and no steering NULL, may be reported using this
   judge.** At AUC 0.665 the instrument cannot resolve the effect sizes this program
   works with (the E7 direction effect was +0.004). This is precisely the error that
   invalidated the prior 124 experiments; it must not be repeated with a new model.
2. **H1 (the certified gate) is BLOCKED** on a judge-dependent endpoint. It may proceed
   only on judge-independent endpoints (see below).
3. Any prior result whose endpoint was this rubric inherits the same caveat.

### Judge-independent endpoints (the way forward)

The revised program's surviving legs are largely measurable **without** an LLM judge:

- **Gate quality** — ROC-AUC of the activation probe against *ground-truth* harmful/
  benign labels. No judge involved.
- **Over-refusal / refusal rate** — a rule-based refusal detector (or human-verified
  sample), not a concept-scoring LLM.
- **Coherence** — perplexity on real WikiText-2. Judge-free.
- **Matched-budget attribution** — displacement (geometry) vs perplexity. Judge-free.

**Recommendation:** re-scope the immediate program to judge-independent endpoints, and
treat judge repair as a separate, parallel instrument-engineering task.

---

## H0-2 — Qwen3-4B-Instruct-2507, **continuous readout** — **FAILED (but diagnostic)**

| field | value |
|---|---|
| date | 2026-07-25 |
| change vs H0-1 | **readout only.** Same model, same prompt, same 300 items, same labels. Integer argmax replaced by the expected value under the model's own distribution over `{"0","1","2"}` (`LocalJudge.score_axbench_expected`) |
| **ROC-AUC** | **0.7508** (H0-1: 0.665, **Δ +0.0858**) |
| distinct score values | **68** (H0-1: 3) |
| positives / negatives | 1.018 / 0.457 (separation +0.561) |
| runtime | **94 s** (H0-1: 1208.5 s — **12.8× faster**) |
| **gate** | **≥ 0.85 → FAILED** |
| artifact | `autoresearch_results/H0-2_continuous_judge.json` |

### What this establishes

The quantization hypothesis was **partially correct, and is now quantified**: recovering
the discarded resolution is worth **+0.086 AUC**, roughly 40% of the gap between H0-1 and
the 0.85 gate. Because only the readout changed, that delta is cleanly attributable to
quantization and nothing else.

**But the judge still fails at 0.751 with an optimal readout.** The residual deficit is
genuine mis-scoring of concept presence, not lost resolution. Combined with H0-1's
finding that a newer model generation did not help (0.68 → 0.665), the conclusion is:

> **An open ~4B judge cannot resolve AxBench concept presence to the standard this
> program requires. This is a property of the task and the model tier, not of the
> prompt, the rubric's scale, or the model's vintage.**

### Two things worth keeping regardless

1. **`score_axbench_expected` is strictly better than the generate-and-parse path** —
   higher AUC (+0.086) *and* 12.8× faster (one forward pass, no 16-token generate). It
   should be the default readout wherever this judge is used at all.
2. A judge at AUC 0.75 is still usable for **coarse triage** (e.g. filtering obvious
   gibberish), but never for resolving small effects. Any such use must cite this card.

---

## PROGRAM DECISION (2026-07-25): re-scope to judge-independent endpoints

Two calibrations have failed. The program does **not** get a third attempt at shopping
for a judge before doing science — that would be the sunk-cost version of the original
error. The surviving contributions are re-scoped onto endpoints that need no judge:

| leg | endpoint | judge? |
|---|---|---|
| H1 certified gate | probe ROC-AUC vs **ground-truth** harmful/benign labels; realised over-refusal vs the conformal bound | **no** |
| Matched-budget attribution | displacement (geometry) vs **WikiText-2 perplexity** | **no** |
| Coherence throughout | perplexity | **no** |
| Refusal detection | rule-based detector, human-verified on a sample | not an LLM concept-judge |

**Judge-dependent claims (behaviour efficacy, concept steering strength) are suspended
indefinitely**, and every historical result resting on them inherits this card's caveat.

### Remaining options, deliberately NOT taken now

| id | change | why not now |
|---|---|---|
| H0-3 | finer rubric (0–10) | H0-2 already shows resolution is not the binding constraint |
| H0-4 | Qwen2.5-7B (~15 GB download) | 7B→4B moved AUC by 0.015; low expected value for high cost |
| H0-5 | frontier API judge | breaks the offline/local constraint; revisit only if a claim truly requires it |

### Superseded plan (kept for the record)

| id | change | rationale | cost |
|---|---|---|---|
| H0-2 | continuous score from token log-probabilities (e.g. expected value over `{0,1,2}`, or `P(2) − P(0)`) instead of the quantized integer | recovers resolution the rubric discards; tests the quantization component directly | ~20 min GPU + small code change |
| H0-3 | finer rubric (0–10) with the same model | same goal, cruder mechanism | ~20 min GPU |
| H0-4 | Qwen2.5-7B-Instruct | larger judge — **requires a ~15 GB download** (its cache entry here is a 16 KB stub) | download + ~40 min GPU |

**Do not run H1 on a judge-dependent endpoint until one of these passes ≥0.85.**
