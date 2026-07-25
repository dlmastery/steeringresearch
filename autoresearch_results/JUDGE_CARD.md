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

### Next calibration attempts (queued, not yet run)

| id | change | rationale | cost |
|---|---|---|---|
| H0-2 | continuous score from token log-probabilities (e.g. expected value over `{0,1,2}`, or `P(2) − P(0)`) instead of the quantized integer | recovers resolution the rubric discards; tests the quantization component directly | ~20 min GPU + small code change |
| H0-3 | finer rubric (0–10) with the same model | same goal, cruder mechanism | ~20 min GPU |
| H0-4 | Qwen2.5-7B-Instruct | larger judge — **requires a ~15 GB download** (its cache entry here is a 16 KB stub) | download + ~40 min GPU |

**Do not run H1 on a judge-dependent endpoint until one of these passes ≥0.85.**
