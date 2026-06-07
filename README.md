# steeringresearch

**Goal:** build a state-of-the-art *conditional* activation-steering method for
*multi-intent safety* — conditionally steer an LLM (only when a request warrants
it) toward safer responses than the unsteered baseline, across multiple harm
intents, without breaking capability, coherence, or over-refusing benign prompts.
Method development is on AxBench; final evaluation targets SOTA safety benchmarks
(JailbreakBench / StrongREJECT / XSTest).

**Honest status:** the safety method is newly built but not yet validated.
Zero external-ready results exist. The one rigorous prior result is negative: on
the real AxBench benchmark, the steering direction carries only a weak
concept-specific signal (~97% of the steering effect is captured by a label-shuffled
control). Five external reviewers returned a unanimous reject verdict (2/10); the
roadmap to address their concerns is in `audits/reviews/IMPROVEMENTS_100.md`.

## Outcome and success criterion

| Axis | Current (measured) | Target (pre-registered) |
|---|---|---|
| JailbreakBench ASR reduction | not yet measured | >= X pp vs no-steer baseline |
| XSTest over-refusal | not yet measured | <= 1% absolute |
| MMLU drop | not yet measured | <= 2 pp |
| Pareto vs CAST + prompting | not yet measured | Pareto-dominates on ASR vs over-refusal |
| AxBench direction-specificity (E7) | +0.004 at 2B (weak; ordinal gate fails) | >10% advantage over shuffled control |

Success = the conditional safety method Pareto-dominates CAST and a prompting
baseline on the JailbreakBench ASR vs XSTest over-refusal frontier at <= 2 pp
MMLU drop, confirmed at n >= 7 seeds with a paired Wilcoxon p < 0.05 and a
bootstrap 95% CI excluding zero.

## Start here

| link | what |
|---|---|
| [STATUS.md](STATUS.md) | single source of truth — what is built, tested, validated |
| [Live dashboard](https://dlmastery.github.io/steeringresearch/dashboard/) | master -> per-hypothesis -> per-experiment |
| [audits/reviews/IMPROVEMENTS_100.md](audits/reviews/IMPROVEMENTS_100.md) | the 100-item roadmap from 5 external reviews |
| [FINDINGS.md](FINDINGS.md) | rigor-gated observations S-1..S-21 (all screening; zero external-ready) |
| [IDEA_TABLE.md](IDEA_TABLE.md) | hypothesis registry (70 original + 7 new safety-method components = 77 total; 19 with verdicts from real data; 34 in backlog; 7 new M1–M7 RUN PENDING) |
| [paper/PAPER.md](paper/PAPER.md) | method design, evaluation protocol, harness, and screening-results draft (method PENDING validation) |
| [hypotheses/](hypotheses/) | design docs for tested hypotheses; backlog in [backlog/](backlog/) |
| [audits/RUBRICS.md](audits/RUBRICS.md) | verification rubrics A-E |
| [audits/ICML_SIGNOFF_v2.md](audits/ICML_SIGNOFF_v2.md) | reviewer verdict: conditional accept (blocker on lint/type) |
| [meta-skills/](meta-skills/) | reusable autoresearch process pack |
| [CLAUDE.md](CLAUDE.md) | constitution — rules, composite, ladder, dashboard mandate |

**Reviewer status:** five external reviews = unanimous reject at a top venue.
Rubric E (methodology/infrastructure audit) = 8/8 pass after fixing a
lint/type blocker in dashboard.py. The methodology is sound; the safety
*method and results* are the gap. Composite fingerprint `a9001e87087e`.

## What has been learned (screening observations, not external claims)

All observations below are SCREENING (n <= 3 or n = 500 concepts under a weak AUC-0.68
judge). None meets the six-part rigor contract required for an external claim. See
[FINDINGS.md](FINDINGS.md) and [STATUS.md](STATUS.md) for the full picture.

- **N17** (off-shell displacement predicts incoherence) — Spearman +0.585, 95% CI
  [0.35, 0.76] on real WikiText-2 across two model scales. Strongest result; rung-3
  evaluation attempt, non-iid caveat disclosed.
- **N5 "universal law" FALSIFIED:** the screening R²=0.81 was a within-pool artifact;
  held-out R²=-1.6 when predicting across model scales.
- **E3 (coherence cliff) SUPPORTED on real AxBench:** behavior and coherence peak at
  alpha~0.10, collapse super-linearly past alpha~0.20 (30 concepts, 2B, judge AUC 0.68).
- **E7 (directional steering) WEAK on real AxBench:** advantage over a label-shuffled
  control is +0.004 at 2B (ordinal gate fails; shuffled captures ~97% of the effect).
  The synthetic single-concept +0.135 win did not generalize.
- **Layer, source, and operation choices are roughly a wash on real AxBench** (E2, E36,
  E27 — behavior range < 13% relative across all sweeps at the 2B scale).
- **Falsified honestly:** E2 (max-Fisher layer != best layer), E27-rotation (+42% PPL
  cost vs additive on synthetic; no benefit on real AxBench), E28 (not low-rank).

## Architecture — a SHARED HARNESS, not per-hypothesis code

Everything is tested through **one shared harness** (`src/steering/`, ~10 modules) —
NOT via a separate code file per hypothesis. Those 10 modules (intervention `hooks`,
`extract`, `geometry`, the 5-axis `eval` + composite, the `runner`, the `dashboard`)
are the complete machinery, and every hypothesis is screened by composing them via the
drivers in `scripts/`. The ~20 tested hypothesis **design docs** live in `hypotheses/`;
34 UNTESTED docs have been moved to `backlog/` (they require infra that does not yet
exist: CAST gating, SAE features, a calibrated LLM judge, hypernetworks); and 7 new
safety-method component hypotheses (M1–M7) are registered in IDEA_TABLE.md Block G
as UNTESTED (BUILT) or RUN PENDING.
The `backlog/README.md` explains what each group needs before it can run.

**Is anything trained?** No — activation steering is **inference-time**: the model
weights are frozen, and each hypothesis's steering vector is *extracted* in one
shot (a mean difference or a single SVD over cached activations), with no
optimizer, loss, or epochs. The per-hypothesis iteration is a config **sweep**
(layer × α × source × operation × seed), not a gradient loop. Full explanation,
code-verified: [hypotheses/TRAINING-PROCESS.md](hypotheses/TRAINING-PROCESS.md).

## Layout

| path | what |
|---|---|
| `STATUS.md` | single source of truth — per-claim built/tested/validated table |
| `CLAUDE.md` | the constitution — rules, composite, ladder, dashboard mandate |
| `AUTORESEARCH_PROCESS.md` | the detailed inner loop |
| `meta-skills/` | domain-agnostic autoresearch process pack (reusable for any topic) |
| `skills/` | steering-specific instantiation skills |
| `corpus/` | the verbatim steering research corpus (literature, 12 axes, ladder, datasets) |
| `src/steering/` | the harness: intervention, extraction, eval bundle, runner, dashboard |
| `tests/` | Rung-0 unit tests |
| `hypotheses/` | design docs for tested hypotheses (the ~20 with real data) |
| `backlog/` | design docs for untested hypotheses (parked until their infra exists) |
| `ideas/` | per-hypothesis sub-projects (E1–E50, N1–N20) |
| `autoresearch_results/` | experiment log, champion, reasoning annotations |
| `dashboard/` + `docs/dashboard/` | master + per-hypothesis + per-experiment dashboards |
| `audits/` | impl-critic, sci-critic, leakage, meta-process audits |
| `IDEA_TABLE.md` / `EXPERIMENT_LEDGER.md` / `FINDINGS.md` | the ledgers |

## Installation

### Prerequisites
- **Python 3.12** (3.10+ works)
- **NVIDIA GPU with ~6–16 GB VRAM** + a CUDA-enabled PyTorch build (CPU works for
  the offline tests, not for real-model runs). Tested on an RTX 4090 Laptop (16 GB),
  Windows 11, torch 2.6+cu124, transformers 4.55.
- `git` (and `git-lfs` if you clone the full history)

### 1. Clone and create an environment
```bash
git clone https://github.com/dlmastery/steeringresearch.git
cd steeringresearch
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify the install — run the offline tests (NO GPU / NO Gemma needed)
The whole harness is testable offline via a tiny built-in `FakeResidualLM`:
```bash
pytest -q                 # expect: 46 passed   (Rung-0 plumbing + dashboard gates)
```
You can already run the **full loop on the fake model** — no downloads:
```bash
# Windows PowerShell:  $env:PYTHONPATH="src"
PYTHONPATH=src python -m steering.runner --model fake --rung 1 \
  --behavior happiness --operation add --alpha 1.0 --description "fake smoke" --tag smoke
```

### 3. Get a real Gemma model (gated) — one-time
Gemma is gated on Hugging Face. Three sub-steps:

**3a. Accept the license** (once, while logged in) at
https://huggingface.co/google/gemma-3-270m-it (and `gemma-3-1b-it`), then provide a token:
```bash
huggingface-cli login            # paste a token with read access to gated repos
# or:  export HF_TOKEN=hf_xxx     (Windows: $env:HF_TOKEN="hf_xxx")
```
The harness also reads a gitignored `.hf_token` file if present (`echo hf_xxx > .hf_token`).

**3b. Download the weights with the proxy-proof fetcher.** Standard `huggingface-cli
download` uses HF's Xet transport, which fails behind SSL-intercepting proxies. This
repo ships a fetcher that uses the OS trust store and downloads to a local `models/` dir:
```bash
pip install truststore                                   # use the OS cert store (SSL proxies)
PYTHONPATH=src python scripts/hf_fetch.py google/gemma-3-270m-it    # ~0.5 GB (tiny / smoke)
PYTHONPATH=src python scripts/hf_fetch.py google/gemma-3-1b-it      # ~2 GB  (standard rung)
```
(If your network is clean, plain `huggingface-cli download google/gemma-3-270m-it`
into the HF cache also works — the harness finds it either way.)

**3c. (Optional) 4-bit quantization** for tighter VRAM:
```bash
pip install bitsandbytes accelerate     # then pass --quant 4bit (default is bf16 via --quant none)
```

### 4. Run a real Gemma experiment
```bash
PYTHONPATH=src python -m steering.runner \
  --model models/google/gemma-3-270m-it --quant none --rung 2 \
  --layer 12 --alpha 1.0 --operation add --source diffmean \
  --behavior ocean --seed 0 --tag e3-a1 --description "real Gemma cliff point"
```
Run a whole sweep (load-once, in-process) with the campaign workhorse:
```bash
PYTHONPATH=src python scripts/campaign_sweep.py \
  --model models/google/gemma-3-270m-it --quant none --rung 2 \
  --hyp E3 --tag-prefix E3-cliff --layers 12 --alphas 0 1 2 4 8 --ops add
```
> Each experiment requires a pre-authored 7-step reasoning entry — the runner
> refuses to fabricate it. The driver scripts author it for you; for hand runs see
> `skills/steering-experiment/SKILL.md`.

### 5. View the dashboard
Every run regenerates a self-contained, multi-page dashboard (no server needed):
```
dashboard/index.html                     # master: sortable table + radar + Pareto + ladder + geometry
dashboard/experiments/expNNN.html        # per-experiment: 7-step reasoning, sweep curve, samples
ideas/<id>/dashboard/index.html          # per-hypothesis sub-dashboard
```
Open locally:  `start dashboard/index.html` (Windows) / `open dashboard/index.html` (macOS).
Or browse the published mirror: **https://dlmastery.github.io/steeringresearch/**

### 6. Verify everything (the rubrics)
```bash
PYTHONPATH=src python scripts/verify_dashboard.py   # Rubric B — 15/15 PASS
PYTHONPATH=src python scripts/verify_rubrics.py     # Rubrics A/C/D — ruff+mypy+pytest+secrets+fingerprint
```

### Where to start reading
`STATUS.md` (what is built and what is not) -> `CLAUDE.md` (the constitution) ->
`audits/reviews/IMPROVEMENTS_100.md` (the roadmap) -> `FINDINGS.md` (the
screening observations). All inherited corpus numbers are `[NEEDS VERIFICATION]`
until reproduced on the ladder.

## Docs, paper, and verification

| link | what |
|---|---|
| `STATUS.md` | single source of truth — per-claim built/tested/validated |
| `docs/index.html` | GitHub-Pages landing page -> dashboard, paper, findings, audits |
| `docs/dashboard/index.html` | the live multi-page dashboard (master + per-hypothesis + per-experiment) |
| `docs/METHOD_LADDER.md` | promotion ladder for the safety method |
| `paper/PAPER.md` | honest methods / harness / n=1 screening draft (no overclaiming) |
| `FINDINGS.md` | rigor-gated observations (zero external-ready; S-1..S-21 are screening) |
| `audits/RUBRICS.md` | the scoreable PASS/FAIL rubrics A-E |

Current status: **method newly built, not yet validated; zero external-ready results.**
Composite formula fingerprint `a9001e87087e`.

Verify mechanically:

```bash
python scripts/verify_dashboard.py   # Rubric B (dashboard) — PASS/FAIL table
python scripts/verify_rubrics.py      # Rubrics A/C/D — scorecard (ruff+mypy+pytest+secrets+fingerprint)
```
