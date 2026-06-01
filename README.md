# steeringresearch

**The goal: discover and engineer state-of-the-art conditional / activation
steering for LLMs** — by distilling the strengths of the existing literature
(CAA, CAST, the geometry wave, the Rogue-Scalpel safety work, SAE steering) into
testable mechanisms, then finding which ones actually hold up. We pursue this as
an **autonomous AI research program** (driven by Claude Code) on **small Gemma
models** (Gemma-3-270M-it / Gemma-3-1B-it) on a single **RTX 4090 Laptop (16 GB)**:
it forms 70 falsifiable hypotheses drawn from the literature, screens them on a
CIFAR-style benchmark ladder, prices every result with a Goodhart-resistant
composite that no method can cheat, and publishes the entire evidence trail to a
transparent, clickable, multi-page dashboard.

**What "SOTA steering" means here, concretely:** a recipe that maximizes the
*intended behavior change* while paying the least *capability, coherence, safety,
and selectivity* tax — i.e. control that doesn't break the model. The program's
job is to find, falsify, and combine the mechanisms (which layer, which direction,
how much, add-vs-rotate, when to gate, how to stack) that win on that trade-off,
and to do so with publication-grade rigor so the winners are real, not noise.

## 🚀 Start here — everything is reachable from these links

| | link |
|---|---|
| 📊 **Live dashboard** (master → per-hypothesis → per-experiment) | https://dlmastery.github.io/steeringresearch/dashboard/ |
| 🏠 **Landing page** | https://dlmastery.github.io/steeringresearch/ |
| 📚 **Awesome LLM Steering** — curated literature survey (60 papers, arXiv links) | https://dlmastery.github.io/steeringresearch/awesome-steering.html |
| 🧠 **Mindmap** — the steering landscape (12 axes, method families, hypotheses) | https://dlmastery.github.io/steeringresearch/mindmap.html |
| 📄 **Paper** (honest ICML methods / harness / screening draft) | [paper/PAPER.md](paper/PAPER.md) |
| 🔬 **FINDINGS** (rigor-gated; S-1…S-14 + rung-3) | [FINDINGS.md](FINDINGS.md) |
| 🧪 **Hypothesis registry** (70 hypotheses E1–E50 + N1–N20 + verdicts) | [IDEA_TABLE.md](IDEA_TABLE.md) |
| 📐 **70 hypothesis design docs** (12-section each) | [hypotheses/](hypotheses/) |
| ✅ **Verification rubrics** (A–E) + the ICML sign-off | [audits/RUBRICS.md](audits/RUBRICS.md) · [audits/ICML_SIGNOFF_v2.md](audits/ICML_SIGNOFF_v2.md) |
| ♻️ **Reusable meta-skill pack** (domain-agnostic autoresearch process) | [meta-skills/](meta-skills/) |
| 🛠️ **Constitution** (rules, composite, ladder, dashboard mandate) | [CLAUDE.md](CLAUDE.md) |

**Status:** methodology + reproducible harness + **n=1 SCREENING** results across
**19 hypotheses with two-sided verdicts** (11 SUPPORTED · 3 FALSIFIED · 3 PARTIAL ·
1 DIRECTIONAL · 1 INCONCLUSIVE) plus the program's **first rung-3 evaluation**
(N17 on real WikiText-2). External steering claims remain gated on the required
experiments in `FINDINGS.md`. **ICML reviewer: unconditional sign-off, Rubric E
8/8.** Composite fingerprint `a9001e87087e`.

## What's been learned (headline screening results)

- **N17** (off-shell Δ‖h‖ predicts incoherence) — the strongest result; survives the
  **rung-3 evaluation on real WikiText-2** (Spearman +0.585, 95% CI [0.35, 0.76]).
- **N5's "universal law"** — honestly **falsified across scale** (held-out R²=−1.6);
  the screening R²=0.81 was a within-pool artifact that rung-3 rigor caught.
- **E4** (DiffMean ≈ PCA-top1) holds across **4 behaviors × 3 models** (cos 0.995–0.999).
- **E7** relative steering — a clean ‖h‖-independent cliff; the knee is **scale-invariant
  at ~10 % of ‖h‖**. **E17** two-vector stacking retains 101 %/110 % (no interference).
- **Falsified honestly:** **E2** (max-Fisher ≠ best layer), **E27**-rotation, **E28** (not low-rank).

## Architecture — a SHARED HARNESS, not per-hypothesis code

Everything is tested through **one shared harness** (`src/steering/`, ~10 modules),
the same design as the reference `dlmastery/autoresearch*` projects — NOT via a
separate code file per hypothesis. So "few files in `src/`" is by design: those 10
modules (intervention `hooks`, `extract`, `geometry`, the 5-axis `eval` + composite,
the `runner`, the `dashboard`) are the complete machinery, and every hypothesis is
screened by composing them via the drivers in `scripts/` (`campaign_sweep`,
`run_hillclimb`, `rung3_n17`). The 70 hypothesis **design docs** live in
`hypotheses/`; the **dashboard pages** (72) are generated from the experiment log.
The ~50 still-UNTESTED hypotheses are honest about *what new harness code they need*
(CAST gating, multi-vector orchestration, SAE features, a calibrated LLM judge,
hypernetworks) — that missing infra is exactly why they are marked UNTESTED, not
silently skipped.

## Layout

| path | what |
|---|---|
| `CLAUDE.md` | the constitution — rules, winner def, dashboard mandate, rigor floor |
| `AUTORESEARCH_PROCESS.md` | the detailed inner loop |
| `meta-skills/` | **domain-agnostic** autoresearch process pack (reusable for any topic) |
| `skills/` | steering-specific instantiation skills |
| `corpus/` | the verbatim steering research corpus (literature, 12 axes, ladder, datasets, Rogue Scalpel, N1–N20) |
| `src/steering/` | the harness: intervention, extraction, eval bundle, runner, dashboard |
| `tests/` | Rung-0 unit tests |
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
`meta-skills/autoresearch-meta/SKILL.md` (the portable process spine) → `CLAUDE.md`
(the constitution) → `IDEA_TABLE.md` (the 70 hypotheses + status) → `FINDINGS.md`
(the screening observations). All inherited corpus numbers are `[NEEDS VERIFICATION]`
until reproduced on the ladder.

## Docs, paper, and verification

| link | what |
|---|---|
| `docs/index.html` | GitHub-Pages landing page → dashboard, paper, findings, audits |
| `docs/dashboard/index.html` | the live multi-page dashboard (master + per-hypothesis + per-experiment) |
| `paper/PAPER.md` | honest ICML-format methods / harness / **n=1 screening** draft (no overclaiming) |
| `FINDINGS.md` | rigor-gated findings (currently zero external-ready; S-1..S-8 are screening-only) |
| `audits/RUBRICS.md` | the scoreable PASS/FAIL rubrics A–E |

Status: **methodology + harness + n=1 screening; external steering claims are gated**
on the required experiments enumerated in the paper and `FINDINGS.md`. Composite
formula fingerprint `a9001e87087e`.

Verify mechanically:

```bash
python scripts/verify_dashboard.py   # Rubric B (dashboard) — PASS/FAIL table
python scripts/verify_rubrics.py      # Rubrics A/C/D — scorecard (ruff+mypy+pytest+secrets+fingerprint)
```
