# steeringresearch

Autonomous, principled autoresearch on **conditional / activation steering of
small Gemma models** (Gemma-3-270M-it / Gemma-3-1B-it) on a single **RTX 4090
Laptop (16 GB)**. Publication-grade rigor, a CIFAR-style benchmark ladder, and a
transparent multi-page dashboard.

- **Repo:** https://github.com/dlmastery/steeringresearch
- **Live dashboard (GitHub Pages):** https://dlmastery.github.io/steeringresearch/
- **Status:** methodology + harness + **n=1 screening** results; external steering
  claims are gated on the required experiments (see `FINDINGS.md`). Composite
  fingerprint `a9001e87087e`.

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
