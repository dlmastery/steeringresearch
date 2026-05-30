# NEXT STEPS — unlocking the first scientific experiments

The harness, skills, ladder, ledger, and rich dashboard are live and the
**Rung-0/1 plumbing gate is cleared** (exp#1 on the offline FakeResidualLM:
intervention perturbs the residual stream, state restores exactly, the full
loop + 3-tier dashboard execute end-to-end). The first *scientific* steering
runs (E1–E3 on **Gemma-3-1B-it**) are blocked on two one-time, user-side
prerequisites that I cannot do non-interactively:

## 1. Install the 4-bit quantization dependency
```powershell
pip install bitsandbytes accelerate
```
(torch 2.6+cu124, transformers 4.55 are already present; the RTX 4090 Laptop
16 GB is detected.)

## 2. Accept the Gemma license + log in to Hugging Face (gated model)
In the Claude Code prompt, run it as a session command so the output lands here:
```
! huggingface-cli login
```
Then accept the license once at https://huggingface.co/google/gemma-3-1b-it
(and gemma-2-2b-it for the STANDARD rung). Alternatively set a token:
```powershell
$env:HF_TOKEN = "hf_..."
```

## 3. Smoke the real model (Rung-1)
```powershell
$env:PYTHONPATH="src"
python -m steering.runner --model google/gemma-3-1b-it --rung 1 `
  --behavior happiness --operation add --alpha 1.0 --source diffmean `
  --seed 0 --tag E0-smoke-gemma3-1b --description "first real Gemma smoke"
```
If it loads and the monotonicity/PPL/safety tripwires pass, we are cleared to
start the science.

## First three scientific experiments (already scaffolded under `ideas/`)

| id | dir | one-line falsifier | rung path |
|----|-----|--------------------|-----------|
| **E1** | `ideas/10_diffmean_paircount_knee/` | behavior at N=50 pairs ≥ 90% of N=256 asymptote | 1→2 |
| **E2** | `ideas/20_fisher_layer_selection/` | Spearman(Fisher-ratio, efficacy) ≥ 0.7 across 3 behaviors | 1→2 |
| **E3** | `ideas/30_alpha_coherence_cliff/` | identifiable α cliff: MMLU <2pp below it, super-linear PPL above | 1→2 |

Each already has a pre-registered `IDEA.md` (falsifier + predicted Δ +
composite fingerprint `a9001e87087e`). The execution order (CLAUDE.md /
IDEA_TABLE) is **E1–E8 first** (tooling), then the geometry priors N5/N3/N1,
then the conditional/stacking blocks.

## What runs without Gemma (today)
- `python -m pytest tests/` — 28 tests, the Rung-0 plumbing + dashboard gates.
- The full reasoning→runner→ledger→dashboard loop on `--model fake`.
- All authoring/audit/critic agent work (docs, skills, dashboards).

*Nothing in `corpus/` is assumed true — every inherited Gemma number is
`[NEEDS VERIFICATION]` until reproduced here on the 4090 ladder.*
