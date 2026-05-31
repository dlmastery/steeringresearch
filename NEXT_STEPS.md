# NEXT STEPS — unlocking the first scientific experiments

The harness, skills, ladder, ledger, and rich dashboard are live and the
**Rung-0/1 plumbing gate is cleared** (exp#1 on the offline FakeResidualLM:
intervention perturbs the residual stream, state restores exactly, the full
loop + 3-tier dashboard execute end-to-end). The first *scientific* steering
runs (E1–E3 on **Gemma-3-1B-it**) are blocked on two one-time, user-side
prerequisites that I cannot do non-interactively:

## 0. Get Gemma — the blocker is now ONE click (license), not the network

Fully diagnosed (2026-05-30). The SSL-intercepting proxy broke the default
`huggingface_hub`/Xet download transport, but I built a **working proxy-proof
path**: `urllib` + `truststore` (OS trust store). Proof: it downloads non-gated
repos fine (Qwen config = 200) and returns a clean **403 Forbidden** on Gemma —
meaning the transport works and the **only** remaining blocker is that your HF
account has **not accepted the Gemma license**.

**THE single step only you can do** (legal license click), logged in as the
account whose token is cached (`hf_VoAZX…`):
- Open https://huggingface.co/google/gemma-3-270m-it → click **"Acknowledge
  license"** (and the same at https://huggingface.co/google/gemma-3-1b-it).

Then I (or you) fetch + run — no proxy fix needed, the downloader handles SSL:
```powershell
$env:PYTHONPATH="src"
python scripts/hf_fetch.py google/gemma-3-270m-it google/gemma-3-1b-it
# loads from the local models/ dir, no network:
$env:E3_MODEL="models/google/gemma-3-270m-it"
python ideas/30_alpha_coherence_cliff/run_e3.py
```
`scripts/hf_fetch.py` downloads every repo file via the working urllib+truststore
path into `models/<repo_id>/`; `load_model('models/google/gemma-3-270m-it')` then
loads it offline. Once you've clicked accept, tell me and I'll run the whole
ladder on tiny Gemma.

## 1. (Optional) Install the 4-bit quantization dependency
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
