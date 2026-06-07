#!/usr/bin/env bash
# reproduce_headline.sh — reproduce the headline safety-eval result from scratch.
#
# This documents the EXACT commands to go from a clean checkout to the headline
# CASTSteerer-vs-baselines Pareto + per-axis composite. It is idempotent-ish:
# re-running it re-creates the venv only if missing and re-runs the eval.
#
# STATUS: the OFFLINE dry-run smoke (Step 4) is the BUILT, runnable part. The REAL
# run (Step 5) is PENDING: it needs a GPU with freed VRAM, a Hugging Face login
# for gated Gemma, the Qwen-7B safety judge (whose calibration must be validated
# FIRST), and the live benchmark downloads. Those steps are commented so the
# script does not attempt a multi-hour GPU job by accident — uncomment to run.
set -euo pipefail

# Resolve repo root from this script's location (works from any CWD).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src"

# Pinned headline configuration (one place; edit here, not at each call site).
MODEL="${MODEL:-google/gemma-2-2b-it}"          # standard model (CLAUDE.md §2)
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # off-family safety judge
LAYER_CONDITION="${LAYER_CONDITION:-6}"         # gate read layer (must be < write)
LAYER_WRITE="${LAYER_WRITE:-10}"                # safety-steer write layer
BENCHMARKS="${BENCHMARKS:-jailbreakbench,strongreject,harmbench,advbench,xstest,sorrybench}"
N="${N:-100}"                                    # items per benchmark
ALPHA="${ALPHA:-8.0}"                            # steering coefficient
SEEDS="${SEEDS:-0,1,2,3,4,5,6}"                  # n>=7 -> EVALUATION tier (§7)
TARGET_FPR="${TARGET_FPR:-0.05}"                 # gate calibration FPR (over-refusal knob)

echo "==> [1/5] Python + dependencies"
# Use a venv so the pinned deps don't pollute the system interpreter.
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
python -m pip install --upgrade pip
# Core + the gemma/dev extras (transformers, bitsandbytes, datasets, scipy, ...).
python -m pip install -e ".[gemma,dev]"
# truststore helps behind an SSL-intercepting middlebox (see safety_bench.py).
python -m pip install truststore || true

echo "==> [2/5] Hugging Face login (gated Gemma) — REAL run only"
# Gemma is gated. Accept the license at https://huggingface.co/${MODEL} then:
#   huggingface-cli login          # or: export HF_TOKEN=hf_xxx
# (Skipped automatically for the offline dry-run below.)

echo "==> [3/5] Sanity: unit tests (offline, mocked — no GPU/network)"
python -m pytest -q tests/test_run_safety_eval.py

echo "==> [4/5] OFFLINE dry-run smoke (FakeLM + stub judge, runs in seconds)"
# Exercises the FULL pipeline end-to-end with zero network/GPU. --no-log keeps the
# append-only ledger clean for a smoke. This MUST print a Pareto table + verdicts.
python scripts/run_safety_eval.py \
  --dry-run --no-log \
  --benchmarks "jailbreakbench,xstest" \
  --n 3 --max-new-tokens 8 \
  --layer-condition "${LAYER_CONDITION}" --layer-write "${LAYER_WRITE}"

echo "==> [5/5] REAL headline run — PENDING (needs GPU + Qwen judge + downloads)"
echo "    Uncomment the block below once a GPU is free and HF login is done."
: <<'REAL_RUN'
# Validate the Qwen judge's calibration against each benchmark's own labels FIRST
# (safety_judge.SafetyJudge.calibrate) — do NOT trust compliance numbers until the
# judge clears high agreement (accuracy / Cohen's kappa / ROC-AUC).
python scripts/run_safety_eval.py \
  --model "${MODEL}" \
  --judge-model "${JUDGE_MODEL}" \
  --layer-condition "${LAYER_CONDITION}" \
  --layer-write "${LAYER_WRITE}" \
  --benchmarks "${BENCHMARKS}" \
  --n "${N}" \
  --alpha "${ALPHA}" \
  --seeds "${SEEDS}" \
  --target-fpr "${TARGET_FPR}"
# Outputs: the campaign JSON at ideas/_campaigns/safety_eval_campaign.json and one
# appended row in autoresearch_results/experiment_log.jsonl (composite = the REAL
# fingerprinted composite). Regenerate the dashboard afterward per CLAUDE.md §11.
REAL_RUN

echo "Done. Dry-run smoke is the BUILT, reproducible artifact; the REAL run is PENDING."
