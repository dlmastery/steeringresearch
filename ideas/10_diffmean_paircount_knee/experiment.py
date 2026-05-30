"""
Experiment driver for H<NN> — <one-line idea title>

This script runs the ladder progression (Rung 0 -> 1 -> 2 -> ...) for H<NN>.
It enforces the 7-step ritual gate: the pre-run reasoning entry (Diagnose, Cite,
Hypothesize, Predict) must be filled in autoresearch_results/reasoning_annotations.json
BEFORE any experiment row is written to EXPERIMENT_LEDGER.md.

Usage:
    python experiment.py --rung 1 --seed 42
    python experiment.py --rung 2 --seeds 42 43 44
    python experiment.py --rung 3 --seeds 42 43 44 45 46 47 48

Ladder gates:
    Rung 0 (UNIT):     plumbing check; no model load required
    Rung 1 (SMOKE):    monotone effect + bounded PPL + no safety leak; 1-3 min
    Rung 2 (DEV):      beats baseline on held-out concepts; 10-20 min
    Rung 3 (STANDARD): Pareto-dominates prior method; 1-3 h; n >= 7
    Rung 4 (FULL):     full multi-axis win + ablations + red-team; half-day+

Outputs:
    experiments/<timestamp>/                -- per-run archive
        config.json                         -- full config snapshot
        metrics.json                        -- five-axis + geometry metrics
        reasoning_entry.json                -- 7-step entry snapshot
        samples.jsonl                       -- steered vs unsteered generations
    results.md is updated automatically after each run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Placeholder imports (will resolve once src/ modules are implemented)
# from src.steering.extract import extract_diffmean, compute_fisher_ratio_all_layers
# from src.steering.hooks import steer_context
# from src.steering.eval import evaluate_bundle, COMPOSITE_FORMULA

COMPOSITE_FINGERPRINT = "a9001e87087e"  # placeholder; update after eval.py written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H<NN> experiment driver")
    p.add_argument("--rung", type=int, required=True, choices=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--model", default="google/gemma-3-1b-it")
    p.add_argument("--behavior", default="refusal")
    p.add_argument("--layer", type=int, default=-1, help="-1 = use E2 optimal layer")
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--dry-run", action="store_true", help="validate args; do not run")
    return p.parse_args()


def _gate_check(rung: int, n_seeds: int) -> None:
    """Enforce ladder gate rules."""
    if rung >= 3 and n_seeds < 7:
        raise ValueError(
            f"Rung {rung} requires n >= 7 seeds for statistical validity; "
            f"got {n_seeds}. n <= 3 is SCREENING only."
        )
    if rung >= 3 and n_seeds < 7:
        raise ValueError("Rung 3+ requires n>=7 seeds (rigor contract)")


def _pre_run_gate() -> None:
    """Verify that the pre-run 7-step reasoning entry exists for this experiment."""
    reasoning_file = Path("autoresearch_results/reasoning_annotations.json")
    if not reasoning_file.exists():
        raise FileNotFoundError(
            "autoresearch_results/reasoning_annotations.json not found. "
            "Fill in the pre-run reasoning entry (Diagnose, Cite, Hypothesize, Predict) "
            "BEFORE launching the experiment. No --bypass."
        )
    # TODO: validate that the entry for this hypothesis/rung is present and complete


def run_rung(args: argparse.Namespace) -> dict:
    """Run the experiment at the specified rung.

    Returns a metrics dict matching the EXPERIMENT_LEDGER schema.
    """
    _gate_check(args.rung, len(args.seeds))
    _pre_run_gate()

    print(f"[H<NN>] Starting Rung {args.rung} | model={args.model} | "
          f"behavior={args.behavior} | alpha={args.alpha} | seeds={args.seeds}")
    print(f"[H<NN>] Composite fingerprint: {COMPOSITE_FINGERPRINT}")

    # TODO: implement the actual experiment
    # 1. Load model (args.model) at 4-bit
    # 2. Extract or load cached vector (implementation.extract())
    # 3. Apply steering (implementation.apply())
    # 4. Run eval bundle (implementation.evaluate())
    # 5. Log metrics to experiments/<timestamp>/metrics.json
    # 6. Append row to EXPERIMENT_LEDGER.md

    raise NotImplementedError(
        "implement run_rung() after implementation.py extract/apply/evaluate are done"
    )


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run:
        print(f"[H<NN>] Dry run OK — args: {args}")
        sys.exit(0)
    metrics = run_rung(args)
    print(json.dumps(metrics, indent=2))
