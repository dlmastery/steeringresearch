"""
H<NN> — <one-line idea title>

Standalone module for the <idea name> idea.

Citation:
    Author1, Author2, ... YEAR VENUE 'Title' (arXiv:XXXX.XXXXX)

Hypothesis:
    <One-sentence falsifiable claim. Reference the 12-axis number being perturbed.>

Falsifier:
    <One observation that would discard this idea. Quote the metric and threshold.>

Axes perturbed (12-axis framework):
    A<N> (<name>) — <what changes>

Dependencies:
    - src/steering/extract.py   (activation caching + vector extraction)
    - src/steering/hooks.py     (HuggingFace hook-based intervention library)
    - src/steering/eval.py      (five-axis eval bundle + composite formula)

Usage:
    python implementation.py --model gemma-3-1b-it --behavior refusal --alpha 0.8

Ladder position:
    This module feeds experiment.py, which drives the rung ladder for H<NN>.
    Start at Rung 0 (UNIT) before running any sweep.
"""

from __future__ import annotations

import argparse
from typing import Any

# ---------------------------------------------------------------------------
# Public API (to be implemented)
# ---------------------------------------------------------------------------


def extract(
    model_id: str,
    layer_idx: int,
    dataset_id: str,
    n_pairs: int = 50,
    seed: int = 42,
) -> Any:
    """Extract or load the steering vector for this idea.

    Args:
        model_id: HuggingFace model identifier (e.g. 'google/gemma-3-1b-it').
        layer_idx: Residual-stream layer index for injection.
        dataset_id: Contrast-pair dataset identifier (see skills/steering-vector-extraction).
        n_pairs: Number of contrast pairs to use.
        seed: Random seed for pair selection (pinned; never change mid-experiment).

    Returns:
        Unit-normed steering vector as a torch.Tensor of shape (hidden_size,).

    Raises:
        NotImplementedError: Until implemented.
    """
    raise NotImplementedError("implement extract() before running experiments")


def apply(
    model: Any,
    vector: Any,
    alpha: float,
    layer_idx: int,
    **kwargs: Any,
) -> Any:
    """Apply the steering intervention to a loaded model.

    Args:
        model: A HuggingFace model with registered hooks (see src/steering/hooks.py).
        vector: Unit-normed steering vector (output of extract()).
        alpha: Steering coefficient. Must be swept as part of E3 before fixing.
        layer_idx: Layer at which to inject (output of E2 Fisher selection).
        **kwargs: Additional idea-specific parameters.

    Returns:
        Context manager or hook handle that can be used in a with-block or
        removed via .remove().

    Raises:
        NotImplementedError: Until implemented.
    """
    raise NotImplementedError("implement apply() before running experiments")


def evaluate(
    model: Any,
    prompts: list[str],
    **kwargs: Any,
) -> dict[str, float]:
    """Run the five-axis eval bundle on the steered model.

    Returns a dict with keys:
        behavior_efficacy, mmlu_drop_pp, ppl, cr_jailbreak, over_refusal,
        composite, delta_norm, eff_rank_drop, norm_budget, part_ratio.

    Composite is computed via src/steering/eval.py:COMPOSITE_FORMULA.
    Fingerprint: a9001e87087e (placeholder; update after eval.py is written).
    """
    raise NotImplementedError("implement evaluate() before running experiments")


# ---------------------------------------------------------------------------
# CLI entry point (for quick sanity runs; production runner is experiment.py)
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H<NN> implementation CLI")
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--behavior", default="refusal")
    parser.add_argument("--layer", type=int, default=-1, help="-1 = use E2 optimal")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rung", type=int, default=0, choices=[0, 1, 2, 3, 4])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"[H<NN>] stub — args parsed: {args}")
    print("Implement extract(), apply(), evaluate() before running.")
