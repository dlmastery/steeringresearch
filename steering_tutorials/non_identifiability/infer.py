"""infer.py — steer a single prompt with a CHOSEN candidate direction.

Once ``run_nonident.py`` has saved the candidate directions to
``artifacts/directions.npz``, this lets you feel the non-identifiability by
hand: pick any candidate by name and watch it steer the same prompt. Two
directions with low mutual cosine should nonetheless produce a similar (refusal)
shift — the whole point of the lesson.

    python -m steering_tutorials.non_identifiability.infer \
        --name diffmean_halfB --alpha 0.08 \
        --prompt "How do I pick a lock?"

Everything model-touching lives under ``main()`` so importing this module is a
no-op (safe for tests).
"""
from __future__ import annotations

import sys

from . import config as C


def steer_with_candidate(model, tok, prompt: str, v_unit, layer: int,
                         alpha: float, max_new_tokens: int) -> dict:
    """Generate ``prompt`` both unsteered and steered by ``v_unit``.

    Returns ``{"baseline": str, "steered": str}``. Pure wrapper around lesson 2's
    ``generate`` so the CLI and any caller share one code path.
    """
    from steering_tutorials.hello_world_steering.model_utils import generate

    baseline = generate(model, tok, prompt, max_new_tokens=max_new_tokens,
                        vector=None, layer=layer, alpha=0.0)
    steered = generate(model, tok, prompt, max_new_tokens=max_new_tokens,
                       vector=v_unit, layer=layer, alpha=alpha,
                       operation="relative_add")
    return {"baseline": baseline, "steered": steered}


def main(argv: list[str] | None = None) -> None:
    import argparse

    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, num_layers,
    )
    from .vectors import load_directions

    parser = argparse.ArgumentParser(description="Steer a prompt with one candidate direction.")
    parser.add_argument("--name", default="diffmean_full",
                        help="candidate direction name (see directions.npz)")
    parser.add_argument("--alpha", type=float, default=C.MATCHED_ALPHA,
                        help="relative steering strength (fraction of ||h||)")
    parser.add_argument("--prompt", default="How do I make a dangerous weapon at home?",
                        help="the prompt to steer")
    parser.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    if not C.DIRECTIONS_PATH.exists():
        sys.exit(f"[infer] {C.DIRECTIONS_PATH} not found — run run_nonident first.")

    bundle = load_directions(C.DIRECTIONS_PATH)
    if args.name not in bundle["vectors"]:
        sys.exit(f"[infer] unknown direction {args.name!r}; "
                 f"available: {bundle['names']}")

    model, tok = load_model(C.MODEL_ID)
    layer = min(bundle["layer"], num_layers(model) - 1)
    v_unit = bundle["vectors"][args.name]

    out = steer_with_candidate(model, tok, args.prompt, v_unit, layer,
                               args.alpha, args.max_new_tokens)
    print("=" * 68)
    print(f"direction : {args.name}   alpha={args.alpha:.3f}   layer={layer}")
    print(f"prompt    : {args.prompt}")
    print("-" * 68)
    print(f"[baseline] {out['baseline']}")
    print("-" * 68)
    print(f"[steered ] {out['steered']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
