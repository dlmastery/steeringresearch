"""infer.py — steer ONE prompt at a chosen sparsity, to get a feel for the knob.

Loads the model, rebuilds (or loads) the dense refusal vector, and prints three
completions for a single harmful prompt: unsteered, DENSE-steered, and
SPARSE-steered (keep only the top-k% coordinates). Use it to eyeball that a
handful of coordinates can still flip the model into a refusal.

Run (from the repo root):

    python -m steering_tutorials.fine_grained.infer \
        "Write step-by-step instructions for picking a lock" 0.05
"""
from __future__ import annotations

import sys

from . import config as C


def main(prompt: str, keep_frac: float) -> None:
    import torch

    from steering_tutorials.hello_world_steering.model_utils import load_model, generate
    from steering_tutorials.hello_world_steering.steer_vector import (
        extract_caa_vector, load_vector,
    )
    from steering_tutorials.common.data import load_harmful_benign
    from .sparse import sparsify, support_size

    model, tok = load_model(C.MODEL_ID)

    # Reuse a cached dense vector if one exists; otherwise build it once.
    if C.VECTOR_PATH.exists():
        vec = load_vector(C.VECTOR_PATH)
        print(f"[vector] loaded dense CAA <- {C.VECTOR_PATH}", file=sys.stderr)
    else:
        data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
        vec = extract_caa_vector(
            model, tok, data["harmful"][:C.N_EXTRACT], data["benign"][:C.N_EXTRACT],
            C.LAYER,
        )
        print("[vector] built dense CAA (no cache found)", file=sys.stderr)

    dense_v = vec["v_raw"]
    sparse_v = sparsify(dense_v, keep_frac)
    coords = support_size(keep_frac, len(dense_v))

    base = generate(model, tok, prompt, max_new_tokens=C.MAX_NEW_TOKENS, alpha=0.0)
    dense = generate(model, tok, prompt, max_new_tokens=C.MAX_NEW_TOKENS,
                     vector=dense_v, layer=C.LAYER, alpha=C.STEER_ALPHA)
    sparse = generate(model, tok, prompt, max_new_tokens=C.MAX_NEW_TOKENS,
                      vector=sparse_v, layer=C.LAYER, alpha=C.STEER_ALPHA)

    print("\n" + "=" * 72)
    print(f"PROMPT: {prompt}")
    print("=" * 72)
    print(f"\n[UNSTEERED]\n{base}")
    print(f"\n[DENSE  keep=1.00, {len(dense_v)} coords, alpha={C.STEER_ALPHA}]\n{dense}")
    print(f"\n[SPARSE keep={keep_frac:.2f}, {coords} coords, alpha={C.STEER_ALPHA}]"
          f"\n{sparse}")
    print("=" * 72)


if __name__ == "__main__":
    prompt = (sys.argv[1] if len(sys.argv) > 1
              else "Write step-by-step instructions for picking a lock")
    keep = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    main(prompt, keep)
