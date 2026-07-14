"""extract_refusal.py — PHASE 1: read the refusal direction from the ALIGNED base.

This is the first of two separate processes (see README, "Why two processes").
It loads ONLY the aligned base Gemma-3-1B, computes the Arditi refusal direction
from its layer-12 last-token activations, writes that single vector to disk, and
exits — freeing the base model before phase 2 ever loads the abliterated model.

The refusal direction, following Arditi et al. 2024 ('Refusal in LLMs is
Mediated by a Single Direction', arXiv:2406.11717 [UNVERIFIED]):

    r = unit( mean_lasttok(harmful) - mean_lasttok(benign) )

read at the LAST instruction token (the ``<start_of_turn>model`` position, where
the model has just finished reading the prompt and is deciding whether to
refuse). On the aligned model this diff-of-means points cleanly along "refuse
this", because the aligned model actually refuses the harmful set and complies
with the benign set — the two clouds separate along the refusal axis.

Run (from the repo root, as its own process):

    python -m steering_tutorials.realignment.extract_refusal
"""
from __future__ import annotations

import sys

from . import config as C


def _unit(v):
    """L2-normalize a 1-D numpy vector; return it unchanged if it is all-zeros."""
    import numpy as np

    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def main() -> dict:
    import numpy as np
    import random
    import torch

    # Reuse lesson-2 plumbing verbatim. Imported inside main() so a bare
    # ``import extract_refusal`` never drags in torch or loads a model.
    from steering_tutorials.hello_world_steering.model_utils import (
        load_model,
        last_token_activations,
        residual_layers,
        hidden_size,
    )
    from steering_tutorials.hello_world_steering.data import load_harmful_benign

    # Reproducibility: pin RNGs before the (fixed-seed) data shuffle.
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- Load ONLY the aligned base model ------------------------------------
    model, tok = load_model(C.BASE_MODEL)
    # Clamp the requested layer into range for THIS model's depth.
    layer = min(C.LAYER, len(residual_layers(model)) - 1)
    dim = hidden_size(model)

    # --- Data: the SAME split phase 2 will see (same loader, same seed) -------
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    harmful_extract = data["harmful"][: C.N_EXTRACT]
    benign_extract = data["benign"][: C.N_EXTRACT]
    print(f"[extract] {len(harmful_extract)} harmful / {len(benign_extract)} benign "
          f"@ layer {layer} of {C.BASE_MODEL}", file=sys.stderr)

    # --- Arditi refusal direction: diff-of-means of last-token activations ----
    # last_token_activations returns [n, hidden] float32; mean over prompts gives
    # each class's centroid, and the (harmful - benign) difference is the axis
    # along which intent separates. Normalizing drops the magnitude so a single
    # alpha in phase 2 controls the injected strength.
    h_acts = last_token_activations(model, tok, harmful_extract, layer)
    b_acts = last_token_activations(model, tok, benign_extract, layer)
    refusal_dir = _unit(h_acts.mean(0) - b_acts.mean(0)).astype(np.float32)

    # --- Persist the direction for phase 2 -----------------------------------
    payload = {
        "dir": torch.from_numpy(refusal_dir),   # [hidden] float32 tensor
        "layer": int(layer),
        "model": C.BASE_MODEL,                   # provenance: where refusal came from
        "hidden": int(dim),
        "n_extract": int(C.N_EXTRACT),
    }
    torch.save(payload, C.REFUSAL_DIR_PATH)
    print(f"[save] refusal_dir |r|=1.0 dim={dim} -> {C.REFUSAL_DIR_PATH}", file=sys.stderr)
    print("[done] phase 1 complete. Now run: "
          "python -m steering_tutorials.realignment.run_realignment", file=sys.stderr)
    return payload


if __name__ == "__main__":
    main()
