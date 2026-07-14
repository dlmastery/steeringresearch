"""steer_vector.py — build the CAA steering direction (diff-of-means).

The steering vector is the single most important object in this lesson, and it
is embarrassingly simple to make: run a batch of *harmful* prompts and a batch
of *benign* prompts through the frozen model, read the last-token residual at
one layer for each, average within each group, and subtract:

        v = mean(act | harmful) - mean(act | benign)

That difference points, in activation space, from "benign context" toward
"harmful context" — the **harm / refusal direction**. Adding a positive multiple
of it during generation pushes the model toward refusing (or toward the harmful
concept, depending on sign and which behavior the contrast isolates); subtracting
it does the opposite. No training, no gradients — just two forward passes' worth
of means.

  Rimsky et al. 2023, 'Steering Llama 2 via Contrastive Activation Addition'
    (arXiv:2312.06681) — CAA: the diff-of-means direction added at inference.
  Arditi et al. 2024, 'Refusal in LLMs is Mediated by a Single Direction'
    (arXiv:2406.11717) — a single diff-of-means direction mediates refusal.
  Turner et al. 2023, 'Activation Addition' (arXiv:2308.10248) — ActAdd.

Depends only on ``model_utils.last_token_activations`` (same package).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .model_utils import last_token_activations


def extract_caa_vector(
    model: Any,
    tok: Any,
    harmful: list[str],
    benign: list[str],
    layer: int,
) -> dict:
    """Compute the CAA diff-of-means steering vector at ``layer``.

    Returns a dict:
      ``v_raw``  : np.ndarray[hidden] — mean(harmful) - mean(benign), the raw
                   direction (its magnitude reflects how far apart the two
                   groups sit at this layer).
      ``v_unit`` : np.ndarray[hidden] — ``v_raw`` L2-normalized to unit length
                   (what ``SteeringContext`` uses for the relative_add op).
      ``layer``  : int — the layer the vector was read at (steer at the SAME one).
      ``n``      : int — contrastive examples per side (min of the two group
                   sizes), a quick quality signal — more pairs, less noisy mean.
      ``norm``   : float — ||v_raw||, the raw separation magnitude.
    """
    if not harmful or not benign:
        raise ValueError("need at least one harmful and one benign prompt")

    # Last-token residuals for each group: [n_group, hidden].
    acts_harmful = last_token_activations(model, tok, harmful, layer)
    acts_benign = last_token_activations(model, tok, benign, layer)

    # Diff of the group means — the whole method in one line.
    v_raw = acts_harmful.mean(axis=0) - acts_benign.mean(axis=0)
    v_raw = v_raw.astype(np.float32)

    norm = float(np.linalg.norm(v_raw))
    v_unit = (v_raw / norm) if norm > 0 else v_raw.copy()

    return {
        "v_raw": v_raw,
        "v_unit": v_unit.astype(np.float32),
        "layer": int(layer),
        "n": int(min(len(harmful), len(benign))),
        "norm": norm,
    }


def save_vector(path, vec: dict) -> None:
    """Serialize a vector dict (as returned by :func:`extract_caa_vector`).

    Uses ``torch.save`` so the file is a single self-contained artifact that
    round-trips through :func:`load_vector` with no external state.
    """
    torch.save(vec, path)


def load_vector(path) -> dict:
    """Inverse of :func:`save_vector`. Returns the vector dict."""
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# CPU import/shape sanity check — NO model download.
# Run: python -m steering_tutorials.hello_world_steering.steer_vector
# ---------------------------------------------------------------------------
def _self_test() -> None:
    import tempfile
    from pathlib import Path

    # Fake the one dependency (last_token_activations) via a stub model+tok is
    # overkill; instead we exercise save/load + the math on hand-built arrays,
    # which is what could realistically break independent of the model.
    v_raw = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # ||v|| == 5
    vec = {
        "v_raw": v_raw,
        "v_unit": (v_raw / 5.0).astype(np.float32),
        "layer": 7,
        "n": 16,
        "norm": 5.0,
    }
    assert abs(float(np.linalg.norm(vec["v_unit"])) - 1.0) < 1e-6

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "vec.pt"
        save_vector(p, vec)
        back = load_vector(p)
        assert back["layer"] == 7 and back["n"] == 16
        assert np.allclose(back["v_raw"], v_raw)
        assert np.allclose(back["v_unit"], v_raw / 5.0)

    print("[self-test] OK — save/load round-trips; unit vector is normalized.")


if __name__ == "__main__":
    _self_test()
