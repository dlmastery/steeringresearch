r"""decompose.py -- the math core: split prompting's activation delta into a
steering-vector-like component + an off-direction residual, and measure how
consistent it is across prompts.

Nothing here is Gemma-specific except one thin extraction helper at the bottom
(``prompt_deltas``) that reads activations. The decomposition itself is pure
NumPy on ``[n_prompts, hidden]`` arrays, so it is fully CPU-testable with no
model (see the ``__main__`` self-test).

The picture (Cheng & Kriegeskorte 2026, arXiv:2606.03093)
---------------------------------------------------------
For prompt x let

    d(x) = act(x WITH the refusal instruction) - act(x WITHOUT it)      [hidden]

be the activation change the instruction induces at one layer. Given a unit
refusal direction ``u`` (a diff-of-means CAA vector) we write

    d(x) = <d(x), u> u        +      ( d(x) - <d(x), u> u )
           \_______________/         \_______________________/
           on-direction (parallel)   off-direction residual
           == a steering-vector shift  == the richer, input-specific part

Two summaries tell us how "steering-vector-like" prompting is:

  * on-direction energy fraction  = mean_x <d(x),u>^2 / ||d(x)||^2
        how much of each delta lies along the single refusal axis.
  * consistency  = mean pairwise cosine of the d(x)
        1.0 would mean every prompt gets the SAME shift (a pure translation --
        the tier the paper finds explains most of prompting); lower means the
        shift is input-dependent (an affine / nonlinear transform).

We also report the shared-translation fraction ``||mean_x d(x)||^2 / mean_x
||d(x)||^2`` (how much of the delta is one common vector) and hand back two raw
reconstruction vectors the WRITE check steers with:
  * ``v_shared`` = mean_x d(x)                      (the full common translation)
  * ``v_proj``   = <mean-delta, u> u  ... actually the mean of the *parallel*
                   components, i.e. mean_x (<d(x),u> u) -- the on-direction part
                   of the common translation only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# model_utils imports torch but does NOT load a model at import time (same as
# steer_vector.py's dependency), so importing this stays CPU/model-free.
from steering_tutorials.hello_world_steering.model_utils import last_token_activations


# --------------------------------------------------------------------------- #
# Pure math -- operates on plain arrays, unit-tested on CPU with no model.
# --------------------------------------------------------------------------- #
def mean_pairwise_cosine(deltas: np.ndarray) -> float:
    """Mean cosine similarity over all distinct prompt-delta pairs.

    A single shared translation makes every delta point the same way -> ~1.0.
    An input-dependent transform spreads their directions -> lower. We compute it
    from the normalized rows' Gram matrix and average its off-diagonal (each
    unordered pair counted once). Returns 0.0 for fewer than two rows.
    """
    d = np.asarray(deltas, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = d / norms
    gram = unit @ unit.T                       # [n, n] pairwise cosines
    n = d.shape[0]
    off_sum = float(gram.sum() - np.trace(gram))
    return off_sum / (n * (n - 1))             # mean over ordered == unordered


def decompose_prompt_deltas(deltas: np.ndarray, direction: np.ndarray) -> dict:
    """Decompose per-prompt deltas against a (refusal) ``direction``.

    Parameters
    ----------
    deltas : ``[n_prompts, hidden]`` -- one activation delta per prompt.
    direction : ``[hidden]`` -- the axis to project onto (need not be unit; we
        normalize it here so the caller can pass a raw diff-of-means vector).

    Returns a dict of scalars (all Python floats) plus two raw reconstruction
    vectors for the WRITE check:
      mean_delta_norm            : mean_x ||d(x)||
      mean_abs_proj              : mean_x |<d(x), u>|           (on-axis size)
      mean_signed_proj           : mean_x  <d(x), u>           (sign => which way)
      mean_residual_norm         : mean_x ||d(x) - <d(x),u> u||
      on_direction_frac          : mean_x <d(x),u>^2 / ||d(x)||^2   in [0, 1]
      consistency                : mean pairwise cosine of the deltas
      shared_translation_frac    : ||mean_x d(x)||^2 / mean_x ||d(x)||^2  in [0,1]
      v_shared                   : ``[hidden]`` mean_x d(x)  (raw units)
      v_proj                     : ``[hidden]`` mean_x (<d(x),u> u)  (raw units)
      u_unit                     : ``[hidden]`` the normalized direction used
      n                          : number of prompts
    """
    d = np.asarray(deltas, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] == 0:
        raise ValueError("deltas must be a non-empty [n_prompts, hidden] array")
    u = np.asarray(direction, dtype=np.float64).reshape(-1)
    u_norm = np.linalg.norm(u)
    if u_norm == 0.0:
        raise ValueError("direction has zero norm")
    u = u / u_norm

    proj_coef = d @ u                           # [n] signed length along u
    parallel = proj_coef[:, None] * u           # [n, hidden] on-direction part
    residual = d - parallel                     # [n, hidden] off-direction part

    delta_norm = np.linalg.norm(d, axis=1)      # [n]
    residual_norm = np.linalg.norm(residual, axis=1)
    safe_sq = np.where(delta_norm > 0, delta_norm ** 2, 1.0)
    on_frac = (proj_coef ** 2) / safe_sq        # [n] in [0, 1]

    d_bar = d.mean(axis=0)                       # shared translation (full)
    v_proj = parallel.mean(axis=0)               # shared translation (on-axis only)
    shared_frac = float((d_bar @ d_bar) / max(1e-12, float((delta_norm ** 2).mean())))

    return {
        "n": int(d.shape[0]),
        "mean_delta_norm": float(delta_norm.mean()),
        "mean_abs_proj": float(np.abs(proj_coef).mean()),
        "mean_signed_proj": float(proj_coef.mean()),
        "mean_residual_norm": float(residual_norm.mean()),
        "on_direction_frac": float(on_frac.mean()),
        "consistency": mean_pairwise_cosine(d),
        "shared_translation_frac": shared_frac,
        "v_shared": d_bar.astype(np.float32),
        "v_proj": v_proj.astype(np.float32),
        "u_unit": u.astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# Thin model-using extractor -- the ONLY function here that touches Gemma.
# --------------------------------------------------------------------------- #
def prompt_deltas(model: Any, tok: Any, prompts: list[str], layer: int,
                  instruction: str) -> np.ndarray:
    """Return ``[n_prompts, hidden]`` per-prompt deltas act(WITH) - act(WITHOUT).

    ``WITHOUT`` is the bare prompt; ``WITH`` prepends ``instruction`` to the same
    user turn. Both are read at the last token (the ``<start_of_turn>model``
    decision position, via ``add_generation_prompt=True`` inside
    ``last_token_activations``), so the delta captures how the instruction
    reshaped the state right where the model is about to answer.
    """
    if not prompts:
        raise ValueError("need at least one prompt")
    with_prompts = [f"{instruction}\n\n{p}" for p in prompts]
    acts_without = last_token_activations(model, tok, prompts, layer)     # [n, h]
    acts_with = last_token_activations(model, tok, with_prompts, layer)   # [n, h]
    return (acts_with - acts_without).astype(np.float32)


# --------------------------------------------------------------------------- #
# CPU self-test -- NO model download. Verifies the decomposition math on
# synthetic deltas with known structure.
# Run: python -m steering_tutorials.decomposing_prompting.decompose
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    rng = np.random.default_rng(0)
    hidden = 32
    n = 40
    u = rng.standard_normal(hidden)
    u = u / np.linalg.norm(u)

    # Case A: every delta == the SAME on-direction shift (pure translation along
    # u). Expect on_direction_frac ~ 1, residual ~ 0, consistency ~ 1,
    # shared_translation_frac ~ 1.
    shift = 3.0 * u
    deltas_a = np.tile(shift, (n, 1))
    a = decompose_prompt_deltas(deltas_a, u)
    assert a["on_direction_frac"] > 0.99, a["on_direction_frac"]
    assert a["mean_residual_norm"] < 1e-4, a["mean_residual_norm"]
    assert a["consistency"] > 0.99, a["consistency"]
    assert a["shared_translation_frac"] > 0.99, a["shared_translation_frac"]
    # v_proj reconstructs the shift; v_shared equals it too (no off-axis part).
    assert np.allclose(a["v_proj"], shift, atol=1e-3)
    assert np.allclose(a["v_shared"], shift, atol=1e-3)

    # Case B: shared on-direction shift + large per-prompt OFF-direction noise
    # (orthogonal complement of u). Expect low on_direction_frac, big residual,
    # lower consistency, but shared_translation_frac stays moderate because the
    # common part still survives averaging while the noise cancels.
    noise = rng.standard_normal((n, hidden))
    noise = noise - (noise @ u)[:, None] * u          # strip the u-component
    noise *= 5.0
    deltas_b = shift + noise
    b = decompose_prompt_deltas(deltas_b, u)
    assert b["on_direction_frac"] < a["on_direction_frac"]
    assert b["mean_residual_norm"] > a["mean_residual_norm"]
    assert b["consistency"] < a["consistency"]
    # the on-axis reconstruction should still recover ~the shared shift.
    assert np.allclose(b["v_proj"], shift, atol=0.5), b["v_proj"][:3]

    # mean_pairwise_cosine edge cases.
    assert mean_pairwise_cosine(np.zeros((1, 4))) == 0.0
    opp = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert abs(mean_pairwise_cosine(opp) + 1.0) < 1e-9    # anti-parallel -> -1

    print("[self-test] OK -- decomposition recovers translation vs residual; "
          "consistency + on-direction fraction behave as expected.")


if __name__ == "__main__":
    _self_test()
