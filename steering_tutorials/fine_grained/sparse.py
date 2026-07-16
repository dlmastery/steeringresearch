"""sparse.py — the one new idea: sparsify a steering vector before injecting it.

Dense activation steering (lesson 2) adds a full-rank direction ``v`` to the
residual stream: every one of the model's ~1000+ hidden coordinates is nudged.
Fine-grained steering asks a sharper question — *do you need all of them?* Most
of ``v``'s coordinates are small; the behavior may be carried by a few large
ones. So we KEEP only the top-``keep_frac`` coordinates by magnitude, ZERO the
rest, and RENORMALIZE the survivor back to the dense vector's norm so the edit's
*strength* is unchanged — only its *support* shrinks.

        v_sparse = renorm( v ⊙ mask_topk(|v|, keep_frac) ,  to ‖v‖ )

This lesson is a simplified reconstruction INSPIRED BY AUSteer
(arXiv:2602.04428, Feng et al., ICLR 2026): the paper selects units by an
activation-momentum discriminativeness metric with adaptive per-input strength;
we use a simpler top-k magnitude mask. Why this can help: the small
coordinates are the ones most likely to encode *incidental* correlates of the
contrast (topic, length, style) rather than the behavior itself. Zeroing them
removes off-target pressure on the residual stream, so benign prompts get
over-steered less and the model tips into gibberish less — "steering less,
achieving more."

Two objects:
  * ``sparsify(v, keep_frac)`` — the pure numpy transform (unit-tested, no model).
  * ``SparseSteeringContext``  — lesson 2's steering hook, fed the sparse vector.

Depends only on numpy here; the context delegates all hook mechanics to
``hello_world_steering.model_utils.SteeringContext`` (imported lazily).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def sparsify(v: "np.ndarray", keep_frac: float) -> "np.ndarray":
    """Keep the top ``keep_frac`` coordinates of ``v`` by |magnitude|; zero the rest.

    The survivor is renormalized back to ``‖v‖`` so the sparse vector has the
    SAME L2 norm as the dense one — this is what "matched strength" means: the
    step size does not change, only how many dimensions carry it.

    Parameters
    ----------
    v : np.ndarray, shape ``[hidden]`` — the dense steering direction.
    keep_frac : float in (0, 1] — fraction of coordinates to keep. ``1.0`` returns
        a plain copy (dense). ``0.05`` keeps the 5% largest-magnitude coordinates.

    Returns
    -------
    np.ndarray, shape ``[hidden]``, float32 — the sparsified, renormalized vector.
    The number of nonzero coordinates is exactly ``round(keep_frac * hidden)``
    (clamped to at least 1), and ``‖out‖ == ‖v‖`` up to float error.
    """
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    dim = v.shape[0]
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")

    dense_norm = float(np.linalg.norm(v))
    if keep_frac >= 1.0 or dim == 0:
        return v.copy()

    # How many coordinates survive. round() matches the paper's "top-k%" framing;
    # clamp to >=1 so we never return an all-zero (no-op) vector.
    k = max(1, int(round(keep_frac * dim)))
    if k >= dim:
        return v.copy()

    # Indices of the k largest-magnitude coordinates. argpartition is O(dim) and
    # picks exactly k indices regardless of ties, so the nonzero count is exact.
    keep_idx = np.argpartition(np.abs(v), dim - k)[-k:]

    out = np.zeros_like(v)
    out[keep_idx] = v[keep_idx]

    # Renormalize the survivor back to the dense norm (matched strength).
    sparse_norm = float(np.linalg.norm(out))
    if sparse_norm > 0.0:
        out *= dense_norm / sparse_norm
    return out.astype(np.float32)


def support_size(keep_frac: float, dim: int) -> int:
    """The number of nonzero coordinates ``sparsify`` will produce for ``dim``.

    Exposed so callers (and the README table) can report the exact coordinate
    count behind each ``keep_frac`` without re-deriving the rounding rule.
    """
    if keep_frac >= 1.0:
        return dim
    return max(1, int(round(keep_frac * dim)))


class SparseSteeringContext:
    """Steer with the SPARSIFIED vector, using lesson 2's hook unchanged.

    This is a thin wrapper: it sparsifies ``vector`` once (``sparsify`` above)
    and then hands the result to ``hello_world_steering.model_utils``'s
    ``SteeringContext``, so all the delicate hook mechanics — the ``relative_add``
    norm-relative step, the special-token guard, exact restoration on exit — are
    reused verbatim rather than re-implemented. ``keep_frac=1.0`` reproduces
    dense lesson-2 steering exactly.

    Parameters mirror ``SteeringContext`` plus ``keep_frac`` (the sparsity knob).
    The context manager protocol (``with SparseSteeringContext(...):``) is
    delegated to the wrapped context.
    """

    def __init__(
        self,
        model: Any,
        vector: "np.ndarray",
        layer: int,
        alpha: float,
        keep_frac: float,
        operation: str = "relative_add",
        special_ids: "set[int] | None" = None,
    ) -> None:
        # Lazy import so ``import sparse`` never pulls in torch / transformers.
        from steering_tutorials.hello_world_steering.model_utils import SteeringContext

        self.keep_frac = float(keep_frac)
        v_sparse = sparsify(vector, keep_frac)
        self._inner = SteeringContext(
            model, v_sparse, layer, alpha, operation, special_ids
        )

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)


# ---------------------------------------------------------------------------
# CPU unit test — NO model download. Verifies the sparsify contract:
#   (a) nonzero count == round(keep_frac * dim),
#   (b) the kept coordinates are the largest-magnitude ones,
#   (c) the output norm equals the dense norm (matched strength),
#   (d) keep_frac == 1.0 is an identity (dense passthrough).
# Run: python -m steering_tutorials.fine_grained.sparse
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # Use a dim/keep_frac pair whose product is an exact integer with no rounding
    # ambiguity, and distinct nonzero magnitudes so the top-k set is unique.
    dim = 1000
    v = rng.standard_normal(dim).astype(np.float32)
    # Guarantee distinct magnitudes (no ties) so the assertions are exact.
    v = (np.sign(v) * (np.abs(v) + np.arange(dim) * 1e-4)).astype(np.float32)
    dense_norm = float(np.linalg.norm(v))

    for keep_frac in (0.5, 0.25, 0.1, 0.05, 0.02):
        out = sparsify(v, keep_frac)
        k_expected = round(keep_frac * dim)

        # (a) exact support size.
        nnz = int(np.count_nonzero(out))
        assert nnz == k_expected, f"keep_frac={keep_frac}: nnz {nnz} != {k_expected}"
        assert support_size(keep_frac, dim) == k_expected

        # (b) the survivors are exactly the top-k by magnitude.
        kept = set(np.nonzero(out)[0].tolist())
        topk = set(np.argsort(np.abs(v))[-k_expected:].tolist())
        assert kept == topk, f"keep_frac={keep_frac}: kept set != true top-k"

        # (c) matched strength: norm preserved.
        assert abs(float(np.linalg.norm(out)) - dense_norm) < 1e-3, "norm not preserved"

    # (d) dense passthrough is an identity.
    dense = sparsify(v, 1.0)
    assert np.array_equal(dense, v), "keep_frac=1.0 should be identity"
    assert int(np.count_nonzero(dense)) == dim

    # Guard rails.
    for bad in (0.0, -0.1, 1.5):
        try:
            sparsify(v, bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"keep_frac={bad} should have raised ValueError")

    print("[self-test] OK - sparsify keeps exact top-k, preserves norm, "
          "dense passthrough is identity.")


if __name__ == "__main__":
    _self_test()
