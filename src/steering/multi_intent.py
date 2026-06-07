"""multi_intent.py — compose K safety directions under a Gram-mass budget.

When several intents fire at once (e.g. a prompt is both "self-harm" and
"privacy"), the method must combine their safety directions into a single write.
Naively summing competing directions wastes the residual norm budget (N5) and
can cancel; the §9 stacking discipline says near-orthogonal directions stack but
overlapping ones interfere. This module supplies the three primitives that make
that discipline operational:

  * ``gram_schmidt``           — orthogonalise a set of directions so they stack
                                 cleanly (returns a unit-norm orthogonal basis).
  * ``compose``                — the actual combine: ``Σ alpha_i v_i``.
  * ``interference_gram_mass`` — read out how much overlap a candidate set
                                 carries (the off-diagonal mass of the Gram
                                 matrix). Zero ⇒ orthogonal ⇒ free to stack;
                                 large ⇒ the directions fight ⇒ budget is spent.

All math is numpy float32 (the extraction/calibration side of the numpy↔torch
seam; see DESIGN.md §3). Pure functions, deterministic.

STATUS: BUILT but NOT YET VALIDATED on real models / benchmarks. The unit tests
verify orthogonality and the Gram-mass formula on synthetic vectors; no real
multi-intent safety basis has been measured on Gemma yet.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_EPS = 1e-8


def _as_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Stack a sequence of equal-length 1-D vectors into ``[k, dim]`` float32."""
    if len(vectors) == 0:
        raise ValueError("need at least one vector")
    mats = [np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors]
    dim = mats[0].shape[0]
    for i, m in enumerate(mats):
        if m.shape[0] != dim:
            raise ValueError(
                f"all vectors must share a dimension; vector {i} has {m.shape[0]}, "
                f"expected {dim}"
            )
    return np.stack(mats, axis=0)


def gram_schmidt(vectors: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Orthonormalise ``vectors`` via modified Gram-Schmidt.

    Returns a list of unit-norm vectors that are mutually orthogonal (pairwise
    cosine ~0). A vector that is (numerically) in the span of the already-chosen
    basis collapses to ~0 after subtraction; such a degenerate direction is
    dropped rather than returned as noise, so the output may be shorter than the
    input when the inputs are linearly dependent.

    The order of ``vectors`` is preserved (earlier vectors keep more of their
    original direction) so a priority ordering of intents is respected.
    """
    mat = _as_matrix(vectors)
    basis: list[np.ndarray] = []
    for row in mat:
        w = row.astype(np.float32).copy()
        for b in basis:
            w = w - float(np.dot(w, b)) * b
        norm = float(np.linalg.norm(w))
        if norm < 1e-6:
            # In the span of the existing basis -> degenerate, drop it.
            continue
        basis.append((w / norm).astype(np.float32))
    return basis


def compose(
    active_vectors: Sequence[np.ndarray],
    alphas: Sequence[float],
) -> np.ndarray:
    """Linear combination ``Σ alpha_i v_i`` of the active safety directions.

    This is the write the CAST steerer applies when multiple intents fire: each
    fired intent contributes its safety direction scaled by its weight. With
    ``alphas`` all equal it is a plain sum of directions (the steering
    coefficient is then applied once, in ``hooks.apply_operation``). Returns a
    ``[dim]`` float32 vector (NOT renormalised — the caller controls magnitude
    via the operation's alpha).
    """
    mat = _as_matrix(active_vectors)
    a = np.asarray(alphas, dtype=np.float32).reshape(-1)
    if a.shape[0] != mat.shape[0]:
        raise ValueError(
            f"alphas ({a.shape[0]}) must match the number of vectors ({mat.shape[0]})"
        )
    return (a @ mat).astype(np.float32)


def interference_gram_mass(vectors: Sequence[np.ndarray]) -> float:
    """Off-diagonal mass of the unit-vector Gram matrix (the N5 overlap readout).

    Each vector is unit-normalised; the Gram matrix ``G = U Uᵀ`` then holds the
    pairwise cosines (diagonal = 1). The interference mass is the Frobenius norm
    of the off-diagonal block::

        mass = sqrt( Σ_{i≠j} G_ij² ) = sqrt( 2 · Σ_{i<j} cos²(v_i, v_j) )

    A perfectly orthogonal set ⇒ 0 (free to stack); two identical directions ⇒
    ``sqrt(2)`` (the two symmetric off-diagonal 1's); larger values mean the
    candidate intents fight over the same subspace and the norm budget is being
    spent. A single vector has no pairs ⇒ 0.
    """
    mat = _as_matrix(vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    units = mat / (norms + _EPS)
    gram = units @ units.T
    off = gram - np.diag(np.diag(gram))
    return float(np.sqrt(float(np.sum(off * off))))
