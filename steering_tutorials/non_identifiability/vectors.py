"""vectors.py — build K different "refusal-ish" directions from the SAME data.

This is the heart of the lesson. From one harmful/benign contrast at one layer
we compute several candidate steering directions, each by a recipe that a
practitioner might reasonably reach for and each of which someone could call
"the refusal direction". The point is to then *measure* that they are NOT the
same vector (low pairwise cosine) even though — as ``run_nonident.py`` shows —
they steer to a similar effect.

The K recipes
-------------
  (a) ``diffmean_halfA``   diff-of-means (last token) on HALF A of the prompts.
  (b) ``diffmean_halfB``   diff-of-means (last token) on the DISJOINT HALF B.
                           (a) vs (b): same recipe, different data sample — the
                           cheapest source of non-identifiability, sampling.
  (c) ``pca_top1``         top-1 principal component of the paired residual
                           differences (harmful_i - benign_i). A *variance* axis,
                           not a *mean* axis — a different estimator of "the
                           contrast direction".
  (d) ``diffmean_full``    diff-of-means (last token) on ALL prompts. The
                           canonical CAA vector (Rimsky et al. 2023); our anchor.
  (e) ``diffmean_meanpool``diff-of-means but from MEAN-POOLED residuals instead
                           of the last token — a different pooling choice.
  (f) ``random_in_pcspan`` a RANDOM unit vector drawn inside the span of the top
                           principal components of the residuals. The control:
                           a direction that encodes no contrast at all, only
                           "lives where the activations live".

Every candidate is returned as a UNIT vector; ``model_utils`` steering is
norm-relative, so unit direction is all that matters for a matched-alpha effect.

Recipes (a)-(e) are sign-aligned to (d) so a positive alpha pushes the same
(refusal) way for all of them; (f) is a control and keeps its random sign.

  Rimsky et al. 2023, 'Steering Llama 2 via Contrastive Activation Addition'
    (arXiv:2312.06681) — diff-of-means at the last token (recipes a,b,d).
  Arditi et al. 2024, 'Refusal in LLMs is Mediated by a Single Direction'
    (arXiv:2406.11717) — the single-direction framing this lesson complicates.
  Venkatesh & Kurapath (Manipal Institute of Technology) 2026, 'On the
    Non-Identifiability of Steering Vectors in Large Language Models'
    (arXiv:2602.06801) — the claim under test.

Depends only on ``hello_world_steering.model_utils`` (activation reads) + numpy.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

# Absolute imports: this lesson reuses lesson 2's plumbing without duplicating it.
from steering_tutorials.hello_world_steering.model_utils import (
    last_token_activations,
    mean_pool_activation,
)


# --------------------------------------------------------------------------- #
# Pure linear algebra — no model, unit-tested below.
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` L2-normalized to unit length (or a copy if it is all-zero)."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.copy()


def _diff_of_means(harm: np.ndarray, ben: np.ndarray) -> np.ndarray:
    """The CAA direction: ``mean(harmful) - mean(benign)`` (raw, un-normalized)."""
    return (harm.mean(axis=0) - ben.mean(axis=0)).astype(np.float32)


def _pca_top1(diffs: np.ndarray) -> np.ndarray:
    """Top-1 principal component (unit) of a ``[n, hidden]`` difference matrix.

    We center the differences and take the leading right-singular vector via
    SVD — the axis along which the harmful-vs-benign differences vary most. This
    is a *different* estimator of the contrast than the mean, so it is expected
    to point somewhere similar-but-not-identical to diff-of-means.
    """
    diffs = np.asarray(diffs, dtype=np.float32)
    centered = diffs - diffs.mean(axis=0, keepdims=True)
    # full_matrices=False keeps it cheap; Vt[0] is the top principal direction.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return _unit(vt[0])


def _top_pcs(acts: np.ndarray, k: int) -> np.ndarray:
    """Return the top ``k`` principal components (``[k, hidden]``, orthonormal).

    These span the ``k``-dimensional subspace where the activations vary most —
    the "active subspace" the random control direction (f) is drawn from.
    """
    acts = np.asarray(acts, dtype=np.float32)
    centered = acts - acts.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    k = max(1, min(k, vt.shape[0]))
    return vt[:k].astype(np.float32)


def _random_in_span(basis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random UNIT vector inside the row-span of ``basis`` (``[k, hidden]``).

    Draw random coefficients, mix the basis vectors, normalize. Because ``basis``
    is orthonormal (SVD right-singular vectors) the result is a uniformly-random
    direction *within the active subspace* — a far stronger control than a random
    direction in all of R^hidden, most of which would be near-orthogonal to
    everything the model actually uses.
    """
    basis = np.asarray(basis, dtype=np.float32)
    coeffs = rng.standard_normal(basis.shape[0]).astype(np.float32)
    return _unit(coeffs @ basis)


def _align_sign(v_unit: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip ``v_unit`` if it points away from ``reference`` (so +alpha agrees).

    PCA and diff-of-means fix a direction only up to sign. We anchor every
    contrast recipe to the canonical diff-of-means so a positive alpha pushes all
    of them the same (refusal) way — otherwise a sign flip would masquerade as a
    huge behavioral difference and confound the comparison.
    """
    return v_unit if float(np.dot(v_unit, reference)) >= 0 else (-v_unit).astype(np.float32)


def cosine_matrix(unit_vectors: list[np.ndarray]) -> np.ndarray:
    """Pairwise cosine similarity of a list of (unit) vectors -> ``[K, K]``.

    Diagonal is 1.0; the matrix is symmetric. Because the inputs are unit
    vectors this is just their Gram matrix.
    """
    M = np.stack([_unit(v) for v in unit_vectors]).astype(np.float32)
    return (M @ M.T).astype(np.float32)


# --------------------------------------------------------------------------- #
# Activation collection — the only model-touching helpers.
# --------------------------------------------------------------------------- #
def _collect_last_token(model: Any, tok: Any, prompts: list[str],
                        layer: int) -> np.ndarray:
    """``[n, hidden]`` last-token residuals at ``layer`` (delegates to lesson 2)."""
    return last_token_activations(model, tok, prompts, layer)


def _collect_mean_pooled(model: Any, tok: Any, prompts: list[str],
                         layer: int) -> np.ndarray:
    """``[n, hidden]`` MEAN-POOLED residuals at ``layer``.

    Lesson 2 only exposes single-prompt mean pooling, so we loop it. Mean pooling
    averages over all positions rather than reading the last token — a different
    "where do I read the concept?" choice that yields recipe (e).
    """
    feats = [mean_pool_activation(model, tok, p, layer) for p in prompts]
    return np.stack(feats).astype(np.float32)


# --------------------------------------------------------------------------- #
# The public builder.
# --------------------------------------------------------------------------- #
def build_candidate_directions(
    model: Any,
    tok: Any,
    harmful: list[str],
    benign: list[str],
    layer: int,
    n_pc: int = 10,
    seed: int = 0,
) -> dict:
    """Build the K candidate directions and their cosine-similarity matrix.

    Returns a dict::

        {
          "candidates": {name: {"v_unit": np.ndarray[hidden],
                                "recipe": str, "pooling": str}},
          "names":      [name, ...],            # stable order for the matrix
          "cosine":     np.ndarray[K, K],       # pairwise cosine of v_unit's
          "layer":      int,
          "n_extract":  int,                    # min(#harmful, #benign) used
        }

    All contrast recipes (a-e) are sign-aligned to the canonical diff-of-means
    (d); the random control (f) keeps its own sign.
    """
    if not harmful or not benign:
        raise ValueError("need at least one harmful and one benign prompt")

    rng = np.random.default_rng(seed)

    # --- read activations once, reuse everywhere ----------------------------
    H_last = _collect_last_token(model, tok, harmful, layer)   # [nh, hidden]
    B_last = _collect_last_token(model, tok, benign, layer)    # [nb, hidden]
    H_mean = _collect_mean_pooled(model, tok, harmful, layer)  # [nh, hidden]
    B_mean = _collect_mean_pooled(model, tok, benign, layer)   # [nb, hidden]

    m = min(len(H_last), len(B_last))          # paired count for halves + PCA
    half = m // 2

    # --- (d) canonical diff-of-means (last token, full data): the anchor -----
    v_full_raw = _diff_of_means(H_last, B_last)
    v_full = _unit(v_full_raw)

    # --- (a,b) diff-of-means on two disjoint halves --------------------------
    v_halfA = _align_sign(_unit(_diff_of_means(H_last[:half], B_last[:half])), v_full)
    v_halfB = _align_sign(_unit(_diff_of_means(H_last[half:m], B_last[half:m])), v_full)

    # --- (c) PCA top-1 of the paired residual differences --------------------
    diffs = (H_last[:m] - B_last[:m]).astype(np.float32)
    v_pca = _align_sign(_pca_top1(diffs), v_full)

    # --- (e) diff-of-means from MEAN-POOLED residuals ------------------------
    v_meanpool = _align_sign(_unit(_diff_of_means(H_mean, B_mean)), v_full)

    # --- (f) random unit vector inside the top-PC span (control) -------------
    stack = np.concatenate([H_last, B_last], axis=0)
    pcs = _top_pcs(stack, n_pc)
    v_random = _random_in_span(pcs, rng)       # NOT sign-aligned: it's a control

    candidates = {
        "diffmean_halfA":    {"v_unit": v_halfA,    "recipe": "diff-of-means, last-token, data half A", "pooling": "last"},
        "diffmean_halfB":    {"v_unit": v_halfB,    "recipe": "diff-of-means, last-token, data half B", "pooling": "last"},
        "pca_top1":          {"v_unit": v_pca,      "recipe": "PCA top-1 of paired (harmful-benign) diffs", "pooling": "last"},
        "diffmean_full":     {"v_unit": v_full,     "recipe": "diff-of-means, last-token, ALL data (CAA anchor)", "pooling": "last"},
        "diffmean_meanpool": {"v_unit": v_meanpool, "recipe": "diff-of-means, MEAN-POOLED, ALL data", "pooling": "mean"},
        "random_in_pcspan":  {"v_unit": v_random,   "recipe": f"random unit vector in top-{pcs.shape[0]} PC span (control)", "pooling": "last"},
    }

    names = list(candidates.keys())
    cos = cosine_matrix([candidates[n]["v_unit"] for n in names])

    print(f"[vectors] built {len(names)} candidate directions at layer {layer} "
          f"(n_extract={m}/class)", file=sys.stderr)
    return {
        "candidates": candidates,
        "names": names,
        "cosine": cos,
        "layer": int(layer),
        "n_extract": int(m),
    }


def save_directions(path, built: dict) -> None:
    """Persist candidate unit vectors + names to a single ``.npz`` file.

    ``infer.py`` reloads this so you can steer with any chosen candidate without
    re-reading activations. Metadata (recipe strings, cosine matrix) is small so
    we store it too.
    """
    names = built["names"]
    arrays = {f"v__{n}": built["candidates"][n]["v_unit"] for n in names}
    np.savez(
        path,
        names=np.array(names, dtype=object),
        cosine=built["cosine"],
        layer=np.int64(built["layer"]),
        **arrays,
    )


def load_directions(path) -> dict:
    """Inverse of :func:`save_directions`. Returns ``{name: v_unit}`` + metadata."""
    data = np.load(path, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    vectors = {n: data[f"v__{n}"].astype(np.float32) for n in names}
    return {"names": names, "vectors": vectors,
            "cosine": data["cosine"], "layer": int(data["layer"])}


# --------------------------------------------------------------------------- #
# CPU self-test — NO model download. Exercises the pure math on synthetic data.
# Run: python -m steering_tutorials.non_identifiability.vectors
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    rng = np.random.default_rng(0)
    hidden = 32
    n = 40

    # Two clearly separated Gaussian blobs so diff-of-means is well-defined.
    true_dir = _unit(rng.standard_normal(hidden))
    benign = rng.standard_normal((n, hidden)).astype(np.float32)
    harmful = benign + 3.0 * true_dir + 0.3 * rng.standard_normal((n, hidden))

    # diff-of-means recovers roughly the true direction.
    v = _unit(_diff_of_means(harmful, benign))
    assert abs(np.dot(v, true_dir)) > 0.9, "diff-of-means should recover the axis"

    # unit vectors really are unit length.
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    # PCA top-1 of the differences is a valid unit vector.
    v_pca = _pca_top1(harmful[:n] - benign[:n])
    assert abs(np.linalg.norm(v_pca) - 1.0) < 1e-5

    # sign alignment: an anti-aligned vector gets flipped toward the reference.
    flipped = _align_sign(-v, v)
    assert np.dot(flipped, v) > 0, "sign alignment should flip toward reference"

    # random-in-span stays inside the span (its residual off the basis is ~0).
    basis = _top_pcs(np.concatenate([harmful, benign]), 5)   # [5, hidden]
    r = _random_in_span(basis, rng)
    recon = basis.T @ (basis @ r)          # project r onto the span, then back
    assert np.linalg.norm(r - recon) < 1e-4, "random control must lie in the span"
    assert abs(np.linalg.norm(r) - 1.0) < 1e-5

    # cosine matrix: symmetric, unit diagonal, entries in [-1, 1].
    mat = cosine_matrix([v, v_pca, r])
    assert mat.shape == (3, 3)
    assert np.allclose(np.diag(mat), 1.0, atol=1e-5)
    assert np.allclose(mat, mat.T, atol=1e-6)
    assert mat.min() >= -1.0001 and mat.max() <= 1.0001

    print("[self-test] OK - diff-of-means, PCA, span-random, sign-align, "
          "cosine matrix all behave.")


if __name__ == "__main__":
    _self_test()
