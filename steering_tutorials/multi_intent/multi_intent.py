"""multi_intent.py — compose K steering directions without them fighting.

This is the mechanical core of lesson 9. It answers three questions:

  1. How do we get K concept directions? -> ``extract_concept_vectors``: one
     diff-of-means per concept, each contrasted against a SHARED benign baseline
     so the K raw directions share an origin and their overlap is meaningful.

  2. How do we stop them interfering? -> ``gram_schmidt``: orthonormalize the K
     directions so each concept steers along its own axis. After
     orthogonalization, adding concept A's vector leaves concept B's projection
     (almost) unchanged — that is what kills cross-talk.

  3. How do we apply all K at once, and how much "room" is left? ->
     ``apply_multi`` injects Σ αᵢ ||h|| unit(vᵢ) in a single hook, and
     ``norm_budget`` reports the total displacement being spent (the N5 budget).

Why interference happens (the mechanism)
----------------------------------------
Two diff-of-means directions vᵢ, vⱼ are almost never orthogonal — real concepts
share features ("malware" and "fraud" both involve deception, money, systems).
If you add unit(vᵢ) + unit(vⱼ), the component of vⱼ that lies ALONG vᵢ adds to
vᵢ's push (over-steering that concept) while the shared direction eats budget
that neither concept "meant" to spend. Orthogonalizing removes each later
vector's projection onto the earlier ones, so what remains is the part of vⱼ
that is genuinely NEW relative to vᵢ.

Reuses lesson 2's steering hook verbatim (``SteeringContext``) — we do not
reimplement steering, we drive it with a summed vector.

  Rimsky et al. 2023, 'Steering Llama 2 via Contrastive Activation Addition'
    (arXiv:2312.06681) — the diff-of-means direction, per concept.
  Arditi et al. 2024, 'Refusal in LLMs is Mediated by a Single Direction'
    (arXiv:2406.11717) — a concept can be one linear direction (so K concepts
    are K directions we can orthogonalize).
  The norm budget N5 is this project's own leading-indicator (see CLAUDE.md §3):
    cumulative ‖Δh‖/‖h‖ is finite before coherence breaks.

Depends only on numpy + lesson 2's ``model_utils`` (same tutorials package).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from steering_tutorials.hello_world_steering.model_utils import (
    SteeringContext,
    generate,
    last_token_activations,
)


# --------------------------------------------------------------------------- #
# 1. Extract one diff-of-means direction per concept (shared benign baseline).
# --------------------------------------------------------------------------- #
def extract_concept_vectors(
    model: Any,
    tok: Any,
    concept_prompts: dict[str, list[str]],
    layer: int,
    baseline_prompts: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build one CAA steering direction per concept.

    Parameters
    ----------
    concept_prompts : ``{concept_name: [harmful exemplar prompts]}``. If a key
        named ``"__baseline__"`` is present it is popped and used as the shared
        benign baseline (convenient when the caller carries everything in one
        dict); otherwise pass ``baseline_prompts`` explicitly.
    baseline_prompts : the SHARED benign set every concept is contrasted against.

    Returns
    -------
    ``{concept_name: v_raw}`` where each ``v_raw`` is a float32 ``[hidden]``
    array = ``mean(act | concept) - mean(act | baseline)`` at ``layer``. These
    are RAW (un-normalized, un-orthogonalized) directions — normalization and
    orthogonalization are separate, composable steps (``gram_schmidt``).

    Using ONE baseline for all concepts is deliberate: it gives the K directions
    a common origin, so ``cos(vᵢ, vⱼ)`` measures how similar the concepts are
    (not how far their baselines drifted). That similarity is exactly the
    interference we later orthogonalize away.
    """
    prompts = dict(concept_prompts)  # shallow copy — we may pop the baseline key
    if baseline_prompts is None:
        baseline_prompts = prompts.pop("__baseline__", None)
    if not baseline_prompts:
        raise ValueError("need a shared benign baseline (baseline_prompts or "
                         "a '__baseline__' key in concept_prompts)")

    # Read the baseline ONCE — it is shared across every concept's contrast.
    baseline_acts = last_token_activations(model, tok, baseline_prompts, layer)
    baseline_mean = baseline_acts.mean(axis=0)  # [hidden]

    vectors: dict[str, np.ndarray] = {}
    for name, exemplars in prompts.items():
        if not exemplars:
            raise ValueError(f"concept {name!r} has no exemplar prompts")
        acts = last_token_activations(model, tok, exemplars, layer)  # [n, hidden]
        v_raw = (acts.mean(axis=0) - baseline_mean).astype(np.float32)
        vectors[name] = v_raw
    return vectors


# --------------------------------------------------------------------------- #
# 2. Gram-Schmidt: turn K overlapping directions into K orthonormal axes.
# --------------------------------------------------------------------------- #
def gram_schmidt(vectors: list[np.ndarray], eps: float = 1e-8) -> list[np.ndarray]:
    """Orthonormalize a list of vectors with the (modified) Gram-Schmidt process.

    Given directions ``v0, v1, ..., v_{K-1}``, produce unit vectors
    ``u0, u1, ..., u_{K-1}`` such that ``uᵢ · uⱼ = 0`` for ``i != j`` and
    ``||uᵢ|| = 1``. Each ``uᵢ`` keeps only the part of ``vᵢ`` that is NEW — the
    component orthogonal to everything already accepted.

    The recipe, one vector at a time:

        w = vᵢ                                  # start from the raw direction
        for each previously accepted axis u:    # subtract what's already covered
            w = w - (w · u) u                    #   remove w's projection onto u
        uᵢ = w / ||w||                           # normalize what remains

    The inner loop is the crux: ``(w · u) u`` is the shadow ``w`` casts on an
    existing axis; subtracting it leaves the residual that no earlier concept
    could express. We use the MODIFIED form (subtract against the running ``w``,
    not the original ``vᵢ``), which is numerically more stable than the classical
    form when the inputs are nearly collinear.

    Order matters: earlier vectors keep their full direction; later ones are
    trimmed to their novel part. Feed the most important / most distinct concept
    first (see ``config.CONCEPTS``). If a vector is (numerically) a linear
    combination of earlier ones, its residual norm falls below ``eps`` and we
    emit a zero vector for it — an honest signal that "this concept adds no new
    axis" rather than a silent NaN from dividing by ~0.
    """
    basis: list[np.ndarray] = []
    for v in vectors:
        w = v.astype(np.float64).copy()          # float64 for a stable dot/subtract
        for u in basis:
            w = w - np.dot(w, u) * u             # remove the shadow on each axis
        norm = float(np.linalg.norm(w))
        if norm < eps:
            # Degenerate: v lies in the span of earlier vectors -> no new axis.
            basis.append(np.zeros_like(w))
        else:
            basis.append(w / norm)               # accept a fresh unit axis
    return [u.astype(np.float32) for u in basis]


def cosine_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Return the ``[K, K]`` matrix of pairwise cosine similarities.

    A diagnostic for interference: off-diagonal entries near 0 mean the concepts
    are already near-orthogonal (little to gain from Gram-Schmidt); entries near
    ±1 mean heavy overlap (orthogonalization matters a lot). Zero vectors map to
    a zero row/column rather than NaN.
    """
    k = len(vectors)
    units = []
    for v in vectors:
        n = float(np.linalg.norm(v))
        units.append(v / n if n > 0 else np.zeros_like(v))
    M = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            M[i, j] = float(np.dot(units[i], units[j]))
    return M


# --------------------------------------------------------------------------- #
# 3. The norm budget (N5): total displacement being injected.
# --------------------------------------------------------------------------- #
def norm_budget(vectors: list[np.ndarray], alphas: list[float]) -> float:
    """Total steering displacement, in units of ||h|| (the N5 norm budget).

    Under ``relative_add`` each concept injects ``αᵢ * ||h|| * unit(vᵢ)``. When
    the ``unit(vᵢ)`` are ORTHONORMAL the injected deltas are perpendicular, so
    the combined step length is the Euclidean (root-sum-square) combination:

        budget = ||Σ αᵢ ||h|| uᵢ|| / ||h|| = sqrt(Σ αᵢ²)     (orthonormal uᵢ)

    We report this orthonormal-case budget: sqrt(Σ αᵢ²). It is the honest cost of
    stacking K orthogonal concepts and the number to watch against the coherence
    cliff — once it climbs past the single-concept gibberish threshold, adding
    more concepts starts breaking the model regardless of orthogonality. (For raw
    NON-orthogonal vectors the true displacement can be larger, because the
    shared components add linearly rather than in quadrature — another reason
    orthogonalizing is the budget-efficient choice.)
    """
    if len(vectors) != len(alphas):
        raise ValueError("vectors and alphas must have the same length")
    return float(np.sqrt(sum(a * a for a in alphas)))


# --------------------------------------------------------------------------- #
# 4. Apply K directions at once, in a single steering hook.
# --------------------------------------------------------------------------- #
def _combined_vector(vectors: list[np.ndarray], alphas: list[float]) -> np.ndarray:
    """Fold K (alpha, unit-direction) pairs into ONE vector for the hook.

    ``SteeringContext`` (relative_add) injects ``alpha_total * ||h|| * unit(V)``
    for a single ``V``. To apply K concepts with per-concept alphas we build

        V = Σ αᵢ unit(vᵢ)                      (a weighted sum of unit dirs)

    and hand the hook ``V`` with ``alpha_total = 1.0``. Then the hook injects
    ``||h|| * unit(V)`` scaled correctly ONLY if ||V|| carries the alphas — so we
    do NOT re-normalize V here; we return the raw weighted sum and let the caller
    pass it with the matching alpha (see ``apply_multi``). Concretely, the hook
    computes ``alpha * ||h|| * unit(V)`` = ``||h|| * unit(V)`` at alpha=1, which
    for orthonormal uᵢ has the intended per-concept ratios and total length
    ``sqrt(Σαᵢ²)`` relative to the unit — matching ``norm_budget``.
    """
    if len(vectors) != len(alphas):
        raise ValueError("vectors and alphas must have the same length")
    hidden = vectors[0].shape[0]
    combined = np.zeros(hidden, dtype=np.float32)
    for v, a in zip(vectors, alphas):
        n = float(np.linalg.norm(v))
        if n > 0:
            combined += np.float32(a) * (v / n)   # alpha-weighted UNIT direction
    return combined


def apply_multi(
    model: Any,
    tok: Any,
    prompt: str,
    vectors: list[np.ndarray],
    alphas: list[float],
    layer: int,
    max_new_tokens: int = 48,
) -> str:
    """Generate ``prompt`` while steering along K directions simultaneously.

    Builds the alpha-weighted sum ``V = Σ αᵢ unit(vᵢ)`` and runs one
    ``relative_add`` hook with ``alpha=1.0`` on ``V`` — so all K concepts are
    injected in a single pass at ``||h|| * unit(V)``. Returns ONLY the new text.

    Pass ORTHONORMALIZED ``vectors`` (from ``gram_schmidt``) to steer each
    concept along its own axis with minimal cross-talk; pass the RAW diff-of-mean
    vectors to reproduce the naive summed-steering baseline and watch it
    interfere. The rest of the pipeline (which prompts, which judge) is identical
    for both, so any difference is attributable to orthogonalization alone.
    """
    if not vectors:
        return generate(model, tok, prompt, max_new_tokens=max_new_tokens, alpha=0.0)

    combined = _combined_vector(vectors, alphas)
    if float(np.linalg.norm(combined)) == 0.0:
        # Every alpha was 0 (or all vectors degenerate) -> unsteered baseline.
        return generate(model, tok, prompt, max_new_tokens=max_new_tokens, alpha=0.0)

    # One hook, one pass: the hook injects ||h|| * unit(combined). Because
    # ``combined`` already carries the per-concept alpha weights, its direction
    # encodes the K-way mixture and its magnitude sets the relative strengths.
    special = set(getattr(tok, "all_special_ids", []) or [])
    device_prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    import torch  # local import keeps ``import multi_intent`` torch-free at parse

    with torch.no_grad():
        device = next(model.parameters()).device
        ids = device_prompt_ids.to(device)
        prompt_len = ids.shape[1]
        with SteeringContext(model, combined, layer, alpha=1.0,
                             operation="relative_add", special_ids=special):
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None
                              else tok.eos_token_id),
            )
    new_tokens = out[0][prompt_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# CPU unit test — NO model download. Verifies the pure-math pieces:
#   Gram-Schmidt orthonormality, the norm budget, and the combined-vector fold.
# Run: python -m steering_tutorials.multi_intent.multi_intent
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # (a) Gram-Schmidt on random (correlated) vectors -> orthonormal basis.
    #     Build correlated inputs so the test exercises real projection removal,
    #     not a lucky already-orthogonal set.
    base = rng.standard_normal((5, 16)).astype(np.float32)
    correlated = [base[0]]
    for i in range(1, 5):
        # Each new vector shares 60% of the previous one -> guaranteed overlap.
        correlated.append(0.6 * correlated[-1] + 0.4 * base[i])
    U = gram_schmidt(correlated)

    G = np.stack(U) @ np.stack(U).T          # Gram matrix of the basis
    off_diag = G - np.diag(np.diag(G))
    assert np.max(np.abs(off_diag)) < 1e-5, "Gram-Schmidt: axes not orthogonal"
    for u in U:
        n = float(np.linalg.norm(u))
        assert abs(n - 1.0) < 1e-5, "Gram-Schmidt: axis not unit-norm"

    # (b) Degenerate input: a vector inside the span of earlier ones -> zero axis.
    v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v2 = np.array([2.0, 3.0, 0.0], dtype=np.float32)  # = 2*v0 + 3*v1 (no new axis)
    U2 = gram_schmidt([v0, v1, v2])
    assert float(np.linalg.norm(U2[2])) < 1e-6, "collinear vector should give 0 axis"

    # (c) Norm budget of orthonormal axes = sqrt(sum alpha^2).
    alphas = [0.06, 0.06, 0.06]
    b = norm_budget(U[:3], alphas)
    assert abs(b - np.sqrt(3) * 0.06) < 1e-6, "norm budget mismatch"

    # (d) Combined-vector fold: for ORTHONORMAL axes, the mixed vector's length is
    #     exactly the norm budget (quadrature), and its projection on each axis
    #     recovers that axis's alpha.
    mixed = _combined_vector(U[:3], alphas)
    assert abs(float(np.linalg.norm(mixed)) - b) < 1e-5, "mixed length != budget"
    for u, a in zip(U[:3], alphas):
        proj = float(np.dot(mixed, u))
        assert abs(proj - a) < 1e-5, "per-axis alpha not recovered from mixture"

    # (e) cosine_matrix: orthonormal basis -> identity.
    C = cosine_matrix(U[:3])
    assert np.allclose(C, np.eye(3), atol=1e-5), "cosine matrix of ortho basis != I"

    print("[self-test] OK - Gram-Schmidt orthonormal, degenerate->0, "
          "norm budget = sqrt(sum alpha^2), mixture recovers per-axis alphas.")


if __name__ == "__main__":
    _self_test()
