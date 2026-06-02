"""controls.py — steering-vector CONTROL directions (the confound killers).

The project's dominant confound: a measured "behavior effect" at some alpha is
reported only against the alpha=0 baseline, so it cannot be distinguished from
generic NORM-INDUCED activation perturbation. Pushing the residual stream by ANY
vector of the same magnitude degrades/alters generations; without a matched
control you are measuring that, not your concept direction.

This module builds drop-in [dim] np.ndarray directions that slot into the
existing steering pipeline (hooks.SteeringContext / apply_operation) in place of
a real DiffMean vector. Each is a different null hypothesis:

  - ``random_direction``      : an isotropic random unit vector (scaled).
  - ``matched_norm_random``   : random direction at EXACTLY ||reference|| — the
                                single most important steering baseline. A
                                behavior delta must be reported over THIS push,
                                not over alpha=0.
  - ``shuffled_label_vector`` : DiffMean of the SAME activations after the
                                pos/neg labels are destroyed by a random
                                re-partition — tests whether the real label
                                structure matters vs any random split direction.
  - ``extraction_stability``  : bootstrap gate that the extracted DiffMean is
                                STABLE (cosine-to-full near 1.0) before anything
                                downstream is trusted.

House style mirrors extract.py / geometry.py: ``from __future__ import
annotations``, typed, numpy float32, seeded determinism. The DiffMean convention
(mean(pos) - mean(neg)) and the cosine helper are reused from extract.py so the
controls share the champion pipeline's exact vector conventions.
"""

from __future__ import annotations

import numpy as np

from .extract import cosine, diffmean_vector


def random_direction(dim: int, *, norm: float = 1.0, seed: int = 0) -> np.ndarray:
    """Isotropic random unit vector scaled to ``norm`` — [dim] float32.

    The MATCHED-NORM RANDOM control in its bare form: a direction drawn from the
    rotationally-symmetric Gaussian (so uniform on the sphere), normalised, then
    scaled. Catches the "you're just measuring norm-induced degradation" failure
    mode — steering along this vector applies the SAME push magnitude as a real
    vector but carries no concept information.

    Determinism: a fixed ``seed`` reproduces the exact vector (numpy default-RNG,
    no global-state mutation).

    dim  : ambient residual-stream dimension.
    norm : target L2 norm of the returned vector (default 1.0 ⇒ unit vector).
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-12:  # pragma: no cover - astronomically unlikely
        # Degenerate all-zero draw: fall back to a canonical axis.
        v = np.zeros(dim, dtype=np.float32)
        v[0] = 1.0
        n = 1.0
    return (v / n * np.float32(norm)).astype(np.float32)


def matched_norm_random(reference_vector: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Random direction scaled to EXACTLY ‖reference_vector‖ — [dim] float32.

    The norm-matched random control: same dimension, same push magnitude as the
    real (e.g. DiffMean) vector, random direction. Reporting a behavior delta
    over this control instead of over alpha=0 isolates the contribution of the
    DIRECTION from the contribution of the raw displacement size. In high
    dimension this vector is ~orthogonal to the reference in expectation
    (cosine ≈ 0), so any residual alignment is pure chance.

    reference_vector : [dim] the vector whose norm is matched.
    """
    ref = np.asarray(reference_vector, dtype=np.float32).reshape(-1)
    target = float(np.linalg.norm(ref))
    return random_direction(ref.shape[0], norm=target, seed=seed)


def shuffled_label_vector(pos: np.ndarray, neg: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """DiffMean of a random RE-PARTITION of pos+neg — the LABELS-DESTROYED control.

    Pool the pos and neg activation rows, randomly re-partition them into two
    groups of the SAME sizes (|pos|, |neg|), and return the DiffMean of the
    shuffled partition. The vector is built from the identical data with the
    label structure destroyed, so it answers: does the real pos-vs-neg labelling
    carry a direction, or would any random split of the same rows produce a
    comparably-aligned vector? On data with genuine separation the true DiffMean
    aligns strongly with itself while this shuffled vector does not — the key
    scientific check that label structure (not just the data cloud) drives the
    extracted direction.

    pos, neg : [n_pos, dim], [n_neg, dim] activation matrices.
    """
    pos = np.asarray(pos, dtype=np.float32)
    neg = np.asarray(neg, dtype=np.float32)
    n_pos = pos.shape[0]
    pool = np.concatenate([pos, neg], axis=0)  # [n_pos+n_neg, dim]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool.shape[0])
    shuffled = pool[perm]
    fake_pos = shuffled[:n_pos]
    fake_neg = shuffled[n_pos:]
    return diffmean_vector(fake_pos, fake_neg).astype(np.float32)


def extraction_stability(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Bootstrap STABILITY gate for an extracted DiffMean direction.

    Resample the contrast rows WITH replacement (paired-independently within the
    pos and neg groups, preserving group sizes), recompute the DiffMean on each
    resample, and measure its cosine to the full-data DiffMean. A direction you
    can trust downstream is one that barely moves under resampling
    (``mean_cosine_to_full`` near 1.0, tight CI). On pure noise — no real
    separation — successive resamples point every which way and the mean cosine
    collapses toward 0 with a wide band. This is the gate to run BEFORE trusting
    anything downstream of the extracted vector.

    pos, neg : [n_pos, dim], [n_neg, dim] activation matrices.
    n_boot   : number of bootstrap resamples.
    seed     : RNG seed (deterministic).

    Returns {"mean_cosine_to_full", "std", "ci95": (lo, hi)} (all float).
    """
    pos = np.asarray(pos, dtype=np.float32)
    neg = np.asarray(neg, dtype=np.float32)
    full = diffmean_vector(pos, neg)
    rng = np.random.default_rng(seed)
    n_pos, n_neg = pos.shape[0], neg.shape[0]

    cosines = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        ip = rng.integers(0, n_pos, size=n_pos)
        in_ = rng.integers(0, n_neg, size=n_neg)
        boot_dm = diffmean_vector(pos[ip], neg[in_])
        cosines[b] = cosine(boot_dm, full)

    lo, hi = np.percentile(cosines, [2.5, 97.5])
    return {
        "mean_cosine_to_full": float(cosines.mean()),
        "std": float(cosines.std(ddof=1)) if n_boot > 1 else 0.0,
        "ci95": (float(lo), float(hi)),
    }
