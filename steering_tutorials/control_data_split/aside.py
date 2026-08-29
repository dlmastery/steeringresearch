"""aside.py -- the ASIDE isoclinic rotation, implemented exactly and checked.

REFERENCE (WebFetch-VERIFIED 2026-08-29)
----------------------------------------
Zverev, Kortukov, Panfilov, Volkova, Tabesh, Lapuschkin, Samek, Lampert,
"ASIDE: Architectural Separation of Instructions and Data in Language Models",
arXiv:2503.10566 (v1 13 Mar 2025, v4 9 Feb 2026), ICLR 2026. Abstract, verbatim:
"ASIDE applies an orthogonal rotation to the embeddings of data tokens, thus
creating clearly distinct representations of instructions and data tokens
without introducing any additional parameters."

The pi/2 angle is NOT stated on the abstract page. It is reported to us as the
ablation optimum (SEP 71.4% on Qwen3-8B) but is marked [UNVERIFIED-ANGLE] until
someone reads the full text. The angle is a CONFIG knob here, defaulting to
pi/2, precisely so the claim can be tested rather than assumed -- if pi/2 is not
special in our sweep, that is a finding, not a bug.

WHY THIS METHOD IS WORTH REPRODUCING ON A 4090
-----------------------------------------------
It adds NO parameters. The rotation is a fixed change of basis, so it can be
implemented exactly and verified exactly -- orthogonality, norm preservation and
involution are arithmetic facts, not empirical ones. That makes it one of the
few 2026 defences whose CORE is fully checkable on a small budget.

WHAT WE CAN AND CANNOT REPRODUCE HERE -- READ BEFORE QUOTING ANY NUMBER
------------------------------------------------------------------------
CAN (no training needed):
  * the rotation itself: orthogonal, norm-preserving, parameter-free
  * layer-wise linear-probe separability of instruction vs data tokens
  * the cosine-similarity trajectory of rotated vs vanilla hidden states
  * the causal check: undo the rotation at inference and see what moves
  * rotation vs a learned OFFSET (ISE), which the paper reports collapses
    toward vanilla while the rotation does not

CANNOT (needs SFT on the modified forward, which the paper does and we do not):
  * the SEP scores (Llama-2-7B 68.7->81.0, Mistral-7B 48.0->92.1,
    Qwen3-8B 45.3->71.4)
  * the ASR reductions (BIPIA-text 14.7->4.9, StruQ-ID 45.6->28.1)
  * anything about utility
A rotation applied to a model that was never trained with it will DEGRADE that
model. Reporting our ASR without the SFT would be measuring our own omission.

THE LIMIT THAT MATTERS MOST, AND IT IS NOT A NUMBER
----------------------------------------------------
The rotation is only as strong as the ROLE CHANNEL that decides which tokens are
data. If role is inferred from delimiters inside attacker-controlled text, the
attacker forges the delimiter and the split never happens. The label must come
from the application, out of band -- the "immutable provenance tag" that
arXiv:2606.27567 proves is necessary. `rotate_embeddings` therefore takes an
explicit boolean mask and will NOT parse roles out of text.

CPU-only. numpy/torch-agnostic core. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["isoclinic_matrix", "rotate_vectors", "rotate_embeddings",
           "learned_offset_baseline", "DEFAULT_ANGLE"]

# Reported ablation optimum. [UNVERIFIED-ANGLE]: not on the abstract page.
DEFAULT_ANGLE = math.pi / 2


def isoclinic_matrix(dim: int, theta: float = DEFAULT_ANGLE) -> np.ndarray:
    """Block-diagonal isoclinic rotation: the same 2x2 rotation on every pair.

    At theta = pi/2 each pair (x1, x2) -> (-x2, x1), i.e. swap-and-negate, which
    is the form the method is usually written in.

    `dim` must be even; an odd dimension would leave one coordinate unrotated
    and silently weaken the separation on that axis, so it is refused.
    """
    if dim % 2:
        raise ValueError(
            "isoclinic rotation needs an EVEN dimension, got %d. An odd "
            "dimension leaves one coordinate unpaired and unrotated, which "
            "silently weakens the split on that axis." % dim)
    c, s = math.cos(theta), math.sin(theta)
    R = np.zeros((dim, dim), dtype=np.float64)
    for i in range(0, dim, 2):
        R[i, i] = c
        R[i, i + 1] = -s
        R[i + 1, i] = s
        R[i + 1, i + 1] = c
    return R


def rotate_vectors(X, theta: float = DEFAULT_ANGLE) -> np.ndarray:
    """Apply the rotation to every row of X. Pure, allocates a new array."""
    X = np.asarray(X)
    R = isoclinic_matrix(X.shape[-1], theta)
    return (X @ R.T).astype(X.dtype, copy=False)


def rotate_embeddings(E, is_data_mask, theta: float = DEFAULT_ANGLE) -> np.ndarray:
    """Rotate ONLY the rows flagged as data. Instruction rows are untouched.

    Parameters
    ----------
    E : [n_tokens, dim] embeddings, in sequence order.
    is_data_mask : [n_tokens] booleans from the APPLICATION, never parsed out of
        the text. This function deliberately has no way to infer roles: a role
        channel an attacker can write is not a role channel (arXiv:2606.27567).

    The paper leaves instructions alone and rotates data, which matters: it
    means an un-rotated deployment is bit-identical to vanilla on instruction
    tokens, so any change we measure is attributable to the data channel.
    """
    E = np.asarray(E)
    m = np.asarray(is_data_mask, dtype=bool)
    if m.shape[0] != E.shape[0]:
        raise ValueError(
            "mask length %d does not match %d embedding rows; a misaligned "
            "role mask would rotate the wrong tokens and still look fine."
            % (m.shape[0], E.shape[0]))
    out = np.array(E, copy=True)
    if m.any():
        out[m] = rotate_vectors(E[m], theta)
    return out


def learned_offset_baseline(E, is_data_mask, offset) -> np.ndarray:
    """The ISE-style contrast: ADD a vector to data tokens instead of rotating.

    Kept because the paper's central geometric argument is comparative -- an
    offset is a translation the residual stream can cancel in one bias term,
    while a rotation is a change of basis every later linear map must re-learn.
    Reported there as ISE collapsing toward vanilla (cosine > 0.9) where the
    rotation does not. Without this arm, "the rotation persists" has nothing to
    persist against.
    """
    E = np.asarray(E)
    m = np.asarray(is_data_mask, dtype=bool)
    off = np.asarray(offset, dtype=E.dtype).reshape(1, -1)
    if off.shape[1] != E.shape[1]:
        raise ValueError("offset dim %d != embedding dim %d"
                         % (off.shape[1], E.shape[1]))
    out = np.array(E, copy=True)
    out[m] = out[m] + off
    return out


def _self_test() -> None:
    rng = np.random.default_rng(0)
    dim = 8
    R = isoclinic_matrix(dim)

    assert np.allclose(R @ R.T, np.eye(dim), atol=1e-12)
    assert abs(abs(np.linalg.det(R)) - 1.0) < 1e-12
    print("OK  the matrix is ORTHOGONAL (R R^T = I, |det| = 1)")

    x = rng.normal(size=(64, dim))
    y = rotate_vectors(x)
    assert np.allclose(np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1))
    print("OK  norms are preserved exactly -- no information is destroyed")

    # pi/2 is swap-and-negate per pair
    v = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    got = rotate_vectors(v)[0]
    want = np.array([-2.0, 1.0, -4.0, 3.0, -6.0, 5.0, -8.0, 7.0])
    assert np.allclose(got, want), got
    print("OK  at pi/2 the map is (x1,x2)->(-x2,x1): %s" % got.tolist()[:4])

    # four applications return to identity; two give -I (not identity!)
    assert np.allclose(rotate_vectors(rotate_vectors(rotate_vectors(
        rotate_vectors(x)))), x, atol=1e-12)
    assert not np.allclose(rotate_vectors(rotate_vectors(x)), x)
    print("OK  R^4 = I but R^2 != I -- it is a rotation, not a reflection")

    # the rotated basis is orthogonal to the original for every row
    cos = np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) ** 2)
    assert np.allclose(cos, 0.0, atol=1e-12)
    print("OK  every rotated vector is ORTHOGONAL to its original (cos = 0)")

    E = rng.normal(size=(10, dim))
    mask = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0], dtype=bool)
    Eo = rotate_embeddings(E, mask)
    assert np.allclose(Eo[~mask], E[~mask])
    assert not np.allclose(Eo[mask], E[mask])
    print("OK  ONLY data rows move; instruction rows are bit-identical to vanilla")

    try:
        rotate_embeddings(E, mask[:-1])
    except ValueError as exc:
        assert "misaligned role mask" in str(exc)
        print("OK  a misaligned role mask is REFUSED, not silently applied")
    else:
        raise AssertionError("misaligned mask accepted")

    try:
        isoclinic_matrix(7)
    except ValueError as exc:
        assert "EVEN dimension" in str(exc)
        print("OK  an odd dimension is refused (one axis would stay unrotated)")
    else:
        raise AssertionError("odd dim accepted")

    # the comparative claim: an offset is cancellable by a single bias, a
    # rotation is not. Demonstrated directly rather than asserted.
    off = rng.normal(size=dim)
    Eo_ise = learned_offset_baseline(E, mask, off)
    undo = Eo_ise.copy()
    undo[mask] -= off
    assert np.allclose(undo, E)
    print("OK  the ISE OFFSET is undone exactly by one additive bias")
    resid = rotate_embeddings(E, mask) - E
    assert not np.allclose(resid[mask], resid[mask][0])
    print("OK  the ROTATION's displacement is input-DEPENDENT, so no single "
          "bias can cancel it -- the paper's core geometric argument")

    ang = {}
    for th in (0.0, math.pi / 4, math.pi / 2):
        yy = rotate_vectors(x, th)
        ang[round(th, 4)] = float(np.mean(np.sum(x * yy, axis=1)
                                          / (np.linalg.norm(x, axis=1) ** 2)))
    assert ang[0.0] > 0.99 and abs(ang[round(math.pi / 2, 4)]) < 1e-9
    print("OK  angle sweeps: cos(orig, rotated) = %s"
          % {k: round(v, 3) for k, v in ang.items()})
    print("")
    print("OK -- aside.py: the rotation is orthogonal, norm-preserving, "
          "role-masked, and provably not an offset.")


if __name__ == "__main__":
    _self_test()
