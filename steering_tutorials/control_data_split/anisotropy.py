"""anisotropy.py -- WHY the ASIDE rotation is detectable at all.

THE OBSERVATION THIS FILE EXISTS FOR
------------------------------------
While writing separability.py's synthetic fixture, an agent's first attempt used
isotropic zero-mean Gaussian embeddings, and the assertion "rotated data is
linearly separable from vanilla" FAILED at AUC 0.47. That was not a bug in the
probe. A rotationally-symmetric distribution is invariant under rotation: if
X ~ N(0, sigma^2 I) then RX has the SAME law, so no classifier -- linear or
otherwise -- can tell rotated from unrotated. The probe correctly found nothing.

That failure is worth more than the fixture, and chasing it corrected my own
first explanation. I assumed detectability came from embeddings being
ANISOTROPIC and OFF-ORIGIN, and wrote a test asserting anisotropy alone would do
it. That test FAILED at AUC 0.52, and it was right to.

A LINEAR probe separates by a difference in MEANS. Rotating a centred cloud maps
a zero mean to a zero mean: the covariance changes (Sigma -> R Sigma R^T) but the
means do not, so no linear classifier can tell the two apart however anisotropic
the cloud is. A quadratic probe could -- the covariance genuinely differs -- but
that is not what ASIDE's linear-separability claim is about.

So the condition is narrower and sharper than "anisotropic": the embedding cloud
must sit OFF THE ORIGIN. The centroid displacement under rotation, ||c - Rc||, is
the entire signal a linear probe reads at layer 0. If Gemma's embedding matrix
were centred, ASIDE's rotation would be linearly INVISIBLE there no matter how
structured the embeddings were.

That is a property of the embedding matrix which the paper does not measure and
nobody has to grant, so this module measures it before the expensive arm runs.

WHAT IS MEASURED
----------------
  mean_norm / centroid_norm   how far the cloud sits from the origin. THIS is
                              the load-bearing quantity: a rotation about the
                              origin maps a centred cloud's mean to itself, so
                              only an off-centre cloud is displaced.
  centroid_rotation_displacement  ||c - Rc||, absolute and relative to the mean
                              vector norm -- the direct magnitude of what a
                              layer-0 linear probe has to work with.
  participation_ratio         effective number of dimensions carrying variance.
                              Reported as CONTEXT, not as the mechanism: it is
                              measured below that anisotropy alone does not make
                              the rotation linearly detectable, so a low PR is
                              not evidence that ASIDE will work.
  top1 / top10 variance share how concentrated the spectrum is.
  separability_after_rotation the direct test: fit a linear probe to tell
                              rotated embeddings from originals. This is the
                              quantity ASIDE needs to be high, and the synthetic
                              isotropic case pins it at chance.

The isotropic CONTROL is run beside every real number, at matched shape, so the
reader can see what "no signal" looks like on the same axis rather than being
asked to trust that 0.5 is the floor.

CPU-only: reads the embedding matrix, never runs a forward pass.
ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from steering_tutorials.control_data_split.aside import (DEFAULT_ANGLE,
                                                         rotate_vectors)

__all__ = ["spectrum_stats", "rotation_detectability", "report"]


def spectrum_stats(E) -> dict:
    E = np.asarray(E, dtype=np.float64)
    n, d = E.shape
    centroid = E.mean(axis=0)
    Xc = E - centroid
    # eigenvalues of the covariance, via SVD of the centred matrix
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s ** 2) / max(n - 1, 1)
    tot = float(lam.sum())
    pr = float((tot ** 2) / float((lam ** 2).sum())) if tot > 0 else float("nan")
    norms = np.linalg.norm(E, axis=1)
    return {
        "n": int(n), "dim": int(d),
        "mean_norm": round(float(norms.mean()), 4),
        "centroid_norm": round(float(np.linalg.norm(centroid)), 4),
        # how much of a typical vector's length is the SHARED centroid
        "centroid_over_mean_norm": round(
            float(np.linalg.norm(centroid) / max(norms.mean(), 1e-12)), 4),
        "participation_ratio": round(pr, 2),
        "participation_ratio_over_dim": round(pr / d, 4),
        "top1_variance_share": round(float(lam[0] / tot), 4) if tot > 0 else None,
        "top10_variance_share": round(float(lam[:10].sum() / tot), 4) if tot > 0 else None,
        # the load-bearing quantity: how far the CENTROID moves under rotation.
        # This, not anisotropy, is what a layer-0 linear probe reads.
        "centroid_rotation_displacement": round(float(np.linalg.norm(
            centroid - rotate_vectors(centroid.reshape(1, -1))[0])), 4),
        "centroid_displacement_over_mean_norm": round(float(np.linalg.norm(
            centroid - rotate_vectors(centroid.reshape(1, -1))[0])
            / max(norms.mean(), 1e-12)), 4),
    }


def rotation_detectability(E, theta: float = DEFAULT_ANGLE, seed: int = 0) -> dict:
    """Can a linear probe tell rotated embeddings from the originals?

    This is the quantity ASIDE needs to be HIGH. On an isotropic centred cloud
    it is pinned at chance no matter how much data you give it, because the two
    distributions are identical.
    """
    E = np.asarray(E, dtype=np.float64)
    R = rotate_vectors(E, theta)
    X = np.vstack([E, R])
    y = np.concatenate([np.zeros(len(E)), np.ones(len(R))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    auc = float(roc_auc_score(yte, clf.decision_function(sc.transform(Xte))))
    # and the same after CENTERING, which removes the off-origin advantage
    Ec = E - E.mean(axis=0)
    Rc = rotate_vectors(Ec, theta)
    Xc = np.vstack([Ec, Rc])
    Xtr, Xte, ytr, yte = train_test_split(Xc, y, test_size=0.3,
                                          random_state=seed, stratify=y)
    sc2 = StandardScaler().fit(Xtr)
    clf2 = LogisticRegression(max_iter=2000, random_state=seed).fit(
        sc2.transform(Xtr), ytr)
    auc_c = float(roc_auc_score(yte, clf2.decision_function(sc2.transform(Xte))))
    return {"auc_rotated_vs_original": round(auc, 4),
            "auc_after_centering": round(auc_c, 4),
            "theta": float(theta)}


def report(E, name: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    real = spectrum_stats(E)
    det = rotation_detectability(E, seed=seed)
    # matched-shape ISOTROPIC control: same n, same dim, same mean norm
    n, d = np.asarray(E).shape
    G = rng.normal(size=(n, d))
    G = G / np.linalg.norm(G, axis=1, keepdims=True) * real["mean_norm"]
    ctrl = spectrum_stats(G)
    ctrl_det = rotation_detectability(G, seed=seed)
    return {"name": name, "real": real, "rotation": det,
            "isotropic_control": ctrl, "isotropic_control_rotation": ctrl_det}


def _load_embedding_matrix(model_id: str) -> np.ndarray:
    """The input embedding matrix only -- no forward pass, no GPU."""
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                             device_map="cpu")
    W = m.get_input_embeddings().weight.detach().float().cpu().numpy()
    del m
    return W


def main() -> int:
    import steering_tutorials.control_data_split.config as C

    n_sample = int(os.environ.get("CDS_ANISO_N", "20000"))
    W = _load_embedding_matrix(C.MODEL_ID)
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(W.shape[0], size=min(n_sample, W.shape[0]), replace=False)
    E = W[idx]
    out = report(E, name=C.MODEL_ID, seed=C.SEED)

    p = C.ARTIFACTS / "anisotropy.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    os.replace(tmp, p)

    r, c = out["real"], out["isotropic_control"]
    print("embedding matrix: %s  (sampled %d of %d rows, dim %d)"
          % (C.MODEL_ID, r["n"], W.shape[0], r["dim"]))
    print("")
    print("%-34s %14s %14s" % ("", "REAL", "isotropic ctrl"))
    for k in ("mean_norm", "centroid_norm", "centroid_over_mean_norm",
              "centroid_rotation_displacement",
              "centroid_displacement_over_mean_norm",
              "participation_ratio", "participation_ratio_over_dim",
              "top1_variance_share", "top10_variance_share"):
        print("%-34s %14s %14s" % (k, r[k], c[k]))
    print("")
    print("%-34s %14s %14s" % ("linear probe: rotated vs original",
                               out["rotation"]["auc_rotated_vs_original"],
                               out["isotropic_control_rotation"]["auc_rotated_vs_original"]))
    print("%-34s %14s %14s" % ("  same, after centering",
                               out["rotation"]["auc_after_centering"],
                               out["isotropic_control_rotation"]["auc_after_centering"]))
    print("")
    print("READING: the rotation is detectable only insofar as the real cloud is")
    print("anisotropic and off-origin. The isotropic control is what ASIDE would")
    print("look like on a model whose embeddings had no preferred directions.")
    print("wrote %s" % p.name)
    return 0


def _self_test() -> None:
    rng = np.random.default_rng(0)
    d = 16
    # 1. ISOTROPIC + CENTRED: rotation is undetectable, by symmetry
    G = rng.normal(size=(2000, d))
    r = rotation_detectability(G)
    assert r["auc_rotated_vs_original"] < 0.60, r
    print("OK  isotropic centred cloud: rotation is UNDETECTABLE (AUC %.3f) -- "
          "a rotationally-symmetric law is rotation-invariant"
          % r["auc_rotated_vs_original"])

    # 2. OFF-ORIGIN: the same rotation becomes detectable
    off = G + np.full(d, 5.0)
    r2 = rotation_detectability(off)
    assert r2["auc_rotated_vs_original"] > 0.95, r2
    print("OK  the SAME cloud shifted off the origin: AUC %.3f -- displacement "
          "of the centroid is what a probe reads"
          % r2["auc_rotated_vs_original"])

    # 3. and centering removes exactly that advantage
    assert r2["auc_after_centering"] < 0.60
    print("OK  centering that cloud removes it again (AUC %.3f)"
          % r2["auc_after_centering"])

    # 4. ANISOTROPIC but CENTRED: still UNDETECTABLE by a linear probe. This
    # is the assertion I first wrote backwards. A linear classifier separates by
    # a difference in MEANS, and rotation maps a zero mean to a zero mean; the
    # covariance changes but nothing linear can see it.
    A = rng.normal(size=(2000, d)) * np.linspace(8.0, 0.2, d)
    r3 = rotation_detectability(A)
    assert r3["auc_rotated_vs_original"] < 0.60, r3
    print("OK  centred but strongly ANISOTROPIC: AUC %.3f -- still at chance. "
          "Anisotropy is NOT sufficient; a linear probe reads mean differences "
          "and a rotation preserves a zero mean."
          % r3["auc_rotated_vs_original"])

    # 5. the necessary condition is the OFF-ORIGIN centroid, and the signal
    # grows with how far off it sits
    for shift in (0.5, 2.0, 8.0):
        rr = rotation_detectability(G + np.full(d, shift))
        print("     centroid shift %4.1f -> AUC %.3f"
              % (shift, rr["auc_rotated_vs_original"]))

    s_iso, s_ani = spectrum_stats(G), spectrum_stats(A)
    assert s_iso["participation_ratio"] > s_ani["participation_ratio"]
    print("OK  participation ratio still separates the two clouds (isotropic "
          "%.1f of %d dims vs anisotropic %.1f) -- it measures something real, "
          "just not the thing that makes the rotation visible"
          % (s_iso["participation_ratio"], d, s_ani["participation_ratio"]))
    print("")
    print("OK -- anisotropy.py: the condition ASIDE silently depends on is an "
          "OFF-ORIGIN embedding centroid, not anisotropy, and it is measurable "
          "before the expensive arm is run.")


if __name__ == "__main__":
    if os.environ.get("CDS_SELFTEST") == "1":
        _self_test()
    else:
        sys.exit(main())
