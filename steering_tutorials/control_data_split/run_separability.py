"""run_separability.py -- the GPU pass for ASIDE's training-free geometric claims.

WHAT THIS TESTS, AND WHAT IT CANNOT
------------------------------------
ASIDE's headline numbers (SEP scores, ASR reductions, utility) all require SFT on
the modified forward, which we do not run. What survives without any training is
the GEOMETRY, and that is what this measures:

  1. LAYER-WISE SEPARABILITY. Can a linear probe tell instruction-role from
     data-role representations at layer L? The paper reports the rotated model
     is separable from the embedding layer onward while vanilla only becomes
     separable later.
  2. THE COSINE TRAJECTORY. How far do rotated hidden states stay from vanilla
     ones as depth grows? The paper reports the network never fully undoes the
     rotation.
  3. THE ISE CONTRAST. Same two measurements for an additive OFFSET, which the
     paper reports collapses toward vanilla where the rotation does not.

THE CONTROL THAT DECIDES WHETHER ANY OF IT MEANS ANYTHING
----------------------------------------------------------
A probe separating instruction from data in the VANILLA model is not measuring
the rotation -- it is measuring that the two corpora are different text. Our
instruction role is ultrachat and our data role is toxic-chat/JBB, which differ
in register, topic and length, so vanilla separability will NOT be at chance and
must be reported as the floor. Only the ROTATED-MINUS-VANILLA gap is evidence
about ASIDE, and `provenance_floor` (arXiv:2606.27567's Bayes bound on any
provenance detector for this corpus pair) is reported beside it.

ANGLE IS SWEPT, not assumed. The pi/2 optimum is reported to us but is NOT on
the paper's abstract page, so it is marked [UNVERIFIED-DETAIL] throughout this
lesson. If separability is flat across angles then pi/2 is numerology on this
model, which is a finding.

    C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.control_data_split.run_separability
    CDS_ANGLES=0.7854,1.5708,2.3562   to sweep angles
    CDS_N=200                          to shrink into one window

ASCII stdout only (Windows cp1252).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

import steering_tutorials.common.netboot as netboot
import steering_tutorials.control_data_split.config as C
from steering_tutorials.control_data_split.data import load_role_corpus
from steering_tutorials.control_data_split.separability import (
    cosine_trajectory, diff_of_means_offset, extract_hidden_states,
    layer_separability_sweep, provenance_floor)


def _auc(d):
    """layer_separability_sweep returns {layer: {"auc":..}}; take the AUC."""
    return {k: float(v["auc"] if isinstance(v, dict) else v) for k, v in d.items()}

def _auc2(d):
    """cosine_trajectory may return a scalar or a dict per layer."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = v.get("cosine", v.get("mean_cosine", list(v.values())[0]))
        out[k] = float(v)
    return out


def main() -> int:
    netboot.enable()
    n = int(os.environ.get("CDS_N", C.N_PER_ROLE))
    raw = os.environ.get("CDS_ANGLES", "")
    angles = ([float(a) for a in raw.split(",") if a.strip()] if raw
              else [C.ANGLE])

    t0 = time.time()
    corpus = load_role_corpus(n_per_role=n, seed=C.SEED)
    texts = corpus["texts"]
    is_data = np.asarray(corpus["is_data"], dtype=bool)
    print("corpus: %d texts (%d instruction / %d data), licence %s"
          % (len(texts), int((~is_data).sum()), int(is_data.sum()),
             corpus.get("licence", "?")))

    # The impossibility bound for THIS corpus pair, before any model runs.
    floor = provenance_floor([t for t, d in zip(texts, is_data) if not d],
                             [t for t, d in zip(texts, is_data) if d])
    print("provenance floor (arXiv:2606.27567): TV %s over a same-distribution "
          "floor of %s -> Bayes error %s"
          % (floor.get("tv_raw"), floor.get("tv_same_distribution_floor"),
             floor.get("bayes_error_raw")))
    print("")

    from steering_tutorials.hello_world_steering.model_utils import load_model
    model, tok = load_model(C.MODEL_ID)

    out = {"config": C.as_dict(), "corpus": {
        "n": len(texts), "n_instruction": int((~is_data).sum()),
        "n_data": int(is_data.sum()), "licence": corpus.get("licence"),
        "pool_fingerprint": corpus.get("pool_fingerprint")},
        "provenance_floor": floor, "angles": {}}

    print("extracting VANILLA (the FLOOR: these two corpora differ as TEXT, so "
          "this is not chance)")
    van = extract_hidden_states(model, tok, texts, is_data, C.LAYERS,
                                mode="vanilla")
    van_sep = _auc(layer_separability_sweep(van, is_data.astype(int), seed=C.SEED))
    out["vanilla_separability"] = {str(k): round(float(v), 4)
                                   for k, v in van_sep.items()}
    print("  vanilla separability by layer: %s"
          % {k: round(float(v), 3) for k, v in sorted(van_sep.items())})
    print("")

    for th in angles:
        key = "%.4f" % th
        print("--- angle %.4f rad (%.0f deg) ---" % (th, math.degrees(th)))
        rot = extract_hidden_states(model, tok, texts, is_data, C.LAYERS,
                                    mode="rotate", theta=th)
        rot_sep = _auc(layer_separability_sweep(rot, is_data.astype(int), seed=C.SEED))
        cos_rot = _auc2(cosine_trajectory(rot, van))

        # ISE contrast: an additive offset of the same scale
        off = diff_of_means_offset(van[C.LAYERS[0]], is_data.astype(int))
        ise = extract_hidden_states(model, tok, texts, is_data, C.LAYERS,
                                    mode="offset", offset=off)
        ise_sep = _auc(layer_separability_sweep(ise, is_data.astype(int), seed=C.SEED))
        cos_ise = _auc2(cosine_trajectory(ise, van))

        gap = {str(k): round(float(rot_sep[k] - van_sep[k]), 4) for k in rot_sep}
        out["angles"][key] = {
            "theta": th,
            "rotated_separability": {str(k): round(float(v), 4) for k, v in rot_sep.items()},
            "rotated_minus_vanilla": gap,
            "cosine_rotated_vs_vanilla": {str(k): round(float(v), 4) for k, v in cos_rot.items()},
            "ise_separability": {str(k): round(float(v), 4) for k, v in ise_sep.items()},
            "cosine_ise_vs_vanilla": {str(k): round(float(v), 4) for k, v in cos_ise.items()},
        }
        print("  rotated sep : %s" % {k: round(float(v), 3) for k, v in sorted(rot_sep.items())})
        print("  GAP vs van  : %s" % {int(k): v for k, v in sorted(gap.items(), key=lambda kv: int(kv[0]))})
        print("  cos(rot,van): %s" % {k: round(float(v), 3) for k, v in sorted(cos_rot.items())})
        print("  cos(ise,van): %s" % {k: round(float(v), 3) for k, v in sorted(cos_ise.items())})
        print("")

    p = C.ARTIFACTS / "separability.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    print("[done] %.1f min -> %s" % ((time.time() - t0) / 60.0, p.name))
    print("")
    print("READ THE GAP, NOT THE ROTATED COLUMN. Vanilla separability is the")
    print("floor here because the two roles are drawn from different corpora;")
    print("only rotated-minus-vanilla is evidence about the rotation itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
