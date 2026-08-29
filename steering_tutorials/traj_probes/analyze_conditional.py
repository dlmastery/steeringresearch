"""analyze_conditional.py -- what is the probe actually detecting?

The headline says the activation probe (0.7503) loses to a unigram bar (0.8671).
That is a comparison, not an explanation. This asks what the probe's score is
made of, and tests one specific hypothesis to destruction.

THE HYPOTHESIS, AND WHY IT WAS WORTH TESTING
--------------------------------------------
ATBench's `risk_source` tag is ONE-SIDED: of 248 benign-tagged trajectories, 0
end unsafe. Of 749 threat-tagged, 252 (33.6%) still end safe. So a detector that
knew only "does this prompt carry a threat", and nothing whatever about what the
agent then did, would score AUC 0.7480.

Our probe scores 0.7503. Near-identical. The obvious reading is that the probe
has learned to recognise a risky SETUP rather than to predict an emergent
failure -- which would make "early prediction" mostly prompt classification.

THE TEST THAT SETTLES IT
------------------------
Restrict to the threat-tagged subset. There the tag is constant, so it carries
no information, and only the outcome varies. A pure threat-detector must fall to
chance. Measured: the probe scores 0.7482 there, far above 0.5.

So the hypothesis is REFUTED. The probe is genuinely predicting outcome among
trajectories that all start out risky, and the numeric coincidence was a
coincidence. Recorded because the negative headline should not be allowed to
imply a deflationary story that the data does not support.

The bar, meanwhile, rises to 0.9216 on that same subset -- it beats the probe by
MORE on the harder conditional task, not less.

A CAVEAT ON THE CI, found while reconciling two numbers
-------------------------------------------------------
This script builds CV folds at the TRAJECTORY level; `LinearTrajProbe` builds
them at the ROW level (probes.py:370, `group_folds(bundle.group_id, ...)` over
8,341 rows). Both are group-aware and neither leaks, but they are different
partitions, and they disagree: 0.7719 here vs 0.7503 there.

That 0.0216 gap is LARGER than the reported bootstrap half-width (+/-0.016). The
CI resamples groups at a FIXED fold assignment, so it does not include
fold-construction variance. Quote the two numbers against their own pipeline,
never across -- and read the CI as narrower than the true uncertainty.

CPU-only: reads the cached bundle, loads no model. ASCII stdout.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

import steering_tutorials.common.netboot as netboot
import steering_tutorials.traj_probes.config as C
from steering_tutorials.traj_probes.activations import (ExtractSettings,
                                                        bundle_cache_path,
                                                        data_fingerprint)
from steering_tutorials.traj_probes.data import load_corpus
from steering_tutorials.traj_probes.probes import (_fit_fold, content_bar_control,
                                                   group_folds, trajectory_scores)


def _bar(uids, labels, texts) -> float:
    r = content_bar_control(list(uids), [int(v) for v in labels], texts, seed=C.SEED)
    return float(r["auc"] if isinstance(r, dict) else r)


def main() -> int:
    netboot.enable()
    corpus = load_corpus()
    settings = ExtractSettings()
    path = bundle_cache_path(C.ARTIFACTS, C.MODEL_ID, settings, C.LAYER,
                             data_fingerprint(corpus, settings))
    if not path.exists():
        print("no bundle at %s -- run extract_acts first" % path.name)
        return 1
    z = np.load(path, allow_pickle=True)
    X = z["X"]
    uid = np.array([str(u) for u in z["traj_uid"]])
    y = np.asarray(z["y"]).astype(int)

    # trajectory-level folds (see the CI caveat in the module docstring)
    order, lab, seen = [], [], set()
    for u, yy in zip(uid, y):
        if u not in seen:
            seen.add(u)
            order.append(u)
            lab.append(yy)
    order, lab = np.array(order), np.array(lab)
    fold = {}
    for f, (_tr, te) in enumerate(group_folds(order, lab, n_folds=C.N_FOLDS,
                                              seed=C.SEED)):
        for i in te:
            fold[order[i]] = f
    fidx = np.array([fold[u] for u in uid])
    row = np.full(len(y), np.nan)
    for f in range(C.N_FOLDS):
        te = np.flatnonzero(fidx == f)
        tr = np.flatnonzero(fidx != f)
        row[te] = _fit_fold(X[tr].astype(np.float64), y[tr],
                            X[te].astype(np.float64), 1.0, C.SEED)
    tu, ts, tl = trajectory_scores(uid, row, y, how="max")
    tu = np.asarray([str(u) for u in tu])
    ts = np.asarray(ts, dtype=float)
    tl = np.asarray(tl).astype(int)

    risk = {t.uid: (t.risk_source or "unknown") for t in corpus.trajectories}
    rs = np.array([risk[u] for u in tu])
    threat = rs != "benign"

    oracle = np.where(threat, 1.0, 0.0)
    out = {
        "layer": C.LAYER,
        "n_trajectories": int(len(tu)),
        "probe_auc_trajectory_folds": round(float(roc_auc_score(tl, ts)), 4),
        "oracle_risk_tag_auc": round(float(roc_auc_score(tl, oracle)), 4),
        "benign_tagged": int((~threat).sum()),
        "benign_tagged_unsafe": int(((~threat) & (tl == 1)).sum()),
        "threat_tagged": int(threat.sum()),
        "threat_tagged_safe": int((threat & (tl == 0)).sum()),
        "conditional": {
            "n": int(threat.sum()),
            "probe_auc": round(float(roc_auc_score(tl[threat], ts[threat])), 4),
            "content_bar_auc": round(_bar(
                tu[threat], tl[threat],
                {t.uid: t.text for t in corpus.trajectories
                 if (t.risk_source or "unknown") != "benign"}), 4),
        },
    }

    # where in the trajectory does the SURFACE signal live?
    uids = [t.uid for t in corpus.trajectories]
    ys = [t.label for t in corpus.trajectories]
    out["content_bar_decomposition"] = {
        "opening_user_turn_only": round(_bar(
            uids, ys, {t.uid: (t.turns[0].content if t.turns else "")
                       for t in corpus.trajectories}), 4),
        "everything_after_turn_1": round(_bar(
            uids, ys, {t.uid: "\n".join("%s: %s" % (x.role, x.content)
                                        for x in t.turns[1:])
                       for t in corpus.trajectories}), 4),
        "full_trajectory": round(_bar(
            uids, ys, {t.uid: t.text for t in corpus.trajectories}), 4),
    }

    p = C.ARTIFACTS / ("conditional_L%d.json" % C.LAYER)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    import os
    os.replace(tmp, p)

    print("probe AUC (trajectory-level folds) : %.4f" % out["probe_auc_trajectory_folds"])
    print("ORACLE that knows ONLY the risk tag: %.4f" % out["oracle_risk_tag_auc"])
    print("  benign-tagged %d, of which unsafe %d"
          % (out["benign_tagged"], out["benign_tagged_unsafe"]))
    print("  threat-tagged %d, of which safe   %d"
          % (out["threat_tagged"], out["threat_tagged_safe"]))
    print("")
    print("CONDITIONAL on threat-tagged (tag constant, only outcome varies):")
    print("  n=%d  probe %.4f  bar %.4f   [chance 0.5]"
          % (out["conditional"]["n"], out["conditional"]["probe_auc"],
             out["conditional"]["content_bar_auc"]))
    print("  -> probe is FAR above chance, so it is NOT merely a threat detector;")
    print("     the 0.7503/0.7480 near-identity was a coincidence.")
    print("")
    d = out["content_bar_decomposition"]
    print("where the SURFACE signal lives (TF-IDF):")
    print("  opening user turn only : %.4f" % d["opening_user_turn_only"])
    print("  everything after turn 1: %.4f" % d["everything_after_turn_1"])
    print("  full trajectory        : %.4f" % d["full_trajectory"])
    print("")
    print("wrote %s" % p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
