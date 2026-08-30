"""sweep_pooling.py -- is the negative a property of the POOLING?

The headline probe reads the LAST TOKEN of each turn. That was a default, like
layer 12 was, and it went unexamined for the whole first pass of this lesson
while the layer axis got a six-point sweep. Asked directly whether "various
trajectory probes" had been tried, the honest answer was no: one architecture,
one pooling. This closes the pooling half.

THE THREE POOLINGS, AND WHY ONE OF THEM CANNOT BE HEADLINED
------------------------------------------------------------
  last          the last token of turn k. The decision-relevant position, and
                what reproductions A and B read.
  mean_turn     mean over turn k's OWN tokens. A legitimate alternative: it
                reads the whole turn rather than its final position.
  mean_prefix   mean over every token up to the end of turn k. **CONFOUNDED
                WITH STEP INDEX BY CONSTRUCTION** -- as k grows the averaging
                pool grows, so the feature encodes k directly. It is run here
                so the confound can be SHOWN rather than asserted, and its
                residualised number is the only one worth reading.

Each pooling is scored against the SAME layer-independent content bar and at the
SAME trajectory unit, so the rows are comparable to each other and to the layer
sweep. A pooling clears only if it beats the bar.

Costs one extraction per pooling (the bundle cache key includes pooling), so
about 8 minutes each on this host, journalled against a reap.

    C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.traj_probes.sweep_pooling
    TP_SWEEP_POOLINGS=last,mean_turn   ... to choose

ASCII stdout only (Windows cp1252).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import steering_tutorials.common.netboot as netboot
import steering_tutorials.traj_probes.config as C
from steering_tutorials.common.artifact_paths import keyed_path
from steering_tutorials.traj_probes.activations import (ExtractSettings,
                                                        HFActivationExtractor)
from steering_tutorials.traj_probes.data import load_corpus
from steering_tutorials.traj_probes.probes import (StepResidualiser, _fit_fold,
                                                   content_bar_control,
                                                   group_folds,
                                                   trajectory_scores)

DEFAULT_POOLINGS = ("last", "mean_turn", "mean_prefix")


def _oof_trajectory_auc(bundle, residualise: bool):
    y = np.asarray(bundle.y).astype(int)
    uid = np.array([str(u) for u in bundle.traj_uid])
    step = np.asarray(bundle.step_index).astype(float)
    X = np.asarray(bundle.X, dtype=np.float64)
    splits = group_folds(bundle.group_id, y, n_folds=C.N_FOLDS, seed=C.SEED)
    row = np.full(len(y), np.nan)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, Xte = X[tr], X[te]
        if residualise:
            r = StepResidualiser(degree=1).fit(Xtr, step[tr])
            Xtr, Xte = r.transform(Xtr, step[tr]), r.transform(Xte, step[te])
        row[te] = _fit_fold(Xtr, y[tr], Xte, 1.0, C.SEED)
    good = ~np.isnan(row)
    tu, ts, tl = trajectory_scores(uid[good], row[good], y[good], how="max")
    return float(roc_auc_score(np.asarray(tl).astype(int),
                               np.asarray(ts, dtype=float)))


def main() -> int:
    netboot.enable()
    raw = os.environ.get("TP_SWEEP_POOLINGS", "")
    poolings = ([p.strip() for p in raw.split(",") if p.strip()] if raw
                else list(DEFAULT_POOLINGS))

    corpus = load_corpus()
    uids = [t.uid for t in corpus.trajectories]
    ys = [t.label for t in corpus.trajectories]
    bar_res = content_bar_control(uids, ys,
                                  {t.uid: t.text for t in corpus.trajectories},
                                  seed=C.SEED)
    bar = float(bar_res["auc"] if isinstance(bar_res, dict) else bar_res)
    print("content bar (pooling-independent, computed once): AUC %.4f" % bar)
    print("corpus: %d trajectories, %d turn-rows"
          % (len(corpus.trajectories),
             sum(t.n_turns for t in corpus.trajectories)))
    print("")

    out_path = keyed_path(C.ARTIFACTS, "pooling_sweep", ".json",
                          C.CORPUS, C.MODEL_TAG, "L%d" % C.LAYER)
    rows = []
    if out_path.exists():
        try:
            rows = json.loads(out_path.read_text(encoding="utf-8")).get("poolings", [])
            done = {r["pooling"] for r in rows}
            poolings = [p for p in poolings if p not in done]
            if done:
                print("[resume] already done: %s" % sorted(done))
        except (OSError, ValueError):
            rows = []

    shared = None
    for pool in poolings:
        t0 = time.time()
        settings = ExtractSettings(pooling=pool)
        if shared is None:
            shared = HFActivationExtractor(C.MODEL_ID, settings,
                                           cache_dir=C.ARTIFACTS)
            shared._ensure_model()
        ex = HFActivationExtractor(C.MODEL_ID, settings, cache_dir=C.ARTIFACTS,
                                   model=shared._model, tok=shared._tok)
        bundle = ex.extract(corpus, C.LAYER)
        auc = _oof_trajectory_auc(bundle, residualise=False)
        res = _oof_trajectory_auc(bundle, residualise=True)
        rows.append({
            "pooling": pool, "layer": C.LAYER,
            "trajectory_auc": round(auc, 4),
            "trajectory_auc_residualised": round(res, 4),
            "content_bar_auc": round(bar, 4),
            "margin_vs_bar": round(auc - bar, 4),
            "clears_content_bar": bool(auc > bar),
            "confounded_with_step_index": pool == "mean_prefix",
            "n_rows": int(bundle.X.shape[0]),
            "seconds": round(time.time() - t0, 1),
        })
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"config": C.as_dict(),
                                   "content_bar_auc": round(bar, 4),
                                   "unit": "trajectory (max-pooled)",
                                   "poolings": rows}, indent=1),
                       encoding="utf-8")
        os.replace(tmp, out_path)
        print("%-12s auc=%.4f  residualised=%.4f  margin=%+.4f  clears=%s (%.0fs)%s"
              % (pool, auc, res, auc - bar, auc > bar, time.time() - t0,
                 "   <- CONFOUNDED, read the residualised column"
                 if pool == "mean_prefix" else ""))

    print("")
    print("POOLING SWEEP (Gemma-3-1B / ATBench L%d; trajectory unit, bar %.4f)"
          % (C.LAYER, bar))
    print("%-12s %-9s %-14s %-9s %s"
          % ("pooling", "auc", "residualised", "margin", "clears"))
    for d in sorted(rows, key=lambda r: -r["trajectory_auc"]):
        print("%-12s %-9.4f %-14.4f %+0.4f    %s"
              % (d["pooling"], d["trajectory_auc"],
                 d["trajectory_auc_residualised"], d["margin_vs_bar"],
                 d["clears_content_bar"]))
    honest = [d for d in rows if not d["confounded_with_step_index"]]
    if honest:
        b = max(honest, key=lambda d: d["trajectory_auc"])
        print("")
        print("best UNCONFOUNDED pooling: %s at %.4f; bar %.4f -> %s"
              % (b["pooling"], b["trajectory_auc"], bar,
                 "CLEARS" if b["clears_content_bar"]
                 else "still BELOW -- the negative is not a pooling artifact"))
    print("wrote %s" % out_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
