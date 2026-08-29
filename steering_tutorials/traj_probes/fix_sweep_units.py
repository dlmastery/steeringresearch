"""fix_sweep_units.py -- re-score the layer sweep at the UNIT the bar lives at.

sweep_layers.py records `LinearTrajProbe.auc`, which is a ROW-level AUC over
turn-rows, and compares it to a per-TRAJECTORY content bar. Those are different
units on different n, so `margin_vs_bar` and `clears_content_bar` in the sweep
file are not the comparison they claim to be. Same mistake the headline had --
results.json's own notes flagged it there, and the sweep reproduced it because
it never asked probes.py for the matched number.

This recomputes, per layer, the trajectory-pooled (max) probe AUC from the
CACHED bundle and re-derives the margin against the same layer-independent bar.
CPU-only: every layer's activations are already on disk, so no GPU and no model
load -- which is the point of having cached them.

It writes a SEPARATE file rather than editing the sweep's, so the mismatched
numbers stay visible next to the corrected ones instead of being quietly
replaced.

    C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.traj_probes.fix_sweep_units

ASCII stdout only.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

import steering_tutorials.common.netboot as netboot
import steering_tutorials.traj_probes.config as C
from steering_tutorials.traj_probes.activations import (ExtractSettings,
                                                        bundle_cache_path,
                                                        data_fingerprint)
from steering_tutorials.traj_probes.data import load_corpus
from steering_tutorials.traj_probes.probes import (_fit_fold, group_folds,
                                                   trajectory_scores)


def main() -> int:
    netboot.enable()
    src = sorted(glob.glob(str(C.ARTIFACTS / "layer_sweep_*.json")))
    src = [p for p in src if "unitfixed" not in p]
    if not src:
        print("no layer_sweep_*.json found -- run sweep_layers first")
        return 1
    d = json.loads(open(src[0], encoding="utf-8").read())
    bar = float(d["content_bar_auc"])

    corpus = load_corpus()
    settings = ExtractSettings()
    data_fp = data_fingerprint(corpus, settings)
    expected_rows = sum(t.n_turns for t in corpus.trajectories)
    print("corpus expects %d turn-rows; a bundle with any other row "
          "count is stale" % expected_rows)

    out = []
    print("bar (trajectory-level, layer-independent): %.4f" % bar)
    print("")
    print("%-6s %-11s %-11s %-9s %-9s %s"
          % ("layer", "row_auc", "traj_auc", "margin", "clears", "note"))
    # Re-score EVERY layer that has a cached bundle, not only those the sweep
    # file lists. L12 was produced by run_traj_probes, not sweep_layers, so a
    # sweep-file-driven loop silently omits the headline layer from its own
    # "is this a layer artifact" table -- the one layer the question is about.
    by_layer = {int(r["layer"]): r for r in d["layers"]}
    import re as _re
    for f in C.ARTIFACTS.glob("acts_%s_L*_*.npz" % C.MODEL_TAG):
        m = _re.search(r"_L(\d+)_", f.name)
        if m:
            by_layer.setdefault(int(m.group(1)), {"layer": int(m.group(1)),
                                                  "auc": float("nan")})
    for rec in sorted(by_layer.values(), key=lambda r: int(r["layer"])):
        L = int(rec["layer"])
        # The bundle's data fingerprint covers ExtractSettings, which CARRIES
        # THE LAYER, so one fingerprint cannot address every layer's file.
        # Match on the layer segment of the name instead of recomputing a
        # fingerprint that is only correct for the default layer.
        # Pick by CONTENT, never by sort order. Two L12 bundles exist -- the
        # corrected one and a STALE pre-fix one whose rows were dropped in a
        # label-correlated way -- and `sorted(...)[-1]` picked the stale one on
        # fingerprint spelling alone, inflating L12 from 0.7453 to 0.7967. A
        # glob that returns several files is an AMBIGUOUS ARTIFACT, which is
        # exactly what common/artifact_paths.py refuses by name; it has to be
        # refused here too rather than resolved alphabetically.
        cand = sorted(C.ARTIFACTS.glob("acts_%s_L%d_*.npz" % (C.MODEL_TAG, L)))
        keep = []
        for q in cand:
            try:
                with np.load(q, allow_pickle=True) as zz:
                    if int(zz["X"].shape[0]) == expected_rows:
                        keep.append(q)
            except (OSError, ValueError, KeyError):
                continue
        if len(keep) > 1:
            raise SystemExit(
                "AMBIGUOUS: %d cached bundles for L%d match the current corpus "
                "(%s). Cannot attribute a number to one of them."
                % (len(keep), L, ", ".join(q.name for q in keep)))
        if not keep:
            print("%-6d %-11s %-11s %-9s %-9s no bundle matches the current "
                  "corpus (%d rows); %d stale candidate(s) ignored"
                  % (L, "-", "-", "-", "-", expected_rows, len(cand)))
            continue
        path = keep[0]
        if not path.exists():
            print("%-6d %-11s %-11s %-9s %-9s cached bundle missing"
                  % (L, rec.get("auc", "-"), "-", "-", "-"))
            continue
        z = np.load(path, allow_pickle=True)
        X = z["X"]
        y = np.asarray(z["y"]).astype(int)
        uid = np.array([str(u) for u in z["traj_uid"]])
        gid = z["group_id"]
        # the SAME row-level group folds LinearTrajProbe uses (probes.py:370)
        splits = group_folds(gid, y, n_folds=C.N_FOLDS, seed=C.SEED)
        row = np.full(len(y), np.nan)
        for tr, te in splits:
            if len(np.unique(y[tr])) < 2:
                continue
            row[te] = _fit_fold(X[tr].astype(np.float64), y[tr],
                                X[te].astype(np.float64), 1.0, C.SEED)
        good = ~np.isnan(row)
        tu, ts, tl = trajectory_scores(uid[good], row[good], y[good], how="max")
        traj_auc = float(roc_auc_score(np.asarray(tl).astype(int),
                                       np.asarray(ts, dtype=float)))
        rec2 = dict(rec)
        rec2["row_auc"] = rec.get("auc")
        rec2["trajectory_auc"] = round(traj_auc, 4)
        rec2["margin_vs_bar"] = round(traj_auc - bar, 4)
        rec2["clears_content_bar"] = bool(traj_auc > bar)
        rec2["unit"] = "trajectory (max-pooled), matched to the bar"
        out.append(rec2)
        ra = rec.get("auc")
        print("%-6d %-11s %-11.4f %+0.4f   %-9s %s"
              % (L, "-" if ra is None or ra != ra else "%.4f" % ra,
                 traj_auc, traj_auc - bar, traj_auc > bar, "unit-matched"))

    dst = C.ARTIFACTS / ("layer_sweep_unitfixed_%s_%s.json"
                         % (C.CORPUS, C.MODEL_TAG))
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"content_bar_auc": round(bar, 4),
         "unit_note": ("probe AUCs here are trajectory-pooled (max), the SAME "
                       "unit as the bar. sweep_layers.py's own file records "
                       "ROW-level AUCs against this same bar, which is a unit "
                       "mismatch; both are kept so the error stays visible."),
         "layers": out}, indent=1), encoding="utf-8")
    os.replace(tmp, dst)
    print("")
    if out:
        best = max(out, key=lambda r: r["trajectory_auc"])
        print("best layer L%d at %.4f (trajectory unit); bar %.4f -> %s"
              % (best["layer"], best["trajectory_auc"], bar,
                 "CLEARS" if best["clears_content_bar"]
                 else "still BELOW: the negative is not a layer artifact"))
    print("wrote %s" % dst.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
