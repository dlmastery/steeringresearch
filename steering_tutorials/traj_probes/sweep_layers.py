"""sweep_layers.py -- is the layer-12 negative a property of the LAYER?

The headline result (linear probe 0.7503 vs a 0.8671 unigram bar) was measured at
one layer, chosen as a default rather than found. Before that negative is
reported as "the residual stream does not beat the surface here", the layer axis
has to be ruled out: mid-network is where these papers put the signal, but
"mid" is model-dependent and Gemma-3-1B has 26 layers.

This sweeps layers and reports, for each, the probe AUC against the SAME
trajectory-level TF-IDF bar (which does not depend on the layer, so it is
computed once and reused). A layer clears only if it beats that bar.

One model load per layer, one forward pass per trajectory, journalled -- a reap
costs one trajectory. ~4 min/layer on this host.

    C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.traj_probes.sweep_layers
    TP_SWEEP_LAYERS=4,8,16,20,24  ... to choose the layers

ASCII stdout only (Windows cp1252).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import steering_tutorials.common.netboot as netboot
import steering_tutorials.traj_probes.config as C
from steering_tutorials.common.artifact_paths import keyed_path
from steering_tutorials.traj_probes.activations import (ExtractSettings,
                                                        HFActivationExtractor)
from steering_tutorials.traj_probes.data import load_corpus
from steering_tutorials.traj_probes.probes import (LinearTrajProbe,
                                                   content_bar_control)

DEFAULT_LAYERS = (4, 8, 12, 16, 20, 24)


def main() -> int:
    netboot.enable()
    raw = os.environ.get("TP_SWEEP_LAYERS", "")
    layers = ([int(x) for x in raw.split(",") if x.strip()] if raw
              else list(DEFAULT_LAYERS))

    corpus = load_corpus()
    uids = [t.uid for t in corpus.trajectories]
    y = [t.label for t in corpus.trajectories]
    texts = {t.uid: t.text for t in corpus.trajectories}

    # The bar reads the text, not the activations, so it is layer-INDEPENDENT:
    # computed once and used as the same fixed bar for every layer.
    bar_res = content_bar_control(uids, y, texts, seed=C.SEED)
    bar = float(bar_res["auc"] if isinstance(bar_res, dict) else bar_res)
    print("content bar (layer-independent, computed once): AUC %.4f" % bar)
    print("corpus: %d trajectories, %d turn-rows"
          % (len(corpus.trajectories), sum(t.n_turns for t in corpus.trajectories)))
    print("")

    out_path = keyed_path(C.ARTIFACTS, "layer_sweep", ".json",
                          C.CORPUS, C.MODEL_TAG)
    rows = []
    if out_path.exists():                      # resume across a reap
        try:
            rows = json.loads(out_path.read_text(encoding="utf-8")).get("layers", [])
            done = {r["layer"] for r in rows}
            layers = [L for L in layers if L not in done]
            if done:
                print("[resume] layers already done: %s" % sorted(done))
        except (OSError, ValueError):
            rows = []

    settings = ExtractSettings()
    for L in layers:
        t0 = time.time()
        ex = HFActivationExtractor(C.MODEL_ID, settings, cache_dir=C.ARTIFACTS)
        bundle = ex.extract(corpus, L)
        probe = LinearTrajProbe()
        r = probe.fit_predict_cv(bundle, n_folds=C.N_FOLDS, seed=C.SEED)
        auc = float(r.auc)
        res = None if r.auc_residualised is None else float(r.auc_residualised)
        rows.append({"layer": L, "auc": round(auc, 4),
                     "auc_residualised": None if res is None else round(res, 4),
                     "auc_ci": [round(float(v), 4) for v in r.auc_ci],
                     "content_bar_auc": round(bar, 4),
                     "clears_content_bar": bool(auc > bar),
                     "margin_vs_bar": round(auc - bar, 4),
                     "n_rows": int(r.n_items),
                     "seconds": round(time.time() - t0, 1)})
        # write after EVERY layer: a reap must not cost the layers already done
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"config": C.as_dict(), "content_bar_auc": round(bar, 4),
             "layers": sorted(rows, key=lambda d: d["layer"])},
            indent=1), encoding="utf-8")
        os.replace(tmp, out_path)
        print("L%-3d auc=%.4f res=%s bar=%.4f margin=%+.4f clears=%s (%.0fs)"
              % (L, auc, "None" if res is None else "%.4f" % res, bar,
                 auc - bar, auc > bar, time.time() - t0))

    rows.sort(key=lambda d: d["layer"])
    print("")
    print("SWEEP (Gemma-3-1B / ATBench; bar is the SAME text, layer-independent)")
    print("%-6s %-9s %-9s %-9s %s" % ("layer", "auc", "auc_res", "margin", "clears"))
    for d in rows:
        print("%-6d %-9.4f %-9s %+0.4f    %s"
              % (d["layer"], d["auc"],
                 "None" if d["auc_residualised"] is None else "%.4f" % d["auc_residualised"],
                 d["margin_vs_bar"], d["clears_content_bar"]))
    best = max(rows, key=lambda d: d["auc"]) if rows else None
    if best:
        print("")
        print("best layer L%d at %.4f; bar %.4f; %s"
              % (best["layer"], best["auc"], bar,
                 "CLEARS the bar" if best["clears_content_bar"]
                 else "still BELOW the bar -- the negative is not a layer artifact"))
    print("wrote %s" % out_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
