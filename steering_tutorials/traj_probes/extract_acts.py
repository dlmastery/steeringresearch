"""extract_acts.py -- the GPU half, run on its own so a reap is cheap.

Extraction is the only expensive step in this lesson (~10 s/trajectory on this
host, ~2.8 h for the full 997) and everything after it is CPU-cheap. Splitting
them means the analysis can be re-run any number of times against one cached
bundle, and a reaped extraction resumes instead of restarting.

RESUMABILITY IS NOT OPTIONAL HERE. This host reaps long jobs (CLAUDE.md 18.5:
three kills in one session with RAM healthy each time), so `cache_dir` is always
passed and the extractor journals every trajectory as it completes. Re-running
this script after a reap picks up at the last completed trajectory.

Usage (foreground preferred; background only with the journal, which is default):
    C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.traj_probes.extract_acts

Env caps, for shrinking it into one window:
    TP_RUN_N          per-class request (default: config.N_PER_CLASS)
    TP_LAYER          residual layer (default: config.LAYER)
    TP_MAX_TURNS      turn cap (default: config.MAX_TURNS=16; see leakage.py)

ASCII stdout only (Windows cp1252).
"""
from __future__ import annotations

import os
import sys
import time

import steering_tutorials.common.netboot as netboot
import steering_tutorials.traj_probes.config as C
from steering_tutorials.traj_probes.activations import (ExtractSettings,
                                                        HFActivationExtractor,
                                                        bundle_cache_path,
                                                        data_fingerprint,
                                                        save_bundle)
from steering_tutorials.traj_probes.data import (load_corpus, summarise,
                                                 verify_against_paper)
from steering_tutorials.traj_probes.leakage import deterministic_step_region


def main() -> int:
    netboot.enable()
    n = int(os.environ.get("TP_RUN_N", C.N_PER_CLASS))
    layer = int(os.environ.get("TP_LAYER", C.LAYER))

    t0 = time.time()
    corpus = load_corpus(n_per_class=n)          # gated: refuses the step leak
    print(summarise(corpus))
    print("")

    region = deterministic_step_region(corpus)
    print("step-leak gate     region empty=%s (boundary=%s, max turns by label %s)"
          % (region["is_empty"], region["boundary"], region["max_turns_by_label"]))
    anchor = verify_against_paper(corpus)
    print("parser anchor      mean turns %s vs paper %s -- %s"
          % (anchor["got"]["mean_turns"], anchor["paper"]["mean_turns"],
             "MATCHES" if anchor["mean_turns_matches"] else
             "differs (expected on a capped/sampled pool)"))
    n_rows = sum(t.n_turns for t in corpus.trajectories)
    print("to extract         %d trajectories, %d turn-rows, layer %d, model %s"
          % (len(corpus.trajectories), n_rows, layer, C.MODEL_ID))
    print("")

    settings = ExtractSettings()
    cache_dir = C.ARTIFACTS
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_fp = data_fingerprint(corpus, settings)
    out_path = bundle_cache_path(cache_dir, C.MODEL_ID, settings, layer, data_fp)

    ex = HFActivationExtractor(C.MODEL_ID, settings, cache_dir=cache_dir)
    t1 = time.time()
    bundle = ex.extract(corpus, layer)
    dt = time.time() - t1

    # Save BEFORE the summary print: a late crash must not cost the run.
    save_bundle(out_path, bundle, {
        "config": C.as_dict(),
        "corpus": {"name": corpus.name, "licence": corpus.licence,
                   "pool_fingerprint": corpus.pool_fingerprint,
                   "requested_n_per_class": corpus.requested_n_per_class,
                   "achieved_n_safe": corpus.achieved_n_neg,
                   "achieved_n_unsafe": corpus.achieved_n_pos,
                   "label_provenance": corpus.label_provenance,
                   "turns": corpus.turn_count_summary()},
        "step_leak_region": region,
        "paper_anchor": anchor,
        "extract_seconds": round(dt, 1),
    })
    print("")
    print("[done] %d rows x %d dims over %d trajectories in %.1f min (%.2f s/traj)"
          % (bundle.X.shape[0], bundle.X.shape[1],
             len(set(bundle.traj_uid.tolist())), dt / 60.0,
             dt / max(len(corpus.trajectories), 1)))
    print("[done] bundle  %s" % out_path.name)
    print("[done] fingerprint %s" % bundle.behaviour_fingerprint)
    print("[done] total elapsed %.1f min" % ((time.time() - t0) / 60.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
