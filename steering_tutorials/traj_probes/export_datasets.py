"""export_datasets.py -- ship traj_probes' ATBench slice beside its results.

traj_probes is the one lesson in this batch with its OWN distinct data (every
other lesson in this dispatch draws from the shared `common.data` slices and
gets a `USES_SHARED.json` pointer instead -- see each lesson's `datasets/`).

This exports the EXACT corpus `data.load_corpus()` produces at the CURRENT
config (config.py: CORPUS=atbench, N_PER_CLASS=500, SEED=0, MAX_TURNS=16,
MAX_TURN_CHARS=8000) -- the same ~500 safe / 497 unsafe trajectories every
number in `artifacts/results_atbench_gemma-3-1b-it_L12.json` etc. was computed
from (the corpus cache `artifacts/corpus_atbench_n500_s0_t16_c8000.json.gz`
already on disk is that exact pool, so this reads it rather than re-fetching).

SUPERSEDES the earlier "atbench_n500_s0_t16_c1200" slice: MAX_TURN_CHARS was
raised from 1200 to 8000 (team-lead correction, 2026-08-29) because 1200
truncated ~10% of ALL turns when its only job was clipping a handful of
enormous tool dumps (per-turn chars p50 347, p90 1,088, p99 50,915). At 8000
chars with a 16,384-token extraction budget (MAX_TOKENS, also raised from a
hard-coded 4096) it clips only 1.46% of turns (122 of 8,341) and drops no
rows -- outlier control, not subsampling. The c1200 slice + manifest were
deleted; do not resurrect them.

One row per TRAJECTORY (not per turn -- a turn-row export would be ~8.3k rows
of the same text repeated across every extracted layer's viewer; the trajectory
is the natural unit and turns nest under it), carrying: uid, label, group_id,
risk_source, failure_mode, real_world_harm, rationale, and its turns
(index, role, content, action). `rationale` is the corpus's own post-hoc label
explanation (data.py: "it describes the answer and must never reach a probe") --
it travels with the export for transparency, exactly as it does in the cache,
but no runner in this lesson may feed it to a probe.

CPU-only, no model, no GPU, no network (reads the cache already on disk).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steering_tutorials.common.dataset_export import export_slice
from steering_tutorials.traj_probes import config as C
from steering_tutorials.traj_probes.data import load_corpus


def _traj_row(t) -> dict:
    return {
        "uid": t.uid,
        "label": t.label,
        "group_id": t.group_id,
        "risk_source": t.risk_source,
        "failure_mode": t.failure_mode,
        "real_world_harm": t.real_world_harm,
        "rationale": t.rationale,
        "turns": [
            {"index": u.index, "role": u.role, "content": u.content,
             "action": u.action}
            for u in t.turns
        ],
    }


def main() -> None:
    assert C.CORPUS == "atbench", (
        "export_datasets.py is written for the atbench config (500/500); "
        "CORPUS=%r -- re-check the name/config before exporting." % C.CORPUS)

    corpus = load_corpus()   # config-anchor defaults: n=500, seed=0, max_turns=16
    rows = [_traj_row(t) for t in corpus.trajectories]

    print("[export] loaded %d trajectories (%d safe / %d unsafe), "
          "pool_fingerprint=%s" % (len(rows), corpus.achieved_n_neg,
                                   corpus.achieved_n_pos, corpus.pool_fingerprint))

    man = export_slice(
        Path(__file__).resolve().parent,
        "atbench_n500_s0_t16_c8000",
        C.HF_DATASET,
        rows,
        split=C.HF_SPLIT,
        seed=C.SEED,
        notes=(
            "One row per trajectory (uid, label, group_id, risk_source, "
            "failure_mode, real_world_harm, rationale, turns[index,role,"
            "content,action]). Turns are pre-truncated to config.MAX_TURN_CHARS "
            "(8000 chars/turn) and config.MAX_TURNS (first 16 turns/trajectory) "
            "-- the exact truncation the model and every content-bar control in "
            "this lesson read, per data.py's MAX_TURN_CHARS docstring. At 8000 "
            "chars (with a 16,384-token extraction budget, config.MAX_TOKENS) "
            "this clips only 1.46%% of turns (122 of 8,341) and drops NO rows -- "
            "it is outlier control on a handful of enormous tool dumps (p99 "
            "50,915 chars), not sampling, so this slice is COMPLETE at the turn "
            "level, not a subsample. (Supersedes the earlier 1200-char cap, "
            "which truncated ~10%% of all turns; that slice has been deleted.) "
            "`rationale` is the corpus's post-hoc verdict explanation and must "
            "never be fed to a probe (data.py: 'it describes the answer'). "
            "Sampled without replacement, balanced per class, seed=%d, from the "
            "%s config of %s (%s)." % (C.SEED, C.HF_CONFIG, C.HF_DATASET,
                                       C.DATASET_PAPER)
        ),
        extra={
            "pool_fingerprint": corpus.pool_fingerprint,
            "corpus_config": C.HF_CONFIG,
            "n_per_class_requested": C.N_PER_CLASS,
            "max_turns": C.MAX_TURNS,
            "max_turn_chars": C.MAX_TURN_CHARS,
            "max_tokens": C.MAX_TOKENS,
            "label_provenance": corpus.label_provenance,
            "step_label_provenance": corpus.step_label_provenance,
            "dataset_paper": C.DATASET_PAPER,
        },
    )
    print("[export] wrote %s (%d rows, %.2f MB gz, fp=%s)"
          % (man["file"], man["n_rows"], man["bytes_gz"] / 1e6,
             man["slice_fingerprint"]))


if __name__ == "__main__":
    main()
