"""atbench.py -- `AI45Research/ATBench`, Apache-2.0, TRAJECTORY-level labels only.

*** BORDERLINE ON THE >=500/class FLOOR -- STATE THIS IN ANY RESULT BUILT FROM IT ***
`ATBench/test` = 1000 rows, measured **503 negative / 497 positive**
(DATASETS_VERIFIED.md). Positives land 3 SHORT of the CLAUDE.md section 17 rule-1
floor. Sister repos `ATBench-Claw` (500) / `ATBench-Codex` (248) exist but are
DIFFERENT agent scaffolds -- pooling them would silently confound the corpus, so
this loader does NOT pool them. At `n_per_class=500` the positive class WILL report
a shortfall (`Corpus.shortfall()["shortfall"] is True`, achieved_n_pos<=497) --
that is correct behaviour, not a bug; do not raise it to 500 by pooling in the
sister repos without saying so.

DATA
----
`ATBench/test`, 1000 rows. Columns: `id`, `tool_used` (list of full tool JSON specs
-- the "~2,000 tools" the brief cites; not carried onto `Trajectory`/`Step`, see the
spine-gap note below), `contents` (nested multi-step trajectory, SAME shape as
ASSEBench's -- rendered by the shared `corpora.render_contents`), `label` (0/1),
`risk_source` (e.g. `tool_description_injection`), `failure_mode`, `reason`
(free-text rationale NAMING the decisive step in prose, but with no numeric index --
extracting one would require an LLM judge and would make any lead-time number
judge-derived and circular; ATBench is therefore explicitly NOT a lead-time
benchmark), `real_world_harm`.

STEP LABELS: NONE. Every `Step.on_path` / `Step.would_flag_now` / etc. and every
`Trajectory.ideal_flagging_step` / `.point_of_no_return_step` stays `None` -- a
valid Corpus shape per types.py.

KNOWN GAP AGAINST THE SPINE (reported, not worked around): `tool_used` (the tool
spec list) and `reason` (the free-text rationale naming the decisive step) have no
field on the frozen `Trajectory`/`Step` dataclasses. Both are dropped by this
loader rather than stuffed into `Step.text` or invented as a fake numeric field.

GROUPING: no natural cross-row grouping field was found in the verified schema.
`group_id = uid` (rule-4 fallback).

`Trajectory.source = "atbench"`.

CPU-only to import. `.load()` downloads (network + disk cache under
`../artifacts/corpora/`); nothing happens at import time. ASCII stdout/stderr only.
"""
from __future__ import annotations

import random

from steering_tutorials.streaming_trajectory_aggregation.corpora import (
    cached_build, eprint, render_contents, uids_fingerprint,
)
from steering_tutorials.streaming_trajectory_aggregation.types import Corpus, Step, Trajectory

__all__ = ["ATBenchLoader"]

HF_ID = "AI45Research/ATBench"
CONFIG = "ATBench"
SPLIT = "test"


def _build_trajectories(n_per_class: int, seed: int) -> list:
    from datasets import load_dataset as hf_load

    try:
        ds = hf_load(HF_ID, CONFIG, split=SPLIT)
    except Exception as exc:
        eprint("[atbench] %s/%s FAILED to load: %s" % (HF_ID, CONFIG, exc))
        return []
    eprint("[atbench] %s/%s: %d rows" % (HF_ID, CONFIG, len(ds)))

    pos, neg = [], []
    for i, row in enumerate(ds):
        try:
            label = int(row.get("label"))
        except (TypeError, ValueError):
            continue
        if label not in (0, 1):
            continue
        (pos if label == 1 else neg).append((i, row))

    eprint("[atbench] pool: pos=%d neg=%d (of %d rows; DATASETS_VERIFIED.md measured "
           "503/497 -- positives are BORDERLINE, 3 short of the 500 floor)"
           % (len(pos), len(neg), len(ds)))

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    trajectories = []
    for label, pool in ((1, pos), (0, neg)):
        kept = 0
        for i, row in pool:
            if kept >= n_per_class:
                break
            texts = render_contents(row.get("contents"))
            if not texts:
                eprint("[atbench] row %d: empty/unparsable contents -- skipped" % i)
                continue
            steps = tuple(Step(index=j, text=t) for j, t in enumerate(texts))
            uid = "atbench:%s" % row.get("id", i)
            trajectories.append(Trajectory(
                uid=uid, steps=steps, label=label, group_id=uid, source="atbench",
            ))
            kept += 1
        if kept < n_per_class:
            eprint("=" * 70)
            eprint("[atbench] WARNING: SHORTFALL -- only %d/%d admitted for label=%d "
                   "(pool exhausted -- EXPECTED for label=1, see module docstring)"
                   % (kept, n_per_class, label))
            eprint("=" * 70)

    rng.shuffle(trajectories)
    eprint("[atbench] admitted %d trajectories total" % len(trajectories))
    return trajectories


class ATBenchLoader:
    """CorpusLoader for ATBench (trajectory-level only, Apache-2.0, positives
    borderline on the >=500/class floor -- see module docstring)."""

    name = "atbench"
    hf_id = HF_ID

    def load(self, n_per_class: int = 500, seed: int = 0) -> Corpus:
        def _build():
            return _build_trajectories(n_per_class, seed)

        trajectories = cached_build(self.name, [HF_ID], n_per_class, seed, _build)
        return Corpus(
            name=self.name,
            trajectories=trajectories,
            requested_n_per_class=n_per_class,
            pool_fingerprint=uids_fingerprint([t.uid for t in trajectories]),
            licence="Apache-2.0",
            label_provenance="ATBench dataset-author annotation (trajectory-level; no per-step judge)",
        )


if __name__ == "__main__":
    # Small manual smoke -- NOT run by this agent (network + a real HF download).
    # `python -m steering_tutorials.streaming_trajectory_aggregation.corpora.atbench`
    eprint("[smoke] ATBenchLoader().load(n_per_class=5, seed=0) ...")
    corpus = ATBenchLoader().load(n_per_class=5, seed=0)
    s = corpus.shortfall()
    print("ATBENCH smoke: achieved pos=%d neg=%d groups=%d shortfall=%s clears_rule_1=%s"
          % (corpus.achieved_n_pos, corpus.achieved_n_neg, corpus.n_distinct_groups,
             s["shortfall"], s["clears_rule_1"]))
    print("step_count_summary:", corpus.step_count_summary())
    print("licence=%r label_provenance=%r" % (corpus.licence, corpus.label_provenance))
