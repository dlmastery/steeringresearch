"""assebench.py -- `Atarogic/ASSEBench`, Apache-2.0, TRAJECTORY-level labels only.

DATA
----
Four configs, all `test` split. `strict`/`loose`/`safety` are the SAME 1476 items
RELABELLED under different annotation standards (DATASETS_VERIFIED.md confirms
identical byte counts) -- **do not pool them**. This loader uses only the two
GENUINELY DISTINCT pools: `safety` (1476: 641 neg / 835 pos) and `security` (817:
408 neg / 409 pos) -- a real combined pool of 2293 rows, comfortably clearing the
CLAUDE.md section 17 rule-1 floor (>=500/class) on both classes.

Columns: `id`, `profile` (system prompt), `contents` (nested list of
`{role, content|thought|action}` -- the multi-step agent trajectory, rendered by the
shared `corpora.render_contents`), `label` (0/1), `application_scenario`,
`risk_type`, `failure_mode`, `ambiguous` (int flag), `risk_description`.

STEP LABELS: NONE. Every `Step.on_path` / `Step.would_flag_now` / etc. stays `None`
-- that is a VALID Corpus per types.py's own docstring ("a corpus with only
trajectory-level labels still yields valid Steps with every optional field None").
A streaming monitor trains fine on the rendered steps; no lead-time claim is
possible (no oracle flag step, no point-of-no-return field) -- `Trajectory.
ideal_flagging_step` / `.point_of_no_return_step` are always `None` here.

KNOWN GAP AGAINST THE SPINE (reported, not worked around): the `ambiguous` flag --
the row-level marker of where the strict/lenient standards disagree, a genuinely
useful free robustness cut -- has NO field on the frozen `Trajectory`/`Step`
dataclasses to carry it. This loader does not invent one; it is dropped. If a
robustness cut on `ambiguous` matters later, that is a spine change for the lead,
not a workaround here.

GROUPING: no natural cross-row grouping field was found in the verified schema
(each row is independently annotated). `group_id = uid` (rule-4 fallback), so
group-aware CV degenerates to ordinary row-aware CV for this corpus -- documented,
not hidden.

`Trajectory.source` carries the CONFIG (`assebench_safety` / `assebench_security`)
so a caller can separate or recombine them without re-fetching.

CPU-only to import. `.load()` downloads (network + disk cache under
`../artifacts/corpora/`); nothing happens at import time. ASCII stdout/stderr only.
"""
from __future__ import annotations

import random

from steering_tutorials.streaming_trajectory_aggregation.corpora import (
    cached_build, eprint, render_contents, uids_fingerprint,
)
from steering_tutorials.streaming_trajectory_aggregation.types import Corpus, Step, Trajectory

__all__ = ["ASSEBenchLoader"]

HF_ID = "Atarogic/ASSEBench"
CONFIGS = ("safety", "security")  # the two DISTINCT pools; strict/loose == safety, excluded


def _build_trajectories(n_per_class: int, seed: int) -> list:
    from datasets import load_dataset as hf_load

    rows = []  # (config, row_index, row)
    for cfg in CONFIGS:
        try:
            ds = hf_load(HF_ID, cfg, split="test")
        except Exception as exc:
            eprint("[assebench] config %s FAILED to load: %s" % (cfg, exc))
            continue
        for i, row in enumerate(ds):
            rows.append((cfg, i, row))
        eprint("[assebench] config=%s rows=%d" % (cfg, len(ds)))

    pos, neg = [], []
    for cfg, i, row in rows:
        try:
            label = int(row.get("label"))
        except (TypeError, ValueError):
            continue
        if label not in (0, 1):
            continue
        (pos if label == 1 else neg).append((cfg, i, row))

    eprint("[assebench] combined pool (safety+security, strict/loose excluded as "
           "duplicates of safety): pos=%d neg=%d" % (len(pos), len(neg)))

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    trajectories = []
    for label, pool in ((1, pos), (0, neg)):
        kept = 0
        for cfg, i, row in pool:
            if kept >= n_per_class:
                break
            texts = render_contents(row.get("contents"))
            if not texts:
                eprint("[assebench] %s row %d: empty/unparsable contents -- skipped" % (cfg, i))
                continue
            steps = tuple(Step(index=j, text=t) for j, t in enumerate(texts))
            uid = "assebench:%s:%s" % (cfg, row.get("id", i))
            trajectories.append(Trajectory(
                uid=uid, steps=steps, label=label, group_id=uid,
                source="assebench_%s" % cfg,
            ))
            kept += 1
        if kept < n_per_class:
            eprint("=" * 70)
            eprint("[assebench] WARNING: SHORTFALL -- only %d/%d admitted for label=%d "
                   "(pool exhausted)" % (kept, n_per_class, label))
            eprint("=" * 70)

    rng.shuffle(trajectories)
    eprint("[assebench] admitted %d trajectories total" % len(trajectories))
    return trajectories


class ASSEBenchLoader:
    """CorpusLoader for ASSEBench (trajectory-level only, Apache-2.0)."""

    name = "assebench"
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
            label_provenance="ASSEBench dataset-author annotation (trajectory-level; no per-step judge)",
        )


if __name__ == "__main__":
    # Small manual smoke -- NOT run by this agent (network + a real HF download).
    # `python -m steering_tutorials.streaming_trajectory_aggregation.corpora.assebench`
    eprint("[smoke] ASSEBenchLoader().load(n_per_class=5, seed=0) ...")
    corpus = ASSEBenchLoader().load(n_per_class=5, seed=0)
    s = corpus.shortfall()
    print("ASSEBENCH smoke: achieved pos=%d neg=%d groups=%d shortfall=%s clears_rule_1=%s"
          % (corpus.achieved_n_pos, corpus.achieved_n_neg, corpus.n_distinct_groups,
             s["shortfall"], s["clears_rule_1"]))
    print("step_count_summary:", corpus.step_count_summary())
    print("licence=%r label_provenance=%r" % (corpus.licence, corpus.label_provenance))
