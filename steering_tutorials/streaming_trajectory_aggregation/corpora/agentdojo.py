"""agentdojo.py -- the SHORT-HORIZON CONTROL: `control_arena_agentdojo` + monitor-labels.

THIS CORPUS'S WHOLE JOB IS THE CONTRAST WITH `shade.py` (types.py's module
docstring): AgentDojo trajectories have median **8** steps vs SHADE's median 131.
Running the same aggregator ladder on both turns "does mean pooling collapse on long
horizons" from an assertion into a measurement with a built-in control -- if mean
pooling degrades on SHADE but not here, the HORIZON is the cause.

DATA -- IDENTICAL SCHEMA to `shade.py` (DATASETS_VERIFIED.md: "the SAME construction
as the SHADE one"), so this loader is a thin wrapper around the shared
`corpora.build_control_arena_pair`:

  `adityaasinha28/control_arena_agentdojo` (1500 rows, 750/750, 3.7 MB -- much
  smaller per-trajectory than SHADE's 151 MB, i.e. much shorter trajectories) --
  same 11 columns: `prompt`, `ground_truth`, `trajectory_data` (JSON-string
  `messages`), `extra_info.main_task_description`, etc.

  `aksh-n/control_arena_agentdojo-monitor-labels` (1500 JSON judge logs) -- same
  four label tracks as SHADE's.

*** LEAD TIME IS ZERO ON THIS CORPUS -- BY CONSTRUCTION, AND THAT IS THE POINT ***
DATASETS_VERIFIED.md measured this directly (38 sampled rows, 18 positives, 17 with
valid oracle fields): `ideal_flagging_step == point_of_no_return_step` IDENTICALLY
in all 17 valid positives (mean/median/min/max lead all 0.00). This loader does
**NOT** synthesise a lead that is not there -- it reads `candidate_ideal_flagging_step`
and `adjudicator.point_of_no_return_step` from the log exactly as `shade.py` does,
with no special-casing, and if the real download reproduces the DATASETS_VERIFIED.md
sample, `Trajectory.available_lead` will come out 0 for every positive here.
**AgentDojo cannot support a lead-time/advance-warning claim.** It CAN validate
streaming-monitor ACCURACY (it has the same per-step `on_path`/`would_flag_now`
tracks as SHADE) -- that is what it is for.

LICENCE / PROVENANCE -- same as SHADE: neither repo declares a licence
(`Corpus.licence = "unstated"`); every per-step label is Gemini-3.1-Pro output, not
human (`Corpus.label_provenance` says so).

CPU-only to import. `.load()` downloads (network + disk cache); nothing happens at
import time. ASCII stdout/stderr only.
"""
from __future__ import annotations

from steering_tutorials.streaming_trajectory_aggregation.corpora import (
    build_control_arena_pair, cached_build, eprint, uids_fingerprint,
)
from steering_tutorials.streaming_trajectory_aggregation.types import Corpus

__all__ = ["AgentDojoLoader"]

DATASET_ID = "adityaasinha28/control_arena_agentdojo"
LABELS_ID = "aksh-n/control_arena_agentdojo-monitor-labels"


class AgentDojoLoader:
    """CorpusLoader for the AgentDojo ControlArena pair -- the short-horizon
    (median 8 steps, lead=0) control for `ShadeLoader`."""

    name = "agentdojo"
    hf_id = DATASET_ID

    def load(self, n_per_class: int = 500, seed: int = 0) -> Corpus:
        def _build():
            return build_control_arena_pair(self.name, DATASET_ID, LABELS_ID, n_per_class, seed)

        trajectories = cached_build(self.name, [DATASET_ID, LABELS_ID], n_per_class, seed, _build)
        return Corpus(
            name=self.name,
            trajectories=trajectories,
            requested_n_per_class=n_per_class,
            pool_fingerprint=uids_fingerprint([t.uid for t in trajectories]),
            licence="unstated",
            label_provenance="gemini-3.1-pro-preview judge log (LLM-generated, not human)",
        )


if __name__ == "__main__":
    # Small manual smoke -- NOT run by this agent (network + a real HF download).
    # `python -m steering_tutorials.streaming_trajectory_aggregation.corpora.agentdojo`
    eprint("[smoke] AgentDojoLoader().load(n_per_class=5, seed=0) ...")
    corpus = AgentDojoLoader().load(n_per_class=5, seed=0)
    s = corpus.shortfall()
    print("AGENTDOJO smoke: achieved pos=%d neg=%d groups=%d shortfall=%s clears_rule_1=%s"
          % (corpus.achieved_n_pos, corpus.achieved_n_neg, corpus.n_distinct_groups,
             s["shortfall"], s["clears_rule_1"]))
    print("step_count_summary:", corpus.step_count_summary())
    leads = [t.available_lead for t in corpus.trajectories if t.available_lead is not None]
    print("available_lead values seen: %s (expect all 0, per DATASETS_VERIFIED.md)" % leads)
    print("licence=%r label_provenance=%r pool_fingerprint=%s..."
          % (corpus.licence, corpus.label_provenance, corpus.pool_fingerprint[:16]))
