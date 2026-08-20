"""shade.py -- the LONG-HORIZON corpus: `control_arena_shade` + its monitor-labels.

*** This is the corpus that DECIDES the lesson (types.py's module docstring). ***
It is the only pairing in the whole DATASETS_VERIFIED.md sweep with per-step labels,
an oracle flag step, AND a point-of-no-return step -- the only corpus that can carry
a lead-time claim at all.

DATA
----
Two HF repos, joined by ROW INDEX (row N of one aligns to row N of the other, per
DATASETS_VERIFIED.md):

  `adityaasinha28/control_arena_shade` (1500 rows, 750/750) -- ONE row = one FULL
  trajectory: `prompt`, `ground_truth` (0/1), `trajectory_data` (a JSON STRING
  holding a `messages` list -- where the per-step structure lives), `transcript`,
  `cleaned_misalignment_evidence`, `extra_info.main_task_description`,
  `setting_description`.

  `aksh-n/control_arena_shade-monitor-labels` (1497 judge-log JSON files, a FILE
  repo -- not parquet, so `datasets-server` cannot page it; fetched by URL via
  `huggingface_hub.hf_hub_download`) -- one Gemini-3.1-Pro judge log per row:
    * `priv_judge.with_evidence.step_analyses[]`      -> Step.on_path,
      Step.severity_if_confirmed, Step.irreversibility_imminent
    * `priv_judge.with_evidence.candidate_ideal_flagging_step` (-1 for benign in the
      raw data -> stored as `None` on Trajectory, per types.py's contract)
    * `nonpriv_streaming[]`                            -> Step.would_flag_now,
      Step.cumulative_confidence (a genuinely CAUSAL online reference monitor --
      predicting THIS is monitor DISTILLATION, not scheming detection; keep that arm
      separate from the ground_truth arm, per types.py)
    * `adjudicator.point_of_no_return_step`            -> Trajectory.point_of_no_return_step
    * `verification.passed`                            -> the ADMISSION FILTER

Measured (DATASETS_VERIFIED.md, uniform 60-row sample): class balance ~750/750,
steps/trajectory min 30 / median 131 / mean 132.2 / max 422 -- long trajectories,
where the streaming-vs-mean-pooling argument has real room to matter.

LEAD-TIME CAVEAT (load-bearing, do not drop when reporting on this corpus)
---------------------------------------------------------------------------
A wider audit (DATASETS_VERIFIED.md "LEAD-TIME FEASIBILITY") measured lead = PoNR -
ideal_flag on a 188-row sample: MEDIAN 0, mean 7.61 carried almost entirely by a
five-point tail (132, 112, 56, 54, 50), and 60% of positives offer ZERO or NEGATIVE
advance warning. Report lead as a DISTRIBUTION (`LeadTimeResult`'s percentiles), and
as "fraction detected strictly before point_of_no_return_step" -- never as a mean.

LICENCE / PROVENANCE (load-bearing, honest per types.py's CorpusLoader contract)
----------------------------------------------------------------------------------
Neither repo declares a licence -- `Corpus.licence = "unstated"`, not "MIT" (the
upstream SHADE-Arena CODE is MIT-licensed; these community re-dumps assert nothing).
Every per-step label is Gemini-3.1-Pro OUTPUT, not human ground truth --
`Corpus.label_provenance` says so explicitly. Gemini is off-family from Gemma
(CLAUDE.md section 17 rule 3), which is necessary but not sufficient: lead time is
measured against a judge ORACLE, never against truth.

CPU-only to import. `.load()` downloads (network + disk cache under
`../artifacts/corpora/`); nothing happens at import time. ASCII stdout/stderr only.
"""
from __future__ import annotations

from steering_tutorials.streaming_trajectory_aggregation.corpora import (
    build_control_arena_pair, cached_build, eprint, uids_fingerprint,
)
from steering_tutorials.streaming_trajectory_aggregation.types import Corpus

__all__ = ["ShadeLoader"]

DATASET_ID = "adityaasinha28/control_arena_shade"
LABELS_ID = "aksh-n/control_arena_shade-monitor-labels"


class ShadeLoader:
    """CorpusLoader for the SHADE ControlArena pair (satisfies `types.CorpusLoader`
    structurally -- `name`, `hf_id`, `load(n_per_class, seed) -> Corpus`)."""

    name = "shade"
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
            label_provenance="ControlArena `ground_truth` 0/1 field -- the task's own attack/benign designation, JUDGE-FREE (this is what the AUC ladder is scored against)",
            step_label_provenance="gemini-3.1-pro-preview judge log (LLM-generated, not human) -- oracle flagging step and point-of-no-return; lead time is measured against THIS, never against truth",
        )


if __name__ == "__main__":
    # Small manual smoke -- NOT run by this agent (network + a real HF download).
    # `python -m steering_tutorials.streaming_trajectory_aggregation.corpora.shade`
    eprint("[smoke] ShadeLoader().load(n_per_class=5, seed=0) ...")
    corpus = ShadeLoader().load(n_per_class=5, seed=0)
    s = corpus.shortfall()
    print("SHADE smoke: achieved pos=%d neg=%d groups=%d shortfall=%s clears_rule_1=%s"
          % (corpus.achieved_n_pos, corpus.achieved_n_neg, corpus.n_distinct_groups,
             s["shortfall"], s["clears_rule_1"]))
    print("step_count_summary:", corpus.step_count_summary())
    if corpus.trajectories:
        t = corpus.trajectories[0]
        print("sample uid=%s label=%d n_steps=%d ideal_flag=%s ponr=%s available_lead=%s"
              % (t.uid, t.label, t.n_steps, t.ideal_flagging_step,
                 t.point_of_no_return_step, t.available_lead))
    print("licence=%r label_provenance=%r pool_fingerprint=%s..."
          % (corpus.licence, corpus.label_provenance, corpus.pool_fingerprint[:16]))
