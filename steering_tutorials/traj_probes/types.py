"""types.py -- FIXED SPINE for the long-trajectory ACTIVATION-PROBE series.

The lead owns this file. Sub-lessons implement AGAINST it and never edit it
(CLAUDE.md section 17 rule 9).

WHAT THIS SERIES IS, AND HOW IT DIFFERS FROM streaming_trajectory_aggregation
-------------------------------------------------------------------------------
STA embeds trajectory TEXT with a sentence encoder and classifies the embedding.
This series probes the RESIDUAL STREAM of a model that is *reading* the
trajectory. The claim under test is not "is this text separable" but "does the
model's own internal state carry the failure signal, earlier and more
generalisably than anything observable in the text".

That difference is the point, and it is also why the confound bar matters MORE
here, not less: an activation probe that merely rediscovers the bag-of-words
signal has demonstrated nothing.

THE THREE REPRODUCTIONS
-----------------------
  A. EARLY ABORT       Ruan, Huang, Zhou, Wei, Lin, Wang and Sun, 7 Jul 2026,
                       "Doomed from the Start: Early Abort of LLM Agent Episodes
                       via a Recall-Controlled Probe Cascade" (arXiv:2607.06503)
                       -- WebFetch-VERIFIED 2026-08-27.
                       Linear probes on one fixed layer predict eventual failure
                       from the first round; a recall-controlled cascade aborts
                       doomed episodes. Reported: 60.2% / 54.9% token reduction
                       at 90% recall on TextCraft / WebShop.

  B. GOAL DRIFT        Chen, ICML 2026 AIWILD workshop, "Goal-Drift Probes:
                       Anticipating Multi-Turn LLM Agent Failure From Mid-Network
                       Activations" -- [UNVERIFIED]: workshop/OpenReview item, no
                       arXiv id supplied, NOT WebFetch-confirmed. Cite as
                       unverified until someone resolves it. Reported: failure
                       predicted 3 steps ahead at AUC 0.989, and >= 0.939 after
                       FULL step-index residualisation.

  C. TOOL-CALL GRAPH   Sun and Kazakov, 25 May 2026, "Tool-Call Dependency
                       Structure is Linearly Decodable in LLM Agent Residual
                       Streams" (arXiv:2605.25310) -- WebFetch-VERIFIED
                       2026-08-27. Low-capacity edge probes decode which earlier
                       tool output feeds which later argument, against a
                       Hewitt-Liang random-label control and a positional
                       baseline.

WE DO NOT REPRODUCE THEIR NUMBERS AND MUST NOT PRINT OURS BESIDE THEIRS. They use
Qwen-2.5-7B / Qwen3-32B / Llama-3.3-70B on TextCraft, WebShop, ALFWorld and
tau-bench. We run Gemma-3-1B on one 4090 over ATBench. What transfers is the
METHOD and its controls, not the magnitudes.

THE TWO CONTROLS THAT DECIDE WHETHER ANY OF IT MEANS ANYTHING
--------------------------------------------------------------
1. STEP-INDEX RESIDUALISATION. A probe that "predicts failure" from activations
   at step k may simply be reading k. Failing trajectories are often longer, so
   step index correlates with the label, and a position detector scores well
   while knowing nothing. Every reported AUC here must be accompanied by its
   value AFTER linear step-index information is residualised out of the
   features. Paper B reports 0.989 -> >= 0.939 under this control; a probe that
   collapses to chance under it has measured position, not drift.

2. THE CONTENT BAR. common/confound.py's bag-of-words bar, computed on the SAME
   trajectory text the model read. If a TF-IDF centroid over the text matches the
   activation probe, the internal state has added nothing. This is not optional
   scepticism: within this repo, biencoder_guard's harm detector, STA's F1 arms
   and cross_trajectory's attn_pool all cleared a nominal threshold and then
   failed a binding content bar.

CPU-only to import. Loads NO model. ASCII stdout only (Windows cp1252).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Turn", "AgentTrajectory", "TrajCorpus",
    "ActivationBundle", "ProbeResult", "CascadeResult",
    "TrajLoader", "ActivationExtractor", "TrajProbe",
]


# --- data models -------------------------------------------------------------
@dataclass(frozen=True)
class Turn:
    """One turn of one agent trajectory: what the agent did and what came back."""

    index: int                       # 0-based position WITHIN its trajectory
    role: str                        # "user" | "assistant" | "tool" | "system"
    content: str
    action: str | None = None        # tool name invoked at this turn, if any
    tool_output: str | None = None
    # tool-call dependency edges (reproduction C; empty where unavailable).
    # (i in consumes_from) means THIS turn's arguments consume turn i's output.
    consumes_from: tuple = ()


@dataclass(frozen=True)
class AgentTrajectory:
    """One agent episode. The unit a trajectory-level label attaches to."""

    uid: str
    turns: tuple
    label: int                       # 1 = unsafe / eventual failure, 0 = safe
    group_id: str                    # group-aware CV; NEVER split a group
    source: str
    # ATBench three-axis taxonomy (None on corpora that lack it)
    risk_source: str | None = None
    failure_mode: str | None = None
    real_world_harm: str | None = None
    rationale: str | None = None     # the corpus's own `reason` field
    # root-cause localisation (LongRCA; None elsewhere)
    mistake_step: int | None = None
    mistake_agent: str | None = None

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def text(self) -> str:
        """The trajectory as the model reads it -- and as the content bar reads it.

        The SAME string must feed both, or the confound comparison is not a
        comparison.
        """
        return "\n".join("%s: %s" % (t.role, t.content) for t in self.turns)


@dataclass
class TrajCorpus:
    """A loaded corpus plus the provenance that makes its numbers auditable."""

    name: str
    trajectories: list
    requested_n_per_class: int
    pool_fingerprint: str
    licence: str                     # "unstated" is a legitimate, important value
    label_provenance: str            # what the TRAJECTORY label came from
    step_label_provenance: str = ""  # what any PER-STEP label came from

    @property
    def achieved_n_pos(self) -> int:
        return sum(1 for t in self.trajectories if t.label == 1)

    @property
    def achieved_n_neg(self) -> int:
        return sum(1 for t in self.trajectories if t.label == 0)

    def turn_count_summary(self) -> dict:
        n = [t.n_turns for t in self.trajectories]
        if not n:
            return {"n": 0}
        s = sorted(n)
        return {"n": len(s), "min": s[0], "median": float(np.median(s)),
                "mean": float(np.mean(s)), "max": s[-1]}


@dataclass
class ActivationBundle:
    """Residual-stream activations for one corpus at one layer.

    X is [n_items, hidden]. step_index is the per-item turn index and is
    REQUIRED: residualisation cannot be done without it, and a bundle that
    cannot be residualised cannot support a claim in this series.
    """

    X: np.ndarray                    # [n_items, hidden]
    y: np.ndarray                    # [n_items] trajectory label, broadcast
    step_index: np.ndarray           # [n_items] turn index within its trajectory
    traj_uid: np.ndarray             # [n_items] owning trajectory uid
    group_id: np.ndarray             # [n_items] for group-aware CV
    layer: int
    model_id: str
    behaviour_fingerprint: str       # model + library + MEASURED behaviour

    def __post_init__(self) -> None:
        n = len(self.X)
        for name in ("y", "step_index", "traj_uid", "group_id"):
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(
                    "ActivationBundle length mismatch: X has %d rows but %s has "
                    "%d. A misaligned bundle scores one item's activations "
                    "against another item's label." % (n, name, got))


@dataclass
class ProbeResult:
    """One probe cell, with the controls that make it readable."""

    method: str
    layer: int
    auc: float
    auc_ci: tuple
    # CONTROL 1: the same probe after linear step-index residualisation
    auc_residualised: float | None = None
    # CONTROL 2: the bag-of-words bar on the SAME text
    content_bar_auc: float | None = None
    clears_content_bar: bool | None = None
    # CONTROL 3: Hewitt-Liang random-label ceiling
    random_label_auc: float | None = None
    n_items: int = 0
    n_trajectories: int = 0
    notes: str = ""


@dataclass
class CascadeResult:
    """Reproduction A: per-round abort gates under a global recall target."""

    target_recall: float
    achieved_recall: float
    rounds: tuple
    per_round_threshold: tuple
    frac_aborted: float
    token_saving: float              # fraction of generated tokens not spent
    n_episodes: int
    calibration_note: str = ""


# --- protocols ---------------------------------------------------------------
@runtime_checkable
class TrajLoader(Protocol):
    """Load a corpus. MUST record licence and label provenance, both honestly."""

    name: str

    def load(self, n_per_class: int = 500, seed: int = 0): ...


@runtime_checkable
class ActivationExtractor(Protocol):
    """Replay trajectories through the target model, cache residual states.

    The bundle's behaviour_fingerprint must cover the model AND the measured
    behaviour of the stack that produced it -- a version string alone is not
    enough (audits/AUDIT_2026-08-17_embeddinggemma_causal.md).
    """

    def extract(self, corpus, layer: int): ...


@runtime_checkable
class TrajProbe(Protocol):
    """Fit on activations, score held-out GROUPS, never held-out rows."""

    name: str

    def fit_predict_cv(self, bundle, n_folds: int = 5, seed: int = 0): ...


def _self_test() -> None:
    t = tuple(Turn(index=i, role="assistant", content="step %d" % i)
              for i in range(4))
    tr = AgentTrajectory(uid="u1", turns=t, label=1, group_id="g1",
                         source="smoke", risk_source="tool_description_injection")
    assert tr.n_turns == 4 and "assistant: step 3" in tr.text
    c = TrajCorpus(name="smoke", trajectories=[tr], requested_n_per_class=1,
                   pool_fingerprint="0" * 8, licence="unstated",
                   label_provenance="smoke")
    assert c.achieved_n_pos == 1 and c.achieved_n_neg == 0
    print("OK  data models:", c.turn_count_summary())

    X = np.zeros((4, 8), dtype=np.float32)
    b = ActivationBundle(X=X, y=np.ones(4), step_index=np.arange(4),
                         traj_uid=np.array(["u1"] * 4),
                         group_id=np.array(["g1"] * 4),
                         layer=12, model_id="smoke", behaviour_fingerprint="abc")
    assert b.X.shape == (4, 8)
    print("OK  ActivationBundle accepts aligned input")

    try:
        ActivationBundle(X=X, y=np.ones(3), step_index=np.arange(4),
                         traj_uid=np.array(["u1"] * 4),
                         group_id=np.array(["g1"] * 4),
                         layer=12, model_id="smoke", behaviour_fingerprint="abc")
    except ValueError as exc:
        assert "length mismatch" in str(exc)
        print("OK  a MISALIGNED bundle is refused, not silently scored")
    else:
        raise AssertionError("misaligned bundle was accepted")

    r = ProbeResult(method="linear", layer=12, auc=0.9, auc_ci=(0.85, 0.94),
                    auc_residualised=0.62, content_bar_auc=0.88,
                    clears_content_bar=False)
    assert r.auc_residualised < r.auc and r.clears_content_bar is False
    print("OK  ProbeResult carries both controls alongside the headline AUC")
    print("")
    print("OK -- traj_probes spine imports and the dataclasses behave.")


if __name__ == "__main__":
    _self_test()
