"""leakage.py -- the corpus-level step-index leak that CONTROL 1 cannot remove.

WHY THIS MODULE EXISTS
----------------------
`probes.StepResidualiser` removes LINEAR step-index information. Its own
self-test is honest that this is all it does: a planted QUADRATIC position code
survives degree-1 residualisation at AUC 0.6232. So a position signal that is
not linear passes CONTROL 1 untouched.

ATBench contains exactly such a signal, and it is not a trend -- it is a
DETERMINISTIC THRESHOLD. Measured on all 997 loaded trajectories, 2026-08-29:

    longest SAFE   trajectory : 18 turns
    longest UNSAFE trajectory : 62 turns
    -> any turn whose 0-based index is >= 18 belongs to a trajectory longer
       than EVERY safe trajectory in the corpus, so its label is decided before
       a single activation is read.

    32 trajectories (all unsafe) run past 18 turns
    570 of 8,981 turn-rows (6.3%) sit in that region, 100% label=1

WHY THE OBVIOUS CONTROL DOES NOT CATCH IT
-----------------------------------------
Step index over the whole corpus reaches AUC 0.5686, and fractional position
reaches 0.5000. Both look harmless. The leak is invisible globally because it is
LOCAL: it is confined to the top 6.3% of rows, where it is perfect.

    turns at step >=  8 : n=2051  P(unsafe)=0.633
    turns at step >= 16 : n= 640  P(unsafe)=0.997
    turns at step >= 24 : n= 408  P(unsafe)=1.000

This is the same shape as the note already in CLAUDE.md 18.2 about the conformal
gate -- feasibility is a LOCAL ROC property while we report global AUC
everywhere. A global summary is the wrong instrument for a local defect.

It bites reproduction A hardest. An early-abort cascade is *defined* by operating
at high precision on a subset, which is precisely where this region lives: a
cascade could report an excellent precision-at-recall while having learned
nothing but "long trajectories are unsafe".

THE FIX IS AT THE DATA LEVEL, BECAUSE THE CONTROL CANNOT REACH IT
----------------------------------------------------------------
Capping turns at MAX_TURNS <= 18 makes the region empty by construction. That is
why `config.MAX_TURNS` now defaults to 16 rather than 0 (unlimited): at 16 the
highest step index is 15, below the 18 boundary, and 8,341 of 8,981 turn-rows
(92.9%) survive.

The cap is also FAITHFUL to reproduction A rather than a concession to it: the
paper's claim is that failure is predictable EARLY, so evaluating on early turns
is the honest test of it, not a weakened one.

CPU-only. Imports no model. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

__all__ = ["deterministic_step_region", "assert_no_deterministic_region"]


def deterministic_step_region(corpus) -> dict:
    """Measure the step index beyond which one label becomes certain.

    Returns a dict with the boundary, the rows past it, and which label they
    all carry. `is_empty` False means a probe can score those rows correctly
    from position alone.
    """
    trs = list(corpus.trajectories)
    out = {"n_trajectories": len(trs), "n_rows": sum(t.n_turns for t in trs),
           "boundary": None, "certain_label": None, "n_rows_beyond": 0,
           "n_traj_beyond": 0, "frac_rows_beyond": 0.0, "is_empty": True,
           "region_is_pure": None, "max_turns_by_label": {}}
    labels = {t.label for t in trs}
    if len(labels) < 2:
        out["note"] = "only one class present; no cross-label boundary exists"
        return out

    max_by = {lab: max(t.n_turns for t in trs if t.label == lab) for lab in labels}
    out["max_turns_by_label"] = {str(k): int(v) for k, v in max_by.items()}

    # The boundary is the SHORTER class's maximum: past it, only the longer
    # class can supply a row.
    boundary = min(max_by.values())
    certain = [lab for lab, m in max_by.items() if m > boundary]
    out["boundary"] = int(boundary)
    if not certain:
        return out                                   # classes tie; no region

    beyond = [t for t in trs if t.n_turns > boundary]
    rows_beyond = sum(t.n_turns - boundary for t in beyond)
    out.update(certain_label=int(certain[0]) if len(certain) == 1 else None,
               n_traj_beyond=len(beyond), n_rows_beyond=int(rows_beyond),
               frac_rows_beyond=(rows_beyond / out["n_rows"]) if out["n_rows"] else 0.0,
               is_empty=rows_beyond == 0)
    # Purity is the point: if the region is not 100% one label it is a skew,
    # not a determinism, and it should not be reported as one.
    out["region_is_pure"] = len({t.label for t in beyond}) == 1
    return out


def assert_no_deterministic_region(corpus, acknowledge: bool = False) -> dict:
    """Raise unless the deterministic region is empty (or explicitly accepted).

    This is a BUILD-TIME gate on purpose. The residualiser removes linear
    position only, so nothing downstream will notice this region -- a check that
    must be remembered will eventually be skipped.
    """
    r = deterministic_step_region(corpus)
    if r["is_empty"] or acknowledge:
        return r
    raise SystemExit(
        "DETERMINISTIC STEP-INDEX REGION in corpus %r.\n"
        "  longest trajectory by label : %s\n"
        "  every turn at step index >= %d is label %s, decided BEFORE any\n"
        "  activation is read: %d rows (%.1f%%) across %d trajectories.\n"
        "  CONTROL 1 CANNOT REMOVE THIS. StepResidualiser removes LINEAR step\n"
        "  information; this is a threshold, and its own self-test shows a\n"
        "  non-linear position code surviving at AUC 0.62.\n"
        "  Fix it in the DATA: set TP_MAX_TURNS <= %d (config default is 16).\n"
        "  Pass acknowledge=True only to study the region deliberately."
        % (corpus.name, r["max_turns_by_label"], r["boundary"],
           r["certain_label"], r["n_rows_beyond"], 100 * r["frac_rows_beyond"],
           r["n_traj_beyond"], r["boundary"]))


def _self_test() -> None:
    from steering_tutorials.traj_probes.types import (AgentTrajectory, Turn,
                                                      TrajCorpus)

    def mk(uid, n, lab):
        return AgentTrajectory(
            uid=uid, turns=tuple(Turn(index=i, role="assistant", content="x")
                                 for i in range(n)),
            label=lab, group_id=uid, source="smoke")

    def corp(trs, name="smoke"):
        return TrajCorpus(name=name, trajectories=trs, requested_n_per_class=2,
                          pool_fingerprint="0" * 8, licence="smoke",
                          label_provenance="smoke")

    # a corpus where the positive class owns the long tail -- the ATBench shape
    leaky = corp([mk("s%d" % i, 4, 0) for i in range(5)] +
                 [mk("u%d" % i, 4, 1) for i in range(4)] + [mk("long", 10, 1)])
    r = deterministic_step_region(leaky)
    assert not r["is_empty"] and r["boundary"] == 4 and r["certain_label"] == 1
    assert r["n_rows_beyond"] == 6 and r["region_is_pure"]
    print("OK  a long-tailed positive class is detected: step >= %d is always "
          "label %d (%d rows)" % (r["boundary"], r["certain_label"],
                                  r["n_rows_beyond"]))
    try:
        assert_no_deterministic_region(leaky)
    except SystemExit as exc:
        assert "CANNOT REMOVE THIS" in str(exc)
        print("OK  the gate REFUSES it, and says why the residualiser will not save it")
    else:
        raise AssertionError("leaky corpus was accepted")

    assert assert_no_deterministic_region(leaky, acknowledge=True)["boundary"] == 4
    print("OK  acknowledge=True returns the measurement instead of raising")

    balanced = corp([mk("s%d" % i, 6, 0) for i in range(5)] +
                    [mk("u%d" % i, 6, 1) for i in range(5)])
    assert deterministic_step_region(balanced)["is_empty"]
    assert_no_deterministic_region(balanced)
    print("OK  equal-length classes leave NO region, and pass the gate")

    # truncation is the fix: cap the leaky corpus at its boundary
    capped = corp([mk(t.uid, min(t.n_turns, 4), t.label)
                   for t in leaky.trajectories])
    assert deterministic_step_region(capped)["is_empty"]
    print("OK  capping turns at the boundary EMPTIES the region (the MAX_TURNS fix)")

    one = corp([mk("a", 3, 1), mk("b", 9, 1)])
    assert deterministic_step_region(one)["is_empty"]
    print("OK  a single-class corpus reports no region rather than dividing by zero")
    print("")
    print("OK -- leakage.py: the region is measured, the gate refuses it, and "
          "capping turns is verified to be the fix.")


if __name__ == "__main__":
    _self_test()
