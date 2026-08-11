"""evaluate.py -- the core of the streaming-trajectory-aggregation lesson: turns a
`Corpus` + per-trajectory embeddings + an aggregator ladder into priced, group-aware,
bootstrap-CI'd results.

Three entry points, per the lead's brief:

  evaluate_offline(corpus, embeddings, aggregators, seed, n_folds)  -> list[AggregatorResult]
  evaluate_lead_time(corpus, embeddings, monitor, seed, test_frac)  -> (LeadTimeResult, diagnostics)
  confound_bars(corpus, seed, n_folds)                              -> dict

GROUP-AWARE, ALWAYS
--------------------
`Trajectory.group_id` exists so a split NEVER straddles a group (SHADE/AgentDojo group
by `main_task_description`; several rows/epochs can share one). Every CV here uses
`StratifiedGroupKFold` (falling back to plain `GroupKFold` if the stratified variant is
unavailable in the installed sklearn), never a group-blind KFold.

WHY `evaluate_lead_time` RETURNS A TUPLE, NOT A BARE `LeadTimeResult`
------------------------------------------------------------------------
`types.py` is the frozen spine and `LeadTimeResult` has no `unreliable_calibration`
field -- and this module must not edit that file. The lead's brief names a REAL defect
the streaming agent found: `SafetyDriftMonitor`'s score is a lookup over only
`n_states` (default 8) discrete table entries, so a 95th-percentile threshold fit from
~30 benign trajectories can TIE the observed max by construction (few distinct values,
many ties at the top), silently making alarms rare. That must be reported LOUDLY, not
folded into `alarm_step=None` and left for a reader to notice. So `evaluate_lead_time`
returns `(LeadTimeResult, diagnostics)` -- the dataclass stays exactly spine-shaped, and
`diagnostics["unreliable_calibration"]` (plus the threshold/peak numbers that justify
it) travels alongside it. `LeadTimeResult.oracle_note` ALSO gets the warning appended
in-line, so a caller that only keeps the dataclass (e.g. after a JSON round-trip that
drops the sibling dict) still sees it.

CPU-only to import. Loads NO model (aggregators are already-embedded arrays; a monitor
is a plain object). ASCII stdout/stderr only (Windows cp1252 console).
"""
from __future__ import annotations

import sys
import time
from typing import Sequence

import numpy as np

from steering_tutorials.streaming_trajectory_aggregation import config as C
from steering_tutorials.streaming_trajectory_aggregation.types import (
    Aggregator, AggregatorResult, Corpus, LeadTimeResult, StreamingAggregator,
)

__all__ = [
    "bootstrap_auc_ci", "group_aware_folds", "group_aware_split",
    "evaluate_offline", "evaluate_lead_time", "confound_bars",
]


def _eprint(*args) -> None:
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


# --- metric primitive --------------------------------------------------------------
def bootstrap_auc_ci(y_true, y_score, n: int = 10000, seed: int = 0):
    """Percentile bootstrap 95% CI on ROC-AUC. -> (auc, lo, hi); NaNs if AUC undefined
    (a resample or the full pool has only one class present)."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or len(set(y_true.tolist())) < 2:
        return (float("nan"), float("nan"), float("nan"))
    point = float(roc_auc_score(y_true, y_score))

    rng = np.random.default_rng(seed)
    m = len(y_true)
    boots = []
    for _ in range(int(n)):
        idx = rng.integers(0, m, m)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(roc_auc_score(yt, y_score[idx]))
    if not boots:
        return (point, point, point)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return (point, lo, hi)


# --- group-aware splitting -----------------------------------------------------------
def _make_splitter(n_splits: int, seed: int):
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), True
    except Exception:
        from sklearn.model_selection import GroupKFold

        return GroupKFold(n_splits=n_splits), False


def group_aware_folds(labels: Sequence[int], groups: Sequence, n_folds: int, seed: int):
    """K group-aware folds. `Trajectory.group_id` NEVER straddles train/test.

    `n_splits` is clamped to <= (distinct groups) and <= (minority-class count) so the
    splitter has a chance of producing a valid partition on a small corpus; floored at 2.
    """
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    n_groups = len(set(groups))
    bincount = np.bincount(labels) if n else np.array([0])
    minority = int(bincount[bincount > 0].min()) if bincount.sum() else 0
    k = max(2, min(int(n_folds), n_groups, max(minority, 2)))
    splitter, stratified = _make_splitter(k, seed)
    X = np.zeros((n, 1))
    folds = [(tr, te) for tr, te in splitter.split(X, labels, groups)]
    if not stratified:
        _eprint("[evaluate] StratifiedGroupKFold unavailable in this sklearn -- "
                "using plain GroupKFold (group-aware, not label-stratified)")
    return folds


def group_aware_split(labels: Sequence[int], groups: Sequence, test_frac: float, seed: int):
    """ONE group-aware train/test split (the first fold of a k-fold chosen so its test
    share is close to `test_frac`). Used for the one-class streaming-monitor arm, where
    calibration must be TRAIN-ONLY and evaluation happens on a disjoint TEST split."""
    test_frac = float(test_frac)
    n_splits = max(2, int(round(1.0 / test_frac))) if test_frac > 0 else 2
    folds = group_aware_folds(labels, groups, n_splits, seed)
    return folds[0]  # deterministic given `seed`; the splitter already shuffled


# ======================================================================================
# 1. evaluate_offline
# ======================================================================================
def evaluate_offline(corpus: Corpus, embeddings: list[np.ndarray],
                      aggregators: Sequence[Aggregator], seed: int = 0, n_folds: int = 5,
                      bootstrap: int = None) -> list[AggregatorResult]:
    """Group-aware out-of-fold AUC (+ 95% bootstrap CI) + per-step latency, for every
    aggregator in `aggregators`.

    Each `Aggregator` instance is reused fold-to-fold: every implementation in
    `aggregators/pooling.py` and `aggregators/sequence.py` OVERWRITES its fitted state
    on `.fit()` (no incremental carry-over), so this is equivalent to a fresh instance
    per fold without the extra construction. A fold whose TRAIN split lands single-class
    (rare on a group-aware split of a balanced corpus, but possible on a small one) is
    skipped for that method and counted in the result's `notes` -- never silently
    dropped from the record.
    """
    bootstrap = C.BOOTSTRAP if bootstrap is None else bootstrap
    labels = np.asarray([t.label for t in corpus.trajectories], dtype=int)
    groups = [t.group_id for t in corpus.trajectories]
    n = len(embeddings)
    if n != len(corpus.trajectories):
        raise ValueError("len(embeddings)=%d != len(corpus.trajectories)=%d" %
                          (n, len(corpus.trajectories)))
    folds = group_aware_folds(labels, groups, n_folds, seed)

    results = []
    for agg in aggregators:
        pooled_score = np.full(n, np.nan, dtype=np.float64)
        total_latency_s = 0.0
        total_steps = 0
        skipped_folds = 0
        for tr, te in folds:
            y_tr = labels[tr]
            if len(set(y_tr.tolist())) < 2:
                skipped_folds += 1
                continue
            try:
                agg.fit([embeddings[i] for i in tr], y_tr)
            except Exception as exc:
                skipped_folds += 1
                _eprint("[evaluate:%s] fold FAILED to fit (%s) -- skipped" % (agg.name, exc))
                continue
            for i in te:
                t0 = time.perf_counter()
                try:
                    s = agg.score(embeddings[i])
                except Exception as exc:
                    _eprint("[evaluate:%s] score FAILED on trajectory %d (%s)"
                            % (agg.name, i, exc))
                    continue
                total_latency_s += time.perf_counter() - t0
                pooled_score[i] = float(s)
                total_steps += max(1, int(np.asarray(embeddings[i]).shape[0]))

        mask = ~np.isnan(pooled_score)
        n_scored = int(mask.sum())
        auc, lo, hi = bootstrap_auc_ci(labels[mask], pooled_score[mask], n=bootstrap, seed=seed)
        latency_us = (total_latency_s / total_steps * 1e6) if total_steps else None
        notes = ("is_causal=%s n_scored=%d/%d folds_used=%d/%d skipped_folds=%d"
                 % (getattr(agg, "is_causal", None), n_scored, n, len(folds) - skipped_folds,
                    len(folds), skipped_folds))
        results.append(AggregatorResult(
            method=agg.name, auc=auc, auc_ci=(lo, hi), scores=pooled_score,
            latency_us_per_step=latency_us, notes=notes,
        ))
        _eprint("[evaluate:%s] auc=%.4f ci=[%.4f,%.4f] latency=%s us/step (%s)"
                % (agg.name, auc, lo, hi,
                   ("%.3f" % latency_us) if latency_us is not None else "n/a", notes))
    return results


# ======================================================================================
# 2. evaluate_lead_time
# ======================================================================================
def evaluate_lead_time(corpus: Corpus, embeddings: list[np.ndarray],
                        monitor: StreamingAggregator, seed: int = 0,
                        test_frac: float = None):
    """Fit `monitor` ONE-CLASS on a TRAIN-ONLY split's benign trajectories (the monitor's
    own `.fit()` filters `label==0`), then measure lead time + false-alarm rate on the
    DISJOINT TEST split. -> `(LeadTimeResult, diagnostics)` -- see module docstring for
    why the degenerate-calibration flag travels in the sibling `diagnostics` dict rather
    than being invented on the frozen `LeadTimeResult` dataclass.
    """
    test_frac = C.LEAD_TIME_TEST_FRAC if test_frac is None else test_frac
    labels = np.asarray([t.label for t in corpus.trajectories], dtype=int)
    groups = [t.group_id for t in corpus.trajectories]
    tr_idx, te_idx = group_aware_split(labels, groups, test_frac, seed)

    train_trajs = [embeddings[i] for i in tr_idx]
    train_labels = labels[tr_idx]
    monitor.fit(train_trajs, train_labels)  # one-class: filters label==0 internally

    # --- degenerate-threshold detection (train-only, independent of the test split) ---
    threshold = getattr(monitor, "_threshold", None)
    train_benign_idx = [int(i) for i, y in zip(tr_idx, train_labels) if int(y) == 0]
    peaks = []
    for i in train_benign_idx:
        monitor.reset()
        peak = None
        for step_vec in np.asarray(embeddings[i]):
            monitor.update(step_vec)
        peak = getattr(monitor, "_peak", None)
        if peak is not None:
            peaks.append(float(peak))
    monitor.reset()
    peak_max = max(peaks) if peaks else None
    n_distinct = len({round(p, 9) for p in peaks}) if peaks else 0
    tied_at_max = bool(
        threshold is not None and peak_max is not None
        and abs(float(threshold) - float(peak_max)) <= max(1e-9, 1e-6 * abs(peak_max))
    )
    low_resolution = bool(peaks and n_distinct <= 3)
    unreliable = tied_at_max or low_resolution
    diagnostics = {
        "monitor": getattr(monitor, "name", monitor.__class__.__name__),
        "threshold": threshold,
        "train_benign_n": len(peaks),
        "train_benign_peak_max": peak_max,
        "n_distinct_peak_values": n_distinct,
        "tied_at_max": tied_at_max,
        "low_resolution_calibration": low_resolution,
        "unreliable_calibration": unreliable,
    }
    if unreliable:
        _eprint("=" * 70)
        _eprint("[evaluate:%s] WARNING: UNRELIABLE CALIBRATION -- alarm threshold "
                "(%.6g) %s the observed train-benign peak max (%.6g over %d "
                "trajectories, %d distinct value(s)). Alarms may be rare BY "
                "CONSTRUCTION, not because the monitor genuinely separates classes. "
                "Reporting alarm_step=None honestly -- NOT treating a quiet monitor "
                "as a finding." % (
                    getattr(monitor, "name", monitor.__class__.__name__),
                    float(threshold) if threshold is not None else float("nan"),
                    "TIES" if tied_at_max else "is within",
                    float(peak_max) if peak_max is not None else float("nan"),
                    len(peaks), n_distinct))
        _eprint("=" * 70)

    # --- streaming lead-time + detection on the TEST split's positives -----------------
    lead_steps = []
    n_eligible = 0
    n_detected = 0
    for i in te_idx:
        traj = corpus.trajectories[i]
        if traj.label != 1 or traj.available_lead is None:
            continue
        n_eligible += 1
        ponr = traj.point_of_no_return_step
        monitor.reset()
        for step_vec in np.asarray(embeddings[i]):
            monitor.update(step_vec)
        alarm_step = getattr(monitor, "alarm_step", None)
        if alarm_step is not None and ponr is not None and int(alarm_step) <= int(ponr):
            n_detected += 1
            lead_steps.append(int(ponr) - int(alarm_step))

    # --- false-alarm rate on the TEST split's benign trajectories ---------------------
    test_neg_idx = [int(i) for i in te_idx if labels[i] == 0]
    n_false = 0
    for i in test_neg_idx:
        monitor.reset()
        for step_vec in np.asarray(embeddings[i]):
            monitor.update(step_vec)
        if getattr(monitor, "alarm_step", None) is not None:
            n_false += 1
    false_alarm_rate = (n_false / len(test_neg_idx)) if test_neg_idx else float("nan")

    lead_arr = np.asarray(lead_steps, dtype=np.float64)
    p10 = float(np.percentile(lead_arr, 10)) if lead_arr.size else 0.0
    med = float(np.percentile(lead_arr, 50)) if lead_arr.size else 0.0
    p90 = float(np.percentile(lead_arr, 90)) if lead_arr.size else 0.0

    oracle_note = "lead measured against a judge oracle, NOT ground truth"
    if unreliable:
        oracle_note += ("; UNRELIABLE_CALIBRATION: threshold %s the train-benign peak "
                         "max (n=%d, %d distinct value(s)) -- alarms may be rare by "
                         "construction; do not read a low n_detected here as evidence "
                         "of poor separation" % ("ties" if tied_at_max else "is close to",
                                                  len(peaks), n_distinct))

    result = LeadTimeResult(
        method=getattr(monitor, "name", monitor.__class__.__name__),
        n_positives_evaluated=n_eligible,
        n_detected=n_detected,
        lead_steps=lead_arr,
        lead_p10=p10,
        lead_median=med,
        lead_p90=p90,
        false_alarm_rate=false_alarm_rate,
        oracle_note=oracle_note,
    )
    _eprint("[evaluate_lead_time:%s] n_eligible=%d n_detected=%d lead_p10/med/p90=%.1f/%.1f/%.1f "
            "false_alarm_rate=%.4f unreliable_calibration=%s"
            % (result.method, n_eligible, n_detected, p10, med, p90, false_alarm_rate,
               unreliable))
    return result, diagnostics


# ======================================================================================
# 3. confound_bars
# ======================================================================================
def confound_bars(corpus: Corpus, seed: int = 0, n_folds: int = 5) -> dict:
    """The shared confound spine (`common/confound.py`) on FULL trajectory text, PLUS a
    FIRST-STEP-ONLY bag-of-words rival (mirroring `trajguard`'s prompt-channel: "can a
    dumb model already tell from just the opening move?" is a rival METHOD, not a
    confound to subtract).

    Returns a dict with `full_trajectory` (the 4-bar `confound_report`), `first_step_rival`
    (a `content_bar` on step-0 text only), and `binding_bar`/`binding_bar_name` = the
    larger of the two (also exposed as `worst_auc`/`worst_name` so this dict is a
    drop-in `report` argument to `common.confound.margin_over_bar`).
    """
    from steering_tutorials.common.confound import confound_report, content_bar

    labels = [t.label for t in corpus.trajectories]
    texts = [" ".join(step.text for step in t.steps) for t in corpus.trajectories]
    units = [t.steps for t in corpus.trajectories]
    full = confound_report(texts, labels, units=units, seed=seed, n_folds=n_folds)

    first_step_texts = [(t.steps[0].text if t.steps else "") for t in corpus.trajectories]
    first_step_content = content_bar(first_step_texts, labels, seed=seed, n_folds=n_folds)

    full_bar = float(full.get("worst_auc", 0.5))
    rival_auc = float(first_step_content.get("auc", 0.5))
    if rival_auc > full_bar:
        binding_bar, binding_name = rival_auc, "first_step_rival_content"
    else:
        binding_bar, binding_name = full_bar, full.get("worst_name", "none")

    return {
        "full_trajectory": full,
        "first_step_rival": {"content": first_step_content},
        "binding_bar": binding_bar,
        "binding_bar_name": binding_name,
        # aliases so this dict is a drop-in `report` for common.confound.margin_over_bar
        "worst_auc": binding_bar,
        "worst_name": binding_name,
        "note": ("first_step_rival is a RIVAL METHOD (can a bag-of-words model on just "
                 "the FIRST step already separate the classes?), not a bar to fold in "
                 "quietly -- mirroring trajguard's prompt-channel treatment. Every "
                 "headline aggregator AUC must be priced against binding_bar = "
                 "max(full_trajectory.worst_auc, first_step_rival.content.auc)."),
    }


# --- CPU self-test: python -m ...streaming_trajectory_aggregation.evaluate -----------
if __name__ == "__main__":
    from steering_tutorials.streaming_trajectory_aggregation.aggregators.pooling import (
        LastStep, MeanPool,
    )
    from steering_tutorials.streaming_trajectory_aggregation.streaming.esn_cusum import (
        ESNCusum,
    )
    from steering_tutorials.streaming_trajectory_aggregation.types import (
        Corpus, Step, Trajectory,
    )

    rng = np.random.default_rng(0)
    dim = 8

    def _make_traj(uid, n_steps, drift, label, gid, ponr=None, ideal=None):
        base = rng.normal(0.0, 1.0, size=(n_steps, dim)).astype(np.float32)
        ramp = np.linspace(0.0, drift, n_steps)[:, None].astype(np.float32)
        base = base + ramp
        steps = tuple(Step(index=i, text="step %d of %s" % (i, uid)) for i in range(n_steps))
        return Trajectory(uid=uid, steps=steps, label=label, group_id=gid, source="smoke",
                           ideal_flagging_step=ideal, point_of_no_return_step=ponr), base

    trajs, embs = [], []
    for i in range(40):
        n = int(rng.integers(10, 20))
        if i % 2 == 0:
            t, e = _make_traj("pos-%d" % i, n, drift=5.0, label=1, gid="g%d" % i,
                               ponr=n - 2, ideal=max(0, n - 6))
        else:
            t, e = _make_traj("neg-%d" % i, n, drift=0.0, label=0, gid="g%d" % i)
        trajs.append(t)
        embs.append(e)

    corpus = Corpus(name="smoke", trajectories=trajs, requested_n_per_class=20,
                     pool_fingerprint="0" * 64, licence="unstated", label_provenance="smoke")

    results = evaluate_offline(corpus, embs, [MeanPool(), LastStep()], seed=0, n_folds=4,
                                bootstrap=200)
    for r in results:
        print("offline %-12s auc=%.3f ci=[%.3f,%.3f] latency=%s notes=%s"
              % (r.method, r.auc, r.auc_ci[0], r.auc_ci[1],
                 r.latency_us_per_step, r.notes))
    assert all(np.isfinite(r.auc) for r in results), "offline AUC came back non-finite"

    mon = ESNCusum(reservoir_dim=16, seed=0)
    lead_result, diag = evaluate_lead_time(corpus, embs, mon, seed=0, test_frac=0.3)
    print("lead_time n_eligible=%d n_detected=%d p10/med/p90=%.1f/%.1f/%.1f false_alarm=%.3f"
          % (lead_result.n_positives_evaluated, lead_result.n_detected, lead_result.lead_p10,
             lead_result.lead_median, lead_result.lead_p90, lead_result.false_alarm_rate))
    print("diagnostics:", diag)
    assert isinstance(diag, dict) and "unreliable_calibration" in diag

    conf = confound_bars(corpus, seed=0, n_folds=3)
    print("confound binding_bar=%.4f (%s)" % (conf["binding_bar"], conf["binding_bar_name"]))
    assert "full_trajectory" in conf and "first_step_rival" in conf

    print("OK -- evaluate.py self-test: offline CV, lead-time + degenerate-threshold "
          "diagnostics, and confound_bars all run end to end on synthetic data.")
