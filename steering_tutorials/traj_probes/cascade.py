"""cascade.py -- the recall-controlled probe CASCADE, reproduction A's actual
claim (Ruan, Huang, Zhou, Wei, Lin, Wang and Sun, 7 Jul 2026, "Doomed from the
Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade",
arXiv:2607.06503).

WHY THIS FILE EXISTS
--------------------
`types.CascadeResult` has sat in the fixed spine, unused, since the spine was
written. `run_traj_probes.early_abort_curve` measures AUC-vs-k -- "how early is
failure predictable" -- which is a real and useful number, but it is NOT the
paper's mechanism. The paper aborts episodes: at each round it scores every
still-running episode and kills the ones that look doomed, subject to a GLOBAL
recall floor on the unsafe class, and the payoff is TOKENS NEVER SPENT because
the episode was never continued. An AUC curve says nothing about how many
tokens that would have saved. This module builds the actual cascade and reports
that number.

THE DESIGN (read before trusting a number)
-------------------------------------------
A cascade needs, per round k (turns visible so far), an out-of-fold score for
every still-running episode, and a threshold theta_k such that aborting anyone
who crosses it preserves the target recall. The naive version of "target
recall at every round" is a trap: if every round independently demands "catch
>= r of whatever unsafe episodes are still active", the CUMULATIVE recall
compounds toward 1 - (1-r)^k within a handful of rounds regardless of whether
the probe carries any signal at all -- rounds 2-3 alone would already clear
0.99 recall for r=0.90, which is not a controlled target, it is a runaway.

So this module uses a single calibrated cutoff instead: for a target recall r,
theta is the (1 - r) quantile of CALIBRATION-set unsafe trajectories' PEAK
score (the max of their own out-of-fold scores across every round they
reached). An episode is aborted the first round any of its own round scores
exceeds theta; by construction, approximately r-fraction of a same-distributed
unsafe population will have a peak crossing theta, so the ACHIEVED recall
tracks the TARGET directly, without the per-round compounding trap. The bar is
the same value at every round (see `CascadeResult.per_round_threshold`, which
therefore reports one repeated number, not a genuinely-varying schedule) --
this is a real simplification versus whatever stage-by-stage optimisation the
paper's own recipe uses (this repo has not reproduced that optimisation), and
it is documented as a simplification rather than passed off as the paper's
method. What it keeps faithfully is the actual claim under test: a global
recall floor, held out-of-fold, converted into early-abort token savings.

WHERE THE EARLINESS COMES FROM, AND WHY token_saving ALONE CAN LIE
--------------------------------------------------------------------
Because every round is checked in sequence and the SAME theta is used
throughout, an episode's own first crossing tends to happen earlier when its
scores are tightly, reliably separated from the other class (the separable
self-test fixture: mean relative catch position 0.24 of an episode's own
length) and later when the score is round-to-round noise with real within-
class spread (the pure-noise fixture: 0.40) -- measured directly in
`_self_test`, not asserted from theory.

But a SECOND effect matters just as much and is easy to miss: at a HIGH
target recall, a no-signal probe can only reach that recall by aborting
almost every episode, safe or unsafe, because its scores do not distinguish
the classes at all (measured: 38/40 safe trajectories aborted at 0.90 target
recall on the pure-noise fixture, vs 0/40 on the separable one). That
indiscriminate aborting still counts as "tokens saved" in the raw
`token_saving` number, so a genuinely useless cascade can show a HIGHER
token_saving than a genuinely good one at high recall targets simply by being
maximally aggressive. `_aggregate_cascade` therefore folds
`frac_safe_aborted` and `mean_relative_catch_unsafe` into `calibration_note` --
read `token_saving` next to those two, never alone.

THE TWO CASCADES BUILT HERE, AND THE ONE NOT BUILT
----------------------------------------------------
  activation cascade     LinearTrajProbe-style OOF scores (probes._fit_fold,
                          reusing run_traj_probes._trajectory_fold_assignment /
                          _cv_indices_from_fold_map so the fold partition is
                          IDENTICAL to the rest of this lesson's reporting).
  step-index baseline     the SAME peak-score machinery, but the "score" at
                          round k is just k itself -- i.e. "abort long
                          episodes." Cheap, and the honest baseline this
                          corpus's own length confound (config.py, README
                          Section 3) demands: if the activation cascade cannot
                          beat "abort once an episode runs long", it has not
                          demonstrated anything the turn counter did not
                          already know.
  content-bar cascade     NOT built. common.confound.content_bar fits its own
                          internal group-free K-fold CV and returns one
                          aggregate AUC -- not a per-trajectory out-of-fold
                          score usable to decide WHEN to abort. Plugging it
                          into this cascade needs a parallel per-round,
                          nested-fold TF-IDF fit/score pipeline (vocabulary +
                          IDF fit on calibration trajectories' first-k-turn
                          text only, scored on the held-out fold, repeated for
                          every k and every outer fold) -- a second bespoke
                          pipeline, not a cheap reuse of the existing one.
                          Reported as SKIPPED, with this reason, in every
                          output payload (`content_bar_cascade_note`).

Out-of-fold only. Group-aware CV throughout (never split a trajectory).
Thresholds calibrated on the 4 CALIBRATION folds, applied to the 1 held-out
TEST fold -- never the reverse; calibrating on the fold being scored is the
exact failure mode this design exists to avoid.

CPU-only. Loads NO model -- reads the cached activation bundle already on
disk. ASCII stdout only (Windows cp1252).

Run it (real corpus, cached bundle, no GPU):
    C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.cascade

CPU self-test, synthetic bundle, no model, no network, no GPU:
    TP_SELFTEST=1 C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.cascade
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:  # package form
    from . import config as C
    from .probes import group_folds  # noqa: F401 -- re-exported for callers/tests
    from .run_traj_probes import (
        _cv_indices_from_fold_map, _fit_score, _trajectory_fold_assignment,
    )
    from .types import ActivationBundle, CascadeResult
except ImportError:  # pragma: no cover - direct-script form
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent.parent))
    from steering_tutorials.traj_probes import config as C
    from steering_tutorials.traj_probes.probes import group_folds  # noqa: F401
    from steering_tutorials.traj_probes.run_traj_probes import (
        _cv_indices_from_fold_map, _fit_score, _trajectory_fold_assignment)
    from steering_tutorials.traj_probes.types import ActivationBundle, CascadeResult

try:
    from ..common.artifact_paths import keyed_path
except (ImportError, ValueError):  # pragma: no cover
    from steering_tutorials.common.artifact_paths import keyed_path

__all__ = [
    "find_bundle", "per_round_oof_scores", "run_cascade",
    "content_bar_cascade_note", "run", "_self_test",
]

# Measured on artifacts/acts_gemma-3-1b-it_L12_last_*.npz, 2026-08-30: 997
# trajectories, TP_MAX_TURNS=16 cap, X.shape == (8341, 1152). This is a fact
# about the CURRENT atbench/MAX_TURNS=16 config, not a universal constant --
# override via TP_CASCADE_EXPECTED_N_ROWS if the corpus config changes.
EXPECTED_N_ROWS = int(os.environ.get("TP_CASCADE_EXPECTED_N_ROWS") or 8341)

DEFAULT_TARGET_RECALLS = (0.99, 0.95, 0.90, 0.80, 0.70)


def _print(*args) -> None:
    """ASCII-only stdout. This host's cp1252 console crashes on unicode."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg)
    except Exception:  # pragma: no cover
        print(msg.encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# 1. Find the ONE bundle that matches this corpus, by CONTENT not by luck
# ---------------------------------------------------------------------------
def find_bundle(artifacts_dir, model_tag: str, layer: int,
                expected_n_rows: int = EXPECTED_N_ROWS, pooling: str = "last"):
    """-> (ActivationBundle, path, meta). Selects by X.shape[0], not by mtime.

    Globs `acts_<model_tag>_L<layer>_<pooling>_*.npz` (the extractor's own
    naming, see activations.bundle_cache_path) and REFUSES to guess if more
    than one file on disk matches both the name pattern and the expected row
    count -- a stale bundle has silently inflated a result in this lesson once
    already (config.py's CACHE_REFRESH note), and picking "the newest one"
    would repeat exactly that mistake with a different justification.

    `pooling` defaults to "last" -- `activations.ExtractSettings()`'s own
    default and the pooling every number in README Section 7 is reported
    against -- because this artifacts directory holds MORE than one pooling's
    cache (`sweep_pooling.py` builds "mean_turn" and "mean_prefix" bundles
    alongside it) and two of them coincide in row count at MAX_TURNS=16,
    which would otherwise be a silent, wrong pick rather than a raised one.
    """
    pattern = str(Path(artifacts_dir)
                 / ("acts_%s_L%d_%s_*.npz" % (model_tag, int(layer), pooling)))
    candidates = sorted(glob.glob(pattern))
    matches = []
    seen = []
    for p in candidates:
        try:
            z = np.load(p, allow_pickle=False)
            n_rows = int(z["X"].shape[0])
        except Exception as exc:  # pragma: no cover - corrupt/unreadable file
            seen.append("%s (unreadable: %s)" % (Path(p).name, exc))
            continue
        seen.append("%s (%d rows)" % (Path(p).name, n_rows))
        if n_rows == int(expected_n_rows):
            matches.append((p, z))
    if not matches:
        raise FileNotFoundError(
            "no bundle matched %r with X.shape[0]==%d. Candidates seen: %s"
            % (pattern, expected_n_rows, "; ".join(seen) if seen else "none found"))
    if len(matches) > 1:
        raise RuntimeError(
            "AMBIGUOUS BUNDLE: %d files match %r AND have X.shape[0]==%d (%s). "
            "Refusing to pick one -- a stale bundle silently inflated a result "
            "in this lesson once already. Delete or rename the stale file."
            % (len(matches), pattern, expected_n_rows,
               ", ".join(Path(p).name for p, _z in matches)))
    path, z = matches[0]
    meta = json.loads(str(z["meta"]))
    bundle = ActivationBundle(
        X=z["X"], y=z["y"], step_index=z["step_index"],
        traj_uid=z["traj_uid"], group_id=z["group_id"],
        layer=int(meta["layer"]), model_id=str(meta["model_id"]),
        behaviour_fingerprint=str(meta.get("behaviour_key", "")))
    return bundle, Path(path), meta


# ---------------------------------------------------------------------------
# 2. Per-trajectory bookkeeping
# ---------------------------------------------------------------------------
def _trajectory_lengths(bundle: ActivationBundle):
    """-> (n_turns, labels), both {traj_uid(str): value}."""
    traj = np.asarray([str(u) for u in bundle.traj_uid])
    step = np.asarray(bundle.step_index)
    y = np.asarray(bundle.y).astype(np.int64)
    n_turns, labels = {}, {}
    for t in np.unique(traj):
        m = traj == t
        n_turns[str(t)] = int(step[m].max()) + 1
        labels[str(t)] = int(y[m][0])
    return n_turns, labels


# ---------------------------------------------------------------------------
# 3. Per-round out-of-fold activation-probe scores
# ---------------------------------------------------------------------------
def per_round_oof_scores(bundle: ActivationBundle, n_folds: int = 5, seed: int = 0):
    """-> (dict[k] = {"traj_uid", "score", "y", "n_folds_used"}, fold_of).

    Round k's rows are exactly the ones with `step_index == k - 1` (the
    activation AFTER reading turn k). Every round is scored with the SAME
    per-trajectory fold assignment (`_trajectory_fold_assignment`), so a
    trajectory sits in the same fold at round 1 as at round 16 -- required
    for the cascade's own calibration/evaluation split (Section 5 below) to
    mean anything across rounds. Delegates the estimator to
    `run_traj_probes._fit_score` (-> `probes._fit_fold`) rather than
    reimplementing it, so this cascade's probe is provably the SAME probe as
    the rest of the lesson's reporting.
    """
    fold_of = _trajectory_fold_assignment(bundle, n_folds, seed)
    step = np.asarray(bundle.step_index)
    y_all = np.asarray(bundle.y).astype(np.int64)
    traj = np.asarray([str(u) for u in bundle.traj_uid])
    k_max = int(step.max()) + 1

    out = {}
    for k in range(1, k_max + 1):
        mask = step == (k - 1)
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue
        Xk = np.asarray(bundle.X[mask], dtype=np.float64)
        yk = y_all[mask]
        gk = traj[mask]
        splits = _cv_indices_from_fold_map(gk, fold_of, n_folds)
        scores = np.full(n_rows, np.nan)
        used = 0
        for tr, te in splits:
            if len(np.unique(yk[tr])) < 2:
                continue
            scores[te] = _fit_score(Xk[tr], yk[tr], Xk[te], seed)
            used += 1
        out[k] = {"traj_uid": gk, "score": scores, "y": yk, "n_folds_used": used}
    return out, fold_of


def _scores_by_round_from_oof(oof_by_round: dict) -> dict:
    """-> {traj_uid: {k: score}}, dropping any (traj, k) whose fold was
    unusable (NaN) rather than treating a missing round as a zero score."""
    out = {}
    for k, d in oof_by_round.items():
        for t, s in zip(d["traj_uid"], d["score"]):
            if not np.isfinite(s):
                continue
            out.setdefault(str(t), {})[k] = float(s)
    return out


def _step_index_scores_by_round(n_turns: dict) -> dict:
    """The baseline: score(traj, k) = k itself. Peak score = the trajectory's
    OWN length, so calibrating a cutoff on this is exactly "abort once an
    episode has run longer than some threshold" -- the length confound
    config.py and README Section 3 already document for this corpus, made
    into an explicit competitor rather than an implicit one.
    """
    return {t: {k: float(k) for k in range(1, L + 1)} for t, L in n_turns.items()}


# ---------------------------------------------------------------------------
# 4. The recall-controlled cascade itself
# ---------------------------------------------------------------------------
def _peak_score_cascade(scores_by_round: dict, labels: dict, fold_of: dict,
                        target_recall: float, n_folds: int):
    """-> (abort_round: {traj: k or None}, fold_thetas: list[float]).

    Nested calibrate/apply: for each outer fold f, theta is calibrated ONLY
    from trajectories in the OTHER 4 folds (their unsafe peak scores' (1 -
    target_recall) quantile), then applied to fold f's trajectories by
    aborting at the first round any of their own scores exceeds theta. Fold
    f's own labels never enter the calibration that scores fold f -- that is
    the entire point of doing this per outer fold rather than once globally.
    """
    all_trajs = list(labels.keys())
    abort_round = {}
    fold_thetas = []
    for f in range(n_folds):
        calib = [t for t in all_trajs if fold_of.get(t) != f]
        test = [t for t in all_trajs if fold_of.get(t) == f]
        calib_unsafe_peaks = [max(scores_by_round[t].values())
                              for t in calib
                              if labels[t] == 1 and scores_by_round.get(t)]
        if calib_unsafe_peaks:
            theta = float(np.quantile(calib_unsafe_peaks, 1.0 - target_recall))
        else:  # pragma: no cover - degenerate fold, no calibration signal
            theta = float("inf")
        fold_thetas.append(theta)
        for t in test:
            rd = scores_by_round.get(t, {})
            hit = None
            for k in sorted(rd.keys()):
                if rd[k] > theta:
                    hit = k
                    break
            abort_round[t] = hit
    return abort_round, fold_thetas


def _aggregate_cascade(abort_round: dict, labels: dict, n_turns: dict,
                       target_recall: float, fold_thetas: list,
                       method_note: str) -> CascadeResult:
    n_unsafe = sum(1 for t in labels if labels[t] == 1)
    n_caught = sum(1 for t, r in abort_round.items() if labels[t] == 1 and r is not None)
    achieved_recall = (n_caught / n_unsafe) if n_unsafe else float("nan")

    n_total = len(labels)
    n_aborted = sum(1 for r in abort_round.values() if r is not None)
    frac_aborted = (n_aborted / n_total) if n_total else float("nan")

    saved = sum(max(0, n_turns[t] - abort_round[t])
               for t in labels if abort_round.get(t) is not None)
    total_rows = sum(n_turns.values())
    token_saving = (saved / total_rows) if total_rows else float("nan")

    # HONESTY DIAGNOSTICS, not part of the CascadeResult schema (types.py is
    # owned by the lead) but load-bearing for reading token_saving correctly:
    # a no-signal cascade can only reach a HIGH recall target by aborting
    # almost EVERY episode regardless of class, which inflates raw
    # token_saving despite being useless. Measured on the synthetic self-test
    # at target_recall=0.90: a genuinely separable probe left 0/40 safe
    # trajectories falsely aborted and caught unsafe ones at 24% of their own
    # length on average; a pure-noise probe hitting the SAME 0.90 recall
    # target aborted 38/40 safe trajectories too (95%) and caught unsafe ones
    # at 40% of their length on average -- LATER, and indiscriminately. So
    # token_saving alone can make a useless cascade look BETTER than a real
    # one at high recall; frac_safe_aborted and the mean relative catch round
    # are the numbers that catch that.
    n_safe = sum(1 for t in labels if labels[t] == 0)
    n_safe_aborted = sum(1 for t, r in abort_round.items()
                         if labels[t] == 0 and r is not None)
    frac_safe_aborted = (n_safe_aborted / n_safe) if n_safe else float("nan")
    rel_catch = [abort_round[t] / n_turns[t] for t in labels
                if labels[t] == 1 and abort_round.get(t) is not None]
    mean_rel_catch_unsafe = (float(np.mean(rel_catch)) if rel_catch
                             else float("nan"))

    k_max = max(n_turns.values()) if n_turns else 0
    rounds = tuple(range(1, k_max + 1))
    finite_thetas = [t for t in fold_thetas if np.isfinite(t)]
    mean_theta = float(np.mean(finite_thetas)) if finite_thetas else float("nan")
    per_round_threshold = tuple([mean_theta] * len(rounds))

    note = (
        "%s: peak-score recall-controlled cascade. theta = the (1 - "
        "target_recall) quantile of CALIBRATION-fold unsafe trajectories' "
        "PEAK score across all rounds they reached, calibrated independently "
        "per outer fold (5-fold, group=trajectory) on the OTHER 4 folds and "
        "applied only to that held-out fold; an episode aborts at the first "
        "round its own score exceeds theta, or never (a miss) if it does not. "
        "per_round_threshold reports the across-fold MEAN of one shared "
        "cutoff used at every round -- this design does not vary the bar by "
        "round (see cascade.py module docstring for why a naive per-round "
        "recall floor compounds past the target instead of tracking it). "
        "READ token_saving ALONGSIDE frac_safe_aborted=%s (fraction of SAFE "
        "trajectories also aborted -- high here means the cascade is "
        "indiscriminate, not that it is good) and mean_relative_catch_unsafe=%s "
        "(mean abort_round/n_turns among caught unsafe episodes -- low means "
        "genuinely early). fold thetas: [%s]"
        % (method_note,
           "%.4f" % frac_safe_aborted if np.isfinite(frac_safe_aborted) else "n/a",
           "%.4f" % mean_rel_catch_unsafe if np.isfinite(mean_rel_catch_unsafe) else "n/a",
           ", ".join("%.4f" % t if np.isfinite(t) else "inf" for t in fold_thetas))
    )
    return CascadeResult(
        target_recall=float(target_recall), achieved_recall=float(achieved_recall),
        rounds=rounds, per_round_threshold=per_round_threshold,
        frac_aborted=float(frac_aborted), token_saving=float(token_saving),
        n_episodes=int(n_total), calibration_note=note)


def run_cascade(bundle: ActivationBundle, target_recalls=DEFAULT_TARGET_RECALLS,
               n_folds: int = 5, seed: int = 0):
    """-> (activation_results, step_index_results, diagnostics).

    Both lists are `CascadeResult`, one per `target_recalls` entry, computed
    with the IDENTICAL fold partition and aggregation logic -- the step-index
    baseline is not a different pipeline with different bookkeeping, it is
    the same `_peak_score_cascade` fed a different score function, so the
    comparison is apples-to-apples by construction.
    """
    n_turns, labels = _trajectory_lengths(bundle)
    oof_by_round, fold_of = per_round_oof_scores(bundle, n_folds=n_folds, seed=seed)
    act_scores = _scores_by_round_from_oof(oof_by_round)
    step_scores = _step_index_scores_by_round(n_turns)

    activation_results, step_results = [], []
    for r in target_recalls:
        ab_act, th_act = _peak_score_cascade(act_scores, labels, fold_of, r, n_folds)
        activation_results.append(_aggregate_cascade(
            ab_act, labels, n_turns, r, th_act,
            "activation probe, layer %d" % int(bundle.layer)))

        ab_step, th_step = _peak_score_cascade(step_scores, labels, fold_of, r, n_folds)
        step_results.append(_aggregate_cascade(
            ab_step, labels, n_turns, r, th_step, "step-index-only baseline"))

    diag = {
        "n_trajectories": len(labels),
        "n_unsafe": int(sum(labels.values())),
        "n_safe": int(len(labels) - sum(labels.values())),
        "rounds_with_oof": sorted(oof_by_round.keys()),
        "n_folds_used_by_round": {int(k): int(d["n_folds_used"])
                                  for k, d in oof_by_round.items()},
    }
    return activation_results, step_results, diag


def content_bar_cascade_note() -> dict:
    """Reproduction requirement 5's explicit escape valve, taken. See the
    module docstring's "THE TWO CASCADES BUILT HERE, AND THE ONE NOT BUILT".
    """
    return {
        "skipped": True,
        "reason": (
            "common.confound.content_bar fits its own internal group-free "
            "K-fold CV and returns one aggregate AUC, not a per-trajectory "
            "out-of-fold score usable to decide WHEN to abort. Reusing it for "
            "a recall-controlled cascade needs a parallel per-round, "
            "nested-fold TF-IDF fit/score pipeline (vocabulary + IDF fit on "
            "calibration trajectories' first-k-turn text only, scored on the "
            "held-out fold, repeated for every k in 1..16 and every outer "
            "fold) -- a second bespoke pipeline, not a cheap reuse of the "
            "existing bar. Not built in this pass; the activation-vs-"
            "step-index comparison is the one that was cheap to build "
            "correctly."
        ),
    }


def _cascade_result_to_dict(r: CascadeResult) -> dict:
    return {
        "target_recall": r.target_recall, "achieved_recall": r.achieved_recall,
        "rounds": list(r.rounds), "per_round_threshold": list(r.per_round_threshold),
        "frac_aborted": r.frac_aborted, "token_saving": r.token_saving,
        "n_episodes": r.n_episodes, "calibration_note": r.calibration_note,
    }


# ---------------------------------------------------------------------------
# 5. The real run: cached bundle, no model, no GPU
# ---------------------------------------------------------------------------
def run() -> dict:
    C.ensure_artifacts()
    bundle, bundle_path, bundle_meta = find_bundle(C.ARTIFACTS, C.MODEL_TAG, C.LAYER)
    assert int(bundle.layer) == int(C.LAYER), (
        "found bundle layer %d != config.LAYER %d -- the glob pattern should "
        "have prevented this" % (bundle.layer, C.LAYER))
    n_traj = len(set(str(u) for u in bundle.traj_uid))
    _print("[cascade] bundle: %s (X=%s, %d trajectories)"
          % (bundle_path.name, bundle.X.shape, n_traj))

    t0 = time.time()
    act, step, diag = run_cascade(bundle, target_recalls=DEFAULT_TARGET_RECALLS,
                                  n_folds=C.N_FOLDS, seed=C.SEED)
    elapsed = time.time() - t0

    payload = {
        "corpus": C.CORPUS, "model_id": bundle.model_id, "model_tag": C.MODEL_TAG,
        "layer": int(bundle.layer), "bundle_path": str(bundle_path),
        "bundle_data_fingerprint": bundle_meta.get("data_fingerprint"),
        "target_recalls": list(DEFAULT_TARGET_RECALLS),
        "n_folds": int(C.N_FOLDS), "seed": int(C.SEED),
        "diagnostics": diag,
        "activation_cascade": [_cascade_result_to_dict(r) for r in act],
        "step_index_baseline": [_cascade_result_to_dict(r) for r in step],
        "content_bar_cascade": content_bar_cascade_note(),
        "elapsed_sec": elapsed,
    }

    out_path = keyed_path(C.ARTIFACTS, "cascade", ".json", C.CORPUS, C.MODEL_TAG,
                          "L%d" % int(bundle.layer))
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(out_path))
    _print("[cascade] wrote %s" % out_path)

    _print("")
    _print("recall-controlled cascade -- %s, layer %d, %d trajectories, %.1fs"
          % (C.CORPUS, int(bundle.layer), diag["n_trajectories"], elapsed))
    _print("%6s | %8s %8s %8s | %8s %8s %8s"
          % ("target", "act_rec", "act_abrt", "act_save",
             "step_rec", "step_abrt", "step_save"))
    for r_act, r_step in zip(act, step):
        _print("%6.2f | %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f"
              % (r_act.target_recall, r_act.achieved_recall, r_act.frac_aborted,
                 r_act.token_saving, r_step.achieved_recall, r_step.frac_aborted,
                 r_step.token_saving))
    beats = all(a.token_saving > s.token_saving for a, s in zip(act, step))
    _print("")
    _print("activation cascade beats step-index baseline at EVERY target: %s" % beats)
    _print("content-bar cascade: SKIPPED (%s)" % content_bar_cascade_note()["reason"][:80])
    return payload


# ---------------------------------------------------------------------------
# 6. CPU self-test -- NO model, NO GPU, NO network. Synthetic bundle only.
#   TP_SELFTEST=1 python -m steering_tutorials.traj_probes.cascade
# ---------------------------------------------------------------------------
def _build_synthetic_bundle(rng, n_per_class: int, separable: bool) -> ActivationBundle:
    """Trajectory LENGTH is independent of label here on purpose -- this
    fixture isolates the score-vs-noise question the self-test is about,
    leaving the length-confound question to the real corpus (README Section 3
    already covers that ground).
    """
    hidden = 4
    X, y, step, uid, grp = [], [], [], [], []
    for i in range(n_per_class * 2):
        label = i % 2
        n = int(rng.integers(3, 11))
        u = "traj%03d" % i
        for k in range(n):
            if separable:
                h = rng.normal(scale=0.2, size=hidden) + (5.0 if label else -5.0)
            else:
                h = rng.normal(scale=1.0, size=hidden)
            X.append(h)
            y.append(label)
            step.append(k)
            uid.append(u)
            grp.append(u)
    return ActivationBundle(
        X=np.asarray(X, dtype=np.float64), y=np.asarray(y),
        step_index=np.asarray(step), traj_uid=np.asarray(uid),
        group_id=np.asarray(grp), layer=12, model_id="synthetic-selftest",
        behaviour_fingerprint="selftest-%s" % ("sep" if separable else "noise"))


def _self_test() -> None:  # noqa: C901 - a test, read top to bottom
    _print("=" * 74)
    _print("cascade.py self-test (TP_SELFTEST=1) -- synthetic bundle, no model, "
          "no network, no GPU")
    _print("=" * 74)

    rng = np.random.default_rng(0)
    n_folds, seed = 5, 0
    targets = (0.90, 0.70)

    # --- thresholds calibrated on TRAIN only, structurally verified --------
    b_sep = _build_synthetic_bundle(rng, n_per_class=40, separable=True)
    _, labels = _trajectory_lengths(b_sep)
    fold_of = _trajectory_fold_assignment(b_sep, n_folds, seed)
    for f in range(n_folds):
        calib = set(t for t in labels if fold_of[t] != f)
        test = set(t for t in labels if fold_of[t] == f)
        assert calib.isdisjoint(test), "fold %d calibration/test overlap" % f
        assert calib | test == set(labels), "fold %d does not cover every trajectory" % f
    _print("OK  every outer fold's calibration set is disjoint from its test "
          "set, and together they cover every trajectory (calibration never "
          "sees the fold it is about to score)")

    act_sep, step_sep, diag_sep = run_cascade(
        b_sep, target_recalls=targets, n_folds=n_folds, seed=seed)
    assert diag_sep["n_trajectories"] == 80 and diag_sep["n_unsafe"] == 40
    for r in act_sep:
        assert abs(r.achieved_recall - r.target_recall) <= 0.20, (
            "separable case: achieved recall %.4f far from target %.4f"
            % (r.achieved_recall, r.target_recall))
    _print("OK  [separable] achieved recall tracks target: %s"
          % ", ".join("%.2f->%.4f" % (r.target_recall, r.achieved_recall)
                     for r in act_sep))

    b_noise = _build_synthetic_bundle(rng, n_per_class=40, separable=False)
    act_noise, step_noise, diag_noise = run_cascade(
        b_noise, target_recalls=targets, n_folds=n_folds, seed=seed)
    for r in act_noise:
        assert abs(r.achieved_recall - r.target_recall) <= 0.25, (
            "noise case: achieved recall %.4f far from target %.4f"
            % (r.achieved_recall, r.target_recall))
    _print("OK  [noise] achieved recall still tracks target -- recall control "
          "is honoured by construction regardless of whether the score "
          "carries any real signal: %s"
          % ", ".join("%.2f->%.4f" % (r.target_recall, r.achieved_recall)
                     for r in act_noise))

    # Raw token_saving is NOT the right comparison here (see the module
    # docstring's "WHERE THE EARLINESS COMES FROM" section): at a fixed high
    # recall target a no-signal probe can only hit that recall by aborting
    # almost everyone, which can make token_saving LOOK better while being
    # useless. The properties that actually distinguish a good cascade from a
    # useless one are (a) it does not falsely abort safe episodes, and (b) it
    # catches unsafe episodes genuinely EARLY relative to their own length --
    # both recomputed directly here (not parsed out of calibration_note) so
    # the assertions are on real numbers, not on prose.
    def _diagnostics(bundle, target_recall):
        n_turns, labels = _trajectory_lengths(bundle)
        oof, fold_of = per_round_oof_scores(bundle, n_folds=n_folds, seed=seed)
        scores = _scores_by_round_from_oof(oof)
        abort_round, _thetas = _peak_score_cascade(
            scores, labels, fold_of, target_recall, n_folds)
        n_safe = sum(1 for t in labels if labels[t] == 0)
        n_safe_aborted = sum(1 for t, r in abort_round.items()
                             if labels[t] == 0 and r is not None)
        frac_safe_aborted = (n_safe_aborted / n_safe) if n_safe else float("nan")
        rel_catch = [abort_round[t] / n_turns[t] for t in labels
                    if labels[t] == 1 and abort_round.get(t) is not None]
        mean_rel_catch = float(np.mean(rel_catch)) if rel_catch else float("nan")
        return frac_safe_aborted, mean_rel_catch

    fsa_sep, catch_sep = _diagnostics(b_sep, 0.90)
    fsa_noise, catch_noise = _diagnostics(b_noise, 0.90)
    assert fsa_sep <= 0.10, (
        "separable case should almost never falsely abort a safe episode, "
        "got frac_safe_aborted=%.4f" % fsa_sep)
    assert fsa_noise > fsa_sep, (
        "a no-signal probe hitting the SAME high recall target should be "
        "MORE indiscriminate about safe episodes than a separable one "
        "(%.4f vs %.4f)" % (fsa_noise, fsa_sep))
    assert catch_sep < catch_noise, (
        "separable case should catch unsafe episodes EARLIER, relative to "
        "their own length, than the noise case (%.4f vs %.4f)"
        % (catch_sep, catch_noise))
    _print("OK  [selectivity @ target_recall=0.90] frac_safe_aborted "
          "separable=%.4f vs noise=%.4f -- a no-signal probe can only reach "
          "high recall by aborting almost everyone" % (fsa_sep, fsa_noise))
    _print("OK  [earliness @ target_recall=0.90] mean_relative_catch_unsafe "
          "separable=%.4f vs noise=%.4f -- lower means genuinely earlier"
          % (catch_sep, catch_noise))
    for r in list(act_sep) + list(act_noise):
        assert 0.0 <= r.token_saving <= 1.0 and 0.0 <= r.frac_aborted <= 1.0

    # step-index baseline runs and returns sane numbers on both fixtures
    # (length is independent of label here by construction, so it is not
    # expected to win -- just to execute cleanly and report bounded values).
    for lst in (step_sep, step_noise):
        for r in lst:
            assert 0.0 <= r.achieved_recall <= 1.0
            assert 0.0 <= r.frac_aborted <= 1.0
            assert 0.0 <= r.token_saving <= 1.0
    _print("OK  step-index baseline returns bounded recall/abort/saving on "
          "both fixtures")

    payload = [_cascade_result_to_dict(r) for r in act_sep]
    blob = json.dumps(payload, sort_keys=True)
    back = json.loads(blob)
    assert back[0]["target_recall"] == act_sep[0].target_recall
    assert len(back[0]["rounds"]) == len(act_sep[0].rounds)
    _print("OK  CascadeResult payload round-trips through JSON")

    note = content_bar_cascade_note()
    assert note["skipped"] is True and note["reason"]
    _print("OK  content-bar cascade honestly reports SKIPPED, with a reason")

    _print("")
    _print("OK -- cascade.py self-test passed CPU-only, no model, no GPU, no "
          "network.")


def main() -> None:
    if os.environ.get("TP_SELFTEST") == "1":
        _self_test()
        return
    run()


if __name__ == "__main__":
    main()
