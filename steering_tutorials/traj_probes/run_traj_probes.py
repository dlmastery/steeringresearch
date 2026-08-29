"""run_traj_probes.py -- reproduction A ("Doomed from the Start: Early Abort of
LLM Agent Episodes via a Recall-Controlled Probe Cascade", Ruan, Huang, Zhou,
Wei, Lin, Wang and Sun, 7 Jul 2026, arXiv:2607.06503).

WHAT THIS SCRIPT DOES
----------------------
Loads ATBench, extracts Gemma-3-1B residual-stream activations at every turn
boundary (`activations.HFActivationExtractor`), then runs FOUR arms on the same
bundle and the same group-aware CV folds:

  linear        LinearTrajProbe on the activations -- the headline. Its
                ProbeResult is filled COMPLETELY: raw auc, auc_ci,
                auc_residualised (CONTROL 1: linear step-index residualisation),
                content_bar_auc / clears_content_bar (CONTROL 2: the unigram
                TF-IDF bar on the SAME text the model read), random_label_auc
                (CONTROL 3: Hewitt-Liang capacity ceiling, arXiv:1909.03368).
  step_only     StepIndexProbe -- what step index ALONE gets. This is the floor
                an activation probe must clear, never 0.5: failing trajectories
                run longer on ATBench (config.py measured this directly), so a
                position detector scores well while reading nothing.
  content_bar   The standalone unigram TF-IDF bar (`probes.content_bar_control`)
                on the same (trajectory_uid, label, text) triples the linear
                arm's own internal bar used -- reported in full (auc, auc_raw,
                n_folds) rather than only the paired auc/clears fields.
  random_label  The standalone Hewitt-Liang ceiling (`probes.random_label_control`)
                on the same bundle.

Then the EARLY-ABORT CURVE: using only rows with step_index < k for
k in (1, 2, 3, 4, 6, 8), fit/score a linear probe and report AUC (raw and
residualised) vs k -- "how early is failure predictable", the paper's actual
claim. The train/test GROUP assignment is decided ONCE on the full bundle and
reused unchanged at every k (`_trajectory_fold_assignment` /
`_cv_indices_from_fold_map`), rather than re-computed per k. That matters:
StratifiedGroupKFold balances per-group SAMPLE COUNTS as well as labels, and
those counts change with k, so recomputing per k would silently compare
different train/test partitions and call the difference "early abort".

WHAT THIS DOES NOT DO
---------------------
It does not print our numbers beside the paper's. Paper A runs Qwen-2.5-7B /
Qwen3-32B / Llama-3.3-70B on TextCraft / WebShop; we run Gemma-3-1B on ATBench.
Different model, different corpus. What transfers is the METHOD and its
controls (step-index residualisation, the content bar, the random-label
ceiling), not the magnitudes -- see `types.py`'s module docstring.

No generation judge is used or imported anywhere in this file: this is a
detection lesson, scored against the corpus's own ground-truth labels.

results.json is written BEFORE the summary print (config.preflight() +
`common.artifact_paths.keyed_path`; a late crash must not lose the run) and is
resumable via the activation cache's row journal (activations.RowJournal) --
this host reaps long background jobs.

ASCII stdout/stderr ONLY. Never print alpha/Delta/||/arrows: this host's
Windows cp1252 console crashes on them.

Run it (GPU, real corpus):
    C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.run_traj_probes

Shrink it into one foreground window (env caps; default to the config values):
    TP_RUN_N=60 TP_RUN_LAYER=12 TP_RUN_MAX_TURNS=16 \\
        C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.run_traj_probes

CPU self-test, no model, no GPU, no network:
    TP_SELFTEST=1 C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.run_traj_probes
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:  # package form
    from . import config as C
    from . import data
    from . import leakage
    from .activations import ExtractSettings, HFActivationExtractor
    from .probes import (
        EVAL_N_BOOT, LinearTrajProbe, StepIndexProbe, StepResidualiser, auc,
        bootstrap_auc_ci, content_bar_control, group_folds,
        random_label_control,
    )
    from .types import ActivationBundle, ProbeResult
except ImportError:  # pragma: no cover - direct-script form
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent.parent))
    from steering_tutorials.traj_probes import config as C
    from steering_tutorials.traj_probes import data
    from steering_tutorials.traj_probes import leakage
    from steering_tutorials.traj_probes.activations import (ExtractSettings,
                                                            HFActivationExtractor)
    from steering_tutorials.traj_probes.probes import (
        EVAL_N_BOOT, LinearTrajProbe, StepIndexProbe, StepResidualiser, auc,
        bootstrap_auc_ci, content_bar_control, group_folds,
        random_label_control)
    from steering_tutorials.traj_probes.types import ActivationBundle, ProbeResult

try:
    from ..common.artifact_paths import keyed_path
except (ImportError, ValueError):  # pragma: no cover
    from steering_tutorials.common.artifact_paths import keyed_path

__all__ = ["run", "early_abort_curve", "_self_test"]

# The paper's own claim is EARLY predictability; these are turn counts to
# truncate to, not a cost knob (see the module docstring and leakage.py: the
# corpus's own deterministic region starts at 18, so every k here stays inside
# the region the config's TP_MAX_TURNS=16 default already keeps clean).
EARLY_ABORT_KS = (1, 2, 3, 4, 6, 8)


def _print(*args) -> None:
    """ASCII-only stdout. This host's cp1252 console crashes on unicode."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg)
    except Exception:  # pragma: no cover
        print(msg.encode("ascii", "replace").decode("ascii"))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else int(default)


# ---------------------------------------------------------------------------
# Small helpers shared by the four arms and the early-abort curve
# ---------------------------------------------------------------------------
def _trajectory_labels(bundle: ActivationBundle):
    """-> (uids, labels) at the trajectory level, first-seen order in
    `bundle.traj_uid`. Mirrors the exact ordering `probes.trajectory_scores`
    uses internally, so a standalone content-bar/random-label call here sees
    the SAME (uid, label) set the linear arm's own internal controls used.
    """
    uid = np.asarray([str(u) for u in bundle.traj_uid])
    y = np.asarray(bundle.y).astype(np.int64)
    order, labels = [], []
    for u in dict.fromkeys(uid.tolist()):
        order.append(u)
        labels.append(int(y[uid == u][0]))
    return np.asarray(order), np.asarray(labels)


def _probe_result_to_dict(r: ProbeResult) -> dict:
    return {
        "method": r.method, "layer": int(r.layer), "auc": r.auc,
        "auc_ci": [float(r.auc_ci[0]), float(r.auc_ci[1])],
        "auc_residualised": r.auc_residualised,
        "content_bar_auc": r.content_bar_auc,
        "clears_content_bar": r.clears_content_bar,
        "random_label_auc": r.random_label_auc,
        "n_items": int(r.n_items), "n_trajectories": int(r.n_trajectories),
        "notes": r.notes,
    }


PROBE_C = 1.0          # single source of truth; probes._fit_fold takes it as an arg


def _fit_score(Xtr, ytr, Xte, seed: int, C: float = PROBE_C) -> np.ndarray:
    """Standardise on train only, L2 logistic regression, score test.

    DELEGATES to `probes._fit_fold` rather than reimplementing it. The
    early-abort curve must use the SAME estimator as the `linear` arm, and a
    second copy of the hyper-parameters is a copy that drifts: change C or the
    solver in probes.py and a duplicate here would keep the old value while
    still being labelled "the same probe". Reaching past the underscore is the
    lesser evil -- the alternative is two definitions of the estimator whose
    agreement nothing checks.
    """
    from steering_tutorials.traj_probes.probes import _fit_fold

    return _fit_fold(Xtr, ytr, Xte, C, seed)


# ---------------------------------------------------------------------------
# The early-abort curve, with the fold assignment FIXED once and reused
# ---------------------------------------------------------------------------
def _trajectory_fold_assignment(bundle: ActivationBundle, n_folds: int,
                                seed: int) -> dict:
    """Decide ONCE, on the full (un-truncated-by-k) bundle, which fold each
    trajectory belongs to. Reused unchanged at every k so the early-abort curve
    compares the same train/test partition as k grows, not a different one.

    `group_folds` is called with one row per trajectory (each its own group by
    construction, per data.py's GROUPING section), so the partition depends only
    on trajectory identity and label -- never on how many turn-rows a
    trajectory happens to contribute at a given k.
    """
    order, labels = _trajectory_labels(bundle)
    splits = group_folds(order, labels, n_folds=n_folds, seed=seed)
    fold_of = {}
    for f, (_tr, te) in enumerate(splits):
        for i in te:
            fold_of[str(order[i])] = f
    return fold_of


def _cv_indices_from_fold_map(group_id, fold_of: dict, n_folds: int) -> list:
    """-> [(train_idx, test_idx), ...] for a row subset, from the FIXED
    trajectory->fold map. A trajectory absent from `fold_of` (should not
    happen; every group appears in the full bundle) is dropped from both sides
    rather than silently mis-assigned.
    """
    g = np.asarray([str(x) for x in group_id])
    fidx = np.asarray([fold_of.get(x, -1) for x in g])
    out = []
    for f in range(n_folds):
        te = np.flatnonzero(fidx == f)
        tr = np.flatnonzero((fidx != f) & (fidx >= 0))
        if len(te) == 0 or len(tr) == 0:
            continue
        out.append((tr, te))
    return out


def early_abort_curve(bundle: ActivationBundle, n_folds: int = 5,
                      seed: int = 0, n_boot: int = EVAL_N_BOOT,
                      ks=EARLY_ABORT_KS) -> list:
    """-> list of {k, n_rows, n_folds_used, auc, auc_ci, auc_residualised}.

    Filters to rows with step_index < k, scores with the SAME linear probe as
    the `linear` arm (raw and degree-1-residualised), using the fold assignment
    fixed by `_trajectory_fold_assignment`. A k with fewer than 2 usable folds
    or a single-class fold is reported with `auc=None` and a `note`, never
    silently dropped from the list.
    """
    fold_of = _trajectory_fold_assignment(bundle, n_folds, seed)
    y_all = np.asarray(bundle.y).astype(np.int64)
    step_all = np.asarray(bundle.step_index)
    out = []
    for k in ks:
        mask = step_all < k
        n_rows = int(mask.sum())
        row = {"k": int(k), "n_rows": n_rows, "auc": None, "auc_ci": None,
              "auc_residualised": None}
        if n_rows == 0:
            row["note"] = "no rows at this k"
            out.append(row)
            continue

        Xk = np.asarray(bundle.X[mask], dtype=np.float64)
        yk = y_all[mask]
        stepk = step_all[mask]
        gk = bundle.group_id[mask]
        splits = _cv_indices_from_fold_map(gk, fold_of, n_folds)
        if len(splits) < 2:
            row["note"] = "fewer than 2 usable folds at this k"
            out.append(row)
            continue

        raw_scores = np.full(n_rows, np.nan)
        res_scores = np.full(n_rows, np.nan)
        used = 0
        for tr, te in splits:
            if len(np.unique(yk[tr])) < 2 or len(np.unique(yk[te])) < 2:
                continue
            raw_scores[te] = _fit_score(Xk[tr], yk[tr], Xk[te], seed)
            try:
                r = StepResidualiser(degree=1).fit(Xk[tr], stepk[tr])
                res_scores[te] = _fit_score(
                    r.transform(Xk[tr], stepk[tr]), yk[tr],
                    r.transform(Xk[te], stepk[te]), seed)
            except ValueError:
                pass  # constant step_index within this fold/k -- leave NaN
            used += 1
        row["n_folds_used"] = used

        ok = np.isfinite(raw_scores)
        if used == 0 or not ok.any() or len(np.unique(yk[ok])) < 2:
            row["note"] = "no usable fold (single-class) at this k"
            out.append(row)
            continue

        a = auc(raw_scores[ok], yk[ok])
        ci = bootstrap_auc_ci(raw_scores[ok], yk[ok], gk[ok], n_boot=n_boot,
                              seed=seed)
        row["auc"] = float(a)
        row["auc_ci"] = [float(ci[0]), float(ci[1])]

        okr = np.isfinite(res_scores)
        if okr.any() and len(np.unique(yk[okr])) >= 2:
            row["auc_residualised"] = float(auc(res_scores[okr], yk[okr]))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def run() -> dict:
    t0 = time.time()
    C.preflight()

    corpus_name = C.CORPUS
    n_per_class = _env_int("TP_RUN_N", C.N_PER_CLASS)
    layer = _env_int("TP_RUN_LAYER", C.LAYER)
    max_turns = _env_int("TP_RUN_MAX_TURNS", C.MAX_TURNS)
    seed = C.SEED
    n_folds = C.N_FOLDS
    n_boot = C.BOOTSTRAP

    _print("=" * 74)
    _print("traj_probes -- reproduction A (Doomed from the Start, arXiv:2607.06503)")
    _print("Gemma-3-1B / ATBench. Paper A runs Qwen-2.5-7B / Qwen3-32B / "
          "Llama-3.3-70B on TextCraft / WebShop / ALFWorld / tau-bench -- "
          "different model, different corpus. The METHOD and its controls "
          "transfer; the MAGNITUDES do not, and are never printed beside the "
          "paper's.")
    _print("corpus=%s n_per_class=%d layer=%d max_turns=%d seed=%d n_folds=%d "
          "n_boot=%d" % (corpus_name, n_per_class, layer, max_turns, seed,
                        n_folds, n_boot))
    _print("=" * 74)

    corpus = data.load_corpus(n_per_class=n_per_class, seed=seed,
                             corpus=corpus_name, max_turns=max_turns)
    _print(data.summarise(corpus))

    leak = leakage.deterministic_step_region(corpus)
    _print("deterministic_step_region: %s" % leak)
    if not leak.get("is_empty", True):
        _print("!! non-empty deterministic region above -- this run's numbers "
              "would include the leak leakage.py exists to catch; check "
              "TP_RUN_MAX_TURNS")

    texts = {t.uid: t.text for t in corpus.trajectories}

    extractor = HFActivationExtractor(
        C.MODEL_ID, settings=ExtractSettings(pooling="last", layer=layer),
        cache_dir=C.ARTIFACTS)
    bundle = extractor.extract(corpus, layer=layer)
    _print("bundle: %d rows, %d trajectories, behaviour_fingerprint=%s"
          % (len(bundle.X), len(set(str(u) for u in bundle.traj_uid)),
             bundle.behaviour_fingerprint))

    # --- the four arms, all on the SAME bundle / SAME group folds ----------
    linear_probe = LinearTrajProbe(C=1.0, residual_degree=1, traj_pool="max")
    r_linear = linear_probe.fit_predict_cv(bundle, n_folds=n_folds, seed=seed,
                                           texts=texts, n_boot=n_boot,
                                           run_controls=True)
    _print("[linear]       auc=%.4f ci=%s auc_res=%s bar=%s clears=%s rand=%s"
          % (r_linear.auc, r_linear.auc_ci, r_linear.auc_residualised,
             r_linear.content_bar_auc, r_linear.clears_content_bar,
             r_linear.random_label_auc))

    r_step = StepIndexProbe(C=1.0).fit_predict_cv(bundle, n_folds=n_folds,
                                                  seed=seed, n_boot=n_boot)
    _print("[step_only]    auc=%.4f ci=%s (the floor a probe must beat, NOT 0.5)"
          % (r_step.auc, r_step.auc_ci))

    uids, traj_labels = _trajectory_labels(bundle)
    bar = content_bar_control(uids, traj_labels, texts, seed=seed)
    _print("[content_bar]  auc=%.4f auc_raw=%.4f n_folds=%d (unigram TF-IDF "
          "bar on the SAME text the model read; the linear arm's own paired "
          "content_bar_auc=%s / clears_content_bar=%s above is the binding "
          "comparison)"
          % (bar["auc"], bar["auc_raw"], bar["n_folds"],
             r_linear.content_bar_auc, r_linear.clears_content_bar))

    rand_auc = random_label_control(bundle, splits=None, C=1.0, seed=seed,
                                    degree=1, n_folds=n_folds)
    _print("[random_label] auc=%.4f (Hewitt-Liang capacity ceiling, "
          "arXiv:1909.03368; the linear arm's own random_label_auc=%s above "
          "used the same control)" % (rand_auc, r_linear.random_label_auc))

    curve = early_abort_curve(bundle, n_folds=n_folds, seed=seed, n_boot=n_boot)
    _print("early-abort curve (k = turns kept, step_index < k; SAME fold "
          "assignment reused at every k):")
    for row in curve:
        _print("  k=%-2d n_rows=%-5d auc=%s auc_res=%s%s"
              % (row["k"], row["n_rows"], row.get("auc"),
                 row.get("auc_residualised"),
                 ("  (%s)" % row["note"]) if row.get("note") else ""))

    results = {
        "config": dict(C.as_dict(), n_per_class=n_per_class, layer=layer,
                       max_turns=max_turns, seed=seed, n_folds=n_folds,
                       n_boot=n_boot, corpus=corpus_name),
        "note": "Gemma-3-1B / ATBench. Do not compare these numbers to "
                "arXiv:2607.06503's Qwen/TextCraft/WebShop figures -- different "
                "model, different corpus. Only the method and controls "
                "transfer.",
        "corpus_provenance": {
            "pool_fingerprint": corpus.pool_fingerprint,
            "licence": corpus.licence,
            "requested_n_per_class": corpus.requested_n_per_class,
            "achieved_n_safe": corpus.achieved_n_neg,
            "achieved_n_unsafe": corpus.achieved_n_pos,
            "label_provenance": corpus.label_provenance,
            "step_label_provenance": corpus.step_label_provenance,
            "turns": corpus.turn_count_summary(),
        },
        "bundle": {
            "n_rows": int(len(bundle.X)),
            "n_trajectories": int(len(set(str(u) for u in bundle.traj_uid))),
            "layer": int(bundle.layer), "model_id": bundle.model_id,
            "behaviour_fingerprint": bundle.behaviour_fingerprint,
        },
        "leakage": leak,
        "arms": {
            "linear": _probe_result_to_dict(r_linear),
            "step_only": _probe_result_to_dict(r_step),
            "content_bar": {
                "method": "content_bar", "auc": bar["auc"],
                "auc_raw": bar["auc_raw"], "n_folds": bar["n_folds"],
                "n_trajectories": int(len(uids)),
                "notes": "unigram TF-IDF centroid bar (CLAUDE.md section 17 "
                        "confound discipline; common.confound.content_bar), "
                        "directionless, on the SAME text AgentTrajectory.text "
                        "renders for the model. Paired against the linear "
                        "arm's own content_bar_auc/clears_content_bar above.",
            },
            "random_label": {
                "method": "random_label", "auc": rand_auc,
                "n_trajectories": int(len(uids)),
                "notes": "Hewitt and Liang, EMNLP 2019, 'Designing and "
                        "Interpreting Probes with Control Tasks' "
                        "(arXiv:1909.03368) -- labels permuted BETWEEN "
                        "trajectories, same probe/CV settings as the linear "
                        "arm.",
            },
        },
        "early_abort_curve": curve,
        "wall_clock_sec": round(time.time() - t0, 2),
    }

    results_path = keyed_path(C.ARTIFACTS, "results", ".json", corpus_name,
                             C.MODEL_TAG, "L%d" % layer)
    tmp = results_path.with_suffix(results_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    os.replace(tmp, results_path)
    _print("wrote %s" % results_path)

    _print("")
    _print("SUMMARY (Gemma-3-1B/ATBench; method transfers from arXiv:2607.06503, "
          "magnitudes do not):")
    _print("  linear       auc=%.4f  auc_residualised=%s  clears_content_bar=%s"
          % (r_linear.auc, r_linear.auc_residualised, r_linear.clears_content_bar))
    _print("  step_only    auc=%.4f  (the floor)" % r_step.auc)
    _print("  content_bar  auc=%.4f  (the binding bar)" % bar["auc"])
    _print("  random_label auc=%.4f  (the capacity ceiling)" % rand_auc)
    _print("  wall_clock_sec=%.1f" % results["wall_clock_sec"])
    return results


# ---------------------------------------------------------------------------
# CPU self-test -- NO model, NO GPU, NO network. Synthetic bundle only.
#   TP_SELFTEST=1 python -m steering_tutorials.traj_probes.run_traj_probes
# ---------------------------------------------------------------------------
def _self_test() -> None:  # noqa: C901 - a test, read top to bottom
    _print("=" * 74)
    _print("run_traj_probes.py self-test (TP_SELFTEST=1) -- synthetic bundle, "
          "no model, no network, no GPU")
    _print("=" * 74)

    rng = np.random.default_rng(0)
    hidden, n_traj = 16, 30
    content_dir = rng.normal(size=hidden)   # a genuine, step-INDEPENDENT signal
    X, y, step, uid, grp = [], [], [], [], []
    texts = {}
    for i in range(n_traj):
        label = i % 2
        # failing (label=1) trajectories run longer -- the real-world
        # correlation that makes CONTROL 1 necessary in the first place.
        n = int(rng.integers(10, 16) if label else rng.integers(3, 6))
        u = "traj%03d" % i
        texts[u] = (("the agent looped and retried and failed " * 6) if label
                   else ("the agent fetched the record and finished " * 6))
        for k in range(n):
            h = rng.normal(scale=0.5, size=hidden) + content_dir * (2.0 * label - 1.0)
            X.append(h)
            y.append(label)
            step.append(k)
            uid.append(u)
            grp.append(u)
    bundle = ActivationBundle(
        X=np.asarray(X, dtype=np.float32), y=np.asarray(y),
        step_index=np.asarray(step), traj_uid=np.asarray(uid),
        group_id=np.asarray(grp), layer=12, model_id="synthetic-selftest",
        behaviour_fingerprint="selftest")
    assert isinstance(bundle, ActivationBundle)
    _print("OK  synthetic bundle: %d rows, %d trajectories" % (len(bundle.X), n_traj))

    n_folds, seed, n_boot = 5, 0, 300

    r_linear = LinearTrajProbe(C=1.0).fit_predict_cv(
        bundle, n_folds=n_folds, seed=seed, texts=texts, n_boot=n_boot)
    assert r_linear.content_bar_auc is not None, "linear arm did not run CONTROL 2"
    assert r_linear.random_label_auc is not None, "linear arm did not run CONTROL 3"
    assert r_linear.auc_residualised is not None, "linear arm did not run CONTROL 1"
    _print("OK  [linear] auc=%.4f auc_residualised=%.4f content_bar_auc=%.4f "
          "clears=%s random_label_auc=%.4f"
          % (r_linear.auc, r_linear.auc_residualised, r_linear.content_bar_auc,
             r_linear.clears_content_bar, r_linear.random_label_auc))

    r_step = StepIndexProbe().fit_predict_cv(bundle, n_folds=n_folds, seed=seed,
                                             n_boot=n_boot)
    assert r_step.auc_residualised == 0.5
    _print("OK  [step_only] auc=%.4f (floor)" % r_step.auc)

    uids, labels = _trajectory_labels(bundle)
    assert len(uids) == n_traj
    bar = content_bar_control(uids, labels, texts, seed=seed)
    assert 0.0 <= bar["auc"] <= 1.0
    _print("OK  [content_bar] auc=%.4f auc_raw=%.4f n_folds=%d"
          % (bar["auc"], bar["auc_raw"], bar["n_folds"]))

    rand_auc = random_label_control(bundle, seed=seed, n_folds=n_folds)
    assert np.isnan(rand_auc) or 0.0 <= rand_auc <= 1.0
    _print("OK  [random_label] auc=%.4f" % rand_auc)

    curve = early_abort_curve(bundle, n_folds=n_folds, seed=seed, n_boot=n_boot)
    assert len(curve) == len(EARLY_ABORT_KS)
    required = {"k", "n_rows", "auc", "auc_ci", "auc_residualised"}
    for row in curve:
        assert required <= set(row), row
        assert row["n_rows"] >= 0
    # k=1 keeps only step_index==0 -- one row per trajectory, still >= 2 folds.
    assert curve[0]["k"] == 1 and curve[0]["n_rows"] == n_traj
    # the curve is monotonically non-decreasing in n_rows as k grows.
    assert all(curve[i]["n_rows"] <= curve[i + 1]["n_rows"]
              for i in range(len(curve) - 1))
    _print("OK  early-abort curve: %d k-values, fold assignment fixed once and "
          "reused at every k, n_rows monotone in k" % len(curve))
    for row in curve:
        _print("    k=%-2d n_rows=%-4d auc=%s auc_res=%s"
              % (row["k"], row["n_rows"], row.get("auc"),
                 row.get("auc_residualised")))

    # the fixed fold map really is fixed: re-deriving it must be identical.
    fm1 = _trajectory_fold_assignment(bundle, n_folds, seed)
    fm2 = _trajectory_fold_assignment(bundle, n_folds, seed)
    assert fm1 == fm2
    _print("OK  _trajectory_fold_assignment is deterministic given (bundle, "
          "n_folds, seed)")

    payload = {
        "arms": {
            "linear": _probe_result_to_dict(r_linear),
            "step_only": _probe_result_to_dict(r_step),
            "content_bar": {"auc": bar["auc"], "auc_raw": bar["auc_raw"],
                           "n_folds": bar["n_folds"]},
            "random_label": {"auc": rand_auc},
        },
        "early_abort_curve": curve,
    }
    blob = json.dumps(payload, sort_keys=True)
    back = json.loads(blob)
    assert back["arms"]["linear"]["auc"] == _probe_result_to_dict(r_linear)["auc"]
    assert back["early_abort_curve"][0]["k"] == 1
    _print("OK  the whole reporting payload (arms + early-abort curve) "
          "round-trips through JSON")

    _print("")
    _print("OK -- run_traj_probes.py self-test passed CPU-only, no model, no "
          "GPU, no network.")


def main() -> None:
    if os.environ.get("TP_SELFTEST") == "1":
        _self_test()
        return
    run()


if __name__ == "__main__":
    main()
