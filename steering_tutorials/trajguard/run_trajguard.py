"""run_trajguard.py -- orchestrator for the TRAJGUARD (streaming decoding-time
jailbreak detection) lesson.

Builds per-token hidden-state trajectories for harmful vs benign completions of the
abliterated Gemma-3-1B (data.load_or_build), then runs 5-fold stratified CV over four
sequence detectors:

  threshold_freeform -> trajectory.ThresholdDetector  (TRAINING-FREE sliding-window
                        projection onto the harm direction -- the paper's own method)
  per_turn_max / trajectory_mlp / seq_gru -> multiturn_jailbreak.models.{PerTurnMaxProbe,
                        TrajectoryMLP, SeqGRU}  (REUSED unchanged: a token trajectory is
                        the same [n_steps, dim] object a turn trajectory is)

Out-of-fold predictions are pooled per method and scored with AUC (+ bootstrap 95% CI),
F1, ACC, TPR@FPR<=0.10. The EARLY-DETECTION curve reports out-of-fold AUC as a function
of how many generated tokens K we have seen -- the streaming / early-warning value --
for both the training-free threshold detector and a seq_gru trained on first-K-truncated
trajectories. Three PNGs (ROC, early-AUC vs K, risk-drift example) render on the Agg
backend; results.json is written BEFORE the ASCII summary so a late crash keeps the data.

Sibling modules (data / trajectory) and the reused multiturn_jailbreak.models are imported
LAZILY inside main() so `python -c "import ...run_trajguard"` passes against bare stubs.
CPU-only; ASCII stdout (Windows cp1252). Screening tier -- see README caveats.
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# CV + metric helpers (no sibling-module / model dependency -> import-safe)
# ---------------------------------------------------------------------------
def kfold_indices(n, n_folds, seed, labels=None):
    """K-fold split. Returns a list of (train_idx, test_idx) ndarrays.

    When `labels` is supplied (the caller always does) uses sklearn StratifiedKFold so
    each fold preserves the harmful/benign base rate; otherwise a plain shuffled KFold
    over range(n). `labels` is an OPTIONAL 4th arg -- the required 3-positional
    signature `(n, n_folds, seed)` is preserved for the interface contract.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if labels is not None:
        y = np.asarray(labels)
        n = len(y)
        k = int(min(n_folds, np.min(np.bincount(y.astype(int))))) if len(y) else n_folds
        k = max(2, k)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in skf.split(np.zeros((n, 1)), y)]
    kf = KFold(n_splits=int(max(2, n_folds)), shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in kf.split(np.arange(n))]


def bootstrap_auc_ci(y_true, y_score, n=C.BOOTSTRAP, seed=0):
    """Percentile bootstrap 95% CI on ROC-AUC. Returns (auc, lo, hi)."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    try:
        point = float(roc_auc_score(y_true, y_score))
    except ValueError:
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    m = len(y_true)
    boots = []
    for _ in range(int(n)):
        idx = rng.integers(0, m, m)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:  # need both classes for a defined AUC
            continue
        boots.append(roc_auc_score(yt, y_score[idx]))
    if not boots:
        return (point, point, point)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return (point, lo, hi)


def _tpr_at_fpr10(y_true, y_score):
    """TPR at the score threshold giving FPR <= 0.10 on this set."""
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= 0.10
    if not np.any(ok):
        return 0.0
    return float(np.max(tpr[ok]))


def _metrics(y_true, y_score):
    """Full metric bundle for one pooled out-of-fold prediction vector.

    Thresholds probabilities at 0.5 for F1/ACC; TPR is read at FPR<=0.10.
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc, lo, hi = bootstrap_auc_ci(y_true, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    return {
        "auc": float(auc),
        "auc_ci": [float(lo), float(hi)],
        "f1": f1,
        "acc": acc,
        "tpr_at_fpr10": _tpr_at_fpr10(y_true, y_score),
    }


def _safe_auc(y_true, y_score):
    """Point ROC-AUC or NaN if undefined (single-class pool)."""
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))
    except ValueError:
        return float("nan")


def _truncate(traj, k):
    """First-K generated-token slice of a [n_tokens, dim] trajectory (>=1 row)."""
    a = np.asarray(traj, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    out = a[: int(max(1, k))]
    return out if out.shape[0] > 0 else a[:1]


# ---------------------------------------------------------------------------
# Plotting (Agg backend, PNG only)
# ---------------------------------------------------------------------------
def _plot_roc(roc_scores, out_path):
    """Pooled out-of-fold ROC per method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for method, (yt, ys) in roc_scores.items():
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        auc = roc_auc_score(yt, ys)
        ax.plot(fpr, tpr, label="%s (AUC=%.3f)" % (method, auc))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("TrajGuard token-trajectory detection ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_early(early, out_path):
    """Out-of-fold AUC vs number of generated tokens K, both methods."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for method, blk in early.items():
        ks, aucs = [], []
        for k in C.EARLY_KS:
            v = blk.get(str(k))
            if isinstance(v, (int, float)) and v == v:  # finite, not NaN
                ks.append(k)
                aucs.append(v)
        if ks:
            ax.plot(ks, aucs, marker="o", label=method)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance")
    ax.set_xlabel("Generated tokens seen (K)")
    ax.set_ylabel("Out-of-fold AUC")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Early-detection: how few tokens flag the jailbreak?")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_drift(examples, out_path):
    """Per-token sliding-window RISK: one harmful (drifts up) vs one benign (flat)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for ex in examples:
        curve = ex.get("risk_curve") or []
        if not curve:
            continue
        x = np.arange(1, len(curve) + 1)
        lbl = "harmful" if ex.get("label") == 1 else "benign"
        ax.plot(x, curve, marker=".", label=lbl)
    ax.set_xlabel("Generated token index")
    ax.set_ylabel("Sliding-window harm risk")
    ax.set_title("Decoding-time risk drift (harmful vs benign)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _make_model(name, trajectory, MJ):
    if name == "threshold_freeform":
        return trajectory.ThresholdDetector()
    if name == "per_turn_max":
        return MJ.PerTurnMaxProbe()
    if name == "trajectory_mlp":
        return MJ.TrajectoryMLP()
    if name == "seq_gru":
        return MJ.SeqGRU()
    raise ValueError("unknown method: %s" % name)


def main():
    # Lazy sibling / reuse imports (guarded here, NOT at module top, so the import-check
    # passes even while data/trajectory are bare stubs).
    from . import data, trajectory
    from steering_tutorials.multiturn_jailbreak import models as MJ

    ds = data.load_or_build()
    trajs = list(ds["trajectories"])
    labels = np.asarray(ds["labels"]).astype(int)
    prompts = list(ds["prompts"])
    n = len(trajs)
    n_harmful = int((labels == 1).sum())
    n_benign = int((labels == 0).sum())
    lens = np.array([int(np.asarray(t).reshape(-1, np.asarray(t).shape[-1]).shape[0])
                     for t in trajs]) if n else np.array([])
    mean_len_h = float(lens[labels == 1].mean()) if n_harmful else float("nan")
    mean_len_b = float(lens[labels == 0].mean()) if n_benign else float("nan")
    print("[data] trajectories=%d harmful=%d benign=%d mean_len_h=%.1f mean_len_b=%.1f"
          % (n, n_harmful, n_benign, mean_len_h, mean_len_b))

    folds = kfold_indices(n, C.N_FOLDS, C.SEED, labels=labels)

    # --- per-method 5-fold pooled out-of-fold predictions --------------------
    methods_block = {}
    roc_scores = {}
    for name in C.METHODS:
        try:
            pooled_t, pooled_s = [], []
            for tr, te in folds:
                mdl = _make_model(name, trajectory, MJ)
                mdl.fit([trajs[i] for i in tr], labels[tr])
                proba = np.asarray(mdl.predict_proba([trajs[i] for i in te])).reshape(-1)
                pooled_t.append(labels[te])
                pooled_s.append(proba)
            yt = np.concatenate(pooled_t)
            ys = np.concatenate(pooled_s)
            methods_block[name] = _metrics(yt, ys)
            roc_scores[name] = (yt, ys)
            m = methods_block[name]
            print("[method:%s] auc=%.3f ci=[%.3f,%.3f] f1=%.3f acc=%.3f tpr@fpr10=%.3f"
                  % (name, m["auc"], m["auc_ci"][0], m["auc_ci"][1],
                     m["f1"], m["acc"], m["tpr_at_fpr10"]))
        except Exception as exc:
            methods_block[name] = {"error": str(exc)}
            print("[method:%s] FAILED: %s" % (name, exc))

    # --- early-detection curve: out-of-fold AUC vs K -------------------------
    early = {"threshold_freeform": {}, "seq_gru": {}}
    for k in C.EARLY_KS:
        # (a) training-free threshold detector, scored on the FIRST K tokens.
        try:
            pt, ps = [], []
            for tr, te in folds:
                det = trajectory.ThresholdDetector()
                det.fit([trajs[i] for i in tr], labels[tr])
                proba = np.asarray(
                    det.predict_proba_earlyK([trajs[i] for i in te], k)).reshape(-1)
                pt.append(labels[te])
                ps.append(proba)
            early["threshold_freeform"][str(k)] = _safe_auc(
                np.concatenate(pt), np.concatenate(ps))
        except Exception as exc:
            early["threshold_freeform"][str(k)] = float("nan")
            print("[early:threshold_freeform:K=%d] FAILED: %s" % (k, exc))
        # (b) seq_gru trained AND tested on first-K-truncated trajectories.
        try:
            pt, ps = [], []
            for tr, te in folds:
                gru = MJ.SeqGRU()
                gru.fit([_truncate(trajs[i], k) for i in tr], labels[tr])
                proba = np.asarray(
                    gru.predict_proba([_truncate(trajs[i], k) for i in te])).reshape(-1)
                pt.append(labels[te])
                ps.append(proba)
            early["seq_gru"][str(k)] = _safe_auc(np.concatenate(pt), np.concatenate(ps))
        except Exception as exc:
            early["seq_gru"][str(k)] = float("nan")
            print("[early:seq_gru:K=%d] FAILED: %s" % (k, exc))
    for k in C.EARLY_KS:
        print("[early] K=%2d  threshold=%.3f  seq_gru=%.3f"
              % (k, early["threshold_freeform"].get(str(k), float("nan")),
                 early["seq_gru"].get(str(k), float("nan"))))

    # --- demo risk-drift curves (detector fit on ALL data for the illustration) --
    examples = []
    try:
        center, unit_dir = trajectory.harm_direction(trajs, labels)
        hi = next((i for i in range(n) if labels[i] == 1), None)
        bi = next((i for i in range(n) if labels[i] == 0), None)
        for i in (hi, bi):
            if i is None:
                continue
            scores = trajectory.token_scores(trajs[i], center, unit_dir)
            risk = np.asarray(
                trajectory.sliding_window_risk(scores, C.WINDOW)).reshape(-1)
            examples.append({
                "label": int(labels[i]),
                "prompt": str(prompts[i]),
                "risk_curve": [float(x) for x in risk],
            })
    except Exception as exc:
        print("[examples] FAILED: %s" % exc)

    # --- assemble + WRITE results.json BEFORE the summary print --------------
    results = {
        "model_id": C.MODEL_ID,
        "layer": int(C.LAYER),
        "n_harmful": n_harmful,
        "n_benign": n_benign,
        "max_new_tokens": int(C.MAX_NEW_TOKENS),
        "window": int(C.WINDOW),
        "seed": int(C.SEED),
        "n_folds": int(C.N_FOLDS),
        "judge": None,
        "methods": methods_block,
        "early_detection": early,
        "mean_traj_len_harmful": mean_len_h,
        "mean_traj_len_benign": mean_len_b,
        "examples": examples,
        "plots": [],
    }

    plots = []
    for fn, png, arg in (("roc", C.ROC_PNG, roc_scores),
                         ("early", C.EARLY_PNG, early),
                         ("drift", C.DRIFT_PNG, examples)):
        try:
            if fn == "roc" and arg:
                _plot_roc(arg, png)
                plots.append(str(png))
            elif fn == "early":
                _plot_early(arg, png)
                plots.append(str(png))
            elif fn == "drift" and arg:
                _plot_drift(arg, png)
                plots.append(str(png))
        except Exception as exc:
            print("[plot:%s] FAILED: %s" % (fn, exc))
    results["plots"] = plots

    C.ARTIFACTS.mkdir(exist_ok=True)
    with open(C.RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("[write] %s" % C.RESULTS_PATH)

    _print_summary(results)
    return results


def _print_summary(results):
    line = "-" * 78
    print("")
    print(line)
    print("TRAJGUARD -- streaming decoding-time jailbreak detection (SCREENING TIER)")
    print("model=%s layer=%d  harmful=%d benign=%d  window=%d folds=%d seed=%d"
          % (results["model_id"], results["layer"], results["n_harmful"],
             results["n_benign"], results["window"], results["n_folds"],
             results["seed"]))
    print("mean trajectory length: harmful=%.1f benign=%.1f tokens"
          % (results["mean_traj_len_harmful"], results["mean_traj_len_benign"]))
    print(line)
    print("%-20s %7s %-15s %6s %6s %8s"
          % ("method", "AUC", "95% CI", "F1", "ACC", "TPR@10"))
    for name in C.METHODS:
        cell = results["methods"].get(name)
        if not isinstance(cell, dict):
            continue
        if "error" in cell:
            print("%-20s  [FAILED]" % name)
            continue
        ci = "[%.2f,%.2f]" % (cell["auc_ci"][0], cell["auc_ci"][1])
        print("%-20s %7.3f %-15s %6.2f %6.2f %8.2f"
              % (name, cell["auc"], ci, cell["f1"], cell["acc"], cell["tpr_at_fpr10"]))
    print(line)
    print("EARLY DETECTION -- out-of-fold AUC vs generated tokens seen (K):")
    print("%-20s %s" % ("K:", "  ".join("%6d" % k for k in C.EARLY_KS)))
    for method in ("threshold_freeform", "seq_gru"):
        blk = results["early_detection"].get(method, {})
        row = "  ".join("%6.3f" % blk.get(str(k), float("nan")) for k in C.EARLY_KS)
        print("%-20s %s" % (method, row))
    print(line)
    print("READ: threshold_freeform is the TRAINING-FREE sliding-window projection (the "
          "paper's own method); the reused seq_gru/trajectory_mlp are the learned "
          "sequence detectors. If AUC rises with K, the jailbreak signal accumulates "
          "token-by-token -- it can be flagged BEFORE the harmful content is fully "
          "emitted. Screening tier (n small; abliterated model; label=prompt class).")
    print(line)


if __name__ == "__main__":
    main()
