"""run_multiturn.py — orchestrator for the multi-turn jailbreak DETECTION lesson.

Loads the Attack_600 (positive) + topic-matched UltraChat (negative) conversations,
embeds each turn with one or both embedders (gemma / minilm), then runs group-aware
N-fold CV over four sequence classifiers (per-turn-max baseline, trajectory-MLP,
seq-GRU, hierarchical-attention). Out-of-fold scores are pooled per (embedder x
method) and scored with AUC (+ bootstrap 95% CI), F1, ACC, and TPR@FPR<=0.10.
Results are written to results.json (schema in the interface contract) BEFORE the
ASCII summary print; three PNGs are rendered with the matplotlib Agg backend.

The sibling modules (data / embed / models) are imported lazily INSIDE main() so
`python -c "import ...run_multiturn"` succeeds even while those modules are stubs.
CPU-only; env caps (MJ_N_POS / MJ_N_NEG / MJ_EMBED / MJ_FOLDS) already live in config.
Stdout is ASCII only (Windows cp1252).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# CV + metric helpers (no sibling-module / model dependency -> import-safe)
# ---------------------------------------------------------------------------
def group_kfold_indices(groups, n_folds, seed):
    """Group-aware K-fold splits. Returns list of (train_idx, test_idx) ndarrays.

    Groups are shuffled with `seed` so folds are deterministic but not ordered by
    the raw group ids; a whole group stays inside a single fold (no target leakage
    across CV folds). Falls back to the number of distinct groups if it is < n_folds.
    """
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n = len(groups)
    n_groups = len(np.unique(groups))
    k = int(min(n_folds, n_groups))
    if k < 2:
        # Degenerate: not enough groups to split. One fold = train==test.
        idx = np.arange(n)
        return [(idx, idx)]

    # Shuffle the group labels deterministically by remapping each group id to a
    # random rank; GroupKFold itself is order-deterministic, so this injects seed.
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = rng.permutation(len(uniq))
    remap = {g: perm[i] for i, g in enumerate(uniq)}
    shuffled = np.array([remap[g] for g in groups])

    X_dummy = np.zeros((n, 1))
    gkf = GroupKFold(n_splits=k)
    return [(tr, te) for tr, te in gkf.split(X_dummy, groups=shuffled)]


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
    """Full metric bundle for one pooled out-of-fold prediction vector."""
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


def _make_model(name):
    """Instantiate one classifier by its stable config key (imported lazily)."""
    from . import models

    if name == "per_turn_max":
        return models.PerTurnMaxProbe()
    if name == "trajectory_mlp":
        return models.TrajectoryMLP()
    if name == "seq_gru":
        return models.SeqGRU()
    if name == "hier_attn":
        return models.HierAttn()
    raise ValueError("unknown method: %s" % name)


def _cv_pool(seqs, labels, groups, method):
    """Group-aware N-fold CV for one method. Returns (y_true_pooled, y_score_pooled)."""
    labels = np.asarray(labels)
    folds = group_kfold_indices(groups, C.N_FOLDS, C.SEED)
    pooled_true, pooled_score = [], []
    for tr, te in folds:
        train_seqs = [seqs[i] for i in tr]
        train_labels = labels[tr]
        test_seqs = [seqs[i] for i in te]
        model = _make_model(method)
        model.fit(train_seqs, train_labels)
        proba = np.asarray(model.predict_proba(test_seqs)).reshape(-1)
        pooled_true.append(labels[te])
        pooled_score.append(proba)
    return np.concatenate(pooled_true), np.concatenate(pooled_score)


# ---------------------------------------------------------------------------
# Plotting (Agg backend, PNG only)
# ---------------------------------------------------------------------------
def _plot_roc(per_method_scores, out_path):
    """ROC curve per method for one embedder (gemma by default)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for method, (yt, ys) in per_method_scores.items():
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        auc = roc_auc_score(yt, ys)
        ax.plot(fpr, tpr, label="%s (AUC=%.3f)" % (method, auc))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Multi-turn jailbreak detection ROC (gemma embedder)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_auc_bar(embedders_block, out_path):
    """Grouped AUC bar chart: method (x) x embedder (series)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = C.METHODS
    embs = list(embedders_block.keys())
    x = np.arange(len(methods))
    width = 0.8 / max(1, len(embs))

    fig, ax = plt.subplots(figsize=(7, 5))
    for j, emb in enumerate(embs):
        vals = []
        for m in methods:
            cell = embedders_block.get(emb, {}).get(m, {})
            v = cell.get("auc", float("nan")) if isinstance(cell, dict) else float("nan")
            vals.append(v if v == v else 0.0)  # NaN -> 0 bar
        ax.bar(x + j * width, vals, width, label=emb)
    ax.set_xticks(x + width * (len(embs) - 1) / 2)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Pooled out-of-fold AUC")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
    ax.set_title("AUC by method x embedder")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_trajectory(examples, out_path):
    """Per-turn running-risk trajectory: one attack vs one benign conversation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for ex in examples:
        traj = ex.get("gru_risk_trajectory") or []
        if not traj:
            continue
        turns = np.arange(1, len(traj) + 1)
        lbl = "attack" if ex.get("label") == 1 else "benign"
        ax.plot(turns, traj, marker="o", label="%s (%s)" % (lbl, ex.get("source", "?")))
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="threshold=0.5")
    ax.set_xlabel("User turn index")
    ax.set_ylabel("GRU running P(attack)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-turn risk trajectory (escalation vs flat)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _embedder_list():
    sel = (C.EMBEDDERS or "both").strip().lower()
    if sel == "both":
        return ["gemma", "minilm"]
    if sel in ("gemma", "minilm"):
        return [sel]
    return ["gemma", "minilm"]


def _condition_list():
    sel = (os.environ.get("MJ_CONDITION") or "both").strip().lower()
    if sel in ("easy", "hard"):
        return [sel]
    return ["easy", "hard"]


def _run_condition(condition, data, embed, models):
    """Load one condition (easy|hard), embed, CV the four methods per embedder,
    build demo trajectories + per-condition plots. Returns a result block."""
    ds = data.load_dataset(condition=condition)
    convs = ds["conversations"]
    labels = list(ds["labels"])
    groups = list(ds["groups"])
    sources = ds.get("sources", ["?"] * len(convs))
    n_pos = int(sum(1 for y in labels if y == 1))
    n_neg = int(sum(1 for y in labels if y == 0))
    print("[%s] conversations=%d  pos=%d  neg=%d" % (condition, len(convs), n_pos, n_neg))

    conf = data.length_confound_report(convs, labels)
    print("[%s/confound] turncount_auc=%.3f totalchar_auc=%.3f turns_pos=%.2f turns_neg=%.2f"
          % (condition, conf.get("turncount_auc", float("nan")),
             conf.get("totalchar_auc", float("nan")),
             conf.get("turncount_pos_mean", float("nan")),
             conf.get("turncount_neg_mean", float("nan"))))

    embedders_block = {}
    roc_scores_gemma = {}
    for emb_name in _embedder_list():
        # Cache keyed by (condition, embedder) -- easy/hard have DIFFERENT convs.
        cache = C.ARTIFACTS / ("seqs_%s_%s.npz" % (condition, emb_name))
        try:
            seqs = embed.load_or_build(convs, emb_name, cache)
            dim = int(seqs[0].shape[1]) if len(seqs) and seqs[0].ndim == 2 else 0
        except Exception as exc:
            embedders_block[emb_name] = {"error": str(exc)}
            print("[%s/embed:%s] FAILED: %s" % (condition, emb_name, exc))
            continue
        cell_block = {"dim": dim}
        for method in C.METHODS:
            try:
                yt, ys = _cv_pool(seqs, labels, groups, method)
                cell_block[method] = _metrics(yt, ys)
                if emb_name == "gemma":
                    roc_scores_gemma[method] = (yt, ys)
                m = cell_block[method]
                print("[%s/%s/%s] auc=%.3f ci=[%.3f,%.3f] f1=%.3f tpr@fpr10=%.3f"
                      % (condition, emb_name, method, m["auc"], m["auc_ci"][0],
                         m["auc_ci"][1], m["f1"], m["tpr_at_fpr10"]))
            except Exception as exc:
                cell_block[method] = {"error": str(exc)}
                print("[%s/%s/%s] FAILED: %s" % (condition, emb_name, method, exc))
        embedders_block[emb_name] = cell_block

    # Demo trajectories (train one SeqGRU on all of this condition's data).
    examples = []
    demo_emb = "gemma" if "gemma" in _embedder_list() else _embedder_list()[0]
    try:
        cache = C.ARTIFACTS / ("seqs_%s_%s.npz" % (condition, demo_emb))
        demo_seqs = embed.load_or_build(convs, demo_emb, cache)
        gru = models.SeqGRU()
        gru.fit(demo_seqs, np.asarray(labels))
        pos_i = next((i for i, y in enumerate(labels) if y == 1), None)
        neg_i = next((i for i, y in enumerate(labels) if y == 0), None)
        for i in (pos_i, neg_i):
            if i is None:
                continue
            traj = np.asarray(gru.risk_trajectory(demo_seqs[i])).reshape(-1).tolist()
            examples.append({"source": str(sources[i]), "label": int(labels[i]),
                             "turns": list(convs[i]),
                             "gru_risk_trajectory": [float(t) for t in traj]})
    except Exception as exc:
        print("[%s/examples] FAILED: %s" % (condition, exc))

    # Per-condition plots (best-effort).
    plots = []
    roc_png = C.ARTIFACTS / ("roc_%s.png" % condition)
    bar_png = C.ARTIFACTS / ("auc_%s.png" % condition)
    traj_png = C.ARTIFACTS / ("risk_trajectory_%s.png" % condition)
    for fn, png, arg in (("roc", roc_png, roc_scores_gemma),
                         ("bar", bar_png, embedders_block),
                         ("traj", traj_png, examples)):
        try:
            if fn == "roc" and arg:
                _plot_roc(arg, png); plots.append(str(png))
            elif fn == "bar":
                _plot_auc_bar(arg, png); plots.append(str(png))
            elif fn == "traj" and arg:
                _plot_trajectory(arg, png); plots.append(str(png))
        except Exception as exc:
            print("[%s/plot:%s] FAILED: %s" % (condition, fn, exc))

    return {
        "n_pos": n_pos, "n_neg": n_neg,
        "confound": {
            "turncount_auc": float(conf.get("turncount_auc", float("nan"))),
            "totalchar_auc": float(conf.get("totalchar_auc", float("nan"))),
            "turncount_pos_mean": float(conf.get("turncount_pos_mean", float("nan"))),
            "turncount_neg_mean": float(conf.get("turncount_neg_mean", float("nan"))),
        },
        "embedders": embedders_block,
        "examples": examples,
        "plots": plots,
    }


def main():
    # Lazy sibling imports (guarded here, NOT at module top, so import-check passes
    # even while data/embed/models are still stubs).
    from . import data, embed, models

    conditions = {}
    for cond in _condition_list():
        conditions[cond] = _run_condition(cond, data, embed, models)

    results = {
        "gemma_model_id": C.GEMMA_MODEL_ID,
        "gemma_layer": int(C.GEMMA_LAYER),
        "minilm_id": C.MINILM_ID,
        "min_turns": int(C.MIN_USER_TURNS),
        "max_turns": int(C.MAX_USER_TURNS),
        "seed": int(C.SEED),
        "n_folds": int(C.N_FOLDS),
        "judge": None,
        "conditions": conditions,
    }
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
    print("MULTI-TURN JAILBREAK DETECTION  (SCREENING TIER, group-aware CV)")
    print("folds=%d seed=%d min/max turns=%d/%d"
          % (results["n_folds"], results["seed"], results["min_turns"],
             results["max_turns"]))
    for cond, block in results.get("conditions", {}).items():
        c = block["confound"]
        tag = "EASY (attack vs UltraChat benign)" if cond == "easy" \
            else "HARD (full attack vs benign PREFIX -- same style, only escalation differs)"
        print(line)
        print("CONDITION: %s   pos=%d neg=%d" % (tag, block["n_pos"], block["n_neg"]))
        print("confound: turncount_auc=%.3f totalchar_auc=%.3f  (~0.5 => no trivial signal)"
              % (c["turncount_auc"], c["totalchar_auc"]))
        print("%-9s %-15s %7s %-15s %6s %8s"
              % ("embedder", "method", "AUC", "95% CI", "F1", "TPR@10"))
        for emb_name, eb in block["embedders"].items():
            if not isinstance(eb, dict) or "error" in eb:
                print("%-9s [EMBEDDER FAILED]" % emb_name)
                continue
            for method in C.METHODS:
                cell = eb.get(method)
                if not isinstance(cell, dict):
                    continue
                if "error" in cell:
                    print("%-9s %-15s  [FAILED]" % (emb_name, method))
                    continue
                ci = "[%.2f,%.2f]" % (cell["auc_ci"][0], cell["auc_ci"][1])
                print("%-9s %-15s %7.3f %-15s %6.2f %8.2f"
                      % (emb_name, method, cell["auc"], ci, cell["f1"],
                         cell["tpr_at_fpr10"]))
    print(line)
    print("READ: per_turn_max is the STATELESS baseline. If it wins on EASY but the "
          "sequence models (seq_gru/hier_attn) beat it on HARD, that is the lesson: "
          "multi-turn attacks hide in the trajectory that per-turn cannot see.")
    print(line)


if __name__ == "__main__":
    main()
