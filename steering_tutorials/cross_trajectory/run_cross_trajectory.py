"""run_cross_trajectory.py — orchestrator for the CROSS-TRAJECTORY LATENT-
AGGREGATION lesson (the multi-agent capstone of the trajectory-detection trilogy).

A harmful goal is DECOMPOSED into K individually-innocuous sub-tasks and spread
across K separate agents / sessions / trajectories, so no single trajectory
carries the payload. This runner builds the constructed decompositions
(SafeMTData Attack_600) vs benign multi-trajectory samples in two conditions
(easy / hard), embeds each of a sample's K trajectory texts with the reused
multiturn_jailbreak Gemma turn-embedder, and runs group-aware N-fold CV over four
permutation-invariant SET classifiers:

  per_traj_max : the DECOMPOSITION-DEFEATED baseline (MAX over per-trajectory
                 P(attack) -- "does ANY single trajectory look harmful?").
  mean_agg / attn_pool / gnn_agg : set-aggregators that pool the K latents and
                 recover the fractured intent the parts hide.

Pooled out-of-fold predictions are scored with AUC (+ bootstrap 95% CI), F1, ACC
and TPR@FPR<=0.10. The real OOD benchmark (CSTM-Bench) is scored by training each
method on ALL of the HARD main set and predicting the cross-session scenarios.
results.json (schema in the interface contract) is written BEFORE the ASCII
summary; three PNGs are rendered with the matplotlib Agg backend.

Sibling modules (data / models) and the multiturn_jailbreak embedder are imported
lazily INSIDE main() so `python -c "import ...run_cross_trajectory"` succeeds even
while those modules are still stubs. CPU-only; env caps (CT_N_POS / CT_N_NEG /
CT_K / CT_CONDITION / CT_FOLDS) already live in config. Stdout is ASCII only
(Windows cp1252).
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# CV + metric helpers (no sibling-module / model dependency -> import-safe)
# ---------------------------------------------------------------------------
def group_kfold_indices(groups, n_folds, seed):
    """Group-aware K-fold splits. Returns list of (train_idx, test_idx) ndarrays.

    Groups are shuffled with `seed` so folds are deterministic but not ordered by
    the raw group ids; a whole group (an attack query_id / a unique benign id)
    stays inside a single fold, so no target leaks across CV folds. Falls back to
    the number of distinct groups if that is < n_folds; a single group => one fold
    with train==test (degenerate but never crashes).
    """
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n = len(groups)
    n_groups = len(np.unique(groups))
    k = int(min(n_folds, n_groups))
    if k < 2:
        idx = np.arange(n)
        return [(idx, idx)]

    # Inject the seed by remapping each group id to a random rank (GroupKFold is
    # order-deterministic on its own).
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
    """Full metric bundle for one pooled prediction vector.

    Returns {"auc","auc_ci","f1","acc","tpr_at_fpr10"} (schema-verbatim keys).
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


# ---------------------------------------------------------------------------
# Model factory + CV pooling
# ---------------------------------------------------------------------------
def _make_model(name):
    """Instantiate one set-classifier by its stable config key (lazy import)."""
    from . import models

    if name == "per_traj_max":
        return models.PerTrajMax()
    if name == "mean_agg":
        return models.MeanAgg()
    if name == "attn_pool":
        return models.AttnPool()
    if name == "gnn_agg":
        return models.GnnAgg()
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
    """ROC curve per method on the HARD condition (the headline discriminator)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for method, (yt, ys) in per_method_scores.items():
        yt = np.asarray(yt)
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        auc = roc_auc_score(yt, ys)
        ax.plot(fpr, tpr, label="%s (AUC=%.3f)" % (method, auc))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Cross-trajectory detection ROC (HARD, %s embedder)" % C.EMBEDDER)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_auc_bar(conditions_block, out_path):
    """Grouped AUC bar chart: method (x) x condition (series)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = C.METHODS
    conds = list(conditions_block.keys())
    x = np.arange(len(methods))
    width = 0.8 / max(1, len(conds))

    fig, ax = plt.subplots(figsize=(7, 5))
    for j, cond in enumerate(conds):
        vals = []
        for m in methods:
            cell = conditions_block.get(cond, {}).get("methods", {}).get(m, {})
            v = cell.get("auc", float("nan")) if isinstance(cell, dict) else float("nan")
            vals.append(v if v == v else 0.0)  # NaN -> 0 bar
        ax.bar(x + j * width, vals, width, label=cond)
    ax.set_xticks(x + width * (len(conds) - 1) / 2)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Pooled out-of-fold AUC")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
    ax.set_title("AUC by method x condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_ood_bar(ood_methods, out_path):
    """AUC bar chart on the real OOD benchmark (CSTM-Bench)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = C.METHODS
    x = np.arange(len(methods))
    vals = []
    for m in methods:
        cell = ood_methods.get(m, {})
        v = cell.get("auc", float("nan")) if isinstance(cell, dict) else float("nan")
        vals.append(v if v == v else 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, vals, 0.6, color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("OOD AUC (CSTM-Bench)")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
    ax.set_title("Out-of-distribution AUC on CSTM-Bench")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------
def _condition_list():
    sel = (C.CONDITION or "both").strip().lower()
    if sel in ("easy", "hard"):
        return [sel]
    return ["easy", "hard"]


def _embed_samples(MJE, samples, tag):
    """Embed every sample's K trajectory texts -> list of [K, dim] float32.

    Reuses the multiturn_jailbreak ragged .npz cache (one sample == one
    "conversation" of K trajectory strings). Cached per (tag/condition, embedder).
    """
    cache = C.ARTIFACTS / ("traj_%s_%s.npz" % (tag, C.EMBEDDER))
    return MJE.load_or_build(list(samples), C.EMBEDDER, cache)


def _per_traj_scores(ptm_model, seq):
    """Per-trajectory P(attack) from the per_traj_max model.

    API-agnostic: score each single trajectory as a 1-element SET; for per_traj_max
    (MAX over the set) that returns exactly that trajectory's P(attack). Shows the
    individual trajectories look benign while the aggregate flags the attack.
    """
    out = []
    for j in range(int(seq.shape[0])):
        one = np.asarray(seq[j:j + 1], dtype=np.float32)
        p = float(np.asarray(ptm_model.predict_proba([one])).reshape(-1)[0])
        out.append(p)
    return out


def _run_condition(cond, data, models, MJE):
    """Load one condition (easy|hard), embed, CV the four methods, return a block
    plus the per-method (y_true, y_score) for ROC plotting."""
    ds = data.load_dataset(condition=cond)
    samples = ds["samples"]
    labels = list(ds["labels"])
    groups = list(ds["groups"])
    n_pos = int(sum(1 for y in labels if y == 1))
    n_neg = int(sum(1 for y in labels if y == 0))
    print("[%s] samples=%d  pos=%d  neg=%d" % (cond, len(samples), n_pos, n_neg))

    try:
        conf = data.confound_report(samples, labels)
    except Exception as exc:
        conf = {"error": str(exc)}
        print("[%s/confound] FAILED: %s" % (cond, exc))
    if isinstance(conf, dict) and "error" not in conf:
        print("[%s/confound] kcount_auc=%.3f totalchar_auc=%.3f k_pos=%.2f k_neg=%.2f"
              % (cond, conf.get("kcount_auc", float("nan")),
                 conf.get("totalchar_auc", float("nan")),
                 conf.get("k_pos_mean", float("nan")),
                 conf.get("k_neg_mean", float("nan"))))

    methods_block = {}
    roc_scores = {}
    try:
        seqs = _embed_samples(MJE, samples, cond)
    except Exception as exc:
        print("[%s/embed] FAILED: %s" % (cond, exc))
        seqs = None

    if seqs is not None:
        for method in C.METHODS:
            try:
                yt, ys = _cv_pool(seqs, labels, groups, method)
                methods_block[method] = _metrics(yt, ys)
                roc_scores[method] = (yt, ys)
                mm = methods_block[method]
                print("[%s/%s] auc=%.3f ci=[%.3f,%.3f] f1=%.3f tpr@fpr10=%.3f"
                      % (cond, method, mm["auc"], mm["auc_ci"][0],
                         mm["auc_ci"][1], mm["f1"], mm["tpr_at_fpr10"]))
            except Exception as exc:
                methods_block[method] = {"error": str(exc)}
                print("[%s/%s] FAILED: %s" % (cond, method, exc))
    else:
        for method in C.METHODS:
            methods_block[method] = {"error": "embedding failed"}

    block = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "confound": {
            "kcount_auc": float(conf.get("kcount_auc", float("nan"))) if isinstance(conf, dict) else float("nan"),
            "totalchar_auc": float(conf.get("totalchar_auc", float("nan"))) if isinstance(conf, dict) else float("nan"),
            "k_pos_mean": float(conf.get("k_pos_mean", float("nan"))) if isinstance(conf, dict) else float("nan"),
            "k_neg_mean": float(conf.get("k_neg_mean", float("nan"))) if isinstance(conf, dict) else float("nan"),
        },
        "methods": methods_block,
    }
    return block, roc_scores


def _fit_all_on_hard(data, models, MJE):
    """Train every method on ALL of the HARD main set. Returns
    (fitted_models, hard_ds, hard_seqs, hard_labels). Used for OOD + examples."""
    ds = data.load_dataset(condition="hard")
    samples = ds["samples"]
    labels = np.asarray(ds["labels"])
    seqs = _embed_samples(MJE, samples, "hard")
    fitted = {}
    for method in C.METHODS:
        try:
            model = _make_model(method)
            model.fit(seqs, labels)
            fitted[method] = model
        except Exception as exc:
            fitted[method] = None
            print("[ood/fit:%s] FAILED: %s" % (method, exc))
    return fitted, ds, seqs, labels


def _run_ood(data, models, MJE, fitted):
    """Predict the CSTM-Bench cross-session scenarios with the hard-trained models."""
    ds = data.load_ood_cstm()
    samples = ds["samples"]
    labels = np.asarray(ds["labels"])
    n_attack = int(np.sum(labels == 1))
    n_benign = int(np.sum(labels == 0))
    print("[ood] scenarios=%d  attack=%d  benign=%d" % (len(samples), n_attack, n_benign))

    ood_methods = {}
    try:
        seqs = _embed_samples(MJE, samples, "ood")
    except Exception as exc:
        print("[ood/embed] FAILED: %s" % exc)
        seqs = None

    if seqs is not None:
        for method in C.METHODS:
            model = fitted.get(method)
            if model is None:
                ood_methods[method] = {"error": "model not fitted"}
                continue
            try:
                scores = np.asarray(model.predict_proba(seqs)).reshape(-1)
                ood_methods[method] = _metrics(labels, scores)
                mm = ood_methods[method]
                print("[ood/%s] auc=%.3f ci=[%.3f,%.3f] f1=%.3f tpr@fpr10=%.3f"
                      % (method, mm["auc"], mm["auc_ci"][0], mm["auc_ci"][1],
                         mm["f1"], mm["tpr_at_fpr10"]))
            except Exception as exc:
                ood_methods[method] = {"error": str(exc)}
                print("[ood/%s] FAILED: %s" % (method, exc))
    else:
        for method in C.METHODS:
            ood_methods[method] = {"error": "embedding failed"}

    return {
        "dataset": "intrinsec-ai/cstm-bench",
        "n_attack": n_attack,
        "n_benign": n_benign,
        "methods": ood_methods,
    }


def _build_examples(fitted, hard_ds, hard_seqs, hard_labels):
    """1 attack + 1 benign HARD sample: every method's aggregate P(attack) AND the
    per-trajectory P(attack) from per_traj_max (individual trajectories look benign;
    the aggregate flags the fractured intent)."""
    samples = hard_ds["samples"]
    sources = hard_ds.get("sources", ["?"] * len(samples))
    labels = list(hard_labels)
    ptm = fitted.get("per_traj_max")

    examples = []
    pos_i = next((i for i, y in enumerate(labels) if y == 1), None)
    neg_i = next((i for i, y in enumerate(labels) if y == 0), None)
    for i in (pos_i, neg_i):
        if i is None:
            continue
        seq = np.asarray(hard_seqs[i], dtype=np.float32)
        method_proba = {}
        for method in C.METHODS:
            model = fitted.get(method)
            if model is None:
                continue
            try:
                method_proba[method] = float(
                    np.asarray(model.predict_proba([seq])).reshape(-1)[0])
            except Exception as exc:
                method_proba[method] = None
                print("[examples/%s] FAILED: %s" % (method, exc))
        per_traj = None
        if ptm is not None:
            try:
                per_traj = _per_traj_scores(ptm, seq)
            except Exception as exc:
                print("[examples/per_traj] FAILED: %s" % exc)
        examples.append({
            "source": str(sources[i]) if i < len(sources) else "?",
            "label": int(labels[i]),
            "trajectories": list(samples[i]),
            "method_proba": method_proba,
            "per_traj_attack_proba": per_traj,
        })
    return examples


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main():
    # Lazy sibling imports (guarded here, NOT at module top, so import-check passes
    # even while data/models are still stubs).
    from . import data, models
    from steering_tutorials.multiturn_jailbreak import embed as MJE

    C.ARTIFACTS.mkdir(exist_ok=True)

    # --- 1. per-condition group-aware CV ------------------------------------
    conditions = {}
    hard_roc = {}
    for cond in _condition_list():
        try:
            block, roc = _run_condition(cond, data, models, MJE)
            conditions[cond] = block
            if cond == "hard":
                hard_roc = roc
        except Exception as exc:
            conditions[cond] = {"error": str(exc)}
            print("[%s] CONDITION FAILED: %s" % (cond, exc))

    # --- 2. OOD (train on ALL hard, predict CSTM-Bench) + 3. examples -------
    ood = {"dataset": "intrinsec-ai/cstm-bench", "n_attack": 0, "n_benign": 0,
           "methods": {}}
    examples = []
    try:
        fitted, hard_ds, hard_seqs, hard_labels = _fit_all_on_hard(data, models, MJE)
        try:
            ood = _run_ood(data, models, MJE, fitted)
        except Exception as exc:
            ood["error"] = str(exc)
            print("[ood] FAILED: %s" % exc)
        try:
            examples = _build_examples(fitted, hard_ds, hard_seqs, hard_labels)
        except Exception as exc:
            print("[examples] FAILED: %s" % exc)
    except Exception as exc:
        print("[ood/examples/setup] FAILED: %s" % exc)

    # --- 4. plots (best-effort, each wrapped) -------------------------------
    plots = []
    for tag, png, fn in (
        ("roc", C.ROC_PNG, lambda: _plot_roc(hard_roc, C.ROC_PNG)),
        ("bar", C.BAR_PNG, lambda: _plot_auc_bar(conditions, C.BAR_PNG)),
        ("ood", C.OOD_PNG, lambda: _plot_ood_bar(ood.get("methods", {}), C.OOD_PNG)),
    ):
        try:
            if tag == "roc" and not hard_roc:
                continue
            fn()
            plots.append(str(png))
        except Exception as exc:
            print("[plot:%s] FAILED: %s" % (tag, exc))

    # --- 5. results.json (schema-verbatim), written BEFORE the summary ------
    results = {
        "embedder": str(C.EMBEDDER),
        "gemma_layer": int(C.GEMMA_LAYER),
        "k": int(C.K_TRAJ),
        "n_folds": int(C.N_FOLDS),
        "seed": int(C.SEED),
        "judge": None,
        "conditions": conditions,
        "ood": ood,
        "examples": examples,
        "plots": plots,
    }
    with open(C.RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("[write] %s" % C.RESULTS_PATH)

    _print_summary(results)
    return results


def _print_summary(results):
    line = "-" * 78
    print("")
    print(line)
    print("CROSS-TRAJECTORY LATENT AGGREGATION  (SCREENING TIER, group-aware CV)")
    print("embedder=%s gemma_layer=%d k=%d folds=%d seed=%d"
          % (results["embedder"], results["gemma_layer"], results["k"],
             results["n_folds"], results["seed"]))
    for cond, block in results.get("conditions", {}).items():
        if not isinstance(block, dict) or "methods" not in block:
            print(line)
            print("CONDITION: %s   [FAILED]" % cond)
            continue
        c = block.get("confound", {})
        tag = "EASY (attack decomposition vs UltraChat benign)" if cond == "easy" \
            else "HARD (full decomposition vs INCOMPLETE same-style lead-up)"
        print(line)
        print("CONDITION: %s   pos=%d neg=%d" % (tag, block["n_pos"], block["n_neg"]))
        print("confound: kcount_auc=%.3f totalchar_auc=%.3f  (~0.5 => no trivial tell)"
              % (c.get("kcount_auc", float("nan")), c.get("totalchar_auc", float("nan"))))
        print("%-14s %7s %-15s %6s %6s %8s"
              % ("method", "AUC", "95% CI", "F1", "ACC", "TPR@10"))
        for method in C.METHODS:
            cell = block["methods"].get(method)
            if not isinstance(cell, dict):
                continue
            if "error" in cell:
                print("%-14s  [FAILED]" % method)
                continue
            ci = "[%.2f,%.2f]" % (cell["auc_ci"][0], cell["auc_ci"][1])
            print("%-14s %7.3f %-15s %6.2f %6.2f %8.2f"
                  % (method, cell["auc"], ci, cell["f1"], cell["acc"],
                     cell["tpr_at_fpr10"]))

    ood = results.get("ood", {})
    print(line)
    print("OOD: %s   attack=%d benign=%d  (real cross-session benchmark)"
          % (ood.get("dataset", "?"), ood.get("n_attack", 0), ood.get("n_benign", 0)))
    print("%-14s %7s %-15s %6s %6s %8s"
          % ("method", "AUC", "95% CI", "F1", "ACC", "TPR@10"))
    for method in C.METHODS:
        cell = ood.get("methods", {}).get(method)
        if not isinstance(cell, dict):
            continue
        if "error" in cell:
            print("%-14s  [FAILED]" % method)
            continue
        ci = "[%.2f,%.2f]" % (cell["auc_ci"][0], cell["auc_ci"][1])
        print("%-14s %7.3f %-15s %6.2f %6.2f %8.2f"
              % (method, cell["auc"], ci, cell["f1"], cell["acc"],
                 cell["tpr_at_fpr10"]))
    print(line)
    print("READ: per_traj_max is the DECOMPOSITION-DEFEATED baseline. FALSIFIER -- if "
          "the set-aggregators (attn_pool/gnn_agg) do NOT beat per_traj_max on HARD, "
          "'aggregation recovers the fractured intent' is FALSE.")
    print(line)


if __name__ == "__main__":
    main()
