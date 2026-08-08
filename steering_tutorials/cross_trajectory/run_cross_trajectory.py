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
and TPR@FPR<=0.10, and -- the number that is actually claimable -- the MARGIN over
``max{confound bars, per_traj_max}`` from ``common.confound``. The real OOD
benchmark (CSTM-Bench) is scored by training each method on ALL of the HARD main
set and predicting the cross-session scenarios. results.json is written BEFORE the
ASCII summary; three PNGs are rendered with the matplotlib Agg backend.

Three defects the 2026-08 audit found are fixed here:
  * ``RESULTS_PATH`` is now per-embedder, so running the gemma arm no longer
    OVERWRITES the minilm headline (the README's side-by-side table previously
    could not be produced by this code in one run or two).
  * embeddings go through ``embed_ct``, which keys its cache on a SHA-256 of the
    texts (+ seed / K / n / dim / embedder) and ASSERTS on load, instead of the
    old ``(condition, embedder)`` key validated by row count alone.
  * the falsifier is registered against the CONFOUND BAR, not against
    ``per_traj_max`` -- which itself sits BELOW the hard length bar.

Sibling modules (data / models / embed_ct) are imported lazily INSIDE main() so
`python -c "import ...run_cross_trajectory"` succeeds even while those modules are
still stubs. CPU-only apart from the embedder; env caps (CT_N_POS / CT_N_NEG /
CT_K / CT_CONDITION / CT_FOLDS / CT_EMBED / CT_OOD_SELECT) live in config. Stdout
is ASCII only (Windows cp1252).
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


def _embed_samples(samples, tag):
    """Embed every sample's K trajectory texts -> (list of [K, dim] float32, meta).

    Delegates to ``embed_ct.load_or_build``, which keys the cache on a SHA-256 of
    the CONCATENATED TEXTS plus seed / K / n / dim / embedder and **asserts** all of
    them on load. The previous version keyed on ``(condition, embedder)`` only and
    the underlying loader validated the ROW COUNT alone, so changing ``CT_SEED``
    reshuffled which attacks were positive, kept the count identical, and silently
    scored cached vectors against NEW LABELS (CLAUDE.md 18.8). A mismatch now raises.
    """
    from . import embed_ct

    return embed_ct.load_or_build(list(samples), C.EMBEDDER, tag, C.SEED, C.K_TRAJ)


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


def _run_condition(cond, data, models):
    """Load one condition (easy|hard), embed, CV the four methods, return a block
    plus the per-method (y_true, y_score) for ROC plotting.

    Every method AUC is accompanied by its MARGIN OVER THE BINDING BAR --
    ``max{confound bars, per_traj_max}`` -- because that margin, not the raw AUC,
    is the only number this lesson may headline (CLAUDE.md 17 rule 7).
    """
    ds = data.load_dataset(condition=cond)
    samples = ds["samples"]
    labels = list(ds["labels"])
    groups = list(ds["groups"])
    meta = ds.get("meta", {})
    n_pos = int(sum(1 for y in labels if y == 1))
    n_neg = int(sum(1 for y in labels if y == 0))
    print("[%s] samples=%d  pos=%d  neg=%d  distinct_groups=%d"
          % (cond, len(samples), n_pos, n_neg, meta.get("n_distinct_groups", -1)))

    try:
        conf = data.confound_report(samples, labels)
        print(data.format_report(conf))
    except Exception as exc:
        conf = {"error": str(exc), "worst_name": "none", "worst_auc": 0.5}
        print("[%s/confound] FAILED: %s" % (cond, exc))

    methods_block = {}
    roc_scores = {}
    embed_meta = {}
    try:
        seqs, embed_meta = _embed_samples(samples, cond)
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

    # Margin over max{confound bar, per_traj_max}. The baseline is priced INTO the
    # bar, so a method never gets credit for beating whichever is weaker.
    baseline_cell = methods_block.get("per_traj_max") or {}
    baseline_auc = baseline_cell.get("auc") if isinstance(baseline_cell, dict) else None
    for method, cell in methods_block.items():
        if not isinstance(cell, dict) or "auc" not in cell:
            continue
        bl = None if method == "per_traj_max" else baseline_auc
        cell["margin"] = data.margin_over_bar(cell["auc"], conf, baseline_auc=bl)

    block = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "data_meta": meta,
        "embed_meta": embed_meta,
        "confound": conf,
        "methods": methods_block,
    }
    return block, roc_scores


def _fit_all_on_hard(data, models):
    """Train every method on ALL of the HARD main set. Returns
    (fitted_models, hard_ds, hard_seqs, hard_labels). Used for OOD + examples."""
    ds = data.load_dataset(condition="hard")
    samples = ds["samples"]
    labels = np.asarray(ds["labels"])
    seqs, _ = _embed_samples(samples, "hard")
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


def _degeneracy(scores):
    """Is this prediction vector CONSTANT / all-one-side? A different failure from
    'near chance', and the one the CSTM-Bench arm actually exhibited (mean_agg came
    back AUC 0.500 with CI [0.500,0.500] -- a constant, not a coin flip)."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    if s.size == 0:
        return {"n_distinct": 0, "is_constant": True, "pred_positive_frac": float("nan")}
    return {
        "n_distinct": int(np.unique(np.round(s, 6)).size),
        "is_constant": bool(np.unique(np.round(s, 6)).size <= 1),
        "std": float(np.std(s)),
        "pred_positive_frac": float(np.mean(s >= 0.5)),
    }


def _run_ood(data, models, fitted):
    """Predict the CSTM-Bench cross-session scenarios with the hard-trained models."""
    ds = data.load_ood_cstm()
    samples = ds["samples"]
    labels = np.asarray(ds["labels"])
    meta = ds.get("meta", {})
    n_attack = int(np.sum(labels == 1))
    n_benign = int(np.sum(labels == 0))
    print("[ood] scenarios=%d  attack=%d  benign=%d  selection=%s (discarded %.0f%% of sessions)"
          % (len(samples), n_attack, n_benign, meta.get("selection", "?"),
             100.0 * float(meta.get("sessions_discarded_frac", 0.0))))

    ood_methods = {}
    embed_meta = {}
    try:
        seqs, embed_meta = _embed_samples(samples, "ood")
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
                ood_methods[method]["degeneracy"] = _degeneracy(scores)
                # How far outside the TRAIN feature range did this OOD set land?
                clip = getattr(model, "last_clip_frac", None)
                if clip is not None:
                    ood_methods[method]["feature_clip_frac"] = float(clip)
                mm = ood_methods[method]
                print("[ood/%s] auc=%.3f ci=[%.3f,%.3f] f1=%.3f tpr@fpr10=%.3f "
                      "distinct_scores=%d pred_pos_frac=%.2f%s"
                      % (method, mm["auc"], mm["auc_ci"][0], mm["auc_ci"][1],
                         mm["f1"], mm["tpr_at_fpr10"],
                         mm["degeneracy"]["n_distinct"],
                         mm["degeneracy"]["pred_positive_frac"],
                         ("" if clip is None else "  clipped_features=%.3f" % clip)))
                if mm["degeneracy"]["is_constant"]:
                    print("[ood/%s] WARNING: prediction is CONSTANT -- report this as "
                          "SATURATED/DEGENERATE under shift, not as 'near chance'." % method)
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
        "selection_meta": meta,
        "embed_meta": embed_meta,
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

    C.ARTIFACTS.mkdir(exist_ok=True)

    # --- 1. per-condition group-aware CV ------------------------------------
    conditions = {}
    hard_roc = {}
    for cond in _condition_list():
        try:
            block, roc = _run_condition(cond, data, models)
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
        fitted, hard_ds, hard_seqs, hard_labels = _fit_all_on_hard(data, models)
        try:
            ood = _run_ood(data, models, fitted)
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
        "embedder_compliance": (
            "COMPLIANT: google/embeddinggemma-300m, the encoder CLAUDE.md 17 mandates"
            if C.EMBEDDER == "embeddinggemma" else
            "NON-COMPLIANT SUBSTITUTE: CLAUDE.md 17 mandates google/embeddinggemma-300m, "
            "whose weights ARE on disk. %r was chosen for speed/comparison, not necessity."
            % C.EMBEDDER),
        "gemma_layer": int(C.GEMMA_LAYER),
        "attack_configs": list(C.ATTACK_CONFIGS),
        "k": int(C.K_TRAJ),
        "n_folds": int(C.N_FOLDS),
        "seed": int(C.SEED),
        "ood_selection": str(C.OOD_SELECT),
        "ood_k": int(C.OOD_K),
        "feature_clip": float(C.FEATURE_CLIP),
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
    print("embedder=%s gemma_layer=%d k=%d folds=%d seed=%d configs=%s"
          % (results["embedder"], results["gemma_layer"], results["k"],
             results["n_folds"], results["seed"], ",".join(results["attack_configs"])))
    print("embedder note: %s" % results.get("embedder_compliance", ""))
    for cond, block in results.get("conditions", {}).items():
        if not isinstance(block, dict) or "methods" not in block:
            print(line)
            print("CONDITION: %s   [FAILED]" % cond)
            continue
        c = block.get("confound", {}) or {}
        dm = block.get("data_meta", {}) or {}
        tag = "EASY (attack decomposition vs UltraChat benign)" if cond == "easy" \
            else "HARD (full decomposition vs INCOMPLETE same-style lead-up)"
        print(line)
        print("CONDITION: %s" % tag)
        print("  n: pos=%d neg=%d   DISTINCT GROUPS: %d (pos=%d neg=%d)   >=500/class: %s"
              % (block["n_pos"], block["n_neg"], dm.get("n_distinct_groups", -1),
                 dm.get("n_distinct_groups_pos", -1), dm.get("n_distinct_groups_neg", -1),
                 "YES" if dm.get("meets_500_floor") else "NO"))
        for name in ("length", "count", "content", "shuffle"):
            b = c.get(name)
            if isinstance(b, dict) and "auc" in b:
                print("  confound %-8s raw=%.4f  DIRECTIONLESS=%.4f"
                      % (name, b["auc_raw"], b["auc"]))
        print("  BINDING BAR: %s = %.4f  (a method must beat THIS, not 0.5)"
              % (c.get("worst_name", "none"), c.get("worst_auc", 0.5)))
        print("%-14s %7s %-15s %6s %6s %8s %9s %s"
              % ("method", "AUC", "95% CI", "F1", "ACC", "TPR@10", "MARGIN", "over"))
        for method in C.METHODS:
            cell = block["methods"].get(method)
            if not isinstance(cell, dict):
                continue
            if "error" in cell:
                print("%-14s  [FAILED]" % method)
                continue
            ci = "[%.2f,%.2f]" % (cell["auc_ci"][0], cell["auc_ci"][1])
            mg = cell.get("margin") or {}
            print("%-14s %7.3f %-15s %6.2f %6.2f %8.2f %+9.4f %s=%.3f"
                  % (method, cell["auc"], ci, cell["f1"], cell["acc"],
                     cell["tpr_at_fpr10"], mg.get("margin", float("nan")),
                     mg.get("binding_bar_name", "?"), mg.get("binding_bar", float("nan"))))

    ood = results.get("ood", {})
    sm = ood.get("selection_meta", {}) or {}
    print(line)
    print("OOD: %s   attack=%d benign=%d  (real cross-session benchmark)"
          % (ood.get("dataset", "?"), ood.get("n_attack", 0), ood.get("n_benign", 0)))
    print("  session selection=%s k=%s ; scenarios carry %.1f sessions on average, "
          "%.0f%% DISCARDED"
          % (sm.get("selection", "?"), sm.get("ood_k", "?"),
             sm.get("sessions_available_mean", float("nan")),
             100.0 * float(sm.get("sessions_discarded_frac", 0.0))))
    print("%-14s %7s %-15s %6s %6s %8s %9s"
          % ("method", "AUC", "95% CI", "F1", "ACC", "TPR@10", "distinct"))
    for method in C.METHODS:
        cell = ood.get("methods", {}).get(method)
        if not isinstance(cell, dict):
            continue
        if "error" in cell:
            print("%-14s  [FAILED]" % method)
            continue
        ci = "[%.2f,%.2f]" % (cell["auc_ci"][0], cell["auc_ci"][1])
        deg = cell.get("degeneracy", {}) or {}
        print("%-14s %7.3f %-15s %6.2f %6.2f %8.2f %9s"
              % (method, cell["auc"], ci, cell["f1"], cell["acc"],
                 cell["tpr_at_fpr10"],
                 ("CONSTANT" if deg.get("is_constant") else deg.get("n_distinct", "?"))))
    print(line)
    # The falsifier, registered against the CONFOUND BAR -- not against per_traj_max.
    # Under the old (baseline-only) form, a method could "pass" while sitting below a
    # length shortcut. per_traj_max itself is ALREADY below the 0.704 hard bar.
    hard = (results.get("conditions", {}) or {}).get("hard", {}) or {}
    hconf = hard.get("confound", {}) or {}
    bar = float(hconf.get("worst_auc", float("nan")))
    bar_name = hconf.get("worst_name", "?")
    print("PRE-REGISTERED FALSIFIER (confound-bar form): for each aggregator, the claim")
    print("  'aggregation recovers the fractured intent' is FALSE if")
    print("      AUC(aggregator) <= max(confound bars, AUC(per_traj_max))")
    print("  on HARD. Raw AUC over 0.5 is NOT the test; the binding bar is.")
    if bar == bar:
        print("  HARD binding bar this run: %s = %.4f" % (bar_name, bar))
        for method in C.METHODS:
            cell = (hard.get("methods", {}) or {}).get(method)
            if not isinstance(cell, dict) or "margin" not in cell:
                continue
            mg = cell["margin"]
            print("    %-14s AUC=%.4f  margin %+0.4f over %s=%.4f  ->  %s"
                  % (method, cell["auc"], mg["margin"], mg["binding_bar_name"],
                     mg["binding_bar"], "CLEARS" if mg["clears"] else "FAILS THE BAR"))
    print("  No reclassification after the fact, no moving to the EASY condition.")
    print(line)


if __name__ == "__main__":
    main()
