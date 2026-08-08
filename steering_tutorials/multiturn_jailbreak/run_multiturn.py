"""run_multiturn.py -- orchestrator for the multi-turn jailbreak DETECTION lesson.

For each CONDITION (easy | hard) it loads the conversations, runs the SHARED confound
audit (`common.confound` via `data.confound_audit` -- length, count, content/TF-IDF and
a label-shuffle leakage control), embeds each turn with each selected embedder
(embgemma | gemma | minilm), and runs group-aware N-fold CV over five sequence
classifiers: the stateless `per_turn_max` baseline, the `last_turn_only` CONTROL,
`trajectory_mlp`, `seq_gru`, and `hier_attn`. Out-of-fold scores are pooled per
(embedder x method) and scored with AUC (+ bootstrap 95% CI), per-FOLD AUC mean/CI,
F1, ACC and TPR@FPR<=0.10, and each method's MARGIN over the BINDING confound bar.

Three arms beyond the main grid:
  * SHUFFLED TURNS -- the same CV with each conversation's turn order permuted. Runs
    on the CACHED embeddings (CPU, no GPU). The direct test of whether ORDER carries
    the signal, which the lesson asserted and never measured.
  * OOD -- fit on the whole HARD set, score zero-shot on `intrinsec-ai/cstm-bench`.
  * PRE-REGISTRATION -- config.PREREGISTRATION is printed BEFORE any number exists.

`results.json` is written BEFORE the ASCII summary print and stamps the ACHIEVED
config (counts, distinct groups, pool ceiling, data fingerprint, git SHA, timestamp),
not the requested one -- section 18.8.

The sibling modules (data / embed / models) are imported lazily INSIDE main() so
`python -c "import ...run_multiturn"` succeeds even while those modules are stubs.
CPU-only apart from the embedder; env caps live in config. Stdout is ASCII only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

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


def bootstrap_auc_ci(y_true, y_score, n=None, seed=None):
    """Percentile bootstrap 95% CI on ROC-AUC. Returns (auc, lo, hi).

    NOTE this is a SAMPLING-noise CI on one fixed fit, not a seed-variance CI. The
    lesson is single-seed (C.SEED), so nothing here speaks to training-run variance.
    """
    from sklearn.metrics import roc_auc_score

    n = int(C.BOOTSTRAP if n is None else n)
    seed = int(C.SEED if seed is None else seed)
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


def _per_fold_auc(fold_scores):
    """AUC computed WITHIN each fold, then mean + 95% normal CI across folds.

    The pooled AUC concatenates raw `predict_proba` from N independently-fit models,
    so between-fold calibration drift leaks into it. This is the cleaner companion
    number; both are reported.
    """
    from sklearn.metrics import roc_auc_score

    vals = []
    for yt, ys in fold_scores:
        yt = np.asarray(yt)
        if len(np.unique(yt)) < 2:
            continue
        try:
            vals.append(float(roc_auc_score(yt, np.asarray(ys))))
        except ValueError:
            continue
    if not vals:
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")],
                "per_fold": [], "n_folds": 0}
    a = np.asarray(vals, dtype=float)
    mean = float(a.mean())
    if len(a) > 1:
        se = float(a.std(ddof=1) / np.sqrt(len(a)))
    else:
        se = 0.0
    return {"mean": mean, "ci": [mean - 1.96 * se, mean + 1.96 * se],
            "per_fold": [float(v) for v in a], "n_folds": int(len(a))}


def _metrics(y_true, y_score, fold_scores=None):
    """Full metric bundle for one pooled out-of-fold prediction vector."""
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc, lo, hi = bootstrap_auc_ci(y_true, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    out = {
        "auc": float(auc),
        "auc_ci": [float(lo), float(hi)],
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "tpr_at_fpr10": _tpr_at_fpr10(y_true, y_score),
    }
    if fold_scores is not None:
        out["auc_per_fold"] = _per_fold_auc(fold_scores)
    return out


def _make_model(name):
    """Instantiate one classifier by its stable config key (imported lazily)."""
    from . import models

    table = {
        "per_turn_max": models.PerTurnMaxProbe,
        "last_turn_only": models.LastTurnOnly,
        "trajectory_mlp": models.TrajectoryMLP,
        "seq_gru": models.SeqGRU,
        "hier_attn": models.HierAttn,
    }
    if name not in table:
        raise ValueError("unknown method: %s" % name)
    return table[name]()


def _cv_pool(seqs, labels, groups, method):
    """Group-aware N-fold CV for one method.

    Returns (y_true_pooled, y_score_pooled, per_fold_pairs).
    """
    labels = np.asarray(labels)
    folds = group_kfold_indices(groups, C.N_FOLDS, C.SEED)
    pooled_true, pooled_score, per_fold = [], [], []
    for tr, te in folds:
        train_seqs = [seqs[i] for i in tr]
        test_seqs = [seqs[i] for i in te]
        model = _make_model(method)
        model.fit(train_seqs, labels[tr])
        proba = np.asarray(model.predict_proba(test_seqs)).reshape(-1)
        pooled_true.append(labels[te])
        pooled_score.append(proba)
        per_fold.append((labels[te], proba))
    return np.concatenate(pooled_true), np.concatenate(pooled_score), per_fold


def shuffle_turn_order(seqs, seed):
    """Permute the TURN ORDER inside every sequence (fixed seed).

    The set of turn vectors is unchanged; only their order is destroyed. Any model
    whose AUC survives this was never reading the trajectory.
    """
    rng = np.random.default_rng(int(seed))
    out = []
    for s in seqs:
        a = np.asarray(s, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        out.append(a[rng.permutation(a.shape[0])].copy() if a.shape[0] > 1 else a.copy())
    return out


# ---------------------------------------------------------------------------
# Plotting (Agg backend, PNG only)
# ---------------------------------------------------------------------------
def _plot_roc(per_method_scores, out_path, title):
    """ROC curve per method for the headline embedder."""
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
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_auc_bar(embedders_block, out_path, bar=None):
    """Grouped AUC bar chart: method (x) x embedder (series), with the confound bar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = C.METHODS
    embs = [e for e in embedders_block if isinstance(embedders_block[e], dict)
            and "error" not in embedders_block[e]]
    x = np.arange(len(methods))
    width = 0.8 / max(1, len(embs))

    fig, ax = plt.subplots(figsize=(8, 5))
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
    if bar is not None and bar == bar:
        ax.axhline(float(bar), color="crimson", linestyle="-", alpha=0.8,
                   label="binding confound bar (%.3f)" % float(bar))
    ax.set_title("AUC by method x embedder (bars below the red line claim nothing)")
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
    return C.embedder_list()


def _headline_embedder(names):
    return C.HEADLINE_EMBEDDER if C.HEADLINE_EMBEDDER in names else (names[0] if names else None)


def _condition_list():
    sel = (os.environ.get("MJ_CONDITION") or "both").strip().lower()
    if sel in ("easy", "hard"):
        return [sel]
    return ["easy", "hard"]


def _git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(C.ROOT),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def print_preregistration():
    """Print the pre-registered claim + falsifiers BEFORE any number exists.

    Rule 8: the pre-2026-08 falsifier appeared only in the README that also reported
    the result, so it could be neither checked nor, in principle, failed. Printing it
    from `config.PREREGISTRATION` at launch makes it impossible to drop quietly.
    """
    P = C.PREREGISTRATION
    line = "=" * 78
    print(line)
    print("PRE-REGISTRATION (config.PREREGISTRATION, registered %s, condition=%s)"
          % (P["registered"], P["condition"]))
    print(line)
    print("CLAIM: %s" % P["claim"])
    for k in ("falsifier_1_binding_bar", "falsifier_2_last_turn", "falsifier_3_shuffle"):
        print("%s:" % k.upper())
        print("  %s" % P[k])
    print("TIER: %s" % P["tier"])
    print("NOTE: %s" % P["note"])
    print(line)
    print("")


def _evaluate_falsifiers(cells, bar, shuffle_cells=None):
    """Score the pre-registered falsifiers against the measured cells.

    `cells` maps method -> metric dict for ONE embedder. Returns a verdict dict; a
    missing input yields "not_evaluable", never a silent pass.
    """
    def auc(m):
        c = cells.get(m) or {}
        v = c.get("auc")
        return float(v) if isinstance(v, (int, float)) and v == v else None

    seq = {m: auc(m) for m in ("trajectory_mlp", "seq_gru", "hier_attn")}
    best_name, best = None, None
    for m, v in seq.items():
        if v is not None and (best is None or v > best):
            best_name, best = m, v
    lto = auc("last_turn_only")
    ptm = auc("per_turn_max")

    out = {"best_sequence_model": best_name, "best_sequence_auc": best,
           "last_turn_only_auc": lto, "per_turn_max_auc": ptm,
           "binding_bar": (float(bar) if bar is not None else None)}

    if best is None or bar is None:
        out["falsifier_1_binding_bar"] = "not_evaluable"
    else:
        out["falsifier_1_binding_bar"] = "FALSIFIED" if best <= float(bar) else "survives"
        out["margin_over_bar"] = best - float(bar)

    if best is None or lto is None:
        out["falsifier_2_last_turn"] = "not_evaluable"
    else:
        out["falsifier_2_last_turn"] = "FALSIFIED" if best <= lto + 0.02 else "survives"
        out["margin_over_last_turn"] = best - lto

    if shuffle_cells is None:
        out["falsifier_3_shuffle"] = "not_evaluable"
    else:
        s = (shuffle_cells.get("seq_gru") or {}).get("auc")
        t = auc("seq_gru")
        if not isinstance(s, (int, float)) or s != s or t is None:
            out["falsifier_3_shuffle"] = "not_evaluable"
        else:
            out["shuffled_seq_gru_auc"] = float(s)
            out["order_cost"] = t - float(s)
            out["falsifier_3_shuffle"] = "FALSIFIED" if float(s) >= t - 0.02 else "survives"
    return out


def _run_condition(condition, data, embed, models):
    """Load one condition (easy|hard), audit confounds, embed, CV every method per
    embedder, run the shuffled-turn arm, build demo trajectories + plots."""
    from steering_tutorials.common.confound import format_report, margin_over_bar

    ds = data.load_dataset(condition=condition)
    convs = ds["conversations"]
    labels = list(ds["labels"])
    groups = list(ds["groups"])
    sources = ds.get("sources", ["?"] * len(convs))
    meta = ds.get("meta", {})
    print("[%s] conversations=%d pos=%d neg=%d distinct_groups=%d"
          % (condition, len(convs), meta.get("n_pos_achieved", 0),
             meta.get("n_neg_achieved", 0), meta.get("n_distinct_groups", 0)))

    conf = data.confound_audit(convs, labels)
    print(format_report(conf))
    bar = float(conf.get("worst_auc", 0.5))

    embedders_block = {}
    shuffle_block = {}
    names = _embedder_list()
    head = _headline_embedder(names)
    roc_scores_head = {}
    for emb_name in names:
        # Cache keyed by (condition, embedder); the CONTENT fingerprint inside the
        # npz is what actually validates it (embed.load_or_build).
        cache = C.ARTIFACTS / ("seqs_%s_%s.npz" % (condition, emb_name))
        try:
            seqs = embed.load_or_build(convs, emb_name, cache, seed=C.SEED)
            dim = int(seqs[0].shape[1]) if len(seqs) and seqs[0].ndim == 2 else 0
        except Exception as exc:
            embedders_block[emb_name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            print("[%s/embed:%s] FAILED: %s" % (condition, emb_name, exc))
            continue
        cell_block = {"dim": dim, "cache": str(cache),
                      "headline_eligible": emb_name not in C.LEGACY_EMBEDDERS}
        for method in C.METHODS:
            try:
                yt, ys, folds = _cv_pool(seqs, labels, groups, method)
                cell = _metrics(yt, ys, folds)
                cell["vs_bar"] = margin_over_bar(cell["auc"], conf)
                cell_block[method] = cell
                if emb_name == head:
                    roc_scores_head[method] = (yt, ys)
                print("[%s/%s/%s] auc=%.3f ci=[%.3f,%.3f] fold_mean=%.3f f1=%.3f "
                      "tpr@fpr10=%.3f margin_over_%s=%+.3f"
                      % (condition, emb_name, method, cell["auc"], cell["auc_ci"][0],
                         cell["auc_ci"][1], cell["auc_per_fold"]["mean"], cell["f1"],
                         cell["tpr_at_fpr10"], cell["vs_bar"]["binding_bar_name"],
                         cell["vs_bar"]["margin"]))
            except Exception as exc:
                cell_block[method] = {"error": "%s: %s" % (type(exc).__name__, exc)}
                print("[%s/%s/%s] FAILED: %s" % (condition, emb_name, method, exc))
        embedders_block[emb_name] = cell_block

        # --- SHUFFLED-TURN ARM (cached embeddings, CPU only) ---
        if C.SHUFFLE_TURNS:
            shuffled = shuffle_turn_order(seqs, C.SHUFFLE_SEED)
            sblock = {}
            for method in C.METHODS:
                try:
                    yt, ys, folds = _cv_pool(shuffled, labels, groups, method)
                    sblock[method] = _metrics(yt, ys, folds)
                    delta = sblock[method]["auc"] - (
                        embedders_block[emb_name].get(method, {}).get("auc", float("nan")))
                    print("[%s/%s/%s/SHUFFLED] auc=%.3f (delta vs true order %+.3f)"
                          % (condition, emb_name, method, sblock[method]["auc"], delta))
                except Exception as exc:
                    sblock[method] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            sblock["_note"] = (
                "Turn ORDER permuted (seed=%d); the turn vectors themselves are "
                "unchanged. A method that keeps its AUC here was not reading the "
                "trajectory. Informative for %r; near-vacuous for the permutation-"
                "invariant models." % (C.SHUFFLE_SEED, list(C.ORDER_SENSITIVE_METHODS)))
            shuffle_block[emb_name] = sblock

    # Pre-registered falsifiers, evaluated on the headline embedder.
    falsifiers = None
    if head and isinstance(embedders_block.get(head), dict) and "error" not in embedders_block[head]:
        falsifiers = _evaluate_falsifiers(embedders_block[head], bar,
                                          shuffle_block.get(head))
        print("[%s/falsifiers:%s] %s" % (condition, head, json.dumps(falsifiers)))

    # Demo trajectories (train one SeqGRU on all of this condition's data).
    examples = []
    demo_emb = head
    try:
        cache = C.ARTIFACTS / ("seqs_%s_%s.npz" % (condition, demo_emb))
        demo_seqs = embed.load_or_build(convs, demo_emb, cache, seed=C.SEED)
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
    try:
        if roc_scores_head:
            png = C.ARTIFACTS / ("roc_%s.png" % condition)
            _plot_roc(roc_scores_head, png,
                      "Multi-turn jailbreak detection ROC (%s, %s)" % (condition, head))
            plots.append(str(png))
    except Exception as exc:
        print("[%s/plot:roc] FAILED: %s" % (condition, exc))
    try:
        png = C.ARTIFACTS / ("auc_%s.png" % condition)
        _plot_auc_bar(embedders_block, png, bar=bar)
        plots.append(str(png))
    except Exception as exc:
        print("[%s/plot:bar] FAILED: %s" % (condition, exc))
    try:
        if examples:
            png = C.ARTIFACTS / ("risk_trajectory_%s.png" % condition)
            _plot_trajectory(examples, png)
            plots.append(str(png))
    except Exception as exc:
        print("[%s/plot:traj] FAILED: %s" % (condition, exc))

    return {
        "n_pos": int(meta.get("n_pos_achieved", sum(labels))),
        "n_neg": int(meta.get("n_neg_achieved", len(labels) - sum(labels))),
        "data_meta": meta,
        "confound": conf,
        "binding_bar": bar,
        "binding_bar_name": conf.get("worst_name"),
        "embedders": embedders_block,
        "shuffled_turns": shuffle_block or None,
        "falsifiers": falsifiers,
        "headline_embedder": head,
        "examples": examples,
        "plots": plots,
    }


def _run_ood(data, embed, hard_block):
    """Zero-shot OOD: fit on the WHOLE hard set, score `intrinsec-ai/cstm-bench`.

    No CV here -- the OOD set is the test set, so every model sees all of HARD as
    training data and the benchmark exactly once. Its own confound bar is measured
    too: an OOD number priced against nothing is worth as much as an in-domain one.
    """
    from steering_tutorials.common.confound import format_report, margin_over_bar

    ood = data.load_ood()
    if not ood["labels"] or len(set(ood["labels"])) < 2:
        return {"error": "OOD set has fewer than two classes; nothing to score",
                "meta": ood.get("meta")}
    conf = data.confound_audit(ood["conversations"], ood["labels"])
    print(format_report(conf))
    bar = float(conf.get("worst_auc", 0.5))

    train = data.load_dataset(condition="hard")
    names = _embedder_list()
    out = {"meta": ood["meta"], "confound": conf, "binding_bar": bar,
           "binding_bar_name": conf.get("worst_name"), "embedders": {}}
    for emb_name in names:
        try:
            tr_cache = C.ARTIFACTS / ("seqs_hard_%s.npz" % emb_name)
            oo_cache = C.ARTIFACTS / ("seqs_ood_%s.npz" % emb_name)
            tr_seqs = embed.load_or_build(train["conversations"], emb_name, tr_cache,
                                          seed=C.SEED)
            oo_seqs = embed.load_or_build(ood["conversations"], emb_name, oo_cache,
                                          seed=C.SEED)
        except Exception as exc:
            out["embedders"][emb_name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            print("[ood/embed:%s] FAILED: %s" % (emb_name, exc))
            continue
        block = {}
        for method in C.METHODS:
            try:
                model = _make_model(method)
                model.fit(tr_seqs, np.asarray(train["labels"]))
                proba = np.asarray(model.predict_proba(oo_seqs)).reshape(-1)
                cell = _metrics(np.asarray(ood["labels"]), proba)
                cell["vs_bar"] = margin_over_bar(cell["auc"], conf)
                # The in-domain -> OOD drop, reported as prominently as any win.
                ind = ((hard_block or {}).get("embedders", {})
                       .get(emb_name, {}).get(method, {}) or {}).get("auc")
                if isinstance(ind, (int, float)) and ind == ind:
                    cell["in_domain_auc"] = float(ind)
                    cell["ood_drop"] = float(ind) - cell["auc"]
                block[method] = cell
                print("[ood/%s/%s] auc=%.3f margin_over_%s=%+.3f drop_vs_hard=%s"
                      % (emb_name, method, cell["auc"], cell["vs_bar"]["binding_bar_name"],
                         cell["vs_bar"]["margin"],
                         ("%+.3f" % cell["ood_drop"]) if "ood_drop" in cell else "n/a"))
            except Exception as exc:
                block[method] = {"error": "%s: %s" % (type(exc).__name__, exc)}
                print("[ood/%s/%s] FAILED: %s" % (emb_name, method, exc))
        out["embedders"][emb_name] = block
    return out


def main():
    # Lazy sibling imports (guarded here, NOT at module top, so import-check passes
    # even while data/embed/models are still stubs).
    from . import data, embed, models

    print_preregistration()

    conditions = {}
    for cond in _condition_list():
        conditions[cond] = _run_condition(cond, data, embed, models)

    ood = None
    if C.OOD_ENABLED and "hard" in conditions:
        try:
            ood = _run_ood(data, embed, conditions.get("hard"))
        except Exception as exc:
            ood = {"error": "%s: %s" % (type(exc).__name__, exc)}
            print("[ood] FAILED: %s" % exc)

    results = {
        "lesson": "multiturn_jailbreak",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "run_config": C.run_config(),
        "preregistration": C.PREREGISTRATION,
        # Back-compat top-level keys the older schema exposed.
        "gemma_model_id": C.GEMMA_MODEL_ID,
        "gemma_layer": int(C.GEMMA_LAYER),
        "minilm_id": C.MINILM_ID,
        "min_turns": int(C.MIN_USER_TURNS),
        "max_turns": int(C.MAX_USER_TURNS),
        "seed": int(C.SEED),
        "n_folds": int(C.N_FOLDS),
        "judge": None,
        "judge_note": ("Detection lesson: a classifier reads frozen embeddings, nothing "
                       "generates text, so there is no judge to be off-family (rule 3 N/A)."),
        "conditions": conditions,
        "ood": ood,
    }
    C.ARTIFACTS.mkdir(exist_ok=True)
    with open(C.RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("[write] %s" % C.RESULTS_PATH)
    _print_summary(results)
    return results


def _print_summary(results):
    line = "-" * 78
    rc = results["run_config"]
    print("")
    print(line)
    print("MULTI-TURN JAILBREAK DETECTION  (SCREENING TIER, group-aware CV)")
    print("folds=%d seed=%d turns=%d-%d hard_window=%d embedders=%s headline=%s"
          % (rc["n_folds"], rc["seed"], rc["min_turns"], rc["max_turns"],
             rc["hard_window"], ",".join(rc["embedders"]), rc["headline_embedder"]))
    for cond, block in results.get("conditions", {}).items():
        meta = block.get("data_meta", {})
        tag = ("EASY (attack vs %s benign)" % meta.get("neg_source", "?")) if cond == "easy" \
            else "HARD (attack window vs benign PREFIX of a DISJOINT group half)"
        print(line)
        print("CONDITION: %s" % tag)
        print("  n=%d pos=%d neg=%d distinct_groups=%d  rule1(>=%d/class): %s"
              % (meta.get("n", 0), block["n_pos"], block["n_neg"],
                 meta.get("n_distinct_groups", 0), rc["rule1_floor"],
                 "MET" if meta.get("meets_rule1") else "NOT MET"))
        ceil = meta.get("pool_ceiling")
        if ceil:
            print("  POOL CEILING: pos~%d neg~%d over %d distinct goals "
                  "(rows>=W %d, rows>W %d, W=%d)"
                  % (ceil["pos_ceiling_approx"], ceil["neg_ceiling_approx"],
                     ceil["distinct_groups"], ceil["rows_ge_window"],
                     ceil["rows_gt_window"], ceil["hard_window"]))
        print("  BINDING CONFOUND BAR: %s = %.4f  (methods must beat THIS, not 0.5)"
              % (block.get("binding_bar_name"), block.get("binding_bar", 0.5)))
        print("%-9s %-15s %7s %-15s %8s %6s %8s %9s"
              % ("embedder", "method", "AUC", "95% CI", "fold_mean", "F1", "TPR@10", "vs bar"))
        for emb_name, eb in block["embedders"].items():
            if not isinstance(eb, dict) or "error" in eb:
                print("%-9s [EMBEDDER FAILED]" % emb_name)
                continue
            legacy = "" if eb.get("headline_eligible", True) else "  [LEGACY - not headline]"
            for method in C.METHODS:
                cell = eb.get(method)
                if not isinstance(cell, dict):
                    continue
                if "error" in cell:
                    print("%-9s %-15s  [FAILED]" % (emb_name, method))
                    continue
                ci = "[%.2f,%.2f]" % (cell["auc_ci"][0], cell["auc_ci"][1])
                print("%-9s %-15s %7.3f %-15s %8.3f %6.2f %8.2f %+9.3f%s"
                      % (emb_name, method, cell["auc"], ci,
                         cell.get("auc_per_fold", {}).get("mean", float("nan")),
                         cell["f1"], cell["tpr_at_fpr10"],
                         cell.get("vs_bar", {}).get("margin", float("nan")), legacy))
        fal = block.get("falsifiers")
        if fal:
            print("  PRE-REGISTERED FALSIFIERS (headline embedder %s):"
                  % block.get("headline_embedder"))
            for k in ("falsifier_1_binding_bar", "falsifier_2_last_turn",
                      "falsifier_3_shuffle"):
                print("    %-26s %s" % (k, fal.get(k)))
    ood = results.get("ood")
    if ood and "error" not in ood:
        m = ood.get("meta", {})
        print(line)
        print("OOD (%s splits=%s): n=%d pos=%d neg=%d scenarios=%d  bar %s=%.4f"
              % (m.get("dataset"), ",".join(m.get("splits", [])), m.get("n", 0),
                 m.get("n_pos_achieved", 0), m.get("n_neg_achieved", 0),
                 m.get("n_distinct_groups", 0), ood.get("binding_bar_name"),
                 ood.get("binding_bar", 0.5)))
        print("  %s" % m.get("mhj_status", ""))
        for emb_name, eb in (ood.get("embedders") or {}).items():
            if not isinstance(eb, dict) or "error" in eb:
                print("  %-9s [FAILED]" % emb_name)
                continue
            for method in C.METHODS:
                cell = eb.get(method)
                if isinstance(cell, dict) and "auc" in cell:
                    print("  %-9s %-15s auc=%.3f  drop_vs_hard=%s"
                          % (emb_name, method, cell["auc"],
                             ("%+.3f" % cell["ood_drop"]) if "ood_drop" in cell else "n/a"))
    print(line)
    print("READ: `per_turn_max` is the stateless baseline and `last_turn_only` is the")
    print("control that separates 'the trajectory escalates' from 'the final turn is the")
    print("ask'. A sequence model only supports the lesson's thesis if it beats BOTH the")
    print("binding confound bar AND `last_turn_only`, and loses AUC under SHUFFLED turns.")
    print(line)


if __name__ == "__main__":
    main()
