"""train_large.py — retrain the safety probe on a LARGER, realistic dataset.

Run:  python -m steering_tutorials.hello_world.train_large   (from the repo root)

The original ``train_probe.py`` learns on 200 curated JailbreakBench prompts
(100 harmful / 100 benign). That set is tiny and topically matched, which is
great for a clean demo but weak as evidence. This script re-runs the SAME probe
recipe on **Toxic-Chat** (``lmsys/toxic-chat``, ~1.4k real, in-the-wild user
prompts hand-labelled for toxicity) and then holds the result to a much stricter
bar, because a bigger real dataset brings a bigger risk of a boring confound:
**toxic prompts are ~3x longer than benign ones** (median ~178 vs ~58 chars), so
a probe could score high just by reading length.

Everything here REUSES the deployed modules so the recipe is provably identical:
  * ``data_large.load_large_dataset``  — the balanced Toxic-Chat loader.
  * ``model_utils.extract_features``   — frozen-Gemma activations at LAYER 12.
  * ``train_probe.train_probe``        — the exact 3-layer-MLP training loop.
  * ``train_probe.evaluate``           — the exact 12-metric suite + curves.
  * ``cross_validate.*``               — the exact 5-fold CV recipe + logreg ref.
  * ``audit_leakage._fit_lr_report``   — the exact train-only LogReg proxy.

It writes ONLY ``*_large`` artifacts and never touches the JailbreakBench files
(``metrics.json``, ``features.npz``, ``probe.pt``, ...).

Sections:
    1. Load Toxic-Chat (1390 balanced prompts); print balance + natural base rate.
    2. Extract + cache activation features  (artifacts/features_large.npz).
    3. Stratified 70/15/15 split (label-stratified, seed 0, leakage-safe).
    4. Standardize (train-only) + train the 3-layer MLP; save probe_large.pt.
    5. Full 12-metric suite on held-out test + 5 plots + metrics_large.json.
    6. 5-fold stratified CV (mean ± 95% CI) + LogReg reference (cv_large.*).
    7. Leakage + length-confound audit with an explicit verdict (audit_large.*).
    8. Cross-dataset OOD: the Toxic-Chat probe, zero-shot, on JailbreakBench.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from . import config as C
from .data_large import load_large_dataset
from .model_utils import extract_features, hidden_size, load_model, num_layers
from .probe import MLPProbe, Scaler, predict_proba, save_probe
# Reuse the DEPLOYED training loop + metric suite verbatim (no reimplementation).
from .train_probe import train_probe as train_mlp, evaluate
# Reuse the DEPLOYED cross-validation recipe verbatim.
from .cross_validate import (K_FOLDS, METRIC_NAMES, aggregate, compute_metrics,
                             logreg_reference, mlp_proba, stratified_val_split,
                             train_one_mlp)
# Reuse the DEPLOYED leakage proxy (train-only-standardized LogReg) verbatim.
from .audit_leakage import _fit_lr_report, _label_balance

# --- Run configuration ------------------------------------------------------
N_PER_CLASS = 750           # loader caps at ~695/class (toxic class is scarce)
TEST_FRACTION = 0.15        # task-mandated 70/15/15 split
VAL_FRACTION = 0.15         # carved from the training portion for early stopping
NATURAL_BASE_RATE = 0.07    # ~7% of real Toxic-Chat traffic is toxic (imbalanced)

# --- Paths (ALL suffixed _large; the JBB artifacts are never overwritten) ----
FEATURES_LARGE = C.ARTIFACTS / "features_large.npz"
PROBE_LARGE = C.ARTIFACTS / "probe_large.pt"
METRICS_LARGE = C.ARTIFACTS / "metrics_large.json"
ROC_LARGE = C.ARTIFACTS / "roc_large.png"
PR_LARGE = C.ARTIFACTS / "pr_large.png"
CALIBRATION_LARGE = C.ARTIFACTS / "calibration_large.png"
CONFUSION_LARGE = C.ARTIFACTS / "confusion_large.png"
HISTORY_LARGE = C.ARTIFACTS / "history_large.png"
CV_LARGE_JSON = C.ARTIFACTS / "cv_large.json"
CV_LARGE_MD = C.ARTIFACTS / "cv_large.md"
AUDIT_LARGE_JSON = C.ARTIFACTS / "audit_large.json"
AUDIT_LARGE_MD = C.ARTIFACTS / "audit_large.md"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================== #
# Step 2 — extract-or-load cached features                                    #
# =========================================================================== #
def get_features_large(prompts: list[str], labels: list[int]):
    """Return (X, y, layer). Uses features_large.npz if it matches this config.

    Mirrors ``train_probe.get_features`` but writes to the ``_large`` cache and
    additionally stores the prompt strings (the audit needs them for the
    text/length confound checks without re-downloading anything).
    """
    if FEATURES_LARGE.exists():
        cache = np.load(FEATURES_LARGE, allow_pickle=True)
        if (int(cache["layer"]) == C.LAYER
                and str(cache["model_id"]) == C.MODEL_ID
                and cache["X"].shape[0] == len(prompts)):
            print(f"[features] cache hit {FEATURES_LARGE.name} X={cache['X'].shape}",
                  file=sys.stderr)
            return cache["X"], cache["y"], int(cache["layer"])

    # Cache miss → load the frozen model, extract, cache, then free VRAM.
    model, tok = load_model(C.MODEL_ID)
    layer = max(0, min(C.LAYER, num_layers(model) - 1))
    X = extract_features(model, tok, prompts, layer, pooling=C.POOLING, log_every=25)
    y = np.asarray(labels, dtype=np.int64)
    np.savez(FEATURES_LARGE, X=X, y=y, layer=layer, model_id=C.MODEL_ID,
             hidden=hidden_size(model), prompts=np.array(prompts, dtype=object))
    del model                                   # free VRAM — only X is needed now
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[features] extracted + cached {FEATURES_LARGE.name} X={X.shape}",
          file=sys.stderr)
    return X, y, layer


# =========================================================================== #
# Step 3 — stratified 70/15/15 split                                          #
# =========================================================================== #
def stratified_split_3way(y: np.ndarray, rng: np.random.Generator,
                          test_frac: float, val_frac: float):
    """Return (train, val, test) index arrays, balanced across both classes.

    Identical logic to ``train_probe.stratified_split`` but with the 15/15
    fractions passed in explicitly so we land on the mandated 70/15/15.
    """
    train_idx, val_idx, test_idx = [], [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_idx += idx[:n_test].tolist()
        val_idx += idx[n_test:n_test + n_val].tolist()
        train_idx += idx[n_test + n_val:].tolist()
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# =========================================================================== #
# Step 5 — plots (mirror of train_probe.make_plots, to the _large paths)      #
# =========================================================================== #
def make_plots_large(metrics: dict, history: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = metrics["curves"]
    m = metrics["metrics"]

    # ROC curve
    roc = curves["roc"]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(roc["fpr"], roc["tpr"], color="#2563eb", lw=2,
            label=f"AUC = {m['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — Toxic-Chat held-out test"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(ROC_LARGE, dpi=110); plt.close(fig)

    # Precision-Recall curve
    pr = curves["pr"]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(pr["recall"], pr["precision"], color="#7c3aed", lw=2,
            label=f"PR-AUC = {m['pr_auc']:.3f}")
    cm = np.array(metrics["confusion_matrix"])
    prevalence = cm[1].sum() / max(1, cm.sum())
    ax.axhline(prevalence, ls="--", color="#9ca3af", lw=1,
               label=f"chance = {prevalence:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 1.0)
    ax.set_title("Precision-Recall — Toxic-Chat test"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(PR_LARGE, dpi=110); plt.close(fig)

    # Calibration / reliability diagram
    cal = curves["calibration"]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1, label="perfect")
    ax.plot(cal["prob_pred"], cal["prob_true"], "o-", color="#059669", lw=2,
            label=f"probe (Brier={m['brier_score_loss']:.3f})")
    ax.set_xlabel("Mean predicted P(harmful)"); ax.set_ylabel("Observed harmful fraction")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Calibration — Toxic-Chat test"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(CALIBRATION_LARGE, dpi=110); plt.close(fig)

    # Training history
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.plot(history["train_loss"], label="train loss", color="#2563eb")
    ax.plot(history["val_loss"], label="val loss", color="#dc2626")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.set_title("Training history")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(HISTORY_LARGE, dpi=110); plt.close(fig)

    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["safe", "harmful"]); ax.set_yticklabels(["safe", "harmful"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout(); fig.savefig(CONFUSION_LARGE, dpi=110); plt.close(fig)
    print(f"[plots] wrote {ROC_LARGE.name}, {PR_LARGE.name}, "
          f"{CALIBRATION_LARGE.name}, {HISTORY_LARGE.name}, {CONFUSION_LARGE.name}",
          file=sys.stderr)


# =========================================================================== #
# Step 6 — 5-fold cross-validation (mirror of cross_validate.main)            #
# =========================================================================== #
def run_cross_validation(X: np.ndarray, y: np.ndarray, single: dict) -> dict:
    """5-fold stratified CV of the deployed MLP + a LogReg linear-probe reference.

    Reuses ``cross_validate``'s helpers verbatim (train_one_mlp, compute_metrics,
    aggregate, logreg_reference, stratified_val_split) so the recipe is identical
    to the deployed CV — only the artifacts differ (cv_large.*). CPU-only.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    device = "cpu"
    in_dim = X.shape[1]
    cv_seed = C.SEED
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=cv_seed)

    per_fold = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler().fit(X[tr])                 # train-only fit
        Xtr_all = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)

        rng = np.random.default_rng(cv_seed + fold)
        fit_pos, val_pos = stratified_val_split(y[tr], C.VAL_FRACTION, rng)
        set_seed(cv_seed + fold)                             # deterministic init
        probe = train_one_mlp(Xtr_all[fit_pos], y[tr][fit_pos],
                              Xtr_all[val_pos], y[tr][val_pos], in_dim, device)
        p_te = mlp_proba(probe, Xte, device)
        m = compute_metrics(y[te], p_te, C.DECISION_THRESHOLD)
        per_fold.append(m)
        print(f"[cv fold {fold}/{K_FOLDS}] n_test={len(te)} acc={m['accuracy']:.3f} "
              f"auc={m['roc_auc']:.3f} f1={m['f1']:.3f}", file=sys.stderr)

    mlp_agg = aggregate(per_fold)
    logreg = logreg_reference(X, y)

    report = {
        "config": {"k_folds": K_FOLDS, "cv_seed": cv_seed,
                   "estimator": "MLPProbe (in->128->32->1, dropout %.2f)" % C.DROPOUT,
                   "recipe": {"lr": C.LR, "weight_decay": C.WEIGHT_DECAY,
                              "epochs_max": C.EPOCHS, "patience": C.PATIENCE,
                              "val_fraction": C.VAL_FRACTION,
                              "threshold": C.DECISION_THRESHOLD},
                   "scaler": "StandardScaler, fit per-fold on train only",
                   "device": device},
        "data": {"n_total": int(len(y)), "in_dim": int(in_dim),
                 "class_balance": np.bincount(y).tolist()},
        "mlp_per_fold": per_fold,
        "mlp_aggregate": mlp_agg,
        "logreg_reference": logreg,
        "single_split": single,
    }
    CV_LARGE_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[cv] wrote {CV_LARGE_JSON.name}", file=sys.stderr)

    # --- readable markdown --------------------------------------------------
    lines = [f"# {K_FOLDS}-Fold Cross-Validation — Safety Probe (Toxic-Chat)\n"]
    lines.append(f"Frozen-LLM activations `X` = **{len(y)}x{in_dim}**, balanced "
                 f"({np.bincount(y).tolist()}). CPU-only; the Gemma model is never "
                 "loaded — we reuse `artifacts/features_large.npz`. This mean ± CI "
                 "across folds is the TRUSTWORTHY headline (every example is held "
                 "out exactly once).\n")
    lines.append(f"Estimator: the deployed 3-layer `MLPProbe` ({in_dim}->128->32->1, "
                 f"dropout {C.DROPOUT}), trained with the exact `train_probe.py` "
                 f"recipe (Adam lr={C.LR}, weight_decay={C.WEIGHT_DECAY}, BCE, early "
                 "stop). StandardScaler fit per-fold on train only.\n")
    lines.append("## MLP probe — mean ± 95% CI across folds\n")
    lines.append("| metric | mean | std | 95% CI | single-split |")
    lines.append("|---|---|---|---|---|")
    for name in METRIC_NAMES:
        a = mlp_agg[name]
        s = single.get(name)
        s_str = f"{s:.4f}" if s is not None else "—"
        lines.append(f"| {name} | {a['mean']:.4f} | {a['std']:.4f} | "
                     f"[{a['ci95_low']:.4f}, {a['ci95_high']:.4f}] | {s_str} |")
    lines.append("")
    lines.append("## Linear-probe reference — LogisticRegression, 5-fold CV\n")
    lines.append("| metric | mean | std |")
    lines.append("|---|---|---|")
    lines.append(f"| accuracy | {logreg['accuracy']['mean']:.4f} | "
                 f"{logreg['accuracy']['std']:.4f} |")
    lines.append(f"| roc_auc | {logreg['roc_auc']['mean']:.4f} | "
                 f"{logreg['roc_auc']['std']:.4f} |")
    lines.append("")
    acc = mlp_agg["accuracy"]
    lines.append("## Trustworthy headline\n")
    lines.append(
        f"Across {K_FOLDS} stratified folds the MLP probe scores "
        f"**accuracy {acc['mean']:.3f} ± {acc['ci95_halfwidth']:.3f}** "
        f"(95% CI [{acc['ci95_low']:.3f}, {acc['ci95_high']:.3f}]) and "
        f"**roc_auc {mlp_agg['roc_auc']['mean']:.3f} ± "
        f"{mlp_agg['roc_auc']['ci95_halfwidth']:.3f}**. A plain logistic-regression "
        f"linear probe reaches {logreg['accuracy']['mean']:.3f} accuracy / "
        f"{logreg['roc_auc']['mean']:.3f} roc_auc, so the harmful-vs-benign signal "
        "is strongly (and largely linearly) decodable from this layer.\n")
    CV_LARGE_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[cv] wrote {CV_LARGE_MD.name}", file=sys.stderr)
    return {"mlp_aggregate": mlp_agg, "logreg_reference": logreg}


# =========================================================================== #
# Step 7 — leakage + length-confound audit                                    #
# =========================================================================== #
def _lr_test_probs(Xtr, ytr, Xte, standardize=True) -> np.ndarray:
    """Fit LogReg on (Xtr,ytr) [train-only standardized] and return P(harmful) on Xte."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if standardize:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def length_confound_audit(X, y, prompts, tr, va, te, p_test_probe) -> dict:
    """Rule out leakage AND settle the length-confound question honestly.

    (a) shuffled-label control on the ACTIVATIONS  — must collapse to ~chance.
    (b) length-only baseline  [char_len, word_len] -> LogReg (the confound test).
    (c) TF-IDF (1,2)-gram baseline (a stronger surface-text confound).
    Then a LENGTH-MATCHED check: bin the test prompts into char-length quartiles
    and, WITHIN each comparable-length bin, compare the deployed probe's accuracy
    to the length-only baseline's accuracy. If the probe stays high in the middle
    bins (where length no longer separates the classes), the probe learned more
    than length.
    """
    from collections import Counter

    from sklearn.metrics import accuracy_score, roc_auc_score

    report: dict = {"checks": {}, "flags": []}
    tr_prompts = [prompts[i] for i in tr]
    te_prompts = [prompts[i] for i in te]

    # ---- Check 0: split disjointness / duplicates (belt-and-suspenders) ----
    set_tr, set_va, set_te = set(tr_prompts), set([prompts[i] for i in va]), set(te_prompts)
    cross_overlaps = len(set_tr & set_te) + len(set_tr & set_va) + len(set_va & set_te)
    dup_counts = Counter(prompts)
    exact_dupes = sum(n - 1 for n in dup_counts.values() if n > 1)
    disjoint = cross_overlaps == 0
    report["checks"]["0_disjointness"] = {
        "cross_split_prompt_overlaps": cross_overlaps,
        "exact_duplicate_prompts_in_dataset": exact_dupes,
        "label_balance": {"train": _label_balance(y[tr]),
                          "val": _label_balance(y[va]),
                          "test": _label_balance(y[te])},
        "pass": bool(disjoint),
        "verdict": ("PASS — splits disjoint, no cross-split prompt leaks"
                    if disjoint else "FAIL — a prompt leaks across splits"),
    }
    if not disjoint:
        report["flags"].append("SPLIT_OVERLAP")

    # ---- Check 1: shuffled-label control on activations --------------------
    true_acc, true_auc = _fit_lr_report(X[tr], y[tr], X[te], y[te], standardize=True)
    shuffle_rng = np.random.default_rng(C.SEED + 1)
    y_tr_shuf = y[tr].copy()
    shuffle_rng.shuffle(y_tr_shuf)
    shuf_acc, shuf_auc = _fit_lr_report(X[tr], y_tr_shuf, X[te], y[te], standardize=True)
    shuffle_ok = shuf_acc <= 0.65
    report["checks"]["1_label_shuffle"] = {
        "true_labels": {"test_accuracy": true_acc, "test_auc": true_auc},
        "shuffled_labels": {"test_accuracy": shuf_acc, "test_auc": shuf_auc},
        "fail_threshold_shuffled_acc": 0.65,
        "pass": bool(shuffle_ok),
        "verdict": (f"PASS — true={true_acc:.3f}, shuffled={shuf_acc:.3f} ~chance; "
                    "no features->label leakage path"
                    if shuffle_ok else
                    f"FAIL — shuffled labels still score {shuf_acc:.3f} > 0.65"),
    }
    if not shuffle_ok:
        report["flags"].append("SHUFFLE_LEAK")

    # ---- Length statistics (the whole point of this dataset's risk) --------
    def char_lens(ps):
        return np.array([len(p) for p in ps], dtype=float)
    all_lens = char_lens(prompts)
    harm_lens = all_lens[y == 1]
    benign_lens = all_lens[y == 0]
    length_stats = {
        "median_chars_harmful": float(np.median(harm_lens)),
        "median_chars_benign": float(np.median(benign_lens)),
        "mean_chars_harmful": float(harm_lens.mean()),
        "mean_chars_benign": float(benign_lens.mean()),
        "length_ratio_harmful_over_benign": float(np.median(harm_lens) /
                                                  max(1.0, np.median(benign_lens))),
    }

    # ---- Check 2: length-only + TF-IDF surface baselines -------------------
    def len_feats(ps):
        return np.array([[len(p), len(p.split())] for p in ps], dtype=np.float64)
    len_acc, len_auc = _fit_lr_report(len_feats(tr_prompts), y[tr],
                                      len_feats(te_prompts), y[te], standardize=True)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr_tfidf = vec.fit_transform(tr_prompts)            # fit on train only
    Xte_tfidf = vec.transform(te_prompts)
    tfidf_clf = LogisticRegression(max_iter=2000).fit(Xtr_tfidf, y[tr])
    p_tfidf = tfidf_clf.predict_proba(Xte_tfidf)[:, 1]
    tfidf_acc = float(accuracy_score(y[te], (p_tfidf >= 0.5).astype(int)))
    tfidf_auc = float(roc_auc_score(y[te], p_tfidf))

    # ---- The deployed probe's own numbers on the SAME test set -------------
    probe_acc = float(accuracy_score(y[te], (p_test_probe >= 0.5).astype(int)))
    probe_auc = float(roc_auc_score(y[te], p_test_probe))

    report["checks"]["2_surface_baselines"] = {
        "length_stats": length_stats,
        "probe": {"test_accuracy": probe_acc, "test_auc": probe_auc},
        "length_only": {"test_accuracy": len_acc, "test_auc": len_auc},
        "tfidf": {"test_accuracy": tfidf_acc, "test_auc": tfidf_auc,
                  "n_features": int(Xtr_tfidf.shape[1])},
        "probe_lift_over_length_acc": round(probe_acc - len_acc, 4),
        "probe_lift_over_tfidf_acc": round(probe_acc - tfidf_acc, 4),
    }

    # ---- Check 3: LENGTH-MATCHED comparison (bin by char-length quartiles) -
    # Within a narrow length band the two classes have similar lengths, so
    # length-only prediction should decay toward chance. If the probe stays
    # high there, it is reading intent, not length.
    te_lens = char_lens(te_prompts)
    edges = np.quantile(te_lens, [0.0, 0.25, 0.50, 0.75, 1.0])
    p_len_test = _lr_test_probs(len_feats(tr_prompts), y[tr], len_feats(te_prompts))
    yte = y[te]
    bins = []
    for b in range(4):
        lo, hi = edges[b], edges[b + 1]
        mask = (te_lens >= lo) & (te_lens <= hi) if b == 3 else (te_lens >= lo) & (te_lens < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        yb = yte[mask]
        n_harm = int(yb.sum())
        # accuracy is only meaningful if both classes are present in the bin.
        both = 0 < n_harm < n
        probe_bin_acc = float(accuracy_score(yb, (p_test_probe[mask] >= 0.5).astype(int)))
        len_bin_acc = float(accuracy_score(yb, (p_len_test[mask] >= 0.5).astype(int)))
        bins.append({
            "bin": b + 1,
            "char_len_range": [float(lo), float(hi)],
            "n": n, "n_harmful": n_harm, "n_benign": n - n_harm,
            "both_classes_present": both,
            "probe_accuracy": round(probe_bin_acc, 4),
            "length_only_accuracy": round(len_bin_acc, 4),
            "probe_minus_length": round(probe_bin_acc - len_bin_acc, 4),
        })

    # Focus on the two MIDDLE bins — the region where lengths overlap most and
    # length-only prediction has the least to work with.
    mid = [b for b in bins if b["bin"] in (2, 3) and b["both_classes_present"]]
    mid_probe = float(np.mean([b["probe_accuracy"] for b in mid])) if mid else float("nan")
    mid_len = float(np.mean([b["length_only_accuracy"] for b in mid])) if mid else float("nan")
    mid_lift = round(mid_probe - mid_len, 4) if mid else float("nan")
    report["checks"]["3_length_matched"] = {
        "bins": bins,
        "middle_bins_probe_accuracy": round(mid_probe, 4) if mid else None,
        "middle_bins_length_only_accuracy": round(mid_len, 4) if mid else None,
        "middle_bins_probe_minus_length": mid_lift if mid else None,
        "note": ("Middle (overlap-length) bins are the honest test: length carries "
                 "little signal there, so any probe accuracy above the length-only "
                 "baseline is genuine intent signal."),
    }

    # ---- Overall verdict (derived from the numbers) ------------------------
    length_is_strong = len_acc >= 0.80
    if not shuffle_ok or not disjoint:
        verdict = ("FAIL — a genuine leakage path was detected; the large-set "
                   "headline is NOT trustworthy until fixed.")
        report["flags"].append("LEAKAGE")
    elif mid and mid_lift >= 0.10 and mid_probe >= 0.75:
        verdict = (
            "LEGITIMATE — no leakage (shuffle collapses to chance, splits disjoint). "
            f"Toxic prompts ARE ~{length_stats['length_ratio_harmful_over_benign']:.1f}x "
            "longer, and a length-only baseline is non-trivial "
            f"(acc {len_acc:.3f}), so length is a PARTIAL confound. BUT within "
            f"comparable-length (middle) bins the probe scores {mid_probe:.3f} vs the "
            f"length-only baseline's {mid_len:.3f} (+{mid_lift:.3f}) — the probe reads "
            "toxicity, not merely length.")
    elif mid and mid_lift < 0.05 and length_is_strong:
        verdict = (
            "LENGTH-DRIVEN — no leakage, but within comparable-length bins the probe "
            f"({mid_probe:.3f}) barely beats the length-only baseline ({mid_len:.3f}, "
            f"+{mid_lift:.3f}) and length-only alone reaches {len_acc:.3f}. Treat the "
            "large-set headline as substantially a LENGTH artifact.")
    else:
        verdict = (
            "LEGITIMATE-BUT-CAVEATED — no leakage. Length is a partial confound "
            f"(length-only acc {len_acc:.3f}; toxic prompts "
            f"~{length_stats['length_ratio_harmful_over_benign']:.1f}x longer). The "
            f"probe's overall accuracy ({probe_acc:.3f}) exceeds every surface "
            f"baseline (length {len_acc:.3f}, TF-IDF {tfidf_acc:.3f}); within-bin lift "
            f"is +{mid_lift if mid else float('nan')}. Report the probe's LIFT over "
            "the length baseline, not the raw number.")
    report["overall_verdict"] = verdict

    # ---- Write JSON + markdown --------------------------------------------
    AUDIT_LARGE_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    c0 = report["checks"]["0_disjointness"]
    c1 = report["checks"]["1_label_shuffle"]
    c2 = report["checks"]["2_surface_baselines"]
    c3 = report["checks"]["3_length_matched"]
    lines = [
        "# Large-set (Toxic-Chat) leakage & length-confound audit",
        "",
        f"Dataset: Toxic-Chat balanced (n={X.shape[0]}), split "
        f"train={len(tr)}/val={len(va)}/test={len(te)}, seed={C.SEED}. "
        f"Natural base rate ~{int(NATURAL_BASE_RATE * 100)}% toxic (we evaluate on a "
        "balanced set for readable metrics).",
        f"Deployed probe on this test set: acc={c2['probe']['test_accuracy']:.3f} / "
        f"auc={c2['probe']['test_auc']:.3f}.",
        "",
        "## Check 0 — split disjointness & duplicates",
        f"- cross-split prompt overlaps: {c0['cross_split_prompt_overlaps']}",
        f"- exact-duplicate prompts in dataset: {c0['exact_duplicate_prompts_in_dataset']}",
        f"- label balance: train {c0['label_balance']['train']}, "
        f"val {c0['label_balance']['val']}, test {c0['label_balance']['test']}",
        f"- VERDICT: {c0['verdict']}",
        "",
        "## Check 1 — label-shuffle control on activations",
        f"- TRUE labels:     acc={c1['true_labels']['test_accuracy']:.3f}, "
        f"auc={c1['true_labels']['test_auc']:.3f}",
        f"- SHUFFLED labels: acc={c1['shuffled_labels']['test_accuracy']:.3f}, "
        f"auc={c1['shuffled_labels']['test_auc']:.3f}  (must collapse; fail if acc>0.65)",
        f"- VERDICT: {c1['verdict']}",
        "",
        "## Check 2 — surface baselines (the length confound)",
        f"- median length: harmful {c2['length_stats']['median_chars_harmful']:.0f} "
        f"chars vs benign {c2['length_stats']['median_chars_benign']:.0f} chars "
        f"({c2['length_stats']['length_ratio_harmful_over_benign']:.1f}x)",
        f"- deployed probe:      acc={c2['probe']['test_accuracy']:.3f}, "
        f"auc={c2['probe']['test_auc']:.3f}",
        f"- length-only [char,word]: acc={c2['length_only']['test_accuracy']:.3f}, "
        f"auc={c2['length_only']['test_auc']:.3f}",
        f"- TF-IDF (1,2)-gram:       acc={c2['tfidf']['test_accuracy']:.3f}, "
        f"auc={c2['tfidf']['test_auc']:.3f} ({c2['tfidf']['n_features']} feats)",
        f"- probe lift over length: +{c2['probe_lift_over_length_acc']:.3f} acc; "
        f"over TF-IDF: +{c2['probe_lift_over_tfidf_acc']:.3f} acc",
        "",
        "## Check 3 — length-matched (within comparable-length bins)",
        "| bin | char-len range | n | harm/benign | probe acc | length-only acc | probe−length |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in c3["bins"]:
        lines.append(
            f"| {b['bin']} | {b['char_len_range'][0]:.0f}–{b['char_len_range'][1]:.0f} | "
            f"{b['n']} | {b['n_harmful']}/{b['n_benign']} | {b['probe_accuracy']:.3f} | "
            f"{b['length_only_accuracy']:.3f} | {b['probe_minus_length']:+.3f} |")
    if c3["middle_bins_probe_accuracy"] is not None:
        lines.append("")
        lines.append(f"- middle (overlap-length) bins: probe "
                     f"{c3['middle_bins_probe_accuracy']:.3f} vs length-only "
                     f"{c3['middle_bins_length_only_accuracy']:.3f} "
                     f"(+{c3['middle_bins_probe_minus_length']:.3f})")
    lines += ["", "## Overall verdict", verdict, "",
              f"Flags: {report['flags'] if report['flags'] else 'none'}"]
    AUDIT_LARGE_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] wrote {AUDIT_LARGE_JSON.name}, {AUDIT_LARGE_MD.name}", file=sys.stderr)
    return report


# =========================================================================== #
# Step 8 — cross-dataset OOD: Toxic-Chat probe, zero-shot, on JailbreakBench  #
# =========================================================================== #
def cross_dataset_ood(probe, scaler) -> dict:
    """Score the Toxic-Chat-trained probe zero-shot on the cached JBB features.

    JailbreakBench is a DIFFERENT distribution (curated, topically-matched, short
    prompts) than Toxic-Chat (real, in-the-wild, length-skewed). We reuse the
    cached ``features.npz`` (same model + layer 12 + mean-pool, so the activation
    space matches) and never reload the model. Reports accuracy + roc_auc.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    if not C.FEATURES_CACHE.exists():
        print("[ood] JBB features.npz missing — skipping cross-dataset OOD.",
              file=sys.stderr)
        return {}
    cache = np.load(C.FEATURES_CACHE, allow_pickle=True)
    if int(cache["layer"]) != C.LAYER or str(cache["model_id"]) != C.MODEL_ID:
        print("[ood] JBB cache layer/model mismatch — skipping.", file=sys.stderr)
        return {}
    Xj = cache["X"].astype(np.float32)
    yj = cache["y"].astype(np.int64)
    # Score on whatever device the probe already lives on (avoids a cpu/cuda mix).
    dev = str(next(probe.parameters()).device)
    probs = predict_proba(probe, scaler, Xj, device=dev)        # scaler applied inside
    acc = float(accuracy_score(yj, (probs >= C.DECISION_THRESHOLD).astype(int)))
    auc = float(roc_auc_score(yj, probs))
    out = {
        "dataset": "JailbreakBench harmful vs. benign (zero-shot transfer)",
        "note": "Different distribution: probe trained on Toxic-Chat, evaluated on "
                "curated JBB. Same model/layer/pooling so activations are comparable.",
        "n": int(len(yj)), "n_harmful": int(yj.sum()), "n_safe": int(len(yj) - yj.sum()),
        "accuracy": round(acc, 4), "roc_auc": round(auc, 4),
    }
    print(f"[ood] cross-dataset (Toxic-Chat->JBB): acc={acc:.3f} auc={auc:.3f}",
          file=sys.stderr)
    return out


# =========================================================================== #
# Orchestration                                                               #
# =========================================================================== #
def main() -> None:
    set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) data
    prompts, labels = load_large_dataset(n_per_class=N_PER_CLASS, seed=C.SEED)
    n_harm = int(sum(labels))
    print(f"[data] Toxic-Chat balanced: {len(prompts)} prompts "
          f"({n_harm} harmful / {len(labels) - n_harm} benign). "
          f"Natural base rate in the wild ~{int(NATURAL_BASE_RATE * 100)}% toxic — "
          "we balance for readable, non-degenerate metrics.", file=sys.stderr)

    # 2) features (cached to features_large.npz; frees VRAM after extraction)
    X, y, layer = get_features_large(prompts, labels)
    in_dim = X.shape[1]

    # 3) stratified 70/15/15 split
    rng = np.random.default_rng(C.SEED)
    tr, va, te = stratified_split_3way(y, rng, TEST_FRACTION, VAL_FRACTION)
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)} "
          f"(70/15/15, stratified)", file=sys.stderr)

    # 4) standardize (fit on TRAIN only) + train the deployed MLP recipe
    scaler = Scaler.fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    probe, history = train_mlp(Xtr, y[tr], Xva, y[va], in_dim, device)

    # 5) evaluate on held-out test (12-metric suite + curves), then plots + json
    p_test = predict_proba(probe, scaler, X[te], device=device)
    ev = evaluate(y[te], p_test, C.DECISION_THRESHOLD)
    m = ev["metrics"]

    examples = [{"prompt": prompts[idx][:160], "true": int(y[idx]),
                 "prob_harmful": float(prob)} for idx, prob in zip(te, p_test)]
    examples.sort(key=lambda e: e["prob_harmful"], reverse=True)
    examples = examples[:8] + examples[-8:]     # a few confident-harmful + confident-safe

    metrics_out = {
        "dataset": "Toxic-Chat (lmsys/toxic-chat) harmful vs. benign",
        "source": "lmsys/toxic-chat:data/0124/toxic-chat_annotation_all.csv",
        "model_id": C.MODEL_ID, "layer": int(layer), "pooling": C.POOLING,
        "hidden_dim": int(in_dim), "threshold": C.DECISION_THRESHOLD,
        "n_total": int(len(y)), "class_balance": np.bincount(y).tolist(),
        "natural_base_rate_toxic": NATURAL_BASE_RATE,
        "evaluation_note": "Balanced eval set (50/50). Real traffic is ~7% toxic; "
                           "on the natural base rate precision would be lower and "
                           "PR-AUC is the metric to watch.",
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": ev["n_test"],
        "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],   # back-compat duplicates
        "metrics": m,
        "confusion_matrix": ev["confusion_matrix"],
        "curves": ev["curves"],
        "plots": {"roc": ROC_LARGE.name, "pr": PR_LARGE.name,
                  "calibration": CALIBRATION_LARGE.name,
                  "confusion": CONFUSION_LARGE.name, "history": HISTORY_LARGE.name},
        "history": history,
        "examples": examples,
    }

    save_probe(PROBE_LARGE, probe, scaler,
               meta={"model_id": C.MODEL_ID, "layer": int(layer),
                     "pooling": C.POOLING, "threshold": C.DECISION_THRESHOLD,
                     "dataset": "toxic-chat"})
    make_plots_large(metrics_out, history)

    # 6) 5-fold CV (the trustworthy headline)
    single_for_cv = {}
    for name in METRIC_NAMES:
        key = "brier_score_loss" if name == "brier" else name
        if key in m:
            single_for_cv[name] = float(m[key])
    cv = run_cross_validation(X, y, single_for_cv)

    # 7) leakage + length-confound audit
    audit = length_confound_audit(X, y, prompts, tr, va, te, p_test)

    # 8) cross-dataset OOD (reuses cached JBB features; no model reload)
    ood = cross_dataset_ood(probe, scaler)
    metrics_out["cross_dataset_ood"] = ood
    METRICS_LARGE.write_text(json.dumps(metrics_out, indent=2))
    print(f"[metrics] wrote {METRICS_LARGE.name}", file=sys.stderr)

    # Free VRAM at the very end (belt-and-suspenders; model already freed).
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- console summary ----------------------------------------------------
    mlp_acc = cv["mlp_aggregate"]["accuracy"]
    mlp_auc = cv["mlp_aggregate"]["roc_auc"]
    mlp_f1 = cv["mlp_aggregate"]["f1"]
    lr = cv["logreg_reference"]
    c2 = audit["checks"]["2_surface_baselines"]
    print("\n=== LARGE-SET (TOXIC-CHAT) RESULTS ==========================")
    print(f"  held-out test (n={ev['n_test']}):")
    print(f"    accuracy {m['accuracy']:.3f}  bal_acc {m['balanced_accuracy']:.3f}  "
          f"f1 {m['f1']:.3f}  roc_auc {m['roc_auc']:.3f}  pr_auc {m['pr_auc']:.3f}")
    print(f"    precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
          f"specificity {m['specificity']:.3f}  mcc {m['mcc']:.3f}")
    print(f"  5-fold CV (TRUSTWORTHY headline, mean +/- 95% CI):")
    print(f"    accuracy {mlp_acc['mean']:.3f} +/- {mlp_acc['ci95_halfwidth']:.3f}   "
          f"roc_auc {mlp_auc['mean']:.3f} +/- {mlp_auc['ci95_halfwidth']:.3f}   "
          f"f1 {mlp_f1['mean']:.3f} +/- {mlp_f1['ci95_halfwidth']:.3f}")
    print(f"    logreg ref: acc {lr['accuracy']['mean']:.3f}  auc {lr['roc_auc']['mean']:.3f}")
    print(f"  length audit: probe {c2['probe']['test_accuracy']:.3f}  vs  "
          f"length-only {c2['length_only']['test_accuracy']:.3f}  vs  "
          f"TF-IDF {c2['tfidf']['test_accuracy']:.3f}")
    if ood:
        print(f"  cross-dataset OOD (Toxic-Chat->JBB): acc {ood['accuracy']:.3f}  "
              f"auc {ood['roc_auc']:.3f}")
    print(f"  VERDICT: {audit['overall_verdict']}")
    print("==============================================================")


if __name__ == "__main__":
    main()
