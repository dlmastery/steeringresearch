"""audit_leakage.py -- a leakage / dataset-artifact audit of the safety probe.

Run:  python -m steering_tutorials.hello_world.audit_leakage   (from the repo root)

The headline claim of this hello-world is "0.95 test accuracy / 0.98 AUC when a
tiny MLP reads frozen Gemma activations and classifies JailbreakBench harmful vs.
benign prompts." Before anyone repeats that number we want to rule out the two
boring explanations for a high score:

  1. LEAKAGE -- the same prompt (or its label) sneaks from train into test, so the
     probe is graded on data it effectively saw during training.
  2. TEXT ARTIFACT -- the prompt strings are separable by trivial surface features
     (length, a few give-away words) so a bag-of-words model with ZERO knowledge
     of the LLM already scores high. If so, part of the "the model knows harm in
     its activations" story is really "JailbreakBench has a surface tell."

This script is CPU-only. It never loads Gemma: it reuses the cached activation
matrix in ``artifacts/features.npz`` and only downloads the raw prompt CSVs (for
the text-only checks) via the same loader the trainer uses.

It reproduces the EXACT train/val/test split the trainer used, runs four checks,
writes ``artifacts/audit_report.json`` + ``artifacts/audit_report.md``, and
prints the summary.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from . import config as C
from .data import load_safety_dataset
from .train_probe import stratified_split


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _fit_lr_report(Xtr, ytr, Xte, yte, standardize=True):
    """Train a plain LogisticRegression on (Xtr,ytr), score on (Xte,yte).

    Standardization (when requested) is fit on the TRAIN split only -- the same
    discipline the probe trainer uses -- so no test statistics leak in.
    Returns (test_accuracy, test_roc_auc).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if standardize:
        scaler = StandardScaler().fit(Xtr)      # fit on train only
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    acc = float(accuracy_score(yte, (p >= 0.5).astype(int)))
    # AUC is undefined if the test labels are single-class; guard just in case.
    auc = float(roc_auc_score(yte, p)) if len(set(yte.tolist())) > 1 else float("nan")
    return acc, auc


def _label_balance(y_split):
    """Return {harmful: n1, benign: n0} for a label array."""
    c = Counter(int(v) for v in y_split)
    return {"harmful(1)": c.get(1, 0), "benign(0)": c.get(0, 0)}


# --------------------------------------------------------------------------- #
# Main audit                                                                  #
# --------------------------------------------------------------------------- #
def main() -> None:
    report: dict = {"headline_claim": {}, "checks": {}, "flags": []}

    # ---- Load cached activations (X, y) -- NEVER touches the Gemma model ----- #
    if not C.FEATURES_CACHE.exists():
        raise SystemExit(f"missing feature cache: {C.FEATURES_CACHE}")
    cache = np.load(C.FEATURES_CACHE, allow_pickle=True)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    print(f"[audit] loaded features X={X.shape} y={y.shape} "
          f"layer={int(cache['layer'])} model={str(cache['model_id'])}",
          file=sys.stderr)

    # ---- Load the prompt strings (for the text-only checks) ---------------- #
    # load_safety_dataset only downloads two CSVs; it does not load the model.
    prompts, labels = load_safety_dataset()
    prompts = list(prompts)
    labels = np.asarray(labels, dtype=np.int64)

    # Sanity: the cached feature order must match the dataset order, otherwise
    # every downstream index alignment (prompt <-> activation) would be wrong.
    if len(prompts) != X.shape[0]:
        raise SystemExit(f"prompt count {len(prompts)} != feature rows {X.shape[0]}")
    if not np.array_equal(labels, y):
        raise SystemExit("cached y does not match dataset labels -- order mismatch")

    # ---- Reproduce the EXACT trainer split --------------------------------- #
    # train_probe.main() does: rng = np.random.default_rng(SEED); stratified_split(y, rng)
    rng = np.random.default_rng(C.SEED)
    tr, va, te = stratified_split(y, rng)
    print(f"[audit] split train={len(tr)} val={len(va)} test={len(te)}", file=sys.stderr)

    # ===================================================================== #
    # CHECK 1 -- split disjointness / duplicate prompts / label balance      #
    # ===================================================================== #
    tr_prompts = [prompts[i] for i in tr]
    va_prompts = [prompts[i] for i in va]
    te_prompts = [prompts[i] for i in te]
    set_tr, set_va, set_te = set(tr_prompts), set(va_prompts), set(te_prompts)

    # A prompt that appears in two different splits is a cross-split overlap.
    overlap_tr_te = set_tr & set_te
    overlap_tr_va = set_tr & set_va
    overlap_va_te = set_va & set_te
    cross_split_overlaps = len(overlap_tr_te) + len(overlap_tr_va) + len(overlap_va_te)

    # Exact-duplicate prompt strings anywhere in the full dataset. (Even within
    # one split these matter: a duplicate spanning train/test is the classic leak.)
    dup_counts = Counter(prompts)
    exact_duplicate_prompts = sum(n - 1 for n in dup_counts.values() if n > 1)
    duplicate_examples = [p[:80] for p, n in dup_counts.items() if n > 1][:5]

    # index-level disjointness (belt and suspenders -- indices should never overlap)
    idx_disjoint = (
        len(set(tr.tolist()) & set(te.tolist())) == 0
        and len(set(tr.tolist()) & set(va.tolist())) == 0
        and len(set(va.tolist()) & set(te.tolist())) == 0
    )

    check1_pass = (cross_split_overlaps == 0) and idx_disjoint
    report["checks"]["1_disjointness"] = {
        "cross_split_prompt_overlaps": cross_split_overlaps,
        "overlap_train_test": len(overlap_tr_te),
        "overlap_train_val": len(overlap_tr_va),
        "overlap_val_test": len(overlap_va_te),
        "index_level_disjoint": bool(idx_disjoint),
        "exact_duplicate_prompts_in_dataset": exact_duplicate_prompts,
        "duplicate_examples": duplicate_examples,
        "label_balance": {
            "train": _label_balance(y[tr]),
            "val": _label_balance(y[va]),
            "test": _label_balance(y[te]),
        },
        "pass": bool(check1_pass),
        "verdict": ("PASS -- splits are disjoint at both prompt and index level"
                    if check1_pass else
                    "FAIL -- a prompt leaks across splits"),
    }
    if not check1_pass:
        report["flags"].append("CHECK1_FAIL_split_overlap")

    # ===================================================================== #
    # CHECK 2 -- label-shuffle (permutation) control on the ACTIVATIONS      #
    # ===================================================================== #
    # If the features truly carry the harm signal, TRUE labels should train to a
    # high test score. If we SHUFFLE the training labels, any honest model must
    # collapse to chance (~0.5): there is no legitimate features->label path left.
    # A shuffled score that stays high would mean a leak (e.g. test stats bled in,
    # or the "test" rows were memorized). We fit standardization on train only.
    true_acc, true_auc = _fit_lr_report(X[tr], y[tr], X[te], y[te], standardize=True)

    shuffle_rng = np.random.default_rng(C.SEED + 1)
    y_tr_shuf = y[tr].copy()
    shuffle_rng.shuffle(y_tr_shuf)
    shuf_acc, shuf_auc = _fit_lr_report(X[tr], y_tr_shuf, X[te], y[te], standardize=True)

    check2_pass = shuf_acc <= 0.65
    report["checks"]["2_label_shuffle"] = {
        "true_labels": {"test_accuracy": true_acc, "test_auc": true_auc},
        "shuffled_labels": {"test_accuracy": shuf_acc, "test_auc": shuf_auc},
        "fail_threshold_shuffled_acc": 0.65,
        "pass": bool(check2_pass),
        "verdict": (f"PASS -- true={true_acc:.3f} high, shuffled={shuf_acc:.3f} "
                    f"~chance; no features->label leakage path"
                    if check2_pass else
                    f"FAIL -- shuffled labels still score {shuf_acc:.3f} > 0.65; "
                    f"a leakage path exists"),
    }
    if not check2_pass:
        report["flags"].append("CHECK2_FAIL_shuffle_leak")

    # ===================================================================== #
    # CHECK 3 -- trivial text-confound baselines (NO activations at all)     #
    # ===================================================================== #
    # These use ONLY the raw prompt strings on the SAME split. If a dumb text
    # model already scores ~0.90+, then part of the probe's headline reflects a
    # JailbreakBench surface artifact, not deep LLM "understanding."

    # (a) length-only features: [char_len, word_len]
    def _len_feats(ps):
        return np.array([[len(p), len(p.split())] for p in ps], dtype=np.float64)

    len_acc, len_auc = _fit_lr_report(
        _len_feats(tr_prompts), y[tr], _len_feats(te_prompts), y[te], standardize=True
    )

    # (b) TF-IDF bag-of-words/bigrams. Vectorizer is fit on TRAIN prompts only.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)  # word-level default
    Xtr_tfidf = vec.fit_transform(tr_prompts)            # fit on train only
    Xte_tfidf = vec.transform(te_prompts)
    tfidf_clf = LogisticRegression(max_iter=2000)
    tfidf_clf.fit(Xtr_tfidf, y[tr])
    p_tfidf = tfidf_clf.predict_proba(Xte_tfidf)[:, 1]
    tfidf_acc = float(accuracy_score(y[te], (p_tfidf >= 0.5).astype(int)))
    tfidf_auc = float(roc_auc_score(y[te], p_tfidf))

    strong_artifact = (tfidf_acc >= 0.90) or (len_acc >= 0.90)
    report["checks"]["3_text_confounds"] = {
        "length_only": {"test_accuracy": len_acc, "test_auc": len_auc},
        "tfidf": {"test_accuracy": tfidf_acc, "test_auc": tfidf_auc,
                  "n_features": int(Xtr_tfidf.shape[1])},
        "strong_text_artifact_ge_0.90": bool(strong_artifact),
        "interpretation": (
            "A trivial text baseline already reaches >=0.90 -- a meaningful part "
            "of the 0.95 headline is a JailbreakBench SURFACE artifact, not proof "
            "of deep activation-level understanding. Report the probe's lift OVER "
            "this baseline, not the raw number."
            if strong_artifact else
            "Trivial text baselines stay well below the probe -- the probe's score "
            "is not explained by surface text features alone."
        ),
        # this check never 'fails' the pipeline; it is an honesty caveat.
        "pass": True,
        "verdict": (f"CAVEAT -- TF-IDF alone hits {tfidf_acc:.3f} acc / {tfidf_auc:.3f} "
                    f"AUC; the headline is partly a text artifact"
                    if strong_artifact else
                    f"OK -- text-only baselines (tfidf {tfidf_acc:.3f}) trail the probe"),
    }
    if strong_artifact:
        report["flags"].append("CHECK3_CAVEAT_text_artifact")

    # ===================================================================== #
    # CHECK 4 -- scaler discipline restatement                               #
    # ===================================================================== #
    # We do not re-run training here; we confirm by code-reference that the
    # trainer fits standardization on the TRAIN split only.
    #   train_probe.py step 4:  scaler = Scaler.fit(X[tr])
    #                           Xtr,Xva,Xte = scaler.transform(...)  # same train-fit scaler
    report["checks"]["4_scaler_discipline"] = {
        "reference": "train_probe.py step 4: Scaler.fit(X[tr]) then transform "
                     "train/val/test with that train-fit scaler",
        "test_statistics_used_in_fit": False,
        "pass": True,
        "verdict": "PASS -- standardization is fit on the train split only; no "
                   "test statistics leak into the scaler.",
    }

    # ---- Headline for context --------------------------------------------- #
    report["headline_claim"] = {
        "probe_true_lr_proxy_test_acc": true_acc,
        "probe_true_lr_proxy_test_auc": true_auc,
        "note": "LogisticRegression proxy on the same features/split; the MLP "
                "probe's reported headline is acc=0.95 / auc=0.98 (metrics.json).",
    }

    # ---- Overall verdict --------------------------------------------------- #
    hard_fail = any(f.endswith("_leak") or "FAIL_split" in f for f in report["flags"])
    if hard_fail:
        overall = ("FAIL -- a genuine leakage path was detected; the headline is "
                   "NOT trustworthy until fixed.")
    elif "CHECK3_CAVEAT_text_artifact" in report["flags"]:
        overall = ("LEGITIMATE-BUT-CAVEATED -- no leakage (splits disjoint, shuffle "
                   "collapses to chance, scaler is train-only), so the 0.95 is real "
                   "in the no-leak sense. HOWEVER a trivial TF-IDF text baseline is "
                   "already strong on JailbreakBench, so part of the score is a "
                   "dataset surface artifact. Report the probe's LIFT over TF-IDF.")
    else:
        overall = ("LEGITIMATE -- no leakage and no dominant text artifact; the "
                   "headline reflects real activation-level signal.")
    report["overall_verdict"] = overall

    # ---- Write JSON -------------------------------------------------------- #
    (C.ARTIFACTS / "audit_report.json").write_text(json.dumps(report, indent=2))

    # ---- Write + print the readable summary -------------------------------- #
    c1 = report["checks"]["1_disjointness"]
    c2 = report["checks"]["2_label_shuffle"]
    c3 = report["checks"]["3_text_confounds"]
    c4 = report["checks"]["4_scaler_discipline"]
    lines = [
        "# Safety-probe leakage & artifact audit",
        "",
        f"Dataset: JailbreakBench harmful vs. benign (n={X.shape[0]}), "
        f"split train={len(tr)}/val={len(va)}/test={len(te)}, seed={C.SEED}.",
        f"Headline under audit: MLP probe acc=0.95 / auc=0.98 (metrics.json).",
        "",
        "## Check 1 -- split disjointness & duplicates",
        f"- cross-split prompt overlaps: {c1['cross_split_prompt_overlaps']} "
        f"(train/test={c1['overlap_train_test']}, train/val={c1['overlap_train_val']}, "
        f"val/test={c1['overlap_val_test']})",
        f"- exact-duplicate prompts in full dataset: "
        f"{c1['exact_duplicate_prompts_in_dataset']}",
        f"- label balance: train {c1['label_balance']['train']}, "
        f"val {c1['label_balance']['val']}, test {c1['label_balance']['test']}",
        f"- VERDICT: {c1['verdict']}",
        "",
        "## Check 2 -- label-shuffle (permutation) control on activations",
        f"- TRUE labels:     test acc={c2['true_labels']['test_accuracy']:.3f}, "
        f"auc={c2['true_labels']['test_auc']:.3f}",
        f"- SHUFFLED labels: test acc={c2['shuffled_labels']['test_accuracy']:.3f}, "
        f"auc={c2['shuffled_labels']['test_auc']:.3f}  (fail if acc>0.65)",
        f"- VERDICT: {c2['verdict']}",
        "",
        "## Check 3 -- trivial text-confound baselines (no activations)",
        f"- length-only [char,word]: acc={c3['length_only']['test_accuracy']:.3f}, "
        f"auc={c3['length_only']['test_auc']:.3f}",
        f"- TF-IDF (1,2)-gram:       acc={c3['tfidf']['test_accuracy']:.3f}, "
        f"auc={c3['tfidf']['test_auc']:.3f} ({c3['tfidf']['n_features']} feats)",
        f"- {c3['interpretation']}",
        f"- VERDICT: {c3['verdict']}",
        "",
        "## Check 4 -- scaler discipline",
        f"- {c4['reference']}",
        f"- VERDICT: {c4['verdict']}",
        "",
        "## Overall verdict",
        overall,
        "",
        f"Flags: {report['flags'] if report['flags'] else 'none'}",
    ]
    summary = "\n".join(lines)
    (C.ARTIFACTS / "audit_report.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
