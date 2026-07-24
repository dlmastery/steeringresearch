"""run_biencoder_guard.py -- orchestrator for the DUAL-TOWER (BI-ENCODER) SAFETY
GUARDRAIL lesson (the production-guardrail member of the safety-detection course).

------------------------------------------------------------------------------
THE THESIS (what the six experiments prove)
------------------------------------------------------------------------------
Moderating content against a LARGE policy taxonomy has two shapes:

  * a UNI-encoder fuses the text with each policy description and scores them
    JOINTLY -- accurate, but it must re-encode once PER (text, label) pair, so
    its cost grows linearly with the number of labels (it collapses past a few
    dozen policies).
  * a BI-encoder decouples the towers: a CONTENT tower embeds the text once, a
    POLICY tower embeds each policy DESCRIPTION once, and compatibility is a
    cheap cosine in the shared space. The policy vectors are text-INDEPENDENT,
    so they are embedded ONCE and cached -> ~constant per-request cost at ANY
    label count, and a brand-NEW policy is added ZERO-SHOT from its description
    alone, with no retraining.

We benchmark three guards -- bi_encoder (HERO: cached, scales, zero-shot),
uni_encoder (re-encode-per-label; does not scale), trained_head (supervised;
strong on SEEN labels, cannot score an UNSEEN one) -- on a hard, multi-dataset,
many-label safety corpus (BeaverTails + toxic-chat + wildguardmix).

------------------------------------------------------------------------------
THE SIX EXPERIMENTS
------------------------------------------------------------------------------
  EXP-A  seen-policy multilabel : per-method macro/micro AP + F1 on SEEN cols.
  EXP-B  held-out ZERO-SHOT     : score policies never seen in training. bi &
                                  uni report AP/F1; trained_head = "N/A". THE
                                  HEADLINE -- add a policy from a description.
  EXP-C  multi-prototype ablate : bi_encoder on held-out cols with n_proto=1 (a
                                  single description) vs n_proto=P (P paraphrases
                                  averaged) -> AP delta. The 2026 "synthetic
                                  schema expansion" idea.
  EXP-D  latency vs #labels      : bi (embed once + matmul vs K cached vecs) is
                                  FLAT; uni (re-encode texts x K joints) rises
                                  LINEARLY. The "million-label" scaling claim.
  EXP-E  OOD                     : score a disjoint out-of-distribution slice
                                  over the SEEN cols -> binary harm AUC + macro
                                  AP. The real generalization check.
  EXP-F  hard-negative synthesis : the 2026 contrastive-augmentation recipe --
                                  dense-mine look-alike benigns -> ECIsem score
                                  them -> CausalNeg counterfactuals -> ARHN
                                  false-negative filter -> train a small
                                  ContrastiveAdapter -> does it cut FPR@recall
                                  0.90 vs the frozen bi-encoder on hard negatives?

Detection task -> NO generation judge (results["judge"] is null). CPU-only from
the orchestrator's view: the embedder loads inside the encoders module. Sibling
modules (data / encoders / hardneg) are imported LAZILY inside main() so
`python -c "import ...run_biencoder_guard"` succeeds while they are still stubs.
Stdout is ASCII only (Windows cp1252): we write "cos" / "AP" / "FPR" / ">=",
never unicode. results.json is written BEFORE the summary print; each EXP and
each plot is wrapped so a late failure still leaves results.json on disk.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import config as C


# ===========================================================================
# Scoring adapter + metric glue (no sibling / model dependency -> import-safe)
# ===========================================================================
def _guard_scores(guard, Xc, texts, policy_bank, cols):
    """Uniform per-method scoring adapter -> np.ndarray[n, len(cols)] in [0,1].

    The shared surface is `guard.scores(Xc, policy_bank, cols, texts=None)`. The
    bi_encoder and trained_head score PURELY from the precomputed content
    embeddings `Xc` and ignore `texts`; the uni_encoder RE-ENCODES the joint
    "(text, policy)" strings, so it consumes `texts`. We always pass the raw
    strings through (one call site for all three methods), falling back to the
    3-arg form only if a guard were built without the kwarg.
    """
    try:
        return np.asarray(guard.scores(Xc, policy_bank, cols, texts=texts), dtype=float)
    except TypeError:
        return np.asarray(guard.scores(Xc, policy_bank, cols), dtype=float)


def _any_policy_score(S):
    """Reduce a [n, n_cols] policy-score matrix to one per-text harm score.

    "Is this harmful under ANY policy?" == the MAX policy score for the text.
    Used to compute the binary harmful-vs-benign AUC alongside the multi-label
    metrics.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[1] == 0:
        return np.zeros(S.shape[0], dtype=float)
    return np.nanmax(S, axis=1)


def _seen_metrics(encoders, Y_true, S, is_harmful):
    """The EXP-A / EXP-E metric bundle for one method on one column set.

    Returns {"macro_ap","micro_ap","macro_f1","micro_f1","binary_harm_auc"} --
    the multi-label ranking + thresholded metrics plus the any-policy harm AUC.
    """
    mm = encoders.macro_micro(Y_true, S, thresholds=None)
    out = {
        "macro_ap": float(mm.get("macro_ap", float("nan"))),
        "micro_ap": float(mm.get("micro_ap", float("nan"))),
        "macro_f1": float(mm.get("macro_f1", float("nan"))),
        "micro_f1": float(mm.get("micro_f1", float("nan"))),
    }
    try:
        out["binary_harm_auc"] = float(
            encoders.binary_harm_auc(is_harmful, _any_policy_score(S)))
    except Exception:
        out["binary_harm_auc"] = float("nan")
    return out


# ===========================================================================
# Encoding / policy-bank caching helpers
# ===========================================================================
def _encode_content_cached(embedder, texts, split):
    """Embed a batch of texts with the CONTENT tower, cached to disk per split.

    The corpus texts are the SAME across the seen / held-out experiments (only
    the policy COLUMNS differ), so we embed the whole corpus once under the
    "train" cache key and index it by the train / test row splits. OOD texts are
    a genuinely different set and get their own cache key. Cache is keyed by
    (split, embedder-name); a shape mismatch invalidates it.
    """
    key = (split, C.EMBEDDER)
    path = C.EMB_CACHE.get(key)
    if path is not None and path.exists():
        try:
            data = np.load(path)
            X = data["X"].astype(np.float32)
            if X.shape[0] == len(texts):
                print("[embed] loaded cache %s  shape=%s" % (path.name, X.shape))
                return X
        except Exception as exc:
            print("[embed] cache reload failed (%s); re-encoding" % exc)
    X = np.asarray(embedder.encode(list(texts), "content"), dtype=np.float32)
    if path is not None:
        try:
            np.savez_compressed(path, X=X)
        except Exception as exc:
            print("[embed] cache save failed: %s" % exc)
    print("[embed] encoded %d texts -> shape=%s (split=%s)" % (len(texts), X.shape, split))
    return X


def _build_bank_cached(encoders, policies, embedder, n_proto, cache):
    """Build the POLICY tower ([P, dim]) with `n_proto` prototypes per policy.

    The main multi-prototype bank (n_proto == POLICY_PARAPHRASES) is cached to
    disk (it is reused by every experiment); the single-prototype ablation bank
    is cheap and built fresh. This is the tower we cache ONCE and match every
    incoming text against -- the crux of the bi-encoder scaling story.
    """
    path = C.POLICY_CACHE.get(C.EMBEDDER)
    if cache and path is not None and path.exists():
        try:
            data = np.load(path)
            B = data["B"].astype(np.float32)
            if B.shape[0] == len(policies):
                print("[bank] loaded cache %s  shape=%s" % (path.name, B.shape))
                return B
        except Exception as exc:
            print("[bank] cache reload failed (%s); rebuilding" % exc)
    B = np.asarray(encoders.build_policy_bank(policies, embedder, n_proto=n_proto),
                   dtype=np.float32)
    if cache and path is not None:
        try:
            np.savez_compressed(path, B=B)
        except Exception as exc:
            print("[bank] cache save failed: %s" % exc)
    print("[bank] built policy bank n_proto=%d -> shape=%s" % (n_proto, B.shape))
    return B


# ===========================================================================
# Plotting (Agg backend, PNG only) -- each wrapped by the caller
# ===========================================================================
def _plot_pr_by_method(pr_flat, out_path):
    """Micro-averaged precision-recall curve per method on the SEEN policies.

    `pr_flat`: {method: (y_true_flat[0/1], score_flat)} -- the flattened
    multi-hot labels and matching bi/uni/head scores over all seen columns.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(6.2, 5))
    for method, (yt, ys) in pr_flat.items():
        yt = np.asarray(yt).astype(int)
        ys = np.asarray(ys, dtype=float)
        if len(np.unique(yt)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(yt, ys)
        ap = average_precision_score(yt, ys)
        ax.plot(rec, prec, label="%s (AP=%.3f)" % (method, ap))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Precision-Recall by method (SEEN policies, %s)" % C.EMBEDDER)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_heldout_ap(heldout_block, multiproto, out_path):
    """Zero-shot held-out macro-AP bars: bi vs uni, plus the 1-vs-P proto ablation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, vals, colors = [], [], []
    for m, col in (("bi_encoder", "tab:green"), ("uni_encoder", "tab:orange")):
        cell = heldout_block.get(m, {})
        v = cell.get("macro_ap", float("nan")) if isinstance(cell, dict) else float("nan")
        labels.append("%s\n(zero-shot)" % m)
        vals.append(v if v == v else 0.0)
        colors.append(col)
    if isinstance(multiproto, dict) and "single" in multiproto:
        s = multiproto.get("single", {}).get("macro_ap", float("nan"))
        mu = multiproto.get("multi", {}).get("macro_ap", float("nan"))
        labels += ["bi 1-proto", "bi %d-proto" % int(multiproto.get("multi", {}).get("n_proto", 0) or 0)]
        vals += [s if s == s else 0.0, mu if mu == mu else 0.0]
        colors += ["tab:gray", "tab:blue"]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.bar(x, vals, 0.6, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Held-out macro-AP")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance-ish (0.5)")
    ax.set_title("Zero-shot held-out policy detection")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_latency(scaling, out_path):
    """Latency vs #labels (log-y): bi_encoder FLAT, uni_encoder LINEAR."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = scaling.get("labels", [])
    bi = scaling.get("bi_sec", [])
    uni = scaling.get("uni_sec", [])
    fig, ax = plt.subplots(figsize=(6.4, 5))
    if labels and bi:
        ax.plot(labels, bi, "o-", color="tab:green", label="bi_encoder (cached)")
    if labels and uni:
        ax.plot(labels, uni, "s-", color="tab:orange", label="uni_encoder (re-encode)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of policy labels")
    ax.set_ylabel("Seconds to moderate a batch (log)")
    ax.set_title("Moderation latency vs #labels")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_hardneg(hardneg_block, out_path):
    """FPR@recall0.90 bars: frozen bi-encoder vs the trained contrastive adapter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr = hardneg_block.get("fpr_at_recall90", {}) if isinstance(hardneg_block, dict) else {}
    frozen = fpr.get("frozen_bi", float("nan"))
    adapter = fpr.get("adapter", float("nan"))
    labels = ["frozen bi", "adapter"]
    vals = [frozen if frozen == frozen else 0.0, adapter if adapter == adapter else 0.0]
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.bar(labels, vals, 0.5, color=["tab:gray", "tab:blue"])
    ax.set_ylabel("FPR at recall=0.90 (lower=better)")
    ax.set_title("Hard-negative sharpening: frozen vs adapter")
    for i, v in enumerate(vals):
        ax.text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ===========================================================================
# EXP-F helper: the 2026 hard-negative contrastive-augmentation recipe
# ===========================================================================
def _run_hardneg(hardneg, encoders, policies, policy_bank,
                 Xc_tr, Y_tr, ih_tr, texts_tr,
                 Xc_te, Y_te, ih_te, seen_cols):
    """Run the full hard-negative pipeline and compare frozen vs adapter FPR.

    Steps (each links to its 2026 paper in hardneg.py):
      1. dense-mine benign look-alikes per policy (ANCE-style),
      2. ECIsem-score the mined negative set in the frozen geometry,
      3. build CausalNeg templated counterfactuals + ARHN false-negative filter,
      4. train the ContrastiveAdapter on frozen embeddings with the hard negs,
      5. compare FPR@recall0.90 of the FROZEN cosine vs the ADAPTER-projected
         cosine on held-out (TEST) positives vs mined hard negatives, per policy,
         averaged across the seen columns.
    """
    out = {
        "n_mined": 0,
        "eci": {"target_consistency": float("nan"), "locality": float("nan"),
                "lexical_residual": float("nan"), "diversity": float("nan"),
                "eci": float("nan")},
        "fpr_at_recall90": {"frozen_bi": float("nan"), "adapter": float("nan")},
        "delta": float("nan"),
        "n_counterfactuals": 0,
        "n_false_neg_dropped": 0,
    }

    # 1. dense mining on the TRAIN split -> {col: [benign row idx ...]}.
    mined = hardneg.mine_dense_hard_negatives(Xc_tr, policy_bank, ih_tr, seen_cols)
    out["n_mined"] = int(sum(len(v) for v in mined.values()))
    print("[hardneg] mined %d dense hard negatives across %d policies"
          % (out["n_mined"], len(mined)))

    # 2. ECIsem diagnostic, averaged over columns that have both pos and mined neg.
    eci_rows = []
    for col in seen_cols:
        neg_idx = list(mined.get(col, []))
        pos_idx = list(np.where(np.asarray(Y_tr)[:, col] > 0)[0])
        if len(pos_idx) < 2 or len(neg_idx) < 2:
            continue
        try:
            eci_rows.append(hardneg.eci_score(Xc_tr, policy_bank, pos_idx, neg_idx, col))
        except Exception as exc:
            print("[hardneg/eci col=%d] FAILED: %s" % (col, exc))
    if eci_rows:
        for k in out["eci"]:
            out["eci"][k] = float(np.nanmean([r.get(k, float("nan")) for r in eci_rows]))
        print("[hardneg] ECIsem eci=%.3f locality=%.3f target_consistency=%.3f"
              % (out["eci"]["eci"], out["eci"]["locality"], out["eci"]["target_consistency"]))

    # 3a. CausalNeg controlled counterfactuals (templated single-requirement swaps).
    try:
        cfs = hardneg.causal_counterfactuals(texts_tr, policies, seen_cols, n_per=3)
        out["n_counterfactuals"] = int(len(cfs))
        print("[hardneg] built %d CausalNeg counterfactuals" % out["n_counterfactuals"])
    except Exception as exc:
        print("[hardneg/causal] FAILED: %s" % exc)

    # 3b. ARHN false-negative filter: drop mined "negatives" that actually violate.
    dropped = 0
    for col in seen_cols:
        neg_idx = list(mined.get(col, []))
        if not neg_idx:
            continue
        neg_texts = [texts_tr[i] for i in neg_idx]
        try:
            keep = hardneg.arhn_false_negative_filter(neg_texts, policies[col])
            dropped += int(sum(1 for k in keep if not k))
        except Exception as exc:
            print("[hardneg/arhn col=%d] FAILED: %s" % (col, exc))
    out["n_false_neg_dropped"] = int(dropped)
    print("[hardneg] ARHN dropped %d probable false negatives" % dropped)

    # 4. Train the contrastive adapter on frozen embeddings + mined hard negatives.
    adapter = hardneg.ContrastiveAdapter()
    adapter.fit(Xc_tr, policy_bank, Y_tr, seen_cols, mined)

    # 5. FPR@recall0.90, frozen vs adapter, on TEST positives vs mined test negs.
    mined_te = hardneg.mine_dense_hard_negatives(Xc_te, policy_bank, ih_te, seen_cols)
    pb_adj = np.asarray(adapter.transform(policy_bank), dtype=float)
    Xc_te_adj = np.asarray(adapter.transform(Xc_te), dtype=float)
    frozen_fprs, adapter_fprs = [], []
    for col in seen_cols:
        pos_idx = list(np.where(np.asarray(Y_te)[:, col] > 0)[0])
        neg_idx = list(mined_te.get(col, []))
        if len(pos_idx) < 3 or len(neg_idx) < 3:
            continue
        idx = np.array(pos_idx + neg_idx)
        y = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
        s_frozen = Xc_te[idx] @ policy_bank[col]
        s_adapter = Xc_te_adj[idx] @ pb_adj[col]
        try:
            frozen_fprs.append(hardneg.fpr_at_recall(y, s_frozen, recall=0.90))
            adapter_fprs.append(hardneg.fpr_at_recall(y, s_adapter, recall=0.90))
        except Exception as exc:
            print("[hardneg/fpr col=%d] FAILED: %s" % (col, exc))
    if frozen_fprs and adapter_fprs:
        fb = float(np.nanmean(frozen_fprs))
        fa = float(np.nanmean(adapter_fprs))
        out["fpr_at_recall90"] = {"frozen_bi": fb, "adapter": fa}
        out["delta"] = float(fb - fa)  # positive delta => adapter cut false positives
        print("[hardneg] FPR@recall0.90  frozen=%.3f  adapter=%.3f  delta=%.3f"
              % (fb, fa, out["delta"]))
    return out


# ===========================================================================
# Orchestrator
# ===========================================================================
def main():
    # Lazy sibling imports (guarded HERE, not at module top, so `import
    # run_biencoder_guard` never triggers a model load or a dataset download).
    from . import data, encoders

    C.ARTIFACTS.mkdir(exist_ok=True)
    print("[cfg] embedder=%s emb_model=%s emb_dim=%d n_per_class=%d n_benign=%d seed=%d"
          % (C.EMBEDDER, C.EMBED_MODEL, C.EMB_DIM, C.N_PER_CLASS, C.N_BENIGN, C.SEED))

    # --- 1. Data -----------------------------------------------------------
    corpus = data.load_corpus()
    policies = corpus["policies"]
    texts = list(corpus["texts"])
    Y = np.asarray(corpus["Y"], dtype=np.float32)
    is_harmful = np.asarray(corpus["is_harmful"]).astype(int)
    P = len(policies)
    print("[data] corpus n=%d  n_policies=%d  harmful=%d benign=%d"
          % (len(texts), P, int(is_harmful.sum()), int((is_harmful == 0).sum())))

    split = data.split_seen_heldout(corpus)
    seen_cols = list(split["seen_cols"])
    heldout_cols = list(split["heldout_cols"])
    heldout_names = [policies[c]["name"] for c in heldout_cols]
    print("[data] seen_cols=%d  heldout_cols=%d (%s)"
          % (len(seen_cols), len(heldout_cols), ", ".join(heldout_names)))

    tr, te = data.group_train_test(corpus)
    tr = np.asarray(tr)
    te = np.asarray(te)
    print("[data] group split  train=%d  test=%d" % (len(tr), len(te)))

    # length-confound audit (can raw char length separate harmful vs benign?).
    try:
        conf = data.confound_report(texts, is_harmful)
    except Exception as exc:
        conf = {"length_auc": float("nan"), "len_pos_mean": float("nan"),
                "len_neg_mean": float("nan"), "error": str(exc)}
        print("[confound] FAILED: %s" % exc)
    print("[confound] length_auc=%.3f (0.5 => no trivial length tell)"
          % conf.get("length_auc", float("nan")))

    # --- 2. Encode (content tower) + build the cached policy tower ----------
    embedder = encoders.get_embedder()
    Xc_all = _encode_content_cached(embedder, texts, "train")  # whole corpus once
    Xc_tr, Xc_te = Xc_all[tr], Xc_all[te]
    Y_tr, Y_te = Y[tr], Y[te]
    ih_tr, ih_te = is_harmful[tr], is_harmful[te]
    texts_tr = [texts[i] for i in tr]
    texts_te = [texts[i] for i in te]
    policy_bank = _build_bank_cached(encoders, policies, embedder,
                                     n_proto=C.POLICY_PARAPHRASES, cache=True)

    # --- 3. Fit the three guards on the TRAIN split, SEEN cols -------------
    # Construction: only the uni_encoder needs the embedder (it re-encodes joint
    # (text, policy) strings). fit() takes `texts_train` for the same reason;
    # bi/head ignore it. The bi_encoder additionally CALIBRATES per-column F1
    # thresholds against the cached policy bank before we score EXP-A.
    guards = {}
    for method, ctor in (("bi_encoder", encoders.BiEncoderGuard),
                         ("uni_encoder", encoders.UniEncoderGuard),
                         ("trained_head", encoders.TrainedHeadGuard)):
        try:
            g = encoders.UniEncoderGuard(embedder=embedder) if method == "uni_encoder" else ctor()
            g.fit(Xc_tr, Y_tr, seen_cols, policies, texts_train=texts_tr)
            if method == "bi_encoder":
                try:
                    g.calibrate(Xc_tr, Y_tr, seen_cols, policy_bank)
                except Exception as exc:
                    print("[fit] bi_encoder calibrate skipped: %s" % exc)
            guards[method] = g
            print("[fit] %s ready" % method)
        except Exception as exc:
            guards[method] = None
            print("[fit] %s FAILED: %s" % (method, exc))

    results = {
        "embedder": str(C.EMBEDDER),
        "embed_model": str(C.EMBED_MODEL),
        "emb_dim": int(C.EMB_DIM),
        "n_policies": int(P),
        "seen_cols": [int(c) for c in seen_cols],
        "heldout_policies": [str(n) for n in heldout_names],
        "n_per_class": int(C.N_PER_CLASS),
        "n_benign": int(C.N_BENIGN),
        "seed": int(C.SEED),
        "judge": None,
        "confound": {
            "length_auc": float(conf.get("length_auc", float("nan"))),
            "len_pos_mean": float(conf.get("len_pos_mean", float("nan"))),
            "len_neg_mean": float(conf.get("len_neg_mean", float("nan"))),
        },
        "seen": {},
        "heldout_zeroshot": {},
        "multiproto_ablation": {},
        "scaling": {},
        "ood": {},
        "hardneg": {},
        "examples": [],
        "plots": [],
    }

    # cache the per-method SEEN scores (reused by the PR plot).
    seen_scores = {}
    Y_seen_te = Y_te[:, seen_cols]

    # --- EXP-A: seen-policy multilabel ------------------------------------
    for method, g in guards.items():
        if g is None:
            results["seen"][method] = {"error": "fit failed"}
            continue
        try:
            S = _guard_scores(g, Xc_te, texts_te, policy_bank, seen_cols)
            seen_scores[method] = S
            results["seen"][method] = _seen_metrics(encoders, Y_seen_te, S, ih_te)
            m = results["seen"][method]
            print("[EXP-A/%s] macro_ap=%.3f micro_ap=%.3f macro_f1=%.3f harm_auc=%.3f"
                  % (method, m["macro_ap"], m["micro_ap"], m["macro_f1"], m["binary_harm_auc"]))
        except Exception as exc:
            results["seen"][method] = {"error": str(exc)}
            print("[EXP-A/%s] FAILED: %s" % (method, exc))

    # --- EXP-B: held-out ZERO-SHOT (the headline) -------------------------
    Y_held_te = Y_te[:, heldout_cols]
    for method in ("bi_encoder", "uni_encoder"):
        g = guards.get(method)
        if g is None:
            results["heldout_zeroshot"][method] = {"error": "fit failed"}
            continue
        try:
            S = _guard_scores(g, Xc_te, texts_te, policy_bank, heldout_cols)
            mm = encoders.macro_micro(Y_held_te, S, thresholds=None)
            results["heldout_zeroshot"][method] = {
                "macro_ap": float(mm.get("macro_ap", float("nan"))),
                "macro_f1": float(mm.get("macro_f1", float("nan"))),
            }
            print("[EXP-B/%s] zero-shot macro_ap=%.3f macro_f1=%.3f"
                  % (method, results["heldout_zeroshot"][method]["macro_ap"],
                     results["heldout_zeroshot"][method]["macro_f1"]))
        except Exception as exc:
            results["heldout_zeroshot"][method] = {"error": str(exc)}
            print("[EXP-B/%s] FAILED: %s" % (method, exc))
    # trained_head structurally cannot score an unseen policy -> N/A.
    results["heldout_zeroshot"]["trained_head"] = "N/A"

    # --- EXP-C: 1-vs-P multi-prototype ablation (held-out cols) -----------
    if C.MULTIPROTO_ABLATION:
        try:
            g = guards.get("bi_encoder")
            if g is None:
                raise RuntimeError("bi_encoder not fitted")
            bank_single = _build_bank_cached(encoders, policies, embedder,
                                             n_proto=1, cache=False)
            S1 = _guard_scores(g, Xc_te, texts_te, bank_single, heldout_cols)
            SP = _guard_scores(g, Xc_te, texts_te, policy_bank, heldout_cols)
            ap1 = float(encoders.macro_micro(Y_held_te, S1).get("macro_ap", float("nan")))
            apP = float(encoders.macro_micro(Y_held_te, SP).get("macro_ap", float("nan")))
            results["multiproto_ablation"] = {
                "single": {"macro_ap": ap1},
                "multi": {"macro_ap": apP, "n_proto": int(C.POLICY_PARAPHRASES)},
            }
            print("[EXP-C] proto ablation  1-proto AP=%.3f  %d-proto AP=%.3f  delta=%.3f"
                  % (ap1, C.POLICY_PARAPHRASES, apP, apP - ap1))
        except Exception as exc:
            results["multiproto_ablation"] = {"error": str(exc)}
            print("[EXP-C] FAILED: %s" % exc)

    # --- EXP-D: latency vs #labels (the scaling claim) --------------------
    try:
        scale_texts = texts[:C.SCALE_BATCH]
        lat = encoders.scaling_latency(embedder, scale_texts, n_labels_grid=C.LABEL_SCALES)
        bi = lat.get("bi", {})
        uni = lat.get("uni", {})
        ks = sorted(int(k) for k in bi.keys())
        results["scaling"] = {
            "labels": ks,
            "bi_sec": [float(bi.get(k, bi.get(str(k), float("nan")))) for k in ks],
            "uni_sec": [float(uni.get(k, uni.get(str(k), float("nan")))) for k in ks],
        }
        print("[EXP-D] labels=%s bi=%s uni=%s"
              % (results["scaling"]["labels"],
                 ["%.3f" % v for v in results["scaling"]["bi_sec"]],
                 ["%.3f" % v for v in results["scaling"]["uni_sec"]]))
    except Exception as exc:
        results["scaling"] = {"error": str(exc)}
        print("[EXP-D] FAILED: %s" % exc)

    # --- EXP-E: OOD (score a disjoint slice over SEEN cols) ---------------
    try:
        ood = data.load_ood()
        ood_texts = list(ood["texts"])
        Y_ood = np.asarray(ood["Y"], dtype=np.float32)[:, seen_cols]
        ih_ood = np.asarray(ood["is_harmful"]).astype(int)
        Xc_ood = _encode_content_cached(embedder, ood_texts, "ood")
        results["ood"] = {"source": str(ood.get("source", "?")), "n": int(len(ood_texts))}
        print("[EXP-E] ood source=%s n=%d harmful=%d"
              % (results["ood"]["source"], results["ood"]["n"], int(ih_ood.sum())))
        for method, g in guards.items():
            if g is None:
                results["ood"][method] = {"error": "fit failed"}
                continue
            try:
                S = _guard_scores(g, Xc_ood, ood_texts, policy_bank, seen_cols)
                mm = encoders.macro_micro(Y_ood, S, thresholds=None)
                results["ood"][method] = {
                    "binary_harm_auc": float(encoders.binary_harm_auc(ih_ood, _any_policy_score(S))),
                    "macro_ap": float(mm.get("macro_ap", float("nan"))),
                }
                print("[EXP-E/%s] harm_auc=%.3f macro_ap=%.3f"
                      % (method, results["ood"][method]["binary_harm_auc"],
                         results["ood"][method]["macro_ap"]))
            except Exception as exc:
                results["ood"][method] = {"error": str(exc)}
                print("[EXP-E/%s] FAILED: %s" % (method, exc))
    except Exception as exc:
        results["ood"] = {"error": str(exc)}
        print("[EXP-E] FAILED: %s" % exc)

    # --- EXP-F: hard-negative contrastive augmentation --------------------
    if C.HARDNEG_MODULE:
        try:
            from . import hardneg
            results["hardneg"] = _run_hardneg(
                hardneg, encoders, policies, policy_bank,
                Xc_tr, Y_tr, ih_tr, texts_tr,
                Xc_te, Y_te, ih_te, seen_cols)
        except Exception as exc:
            results["hardneg"] = {"error": str(exc)}
            print("[EXP-F] FAILED: %s" % exc)

    # --- examples: one harmful + one benign, top matched policies (bi) -----
    try:
        results["examples"] = _build_examples(guards, policy_bank, Xc_te, texts_te,
                                              Y_te, seen_cols, heldout_cols, policies)
    except Exception as exc:
        print("[examples] FAILED: %s" % exc)

    # --- plots (each wrapped; failures do not block results.json) ---------
    plots = []
    # PR: flatten seen scores + labels over the seen columns.
    try:
        pr_flat = {}
        for method, S in seen_scores.items():
            pr_flat[method] = (Y_seen_te.reshape(-1), np.asarray(S).reshape(-1))
        if pr_flat:
            _plot_pr_by_method(pr_flat, C.PR_PNG)
            plots.append(str(C.PR_PNG))
    except Exception as exc:
        print("[plot:pr] FAILED: %s" % exc)
    for tag, png, fn in (
        ("heldout", C.HELDOUT_PNG,
         lambda: _plot_heldout_ap(results["heldout_zeroshot"],
                                  results.get("multiproto_ablation", {}), C.HELDOUT_PNG)),
        ("scaling", C.SCALE_PNG, lambda: _plot_latency(results.get("scaling", {}), C.SCALE_PNG)),
        ("hardneg", C.HARDNEG_PNG, lambda: _plot_hardneg(results.get("hardneg", {}), C.HARDNEG_PNG)),
    ):
        try:
            fn()
            plots.append(str(png))
        except Exception as exc:
            print("[plot:%s] FAILED: %s" % (tag, exc))
    results["plots"] = plots

    # --- write results.json BEFORE the summary print ----------------------
    with open(C.RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("[write] %s" % C.RESULTS_PATH)

    _print_summary(results)
    return results


def _build_examples(guards, policy_bank, Xc_te, texts_te, Y_te,
                    seen_cols, heldout_cols, policies):
    """One harmful + one benign test row: the top policies the bi-encoder matches.

    Also demonstrates the ZERO-SHOT point -- the same content vector, scored
    against a HELD-OUT policy that was never trained on, still fires. Uses raw
    cosine against the cached bank (the bi-encoder's core operation).
    """
    g = guards.get("bi_encoder")
    if g is None:
        return []
    labels = np.asarray(Y_te)
    ih = (labels.sum(1) > 0).astype(int)
    pos_i = next((i for i in range(len(texts_te)) if ih[i] == 1), None)
    neg_i = next((i for i in range(len(texts_te)) if ih[i] == 0), None)

    examples = []
    all_cols = list(seen_cols) + list(heldout_cols)
    for i in (pos_i, neg_i):
        if i is None:
            continue
        try:
            S = _guard_scores(g, Xc_te[i:i + 1], [texts_te[i]], policy_bank, all_cols)[0]
        except Exception:
            continue
        order = np.argsort(-np.asarray(S))[:3]
        top = [{"policy": str(policies[all_cols[j]]["name"]),
                "score": float(S[j]),
                "held_out": bool(all_cols[j] in heldout_cols)} for j in order]
        examples.append({
            "label": int(ih[i]),
            "text": str(texts_te[i])[:300],
            "top_policies": top,
        })
    return examples


# ===========================================================================
# ASCII summary (last)
# ===========================================================================
def _fmt(cell, keys, width=8):
    if not isinstance(cell, dict) or "error" in cell:
        return "  [N/A]"
    return "".join("%*.3f" % (width, cell.get(k, float("nan"))) for k in keys)


def _print_summary(results):
    line = "-" * 78
    print("")
    print(line)
    print("BI-ENCODER SAFETY GUARDRAIL  (SCREENING TIER, EmbeddingGemma dual-tower)")
    print("embedder=%s emb_dim=%d  n_policies=%d  seed=%d"
          % (results["embedder"], results["emb_dim"], results["n_policies"], results["seed"]))
    print("held-out (zero-shot) policies: %s" % ", ".join(results["heldout_policies"]))
    c = results.get("confound", {})
    print("length-confound AUC=%.3f (0.5 => no trivial length tell)"
          % c.get("length_auc", float("nan")))

    # EXP-A
    print(line)
    print("EXP-A  SEEN-policy multilabel (test)")
    print("%-14s %8s %8s %8s %8s %9s"
          % ("method", "macroAP", "microAP", "macroF1", "microF1", "harmAUC"))
    for m in C.METHODS:
        cell = results.get("seen", {}).get(m)
        if not isinstance(cell, dict) or "error" in cell:
            print("%-14s   [FAILED]" % m)
            continue
        print("%-14s %8.3f %8.3f %8.3f %8.3f %9.3f"
              % (m, cell["macro_ap"], cell["micro_ap"], cell["macro_f1"],
                 cell["micro_f1"], cell["binary_harm_auc"]))

    # EXP-B
    print(line)
    print("EXP-B  HELD-OUT zero-shot (the headline -- add a policy from a description)")
    print("%-14s %8s %8s" % ("method", "macroAP", "macroF1"))
    for m in ("bi_encoder", "uni_encoder"):
        cell = results.get("heldout_zeroshot", {}).get(m)
        if not isinstance(cell, dict) or "error" in cell:
            print("%-14s   [FAILED]" % m)
            continue
        print("%-14s %8.3f %8.3f" % (m, cell["macro_ap"], cell["macro_f1"]))
    print("%-14s   %s" % ("trained_head", "N/A (cannot score an unseen policy)"))

    # EXP-C
    mp = results.get("multiproto_ablation", {})
    if isinstance(mp, dict) and "single" in mp:
        print(line)
        print("EXP-C  multi-prototype ablation (held-out cols, bi_encoder)")
        print("  1-proto macroAP=%.3f   %d-proto macroAP=%.3f   delta=%+.3f"
              % (mp["single"]["macro_ap"], int(mp["multi"].get("n_proto", 0)),
                 mp["multi"]["macro_ap"], mp["multi"]["macro_ap"] - mp["single"]["macro_ap"]))

    # EXP-D
    sc = results.get("scaling", {})
    if isinstance(sc, dict) and sc.get("labels"):
        print(line)
        print("EXP-D  latency vs #labels (bi should stay FLAT, uni should RISE)")
        print("  labels : %s" % sc["labels"])
        print("  bi_sec : %s" % ["%.3f" % v for v in sc["bi_sec"]])
        print("  uni_sec: %s" % ["%.3f" % v for v in sc["uni_sec"]])

    # EXP-E
    ood = results.get("ood", {})
    if isinstance(ood, dict) and "source" in ood:
        print(line)
        print("EXP-E  OOD (%s, n=%d)  score over SEEN cols" % (ood.get("source", "?"), ood.get("n", 0)))
        print("%-14s %9s %8s" % ("method", "harmAUC", "macroAP"))
        for m in C.METHODS:
            cell = ood.get(m)
            if not isinstance(cell, dict) or "error" in cell:
                print("%-14s   [FAILED]" % m)
                continue
            print("%-14s %9.3f %8.3f" % (m, cell["binary_harm_auc"], cell["macro_ap"]))

    # EXP-F
    hn = results.get("hardneg", {})
    if isinstance(hn, dict) and "fpr_at_recall90" in hn:
        print(line)
        print("EXP-F  hard-negative augmentation (frozen bi vs contrastive adapter)")
        fpr = hn.get("fpr_at_recall90", {})
        print("  mined=%d  counterfactuals=%d  false_neg_dropped=%d  ECIsem.eci=%.3f"
              % (hn.get("n_mined", 0), hn.get("n_counterfactuals", 0),
                 hn.get("n_false_neg_dropped", 0), hn.get("eci", {}).get("eci", float("nan"))))
        print("  FPR@recall0.90  frozen=%.3f  adapter=%.3f  delta=%+.3f (positive => adapter helps)"
              % (fpr.get("frozen_bi", float("nan")), fpr.get("adapter", float("nan")),
                 hn.get("delta", float("nan"))))

    print(line)
    print("READ: bi_encoder caches the policy tower -> O(1) per request in #labels and "
          "scores UNSEEN policies zero-shot. FALSIFIERS -- (i) uni latency flat => scaling "
          "claim FALSE; (ii) bi held-out macroAP <= 0.5 => zero-shot claim FALSE; (iii) adapter "
          "does NOT lower FPR@recall0.90 => hard-negative sharpening does not help here.")
    print(line)


if __name__ == "__main__":
    main()
