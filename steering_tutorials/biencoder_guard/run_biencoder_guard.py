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
THE SEVEN EXPERIMENTS
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
  EXP-E  TRANSFER (3 arms)       : heldout_split (BeaverTails 30k_test -- rows only,
                                   NOT OOD), cross_annotator (Aegis 2.0 test), and
                                   ood_benchmark (intrinsec-ai/cstm-bench)
                                  over the SEEN cols -> binary harm AUC + macro
                                  AP. The real generalization check.
  EXP-F  hard-negative synthesis : the 2026 contrastive-augmentation recipe --
                                  dense-mine look-alike benigns -> ECIsem score
                                  them -> CausalNeg counterfactuals -> ARHN
                                  false-negative filter -> train a small
                                  ContrastiveAdapter -> does it cut FPR@recall
                                  0.90 vs the frozen bi-encoder on hard negatives?
  EXP-G  ACCURACY vs #labels     : the OTHER half of the arXiv:2602.18487 scaling
                                  claim. EXP-D shows latency stays flat; this asks
                                  whether ACCURACY survives. Pad the 16 real
                                  policies with deterministic synthetic DISTRACTOR
                                  policies (distractors.py) to K in {16,64,256,900},
                                  score the test set against all K, and measure ONLY
                                  the 16 real columns. Per-column AP/F1 are invariant
                                  in K by construction, so the claim is judged on the
                                  COMPETITIVE metrics: rank of the true policy among
                                  all K, top-T routing F1, and argmax steal rate.
                                  (Requested as "EXP-F"; that tag was already used by
                                  the hard-negative module, hence EXP-G.)
  EXP-H  the OPIR TAXONOMY test  : EXP-G's competitors are UNRELATED by construction
                                  (disjoint domain, harm-stem blocklist, Jaccard <=
                                  0.20 vs every real phrasing). Opir (arXiv:2605.29659)
                                  trains on "a three-level taxonomy containing 996
                                  categories across 16 top-level labels, 126 mid-level
                                  labels, and 854 leaf labels", where a label competes
                                  against its own SIBLINGS and DESCENDANTS. taxonomy.py
                                  builds that shape below the 16 real policies, and
                                  EXP-H re-runs EXP-G's protocol against it -- changing
                                  exactly ONE thing, the RELATEDNESS of the competitors.
                                  Because a DESCENDANT outranking its parent is an
                                  over-specification rather than a mis-route, every
                                  rank violation is classified descendant / sibling /
                                  other-branch, against the share a RANDOM violator
                                  would produce.

Detection task -> NO generation judge (results["judge"] is null). CPU-only from
the orchestrator's view: the embedder loads inside the encoders module. Sibling
modules (data / encoders / hardneg) are imported LAZILY inside main() so
`python -c "import ...run_biencoder_guard"` succeeds while they are still stubs.
Stdout is ASCII only (Windows cp1252): we write "cos" / "AP" / "FPR" / ">=",
never unicode. results.json is written BEFORE the summary print; each EXP and
each plot is wrapped so a late failure still leaves results.json on disk.
"""
from __future__ import annotations

import hashlib
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
def _content_fingerprint(texts) -> str:
    """SHA-256 over the ORDERED texts. Order matters: rows index into this matrix."""
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _bank_fingerprint(policies, n_proto) -> str:
    """SHA-256 over the ORDERED policy TEXTS -- the thing the bank actually encodes."""
    h = hashlib.sha256()
    h.update(("n_proto=%d|" % int(n_proto)).encode("utf-8"))
    for p in policies:
        h.update(str(p.get("id", "")).encode("utf-8", "replace"))
        h.update(b"\x1f")
        h.update(str(p.get("description", "")).encode("utf-8", "replace"))
        h.update(b"\x1f")
        for para in p.get("paraphrases", []):
            h.update(str(para).encode("utf-8", "replace"))
            h.update(b"\x1e")
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _encode_content_cached(embedder, texts, split):
    """Embed a batch of texts with the CONTENT tower, cached to disk per split.

    The corpus texts are the SAME across the seen / held-out experiments (only
    the policy COLUMNS differ), so we embed the whole corpus once under the
    "train" cache key and index it by the train / test row splits. Each transfer
    arm is a genuinely different set and gets its own cache key.

    STAMPED, not counted. This used to validate on `X.shape[0] == len(texts)`
    alone, so any change that preserved the row count -- a new dataset mix, a
    reshuffled seed -- silently reused vectors belonging to different text. That
    is verbatim CLAUDE.md section 18.8's "the embedding cache returned stale labels
    under a key that ignored them". The cache now carries a fingerprint of the
    ordered texts and is REJECTED on mismatch, including when it predates
    fingerprinting and cannot prove anything about itself.
    """
    key = (split, C.EMBEDDER)
    path = C.EMB_CACHE.get(key)
    fp = _content_fingerprint(texts)
    if path is not None and path.exists():
        try:
            blob = np.load(path)
            X = blob["X"].astype(np.float32)
            cached_fp = str(blob["fp"]) if "fp" in blob.files else None
            if cached_fp is None:
                print("[embed] cache REJECTED %s: no fingerprint (predates stamping); "
                      "re-encoding" % path.name)
            elif cached_fp != fp:
                print("[embed] cache REJECTED %s: fingerprint %s != %s (the texts "
                      "changed); re-encoding" % (path.name, cached_fp, fp))
            elif X.shape[0] != len(texts):
                print("[embed] cache REJECTED %s: %d rows != %d texts; re-encoding"
                      % (path.name, X.shape[0], len(texts)))
            else:
                print("[embed] loaded cache %s  shape=%s fp=%s"
                      % (path.name, X.shape, fp))
                return X
        except Exception as exc:
            print("[embed] cache reload failed (%s); re-encoding" % exc)
    X = np.asarray(embedder.encode(list(texts), "content"), dtype=np.float32)
    if path is not None:
        try:
            np.savez_compressed(path, X=X, fp=np.array(fp))
        except Exception as exc:
            print("[embed] cache save failed: %s" % exc)
    print("[embed] encoded %d texts -> shape=%s (split=%s fp=%s)"
          % (len(texts), X.shape, split, fp))
    return X


def _build_bank_cached(encoders, policies, embedder, n_proto, cache):
    """Build the POLICY tower ([P, dim]) with `n_proto` prototypes per policy.

    The main multi-prototype bank (n_proto == POLICY_PARAPHRASES) is cached to
    disk (it is reused by every experiment); the single-prototype ablation bank
    is cheap and built fresh. This is the tower we cache ONCE and match every
    incoming text against -- the crux of the bi-encoder scaling story.
    """
    path = C.POLICY_CACHE.get(C.EMBEDDER)
    fp = _bank_fingerprint(policies, n_proto)
    if cache and path is not None and path.exists():
        try:
            blob = np.load(path)
            B = blob["B"].astype(np.float32)
            cached_fp = str(blob["fp"]) if "fp" in blob.files else None
            # A row-count check on 16 rows guards nothing: edit a policy DESCRIPTION
            # or a paraphrase and the count is unchanged, so every cosine in
            # EXP-A/B/C/E/G/H would be against the old text. Fingerprint the TEXTS.
            if cached_fp is None:
                print("[bank] cache REJECTED %s: no fingerprint (predates stamping); "
                      "rebuilding" % path.name)
            elif cached_fp != fp:
                print("[bank] cache REJECTED %s: fingerprint %s != %s (a policy "
                      "description or paraphrase changed); rebuilding"
                      % (path.name, cached_fp, fp))
            elif B.shape[0] != len(policies):
                print("[bank] cache REJECTED %s: %d rows != %d policies; rebuilding"
                      % (path.name, B.shape[0], len(policies)))
            else:
                print("[bank] loaded cache %s  shape=%s fp=%s" % (path.name, B.shape, fp))
                return B
        except Exception as exc:
            print("[bank] cache reload failed (%s); rebuilding" % exc)
    B = np.asarray(encoders.build_policy_bank(policies, embedder, n_proto=n_proto),
                   dtype=np.float32)
    if cache and path is not None:
        try:
            np.savez_compressed(path, B=B, fp=np.array(fp))
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


def _plot_label_scale(block, out_path):
    """Accuracy vs #labels: the ranking metrics that CAN move, for both bi arms.

    Left  -- mean absolute rank of the true policy among all K candidates, drawn
             against the (K+1)/2 CHANCE line so the reader can separate "the bank
             got bigger" from "the encoder got worse".
    Right -- distractor steal rate at top-1: how often a synthetic distractor
             out-scores EVERY real policy on a genuinely harmful row.

    Two metrics are deliberately NOT plotted. Per-column macro-AP/F1 are invariant
    in K by construction, so a flat line would be arithmetic rather than evidence.
    macro_f1_top_t is confounded -- distractors consume top-T slots, cutting the
    number of real predictions per row and raising precision for free -- so
    plotting it would advertise an artifact as a scaling win.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = block.get("arms", {})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    styles = {"bi_encoder": ("o-", "tab:green"), "bi_encoder_trained": ("s-", "tab:blue")}
    chance_ks, chance_v = [], []
    for arm, rows in arms.items():
        if not isinstance(rows, list) or not rows:
            continue
        ks = [r["n_labels"] for r in rows]
        st, col = styles.get(arm, ("^-", "tab:gray"))
        axes[0].plot(ks, [r["mean_rank_true"] for r in rows], st, color=col, label=arm)
        axes[1].plot(ks, [r["distractor_steal_top1"] for r in rows], st, color=col, label=arm)
        chance_ks, chance_v = ks, [r["mean_rank_chance"] for r in rows]
    if chance_ks:
        axes[0].plot(chance_ks, chance_v, "k--", alpha=0.5, label="chance ((K+1)/2)")
        axes[0].set_yscale("log")
    for ax, ylab, title in (
        (axes[0], "mean rank of the true policy (log, lower better)",
         "Rank of the TRUE policy vs bank size"),
        (axes[1], "P(a distractor beats every real policy)",
         "Top-1 steal rate on harmful rows"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of policy labels K (16 real + distractors)")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Large-label-scale ACCURACY (metrics over the 16 REAL columns only)",
                 fontsize=11)
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
    #    The adapter projects vectors of the content/policy embedding dimension, so it
    #    must be constructed with that dim (Xc_tr is [n_train, dim]).
    adapter = hardneg.ContrastiveAdapter(dim=Xc_tr.shape[1])
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
# EXP-G helper: ACCURACY under LARGE-LABEL SCALING (the paper's 2nd claim)
# ===========================================================================
def _distractor_bank_cached(encoders, distractors, embedder, n_needed, policies):
    """Build (and cache) the synthetic distractor half of the policy tower.

    Cached per embedder, and the cache carries the generator FINGERPRINT: if the
    distractor list changes (new grid, new seed, different count) the fingerprint
    moves and the stale matrix is REJECTED rather than silently reused. An
    artifact that cannot be tied to the code beside it is not evidence.
    """
    built = distractors.build_distractors(policies, n_needed)
    dpols = built["policies"]
    fp = built["audit"]["fingerprint"]

    path = C.DISTRACTOR_CACHE.get(C.EMBEDDER)
    if path is not None and path.exists():
        try:
            data = np.load(path, allow_pickle=False)
            B = data["B"].astype(np.float32)
            cached_fp = str(data["fingerprint"].item()) if "fingerprint" in data else ""
            if B.shape[0] == len(dpols) and cached_fp == fp:
                print("[distract] loaded cache %s shape=%s fp=%s" % (path.name, B.shape, fp))
                return B, dpols, built["audit"]
            print("[distract] cache REJECTED (shape %s vs %d, fp %r vs %r); rebuilding"
                  % (B.shape, len(dpols), cached_fp, fp))
        except Exception as exc:
            print("[distract] cache reload failed (%s); rebuilding" % exc)

    t0 = time.perf_counter()
    B = np.asarray(encoders.build_policy_bank(dpols, embedder, n_proto=C.LABEL_SCALE_PROTO),
                   dtype=np.float32)
    print("[distract] embedded %d distractor policies x %d prototypes in %.1fs -> %s"
          % (len(dpols), C.LABEL_SCALE_PROTO, time.perf_counter() - t0, B.shape))
    if path is not None:
        try:
            np.savez_compressed(path, B=B, fingerprint=np.array(fp))
        except Exception as exc:
            print("[distract] cache save failed: %s" % exc)
    return B, dpols, built["audit"]


def _competition_ranks(S):
    """Rank every column per row, 1 = highest score. Returns int32 [n, K].

    Ties are broken by column index, and the 16 REAL columns occupy indices
    0..15, so a tie is resolved in the REAL policy's favour. With float cosines
    exact ties are effectively nonexistent; the convention is stated so the
    (mildly optimistic) direction of the bias is on the record rather than hidden.
    """
    S = np.asarray(S, dtype=np.float32)
    n, K = S.shape
    order = np.argsort(-S, axis=1, kind="stable")          # best -> worst
    ranks = np.empty((n, K), dtype=np.int32)
    np.put_along_axis(ranks, order, np.arange(1, K + 1, dtype=np.int32)[None, :], axis=1)
    return ranks


def _label_scale_metrics(encoders, Y_real, S_K, n_real, top_t, seen_pos):
    """Metrics for ONE arm at ONE label count K, scored over the REAL columns only.

    TWO FAMILIES, and the distinction is the whole point of the experiment:

    * PER-COLUMN metrics (macro_ap, macro_f1 at a fixed per-policy threshold) ask
      "does policy j fire on this text?" independently of every other column. For
      a bi-encoder that score is cosine(content, bank[j]) -- a function of column j
      ALONE -- so adding distractor columns cannot change it. These are therefore
      EXACTLY invariant in K, by construction rather than by measurement, and the
      runner asserts that invariance instead of presenting it as a finding.
    * COMPETITIVE metrics (rank of the true policy among all K, top-T routing F1,
      argmax steal rate) ask "which of the K policies does this text match best?"
      -- the retrieval-shaped decision a production router actually makes. THESE
      can degrade as distractors are added, so this is where the paper's
      "accuracy is maintained at scale" claim carries empirical content.
    """
    from sklearn.metrics import f1_score

    S_K = np.asarray(S_K, dtype=np.float32)
    Y_real = np.asarray(Y_real)
    K = S_K.shape[1]
    S_real = S_K[:, :n_real]

    # --- per-column family (K-invariant by construction) ---
    mm = encoders.macro_micro(Y_real, S_real, thresholds=None)
    out = {
        "n_labels": int(K),
        "n_distractors": int(K - n_real),
        "macro_ap": float(mm.get("macro_ap", float("nan"))),
        "macro_f1": float(mm.get("macro_f1", float("nan"))),
        "micro_ap": float(mm.get("micro_ap", float("nan"))),
    }
    # macro-AP restricted to the SEEN columns, so this row is directly comparable
    # to EXP-A (which scores seen columns only).
    if seen_pos:
        mm_seen = encoders.macro_micro(Y_real[:, seen_pos], S_real[:, seen_pos], thresholds=None)
        out["macro_ap_seen_cols"] = float(mm_seen.get("macro_ap", float("nan")))
    else:
        out["macro_ap_seen_cols"] = float("nan")

    # --- competitive family (this is what K can move) ---
    ranks = _competition_ranks(S_K)
    r_real = ranks[:, :n_real]
    pos = Y_real > 0
    if pos.any():
        rp = r_real[pos].astype(float)
        out["mean_rank_true"] = float(np.mean(rp))
        out["median_rank_true"] = float(np.median(rp))
        # scale-free: 0.0 => the true policy is always ranked first among all K.
        out["mean_pct_rank"] = float(np.mean((rp - 1.0) / max(K - 1, 1)))
        out["n_positive_pairs"] = int(pos.sum())
    else:
        out.update({"mean_rank_true": float("nan"), "median_rank_true": float("nan"),
                    "mean_pct_rank": float("nan"), "n_positive_pairs": 0})

    # chance calibration for the absolute rank: a random bank ranks the true policy
    # at (K+1)/2 on average, so the raw mean rank MUST be read against it.
    out["mean_rank_chance"] = float((K + 1) / 2.0)
    out["rank_lift_vs_chance"] = (float(out["mean_rank_true"] / out["mean_rank_chance"])
                                  if out["mean_rank_true"] == out["mean_rank_true"] else float("nan"))

    # top-T routing: flag policy j iff it is among the T best of ALL K candidates.
    pred = (r_real <= int(top_t)).astype(int)
    f1s = []
    for j in range(n_real):
        y = Y_real[:, j].astype(int)
        if y.sum() == 0:
            continue                      # undefined column -> excluded, as elsewhere
        f1s.append(f1_score(y, pred[:, j], zero_division=0))
    out["macro_f1_top_t"] = float(np.mean(f1s)) if f1s else float("nan")

    # CONFOUND DIAGNOSTIC -- read this before reading macro_f1_top_t.
    # At K = n_real the top-T rule always fires on exactly T REAL columns. As
    # distractors enter, some of those T slots go to distractors, so the rule
    # predicts FEWER real columns per row. Where precision is the binding
    # constraint (T predictions for ~1-2 true labels), that pruning can RAISE F1
    # even as the underlying ranking gets worse -- an improvement that is an
    # artifact of the decision rule, not of the encoder. These three fields expose
    # the mechanism so macro_f1_top_t can never be read naively as "accuracy held".
    out["mean_real_predicted_per_row"] = float(pred.sum(axis=1).mean())
    tp = float((pred * (Y_real > 0)).sum())
    out["micro_precision_top_t"] = float(tp / pred.sum()) if pred.sum() else float("nan")
    out["micro_recall_top_t"] = (float(tp / (Y_real > 0).sum())
                                 if (Y_real > 0).sum() else float("nan"))

    # how often does a DISTRACTOR win the argmax on a row that really is harmful?
    rows = np.where(pos.any(axis=1))[0]
    if len(rows):
        top1 = np.argmin(ranks[rows], axis=1)
        out["distractor_steal_top1"] = float(np.mean(top1 >= n_real))
    else:
        out["distractor_steal_top1"] = float("nan")
    return out


def _run_label_scale_accuracy(encoders, distractors, guards, policies, embedder,
                              policy_bank, Xc_te, texts_te, Y_te, seen_cols):
    """EXP-G: does bi-encoder ACCURACY survive a ~900-label policy bank?

    Method. Pad the 16 real policies with synthetic distractors (distractors.py,
    fixed seed, four disjointness guards) to K in C.LABEL_SCALE_ACC_K. Score the
    test set against the full bank, then compute every metric over the FIRST 16
    (real) columns only -- distractors compete for rank but are never scored as
    answers, so the numbers stay on the same footing as EXP-A/EXP-B.

    Efficiency + exactness. The bank is [16 real | 884 distractors] in a FIXED
    order, so the K-label condition is the first K columns of the K_max score
    matrix. We therefore score ONCE at K_max and SLICE. That is not an
    approximation: it is the same matrix the K-th run would have produced, and it
    also guarantees the nested-scale property (no resampling between K values).
    """
    n_real = len(policies)
    # K = n_real is the UNPADDED row -- the no-distractor baseline every other K is
    # read against, and the row EXP-H's head-to-head looks up by `n_labels == n_real`.
    # Pooling Aegis with extra columns moves n_real off 16, so it is added
    # explicitly rather than left to a hardcoded grid that no longer contains it.
    ks = sorted({int(k) for k in C.LABEL_SCALE_ACC_K} | {n_real})
    dropped = [k for k in ks if k < n_real]
    ks = [k for k in ks if k >= n_real]
    if dropped:
        print("[EXP-G] dropped K < n_real=%d from the grid: %s" % (n_real, dropped))
    if not ks:
        raise ValueError("LABEL_SCALE_ACC_K has no K >= n_real=%d" % n_real)
    k_max = ks[-1]
    n_needed = k_max - n_real

    D, dpols, audit = _distractor_bank_cached(encoders, distractors, embedder,
                                              n_needed, policies)
    real_bank = np.asarray(policy_bank, dtype=np.float32)
    full_bank = np.vstack([real_bank, D]).astype(np.float32)
    assert full_bank.shape[0] == k_max, "bank assembly produced %d rows, expected %d" % (
        full_bank.shape[0], k_max)

    # G4: report-only proximity audit of the distractors in the encoder's geometry.
    audit = dict(audit)
    audit["embedding"] = distractors.embedding_audit(D, real_bank)
    ea = audit["embedding"]
    print("[EXP-G] distractor proximity: mean_max_cos=%.3f p95=%.3f max=%.3f "
          "(n>=0.90: %d, n>=0.95: %d)"
          % (ea.get("mean_max_cos_to_real", float("nan")),
             ea.get("p95_max_cos_to_real", float("nan")),
             ea.get("max_max_cos_to_real", float("nan")),
             ea.get("n_above_0.90", -1), ea.get("n_above_0.95", -1)))

    Y_real = np.asarray(Y_te)
    seen_pos = [int(c) for c in seen_cols if 0 <= int(c) < n_real]

    block = {
        "k_grid": [int(k) for k in ks],
        "n_real_policies": int(n_real),
        "top_t": int(C.LABEL_SCALE_TOP_T),
        "n_test_rows": int(Xc_te.shape[0]),
        "distractor_proto": int(C.LABEL_SCALE_PROTO),
        "distractors": audit,
        "arms": {},
        "notes": {},
    }

    for arm in C.LABEL_SCALE_ARMS:
        g = guards.get(arm)
        if g is None:
            block["arms"][arm] = {"error": "fit failed"}
            continue
        try:
            t0 = time.perf_counter()
            S_max = _guard_scores(g, Xc_te, texts_te, full_bank, list(range(k_max)))
            print("[EXP-G/%s] scored %d texts x %d labels in %.2fs"
                  % (arm, S_max.shape[0], S_max.shape[1], time.perf_counter() - t0))
            rows = []
            for k in ks:
                rows.append(_label_scale_metrics(encoders, Y_real, S_max[:, :k],
                                                 n_real, C.LABEL_SCALE_TOP_T, seen_pos))
            block["arms"][arm] = rows
            for r in rows:
                print("[EXP-G/%s] K=%4d  macroAP=%.3f  top%dF1=%.3f (P=%.3f R=%.3f "
                      "nreal/row=%.2f)  meanRank=%.2f (chance %.1f, lift %.3f)  steal@1=%.3f"
                      % (arm, r["n_labels"], r["macro_ap"], C.LABEL_SCALE_TOP_T,
                         r["macro_f1_top_t"], r["micro_precision_top_t"],
                         r["micro_recall_top_t"], r["mean_real_predicted_per_row"],
                         r["mean_rank_true"], r["mean_rank_chance"],
                         r["rank_lift_vs_chance"], r["distractor_steal_top1"]))
        except Exception as exc:
            block["arms"][arm] = {"error": str(exc)}
            print("[EXP-G/%s] FAILED: %s" % (arm, exc))

    # ASSERT THE ANCHOR. The per-column metrics MUST be bit-identical across K for
    # these arms (cosine against column j ignores every other column). If they ever
    # differ, the scoring path has picked up a cross-column dependency -- a bug that
    # would otherwise masquerade as "accuracy degrades at scale". We verify rather
    # than assume, and we record the verdict so a reader can see it was checked.
    inv_ok, inv_detail = True, {}
    for arm, rows in block["arms"].items():
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        aps = [r["macro_ap"] for r in rows]
        f1s = [r["macro_f1"] for r in rows]
        spread = max(max(aps) - min(aps), max(f1s) - min(f1s))
        inv_detail[arm] = float(spread)
        if spread > 1e-9:
            inv_ok = False
    block["notes"]["per_column_invariance_verified"] = bool(inv_ok)
    block["notes"]["per_column_max_spread_over_k"] = inv_detail
    block["notes"]["per_column_invariance_explanation"] = (
        "macro_ap / macro_f1 are computed per policy column from cosine(content, "
        "bank[j]), which does not depend on any other column, so they are EXACTLY "
        "invariant in K for both bi-encoder arms. Their flatness is arithmetic, not "
        "evidence. The paper's 'accuracy maintained at scale' claim is only "
        "falsifiable under a COMPETITIVE decision rule, which is what mean_rank_true, "
        "mean_pct_rank, macro_f1_top_t and distractor_steal_top1 measure.")
    block["notes"]["top_t_f1_confound"] = (
        "macro_f1_top_t is NOT a clean scaling metric. The top-T rule emits exactly T "
        "predictions per row; at K=n_real all T land on real columns, but as distractors "
        "enter they consume slots, so mean_real_predicted_per_row falls and precision "
        "rises for free. macro_f1_top_t can therefore INCREASE with K while the ranking "
        "underneath degrades. Read it only alongside mean_real_predicted_per_row, "
        "micro_precision_top_t and micro_recall_top_t. The uncontaminated degradation "
        "measures are distractor_steal_top1 and mean_rank_true vs mean_rank_chance.")
    print("[EXP-G] per-column invariance in K verified=%s (max spread %s)"
          % (inv_ok, {k: "%.2e" % v for k, v in inv_detail.items()}))
    return block


# ===========================================================================
# EXP-H helper: the OPIR TAXONOMY test -- 996 RELATED categories
# ===========================================================================
# EXP-G asked whether the guard survives ~900 STRANGERS. EXP-H asks the harder
# question the Opir paper's label space actually poses: can it still find the right
# policy when the ~980 competitors are the true policy's own SIBLINGS and
# DESCENDANTS -- plausible near-misses generated by narrowing a real policy rather
# than by moving to a disjoint domain? See taxonomy.py's header for the contrast.
# ===========================================================================
def _taxonomy_bank_cached(encoders, taxonomy, embedder, policies):
    """Build (and cache) the 980 mid+leaf vectors of the Opir-shaped taxonomy.

    Same cache discipline as EXP-G: the generator FINGERPRINT is stored with the
    matrix and a mismatch REJECTS the cache. An artifact that cannot be tied to the
    code beside it is not evidence.
    """
    built = taxonomy.build_taxonomy(policies)
    nodes = built["nodes"]
    fp = built["audit"]["fingerprint"]

    path = C.TAXONOMY_CACHE.get(C.EMBEDDER)
    if path is not None and path.exists():
        try:
            data = np.load(path, allow_pickle=False)
            B = data["B"].astype(np.float32)
            cached_fp = str(data["fingerprint"].item()) if "fingerprint" in data else ""
            if B.shape[0] == len(nodes) and cached_fp == fp:
                print("[taxo] loaded cache %s shape=%s fp=%s" % (path.name, B.shape, fp))
                return B, nodes, built["audit"]
            print("[taxo] cache REJECTED (shape %s vs %d, fp %r vs %r); rebuilding"
                  % (B.shape, len(nodes), cached_fp, fp))
        except Exception as exc:
            print("[taxo] cache reload failed (%s); rebuilding" % exc)

    t0 = time.perf_counter()
    B = np.asarray(encoders.build_policy_bank(nodes, embedder, n_proto=C.OPIR_PROTO),
                   dtype=np.float32)
    print("[taxo] embedded %d taxonomy nodes x %d prototypes in %.1fs -> %s"
          % (len(nodes), C.OPIR_PROTO, time.perf_counter() - t0, B.shape))
    if path is not None:
        try:
            np.savez_compressed(path, B=B, fingerprint=np.array(fp))
        except Exception as exc:
            print("[taxo] cache save failed: %s" % exc)
    return B, nodes, built["audit"]


def _opir_arm_metrics(encoders, Y_real, S, owner, level, n_real, seen_pos):
    """Every EXP-H number for ONE arm at ONE bank size K, over the REAL columns only.

    THE ERROR TAXONOMY. For each (row, true top-level policy j) pair we take the rank
    of column j among all K, and CLASSIFY every column that outranks it:

      DESCENDANT   owner[c] == j and level[c] > 0  -- a narrower rule under the SAME
                   policy. Routing here is an over-specification, not a mis-route;
                   `mean_rank_excl_descendants` is the rank with these removed.
      SIBLING      level[c] == 0, c != j           -- another TOP-LEVEL policy. A real
                   mis-route, and the same error EXP-A/EXP-G's K=16 row measures.
      OTHER-BRANCH owner[c] != j and level[c] > 0  -- a cousin: some other policy's
                   mid/leaf. The closest analogue of an EXP-G distractor stealing rank.

    Every share is reported against the share a RANDOM violator would produce
    (`*_share_chance`), because the three buckets have wildly different sizes: a
    policy has ~61 descendants but ~919 cousins, so a raw "most violations are
    cousins" would be a statement about bucket sizes, not about the encoder. The
    ENRICHMENT ratio (observed / chance) is the quantity with content.
    """
    from sklearn.metrics import f1_score

    S = np.asarray(S, dtype=np.float32)
    Y_real = np.asarray(Y_real)
    K = S.shape[1]
    owner = np.asarray(owner)[:K]
    level = np.asarray(level)[:K]
    ranks = _competition_ranks(S)                   # [n, K], 1 = best

    out = {
        "n_labels": int(K),
        "n_competitors": int(K - n_real),
        "n_mid": int((level == 1).sum()),
        "n_leaf": int((level == 2).sum()),
    }

    # --- per-column family: invariant in K by construction (see EXP-G) -------
    S_real = S[:, :n_real]
    mm = encoders.macro_micro(Y_real, S_real, thresholds=None)
    out["macro_ap"] = float(mm.get("macro_ap", float("nan")))
    out["macro_f1"] = float(mm.get("macro_f1", float("nan")))
    if seen_pos:
        mm_seen = encoders.macro_micro(Y_real[:, seen_pos], S_real[:, seen_pos],
                                       thresholds=None)
        out["macro_ap_seen_cols"] = float(mm_seen.get("macro_ap", float("nan")))
    else:
        out["macro_ap_seen_cols"] = float("nan")

    # --- competitive family, with the violation breakdown -------------------
    r_all, n_desc_all, n_sib_all, n_oth_all = [], [], [], []
    exp_desc, exp_sib, exp_oth = [], [], []          # per-pair CHANCE shares
    n_desc_avail = []
    for j in range(n_real):
        rows = np.where(Y_real[:, j] > 0)[0]
        if len(rows) == 0:
            continue
        desc_mask = (owner == j) & (level > 0)
        sib_mask = (level == 0).copy()
        sib_mask[j] = False
        oth_mask = (owner != j) & (level > 0)
        rj = ranks[rows, j].astype(np.int64)
        viol = ranks[rows, :] < rj[:, None]           # [m, K] bool
        nd = viol[:, desc_mask].sum(axis=1)
        ns = viol[:, sib_mask].sum(axis=1)
        no = viol[:, oth_mask].sum(axis=1)
        r_all.append(rj)
        n_desc_all.append(nd)
        n_sib_all.append(ns)
        n_oth_all.append(no)
        n_desc_avail.append(np.full(len(rows), int(desc_mask.sum()), dtype=np.int64))
        denom = float(max(K - 1, 1))
        exp_desc.append(np.full(len(rows), desc_mask.sum() / denom))
        exp_sib.append(np.full(len(rows), sib_mask.sum() / denom))
        exp_oth.append(np.full(len(rows), oth_mask.sum() / denom))

    if r_all:
        r = np.concatenate(r_all).astype(float)
        nd = np.concatenate(n_desc_all).astype(float)
        ns = np.concatenate(n_sib_all).astype(float)
        no = np.concatenate(n_oth_all).astype(float)
        navail = np.concatenate(n_desc_avail).astype(float)
        ed = np.concatenate(exp_desc)
        es = np.concatenate(exp_sib)
        eo = np.concatenate(exp_oth)
        nviol = r - 1.0
        tot = float(nviol.sum())
        out.update({
            "n_positive_pairs": int(len(r)),
            "mean_rank_true": float(r.mean()),
            "median_rank_true": float(np.median(r)),
            "mean_pct_rank": float(np.mean((r - 1.0) / max(K - 1, 1))),
            "mean_rank_chance": float((K + 1) / 2.0),
            "rank_lift_vs_chance": float(r.mean() / ((K + 1) / 2.0)),
            "frac_rank_1": float((r == 1).mean()),
            # violations per positive pair, by bucket
            "viol_per_pair": float(nviol.mean()),
            "viol_descendant_per_pair": float(nd.mean()),
            "viol_sibling_per_pair": float(ns.mean()),
            "viol_other_branch_per_pair": float(no.mean()),
            # POOLED shares of all violations (these must sum to 1)
            "viol_descendant_share": float(nd.sum() / tot) if tot else float("nan"),
            "viol_sibling_share": float(ns.sum() / tot) if tot else float("nan"),
            "viol_other_branch_share": float(no.sum() / tot) if tot else float("nan"),
            # the chance shares those must be read against (bucket sizes differ ~15x)
            "viol_descendant_share_chance": (float((nviol * ed).sum() / tot)
                                             if tot else float("nan")),
            "viol_sibling_share_chance": (float((nviol * es).sum() / tot)
                                          if tot else float("nan")),
            "viol_other_branch_share_chance": (float((nviol * eo).sum() / tot)
                                               if tot else float("nan")),
            # THE DESCENDANT CORRECTION: a child outranking its parent is arguably not
            # an error, so this is the rank after deleting the truth's own descendants.
            "mean_rank_excl_descendants": float((r - nd).mean()),
            "mean_pct_rank_excl_descendants": float(
                np.mean((r - nd - 1.0) / np.maximum(K - 1.0 - navail, 1.0))),
            "frac_rank_1_excl_descendants": float(((r - nd) == 1).mean()),
        })
        for key in ("descendant", "sibling", "other_branch"):
            obs = out["viol_%s_share" % key]
            chc = out["viol_%s_share_chance" % key]
            out["viol_%s_enrichment" % key] = (float(obs / chc) if chc and chc == chc
                                               and chc > 0 else float("nan"))
    else:
        out["n_positive_pairs"] = 0

    # --- top-1: what actually wins the argmax on a genuinely harmful row ----
    pos_rows = np.where((Y_real > 0).any(axis=1))[0]
    if len(pos_rows):
        top1 = np.argmin(ranks[pos_rows], axis=1)
        own_top = owner[top1]
        lvl_top = level[top1]
        is_true_owner = Y_real[pos_rows, own_top] > 0
        b_true = (lvl_top == 0) & is_true_owner
        b_desc = (lvl_top > 0) & is_true_owner
        b_sib = (lvl_top == 0) & ~is_true_owner
        b_oth = (lvl_top > 0) & ~is_true_owner
        n = float(len(pos_rows))
        # CHANCE baseline for the top-1 breakdown, computed PER ROW from that row's own
        # true-label set: a uniformly random argmax lands on a descendant of the truth
        # with probability (number of descendants of this row's true policies)/K. The
        # descendant bucket is only ~6% of the bank, so an observed descendant share far
        # above that is the concentration the raw number alone cannot demonstrate.
        n_desc_of = np.array([int(((owner == j) & (level > 0)).sum())
                              for j in range(n_real)], dtype=float)
        Yp = Y_real[pos_rows] > 0
        n_true = Yp.sum(axis=1).astype(float)
        c_true = n_true / K
        c_desc = (Yp * n_desc_of[None, :]).sum(axis=1) / K
        c_sib = (n_real - n_true) / K
        c_oth = 1.0 - c_true - c_desc - c_sib
        out.update({
            "n_harmful_rows": int(n),
            "top1_true_policy": float(b_true.sum() / n),
            "top1_descendant_of_true": float(b_desc.sum() / n),
            "top1_sibling_top_level": float(b_sib.sum() / n),
            "top1_other_branch": float(b_oth.sum() / n),
            "top1_true_policy_chance": float(c_true.mean()),
            "top1_descendant_of_true_chance": float(c_desc.mean()),
            "top1_sibling_top_level_chance": float(c_sib.mean()),
            "top1_other_branch_chance": float(c_oth.mean()),
            "top1_true_policy_enrichment": float((b_true.sum() / n) / max(c_true.mean(), 1e-12)),
            "top1_descendant_enrichment": float((b_desc.sum() / n) / max(c_desc.mean(), 1e-12)),
            "top1_sibling_enrichment": float((b_sib.sum() / n) / max(c_sib.mean(), 1e-12)),
            "top1_other_branch_enrichment": float((b_oth.sum() / n) / max(c_oth.mean(), 1e-12)),
            # naive steal rate: anything but the true TOP-LEVEL column counts as a steal
            "steal_top1": float(1.0 - b_true.sum() / n),
            # forgiving steal rate: routing to a descendant of the truth is NOT an error
            "steal_top1_excl_descendants": float((b_sib.sum() + b_oth.sum()) / n),
            # THE ONE THAT IS COMPARABLE TO EXP-G. EXP-G's `distractor_steal_top1` is
            # P(argmax lands on a SYNTHETIC column), which treats a wrong REAL policy
            # winning as "not a steal". `steal_top1` above uses a different (stricter)
            # rule, so putting the two side by side would compare definitions rather
            # than encoders. This field reproduces EXP-G's rule exactly on EXP-H's
            # bank -- P(argmax is any added mid/leaf) -- and it is the field the
            # head-to-head uses.
            "competitor_steal_top1": float((b_desc.sum() + b_oth.sum()) / n),
            # ...and the same rule with descendant wins forgiven.
            "competitor_steal_top1_excl_descendants": float(b_oth.sum() / n),
        })

    # --- the 16-TOP-LEVEL-ONLY restriction (comparable to EXP-A / EXP-G K=16) --
    ranks16 = _competition_ranks(S_real)
    pos = Y_real > 0
    if pos.any():
        out["mean_rank_true_top16"] = float(ranks16[pos].astype(float).mean())
        t1 = np.argmin(ranks16[pos_rows], axis=1) if len(pos_rows) else np.array([])
        out["top1_acc_top16"] = (float((Y_real[pos_rows, t1] > 0).mean())
                                 if len(pos_rows) else float("nan"))
    # top-T routing over the full bank, with the EXP-G confound diagnostic attached.
    pred = (ranks[:, :n_real] <= int(C.LABEL_SCALE_TOP_T)).astype(int)
    f1s = []
    for j in range(n_real):
        y = Y_real[:, j].astype(int)
        if y.sum() == 0:
            continue
        f1s.append(f1_score(y, pred[:, j], zero_division=0))
    out["macro_f1_top_t"] = float(np.mean(f1s)) if f1s else float("nan")
    out["mean_real_predicted_per_row"] = float(pred.sum(axis=1).mean())
    return out


def _gpu_witness() -> dict:
    """Who else was on the GPU when we measured? Recorded beside EXP-D's latencies.

    WHY THIS EXISTS. EXP-D is the only block in this lesson whose numbers are wall-clock
    rather than deterministic, and on a single shared card they move a lot: across two
    runs of the identical code `bi_encoder` ranged 0.39 -> 0.69 s and `uni_encoder`
    15.8 -> 27.0 s purely because another job was resident. That is survivable for the
    RATIO (contention scales both arms by nearly the same factor) but fatal for the
    absolute seconds -- and a published "64x" in this README turned out to be a
    *contended* uni-encoder divided by an *uncontended* bi-encoder, two machine states
    inside one ratio, with no artifact behind it.

    A timing that does not record its own machine state cannot be compared to another
    timing, and cannot be caught when it silently disagrees. So we stamp it: how many
    compute processes were resident and how much memory was in use. Best-effort -- if
    nvidia-smi is absent this returns {"available": False} rather than failing the run.
    """
    import subprocess

    out = {"available": False}
    try:
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        mem = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        lines = [l.strip() for l in procs.stdout.splitlines() if l.strip()]
        names = [l.split(",", 1)[1].strip() if "," in l else l for l in lines]
        # our own process is one of these; anything else is a co-tenant.
        others = [n for n in names if "python" in n.lower()]
        fields = [f.strip() for f in mem.stdout.strip().splitlines()[0].split(",")] \
            if mem.stdout.strip() else []
        out = {
            "available": True,
            "n_compute_processes": len(lines),
            "n_python_processes": len(others),
            "concurrent_python_jobs_excluding_self": max(len(others) - 1, 0),
            "memory_used_mib": int(float(fields[0])) if len(fields) > 0 else None,
            "memory_total_mib": int(float(fields[1])) if len(fields) > 1 else None,
            "utilization_pct": int(float(fields[2])) if len(fields) > 2 else None,
            "contended": bool(max(len(others) - 1, 0) > 0),
        }
    except Exception as exc:
        out = {"available": False, "error": str(exc)}
    return out


def _embedder_precision(embedder) -> str:
    """The backbone's parameter dtype, e.g. "torch.float32" (or "unknown").

    Recorded in the EXP-H artifact ON PURPOSE. EXP-H's whole claim is a head-to-head
    against EXP-G in which exactly ONE thing differs -- whether the ~900 competitors
    are related to the truth. The content embeddings, the 16-policy bank and the
    884-distractor bank on disk are fp32, so building the taxonomy bank at a different
    precision would put a SECOND uncontrolled variable inside that comparison. It was
    held fixed deliberately, and a future reader tempted to re-run this in fp16 for
    speed needs to see that it was a decision rather than an accident.
    """
    try:
        model = getattr(embedder, "_model", None)
        if model is None:
            return "unknown"
        for p in model.parameters():
            return str(p.dtype)
    except Exception:
        pass
    return "unknown"


def _run_opir_taxonomy(encoders, taxonomy, guards, policies, embedder,
                       policy_bank, Xc_te, texts_te, Y_te, seen_cols, exp_g_block):
    """EXP-H: 996 RELATED categories (Opir's 16/126/854 shape) vs EXP-G's strangers.

    Method. taxonomy.py narrows each of the 16 real policies into 7-8 mids and each mid
    into 6-7 leaves -> 980 competitors, every one a sibling or descendant of a real
    policy. Stack [16 real | 126 mid | 854 leaf] into a 996-column bank, score the test
    set once, and read the metrics over the 16 real columns -- identical scoring
    protocol to EXP-G, so the ONLY thing that changed is the RELATEDNESS of the
    competitors. That single-variable design is what makes the head-to-head legible.
    """
    n_real = len(policies)
    D, nodes, audit = _taxonomy_bank_cached(encoders, taxonomy, embedder, policies)
    real_bank = np.asarray(policy_bank, dtype=np.float32)
    full_bank = np.vstack([real_bank, D]).astype(np.float32)
    K = full_bank.shape[0]
    if K != C.OPIR_TOTAL:
        raise ValueError("assembled %d columns, expected %d" % (K, C.OPIR_TOTAL))

    owner, level = taxonomy.relation_arrays(policies, nodes)
    audit = dict(audit)
    audit["embedding"] = taxonomy.proximity_audit(D, real_bank, owner[n_real:])
    ea = audit["embedding"]
    print("[EXP-H] node proximity (report-only): cos to OWN parent %.3f vs best OTHER "
          "%.3f (%.1f%% closer to own); max cos to any real policy: mean %.3f p95 %.3f"
          % (ea.get("mean_cos_to_own_parent", float("nan")),
             ea.get("mean_cos_to_best_other_parent", float("nan")),
             100 * ea.get("frac_closer_to_own_parent", float("nan")),
             ea.get("mean_max_cos_to_real", float("nan")),
             ea.get("p95_max_cos_to_real", float("nan"))))

    Y_real = np.asarray(Y_te)
    seen_pos = [int(c) for c in seen_cols if 0 <= int(c) < n_real]
    k_match = int(C.OPIR_MATCH_K)

    block = {
        "n_labels": int(K),
        "n_real_policies": int(n_real),
        "n_mid": int(C.OPIR_N_MID),
        "n_leaf": int(C.OPIR_N_LEAF),
        "match_k": k_match,
        "top_t": int(C.LABEL_SCALE_TOP_T),
        "n_test_rows": int(Xc_te.shape[0]),
        "taxonomy": audit,
        "precision": _embedder_precision(embedder),
        "precision_note": (
            "The taxonomy bank is embedded at the SAME precision as the content "
            "embeddings, the 16-policy bank and EXP-G's 884-distractor bank already on "
            "disk. This is deliberate and load-bearing: EXP-H's only claim is a "
            "head-to-head against EXP-G in which exactly ONE variable moves (are the "
            "~900 competitors related to the truth or not). Re-running the taxonomy "
            "bank in fp16 for speed would introduce a second uncontrolled variable "
            "into that comparison while saving memory the 300M encoder does not need. "
            "If this field ever disagrees with the precision of the banks it is scored "
            "against, the EXP-G/EXP-H head-to-head is void."),
        "arms": {},
        "matched_k": {},
        "vs_exp_g": {},
        "notes": {},
    }

    for arm in C.OPIR_ARMS:
        g = guards.get(arm)
        if g is None:
            block["arms"][arm] = {"error": "fit failed"}
            continue
        try:
            t0 = time.perf_counter()
            S = _guard_scores(g, Xc_te, texts_te, full_bank, list(range(K)))
            print("[EXP-H/%s] scored %d texts x %d related labels in %.2fs"
                  % (arm, S.shape[0], S.shape[1], time.perf_counter() - t0))
            block["arms"][arm] = _opir_arm_metrics(encoders, Y_real, S, owner, level,
                                                   n_real, seen_pos)
            # exact-K slice for the head-to-head against EXP-G's 900-column bank.
            if k_match < K:
                block["matched_k"][arm] = _opir_arm_metrics(
                    encoders, Y_real, S[:, :k_match], owner[:k_match], level[:k_match],
                    n_real, seen_pos)
            r = block["arms"][arm]
            print("[EXP-H/%s] K=%d  meanRank=%.2f (chance %.1f, lift %.3f)  "
                  "exclDescendants=%.2f  steal@1=%.3f (excl desc %.3f)"
                  % (arm, K, r["mean_rank_true"], r["mean_rank_chance"],
                     r["rank_lift_vs_chance"], r["mean_rank_excl_descendants"],
                     r["steal_top1"], r["steal_top1_excl_descendants"]))
            print("[EXP-H/%s] violations: descendant %.3f (chance %.3f, %.1fx)  "
                  "sibling %.3f (chance %.3f, %.1fx)  other-branch %.3f (chance %.3f, %.1fx)"
                  % (arm, r["viol_descendant_share"], r["viol_descendant_share_chance"],
                     r["viol_descendant_enrichment"], r["viol_sibling_share"],
                     r["viol_sibling_share_chance"], r["viol_sibling_enrichment"],
                     r["viol_other_branch_share"], r["viol_other_branch_share_chance"],
                     r["viol_other_branch_enrichment"]))
        except Exception as exc:
            block["arms"][arm] = {"error": str(exc)}
            print("[EXP-H/%s] FAILED: %s" % (arm, exc))

    # --- head-to-head against EXP-G: RELATED vs UNRELATED competitors -------
    # Same arms, same test rows, same scoring protocol, same 16 scored columns. The
    # bank sizes differ (996 vs 900), so the primary comparison is the SCALE-FREE
    # percentile rank; the exact-K slice is reported beside it so neither the reader
    # nor the author has to take the normalisation on trust.
    g_rows = (exp_g_block or {}).get("arms", {})
    for arm in C.OPIR_ARMS:
        h = block["arms"].get(arm)
        hm = block["matched_k"].get(arm)
        rows = g_rows.get(arm)
        if not isinstance(h, dict) or "mean_rank_true" not in h or not isinstance(rows, list):
            continue
        g = max((r for r in rows if isinstance(r, dict)),
                key=lambda r: r.get("n_labels", 0), default=None)
        if g is None:
            continue
        cmp = {
            "exp_g_k": int(g["n_labels"]),
            "exp_h_k": int(h["n_labels"]),
            "exp_g_mean_rank": float(g["mean_rank_true"]),
            "exp_h_mean_rank": float(h["mean_rank_true"]),
            "exp_g_pct_rank": float(g["mean_pct_rank"]),
            "exp_h_pct_rank": float(h["mean_pct_rank"]),
            "pct_rank_ratio_h_over_g": float(h["mean_pct_rank"] / g["mean_pct_rank"])
                                       if g["mean_pct_rank"] else float("nan"),
            # SAME RULE ON BOTH SIDES: P(the argmax lands on an ADDED column). EXP-G
            # calls it distractor_steal_top1; EXP-H's competitor_steal_top1 is the
            # identical predicate on the taxonomy bank. `steal_top1` (any non-true
            # column, including a wrong real policy) is a DIFFERENT rule and is
            # reported separately -- never against EXP-G.
            "exp_g_competitor_steal_top1": float(g["distractor_steal_top1"]),
            "exp_h_competitor_steal_top1": float(h["competitor_steal_top1"]),
            "exp_h_competitor_steal_top1_excl_descendants":
                float(h["competitor_steal_top1_excl_descendants"]),
            "exp_h_steal_top1_any_wrong_column": float(h["steal_top1"]),
            "exp_g_macro_ap": float(g["macro_ap"]),
            "exp_h_macro_ap": float(h["macro_ap"]),
            "definition_note": (
                "steal rates compared here use EXP-G's rule on both sides: the share of "
                "harmful rows whose argmax over the whole bank is one of the ADDED "
                "columns. EXP-H's stricter steal_top1 (any column that is not a true "
                "top-level policy) is carried as exp_h_steal_top1_any_wrong_column and "
                "must NOT be read against the EXP-G number."),
        }
        if isinstance(hm, dict) and "mean_rank_true" in hm:
            cmp["exp_h_mean_rank_at_matched_k"] = float(hm["mean_rank_true"])
            cmp["exp_h_competitor_steal_top1_at_matched_k"] = float(
                hm["competitor_steal_top1"])
            cmp["mean_rank_ratio_h_over_g_matched_k"] = (
                float(hm["mean_rank_true"] / g["mean_rank_true"])
                if g["mean_rank_true"] else float("nan"))
        block["vs_exp_g"][arm] = cmp
        print("[EXP-H/%s] vs EXP-G: meanRank %.2f (K=%d unrelated) -> %.2f (K=%d related, "
              "%.2f at matched K=%d)  competitor-steal@1 %.3f -> %.3f (%.3f excl desc)"
              % (arm, cmp["exp_g_mean_rank"], cmp["exp_g_k"], cmp["exp_h_mean_rank"],
                 cmp["exp_h_k"], cmp.get("exp_h_mean_rank_at_matched_k", float("nan")),
                 k_match, cmp["exp_g_competitor_steal_top1"],
                 cmp["exp_h_competitor_steal_top1"],
                 cmp["exp_h_competitor_steal_top1_excl_descendants"]))

    # --- ASSERT THE CROSS-EXPERIMENT ANCHORS -------------------------------
    # Two quantities MUST be identical between EXP-G and EXP-H, because both are
    # functions of the 16 real columns alone and neither experiment touches those:
    #   (1) macro-AP over the real columns (per-column cosine -- the EXP-G tautology),
    #   (2) the mean rank of the true policy RESTRICTED to the 16 top-level columns.
    # If either drifts, the two experiments are not scoring the same thing and no
    # head-to-head is legitimate. We check rather than assume, and record the verdict.
    anchors = {}
    ok = True
    for arm in C.OPIR_ARMS:
        h = block["arms"].get(arm)
        rows = g_rows.get(arm)
        if not isinstance(h, dict) or "macro_ap" not in h or not isinstance(rows, list):
            continue
        g16 = next((r for r in rows if r.get("n_labels") == n_real), None)
        d_ap = abs(h["macro_ap"] - rows[0]["macro_ap"])
        d_r16 = (abs(h.get("mean_rank_true_top16", float("nan")) - g16["mean_rank_true"])
                 if g16 else float("nan"))
        anchors[arm] = {"macro_ap_delta_vs_exp_g": float(d_ap),
                        "top16_mean_rank_delta_vs_exp_g": float(d_r16)}
        if d_ap > 1e-9 or (d_r16 == d_r16 and d_r16 > 1e-6):
            ok = False
    block["notes"]["cross_experiment_anchor_verified"] = bool(ok)
    block["notes"]["cross_experiment_anchor_deltas"] = anchors
    block["notes"]["uninformative_by_construction"] = (
        "macro_ap / macro_f1 / macro_ap_seen_cols are per-column cosine metrics -- "
        "cos(content, bank[j]) ignores every other column -- so they are EXACTLY the "
        "EXP-A and EXP-G values and cannot move when 980 related labels are added. "
        "mean_rank_true_top16 and top1_acc_top16 are likewise identical to EXP-G's "
        "K=16 row: restricting to the 16 real columns deletes every competitor both "
        "experiments added. All five are reported ONLY as anchors proving the two "
        "experiments score the same thing; none of them is evidence about relatedness. "
        "macro_f1_top_t inherits EXP-G's top-T confound (competitors consume top-T "
        "slots, so real predictions per row fall and precision rises for free) and is "
        "excluded from the plot for the same reason.")
    block["notes"]["descendant_semantics"] = (
        "A DESCENDANT outranking its parent is not obviously an error: the router has "
        "chosen a narrower rule under the SAME policy, which is an over-specification "
        "rather than a mis-route. steal_top1_excl_descendants and "
        "mean_rank_excl_descendants price it as correct; steal_top1 and mean_rank_true "
        "price it as an error. Both are reported because the right answer depends on "
        "whether the downstream system acts on the top-level policy or on the leaf.")
    print("[EXP-H] cross-experiment anchors verified=%s %s"
          % (ok, {a: "AP d=%.1e r16 d=%.1e" % (v["macro_ap_delta_vs_exp_g"],
                                               v["top16_mean_rank_delta_vs_exp_g"])
                  for a, v in anchors.items()}))
    return block


def _plot_opir(block, exp_g_block, out_path):
    """EXP-H: related competitors vs EXP-G's unrelated ones, and where the errors go.

    Left  -- mean rank of the true policy: EXP-G at its largest K (unrelated) beside
             EXP-H at K=996 (related) and EXP-H with the truth's own DESCENDANTS
             removed, per arm. The third bar is the honest one if a descendant win is
             not counted as an error.
    Right -- the TOP-1 breakdown: on a genuinely harmful row, what wins the argmax --
             the true top-level policy, a DESCENDANT of it, a SIBLING top-level policy,
             or an unrelated cousin -- each beside the share a UNIFORMLY RANDOM argmax
             would give. The chance bars are not decoration: descendants are only ~6% of
             the 996 columns, so "half the wins are descendants" is only meaningful
             against that 6%.

    The full-violation-pool breakdown (viol_*_share) is NOT plotted: ~92% of the bank is
    other-branch cousins, so those bars are a picture of bucket sizes with the signal
    compressed into invisibility. It stays in results.json with its chance columns.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in C.OPIR_ARMS if isinstance(block.get("arms", {}).get(a), dict)
            and "mean_rank_true" in block["arms"][a]]
    if not arms:
        raise RuntimeError("no EXP-H arm produced metrics")
    g_rows = (exp_g_block or {}).get("arms", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(arms), dtype=float)
    w = 0.26

    def _g_rank(arm):
        rows = g_rows.get(arm)
        if not isinstance(rows, list) or not rows:
            return float("nan")
        r = max((q for q in rows if isinstance(q, dict)),
                key=lambda q: q.get("n_labels", 0))
        return float(r.get("mean_rank_true", float("nan")))

    g_vals = [_g_rank(a) for a in arms]
    h_vals = [block["arms"][a]["mean_rank_true"] for a in arms]
    h_excl = [block["arms"][a]["mean_rank_excl_descendants"] for a in arms]
    axes[0].bar(x - w, g_vals, w, color="tab:gray", label="EXP-G K=900 (UNRELATED)")
    axes[0].bar(x, h_vals, w, color="tab:red", label="EXP-H K=996 (RELATED)")
    axes[0].bar(x + w, h_excl, w, color="tab:orange",
                label="EXP-H, own descendants removed")
    for xi, v in zip(np.concatenate([x - w, x, x + w]), g_vals + h_vals + h_excl):
        if v == v:
            axes[0].text(xi, v, "%.1f" % v, ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(arms, fontsize=8)
    axes[0].set_ylabel("mean rank of the true policy (lower better)", fontsize=8)
    axes[0].set_title("Related vs unrelated competitors", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, axis="y", alpha=0.3)

    keys = [("top1_true_policy", "true\ntop-level"),
            ("top1_descendant_of_true", "DESCENDANT\nof the truth"),
            ("top1_sibling_top_level", "sibling\n(other top-level)"),
            ("top1_other_branch", "other branch\n(cousin)")]
    xb = np.arange(len(keys), dtype=float)
    wb = 0.8 / (2 * len(arms))
    colors = {"bi_encoder": "tab:green", "bi_encoder_trained": "tab:blue"}
    for i, arm in enumerate(arms):
        r = block["arms"][arm]
        if "top1_true_policy" not in r:
            continue
        obs = [r[k] for k, _ in keys]
        chc = [r[k + "_chance"] for k, _ in keys]
        axes[1].bar(xb + (2 * i - len(arms)) * wb, obs, wb,
                    color=colors.get(arm, "tab:purple"), label="%s observed" % arm)
        axes[1].bar(xb + (2 * i + 1 - len(arms)) * wb, chc, wb, color="none",
                    edgecolor=colors.get(arm, "tab:purple"), hatch="//",
                    label="%s chance (random argmax)" % arm)
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels([lab for _, lab in keys], fontsize=8)
    axes[1].set_ylabel("share of harmful rows", fontsize=8)
    axes[1].set_title("What wins the argmax among all 996", fontsize=10)
    axes[1].legend(fontsize=6)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("EXP-H: the Opir 3-level taxonomy (16 top / 126 mid / 854 leaf = 996)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


class _SkipExperiment(Exception):
    """Raised to leave an experiment block with a STATED reason, not an error."""


def _opir_fit_shape(n_top: int) -> dict:
    """Keep Opir's 996-category TOTAL exact when the real taxonomy is not 16 wide.

    Opir (arXiv:2605.29659) reports "996 categories across 16 top-level labels, 126
    mid-level labels, and 854 leaf labels". `taxonomy.build_taxonomy` ASSERTS that
    sum rather than trusting it, which is correct -- and which means it raises the
    moment pooling Aegis adds policy columns and n_top leaves 16.

    Two honest options, and silently mis-shaping the taxonomy is neither:
      * AUTOFIT (default): hold the 996 TOTAL and the 126:854 ratio, re-deriving
        mid/leaf at the new n_top. The total is the number the paper's scaling claim
        is about; the exact 126/854 split is not. The deviation is recorded in
        results.json under `opir_shape` and printed, never assumed.
      * OFF: skip EXP-H and say why.

    Mutates C.OPIR_N_MID / C.OPIR_N_LEAF, which `taxonomy` reads at call time.
    Returns the shape record (paper's numbers AND ours).
    """
    paper = {"n_top": int(C.OPIR_PAPER_TOP), "n_mid": 126, "n_leaf": 854,
             "n_total": int(C.OPIR_TOTAL)}
    if int(n_top) == int(C.OPIR_PAPER_TOP):
        return {"autofit": False, "matches_paper": True, "paper": paper,
                "used": {"n_top": int(n_top), "n_mid": int(C.OPIR_N_MID),
                         "n_leaf": int(C.OPIR_N_LEAF), "n_total": int(C.OPIR_TOTAL)}}
    if not C.OPIR_AUTOFIT:
        return {"autofit": False, "matches_paper": False, "paper": paper, "skip": True,
                "reason": ("the real taxonomy has %d top-level policies, not %d, so "
                           "Opir's 16/126/854=996 split is not exact. Set "
                           "BG_OPIR_AUTOFIT=1 to hold the 996 total and re-derive "
                           "mid/leaf, or BG_AEGIS_EXTRA=0 to keep 16 columns."
                           % (n_top, C.OPIR_PAPER_TOP))}
    below = int(C.OPIR_TOTAL) - int(n_top)
    if below < 2:
        return {"autofit": False, "matches_paper": False, "paper": paper, "skip": True,
                "reason": "n_top=%d leaves %d nodes below it; too few to build a "
                          "3-level taxonomy." % (n_top, below)}
    ratio = 126.0 / (126.0 + 854.0)
    n_mid = max(1, int(round(below * ratio)))
    n_leaf = below - n_mid
    C.OPIR_N_MID, C.OPIR_N_LEAF = int(n_mid), int(n_leaf)
    used = {"n_top": int(n_top), "n_mid": int(n_mid), "n_leaf": int(n_leaf),
            "n_total": int(n_top + n_mid + n_leaf)}
    print("[EXP-H] OPIR SHAPE AUTOFIT: n_top=%d (paper: %d) -> mid=%d leaf=%d, "
          "total=%d held at Opir's 996. The 126:854 RATIO is preserved; the exact "
          "126/854 split is NOT the paper's."
          % (n_top, C.OPIR_PAPER_TOP, n_mid, n_leaf, used["n_total"]))
    return {"autofit": True, "matches_paper": False, "paper": paper, "used": used,
            "note": "total held at Opir's 996; mid/leaf re-derived at the new n_top "
                    "by the paper's 126:854 ratio"}


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

    # --- what the corpus ACHIEVED, not what it was asked for ---------------
    # results.json used to write `n_per_class: 500` straight off the config while
    # six of sixteen columns held 109-374 positives, so the rubric failure was
    # invisible from the artifact. Recompute WITH the test split so the per-column
    # eval counts are on the record too, and warn loudly on any shortfall.
    achieved = data.corpus_provenance(corpus, C.N_PER_CLASS, C.N_BENIGN, test_idx=te)
    print(data.format_shortfall(achieved))

    # --- confound audit: FOUR bars, directionless, from the shared spine ----
    # length / word-count / TF-IDF content / label-shuffle, each folded about 0.5.
    # Run on the whole corpus (comparable to prior runs) AND on the TEST split,
    # because the test split is where the methods are scored and therefore where
    # the binding bar has to come from.
    def _confound(tag, sub_texts, sub_labels):
        try:
            rep = data.confound_report(sub_texts, sub_labels, seed=C.SEED)
            print("[confound:%s]\n%s" % (tag, data.format_confound_report(rep)))
            return rep
        except Exception as exc:
            print("[confound:%s] FAILED: %s" % (tag, exc))
            return {"length_auc": float("nan"), "len_pos_mean": float("nan"),
                    "len_neg_mean": float("nan"), "worst_name": "none",
                    "worst_auc": 0.5, "error": str(exc)}

    conf = _confound("corpus", texts, is_harmful)
    conf_test = _confound("test", [texts[i] for i in te], [is_harmful[i] for i in te])

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
                         ("bi_encoder_trained", encoders.ContrastiveBiEncoderGuard),
                         ("uni_encoder", encoders.UniEncoderGuard),
                         ("trained_head", encoders.TrainedHeadGuard)):
        try:
            if method == "bi_encoder_trained":
                # The papers' actual method: a CONTRASTIVELY TRAINED projection over the
                # frozen backbone. Trained on SEEN policies only, and it still scores an
                # unseen policy from that policy's TEXT -- so the held-out evaluation
                # remains a real zero-shot test. `bi_encoder` (frozen cosine) is kept
                # beside it as the ablation it actually is.
                g = encoders.ContrastiveBiEncoderGuard(embedder=embedder, policies=policies)
                g.fit(Xc_tr, Y_tr, policy_bank, seen_cols)
                guards[method] = g
                print("[fit] %s ready (InfoNCE, final loss %.4f)" % (method, g.final_loss))
                continue
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
        # REQUESTED config. The achieved counts live under "achieved" -- these two
        # are named apart on purpose; conflating them is what made the rubric
        # failure unauditable from the artifact (AUDIT_2026-08.md section A2).
        "n_per_class_requested": int(C.N_PER_CLASS),
        "n_benign_requested": int(C.N_BENIGN),
        # legacy keys, kept so an older reader does not KeyError -- but they are the
        # REQUESTED values and must not be read as compliance claims.
        "n_per_class": int(C.N_PER_CLASS),
        "n_benign": int(C.N_BENIGN),
        "seed": int(C.SEED),
        "judge": None,
        # --- ACHIEVED: per-column positives (corpus AND test), realised benign
        # count, class balance, measured source distribution, pool fingerprint,
        # and a per-column requested-vs-achieved shortfall flag.
        "achieved": achieved,
        "source_distribution": achieved.get("source_distribution", {}),
        "pool_fingerprint": achieved.get("pool_fingerprint", ""),
        "requested_vs_achieved": achieved.get("requested_vs_achieved", {}),
        "datasets": {
            "beavertails": {"id": C.BEAVERTAILS_DATASET,
                            "split": C.BEAVERTAILS_TRAIN_SPLIT, "gated": False},
            "aegis": {"id": C.AEGIS_DATASET, "split": C.AEGIS_TRAIN_SPLIT,
                      "gated": False, "enabled": bool(C.AEGIS_ON),
                      "extra_columns": bool(C.AEGIS_EXTRA_COLUMNS),
                      "stats": corpus.get("aegis", {})},
            "toxicchat": {"id": C.TOXICCHAT_DATASET, "config": C.TOXICCHAT_CONFIG,
                          "gated": False},
            "wildguardmix": {"id": C.WILDGUARD_DATASET, "gated": True,
                             "note": "GATED; HTTP 403 without an HF token. Contributes "
                                     "0 rows on this host -- see source_distribution."},
        },
        "confound": {
            # folded (directionless) length AUC, plus the raw value it folds from
            "length_auc": float(conf.get("length_auc", float("nan"))),
            "length_auc_raw": float(conf.get("length_auc_raw", float("nan"))),
            "len_pos_mean": float(conf.get("len_pos_mean", float("nan"))),
            "len_neg_mean": float(conf.get("len_neg_mean", float("nan"))),
            "corpus": conf,
            "test": conf_test,
            "binding_bar": float(conf_test.get("worst_auc", 0.5)),
            "binding_bar_name": str(conf_test.get("worst_name", "none")),
        },
        "margins": {},
        "seen": {},
        "heldout_zeroshot": {},
        "multiproto_ablation": {},
        "scaling": {},
        "transfer": {},
        "ood": {},
        "hardneg": {},
        "label_scale_accuracy": {},
        "opir_taxonomy": {},
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

    # --- the only number a method may headline: margin over the BINDING bar --
    # CLAUDE.md section 17 rule 7: claim only the margin ABOVE the larger of
    # {confound bar, method baseline}. The bar comes from the TEST split, which is
    # where binary_harm_auc is measured. A negative margin is printed, not hidden.
    for method, cell in results["seen"].items():
        if not isinstance(cell, dict) or "binary_harm_auc" not in cell:
            continue
        try:
            m = data.margin_over_bar(float(cell["binary_harm_auc"]), conf_test)
            results["margins"][method] = m
            print("[margin/%s] harm_auc=%.4f  bar(%s)=%.4f  margin=%+.4f  clears=%s"
                  % (method, m["method_auc"], m["binding_bar_name"],
                     m["binding_bar"], m["margin"], m["clears"]))
        except Exception as exc:
            results["margins"][method] = {"error": str(exc)}

    # --- EXP-B: held-out ZERO-SHOT (the headline) -------------------------
    Y_held_te = Y_te[:, heldout_cols]
    # bi_encoder_trained belongs here above all: the whole claim of a bi-encoder guard
    # is that it handles an UNSEEN policy from that policy's text. A trained
    # projection that only worked on seen policies would be trained_head with extra
    # steps, so this is the arm's real test, not a bonus column.
    for method in ("bi_encoder", "bi_encoder_trained", "uni_encoder"):
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
        witness_before = _gpu_witness()
        lat = encoders.scaling_latency(embedder, scale_texts, n_labels_grid=C.LABEL_SCALES)
        witness_after = _gpu_witness()
        bi = lat.get("bi", {})
        uni = lat.get("uni", {})
        ks = sorted(int(k) for k in bi.keys())
        bi_sec = [float(bi.get(k, bi.get(str(k), float("nan")))) for k in ks]
        uni_sec = [float(uni.get(k, uni.get(str(k), float("nan")))) for k in ks]
        contended = bool(witness_before.get("contended") or witness_after.get("contended"))
        results["scaling"] = {
            "labels": ks,
            "bi_sec": bi_sec,
            "uni_sec": uni_sec,
            # THE ONLY WALL-CLOCK BLOCK IN THIS LESSON -- stamp how it was measured.
            "ratio_at_max_labels": (float(uni_sec[-1] / bi_sec[-1])
                                    if bi_sec and bi_sec[-1] else float("nan")),
            "gpu_witness_before": witness_before,
            "gpu_witness_after": witness_after,
            "contended": contended,
            "measurement_note": (
                "Wall-clock on one shared GPU. The ABSOLUTE seconds are only comparable "
                "across runs measured in the same machine state -- two runs of identical "
                "code gave bi 0.39 vs 0.69 s and uni 15.8 vs 27.0 s purely from a "
                "co-tenant job. Contention scales BOTH arms by nearly the same factor, so "
                "compare runs on `ratio_at_max_labels`, not on seconds. `contended`=true "
                "means another python GPU job was resident during the measurement and the "
                "absolute seconds should not be quoted. NEVER form a ratio from a "
                "bi_encoder and a uni_encoder measured in different states -- a published "
                "'64x' in this lesson's README was exactly that error."),
        }
        print("[EXP-D] labels=%s bi=%s uni=%s"
              % (results["scaling"]["labels"],
                 ["%.3f" % v for v in results["scaling"]["bi_sec"]],
                 ["%.3f" % v for v in results["scaling"]["uni_sec"]]))
        print("[EXP-D] ratio at %d labels = %.1fx   CONTENDED=%s (%d co-tenant python "
              "job(s), %s MiB in use) -- if contended, quote the RATIO, not the seconds"
              % (ks[-1] if ks else -1, results["scaling"]["ratio_at_max_labels"],
                 contended,
                 witness_before.get("concurrent_python_jobs_excluding_self", -1),
                 witness_before.get("memory_used_mib", "?")))
    except Exception as exc:
        results["scaling"] = {"error": str(exc)}
        print("[EXP-D] FAILED: %s" % exc)

    # --- EXP-E: TRANSFER, in three arms that shift different things -------
    # The arm this lesson used to call "OOD" was BeaverTails/30k_test -- the same
    # dataset, annotators, taxonomy and rendering as most of train, with only the
    # rows changed. It is kept and renamed `heldout_split`, and two arms where
    # something real shifts sit beside it: `cross_annotator` (Aegis 2.0's held-out
    # test split -- different annotators and taxonomy) and `ood_benchmark`
    # (intrinsec-ai/cstm-bench -- a released external benchmark, the one CLAUDE.md
    # section 17 rule 8 names for this lesson family).
    try:
        arms = data.load_transfer_arms()
        results["transfer"] = {}
        for arm_name, arm in arms.items():
            arm_texts = list(arm.get("texts", []))
            block = {
                "source": str(arm.get("source", "?")),
                "n": int(len(arm_texts)),
                "shift": str(arm.get("shift", "")),
            }
            for key in ("scored_columns", "label_granularity", "error"):
                if key in arm:
                    block[key] = arm[key]
            results["transfer"][arm_name] = block
            if not arm_texts:
                print("[EXP-E/%s] NO ROWS (%s) -- arm reported, not silently dropped"
                      % (arm_name, arm.get("error", "empty")))
                continue
            Y_arm = np.asarray(arm["Y"], dtype=np.float32)[:, seen_cols]
            ih_arm = np.asarray(arm["is_harmful"]).astype(int)
            Xc_arm = _encode_content_cached(embedder, arm_texts, arm_name)
            block["n_harmful"] = int(ih_arm.sum())
            print("[EXP-E/%s] source=%s n=%d harmful=%d  (%s)"
                  % (arm_name, block["source"], block["n"], block["n_harmful"],
                     block["shift"]))
            for method, g in guards.items():
                if g is None:
                    block[method] = {"error": "fit failed"}
                    continue
                try:
                    S = _guard_scores(g, Xc_arm, arm_texts, policy_bank, seen_cols)
                    mm = encoders.macro_micro(Y_arm, S, thresholds=None)
                    block[method] = {
                        "binary_harm_auc": float(
                            encoders.binary_harm_auc(ih_arm, _any_policy_score(S))),
                        "macro_ap": float(mm.get("macro_ap", float("nan"))),
                    }
                    print("[EXP-E/%s/%s] harm_auc=%.3f macro_ap=%.3f"
                          % (arm_name, method, block[method]["binary_harm_auc"],
                             block[method]["macro_ap"]))
                except Exception as exc:
                    block[method] = {"error": str(exc)}
                    print("[EXP-E/%s/%s] FAILED: %s" % (arm_name, method, exc))
        # legacy key: `ood` was always the held-out split. Keep it pointing there so
        # an older reader gets the same numbers under the same name, and never
        # silently gets the new external benchmark under the old label.
        results["ood"] = dict(results["transfer"].get("heldout_split", {}))
        results["ood"]["note"] = ("This is the BeaverTails held-out SPLIT, not "
                                  "out-of-distribution. See results['transfer'].")
    except Exception as exc:
        results["transfer"] = {"error": str(exc)}
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

    # --- EXP-G: ACCURACY under large-label scaling ------------------------
    # Requested as "EXP-F"; that tag was already taken by the hard-negative
    # module above, so this one is EXP-G. It closes the other half of the
    # arXiv:2602.18487 scaling claim -- EXP-D tests LATENCY at scale, this tests
    # ACCURACY at scale.
    if C.LABEL_SCALE_ACC:
        try:
            from . import distractors
            results["label_scale_accuracy"] = _run_label_scale_accuracy(
                encoders, distractors, guards, policies, embedder,
                policy_bank, Xc_te, texts_te, Y_te, seen_cols)
        except Exception as exc:
            results["label_scale_accuracy"] = {"error": str(exc)}
            print("[EXP-G] FAILED: %s" % exc)

    # --- EXP-H: the OPIR TAXONOMY test (RELATED competitors) --------------
    # EXP-G padded the bank with STRANGERS. Opir (arXiv:2605.29659) trains on a
    # three-level 996-category taxonomy where a label's competitors are its own
    # siblings and descendants. EXP-H rebuilds that shape below the 16 real policies
    # and re-runs EXP-G's protocol against it, changing exactly ONE thing: whether the
    # ~980 competitors are unrelated or adjacent.
    if C.OPIR_MODULE:
        try:
            from . import taxonomy
            results["opir_shape"] = _opir_fit_shape(len(policies))
            if results["opir_shape"].get("skip"):
                results["opir_taxonomy"] = {
                    "skipped": True, "reason": results["opir_shape"]["reason"]}
                print("[EXP-H] SKIPPED: %s" % results["opir_shape"]["reason"])
                raise _SkipExperiment()
            results["opir_taxonomy"] = _run_opir_taxonomy(
                encoders, taxonomy, guards, policies, embedder,
                policy_bank, Xc_te, texts_te, Y_te, seen_cols,
                results.get("label_scale_accuracy", {}))
        except _SkipExperiment:
            pass
        except Exception as exc:
            results["opir_taxonomy"] = {"error": str(exc)}
            print("[EXP-H] FAILED: %s" % exc)

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
        ("label_scale", C.LABEL_SCALE_PNG,
         lambda: _plot_label_scale(results.get("label_scale_accuracy", {}), C.LABEL_SCALE_PNG)),
        ("opir", C.OPIR_PNG,
         lambda: _plot_opir(results.get("opir_taxonomy", {}),
                            results.get("label_scale_accuracy", {}), C.OPIR_PNG)),
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
    ach = results.get("achieved", {})
    if ach:
        print("ACHIEVED: %d rows  harmful=%d benign=%d (harmful frac %.3f)  "
              "%d/%d columns under the requested %d"
              % (ach.get("n_rows", 0), ach.get("n_harmful", 0),
                 ach.get("n_benign_achieved", 0),
                 ach.get("class_balance_harmful_frac", 0.0),
                 ach.get("n_columns_short", 0), ach.get("n_policies", 0),
                 results.get("n_per_class_requested", 0)))
        print("sources: %s" % ach.get("source_distribution", {}))
        print("pool_fingerprint=%s" % str(ach.get("pool_fingerprint", ""))[:16])
    c = results.get("confound", {})
    print("confound (folded, directionless): length AUC=%.3f  BINDING BAR %s=%.3f"
          % (c.get("length_auc", float("nan")),
             c.get("binding_bar_name", "none"), c.get("binding_bar", 0.5)))
    for m, mg in sorted((results.get("margins") or {}).items()):
        if isinstance(mg, dict) and "margin" in mg:
            print("  margin %-18s harmAUC %.4f - %s %.4f = %+.4f  clears=%s"
                  % (m, mg["method_auc"], mg["binding_bar_name"],
                     mg["binding_bar"], mg["margin"], mg["clears"]))

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

    # EXP-E -- three arms, each labelled with WHAT SHIFTED. The first is a held-out
    # split of the training dataset and is not out-of-distribution; the name says so.
    transfer = results.get("transfer", {})
    if isinstance(transfer, dict) and transfer and "error" not in transfer:
        print(line)
        print("EXP-E  TRANSFER, scored over SEEN cols")
        for arm_name, arm in transfer.items():
            if not isinstance(arm, dict):
                continue
            print("  [%s] %s  n=%d  -- shift: %s"
                  % (arm_name, arm.get("source", "?"), arm.get("n", 0),
                     arm.get("shift", "?")))
            if arm.get("error"):
                print("      [NO ROWS] %s" % arm["error"])
                continue
            print("      %-14s %9s %8s" % ("method", "harmAUC", "macroAP"))
            for m in C.METHODS:
                cell = arm.get(m)
                if not isinstance(cell, dict) or "error" in cell:
                    print("      %-14s   [FAILED]" % m)
                    continue
                print("      %-14s %9.3f %8.3f"
                      % (m, cell["binary_harm_auc"], cell["macro_ap"]))

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

    # EXP-G
    ls = results.get("label_scale_accuracy", {})
    if isinstance(ls, dict) and ls.get("arms"):
        t = int(ls.get("top_t", 3))
        print(line)
        print("EXP-G  ACCURACY vs #labels (16 real policies padded with synthetic distractors)")
        da = ls.get("distractors", {})
        ea = da.get("embedding", {}) if isinstance(da, dict) else {}
        print("  distractors: n=%d seed=%s fp=%s  max lexical Jaccard to a real policy=%.3f"
              % (da.get("n_used", 0), da.get("seed", "?"), da.get("fingerprint", "?"),
                 da.get("max_jaccard_observed", float("nan"))))
        print("  distractor-to-real cosine: mean_max=%.3f p95=%.3f max=%.3f (n>=0.95: %s)"
              % (ea.get("mean_max_cos_to_real", float("nan")),
                 ea.get("p95_max_cos_to_real", float("nan")),
                 ea.get("max_max_cos_to_real", float("nan")),
                 ea.get("n_above_0.95", "?")))
        print("%-20s %5s %8s %8s %8s %8s %8s %8s %8s"
              % ("arm", "K", "macroAP", "macroF1", "top%dF1" % t, "nreal/row",
                 "meanRank", "chance", "steal@1"))
        for arm in C.LABEL_SCALE_ARMS:
            rows = ls["arms"].get(arm)
            if not isinstance(rows, list) or not rows:
                print("%-20s   [FAILED]" % arm)
                continue
            for r in rows:
                print("%-20s %5d %8.3f %8.3f %8.3f %8.2f %8.2f %8.1f %8.3f"
                      % (arm, r["n_labels"], r["macro_ap"], r["macro_f1"],
                         r["macro_f1_top_t"], r["mean_real_predicted_per_row"],
                         r["mean_rank_true"], r["mean_rank_chance"],
                         r["distractor_steal_top1"]))
        nt = ls.get("notes", {})
        print("  per-column invariance in K verified: %s" % nt.get("per_column_invariance_verified"))
        print("  HOW TO READ -- two traps, both live in this table:")
        print("   1. macroAP/macroF1 are per-column cosine metrics: EXACTLY invariant in K")
        print("      by construction. Their flatness is arithmetic, not evidence.")
        print("   2. top%dF1 can RISE with K without accuracy improving: distractors take"
              % t)
        print("      slots in the top-%d, so fewer REAL columns are predicted per row"
              % t)
        print("      (see nreal/row) and precision goes up for free. Not a scaling win.")
        print("  The uncontaminated verdict is steal@1 (a distractor beating every real")
        print("  policy on a genuinely harmful row) and meanRank read against chance.")

    # EXP-H
    op = results.get("opir_taxonomy", {})
    if isinstance(op, dict) and op.get("arms"):
        tx = op.get("taxonomy", {})
        lx = tx.get("lexical", {})
        ep = tx.get("embedding", {})
        print(line)
        print("EXP-H  the OPIR 3-LEVEL TAXONOMY (%d top + %d mid + %d leaf = %d categories)"
              % (tx.get("n_top", 0), tx.get("n_mid", 0), tx.get("n_leaf", 0),
                 tx.get("n_total", 0)))
        print("  competitors are SIBLINGS/DESCENDANTS of the truth, not strangers:")
        print("   lexical Jaccard to OWN parent %.3f vs %.3f to the nearest other parent"
              % (lx.get("mean_jaccard_own_parent", float("nan")),
                 lx.get("mean_jaccard_best_other_parent", float("nan"))))
        print("   (EXP-G DROPPED any distractor above %.2f; here %.0f%% of nodes are above it"
              % (lx.get("exp_g_gate_for_reference", float("nan")),
                 100 * lx.get("frac_above_exp_g_gate", float("nan"))))
        print("   embedding cos to own parent %.3f vs %.3f best other (%.0f%% closer to own)"
              % (ep.get("mean_cos_to_own_parent", float("nan")),
                 ep.get("mean_cos_to_best_other_parent", float("nan")),
                 100 * ep.get("frac_closer_to_own_parent", float("nan"))))
        print("  taxonomy fp=%s  n_test_rows=%d  bank precision=%s (held EQUAL to the"
              " EXP-G banks on purpose; a mismatch voids the head-to-head)"
              % (tx.get("fingerprint", "?"), op.get("n_test_rows", 0),
                 op.get("precision", "?")))
        print("%-20s %8s %9s %9s %9s %9s %9s"
              % ("arm", "meanRank", "chance", "exclDesc", "cSteal@1", "cSteal-xD",
                 "steal@1"))
        for arm in C.OPIR_ARMS:
            r = op["arms"].get(arm)
            if not isinstance(r, dict) or "mean_rank_true" not in r:
                print("%-20s   [FAILED]" % arm)
                continue
            print("%-20s %8.2f %9.1f %9.2f %9.3f %9.3f %9.3f"
                  % (arm, r["mean_rank_true"], r["mean_rank_chance"],
                     r["mean_rank_excl_descendants"], r["competitor_steal_top1"],
                     r["competitor_steal_top1_excl_descendants"], r["steal_top1"]))
        print("  cSteal@1 = EXP-G's rule (an ADDED column wins the argmax) -- the only")
        print("  steal figure comparable to EXP-G. cSteal-xD forgives a descendant win.")
        print("  steal@1 is the stricter rule (ANY column that is not a true top-level")
        print("  policy) and must NOT be read against EXP-G's number.")
        print("  WHERE THE RANK VIOLATIONS COME FROM (share of all violations; chance in")
        print("  brackets -- a policy has ~61 descendants but ~919 cousins, so the raw")
        print("  shares are dominated by bucket SIZE and only the x-enrichment has content):")
        print("%-20s %22s %22s %22s" % ("arm", "descendant", "sibling(top-level)",
                                        "other-branch(cousin)"))
        for arm in C.OPIR_ARMS:
            r = op["arms"].get(arm)
            if not isinstance(r, dict) or "viol_descendant_share" not in r:
                continue
            cells = []
            for k in ("descendant", "sibling", "other_branch"):
                cells.append("%.3f [%.3f] %5.1fx" % (r["viol_%s_share" % k],
                                                     r["viol_%s_share_chance" % k],
                                                     r["viol_%s_enrichment" % k]))
            print("%-20s %22s %22s %22s" % (arm, cells[0], cells[1], cells[2]))
        print("  TOP-1 on a genuinely harmful row -- what actually wins the argmax")
        print("  (share [chance] enrichment; descendants are only ~6% of the 996 columns):")
        print("%-20s %22s %22s %22s %22s"
              % ("arm", "true top-level", "descendant", "sibling", "other-branch"))
        for arm in C.OPIR_ARMS:
            r = op["arms"].get(arm)
            if not isinstance(r, dict) or "top1_true_policy" not in r:
                continue
            cells = []
            for k, e in (("top1_true_policy", "top1_true_policy_enrichment"),
                         ("top1_descendant_of_true", "top1_descendant_enrichment"),
                         ("top1_sibling_top_level", "top1_sibling_enrichment"),
                         ("top1_other_branch", "top1_other_branch_enrichment")):
                cells.append("%.3f [%.3f] %5.1fx" % (r[k], r[k + "_chance"], r[e]))
            print("%-20s %22s %22s %22s %22s" % (arm, cells[0], cells[1], cells[2], cells[3]))
        vg = op.get("vs_exp_g", {})
        if vg:
            print("  HEAD-TO-HEAD vs EXP-G (same arms, same rows, same 16 scored columns;")
            print("  the ONLY change is whether the ~900 competitors are related):")
            print("  (steal figures use EXP-G's own rule on both sides: an ADDED column")
            print("  wins the argmax on a genuinely harmful row.)")
            print("%-20s %11s %11s %11s %10s %10s %10s"
                  % ("arm", "G rank(900)", "H rank(996)", "H rank@900", "G cSteal",
                     "H cSteal", "H cSt-xD"))
            for arm in C.OPIR_ARMS:
                c = vg.get(arm)
                if not c:
                    continue
                print("%-20s %11.2f %11.2f %11.2f %10.3f %10.3f %10.3f"
                      % (arm, c["exp_g_mean_rank"], c["exp_h_mean_rank"],
                         c.get("exp_h_mean_rank_at_matched_k", float("nan")),
                         c["exp_g_competitor_steal_top1"],
                         c["exp_h_competitor_steal_top1"],
                         c["exp_h_competitor_steal_top1_excl_descendants"]))
        nt = op.get("notes", {})
        print("  cross-experiment anchors (macroAP and top-16 mean rank identical to")
        print("  EXP-G, since both depend only on the 16 real columns) verified: %s"
              % nt.get("cross_experiment_anchor_verified"))
        print("  HOW TO READ -- a DESCENDANT outranking its parent is arguably NOT an")
        print("  error (the router picked a narrower rule under the SAME policy), so the")
        print("  exclDesc / steal-xD columns price it as correct and the plain columns")
        print("  price it as wrong. Report both; which one is right depends on whether")
        print("  the downstream system acts on the top-level policy or on the leaf.")

    print(line)
    print("READ: bi_encoder caches the policy tower -> O(1) per request in #labels and "
          "scores UNSEEN policies zero-shot. FALSIFIERS -- (i) uni latency flat => scaling "
          "claim FALSE; (ii) bi held-out macroAP <= 0.5 => zero-shot claim FALSE; (iii) adapter "
          "does NOT lower FPR@recall0.90 => hard-negative sharpening does not help here; "
          "(iv) the top-1 distractor STEAL RATE rises materially from K=16 to K=900 "
          "=> the 'accuracy is MAINTAINED at scale' half of arXiv:2602.18487 is FALSE "
          "under a competitive decision rule. Judge (iv) on steal@1 and on meanRank read "
          "against its chance line -- NOT on top-T F1, which distractors inflate for free.")
    print(line)


if __name__ == "__main__":
    main()
