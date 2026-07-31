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
  EXP-E  OOD                     : score a disjoint out-of-distribution slice
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
    ks = sorted({int(k) for k in C.LABEL_SCALE_ACC_K})
    n_real = len(policies)
    ks = [k for k in ks if k >= n_real]
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
        "label_scale_accuracy": {},
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
