"""ltd.py -- the September 2026 SOTA probe, run against the control it never met.

REFERENCE (WebFetch-VERIFIED 2026-09-11)
----------------------------------------
Mammen, Joswin, Medicherla, "Do Agents Know When They Succeed? Calibrating Agent
Confidence from Internal Representations", arXiv:2609.09448, 8 Sep 2026.
Two methods on Qwen-14B / Qwen-7B / DeepSeek-6.7B over Bash / SQL / Python:

  LTD  Latent Trajectory Dynamics -- 28 "kinematic" features from how the
       final-layer residual state MOVES across turns: cosine displacement
       1 - cos(h_t, h_{t+1}), relative displacement ||h_{t+1}-h_t|| / ||h_t||,
       their mean/std/min/max stratified by transition type, plus path
       efficiency and drift ratios. Fed to an L2 logistic classifier.
  ARP  Action Representation Probe -- final-layer state at each ACTION endpoint,
       mean-pooled over actions (terminal excluded), standardised on train
       stats, PCA to 64, L2 logistic, then monotone Platt calibration.

Reported: ARP 0.842 AUROC on Qwen-14B/SQL vs 0.743 for their strongest baseline.

WHY THIS FILE EXISTS -- THE GAP IN THEIR EVALUATION
----------------------------------------------------
Their baselines are TOKEN-PROBABILITY based: "Calibrated Logprob" and "HTC"
(48 features derived from token probabilities). There is NO bag-of-words or
TF-IDF text baseline anywhere in the paper. That is exactly the control that
decides every verdict in this lesson -- and on ATBench it beat our per-turn
snapshot probe (0.8418 vs 0.8070 at `last`; parity at `mean_turn`).

So the question this module answers is one their paper cannot: do LTD and ARP
beat UNIGRAMS OVER THE SAME TEXT? If they do, the residual stream carries
something the surface does not, and our earlier negative was an architecture
artifact. If they do not, then a method reported as beating "surface baselines"
loses to the cheapest surface baseline there is, and the evaluation gap is the
finding.

This runs on the ALREADY-CACHED per-turn activations, so it costs no GPU. Each
row of the bundle is the last-token residual of one turn, which is precisely
the sequence h_0, h_1, ... h_T that LTD differences.

DEVIATIONS FROM THE PAPER, STATED
---------------------------------
  * Layer: they use the FINAL layer. The cached bundles go to L24 of Gemma-3-1B's
    26; L24 is used and named. (Extracting L25 is a 4-minute GPU job if the
    result turns on it.)
  * Transition types: they stratify by reasoning / commitment / action /
    feedback. ATBench turns carry roles user / assistant / tool, so transitions
    are stratified by (from_role -> to_role) pair, which is the same idea over
    the roles this corpus actually has.
  * Their logprob baselines are NOT reproduced -- token probabilities were never
    cached here. The comparison that matters is against the content bar.
  * Models: 1B here vs 7B-14B there. Magnitudes do not transfer; the method and
    its control do.

CPU-only. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import steering_tutorials.common.netboot as netboot
import steering_tutorials.traj_probes.config as C
from steering_tutorials.traj_probes.data import load_corpus
from steering_tutorials.traj_probes.probes import content_bar_control, group_folds

EPS = 1e-8


# ---------------------------------------------------------------------------
# 1. Align cached rows to turns and roles
# ---------------------------------------------------------------------------
def load_bundle_for_layer(layer: int):
    """Pick the `last`-pooled bundle for `layer` by CONTENT (row count), and
    refuse ambiguity -- two same-count bundles once inflated a result here."""
    corpus = load_corpus()
    expected = sum(t.n_turns for t in corpus.trajectories)
    cands = sorted(C.ARTIFACTS.glob("acts_%s_L%d_last_*.npz" % (C.MODEL_TAG, layer)))
    keep = []
    for q in cands:
        with np.load(q, allow_pickle=True) as z:
            if int(z["X"].shape[0]) == expected:
                keep.append(q)
    if len(keep) != 1:
        raise SystemExit("need exactly one L%d bundle with %d rows; found %d (%s)"
                         % (layer, expected, len(keep), [k.name for k in keep]))
    z = np.load(keep[0], allow_pickle=True)
    return corpus, z, keep[0].name


def per_trajectory_sequences(corpus, z):
    """-> {uid: (H [T, d] in step order, roles [T], label)}."""
    X = np.asarray(z["X"], dtype=np.float64)
    uid = np.array([str(u) for u in z["traj_uid"]])
    step = np.asarray(z["step_index"]).astype(int)
    y = np.asarray(z["y"]).astype(int)
    roles = {t.uid: [x.role for x in t.turns] for t in corpus.trajectories}
    out = {}
    for u in np.unique(uid):
        m = uid == u
        order = np.argsort(step[m])
        H = X[m][order]
        st = step[m][order]
        r = [roles[u][s] for s in st]
        out[u] = (H, r, int(y[m][0]))
    return out


# ---------------------------------------------------------------------------
# 2. LTD: 28 kinematic features
# ---------------------------------------------------------------------------
TRANSITIONS = ("assistant->assistant", "assistant->tool", "tool->assistant",
               "user->assistant")


def _stats(v) -> list:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(v.mean()), float(v.std()), float(v.min()), float(v.max())]


def ltd_features(H: np.ndarray, roles) -> np.ndarray:
    """The kinematic summary of one trajectory's latent path.

    Feature layout (28):
      [0:4]   cosine displacement  mean/std/min/max, all transitions
      [4:8]   relative displacement mean/std/min/max, all transitions
      [8:24]  cosine displacement mean/max + relative displacement mean/max,
              for each of the 4 role transitions (4 x 4 = 16)
      [24]    path efficiency  ||h_T - h_0|| / sum_t ||h_{t+1} - h_t||
      [25]    drift ratio      ||h_T - h_0|| / ||h_0||
      [26]    mean state norm
      [27]    T (number of turns) -- INCLUDED DELIBERATELY so the step-index
              confound is IN the feature set and can be measured, not hidden.
    """
    T = H.shape[0]
    if T < 2:
        return np.zeros(28)
    n = np.linalg.norm(H, axis=1) + EPS
    d = H[1:] - H[:-1]
    cos_disp = 1.0 - np.sum(H[:-1] * H[1:], axis=1) / (n[:-1] * n[1:])
    rel_disp = np.linalg.norm(d, axis=1) / n[:-1]
    f = _stats(cos_disp) + _stats(rel_disp)
    trans = ["%s->%s" % (roles[t], roles[t + 1]) for t in range(T - 1)]
    for tr in TRANSITIONS:
        idx = [i for i, x in enumerate(trans) if x == tr]
        c = cos_disp[idx] if idx else np.array([])
        r = rel_disp[idx] if idx else np.array([])
        f += [float(c.mean()) if c.size else 0.0, float(c.max()) if c.size else 0.0,
              float(r.mean()) if r.size else 0.0, float(r.max()) if r.size else 0.0]
    path_len = float(np.linalg.norm(d, axis=1).sum()) + EPS
    chord = float(np.linalg.norm(H[-1] - H[0]))
    f += [chord / path_len, chord / (n[0]), float(n.mean()), float(T)]
    return np.asarray(f, dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. ARP: mean over action endpoints, PCA-64, logistic, Platt
# ---------------------------------------------------------------------------
def arp_vector(H: np.ndarray, roles) -> np.ndarray:
    """Mean of final-layer states at ACTION turns (assistant), terminal excluded."""
    idx = [i for i, r in enumerate(roles) if r == "assistant"]
    if len(idx) > 1:
        idx = idx[:-1]                       # exclude the terminal submission
    if not idx:
        idx = list(range(H.shape[0]))
    return H[idx].mean(axis=0)


# ---------------------------------------------------------------------------
# 4. Evaluation, group-aware, matched to the lesson's other arms
# ---------------------------------------------------------------------------
def _cv_auc(F, y, groups, seed, n_folds, pca_dim=None, platt=False):
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y).astype(int)
    splits = group_folds(groups, y, n_folds=n_folds, seed=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(F[tr])
        Xtr, Xte = sc.transform(F[tr]), sc.transform(F[te])
        if pca_dim and Xtr.shape[1] > pca_dim:
            p = PCA(n_components=pca_dim, random_state=seed).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        clf.fit(Xtr, y[tr])
        s = clf.decision_function(Xte)
        if platt:
            # monotone Platt on a held-in slice of TRAIN, never on test
            cut = max(2, int(0.2 * len(tr)))
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(tr))
            cal, fit = tr[perm[:cut]], tr[perm[cut:]]
            if len(np.unique(y[fit])) == 2 and len(np.unique(y[cal])) == 2:
                clf2 = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
                Xfit = sc.transform(F[fit]); Xcal = sc.transform(F[cal])
                if pca_dim and Xfit.shape[1] > pca_dim:
                    Xfit, Xcal = p.transform(Xfit), p.transform(Xcal)
                clf2.fit(Xfit, y[fit])
                pl = LogisticRegression(max_iter=1000).fit(
                    clf2.decision_function(Xcal).reshape(-1, 1), y[cal])
                s = pl.decision_function(s.reshape(-1, 1))
        oof[te] = s
    good = ~np.isnan(oof)
    return float(roc_auc_score(y[good], oof[good])), oof


def _paired_boot(y, a, b, seed=0, n=10000):
    """Bootstrap CI on AUC(a) - AUC(b) over trajectories."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int); a = np.asarray(a, float); b = np.asarray(b, float)
    good = ~(np.isnan(a) | np.isnan(b))
    y, a, b = y[good], a[good], b[good]
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_a_gt_b": round(float((d > 0).mean()), 3), "excludes_zero": bool(hi < 0 or lo > 0)}


def main() -> int:
    netboot.enable()
    layer = int(os.environ.get("TP_LTD_LAYER", "24"))
    corpus, z, bname = load_bundle_for_layer(layer)
    seqs = per_trajectory_sequences(corpus, z)
    uids = sorted(seqs)
    y = np.array([seqs[u][2] for u in uids])
    print("bundle %s  (%d trajectories, layer %d)" % (bname, len(uids), layer))

    # --- the control they never ran: unigrams over the SAME text ------------
    texts = {t.uid: t.text for t in corpus.trajectories}
    bar = content_bar_control(uids, [int(v) for v in y], texts, seed=C.SEED)
    bar_auc = float(bar["auc"]); bar_scores = np.asarray(bar["scores"], float)
    print("TF-IDF content bar (same text) : AUC %.4f" % bar_auc)

    # --- LTD ----------------------------------------------------------------
    F_ltd = np.stack([ltd_features(seqs[u][0], seqs[u][1]) for u in uids])
    ltd_auc, ltd_oof = _cv_auc(F_ltd, y, uids, C.SEED, C.N_FOLDS)
    # the same features WITHOUT the turn-count column (feature 27), to price
    # the step-index confound rather than hide it
    ltd_noT_auc, ltd_noT_oof = _cv_auc(F_ltd[:, :27], y, uids, C.SEED, C.N_FOLDS)
    T_only_auc, _ = _cv_auc(F_ltd[:, 27:28], y, uids, C.SEED, C.N_FOLDS)

    # --- ARP ----------------------------------------------------------------
    F_arp = np.stack([arp_vector(seqs[u][0], seqs[u][1]) for u in uids])
    arp_auc, arp_oof = _cv_auc(F_arp, y, uids, C.SEED, C.N_FOLDS, pca_dim=64, platt=True)
    arp_nopca_auc, _ = _cv_auc(F_arp, y, uids, C.SEED, C.N_FOLDS)

    # --- LTD + ARP together (the paper reports them separately; this is extra)
    F_both = np.hstack([F_ltd, F_arp])
    both_auc, both_oof = _cv_auc(F_both, y, uids, C.SEED, C.N_FOLDS, pca_dim=64)

    rows = {
        "content_bar_tfidf": {"auc": round(bar_auc, 4), "n_features": "unigram vocab"},
        "LTD_28_kinematic": {"auc": round(ltd_auc, 4), "n_features": 28,
                             "vs_bar": _paired_boot(y, ltd_oof, bar_scores)},
        "LTD_without_turn_count": {"auc": round(ltd_noT_auc, 4), "n_features": 27},
        "turn_count_only": {"auc": round(T_only_auc, 4), "n_features": 1,
                            "note": "the step-index confound, priced explicitly"},
        "ARP_pca64_platt": {"auc": round(arp_auc, 4), "n_features": 64,
                            "vs_bar": _paired_boot(y, arp_oof, bar_scores)},
        "ARP_no_pca": {"auc": round(arp_nopca_auc, 4), "n_features": int(F_arp.shape[1])},
        "LTD_plus_ARP": {"auc": round(both_auc, 4),
                         "vs_bar": _paired_boot(y, both_oof, bar_scores)},
    }
    out = {"reference": "arXiv:2609.09448 (Mammen, Joswin, Medicherla, 8 Sep 2026)",
           "layer": layer, "layer_note": ("paper uses the FINAL layer; L%d of 26 used here "
                                          "from the cached bundle" % layer),
           "n_trajectories": int(len(uids)), "n_folds": C.N_FOLDS, "seed": C.SEED,
           "content_bar_auc": round(bar_auc, 4), "arms": rows,
           "for_comparison_from_earlier_runs": {
               "last_token_snapshot_L12": 0.8070, "mean_turn_L12": 0.8565,
               "note": "trajectory unit, same corpus and bar; see README section 7/7b"}}
    p = C.ARTIFACTS / ("ltd_arp_L%d.json" % layer)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    os.replace(tmp, p)

    print("")
    print("%-26s %-8s %s" % ("arm", "AUC", "vs TF-IDF bar (paired bootstrap)"))
    for k, v in rows.items():
        vb = v.get("vs_bar")
        tail = ("%+.4f  CI %s  P(>bar)=%.3f  %s" % (vb["mean"], vb["ci95"], vb["p_a_gt_b"],
                "EXCLUDES 0" if vb["excludes_zero"] else "includes 0")) if vb else ""
        print("%-26s %-8.4f %s" % (k, v["auc"], tail))
    print("")
    print("earlier arms on this corpus: last-token L12 0.8070 | mean_turn L12 0.8565 | bar %.4f" % bar_auc)
    print("wrote %s" % p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
