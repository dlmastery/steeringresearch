"""run_meerkat.py -- orchestrator for the MEERKAT SPARSE-VIOLATION-LOCALIZATION
lesson (the UNSUPERVISED clustering sibling of `cross_trajectory`).

THE EXPERIMENT (Meerkat, Stein/Brown/Hassani/Naik/Wong, arXiv:2604.11806)
------------------------------------------------------------------------
Safety violations in agentic systems are often SPARSE (~5% of a repository of
traces), COMPLEX, and adversarially DILUTED among benign traffic -- a misuse
campaign whose individual traces look unremarkable one-at-a-time. A per-trace
monitor (score each trace ALONE) drowns in the benign majority and misses the
campaign. Meerkat instead EMBEDS every trace (bge-base-en-v1.5), K-MEANS
CLUSTERS the whole repository, and scores clusters by their violation
enrichment -- the campaign concentrates into a cluster and lights up even at 5%.

This runner asks one falsifiable question:

    At a 5% sparse base rate, does CLUSTER-ENRICHMENT (kmeans_enrich) reach a
    higher Average Precision than a PER-TRACE monitor (per_trace)?  If it does
    NOT, "clustering surfaces the distributed campaign" is FALSE for our setup.

WHAT main() DOES
----------------
  1. Build the trace POOL (>=500/class: SafeMTData Attack_600 decompositions as
     positives, UltraChat conversations as negatives) and EMBED it ONCE (cached
     to C.EMB_CACHE[C.EMBEDDER]); report the raw-length confound.
  2. For base_rate in {C.BASE_RATE (sparse ~5%), 0.5 (balanced)}: sample
     C.N_REPOS repositories; on each, fit every localizer on a small labelled
     SEED (SEED_FRAC of the repo, mimicking an analyst who labels a handful),
     score the held-out rest, and record AP + ROC-AUC. Average over repos with
     a bootstrap CI on the mean AP.
  3. Cluster quality on one sparse repository: silhouette-chosen K, cluster
     purity, and how much of the attack campaign concentrates in ONE cluster.
  4. OOD: score the three localizers on the real CSTM-Bench cross-session
     scenarios (seeded with a few labelled CSTM traces).
  5. Write results.json (schema in the contract) BEFORE the ASCII summary, and
     render three Agg PNGs (PCA scatter with the attack cluster highlighted, AP
     vs base-rate bars, silhouette vs K). Every cell + plot is wrapped so a late
     failure still leaves results.json + whatever plots succeeded.

Sibling modules (data / cluster) are imported LAZILY inside main() so
`python -c "import ...run_meerkat"` succeeds even while they are still stubs and
NEVER triggers a model load at import. CPU-only orchestration; env caps
(MK_N_ATTACK / MK_N_BENIGN / MK_BASE_RATE / MK_REPO_SIZE / MK_N_REPOS / MK_EMBED)
already live in config. Stdout is ASCII only (Windows cp1252).
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# Metric helpers -- sklearn only, so they stay import-safe (no sibling/model dep)
# ---------------------------------------------------------------------------
def _ap(y_true, y_score):
    """Average Precision (the Meerkat headline metric for the sparse regime)."""
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:  # AP is undefined without both classes
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_score)))


def _auc(y_true, y_score):
    """ROC-AUC (the balanced-regime metric)."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_score)))


def _boot_ap_ci(y_true, y_score, n=C.BOOTSTRAP, seed=0):
    """Percentile bootstrap 95% CI on AP over the SAMPLES of one prediction set.

    Resamples (y, s) pairs with replacement; skips a resample that lost a class
    (AP would be undefined). Used for the single-repository OOD block.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    m = len(y_true)
    if m == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n)):
        idx = rng.integers(0, m, m)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(_ap(yt, y_score[idx]))
    if not boots:
        return [float("nan"), float("nan")]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def _metrics(y, s):
    """Full metric bundle for one pooled prediction vector.

    Returns {"ap","roc_auc","ap_ci"} (bootstrap CI on AP). The OOD block stores
    only {"ap","roc_auc"} per the schema; the CI is available for pages/plots.
    """
    return {"ap": _ap(y, s), "roc_auc": _auc(y, s), "ap_ci": _boot_ap_ci(y, s)}


def _mean_ci(vals, n=C.BOOTSTRAP, seed=0):
    """Bootstrap 95% CI on the MEAN of a list of per-repository AP values.

    The sparse-regime headline averages AP across C.N_REPOS sampled
    repositories; this reports the spread of that mean across repositories.
    Returns (mean, [lo, hi]).
    """
    vals = np.asarray([v for v in vals if v == v], dtype=float)  # drop NaNs
    if len(vals) == 0:
        return float("nan"), [float("nan"), float("nan")]
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, [mean, mean]
    rng = np.random.default_rng(seed)
    boots = [float(vals[rng.integers(0, len(vals), len(vals))].mean())
             for _ in range(int(n))]
    return mean, [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


# ---------------------------------------------------------------------------
# Localizer factory + seed/eval split
# ---------------------------------------------------------------------------
def _make_model(cluster, name):
    """Instantiate one localizer by its stable config key (lazy sibling import).

    Each exposes fit(emb, seed_idx, seed_labels) then score(emb) -> [0,1].
    """
    if name == "per_trace":
        return cluster.PerTraceMonitor()   # BASELINE: logistic on each trace alone
    if name == "kmeans_enrich":
        return cluster.KMeansEnrich()      # MEERKAT-style: cluster + seed-enrichment
    if name == "knn_purity":
        return cluster.KnnPurity()         # clustering-free density proxy
    raise ValueError("unknown method: %s" % name)


def _seed_split(labels, seed_frac, seed):
    """Pick a small labelled SEED and the held-out rest.

    Mimics an analyst who labels a handful of traces. Guarantees BOTH classes
    appear in the seed whenever both exist in the repository, so the logistic
    baseline and the cluster-enrichment estimator are both well-defined. Returns
    (seed_idx, rest_idx) as int ndarrays.
    """
    labels = np.asarray(labels)
    n = len(labels)
    n_seed = min(n, max(2, int(round(seed_frac * n))))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    seed_idx = list(perm[:n_seed])
    rest_idx = list(perm[n_seed:])
    # Force at least one member of any present-but-missing class into the seed.
    for cls in (0, 1):
        cls_all = np.where(labels == cls)[0]
        if len(cls_all) and not np.any(labels[np.asarray(seed_idx)] == cls):
            take = int(cls_all[0])
            seed_idx.append(take)
            rest_idx = [i for i in rest_idx if i != take]
    return np.asarray(seed_idx, dtype=int), np.asarray(rest_idx, dtype=int)


def _campaign_max_cluster_recall(cluster_ids, labels):
    """Largest fraction of the WHOLE attack campaign that lands in one cluster.

    High => the diluted campaign concentrates into a single cluster (exactly what
    lets cluster-enrichment beat the per-trace monitor at a sparse base rate).
    """
    cluster_ids = np.asarray(cluster_ids)
    labels = np.asarray(labels)
    total = int(np.sum(labels == 1))
    if total == 0:
        return 0.0
    best = 0
    for c in np.unique(cluster_ids):
        best = max(best, int(np.sum((labels == 1) & (cluster_ids == c))))
    return float(best) / float(total)


# ---------------------------------------------------------------------------
# Embedding (embed the POOL once; look repo traces back up by their string)
# ---------------------------------------------------------------------------
def _embed_pool(cluster, traces):
    """Embed the whole trace pool ONCE, cached to C.EMB_CACHE[C.EMBEDDER].

    The cache is keyed on the embedder + the pool size; if the count changed (env
    caps), it is rebuilt. Returns [n, dim] float32.
    """
    cache = C.EMB_CACHE[C.EMBEDDER]
    if cache.exists():
        try:
            npz = np.load(cache)
            emb = npz["emb"]
            if emb.shape[0] == len(traces):
                print("[embed] loaded cache %s (%d x %d)"
                      % (cache.name, emb.shape[0], emb.shape[1]))
                return emb.astype(np.float32)
            print("[embed] cache size mismatch (%d != %d); re-embedding"
                  % (emb.shape[0], len(traces)))
        except Exception as exc:
            print("[embed] cache unreadable (%s); re-embedding" % exc)
    embed_text, dim = cluster.get_embedder(C.EMBEDDER)
    emb = np.asarray(cluster.embed_traces(traces, embed_text), dtype=np.float32)
    try:
        np.savez(cache, emb=emb)
        print("[embed] built + cached %s (%d x %d)" % (cache.name, emb.shape[0], emb.shape[1]))
    except Exception as exc:
        print("[embed] cache save FAILED: %s" % exc)
    return emb


def _repo_emb(pool_emb, t2i, repo):
    """Slice the pool embedding matrix for one sampled repository (by trace str)."""
    return np.asarray([pool_emb[t2i[t]] for t in repo["traces"]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Regime sweep: average AP + ROC-AUC over N_REPOS sampled repositories
# ---------------------------------------------------------------------------
def _run_regime(cluster, data, pool, pool_emb, t2i, base_rate):
    """Sample C.N_REPOS repositories at `base_rate`; fit+score each localizer on
    each; average AP/ROC-AUC over repositories. Returns the regime block."""
    per_ap = {m: [] for m in C.METHODS}
    per_auc = {m: [] for m in C.METHODS}

    for r in range(C.N_REPOS):
        try:
            repo = data.sample_repository(pool, C.REPO_SIZE, base_rate, seed=C.SEED + r)
        except Exception as exc:
            print("[regime %.2f/repo %d] sample FAILED: %s" % (base_rate, r, exc))
            continue
        try:
            remb = _repo_emb(pool_emb, t2i, repo)
        except Exception as exc:
            print("[regime %.2f/repo %d] embed lookup FAILED: %s" % (base_rate, r, exc))
            continue
        rlabels = np.asarray(repo["labels"])
        seed_idx, rest_idx = _seed_split(rlabels, C.SEED_FRAC, C.SEED + r)
        if len(rest_idx) == 0 or len(np.unique(rlabels[rest_idx])) < 2:
            continue  # nothing to score against
        for m in C.METHODS:
            try:
                model = _make_model(cluster, m)
                model.fit(remb, seed_idx, rlabels[seed_idx])
                scores = np.asarray(model.score(remb)).reshape(-1)
                y = rlabels[rest_idx]
                s = scores[rest_idx]
                per_ap[m].append(_ap(y, s))
                per_auc[m].append(_auc(y, s))
            except Exception as exc:
                print("[regime %.2f/repo %d/%s] FAILED: %s" % (base_rate, r, m, exc))

    block = {"repo_size": int(C.REPO_SIZE), "n_repos": int(C.N_REPOS)}
    for m in C.METHODS:
        ap_mean, ap_ci = _mean_ci(per_ap[m], seed=C.SEED)
        aucs = [v for v in per_auc[m] if v == v]
        auc_mean = float(np.mean(aucs)) if aucs else float("nan")
        block[m] = {"ap": ap_mean, "ap_ci": ap_ci, "roc_auc": auc_mean}
        print("[regime %.2f/%s] AP=%.3f CI=[%.3f,%.3f] AUC=%.3f (n_repos_scored=%d)"
              % (base_rate, m, ap_mean, ap_ci[0], ap_ci[1], auc_mean,
                 len([v for v in per_ap[m] if v == v])))
    return block


# ---------------------------------------------------------------------------
# Cluster quality on one sparse repository
# ---------------------------------------------------------------------------
def _run_clustering(cluster, sparse_emb, sparse_labels):
    """Silhouette-chosen K, cluster purity, and campaign concentration."""
    best_k, sil = cluster.choose_k(sparse_emb, C.K_GRID, C.SEED)
    ids = cluster.kmeans_labels(sparse_emb, best_k, C.SEED)
    purity = float(cluster.cluster_purity(ids, sparse_labels))
    recall = _campaign_max_cluster_recall(ids, sparse_labels)
    print("[cluster] best_k=%d purity=%.3f campaign_max_cluster_recall=%.3f"
          % (best_k, purity, recall))
    return {
        "best_k": int(best_k),
        "silhouette": {str(k): float(v) for k, v in sil.items()},
        "cluster_purity": purity,
        "campaign_max_cluster_recall": recall,
    }, ids


# ---------------------------------------------------------------------------
# OOD: real CSTM-Bench cross-session scenarios
# ---------------------------------------------------------------------------
def _run_ood(cluster, data):
    """Embed CSTM-Bench, seed with a few labelled traces, score the rest."""
    ds = data.load_ood_cstm(C.SEED)
    traces = ds["traces"]
    labels = np.asarray(ds["labels"])
    n_attack = int(np.sum(labels == 1))
    n_benign = int(np.sum(labels == 0))
    print("[ood] CSTM-Bench traces=%d attack=%d benign=%d"
          % (len(traces), n_attack, n_benign))

    out = {"dataset": C.OOD_DATASET, "n_attack": n_attack, "n_benign": n_benign}
    embed_text, _dim = cluster.get_embedder(C.EMBEDDER)
    emb = np.asarray(cluster.embed_traces(traces, embed_text), dtype=np.float32)
    seed_idx, rest_idx = _seed_split(labels, C.SEED_FRAC, C.SEED)
    if len(rest_idx) == 0 or len(np.unique(labels[rest_idx])) < 2:
        rest_idx = np.arange(len(labels))  # tiny bench -> score everything
    for m in C.METHODS:
        try:
            model = _make_model(cluster, m)
            model.fit(emb, seed_idx, labels[seed_idx])
            scores = np.asarray(model.score(emb)).reshape(-1)
            mm = _metrics(labels[rest_idx], scores[rest_idx])
            out[m] = {"ap": mm["ap"], "roc_auc": mm["roc_auc"]}
            print("[ood/%s] AP=%.3f AUC=%.3f" % (m, mm["ap"], mm["roc_auc"]))
        except Exception as exc:
            out[m] = {"error": str(exc)}
            print("[ood/%s] FAILED: %s" % (m, exc))
    return out


# ---------------------------------------------------------------------------
# Examples: fit all three on the sparse repo, show scores for a few traces
# ---------------------------------------------------------------------------
def _build_examples(cluster, sparse_repo, sparse_emb):
    """A few attack + benign traces with EVERY localizer's P(violation).

    Shows the per-trace monitor scoring the diluted campaign low while
    cluster-enrichment surfaces it -- the whole point, rendered per-example.
    """
    labels = np.asarray(sparse_repo["labels"])
    traces = sparse_repo["traces"]
    sources = sparse_repo.get("sources", ["?"] * len(traces))
    seed_idx, rest_idx = _seed_split(labels, C.SEED_FRAC, C.SEED)

    fitted = {}
    for m in C.METHODS:
        try:
            model = _make_model(cluster, m)
            model.fit(sparse_emb, seed_idx, labels[seed_idx])
            fitted[m] = np.asarray(model.score(sparse_emb)).reshape(-1)
        except Exception as exc:
            print("[examples/%s] FAILED: %s" % (m, exc))

    pos = [i for i in rest_idx if labels[i] == 1][:3]
    neg = [i for i in rest_idx if labels[i] == 0][:3]
    examples = []
    for i in list(pos) + list(neg):
        examples.append({
            "source": str(sources[i]) if i < len(sources) else "?",
            "label": int(labels[i]),
            "trace_preview": str(traces[i])[:300],
            "scores": {m: float(fitted[m][i]) for m in fitted},
        })
    return examples


# ---------------------------------------------------------------------------
# Plots (Agg backend, PNG only) -- each called under its own try/except
# ---------------------------------------------------------------------------
def _plot_pca_scatter(sparse_emb, sparse_labels, cluster_ids, out_path):
    """2-D PCA of the sparse repository; benign grey, attacks red, and the
    highest-attack-recall cluster outlined (the campaign clustering surfaces)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    labels = np.asarray(sparse_labels)
    ids = np.asarray(cluster_ids)
    xy = PCA(n_components=2, random_state=C.SEED).fit_transform(sparse_emb)

    # Which cluster holds the most of the attack campaign?
    attack_cluster, best = None, -1
    for c in np.unique(ids):
        atk = int(np.sum((labels == 1) & (ids == c)))
        if atk > best:
            best, attack_cluster = atk, c

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ben = labels == 0
    atk = labels == 1
    ax.scatter(xy[ben, 0], xy[ben, 1], s=10, c="0.7", alpha=0.6, label="benign")
    ax.scatter(xy[atk, 0], xy[atk, 1], s=26, c="tab:red", edgecolor="k",
               linewidth=0.3, label="violation (attack)")
    if attack_cluster is not None:
        inc = ids == attack_cluster
        ax.scatter(xy[inc, 0], xy[inc, 1], s=90, facecolors="none",
                   edgecolors="tab:blue", linewidth=1.1,
                   label="attack cluster (k-means)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Sparse repository (%.0f%% attacks): the campaign clusters"
                 % (100.0 * C.BASE_RATE))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_ap_vs_baserate(regimes, out_path):
    """Grouped bars: AP per method at each base rate (the headline comparison)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = list(regimes.keys())
    methods = C.METHODS
    x = np.arange(len(methods))
    width = 0.8 / max(1, len(rates))

    fig, ax = plt.subplots(figsize=(7, 5))
    for j, rate in enumerate(rates):
        vals, errs = [], []
        for m in methods:
            cell = regimes.get(rate, {}).get(m, {})
            v = cell.get("ap", float("nan")) if isinstance(cell, dict) else float("nan")
            ci = cell.get("ap_ci", [v, v]) if isinstance(cell, dict) else [v, v]
            vals.append(v if v == v else 0.0)
            lo = ci[0] if ci and ci[0] == ci[0] else v
            hi = ci[1] if ci and ci[1] == ci[1] else v
            errs.append([max(0.0, (v - lo)) if v == v else 0.0,
                         max(0.0, (hi - v)) if v == v else 0.0])
        errs = np.asarray(errs).T if errs else None
        ax.bar(x + j * width, vals, width, yerr=errs, capsize=3,
               label="base_rate=%s" % rate)
    ax.set_xticks(x + width * (len(rates) - 1) / 2)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Average Precision (mean over repos)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("AP by method x base rate  (sparse is the Meerkat regime)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_silhouette(silhouette, best_k, out_path):
    """Silhouette vs K, with the chosen K marked (how choose_k picks the grid)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(int(k) for k in silhouette.keys())
    vals = [silhouette[str(k)] for k in ks]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(ks, vals, "o-", color="tab:green")
    if best_k in ks:
        ax.axvline(best_k, color="k", linestyle="--", alpha=0.5,
                   label="chosen k=%d" % best_k)
        ax.legend(fontsize=8)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("mean silhouette")
    ax.set_title("Silhouette vs k on the sparse repository")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main():
    # Lazy sibling imports (guarded here, NOT at module top, so the import-check
    # passes while data/cluster are still stubs and no model loads at import).
    from . import cluster, data

    C.ARTIFACTS.mkdir(exist_ok=True)

    # --- 1. pool + embed-once + confound ------------------------------------
    pool = data.load_trace_pool(C.N_ATTACK, C.N_BENIGN, C.SEED)
    pool_labels = np.asarray(pool["labels"])
    n_attack = int(np.sum(pool_labels == 1))
    n_benign = int(np.sum(pool_labels == 0))
    print("[pool] traces=%d attack=%d benign=%d"
          % (len(pool["traces"]), n_attack, n_benign))

    try:
        pool_emb = _embed_pool(cluster, pool["traces"])
    except Exception as exc:
        print("[embed] FATAL: %s" % exc)
        raise
    # Look repo traces back up by their string (embed once, reuse everywhere).
    t2i = {t: i for i, t in enumerate(pool["traces"])}

    try:
        conf = data.confound_report(pool["traces"], pool["labels"])
    except Exception as exc:
        conf = {"length_auc": float("nan"), "len_pos_mean": float("nan"),
                "len_neg_mean": float("nan"), "error": str(exc)}
        print("[confound] FAILED: %s" % exc)
    print("[confound] length_auc=%.3f len_pos=%.0f len_neg=%.0f  (~0.5 => no trivial length tell)"
          % (conf.get("length_auc", float("nan")), conf.get("len_pos_mean", float("nan")),
             conf.get("len_neg_mean", float("nan"))))

    # --- 2. regime sweep (sparse + balanced) --------------------------------
    regimes = {}
    for base_rate in (C.BASE_RATE, 0.5):
        try:
            regimes[str(base_rate)] = _run_regime(
                cluster, data, pool, pool_emb, t2i, base_rate)
        except Exception as exc:
            regimes[str(base_rate)] = {"repo_size": int(C.REPO_SIZE),
                                       "n_repos": int(C.N_REPOS), "error": str(exc)}
            print("[regime %.2f] FAILED: %s" % (base_rate, exc))

    # --- 3. cluster quality on ONE sparse repository ------------------------
    #     (also reused for the PCA plot + the worked examples)
    clustering = {}
    sparse_repo = None
    sparse_emb = None
    cluster_ids = None
    try:
        sparse_repo = data.sample_repository(pool, C.REPO_SIZE, C.BASE_RATE, seed=C.SEED)
        sparse_emb = _repo_emb(pool_emb, t2i, sparse_repo)
        clustering, cluster_ids = _run_clustering(
            cluster, sparse_emb, np.asarray(sparse_repo["labels"]))
    except Exception as exc:
        clustering = {"error": str(exc)}
        print("[cluster] FAILED: %s" % exc)

    # --- 4. OOD on CSTM-Bench -----------------------------------------------
    try:
        ood = _run_ood(cluster, data)
    except Exception as exc:
        ood = {"dataset": C.OOD_DATASET, "n_attack": 0, "n_benign": 0, "error": str(exc)}
        print("[ood] FAILED: %s" % exc)

    # --- 4b. worked examples ------------------------------------------------
    examples = []
    if sparse_repo is not None and sparse_emb is not None:
        try:
            examples = _build_examples(cluster, sparse_repo, sparse_emb)
        except Exception as exc:
            print("[examples] FAILED: %s" % exc)

    # --- 5a. plots (each wrapped; best-effort) ------------------------------
    plots = []
    if sparse_emb is not None and cluster_ids is not None:
        try:
            _plot_pca_scatter(sparse_emb, np.asarray(sparse_repo["labels"]),
                              cluster_ids, C.CLUSTER_PNG)
            plots.append(str(C.CLUSTER_PNG))
        except Exception as exc:
            print("[plot:pca] FAILED: %s" % exc)
    try:
        _plot_ap_vs_baserate(regimes, C.AP_PNG)
        plots.append(str(C.AP_PNG))
    except Exception as exc:
        print("[plot:ap] FAILED: %s" % exc)
    if isinstance(clustering, dict) and "silhouette" in clustering:
        try:
            _plot_silhouette(clustering["silhouette"], clustering.get("best_k", -1),
                             C.SILHOUETTE_PNG)
            plots.append(str(C.SILHOUETTE_PNG))
        except Exception as exc:
            print("[plot:silhouette] FAILED: %s" % exc)

    # --- 5b. results.json (schema-verbatim), written BEFORE the summary -----
    results = {
        "embed_model": str(C.EMBED_MODEL),
        "embedder": str(C.EMBEDDER),
        "n_attack": n_attack,
        "n_benign": n_benign,
        "seed": int(C.SEED),
        "confound": {
            "length_auc": float(conf.get("length_auc", float("nan"))),
            "len_pos_mean": float(conf.get("len_pos_mean", float("nan"))),
            "len_neg_mean": float(conf.get("len_neg_mean", float("nan"))),
        },
        "regimes": regimes,
        "clustering": clustering,
        "ood": ood,
        "examples": examples,
        "plots": plots,
    }
    try:
        with open(C.RESULTS_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print("[write] %s" % C.RESULTS_PATH)
    except Exception as exc:
        print("[write] FAILED: %s" % exc)

    _print_summary(results)
    return results


def _print_summary(results):
    line = "-" * 78
    print("")
    print(line)
    print("MEERKAT SPARSE-VIOLATION LOCALIZATION  (SCREENING TIER, cluster-density)")
    print("embed_model=%s embedder=%s seed=%d  pool: attack=%d benign=%d"
          % (results["embed_model"], results["embedder"], results["seed"],
             results["n_attack"], results["n_benign"]))
    c = results.get("confound", {})
    print("confound: length_auc=%.3f  (~0.5 => raw trace length is NOT a trivial tell)"
          % c.get("length_auc", float("nan")))

    for rate, block in results.get("regimes", {}).items():
        print(line)
        if not isinstance(block, dict) or "error" in block:
            print("BASE RATE %s   [FAILED]" % rate)
            continue
        tag = "SPARSE (Meerkat regime)" if float(rate) < 0.5 else "BALANCED"
        print("BASE RATE %s  %s   repo_size=%d n_repos=%d"
              % (rate, tag, block.get("repo_size", 0), block.get("n_repos", 0)))
        print("%-16s %7s %-17s %8s" % ("method", "AP", "95% CI", "ROC-AUC"))
        for m in C.METHODS:
            cell = block.get(m)
            if not isinstance(cell, dict):
                continue
            ci = cell.get("ap_ci", [float("nan"), float("nan")])
            print("%-16s %7.3f [%.3f,%.3f]   %8.3f"
                  % (m, cell.get("ap", float("nan")), ci[0], ci[1],
                     cell.get("roc_auc", float("nan"))))

    cl = results.get("clustering", {})
    print(line)
    if isinstance(cl, dict) and "best_k" in cl:
        print("CLUSTERING (sparse repo): best_k=%d purity=%.3f campaign_max_cluster_recall=%.3f"
              % (cl.get("best_k", -1), cl.get("cluster_purity", float("nan")),
                 cl.get("campaign_max_cluster_recall", float("nan"))))
    else:
        print("CLUSTERING: [FAILED]")

    ood = results.get("ood", {})
    print(line)
    print("OOD: %s   attack=%d benign=%d  (real cross-session benchmark)"
          % (ood.get("dataset", "?"), ood.get("n_attack", 0), ood.get("n_benign", 0)))
    print("%-16s %7s %8s" % ("method", "AP", "ROC-AUC"))
    for m in C.METHODS:
        cell = ood.get(m)
        if not isinstance(cell, dict):
            continue
        if "error" in cell:
            print("%-16s  [FAILED]" % m)
            continue
        print("%-16s %7.3f %8.3f"
              % (m, cell.get("ap", float("nan")), cell.get("roc_auc", float("nan"))))

    print(line)
    print("READ: per_trace is the diluted-campaign-DEFEATED baseline. FALSIFIER -- if "
          "kmeans_enrich AP <= per_trace AP at the sparse base rate, 'clustering "
          "surfaces the distributed campaign' is FALSE for this setup.")
    print(line)


if __name__ == "__main__":
    main()
