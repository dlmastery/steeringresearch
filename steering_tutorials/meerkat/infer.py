"""infer.py — watch a sparse misuse campaign HIDE from the per-trace monitor and
get SURFACED by clustering the whole repository.

Run:
    python -m steering_tutorials.meerkat.infer

The default builds ONE small repository at the paper's sparse base rate (~5%
attack traces diluted among benign traffic), embeds every trace, k-means clusters
them, and then prints the contrast that IS the lesson:

  * the PER-TRACE MONITOR (a logistic head on each trace embedding alone) top-
    scored traces -- it scores traces in isolation, so the diluted campaign never
    stands out and its top-K is mostly benign false positives; it MISSES.
  * the K-MEANS CLUSTER the attack campaign concentrates in -- because the campaign
    traces share an escalation shape, they land in ONE tight cluster. Cluster-
    enrichment (Meerkat-style) scores a trace by its cluster's violation fraction,
    so the whole campaign lights up at once; it SURFACES.

Per-trace looks at one dot at a time; clustering looks at the shape of the cloud.
That difference is the entire lesson. All heavy imports (numpy / the embedder /
the datasets) live inside main(), so `python -c "import ...infer"` stays CPU-cheap
and model-free.

Env caps (shrink the demo into one foreground window -- host RAM is the wall):
    MK_INFER_REPO   traces in the demo repository        (default 200)
    MK_BASE_RATE    sparse attack fraction               (default 0.05, from config)
    MK_EMBED        "bge" | "minilm"                      (default from config)
    MK_TOPK         rows to print per localizer          (default 10)
"""
from __future__ import annotations

import math
import os

BAR = "=" * 72


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _short(text: str, width: int = 52) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _print_top(title, order, traces, labels, scores, k):
    """Print the top-k traces a localizer ranks highest, flagging true attacks."""
    print("")
    print("--- %s ---" % title)
    print("  rank  score  truth   trace")
    hits = 0
    for rank, idx in enumerate(order[:k], start=1):
        is_atk = int(labels[idx]) == 1
        hits += is_atk
        print("  %4d  %.3f  %-6s  %s"
              % (rank, float(scores[idx]),
                 "ATTACK" if is_atk else "benign", _short(traces[idx])))
    print("  ................................................................")
    print("  attacks in top-%d: %d / %d" % (k, hits, k))
    return hits


def main() -> None:
    import numpy as np

    from . import config as C
    from . import data, cluster

    repo_size = _env_int("MK_INFER_REPO", 200)
    base_rate = C.BASE_RATE
    topk = _env_int("MK_TOPK", 10)

    # traces needed at the sparse rate, plus a small margin so the pool is not
    # exactly the repository (sampling should have something to choose from).
    n_atk_repo = max(1, int(round(repo_size * base_rate)))
    n_ben_repo = repo_size - n_atk_repo

    print(BAR)
    print(" meerkat infer -- a sparse misuse campaign hides from a per-trace monitor")
    print(BAR)
    print("Building a repository of %d traces at a %.0f%% sparse attack rate"
          % (repo_size, 100.0 * base_rate))
    print("(~%d attack traces diluted among ~%d benign traces)."
          % (n_atk_repo, n_ben_repo))
    print("Loading the '%s' trace embedder (local model, one lazy load)..."
          % C.EMBEDDER)

    # 1) Build a POOL, then SAMPLE one sparse repository from it.
    pool = data.load_trace_pool(
        n_attack=n_atk_repo + 8,
        n_benign=n_ben_repo + 8,
        seed=C.SEED,
    )
    repo = data.sample_repository(pool, size=repo_size, base_rate=base_rate, seed=0)
    traces = list(repo["traces"])
    labels = np.asarray(repo["labels"], dtype=int)
    n = len(traces)
    n_atk = int(labels.sum())
    print("  repository: %d traces (%d attack / %d benign)"
          % (n, n_atk, n - n_atk))

    # 2) Embed every trace (one model load inside get_embedder).
    embed_text, dim = cluster.get_embedder(C.EMBEDDER)
    emb = cluster.embed_traces(traces, embed_text)
    print("  embedder=%s  dim=%d  embeddings=%s" % (C.EMBEDDER, dim, tuple(emb.shape)))

    # 3) A small labelled SEED -- the analyst who hand-labels a handful of traces.
    #    Both localizers see the SAME seed labels; only how they GENERALIZE differs.
    rng = np.random.default_rng(C.SEED)
    n_seed = max(min(int(math.ceil(C.SEED_FRAC * n)), n), 8)
    seed_idx = rng.choice(n, size=n_seed, replace=False)
    seed_labels = labels[seed_idx]
    print("  labelled seed: %d / %d traces (%d attack)"
          % (n_seed, n, int(seed_labels.sum())))

    # 4) Fit both localizers on the seed, score ALL traces.
    per_trace = cluster.PerTraceMonitor()
    per_trace.fit(emb, seed_idx, seed_labels)
    s_per = np.asarray(per_trace.score(emb), dtype=float).reshape(-1)

    enrich = cluster.KMeansEnrich()
    enrich.fit(emb, seed_idx, seed_labels)
    s_enr = np.asarray(enrich.score(emb), dtype=float).reshape(-1)

    order_per = np.argsort(-s_per)
    order_enr = np.argsort(-s_enr)

    # 5) THE BASELINE: per-trace monitor top-K -- scores each trace alone, so the
    #    diluted campaign never stands out.
    hits_per = _print_top(
        "PER-TRACE MONITOR top-%d (scores each trace ALONE -> misses the campaign)" % topk,
        order_per, traces, labels, s_per, topk)

    # 6) THE CAMPAIGN CLUSTER: k-means the whole repo, find the cluster the attack
    #    campaign concentrates in (true labels used here only to NARRATE the demo).
    best_k, sil = cluster.choose_k(emb, k_grid=C.K_GRID, seed=C.SEED)
    cluster_ids = np.asarray(cluster.kmeans_labels(emb, best_k, seed=C.SEED), dtype=int)
    frac, campaign_cluster = -1.0, -1
    for cid in range(best_k):
        members = cluster_ids == cid
        if not members.any():
            continue
        f = float(labels[members].mean())
        if f > frac:
            frac, campaign_cluster = f, cid
    members = cluster_ids == campaign_cluster
    n_members = int(members.sum())
    n_atk_in = int(labels[members].sum())
    recall = (n_atk_in / n_atk) if n_atk else 0.0
    print("")
    print("--- K-MEANS found %d clusters (best silhouette over %s) ---"
          % (best_k, C.K_GRID))
    print("  the attack campaign concentrates in cluster #%d:" % campaign_cluster)
    print("    size=%d  attacks=%d  purity=%.0f%% attack  campaign-recall=%.0f%%"
          % (n_members, n_atk_in, 100.0 * frac, 100.0 * recall))

    # 7) CLUSTER-ENRICHMENT top-K -- one seed label per cluster lifts the WHOLE
    #    campaign at once.
    hits_enr = _print_top(
        "CLUSTER-ENRICHMENT top-%d (scores by cluster violation fraction -> surfaces it)" % topk,
        order_enr, traces, labels, s_enr, topk)

    # 8) The headline single number: Average Precision (the Meerkat metric).
    ap_per = float(cluster.average_precision(labels, s_per))
    ap_enr = float(cluster.average_precision(labels, s_enr))

    print("")
    print(BAR)
    print("  Average Precision (the campaign is %.0f%% of the repository):" % (100.0 * base_rate))
    print("    per-trace monitor    AP = %.3f   (attacks in top-%d: %d)"
          % (ap_per, topk, hits_per))
    print("    cluster-enrichment   AP = %.3f   (attacks in top-%d: %d)"
          % (ap_enr, topk, hits_enr))
    print(BAR)
    print("The lesson: a per-trace monitor scores each trace in ISOLATION, so a")
    print("sparse campaign diluted to ~5%% never stands out -- its top-K is mostly")
    print("benign false positives. Clustering reads the SHAPE of the whole cloud:")
    print("the campaign's shared escalation lands in ONE cluster, and enrichment")
    print("scoring lifts every member at once. This is the UNSUPERVISED sibling of")
    print("cross_trajectory (which LEARNS a set-aggregator over K agent traces).")
    print(BAR)


if __name__ == "__main__":
    main()
