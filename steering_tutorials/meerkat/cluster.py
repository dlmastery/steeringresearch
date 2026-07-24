"""cluster.py — the embedder, k-means machinery, and the THREE localizers.

This is the analytic heart of the MEERKAT lesson. The pedagogical claim is a
single sentence:

    A safety-violation *campaign* in an agentic repository is SPARSE (~5% of
    traces) and DISTRIBUTED (many near-duplicate escalations). Any monitor that
    scores each trace ALONE sees a faint minority signal drowned in benign
    traffic and misses it. But when you EMBED every trace and CLUSTER the whole
    repository together, the campaign — being self-similar — collapses into one
    tight region of embedding space, and a handful of labelled seeds is enough
    to light that region up. Clustering is what recovers the campaign.

We implement three localizers to make that contrast measurable:

  * ``PerTraceMonitor``  — the BASELINE. A logistic regression trained on the
    labelled seed embeddings, applied to each trace independently. No use of
    cross-trace structure, so at a 5% base rate it has almost no positives to
    learn from and misses the diluted campaign.
  * ``KMeansEnrich``     — the MEERKAT-style localizer. k-means clusters ALL
    repository traces (unsupervised, using every trace), THEN each trace's score
    is the seed-estimated violation FRACTION of the cluster it fell into. The
    clustering is done with all the data; the seed only *labels* the clusters.
  * ``KnnPurity``        — a clustering-free density proxy. Each trace is scored
    by the attack-fraction among its k nearest SEED-labelled neighbours. Same
    intuition (attacks are self-similar so they clump) without committing to a
    hard partition.

All three share the same tiny API so the runner can treat them uniformly:

    m.fit(emb, seed_idx, seed_labels)   # learn from the labelled seed
    scores = m.score(emb)               # P(violation) in [0,1] for EVERY trace

Everything here is CPU-only and model-free EXCEPT ``get_embedder``, which lazily
loads bge/minilm exactly once. The ``__main__`` self-test never touches a model:
it runs on synthetic embeddings and asserts the clustering localizers beat the
per-trace baseline at a 5% base rate — the whole thesis in a unit test.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    silhouette_score,
)

from steering_tutorials.meerkat import config as C


# --- Embedder (the ONLY place a model is loaded; lazy, once) -----------------
def get_embedder(method: str = C.EMBEDDER):
    """Return ``(embed_text, dim)`` for ``method`` in {"bge","minilm"}.

    ``embed_text(text:str) -> np.ndarray[dim]`` (float32, 1-D). Meerkat embeds
    each trace with ``bge-base-en-v1.5``; we load it via plain ``transformers``
    AutoModel + attention-mask mean pooling (NO sentence-transformers
    dependency), matching the multiturn_jailbreak embedder idiom. "minilm" is a
    faster substitute. The model is loaded ONCE here, lazily, and closed over by
    ``embed_text`` — importing this module stays CPU-cheap and model-free.
    """
    if method == "bge":
        model_id = C.EMBED_MODEL
    elif method == "minilm":
        model_id = C.MINILM_ID
    else:
        raise ValueError("method must be 'bge' or 'minilm', got %r" % (method,))

    # Heavy imports live INSIDE the fn so `import cluster` never pulls in torch.
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def embed_text(text: str) -> np.ndarray:
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,   # traces are multi-turn; allow a longer window
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state              # [1, seq, hid]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [1, seq, 1]
        summed = (hidden * mask).sum(dim=1)         # [1, hid]  (sum over real tokens)
        counts = mask.sum(dim=1).clamp(min=1e-9)    # [1, 1]   (# of real tokens)
        pooled = (summed / counts).squeeze(0)       # [hid]    (mean-pool)
        return pooled.detach().cpu().float().numpy().reshape(-1)

    # Probe the true hidden dim from one cheap forward (bge=768, minilm=384).
    dim = int(embed_text("hello").shape[0])
    return embed_text, dim


def embed_traces(traces, embed_text) -> np.ndarray:
    """Embed a list of trace strings -> ``[n, dim]`` float32, L2-normalized rows.

    L2 normalization makes Euclidean k-means behave like cosine clustering (the
    natural geometry for these embeddings): after normalizing, squared Euclidean
    distance is a monotone function of cosine distance, so tight cosine clusters
    stay tight for KMeans and KNN.
    """
    if len(traces) == 0:
        raise ValueError("cannot embed an empty trace list")
    vecs = [np.asarray(embed_text(t), dtype=np.float32).reshape(-1) for t in traces]
    emb = np.stack(vecs, axis=0).astype(np.float32)
    return _l2_normalize(emb)


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)   # never divide by zero
    return (emb / norms).astype(np.float32)


# --- k-means machinery -------------------------------------------------------
def choose_k(emb, k_grid=C.K_GRID, seed=C.SEED):
    """k-means for each k in ``k_grid``; pick the k with the best silhouette.

    Silhouette measures how tight-and-separated the clusters are (in [-1, 1],
    higher is better) WITHOUT any labels — so choosing k this way is fully
    unsupervised, exactly the setting Meerkat operates in. Returns
    ``(best_k, {k: silhouette})``. k's that can't be scored (e.g. k >= n, or a
    degenerate single-cluster solution) are skipped.
    """
    emb = np.asarray(emb, dtype=np.float32)
    n = emb.shape[0]
    scores = {}
    for k in k_grid:
        if k < 2 or k >= n:
            continue
        labels = kmeans_labels(emb, k, seed=seed)
        if len(np.unique(labels)) < 2:
            continue   # silhouette is undefined for a single populated cluster
        try:
            scores[int(k)] = float(silhouette_score(emb, labels))
        except ValueError:
            continue
    if not scores:
        # Fallback: no k scored -> smallest valid k (keeps the pipeline alive).
        best_k = int(min(k for k in k_grid if 2 <= k < n)) if n > 2 else 2
        return best_k, scores
    best_k = int(max(scores, key=scores.get))
    return best_k, scores


def kmeans_labels(emb, k, seed=C.SEED) -> np.ndarray:
    """Return the k-means cluster id (0..k-1) for every trace."""
    emb = np.asarray(emb, dtype=np.float32)
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10)
    return km.fit_predict(emb).astype(int)


# --- The three localizers ----------------------------------------------------
# Shared API:  fit(emb, seed_idx, seed_labels) -> self ; score(emb) -> [n] in [0,1].
#   emb         : [n, dim] L2-normalized embeddings of the WHOLE repository.
#   seed_idx    : indices (into emb) of the small labelled analyst seed.
#   seed_labels : the 0/1 labels for those seed rows (1 = violation/attack).
# score() returns a per-trace P(violation) for EVERY row of emb (seed + rest).

class PerTraceMonitor:
    """BASELINE: logistic regression scoring each trace in ISOLATION.

    ONE idea: treat violation detection as ordinary supervised classification on
    individual trace embeddings, trained on the labelled seed. It has no notion
    that traces come in a repository or that a campaign is a *group* of similar
    traces. At a 5% base rate the seed carries only a couple of positives, so the
    decision boundary is starved of signal and the diluted campaign — spread thin
    across many benign look-alikes — is missed. This is precisely the failure
    Meerkat's clustering is designed to fix, so it is our reference point.
    """

    def fit(self, emb, seed_idx, seed_labels):
        emb = np.asarray(emb, dtype=np.float32)
        seed_idx = np.asarray(seed_idx, dtype=int)
        y = np.asarray(seed_labels, dtype=int)
        self._degenerate = None
        if len(np.unique(y)) < 2:
            # Only one class in the seed -> logistic is undefined. Fall back to a
            # constant score equal to the seed base rate (an honest prior).
            self._degenerate = float(y.mean()) if len(y) else 0.0
            self._clf = None
            return self
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._clf.fit(emb[seed_idx], y)
        # Column index of the positive class, so score() reads P(y=1) correctly.
        self._pos_col = int(np.where(self._clf.classes_ == 1)[0][0])
        return self

    def score(self, emb) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        if self._clf is None:
            return np.full(emb.shape[0], self._degenerate, dtype=np.float32)
        return self._clf.predict_proba(emb)[:, self._pos_col].astype(np.float32)


class KMeansEnrich:
    """MEERKAT-style: cluster ALL traces, score each by its cluster's violation
    enrichment estimated from the seed.

    ONE idea: the repository's own geometry does the heavy lifting. We k-means
    cluster EVERY trace (choose_k picks k by silhouette — unsupervised, using all
    the data), so a self-similar campaign lands together in one cluster REGARDLESS
    of how rare it is. Only THEN do the seed labels enter: within each cluster we
    estimate the violation fraction from whatever seed traces fell into it, and
    every trace in that cluster inherits that fraction as its score.

    Why this recovers a sparse distributed campaign that per-trace scoring can't:
    the signal is in the *co-location* of the campaign traces, which is visible
    only when the whole repo is clustered together, not when each trace is judged
    alone.

    Enrichment math (Laplace / additive smoothing) for cluster c:

        score(c) = (attacks_in_seed(c) + alpha) / (seed_in(c) + 2*alpha)

    with alpha = 1. The +alpha in the numerator and +2*alpha in the denominator
    are a Beta(1,1) uniform prior: a cluster with NO seed labels gets 0.5 (honest
    "unknown"), not 0/0; a cluster with a few all-attack seeds is pulled toward
    but not pinned at 1.0, so tiny seed counts can't produce over-confident 1.0s.
    """

    def __init__(self, k_grid=C.K_GRID, seed=C.SEED, alpha=1.0):
        self.k_grid = k_grid
        self.seed = seed
        self.alpha = float(alpha)

    def fit(self, emb, seed_idx, seed_labels):
        emb = np.asarray(emb, dtype=np.float32)
        seed_idx = np.asarray(seed_idx, dtype=int)
        seed_labels = np.asarray(seed_labels, dtype=int)

        # 1. Cluster the WHOLE repository (unsupervised — uses every trace).
        self.best_k, self.silhouette = choose_k(emb, self.k_grid, self.seed)
        self.cluster_ids = kmeans_labels(emb, self.best_k, seed=self.seed)

        # 2. Laplace-smoothed violation fraction per cluster, from the seed only.
        a = self.alpha
        self.cluster_score = {}
        for c in range(self.best_k):
            in_c = seed_idx[self.cluster_ids[seed_idx] == c]   # seed rows in cluster c
            if len(in_c) == 0:
                self.cluster_score[c] = 0.5     # no evidence -> uniform prior
                continue
            attacks = float(seed_labels[np.isin(seed_idx, in_c)].sum())
            total = float(len(in_c))
            self.cluster_score[c] = (attacks + a) / (total + 2.0 * a)
        return self

    def score(self, emb) -> np.ndarray:
        # Each trace inherits the enrichment of the cluster it was assigned to.
        return np.array(
            [self.cluster_score[int(c)] for c in self.cluster_ids],
            dtype=np.float32,
        )


class KnnPurity:
    """Clustering-free density proxy: score by the attack-fraction of the k
    nearest SEED-labelled traces.

    ONE idea: skip the hard partition entirely. For each trace, look at its
    ``C.KNN_K`` nearest neighbours AMONG the labelled seed and report the fraction
    of them that are attacks. Because a campaign is self-similar, a genuine attack
    trace sits in a dense neighbourhood of other attacks (even a few seeded ones),
    so its purity is high; an isolated benign trace's nearest seeds are benign.
    This captures the same 'attacks clump together' intuition as KMeansEnrich but
    without choosing k or committing every point to one cluster — a softer,
    local-density view. On L2-normalized embeddings Euclidean nearest = cosine
    nearest, so this is a cosine-kNN over the seed set.
    """

    def __init__(self, knn_k=C.KNN_K):
        self.knn_k = int(knn_k)

    def fit(self, emb, seed_idx, seed_labels):
        emb = np.asarray(emb, dtype=np.float32)
        self.seed_emb = emb[np.asarray(seed_idx, dtype=int)]
        self.seed_labels = np.asarray(seed_labels, dtype=int)
        # Cap k at the seed size so we never ask for more neighbours than exist.
        self.k = int(min(self.knn_k, len(self.seed_labels)))
        return self

    def score(self, emb) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        if self.k == 0:
            return np.zeros(emb.shape[0], dtype=np.float32)
        # Cosine similarity to every seed (rows are L2-normalized -> dot = cosine).
        sims = emb @ self.seed_emb.T                 # [n, n_seed]
        # Indices of the k most-similar seeds for each trace (unsorted top-k).
        topk = np.argpartition(-sims, self.k - 1, axis=1)[:, : self.k]  # [n, k]
        neigh_labels = self.seed_labels[topk]        # [n, k] of 0/1
        return neigh_labels.mean(axis=1).astype(np.float32)   # attack-fraction


# --- Metrics -----------------------------------------------------------------
def average_precision(y_true, y_score) -> float:
    """Average Precision — the Meerkat headline metric.

    AP (area under the precision-recall curve) is the right metric for a SPARSE
    positive class: unlike ROC-AUC it is not flattered by the huge benign
    majority, so a detector that only surfaces the 5% campaign near the top of
    the ranking is properly rewarded.
    """
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")   # AP undefined with a single class present
    return float(average_precision_score(y_true, np.asarray(y_score, dtype=float)))


def roc_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))


def cluster_purity(cluster_ids, labels) -> float:
    """Weighted mean of each cluster's majority-class fraction.

    For every cluster, take the fraction of its members belonging to its most
    common label; average those over clusters weighted by cluster size. 1.0 means
    every cluster is label-pure (the campaign is perfectly isolated); ~base-rate
    means clustering mixed attacks and benign together.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0:
        return float("nan")
    total = 0.0
    for c in np.unique(cluster_ids):
        members = labels[cluster_ids == c]
        counts = np.bincount(members)
        total += counts.max()   # size * (majority fraction) = majority count
    return float(total / n)


# --- CPU self-test on SYNTHETIC embeddings (NO model) ------------------------
def _make_synthetic(n=400, dim=32, base_rate=0.05, seed=0, n_modes=4):
    """A sparse (5%) MULTI-MODAL attack campaign a linear monitor cannot catch.

    If the campaign were a single blob on one side of embedding space, the
    per-trace logistic would trivially separate it and there would be no lesson.
    Real distributed misuse has SEVERAL escalation shapes (ActorAttack rewrites a
    goal many ways), so the campaign is MULTI-MODAL: several tight blobs pointing
    in DIFFERENT (here mutually orthogonal / antipodal) directions, with the
    benign cloud sitting in the MIDDLE of them.

    Why this is the honest failure of a per-trace monitor: logistic regression is
    LINEAR. One hyperplane cannot put four blobs pointing in opposite directions
    all on the far side of the benign cloud between them -- scoring one mode high
    forces its antipode low. So the per-trace monitor recovers at most a couple of
    modes and its Average-Precision collapses. k-means (one centroid PER mode) and
    kNN-purity (a purely LOCAL vote) are naturally multi-modal, so they light up
    every mode. That multi-modal density structure is exactly what per-trace
    scoring discards.
    """
    rng = np.random.default_rng(seed)
    n_pos = max(n_modes, int(round(n * base_rate)))
    n_neg = n - n_pos

    # Mode directions: +e0, -e0, +e1, -e1, ... -> mutually orthogonal/antipodal,
    # with benign centred at the origin BETWEEN them (no separating hyperplane).
    centers = np.zeros((n_modes, dim), dtype=np.float32)
    for m in range(n_modes):
        axis = m // 2
        sign = 1.0 if (m % 2 == 0) else -1.0
        centers[m, axis] = sign * 4.0

    # Spread the positives across the modes (a self-similar sub-campaign each).
    per_mode = [n_pos // n_modes + (1 if i < n_pos % n_modes else 0)
                for i in range(n_modes)]
    attack_parts, attack_mode = [], []
    for m, cnt in enumerate(per_mode):
        blob = centers[m] + rng.standard_normal((cnt, dim)).astype(np.float32) * 0.10
        attack_parts.append(blob)
        attack_mode.extend([m] * cnt)
    attack = np.vstack(attack_parts).astype(np.float32)

    benign = rng.standard_normal((n_neg, dim)).astype(np.float32) * 1.0

    emb = np.vstack([attack, benign]).astype(np.float32)
    labels = np.concatenate([np.ones(n_pos, int), np.zeros(n_neg, int)])
    modes = np.concatenate([np.asarray(attack_mode, int), -np.ones(n_neg, int)])

    order = rng.permutation(n)   # shuffle so attacks are not contiguous
    emb, labels, modes = emb[order], labels[order], modes[order]
    return _l2_normalize(emb), labels, modes


def _self_test() -> None:
    emb, labels, modes = _make_synthetic(n=400, dim=32, base_rate=0.05, seed=0)
    n = len(labels)
    print("synthetic repo: n=%d  attacks=%d (modes=%d)  base_rate=%.3f"
          % (n, int(labels.sum()), len(np.unique(modes[modes >= 0])), labels.mean()))

    # Labelled seed = an analyst who has confirmed a couple of traces PER mode
    # (so every escalation shape has at least one seed positive) plus a sample of
    # benign. Coverage-per-mode is what lets kNN/enrich light up all modes; the
    # per-trace monitor gets the SAME labels yet still cannot separate them.
    rng = np.random.default_rng(1)
    seed_pos = []
    for m in np.unique(modes[modes >= 0]):
        idx_m = np.where(modes == m)[0]
        take = max(3, int(round(len(idx_m) * 0.8)))
        seed_pos.append(rng.choice(idx_m, size=min(take, len(idx_m)), replace=False))
    seed_pos = np.concatenate(seed_pos)
    neg = np.where(labels == 0)[0]
    seed_neg = rng.choice(neg, size=int(round(len(neg) * 0.10)), replace=False)
    seed_idx = np.concatenate([seed_pos, seed_neg])
    seed_labels = labels[seed_idx]

    # NOTE on the kNN k: the real repos hold hundreds of attacks so C.KNN_K=15
    # neighbours is sensible; this TINY synthetic has only ~20 positives, so a
    # 15-vote window is larger than the per-mode seed count and washes the signal
    # out. We use a synthetic-appropriate small k here (the mechanism is identical
    # — a local same-mode density vote).
    results = {}
    for name, m in [
        ("per_trace", PerTraceMonitor()),
        ("kmeans_enrich", KMeansEnrich()),
        ("knn_purity", KnnPurity(knn_k=6)),
    ]:
        m.fit(emb, seed_idx, seed_labels)
        ap = average_precision(labels, m.score(emb))
        results[name] = ap
        print("  %-14s AP=%.4f" % (name, ap))

    # Cluster-quality sanity print (does the campaign concentrate in one cluster?).
    best_k, sil = choose_k(emb)
    cids = kmeans_labels(emb, best_k)
    print("  choose_k -> k=%d  cluster_purity=%.4f" % (best_k, cluster_purity(cids, labels)))

    # The thesis: BOTH clustering localizers must beat the per-trace baseline at 5%.
    margin = 0.10
    assert results["kmeans_enrich"] > results["per_trace"] + margin, (
        "KMeansEnrich (%.4f) must exceed PerTraceMonitor (%.4f) by >%.2f"
        % (results["kmeans_enrich"], results["per_trace"], margin))
    assert results["knn_purity"] > results["per_trace"] + margin, (
        "KnnPurity (%.4f) must exceed PerTraceMonitor (%.4f) by >%.2f"
        % (results["knn_purity"], results["per_trace"], margin))
    print("cluster.py self-test OK: clustering localizers >> per-trace at 5% base rate")


if __name__ == "__main__":
    _self_test()
