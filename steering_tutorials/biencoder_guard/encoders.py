"""encoders.py -- the three guardrail towers + the bi-vs-uni scaling benchmark.

This is the mechanical heart of the lesson. It teaches ONE contrast three ways:

    How do you decide whether a piece of content violates each of P safety
    policies, when P is large and new policies appear all the time?

  * BI-ENCODER (the hero).  Two INDEPENDENT towers. The CONTENT tower embeds the
    text once; the POLICY tower embeds each policy DESCRIPTION once. Compatibility
    is a cheap cosine in the shared space. Because a policy vector does not depend
    on the text, the whole policy bank is embedded ONCE and cached -> per-request
    cost is ~constant in the number of labels, and a brand-new policy is scored
    ZERO-SHOT the moment you write its description. This is why it scales and why
    it generalizes to unseen columns.

  * UNI-ENCODER (cross-encoder-lite).  ONE tower that FUSES text + policy: it
    embeds the joint string "moderate: {text}\npolicy: {desc}" and reads a
    compatibility score off a small head. Fusing the two sides lets the encoder
    attend text against policy (often more accurate), but the joint vector depends
    on BOTH sides, so nothing can be cached: scoring n texts against K policies
    costs n*K encoder calls. That linear-in-K cost is the scaling bottleneck this
    lesson demonstrates. It can still score a held-out policy (rebuild the joint,
    reuse the head), just never cheaply.

  * TRAINED-HEAD (the supervised ceiling).  A one-vs-rest logistic regression on
    the frozen content embedding, one weight vector per SEEN policy. Strong where
    it was trained -- but a policy it never saw has NO weight vector, so it must
    ABSTAIN (returns NaN). This is the concrete price of supervised specialization:
    no zero-shot.

All three expose the same tiny surface so the runner can swap them:
    guard.fit(Xc_train, Y_train, seen_cols, policies, texts_train=None)
    guard.scores(Xc, policy_bank, cols, texts=None) -> [n, len(cols)] in [0,1]
`Xc` is a matrix of PRECOMPUTED, L2-normalized CONTENT embeddings; `cols` selects
which policy columns to score. `texts_train`/`texts` are optional and used ONLY by
the uni-encoder, which must re-embed joint strings (the bi-encoder and trained-head
ignore them). Scores are always mapped to [0,1] with higher = "policy applies".

ASCII-only stdout (Windows cp1252): we write "cos", "F1", ">=", never unicode.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from steering_tutorials.biencoder_guard import config as C


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize so a dot product IS the cosine similarity.

    Every embedding in this lesson lives on the unit sphere; that is what lets us
    replace the (expensive) cosine formula with a single matrix multiply.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(norms, eps)).astype(np.float32)


def _truncate(X: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation: keep the first `dim` coordinates (or fewer)."""
    X = np.asarray(X, dtype=np.float32)
    d = min(int(dim), X.shape[-1])
    return X[..., :d]


def _cos_to_unit(cos: np.ndarray) -> np.ndarray:
    """Map a cosine in [-1, 1] to a probability-like score in [0, 1].

    cosine measures direction agreement; (cos + 1) / 2 rescales "opposite" -> 0,
    "orthogonal" -> 0.5, "identical" -> 1 so the bi-encoder emits the same [0,1]
    range as the two learned heads and all three are directly comparable.
    """
    return np.clip((np.asarray(cos, dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# the shared backbone: one embedder, two asymmetric "kinds"
# ---------------------------------------------------------------------------
class _SentenceEmbedder:
    """Thin wrapper over a sentence-transformers model exposing `.encode(texts,kind)`.

    ONE idea: give both towers the SAME backbone but let them speak with different
    task PROMPTS. EmbeddingGemma is trained for asymmetric retrieval -- content
    ("query") and policy ("document") are encoded under different prompts so they
    land in a comparable-but-role-aware space. `kind` in {"content","policy"} picks
    the prompt; the returned rows are truncated to C.EMB_DIM and L2-normalized.
    """

    # Documented task-prefix fallbacks (from the EmbeddingGemma model card) used
    # when the loaded model does not register named prompts (e.g. MiniLM).
    _FALLBACK_PREFIX = {
        "content": "task: search result | query: ",
        "policy": "title: none | text: ",
    }

    def __init__(self, model, use_prompts: bool, dim: int):
        self._model = model            # a sentence_transformers.SentenceTransformer
        self._use_prompts = use_prompts  # True for EmbeddingGemma, False for MiniLM
        self._dim = int(dim)

    def _prompt_name(self, kind: str) -> str:
        return C.CONTENT_PROMPT if kind == "content" else C.POLICY_PROMPT

    def encode(self, texts, kind: str) -> np.ndarray:
        if kind not in ("content", "policy"):
            raise ValueError("kind must be 'content' or 'policy', got %r" % (kind,))
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        emb = None
        if self._use_prompts:
            # Preferred path: named task prompts baked into the model.
            try:
                emb = self._model.encode(
                    texts,
                    prompt_name=self._prompt_name(kind),
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            except (ValueError, KeyError, TypeError):
                emb = None  # model lacks that named prompt -> fall back below
        if emb is None:
            # Fallback: prepend the documented task-prefix string ourselves.
            prefix = self._FALLBACK_PREFIX[kind] if self._use_prompts else ""
            emb = self._model.encode(
                [prefix + t for t in texts],
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        emb = _truncate(np.asarray(emb, dtype=np.float32), self._dim)
        return _l2_normalize(emb)


def get_embedder(method: str = C.EMBEDDER):
    """Load the shared bi-encoder backbone ONCE (lazily) and wrap it.

    "embeddinggemma" -> google/embeddinggemma-300m via sentence-transformers, with
    the gated HF id retried from the local snapshot (C.EMBED_LOCAL) on failure.
    "minilm" -> all-MiniLM-L6-v2, a fast ungated substitute for dry runs (no task
    prompts). The heavy import + model load happen HERE, never at module import, so
    `import encoders` stays CPU-cheap and the synthetic self-test never touches a
    model. Returns an object exposing `.encode(texts, kind) -> [n, C.EMB_DIM]`.
    """
    from sentence_transformers import SentenceTransformer  # lazy, heavy

    if method == "minilm":
        model = SentenceTransformer(C.MINILM_ID)
        return _SentenceEmbedder(model, use_prompts=False, dim=C.EMB_DIM)

    if method == "embeddinggemma":
        try:
            model = SentenceTransformer(C.EMBED_MODEL)   # gated HF id
        except Exception as exc:  # 401 / gated / offline -> local snapshot
            print("[encoders] HF load failed (%s); retrying local %s"
                  % (type(exc).__name__, C.EMBED_LOCAL))
            model = SentenceTransformer(C.EMBED_LOCAL)
        return _SentenceEmbedder(model, use_prompts=True, dim=C.EMB_DIM)

    raise ValueError("method must be 'embeddinggemma' or 'minilm', got %r" % (method,))


# ---------------------------------------------------------------------------
# the cached POLICY tower (multi-prototype = synthetic schema expansion)
# ---------------------------------------------------------------------------
def build_policy_bank(policies, embedder, n_proto: int = C.POLICY_PARAPHRASES) -> np.ndarray:
    """Embed every policy into ONE robust vector -> the cacheable [P, dim] bank.

    Multi-PROTOTYPE idea (the 2026 "synthetic schema expansion" practice): a single
    hand-written description is a noisy sample of a policy's meaning. Embedding the
    description PLUS several paraphrases and AVERAGING them cancels phrasing-specific
    noise and yields a more central, more transferable policy vector -- exactly the
    kind of robustness that pays off ZERO-SHOT on unseen policies. n_proto == 1
    reduces to "description only" (the ablation baseline). The averaged vector is
    re-normalized so it stays a unit direction for cosine scoring.
    """
    n_proto = max(1, int(n_proto))
    vecs = []
    for pol in policies:
        phrasings = [pol["description"]]
        if n_proto > 1:
            phrasings += list(pol.get("paraphrases", []))[: n_proto - 1]
        emb = embedder.encode(phrasings, "policy")   # [n_phrasings, dim], unit rows
        proto = emb.mean(axis=0, keepdims=True)      # average the prototypes
        vecs.append(_l2_normalize(proto)[0])         # re-normalize -> unit vector
    return np.asarray(np.stack(vecs, axis=0), dtype=np.float32)


# ---------------------------------------------------------------------------
# guard 1: BI-ENCODER  (hero: cached bank, zero-shot on any column)
# ---------------------------------------------------------------------------
class BiEncoderGuard:
    """Score = cosine(content, cached policy vector), mapped to [0,1].

    ONE idea: the two towers are DECOUPLED, so the policy bank is text-independent
    and cached. Scoring is a single matmul of content vectors against the bank ->
    O(1) encoder work per request regardless of label count, and ANY column can be
    scored -- including HELD-OUT policies never seen in training (zero-shot from the
    description alone). `fit` learns NO weights; it only calibrates a per-column
    decision threshold on the training scores (used for F1 reporting), so the method
    stays fully zero-shot on unseen columns (which simply reuse the global default).
    """

    def __init__(self):
        self.thresholds: dict[int, float] = {}
        self._default_threshold = 0.5

    def fit(self, Xc_train, Y_train, seen_cols, policies, texts_train=None):
        # No parameters are learned -- the bi-encoder is a fixed geometry. `fit`
        # only seeds a per-SEEN-column decision threshold (default 0.5). Proper
        # F1 calibration needs the train POLICY BANK, which is produced by an
        # embedder the runner owns, so the runner may call `calibrate(...)` with a
        # precomputed bank to sharpen these thresholds. Unseen columns keep the
        # default, which is exactly why scoring stays fully zero-shot.
        self._thr_grid = np.asarray(C.THRESHOLD_GRID, dtype=np.float32)
        self.thresholds = {int(c): self._default_threshold for c in seen_cols}
        return self

    def calibrate(self, Xc_train, Y_train, seen_cols, policy_bank):
        """Optional: sharpen per-column thresholds using a precomputed train bank."""
        S = self.scores(Xc_train, policy_bank, list(seen_cols))
        for j, col in enumerate(seen_cols):
            y = np.asarray(Y_train)[:, col]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            best_t, best_f1 = self._default_threshold, -1.0
            for t in self._thr_grid:
                f1 = f1_score(y, (S[:, j] >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            self.thresholds[int(col)] = best_t
        return self

    def scores(self, Xc, policy_bank, cols, texts=None):
        # Pure geometry: unit content rows dotted with unit policy rows == cosine.
        Xc = _l2_normalize(np.asarray(Xc, dtype=np.float32))
        bank = _l2_normalize(np.asarray(policy_bank, dtype=np.float32))
        cos = Xc @ bank[list(cols)].T          # [n, len(cols)] cosine similarities
        return _cos_to_unit(cos)               # -> [0,1], higher = policy applies


# ---------------------------------------------------------------------------
# guard 2: UNI-ENCODER  (cross-encoder-lite: fuse text+policy; does NOT scale)
# ---------------------------------------------------------------------------
class UniEncoderGuard:
    """Fuse text + policy into ONE joint string, embed it, read a logistic head.

    ONE idea: instead of two cached towers, build the JOINT string
    "moderate: {text}\npolicy: {desc}" and embed THAT, then map the joint vector to
    P(policy applies) with a single shared logistic head. Fusing the sides lets the
    encoder relate the content to the specific policy -- but the joint vector depends
    on BOTH text and policy, so NOTHING is cacheable: scoring n texts x K policies
    needs n*K encoder calls. That is the scaling bottleneck. It can still score a
    HELD-OUT policy (rebuild the joint, reuse the head), just never cheaply, and it
    needs the raw `texts` (not just precomputed content embeddings) to work.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder    # the shared backbone (needed to embed joints)
        self.head = None            # one LogisticRegression over joint embeddings
        self.policies = None

    @staticmethod
    def _joint(text: str, policy) -> str:
        # The fusion template. desc carries the policy semantics into the encoder.
        return "moderate: %s\npolicy: %s" % (str(text), policy["description"])

    def encode_joint(self, texts, policies, cols) -> dict:
        """Embed the (text x col) joint strings -> {col: [n, dim]} content embeddings.

        This is where the cost lives: one encoder call per (text, policy). We expose
        it so the runner (and the scaling benchmark's intuition) can see the fusion
        explicitly. Requires a live embedder.
        """
        if self.embedder is None:
            raise RuntimeError("UniEncoderGuard needs an embedder to encode joints")
        out = {}
        for col in cols:
            strs = [self._joint(t, policies[col]) for t in texts]
            out[int(col)] = self.embedder.encode(strs, "content")
        return out

    def fit(self, Xc_train, Y_train, seen_cols, policies, texts_train=None):
        # Train ONE shared head on joint embeddings pooled over all SEEN columns.
        # For each seen policy we take its positives + an equal random sample of
        # negatives (keeps the joint-encoding cost bounded and the head balanced).
        if texts_train is None:
            raise ValueError("UniEncoderGuard.fit requires texts_train (raw strings)")
        if self.embedder is None:
            raise RuntimeError("UniEncoderGuard needs an embedder to fit")
        self.policies = policies
        Y_train = np.asarray(Y_train)
        rng = np.random.default_rng(C.SEED)
        Xj, yj = [], []
        for col in seen_cols:
            pos = np.where(Y_train[:, col] == 1)[0]
            neg = np.where(Y_train[:, col] == 0)[0]
            if len(pos) == 0:
                continue
            k = min(len(neg), max(len(pos), 1))
            if len(neg) > k:
                neg = rng.choice(neg, size=k, replace=False)
            idx = np.concatenate([pos, neg])
            strs = [self._joint(texts_train[i], policies[col]) for i in idx]
            emb = self.embedder.encode(strs, "content")
            Xj.append(emb)
            yj.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
        if not Xj:
            raise ValueError("UniEncoderGuard.fit: no positive examples in seen_cols")
        Xj = np.vstack(Xj)
        yj = np.concatenate(yj)
        self.head = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xj, yj)
        return self

    def scores(self, Xc, policy_bank, cols, texts=None):
        # Re-encode a joint string per (text, col) -> the n*K cost. `texts` required.
        if texts is None:
            raise ValueError("UniEncoderGuard.scores requires the raw `texts`")
        if self.head is None:
            raise RuntimeError("UniEncoderGuard.scores called before fit")
        joint = self.encode_joint(texts, self.policies, cols)
        out = np.zeros((len(texts), len(cols)), dtype=np.float32)
        for j, col in enumerate(cols):
            out[:, j] = self.head.predict_proba(joint[int(col)])[:, 1]
        return out


# ---------------------------------------------------------------------------
# guard 3: TRAINED-HEAD  (supervised OvR; strong on seen, abstains on unseen)
# ---------------------------------------------------------------------------
class TrainedHeadGuard:
    """One-vs-rest logistic regression on the frozen content embedding.

    ONE idea: learn a dedicated weight vector per SEEN policy directly on the
    content embedding. This is the supervised ceiling -- it can fit the training
    labels closely -- but each weight vector is tied to a specific column, so a
    HELD-OUT policy has NO classifier and the guard must ABSTAIN, returning np.nan
    for that column. That NaN is the whole point: supervised specialization buys
    accuracy on seen labels at the cost of any zero-shot ability.
    """

    def __init__(self):
        self.heads: dict[int, object] = {}   # col -> fitted classifier (or constant)

    def fit(self, Xc_train, Y_train, seen_cols, policies, texts_train=None):
        Xc_train = _l2_normalize(np.asarray(Xc_train, dtype=np.float32))
        Y_train = np.asarray(Y_train)
        for col in seen_cols:
            y = Y_train[:, col].astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                # Degenerate column: no two classes -> store the constant rate.
                self.heads[int(col)] = float(y.mean())
                continue
            self.heads[int(col)] = LogisticRegression(
                max_iter=1000, class_weight="balanced").fit(Xc_train, y)
        return self

    def scores(self, Xc, policy_bank, cols, texts=None):
        Xc = _l2_normalize(np.asarray(Xc, dtype=np.float32))
        out = np.full((Xc.shape[0], len(cols)), np.nan, dtype=np.float32)
        for j, col in enumerate(cols):
            head = self.heads.get(int(col))
            if head is None:
                continue                      # unseen policy -> abstain (NaN column)
            if isinstance(head, float):
                out[:, j] = head              # degenerate constant predictor
            else:
                out[:, j] = head.predict_proba(Xc)[:, 1]
        return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def per_col_ap(Y_true, S) -> np.ndarray:
    """Average precision per policy column (NaN where a column is unscored/empty).

    AP is the area under precision-recall -- the right metric for rare, imbalanced
    safety labels. A column that has no positives (undefined AP) or is all-NaN
    (an abstaining head) returns NaN so callers can skip it in the macro average.
    """
    Y_true = np.asarray(Y_true)
    S = np.asarray(S, dtype=np.float32)
    P = S.shape[1]
    aps = np.full(P, np.nan, dtype=np.float32)
    for j in range(P):
        col = S[:, j]
        if np.all(np.isnan(col)):
            continue
        y = Y_true[:, j]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aps[j] = average_precision_score(y, col)
    return aps


def macro_micro(Y_true, S, thresholds=None) -> dict:
    """Macro/micro AP + macro/micro F1 over the scored columns.

    macro_* averages the per-column metric (each policy counts equally -- exposes
    rare-label failure); micro_* pools all (text, policy) decisions (dominated by
    common labels). NaN columns (abstentions / empty) are excluded. `thresholds`
    may be a scalar or per-column array for binarizing F1; default 0.5.
    """
    Y_true = np.asarray(Y_true)
    S = np.asarray(S, dtype=np.float32)
    P = S.shape[1]
    aps = per_col_ap(Y_true, S)
    valid = ~np.isnan(aps)

    # --- F1 thresholds ---
    if thresholds is None:
        thr = np.full(P, 0.5, dtype=np.float32)
    elif np.isscalar(thresholds):
        thr = np.full(P, float(thresholds), dtype=np.float32)
    else:
        thr = np.asarray(thresholds, dtype=np.float32)

    macro_f1s, micro_y, micro_s, micro_pred = [], [], [], []
    for j in range(P):
        col = S[:, j]
        if np.all(np.isnan(col)):
            continue
        y = Y_true[:, j].astype(int)
        pred = (col >= thr[j]).astype(int)
        macro_f1s.append(f1_score(y, pred, zero_division=0))
        micro_y.append(y)
        micro_s.append(col)
        micro_pred.append(pred)

    macro_ap = float(np.nanmean(aps[valid])) if valid.any() else float("nan")
    if micro_y:
        my = np.concatenate(micro_y)
        ms = np.concatenate(micro_s)
        mp = np.concatenate(micro_pred)
        micro_ap = float(average_precision_score(my, ms)) if 0 < my.sum() < len(my) \
            else float("nan")
        micro_f1 = float(f1_score(my, mp, zero_division=0))
        macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else float("nan")
    else:
        micro_ap = micro_f1 = macro_f1 = float("nan")

    return {"macro_ap": macro_ap, "micro_ap": micro_ap,
            "macro_f1": macro_f1, "micro_f1": micro_f1}


def binary_harm_auc(is_harmful, S_any) -> float:
    """ROC-AUC of the any-policy harmful score vs the binary harmful label.

    S_any is the per-text "does ANY policy fire" score (e.g. the row-max over
    columns). AUC asks: ranked by that score, are harmful texts above benign ones?
    """
    is_harmful = np.asarray(is_harmful).astype(int)
    S_any = np.asarray(S_any, dtype=np.float32)
    if is_harmful.sum() == 0 or is_harmful.sum() == len(is_harmful):
        return float("nan")
    return float(roc_auc_score(is_harmful, S_any))


# ---------------------------------------------------------------------------
# the scaling microbenchmark: bi FLAT vs uni LINEAR in #labels
# ---------------------------------------------------------------------------
def scaling_latency(embedder, texts, n_labels_grid=C.LABEL_SCALES) -> dict:
    """Time moderation as the label bank grows -> the "million-label" claim.

    bi_encoder: embed the text batch ONCE, then score against K CACHED policy
      vectors with a single matmul -> cost is ~flat in K (only the matmul grows,
      and it is negligible next to the one-time text embedding).
    uni_encoder: must embed a JOINT string per (text, policy) -> n_texts * K encoder
      calls, so wall time grows ~linearly in K. We sub-sample the text batch for the
      uni timing (its cost explodes at large K); the SHAPE flat-vs-linear is the
      teaching point, not the absolute seconds. The real taxonomy is padded with
      synthetic "policy N" descriptions up to max(grid).
    Returns {"bi": {K: sec}, "uni": {K: sec}}.
    """
    texts = [str(t) for t in list(texts)[: C.SCALE_BATCH]]
    grid = sorted(int(k) for k in n_labels_grid)
    max_k = grid[-1]
    # Synthetic policy descriptions padding the bank up to the largest K.
    synth = ["policy number %d: content that violates safety rule %d in some way"
             % (i, i) for i in range(max_k)]

    bi: dict[int, float] = {}
    uni: dict[int, float] = {}

    # --- bi: one-time content embed + one-time full policy bank, then matmul ---
    t0 = time.perf_counter()
    Xc = embedder.encode(texts, "content")            # embed the batch ONCE
    embed_sec = time.perf_counter() - t0
    bank_all = embedder.encode(synth, "policy")       # cached policy tower (once)

    # Uni sub-samples texts so n*K stays runnable even at K=1024.
    n_uni = min(len(texts), max(1, C.SCALE_BATCH // 16))

    for K in grid:
        bank = bank_all[:K]
        t0 = time.perf_counter()
        _ = Xc @ bank.T                               # the only per-K bi work
        matmul_sec = time.perf_counter() - t0
        bi[K] = float(embed_sec + matmul_sec)         # flat: embed dominates

        joints = ["moderate: %s\npolicy: %s" % (texts[i], synth[k])
                  for i in range(n_uni) for k in range(K)]
        t0 = time.perf_counter()
        _ = embedder.encode(joints, "content")        # n_uni * K encoder calls
        uni[K] = float(time.perf_counter() - t0)      # linear in K

    return {"bi": bi, "uni": uni, "n_uni_texts": n_uni, "n_bi_texts": len(texts)}


# ---------------------------------------------------------------------------
# CPU self-test: synthetic embeddings only, NO model, NO data
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    """Look-up embedder for the self-test: maps known strings -> preset vectors.

    Lets us exercise build_policy_bank's multi-prototype averaging on SYNTHETIC
    vectors without loading any real model.
    """

    def __init__(self, table: dict, dim: int):
        self._table = table
        self._dim = dim

    def encode(self, texts, kind):
        rows = [self._table[str(t)] for t in texts]
        return _l2_normalize(_truncate(np.asarray(rows, dtype=np.float32), self._dim))


def _self_test() -> None:
    rng = np.random.default_rng(C.SEED)
    dim = 64
    P = 8

    # --- (1) BiEncoder cosine recovers nearest-policy labels ---
    # P well-separated unit policy directions; each text = its policy + small noise.
    policy_vecs = _l2_normalize(rng.standard_normal((P, dim)).astype(np.float32))
    per = 20
    Xc, Y = [], []
    for c in range(P):
        for _ in range(per):
            Xc.append(policy_vecs[c] + 0.25 * rng.standard_normal(dim))
            onehot = np.zeros(P, dtype=np.float32)
            onehot[c] = 1.0
            Y.append(onehot)
    Xc = _l2_normalize(np.asarray(Xc, dtype=np.float32))
    Y = np.asarray(Y, dtype=np.float32)

    bi = BiEncoderGuard()
    S = bi.scores(Xc, policy_vecs, list(range(P)))
    mm = macro_micro(Y, S)
    print("bi_encoder synthetic: macro_ap=%.4f micro_ap=%.4f (expect near 1.0)"
          % (mm["macro_ap"], mm["micro_ap"]))
    assert mm["macro_ap"] > 0.9, "BiEncoder failed to recover nearest-policy labels"

    # TrainedHead should also do well on these SEEN cols; abstains (NaN) on unseen.
    th = TrainedHeadGuard().fit(Xc, Y, seen_cols=list(range(P - 2)), policies=None)
    Sth = th.scores(Xc, policy_vecs, list(range(P)))
    assert np.all(np.isnan(Sth[:, P - 1])), "TrainedHead must abstain on unseen col"
    assert not np.all(np.isnan(Sth[:, 0])), "TrainedHead must score a seen col"
    print("trained_head synthetic: abstains on 2 held-out cols (NaN) as designed")

    # --- (2) multi-prototype averaging >= single-prototype (denoising) ---
    # Build a fake embedder: description + paraphrases are noisy views of a clean
    # policy direction; averaging them should sit closer to the clean center.
    clean = _l2_normalize(rng.standard_normal((P, dim)).astype(np.float32))
    n_para = C.POLICY_PARAPHRASES
    table = {}
    policies = []
    noise = 0.6
    for c in range(P):
        desc_key = "desc_%d" % c
        table[desc_key] = clean[c] + noise * rng.standard_normal(dim)
        paraphrases = []
        for p in range(n_para - 1):
            key = "para_%d_%d" % (c, p)
            table[key] = clean[c] + noise * rng.standard_normal(dim)
            paraphrases.append(key)
        policies.append({"description": desc_key, "paraphrases": paraphrases})
    fake = _FakeEmbedder(table, dim)

    bank_single = build_policy_bank(policies, fake, n_proto=1)
    bank_multi = build_policy_bank(policies, fake, n_proto=n_para)

    # Test texts sit near the CLEAN centers; the better bank scores them higher.
    Xt, Yt = [], []
    for c in range(P):
        for _ in range(per):
            Xt.append(clean[c] + 0.25 * rng.standard_normal(dim))
            oh = np.zeros(P, dtype=np.float32)
            oh[c] = 1.0
            Yt.append(oh)
    Xt = _l2_normalize(np.asarray(Xt, dtype=np.float32))
    Yt = np.asarray(Yt, dtype=np.float32)

    ap_single = macro_micro(Yt, bi.scores(Xt, bank_single, list(range(P))))["macro_ap"]
    ap_multi = macro_micro(Yt, bi.scores(Xt, bank_multi, list(range(P))))["macro_ap"]
    print("multi-proto ablation: single macro_ap=%.4f  multi(%d) macro_ap=%.4f"
          % (ap_single, n_para, ap_multi))
    assert ap_multi >= ap_single - 1e-6, "multi-prototype should not hurt vs single"

    print("encoders.py self-test OK: bi recovers labels; multi-proto >= single;"
          " trained_head abstains on held-out columns.")


if __name__ == "__main__":
    _self_test()
