"""hardneg.py -- the 2026 HARD-NEGATIVE contrastive-augmentation module.

WHY THIS FILE EXISTS
--------------------
A bi-encoder guardrail lives or dies by the QUALITY of its hard negatives. A
"hard negative" is a BENIGN piece of content that *looks* adversarial -- it sits
close to a policy vector in embedding space even though the policy does not apply
to it. If the model never sees these look-alikes it will over-fire (high false-
positive rate) on innocent, topically-adjacent text: a chemistry lecture flagged
as "drug_weapon", a news report flagged as "terrorism". Quality-controlling the
negative set is therefore a SAFETY problem, not a bookkeeping one -- a guardrail
that cries wolf gets turned off, and a guardrail trained on FALSE negatives
(near-miss jailbreaks mislabeled benign) is actively dangerous.

This module teaches the four-stage 2026 recipe, each stage from one paper:

  1. dense mining (ANCE-style)           -> mine_dense_hard_negatives()
       Let the content tower retrieve its OWN look-alikes: the benign texts most
       cosine-similar to each policy vector. These are the hardest negatives
       *because the model itself confuses them* -- far more useful than random.

  2. ECIsem diagnostic (arXiv:2603.20990) -> eci_score()
       A TRAINING-FREE score of a mined negative set in the frozen geometry.
       Measure informativeness BEFORE you spend a training run.

  3. CausalNeg counterfactuals (arXiv:2606.01304) -> causal_counterfactuals()
       Manufacture *controlled* negatives by perturbing exactly ONE requirement
       of a violating text with TEMPLATED string ops (no free-form LLM) -- so the
       negative stays fluent and on-topic but no longer violates. Avoiding the
       generative-discriminative gap = do not let a generator invent negatives a
       discriminator can trivially separate on surface artifacts.

  4. ARHN false-negative filter (arXiv:2604.11092) -> arhn_false_negative_filter()
       Before accepting a candidate negative, CHECK it does not actually support
       the policy. A "benign" example that still violates is a FALSE negative and
       poisons the contrastive signal (and safety).

Then a small CONTRASTIVE ADAPTER (InfoNCE, adaptive hardness weighting) sharpens
the frozen geometry on the validated negatives -> ContrastiveAdapter.

Everything here operates on PRECOMPUTED vectors (the caller passes content
embeddings Xc [n,dim] and the policy bank [P,dim]); this file loads NO embedding
model. Only causal_counterfactuals() touches raw strings, and only with templates.

ASCII-only stdout (Windows cp1252): we write "cos", ">=", "||", never unicode.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from steering_tutorials.biencoder_guard import config as C


# ---------------------------------------------------------------------------
# small geometry helpers (frozen vectors are assumed ~unit; we renormalize to be
# safe so every "cos" below is a genuine cosine in [-1, 1]).
# ---------------------------------------------------------------------------
def _l2norm(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalize so a dot product equals a cosine similarity."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)


# ===========================================================================
# 1. DENSE HARD-NEGATIVE MINING  (ANCE-style)
# ===========================================================================
def mine_dense_hard_negatives(Xc, policy_bank, is_harmful, cols, k=C.HARDNEG_PER_POLICY):
    """ANCE idea in one line: the encoder mines its OWN hardest negatives.

    ANCE (Approximate nearest neighbour Negative Contrastive Estimation) observed
    that random negatives are almost always easy -- the gradient they give is tiny
    because the model already separates them. The informative negatives are the
    ones the CURRENT encoder ranks close to the query. So for each policy column
    we take the policy vector as the "query" and retrieve the BENIGN texts
    (is_harmful == 0) with the HIGHEST cosine to it: the true look-alikes the guard
    is at risk of mis-flagging.

    Returns {col: [row_idx, ...]} -- the top-k benign look-alike indices per policy,
    sorted hardest-first (highest cosine). These feed the ECIsem diagnostic and the
    contrastive adapter below.
    """
    Xc = _l2norm(Xc)
    P = _l2norm(policy_bank)
    is_harmful = np.asarray(is_harmful).astype(bool)
    benign_idx = np.where(~is_harmful)[0]                 # candidate negatives only
    out: dict[int, list[int]] = {}
    for col in cols:
        if benign_idx.size == 0:
            out[int(col)] = []
            continue
        # cosine of every benign text to this one policy vector.
        sims = Xc[benign_idx] @ P[col]                    # [n_benign]
        kk = int(min(k, benign_idx.size))
        # argpartition for the top-kk, then sort those descending by cosine.
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[np.argsort(-sims[top])]
        out[int(col)] = [int(benign_idx[t]) for t in top]
    return out


# ===========================================================================
# 2. ECIsem  TRAINING-FREE DIAGNOSTIC  (arXiv:2603.20990)
# ===========================================================================
def eci_score(Xc, policy_bank, pos_idx, neg_idx, col) -> dict:
    """ECIsem: "is this negative set worth training on?", answered geometrically.

    "Semantic Residual Effective Contrastive Information" scores a hard-negative
    set for ONE policy purely in the frozen embedding geometry -- no training, no
    labels beyond pos/neg membership. We report four interpretable terms (each
    commented with its formula) and a combined `eci` where HIGHER = a more
    informative negative set. Let p = unit policy vector, c = unit positive
    centroid, and n_i the unit negative vectors.

      target_consistency
        = mean_i cos(pos_i, p)  -  mean_j cos(neg_j, p)
        The ranking MARGIN. Positives should still out-score negatives against the
        policy; if this goes <= 0 the "negatives" are behaving like positives
        (a labeling problem -- see the ARHN false-negative filter).

      locality
        = mean_j cos(neg_j, p)
        How CLOSE the negatives sit to the policy. Higher = harder = more useful
        gradient. Random far negatives score ~0 here; genuine look-alikes score high.

      lexical_residual  (a PENALTY, higher = worse)
        Remove the policy component from each negative: r_j = n_j - (n_j.p) p, then
        measure how much of that residual aligns with the positive centroid's own
        residual  c_res = c - (c.p) p:
            lexical_residual = mean_j | cos(r_j, c_res) |
        This proxies "the negative is just copying the positives' surface features
        off the policy axis" -- trivial overlap the model can exploit without
        learning the policy. (In the paper this is a lexical-overlap term; on
        precomputed vectors we use this geometric residual proxy.)

      diversity
        = mean over pairs (1 - cos(r_i, r_j)) of the residual negative directions.
        A set of near-duplicate negatives teaches one thing many times; a diverse
        set covers more of the benign look-alike manifold. Higher = better.

      eci  (combined; higher = better)
        = locality + 0.5*target_consistency + 0.3*diversity - 0.5*lexical_residual
        Dominated by LOCALITY (hardness) but rewards a correctly-ranked, diverse,
        non-trivial set and penalizes surface-copy overlap.
    """
    Xc = _l2norm(Xc)
    P = _l2norm(policy_bank)
    p = P[col]
    pos = Xc[np.asarray(pos_idx, dtype=int)]
    neg = Xc[np.asarray(neg_idx, dtype=int)]

    cos_pos = pos @ p                                     # cos(pos_i, p)
    cos_neg = neg @ p                                     # cos(neg_j, p)
    target_consistency = float(cos_pos.mean() - cos_neg.mean())
    locality = float(cos_neg.mean())

    # residual = component orthogonal to the policy axis (the "semantic residual").
    def _residual(V):
        proj = (V @ p)[:, None] * p[None, :]
        return _l2norm(V - proj)

    r_neg = _residual(neg)
    c = _l2norm(pos.mean(axis=0))[0]                      # unit positive centroid
    c_res = c - (c @ p) * p
    c_res_n = c_res / max(np.linalg.norm(c_res), 1e-8)
    lexical_residual = float(np.mean(np.abs(r_neg @ c_res_n)))

    # diversity = mean pairwise angular spread of residual negative directions.
    if r_neg.shape[0] >= 2:
        G = r_neg @ r_neg.T                               # pairwise cos
        iu = np.triu_indices(G.shape[0], k=1)
        diversity = float(np.mean(1.0 - G[iu]))
    else:
        diversity = 0.0

    eci = (locality
           + 0.5 * target_consistency
           + 0.3 * diversity
           - 0.5 * lexical_residual)
    return {"target_consistency": target_consistency,
            "locality": locality,
            "lexical_residual": lexical_residual,
            "diversity": diversity,
            "eci": float(eci)}


# ===========================================================================
# 3. CausalNeg  CONTROLLED COUNTERFACTUALS  (arXiv:2606.01304)
# ===========================================================================
# The CausalNeg lesson: a good hard negative should differ from the positive on
# EXACTLY ONE requirement of the policy, so the model learns the DECISION BOUNDARY
# rather than a spurious surface artifact. Free-form LLM generation creates the
# "generative-discriminative gap": the generator leaves fingerprints (phrasing,
# length, boilerplate) that a discriminator learns instead of the real concept. So
# we perturb with deterministic TEMPLATES -- topic preserved, one requirement flipped.

# entity swaps: turn a harmful actor/object into a benign one (topic-preserving).
_ENTITY_SWAPS = [
    ("bomb", "cake"), ("explosive", "baking"), ("weapon", "tool"),
    ("poison", "spice"), ("hack", "log into"), ("malware", "software"),
    ("kill", "photograph"), ("attack", "visit"), ("steal", "borrow"),
    ("drug", "vitamin"), ("gun", "camera"),
]


def _apply_entity_swap(text: str):
    """Flip the first harmful entity we find to a benign one (one requirement)."""
    low = text.lower()
    for bad, good in _ENTITY_SWAPS:
        i = low.find(bad)
        if i >= 0:
            return text[:i] + good + text[i + len(bad):]
    return None


def _apply_negation(text: str):
    """Insert a refusal/avoidance frame: the topic stays, the INTENT flips benign."""
    return "Explain why one should never " + text[0].lower() + text[1:]


def _apply_softening(text: str):
    """Recast as a safety/awareness question -- on-topic, no longer an instruction."""
    return ("For a public-safety awareness class, describe the warning signs of "
            + text[0].lower() + text[1:])


_CAUSAL_OPS = [("entity_swap", _apply_entity_swap),
               ("negation", _apply_negation),
               ("softening", _apply_softening)]


def causal_counterfactuals(texts, policies, cols, n_per=3) -> list[dict]:
    """Templated single-requirement counterfactual negatives (CausalNeg).

    Take each VIOLATING text and apply up to `n_per` distinct templated ops, each
    of which violates exactly one policy requirement so the result stays fluent and
    on-topic but no longer applies. Returns [{"text", "from_col", "op"}]. Purely
    string templates -- no model -- which is the whole point: deterministic,
    auditable, and free of the generative-discriminative gap. `policies`/`cols`
    are carried through for provenance (which policy the counterfactual was mined
    from); a real pipeline would decompose the policy into requirements first, but
    the teaching version keeps the requirement set implicit in the op templates.
    """
    cols = list(cols)
    out: list[dict] = []
    for t_i, text in enumerate(texts):
        # attribute this text to a source policy column (round-robin for the demo).
        from_col = int(cols[t_i % len(cols)]) if cols else -1
        made = 0
        for op_name, op_fn in _CAUSAL_OPS:
            if made >= n_per:
                break
            new = op_fn(text)
            if new and new != text:
                out.append({"text": new, "from_col": from_col, "op": op_name})
                made += 1
    return out


# ===========================================================================
# 4. ARHN  FALSE-NEGATIVE FILTER  (arXiv:2604.11092)
# ===========================================================================
def _lexical_policy_support(text: str, policy: dict) -> bool:
    """Cheap stand-in for ARHN's LLM answerability check.

    ARHN asks an open-source LLM "does this passage actually support/answer the
    query?" and RELABELS the ones that do (they are false negatives). We cannot run
    an LLM here, so we approximate support by keyword overlap: if the text contains
    salient policy tokens (from the policy name + description), we conservatively
    treat it as SUPPORTING the policy. This is deliberately documented as a proxy --
    it errs toward caution, which for a SAFETY filter is the right direction (drop a
    borderline example rather than train on a possible violation labeled benign).
    """
    words = set()
    for field in ("name", "description"):
        for w in str(policy.get(field, "")).lower().replace("_", " ").split():
            if len(w) >= 4:                               # skip stopword-ish shorties
                words.add(w)
    low = text.lower()
    hits = sum(1 for w in words if w in low)
    return hits >= 2                                      # >=2 policy tokens -> supports


def arhn_false_negative_filter(neg_texts, policy, support_fn=None) -> list[bool]:
    """Keep a candidate negative ONLY if it does not support the policy.

    Returns a keep-mask (True = safe to use as a negative, False = drop as a
    probable FALSE negative). A false negative here is the dangerous case: a
    near-miss jailbreak that still violates the policy but got labeled benign --
    training on it teaches the guard to PASS the very thing it should catch. So we
    run each candidate through a policy-support check and drop the ones that pass.
    Default check = the documented lexical heuristic above; a caller may inject a
    stronger `support_fn(text, policy) -> bool` (e.g. an LLM answerability judge).
    """
    fn = support_fn or _lexical_policy_support
    return [not bool(fn(t, policy)) for t in neg_texts]


# ===========================================================================
# 5. CONTRASTIVE ADAPTER  (InfoNCE + adaptive hardness weighting)
# ===========================================================================
class ContrastiveAdapter(nn.Module):
    """A small projection that SHARPENS the frozen geometry on validated negatives.

    Architecture (shared two-tower: the SAME net projects both content and policy
    vectors, preserving the cosine comparability the bi-encoder relies on):
        Linear(dim -> ADAPTER_DIM) -> tanh -> Linear(ADAPTER_DIM -> ADAPTER_DIM)
    The backbone stays frozen; we only learn this thin projection on top of cached
    vectors -- cheap, CPU-friendly, and it cannot damage the base model.

    TRAINING OBJECTIVE -- InfoNCE with adaptive hardness weighting.
    For a policy p (anchor) with a positive content vector x+ and mined hard
    negatives {x-_j}, cosine similarities s = cos(proj(p), proj(x)) at temperature
    T = C.CONTRASTIVE_TEMP give:

        L = -log(  exp(s+/T)  /  ( exp(s+/T) + sum_j a_j * exp(s-_j/T) )  )

    where the ADAPTIVE HARDNESS WEIGHT
        a_j = 1 + beta * relu(s-_j - s+).detach()
    up-weights exactly the negatives that VIOLATE the margin (score above the
    positive) -- the hardest, most informative look-alikes get the strongest push,
    while easy negatives (a_j ~ 1) are left alone. `.detach()` keeps the weight a
    scalar schedule, not a second gradient path.

    We hold out a validation slice of positives+negatives and BEST-CHECKPOINT on
    the lowest val InfoNCE loss (the stiff contrastive loss oscillates, so the last
    step is rarely the best -- a hard-won lesson from this course).
    """

    def __init__(self, dim: int, adapter_dim: int = C.ADAPTER_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.Tanh(),
            nn.Linear(adapter_dim, adapter_dim),
        )
        self.temp = float(C.CONTRASTIVE_TEMP)
        self.beta = 4.0                                   # hardness-weight strength

    # -- forward / transform -------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)   # renormalize -> cosine space

    @torch.no_grad()
    def transform(self, X) -> np.ndarray:
        """Project numpy vectors X [n,dim] -> renormalized numpy [n,ADAPTER_DIM]."""
        self.eval()
        t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return self.forward(t).cpu().numpy()

    # -- loss for one (policy, positives, negatives) group -------------------
    def _group_loss(self, pol_vec, pos_vecs, neg_vecs) -> torch.Tensor:
        zp = self.forward(pol_vec[None, :])[0]            # projected policy anchor
        zpos = self.forward(pos_vecs)                     # [n_pos, d]
        zneg = self.forward(neg_vecs)                     # [n_neg, d]
        s_pos = (zpos @ zp) / self.temp                   # [n_pos]
        s_neg = (zneg @ zp) / self.temp                   # [n_neg]
        # adaptive hardness weight on each negative (detached scalar schedule).
        losses = []
        for sp in s_pos:
            margin = (s_neg - sp).clamp_min(0.0).detach()
            a = 1.0 + self.beta * margin                  # a_j >= 1, hardest weighted most
            denom = torch.exp(sp) + (a * torch.exp(s_neg)).sum()
            losses.append(-(sp - torch.log(denom + 1e-8)))
        return torch.stack(losses).mean()

    # -- fit -----------------------------------------------------------------
    def fit(self, Xc, policy_bank, Y, cols, hardnegs, epochs: int = C.ADAPTER_EPOCHS,
            val_frac: float = 0.3, verbose: bool = False):
        """Train on frozen vectors. hardnegs = {col: [benign_idx,...]} from mining.

        For each policy col we form (anchor=policy_bank[col], positives=rows with
        Y[:,col]==1, negatives=hardnegs[col]) and split positives+negatives into
        train/val for best-checkpointing. Returns self.
        """
        torch.manual_seed(C.SEED)
        rng = np.random.default_rng(C.SEED)
        Xc = _l2norm(Xc)
        P = _l2norm(policy_bank)
        Y = np.asarray(Y)
        Xc_t = torch.as_tensor(Xc)
        P_t = torch.as_tensor(P)

        # build per-col train/val index groups (skip cols lacking pos or neg).
        train_groups, val_groups = [], []
        for col in cols:
            pos = np.where(Y[:, col] == 1)[0]
            neg = np.asarray(hardnegs.get(int(col), []), dtype=int)
            if pos.size < 2 or neg.size < 2:
                continue
            pos = pos.copy(); neg = neg.copy()
            rng.shuffle(pos); rng.shuffle(neg)
            pv = max(1, int(val_frac * pos.size)); nv = max(1, int(val_frac * neg.size))
            train_groups.append((int(col), pos[pv:], neg[nv:]))
            val_groups.append((int(col), pos[:pv], neg[:nv]))
        if not train_groups:
            return self                                   # nothing trainable

        opt = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        best_val = float("inf")
        best_state = {k: v.clone() for k, v in self.state_dict().items()}

        for ep in range(int(epochs)):
            self.train()
            opt.zero_grad()
            loss = torch.stack([
                self._group_loss(P_t[c], Xc_t[pos], Xc_t[neg])
                for (c, pos, neg) in train_groups
            ]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)   # stiff loss -> clip
            opt.step()

            # validation InfoNCE -> best-checkpoint on the lowest val loss.
            self.eval()
            with torch.no_grad():
                vloss = torch.stack([
                    self._group_loss(P_t[c], Xc_t[pos], Xc_t[neg])
                    for (c, pos, neg) in val_groups
                ]).mean().item()
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            if verbose:
                print("epoch %02d  train=%.4f  val=%.4f" % (ep, loss.item(), vloss))

        self.load_state_dict(best_state)                  # restore best, not last
        return self


# ===========================================================================
# METRIC:  false-positive rate at a fixed recall
# ===========================================================================
def fpr_at_recall(y_true, score, recall=0.90) -> float:
    """FPR at the operating point where recall (TPR) first reaches `recall`.

    A guardrail is deployed at a chosen recall (catch >= 90% of violations); the
    question that matters is then "how many BENIGN inputs do I wrongly block?".
    Lower FPR@recall = fewer false alarms at the same safety coverage. We walk the
    ROC curve and return the smallest FPR whose TPR >= recall.
    """
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return float("nan")                               # undefined without both classes
    fpr, tpr, _ = roc_curve(y_true, score)
    mask = tpr >= recall
    return float(fpr[mask].min()) if mask.any() else 1.0


# ===========================================================================
# CPU-ONLY SYNTHETIC SELF-TEST  (no model, no data, dim=48)
# ===========================================================================
if __name__ == "__main__":
    print("== hardneg.py self-test (synthetic, CPU, dim=48) ==")
    rng = np.random.default_rng(C.SEED)
    dim = 48

    # one policy vector; positives cluster tightly around it (high cos); a RING of
    # hard-negative benigns sits at moderate cos (look-alikes); random FAR negatives
    # sit near cos ~0.  is_harmful marks only the positives.
    p = _l2norm(rng.standard_normal(dim))[0]

    def _mix(anchor, w, n):
        V = w * anchor[None, :] + (1.0 - w) * rng.standard_normal((n, dim))
        return _l2norm(V)

    n_pos, n_ring, n_far = 60, 40, 40
    pos = _mix(p, 0.80, n_pos)                            # cos ~0.8   (positives)
    ring = _mix(p, 0.50, n_ring)                          # cos ~0.5   (hard negatives)
    far = _l2norm(rng.standard_normal((n_far, dim)))      # cos ~0     (easy negatives)

    Xc = np.vstack([pos, ring, far]).astype(np.float32)
    policy_bank = p[None, :].astype(np.float32)
    is_harmful = np.array([1] * n_pos + [0] * (n_ring + n_far))
    Y = is_harmful.reshape(-1, 1)
    pos_idx = np.arange(0, n_pos)
    ring_idx = np.arange(n_pos, n_pos + n_ring)
    far_idx = np.arange(n_pos + n_ring, n_pos + n_ring + n_far)

    # (a) dense mining should recover the RING, not the FAR negatives.
    mined = mine_dense_hard_negatives(Xc, policy_bank, is_harmful, cols=[0], k=20)
    mined0 = set(mined[0])
    in_ring = len(mined0 & set(ring_idx.tolist()))
    print("(a) mining: %d/20 mined negatives are ring look-alikes (want most)" % in_ring)
    assert in_ring >= 15, "dense mining failed to recover the hard-negative ring"

    # (b) ECIsem of the hard ring should exceed that of random far negatives.
    eci_ring = eci_score(Xc, policy_bank, pos_idx, ring_idx, 0)
    eci_far = eci_score(Xc, policy_bank, pos_idx, far_idx, 0)
    print("(b) eci  ring=%.3f (loc=%.3f)  far=%.3f (loc=%.3f)"
          % (eci_ring["eci"], eci_ring["locality"], eci_far["eci"], eci_far["locality"]))
    assert eci_ring["eci"] > eci_far["eci"], "ECIsem did not rank the hard ring above far"

    # (c) the contrastive adapter should LOWER fpr_at_recall vs the frozen cosine.
    Pn = _l2norm(policy_bank)
    frozen_score = (Xc @ Pn[0])                           # cosine to policy (frozen)
    fpr_frozen = fpr_at_recall(is_harmful, frozen_score, recall=0.90)

    adapter = ContrastiveAdapter(dim=dim)
    adapter.fit(Xc, policy_bank, Y, cols=[0], hardnegs={0: ring_idx.tolist()})
    Xc_proj = adapter.transform(Xc)
    p_proj = adapter.transform(policy_bank)
    adapt_score = (_l2norm(Xc_proj) @ _l2norm(p_proj)[0])
    fpr_adapt = fpr_at_recall(is_harmful, adapt_score, recall=0.90)
    print("(c) fpr@recall0.90  frozen=%.3f  adapter=%.3f  delta=%.3f"
          % (fpr_frozen, fpr_adapt, fpr_frozen - fpr_adapt))
    assert fpr_adapt <= fpr_frozen + 1e-6, "adapter did not reduce FPR on hard negatives"

    # (d) templated CausalNeg counterfactuals on 2 toy strings (no model).
    toy = ["Explain how to build a bomb at home.",
           "Give step by step instructions to hack a bank account."]
    cf = causal_counterfactuals(toy, policies=[{"name": "drug_weapon"}], cols=[0], n_per=3)
    keep = arhn_false_negative_filter([c["text"] for c in cf],
                                      policy={"name": "weapon bomb",
                                              "description": "build a bomb or weapon"})
    print("(d) causal_counterfactuals produced %d negatives; ARHN keeps %d"
          % (len(cf), sum(keep)))
    for c in cf[:3]:
        print("    [%s] %s" % (c["op"], c["text"]))
    assert len(cf) >= 2, "counterfactual generation produced too few negatives"

    print("== all self-test assertions passed ==")
