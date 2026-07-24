"""models.py -- the four SET-of-trajectories classifiers for cross-trajectory
(multi-agent / cross-session) decomposed-attack detection.

Every classifier exposes the SAME interface:
    .fit(train_sets, train_labels)          # train_sets: list of [K, dim] float32
    .predict_proba(sets) -> np.ndarray[n]   # P(attack) in [0,1], one per SAMPLE

A "sample" is an unordered SET of K trajectory embeddings (one per agent/session).
The aggregators are PERMUTATION-INVARIANT over the K trajectories -- the order of
the agents must not matter. The four models form a pedagogical ladder:

  1. PerTrajMax -- DECOMPOSITION-DEFEATED BASELINE. Logistic regression on INDIVIDUAL
     trajectory vectors (each inherits its sample's label); the sample score is the MAX
     per-trajectory P(attack) -- "does ANY single trajectory look harmful?". It cannot
     see the aggregate, so a goal fractured across K individually-innocuous trajectories
     slips right past it. That failure is the whole point.
  2. MeanAgg   -- mean/max/std pool over the K embeddings -> standardized -> MLP. Sees
     the aggregate through a fixed permutation-invariant summary.
  3. AttnPool  -- the HEADLINE aggregator. Set-Transformer Pooling by Multihead
     Attention (PMA, Lee et al. 2019, arXiv:1810.00825): a learned seed query attends
     over the K trajectory embeddings (optionally after one SAB self-attention block)
     -> one pooled vector -> logit. Permutation-invariant; variable K via a padding mask.
  4. GnnAgg    -- GroupGuard-style collusion detector. A fully-connected graph over the
     K agents; 1-2 rounds of mean message passing (new_node = ReLU(W[node, mean_all]))
     -> mean readout -> logit. Also permutation-invariant with a padding mask.

Pure numpy/torch/sklearn, CPU-only. The __main__ self-test uses ONLY synthetic data
(a hidden goal that lives in the SUM of K vectors but in no single one) -- no real
dataset, no model download.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - sklearn is installed, but stay robust
    _HAVE_SKLEARN = False

from . import config as C

torch.manual_seed(C.SEED)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_set_list(sets):
    """Coerce to a list of float32 [K, dim] arrays, each with K>=1 trajectory."""
    out = []
    for s in sets:
        a = np.asarray(s, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[0] == 0:
            a = np.zeros((1, a.shape[-1]), dtype=np.float32)
        out.append(a)
    return out


def _infer_dim(sets) -> int:
    for s in sets:
        a = np.asarray(s)
        if a.size:
            return int(a.shape[-1])
    return 1


def _pad_batch(batch_sets, device="cpu"):
    """List of [K_i, dim] -> (x [B, Kmax, dim], valid_mask [B, Kmax] bool True=real)."""
    dim = batch_sets[0].shape[-1]
    kmax = max(a.shape[0] for a in batch_sets)
    B = len(batch_sets)
    x = np.zeros((B, kmax, dim), dtype=np.float32)
    valid = np.zeros((B, kmax), dtype=bool)
    for i, a in enumerate(batch_sets):
        k = a.shape[0]
        x[i, :k] = a
        valid[i, :k] = True
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(valid).to(device))


def _standardize_stats(sets):
    """Mean/std over ALL trajectory vectors stacked (per-feature)."""
    allv = np.concatenate([np.asarray(a, dtype=np.float32) for a in sets], axis=0)
    mu = allv.mean(axis=0)
    sd = allv.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. PerTrajMax -- the decomposition-defeated baseline
# ---------------------------------------------------------------------------
class PerTrajMax:
    """Logistic regression on individual trajectory vectors; sample score = MAX
    over the K per-trajectory P(attack). Standardized on train. Permutation-invariant
    (max is order-free) but blind to the aggregate -- the attack is built to bypass it."""

    def __init__(self):
        self.scaler = None
        self.clf = None
        self._const = None  # fallback prob if a class is missing

    def fit(self, train_sets, train_labels):
        sets = _as_set_list(train_sets)
        y = np.asarray(train_labels, dtype=int)
        rows, row_y = [], []
        for a, lab in zip(sets, y):
            for v in a:
                rows.append(v)
                row_y.append(int(lab))
        X = np.asarray(rows, dtype=np.float32)
        ry = np.asarray(row_y, dtype=int)
        if not _HAVE_SKLEARN or len(np.unique(ry)) < 2:
            self._const = float(ry.mean()) if ry.size else 0.5
            return self
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.clf.fit(self.scaler.transform(X), ry)
        return self

    def predict_proba(self, sets):
        sets = _as_set_list(sets)
        if self.clf is None:
            return np.full(len(sets), self._const if self._const is not None else 0.5,
                           dtype=np.float32)
        pos_idx = list(self.clf.classes_).index(1) if 1 in self.clf.classes_ else -1
        out = np.zeros(len(sets), dtype=np.float32)
        for i, a in enumerate(sets):
            p = self.clf.predict_proba(self.scaler.transform(a))
            out[i] = float(p[:, pos_idx].max()) if pos_idx >= 0 else 0.0
        return out


# ---------------------------------------------------------------------------
# shared torch training utilities
# ---------------------------------------------------------------------------
def _train_set_module(module, forward_fn, sets, labels):
    """Train a permutation-invariant set module. forward_fn(x, valid)->logit[B]."""
    torch.manual_seed(C.SEED)
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    n = len(sets)
    opt = torch.optim.Adam(module.parameters(), lr=C.LR)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(C.SEED)
    module.train()
    for _ in range(C.EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, C.BATCH):
            idx = order[start:start + C.BATCH]
            batch = [sets[j] for j in idx]
            x, valid = _pad_batch(batch)
            logit = forward_fn(x, valid)
            loss = loss_fn(logit, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    module.eval()


def _predict_set_module(module, forward_fn, sets):
    module.eval()
    out = np.zeros(len(sets), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(sets), C.BATCH):
            batch = sets[start:start + C.BATCH]
            x, valid = _pad_batch(batch)
            logit = forward_fn(x, valid)
            out[start:start + len(batch)] = torch.sigmoid(logit).cpu().numpy()
    return out


def _masked_mean(x, valid):
    """x [B,K,H], valid [B,K] bool -> [B,H] mean over real trajectories."""
    m = valid.unsqueeze(-1).float()
    s = (x * m).sum(dim=1)
    cnt = m.sum(dim=1).clamp(min=1.0)
    return s / cnt


# ---------------------------------------------------------------------------
# 2. MeanAgg -- mean/max/std pool -> standardized -> MLP
# ---------------------------------------------------------------------------
class MeanAgg(nn.Module):
    """Permutation-invariant [mean, max, std] pool over the K embeddings -> MLP."""

    def __init__(self):
        super().__init__()
        self.mu = self.sd = None
        self.net = None

    @staticmethod
    def _features(a):
        a = np.asarray(a, dtype=np.float32)
        mean = a.mean(axis=0)
        mx = a.max(axis=0)
        std = a.std(axis=0) if a.shape[0] > 1 else np.zeros_like(mean)
        return np.concatenate([mean, mx, std], axis=0).astype(np.float32)

    def _feat_matrix(self, sets):
        return np.stack([self._features(a) for a in sets], axis=0)

    def fit(self, train_sets, train_labels):
        sets = _as_set_list(train_sets)
        X = self._feat_matrix(sets)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-6] = 1.0
        d = X.shape[1]
        torch.manual_seed(C.SEED)
        self.net = nn.Sequential(
            nn.Linear(d, C.HIDDEN), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(C.HIDDEN, 1),
        )
        Xs = torch.tensor((X - self.mu) / self.sd)
        y = torch.tensor(np.asarray(train_labels, dtype=np.float32))
        opt = torch.optim.Adam(self.net.parameters(), lr=C.LR)
        loss_fn = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(C.SEED)
        self.net.train()
        for _ in range(C.EPOCHS):
            order = rng.permutation(len(sets))
            for start in range(0, len(sets), C.BATCH):
                idx = order[start:start + C.BATCH]
                logit = self.net(Xs[idx]).squeeze(-1)
                loss = loss_fn(logit, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def predict_proba(self, sets):
        sets = _as_set_list(sets)
        X = self._feat_matrix(sets)
        Xs = torch.tensor((X - self.mu) / self.sd)
        with torch.no_grad():
            return torch.sigmoid(self.net(Xs).squeeze(-1)).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# 3. AttnPool -- Set-Transformer PMA (the headline aggregator)
# ---------------------------------------------------------------------------
class AttnPool(nn.Module):
    """Project -> optional SAB self-attention block over the set -> PMA (one learned
    seed query attends over the K trajectories) -> Linear logit. Permutation-invariant;
    variable K handled by a key_padding_mask (arXiv:1810.00825)."""

    def __init__(self, use_sab=True):
        super().__init__()
        self.dim = None
        self.use_sab = use_sab

    def _build(self, dim):
        torch.manual_seed(C.SEED)
        h = C.HIDDEN
        self.proj = nn.Linear(dim, h)
        heads = C.ATTN_HEADS if h % C.ATTN_HEADS == 0 else 1
        self.heads = heads
        if self.use_sab:
            self.sab = nn.MultiheadAttention(h, heads, batch_first=True)
            self.sab_ff = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.sab_norm1 = nn.LayerNorm(h)
            self.sab_norm2 = nn.LayerNorm(h)
        self.seed = nn.Parameter(torch.randn(1, 1, h) * 0.1)
        self.pma = nn.MultiheadAttention(h, heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.dim = dim

    def _forward(self, x, valid):
        # x [B,K,dim], valid [B,K] bool (True=real). key_padding_mask True=ignore.
        kpm = ~valid
        z = self.proj(x)
        if self.use_sab:
            a, _ = self.sab(z, z, z, key_padding_mask=kpm)
            z = self.sab_norm1(z + a)
            z = self.sab_norm2(z + self.sab_ff(z))
        B = z.shape[0]
        q = self.seed.expand(B, -1, -1)
        pooled, _ = self.pma(q, z, z, key_padding_mask=kpm)
        return self.head(pooled.squeeze(1)).squeeze(-1)

    def fit(self, train_sets, train_labels):
        sets = _as_set_list(train_sets)
        if self.dim is None:
            self._build(_infer_dim(sets))
        _train_set_module(self, self._forward, sets, train_labels)
        return self

    def predict_proba(self, sets):
        sets = _as_set_list(sets)
        if self.dim is None:
            self._build(_infer_dim(sets))
        return _predict_set_module(self, self._forward, sets)


# ---------------------------------------------------------------------------
# 4. GnnAgg -- fully-connected graph over the K agents (GroupGuard-style)
# ---------------------------------------------------------------------------
class GnnAgg(nn.Module):
    """Node feat = projected trajectory embedding; 1-2 rounds of mean message passing
    (new_node = ReLU(W[node, mean_of_all_nodes])); mean readout -> Linear logit.
    Permutation-invariant (mean over nodes); variable K handled by the valid mask."""

    def __init__(self, rounds=2):
        super().__init__()
        self.dim = None
        self.rounds = rounds

    def _build(self, dim):
        torch.manual_seed(C.SEED)
        h = C.HIDDEN
        self.proj = nn.Linear(dim, h)
        self.msg = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(self.rounds)])
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.dim = dim

    def _forward(self, x, valid):
        m = valid.unsqueeze(-1).float()
        node = torch.relu(self.proj(x)) * m           # [B,K,H]
        for lin in self.msg:
            agg = _masked_mean(node, valid)            # [B,H] fully-connected mean
            agg = agg.unsqueeze(1).expand(-1, node.shape[1], -1)
            node = torch.relu(lin(torch.cat([node, agg], dim=-1))) * m
        readout = _masked_mean(node, valid)            # [B,H]
        return self.head(readout).squeeze(-1)

    def fit(self, train_sets, train_labels):
        sets = _as_set_list(train_sets)
        if self.dim is None:
            self._build(_infer_dim(sets))
        _train_set_module(self, self._forward, sets, train_labels)
        return self

    def predict_proba(self, sets):
        sets = _as_set_list(sets)
        if self.dim is None:
            self._build(_infer_dim(sets))
        return _predict_set_module(self, self._forward, sets)


# ---------------------------------------------------------------------------
# synthetic self-test: the whole lesson in miniature
# ---------------------------------------------------------------------------
def _make_synthetic(n, dim, rng):
    """Positives: K vectors whose SUM aligns with a fixed target, but NO single one
    does (each carries target/K + orthogonal noise). Negatives: K vectors with a
    random per-vector target-component that averages to ~0 (individually the SAME
    magnitude along target, but the SUM does NOT align)."""
    target = np.zeros(dim, dtype=np.float32)
    target[0] = 1.0  # fixed goal direction (unit)
    sets, labels = [], []
    for _ in range(n):
        k = int(rng.integers(3, 7))               # K in {3,4,5,6}
        lab = int(rng.integers(0, 2))
        # orthogonal noise (components 1..dim-1); target lives on axis 0
        noise = rng.normal(0, 0.3, size=(k, dim)).astype(np.float32)
        noise[:, 0] = 0.0
        along = np.empty(k, dtype=np.float32)
        if lab == 1:
            along[:] = 1.0 / k                     # every vector: same +1/k -> sum=1
        else:
            signs = rng.choice([-1.0, 1.0], size=k)  # +-1/k -> sum ~ 0
            along[:] = signs * (1.0 / k)
        a = noise.copy()
        a[:, 0] = along
        sets.append(a)
        labels.append(lab)
    return sets, np.asarray(labels, dtype=int)


def _auc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    except Exception:
        # rank-based fallback
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(s) + 1)
        pos = y == 1
        npos, nneg = pos.sum(), (~pos).sum()
        if npos == 0 or nneg == 0:
            return 0.5
        return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


if __name__ == "__main__":
    rng = np.random.default_rng(C.SEED)
    dim = 16
    train_sets, train_y = _make_synthetic(120, dim, rng)
    test_sets, test_y = _make_synthetic(60, dim, rng)

    models = {
        "per_traj_max": PerTrajMax(),
        "mean_agg": MeanAgg(),
        "attn_pool": AttnPool(),
        "gnn_agg": GnnAgg(),
    }
    aucs = {}
    for name, mdl in models.items():
        mdl.fit(train_sets, train_y)
        aucs[name] = _auc(test_y, mdl.predict_proba(test_sets))
        print("%-14s test AUC = %.3f" % (name, aucs[name]))

    # K=1 robustness smoke (must not crash)
    for mdl in models.values():
        _ = mdl.predict_proba([np.asarray(train_sets[0][:1], dtype=np.float32)])

    assert aucs["attn_pool"] > 0.85, "attn_pool must aggregate the set (AUC>0.85)"
    assert aucs["gnn_agg"] > 0.85, "gnn_agg must aggregate the set (AUC>0.85)"
    assert aucs["per_traj_max"] < 0.70, "per_traj_max must miss the joint signal (AUC<0.70)"
    print("OK: aggregators recover the fractured goal; per-trajectory max cannot.")
