"""models.py -- the four conversation classifiers for multi-turn jailbreak detection.

Every classifier exposes the SAME interface:
    .fit(train_seqs, train_labels)          # train_seqs: list of [n_turns, dim] float32
    .predict_proba(seqs) -> np.ndarray[n]   # P(attack) in [0,1], one per conversation

The four models form a pedagogical ladder from stateless to stateful:

  1. PerTurnMaxProbe  -- STATELESS BASELINE. A logistic regression on INDIVIDUAL turn
     vectors; the conversation score is the MAX per-turn P(attack). It literally cannot
     see turn order, so a multi-turn attack whose every single turn looks benign but
     whose TRAJECTORY escalates slips right past it. That failure is the whole point.
  2. TrajectoryMLP    -- hand-crafted trajectory features (mean/last/max/std over turns
     plus the mean consecutive delta and the max drift-from-turn0 vectors) -> small MLP.
     Sees the shape of the trajectory but through a fixed summary.
  3. SeqGRU           -- the HEADLINE model. A GRU over the turn sequence; its last hidden
     state -> logit. Exposes risk_trajectory(seq): the running per-turn risk that a
     stateful detector reports as an attack escalates.
  4. HierAttn         -- per-turn encoder + additive attention pool over turns. Exposes
     attention_weights(seq): which turns the detector focused on.

Pure numpy/torch/sklearn, CPU-only. The __main__ self-test uses ONLY synthetic data
(escalating vs flat sequences) -- no real dataset, no model download.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - sklearn is installed, but stay robust
    _HAVE_SKLEARN = False

from . import config as C


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_seq_list(seqs):
    """Coerce to a list of float32 [n_turns, dim] arrays, each with >=1 turn."""
    out = []
    for s in seqs:
        a = np.asarray(s, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[0] == 0:
            a = np.zeros((1, a.shape[-1]), dtype=np.float32)
        out.append(a)
    return out


def _infer_dim(seqs) -> int:
    for s in seqs:
        a = np.asarray(s)
        if a.ndim == 2 and a.shape[0] > 0:
            return int(a.shape[1])
        if a.ndim == 1:
            return int(a.shape[0])
    raise ValueError("cannot infer feature dim from empty seqs")


def _set_seeds():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)


def _iter_minibatches(n, batch, seed):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    for i in range(0, n, batch):
        yield idx[i:i + batch]


# ---------------------------------------------------------------------------
# 1. PerTurnMaxProbe -- stateless baseline
# ---------------------------------------------------------------------------
class PerTurnMaxProbe:
    """Logistic regression on individual turns; conversation score = MAX per-turn P.

    Each turn inherits its conversation's label at fit time. It has no notion of order
    or accumulation across turns -- exactly the blind spot multi-turn attacks exploit.
    """

    def __init__(self):
        self.scaler = None
        self.clf = None
        self._torch_fallback = None
        self.dim = None

    def fit(self, train_seqs, train_labels):
        seqs = _as_seq_list(train_seqs)
        labels = np.asarray(train_labels).astype(np.float32)
        self.dim = _infer_dim(seqs)

        # explode conversations into (turn_vector, conv_label) rows
        X_rows, y_rows = [], []
        for seq, lab in zip(seqs, labels):
            for turn in seq:
                X_rows.append(turn)
                y_rows.append(lab)
        X = np.asarray(X_rows, dtype=np.float32)
        y = np.asarray(y_rows, dtype=np.float32)

        self.scaler = StandardScaler().fit(X) if _HAVE_SKLEARN else _NumpyScaler().fit(X)
        Xs = self.scaler.transform(X)

        # both classes present? use logistic; else degenerate constant.
        if len(np.unique(y)) < 2:
            self._const = float(y[0]) if len(y) else 0.0
            self.clf = None
            return self
        self._const = None

        if _HAVE_SKLEARN:
            self.clf = LogisticRegression(max_iter=1000, C=1.0)
            self.clf.fit(Xs, y)
        else:  # pragma: no cover
            self._fit_torch_logistic(Xs, y)
        return self

    def _fit_torch_logistic(self, Xs, y):  # pragma: no cover - fallback path
        _set_seeds()
        model = nn.Linear(Xs.shape[1], 1)
        opt = torch.optim.Adam(model.parameters(), lr=C.LR)
        lossf = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(Xs.astype(np.float32))
        yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
        model.train()
        for _ in range(max(C.EPOCHS, 200)):
            opt.zero_grad()
            loss = lossf(model(Xt), yt)
            loss.backward()
            opt.step()
        model.eval()
        self._torch_fallback = model

    def _turn_probs(self, turns):
        Xs = self.scaler.transform(np.asarray(turns, dtype=np.float32))
        if self._const is not None:
            return np.full(len(Xs), self._const, dtype=np.float32)
        if self.clf is not None:
            return self.clf.predict_proba(Xs)[:, 1].astype(np.float32)
        with torch.no_grad():  # pragma: no cover
            logit = self._torch_fallback(torch.from_numpy(Xs.astype(np.float32)))
            return torch.sigmoid(logit).squeeze(1).numpy().astype(np.float32)

    def predict_proba(self, seqs):
        seqs = _as_seq_list(seqs)
        out = np.zeros(len(seqs), dtype=np.float32)
        for i, seq in enumerate(seqs):
            out[i] = float(np.max(self._turn_probs(seq)))
        return out


class _NumpyScaler:  # pragma: no cover - only if sklearn missing
    def fit(self, X):
        self.mean_ = X.mean(0)
        self.std_ = X.std(0) + 1e-8
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float32) - self.mean_) / self.std_


# ---------------------------------------------------------------------------
# 2. TrajectoryMLP -- hand-crafted trajectory features -> MLP
# ---------------------------------------------------------------------------
def _trajectory_features(seq):
    """concat[mean, last, max, std, mean-consecutive-delta, max-drift-from-turn0] -> 6*dim."""
    a = np.asarray(seq, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    n, d = a.shape
    mean = a.mean(0)
    last = a[-1]
    mx = a.max(0)
    std = a.std(0) if n > 1 else np.zeros(d, dtype=np.float32)
    if n > 1:
        deltas = a[1:] - a[:-1]
        mean_delta = deltas.mean(0)
        drift = a - a[0]                      # drift of each turn from turn 0
        max_drift = drift[np.argmax(np.abs(drift), axis=0), np.arange(d)]
    else:
        mean_delta = np.zeros(d, dtype=np.float32)
        max_drift = np.zeros(d, dtype=np.float32)
    return np.concatenate([mean, last, mx, std, mean_delta, max_drift]).astype(np.float32)


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class TrajectoryMLP:
    def __init__(self, hidden=None, epochs=None):
        self.hidden = hidden or C.MLP_HIDDEN
        self.epochs = epochs or C.EPOCHS
        self.scaler = None
        self.model = None

    def fit(self, train_seqs, train_labels):
        _set_seeds()
        seqs = _as_seq_list(train_seqs)
        y = np.asarray(train_labels).astype(np.float32)
        X = np.stack([_trajectory_features(s) for s in seqs]).astype(np.float32)

        self.scaler = StandardScaler().fit(X) if _HAVE_SKLEARN else _NumpyScaler().fit(X)
        Xs = self.scaler.transform(X).astype(np.float32)

        self.model = _MLP(Xs.shape[1], self.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=C.LR)
        lossf = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(y).unsqueeze(1)
        self.model.train()
        for ep in range(self.epochs):
            for bidx in _iter_minibatches(len(Xt), C.BATCH, C.SEED + ep):
                opt.zero_grad()
                loss = lossf(self.model(Xt[bidx]), yt[bidx])
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def predict_proba(self, seqs):
        seqs = _as_seq_list(seqs)
        X = np.stack([_trajectory_features(s) for s in seqs]).astype(np.float32)
        Xs = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            logit = self.model(torch.from_numpy(Xs))
            return torch.sigmoid(logit).squeeze(1).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# 3. SeqGRU -- the headline stateful model
# ---------------------------------------------------------------------------
class _GRUNet(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gru = nn.GRU(input_size=dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward_packed(self, padded, lengths):
        packed = pack_padded_sequence(padded, lengths, batch_first=True,
                                      enforce_sorted=False)
        _, h_n = self.gru(packed)          # h_n: [1, batch, hidden]
        return self.head(h_n[-1])          # [batch, 1]

    def forward_steps(self, seq):
        """Return per-step hidden states [n_turns, hidden] for one [n_turns, dim] seq."""
        out, _ = self.gru(seq.unsqueeze(0))   # [1, n_turns, hidden]
        return out.squeeze(0)


class SeqGRU:
    """GRU over the turn sequence; last hidden -> logit. Headline stateful detector."""

    def __init__(self, hidden=None, epochs=None):
        self.hidden = hidden or C.GRU_HIDDEN
        self.epochs = epochs or C.EPOCHS
        self.dim = None
        self.model = None

    def fit(self, train_seqs, train_labels):
        _set_seeds()
        seqs = _as_seq_list(train_seqs)
        y = np.asarray(train_labels).astype(np.float32)
        self.dim = _infer_dim(seqs)
        tensors = [torch.from_numpy(s) for s in seqs]
        lengths = np.array([t.shape[0] for t in tensors])

        self.model = _GRUNet(self.dim, self.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=C.LR)
        lossf = nn.BCEWithLogitsLoss()
        yt = torch.from_numpy(y)
        self.model.train()
        for ep in range(self.epochs):
            for bidx in _iter_minibatches(len(tensors), C.BATCH, C.SEED + ep):
                batch = [tensors[j] for j in bidx]
                lens = lengths[bidx]
                padded = pad_sequence(batch, batch_first=True)  # [b, maxlen, dim]
                opt.zero_grad()
                logits = self.model.forward_packed(padded,
                                                   torch.as_tensor(lens, dtype=torch.long))
                loss = lossf(logits.squeeze(1), yt[bidx])
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def predict_proba(self, seqs):
        seqs = _as_seq_list(seqs)
        out = np.zeros(len(seqs), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for i, s in enumerate(seqs):
                t = torch.from_numpy(s).unsqueeze(0)          # [1, n, dim]
                lens = torch.as_tensor([s.shape[0]], dtype=torch.long)
                logit = self.model.forward_packed(t, lens)
                out[i] = float(torch.sigmoid(logit).item())
        return out

    def risk_trajectory(self, seq):
        """Running risk sigmoid(head(h_t)) at each turn -- the escalation curve."""
        s = _as_seq_list([seq])[0]
        self.model.eval()
        with torch.no_grad():
            steps = self.model.forward_steps(torch.from_numpy(s))   # [n, hidden]
            logits = self.model.head(steps).squeeze(1)              # [n]
            return torch.sigmoid(logits).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# 4. HierAttn -- per-turn encoder + additive attention pool
# ---------------------------------------------------------------------------
class _AttnNet(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.enc = nn.Linear(dim, hidden)
        self.attn = nn.Linear(hidden, 1, bias=False)  # additive attention scorer
        self.head = nn.Linear(hidden, 1)

    def encode(self, seq):
        """seq [n, dim] -> (weighted_pool [hidden], weights [n], logit scalar)."""
        h = torch.tanh(self.enc(seq))          # [n, hidden]
        scores = self.attn(h).squeeze(-1)      # [n]
        weights = torch.softmax(scores, dim=0)  # [n]
        pooled = (weights.unsqueeze(-1) * h).sum(0)  # [hidden]
        logit = self.head(pooled).squeeze(-1)
        return pooled, weights, logit

    def forward_batch(self, padded, mask):
        """padded [b, L, dim], mask [b, L] (1=real turn) -> logits [b]."""
        h = torch.tanh(self.enc(padded))               # [b, L, hidden]
        scores = self.attn(h).squeeze(-1)              # [b, L]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)         # [b, L]
        pooled = (weights.unsqueeze(-1) * h).sum(1)    # [b, hidden]
        return self.head(pooled).squeeze(-1)


class HierAttn:
    """Per-turn tanh encoder -> additive attention over turns -> classifier."""

    def __init__(self, hidden=None, epochs=None):
        self.hidden = hidden or C.ATTN_HIDDEN
        self.epochs = epochs or C.EPOCHS
        self.dim = None
        self.model = None

    def fit(self, train_seqs, train_labels):
        _set_seeds()
        seqs = _as_seq_list(train_seqs)
        y = np.asarray(train_labels).astype(np.float32)
        self.dim = _infer_dim(seqs)
        tensors = [torch.from_numpy(s) for s in seqs]
        lengths = np.array([t.shape[0] for t in tensors])

        self.model = _AttnNet(self.dim, self.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=C.LR)
        lossf = nn.BCEWithLogitsLoss()
        yt = torch.from_numpy(y)
        self.model.train()
        for ep in range(self.epochs):
            for bidx in _iter_minibatches(len(tensors), C.BATCH, C.SEED + ep):
                batch = [tensors[j] for j in bidx]
                lens = lengths[bidx]
                padded = pad_sequence(batch, batch_first=True)   # [b, L, dim]
                L = padded.shape[1]
                mask = torch.zeros(len(batch), L)
                for k, ln in enumerate(lens):
                    mask[k, :ln] = 1.0
                opt.zero_grad()
                logits = self.model.forward_batch(padded, mask)
                loss = lossf(logits, yt[bidx])
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def predict_proba(self, seqs):
        seqs = _as_seq_list(seqs)
        out = np.zeros(len(seqs), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for i, s in enumerate(seqs):
                _, _, logit = self.model.encode(torch.from_numpy(s))
                out[i] = float(torch.sigmoid(logit).item())
        return out

    def attention_weights(self, seq):
        """Softmax attention weight per turn -- which turns the detector focused on."""
        s = _as_seq_list([seq])[0]
        self.model.eval()
        with torch.no_grad():
            _, weights, _ = self.model.encode(torch.from_numpy(s))
            return weights.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# synthetic self-test -- NO real data, NO model download
# ---------------------------------------------------------------------------
def _auc(y_true, y_score):
    """Rank-based AUC (Mann-Whitney), no sklearn dependency for the test print."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    sum_pos = ranks[y_true == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _make_synthetic(n, dim, rng, direction):
    """Positives: turn vectors drift monotonically along a FIXED, SHARED direction
    (escalation). Negatives: turns are random noise around 0 with no trajectory.

    `direction` is passed in (not drawn here) so train and test share the SAME attack
    direction -- otherwise the model learns one direction and is tested on another.
    """
    seqs, labels = [], []
    for i in range(n):
        n_turns = int(rng.randint(3, 7))          # {3,4,5,6}
        lab = i % 2
        if lab == 1:
            # escalating: each turn moves further along the shared `direction`.
            # Low per-turn noise so the monotonic directional drift dominates.
            steps = np.linspace(0.0, 4.0, n_turns).astype(np.float32)
            base = rng.randn(n_turns, dim).astype(np.float32) * 0.2
            seq = base + steps[:, None] * direction[None, :]
        else:
            # flat: random around 0, no directional drift.
            seq = rng.randn(n_turns, dim).astype(np.float32) * 0.5
        seqs.append(seq.astype(np.float32))
        labels.append(lab)
    return seqs, np.array(labels)


def _self_test():
    rng = np.random.RandomState(C.SEED)
    dim = 16
    # ONE fixed attack direction, shared by train and test.
    direction = rng.randn(dim).astype(np.float32)
    direction /= (np.linalg.norm(direction) + 1e-8)
    train_seqs, train_y = _make_synthetic(120, dim, rng, direction)
    test_seqs, test_y = _make_synthetic(60, dim, rng, direction)

    models = {
        "per_turn_max": PerTurnMaxProbe(),
        "trajectory_mlp": TrajectoryMLP(),
        "seq_gru": SeqGRU(),
        "hier_attn": HierAttn(),
    }
    aucs = {}
    for name, m in models.items():
        m.fit(train_seqs, train_y)
        p = m.predict_proba(test_seqs)
        aucs[name] = _auc(test_y, p)
        print("  %-16s test AUC = %.4f" % (name, aucs[name]))

    # extra: show the GRU running-risk escalation on one attack vs one benign
    atk = next(s for s, y in zip(test_seqs, test_y) if y == 1)
    ben = next(s for s, y in zip(test_seqs, test_y) if y == 0)
    print("  seq_gru risk_trajectory  attack:", np.round(models["seq_gru"].risk_trajectory(atk), 3))
    print("  seq_gru risk_trajectory  benign:", np.round(models["seq_gru"].risk_trajectory(ben), 3))
    print("  hier_attn attention_weights attack:",
          np.round(models["hier_attn"].attention_weights(atk), 3))

    # the stateful models MUST catch the trajectory; the stateless baseline may lag.
    assert aucs["seq_gru"] > 0.85, "seq_gru AUC too low: %.4f" % aucs["seq_gru"]
    assert aucs["hier_attn"] > 0.85, "hier_attn AUC too low: %.4f" % aucs["hier_attn"]
    print("SELF-TEST PASSED: stateful models (seq_gru, hier_attn) clear AUC>0.85.")
    print("  (per_turn_max AUC = %.4f -- the stateless baseline, shown for contrast)"
          % aucs["per_turn_max"])


if __name__ == "__main__":
    _self_test()
