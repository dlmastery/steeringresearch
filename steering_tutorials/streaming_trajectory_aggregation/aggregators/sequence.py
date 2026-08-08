"""sequence.py -- the learned-encoder arm of the offline aggregator ladder.

Where `pooling.py` fixes the classifier (a shared sklearn LogisticRegression) and lets
only the pooling FUNCTION vary, the two aggregators here are trained END TO END: the
encoder and the linear head are fit jointly by backprop. That is an intentionally
different axis of comparison, not an oversight -- see each class's docstring for the
paper it stands in for.

Both networks are deliberately SMALL (default hidden=32) so a full ladder run trains
in minutes on a 4090, not hours -- this repo has run CPU-bound learned baselines
before and the lesson stuck (CLAUDE.md section 17 operational playbook item 2 on host
RAM/GPU budgets).

TRAINING DISCIPLINE (CLAUDE.md section 17, "the stiff two-term loss" lesson, applied
here even though this loss is single-term): fixed seeds, gradient-norm clipping, and
BEST-checkpointing on a held-out validation split's loss, never the last epoch. A
recurrent/attention classifier trained on a few dozen trajectories can and does
oscillate; saving whatever the last epoch happened to land on silently ships a worse
model than one already seen during training.

CPU-only to import: `torch` is imported but nothing here touches `.cuda()` or moves a
tensor to a non-CPU device unless the caller explicitly passes `device=...`. Loads no
pretrained model, downloads nothing. ASCII stdout only.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

__all__ = ["GRUAggregator", "QueryTokenCompressor"]


class _SeqAggregatorBase:
    """Shared train/score scaffolding for the two learned sequence aggregators.

    Subclasses implement `_build_net(dim) -> nn.Module` and `_forward_score(net, x)
    -> torch.Tensor` (a scalar logit for one trajectory, `x` shaped `[n_steps, dim]`,
    no batch dimension). Everything else -- the train/val split, the training loop,
    gradient clipping, best-checkpointing, and `score()` -- is identical across both,
    so the comparison between them is about the ENCODER, not about who got a better
    training loop.
    """

    name = "seq_aggregator_base"
    is_causal = True

    def __init__(
        self,
        hidden: int = 32,
        lr: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 16,
        val_frac: float = 0.15,
        patience: int = 8,
        grad_clip: float = 1.0,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.patience = patience
        self.grad_clip = grad_clip
        self.seed = seed
        self.device = device
        self.dim_: int | None = None
        self.net_: nn.Module | None = None

    # -- subclasses implement these two -----------------------------------------
    def _build_net(self, dim: int) -> nn.Module:
        raise NotImplementedError

    def _forward_score(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # -- Aggregator protocol ------------------------------------------------------
    def fit(self, trajectories, labels) -> None:
        trajs = [np.asarray(t, dtype=np.float32) for t in trajectories]
        labels_int = [int(y) for y in labels]
        if len(trajs) == 0:
            raise ValueError("%s.fit called with zero trajectories" % self.name)
        if len(set(labels_int)) < 2:
            raise ValueError(
                "%s.fit called with a single class present -- fix the caller's split, "
                "do not silently degrade to a constant predictor." % self.name
            )
        dim = trajs[0].shape[1]
        self.dim_ = dim

        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(trajs))
        rng.shuffle(idx)
        n_val = int(round(len(idx) * self.val_frac))
        # need at least one example per split; fall back to reusing train as val
        # (a degraded-but-honest signal, not a crash) when the fold is too small.
        if len(idx) < 4 or n_val < 1 or n_val >= len(idx):
            train_idx, val_idx = idx, idx
        else:
            val_idx, train_idx = idx[:n_val], idx[n_val:]

        torch.manual_seed(self.seed)
        net = self._build_net(dim).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        best_state = copy.deepcopy(net.state_dict())
        best_val = float("inf")
        stall = 0
        train_list = list(train_idx)

        for _epoch in range(self.epochs):
            net.train()
            rng.shuffle(train_list)
            for start in range(0, len(train_list), self.batch_size):
                batch = train_list[start : start + self.batch_size]
                if not batch:
                    continue
                opt.zero_grad()
                losses = []
                for i in batch:
                    x = torch.from_numpy(trajs[i]).to(self.device)
                    logit = self._forward_score(net, x)
                    y = torch.tensor(float(labels_int[i]), device=self.device)
                    losses.append(loss_fn(logit, y))
                loss = torch.stack(losses).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                opt.step()

            net.eval()
            with torch.no_grad():
                vlosses = []
                for i in val_idx:
                    x = torch.from_numpy(trajs[i]).to(self.device)
                    logit = self._forward_score(net, x)
                    y = torch.tensor(float(labels_int[i]), device=self.device)
                    vlosses.append(loss_fn(logit, y).item())
                vloss = float(np.mean(vlosses)) if vlosses else float("inf")

            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = copy.deepcopy(net.state_dict())
                stall = 0
            else:
                stall += 1
                if stall >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net

    def score(self, trajectory) -> float:
        if self.net_ is None:
            raise RuntimeError("%s.fit must be called before score" % self.name)
        a = np.asarray(trajectory, dtype=np.float32)
        with torch.no_grad():
            x = torch.from_numpy(a).to(self.device)
            logit = self._forward_score(self.net_, x)
            return float(torch.sigmoid(logit).item())


# --- GRUAggregator -----------------------------------------------------------------
class _GRUNet(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(input_size=dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, n_steps, dim] -> final hidden state -> logit
        _, h_n = self.gru(x)
        return self.head(h_n[-1]).squeeze(-1)


class GRUAggregator(_SeqAggregatorBase):
    """GRU encoder over the step sequence; final hidden state -> linear head.

    This is the Trajectory Guard family: Advani, L., 2 Jan 2026, AAAI TrustAgent 2026,
    "Trajectory Guard -- A Lightweight, Sequence-Aware Model for Real-Time Anomaly
    Detection in Agentic AI" (arXiv:2601.00516). Its own abstract states the mechanism
    this whole ladder is built to test: "mean-pooling embeddings dilutes anomalous
    steps, while contrastive-only approaches ignore sequential structure" -- the paper's
    proposed fix is a Siamese Recurrent Autoencoder with a hybrid contrastive +
    reconstruction loss, reporting F1 0.88-0.94 at ~32ms inference latency on its own
    corpus. This class is a plain GRU classifier, not a reproduction of that Siamese-
    autoencoder architecture or its loss -- it isolates the STRUCTURAL claim (recurrent
    state beats fixed pooling) from that paper's specific training recipe. Do not
    report this class's numbers alongside the paper's F1 0.88-0.94 / 32ms as though
    they were measuring the same thing; those come from their model, their corpus, and
    their latency setup, not this one.

    `is_causal = True`: a GRU's hidden state after step t is a pure function of steps
    0..t. Handed a genuine prefix, `score()` reads the hidden state at the prefix's
    end and never touches a later step -- unlike `QueryTokenCompressor` below, whose
    attention is NOT masked. This is the one aggregator in this file, besides the
    pooling.py control (`LastStep`), whose causality claim rests on the architecture
    itself rather than on a convention about how it happens to be called.
    """

    name = "gru"
    is_causal = True

    def _build_net(self, dim: int) -> nn.Module:
        return _GRUNet(dim, self.hidden)

    def _forward_score(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return net(x.unsqueeze(0)).squeeze(0)


# --- QueryTokenCompressor -----------------------------------------------------------
class _QueryTokenNet(nn.Module):
    """K learnable query tokens cross-attend over the full step sequence.

    Deliberately FULL (unmasked) attention: every query attends to every step
    regardless of position. That is what makes `QueryTokenCompressor.is_causal =
    False` a fact about the architecture, not a formality -- there is no masking here
    to remove.
    """

    def __init__(self, dim: int, hidden: int, k: int):
        super().__init__()
        self.k = k
        self.queries = nn.Parameter(torch.randn(k, hidden) * 0.02)
        self.in_proj = nn.Linear(dim, hidden)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=1, batch_first=True)
        self.head = nn.Linear(hidden * k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, n_steps, dim]
        kv = self.in_proj(x)  # [1, n_steps, hidden]
        q = self.queries.unsqueeze(0)  # [1, k, hidden]
        out, _ = self.attn(q, kv, kv)  # [1, k, hidden], full attention, no mask
        flat = out.reshape(out.shape[0], -1)  # [1, k*hidden]
        return self.head(flat).squeeze(-1)


class QueryTokenCompressor(_SeqAggregatorBase):
    """`k` learnable query tokens attend over ALL step embeddings; the queries' final
    states are concatenated and fed to a linear head.

    This is the TRACE family: Hong, Z., Wang, L., Li, L., Ma, H., Fang, J., Shen, F.,
    Zhang, D., Wang, X., 30 May 2026, "TRACE: Trajectory Risk-Aware Compression for
    Long-Horizon Agent Safety" (arXiv:2606.00611). It frames long-horizon agent-safety
    detection as compressing a full trajectory into a compact latent evidence state
    (a Compressor) that a Reader then judges, reporting up to +12.6 percentage points
    across ASSEBench / Pre-Ex-Bench / R-Judge with better resilience as context length
    grows. This class is a small standalone approximation of the "compress via learned
    queries" idea, not a reproduction of their Compressor+Reader pipeline or their
    benchmark numbers -- do not print this class's AUCs beside the paper's +12.6pp as
    though comparable; none of ASSEBench, Pre-Ex-Bench or R-Judge is reproducible on
    this repo's substrate (see `../types.py`'s "WHAT WE CANNOT DO" section).

    `k=16` IS NOT CONFIRMED FROM THE PAPER'S ABSTRACT.
    `CITATIONS_VERIFIED.md`'s "Numbers that are NOT in the abstract" table marks
    "K=16 query tokens" `[NEEDS VERIFICATION -- not in abstract]` for arXiv:2606.00611
    -- it is plausibly a method-section implementation detail that was never confirmed
    against the paper body. `k` is therefore a REQUIRED, explicit config parameter
    here (default below is a harness convenience, not an assertion that 16 is TRACE's
    own value) -- callers sweeping this ladder should treat `k` as a hyperparameter to
    search, not a fact to cite.

    `is_causal = False`: `nn.MultiheadAttention` runs full, unmasked attention -- every
    query sees every step regardless of position. A score computed by handing this
    class a "prefix" is not what an online monitor watching in real time would see,
    because the network is free to use whichever steps it is given, in any order. If
    the ladder ever scores prefixes with this class for a horizon-truncation curve,
    label that comparison explicitly as "full-trajectory-visibility vs honest-prefix"
    rather than folding it into the causal aggregators' curve.
    """

    is_causal = False

    def __init__(self, k: int = 16, **kwargs):
        super().__init__(**kwargs)
        if k < 1:
            raise ValueError("k must be >= 1, got %r" % (k,))
        self.k = int(k)
        self.name = "query_token_compressor_k%d" % self.k

    def _build_net(self, dim: int) -> nn.Module:
        return _QueryTokenNet(dim, self.hidden, self.k)

    def _forward_score(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return net(x.unsqueeze(0)).squeeze(0)


# --- CPU smoke: python -m streaming_trajectory_aggregation.aggregators.sequence ------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dim = 6

    def _make_traj(n_steps, spike):
        a = rng.normal(0.0, 1.0, size=(n_steps, dim)).astype(np.float32)
        if spike is not None:
            a[spike] += 6.0
        return a

    trajs, labels = [], []
    for _ in range(16):
        n = int(rng.integers(5, 10))
        trajs.append(_make_traj(n, spike=int(rng.integers(0, n))))
        labels.append(1)
    for _ in range(16):
        n = int(rng.integers(5, 10))
        trajs.append(_make_traj(n, spike=None))
        labels.append(0)

    for agg in (
        GRUAggregator(hidden=8, epochs=5, batch_size=4, seed=0),
        QueryTokenCompressor(k=4, hidden=8, epochs=5, batch_size=4, seed=0),
    ):
        agg.fit(trajs, labels)
        scores = [agg.score(t) for t in trajs]
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        print(
            "%-28s is_causal=%-5s mean_score pos=%.3f neg=%.3f"
            % (agg.name, agg.is_causal, float(np.mean(pos)), float(np.mean(neg)))
        )
    print("OK -- sequence.py aggregators fit/score end to end")
