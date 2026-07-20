"""trajectory.py — decoding-time token trajectories + the training-free detector.

TRAJGUARD's thesis (Liu et al., ACL 2026 Findings, arXiv:2604.07727): a jailbroken
generation is visible in the model's OWN residual-stream trajectory *as it decodes*.
Token by token, the layer-`C.LAYER` hidden state of a harmful completion drifts toward
a high-risk region; a benign completion stays put. So we can DETECT (and stop) the
jailbreak in real time by projecting each generated-token state onto a fixed HARM
DIRECTION and watching a short sliding window cross a calibrated threshold -- before the
harmful content is fully emitted, and with no prompt-side classifier.

This module owns:
  - generate_and_capture: greedily decode a completion, then ONE forward pass over
    prompt+completion with output_hidden_states -> the per-token trajectory
    [n_gen_tokens, dim] at layer `C.LAYER` (the COMPLETION positions only).
  - harm_direction / token_scores / sliding_window_risk: the projection pipeline.
  - ThresholdDetector: the paper's TRAINING-FREE detector (fit a direction + a tau),
    with an early-K variant for the streaming / early-warning curve.

CPU-only + import-safe: the only heavy imports (torch/transformers via
hello_world_steering.model_utils) live INSIDE generate_and_capture, so
`import steering_tutorials.trajguard.trajectory` never loads a model. The __main__
self-test runs entirely on SYNTHETIC trajectories -- no model, no GPU.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from steering_tutorials.trajguard import config as C


# ---------------------------------------------------------------------------
# 1. Generate a completion and capture its per-token trajectory.
# ---------------------------------------------------------------------------
def generate_and_capture(
    model: Any,
    tok: Any,
    prompt: str,
    max_new_tokens: int = C.MAX_NEW_TOKENS,
    layer: int = C.LAYER,
) -> Tuple[str, np.ndarray]:
    """Greedy-decode ``prompt`` then capture the layer-``layer`` state at each
    GENERATED-token position.

    Steps:
      1. chat-template + greedily generate up to ``max_new_tokens`` (robust to an
         early EOS -> fewer tokens).
      2. ONE forward pass over the full prompt+completion token ids with
         ``output_hidden_states=True``; take ``hidden_states[layer]`` at the
         COMPLETION positions only (everything after the prompt).

    Returns ``(completion_text, trajectory)`` where ``trajectory`` is a
    ``[n_gen_tokens, dim]`` float32 CPU numpy array. If nothing was generated
    (immediate EOS) the trajectory is an empty ``[0, dim]`` array.

    Heavy imports are kept inside the function so importing this module never
    pulls in torch/transformers or loads a model.
    """
    import torch

    device = next(model.parameters()).device
    model.eval()

    # 1) Build the chat-templated prompt ids and greedily generate.
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    prompt_len = int(ids.shape[1])

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: deterministic, reproducible trajectories
            num_beams=1,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None
                          else tok.eos_token_id),
        )

    full_ids = out[0].unsqueeze(0)  # [1, total_len]
    total_len = int(full_ids.shape[1])
    new_ids = full_ids[0, prompt_len:]
    completion_text = tok.decode(new_ids, skip_special_tokens=True).strip()

    dim = int(getattr(model.config, "hidden_size"))
    n_gen = total_len - prompt_len
    if n_gen <= 0:
        return completion_text, np.zeros((0, dim), dtype=np.float32)

    # 2) One forward pass over the full sequence, read hidden states at `layer`.
    with torch.no_grad():
        hs = model(full_ids, output_hidden_states=True).hidden_states
    layer = max(0, min(int(layer), len(hs) - 1))
    # hidden_states[layer]: [1, total_len, dim]; keep the completion positions only.
    traj = hs[layer][0, prompt_len:total_len, :].float().cpu().numpy().astype(np.float32)
    return completion_text, traj


# ---------------------------------------------------------------------------
# 2. The training-free projection pipeline.
# ---------------------------------------------------------------------------
def harm_direction(
    train_trajs: List[np.ndarray], train_labels: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the (center, unit harm direction) from labelled token trajectories.

    center   = mean of ALL train token states (every token of every trajectory).
    unit_dir = unit( mean(harmful token states) - mean(benign token states) ).

    Returns ``(center[dim], unit_dir[dim])`` as float32 arrays.
    """
    labels = np.asarray(train_labels)
    harmful_tok, benign_tok, all_tok = [], [], []
    for traj, lab in zip(train_trajs, labels):
        traj = np.asarray(traj, dtype=np.float32)
        if traj.shape[0] == 0:
            continue
        all_tok.append(traj)
        (harmful_tok if int(lab) == 1 else benign_tok).append(traj)

    all_stack = np.vstack(all_tok)
    dim = all_stack.shape[1]
    center = all_stack.mean(axis=0).astype(np.float32)

    mean_h = np.vstack(harmful_tok).mean(axis=0) if harmful_tok else np.zeros(dim)
    mean_b = np.vstack(benign_tok).mean(axis=0) if benign_tok else np.zeros(dim)
    diff = (mean_h - mean_b).astype(np.float32)
    norm = float(np.linalg.norm(diff))
    unit_dir = (diff / norm).astype(np.float32) if norm > 0 else diff
    return center, unit_dir


def token_scores(traj: np.ndarray, center: np.ndarray, unit_dir: np.ndarray) -> np.ndarray:
    """Per-token harm score: the (centered) state projected onto the harm dir.

    Returns ``[n_tokens]`` = ``(traj - center) @ unit_dir``.
    """
    traj = np.asarray(traj, dtype=np.float32)
    if traj.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return ((traj - center) @ unit_dir).astype(np.float32)


def sliding_window_risk(scores: np.ndarray, window: int = C.WINDOW) -> np.ndarray:
    """Causal running mean of ``scores`` over the last ``window`` tokens.

    Position ``i`` gets ``mean(scores[max(0, i-window+1) : i+1])`` -- only past and
    present tokens (streaming-safe). Returns ``[n_tokens]``.
    """
    scores = np.asarray(scores, dtype=np.float32)
    n = scores.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    window = max(1, int(window))
    # Cumulative-sum trick for an O(n) causal windowed mean.
    csum = np.concatenate([[0.0], np.cumsum(scores)])
    idx = np.arange(1, n + 1)
    lo = np.maximum(0, idx - window)
    out = (csum[idx] - csum[lo]) / (idx - lo)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. The paper's TRAINING-FREE detector.
# ---------------------------------------------------------------------------
class ThresholdDetector:
    """Training-free streaming detector: project onto the harm direction, take the
    causal sliding-window risk, and flag when its per-completion MAX crosses tau.

    ``fit`` learns only (center, unit_dir) + a calibrated ``tau`` (the risk giving
    <= ``C.TARGET_FPR`` benign FPR on train) and ``scale`` (std of train completion
    scores). ``predict_proba`` squashes ``(completion_score - tau)/scale`` through a
    sigmoid so the ranking (and thus AUC) is meaningful. Same call signature as the
    learned models it is baselined against.
    """

    def __init__(self, window: int = C.WINDOW, target_fpr: float = C.TARGET_FPR) -> None:
        self.window = int(window)
        self.target_fpr = float(target_fpr)
        self.center: np.ndarray | None = None
        self.unit_dir: np.ndarray | None = None
        self.tau: float = 0.0
        self.scale: float = 1.0

    # -- internals ---------------------------------------------------------
    def _completion_score(self, traj: np.ndarray) -> float:
        """MAX sliding-window risk over a completion's tokens (its flag score)."""
        s = token_scores(traj, self.center, self.unit_dir)
        if s.shape[0] == 0:
            return 0.0
        return float(sliding_window_risk(s, self.window).max())

    def _scores(self, trajs: List[np.ndarray]) -> np.ndarray:
        return np.array([self._completion_score(t) for t in trajs], dtype=np.float32)

    # -- API ---------------------------------------------------------------
    def fit(self, train_trajs: List[np.ndarray], train_labels: List[int]) -> "ThresholdDetector":
        self.center, self.unit_dir = harm_direction(train_trajs, train_labels)
        labels = np.asarray(train_labels)
        comp = self._scores(train_trajs)

        # tau: benign quantile so that <= target_fpr of benign completions exceed it.
        benign = comp[labels == 0]
        if benign.size > 0:
            self.tau = float(np.quantile(benign, 1.0 - self.target_fpr))
        else:
            self.tau = float(np.median(comp)) if comp.size else 0.0

        # scale: spread of train completion scores (guard against 0).
        std = float(comp.std()) if comp.size else 0.0
        self.scale = std if std > 1e-8 else 1.0
        return self

    def predict_proba(self, trajs: List[np.ndarray]) -> np.ndarray:
        comp = self._scores(trajs)
        return _sigmoid((comp - self.tau) / self.scale).astype(np.float32)

    def predict_proba_earlyK(self, trajs: List[np.ndarray], K: int) -> np.ndarray:
        """Same as ``predict_proba`` but using only the FIRST ``K`` tokens of each
        trajectory -- the streaming / early-warning view (how early can we flag?)."""
        K = max(1, int(K))
        truncated = [np.asarray(t, dtype=np.float32)[:K] for t in trajs]
        comp = self._scores(truncated)
        return _sigmoid((comp - self.tau) / self.scale).astype(np.float32)


# ---------------------------------------------------------------------------
# small helpers (no sklearn dependency).
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U (rank) statistic. Handles ties."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks over tie groups
    s = y_score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# CPU self-test on SYNTHETIC trajectories — NO model download.
# Run: python -m steering_tutorials.trajguard.trajectory
# ---------------------------------------------------------------------------
def _make_synthetic(n_per_class: int, dim: int, rng: np.random.Generator):
    """Harmful trajs drift along a fixed unit dir; benign trajs are flat noise."""
    d = rng.standard_normal(dim)
    d = d / np.linalg.norm(d)
    trajs: List[np.ndarray] = []
    labels: List[int] = []
    noise = 0.15
    for _ in range(n_per_class):  # harmful: state_t = (t/ntok) * drift * dir + noise
        ntok = int(rng.integers(8, 31))
        drift = 3.0
        t = (np.arange(ntok) / max(1, ntok - 1)).reshape(-1, 1)
        traj = t * drift * d.reshape(1, -1) + noise * rng.standard_normal((ntok, dim))
        trajs.append(traj.astype(np.float32))
        labels.append(1)
    for _ in range(n_per_class):  # benign: flat noise, no drift
        ntok = int(rng.integers(8, 31))
        traj = noise * rng.standard_normal((ntok, dim))
        trajs.append(traj.astype(np.float32))
        labels.append(0)
    return trajs, labels


def _self_test() -> None:
    rng = np.random.default_rng(0)
    dim = 16
    n_per_class = 60

    tr_trajs, tr_labels = _make_synthetic(n_per_class, dim, rng)
    te_trajs, te_labels = _make_synthetic(n_per_class, dim, rng)

    det = ThresholdDetector().fit(tr_trajs, tr_labels)

    # (a) held-out AUC on full trajectories.
    proba = det.predict_proba(te_trajs)
    auc_full = _auc(np.array(te_labels), proba)
    print("[self-test] held-out full-trajectory AUC = %.4f" % auc_full)
    assert auc_full > 0.9, "expected held-out AUC > 0.9, got %.4f" % auc_full

    # (b) early-K risk on harmful completions should rise (non-decreasing-ish) with K.
    te = np.array(te_labels)
    harmful = [t for t, l in zip(te_trajs, te_labels) if l == 1]
    Ks = [2, 4, 8, 16]
    means = [float(det.predict_proba_earlyK(harmful, K).mean()) for K in Ks]
    print("[self-test] mean harmful early-K risk:",
          " ".join("K=%d:%.4f" % (k, m) for k, m in zip(Ks, means)))
    tol = 1e-6
    for a, b in zip(means, means[1:]):
        assert b >= a - tol, "early-K risk should be non-decreasing in K: %s" % means

    # (c) early-K AUC improves (non-decreasing-ish) as we see more tokens --
    #     the streaming value: separability grows with the observed prefix.
    aucs = []
    for K in Ks:
        p = det.predict_proba_earlyK(te_trajs, K)
        aucs.append(_auc(te, p))
    print("[self-test] early-K held-out AUC:",
          " ".join("K=%d:%.4f" % (k, a) for k, a in zip(Ks, aucs)))
    for a, b in zip(aucs, aucs[1:]):
        assert b >= a - 0.02, "early-K AUC should rise with K: %s" % aucs
    assert aucs[-1] >= auc_full - 0.1, "long-prefix AUC should approach full AUC"

    print("[self-test] OK -- training-free detector separates drift vs flat; "
          "risk rises with K.")


if __name__ == "__main__":
    _self_test()
