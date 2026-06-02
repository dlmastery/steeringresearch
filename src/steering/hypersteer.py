"""hypersteer.py — E45 description→vector hypernetwork (AXIS 7, the SOURCE).

This module tests E45 (HyperSteer zero-shot, arXiv:2506.03292): instead of
curating a contrast-pair dataset per behavior and running DiffMean
(``extract.diffmean_vector`` — a CLOSED-FORM source), we train a small
hypernetwork ``H: description -> steering_vector`` that maps a natural-language
*description* of a behavior to its steering vector in the residual stream. The
supervised DiffMean vectors are the regression TARGETS; the description
embeddings are the inputs.

Two axis-7 sources contrasted in one file:

  * DiffMean (``extract.py``): closed-form ``mean(pos) - mean(neg)`` — needs a
    contrast set per behavior.
  * HyperSteer (here): GRADIENT optimization of an MLP that *generalizes* to
    held-out behaviors it never saw a contrast set for. The objective (spec §5)
    is vector-space regression::

        L = sum_i || H(embed(description_i)) - v_i ||^2   (+ L2 weight decay)

The description encoder (``encode_descriptions``) follows the spec/SciCritic-F
recommendation to use *Gemma's own hidden state* rather than a new all-MiniLM
dependency: each description is run through the (frozen) model and its residual
activation at a chosen layer is mean-pooled. This keeps the module
dependency-free and works identically on ``FakeResidualLM`` (offline tests) and
real Gemma.

Everything is float32, deterministic via an explicit ``seed``, and offline.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .hooks import probe_activations
from .model import encode_to_device


def encode_descriptions(
    model: nn.Module,
    tokenizer,
    descriptions: Sequence[str],
    layer: int,
) -> np.ndarray:
    """Embed each behavior description as a mean-pooled residual activation.

    Runs every ``description`` through the frozen ``model`` (no grad) and
    mean-pools the residual stream at ``layer`` over the sequence axis, giving
    one fixed-length vector per description. This is the E45 description encoder
    — the spec allows "Gemma's own encoder" in place of all-MiniLM, so we use
    the model's own hidden state and add NO new dependency.

    Parameters
    ----------
    model        : FakeResidualLM (tests) or real Gemma (frozen).
    tokenizer    : matching tokenizer (the offline ``_FakeTokenizer`` works).
    descriptions : list of natural-language behavior descriptions.
    layer        : residual-layer index whose activation is pooled.

    Returns
    -------
    np.ndarray ``[n, dim]`` float32 — one embedding row per description.
    """
    embeddings: list[np.ndarray] = []
    for desc in descriptions:
        ids = encode_to_device(tokenizer, desc, model)
        acts = probe_activations(model, ids, [layer])[layer]  # [1, seq, dim]
        # Mean-pool over the sequence axis. .float().cpu(): real Gemma is bf16
        # on CUDA, neither of which numpy accepts.
        pooled = acts[0].mean(dim=0).float().cpu().numpy()
        embeddings.append(pooled.astype(np.float32))
    return np.stack(embeddings, axis=0)


class HyperNet(nn.Module):
    """Small MLP hypernetwork mapping description embeddings -> steering vectors.

    Architecture (spec §7): ``embed_dim -> hidden -> ... -> vector_dim`` with
    ReLU activations and a configurable number of hidden layers (``depth``,
    2-3 per the spec). The final linear layer is unactivated so the output can
    take any sign/scale in residual-stream space. float32 throughout.
    """

    def __init__(
        self,
        embed_dim: int,
        vector_dim: int,
        hidden: int = 256,
        depth: int = 2,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1 hidden layer, got {depth}")
        layers: list[nn.Module] = [nn.Linear(embed_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, vector_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[n, embed_dim]`` -> ``[n, vector_dim]``."""
        return self.net(x)


def _to_float32_tensor(arr: np.ndarray) -> torch.Tensor:
    """numpy -> contiguous float32 CPU tensor (the harness trains on CPU)."""
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def train_hypernet(
    embeddings: np.ndarray,
    target_vectors: np.ndarray,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    l2: float = 1e-4,
    hidden: int = 256,
    depth: int = 2,
    seed: int = 0,
    normalize_targets: bool = False,
) -> HyperNet:
    """Train the hypernetwork by MSE regression onto the supervised vectors.

    This is the GRADIENT-optimization source of a steering vector — contrast it
    with the closed-form DiffMean in ``extract.diffmean_vector``. The loss is
    the spec §5 objective ``L = sum_i || H(embed_i) - v_i ||^2`` realised as a
    mean-squared-error, with Adam ``weight_decay=l2`` providing the L2
    regularisation the spec's committee Q&A calls for.

    Parameters
    ----------
    embeddings       : ``[n, embed_dim]`` description embeddings (inputs).
    target_vectors   : ``[n, vector_dim]`` supervised DiffMean vectors (targets).
    epochs, lr, l2   : Adam optimisation hyperparameters (l2 -> weight_decay).
    hidden, depth    : HyperNet architecture.
    seed             : determinism — same seed reproduces identical weights.
    normalize_targets: if True, unit-normalise each target vector before fitting
                       (the spec normalises vectors; off by default so callers
                       opt in explicitly). Direction-only fitting is what a
                       cosine eval rewards; magnitude is recovered by alpha.

    Returns
    -------
    A trained ``HyperNet`` in ``eval()`` mode.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    target_vectors = np.asarray(target_vectors, dtype=np.float32)
    if embeddings.shape[0] != target_vectors.shape[0]:
        raise ValueError(
            f"row mismatch: {embeddings.shape[0]} embeddings vs "
            f"{target_vectors.shape[0]} targets"
        )
    if normalize_targets:
        norms = np.linalg.norm(target_vectors, axis=1, keepdims=True)
        target_vectors = target_vectors / (norms + 1e-8)

    # Full determinism: seed torch before constructing the net (weight init) and
    # before the optimiser sees any data.
    torch.manual_seed(seed)
    net = HyperNet(
        embed_dim=embeddings.shape[1],
        vector_dim=target_vectors.shape[1],
        hidden=hidden,
        depth=depth,
    )
    net.train()

    x = _to_float32_tensor(embeddings)
    y = _to_float32_tensor(target_vectors)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    net.eval()
    return net


def train_hypernet_with_history(
    embeddings: np.ndarray,
    target_vectors: np.ndarray,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    l2: float = 1e-4,
    hidden: int = 256,
    depth: int = 2,
    seed: int = 0,
    normalize_targets: bool = False,
) -> tuple[HyperNet, list[float]]:
    """Like :func:`train_hypernet` but also returns the per-epoch loss curve.

    Used to assert that optimisation actually happens (loss decreases). The
    returned net is identical (same seed/hyperparameters) to the one
    :func:`train_hypernet` would produce.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    target_vectors = np.asarray(target_vectors, dtype=np.float32)
    if embeddings.shape[0] != target_vectors.shape[0]:
        raise ValueError(
            f"row mismatch: {embeddings.shape[0]} embeddings vs "
            f"{target_vectors.shape[0]} targets"
        )
    if normalize_targets:
        norms = np.linalg.norm(target_vectors, axis=1, keepdims=True)
        target_vectors = target_vectors / (norms + 1e-8)

    torch.manual_seed(seed)
    net = HyperNet(
        embed_dim=embeddings.shape[1],
        vector_dim=target_vectors.shape[1],
        hidden=hidden,
        depth=depth,
    )
    net.train()

    x = _to_float32_tensor(embeddings)
    y = _to_float32_tensor(target_vectors)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.MSELoss()

    history: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    net.eval()
    return net, history


def predict_vector(net: HyperNet, embedding: np.ndarray) -> np.ndarray:
    """Predict a steering vector for ONE description embedding.

    Parameters
    ----------
    net       : a trained HyperNet.
    embedding : ``[embed_dim]`` (1-D) or ``[1, embed_dim]`` description embedding.

    Returns
    -------
    np.ndarray ``[vector_dim]`` float32 — the predicted steering vector.
    """
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    net.eval()
    with torch.no_grad():
        out = net(_to_float32_tensor(arr))
    return out[0].cpu().numpy().astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity between two vectors (same convention as extract.py)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def evaluate_holdout(
    net: HyperNet,
    holdout_embeddings: np.ndarray,
    holdout_targets: np.ndarray,
) -> dict[str, object]:
    """Secondary geometric check (spec §7): cosine(predicted, supervised).

    For each held-out behavior, predict a vector from its description embedding
    and measure cosine similarity to the supervised DiffMean target. A high mean
    cosine indicates the hypernetwork learned a *generalizing* direction map (not
    just memorised the training behaviors).

    Parameters
    ----------
    net                : trained HyperNet.
    holdout_embeddings : ``[m, embed_dim]`` held-out description embeddings.
    holdout_targets    : ``[m, vector_dim]`` held-out supervised vectors.

    Returns
    -------
    dict with ``"per_behavior"`` (list[float], one cosine per held-out row) and
    ``"mean_cosine"`` (float, their mean).
    """
    holdout_embeddings = np.asarray(holdout_embeddings, dtype=np.float32)
    holdout_targets = np.asarray(holdout_targets, dtype=np.float32)
    per_behavior: list[float] = []
    for i in range(holdout_embeddings.shape[0]):
        pred = predict_vector(net, holdout_embeddings[i])
        per_behavior.append(cosine(pred, holdout_targets[i]))
    mean_cosine = float(np.mean(per_behavior)) if per_behavior else 0.0
    return {"per_behavior": per_behavior, "mean_cosine": mean_cosine}
