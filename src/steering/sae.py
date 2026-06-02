"""sae.py — Sparse Autoencoder + SAE-TS steering-vector optimizer (E20).

This module implements AXIS feature-space steering: instead of steering in raw
activation space (where DiffMean lives, `extract.diffmean_vector`), we decompose
activations into an interpretable, sparse, overcomplete feature basis (a Sparse
Autoencoder) and then OPTIMIZE a steering vector to drive a chosen set of target
features while suppressing every other ("side-effect") feature.

The E20 hypothesis
------------------
`hypotheses/C_stacking/E20_sae_ts_vs_diffmean_3stack.md`. SAE-TS
(arXiv:2411.02193) targets only the SAE features causally responsible for a
behavior and minimizes side-effect features. The mechanistic claim:

    cos(v_B1_SAETS, v_B2_SAETS)  <<  cos(v_B1_DiffMean, v_B2_DiffMean)

i.e. SAE-TS vectors for distinct behaviors are MORE ORTHOGONAL than DiffMean
vectors, so a 3-vector stack has lower off-diagonal Gram mass (§5.2,
`gram_mass`), lower interference (E18 monotone curve), and better joint
coherence. The falsifier is a 3-stack coherence gap < 0.10.

Contrast with DiffMean (the baseline this competes against)
-----------------------------------------------------------
`extract.diffmean_vector` is CLOSED-FORM: v = mean(pos) - mean(neg), one matmul,
no optimization, no notion of "side effects". `sae_ts_vector` (below) is the
OPPOSITE: the steering vector ITSELF is a trainable parameter optimized by
gradient ASCENT on the SAE feature-activation objective

    v* = argmax_v [ score(F_target)  -  lambda * score(F_side) ]

where score(F) is the mean ReLU SAE-encoder activation of the feature set F when
v is fed through the encoder. THIS module is the experiment where the steering
vector is learned rather than averaged.

GemmaScope production swap
--------------------------
The E20 design doc references the pretrained GemmaScope SAE
(``google/gemma-scope-2b-pt-res``). That is a large, gated HuggingFace download
and is NOT available offline, so the test suite cannot depend on it. We therefore
ship a SELF-CONTAINED small SAE (`SparseAutoencoder`) trained on-the-fly on
cached activations (`train_sae`) — the faithful, 4090-budget, hermetic version.

GemmaScope is a drop-in replacement: it exposes the SAME interface this module
relies on — ``encode(activations) -> sparse_features`` and
``decode(features) -> activations``. To use it in production, load the
GemmaScope SAE for the matching layer, wrap it so it exposes ``.encode`` /
``.decode`` returning float32 tensors, and pass it to `sae_ts_vector` instead of
a locally trained `SparseAutoencoder`. Every downstream function
(`feature_activation`, `sae_ts_vector`, `gram_mass`) is agnostic to which SAE
produced the codes.

Everything here is offline-capable, float32, and deterministic via an explicit
seed. No network, no Gemma, no GemmaScope required for the test suite.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

# Single source of truth for the unit-norm / cosine numerical guard (matches
# hooks._EPS / extract.cosine conventions).
_EPS = 1e-8


def _as_float32_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Coerce an array/tensor to a contiguous float32 CPU-or-same-device tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


class SparseAutoencoder(nn.Module):
    """A small overcomplete Sparse Autoencoder (the offline GemmaScope stand-in).

    Architecture (the standard SAE shape):

        z = ReLU(W_enc (x - b_pre) + b_enc)     encode : dim -> n_features (sparse)
        x_hat = W_dec z + b_pre                  decode : n_features -> dim

    ``z`` are the SPARSE CODES (one non-negative scalar per learned feature);
    ``x_hat`` is the reconstruction. With ``tied=True`` the decoder weight is the
    transpose of the encoder weight (``W_dec = W_enc^T``), the classic tied-weight
    autoencoder; with ``tied=False`` encoder and decoder are independent linear
    maps. ``b_pre`` is the pre-encoder bias (the learned activation centroid that
    is subtracted before encoding and added back after decoding) — standard in
    SAE training (Anthropic / GemmaScope) so the codes model deviations from the
    mean activation rather than the raw activation.

    Parameters
    ----------
    dim        : residual-stream width (d_model).
    n_features : SAE dictionary size (overcomplete ⇒ n_features > dim).
    tied       : if True, decoder weight is the encoder weight's transpose.
    seed       : determinism seed for weight init.
    """

    def __init__(
        self,
        dim: int,
        n_features: int,
        *,
        tied: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.n_features = n_features
        self.tied = tied

        g = torch.Generator().manual_seed(seed)
        # Encoder: dim -> n_features. Decoder: n_features -> dim.
        self.encoder = nn.Linear(dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, dim, bias=False)
        # Pre-encoder bias (subtract before encode, add back after decode).
        self.b_pre = nn.Parameter(torch.zeros(dim))

        # Deterministic init from the seeded generator (no global RNG touch).
        enc_w = torch.randn(n_features, dim, generator=g) * (1.0 / dim**0.5)
        self.encoder.weight.data = enc_w.clone()
        self.encoder.bias.data = torch.zeros(n_features)
        if tied:
            # Tied decoder shares the encoder weight's transpose; the Linear's own
            # weight is kept as a buffer-like clone so .decode stays a pure matmul.
            self.decoder.weight.data = enc_w.t().clone()
        else:
            dec_w = torch.randn(dim, n_features, generator=g) * (1.0 / n_features**0.5)
            self.decoder.weight.data = dec_w

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x : [..., dim] -> sparse non-negative codes z : [..., n_features]."""
        x = x.to(dtype=torch.float32)
        return torch.relu(self.encoder(x - self.b_pre))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z : [..., n_features] -> reconstruction x_hat : [..., dim]."""
        w = self.encoder.weight.t() if self.tied else self.decoder.weight
        return torch.nn.functional.linear(z, w) + self.b_pre

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full pass: decode(encode(x)) — the reconstruction x_hat."""
        return self.decode(self.encode(x))


def train_sae(
    activations: np.ndarray | torch.Tensor,
    *,
    n_features: int,
    l1: float = 1e-3,
    epochs: int = 200,
    lr: float = 1e-3,
    tied: bool = False,
    seed: int = 0,
) -> SparseAutoencoder:
    """Train a `SparseAutoencoder` on cached activations (the SAE training infra).

    Adam minimizes reconstruction MSE + an L1 sparsity penalty on the codes:

        loss = mean ‖x - x_hat‖²  +  l1 * mean Σ_features |z|

    The L1 term is what makes the learned codes SPARSE (few active features per
    input), which is the whole point of an SAE — interpretable, mostly-disjoint
    feature directions. ``b_pre`` is initialized to the activation mean so the SAE
    immediately models deviations from the centroid.

    Parameters
    ----------
    activations : [n_samples, dim] cached residual activations to fit.
    n_features  : SAE dictionary size (overcomplete ⇒ > dim).
    l1          : sparsity penalty weight on the codes.
    epochs      : full-batch Adam steps.
    lr          : Adam learning rate.
    tied        : tied vs untied decoder (see `SparseAutoencoder`).
    seed        : determinism seed (weight init + Adam are fully reproducible).

    Returns
    -------
    The trained `SparseAutoencoder` in eval mode.
    """
    torch.manual_seed(seed)
    x = _as_float32_tensor(activations)
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    dim = x.shape[-1]

    sae = SparseAutoencoder(dim, n_features, tied=tied, seed=seed)
    # Initialize the pre-encoder bias at the data centroid (standard SAE practice).
    with torch.no_grad():
        sae.b_pre.data = x.mean(dim=0).clone()

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    sae.train()
    for _ in range(epochs):
        opt.zero_grad()
        z = sae.encode(x)
        x_hat = sae.decode(z)
        recon = torch.mean((x - x_hat) ** 2)
        sparsity = torch.mean(torch.sum(torch.abs(z), dim=-1))
        loss = recon + l1 * sparsity
        loss.backward()
        opt.step()

    sae.eval()
    return sae


def feature_activation(
    sae: SparseAutoencoder,
    vector: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Per-feature SAE activation of a SINGLE steering direction -> [n_features].

    Encodes ``vector`` through the SAE encoder (ReLU codes) and returns the
    resulting non-negative per-feature activation as numpy float32. Used by
    `sae_ts_vector` (as the differentiable objective) and by tests to assert that
    an optimized vector lights up its target features.
    """
    v = _as_float32_tensor(vector)
    with torch.no_grad():
        z = sae.encode(v)
    return z.detach().cpu().numpy().astype(np.float32)


def sae_ts_vector(
    sae: SparseAutoencoder,
    target_feature_ids: Sequence[int],
    *,
    lam: float = 1.0,
    steps: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """OPTIMIZE a unit-norm steering vector by gradient ascent (the SAE-TS core).

    This is the experiment where the steering VECTOR ITSELF is the trainable
    parameter — the explicit contrast with closed-form DiffMean
    (`extract.diffmean_vector`, which is just mean(pos) - mean(neg)). Here ``v``
    starts random and is updated by Adam to maximize the SAE-TS objective (§5.1 of
    the E20 design doc):

        v* = argmax_v [ score(F_target)  -  lambda * score(F_side) ]

    with

        score(F_target) = mean_{i in F_target}   ReLU(encode(v))_i
        score(F_side)   = mean_{i not in F_target} ReLU(encode(v))_i

    i.e. drive the chosen target SAE features HIGH while suppressing every other
    ("side-effect") feature. The side-effect suppression is exactly what makes two
    SAE-TS vectors for disjoint target sets MORE ORTHOGONAL than two DiffMean
    vectors that may share activation dimensions — the E20 mechanism that lowers
    `gram_mass` and improves 3-stack coherence.

    The vector is re-normalized to unit norm after every step (projection onto the
    unit sphere) so the objective optimizes DIRECTION, not magnitude — the steering
    scale alpha is applied separately at injection time (`hooks.apply_operation`).

    Parameters
    ----------
    sae                : a trained `SparseAutoencoder` (or GemmaScope drop-in).
    target_feature_ids : indices of the SAE features to drive (F_target).
    lam                : side-effect penalty weight (lambda); higher ⇒ more
                         aggressive suppression of non-target features.
    steps              : gradient-ascent steps.
    lr                 : Adam learning rate on the vector parameter.
    seed               : determinism seed for the random init.
    device             : device to optimize on.

    Returns
    -------
    The optimized unit-norm steering vector as numpy float32, shape [dim].
    """
    device = torch.device(device)
    dim = sae.dim
    n_features = sae.n_features
    sae = sae.to(device)
    for p in sae.parameters():
        p.requires_grad_(False)

    target_ids = sorted(set(int(i) for i in target_feature_ids))
    if not target_ids:
        raise ValueError("target_feature_ids must be non-empty")
    if min(target_ids) < 0 or max(target_ids) >= n_features:
        raise ValueError(
            f"target_feature_ids out of range [0, {n_features}): {target_ids}"
        )

    target_mask = torch.zeros(n_features, dtype=torch.bool, device=device)
    target_mask[target_ids] = True
    side_mask = ~target_mask

    g = torch.Generator(device="cpu").manual_seed(seed)
    v0 = torch.randn(dim, generator=g).to(device)
    v0 = v0 / (v0.norm() + _EPS)
    v = nn.Parameter(v0.clone())

    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        z = sae.encode(v)  # [n_features], ReLU codes (differentiable in v)
        target_score = z[target_mask].mean()
        # mean over side features; guard the (degenerate) all-target case.
        side_score = z[side_mask].mean() if bool(side_mask.any()) else z.sum() * 0.0
        # Gradient ASCENT on (target - lam*side) == DESCENT on its negation.
        loss = -(target_score - lam * side_score)
        loss.backward()
        opt.step()
        with torch.no_grad():
            v.data = v.data / (v.data.norm() + _EPS)

    with torch.no_grad():
        out = (v.detach() / (v.detach().norm() + _EPS)).cpu().numpy().astype(np.float32)
    return out


def pairwise_cosines(vectors: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """All i<j pairwise cosine similarities of a vector list -> [n*(n-1)/2].

    Returns the flat upper-triangle (excluding the diagonal) of the cosine matrix,
    in row-major (i, j) order. Matches `extract.cosine` conventions (eps-guarded).
    """
    mat = np.asarray([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
    if mat.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1) + _EPS
    unit = mat / norms[:, None]
    cos = unit @ unit.T
    iu = np.triu_indices(mat.shape[0], k=1)
    return cos[iu].astype(np.float64)


def gram_mass(vectors: Sequence[np.ndarray] | np.ndarray) -> float:
    """Off-diagonal Gram mass M = Σ_{i<j} |cos(v_i, v_j)|  (E18 / E20 §5.2).

    The total absolute off-diagonal cosine of a vector set — the scalar that E18's
    monotone curve maps to behavioral interference. Orthogonal set ⇒ 0; an
    n-vector set of identical (or anti-parallel) directions ⇒ n*(n-1)/2 (every
    pair has |cos| = 1). Lower M ⇒ more orthogonal stack ⇒ (per E18) lower
    interference and better joint coherence; the E20 claim is M_SAETS < M_DM.
    """
    cosines = pairwise_cosines(vectors)
    if cosines.size == 0:
        return 0.0
    return float(np.abs(cosines).sum())
