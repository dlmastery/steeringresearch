"""extract.py — build steering vectors from contrast pairs.

Given contrast pairs [(pos_prompt, neg_prompt), ...] (AXIS 7, the SOURCE), we:

  1. cache per-layer activations (mean over answer tokens) to disk and reuse;
  2. compute the DiffMean vector per layer (Step 1 KEY INSIGHT 2:
     mean(pos) - mean(neg) cancels noise, leaves the concept direction);
  3. compute the PCA-top1 direction of the (pos - neg) difference set per layer;
  4. report cosine(diffmean, pca-top1) per layer;
  5. report the Fisher ratio per layer (E2: between-class / within-class
     separation) to locate the most separable layer;
  6. save / load a vector bank to .npz.

Everything is offline-capable: it consumes a model (FakeResidualLM in tests),
a tokenizer, and a list of contrast pairs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .hooks import probe_activations
from .model import get_residual_layers


def _encode(tokenizer, text: str) -> torch.Tensor:
    """Encode a prompt to an input_ids tensor [1, seq]."""
    out = tokenizer(text, return_tensors="pt")
    if isinstance(out, dict):
        return out["input_ids"]
    return out


def _mean_over_answer_tokens(acts: torch.Tensor) -> torch.Tensor:
    """Mean over the answer tokens (all non-padding positions) -> [dim].

    acts : [1, seq, dim]. We mean over the sequence axis. (For the FakeLM
    tokenizer there is no padding within a single prompt.)
    """
    return acts[0].mean(dim=0)


def collect_activations(
    model: nn.Module,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    layers: Optional[Sequence[int]] = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Collect mean-pooled activations for positive and negative prompts.

    Returns {layer: {"pos": [n, dim], "neg": [n, dim]}} as numpy arrays.
    """
    n_layers = len(get_residual_layers(model))
    if layers is None:
        layers = list(range(n_layers))
    layers = list(layers)

    pos_by_layer: dict[int, list] = {li: [] for li in layers}
    neg_by_layer: dict[int, list] = {li: [] for li in layers}

    for pos_text, neg_text in pairs:
        pos_ids = _encode(tokenizer, pos_text)
        neg_ids = _encode(tokenizer, neg_text)
        pos_acts = probe_activations(model, pos_ids, layers)
        neg_acts = probe_activations(model, neg_ids, layers)
        for li in layers:
            pos_by_layer[li].append(_mean_over_answer_tokens(pos_acts[li]).numpy())
            neg_by_layer[li].append(_mean_over_answer_tokens(neg_acts[li]).numpy())

    return {
        li: {
            "pos": np.stack(pos_by_layer[li], axis=0),
            "neg": np.stack(neg_by_layer[li], axis=0),
        }
        for li in layers
    }


def _pairs_signature(pairs: Sequence[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for p, n in pairs:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
        h.update(n.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


def collect_activations_cached(
    model: nn.Module,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    cache_dir: str | Path,
    layers: Optional[Sequence[int]] = None,
    model_tag: str = "fake",
) -> dict[int, dict[str, np.ndarray]]:
    """Cache activations to disk (CLAUDE.md §2: cache once, reuse across ladder).

    The cache key is derived from the model tag + the contrast-pair content, so
    identical inputs hit the cache.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = _pairs_signature(pairs)
    layer_tag = "all" if layers is None else "-".join(map(str, layers))
    cache_path = cache_dir / f"acts_{model_tag}_{sig}_{layer_tag}.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        out: dict[int, dict[str, np.ndarray]] = {}
        meta = json.loads(str(data["__layers__"]))
        for li in meta:
            out[int(li)] = {"pos": data[f"{li}_pos"], "neg": data[f"{li}_neg"]}
        return out

    acts = collect_activations(model, tokenizer, pairs, layers)
    save_blob: dict[str, np.ndarray] = {}
    for li, d in acts.items():
        save_blob[f"{li}_pos"] = d["pos"]
        save_blob[f"{li}_neg"] = d["neg"]
    save_blob["__layers__"] = np.array(json.dumps(list(acts.keys())))
    np.savez(cache_path, **save_blob)
    return acts


def diffmean_vector(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """DiffMean = mean(pos) - mean(neg). [dim]."""
    return pos.mean(axis=0) - neg.mean(axis=0)


def pca_top1_vector(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Top-1 PCA direction of the per-pair difference set (pos_i - neg_i).

    We take PCA of the UNCENTERED differences. The shared concept direction is
    the dominant (largest-energy) axis of {pos_i - neg_i}; centering would
    subtract exactly that shared component and leave only noise, so we keep the
    raw second-moment matrix (Σ d_i d_iᵀ) whose top eigenvector is the concept
    direction. Sign-aligned to the diffmean so cosine comparison is meaningful.
    """
    diffs = pos - neg  # [n, dim]
    # Top right-singular vector of the uncentered difference matrix = leading
    # axis of Σ d d^T = the shared concept direction.
    _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    pc1 = vt[0]
    # sign-align to the raw diffmean direction
    dm = diffmean_vector(pos, neg)
    if float(np.dot(pc1, dm)) < 0:
        pc1 = -pc1
    return pc1


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def fisher_ratio(pos: np.ndarray, neg: np.ndarray, eps: float = 1e-8) -> float:
    """Fisher discriminant ratio for a layer (E2).

    Projects onto the diffmean axis and computes
        (mean_pos - mean_neg)² / (var_pos + var_neg)
    along that axis — the 1-D Fisher ratio. Higher ⇒ more separable layer.
    """
    w = diffmean_vector(pos, neg)
    w = w / (np.linalg.norm(w) + eps)
    pp = pos @ w
    pn = neg @ w
    between = (pp.mean() - pn.mean()) ** 2
    within = pp.var() + pn.var() + eps
    return float(between / within)


def build_vector_bank(
    acts: dict[int, dict[str, np.ndarray]],
) -> dict[int, dict[str, object]]:
    """Compute per-layer DiffMean, PCA-top1, cosine, Fisher ratio.

    Returns {layer: {"diffmean", "pca", "cosine_dm_pca", "fisher"}}.
    """
    bank: dict[int, dict[str, object]] = {}
    for li, d in acts.items():
        pos, neg = d["pos"], d["neg"]
        dm = diffmean_vector(pos, neg)
        pca = pca_top1_vector(pos, neg)
        bank[li] = {
            "diffmean": dm,
            "pca": pca,
            "cosine_dm_pca": cosine(dm, pca),
            "fisher": fisher_ratio(pos, neg),
        }
    return bank


def best_layer(bank: dict[int, dict[str, object]]) -> int:
    """Layer with the highest Fisher ratio (most separable)."""
    return max(bank, key=lambda li: bank[li]["fisher"])


def save_vector_bank(bank: dict[int, dict[str, object]], path: str | Path) -> None:
    """Save a vector bank to .npz (diffmean + pca per layer, plus scalars)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, np.ndarray] = {}
    meta = {}
    for li, d in bank.items():
        blob[f"{li}_diffmean"] = np.asarray(d["diffmean"])
        blob[f"{li}_pca"] = np.asarray(d["pca"])
        meta[str(li)] = {
            "cosine_dm_pca": float(d["cosine_dm_pca"]),
            "fisher": float(d["fisher"]),
        }
    blob["__meta__"] = np.array(json.dumps(meta))
    np.savez(path, **blob)


def load_vector_bank(path: str | Path) -> dict[int, dict[str, object]]:
    """Load a vector bank saved by save_vector_bank."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["__meta__"]))
    bank: dict[int, dict[str, object]] = {}
    for li_str, scalars in meta.items():
        li = int(li_str)
        bank[li] = {
            "diffmean": data[f"{li}_diffmean"],
            "pca": data[f"{li}_pca"],
            "cosine_dm_pca": scalars["cosine_dm_pca"],
            "fisher": scalars["fisher"],
        }
    return bank


def extract_bank(
    model: nn.Module,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    layers: Optional[Sequence[int]] = None,
    cache_dir: Optional[str | Path] = None,
    model_tag: str = "fake",
) -> dict[int, dict[str, object]]:
    """End-to-end: collect (optionally cached) activations and build the bank."""
    if cache_dir is not None:
        acts = collect_activations_cached(
            model, tokenizer, pairs, cache_dir, layers, model_tag
        )
    else:
        acts = collect_activations(model, tokenizer, pairs, layers)
    return build_vector_bank(acts)
