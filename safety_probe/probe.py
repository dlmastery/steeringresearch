"""probe.py — the classifier: a 3-layer MLP on top of frozen activations.

The probe is tiny. Its whole job is to draw a decision surface through the
activation vectors so that harmful prompts land on one side and safe prompts on
the other. Because the LLM has already done the hard representational work, a
small network with heavy regularization is plenty.

Architecture (3 linear layers):
    hidden(1152) -> 128 -> 32 -> 1 logit    (ReLU + dropout between layers)

We also keep a StandardScaler (per-feature mean/std) INSIDE the checkpoint so
inference reproduces training exactly with no external state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class MLPProbe(nn.Module):
    """A 3-linear-layer MLP mapping an activation vector to one harmful-logit."""

    def __init__(self, in_dim: int, h1: int = 128, h2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.h1 = h1
        self.h2 = h2
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [batch] logits


@dataclass
class Scaler:
    """Per-feature standardization: (x - mean) / std. Stored with the probe."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        std = x.std(axis=0)
        std[std < 1e-6] = 1e-6  # guard constant features
        return cls(mean=x.mean(axis=0).astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def save_probe(path, probe: MLPProbe, scaler: Scaler, meta: dict) -> None:
    """Serialize the probe weights + scaler + metadata to one file."""
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "in_dim": probe.in_dim,
            "h1": probe.h1,
            "h2": probe.h2,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "meta": meta,  # model_id, layer, pooling, threshold, ...
        },
        path,
    )


def load_probe(path, device: str = "cpu") -> tuple[MLPProbe, Scaler, dict]:
    """Inverse of :func:`save_probe`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    probe = MLPProbe(ckpt["in_dim"], ckpt["h1"], ckpt["h2"])
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval().to(device)
    scaler = Scaler(mean=ckpt["scaler_mean"], std=ckpt["scaler_std"])
    return probe, scaler, ckpt["meta"]


@torch.no_grad()
def predict_proba(probe: MLPProbe, scaler: Scaler, features: np.ndarray,
                  device: str = "cpu") -> np.ndarray:
    """Return P(harmful) in [0,1] for each row of ``features`` (raw activations)."""
    x = torch.from_numpy(scaler.transform(features)).to(device)
    logits = probe(x)
    return torch.sigmoid(logits).cpu().numpy()
