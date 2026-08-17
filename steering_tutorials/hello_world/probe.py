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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ProbeDimensionMismatch(RuntimeError):
    """A probe is being fed activations of a width it was not trained on.

    This is ALWAYS a model mix-up, never something to paper over: a Gemma-3-1B
    activation is 1152 numbers wide and a Gemma-3-4B activation is 2560, so a
    probe trained on one cannot score the other. There is no truncation,
    padding, or projection that would make the answer meaningful — the only
    correct responses are to load the probe that matches the model, or to train
    one at this width.
    """


def _fix_hint(probe_dim: int, feature_dim: int, meta: dict | None) -> str:
    """The 'what do I do about it' half of a mismatch message (ASCII only)."""
    trained_on = (meta or {}).get("model_id", "unknown model")
    return (
        f"The probe was trained on {trained_on} ({probe_dim}-d activations) and "
        f"is being given {feature_dim}-d activations from a different model.\n"
        f"Fix it one of two ways:\n"
        f"  1. Load the probe that matches the model you are running. Artifacts "
        f"are model-tagged: probe.pt is 1B/1152-d, probe_4b.pt is 4B/2560-d.\n"
        f"  2. Train a probe at this width first:\n"
        f"       set STEER_MODEL_ID=<the model id>  (and STEER_LOAD_4BIT=1 for 4B)\n"
        f"       python -m steering_tutorials.hello_world.train_probe\n"
        f"     which writes a separately-named probe/features/metrics triple.\n"
        f"Do NOT truncate, pad, or project the activations to make the shapes "
        f"agree - the resulting score would be meaningless."
    )


def check_feature_width(probe_dim: int, feature_dim: int, meta: dict | None = None,
                        where: str = "probe") -> None:
    """Raise :class:`ProbeDimensionMismatch` unless the two widths agree."""
    if probe_dim != feature_dim:
        raise ProbeDimensionMismatch(
            f"[{where}] activation width mismatch: expected {probe_dim}, got "
            f"{feature_dim}.\n" + _fix_hint(probe_dim, feature_dim, meta)
        )


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
        # Set by load_probe so a mismatch message can name the model this probe
        # was trained on. Empty for a freshly-constructed (untrained) probe.
        self.meta: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch's own matmul error ("mat1 and mat2 shapes cannot be multiplied")
        # does not say WHICH model each width came from; this one does.
        check_feature_width(self.in_dim, int(x.shape[-1]), self.meta, where="MLPProbe")
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
        # numpy would broadcast-error here anyway, but with a message that names
        # neither the probe nor the model. Say what actually went wrong.
        check_feature_width(int(self.mean.shape[-1]), int(np.asarray(x).shape[-1]),
                            where="Scaler")
        return ((x - self.mean) / self.std).astype(np.float32)


def save_probe(path, probe: MLPProbe, scaler: Scaler, meta: dict,
               allow_model_overwrite: bool = False) -> None:
    """Serialize the probe weights + scaler + metadata to one file.

    Refuses to overwrite a checkpoint that was trained on a DIFFERENT model
    unless ``allow_model_overwrite``: silently replacing probe.pt with a 4B
    probe is what turns a config slip into a mystery crash in lesson 2.
    """
    path = Path(path)
    meta = dict(meta)
    meta.setdefault("in_dim", int(probe.in_dim))
    meta.setdefault("hidden_dim", int(probe.in_dim))

    if path.exists() and not allow_model_overwrite:
        try:
            old = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # pragma: no cover - unreadable/foreign file
            old = None
        if isinstance(old, dict):
            old_model = (old.get("meta") or {}).get("model_id")
            new_model = meta.get("model_id")
            if old_model and new_model and old_model != new_model:
                raise ProbeDimensionMismatch(
                    f"[save_probe] refusing to overwrite {path.name}: it holds a "
                    f"{old.get('in_dim')}-d probe trained on {old_model}, and this "
                    f"run trained a {probe.in_dim}-d probe on {new_model}.\n"
                    f"Artifacts are model-tagged for exactly this reason - unset "
                    f"STEER_PROBE_NAME so the tagged default is used, or pass an "
                    f"explicit name that is not already taken."
                )

    torch.save(
        {
            "state_dict": probe.state_dict(),
            "in_dim": probe.in_dim,
            "h1": probe.h1,
            "h2": probe.h2,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "meta": meta,  # model_id, layer, pooling, threshold, in_dim, ...
        },
        path,
    )


def load_probe(path, device: str = "cpu", expect_dim: int | None = None,
               expect_model_id: str | None = None) -> tuple[MLPProbe, Scaler, dict]:
    """Inverse of :func:`save_probe`.

    ``expect_dim`` / ``expect_model_id`` let a caller that already knows which
    model it is running assert the match UP FRONT, instead of discovering it as
    a shape error inside a matmul several frames later.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = dict(ckpt["meta"])
    meta.setdefault("in_dim", int(ckpt["in_dim"]))

    if expect_model_id is not None and meta.get("model_id") not in (None, expect_model_id):
        raise ProbeDimensionMismatch(
            f"[load_probe] {Path(path).name} was trained on {meta['model_id']} "
            f"({ckpt['in_dim']}-d) but is being loaded for {expect_model_id}.\n"
            + _fix_hint(int(ckpt["in_dim"]), int(expect_dim or ckpt["in_dim"]), meta)
        )
    if expect_dim is not None:
        check_feature_width(int(ckpt["in_dim"]), int(expect_dim), meta, where="load_probe")

    probe = MLPProbe(ckpt["in_dim"], ckpt["h1"], ckpt["h2"])
    probe.load_state_dict(ckpt["state_dict"])
    probe.meta = meta
    probe.eval().to(device)
    scaler = Scaler(mean=ckpt["scaler_mean"], std=ckpt["scaler_std"])
    return probe, scaler, meta


@torch.no_grad()
def predict_proba(probe: MLPProbe, scaler: Scaler, features: np.ndarray,
                  device: str = "cpu") -> np.ndarray:
    """Return P(harmful) in [0,1] for each row of ``features`` (raw activations)."""
    features = np.asarray(features)
    check_feature_width(int(probe.in_dim), int(features.shape[-1]),
                        getattr(probe, "meta", None), where="predict_proba")
    x = torch.from_numpy(scaler.transform(features)).to(device)
    logits = probe(x)
    return torch.sigmoid(logits).cpu().numpy()
