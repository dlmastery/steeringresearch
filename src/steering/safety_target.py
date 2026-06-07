"""safety_target.py — extract the safety-steering direction (the "WHAT").

This is the target half of the conditional safety-steering method: the direction
in the residual stream that, when added, pushes the model toward a *safe*
completion. It is AXIS 7 (the SOURCE) specialised to safety: a DiffMean contrast
between harmful-prompt activations and harmless-prompt activations, mean-pooled
over the sequence and unit-normalised so the raw norm is decoupled from the
steering coefficient (alpha) downstream (cf. ``hooks.relative_add``).

Two targets are provided:

  * ``extract_refusal_direction`` — harmful − harmless. Steering ALONG this
    direction pushes activations toward the region the model occupies when it is
    about to refuse / decline a harmful request. This is the blunt "refuse"
    target.
  * ``extract_safe_completion_direction`` — safe-completion − refusal. A gentler
    target that steers toward a *helpful but safe* completion (e.g. a harm-
    reduction answer) rather than a hard refusal, for the over-refusal-controlled
    regime. Same DiffMean machinery, different contrast set.

The math mirrors ``extract.diffmean_vector`` (mean(pos) − mean(neg) cancels
shared noise and leaves the concept axis) but consumes two *separate* text lists
rather than aligned pairs, because harmful/harmless safety corpora are rarely
paired one-to-one.

Everything is offline-capable: it consumes a model (``FakeResidualLM`` in tests)
and a tokenizer, reads activations through ``hooks.probe_activations``, and
returns numpy float32 (the numpy↔torch seam — see DESIGN.md §3).

STATUS: BUILT but NOT YET VALIDATED on real models / benchmarks. The unit tests
exercise the sign/projection contract on ``FakeResidualLM``; no real refusal
direction has been measured on Gemma yet.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch.nn as nn

from .hooks import probe_activations
from .model import encode_to_device

_EPS = 1e-8


def _pooled_reps(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    layer: int,
) -> np.ndarray:
    """Mean-pooled residual activations at ``layer`` for each text -> ``[n, dim]``.

    One forward pass per text; the activation at ``layer`` is mean-pooled over the
    sequence axis (the same pooling ``gate.condition_features`` and
    ``extract`` use). Returned as numpy float32 (real models are bf16 on CUDA;
    numpy supports neither, hence ``.float().cpu()``).
    """
    if len(texts) == 0:
        raise ValueError("need at least one text to pool activations from")
    reps: list[np.ndarray] = []
    for text in texts:
        input_ids = encode_to_device(tokenizer, text, model)
        acts = probe_activations(model, input_ids, [layer])
        pooled = acts[layer][0].mean(dim=0).float().cpu().numpy()
        reps.append(pooled.astype(np.float32))
    return np.stack(reps, axis=0)


def _unit(v: np.ndarray) -> np.ndarray:
    """Unit-normalise a vector (float32), guarding the zero vector."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return (v / (float(np.linalg.norm(v)) + _EPS)).astype(np.float32)


def extract_refusal_direction(
    model: nn.Module,
    tok,
    harmful_texts: Sequence[str],
    harmless_texts: Sequence[str],
    layer: int,
) -> np.ndarray:
    """Unit DiffMean direction harmful − harmless at ``layer`` (the safety target).

    Steering a residual stream ALONG this unit vector moves it toward the region
    occupied by harmful prompts — i.e. the region where a safety-tuned model
    forms a refusal. Callers therefore steer with a POSITIVE alpha to push toward
    refusal/safe-completion (and could project it out to ablate the refusal
    direction — the Rogue-Scalpel guard's AXIS A).

    Parameters
    ----------
    model         : ``FakeResidualLM`` or real Gemma.
    tok           : matching tokenizer (real HF or offline ``_FakeTokenizer``).
    harmful_texts : prompts that should elicit refusal/safe-completion.
    harmless_texts: benign prompts.
    layer         : residual layer index to read.

    Returns
    -------
    np.ndarray ``[dim]`` float32, unit-normalised. ``np.dot(h, dir)`` is then a
    signed "how harmful does this activation look" projection.
    """
    harmful = _pooled_reps(model, tok, harmful_texts, layer)
    harmless = _pooled_reps(model, tok, harmless_texts, layer)
    diffmean = harmful.mean(axis=0) - harmless.mean(axis=0)
    return _unit(diffmean)


def extract_safe_completion_direction(
    model: nn.Module,
    tok,
    safe_texts: Sequence[str],
    refusal_texts: Sequence[str],
    layer: int,
) -> np.ndarray:
    """Unit DiffMean direction safe-completion − refusal at ``layer`` (gentler target).

    The over-refusal-controlled variant of the safety target. Where
    ``extract_refusal_direction`` steers toward a blunt refusal, this steers
    toward a *helpful but safe* completion: ``safe_texts`` are harm-reduced,
    on-topic answers; ``refusal_texts`` are flat declines. Steering with a
    positive alpha then nudges the model away from over-refusal and toward a
    useful-yet-safe response, which is the §10 selectivity / over-refusal axis.

    Same machinery as ``extract_refusal_direction`` (unit DiffMean), different
    contrast set. Returns ``[dim]`` float32 unit vector.
    """
    safe = _pooled_reps(model, tok, safe_texts, layer)
    refusal = _pooled_reps(model, tok, refusal_texts, layer)
    diffmean = safe.mean(axis=0) - refusal.mean(axis=0)
    return _unit(diffmean)
