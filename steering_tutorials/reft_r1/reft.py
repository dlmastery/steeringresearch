"""reft.py — the learned rank-1 LoReFT intervention (lesson 3 core).

Lesson 2 added ONE fixed direction ``v = mean(harmful) - mean(benign)`` to the
residual stream. Lesson 3, following AxBench, learns a **rank-1 representation
finetune**: instead of a constant vector, we learn a direction ``r`` and an
affine readout ``(w, b)``, and REPLACE that direction's component of the residual
with the learned affine function of the hidden state ``h``::

    r_unit = r / ||r||                                   # unit direction, [hidden]
    h' = h + r_unit * ( (w·h + b) - (r_unit·h) )

Read the edit as: project ``h`` onto ``r_unit`` (that is ``r_unit·h``), then swap
that scalar for a learned affine readout ``w·h + b``. Everything off the ``r_unit``
axis is untouched — this is exactly LoReFT's low-rank subspace edit at rank 1,
``h' = h + Rᵀ(Wh + b − Rh)`` with the single row ``R = r_unit``. Three things fall
out that lesson 2's fixed vector cannot give:

  * the edit is *input-dependent* (it reads ``h`` through ``w``, not a constant),
  * it is *trained end-to-end* by gradient descent (``r``, ``w``, ``b`` all learn), and
  * the learned direction doubles as a *concept detector*: ``r_unit·h`` is a readout.

  Wu et al. 2025, 'AxBench: Steering LLMs? Even Simple Baselines Outperform
    Sparse Autoencoders' (arXiv:2501.17148) [UNVERIFIED] — the ReFT-r1 steering
    method (and the DiffMean baseline it is contrasted with).
  Wu et al. 2024, 'ReFT: Representation Finetuning for Language Models'
    (arXiv:2404.03592) [UNVERIFIED] — LoReFT, whose rank-1 case this is.

This module owns three moving parts:

  1. :class:`ReftR1` — the intervention itself (learnable ``r``, ``w``, ``b``).
  2. :func:`grad_reft_forward` — a TRAINING-time intervened forward pass that keeps
     autograd intact, so the loss can flow gradients back into ``r``, ``w``, ``b``.
  3. :class:`ReftContext` — an INFERENCE-time context manager that applies the same
     (now frozen) intervention during generation.

Contrast with lesson 2's ``SteeringContext``: that adds a fixed vector under
``torch.no_grad``. Training here needs the graph alive, so :func:`grad_reft_forward`
never detaches the intervention's parameters.

Standalone: reuses only the lesson-2 model plumbing; third-party deps are
``torch``, ``transformers`` (transitively, for the real model) and ``numpy``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Reuse lesson-2 plumbing verbatim — do NOT reimplement model access here.
from steering_tutorials.hello_world_steering.model_utils import (  # noqa: F401
    hidden_size,
    load_model,
    num_layers,
    residual_layers,
)


# ---------------------------------------------------------------------------
# 1. The rank-1 intervention
# ---------------------------------------------------------------------------
class ReftR1(nn.Module):
    """A learned rank-1 LoReFT edit of the residual stream: ``[..., hidden] -> [..., hidden]``.

    Parameters (all trainable)::

        r : [hidden]   the (un-normalised) intervention direction. Init small
                       random so ``r_unit = r/||r||`` is well-defined and the
                       direction can rotate freely during training.
        w : [hidden]   the linear part of the affine readout ``w·h + b``. Init
                       ZERO so training starts from a clean "project out the
                       r-component" edit and grows the readout deliberately.
        b : scalar     the bias of the affine readout. Init 0.

    The forward map is :meth:`intervention`. With the zero init, ``w·h + b == 0``
    so ``h' = h - r_unit·(r_unit·h)`` — it merely removes the ``r_unit`` component;
    as ``w`` and ``b`` learn, that component is REPLACED by the learned readout.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        # Small random direction: nonzero so r_unit is defined; small so the
        # initial edit is gentle. w=0, b=0 => readout starts at 0 (see docstring).
        self.r = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.w = nn.Parameter(torch.zeros(hidden_dim))
        self.b = nn.Parameter(torch.zeros(()))

    @property
    def r_unit(self) -> torch.Tensor:
        """The unit direction ``r / ||r||`` — grad-carrying (division is smooth)."""
        return self.r / self.r.norm().clamp_min(1e-8)

    def intervention(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the rank-1 edit to a residual ``h`` of shape ``[..., hidden]``.

        Broadcasts over any leading dims (``[hidden]``, ``[seq, hidden]``,
        ``[batch, seq, hidden]`` all work). Differentiable in ``r``, ``w``, ``b``.
        """
        r_unit = self.r_unit.to(dtype=h.dtype, device=h.device)     # [hidden]
        w = self.w.to(dtype=h.dtype, device=h.device)               # [hidden]
        b = self.b.to(dtype=h.dtype, device=h.device)               # scalar
        # Current component along r_unit, and the learned affine readout that
        # should replace it. Both reduce the last dim -> [..., 1] for broadcast.
        proj = (h * r_unit).sum(dim=-1, keepdim=True)               # r_unit·h
        readout = (h * w).sum(dim=-1, keepdim=True) + b             # w·h + b
        return h + r_unit * (readout - proj)


# ---------------------------------------------------------------------------
# 2. Training-time intervened forward (autograd stays alive)
# ---------------------------------------------------------------------------
def grad_reft_forward(
    model: Any, input_ids: torch.Tensor, reft: ReftR1, layer: int
) -> torch.Tensor:
    """One intervened forward pass that KEEPS the graph, returning logits.

    Registers a forward hook on ``residual_layers(model)[layer]`` that runs
    ``reft.intervention`` on the layer's output hidden state — with NO
    ``torch.no_grad`` and NO ``.detach()`` on the intervention's parameters — so
    ``loss.backward()`` on the returned logits flows all the way back into
    ``reft.r``, ``reft.w`` and ``reft.b``.

    The hook is ALWAYS removed in a ``finally`` so a training step leaves no trace
    on the model — exactly like lesson-2's ``SteeringContext.__exit__``.

    Parameters
    ----------
    model : the loaded causal LM.
    input_ids : ``[batch, seq]`` token ids for the (chat-templated) prompt+target.
    reft : the :class:`ReftR1` intervention whose params we want gradients into.
    layer : index into ``residual_layers(model)``.
    """
    layers = residual_layers(model)
    layer = max(0, min(layer, len(layers) - 1))
    target = layers[layer]

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output          # [batch, seq, hidden]
        h_new = reft.intervention(h)                   # grad-carrying, no detach
        if is_tuple:
            return (h_new, *output[1:])
        return h_new

    handle = target.register_forward_hook(hook)
    try:
        out = model(input_ids)
        return out.logits if hasattr(out, "logits") else out
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# 3. Inference-time intervention (frozen params, for generation)
# ---------------------------------------------------------------------------
class ReftContext:
    """Context manager that applies a (frozen) :class:`ReftR1` during generation.

    On ``__enter__`` it registers a forward hook on ``residual_layers[layer]`` that
    rewrites the layer's output via ``reft.intervention``; on ``__exit__`` it
    removes the hook, restoring the model exactly (the intervention leaves no
    trace). At inference the params are fixed, so callers may run this inside
    ``torch.no_grad`` — but the hook itself imposes nothing, mirroring lesson-2's
    ``SteeringContext`` interface.
    """

    def __init__(self, model: nn.Module, reft: ReftR1, layer: int) -> None:
        self.model = model
        self.reft = reft
        layers = residual_layers(model)
        self.layer = max(0, min(layer, len(layers) - 1))
        self._handles: list[Any] = []

    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h_new = self.reft.intervention(h)
        if is_tuple:
            return (h_new, *output[1:])
        return h_new

    def __enter__(self) -> "ReftContext":
        target = residual_layers(self.model)[self.layer]
        # prepend=True so any downstream probe hook observes the intervened
        # residual — matches lesson-2's SteeringContext.
        self._handles.append(target.register_forward_hook(self._hook, prepend=True))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return None


# ---------------------------------------------------------------------------
# 4. The learned direction as a concept detector
# ---------------------------------------------------------------------------
def detector_score(reft: ReftR1, h_vec: "np.ndarray | torch.Tensor") -> float:
    """AxBench's detection readout: the projection ``r_unit · h`` for one ``h``.

    The same learned direction that STEERS the model also DETECTS the concept:
    a high projection of a hidden state onto ``r_unit`` means the concept is
    present. Accepts a ``[hidden]`` numpy array or tensor; returns a python float.
    """
    if isinstance(h_vec, np.ndarray):
        h = torch.from_numpy(h_vec)
    else:
        h = h_vec
    h = h.detach().reshape(-1).float()
    r_unit = reft.r_unit.detach().reshape(-1).float().to(h.device)
    return float(torch.dot(r_unit, h).item())


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------
def save_reft(path: Any, reft: ReftR1, meta: dict) -> None:
    """Save the intervention's weights + a metadata dict."""
    torch.save(
        {
            "state_dict": reft.state_dict(),
            "hidden_dim": reft.hidden_dim,
            "meta": dict(meta),
        },
        path,
    )


def load_reft(path: Any, hidden_dim: int | None = None) -> tuple[ReftR1, dict]:
    """Rebuild a :class:`ReftR1` from ``path``. -> (reft, meta)."""
    # weights_only=False: our own local checkpoint; ``meta`` may carry numpy.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hd = hidden_dim if hidden_dim is not None else ckpt["hidden_dim"]
    reft = ReftR1(hd)
    reft.load_state_dict(ckpt["state_dict"])
    reft.eval()
    return reft, ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# CPU self-test — NO Gemma download. Verifies autograd flows loss -> r,w,b and
# that the intervention hook is removed afterwards.
# Run: python -m steering_tutorials.reft_r1.reft
# ---------------------------------------------------------------------------
def _self_test() -> None:
    from types import SimpleNamespace

    torch.manual_seed(0)
    hidden, vocab, n_layers = 16, 32, 3

    # A tiny stand-in LM: residual_layers(model) resolves to model.model.layers,
    # exactly the path used for real Gemma. The lm_head makes the hooked layer's
    # output feed the logits, so a gradient can travel back through the hook.
    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(hidden, hidden) for _ in range(n_layers)]
            )

        def forward(self, x):
            for blk in self.layers:
                x = blk(x)
            return x

    class _TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _Inner()
            self.lm_head = nn.Linear(hidden, vocab)
            self.config = SimpleNamespace(hidden_size=hidden)

        def forward(self, x):
            # In the real model `x` is token ids; here we pass a float embedding
            # tensor directly so the Linear "blocks" accept it unchanged.
            return self.lm_head(self.model(x))

    model = _TinyLM().eval()
    assert num_layers(model) == n_layers
    assert hidden_size(model) == hidden

    reft = ReftR1(hidden_dim=hidden)

    # (a) intervention actually MOVES a hidden state, and only along r_unit.
    h = torch.randn(2, 4, hidden)
    h2 = reft.intervention(h)
    assert h2.shape == h.shape
    assert (h2 - h).norm().item() > 0, "intervention left h unchanged"
    # The edit is rank-1: the change lies along r_unit (orthogonal part is zero).
    delta = (h2 - h).reshape(-1, hidden)
    r_unit = reft.r_unit.detach()
    ortho = delta - (delta @ r_unit).unsqueeze(-1) * r_unit
    assert ortho.norm().item() < 1e-4, "edit is not rank-1 along r_unit"

    # (b) detector_score runs on both a tensor and a numpy vector.
    s_t = detector_score(reft, torch.randn(hidden))
    s_n = detector_score(reft, np.random.randn(hidden).astype(np.float32))
    assert isinstance(s_t, float) and isinstance(s_n, float)

    # (c) the whole point: a loss on the INTERVENED logits trains r, w, b.
    fake_ids = torch.randn(2, 4, hidden)  # float "input_ids" flow through Linears
    logits = grad_reft_forward(model, fake_ids, reft, layer=1)
    assert logits.shape == (2, 4, vocab)
    loss = logits.float().pow(2).mean()
    loss.backward()
    for name, p in (("r", reft.r), ("w", reft.w), ("b", reft.b)):
        assert p.grad is not None, f"no gradient reached reft.{name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in reft.{name}"

    # (d) the hook must be gone after the intervened forward.
    _ = model(fake_ids)
    n_hooks = len(model.model.layers[1]._forward_hooks)
    assert n_hooks == 0, "intervention hook was not removed"

    # (e) ReftContext applies then removes its hook too.
    with ReftContext(model, reft, layer=1):
        assert len(model.model.layers[1]._forward_hooks) == 1
    assert len(model.model.layers[1]._forward_hooks) == 0, "ReftContext left a hook"

    # (f) save/load round-trips the weights.
    import os
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "reft_r1_selftest.pt")
    save_reft(tmp, reft, {"note": "self-test"})
    reft2, meta = load_reft(tmp, hidden_dim=hidden)
    with torch.no_grad():
        assert torch.allclose(reft2.intervention(h), h2, atol=1e-5), "save/load changed the map"
    assert meta.get("note") == "self-test"
    os.remove(tmp)

    print("[reft_r1 self-test] OK autograd into r,w,b; hook removed.")


if __name__ == "__main__":
    _self_test()
