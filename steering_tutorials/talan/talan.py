"""talan.py -- the learned latent-adapter intervention (TALAN-inspired core).

Lesson 2 added a CONSTANT vector to the residual stream. Lesson 3 learned a
RANK-1 edit (a linear readout along one direction). This lesson learns a small
NONLINEAR bottleneck adapter -- a "latent side path" -- that reads the residual
``h`` at layer L and writes back a task-aligned correction ``delta``::

    z     = down(h)               # compress the token into latent memory  [.., m]
    z     = act(mix(z))           # remix within the latent memory         [.., m]
    delta = scale * up(z)         # write the perturbation back to hidden  [.., H]
    h'    = h + delta             # additive residual writeback

Read the three stages as TALAN's own description: "compress the active sequence
into latent memory, remix it into token-level perturbations, and write them back
through a controlled residual update". Compared with lesson 3's rank-1 edit, the
adapter is (a) higher-capacity (a full ``m``-dim bottleneck, not a single
direction), (b) NONLINEAR (the ``act`` in the middle), and (c) still cheap
(``2*H*m + m*m`` params, well under 1% of the backbone for small ``m``).

METHOD PROVENANCE -- honest labelling (see config.py for the full note):
  TALAN (Zhang et al. 2026, arXiv:2606.06902, 'TALAN: Task-Aligned Latent
  Adaptation Networks for Targeted Post-Training of Large Language Models') is a
  POST-TRAINING method that co-trains a backbone low-rank adapter AND a latent
  side path in one SFT loop. This module is our own INFERENCE-TIME analogue: we
  keep only the latent side path and FREEZE the whole LLM. It is inspired by, not
  identical to, the paper's method.

This module owns three moving parts, mirroring lesson 3's structure exactly so the
two are easy to diff:

  1. :class:`TalanAdapter` -- the adapter itself (learnable down / mix / up / scale).
  2. :func:`grad_talan_forward` -- a TRAINING-time adapted forward pass that keeps
     autograd intact, so the loss flows gradients back into the adapter only.
  3. :class:`TalanContext` -- an INFERENCE-time context manager that applies the
     same (now frozen) adapter during generation and removes its hook on exit.

Standalone: reuses only the lesson-2 model plumbing; third-party deps are
``torch``, ``transformers`` (transitively, for the real model) and ``numpy``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Reuse lesson-2 plumbing verbatim -- do NOT reimplement model access here.
from steering_tutorials.hello_world_steering.model_utils import (  # noqa: F401
    hidden_size,
    load_model,
    num_layers,
    residual_layers,
)

# The activations TALAN's "mixer" axis can select. Kept tiny and explicit so the
# config string maps to exactly one nn.Module.
_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


# ---------------------------------------------------------------------------
# 1. The latent-adapter intervention
# ---------------------------------------------------------------------------
class TalanAdapter(nn.Module):
    """A learned nonlinear bottleneck adapter: ``[.., hidden] -> [.., hidden]``.

    Parameters (all trainable; the LLM stays frozen)::

        down : Linear(hidden, memory)   compress the token into latent memory.
        mix  : Linear(memory, memory)   remix within the latent memory.
        up   : Linear(memory, hidden)   write the perturbation back to the stream.
        scale: scalar                   learned writeback gate (TALAN axis 6).

    The forward map is :meth:`intervention`. ``up`` is ZERO-initialised (weight and
    bias both 0) so at the start ``delta == 0`` and ``h' == h`` -- training begins
    from the frozen model's exact behaviour and GROWS the correction deliberately.
    This is the same "start at identity" choice lesson 3 makes with ``w = 0``; it
    is what stops the very first optimisation steps from wrecking benign prompts.

    ``memory`` is the bottleneck width (TALAN's "memory size" axis): small forces a
    compact, low-rank-ish correction and keeps the parameter count under 1% of the
    backbone, matching the paper's efficiency claim.
    """

    def __init__(self, hidden_dim: int, memory: int = 16,
                 mixer: str = "gelu", init_scale: float = 1.0) -> None:
        super().__init__()
        if mixer not in _ACTIVATIONS:
            raise ValueError(f"unknown mixer {mixer!r}; choose from {sorted(_ACTIVATIONS)}")
        self.hidden_dim = int(hidden_dim)
        self.memory = int(memory)
        self.mixer = str(mixer)

        self.down = nn.Linear(hidden_dim, memory)   # compress
        self.mix = nn.Linear(memory, memory)        # remix
        self.up = nn.Linear(memory, hidden_dim)     # write back
        self.act = _ACTIVATIONS[mixer]()
        # Learned writeback gate. Starts at init_scale but, because ``up`` is zero,
        # the initial delta is zero regardless -- the gate only matters once ``up``
        # has grown a nonzero map.
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        # Zero the writeback so the adapter starts as the identity map (see above).
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def delta(self, h: torch.Tensor) -> torch.Tensor:
        """The correction ``delta`` the adapter would add to ``h`` (same shape).

        Broadcasts over any leading dims (``[hidden]``, ``[seq, hidden]``,
        ``[batch, seq, hidden]`` all work). Differentiable in every adapter param.
        """
        # Cast the adapter to h's dtype/device on the fly so a bf16 model and an
        # fp32 adapter interoperate without the caller having to line them up.
        dtype, device = h.dtype, h.device
        z = self.down.to(device=device, dtype=dtype)(h)     # [.., memory]
        z = self.act(self.mix.to(device=device, dtype=dtype)(z))
        d = self.up.to(device=device, dtype=dtype)(z)       # [.., hidden]
        return self.scale.to(device=device, dtype=dtype) * d

    def intervention(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the additive residual writeback: ``h' = h + delta(h)``."""
        return h + self.delta(h)

    def num_params(self) -> int:
        """Trainable parameter count -- for the '< 1% of the backbone' check."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 2. Training-time adapted forward (autograd stays alive)
# ---------------------------------------------------------------------------
def grad_talan_forward(
    model: Any, input_ids: torch.Tensor, adapter: TalanAdapter, layer: int
) -> torch.Tensor:
    """One adapted forward pass that KEEPS the graph, returning logits.

    Registers a forward hook on ``residual_layers(model)[layer]`` that runs
    ``adapter.intervention`` on the layer's output hidden state -- with NO
    ``torch.no_grad`` and NO ``.detach()`` on the adapter's parameters -- so
    ``loss.backward()`` on the returned logits flows all the way back into the
    adapter's ``down / mix / up / scale`` (and nowhere else: the LLM is frozen).

    The hook is ALWAYS removed in a ``finally`` so a training step leaves no trace
    on the model -- exactly like lesson-2's ``SteeringContext.__exit__`` and
    lesson-3's ``grad_reft_forward``.
    """
    layers = residual_layers(model)
    layer = max(0, min(layer, len(layers) - 1))
    target = layers[layer]

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output          # [batch, seq, hidden]
        h_new = adapter.intervention(h)                # grad-carrying, no detach
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
class TalanContext:
    """Context manager that applies a (frozen) :class:`TalanAdapter` during generation.

    On ``__enter__`` it registers a forward hook on ``residual_layers[layer]`` that
    rewrites the layer's output via ``adapter.intervention``; on ``__exit__`` it
    removes the hook, restoring the model exactly (the adapter leaves no trace). At
    inference the params are fixed, so callers may run this inside ``torch.no_grad``
    -- the hook imposes nothing, mirroring lesson-2's ``SteeringContext`` and
    lesson-3's ``ReftContext`` interfaces.
    """

    def __init__(self, model: nn.Module, adapter: TalanAdapter, layer: int) -> None:
        self.model = model
        self.adapter = adapter
        layers = residual_layers(model)
        self.layer = max(0, min(layer, len(layers) - 1))
        self._handles: list[Any] = []

    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h_new = self.adapter.intervention(h)
        if is_tuple:
            return (h_new, *output[1:])
        return h_new

    def __enter__(self) -> "TalanContext":
        target = residual_layers(self.model)[self.layer]
        # prepend=True so any downstream probe hook observes the adapted residual --
        # matches lesson-2/3 contexts.
        self._handles.append(target.register_forward_hook(self._hook, prepend=True))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return None


# ---------------------------------------------------------------------------
# 4. Persistence
# ---------------------------------------------------------------------------
def save_talan(path: Any, adapter: TalanAdapter, meta: dict) -> None:
    """Save the adapter's weights + a metadata dict (shape + provenance)."""
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "hidden_dim": adapter.hidden_dim,
            "memory": adapter.memory,
            "mixer": adapter.mixer,
            "meta": dict(meta),
        },
        path,
    )


def load_talan(path: Any, hidden_dim: int | None = None) -> tuple[TalanAdapter, dict]:
    """Rebuild a :class:`TalanAdapter` from ``path``. -> (adapter, meta)."""
    # weights_only=False: our own local checkpoint; ``meta`` may carry numpy.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hd = hidden_dim if hidden_dim is not None else ckpt["hidden_dim"]
    adapter = TalanAdapter(
        hd, memory=ckpt.get("memory", 16), mixer=ckpt.get("mixer", "gelu")
    )
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    return adapter, ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# CPU self-test -- NO Gemma download. Verifies autograd flows loss -> adapter
# params, that the adapter starts at identity, and that hooks are removed.
# Run: python -m steering_tutorials.talan.talan
# ---------------------------------------------------------------------------
def _self_test() -> None:
    from types import SimpleNamespace

    torch.manual_seed(0)
    hidden, vocab, n_layers, mem = 16, 32, 3, 8

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
    for p in model.parameters():  # FREEZE the backbone, exactly like the trainer
        p.requires_grad_(False)
    assert num_layers(model) == n_layers
    assert hidden_size(model) == hidden

    adapter = TalanAdapter(hidden_dim=hidden, memory=mem, mixer="gelu")

    # (a) ZERO-INIT identity: with up=0, the adapter starts as the identity map,
    #     so delta is exactly zero and h' == h. This is the "start safe" property.
    h = torch.randn(2, 4, hidden)
    with torch.no_grad():
        assert adapter.delta(h).abs().max().item() == 0.0, "adapter should start at identity"
        assert torch.equal(adapter.intervention(h), h), "identity init: h' must equal h"

    # (b) after perturbing the writeback, the adapter MOVES a hidden state, and the
    #     move is a genuine nonlinear function of the input (differs across inputs).
    with torch.no_grad():
        adapter.up.weight.normal_(0, 0.1)
        adapter.up.bias.normal_(0, 0.1)
    h2 = adapter.intervention(h)
    assert h2.shape == h.shape
    assert (h2 - h).norm().item() > 0, "adapter left h unchanged after perturbation"
    d_a = adapter.delta(torch.randn(1, hidden))
    d_b = adapter.delta(torch.randn(1, hidden))
    assert not torch.allclose(d_a, d_b), "delta should depend on the input (it is input-conditioned)"

    # (c) the whole point: a loss on the ADAPTED logits trains ALL adapter params
    #     (down, mix, up, scale) and NOTHING else.
    fake_ids = torch.randn(2, 4, hidden)  # float "input_ids" flow through Linears
    logits = grad_talan_forward(model, fake_ids, adapter, layer=1)
    assert logits.shape == (2, 4, vocab)
    loss = logits.float().pow(2).mean()
    loss.backward()
    for name, p in adapter.named_parameters():
        assert p.grad is not None, f"no gradient reached adapter.{name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in adapter.{name}"
    # the frozen LM must NOT have received gradients from this path
    for name, p in model.named_parameters():
        assert p.grad is None, f"gradient leaked into frozen model.{name}"

    # (d) the hook must be gone after the adapted forward.
    _ = model(fake_ids)
    n_hooks = len(model.model.layers[1]._forward_hooks)
    assert n_hooks == 0, "adapter hook was not removed"

    # (e) TalanContext applies then removes its hook too.
    with TalanContext(model, adapter, layer=1):
        assert len(model.model.layers[1]._forward_hooks) == 1
    assert len(model.model.layers[1]._forward_hooks) == 0, "TalanContext left a hook"

    # (f) save/load round-trips the weights.
    import os
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "talan_selftest.pt")
    save_talan(tmp, adapter, {"note": "self-test"})
    adapter2, meta = load_talan(tmp, hidden_dim=hidden)
    with torch.no_grad():
        assert torch.allclose(adapter2.intervention(h), h2, atol=1e-5), "save/load changed the map"
    assert meta.get("note") == "self-test"
    os.remove(tmp)

    print("[talan self-test] OK identity init; autograd into adapter only; hooks removed.")


if __name__ == "__main__":
    _self_test()
