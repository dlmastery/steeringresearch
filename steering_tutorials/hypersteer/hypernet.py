"""hypernet.py — the hypernetwork that GENERATES a steering vector (lesson 3 core).

Lesson 2 added ONE fixed direction ``v = mean(harmful) - mean(benign)`` to the
residual stream. Lesson 3 learns a small network ``H_theta`` so that

    v = H(concept_embedding)

where ``concept_embedding`` is just the mean last-token activation over a few
exemplar prompts describing the concept. Two things fall out of this:

  * a learned, *nonlinear* concept->direction map (strictly more expressive than
    a single diff-of-means), and
  * *amortisation* — feed the hypernet a NEW concept's exemplars and it emits a
    steering vector immediately, with no fresh contrastive extraction pass.

  Sun et al. 2025, 'HyperSteer: Activation Steering at Scale with Hypernetworks'
    (arXiv:2506.03292) [UNVERIFIED] — hypernetworks that emit steering vectors.

This module owns three moving parts:

  1. :class:`HyperSteerNet` — the hypernetwork itself (concept_emb -> vector).
  2. :func:`concept_embedding` — build the concept "description" fed to (1).
  3. :func:`grad_steer_forward` — a TRAINING-time steered forward pass that keeps
     autograd intact, so the language-modelling loss can flow gradients back
     through the steering hook into the hypernetwork's parameters.

Contrast with lesson 2's ``SteeringContext`` / ``generate``: those run under
``torch.no_grad`` and are for INFERENCE. Training needs the graph alive, so we
use a bespoke hook here that never detaches the generated vector.

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
    last_token_activations,
    load_model,
    num_layers,
    residual_layers,
)


# ---------------------------------------------------------------------------
# 1. The hypernetwork
# ---------------------------------------------------------------------------
class HyperSteerNet(nn.Module):
    """Maps a concept embedding to a steering vector: ``[..., hidden] -> [..., hidden]``.

    Architecture (a bottleneck MLP with a learnable output scale)::

        LayerNorm(hidden) -> Linear(hidden, bottleneck) -> GELU
                          -> Linear(bottleneck, hidden) -> * exp(log_scale)

    The LayerNorm keeps the map insensitive to the absolute scale of the concept
    embedding (activation norms vary a lot across prompts); the GELU makes the
    map nonlinear (strictly more expressive than lesson-2's linear diff-of-means);
    and ``exp(log_scale)`` — a single learnable scalar initialised at 0, so scale
    1.0 — lets the network learn *how big* a steering vector to emit without
    fighting the L2-normalisation that the downstream steering hook applies.

    The returned vector is NOT detached: it carries the graph so a downstream
    loss can train ``H_theta`` end-to-end via :func:`grad_steer_forward`.
    """

    def __init__(self, hidden_dim: int, bottleneck: int = 256) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.bottleneck = int(bottleneck)
        self.norm = nn.LayerNorm(hidden_dim)
        self.up = nn.Linear(hidden_dim, bottleneck)
        self.act = nn.GELU()
        self.down = nn.Linear(bottleneck, hidden_dim)
        # Learnable output magnitude: init 0 => exp(0) == 1.0 (identity scale).
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, concept_emb: torch.Tensor) -> torch.Tensor:
        x = self.norm(concept_emb)
        x = self.up(x)
        x = self.act(x)
        x = self.down(x)
        return x * torch.exp(self.log_scale)


# ---------------------------------------------------------------------------
# 2. Building the concept embedding (the hypernet's input)
# ---------------------------------------------------------------------------
def concept_embedding(
    model: Any, tok: Any, exemplars: list[str], layer: int
) -> np.ndarray:
    """The concept "description" fed to the hypernet: mean last-token activation.

    We read the last-token residual at ``layer`` for each exemplar prompt (via
    the lesson-2 helper) and average them. This is a cheap, fixed-size summary of
    "what this concept looks like in activation space" — the hypernetwork's input,
    analogous to lesson-2's ``mean(harmful)`` term but without subtracting a
    baseline (the network learns any contrast it needs internally).

    Returns a ``[hidden]`` float32 vector.
    """
    acts = last_token_activations(model, tok, exemplars, layer)  # [n, hidden]
    return acts.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Training-time steered forward (autograd stays alive)
# ---------------------------------------------------------------------------
def grad_steer_forward(
    model: Any,
    input_ids: torch.Tensor,
    v: torch.Tensor,
    layer: int,
    alpha: float,
) -> torch.Tensor:
    """One steered forward pass that KEEPS the graph, returning logits.

    Registers a forward hook on ``residual_layers(model)[layer]`` that performs
    the same relative-add as lesson 2 —

        h[p] <- h[p] + alpha * ||h[p]|| * unit(v)

    — but crucially never calls ``.detach()`` on ``v`` and never wraps the model
    call in ``torch.no_grad``. The per-position norm ``||h[p]||`` is treated as a
    (constant) magnitude, while the direction ``unit(v)`` carries the gradient, so
    ``loss.backward()`` on the returned logits flows all the way back into the
    hypernetwork that produced ``v``. ``v`` is moved to the residual's dtype/device
    with ``.to(...)`` (graph-preserving) rather than reconstructed.

    The hook is ALWAYS removed in a ``finally`` so a training step leaves no trace
    on the model — exactly like lesson-2's ``SteeringContext.__exit__``.

    Parameters
    ----------
    model : the loaded causal LM.
    input_ids : ``[batch, seq]`` token ids for the (chat-templated) prompt+target.
    v : ``[hidden]`` steering vector, REQUIRING GRAD (from :class:`HyperSteerNet`).
    layer : index into ``residual_layers(model)``.
    alpha : step size as a fraction of the local hidden norm.
    """
    layers = residual_layers(model)
    layer = max(0, min(layer, len(layers) - 1))
    target = layers[layer]

    v = v.reshape(-1)
    v_unit = v / v.norm()  # keeps grad: division is differentiable

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output          # [batch, seq, hidden]
        # Move the (grad-carrying) direction onto the residual; do NOT detach.
        vv = v_unit.to(device=h.device, dtype=h.dtype)
        per_pos_norm = h.norm(dim=-1, keepdim=True)     # magnitude only
        h_new = h + alpha * per_pos_norm * vv
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
# 4. Persistence
# ---------------------------------------------------------------------------
def save_hypernet(path: Any, net: HyperSteerNet, meta: dict) -> None:
    """Save the hypernet's weights + a metadata dict (may carry ``concept_emb``)."""
    torch.save(
        {
            "state_dict": net.state_dict(),
            "hidden_dim": net.hidden_dim,
            "bottleneck": net.bottleneck,
            "meta": dict(meta),
        },
        path,
    )


def load_hypernet(
    path: Any, hidden_dim: int | None = None
) -> tuple[HyperSteerNet, dict]:
    """Rebuild a :class:`HyperSteerNet` from ``path``. -> (net, meta)."""
    # weights_only=False: our own local checkpoint, and ``meta`` may legitimately
    # carry a numpy ``concept_emb`` (not an allowed global under the 2.6 default).
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hd = hidden_dim if hidden_dim is not None else ckpt["hidden_dim"]
    bn = ckpt.get("bottleneck", 256)
    net = HyperSteerNet(hd, bn)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# CPU self-test — NO Gemma download. Verifies autograd flows H -> loss and that
# the steering hook is removed afterwards.
# Run: python -m steering_tutorials.hypersteer.hypernet
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

    net = HyperSteerNet(hidden_dim=hidden, bottleneck=8)
    emb = torch.randn(hidden, requires_grad=True)  # a fake concept embedding
    v = net(emb)                                    # grad-carrying steering vector
    assert v.shape == (hidden,)
    assert v.requires_grad, "hypernet output must carry grad"

    # A float "input_ids" of shape [batch, seq, hidden] flows through the Linears.
    fake_ids = torch.randn(2, 4, hidden)
    logits = grad_steer_forward(model, fake_ids, v, layer=1, alpha=0.08)
    assert logits.shape == (2, 4, vocab)

    # The whole point: a loss on the STEERED logits trains the hypernetwork.
    loss = logits.float().pow(2).mean()
    loss.backward()

    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradient reached the hypernetwork — autograd was broken"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    assert net.log_scale.grad is not None, "log_scale did not receive grad"

    # The hook must be gone: a plain forward now equals itself with no steering
    # side effects, and re-running does not raise (handle already removed).
    _ = model(fake_ids)
    n_hooks = len(model.model.layers[1]._forward_hooks)
    assert n_hooks == 0, "steering hook was not removed"

    # save/load round-trips the weights.
    import tempfile
    import os

    tmp = os.path.join(tempfile.gettempdir(), "hypersteer_selftest.pt")
    save_hypernet(tmp, net, {"note": "self-test", "concept_emb": emb.detach().numpy()})
    net2, meta = load_hypernet(tmp, hidden_dim=hidden)
    with torch.no_grad():
        v2 = net2(emb.detach())
    assert torch.allclose(v2, v.detach(), atol=1e-5), "save/load changed the map"
    assert meta.get("note") == "self-test"
    os.remove(tmp)

    print("[hypersteer self-test] OK autograd flows into H; hook removed.")


if __name__ == "__main__":
    _self_test()
