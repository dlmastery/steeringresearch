"""flow.py — the concept-conditioned velocity field that TRANSPORTS activations
(the FLAS core).

Lesson 2 added ONE fixed vector to the residual stream. Lesson 3 (``reft_r1``)
learned a rank-1 edit ``h' = h + r_unit*((w·h+b) - r_unit·h)``. Both are one-shot
maps ``h -> h'`` evaluated a single time. FLAS instead learns a **velocity field**
``v_theta(h, t, c)`` over activations and STEERS by integrating a flow ODE::

    dx/dt = v_theta(x, t, c),   x(0) = h            # transport, not a jump
    h' = x(T) = h + integral_0^T v_theta(phi_t(h), t, c) dt

We integrate with explicit Euler in ``n_steps`` (the trajectories we target are
near-straight — rectified flow — so few steps suffice). Two payoffs:

  * **Flow-time ``T`` is a continuous, zero-shot STRENGTH dial.** Integrating to a
    smaller ``T`` transports the activation less far along the SAME learned path;
    ``T = 0`` is the identity. No retraining, no raw-magnitude sweep.
  * **One field, many concepts.** The velocity is conditioned on a concept
    embedding ``c`` — the mean last-token activation of a handful of exemplars
    (the "ConceptEncoder") — so a single trained ``v_theta`` steers toward any
    concept whose exemplars you can encode.

  FLAS — Flow-based Activation Steering (github.com/flas-ai/FLAS) [UNVERIFIED] —
    the velocity-field-over-activations steering method reproduced here.
  Lipman et al. 2023, 'Flow Matching for Generative Modeling' (arXiv:2210.02747)
    [UNVERIFIED] — the continuous-time transport / flow-matching training framing.
  Liu et al. 2023, 'Flow Straight and Fast: Rectified Flow' (arXiv:2209.03003)
    [UNVERIFIED] — straight-line transport, why few-step Euler is accurate.

This module owns four moving parts:

  1. :func:`concept_embedding` — encode a concept as the mean exemplar activation.
  2. :class:`VelocityField` — the network ``v_theta(h, t, c)``.
  3. :func:`integrate_flow` — differentiable explicit-Euler integration of the ODE.
  4. :class:`FlowContext` — an inference-time context manager that transports the
     residual during generation, and :func:`save_flow` / :func:`load_flow`.

Standalone: reuses only the lesson-2 model plumbing; third-party deps are
``torch``, ``transformers`` (transitively) and ``numpy``.
"""
from __future__ import annotations

import math
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
# 1. The ConceptEncoder: a concept as its mean exemplar activation
# ---------------------------------------------------------------------------
def concept_embedding(
    model: Any, tok: Any, exemplars: list[str], layer: int
) -> np.ndarray:
    """Encode a concept as the MEAN last-token activation of its exemplars.

    ``c = mean_i  h_last( exemplar_i )`` at ``layer`` — the same last-token,
    decision-relevant residual lesson 2's diff-of-means reads, just averaged
    over a few positive exemplars of the concept (no negatives here: the field,
    not the embedding, learns the direction of transport). Returns ``[hidden]``.
    """
    # last_token_activations returns [n_exemplars, hidden]; log_every=0 silences it.
    acts = last_token_activations(model, tok, exemplars, layer, log_every=0)
    return acts.mean(axis=0).astype(np.float32)  # [hidden]


# ---------------------------------------------------------------------------
# 2. The velocity field v_theta(h, t, c)
# ---------------------------------------------------------------------------
class VelocityField(nn.Module):
    """The learned velocity ``v_theta(h, t, c) : R^hidden -> R^hidden``.

    Inputs (all broadcast to ``h``'s leading dims — ``[hidden]``, ``[seq, hidden]``
    and ``[batch, seq, hidden]`` all work)::

        h : [..., hidden_dim]   the current activation on the flow trajectory.
        t : scalar or [...]     the flow time in [0, 1] (canonical / normalised).
        c : [concept_dim] or [..., concept_dim]   the concept embedding to steer
                                toward; a single ``[concept_dim]`` vector is
                                broadcast over every position of ``h``.

    Architecture: embed the scalar time ``t`` with a small sinusoidal + MLP block
    to ``time_dim`` features, concatenate ``[h, time_emb, c]``, and run a GELU MLP
    of the given ``width`` down to a velocity in ``R^hidden``. The final layer is
    initialised SMALL so the field starts as a gentle transport (like lesson 3's
    ``w = 0`` init) while still passing gradient to every parameter.
    """

    def __init__(
        self,
        hidden_dim: int,
        concept_dim: int | None = None,
        time_dim: int = 16,
        width: int = 512,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.concept_dim = int(concept_dim if concept_dim is not None else hidden_dim)
        self.time_dim = int(time_dim)
        self.width = int(width)

        # Time: sinusoidal features (frequency encoding of t) -> a tiny learnable
        # MLP so the field can shape its own time-dependence.
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.GELU(),
        )

        # The main field: [h, time_emb, c] -> velocity in R^hidden.
        in_dim = self.hidden_dim + self.time_dim + self.concept_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.hidden_dim),
        )

        # Gentle start: shrink the output layer so the initial velocity (hence the
        # initial transport) is small. Non-zero, so gradients still reach every
        # earlier layer through it (unlike a hard zero-init, which would stall them).
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    # -- time encoding ----------------------------------------------------
    def _sinusoidal(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal frequency encoding of ``t`` -> ``[..., dim]`` (as in diffusion
        time embeddings). ``t`` may be any shape (incl. 0-d scalar)."""
        half = max(dim // 2, 1)
        # Geometrically-spaced frequencies, high-frequency detail near t=0..1.
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[..., None] * freqs                       # [..., half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [..., 2*half]
        if emb.shape[-1] < dim:  # odd dim: pad one zero column
            emb = torch.cat([emb, emb.new_zeros(*emb.shape[:-1], 1)], dim=-1)
        return emb[..., :dim]

    def forward(
        self,
        h: torch.Tensor,
        t: "float | torch.Tensor",
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity at state ``h``, time ``t``, concept ``c`` -> ``[..., hidden]``."""
        p_dtype = self.net[0].weight.dtype
        p_device = self.net[0].weight.device
        lead = tuple(h.shape[:-1])                          # leading dims of h

        h_in = h.to(dtype=p_dtype, device=p_device)         # [*lead, hidden]

        # --- time embedding, broadcast to h's leading dims ---
        if not torch.is_tensor(t):
            t = torch.as_tensor(float(t))
        t = t.to(device=p_device, dtype=p_dtype)
        temb = self.time_mlp(self._sinusoidal(t, self.time_dim))  # [*t.shape, time_dim]
        if temb.dim() == 1:  # scalar t -> [time_dim]; add singleton leading dims
            temb = temb.reshape(*([1] * len(lead)), self.time_dim)
        temb = temb.expand(*lead, self.time_dim)            # [*lead, time_dim]

        # --- concept, broadcast to h's leading dims ---
        c_in = c.to(dtype=p_dtype, device=p_device)
        if c_in.dim() == 1:  # single [concept_dim] vector -> broadcast over positions
            c_in = c_in.reshape(*([1] * len(lead)), self.concept_dim)
        c_in = c_in.expand(*lead, self.concept_dim)         # [*lead, concept_dim]

        feats = torch.cat([h_in, temb, c_in], dim=-1)       # [*lead, in_dim]
        return self.net(feats)                              # [*lead, hidden]


# ---------------------------------------------------------------------------
# 3. Integrate the flow: transport h to its steered position
# ---------------------------------------------------------------------------
def integrate_flow(
    vfield: VelocityField,
    h: torch.Tensor,
    c: torch.Tensor,
    T: float = 1.0,
    n_steps: int = 8,
) -> torch.Tensor:
    """Explicit-Euler integration of ``dx/dt = v(x, t, c)`` from ``t=0`` to ``T``.

    Starting at ``x = h``, take ``n_steps`` Euler steps of size ``dt = T/n_steps``.
    At step ``k`` the field is evaluated at the CANONICAL time
    ``t_norm = (k*dt)/max(T, eps)`` in ``[0, 1)`` — so the network always sees a
    layer-agnostic, ``T``-agnostic clock and ``T`` acts purely as the distance
    travelled (the strength dial). ``T = 0`` returns ``h`` unchanged (identity).

    Fully differentiable (no ``.detach()``), so a training loss can backprop
    through the whole integration into ``vfield``'s parameters; equally usable
    under ``torch.no_grad`` at inference.
    """
    eps = 1e-8
    dt = T / n_steps
    denom = max(float(T), eps)
    x = h
    for k in range(n_steps):
        t_norm = (k * dt) / denom                # canonical flow time in [0, 1)
        v = vfield(x, t_norm, c)                  # velocity at current state
        # Euler step; cast v to x's dtype so x keeps h's dtype (e.g. bf16 on GPU
        # even though the field's params are fp32).
        x = x + dt * v.to(device=x.device, dtype=x.dtype)
    return x


# ---------------------------------------------------------------------------
# 4. Inference-time transport during generation
# ---------------------------------------------------------------------------
class FlowContext:
    """Context manager that transports one residual layer's output via the flow.

    On ``__enter__`` it registers a forward hook on ``residual_layers[layer]`` that
    REPLACES the layer's output hidden state ``h`` with ``integrate_flow(vfield, h,
    concept_vec, T, n_steps)``; on ``__exit__`` it removes the hook, restoring the
    model exactly (the flow leaves no trace). ``T`` is the strength dial — the same
    field can be integrated harder or softer per call with no retraining.

    Mirrors lesson 2's ``SteeringContext`` / lesson 3's ``ReftContext`` interface,
    including ``prepend=True`` so a downstream probe hook observes the transported
    residual (which is also what the next layer sees).
    """

    def __init__(
        self,
        model: nn.Module,
        vfield: VelocityField,
        concept_vec: "np.ndarray | torch.Tensor",
        layer: int,
        T: float = 1.0,
        n_steps: int = 8,
    ) -> None:
        self.model = model
        self.vfield = vfield
        self._concept_in = concept_vec
        layers = residual_layers(model)
        self.layer = max(0, min(layer, len(layers) - 1))
        self.T = float(T)
        self.n_steps = int(n_steps)
        self._handles: list[Any] = []
        self._c: torch.Tensor | None = None  # concept tensor on the model device

    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output       # [batch, seq, hidden]
        h_new = integrate_flow(self.vfield, h, self._c, self.T, self.n_steps)
        if is_tuple:
            return (h_new, *output[1:])
        return h_new

    def __enter__(self) -> "FlowContext":
        device = next(self.model.parameters()).device
        v = self._concept_in
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        # Keep concept in fp32 on the model device; VelocityField.forward casts it
        # to the field's param dtype. reshape(-1) -> a [concept_dim] vector that
        # the field broadcasts over every token position.
        self._c = v.detach().reshape(-1).to(device=device).float()
        # the field must live on the same device as the residual stream it edits
        self.vfield = self.vfield.to(device)

        target = residual_layers(self.model)[self.layer]
        self._handles.append(target.register_forward_hook(self._hook, prepend=True))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._c = None
        return None


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------
def save_flow(path: Any, vfield: VelocityField, meta: dict) -> None:
    """Save the velocity field's weights + shape config + a metadata dict."""
    torch.save(
        {
            "state_dict": vfield.state_dict(),
            "hidden_dim": vfield.hidden_dim,
            "concept_dim": vfield.concept_dim,
            "time_dim": vfield.time_dim,
            "width": vfield.width,
            "meta": dict(meta),
        },
        path,
    )


def load_flow(
    path: Any, hidden_dim: int | None = None
) -> tuple[VelocityField, dict]:
    """Rebuild a :class:`VelocityField` from ``path``. -> (vfield, meta)."""
    # weights_only=False: our own local checkpoint; ``meta`` may carry numpy.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hd = hidden_dim if hidden_dim is not None else ckpt["hidden_dim"]
    vfield = VelocityField(
        hidden_dim=hd,
        concept_dim=ckpt["concept_dim"],
        time_dim=ckpt["time_dim"],
        width=ckpt["width"],
    )
    vfield.load_state_dict(ckpt["state_dict"])
    vfield.eval()
    return vfield, ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# CPU self-test — NO Gemma download. Verifies the flow integrates, backprops
# through the whole integration into the field, and that FlowContext transports
# then cleanly removes its hook.
# Run: python -m steering_tutorials.flas.flow
# ---------------------------------------------------------------------------
def _self_test() -> None:
    from types import SimpleNamespace

    torch.manual_seed(0)
    hidden = 16

    vfield = VelocityField(hidden_dim=hidden)  # concept_dim defaults to hidden

    # (a) integrate_flow runs and actually TRANSPORTS h; the T strength dial works.
    h = torch.randn(2, 4, hidden)              # [batch, seq, hidden]
    c = torch.randn(hidden)                    # one concept vector, broadcast over h
    h_full = integrate_flow(vfield, h, c, T=1.0, n_steps=8)
    assert h_full.shape == h.shape
    assert (h_full - h).norm().item() > 0, "flow left h unchanged"
    # T=0 is the identity (strength dial at zero) — a guaranteed property.
    h_zero = integrate_flow(vfield, h, c, T=0.0, n_steps=8)
    assert torch.allclose(h_zero, h), "T=0 must be a no-op transport"
    # A [hidden] activation (no batch/seq dims) integrates too (broadcast check).
    assert integrate_flow(vfield, torch.randn(hidden), c, T=1.0, n_steps=4).shape == (hidden,)

    # (b) the whole integration is differentiable: a loss on h' trains the field.
    h_req = torch.randn(2, 4, hidden)
    out = integrate_flow(vfield, h_req, c, T=1.0, n_steps=4)
    loss = out.pow(2).mean()
    loss.backward()
    total_grad = 0.0
    for name, p in vfield.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"
        total_grad += p.grad.norm().item()
    assert total_grad > 0, "backward produced only zero gradients"

    # (c) FlowContext transports a fake decoder's residual, then removes the hook.
    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])

        def forward(self, x):
            for blk in self.layers:
                x = blk(x)
            return x

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _Inner()
            self.config = SimpleNamespace(hidden_size=hidden)

        def forward(self, x):
            return self.model(x)

    model = _Tiny().eval()
    assert num_layers(model) == 3 and hidden_size(model) == hidden

    layer = 1
    cap: dict[str, torch.Tensor] = {}

    def probe(_m, _i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    ph = residual_layers(model)[layer].register_forward_hook(probe)
    x = torch.randn(2, 4, hidden)
    try:
        model(x)
        base = cap["h"].clone()
        n_before = len(residual_layers(model)[layer]._forward_hooks)

        concept = torch.randn(hidden)
        with FlowContext(model, vfield, concept, layer, T=1.0, n_steps=8):
            # the flow hook is installed (prepend=True => it fires before `probe`).
            assert len(residual_layers(model)[layer]._forward_hooks) == n_before + 1
            model(x)
            flowed = cap["h"].clone()

        # hook gone -> model restored exactly.
        assert len(residual_layers(model)[layer]._forward_hooks) == n_before
        model(x)
        restored = cap["h"].clone()
    finally:
        ph.remove()

    assert (flowed - base).norm().item() > 0, "FlowContext did not transport h"
    assert (restored - base).norm().item() == 0.0, "FlowContext did not restore exactly"

    # (d) save/load round-trips the field (same transport out).
    import os
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "flas_flow_selftest.pt")
    save_flow(tmp, vfield, {"note": "self-test"})
    vfield2, meta = load_flow(tmp, hidden_dim=hidden)
    with torch.no_grad():
        assert torch.allclose(
            integrate_flow(vfield2, h, c, T=1.0, n_steps=8), h_full, atol=1e-5
        ), "save/load changed the flow"
    assert meta.get("note") == "self-test"
    os.remove(tmp)

    print("[flas self-test] OK flow integrates + autograd + hook removed.")


if __name__ == "__main__":
    _self_test()
