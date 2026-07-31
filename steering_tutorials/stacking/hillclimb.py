"""hillclimb.py — the extra priors the stacking hill-climb needs.

`stacking.py` gives us three priors that differ only in SITE and OPERATION
(A @L12 relative_add, B @L8 relative_add, B' @L12 add). That covers two of the
three clauses of the CLAUDE.md sec.9 decision rule. This module adds the pieces
needed to exercise the rest of it:

  * :func:`orthogonal_direction` — a direction that is EXACTLY orthogonal to the
    refusal direction, so prior C can sit at the SAME site as A and still be a
    "near-orthogonal direction" (the second STACK clause). Contrast with B',
    which sits at the same site with the SAME direction (the COMPETE clause).

  * :func:`norm_clamp` — the norm / manifold clamp: a CONSTRAINT that runs after
    every injector hook in the same forward pass and rescales the composed delta
    so that ||dh|| / ||h_base|| <= cap at each prior layer. This is guard layer B
    of the five-layer Rogue-Scalpel guard and it is the direct actuator of the N5
    norm budget. It is a meta-layer: it injects no direction of its own.

  * :func:`gated_generate` — the CAST conditional gate applied to a WHOLE stack:
    run lesson-1's probe on the prompt, and only open the stack's hooks if it
    fires. Also a meta-layer (it decides IF, not WHAT).

  * :func:`measure_norm_budget` — the N5 measurement, generalised so it can be
    taken with the clamp active (the version in ``run_stacking`` hard-codes a
    bare ``stack_contexts``).

References
----------
  Panickssery/Rimsky et al. 2023, 'Steering Llama 2 via Contrastive Activation
    Addition' (arXiv:2312.06681) — the additive edit each injector applies.
  Lee et al. 2024, 'Programming Refusal with Conditional Activation Steering'
    (arXiv:2409.05907) — CAST, the conditional meta-layer. [UNVERIFIED id]
  corpus/steering-first-principles-v2-with-PSR-and-rogue-scalpel.md — guard
    layer B (norm / manifold clamp) and the N5 norm-budget indicator.

No model is loaded at import time; a CPU self-test at the bottom verifies the
clamp math and the orthogonality construction on a tiny stand-in module.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

import numpy as np

from steering_tutorials.hello_world_steering.model_utils import (
    generate,
    residual_layers,
)

from .stacking import Prior, stack_contexts


# ---------------------------------------------------------------------------
# 1. An orthogonal direction — the "near-orthogonal direction" STACK clause
# ---------------------------------------------------------------------------
def orthogonal_direction(acts: np.ndarray, v_unit: np.ndarray) -> np.ndarray:
    """Top principal direction of ``acts`` AFTER projecting out ``v_unit``.

    Parameters
    ----------
    acts : np.ndarray, shape ``[n, hidden]``
        Last-token residual activations at the target layer (the same pooled
        extract set the refusal diff-of-means was built from).
    v_unit : np.ndarray, shape ``[hidden]``
        The refusal direction, unit length.

    Returns
    -------
    np.ndarray, shape ``[hidden]``, unit length, with ``dot(C, v_unit) == 0``
    to numerical precision.

    Why this and not a second concept vector: this lesson's pool carries exactly
    ONE labelled contrast (harmful vs benign), so a genuine second *concept*
    direction is not available honestly. What IS available is the dominant
    remaining variation once the refusal axis is removed — a real direction in
    the model's activation geometry, exactly orthogonal to the refusal axis by
    construction. It is an ORTHOGONALITY CONTROL, not a second concept, and must
    be reported as such.
    """
    X = np.asarray(acts, dtype=np.float64)
    v = np.asarray(v_unit, dtype=np.float64).reshape(-1)
    v = v / np.linalg.norm(v)

    X = X - X.mean(axis=0, keepdims=True)          # centre
    X = X - np.outer(X @ v, v)                     # project the refusal axis out

    # Top right-singular vector of the deflated matrix.
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    c = vt[0]
    c = c - (c @ v) * v                            # re-orthogonalise (numerical)
    c = c / np.linalg.norm(c)

    assert abs(float(c @ v)) < 1e-8, "orthogonal_direction is not orthogonal"
    return c.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. The norm / manifold clamp — a CONSTRAINT meta-layer
# ---------------------------------------------------------------------------
@contextmanager
def norm_clamp(model: Any, layer_indices: "list[int]", cap: float) -> Iterator[None]:
    """Bound the composed steering delta at each layer to ``cap`` of ``||h||``.

    Mechanics (hook ORDER is the whole trick). ``SteeringContext`` registers its
    edit with ``prepend=True``, so the most recently registered prepended hook
    runs FIRST. We therefore enter this context *after* the injector contexts:

        capture hook   (prepend=True, registered last  -> runs FIRST)  sees h_base
        injector hooks (prepend=True, registered before -> run next)   add their deltas
        rescale hook   (appended                        -> runs LAST)  applies the cap

    The rescale is per position ``p``::

        r_p     = ||h_p - h_base_p|| / ||h_base_p||
        scale_p = min(1, cap / r_p)
        h_p     = h_base_p + scale_p * (h_p - h_base_p)

    so a stack already inside budget is left EXACTLY untouched (``scale = 1``),
    and an over-spent stack is pulled back onto the cap sphere without changing
    the delta's DIRECTION. That is what makes it a constraint rather than a
    fourth injector: it can only ever shrink what the injectors did.

    Usage::

        with stack_contexts(model, priors, special):
            with norm_clamp(model, [p.layer for p in priors], cap=0.10):
                generate(...)
    """
    import torch

    layers = residual_layers(model)
    idxs = sorted(set(int(i) for i in layer_indices))
    base: dict[int, "torch.Tensor"] = {}
    handles = []

    def _mk_capture(i: int):
        def hook(_m, _in, out):
            h = out[0] if isinstance(out, tuple) else out
            base[i] = h.detach().clone()
            return None                      # observe only
        return hook

    def _mk_rescale(i: int):
        def hook(_m, _in, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            h0 = base.get(i)
            if h0 is None or h0.shape != h.shape:
                return None                  # nothing captured -> no-op
            delta = h - h0
            r = delta.norm(dim=-1, keepdim=True) / h0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scale = torch.clamp(cap / r.clamp_min(1e-12), max=1.0)
            h_new = h0 + delta * scale
            return (h_new, *out[1:]) if is_tuple else h_new
        return hook

    try:
        for i in idxs:
            # rescale first (appended -> runs last), capture second (prepended ->
            # runs before the already-registered injector hooks).
            handles.append(layers[i].register_forward_hook(_mk_rescale(i)))
            handles.append(layers[i].register_forward_hook(_mk_capture(i), prepend=True))
        yield
    finally:
        for h in handles:
            h.remove()
        base.clear()


# ---------------------------------------------------------------------------
# 3. Composed application: stack (+ optional clamp), (+ optional CAST gate)
# ---------------------------------------------------------------------------
@contextmanager
def stack_with_clamp(model: Any, priors: "list[Prior]", special: "set[int] | None",
                     clamp_cap: "float | None") -> Iterator[None]:
    """``stack_contexts`` plus, optionally, the norm clamp on the same layers."""
    with ExitStack() as es:
        es.enter_context(stack_contexts(model, priors, special))
        if clamp_cap is not None and priors:
            es.enter_context(norm_clamp(model, [p.layer for p in priors], clamp_cap))
        yield


def apply_config(model: Any, tok: Any, prompt: str, priors: "list[Prior]",
                 max_new_tokens: int = 40, clamp_cap: "float | None" = None,
                 gate: Any = None) -> "tuple[str, bool]":
    """Generate under one ladder configuration. Returns ``(text, gate_fired)``.

    ``gate`` is a lesson-2 :class:`~steering_tutorials.hello_world_steering.gate.HarmGate`
    (or ``None``). When present, the ENTIRE stack — injectors and clamp alike —
    is applied only if the probe fires; otherwise the model runs unsteered. That
    is the CAST meta-layer: it decides IF, never WHAT.
    """
    fired = True
    if gate is not None:
        fired, _prob = gate.is_harmful(prompt)
    if not priors or not fired:
        return generate(model, tok, prompt, max_new_tokens=max_new_tokens), fired
    special = set(getattr(tok, "all_special_ids", []) or [])
    with stack_with_clamp(model, priors, special, clamp_cap):
        return generate(model, tok, prompt, max_new_tokens=max_new_tokens), fired


# ---------------------------------------------------------------------------
# 4. The N5 norm budget, measurable WITH the clamp active
# ---------------------------------------------------------------------------
def measure_norm_budget(model: Any, tok: Any, prompt: str, priors: "list[Prior]",
                        clamp_cap: "float | None" = None) -> float:
    """Cumulative ``||dh||/||h||`` summed over the priors' (unique) layers.

    Two forward passes on the prompt (baseline vs the composed configuration),
    capturing the residual at every prior layer, mean over positions, summed over
    layers. Identical definition to ``run_stacking._measure_norm_budget`` but it
    accepts the clamp, so R4/R5's budget is the *clamped* budget.
    """
    import torch

    device = next(model.parameters()).device
    layers = residual_layers(model)
    target_idx = sorted({p.layer for p in priors})
    if not target_idx:
        return 0.0

    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(device)

    def _capture() -> "dict[int, torch.Tensor]":
        grabbed: dict[int, torch.Tensor] = {}
        handles = []
        for idx in target_idx:
            def mk(i):
                def hook(_m, _in, out):
                    h = out[0] if isinstance(out, tuple) else out
                    grabbed[i] = h.detach()[0]
                return hook
            # appended -> runs after the injectors AND after the clamp rescale,
            # so it observes the FINAL state the next layer actually sees.
            handles.append(layers[idx].register_forward_hook(mk(idx)))
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for h in handles:
                h.remove()
        return grabbed

    base = _capture()
    special = set(getattr(tok, "all_special_ids", []) or [])
    with stack_with_clamp(model, priors, special, clamp_cap):
        steered = _capture()

    total = 0.0
    for idx in target_idx:
        b, s = base[idx].float(), steered[idx].float()
        ratio = (s - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)
        total += float(ratio.mean().item())
    return total


# ---------------------------------------------------------------------------
# CPU self-test — no model download.
# Run: python -m steering_tutorials.stacking.hillclimb
# ---------------------------------------------------------------------------
def _self_test() -> None:
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # ---- (a) orthogonal_direction is exactly orthogonal, and non-degenerate --
    v = rng.normal(size=32).astype(np.float32)
    v /= np.linalg.norm(v)
    acts = rng.normal(size=(64, 32)).astype(np.float32) + 3.0 * np.outer(
        rng.normal(size=64), v)          # deliberately refusal-dominated
    c = orthogonal_direction(acts, v)
    assert abs(float(c @ v)) < 1e-6, "C not orthogonal to v"
    assert abs(float(np.linalg.norm(c)) - 1.0) < 1e-6, "C not unit length"

    # ---- (b) the clamp: in-budget stack untouched, over-budget stack capped --
    class _Tiny(nn.Module):
        def __init__(self, hidden: int = 8, n: int = 4):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n)])
            self.config = SimpleNamespace(hidden_size=hidden, bos_token_id=2,
                                          eos_token_id=1, pad_token_id=0)

        def forward(self, x, **_kw):
            for blk in self.layers:
                x = blk(x)
            return x

    model = _Tiny().eval()
    layers = residual_layers(model)
    vv = torch.randn(8)
    vv = (vv / vv.norm()).numpy().astype(np.float32)
    x = torch.randn(2, 5, 8)

    cap_probe: dict[str, torch.Tensor] = {}

    def _probe(_m, _i, o):
        cap_probe["h"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    def _observe(priors_, clamp_):
        """Run one forward and return layer-2's FINAL output.

        The probe must be registered INSIDE the config's context: appended hooks
        fire in registration order, so a probe registered before the clamp would
        observe the pre-clamp state. This is the same reason
        :func:`measure_norm_budget` registers its capture hooks inside the
        ``with`` block.
        """
        with stack_with_clamp(model, priors_, None, clamp_):
            h = layers[2].register_forward_hook(_probe)
            try:
                model(x)
            finally:
                h.remove()
        return cap_probe["h"].clone()

    base = _observe([], None)

    # A single relative_add of 0.05 -> ratio 0.05, inside a cap of 0.10.
    pr_small = [Prior("small", vv, 2, 0.05, "relative_add")]
    h_small = _observe(pr_small, 0.10)
    r_small = ((h_small - base).norm(dim=-1) / base.norm(dim=-1))
    assert torch.allclose(r_small, torch.full_like(r_small, 0.05), atol=1e-4), \
        f"in-budget stack must pass the clamp untouched (got {r_small.flatten()[0]:.4f})"

    # A relative_add of 0.40 -> ratio 0.40, must be pulled back to the cap.
    pr_big = [Prior("big", vv, 2, 0.40, "relative_add")]
    h_big = _observe(pr_big, 0.10)
    r_big = ((h_big - base).norm(dim=-1) / base.norm(dim=-1))
    assert torch.allclose(r_big, torch.full_like(r_big, 0.10), atol=1e-4), \
        f"over-budget stack must be capped at cap (got {r_big.flatten()[0]:.4f})"

    # Direction preserved: the clamped delta is parallel to the raw delta.
    h_raw = _observe(pr_big, None)
    d_raw, d_cl = (h_raw - base), (h_big - base)
    cos = torch.nn.functional.cosine_similarity(
        d_raw.reshape(-1, 8), d_cl.reshape(-1, 8), dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-3), \
        "clamp changed the delta's direction — it must only shrink it"

    # No residual hooks after every context exited.
    n_hooks = sum(len(l._forward_hooks) for l in layers)
    assert n_hooks == 0, f"clamp left residual hooks (got {n_hooks})"

    print("[self-test] OK - orthogonal direction exact; clamp is a pure shrink "
          "(in-budget untouched, over-budget capped, direction preserved); "
          "no residual hooks.")


if __name__ == "__main__":
    _self_test()
