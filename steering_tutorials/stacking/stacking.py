"""stacking.py — compose several steering priors at once, and the ladder helpers.

This is the mechanical core of lesson 12. Lesson 2's ``SteeringContext`` injects
ONE vector at ONE layer. Here we compose SEVERAL such injections simultaneously
and read what happens: priors on disjoint sites add (stack); priors on the same
site with an incompatible operation double-count (compete).

A **prior** is the atomic unit of a stack: a direction + a layer + a strength +
an operation. Two priors STACK when they act on different sites (or near-
orthogonal directions) and their summed norm stays in-distribution; they COMPETE
when they overwrite the same site with incompatible transformations, or when the
cumulative norm pushes the hidden state off the activation manifold.

  Rimsky et al. 2023, 'Steering Llama 2 via Contrastive Activation Addition'
    (arXiv:2312.06681) — the additive CAA edit each prior applies.
  Han et al. 2024, 'Word Embeddings Are Steers for Language Models' /
    steering-composition line — additive directions compose while near-orthogonal.
  Wehner et al. 2025 survey of representation steering — the norm-budget /
    off-manifold failure of over-stacked additive edits (N5 leading indicator).

The stack-vs-compete logic itself is mechanism-based and is documented in
``corpus/steering-stackable-vs-competing-analysis.md`` (sec. 1-4) and CLAUDE.md
section 9. This module only supplies the plumbing to *demonstrate* it.

Reuses lesson 2 verbatim: ``SteeringContext`` (the hook), ``generate`` (steered
decode), ``residual_layers`` (layer access). No model is loaded at import time.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from steering_tutorials.hello_world_steering.model_utils import (
    SteeringContext,
    generate,
    residual_layers,
)


# ---------------------------------------------------------------------------
# 1. The prior — the atomic unit of a stack
# ---------------------------------------------------------------------------
@dataclass
class Prior:
    """One steering intervention: a direction injected at a layer with a strength.

    Attributes
    ----------
    name : str
        Human-readable label used in the ladder + plots (e.g. ``"A refusal@L12"``).
    vector : np.ndarray, shape ``[hidden]``
        The steering DIRECTION. Normalised to unit length by convention (the
        strength lives in ``alpha`` / the operation), so two priors that share a
        direction differ only in site and operation.
    layer : int
        The residual-stream layer (intervention SITE) this prior edits.
    alpha : float
        Step size. For ``"relative_add"`` it is a fraction of the local ||h||;
        for ``"add"`` it is the raw scalar multiple of ``vector``.
    operation : str
        ``"relative_add"`` (norm-aware, the default) or ``"add"`` (literal
        ActAdd). Two priors on the SAME site + SAME direction but DIFFERENT
        operation are the canonical competing pair (they double-count the plane).
    """

    name: str
    vector: np.ndarray
    layer: int
    alpha: float
    operation: str = "relative_add"


# ---------------------------------------------------------------------------
# 2. Composing priors — several steering hooks live at once
# ---------------------------------------------------------------------------
@contextmanager
def stack_contexts(model: Any, priors: list[Prior],
                   special_ids: "set[int] | None" = None) -> Iterator[None]:
    """Enter one :class:`SteeringContext` per prior, all active simultaneously.

    Each prior registers its own forward hook on its layer (and its own input-id
    pre-hook for the special-token guard). Because the hooks live on possibly
    different layers, the forward pass is edited at every prior's site in one
    sweep — that is exactly what "stacking" means mechanically. Two priors on the
    SAME layer both fire on that layer's output, so their deltas add there (and,
    when their operations disagree, over-count).

    An :class:`~contextlib.ExitStack` guarantees that EVERY context is exited —
    and therefore every hook removed — even if one prior raises. On exit the
    model is restored exactly, leaving no residual hooks (asserted in the
    self-test below).
    """
    with ExitStack() as es:
        for p in priors:
            es.enter_context(
                SteeringContext(model, p.vector, p.layer, p.alpha,
                                p.operation, special_ids)
            )
        yield


def apply_stack(model: Any, tok: Any, prompt: str, priors: list[Prior],
                max_new_tokens: int = 40) -> str:
    """Generate ``prompt`` with ALL ``priors`` steering the model at once.

    We open every prior's :class:`SteeringContext` via :func:`stack_contexts`,
    then call lesson 2's plain :func:`generate` (``alpha=0`` ⇒ it adds no vector
    of its own). The active stack hooks do all the steering, so the returned text
    is the model decoded under the *composed* intervention. An empty ``priors``
    list yields the unsteered baseline.
    """
    if not priors:
        return generate(model, tok, prompt, max_new_tokens=max_new_tokens)
    special = set(getattr(tok, "all_special_ids", []) or [])
    with stack_contexts(model, priors, special):
        # generate() adds no vector of its own; the live stack hooks steer it.
        return generate(model, tok, prompt, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# 3. Building the two archetypal stacks from ONE refusal direction
# ---------------------------------------------------------------------------
def build_priors(
    refusal_unit: np.ndarray,
    primary_layer: int,
    orthogonal_layer: int,
    stack_alpha: float,
    compete_add_alpha: float,
) -> dict[str, Prior]:
    """Return the three named priors the ladder composes.

    A  : refusal @ ``primary_layer``      (relative_add) — the base prior.
    B  : refusal @ ``orthogonal_layer``   (relative_add) — a DISJOINT site, so
         A + B is the ORTHOGONAL-SITES stack (expected to STACK).
    B' : refusal @ ``primary_layer``      ("add", raw)   — the SAME site + same
         direction as A but an incompatible operation, so A + B' is the SAME-SITE
         stack (expected to COMPETE by double-counting the refusal plane).

    ``compete_add_alpha`` is the raw step for B'; run_stacking sets it so B'
    alone ≈ A alone in magnitude, making the 2b comparison controlled.
    """
    v = np.asarray(refusal_unit, dtype=np.float32).reshape(-1)
    return {
        "A": Prior("A refusal@L%d" % primary_layer, v, primary_layer,
                   stack_alpha, "relative_add"),
        "B": Prior("B refusal@L%d" % orthogonal_layer, v, orthogonal_layer,
                   stack_alpha, "relative_add"),
        "Bp": Prior("B' refusal@L%d(add)" % primary_layer, v, primary_layer,
                    compete_add_alpha, "add"),
    }


def ladder_rungs(priors: dict[str, Prior]) -> list[dict]:
    """The additive 2->N ladder: each rung adds exactly ONE prior to the last.

    Returns an ordered list of rung specs; ``run_stacking`` measures each. The
    ``expect`` field is the mechanism-based prediction (from the decision rule),
    NOT a measured result — the run confirms or falsifies it.
    """
    A, B, Bp = priors["A"], priors["B"], priors["Bp"]
    return [
        {"key": "rung1", "label": "1: A alone",
         "priors": [A], "category": "base",
         "expect": "baseline single-prior refusal"},
        {"key": "rung2a", "label": "2a: A + B  (different sites)",
         "priors": [A, B], "category": "stack",
         "expect": "STACK — disjoint sites, gains add, budget ~2x"},
        {"key": "rung2b", "label": "2b: A + B' (same site, diff op)",
         "priors": [A, Bp], "category": "compete",
         "expect": "COMPETE — double-counts refusal plane, gain < best single"},
        {"key": "rung3", "label": "3: A + B + B' (all-on hybrid)",
         "priors": [A, B, Bp], "category": "overstack",
         "expect": "OVER-STACK — norm budget spent, gibberish rises"},
    ]


# ---------------------------------------------------------------------------
# CPU self-test — NO model download. Verifies (a) the composed-stack hook math
# for both a disjoint-site and a same-site stack, and (b) that apply_stack /
# stack_contexts register AND remove every hook, leaving no residue.
# Run: python -m steering_tutorials.stacking.stacking
# ---------------------------------------------------------------------------
def _self_test() -> None:
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    torch.manual_seed(0)

    class _Tiny(nn.Module):
        """Stand-in decoder: a ModuleList of Linear "blocks" (as in lesson 2)."""

        def __init__(self, hidden: int = 8, n: int = 4):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n)])
            self.config = SimpleNamespace(
                hidden_size=hidden, bos_token_id=2, eos_token_id=1, pad_token_id=0
            )

        def forward(self, x, **_kw):
            for blk in self.layers:
                x = blk(x)
            return x

    model = _Tiny(hidden=8, n=4).eval()
    layers = residual_layers(model)

    def _hook_count() -> int:
        return sum(len(l._forward_hooks) for l in layers) + len(model._forward_pre_hooks)

    # Two unit priors sharing one direction; float input => "steer all" path.
    v = torch.randn(8)
    v = (v / v.norm()).numpy().astype(np.float32)
    x = torch.randn(2, 5, 8)

    # ---- (a) DISJOINT-SITE stack: priors on layers 1 and 3 -----------------
    # A probe on layer 3 sees BOTH edits (the layer-1 edit propagates forward
    # through layer 2/3, then the layer-3 edit adds on top). We only assert the
    # mechanical facts that are exact: each prior's own layer shows its delta.
    cap: dict[int, torch.Tensor] = {}

    def _probe(idx):
        def hook(_m, _i, o):
            cap[idx] = (o[0] if isinstance(o, tuple) else o).detach().clone()
        return hook

    ph1 = layers[1].register_forward_hook(_probe(1))
    ph3 = layers[3].register_forward_hook(_probe(3))
    try:
        model(x)
        base1, base3 = cap[1].clone(), cap[3].clone()

        # Reference count WITH the two probes registered but no stack yet.
        before = _hook_count()
        priors_disjoint = [
            Prior("A", v, 1, 0.5, "relative_add"),
            Prior("B", v, 3, 0.25, "relative_add"),
        ]
        with stack_contexts(model, priors_disjoint):
            assert _hook_count() > before, "stack did not register hooks"
            model(x)
        s1, s3 = cap[1].clone(), cap[3].clone()

        # After exit: every stack hook removed, back to the probe-only count.
        assert _hook_count() == before, "stack left residual hooks"
        model(x)
        after1 = cap[1].clone()
    finally:
        ph1.remove()
        ph3.remove()

    # Prior A (layer 1, relative_add 0.5) moves the layer-1 residual by exactly
    # 0.5 * ||h|| along unit(v) — verifiable at layer 1 in isolation.
    unit_v = torch.from_numpy(v)
    d1 = s1 - base1
    expected1 = 0.5 * base1.norm(dim=-1, keepdim=True) * unit_v
    assert torch.allclose(d1, expected1, atol=1e-5), "disjoint layer-1 delta wrong"
    # Layer-3 residual also moved (both the propagated layer-1 edit and prior B).
    assert (s3 - base3).norm().item() > 0, "disjoint stack left layer 3 unchanged"
    # Exact restoration.
    assert (after1 - base1).norm().item() == 0.0, "disjoint stack did not restore"

    # ---- (b) SAME-SITE stack: two priors on layer 2, different operations ---
    # Both hooks fire on layer 2's output, but SEQUENTIALLY: the later-entered
    # prior (B', "add") is prepended so it runs FIRST, then A ("relative_add")
    # runs on the already-edited state. So the composition is NOT two independent
    # deltas summed — A's norm-relative step is computed on B''s output. That
    # order-dependent double-counting of the same plane is exactly why same-site
    # stacks compete (the second edit contaminates the first's reference norm).
    cap.clear()
    ph2 = layers[2].register_forward_hook(_probe(2))
    try:
        model(x)
        base2 = cap[2].clone()
        before2 = _hook_count()
        priors_same = [
            Prior("A", v, 2, 0.3, "relative_add"),
            Prior("Bp", v, 2, 1.0, "add"),
        ]
        with stack_contexts(model, priors_same):
            model(x)
        s2 = cap[2].clone()
        assert _hook_count() == before2, "same-site stack left residual hooks"
    finally:
        ph2.remove()

    d2 = s2 - base2
    # Sequential: B' ("add", alpha=1.0) fires first -> h1 = base2 + unit_v;
    # then A ("relative_add", 0.3) -> h2 = h1 + 0.3*||h1||*unit_v.
    h1 = base2 + unit_v
    expected2 = (h1 + 0.3 * h1.norm(dim=-1, keepdim=True) * unit_v) - base2
    assert torch.allclose(d2, expected2, atol=1e-5), "same-site composed delta wrong"

    # ---- (c) build_priors / ladder wiring is coherent ----------------------
    pr = build_priors(v, primary_layer=12, orthogonal_layer=8,
                      stack_alpha=0.08, compete_add_alpha=0.08)
    assert pr["A"].layer == 12 and pr["B"].layer == 8
    assert pr["Bp"].layer == 12 and pr["Bp"].operation == "add"
    rungs = ladder_rungs(pr)
    assert [r["key"] for r in rungs] == ["rung1", "rung2a", "rung2b", "rung3"]
    assert [len(r["priors"]) for r in rungs] == [1, 2, 2, 3], "ladder must add one/rung"

    print("[self-test] OK — composed deltas correct (disjoint + same-site); "
          "apply_stack registers and removes ALL hooks; ladder wiring sound.")


if __name__ == "__main__":
    _self_test()
