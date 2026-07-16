"""duality.py — the two mechanisms of the lesson.

Half 1 (PROMPT vs VECTOR), pure NumPy, CPU-testable:
  * ``prompt_shift_direction`` — the activation delta a refusal INSTRUCTION
    induces at a layer (mean instructed act - mean plain act).
  * ``cosine`` — direction agreement between that shift and the diff-of-means
    steering vector. This number IS the duality measurement.

Half 2 (ATTENTION vs RESIDUAL), a forward hook:
  * ``AttentionSteeringContext`` — the same relative-add edit lesson 2 applies to
    the residual stream, but applied to the ATTENTION sub-module's output
    (``layer.self_attn``). For Gemma-3 the attention output is added to the
    residual (after post-attention layernorm), so nudging it is a legitimate,
    distinct injection site.
  * ``steered_generate`` — one greedy generator with a ``site`` switch
    ("none" | "residual" | "attention") so the two arms differ by ONE thing:
    where the vector is injected.

The residual arm reuses lesson 2 verbatim (``SteeringContext``); the attention
arm mirrors its relative-add math and special-token guard so the comparison is
fair (both skip BOS / control tokens, both scale by the local norm).

Paper: Kang, Liu, Ma, Huang, Tan & Jiang, 2026, 'Prompt-Activation Duality:
Improving Activation Steering via Attention-Level Interventions'
(arXiv:2605.10664) — the attention-level intervention idea.

Only third-party deps are ``numpy`` / ``torch`` (torch only for the hook).
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ===========================================================================
# Half 1 — PROMPT vs VECTOR (pure math; no model, no torch)
# ===========================================================================
def prompt_shift_direction(
    acts_instructed: np.ndarray, acts_plain: np.ndarray
) -> dict:
    """Direction a prepended INSTRUCTION shifts activations at a layer.

    ``acts_instructed`` and ``acts_plain`` are ``[n, hidden]`` matrices of the
    same prompts read WITH and WITHOUT the instruction. The shift is the
    difference of the two group means::

        shift = mean(instructed) - mean(plain)

    A per-prompt difference would also work; we mean each group first so the
    result is symmetric with the diff-of-means steering vector it is compared to.

    Returns ``{"v_raw", "v_unit", "norm", "n"}`` (same shape of dict as the CAA
    vector), so the two directions are directly comparable.
    """
    a = np.asarray(acts_instructed, dtype=np.float64)
    b = np.asarray(acts_plain, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("acts must be matching [n, hidden] matrices")

    v_raw = (a.mean(axis=0) - b.mean(axis=0)).astype(np.float32)
    norm = float(np.linalg.norm(v_raw))
    v_unit = (v_raw / norm) if norm > 0 else v_raw.copy()
    return {
        "v_raw": v_raw,
        "v_unit": v_unit.astype(np.float32),
        "norm": norm,
        "n": int(a.shape[0]),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors, in [-1, 1] (0 if either is zero)."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0.0 or ny == 0.0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def random_cosine_baseline(
    vector: np.ndarray, n_samples: int = 200, seed: int = 0
) -> float:
    """Mean |cosine| between ``vector`` and random Gaussian directions.

    In high dimension two random vectors are nearly orthogonal, so this is ~0
    (about ``sqrt(2/(pi*hidden))``). It is the yardstick for half 1: a measured
    prompt-shift/steering-vector cosine only counts as "aligned" if it clears
    this random floor by a wide margin.
    """
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_samples, v.size))
    cs = (R @ v) / (np.linalg.norm(R, axis=1) * (np.linalg.norm(v) + 1e-12) + 1e-12)
    return float(np.mean(np.abs(cs)))


# ===========================================================================
# Half 2 — ATTENTION vs RESIDUAL (the forward hook)
# ===========================================================================
def attention_module(layer: Any):
    """Return the attention sub-module of a decoder ``layer``.

    Gemma-3 names it ``self_attn`` (see ``Gemma3DecoderLayer.forward``); we also
    accept ``attn`` / ``attention`` so the hook is robust across architectures.
    """
    for name in ("self_attn", "attn", "attention"):
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise ValueError("could not find an attention sub-module on this layer")


class AttentionSteeringContext:
    """Add a steering vector to the ATTENTION output of one decoder layer.

    Mirrors lesson 2's :class:`SteeringContext` but hooks ``layer.self_attn``
    instead of the whole block. The attention forward returns a tuple
    ``(attn_output, attn_weights, ...)``; we rewrite ``attn_output`` in place:

        a[p] <- a[p] + alpha * ||a[p]|| * unit(v)      (relative_add)

    Because Gemma-3 adds this attention output back into the residual stream
    (after post-attention layernorm), the edit propagates like a residual nudge
    but originates at the attention site — the paper's intervention locus.

    The special-token guard (skip BOS / control positions) matches lesson 2 so
    the residual and attention arms are compared fairly. On ``__exit__`` all hooks
    are removed and the model is restored exactly.
    """

    def __init__(
        self,
        model: Any,
        vector: "np.ndarray | Any",
        layer: int,
        alpha: float,
        special_ids: "set[int] | None" = None,
    ) -> None:
        import torch  # local import keeps ``import duality`` torch-free

        self._torch = torch
        self.model = model
        self.layer = int(layer)
        self.alpha = float(alpha)
        self._vector_in = vector
        self.special_ids = special_ids
        self._handles: list[Any] = []
        self._last_input_ids = None
        self._v_unit = None

    def _default_special_ids(self) -> set:
        cfg = getattr(self.model, "config", None)
        ids: set = set()
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            val = getattr(cfg, name, None)
            if isinstance(val, int):
                ids.add(val)
            elif isinstance(val, (list, tuple)):
                ids.update(int(x) for x in val)
        return ids

    def _capture_input_ids(self, _module, args, kwargs):
        """model forward-pre-hook: stash real 2-D integer input_ids for the mask.

        Only integer 2-D tensors count as token ids, so the float micro-test
        stays on the "steer all positions" path.
        """
        torch = self._torch
        ids = kwargs.get("input_ids", None)
        if ids is None and args and torch.is_tensor(args[0]):
            ids = args[0]
        if torch.is_tensor(ids) and ids.dim() == 2 and not ids.is_floating_point():
            self._last_input_ids = ids
        else:
            self._last_input_ids = None

    def _steer_hook(self, _module, _inputs, output):
        torch = self._torch
        is_tuple = isinstance(output, tuple)
        a = output[0] if is_tuple else output          # [batch, seq, hidden]

        per_pos_norm = a.norm(dim=-1, keepdim=True)     # [batch, seq, 1]
        delta = self.alpha * per_pos_norm * self._v_unit

        ids = self._last_input_ids
        if ids is not None and ids.shape[1] == a.shape[1]:
            special = torch.tensor(sorted(self.special_ids), device=ids.device)
            keep = ~torch.isin(ids, special)            # [batch, seq] bool
            delta = delta * keep.unsqueeze(-1).to(delta.dtype)

        a_new = a + delta
        if is_tuple:
            return (a_new, *output[1:])
        return a_new

    # -- residual_layers helper (avoid importing to keep the hook self-contained)
    def _layers(self):
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return list(inner.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        raise ValueError("could not find decoder layers on this model")

    def __enter__(self) -> "AttentionSteeringContext":
        torch = self._torch
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        v = self._vector_in
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        v = v.detach().to(device=device, dtype=dtype).reshape(-1)
        norm = v.norm()
        self._v_unit = v / norm if norm > 0 else v

        if self.special_ids is None:
            self.special_ids = self._default_special_ids()

        attn = attention_module(self._layers()[self.layer])
        # prepend=True: our edit runs before any downstream probe hook.
        self._handles.append(
            attn.register_forward_hook(self._steer_hook, prepend=True))
        self._handles.append(
            self.model.register_forward_pre_hook(
                self._capture_input_ids, with_kwargs=True, prepend=True))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._last_input_ids = None
        return None


def steered_generate(
    model: Any,
    tok: Any,
    prompt: str,
    vector: "np.ndarray | Any | None",
    layer: int,
    alpha: float,
    site: str = "residual",
    max_new_tokens: int = 48,
) -> str:
    """Greedy, chat-templated generation with a choice of injection ``site``.

    ``site``:
      * ``"none"``      — plain baseline (ignores vector/alpha).
      * ``"residual"``  — lesson 2's residual relative_add (reuses SteeringContext
                          via ``model_utils.generate``).
      * ``"attention"`` — this lesson's attention-output relative_add.

    Returns ONLY the newly generated text. The two steered arms differ by exactly
    one thing — the site — which is the whole point of the comparison.
    """
    import torch

    from steering_tutorials.hello_world_steering.model_utils import generate

    steering = vector is not None and alpha != 0.0 and site != "none"
    if not steering or site == "residual" or site == "none":
        # Baseline and the residual arm both go through lesson 2's generate,
        # which builds the residual SteeringContext when vector+alpha are set.
        return generate(
            model, tok, prompt, max_new_tokens=max_new_tokens,
            vector=(None if not steering else vector),
            layer=layer, alpha=(0.0 if not steering else alpha),
            operation="relative_add",
        )

    if site != "attention":
        raise ValueError(f"unknown site {site!r}")

    # --- attention arm: same greedy generate, wrapped in the attention hook ---
    device = next(model.parameters()).device
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(device)
    prompt_len = ids.shape[1]
    special = set(getattr(tok, "all_special_ids", []) or [])

    with torch.no_grad():
        with AttentionSteeringContext(model, vector, layer, alpha, special):
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None
                              else tok.eos_token_id),
            )
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


# ===========================================================================
# CPU self-test — NO model download. Verifies the shift math + the attention
# hook math + exact restore.  Run: python -m ...prompt_activation_duality.duality
# ===========================================================================
def _self_test() -> None:
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    torch.manual_seed(0)

    # --- Half 1: prompt-shift + cosine math -------------------------------
    hidden = 16
    plain = np.random.RandomState(1).randn(20, hidden).astype(np.float32)
    bump = np.zeros(hidden, dtype=np.float32)
    bump[3] = 5.0                                   # a known shift on axis 3
    instructed = plain + bump
    shift = prompt_shift_direction(instructed, plain)
    assert shift["v_raw"].shape == (hidden,)
    assert abs(shift["v_unit"][3] - 1.0) < 1e-4, "shift should recover axis 3"
    assert abs(cosine(shift["v_raw"], bump) - 1.0) < 1e-5, "cosine of colinear=1"
    assert abs(cosine(bump, -bump) + 1.0) < 1e-6, "cosine of anti-colinear=-1"
    # a random direction should be near-orthogonal in 16-D (well under 0.6).
    assert random_cosine_baseline(bump, n_samples=500) < 0.6

    # --- Half 2: attention hook adds alpha*||a||*unit(v) and restores exactly --
    class _Attn(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.proj = nn.Linear(h, h)

        def forward(self, x, **_kw):
            return (self.proj(x),)          # HF-style tuple: (attn_out, weights)

    class _Layer(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.self_attn = _Attn(h)
            self.mlp = nn.Linear(h, h)

        def forward(self, x, **_kw):
            a = self.self_attn(x)[0]
            return self.mlp(x + a)

    class _Tiny(nn.Module):
        def __init__(self, h=8, n=3):
            super().__init__()
            self.layers = nn.ModuleList([_Layer(h) for _ in range(n)])
            self.config = SimpleNamespace(
                hidden_size=h, bos_token_id=2, eos_token_id=1, pad_token_id=0)

        def forward(self, x, **_kw):
            for blk in self.layers:
                x = blk(x)
            return x

    model = _Tiny(h=8, n=3).eval()
    x = torch.randn(2, 5, 8)               # float -> "steer all positions"
    layer, alpha = 1, 0.5
    v = torch.randn(8)

    cap: dict = {}

    def probe(_m, _i, o):
        cap["a"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    attn = attention_module(model.layers[layer])
    ph = attn.register_forward_hook(probe)
    try:
        model(x)
        a_base = cap["a"].clone()
        with AttentionSteeringContext(model, v, layer, alpha):
            model(x)
        a_steered = cap["a"].clone()
        model(x)                            # after exit -> hook removed
        a_after = cap["a"].clone()
    finally:
        ph.remove()

    delta = a_steered - a_base
    assert delta.norm().item() > 0, "attention steering produced no change"
    unit_v = v / v.norm()
    expected = alpha * a_base.norm(dim=-1, keepdim=True) * unit_v
    assert torch.allclose(delta, expected, atol=1e-5), "attention relative_add math"
    assert (a_after - a_base).norm().item() == 0.0, "attention hook did not restore"

    print("[self-test] OK - prompt-shift + cosine math correct; "
          "attention hook adds alpha*||a||*unit(v) and restores exactly.")


if __name__ == "__main__":
    _self_test()
