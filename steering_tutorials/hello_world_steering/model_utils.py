"""model_utils.py — load a frozen Gemma, read its activations, and STEER it.

This is the mechanical core of lesson 2. Where lesson 1 only *read* the residual
stream (to train a probe), this module also *writes* to it: during generation we
add a steering vector to the hidden state at one layer and watch the behavior
change.

Four jobs, nothing else:
  1. load_model()             — load Gemma-3-1B + tokenizer, once, in bf16.
  2. last_token_activations() — [n, hidden] last-token residual at a layer
                                (feeds the CAA diff-of-means in steer_vector.py).
  3. mean_pool_activation()   — [hidden] mean-pooled residual (feeds the lesson-1
                                gate probe, which was trained on mean-pooled acts).
  4. SteeringContext / generate() — inject the vector during a forward pass.

Steering method: Contrastive Activation Addition / ActAdd. We add a fixed
direction to the residual stream at generation time. The relative-add variant
scales the step by the *local* hidden-state norm, so a single ``alpha`` (a
fraction of ||h||) behaves consistently across layers and exposes the coherence
cliff cleanly.

  Panickssery (formerly Rimsky) et al. 2023, 'Steering Llama 2 via Contrastive
    Activation Addition' (arXiv:2312.06681) — the CAA add-a-diff-of-means recipe.
  Turner et al. 2023, 'Steering Language Models With Activation Engineering'
    (arXiv:2308.10248) — ActAdd, add a vector at inference.
  Arditi et al. 2024, 'Refusal in LLMs is Mediated by a Single Direction'
    (arXiv:2406.11717) — the refusal/harm direction as a diff-of-means.

Standalone: only third-party deps are ``torch`` and ``transformers``.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:  # Some networks sit behind an SSL-intercepting proxy; use the OS trust store.
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - optional
    pass


# ---------------------------------------------------------------------------
# 1. Loading the model
# ---------------------------------------------------------------------------
def load_model(model_id: str, device: str | None = None) -> tuple[Any, Any]:
    """Load ``model_id`` in bf16, eval mode, on GPU if available. -> (model, tok).

    bf16 (not 4-bit) because the model is tiny (~2 GB) and full precision gives
    the cleanest activations to read and the most faithful steering. The two
    guards below stop transformers from ``torch.compile``-ing a Triton CUDA
    kernel, which has no Windows wheel and would otherwise crash.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Guard 1: if anything still tries to compile, fall back to eager.
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.suppress_errors = True
    except Exception:  # pragma: no cover
        pass

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()

    # Guard 2: never select the compiled static KV cache (Windows-safe generate).
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        try:
            gen_cfg.cache_implementation = "dynamic"
        except Exception:  # pragma: no cover
            pass

    print(f"[model] loaded {model_id} on {device} "
          f"({num_layers(model)} layers, hidden={hidden_size(model)})", file=sys.stderr)
    return model, tok


def residual_layers(model: nn.Module) -> list[nn.Module]:
    """The decoder blocks whose forward output is the residual stream.

    A forward hook on block ``l`` reads (or edits) the residual stream after
    layer ``l``. For Gemma-3 the path is ``model.model.layers``.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError("Could not find decoder layers on this model.")


def num_layers(model: nn.Module) -> int:
    return len(residual_layers(model))


def hidden_size(model: nn.Module) -> int:
    return int(model.config.hidden_size)


# ---------------------------------------------------------------------------
# 2. Reading activations (last-token and mean-pooled)
# ---------------------------------------------------------------------------
def _forward_capture(model: Any, tok: Any, prompt: str, layer: int) -> torch.Tensor:
    """Run one chat-templated forward pass; return the residual at ``layer``.

    Returns a ``[seq, hidden]`` tensor (batch dim removed) on the model device.
    Shared by both pooling helpers so they stay byte-for-byte consistent.
    """
    device = next(model.parameters()).device
    layers = residual_layers(model)
    layer = max(0, min(layer, len(layers) - 1))
    target = layers[layer]

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach()

    handle = target.register_forward_hook(hook)
    try:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        model(ids)
    finally:
        handle.remove()
    return captured["h"][0]  # [seq, hidden]


@torch.no_grad()
def last_token_activations(
    model: Any, tok: Any, prompts: list[str], layer: int, log_every: int = 25
) -> np.ndarray:
    """Return an ``[n_prompts, hidden]`` float32 matrix of LAST-token activations.

    The last token sits at the ``<start_of_turn>model`` position (thanks to
    ``add_generation_prompt=True``), where the model has just finished reading
    the prompt and is about to answer — the most decision-relevant position, and
    the one CAA / Arditi et al. use to read the refusal direction.
    """
    feats: list[np.ndarray] = []
    for i, prompt in enumerate(prompts):
        h = _forward_capture(model, tok, prompt, layer)  # [seq, hidden]
        feats.append(h[-1].float().cpu().numpy())         # last token
        if log_every and (i + 1) % log_every == 0:
            print(f"[acts] {i + 1}/{len(prompts)}", file=sys.stderr)
    return np.stack(feats).astype(np.float32)


@torch.no_grad()
def mean_pool_activation(model: Any, tok: Any, prompt: str, layer: int) -> np.ndarray:
    """Return a ``[hidden]`` float32 vector: the residual at ``layer`` mean-pooled
    over all token positions.

    This matches how the lesson-1 probe was trained (mean pooling), so the gate
    probe can be applied to a single prompt to decide *whether* to steer.
    """
    h = _forward_capture(model, tok, prompt, layer)  # [seq, hidden]
    return h.mean(0).float().cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Writing activations: the steering hook
# ---------------------------------------------------------------------------
class SteeringContext:
    """Context manager that adds a steering vector to one residual layer.

    On ``__enter__`` it registers a forward hook on ``residual_layers[layer]``
    that rewrites the layer's output hidden state ``h``; on ``__exit__`` it
    removes the hook, restoring the model exactly (steering leaves no trace).

    Two operations:

      "relative_add"  (default, robust across layers):
            h[p] <- h[p] + alpha * ||h[p]|| * unit(v)
        The step at each position ``p`` is a fraction ``alpha`` of that
        position's own hidden norm, in the unit-vector direction of ``v``.
        Because it is norm-relative, one ``alpha`` transfers across layers and
        sweeping it traces the efficacy-vs-coherence curve smoothly.

      "add"  (literal ActAdd):
            h[p] <- h[p] + alpha * v
        The raw vector, scaled by ``alpha``. Sensitive to ||v|| and the layer's
        activation scale — useful for reproducing the original ActAdd recipe.

    Special-token guard: we never steer BOS / ``<start_of_turn>`` / other control
    positions (steering them tends to derail formatting for no behavioral gain).
    A position mask is built from the model's ``input_ids`` when they are visible
    to the hook (the prefill pass). During decoding each step feeds a single
    freshly generated token, which is never special, so we steer it.

    Parameters
    ----------
    model : the loaded causal LM.
    vector : np.ndarray | torch.Tensor, shape ``[hidden]`` — the steering dir.
    layer : int — index into ``residual_layers(model)``.
    alpha : float — step size (fraction of ||h|| for relative_add; raw for add).
    operation : "relative_add" | "add".
    special_ids : optional set of token ids to skip. When ``None`` we fall back
        to the model config's bos/eos/pad ids. ``generate()`` passes the exact
        set from the tokenizer so the guard is precise; other callers get a
        sensible default.
    """

    def __init__(
        self,
        model: nn.Module,
        vector: "np.ndarray | torch.Tensor",
        layer: int,
        alpha: float,
        operation: str = "relative_add",
        special_ids: "set[int] | None" = None,
    ) -> None:
        if operation not in ("relative_add", "add"):
            raise ValueError(f"unknown operation {operation!r}")
        self.model = model
        self.layer = layer
        self.alpha = float(alpha)
        self.operation = operation
        self._vector_in = vector
        # Where to skip steering. None -> derive from config in __enter__.
        self.special_ids = special_ids
        self._handles: list[Any] = []
        self._last_input_ids: torch.Tensor | None = None
        self._v: torch.Tensor | None = None       # raw vector (model dtype/dev)
        self._v_unit: torch.Tensor | None = None   # L2-normalized vector

    # -- helpers -----------------------------------------------------------
    def _default_special_ids(self) -> set[int]:
        cfg = getattr(self.model, "config", None)
        ids: set[int] = set()
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            val = getattr(cfg, name, None)
            if isinstance(val, int):
                ids.add(val)
            elif isinstance(val, (list, tuple)):
                ids.update(int(x) for x in val)
        return ids

    def _capture_input_ids(self, _module, args, kwargs):
        """forward-pre-hook on the whole model: stash input_ids for the mask.

        Only accept an integer 2-D tensor as real token ids — this keeps the
        fake-tensor micro-test (which passes floats) on the "steer all" path.
        """
        ids = kwargs.get("input_ids", None)
        if ids is None and args and torch.is_tensor(args[0]):
            ids = args[0]
        if torch.is_tensor(ids) and ids.dim() == 2 and not ids.is_floating_point():
            self._last_input_ids = ids
        else:
            self._last_input_ids = None

    def _steer_hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output           # [batch, seq, hidden]

        if self.operation == "relative_add":
            per_pos_norm = h.norm(dim=-1, keepdim=True)  # [batch, seq, 1]
            delta = self.alpha * per_pos_norm * self._v_unit
        else:  # "add"
            delta = self.alpha * self._v

        # Skip special tokens when we can see the ids and they line up in length.
        ids = self._last_input_ids
        if ids is not None and ids.shape[1] == h.shape[1]:
            special = torch.tensor(sorted(self.special_ids), device=ids.device)
            steer = ~torch.isin(ids, special)            # [batch, seq] bool
            delta = delta * steer.unsqueeze(-1).to(delta.dtype)

        h_new = h + delta
        if is_tuple:
            return (h_new, *output[1:])
        return h_new

    # -- context-manager protocol -----------------------------------------
    def __enter__(self) -> "SteeringContext":
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        v = self._vector_in
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        v = v.detach().to(device=device, dtype=dtype).reshape(-1)
        self._v = v
        norm = v.norm()
        self._v_unit = v / norm if norm > 0 else v

        if self.special_ids is None:
            self.special_ids = self._default_special_ids()

        target = residual_layers(self.model)[self.layer]
        # prepend=True: the steering edit runs before any user probe hook, so a
        # downstream hook observes the *steered* residual (also what the model's
        # next layer sees). This is what makes the micro-test measurable.
        self._handles.append(
            target.register_forward_hook(self._steer_hook, prepend=True)
        )
        self._handles.append(
            self.model.register_forward_pre_hook(
                self._capture_input_ids, with_kwargs=True, prepend=True
            )
        )
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._last_input_ids = None
        return None


# ---------------------------------------------------------------------------
# 4. Generation (steered or plain)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: Any,
    tok: Any,
    prompt: str,
    max_new_tokens: int = 48,
    vector: "np.ndarray | torch.Tensor | None" = None,
    layer: int | None = None,
    alpha: float = 0.0,
    operation: str = "relative_add",
) -> str:
    """Greedy, chat-templated generation. Returns ONLY the new text.

    If ``vector`` is given and ``alpha != 0`` the whole generation runs inside a
    :class:`SteeringContext`, so every position (prompt + each new token) is
    steered. Otherwise this is a plain unsteered baseline generation.
    """
    device = next(model.parameters()).device
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    prompt_len = ids.shape[1]

    def _run() -> torch.Tensor:
        return model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy: deterministic gates
            num_beams=1,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None
                          else tok.eos_token_id),
        )

    steering = vector is not None and alpha != 0.0
    if steering:
        steer_layer = num_layers(model) // 2 if layer is None else layer
        special = set(getattr(tok, "all_special_ids", []) or [])
        with SteeringContext(model, vector, steer_layer, alpha, operation, special):
            out = _run()
    else:
        out = _run()

    new_tokens = out[0][prompt_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CPU micro-test — NO model download. Verifies the steering hook math + restore.
# Run: python -m steering_tutorials.hello_world_steering.model_utils
# ---------------------------------------------------------------------------
def _self_test() -> None:
    from types import SimpleNamespace

    torch.manual_seed(0)

    class _Tiny(nn.Module):
        """Stand-in for a decoder: a ModuleList of Linear "blocks"."""

        def __init__(self, hidden: int = 8, n: int = 3):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n)])
            self.config = SimpleNamespace(
                hidden_size=hidden, bos_token_id=2, eos_token_id=1, pad_token_id=0
            )

        def forward(self, x, **_kw):
            for blk in self.layers:
                x = blk(x)
            return x

    model = _Tiny(hidden=8, n=3).eval()
    assert num_layers(model) == 3
    assert hidden_size(model) == 8

    x = torch.randn(2, 5, 8)          # [batch, seq, hidden] — float => "steer all"
    layer = 1
    alpha = 0.5
    v = torch.randn(8)

    # A probe hook on the target layer captures its output. With prepend=True the
    # steering hook fires first, so this probe sees the STEERED residual.
    cap: dict[str, torch.Tensor] = {}

    def probe(_m, _i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    ph = residual_layers(model)[layer].register_forward_hook(probe)
    try:
        model(x)
        h_base = cap["h"].clone()

        with SteeringContext(model, v, layer, alpha, "relative_add"):
            model(x)
        h_steered = cap["h"].clone()

        model(x)                       # after exit — hook removed
        h_after = cap["h"].clone()
    finally:
        ph.remove()

    delta = h_steered - h_base
    # (a) steering actually moved the residual.
    assert delta.norm().item() > 0, "expected a non-zero steering delta"
    # (b) the move is exactly alpha * ||h|| in the unit(v) direction.
    unit_v = v / v.norm()
    expected = alpha * h_base.norm(dim=-1, keepdim=True) * unit_v
    assert torch.allclose(delta, expected, atol=1e-5), "relative_add math mismatch"
    per_pos = delta.norm(dim=-1)
    ref = alpha * h_base.norm(dim=-1)
    assert torch.allclose(per_pos, ref, atol=1e-5), "per-position norm mismatch"
    # (c) exact restoration on exit: post-context output == baseline, ||Δ|| == 0.
    assert (h_after - h_base).norm().item() == 0.0, "hook did not restore exactly"

    # Also exercise the literal "add" op: delta == alpha * v everywhere.
    cap.clear()
    ph2 = residual_layers(model)[layer].register_forward_hook(probe)
    try:
        model(x)
        base2 = cap["h"].clone()
        with SteeringContext(model, v, layer, 1.0, "add"):
            model(x)
        add2 = cap["h"].clone()
    finally:
        ph2.remove()
    assert torch.allclose(add2 - base2, v, atol=1e-5), "add op math mismatch"

    print("[self-test] OK — relative_add + add math correct; hook restores exactly.")


if __name__ == "__main__":
    _self_test()
