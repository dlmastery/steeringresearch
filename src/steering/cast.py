"""cast.py — THE conditional, multi-intent safety-steering method.

This is the core contribution of the repo (the reviewer-P0 "the method does not
exist" answered in code). ``CASTSteerer`` is a *control policy* over the already-
tested residual-stream primitives in ``hooks.py``: it reads the residual stream
at a condition layer, runs a per-intent gate, and — ONLY for intents whose gate
fires — writes a (composed) safety direction at a write layer during generation.
If no intent fires, the forward pass is identical to the unsteered model. That
conditional identity is the whole point (CLAUDE.md §10) and is the KEY unit test.

Pipeline (see DESIGN.md for the full diagram):

    prompt -> [READ hook @ layer_condition] -> gate each intent: cos(h,v) > τ
           -> fired set -> [WRITE hook @ layer_write] compose & apply_operation
           -> {text, fired_intents, gate_scores}

Composition with the existing modules:
  * ``hooks.apply_operation``     — the actual residual edit (never re-implemented)
  * ``hooks.build_position_mask`` — special-token protection in the write hook
  * ``multi_intent.compose``      — combine the safety directions of fired intents
  * ``model.get_residual_layers`` / ``encode_to_device`` — model-agnostic access

The gate decision is LATCHED on the first (prompt) forward and reused across
decode steps, so the read is taken over the full prompt and the policy is
consistent for every generated token.

The numpy↔torch seam (DESIGN.md §3): condition/safety vectors are registered and
stored as numpy float32; they are cast to ``h.dtype/device`` inside the hook, so
a bf16/CUDA real Gemma and an fp32 CPU ``FakeResidualLM`` both work unbranched.
``FakeResidualLM`` has no ``.generate``; on it we run a deterministic greedy
decode with the SAME hooks active, so the gate/write logic is genuinely exercised
offline (and an unfired generation is bit-identical to the unsteered one).

STATUS: BUILT but NOT YET VALIDATED on real models / benchmarks. The unit tests
prove the conditional identity, multi-intent firing, and composition on
``FakeResidualLM``; no efficacy / over-refusal / jailbreak number has been
measured on Gemma yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .hooks import apply_operation
from .model import encode_to_device, get_residual_layers
from .multi_intent import compose

_EPS = 1e-8


@dataclass
class Intent:
    """A registered intent: detector direction + gate threshold + safety target.

    condition_vector : ``[dim]`` unit-ish direction read at ``layer_condition``;
                       the gate fires when ``cos(h_pooled, condition_vector) > threshold``.
    threshold        : per-intent cosine firing threshold.
    safety_vector    : ``[dim]`` direction written at ``layer_write`` when fired.
    """

    name: str
    condition_vector: np.ndarray
    threshold: float
    safety_vector: np.ndarray

    @property
    def condition_unit(self) -> np.ndarray:
        v = self.condition_vector
        return (v / (float(np.linalg.norm(v)) + _EPS)).astype(np.float32)


class _GateState:
    """Per-generation latch for the gate decision (shared by read+write hooks)."""

    def __init__(self) -> None:
        self.latched: bool = False
        self.gate_scores: dict[str, float] = {}
        self.fired: list[str] = []


def _aligned_mask(prompt_mask: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Boolean mask ``[batch, seq, 1]`` aligned to ``h`` (mirrors ``hooks._SteerHook``).

    Prompt positions keep their special-token mask; any positions beyond the
    prompt (generated tokens) are steerable; a short incremental decode step
    (seq < prompt) is all-steerable (a fresh token is never a prompt special).
    """
    pm = prompt_mask.to(device=h.device, dtype=torch.bool)
    b, seq = h.shape[0], h.shape[1]
    pb, pseq = pm.shape[0], pm.shape[1]
    if pb != b:
        if pb == 1:
            pm = pm.expand(b, pseq)
        else:
            return torch.ones(b, seq, 1, dtype=torch.bool, device=h.device)
    if seq == pseq:
        full = pm
    elif seq > pseq:
        extra = torch.ones(b, seq - pseq, dtype=torch.bool, device=h.device)
        full = torch.cat([pm, extra], dim=1)
    else:
        full = torch.ones(b, seq, dtype=torch.bool, device=h.device)
    return full.unsqueeze(-1)


class CASTSteerer:
    """Conditional, multi-intent safety steerer.

    Parameters
    ----------
    model           : ``FakeResidualLM`` or real Gemma.
    tokenizer       : matching tokenizer.
    layer_condition : residual layer read by the gate (must be < ``layer_write``
                      so the read precedes the write in a single forward pass).
    layer_write     : residual layer where the safety steer is written.
    """

    #: when True the gate is bypassed and every intent fires (the ablation
    #: baseline; see ``UnconditionalSteerer``).
    unconditional: bool = False

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layer_condition: int,
        layer_write: int,
    ) -> None:
        if layer_condition >= layer_write:
            raise ValueError(
                f"layer_condition ({layer_condition}) must be < layer_write "
                f"({layer_write}) so the gate read precedes the write in one forward"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.layer_condition = int(layer_condition)
        self.layer_write = int(layer_write)
        self._layers = get_residual_layers(model)
        self.intents: list[Intent] = []

    # ----------------------------------------------------------------------- #
    def add_intent(
        self,
        name: str,
        condition_vector: np.ndarray,
        threshold: float,
        safety_vector: np.ndarray,
    ) -> None:
        """Register an intent: its detector direction, gate threshold, and safety
        steer direction. Vectors are stored as numpy float32 and cast to torch
        inside the hook at generation time."""
        if any(i.name == name for i in self.intents):
            raise ValueError(f"intent '{name}' is already registered")
        self.intents.append(
            Intent(
                name=name,
                condition_vector=np.asarray(condition_vector, dtype=np.float32).reshape(-1),
                threshold=float(threshold),
                safety_vector=np.asarray(safety_vector, dtype=np.float32).reshape(-1),
            )
        )

    # ----------------------------------------------------------------------- #
    def _decide(self, h_pooled: np.ndarray) -> tuple[dict[str, float], list[str]]:
        """Run the per-intent gate on the pooled condition activation.

        Returns ``(gate_scores, fired)`` where each score is
        ``cos(h_pooled, condition_vector)`` and an intent fires when its score
        exceeds its threshold (or always, when ``unconditional``).
        """
        h_norm = float(np.linalg.norm(h_pooled)) + _EPS
        scores: dict[str, float] = {}
        fired: list[str] = []
        for intent in self.intents:
            cos = float(np.dot(h_pooled, intent.condition_unit)) / h_norm
            scores[intent.name] = cos
            if self.unconditional or cos > intent.threshold:
                fired.append(intent.name)
        return scores, fired

    def _make_hooks(self, state: _GateState, prompt_mask: torch.Tensor,
                    alpha: float, operation: str):
        """Build the (read, write) forward-hook callables sharing ``state``."""

        def read_hook(module, inputs, output):
            if not state.latched:
                h = output[0] if isinstance(output, tuple) else output
                pooled = h[0].mean(dim=0).detach().float().cpu().numpy()
                state.gate_scores, state.fired = self._decide(pooled)
                state.latched = True
            return output  # read-only

        def write_hook(module, inputs, output):
            if not state.fired:
                return output  # CONDITIONAL: nothing fired -> identical to base
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None
            active = [
                i.safety_vector for i in self.intents if i.name in state.fired
            ]
            # Equal weights compose the directions; magnitude is set by `alpha`
            # in apply_operation (so a single coefficient controls strength).
            composed = compose(active, [1.0] * len(active))
            v = torch.as_tensor(composed, dtype=h.dtype, device=h.device)
            steered = apply_operation(h, v, operation, alpha)
            mask = _aligned_mask(prompt_mask, h)
            steered = torch.where(mask, steered, h)
            if rest is not None:
                return (steered, *rest)
            return steered

        return read_hook, write_hook

    # ----------------------------------------------------------------------- #
    def generate(
        self,
        prompt: str,
        *,
        alpha: float,
        max_new_tokens: int = 64,
        operation: str = "relative_add",
    ) -> dict:
        """Conditionally steered generation.

        Reads the residual at ``layer_condition``, gates each registered intent,
        and (only for fired intents) writes their composed safety direction at
        ``layer_write`` via ``hooks.apply_operation``. If nothing fires the output
        is identical to the unsteered model.

        Returns ``{"text": str, "fired_intents": list[str], "gate_scores":
        dict[str, float]}``. On ``FakeResidualLM`` (no ``.generate``) ``text`` is a
        deterministic greedy decode (space-joined token ids) produced with the
        SAME hooks active, so the gate/write logic is genuinely exercised offline.
        """
        from .hooks import build_position_mask

        ids = encode_to_device(self.tokenizer, prompt, self.model)
        special = self._special_token_ids()
        prompt_mask = build_position_mask(ids, special)

        state = _GateState()
        read_hook, write_hook = self._make_hooks(state, prompt_mask, alpha, operation)

        cond_module = self._layers[self.layer_condition]
        write_module = self._layers[self.layer_write]
        handles = [
            cond_module.register_forward_hook(read_hook),
            write_module.register_forward_hook(write_hook),
        ]
        try:
            if self._can_generate():
                text = self._real_generate(ids, max_new_tokens)
            else:
                text = self._fake_generate(ids, max_new_tokens)
        finally:
            for hd in handles:
                hd.remove()

        return {
            "text": text,
            "fired_intents": list(state.fired),
            "gate_scores": dict(state.gate_scores),
        }

    # ----------------------------------------------------------------------- #
    def _special_token_ids(self) -> list[int]:
        fn = getattr(self.model, "special_token_ids", None)
        if callable(fn):
            return list(fn())
        ids: list[int] = []
        for attr in ("bos_token_id", "eos_token_id"):
            val = getattr(self.tokenizer, attr, None)
            if isinstance(val, int):
                ids.append(val)
        return ids

    def _can_generate(self) -> bool:
        """True only for a real HF model with ``.generate`` (not ``FakeResidualLM``)."""
        gen = getattr(self.model, "generate", None)
        return callable(gen) and not type(self.model).__name__.startswith("Fake")

    def _real_generate(self, ids: torch.Tensor, max_new_tokens: int) -> str:
        gen_kwargs = dict(max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None and eos_id is not None:
            gen_kwargs["pad_token_id"] = eos_id
        with torch.no_grad():
            out = self.model.generate(ids, **gen_kwargs)  # type: ignore[operator]
        new_ids = out[0, ids.shape[1]:]
        try:
            return self.tokenizer.decode(new_ids, skip_special_tokens=True)
        except Exception:  # pragma: no cover - tokenizer without decode
            return ""

    def _fake_generate(self, ids: torch.Tensor, max_new_tokens: int) -> str:
        """Deterministic greedy decode for ``FakeResidualLM`` with hooks active.

        Runs the model forward once per new token (the hooks fire every step, so
        the gate latches on the prompt forward and the conditional write is
        applied on every subsequent step). Returns the new token ids space-joined
        — a deterministic surrogate for ``text`` that DIFFERS iff steering changed
        the logits, which is exactly what the conditional-identity test needs.
        """
        cur = ids
        generated: list[int] = []
        for _ in range(int(max_new_tokens)):
            with torch.no_grad():
                logits = self.model(cur).logits
            nxt = int(logits[0, -1].argmax().item())
            generated.append(nxt)
            cur = torch.cat(
                [cur, torch.tensor([[nxt]], dtype=cur.dtype, device=cur.device)], dim=1
            )
        return " ".join(str(g) for g in generated)


class UnconditionalSteerer(CASTSteerer):
    """Ablation baseline: identical API, but the gate ALWAYS fires.

    Every registered intent is treated as present regardless of the condition
    read, so the safety steer is applied on every prompt. This is the "is the
    gate doing anything?" control — comparing it against ``CASTSteerer`` isolates
    the value of the conditional policy (capability/over-refusal it saves on
    benign prompts).
    """

    unconditional: bool = True
