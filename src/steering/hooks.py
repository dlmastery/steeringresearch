"""hooks.py — forward-hook-based residual-stream interventions.

This is AXIS 4 (the OPERATION) from
`corpus/steering-first-principles-v2-...md` Part 1.5. We support three
operations on the residual stream h at a chosen layer:

    "add"          h <- h + alpha * v               (CAA / ActAdd; classic)
    "rotate"       norm-preserving rotation of h toward v in the (h_proj, v) plane
    "project_out"  h <- h - (h . v_hat) v_hat        (refusal ablation / guard)

Key invariants (Rung-0 UNIT checks, ladder rung 0):
  - the residual stream actually changes (||Δh|| > 0);
  - special-token positions (BOS / start_of_turn) are NEVER steered;
  - on context exit, ALL hooks are removed and the model state is restored
    EXACTLY (the model is stateless, so "restoration" = no lingering hooks and
    identical forward output).

A read-only `ProbeHook` captures activations without modifying them.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .model import get_residual_layers

VALID_OPERATIONS = ("add", "rotate", "project_out")


def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm() + eps)


def apply_operation(
    h: torch.Tensor,
    v: torch.Tensor,
    operation: str,
    alpha: float,
) -> torch.Tensor:
    """Apply a single steering operation to a residual tensor.

    h : [..., dim] residual stream (any leading shape)
    v : [dim] steering direction
    Returns a tensor of the same shape as h. Pure-functional (no in-place).
    """
    v = v.to(dtype=h.dtype, device=h.device)
    if operation == "add":
        return h + alpha * v

    if operation == "project_out":
        v_hat = _unit(v)
        # h - (h . v_hat) v_hat ; alpha scales how much of the projection we remove
        coeff = torch.tensordot(h, v_hat, dims=([-1], [0]))  # [...]
        return h - alpha * coeff.unsqueeze(-1) * v_hat

    if operation == "rotate":
        # Norm-preserving rotation of h by angle alpha (radians) TOWARD v, inside
        # the 2-D plane spanned by h itself and v. Rotating h within a plane that
        # contains h keeps ||h|| EXACTLY fixed -> hugs the activation sphere
        # (Step 2 of the first-principles doc: rotations preserve coherence).
        #
        # Build an orthonormal basis (e1, e2) of that plane:
        #   e1 = h_hat
        #   e2 = unit(v - (v.e1) e1)   (Gram-Schmidt: v's component orthogonal to h)
        # Then h_rot = ||h|| * (cos(alpha) e1 + sin(alpha) e2).
        v_hat = _unit(v)
        h_norm = h.norm(dim=-1, keepdim=True)  # [...,1]
        e1 = h / (h_norm + 1e-8)               # [...,dim]
        # component of v_hat orthogonal to e1, per position
        v_dot_e1 = torch.tensordot(e1, v_hat, dims=([-1], [0])).unsqueeze(-1)  # [...,1]
        e2_raw = v_hat - v_dot_e1 * e1
        e2 = e2_raw / (e2_raw.norm(dim=-1, keepdim=True) + 1e-8)
        angle = torch.tensor(float(alpha), dtype=h.dtype, device=h.device)
        return h_norm * (torch.cos(angle) * e1 + torch.sin(angle) * e2)

    raise ValueError(f"Unknown operation '{operation}'. Valid: {VALID_OPERATIONS}")


def build_position_mask(
    input_ids: torch.Tensor,
    special_token_ids: Sequence[int],
) -> torch.Tensor:
    """Boolean mask [batch, seq]: True where steering IS allowed.

    Special-token positions (BOS / start_of_turn) are set False so they are
    excluded from steering (Rung-0 requirement).
    """
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for tok in special_token_ids:
        mask &= input_ids != tok
    return mask


class _SteerHook:
    """Internal forward-hook callable that edits a module's residual output."""

    def __init__(
        self,
        vector: torch.Tensor,
        operation: str,
        alpha: float,
        position_mask: Optional[torch.Tensor],
    ):
        self.vector = vector
        self.operation = operation
        self.alpha = alpha
        self.position_mask = position_mask  # [batch, seq] bool or None

    def __call__(self, module, inputs, output):
        # Residual blocks return either a tensor or a tuple whose [0] is the hidden.
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        steered = apply_operation(h, self.vector, self.operation, self.alpha)

        if self.position_mask is not None:
            # Broadcast mask [batch, seq] -> [batch, seq, 1]; only edit allowed positions.
            m = self.position_mask.to(device=h.device)
            if m.shape[:2] == h.shape[:2]:
                m3 = m.unsqueeze(-1)
                steered = torch.where(m3, steered, h)

        if rest is not None:
            return (steered, *rest)
        return steered


class ProbeHook:
    """Read-only forward hook that captures the residual activation.

    Usage:
        probe = ProbeHook()
        handle = layer.register_forward_hook(probe)
        model(input_ids)
        acts = probe.activations   # last captured [batch, seq, dim]
        handle.remove()
    """

    def __init__(self):
        self.activations: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self.activations = h.detach().clone()
        return output  # unchanged — read-only


class SteeringContext:
    """Context manager that registers steering hooks and removes them on exit.

    On __enter__: registers a `_SteerHook` on each requested residual layer.
    On __exit__:  removes EVERY handle, asserting exact state restoration (no
                  lingering hooks). The model itself is stateless, so identical
                  inputs produce identical outputs before and after the block.

    Parameters
    ----------
    model         : FakeResidualLM or real Gemma.
    vector        : [dim] steering direction (or per-layer dict {layer_idx: vec}).
    layers        : iterable of layer indices to hook.
    operation     : one of VALID_OPERATIONS.
    alpha         : steering coefficient.
    position_mask : optional [batch, seq] bool; True where steering allowed.
                    Build via build_position_mask() to exclude special tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        vector,
        layers: Iterable[int],
        operation: str = "add",
        alpha: float = 1.0,
        position_mask: Optional[torch.Tensor] = None,
    ):
        if operation not in VALID_OPERATIONS:
            raise ValueError(f"operation must be one of {VALID_OPERATIONS}")
        self.model = model
        self.vector = vector
        self.layers = list(layers)
        self.operation = operation
        self.alpha = alpha
        self.position_mask = position_mask
        self._handles: list = []
        self._layer_modules = get_residual_layers(model)

    def _vector_for(self, layer_idx: int) -> torch.Tensor:
        if isinstance(self.vector, dict):
            return self.vector[layer_idx]
        return self.vector

    def __enter__(self) -> "SteeringContext":
        for li in self.layers:
            module = self._layer_modules[li]
            hook = _SteerHook(
                vector=self._vector_for(li),
                operation=self.operation,
                alpha=self.alpha,
                position_mask=self.position_mask,
            )
            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False  # never suppress exceptions


def probe_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Capture residual activations at the given layers in one forward pass.

    Returns {layer_idx: [batch, seq, dim]}.
    """
    probes: dict[int, ProbeHook] = {}
    handles = []
    modules = get_residual_layers(model)
    try:
        for li in layers:
            p = ProbeHook()
            probes[li] = p
            handles.append(modules[li].register_forward_hook(p))
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()
    return {li: p.activations for li, p in probes.items()}
