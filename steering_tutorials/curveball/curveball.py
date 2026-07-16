"""curveball.py — the curved (nonlinear) steering path and its straight baseline.

The whole lesson in one geometric picture. Steering moves a residual-stream
activation ``h`` toward a target direction ``v_unit`` (lesson 2's diff-of-means
refusal direction). There are two ways to spend a fixed steering budget:

  STRAIGHT (lesson 2, the baseline):   the CHORD
        h_straight = h + alpha * ||h|| * v_unit
    A single jump of length ``alpha*||h||`` along the fixed direction ``v_unit``.
    It leaves the sphere of radius ``||h||`` — it INFLATES the norm — which is the
    "geometric distortion" the Curveball paper links to inconsistent, gibberish-
    prone behaviour when the push is large.

  CURVED (this lesson):                the ARC
        rotate h toward v_unit, along the great circle, by ``alpha`` radians,
        staying exactly on the sphere of radius ||h|| (the local norm shell).
    We integrate the rotation in ``n_steps`` small great-circle steps. At each
    step the effective push direction is the component of ``v_unit`` TANGENT to the
    current state (orthogonal to ``x``), which changes as ``x`` rotates — so the
    path bends to keep following the geodesic toward ``v``. The arc length equals
    the straight chord's length (``||h||*alpha == alpha*||h||``), so the two arms
    spend the SAME steering budget; only the geometry differs.

Why the arc might win: it reaches a comparable alignment with ``v`` (comparable
refusal efficacy) while adding ZERO net off-shell displacement, so it does not
push the activation off the data manifold — the leading indicator of the coherence
collapse (gibberish) that a large straight push causes.

  Raval, Song, Wu, Harrasse, Phillips, Barez & Abdullah 2026, 'Curveball Steering:
    The Right Direction To Steer Isn't Always Linear' (arXiv:2603.09313) — activation
    spaces are locally curved (geodesic/Euclidean distortion is large and concept-
    dependent), so a geometry-aware nonlinear path beats a global straight line.
    Their method is polynomial-kernel-PCA steering in a feature space; the great-
    circle geodesic below is OUR construction motivated by the same thesis.
  Panickssery (Rimsky) et al. 2023, 'Steering Llama 2 via Contrastive Activation
    Addition' (arXiv:2312.06681) — the straight relative-add step reproduced by the
    ``curved=False`` branch (identical math to lesson 2's ``SteeringContext``).

Conceptually adjacent to the ``flas`` lesson: FLAS also steers by INTEGRATING a
path (a learned velocity field) rather than adding a fixed vector. The difference:
FLAS learns where to flow; Curveball prescribes the geodesic from pure geometry,
no training.

Standalone: reuses only the lesson-2 model plumbing; third-party deps are
``numpy`` and ``torch``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Reuse the lesson-2 model plumbing verbatim — do NOT reimplement model access.
from steering_tutorials.hello_world_steering.model_utils import (  # noqa: F401
    hidden_size,
    num_layers,
    residual_layers,
)

_EPS = 1e-8


# =========================================================================== #
# 1. Pure geometry (NumPy) — no model, unit-tested on CPU below.
#    All functions act on the LAST axis, so a single [hidden] vector, a
#    [seq, hidden] matrix and a [batch, seq, hidden] tensor all work by
#    broadcasting; v_unit is a single [hidden] direction broadcast over positions.
# =========================================================================== #
def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize ``v`` along ``axis`` (a zero vector is returned unchanged)."""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n < _EPS, 1.0, n)


def straight_endpoint(h: np.ndarray, v_unit: np.ndarray, alpha: float) -> np.ndarray:
    """The STRAIGHT (chord) step: ``h + alpha*||h||*v_unit`` (lesson 2 relative-add).

    ``v_unit`` is assumed unit-length; it is broadcast over every leading position
    of ``h``. This is the exact displacement lesson 2 applies — a jump off the
    local norm shell whose length scales with the local norm ``||h||``.
    """
    h = np.asarray(h, dtype=np.float64)
    v_unit = np.asarray(v_unit, dtype=np.float64).reshape(-1)
    r = np.linalg.norm(h, axis=-1, keepdims=True)
    return h + alpha * r * v_unit


def curveball_endpoint(
    h: np.ndarray, v_unit: np.ndarray, alpha: float, n_steps: int = 8
) -> np.ndarray:
    """The CURVED (arc) step: rotate ``h`` toward ``v_unit`` by ``alpha`` radians.

    Integrates a great-circle geodesic on the sphere of radius ``||h||`` in
    ``n_steps`` equal rotations of ``dtheta = alpha / n_steps``. At each step:

        x_hat   = x / ||x||                          # current unit state
        tangent = v_unit - (v_unit . x_hat) x_hat    # part of v orthogonal to x
        x       = ||h|| * (cos(dtheta) x_hat + sin(dtheta) tangent_hat)

    The tangent is recomputed every step, so the effective push direction BENDS as
    ``x`` rotates — the path follows the manifold toward ``v`` instead of shooting
    straight at it. The norm ``||x||`` is held EXACTLY at ``||h||`` (zero off-shell
    displacement). If ``x`` is already (anti-)parallel to ``v_unit`` the tangent
    vanishes and that position stops rotating.

    ``alpha = 0`` (or ``n_steps = 0``) returns ``h`` unchanged (identity).
    """
    h = np.asarray(h, dtype=np.float64)
    v_unit = np.asarray(v_unit, dtype=np.float64).reshape(-1)
    if alpha == 0.0 or n_steps <= 0:
        return h.copy()

    r0 = np.linalg.norm(h, axis=-1, keepdims=True)     # local norm shell radius
    dtheta = float(alpha) / int(n_steps)
    cos_d, sin_d = np.cos(dtheta), np.sin(dtheta)

    x = h.copy()
    for _ in range(int(n_steps)):
        xn = np.linalg.norm(x, axis=-1, keepdims=True)
        x_hat = x / np.where(xn < _EPS, 1.0, xn)
        proj = np.sum(v_unit * x_hat, axis=-1, keepdims=True)     # cos(x, v)
        tangent = v_unit - proj * x_hat                          # bends each step
        tnorm = np.linalg.norm(tangent, axis=-1, keepdims=True)
        t_hat = tangent / np.where(tnorm < _EPS, 1.0, tnorm)
        x_new = r0 * (cos_d * x_hat + sin_d * t_hat)             # on-shell rotation
        # Positions with a vanishing tangent (already aligned) do not move.
        moved = tnorm >= _EPS
        x = np.where(moved, x_new, x)
    return x


def angle_between(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Angle (radians) between ``a`` and ``b`` along ``axis`` — a geometry probe."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=axis, keepdims=True)
    nb = np.linalg.norm(b, axis=axis, keepdims=True)
    cos = np.sum(a * b, axis=axis, keepdims=True) / (na * nb + _EPS)
    return np.arccos(np.clip(cos, -1.0, 1.0)).squeeze(axis)


def relative_offshell(endpoint: np.ndarray, h: np.ndarray) -> np.ndarray:
    """``| ||endpoint|| - ||h|| | / ||h||`` per position — the off-shell budget N5.

    Zero for a norm-preserving (curved) step; positive for the straight chord,
    which inflates the norm. This is the leading indicator the composite prices
    under ``lambda_geo`` (CLAUDE.md section 6) and the quantity this lesson claims
    the curved path drives to ~0.
    """
    endpoint = np.asarray(endpoint, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    hn = np.linalg.norm(h, axis=-1)
    en = np.linalg.norm(endpoint, axis=-1)
    return np.abs(en - hn) / (hn + _EPS)


# =========================================================================== #
# 2. The steering hook (torch) — the same geometry, applied to the live
#    residual stream during generation. One class, two modes (curved flag), so
#    the straight baseline and the curved path steer identical positions and the
#    ONLY difference is the path geometry.
# =========================================================================== #
class CurveballContext:
    """Context manager that steers one residual layer along a straight OR curved path.

    On ``__enter__`` it registers a forward hook on ``residual_layers[layer]`` that
    rewrites the layer's output hidden state; on ``__exit__`` it removes the hook,
    restoring the model exactly (steering leaves no trace). A forward-pre-hook
    stashes ``input_ids`` so we can skip special tokens (BOS / ``<start_of_turn>``
    / pad), exactly as lesson 2 does — steering those derails formatting for no gain.

    Modes (chosen by ``curved``):
      curved=False : straight relative-add ``h + alpha*||h||*v_unit`` (lesson 2).
      curved=True  : great-circle rotation of ``h`` toward ``v_unit`` by ``alpha``
                     radians, integrated in ``n_steps`` (norm-preserving arc).

    Parameters mirror lesson 2's ``SteeringContext`` plus ``curved`` / ``n_steps``.
    """

    def __init__(
        self,
        model: nn.Module,
        vector: "np.ndarray | torch.Tensor",
        layer: int,
        alpha: float,
        curved: bool = True,
        n_steps: int = 8,
        special_ids: "set[int] | None" = None,
    ) -> None:
        self.model = model
        layers = residual_layers(model)
        self.layer = max(0, min(int(layer), len(layers) - 1))
        self.alpha = float(alpha)
        self.curved = bool(curved)
        self.n_steps = int(n_steps)
        self._vector_in = vector
        self.special_ids = special_ids
        self._handles: list[Any] = []
        self._last_input_ids: torch.Tensor | None = None
        self._v_unit: torch.Tensor | None = None

    # -- helpers ----------------------------------------------------------- #
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
        """forward-pre-hook: stash real 2-D integer token ids for the mask.

        Floats (the CPU micro-test feeds a float tensor) are rejected, keeping the
        self-test on the "steer every position" path.
        """
        ids = kwargs.get("input_ids", None)
        if ids is None and args and torch.is_tensor(args[0]):
            ids = args[0]
        if torch.is_tensor(ids) and ids.dim() == 2 and not ids.is_floating_point():
            self._last_input_ids = ids
        else:
            self._last_input_ids = None

    def _target(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the steered hidden state for the whole [batch, seq, hidden] block."""
        v = self._v_unit
        if not self.curved:
            # Straight relative-add: chord of length alpha*||h|| along v_unit.
            r = h.norm(dim=-1, keepdim=True)
            return h + self.alpha * r * v

        # Curved: great-circle rotation toward v_unit by alpha radians, n_steps.
        r0 = h.norm(dim=-1, keepdim=True)                    # local shell radius
        dtheta = self.alpha / max(self.n_steps, 1)
        cos_d = float(np.cos(dtheta))
        sin_d = float(np.sin(dtheta))
        x = h
        for _ in range(self.n_steps):
            xn = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
            x_hat = x / xn
            proj = (v * x_hat).sum(dim=-1, keepdim=True)     # cos(x, v)
            tangent = v - proj * x_hat                        # re-aimed each step
            tnorm = tangent.norm(dim=-1, keepdim=True)
            t_hat = tangent / tnorm.clamp_min(_EPS)
            x_new = r0 * (cos_d * x_hat + sin_d * t_hat)
            moved = (tnorm >= _EPS).to(x.dtype)              # aligned positions stay
            x = moved * x_new + (1.0 - moved) * x
        return x

    def _steer_hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output               # [batch, seq, hidden]

        target = self._target(h)

        # Skip special tokens when we can see the ids and they line up in length:
        # keep h unchanged there, use the steered target everywhere else.
        ids = self._last_input_ids
        if ids is not None and ids.shape[1] == h.shape[1]:
            special = torch.tensor(sorted(self.special_ids), device=ids.device)
            steer = ~torch.isin(ids, special)               # [batch, seq] bool
            mask = steer.unsqueeze(-1).to(h.dtype)
            h_new = mask * target + (1.0 - mask) * h
        else:
            h_new = target

        if is_tuple:
            return (h_new, *output[1:])
        return h_new

    # -- context-manager protocol ------------------------------------------ #
    def __enter__(self) -> "CurveballContext":
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        v = self._vector_in
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        v = v.detach().to(device=device, dtype=dtype).reshape(-1)
        norm = v.norm()
        self._v_unit = v / norm if norm > 0 else v          # steer on the unit dir

        if self.special_ids is None:
            self.special_ids = self._default_special_ids()

        target = residual_layers(self.model)[self.layer]
        # prepend=True: the steering edit runs before any downstream probe hook, so
        # a probe (and the next layer) observes the STEERED residual.
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


# =========================================================================== #
# 3. Generation convenience — one call to steer a prompt straight OR curved.
# =========================================================================== #
@torch.no_grad()
def curveball_generate(
    model: Any,
    tok: Any,
    prompt: str,
    vector: "np.ndarray | torch.Tensor | None",
    layer: int,
    alpha: float,
    curved: bool = True,
    n_steps: int = 8,
    max_new_tokens: int = 48,
) -> str:
    """Greedy, chat-templated generation, steered inside a :class:`CurveballContext`.

    Returns ONLY the newly generated text. If ``vector`` is None or ``alpha == 0``
    this is a plain unsteered baseline. Mirrors lesson 2's ``generate`` but routes
    the residual edit through the curved/straight hook here.
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
            do_sample=False,           # greedy: deterministic gates
            num_beams=1,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None
                          else tok.eos_token_id),
        )

    steering = vector is not None and alpha != 0.0
    if steering:
        special = set(getattr(tok, "all_special_ids", []) or [])
        with CurveballContext(model, vector, layer, alpha, curved, n_steps, special):
            out = _run()
    else:
        out = _run()

    new_tokens = out[0][prompt_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# =========================================================================== #
# CPU self-test — NO model download. Verifies the curved-path geometry and that
# the hook steers then restores exactly.
# Run: python -m steering_tutorials.curveball.curveball
# =========================================================================== #
def _self_test() -> None:
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    hidden = 16
    alpha = 0.6
    n_steps = 8

    # A state h and a target direction v that are NOT aligned, so both paths move.
    h = rng.normal(size=hidden)
    v = rng.normal(size=hidden)
    v_unit = _unit(v)
    r0 = float(np.linalg.norm(h))

    straight = straight_endpoint(h, v_unit, alpha)
    curved = curveball_endpoint(h, v_unit, alpha, n_steps)

    # (a) Identity: alpha=0 (or n_steps=0) is a no-op for the curved path.
    assert np.allclose(curveball_endpoint(h, v_unit, 0.0, n_steps), h)
    assert np.allclose(curveball_endpoint(h, v_unit, alpha, 0), h)

    # (b) The curved arc PRESERVES the norm; the straight chord INFLATES it.
    assert abs(np.linalg.norm(curved) - r0) < 1e-6 * r0, "curved path left the shell"
    assert np.linalg.norm(straight) > r0 + 1e-6, "straight chord should inflate ||h||"

    # (c) Off-shell budget: curved ~0, and strictly less than straight.
    off_c = float(relative_offshell(curved, h))
    off_s = float(relative_offshell(straight, h))
    assert off_c < 1e-6, f"curved off-shell should be ~0, got {off_c}"
    assert off_c < off_s, "curved must have a smaller off-shell displacement"

    # (d) Both paths INCREASE alignment with v (they do steer toward the target).
    cos0 = float(np.dot(_unit(h), v_unit))
    cos_c = float(np.dot(_unit(curved), v_unit))
    cos_s = float(np.dot(_unit(straight), v_unit))
    assert cos_c > cos0 and cos_s > cos0, "both paths should move toward v"

    # (e) The curved path rotates h by ~alpha radians (arc length = ||h||*alpha),
    #     matching the straight chord's length ||straight - h|| = alpha*||h||.
    rot = float(angle_between(h, curved))
    assert abs(rot - alpha) < 1e-3, f"curved rotation {rot:.4f} != alpha {alpha}"
    chord_len = float(np.linalg.norm(straight - h))
    assert abs(chord_len - alpha * r0) < 1e-6, "straight chord length != alpha*||h||"

    # (f) Batched geometry works: [batch, seq, hidden] with a single [hidden] v.
    H = rng.normal(size=(2, 3, hidden))
    C = curveball_endpoint(H, v_unit, alpha, n_steps)
    assert C.shape == H.shape
    norms_in = np.linalg.norm(H, axis=-1)
    norms_out = np.linalg.norm(C, axis=-1)
    assert np.allclose(norms_in, norms_out, atol=1e-6), "batched curve broke a norm"

    # (g) The per-step tangent is orthogonal to the current state (it bends the path).
    x_hat = _unit(h)
    proj = np.dot(v_unit, x_hat)
    tangent = v_unit - proj * x_hat
    assert abs(np.dot(_unit(tangent), x_hat)) < 1e-9, "tangent not orthogonal to x"

    # -- (h) The torch hook: steers a fake decoder, then removes its hook cleanly. --
    torch.manual_seed(0)

    class _Tiny(nn.Module):
        """Stand-in decoder: a ModuleList of Linear "blocks" (like lesson 2's test)."""

        def __init__(self, hid: int = 16, n: int = 3):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(hid, hid) for _ in range(n)])
            self.config = SimpleNamespace(
                hidden_size=hid, bos_token_id=2, eos_token_id=1, pad_token_id=0
            )

        def forward(self, x, **_kw):
            for blk in self.layers:
                x = blk(x)
            return x

    model = _Tiny(hidden, 3).eval()
    assert num_layers(model) == 3 and hidden_size(model) == hidden

    layer = 1
    cap: dict[str, torch.Tensor] = {}

    def probe(_m, _i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    ph = residual_layers(model)[layer].register_forward_hook(probe)
    x = torch.randn(2, 4, hidden)                 # float tensor -> "steer all"
    vt = torch.randn(hidden)
    try:
        model(x)
        base = cap["h"].clone()

        # Curved arm: norm preserved on the steered residual.
        with CurveballContext(model, vt, layer, alpha, curved=True, n_steps=n_steps):
            model(x)
        curved_h = cap["h"].clone()

        # Straight arm: same hook, curved=False -> relative-add (norm inflates).
        with CurveballContext(model, vt, layer, alpha, curved=False):
            model(x)
        straight_h = cap["h"].clone()

        model(x)                                   # after exit — hook removed
        after = cap["h"].clone()
    finally:
        ph.remove()

    assert (curved_h - base).norm().item() > 0, "curved hook did not steer"
    assert (straight_h - base).norm().item() > 0, "straight hook did not steer"
    # Curved keeps the per-position norm; straight (generally) does not.
    bn = base.norm(dim=-1)
    assert torch.allclose(curved_h.norm(dim=-1), bn, atol=1e-4), "curved hook off-shell"
    assert (straight_h.norm(dim=-1) - bn).abs().max().item() > 1e-4, \
        "straight hook should move off the shell"
    # Exact restoration on exit.
    assert (after - base).norm().item() == 0.0, "hook did not restore exactly"

    print("[self-test] OK - curved arc is norm-preserving + rotates by alpha; "
          "straight chord inflates the norm; hook steers then restores exactly.")


if __name__ == "__main__":
    _self_test()
