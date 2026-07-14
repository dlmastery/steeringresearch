"""attack.py — the rogue scalpel: strip refusal out of the residual stream.

DEFENSIVE / EDUCATIONAL RED-TEAM. This module implements the *attack* half of the
lesson: it takes the refusal direction (the exact object lesson 2 *adds* to
install refusal) and, instead, *removes* it — jailbreaking an aligned model into
complying with harmful requests. The point is not to build a working exploit; it
is to stress-test our own guard (``guard.py``) and prove the guard neutralizes
this attack. Everything runs on one tiny local model; nothing here transfers to a
production system.

Why this works (the mechanism)
------------------------------
Arditi et al. 2024 ('Refusal in LLMs is Mediated by a Single Direction',
arXiv:2406.11717) show refusal is governed by ONE direction in activation space.
Lesson 2 exploits that constructively: add ``+v`` and the model refuses. The dual
is immediate — take the same ``v`` and either

  * project it OUT of the residual (directional ablation): ``h <- h - (h·û)û``,
    deleting the model's ability to "notice it should refuse", or
  * add the NEGATIVE direction: ``h <- h - α·||h||·û``, actively pushing away
    from refusal (the mirror image of lesson 2's relative-add).

The refusal direction itself is the CAA diff-of-means from lesson 2, reused
verbatim (Rimsky et al. 2023, arXiv:2312.06681).

Two components live here: the pure delta math (``suppress_refusal_delta``) and a
context manager (``AttackContext``) that installs it as a forward hook — the same
hook shape as lesson 2's ``SteeringContext``, so the guard can subclass it.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Reuse lesson 2's plumbing: the CAA diff-of-means (the refusal direction) and
# the layer-list helper. We do NOT reinvent them.
from steering_tutorials.hello_world_steering.model_utils import residual_layers
from steering_tutorials.hello_world_steering.steer_vector import extract_caa_vector


# --------------------------------------------------------------------------- #
# 1. The refusal direction — reuse lesson 2's diff-of-means verbatim.
# --------------------------------------------------------------------------- #
def refusal_direction(
    model: Any, tok: Any, harmful: list[str], benign: list[str], layer: int
) -> dict:
    """Return the CAA refusal direction at ``layer`` (a ``steer_vector`` dict).

    This is exactly lesson 2's steering vector — ``mean(harmful) −
    mean(benign)`` of the last-token residuals. Lesson 2 ADDS it; this lesson
    SUBTRACTS/ablates it. Same object, opposite sign. Keys of interest:
    ``v_unit`` (the unit refusal direction) and ``layer``.
    """
    return extract_caa_vector(model, tok, harmful, benign, layer)


# --------------------------------------------------------------------------- #
# 2. The attack math — pure, tensor-in/tensor-out (unit-testable, no model).
# --------------------------------------------------------------------------- #
def suppress_refusal_delta(
    h: torch.Tensor,
    refusal_unit: torch.Tensor,
    mode: str = "project_out",
    alpha: float = 0.12,
    project_frac: float = 1.0,
) -> torch.Tensor:
    """Return the residual-stream delta ``Δh`` that SUPPRESSES refusal.

    ``h`` is ``[..., hidden]`` (any leading dims); ``refusal_unit`` is the unit
    refusal direction ``[hidden]``. The returned ``Δh`` has the same shape as
    ``h`` and is meant to be added: ``h_attacked = h + Δh``.

      mode="project_out"  : ``Δh = -project_frac · (h·û) û``. With
          ``project_frac=1`` this deletes the entire refusal component of ``h``
          (directional ablation). The push is proportional to how much refusal
          is currently present, so it targets exactly the refusal subspace.
      mode="negative_add" : ``Δh = -alpha · ||h|| · û``. A fixed fraction of the
          local hidden norm in the anti-refusal direction — lesson 2's
          relative-add with a negative sign.
    """
    u = refusal_unit / (refusal_unit.norm() + 1e-8)          # defensively unit
    u = u.to(dtype=h.dtype, device=h.device)

    if mode == "project_out":
        proj = (h * u).sum(dim=-1, keepdim=True)             # (h·û), [...,1]
        return -float(project_frac) * proj * u
    if mode == "negative_add":
        per_pos_norm = h.norm(dim=-1, keepdim=True)          # ||h||, [...,1]
        return -float(alpha) * per_pos_norm * u
    raise ValueError(f"unknown attack mode {mode!r}")


# --------------------------------------------------------------------------- #
# 3. The attack as a forward hook — same shape as lesson 2's SteeringContext.
# --------------------------------------------------------------------------- #
class AttackContext:
    """Context manager that STRIPS refusal at one residual layer during a forward.

    On ``__enter__`` it hooks ``residual_layers[layer]`` and rewrites that
    layer's output ``h``; on ``__exit__`` it removes the hook, restoring the
    model exactly (the attack leaves no persistent trace — like lesson 2).

    Subclasses override :meth:`_edit` to change what happens to the clean
    residual; the base class applies the pure attack delta. ``guard.py`` uses
    this seam to run attack-then-guard in a single hook, so the guard always
    sees the same clean ``h`` the attacker saw.

    Special-token guard: we never edit BOS / ``<start_of_turn>`` / other control
    positions (editing them derails formatting for no behavioral gain), mirroring
    lesson 2. The mask is built from ``input_ids`` when the hook can see them.
    """

    def __init__(
        self,
        model: nn.Module,
        refusal_unit: "np.ndarray | torch.Tensor",
        layer: int,
        mode: str = "project_out",
        alpha: float = 0.12,
        project_frac: float = 1.0,
        special_ids: "set[int] | None" = None,
    ) -> None:
        self.model = model
        self.layer = int(layer)
        self.mode = mode
        self.alpha = float(alpha)
        self.project_frac = float(project_frac)
        self._u_in = refusal_unit
        self.special_ids = special_ids
        self._handles: list[Any] = []
        self._last_input_ids: torch.Tensor | None = None
        self._u: torch.Tensor | None = None          # unit refusal dir (model dtype/dev)

    # -- the edit seam: base = pure attack; guard subclass overrides this ------
    def _edit(self, h: torch.Tensor) -> torch.Tensor:
        """Map a clean residual ``h`` to the attacked residual (no guards)."""
        delta = suppress_refusal_delta(
            h, self._u, self.mode, self.alpha, self.project_frac
        )
        return h + delta

    # -- helpers (mirrors lesson 2) -------------------------------------------
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
        ids = kwargs.get("input_ids", None)
        if ids is None and args and torch.is_tensor(args[0]):
            ids = args[0]
        if torch.is_tensor(ids) and ids.dim() == 2 and not ids.is_floating_point():
            self._last_input_ids = ids
        else:
            self._last_input_ids = None

    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output           # [batch, seq, hidden]

        h_edited = self._edit(h)

        # Skip special tokens when the ids are visible and lengths line up.
        ids = self._last_input_ids
        if ids is not None and ids.shape[1] == h.shape[1]:
            special = torch.tensor(sorted(self.special_ids), device=ids.device)
            keep = ~torch.isin(ids, special)             # [batch, seq] bool
            mask = keep.unsqueeze(-1).to(h.dtype)
            h_edited = h + (h_edited - h) * mask          # edit only real tokens

        if is_tuple:
            return (h_edited, *output[1:])
        return h_edited

    # -- context-manager protocol ---------------------------------------------
    def __enter__(self) -> "AttackContext":
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        u = self._u_in
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)
        u = u.detach().to(device=device, dtype=dtype).reshape(-1)
        self._u = u / (u.norm() + 1e-8)

        if self.special_ids is None:
            self.special_ids = self._default_special_ids()

        target = residual_layers(self.model)[self.layer]
        self._handles.append(target.register_forward_hook(self._hook, prepend=True))
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


# --------------------------------------------------------------------------- #
# 4. Greedy generation helper — shared by attack_generate and guard.py.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _greedy_generate(model, tok, prompt: str, max_new_tokens: int, ctx) -> str:
    """Greedy, chat-templated generation inside a context manager ``ctx``.

    ``ctx`` is an ``AttackContext`` / ``GuardedAttackContext`` (or
    ``nullcontext()`` for a plain baseline). Returns ONLY the newly generated
    text. Mirrors lesson 2's ``generate`` so the comparison is apples-to-apples.
    """
    device = next(model.parameters()).device
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    prompt_len = ids.shape[1]
    with ctx:
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None
                          else tok.eos_token_id),
        )
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def attack_generate(
    model, tok, prompt: str, refusal_dir: "np.ndarray | torch.Tensor",
    alpha: float = 0.12, layer: int = 12,
    mode: str = "project_out", project_frac: float = 1.0,
    max_new_tokens: int = 48,
) -> str:
    """Generate under the rogue-scalpel attack (refusal STRIPPED). Returns text.

    ``refusal_dir`` is the unit refusal direction from :func:`refusal_direction`
    (``vec["v_unit"]``). With ``alpha == 0`` and no mode effect this reduces to a
    plain baseline generation (used for the un-attacked arm).
    """
    if mode == "negative_add" and alpha == 0.0:
        ctx: Any = nullcontext()                          # no-op: true baseline
    else:
        ctx = AttackContext(model, refusal_dir, layer, mode, alpha, project_frac,
                            special_ids=set(getattr(tok, "all_special_ids", []) or []))
    return _greedy_generate(model, tok, prompt, max_new_tokens, ctx)


# --------------------------------------------------------------------------- #
# CPU self-test — NO model download. Verifies the attack math + hook restore.
# Run: python -m steering_tutorials.rogue_scalpel.attack
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    from types import SimpleNamespace

    torch.manual_seed(0)

    # (1) Pure math: project_out deletes the refusal component; negative_add
    #     points against û with the right magnitude.
    hidden = 16
    u = torch.randn(hidden)
    u = u / u.norm()
    h = torch.randn(4, 7, hidden)

    d_proj = suppress_refusal_delta(h, u, mode="project_out", project_frac=1.0)
    h_ablated = h + d_proj
    resid_proj = (h_ablated * u).sum(-1)                  # should be ~0 everywhere
    assert resid_proj.abs().max().item() < 1e-4, "project_out left refusal behind"

    d_neg = suppress_refusal_delta(h, u, mode="negative_add", alpha=0.2)
    # delta is anti-parallel to û and has norm 0.2*||h|| per position.
    cos = (d_neg * u).sum(-1) / (d_neg.norm(dim=-1) + 1e-8)
    assert torch.allclose(cos, -torch.ones_like(cos), atol=1e-4), "negative_add not anti-refusal"
    ref = 0.2 * h.norm(dim=-1)
    assert torch.allclose(d_neg.norm(dim=-1), ref, atol=1e-4), "negative_add magnitude wrong"

    # (2) Hook installs the attack and restores the model exactly on exit.
    class _Tiny(nn.Module):
        def __init__(self, hid=16, n=3):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(hid, hid) for _ in range(n)])
            self.config = SimpleNamespace(
                hidden_size=hid, bos_token_id=2, eos_token_id=1, pad_token_id=0)

        def forward(self, x, **_kw):
            for blk in self.layers:
                x = blk(x)
            return x

    model = _Tiny(hidden if False else 16).eval()
    x = torch.randn(2, 5, 16)                             # float => "edit all"
    cap: dict[str, torch.Tensor] = {}

    def probe(_m, _i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    ph = residual_layers(model)[1].register_forward_hook(probe)
    try:
        model(x); base = cap["h"].clone()
        with AttackContext(model, u, layer=1, mode="project_out"):
            model(x)
        attacked = cap["h"].clone()
        model(x); after = cap["h"].clone()
    finally:
        ph.remove()

    assert (attacked - base).norm().item() > 0, "attack did not move the residual"
    # refusal component of the attacked residual is ablated toward 0.
    assert (attacked * u).sum(-1).abs().max().item() < (base * u).sum(-1).abs().max().item()
    assert (after - base).norm().item() == 0.0, "hook did not restore exactly"

    print("[self-test] OK — attack math correct; refusal ablated; hook restores exactly.")


if __name__ == "__main__":
    _self_test()
