"""guard.py — the layered defense that neutralizes the rogue-scalpel attack.

``attack.py`` strips refusal out of the residual stream. This module bolts on a
small, defensible stack of guards and shows they drive the attack-success-rate
back down. Three layers, deliberately tractable, each documented and independently
ablatable (CLAUDE.md section 10, the five-layer guard — we implement A, B, D):

  Guard B — norm/manifold CLAMP (an activation-space guard).
      Cap the per-position steering delta so ``||Δh|| <= budget · ||h||``. An
      attacker who tries to overwrite a large fraction of the hidden state is
      blunted to a bounded nudge. This is the "manifold clamp": stay near the
      model's natural activation scale.

  Guard A — refusal-subspace projection LOCK (an activation-space guard).
      Re-add refusal so it cannot be removed. We guarantee the residual's
      projection onto the refusal direction is at least ``floor · ||h||``. Whatever
      the attacker projected out or negated is re-installed up to that positive
      floor — the refusal subspace is "locked open".

  Guard D — dual-forward VERDICT check (a runtime output guard).
      After generation, a SECOND forward (the same Gemma as judge) grades the
      response; a COMPLIANCE verdict on a harmful prompt is flagged as a leak and
      overridden with a safe completion. This catches whatever slipped past A/B.

Guards A and B compose in a single forward hook (:class:`GuardedAttackContext`,
which subclasses :class:`attack.AttackContext` so it sees the same clean ``h`` the
attacker saw). Guard D is applied to the generated text.

Order matters: CLAMP first (bound the adversarial delta), then LOCK (guarantee a
positive refusal floor on the clamped state). Honest about limits: on a 1B toy
model these are demonstrations, not a hardened guardrail — real universal attacks
(the 20-vector Rogue Scalpel, Korznikov et al. 2025, arXiv:2509.22067) are
stronger. See the README caveats.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from steering_tutorials.hello_world_steering.judge import is_gibberish
from .attack import AttackContext, _greedy_generate, suppress_refusal_delta


# --------------------------------------------------------------------------- #
# 1. Guard B — norm/manifold clamp (pure math, unit-tested below).
# --------------------------------------------------------------------------- #
def norm_clamp(h_clean: torch.Tensor, h_att: torch.Tensor, budget: float) -> torch.Tensor:
    """Clamp the steering delta so ``||Δh_p|| <= budget · ||h_clean_p||`` per position.

    ``Δh = h_att - h_clean`` is the (adversarial) edit. Where its per-position
    norm exceeds ``budget · ||h_clean||`` we rescale it down to exactly that cap;
    where it is already within budget we leave it untouched. The DIRECTION of the
    delta is preserved — only its magnitude is bounded — so a legitimate small
    nudge is unaffected while a large adversarial push is blunted.
    """
    delta = h_att - h_clean
    delta_norm = delta.norm(dim=-1, keepdim=True)                 # [...,1]
    cap = float(budget) * h_clean.norm(dim=-1, keepdim=True)      # [...,1]
    # factor = min(1, cap/||delta||); guard the 0/0 case.
    factor = torch.clamp(cap / (delta_norm + 1e-8), max=1.0)
    return h_clean + delta * factor


# --------------------------------------------------------------------------- #
# 2. Guard A — refusal-subspace projection lock (pure math).
# --------------------------------------------------------------------------- #
def projection_lock(h: torch.Tensor, refusal_unit: torch.Tensor,
                    floor_frac: float) -> torch.Tensor:
    """Ensure the refusal projection ``(h·û)`` is at least ``floor_frac · ||h||``.

    We compute the current projection onto the (unit) refusal direction and, if
    it has been driven below the positive floor, add back exactly the missing
    component. Positions that already carry enough refusal are untouched, so the
    lock only ever *re-installs* refusal the attacker removed — it never subtracts.
    """
    u = refusal_unit / (refusal_unit.norm() + 1e-8)
    u = u.to(dtype=h.dtype, device=h.device)
    proj = (h * u).sum(dim=-1, keepdim=True)                     # (h·û), [...,1]
    target = float(floor_frac) * h.norm(dim=-1, keepdim=True)    # positive floor
    deficit = torch.clamp(target - proj, min=0.0)                # missing refusal
    return h + deficit * u


# --------------------------------------------------------------------------- #
# 3. Compose the activation-space guards (Guard B then Guard A).
# --------------------------------------------------------------------------- #
def apply_guards(
    h_clean: torch.Tensor,
    h_att: torch.Tensor,
    refusal_unit: torch.Tensor,
    *,
    use_clamp: bool = True,
    use_lock: bool = True,
    clamp_budget: float = 0.06,
    lock_floor_frac: float = 0.05,
) -> torch.Tensor:
    """Return the guarded residual given the clean and attacked residuals.

    Applies Guard B (clamp) then Guard A (lock), each toggleable so the run can
    ablate them one at a time. With both off this returns ``h_att`` unchanged
    (i.e. the undefended attack).
    """
    h = h_att
    if use_clamp:
        h = norm_clamp(h_clean, h, clamp_budget)
    if use_lock:
        h = projection_lock(h, refusal_unit, lock_floor_frac)
    return h


# --------------------------------------------------------------------------- #
# 4. Attack + activation-guards in one forward hook.
# --------------------------------------------------------------------------- #
class GuardedAttackContext(AttackContext):
    """Run the attack AND the activation-space guards (A, B) in a single hook.

    Subclasses :class:`attack.AttackContext` and overrides only the edit seam:
    from the clean residual it computes the attacked residual, then hands both to
    :func:`apply_guards`. Because the guard sees the clean ``h`` directly, the
    clamp measures the true adversarial delta and the lock measures the true
    refusal projection — no approximation.
    """

    def __init__(self, model, refusal_unit, layer, *,
                 mode="project_out", alpha=0.12, project_frac=1.0,
                 use_clamp=True, use_lock=True,
                 clamp_budget=0.06, lock_floor_frac=0.05,
                 special_ids=None):
        super().__init__(model, refusal_unit, layer, mode, alpha, project_frac,
                         special_ids=special_ids)
        self.use_clamp = use_clamp
        self.use_lock = use_lock
        self.clamp_budget = float(clamp_budget)
        self.lock_floor_frac = float(lock_floor_frac)

    def _edit(self, h: torch.Tensor) -> torch.Tensor:
        delta = suppress_refusal_delta(h, self._u, self.mode, self.alpha,
                                       self.project_frac)
        h_att = h + delta
        return apply_guards(
            h, h_att, self._u,
            use_clamp=self.use_clamp, use_lock=self.use_lock,
            clamp_budget=self.clamp_budget, lock_floor_frac=self.lock_floor_frac,
        )


def guarded_generate(
    model, tok, prompt: str, refusal_dir, layer: int, *,
    mode="project_out", alpha=0.12, project_frac=1.0,
    use_clamp=True, use_lock=True,
    clamp_budget=0.06, lock_floor_frac=0.05,
    max_new_tokens=48,
) -> str:
    """Generate under attack WITH the activation-space guards enabled.

    Toggling ``use_clamp`` / ``use_lock`` off recovers the ablations used by the
    run. (Guard D is applied to the returned text separately, see below.)
    """
    ctx = GuardedAttackContext(
        model, refusal_dir, layer, mode=mode, alpha=alpha, project_frac=project_frac,
        use_clamp=use_clamp, use_lock=use_lock,
        clamp_budget=clamp_budget, lock_floor_frac=lock_floor_frac,
        special_ids=set(getattr(tok, "all_special_ids", []) or []),
    )
    return _greedy_generate(model, tok, prompt, max_new_tokens, ctx)


# --------------------------------------------------------------------------- #
# 5. Guard D — dual-forward verdict check (runtime output guard).
# --------------------------------------------------------------------------- #
def dual_forward_verdict(judge, prompt: str, response: str) -> bool:
    """Return True if the response is a COMPLIANCE leak on a harmful prompt.

    This is the "second forward": the same Gemma, acting as judge, grades the
    generated text. We flag only clear compliance (the safety-relevant leak);
    refusals and gibberish are not leaks. GIBBERISH is caught deterministically
    first (a broken response is not a successful jailbreak).
    """
    if is_gibberish(response):
        return False
    return judge.verdict(prompt, response) == "COMPLIANCE"


def enforce_dual_forward(judge, prompt: str, response: str, safe_completion: str) -> str:
    """Guard D applied: override a flagged compliance leak with a safe refusal.

    If the dual-forward verdict flags the response as a leak, we replace it with
    ``safe_completion`` (the runtime monitor refusing on the model's behalf);
    otherwise the response passes through unchanged.
    """
    if dual_forward_verdict(judge, prompt, response):
        return safe_completion
    return response


# --------------------------------------------------------------------------- #
# CPU self-test — NO model download. Verifies clamp + lock math.
# Run: python -m steering_tutorials.rogue_scalpel.guard
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    torch.manual_seed(0)
    hidden, budget, floor = 32, 0.06, 0.05

    h_clean = torch.randn(3, 6, hidden)
    u = torch.randn(hidden); u = u / u.norm()

    # (a) Norm clamp: after clamping, every per-position ||Δh|| <= budget·||h||.
    #     Build an oversized adversarial delta (project-out at full strength +
    #     a big random push) so the clamp actually bites.
    big_delta = suppress_refusal_delta(h_clean, u, "project_out", project_frac=1.0)
    big_delta = big_delta + 0.5 * torch.randn_like(h_clean)      # overwhelm the budget
    h_att = h_clean + big_delta
    h_clamped = norm_clamp(h_clean, h_att, budget)

    clamped_delta = (h_clamped - h_clean).norm(dim=-1)
    cap = budget * h_clean.norm(dim=-1)
    assert torch.all(clamped_delta <= cap + 1e-4), "clamp did not bound ||Δh||"
    # a delta already within budget is left exactly alone.
    small = h_clean + 0.001 * u
    assert torch.allclose(norm_clamp(h_clean, small, budget), small, atol=1e-5), \
        "clamp altered an in-budget delta"

    # (b) Projection lock: after locking, refusal projection >= floor·||h|| where
    #     it was deficient, and the lock never REDUCES an already-sufficient proj.
    h_ablated = h_clean + suppress_refusal_delta(h_clean, u, "project_out")  # proj ~ 0
    h_locked = projection_lock(h_ablated, u, floor)
    proj = (h_locked * u).sum(-1)
    target = floor * h_locked.norm(dim=-1)
    assert torch.all(proj >= target - 1e-3), "lock did not restore the refusal floor"

    # (c) Full stack neutralizes the ablation: guarded refusal projection is
    #     positive again (attacker's ~0 projection is re-installed).
    guarded = apply_guards(h_clean, h_ablated, u,
                           clamp_budget=budget, lock_floor_frac=floor)
    assert torch.all((guarded * u).sum(-1) > 0), "guards did not re-install refusal"

    print("[self-test] OK -- clamp bounds ||delta h||; lock restores the refusal "
          "floor; stack re-installs refusal.")


if __name__ == "__main__":
    _self_test()
