"""geometry.py — leading-indicator geometry probes (pure torch).

These are the always-log geometry signals from CLAUDE.md §3 and the
high-dimensional sweep:

  - off-shell displacement Δ‖h‖      : how far steering moved the residual norm
  - effective rank                   : exp(entropy of normalised singular values)
  - participation ratio              : (Σσ²)² / Σσ⁴  (N3 at the injection layer)
  - cumulative norm budget ‖Δh‖/‖h‖  : the N5 norm budget

All functions are pure-torch and tested on FakeResidualLM activation batches.
The geometry penalties feed the composite (eval.composite, λ_geo term).
"""

from __future__ import annotations

import torch


def offshell_displacement(h_base: torch.Tensor, h_steer: torch.Tensor) -> float:
    """Δ‖h‖ — change in residual-stream norm caused by steering.

    Defined as the mean over positions of |‖h_steer‖ - ‖h_base‖| / ‖h_base‖,
    a relative, scale-free off-manifold leading indicator. Larger ⇒ pushed
    further off the activation shell (Rogue-Scalpel risk).

    h_base, h_steer : [..., dim] matching shapes.
    """
    base_norm = h_base.norm(dim=-1)
    steer_norm = h_steer.norm(dim=-1)
    rel = (steer_norm - base_norm).abs() / (base_norm + 1e-8)
    return float(rel.mean())


def singular_values(activations: torch.Tensor) -> torch.Tensor:
    """Singular values of a centered activation batch.

    activations : [n, dim] (any leading dims are flattened to rows).
    Returns a 1-D tensor of singular values (descending).
    """
    x = activations.reshape(-1, activations.shape[-1]).float()
    x = x - x.mean(dim=0, keepdim=True)
    # economy SVD; values are non-negative, descending.
    s = torch.linalg.svdvals(x)
    return s


def effective_rank(activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective rank = exp(Shannon entropy of the normalised singular spectrum).

    (Roy & Vetterli, 2007.) For a batch whose activations span exactly k
    orthogonal directions with equal energy, this returns ≈ k.

    activations : [n, dim].
    """
    s = singular_values(activations)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * (p + eps).log()).sum()
    return float(torch.exp(entropy))


def participation_ratio(activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Participation ratio PR = (Σλ)² / Σλ²  where λ = σ² are eigenvalues of the
    covariance (N3). Equals k for k equal-energy orthogonal directions.

    activations : [n, dim].
    """
    s = singular_values(activations)
    lam = s ** 2  # eigenvalues of the (unnormalised) covariance
    denom = (lam ** 2).sum()
    if float(denom) < eps:
        return 0.0
    return float((lam.sum() ** 2) / denom)


def norm_budget(deltas: torch.Tensor, h_base: torch.Tensor) -> float:
    """Cumulative ‖Δh‖/‖h‖ norm budget (N5).

    deltas  : [steps, ..., dim] OR [..., dim] — per-step residual edits.
    h_base  : [..., dim] the baseline activation whose norm we budget against.

    Returns Σ_steps ‖Δh_step‖ / ‖h_base‖, accumulated over the leading "steps"
    axis if present. With a single delta it reduces to ‖Δh‖/‖h‖.
    """
    base = h_base.norm(dim=-1).mean() + 1e-8
    if deltas.dim() == h_base.dim() + 1:
        # leading step axis
        per_step = deltas.reshape(deltas.shape[0], -1, deltas.shape[-1]).norm(dim=-1).mean(dim=-1)
        total = per_step.sum()
    else:
        total = deltas.norm(dim=-1).mean()
    return float(total / base)


def geometry_report(
    h_base: torch.Tensor,
    h_steer: torch.Tensor,
) -> dict[str, float]:
    """Convenience bundle: all geometry probes for a base/steered activation pair.

    h_base, h_steer : [n, dim] (or [batch, seq, dim]).
    """
    delta = h_steer - h_base
    return {
        "offshell_displacement": offshell_displacement(h_base, h_steer),
        "effective_rank_base": effective_rank(h_base),
        "effective_rank_steer": effective_rank(h_steer),
        "participation_ratio_base": participation_ratio(h_base),
        "participation_ratio_steer": participation_ratio(h_steer),
        "norm_budget": norm_budget(delta, h_base),
    }
