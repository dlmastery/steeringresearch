"""Steering research harness — offline-first activation-steering experiment toolkit.

Modules:
    fakelm    — offline FakeResidualLM decoder stub (all unit tests run on this)
    model     — real Gemma loading (gated, graceful failure) + residual-layer helper
    hooks     — forward-hook residual interventions (add/rotate/project_out) + probe
    extract   — contrast-pair vector extraction (DiffMean, PCA), Fisher, vector bank
    geometry  — leading-indicator probes (Δ‖h‖, effective-rank, participation ratio, norm budget)
    eval      — the five measurement axes + Goodhart-resistant composite (fingerprinted)
    datasets  — tiny pinned offline dataset slices + loaders
    runner    — single-experiment executor (mirrors the FX runner conventions)
    dashboard — self-contained HTML runs dashboard

See CLAUDE.md (the project constitution) for the five axes (§3), the ladder (§4),
the composite (§6), and the state-file ledger (§12).
"""

__all__ = [
    "fakelm",
    "model",
    "hooks",
    "extract",
    "geometry",
    "eval",
    "datasets",
]

__version__ = "0.1.0"
