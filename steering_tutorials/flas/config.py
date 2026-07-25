"""config.py — every knob for the FLAS lesson (Flow-based Activation Steering).

Lesson 2 (``hello_world_steering``) added ONE fixed diff-of-means vector to the
residual stream. Lesson 3 (``reft_r1``) replaced it with a LEARNED rank-1 edit
``h' = h + r_unit*((w·h+b) - r_unit·h)``. Both are *displacements* applied in a
single shot: the intervention is a function ``h -> h'`` evaluated once.

FLAS reframes steering as *transport*. Instead of a fixed vector or a one-shot
learned edit, we learn a **concept-conditioned velocity field** ``v_theta(h,t,c)``
and steer by INTEGRATING a flow ODE from the unsteered activation to its steered
position::

    h' = h + integral_0^T v_theta(phi_t(h), t, c) dt          (Euler, n_steps)

Two properties fall out that neither fixed vectors nor rank-1 edits give:

  * **Flow-time ``T`` is a continuous, zero-shot STRENGTH dial.** Integrate to a
    smaller ``T`` and you transport the activation less far along the SAME learned
    trajectory — no retraining, no per-alpha sweep of a raw magnitude.
  * **One field handles many concepts.** The velocity is conditioned on a concept
    embedding ``c`` (the mean-activation "ConceptEncoder"), so a single trained
    ``v_theta`` steers toward any concept whose exemplars you can encode.

  FLAS — Flow-based Activation Steering (github.com/flas-ai/FLAS) [UNVERIFIED] —
    the velocity-field-over-activations steering method this lesson reproduces at
    laptop scale.
  Lipman et al. 2023, 'Flow Matching for Generative Modeling'
    (arXiv:2210.02747) [UNVERIFIED] — the flow-matching / continuous-time
    transport framing the velocity field is trained under.
  Liu et al. 2023, 'Flow Straight and Fast: Rectified Flow'
    (arXiv:2209.03003) [UNVERIFIED] — rectified flow, the straight-line-transport
    special case that makes few-step Euler integration accurate.

Like every lesson in ``steering_tutorials`` this package is DELIBERATELY
standalone: it does not import from ``src/steering``. It reuses only the lesson-2
model plumbing (``hello_world_steering.model_utils``); everything else is local.

v2 — THE DIAL WAS MIS-SCALED (2026-07-25). SUPERSEDES the v1 numbers.
----------------------------------------------------------------------
v1 defined the transport target with the RAW diff-of-means, ``h1 = h0 + delta_c``,
and integrated an ABSOLUTE Euler step ``x <- x + dt*v``. Measured on the shipped
``artifacts/flow.pt`` at layer 12 of Gemma-3-1B: ``||delta_c|| = 561`` (sexual) /
``474`` (violence) against a mean activation norm ``||h|| ~= 5.0e3``. So the v1
flow-time grid ``T in {0, 0.5, 1.0, 1.5, 2.0}`` mapped to RELATIVE displacements
``{0, 5.6%, 11.2%, 16.9%, 22.5%}`` of ``||h||`` — i.e. every point except ``T=0``
sat AT or PAST this course's coherence cliff (lesson 2 sweeps alpha only up to
0.15, and 0.10 is already its practical top). The sub-cliff regime ``T < 0.05``
was never sampled. The v1 sweep therefore dialled GIBBERISH (0.28 -> 0.75), not
the concept: it measured the coherence cliff, which is not what the lesson claims.

Three changes make ``T`` a meaningful dial:

  1. **Norm-relative transport** (:data:`NORM_RELATIVE`, default on). The trainer
     regresses the field onto the UNIT direction ``delta_hat = delta_c/||delta_c||``
     over the interpolant ``h_t = h0 + t*TRAIN_T_MAX*||h0||*delta_hat``, and
     :func:`flow.integrate_flow` takes norm-relative Euler steps
     ``x <- x + dt*||x||*v``. Since ``||v|| ~= 1`` by construction, integrating to
     flow-time ``T`` displaces by ``~= T*||h||`` — so **T is now numerically the
     same dial as lesson 2's ``relative_add`` alpha** and the two lessons' curves
     are directly comparable. Set ``FLAS_NORM_RELATIVE=0`` for the v1 behaviour.
  2. **A grid over the informative regime** — :data:`T_SWEEP` now spans
     ``{0, 0.02, 0.05, 0.10, 0.15}`` (sub-cliff through cliff), mirroring lesson 2.
  3. **A special-token guard** (:data:`SKIP_SPECIAL`, default on). v1's
     ``FlowContext`` transported BOS / ``<start_of_turn>`` positions, which lesson
     2's ``SteeringContext`` explicitly refuses to touch ("steering them tends to
     derail formatting for no behavioral gain"). Those positions carry outsized
     residual norms, so an absolute displacement wrecked them. Off with
     ``FLAS_SKIP_SPECIAL=0``.

The v1 numbers in ``artifacts/results.json`` and the README are SUPERSEDED; the
shipped ``artifacts/flow.pt`` is a v1 (raw-delta) field and must be RETRAINED
before a v2 eval — ``run_flas`` refuses to mix the two conventions.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1-3, so the flow
# is learned on the same residual stream at the same representational depth.
#
# Cross-scale check: STEER_MODEL_ID + STEER_LOAD_4BIT let this lesson run on a
# LARGER model (e.g. Gemma-3-4B abliterated) in 4-bit to test whether a 1B
# negative is a capacity artifact. No env vars set -> IDENTICAL to before (1B, bf16).
MODEL_ID = os.environ.get(
    "STEER_MODEL_ID", "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
)
# "1" -> load_model() quantizes to 4-bit (bitsandbytes nf4) so a 4B model fits
# the RAM-constrained host. Default off -> unchanged bf16 path.
LOAD_4BIT = os.environ.get("STEER_LOAD_4BIT", "0") == "1"

# Which residual-stream layer we hook the flow onto — the same middle-ish layer
# lessons 1-3 read/wrote, where abstract concepts live cleanly and linearly.
LAYER = 12

# --- The flow integrator -----------------------------------------------------
# Number of explicit-Euler steps used to integrate dx/dt = v(x,t,c) from t=0..T.
# 8 steps is enough for the near-straight (rectified-flow) trajectories this
# lesson targets; more steps only refine an already-short transport.
N_STEPS = int(os.environ.get("FLAS_N_STEPS", "8"))

# --- The strength dial: NORM-RELATIVE transport (the v2 fix) -----------------
# When on (default), flow-time T is a FRACTION of the residual norm: integrating
# to T displaces the activation by ~= T * ||h||, exactly like lesson 2's
# ``relative_add`` alpha. When off, the v1 convention returns: the field carries
# the raw ||delta_c|| and T is a multiplier on that raw magnitude.
NORM_RELATIVE = os.environ.get("FLAS_NORM_RELATIVE", "1") == "1"

# Never transport BOS / <start_of_turn> / pad positions (lesson-2 parity). These
# control positions carry outsized residual norms and steering them derails
# formatting for no behavioral gain.
SKIP_SPECIAL = os.environ.get("FLAS_SKIP_SPECIAL", "1") == "1"

# The training interpolant's far end, as a fraction of ||h0||. The field only
# ever SEES states within T <= TRAIN_T_MAX of the origin, so this must cover the
# top of T_SWEEP or the eval integrates the field off its training distribution.
TRAIN_T_MAX = float(os.environ.get("FLAS_TRAIN_T_MAX", "0.15"))

# Default flow-time — the STRENGTH dial. Under NORM_RELATIVE this is a fractional
# displacement, so 0.10 matches lesson 2's mid alpha (its top is 0.15).
_T_DEFAULT_FALLBACK = "0.10" if NORM_RELATIVE else "1.0"
T_DEFAULT = float(os.environ.get("FLAS_T_DEFAULT", _T_DEFAULT_FALLBACK))

# Flow-times swept in payoff 1 (the dose-response curve). T=0.0 is the true
# baseline (no transport). Under NORM_RELATIVE the grid spans the INFORMATIVE
# sub-cliff-through-cliff regime and mirrors lesson 2's ALPHAS exactly, with one
# extra point at 0.02 below the cliff. The v1 grid {0,0.5,1,1.5,2} is what the
# non-norm-relative fallback restores.
_T_SWEEP_FALLBACK = "0.0,0.02,0.05,0.10,0.15" if NORM_RELATIVE else "0.0,0.5,1.0,1.5,2.0"
T_SWEEP = [float(t) for t in os.environ.get("FLAS_T_SWEEP", _T_SWEEP_FALLBACK).split(",")]

# Greedy completion length — long enough to tell refusal from compliance.
MAX_NEW_TOKENS = int(os.environ.get("FLAS_MAX_NEW_TOKENS", "48"))

# --- Eval size caps (fit ONE foreground window on the RAM-constrained host) ---
# 0 = no cap: use every held-out prompt the pool affords. A positive value caps
# the eval prompts used PER CELL, which turns a ~1300-generation full pass into a
# capped SCREENING pass. Any capped run must be reported as screening-tier.
N_EVAL_CAP = int(os.environ.get("FLAS_N_EVAL", "0"))
# Same idea for the benign selectivity arm (its 500 prompts dominate the budget).
N_BENIGN_EVAL_CAP = int(os.environ.get("FLAS_N_BENIGN_EVAL", "0"))

# --- The velocity-field network ----------------------------------------------
# Hidden width of the velocity MLP. The field sees [h, time_emb, c] and outputs
# a velocity in R^hidden; 512 is a comfortable capacity for a 1B-model residual.
WIDTH = 512

# --- Training ----------------------------------------------------------------
# Adam LR, number of optimisation steps, prompts per minibatch, grad-norm clip.
# The training objective (owned by a separate trainer module) regresses the field
# onto the transport that carries unsteered activations to their steered target.
LR = 1e-3
STEPS = 400
BATCH = 16
GRAD_CLIP = 1.0

SEED = 0

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
FLOW_PATH = ARTIFACTS / "flow.pt"
RESULTS_PATH = ARTIFACTS / "results.json"

ARTIFACTS.mkdir(exist_ok=True)
