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
N_STEPS = 8

# Default flow-time — the STRENGTH dial. T=1.0 integrates the full learned
# trajectory; smaller T transports less far along the SAME path (zero-shot).
T_DEFAULT = 1.0

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
