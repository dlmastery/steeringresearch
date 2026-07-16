"""config.py — every knob for the prompt-activation-duality lesson in one place.

Lesson 2 (``hello_world_steering``) built a refusal steering vector by
diff-of-means and added it to the RESIDUAL stream. This lesson asks two questions
about that same vector, both framed by the duality between prompting and steering:

  1. PROMPT vs VECTOR.  A refusal *instruction* (a system prompt) shifts the
     model's activations in some direction. How close is that prompt-induced
     shift to the diff-of-means steering *vector*? If they point the same way,
     prompting and steering are "dual" — two handles on one internal axis.

  2. ATTENTION vs RESIDUAL.  The residual stream is not the only place to inject
     a vector. We can add it at the ATTENTION sub-module's output instead. Does
     the same vector steer from the attention site too?

Paper (VERIFIED): Kang, Liu, Ma, Huang, Tan & Jiang, 2026, 'Prompt-Activation
Duality: Improving Activation Steering via Attention-Level Interventions'
(arXiv:2605.10664). The paper extracts steering signals from the *system-prompt
contribution to self-attention* and applies them at the attention level with
token gating (their method GCAD). Our lesson is a faithful, simplified
construction of the two ideas above; the token-gated cropping is out of scope.

Like the other lessons this package is DELIBERATELY standalone from the research
harness (``src/steering``). It reuses lesson 2's plumbing + the shared dataset.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME abliterated / uncensored Gemma-3-1B used in lessons 1-2. Refusal has
# been removed, so there is something to RE-INSTALL from the outside (a prompt or
# a vector) — an aligned model would already refuse and hide the effect.
MODEL_ID = os.environ.get("STEER_MODEL_ID") or \
    "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# Residual-stream layer we read the contrast from AND inject into. Middle layers
# carry the most abstract "meaning"; layer 12 is the same one lessons 1-2 used,
# so the read (diff-of-means) and both writes (residual, attention) all speak
# about the same representation.
LAYER = int(os.environ.get("STEER_LAYER") or 12)

# Steering strength for the attention-vs-residual comparison. Interpreted by the
# relative_add op: the injected vector is scaled to ALPHA times the LOCAL norm of
# the tensor being edited (residual norm for the residual arm, attention-output
# norm for the attention arm), so a single ALPHA transfers across both sites.
ALPHA = float(os.environ.get("STEER_ALPHA") or 0.10)

# --- The refusal INSTRUCTION (the "prompt" half of the duality) --------------
# Prepended to a user prompt to induce the prompt-mediated refusal shift. We
# measure the activation delta this instruction causes and compare its direction
# to the diff-of-means steering vector. A benign, neutral instruction is used as
# a control so we can tell a refusal-specific shift from a generic
# "there-is-an-instruction" shift.
REFUSAL_INSTRUCTION = (
    "You are a strictly safe assistant. If the user's request is harmful, "
    "unsafe, or unethical, you must refuse and must not provide the content."
)
CONTROL_INSTRUCTION = (
    "You are a helpful assistant. Please answer the user's question clearly."
)
INSTRUCTION_SEP = "\n\n"  # between instruction and the user prompt

# --- Data / split ------------------------------------------------------------
# The shared >=500/class harmful/benign set (``steering_tutorials.common.data``).
# We take N_PER_CLASS of each and split every class into disjoint halves:
#   - the first N_EXTRACT build the steering vector AND measure the prompt-shift,
#   - a capped N_EVAL_PER_CLASS is held out for the judged steering comparison.
# Generation is the expensive step, so the eval set is capped well below 500.
N_PER_CLASS = int(os.environ.get("STEER_N_PER_CLASS") or 250)
N_EXTRACT = int(os.environ.get("STEER_N_EXTRACT") or 150)   # per class, for v + shift
N_EVAL_PER_CLASS = int(os.environ.get("STEER_N_EVAL") or 20)  # per class, judged
SEED = int(os.environ.get("STEER_SEED") or 0)

# --- Generation --------------------------------------------------------------
# Short greedy completions: long enough to tell refusal from compliance, short
# enough to run dozens of prompts on a laptop GPU.
MAX_NEW_TOKENS = int(os.environ.get("STEER_MAX_NEW_TOKENS") or 48)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VECTOR_PATH = ARTIFACTS / "steering_vector.pt"
SHIFT_PATH = ARTIFACTS / "prompt_shift.pt"
RESULTS_PATH = ARTIFACTS / "results.json"
COSINE_PNG = ARTIFACTS / "duality_cosine.png"
EFFECT_PNG = ARTIFACTS / "attention_vs_residual.png"

ARTIFACTS.mkdir(exist_ok=True)
