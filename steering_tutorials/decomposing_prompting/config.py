"""config.py -- every knob for the "decompose prompting" lesson in one place.

The question this lesson answers
--------------------------------
AxBench's honest bake-off (arXiv:2501.17148) found that *prompting* is a very
strong steering baseline -- often as strong as a learned intervention. Why?
Cheng & Kriegeskorte 2026 (arXiv:2606.03093, "Decomposing how prompting steers
behavior") give a mechanistic answer: a prompt reshapes the model's internal
representation toward the instructed task, and a large fraction of that reshaping
is a *shape-preserving map* -- a translation (plus rigid rotation / uniform
scaling) of the activation cloud. A translation of the residual stream is exactly
what an activation-steering vector *is*. So prompting is, in large part, a
steering vector the model applies to itself.

This lesson reproduces that logic at laptop scale. For a refusal instruction we:
  1. measure, per prompt, the layer-L activation delta prompting induces
        d(x) = act(prompt WITH the instruction) - act(prompt WITHOUT it)
  2. decompose d(x) against the diff-of-means refusal direction u into
        - the on-direction (steering-vector-like) component   <d(x), u> u
        - the off-direction residual                          d(x) - <d(x),u> u
     and measure how CONSISTENT d(x) is across prompts (mean pairwise cosine):
     a single shared translation would give near-perfect consistency; an
     input-dependent (affine / nonlinear) transform would not.
  3. WRITE check: steer the *unprompted* model with only the extracted
     translation component and judge how much of prompting's refusal effect a
     plain steering vector reproduces.

Like the other tutorials this package is standalone: it reuses lesson-2's
``hello_world_steering`` mechanics (model load, activation read, CAA vector,
steering hook, Qwen judge) and the shared >=500/class dataset, and imports
nothing from the research harness in ``src/steering``.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- The model we probe + steer ---------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used across the tutorial track.
# It rarely refuses on its own, which is what makes the effect of a refusal
# *instruction* (and of a refusal steering vector) large and easy to measure.
MODEL_ID = os.environ.get("STEER_MODEL_ID") or (
    "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated")

# Residual-stream layer we read the deltas from AND build the refusal direction
# at. Middle-ish layers carry the most abstract "task" structure; 12 matches the
# layer lesson-1/lesson-2 use, so every lesson speaks about the same slice.
STEER_LAYER = int(os.environ.get("DECOMP_LAYER") or 12)

# The refusal instruction whose activation footprint we decompose. Prepended to
# the user turn (Gemma's chat template folds "system" guidance into the first
# user turn, so a prepend is the faithful, template-safe way to add it).
REFUSAL_INSTRUCTION = (
    "You are a safety-conscious assistant. If a request is harmful, dangerous, "
    "unethical, or illegal, you must firmly refuse and must not provide the "
    "requested content."
)

# --- Data / split ------------------------------------------------------------
# The shared >=500/class harmful/benign set (``steering_tutorials.common.data``).
# We draw N_PER_CLASS of each, then split disjointly:
#   - the first N_EXTRACT/class build the diff-of-means refusal direction u,
#   - a held-out slice of harmful prompts is what we DECOMPOSE (n = N_DECOMP),
#   - a small held-out slice drives the generate/judge WRITE check (n = N_WRITE).
# Disjoint extract/analyse/write keeps us from grading a vector on its own inputs.
N_PER_CLASS = int(os.environ.get("DECOMP_N_PER_CLASS") or 250)
N_EXTRACT = int(os.environ.get("DECOMP_N_EXTRACT") or 150)   # per class -> u
N_DECOMP = int(os.environ.get("DECOMP_N_DECOMP") or 80)      # harmful, deltas
N_WRITE = int(os.environ.get("DECOMP_N_WRITE") or 24)        # harmful, judged
SEED = int(os.environ.get("DECOMP_SEED") or 0)

# --- Generation --------------------------------------------------------------
# Short greedy completions: long enough to tell a refusal from compliance, short
# enough to run a few dozen prompts on a laptop GPU.
MAX_NEW_TOKENS = int(os.environ.get("DECOMP_MAX_NEW_TOKENS") or 48)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "results.json"
DECOMP_PNG = ARTIFACTS / "decomposition.png"
RECOVERY_PNG = ARTIFACTS / "recovery.png"
DIRECTION_PATH = ARTIFACTS / "refusal_direction.pt"
# The extracted translation components (raw activation units) that the WRITE
# check steers with; saved so infer.py can reproduce prompting on one prompt.
RECON_PATH = ARTIFACTS / "reconstruction.pt"

ARTIFACTS.mkdir(exist_ok=True)
