"""config.py — every knob for the conditional-steering lesson in one place.

Lesson 1 (``steering_tutorials/hello_world``) trained a probe that READS harm
out of Gemma-3-1B's layer-12 activations. Lesson 2 does the WRITE half: it
builds a *refusal steering vector* by contrastive diff-of-means and adds it to
the residual stream to make the uncensored model refuse harmful prompts — but
only when the lesson-1 probe (the "gate") says the prompt is actually harmful.

The lesson in one sentence:
    Steering every prompt would break harmless requests, so we steer
    CONDITIONALLY — read the concept direction once, then only apply it when a
    lightweight gate fires — and let the same Gemma judge whether each output is
    a REFUSAL (it worked), COMPLIANCE (no effect), or GIBBERISH (we broke it).

Like lesson 1 this package is DELIBERATELY standalone: it does not import from
the research harness in ``src/steering``. Everything it needs lives under
``steering_tutorials/hello_world_steering/`` (plus the lesson-1 probe, reused
verbatim as the gate).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lesson 1. It has had its
# refusal behaviour removed, which is exactly why it makes a good demo: an
# aligned model already refuses harmful prompts, so there would be nothing to
# steer. Here we RE-INSTALL refusal from the outside, with an activation vector.
#
# Cross-scale check: STEER_MODEL_ID + STEER_LOAD_4BIT let this lesson run on a
# LARGER model (e.g. Gemma-3-4B abliterated) in 4-bit to test whether a 1B
# negative is a capacity artifact rather than a property of the method. With no
# env vars set the default is IDENTICAL to before (the 1B model in bf16).
MODEL_ID = os.environ.get(
    "STEER_MODEL_ID", "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
)
# When "1", load_model() quantizes to 4-bit (bitsandbytes nf4) so a 4B model
# fits the 16 GB VRAM / RAM-constrained host. Default off -> unchanged bf16 path.
LOAD_4BIT = os.environ.get("STEER_LOAD_4BIT", "0") == "1"

# Which residual-stream layer we read the contrast from AND write the steering
# vector into. Middle layers carry the most abstract "meaning", so a concept
# like "refuse this" lives here cleanly. Gemma-3-1B has 26 layers; 12 is a touch
# past the middle — the same layer the lesson-1 probe was trained on, so the
# gate (read) and the steer (write) speak about the same representation.
STEER_LAYER = 12

# Steering strengths to sweep in the UNCONDITIONAL arm. Interpreted by
# ``model_utils.generate`` under ``operation="relative_add"``: the injected
# vector is scaled to ``alpha`` times the residual-stream norm, so alpha is a
# fraction of "how much of the hidden state" we overwrite. 0.0 == no steering
# (the baseline). We keep the top end small (0.15) because too much steering
# tips coherent refusals into gibberish — the very failure the judge catches.
ALPHAS = [0.0, 0.05, 0.10, 0.15]

# --- Data / split ------------------------------------------------------------
# The shared >=500/class harmful/benign set (``steering_tutorials.common.data``)
# replaces the old 100-prompt JailbreakBench loader, so we can build the vector
# on a real sample and still hold out a credible eval split. We take N_PER_CLASS
# of each, then split every class into two disjoint halves:
#   - the first N_EXTRACT go into building the steering vector (diff-of-means),
#   - the rest are held out for EVALUATION (never seen during extraction).
# Keeping extraction and evaluation disjoint is what stops us from grading the
# vector on the very prompts that defined it.
N_PER_CLASS = 500
N_EXTRACT = 300         # per class, used only to build the vector (eval = 200/class)
SEED = 0

# --- Generation --------------------------------------------------------------
# Short greedy completions: long enough to tell a refusal from compliance,
# short enough to run dozens of prompts on a laptop GPU.
MAX_NEW_TOKENS = 48

# When choosing the single alpha for the CONDITIONAL arm we prefer the smallest
# alpha whose refusal rate is highest *before* gibberish takes over. A candidate
# alpha is disqualified if its gibberish rate exceeds this tolerance.
GIBBERISH_TOLERANCE = 0.20

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VECTOR_PATH = ARTIFACTS / "steering_vector.pt"
RESULTS_PATH = ARTIFACTS / "results.json"
RATES_PNG = ARTIFACTS / "rates_vs_alpha.png"
CONDITIONAL_PNG = ARTIFACTS / "conditional.png"

ARTIFACTS.mkdir(exist_ok=True)
