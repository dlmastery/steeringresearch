"""config.py — every knob for the CONTEXTUAL (input-adaptive) steering lesson.

Lesson 2 (``hello_world_steering``) added a refusal direction to the residual
stream with ONE fixed strength ``alpha`` for every prompt, and then only applied
it when a *separate* probe (lesson 1) said the prompt was harmful. That explicit
gate works, but it needs a second trained model.

This lesson removes the separate gate. The steering direction ``v`` already
carries the context signal we need: a harmful prompt's mean-pooled hidden state
points ALONG ``v``, a benign prompt's does not. So we scale the strength per
input by that alignment:

    proj_i  = cos(mean_pool(h_i @ LAYER), v_unit)          # in [-1, 1]
    alpha_i = ALPHA_BASE * relu((proj_i - tau) / (ref - tau))

Benign prompts (low or negative projection) get ``alpha_i ≈ 0`` — steered ~not
at all — while harmful prompts (high projection) get up to ``ALPHA_BASE``. The
direction gates itself; no probe required.

  Contextual Linear Activation Steering — CLAS (Hsu, Beaglehole, Radhakrishnan
    & Belkin, arXiv:2604.24693, Apr 2026) — the per-input-adaptive-strength idea;
    our cosine ramp is our own construction, not the paper's learned sensing
    vector.
  Rimsky et al. 2023, 'Steering Llama 2 via Contrastive Activation Addition'
    (arXiv:2312.06681) — the diff-of-means direction we reuse from lesson 2.

Like every tutorial here this package is DELIBERATELY standalone: it REUSES the
mechanical core of lesson 2 (model load, activation read, the steering hook, the
judge) and the shared ≥500/class dataset in ``steering_tutorials.common.data`` —
it does not import the research harness in ``src/steering``.
"""
from __future__ import annotations

from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1-2. Its refusal
# behaviour was removed, so it complies with harmful prompts by default — which
# is exactly what lets us RE-INSTALL refusal from the outside and measure whether
# the contextual schedule keeps benign prompts untouched.
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# Residual-stream layer we read the contrast from, read each prompt's projection
# from, AND write the steering vector into. Middle layers carry the most abstract
# "meaning", so "refuse this" is linearly separable here. Kept at 12 to match
# lessons 1-2 so all three speak about the same representation. Gemma-3-1B has 26.
LAYER = 12

# --- Data / split ------------------------------------------------------------
# The shared foundation ships ≥500 harmful + ≥500 benign prompts. We load the
# full budget and split each class into disjoint halves:
#   - the first N_EXTRACT_PER_CLASS build the steering vector (diff-of-means) AND
#     calibrate the contextual schedule (tau/ref, see below),
#   - a capped N_EVAL_PER_CLASS from the held-out remainder is generated on and
#     judged. Extraction and evaluation stay disjoint so we never grade the
#     vector (or the schedule) on the prompts that defined it.
N_PER_CLASS = 500          # >= 500/class from common.data — the whole point
N_EXTRACT_PER_CLASS = 200  # build the vector + calibrate the schedule
# Generation is the expensive part (RAM/VRAM + greedy decode), so the eval set is
# CAPPED well below N_PER_CLASS. Raise it for a fuller (slower) run.
N_EVAL_PER_CLASS = 60
SEED = 0

# --- Steering strength -------------------------------------------------------
# ALPHA_BASE is the FIXED strength the baseline arm uses for EVERY prompt, and
# the CEILING the contextual arm ramps up to for a maximally-aligned prompt.
# Interpreted by ``model_utils`` under ``operation="relative_add"``: the injected
# vector is scaled to ``alpha`` times the local residual-stream norm. We keep it
# modest (0.10) because too much steering tips coherent refusals into gibberish.
ALPHA_BASE = 0.10

# --- The contextual schedule -------------------------------------------------
#   alpha_i = ALPHA_BASE * clip( relu((proj_i - tau) / (ref - tau)), 0, CAP_MULT )
#
# ``tau`` is the projection below which we do NOT steer (the benign floor) and
# ``ref`` is the projection at which we reach full ALPHA_BASE (the harmful
# anchor). Cosines of mean-pooled activations against a diff-of-means direction
# are typically small in magnitude, so the literal paper choice ref = 1.0 would
# under-steer everything. Instead we CALIBRATE tau and ref on the EXTRACT split
# (never the eval split): tau = a high percentile of the benign projections,
# ref = the mean harmful projection. This is the "documented smooth schedule"
# the paper permits; set CALIBRATE = False to fall back to the literal
# relu((proj - TAU_FALLBACK) / (1 - TAU_FALLBACK)) form.
CALIBRATE = True
TAU_BENIGN_PERCENTILE = 90.0   # tau = this percentile of extract-benign projections
TAU_FALLBACK = 0.0             # used when CALIBRATE = False (ref then = 1.0)
REF_FALLBACK = 1.0
# Cap the ramp multiplier so an unusually-aligned prompt cannot be steered past
# ~ALPHA_BASE (which would risk gibberish). 1.0 == never exceed the fixed alpha.
CAP_MULT = 1.0

# --- Generation --------------------------------------------------------------
# Short greedy completions: long enough to tell a refusal from compliance, short
# enough to run dozens of prompts on a laptop GPU.
MAX_NEW_TOKENS = 48

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VECTOR_PATH = ARTIFACTS / "steering_vector.pt"
RESULTS_PATH = ARTIFACTS / "results.json"
COMPARISON_PNG = ARTIFACTS / "fixed_vs_contextual.png"
SCHEDULE_PNG = ARTIFACTS / "alpha_schedule.png"

ARTIFACTS.mkdir(exist_ok=True)
