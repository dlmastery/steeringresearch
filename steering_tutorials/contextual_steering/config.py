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

import os
from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1-2. Its refusal
# behaviour was removed, so it complies with harmful prompts by default — which
# is exactly what lets us RE-INSTALL refusal from the outside and measure whether
# the contextual schedule keeps benign prompts untouched.
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
N_EXTRACT_PER_CLASS = 300  # build the vector + calibrate the schedule (was 200)
# Generation is the expensive part (RAM/VRAM + greedy decode), so the eval set is
# CAPPED below N_PER_CLASS. Raise it for a fuller (slower) run.
N_EVAL_PER_CLASS = 150  # was 60
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

# --- The GATE: WHICH per-prompt signal drives the schedule -------------------
# The schedule needs a per-instance "how harmful is this?" signal to scale alpha.
# Two are wired here and the run reports BOTH so the comparison is honest:
#
#   "diffmean_cos" : proj = cos(mean_pool(h) - benign_mean, v_unit) — the
#       diff-of-means cosine. This is the honest NEGATIVE baseline: per-instance
#       it barely separates the two classes at this layer (Cohen's d ~ 0.1;
#       harmful and benign both spread across the same cosines), so tau lands
#       ABOVE the harmful mean, ref <= tau, and the ramp collapses to a
#       degenerate hard step that cannot spare benign.
#   "probe"        : proj = the TRAINED lesson-1 probe's raw (pre-sigmoid) LOGIT.
#       The probe is exactly the LEARNED sensing vector CLAS (arXiv:2604.24693)
#       prescribes; it separates harmful/benign at AUC ~0.87 at this layer, so
#       the harmful logits sit well above the benign ones, ref > tau, the linear
#       ramp operates, and benign prompts (low logit) get alpha ~ 0. This is the
#       FIX. The sibling `flas` lesson reuses the same probe as its gate.
GATE = "probe"                            # headline gate -> results["arms"]/schedule
GATES_TO_RUN = ("diffmean_cos", "probe")  # calibrate + report both for comparison

# ref anchor for the ramp. None => ref = the MEAN harmful projection (so a
# typical harmful prompt reaches full ALPHA_BASE — the honest central anchor).
# Set to a percentile in (0,100] for a gentler ramp that only saturates on
# high-confidence-harmful prompts (e.g. 60.0). tau is always TAU_BENIGN_PERCENTILE
# of the benign projections; the probe's separation is what makes ref > tau hold.
REF_HARMFUL_PERCENTILE = None

# --- Generation --------------------------------------------------------------
# Short greedy completions: long enough to tell a refusal from compliance, short
# enough to run dozens of prompts on a laptop GPU.
MAX_NEW_TOKENS = 48

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
# The lesson-1 probe checkpoint reused as the "probe" gate (the LEARNED sensing
# vector). Same artifact the sibling `flas` lesson loads. None here would let
# ProbeGate resolve this exact path itself; we make it explicit for transparency.
PROBE_PATH = ROOT.parent / "hello_world" / "artifacts" / "probe.pt"
VECTOR_PATH = ARTIFACTS / "steering_vector.pt"
RESULTS_PATH = ARTIFACTS / "results.json"
COMPARISON_PNG = ARTIFACTS / "fixed_vs_contextual.png"
SCHEDULE_PNG = ARTIFACTS / "alpha_schedule.png"

ARTIFACTS.mkdir(exist_ok=True)
