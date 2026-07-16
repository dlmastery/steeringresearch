"""config.py — every knob for the CURVEBALL (nonlinear-path) steering lesson.

Lesson 2 (``hello_world_steering``) added a fixed refusal direction ``v`` to the
residual stream as a single straight step ``h -> h + alpha*||h||*v_unit``. That
step is a CHORD: it leaves the local activation-norm shell (it inflates ``||h||``),
which the Curveball paper links to the "geometric distortion" that makes big linear
pushes behave inconsistently and tip into gibberish.

This lesson keeps the SAME steering budget but spends it as a CURVE instead of a
chord. We rotate ``h`` toward ``v`` along a great-circle geodesic on the sphere of
radius ``||h||`` — an equal-length ARC that stays ON the shell (zero net norm
inflation). Both arms share ONE knob, ``ALPHA``, so the only thing that differs is
the geometry of the path (straight vs curved) — a clean single-variable test.

    straight :  h + ALPHA*||h||*v_unit                 (chord; norm inflates)
    curved   :  rotate h toward v_unit by ALPHA radians (arc; norm preserved)

The matched quantity is the STEERING BUDGET: the straight chord length ``ALPHA*||h||``
equals the curved arc length ``||h||*ALPHA``. Whether an equal budget spent as an
arc installs refusal as strongly AND with less gibberish is the empirical question
this lesson measures (see run_curveball.py).

  Raval, Song, Wu, Harrasse, Phillips, Barez & Abdullah 2026, 'Curveball Steering:
    The Right Direction To Steer Isn't Always Linear' (arXiv:2603.09313) — the
    thesis that activation spaces are locally curved so geometry-aware, nonlinear
    steering beats a global straight-line push. Our great-circle path is our own
    construction inspired by that thesis, not the paper's kernel-PCA method.
  Panickssery (Rimsky) et al. 2023, 'Steering Llama 2 via Contrastive Activation
    Addition' (arXiv:2312.06681) — the straight diff-of-means step we reuse from
    lesson 2 as the baseline arm.

Like every tutorial here this package is DELIBERATELY standalone: it reuses the
lesson-2 model plumbing (``hello_world_steering.model_utils``), the lesson-2 CAA
vector (``steer_vector``) and judge, and the shared >=500/class dataset in
``steering_tutorials.common.data`` — it does NOT import the research harness in
``src/steering``.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name) or default)


# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1-3. Its refusal
# behaviour was removed, so it complies with harmful prompts by default — which is
# exactly what lets us RE-INSTALL refusal from the outside and compare how the
# straight vs curved paths pay for it.
MODEL_ID = _env_str("CURVEBALL_MODEL",
                    "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated")

# Residual-stream layer we read the contrast from AND steer at. Middle layers carry
# the most abstract "meaning", so "refuse this" is cleanly separable here. Kept at
# 12 to match lessons 1-3 so all speak about the same representation. Gemma-3-1B has 26.
LAYER = _env_int("CURVEBALL_LAYER", 12)

# --- Data / split ------------------------------------------------------------
# The shared foundation ships >=500 harmful + >=500 benign prompts. We load the
# full budget and split each class into disjoint parts: the first
# N_EXTRACT_PER_CLASS build the diff-of-means direction; a capped N_EVAL_PER_CLASS
# from the held-out remainder is generated on and judged. Extraction and evaluation
# stay disjoint so we never grade the vector on the prompts that defined it.
N_PER_CLASS = _env_int("CURVEBALL_N_PER_CLASS", 500)
N_EXTRACT_PER_CLASS = _env_int("CURVEBALL_N_EXTRACT", 150)
# Generation is the expensive part (RAM/VRAM + greedy decode), so the eval set is
# CAPPED well below N_PER_CLASS. Raise CURVEBALL_N_EVAL for a fuller (slower) run.
N_EVAL_PER_CLASS = _env_int("CURVEBALL_N_EVAL", 40)
SEED = _env_int("CURVEBALL_SEED", 0)

# --- Steering budget (the single shared knob) --------------------------------
# ALPHA is the ONE strength both arms use, so the only difference between them is
# straight-vs-curved geometry:
#   * straight arm: displacement magnitude = ALPHA * ||h||  (a chord along v_unit)
#   * curved  arm: total rotation angle   = ALPHA radians  (an equal-length arc)
# 0.6 is deliberately large enough to actually install refusal — and, for the
# straight chord, large enough to inflate ||h|| and risk gibberish. That risk is
# the whole point: the curved arc spends the SAME budget without leaving the shell.
ALPHA = _env_float("CURVEBALL_ALPHA", 0.6)

# Number of great-circle sub-steps used to integrate the curved path. At each
# sub-step the effective push direction is re-aimed at the tangent toward v, so the
# path bends to keep following the geodesic; more steps trace the true great circle
# more faithfully. 8 is plenty for the modest rotations this lesson uses.
N_CURVE_STEPS = _env_int("CURVEBALL_STEPS", 8)

# --- Generation --------------------------------------------------------------
# Short greedy completions: long enough to tell a refusal from compliance, short
# enough to run dozens of prompts on a laptop GPU.
MAX_NEW_TOKENS = _env_int("CURVEBALL_MAX_NEW_TOKENS", 48)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VECTOR_PATH = ARTIFACTS / "steering_vector.pt"
RESULTS_PATH = ARTIFACTS / "results.json"
COMPARISON_PNG = ARTIFACTS / "straight_vs_curved.png"
GEOMETRY_PNG = ARTIFACTS / "offshell_displacement.png"

ARTIFACTS.mkdir(exist_ok=True)
