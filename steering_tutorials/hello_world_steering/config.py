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

# --- WHERE the direction is EXTRACTED from (the extraction-source fix) --------
# THE BUG THIS FIXES. Until 2026-07-30 this lesson read the diff-of-means from
# the SAME abliterated model it steers. But abliteration surgically removes the
# refusal behaviour, so on that model harmful and benign prompts do NOT separate
# along a "refuse this" axis -- the model refuses neither. The difference of the
# two centroids is then a TOPIC direction (harmful subject matter vs benign
# subject matter), not a refusal-mediating one. Adding a topic direction to the
# residual stream pushes the state off-manifold and produces gibberish, which is
# exactly what the old sweep measured -- and the README then blamed ActAdd / CAA
# / Arditi for our extraction error.
#
# THE FIX (the sibling lesson ``steering_tutorials/realignment`` already does
# this): abliteration only damages the *abliterated* model. The ALIGNED base
# model of the same architecture still refuses perfectly, so its residual stream
# still contains a clean refusal direction. Read the direction there, steer the
# abliterated model with it. Same architecture => same hidden size and the same
# representational depth at layer 12, so the transplant is well-typed.
#
# The aligned base Gemma-3-1B, cached locally under ``models/`` (the HF id 401s
# without a token on this host).
BASE_MODEL = os.environ.get("STEER_BASE_MODEL", "models/google/gemma-3-1b-it")

# Which model supplies the contrast used to build the vector:
#   "base"         -> the aligned BASE_MODEL. THE DEFAULT and the correct recipe.
#   "abliterated"  -> the old (buggy) behaviour, kept ONLY as a labelled ablation.
#   "both"         -> run the unconditional sweep under BOTH vectors so the
#                     lesson can show the contrast side by side. The CONDITIONAL
#                     arm and the headline numbers always use the base vector.
# We steer MODEL_ID in every case; only the read side changes.
EXTRACT_FROM = os.environ.get("STEER_EXTRACT_FROM", "base").strip().lower()
if EXTRACT_FROM not in ("base", "abliterated", "both"):
    raise ValueError(
        f"STEER_EXTRACT_FROM must be base|abliterated|both, got {EXTRACT_FROM!r}")

# The source whose numbers are the lesson's headline (and which the conditional
# arm uses). Under "both" that is the corrected, base-extracted vector.
PRIMARY_SOURCE = "base" if EXTRACT_FROM in ("base", "both") else "abliterated"

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
# Overridable so a large model (e.g. Gemma-3-4B) can run a capped screening pass in
# one foreground window: STEER_ALPHAS="0,0.05,0.1" shrinks the grid.
ALPHAS = [float(a) for a in os.environ.get("STEER_ALPHAS", "0.0,0.05,0.10,0.15").split(",")]

# --- Data / split ------------------------------------------------------------
# The shared >=500/class harmful/benign set (``steering_tutorials.common.data``)
# replaces the old 100-prompt JailbreakBench loader, so we can build the vector
# on a real sample and still hold out a credible eval split. We take N_PER_CLASS
# of each, then split every class into two disjoint halves:
#   - the first N_EXTRACT go into building the steering vector (diff-of-means),
#   - the rest are held out for EVALUATION (never seen during extraction).
# Keeping extraction and evaluation disjoint is what stops us from grading the
# vector on the very prompts that defined it.
# Overridable for a capped cross-scale screening: STEER_N_PER_CLASS=120 -> extract 80,
# eval 40/class (fits a 4B foreground window). Default is the full >=500/class rubric.
N_PER_CLASS = int(os.environ.get("STEER_N_PER_CLASS", "500"))
N_EXTRACT = int(os.environ.get("STEER_N_EXTRACT", str(min(300, N_PER_CLASS * 2 // 3))))
# Optional cap on the held-out EVAL half, so a smoke test fits one foreground
# window on this reap-happy host. 0 / unset == use the whole held-out half.
N_EVAL = int(os.environ.get("STEER_N_EVAL", "0"))
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
VECTOR_PATH = ARTIFACTS / "steering_vector.pt"            # the PRIMARY vector
# Per-extraction-source vectors, so both are on disk and auditable side by side.
VECTOR_PATHS = {
    "base": ARTIFACTS / "steering_vector_from_base.pt",
    "abliterated": ARTIFACTS / "steering_vector_from_abliterated.pt",
}
RESULTS_PATH = ARTIFACTS / "results.json"
RATES_PNG = ARTIFACTS / "rates_vs_alpha.png"
CONDITIONAL_PNG = ARTIFACTS / "conditional.png"
# Refusal-vs-alpha under BOTH extraction sources, on one axis (the bug/fix plot).
EXTRACTION_PNG = ARTIFACTS / "extraction_source_contrast.png"

# Resume checkpoint. This host reaps long jobs, so ``run_steering`` writes every
# completed alpha row and every conditional record here as it goes and picks up
# where it left off on the next launch. The file is keyed by a fingerprint of the
# whole config -- change any knob and the stale checkpoint is discarded, never
# silently reused (the "stamp your inputs" rule). Delete it to force a cold run.
CHECKPOINT_PATH = ARTIFACTS / "_run_checkpoint.json"
# Set STEER_FORCE_EXTRACT=1 to recompute the direction even if a matching cached
# vector is on disk.
FORCE_EXTRACT = os.environ.get("STEER_FORCE_EXTRACT", "0") == "1"

ARTIFACTS.mkdir(exist_ok=True)
