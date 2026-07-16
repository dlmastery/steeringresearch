"""config.py -- every knob for the GAVEL activation-monitoring lesson in one place.

GAVEL is a READ-side guardrail. Where lesson 2 *wrote* a refusal vector into the
residual stream, GAVEL never edits activations: it reads the mean-pooled residual
of an incoming prompt, scores it against a library of interpretable "cognitive
element" (CE) directions, and a predicate RULE decides block-or-pass. That
separation is the paper's whole point -- you can retune a rule (or add a CE)
without retraining the model or any detector.

We demonstrate on the ABLITERATED Gemma-3-1B (same model as lessons 1-3). That
choice is deliberate: an abliterated model has had its refusal removed, so it
*complies* with harmful prompts. Without a guardrail, every harmful prompt leaks.
GAVEL sits in front as an external monitor and blocks harmful prompts before the
model ever sees them -- making the value of a read-only guardrail concrete and
measurable (leaks that slip past the monitor are visible in the pass-through).

Like the other lessons this package is DELIBERATELY standalone: it reuses lesson
2's model plumbing (``hello_world_steering.model_utils`` / ``judge``) and the
shared >=500/class dataset (``common.data``), but adds no dependency on the
research harness in ``src/steering``.
"""
from __future__ import annotations

from pathlib import Path

# --- The model we monitor and (conditionally) let answer ---------------------
# The uncensored / abliterated Gemma-3-1B from lessons 1-3. No built-in refusal,
# so without the monitor a harmful prompt is answered; the monitor is what stops
# it. We only READ its activations to score CEs, and only GENERATE for prompts the
# rule PASSES.
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# Residual-stream layer the monitor reads. Middle layers carry the cleanest
# abstract "what is this prompt about" signal -- the same layer lesson 1's probe
# and lesson 2's steering vector use. Gemma-3-1B has 26 layers; 12 is just past
# the middle. Clamped to the model's depth at runtime.
LAYER = 12

# How a variable-length prompt becomes ONE vector for the monitor: mean-pool the
# residual over all prompt tokens (matches lesson 1's probe features).
POOLING = "mean"

# --- Cognitive elements (the fine-grained detectors) -------------------------
# Each CE is a diff-of-means direction for one harm concept: unit(mean(concept) -
# mean(benign)) at LAYER. The score of a prompt is (h - benign_mean) . direction.
# We build one CE per harm CATEGORY present in the extract split (the categories
# come from toxic-chat's moderation flags: sexual / harassment / violence / hate /
# self_harm). A category needs at least this many extract examples to earn a CE;
# sparser categories are folded into a catch-all "other_harm" CE so nothing is
# left uncovered.
# DATA SUFFICIENCY: a diff-of-means direction from a handful of examples is noise
# (an 8-example CE is meaningless). 30 is a defensible floor for a stable direction
# plus a disjoint calibration, and it aligns with the shared loader's own concept
# gate (MIN_CONCEPT_AVAILABLE=100 keeps sexual/harassment/violence, drops the tiny
# hate/self_harm pools). With N_EXTRACT=200 the kept categories are harassment
# (~43), sexual (~105), violence (~36); hate/self_harm (~8 each) are correctly
# dropped rather than given a noise CE.
MIN_CE_EXAMPLES = 30

# Per-CE benign false-positive budget. Each CE's threshold tau is calibrated on
# the EXTRACT benign activations so that only ~TARGET_FPR of benign prompts trip
# that CE. This is the "configurable precision" knob the paper highlights: tighten
# it and the monitor blocks less benign traffic (at some recall cost); loosen it
# and it catches more harm (at some over-block cost) -- no retraining required.
TARGET_FPR = 0.05

# --- The rule (a predicate over CEs) -----------------------------------------
# The evaluated rule for the harmful/benign split is OR-of-all-harm-CEs: block if
# ANY cognitive element fires. Rule.any_of / all_of / at_least (see monitor.py)
# let you compose stricter predicates (e.g. "violence AND payment") -- the README
# shows one; infer.py demonstrates it live.
RULE = "any_of"          # {"any_of", "at_least_2"} -- the predicate over harm CEs

# What a blocked prompt gets instead of a model answer.
SAFE_REFUSAL = "I can't help with that request."

# --- Data / split (shared >=500/class harmful/benign set) --------------------
# Prompts come from ``common.data`` (toxic-chat + JBB top-up, deduped +
# length-matched, with per-prompt harm category). We build the CE library on the
# first N_EXTRACT/class and evaluate the monitor on the disjoint held-out rest.
N_PER_CLASS = 250
N_EXTRACT = 200          # per class, used only to build + calibrate the CE library
SEED = 0

# --- Generation (only for PASSED prompts) ------------------------------------
# Short greedy completions: long enough to tell a refusal from compliance, short
# enough to run the passed-through subset on a laptop GPU. Only prompts the rule
# PASSES are generated + judged; blocked prompts get SAFE_REFUSAL for free.
MAX_NEW_TOKENS = 48

# Optional caps for a laptop run. Read as ``os.environ.get(name) or DEFAULT`` in
# run_gavel.py so an unset OR empty env var falls back to the default (never 0).
DEFAULT_MAX_EVAL = 60    # cap held-out prompts scored per class (None -> all)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MONITOR_PATH = ARTIFACTS / "monitor.npz"      # CE directions + taus + benign means
RESULTS_PATH = ARTIFACTS / "results.json"
GATE_PNG = ARTIFACTS / "block_vs_falseblock.png"
CE_PNG = ARTIFACTS / "ce_firing_rates.png"

ARTIFACTS.mkdir(exist_ok=True)
