"""config.py — every knob for the MULTI-INTENT (compositional) steering lesson.

Lesson 2 (``hello_world_steering``) built ONE steering direction — "refuse this"
— and added it to the residual stream. Real deployments are rarely so tidy: a
safety layer usually wants to suppress SEVERAL harm categories at once (malware
*and* fraud *and* harassment), or to combine "refuse harm" with "be concise".
That is **compositional / multi-intent steering**: steer K concepts, not one.

The naive recipe is to build K diff-of-means vectors and add their sum. It
mostly works for K=1..2 and then degrades, because the K directions are NOT
orthogonal — they overlap in activation space, so pushing along one also pushes
(or cancels) along another. This cross-talk is **interference**. Two levers fight
it, and this lesson measures both:

  1. Gram-Schmidt ORTHOGONALIZATION of the K directions, so each concept gets its
     own axis and steering A barely moves B (see ``multi_intent.gram_schmidt``).
  2. The NORM BUDGET (N5): the total displacement ``Σ αᵢ ||h||`` you inject is
     finite before coherence collapses. Every concept you stack spends part of
     the budget, which caps how many you can compose (see ``norm_budget``).

Like every tutorial here, this package is DELIBERATELY standalone except that it
REUSES the mechanical core of lesson 2 (model loading, activation reading, the
steering hook, and the judge) — we do not reimplement steering, we compose it.
"""
from __future__ import annotations

from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1-2. Because its
# refusal behaviour was removed, it complies with every harm category out of the
# box — which is exactly what lets us watch K independent "refuse this category"
# vectors switch each category off, and watch them interfere when we stack them.
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# Residual-stream layer we read each concept contrast from AND steer into. Middle
# layers carry the most abstract "meaning", so a category like "malware" or
# "fraud" is linearly separable here. Kept at 12 to match lessons 1-2 so the
# three tutorials speak about the same representation. Gemma-3-1B has 26 layers.
STEER_LAYER = 12

# --- The K concepts we compose ----------------------------------------------
# Each concept is one coarse harm category from the SHARED dataset
# (``steering_tutorials.common.data``): toxic-chat's ``openai_moderation`` flags
# folded to five buckets. We steer "refuse THIS category" for each, contrasting
# that category's harmful prompts against a SHARED benign baseline. Using the same
# baseline for all K is what makes the K raw directions comparable (they share an
# origin), so their overlap is a clean measure of concept similarity rather than
# of baseline drift.
#
# Ordered largest-pool-first (sexual is the richest toxic-chat harm category,
# hate the smallest), so the additive K=1..N ladder adds the smallest, least
# stable concept last — the honest stress test for interference. The two smallest
# categories (self_harm, hate) carry only ~20-30 prompts each, so their eval split
# is small; this is reported honestly at load time.
CONCEPTS = [
    "sexual",
    "harassment",
    "violence",
    "self_harm",
    "hate",
]

# Prompts to draw PER concept before the shared loader's 40/30/30 exemplars/steer/
# eval split (``extract`` = exemplars + steer; ``eval`` = the disjoint 30%). 150
# gives ~40-45 eval prompts for the big concepts; small categories return what
# they have. The benign baseline is a shared, larger, lower-variance origin.
N_PER_CONCEPT = 150
N_BENIGN_BASELINE = 40

# --- Steering strengths ------------------------------------------------------
# Per-concept alpha under ``operation="relative_add"``: each concept's unit
# vector is injected at ``alpha * ||h||``. The SUM over K concepts is the total
# norm budget spent, so with K concepts and per-concept alpha a, the budget is
# ~K*a. We keep a modest so that even at K=4 the budget stays in the coherent
# regime for the raw-sum arm to have a fair chance (the whole point is to see
# WHEN it breaks). 0.0 == the unsteered baseline.
ALPHAS = [0.0, 0.04, 0.08]
PER_CONCEPT_ALPHA = 0.06     # the alpha used in the K=1..N stacking ladder

# --- Evaluation --------------------------------------------------------------
# Held-out prompts per concept used to measure steering success and cross-talk
# (disjoint from the extract prompts that built each vector). NOTE: the shared
# loader now owns the extract/eval split (its disjoint 30% eval share, larger than
# this old fixed 5), so this value is passed through but IGNORED — kept only for
# backward-compatible call sites.
N_EVAL_PER_CONCEPT = 5
SEED = 0
MAX_NEW_TOKENS = 48

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VECTORS_PATH = ARTIFACTS / "concept_vectors.pt"
RESULTS_PATH = ARTIFACTS / "results.json"
LADDER_PNG = ARTIFACTS / "success_vs_k.png"

ARTIFACTS.mkdir(exist_ok=True)
