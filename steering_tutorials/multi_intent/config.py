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

import os
from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1-2. Because its
# refusal behaviour was removed, it complies with every harm category out of the
# box — which is exactly what lets us watch K independent "refuse this category"
# vectors switch each category off, and watch them interfere when we stack them.
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
# Ordered largest-pool-first (sexual is the richest toxic-chat harm category), so
# the additive K=1..N ladder adds the smallest, least stable concept last — the
# honest stress test for interference. Only the WELL-POPULATED concepts survive the
# shared loader's data-sufficiency gate (>=100 available, ``MIN_CONCEPT_AVAILABLE``):
# ``hate`` (~24) and ``self_harm`` (~27) are too small for a stable diff-of-means
# and a disjoint >=30-prompt eval split, so they are EXCLUDED and the ladder runs
# K=1..3 (sexual -> +harassment -> +violence), not K=1..5. Every name below must
# be one the shared loader returns.
CONCEPTS = [
    "sexual",
    "harassment",
    "violence",
]

# Prompts to draw PER concept before the shared loader's 40/30/30 exemplars/steer/
# eval split (``extract`` = exemplars + steer; ``eval`` = the disjoint 30%).
#
# POOL-LIMITED, and we now say so in the artifact rather than only in a comment.
# The kept toxic-chat concepts hold roughly sexual ~388 / harassment ~143 /
# violence ~111 deduped prompts, so the CLAUDE.md sec.17 rubric-1 floor of 500 per
# class is UNREACHABLE here no matter what we request. Rubric item 2 governs
# instead: maximise within the pool and state the cap explicitly. We therefore
# REQUEST 500 (was 150, which left ~60% of the sexual pool unused) and let the
# loader return whatever each concept actually has. Effect on the eval split:
#   sexual      388 pool -> ~116 eval  (was 45)
#   harassment  143 pool -> ~ 43 eval  (unchanged; 150 already exceeded the pool)
#   violence    111 pool -> ~ 33 eval  (was 34)
# Larger ``extract`` halves also mean a better-estimated diff-of-means direction
# (rubric item 6: small-N directions are provisional and have flipped findings).
#
# ``MULTI_INTENT_N_PER_CONCEPT`` shrinks the whole run into one foreground GPU
# window on a RAM-pressured host; anything below the pool is SMOKE tier and the
# runner stamps ``pool_capped`` / ``achieved_n`` so the artifact shows it.
N_PER_CONCEPT = int(os.environ.get("MULTI_INTENT_N_PER_CONCEPT", "500"))
N_BENIGN_BASELINE = 40

# Rubric-1 target, recorded in results.json alongside what we actually achieved so
# the gap between the standard and this lesson's pool ceiling is on the record.
TARGET_PER_CLASS = 500

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
# this value), so this value is passed through but IGNORED — kept only for
# backward-compatible call sites.
N_EVAL_PER_CONCEPT = 30  # was a trivially-tiny 5

# HARD FLOOR, enforced at run time (not a comment, not a hope). If any concept's
# held-out eval split comes back below this, ``run_multi_intent.main()`` ABORTS
# before touching the GPU rather than shipping a rate whose denominator is too
# small to read. CLAUDE.md sec.17 rubric item 2: never build a per-concept number
# from < 30 examples.
MIN_EVAL_PER_CONCEPT = 30

# How many (prompt, response, verdict) triples we keep verbatim in results.json.
# This is a SAMPLE for reading, never a denominator — the rates are computed over
# every eval prompt, and results.json records both numbers so the two cannot be
# confused.
EXAMPLES_CAP = 40

SEED = 0
MAX_NEW_TOKENS = 48

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VECTORS_PATH = ARTIFACTS / "concept_vectors.pt"
RESULTS_PATH = ARTIFACTS / "results.json"
LADDER_PNG = ARTIFACTS / "success_vs_k.png"

ARTIFACTS.mkdir(exist_ok=True)
