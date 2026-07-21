"""config.py — every knob for the fine-grained (sparse) steering lesson.

We reuse lesson 2's WRITE machinery (the CAA diff-of-means refusal vector and the
``relative_add`` steering hook) unchanged. The one new idea lives in ``sparse.py``:
before we inject the vector, we ZERO all but its top-``keep_frac`` highest-
magnitude coordinates and renormalize back to the dense vector's norm. That keeps
the *strength* of the edit matched while shrinking the *number of dimensions* it
touches — the "steering less" knob.

Like every lesson here the package is standalone: it imports lesson-2 plumbing
(``hello_world_steering``) and the shared dataset foundation (``common.data``),
but never the research harness in ``src/steering``.
"""
from __future__ import annotations

from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used across the course. Refusal
# has been removed from its weights, so re-installing it from the outside with a
# steering vector is a clean demo (an aligned model would already refuse).
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# Residual-stream layer we read the contrast from AND write the steering vector
# into (Gemma-3-1B has 26 layers; 12 is a touch past the middle — where abstract
# concepts like "refuse this" live cleanly, and the layer lesson 1/2 both use).
LAYER = 12

# --- The sparsity sweep (the new axis) --------------------------------------
# Fraction of the steering vector's coordinates to KEEP (by |magnitude|); the
# rest are zeroed. 1.0 == dense (lesson-2 baseline). We sweep down to 2% to test
# the paper's claim that a handful of coordinates carry the behavior.
SPARSITY_LEVELS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]

# Steering strengths available; the sparsity sweep runs at STEER_ALPHA (a single
# operating point) so the ONLY thing changing across the frontier is sparsity —
# strength is held matched. 0.0 is the unsteered baseline row.
ALPHAS = [0.0, 0.05, 0.10, 0.15]
STEER_ALPHA = 0.10          # operating point for the sparsity frontier

# --- Data / split ------------------------------------------------------------
# The shared foundation ships >=500 harmful + >=500 benign prompt-level examples.
# We build the vector from the first N_EXTRACT of each class and EVALUATE on a
# disjoint held-out slice, so we never grade the vector on prompts that defined
# it. N_PER_CLASS caps how many we pull per class.
N_PER_CLASS = 500
N_EXTRACT = 300             # per class, used only to build the dense vector (was 200)
N_EVAL = 150                # per class, held-out, used only to measure (was 60)
SEED = 0

# --- Generation --------------------------------------------------------------
MAX_NEW_TOKENS = 48         # long enough to tell refusal from compliance

# A sparse level "qualifies" as a win over dense only if it does not regress the
# collateral axes beyond this slack (over-refusal / gibberish) while matching the
# dense refusal rate within this tolerance.
REFUSAL_MATCH_TOL = 0.05    # sparse refusal within this of dense counts as "matched"
COLLATERAL_SLACK = 0.0      # sparse over-refusal/gibberish must be <= dense + slack

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VECTOR_PATH = ARTIFACTS / "dense_vector.pt"
RESULTS_PATH = ARTIFACTS / "results.json"
FRONTIER_PNG = ARTIFACTS / "sparsity_frontier.png"

ARTIFACTS.mkdir(exist_ok=True)
