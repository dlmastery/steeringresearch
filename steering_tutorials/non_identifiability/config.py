"""config.py — every knob for the non-identifiability lesson in one place.

The thesis (Venkatesh & Kurapath, Manipal Institute of Technology, 2026,
arXiv:2602.06801): a steering
vector is NOT unique. Extract "the refusal direction" by several different but
each individually-reasonable recipes and you get several DIFFERENT vectors —
low pairwise cosine — that nonetheless steer to a SIMILAR behavioral effect.
The direction you happened to compute is one member of a whole equivalence
family; calling it *the* refusal direction over-claims.

This package is DELIBERATELY standalone (like lessons 1-2): it reuses lesson
2's model/steering plumbing (``hello_world_steering.model_utils``) and judge,
plus the shared ``common.data`` loader, but imports nothing from the research
harness in ``src/steering``.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1-2. It has had
# its refusal behaviour removed, which is exactly why it makes a good demo: an
# aligned model already refuses, so there would be nothing to steer. Here we
# RE-INSTALL refusal from the outside with a steering vector — and show that
# many different vectors do the job equally well.
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# The residual-stream layer we read every candidate direction from AND write
# each one back into. Middle-ish layers carry the most abstract "meaning", so a
# concept like "refuse this" lives here cleanly. Gemma-3-1B has 26 layers; 12 is
# a touch past the middle — the layer lessons 1-2 already use, so this lesson
# speaks about the same representation.
LAYER = 12

# --- Data / split ------------------------------------------------------------
# The shared foundation ships >=500 harmful + >=500 benign prompts. We load the
# full set for a natural, low-noise contrast, then carve THREE disjoint roles:
#   - the first N_EXTRACT per class BUILD the candidate directions,
#   - a further N_EVAL harmful prompts are HELD OUT to score the steering effect,
#   - (the rest are unused headroom).
# Keeping build and eval disjoint is what stops us grading a direction on the
# very prompts that defined it.
N_PER_CLASS = 500       # loaded from common.data (rubric: >= 500/class)
N_EXTRACT = 300         # per class, used only to BUILD the directions (was 150)
# Held-out harmful prompts used only to SCORE the effect. On a RAM-constrained
# host the eval is 6 recipes x len(ALPHAS) x N_EVAL generations, so NONIDENT_N_EVAL
# lets a run be shrunk into one foreground window (screening-tier, labelled).
N_EVAL = int(os.environ.get("NONIDENT_N_EVAL") or "150")  # was 60
SEED = 0

# Number of top principal components whose span the RANDOM control direction is
# drawn from (recipe f). A random vector inside the *active* subspace is a much
# stronger control than a random vector in all of R^hidden.
N_PC = 10

# --- Steering strength -------------------------------------------------------
# Interpreted by ``model_utils.generate`` under ``operation="relative_add"``:
# the injected direction is L2-normalized and scaled to ``alpha`` times the
# local residual-stream norm. Because every candidate is unit-normalized the
# SAME alpha applies an equal-magnitude nudge to each — so any difference in
# effect is due to DIRECTION, not magnitude. That is the matched comparison.
MATCHED_ALPHA = 0.08

# An optional small dose sweep for context (the headline table uses MATCHED_ALPHA).
ALPHAS = [0.06, 0.08, 0.10]

# A direction "counts as effective" if its refusal rate reaches at least this
# fraction of the BEST candidate's refusal rate. The payoff statistic is the
# MINIMUM pairwise cosine among the effective directions: if two directions
# that barely resemble each other both steer, the vector is non-identifiable.
EFFECTIVE_FRACTION = 0.80

# --- Generation --------------------------------------------------------------
MAX_NEW_TOKENS = 48

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DIRECTIONS_PATH = ARTIFACTS / "directions.npz"   # all candidate unit vectors
RESULTS_PATH = ARTIFACTS / "results.json"
PLOT_PATH = ARTIFACTS / "nonident.png"           # cosine heatmap + refusal bars

ARTIFACTS.mkdir(exist_ok=True)
