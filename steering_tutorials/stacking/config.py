"""config.py — every knob for the stacking lesson in one place.

Lesson 2 (``hello_world_steering``) built ONE refusal steering vector and asked
"how hard can I push before coherence breaks?". Lesson 12 asks the next
question: when you have MORE than one prior, do their effects **add** (stack) or
**fight** (compete)? The answer is decided almost entirely by WHERE each prior
intervenes and HOW it modifies the signal there — the intervention-site taxonomy
of ``corpus/steering-stackable-vs-competing-analysis.md`` and CLAUDE.md sec. 9.

The lesson in one sentence:
    Build an additive 2->N ladder — start from one prior, add exactly ONE more
    per rung, and read the marginal effect: priors on DIFFERENT sites stack
    (gains add), priors on the SAME site with an incompatible operation compete
    (the second cancels/degrades the first), and piling on everything at once
    over-stacks the norm budget into gibberish.

Like lessons 1-3 this package is DELIBERATELY standalone: it imports only the
mechanical core of lesson 2 (``hello_world_steering``) and nothing from the
research harness in ``src/steering``.
"""
from __future__ import annotations

from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1-2. It has had
# its refusal behaviour removed, so steering refusal back IN is a visible,
# measurable effect — exactly what we need to read a marginal-effect ladder.
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# --- The two intervention SITES (layers) -------------------------------------
# The whole stack-vs-compete distinction is a claim about SITES. To isolate the
# site variable cleanly we reuse ONE direction (the refusal diff-of-means) and
# vary only the layer it is injected at:
#   * PRIMARY_LAYER      — prior A lives here (the lesson-2 layer).
#   * ORTHOGONAL_LAYER   — prior B lives here. A different residual layer is a
#                          DISJOINT site, so A and B compose additively. (A
#                          genuinely different *concept* at this layer would
#                          stack for the identical mechanical reason — disjoint
#                          site; using the same direction just holds everything
#                          but the layer fixed, a cleaner controlled demo.)
PRIMARY_LAYER = 12
ORTHOGONAL_LAYER = 8

# --- Per-prior steering strengths --------------------------------------------
# STACK_ALPHA is the relative_add step for A and B: the injected vector is scaled
# to this fraction of the local residual norm ||h|| (see lesson 2). Small enough
# that ONE prior stays coherent, so a second prior on a disjoint site has room to
# add gain before the norm budget (N5) is spent.
STACK_ALPHA = 0.08

# The competing prior B' re-injects the SAME refusal direction at the SAME site
# as A (PRIMARY_LAYER) but with the literal "add" operation instead of
# "relative_add". "add" is NOT norm-aware, so stacking it on top of A double-
# counts the refusal plane and overshoots off-manifold — the canonical same-site
# competition (corpus sec. 3.1). run_stacking rescales this raw step to match A's
# magnitude at a reference ||h|| so 2b is a controlled comparison; this multiplier
# is the fraction of that reference norm used as the raw step.
COMPETE_ADD_FRACTION = 0.08

# --- Data / split ------------------------------------------------------------
# The shared >=500/class harmful/benign set (``steering_tutorials.common.data``:
# toxic-chat + JBB top-up, deduped + length-matched) replaces the old 100-prompt
# JailbreakBench loader. We extract the refusal direction from a disjoint EXTRACT
# half and read the ladder on a held-out EVAL half of harmful prompts.
N_PER_CLASS = 350
N_EXTRACT = 175         # per class, used only to build the vector
N_EVAL = 175            # held-out harmful prompts the ladder is judged on
SEED = 0

# --- Generation --------------------------------------------------------------
MAX_NEW_TOKENS = 40

# How many held-out prompts to average the cumulative norm budget over. The norm
# budget (N5: cumulative ||Δh||/||h||) is measured directly with two forward
# passes per prompt (baseline vs stacked), so we keep this small.
N_NORM_BUDGET_PROMPTS = 6

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VECTOR_PATH = ARTIFACTS / "refusal_vector.pt"
RESULTS_PATH = ARTIFACTS / "results.json"
LADDER_PNG = ARTIFACTS / "ladder.png"

ARTIFACTS.mkdir(exist_ok=True)
