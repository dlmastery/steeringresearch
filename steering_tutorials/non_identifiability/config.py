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

import hashlib
import os
from pathlib import Path

from steering_tutorials.common.artifact_paths import keyed_path

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
# The DEFAULT is named so a runner can tell "this is the pre-registered size"
# from "this is a shrunken screening slice" -- comparing N_EVAL against itself
# after the env has been applied can never make that distinction.
N_EVAL_DEFAULT = 150                                      # was 60
N_EVAL = int(os.environ.get("NONIDENT_N_EVAL") or N_EVAL_DEFAULT)
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

# A small dose sweep, run as CONTEXT around the headline. The registered headline
# table is and stays the MATCHED_ALPHA cell -- the sweep tells you whether the
# non-identifiability verdict is an artefact of one dose or survives a range, and
# it must never be able to move the number the lesson reports.
#
# This list was DEAD CODE until 2026-08-22: ``ALPHAS`` appeared nowhere in
# run_nonident.py, which referenced only ``MATCHED_ALPHA``. So the lesson was
# catalogued for weeks as "alpha sweep wired but never executed" when there was
# no sweep to execute -- the sec.18.8 pattern in its documentation form: a
# plausible, well-formed, wrong entry in the ledger that never crashed.
ALPHAS_DEFAULT = [0.06, 0.08, 0.10]


def _parse_alphas(raw: str | None) -> list[float]:
    """Parse ``NONIDENT_ALPHAS`` ("0.06,0.08,0.10"), or the default list.

    ``NONIDENT_ALPHAS=none`` (or an explicit single value) shrinks the sweep for
    a short foreground window. An unparseable entry raises rather than silently
    collapsing the sweep to the default -- a typo'd env var that quietly runs a
    DIFFERENT experiment than the one you asked for is the failure this project
    keeps paying for.
    """
    if raw is None or not raw.strip():
        return list(ALPHAS_DEFAULT)
    if raw.strip().lower() in ("none", "off", "matched"):
        return []
    out = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(round(float(part), 4))
        except ValueError:
            raise ValueError(
                "NONIDENT_ALPHAS=%r has an unparseable entry %r. Expected a "
                "comma-separated list of floats, e.g. '0.06,0.08,0.10', or "
                "'none' to run the matched alpha alone." % (raw, part))
    return out


ALPHAS = _parse_alphas(os.environ.get("NONIDENT_ALPHAS"))

# The alphas ACTUALLY generated. MATCHED_ALPHA is forced in: the headline cell is
# then just one member of the sweep, computed exactly once and reused, so the
# sweep costs (len-1) extra doses rather than a whole duplicate pass -- and there
# is no way for the headline and the sweep to disagree about the same alpha.
SWEEP_ALPHAS = sorted({round(float(a), 4) for a in ALPHAS} | {MATCHED_ALPHA})

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


def alpha_tag(alphas) -> str:
    """``[0.06, 0.08, 0.1]`` -> ``"a060-080-100"`` — a filename-safe sweep id."""
    return "a" + "-".join("%03d" % round(float(a) * 1000) for a in alphas)


# Everything that VARIES between runs of this lesson goes in the filename. Two
# things do: the eval size (NONIDENT_N_EVAL) and the alpha grid
# (NONIDENT_ALPHAS). A capped screening run and the pre-registered run are
# different experiments and must not be able to land on the same path -- see
# steering_tutorials/common/artifact_paths.py for the three lessons that
# learned this the expensive way.
RUN_KEY = ("n%d" % N_EVAL, alpha_tag(SWEEP_ALPHAS))
RESULTS_PATH = keyed_path(ARTIFACTS, "results", ".json", *RUN_KEY)
PLOT_PATH = keyed_path(ARTIFACTS, "nonident", ".png", *RUN_KEY)

# The candidate directions depend on the BUILD half only -- model, layer, PC
# count, seed and the extract prompts -- never on n_eval or alpha. So they are
# keyed separately and reused across the whole sweep. The npz also carries a
# content fingerprint (see vectors.save_directions); a cache whose fingerprint
# disagrees with the data in hand is REBUILT, not reused.
DIRECTIONS_PATH = keyed_path(ARTIFACTS, "directions", ".npz",
                             "L%d" % LAYER, "x%d" % N_EXTRACT)

# Per-cell checkpoints for a resumable run (this host reaps long jobs).
CKPT_DIR = ARTIFACTS / ("ckpt_%s" % "_".join(RUN_KEY))

# The pre-keying artifact: n_eval=80, no judge stamp, and not regenerable from
# the code beside it. Renamed out of the bare `results.json` slot on 2026-08-22
# so it stays ATTRIBUTABLE instead of becoming indistinguishable from a current
# run (`assert_no_bare_sibling`, called at pre-flight, enforces that from now on).
# NOTE the name: `audit_lesson_paths` looks for an assignment starting with
# `RESULTS_PATH`, so this constant must not be called that.
LEGACY_RESULTS_PATH = ARTIFACTS / "results_n80_legacy-unstamped.json"

ARTIFACTS.mkdir(exist_ok=True)


def run_fingerprint(eval_prompts, judge_id: str) -> str:
    """SHA-256 over everything a checkpoint must not be resumed across.

    Model, layer, sizes, seed, decode length, the alpha grid, the EXACT eval
    prompts, and the judge that graded them. Resuming a partial run whose judge
    or prompt slice has changed underneath it would splice two experiments into
    one artifact and report the seam as a result.
    """
    h = hashlib.sha256()
    for part in (MODEL_ID, str(LAYER), str(N_EXTRACT), str(N_EVAL), str(SEED),
                 str(N_PC), str(MAX_NEW_TOKENS), alpha_tag(SWEEP_ALPHAS),
                 str(MATCHED_ALPHA), str(judge_id)):
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x1f")
    for p in eval_prompts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x1e")
    return h.hexdigest()
