"""config.py -- the anchor for control_data_split (ASIDE instruction/data separation).

WHAT THIS LESSON MEASURES, AND WHAT IT DOES NOT
------------------------------------------------
`aside.py` and `inseparability.py` are the fixed spine (built and verified
elsewhere in this package): the ISOCLINIC ROTATION (arXiv:2503.10566) and the
IMPOSSIBILITY BOUND (arXiv:2606.27567). Both are checkable WITHOUT training. This
config wires the substrate that turns those two into MEASURED numbers on Gemma:

  * a corpus of documents with an EXPLICIT, CONSTRUCTION-TIME instruction-vs-data
    ROLE label (never inferred from the text -- see data.py),
  * the model + layer set the residual-stream separability sweep runs over,
  * the rotation angle (a CONFIG knob, not a hard-coded constant, precisely so
    "pi/2 is special" is a testable claim rather than an assumption -- aside.py
    docstring),
  * keyed artifact paths, because a bare `results.json` is the exact defect
    `common.artifact_paths` exists to prevent (meerkat/rogue_scalpel/
    cross_trajectory all shipped it independently -- see that module's docstring).

This file loads no model and imports nothing heavy. ASCII stdout only (Windows
cp1252 console).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from steering_tutorials.common.artifact_paths import assert_no_bare_sibling, keyed_path

__all__ = [
    "MODEL_ID", "MODEL_TAG", "HIDDEN_SIZE", "NUM_HIDDEN_LAYERS", "LAYERS",
    "SEED", "N_PER_ROLE", "ANGLE", "TEST_SIZE",
    "HF_INSTRUCTION_SOURCE", "HF_DATA_SOURCES",
    "ARTIFACTS", "RESULTS_PATH", "CORPUS_CACHE_PATH", "ACTIVATION_CACHE_PATH",
    "ensure_artifacts", "preflight", "as_dict",
]


# --- env helpers (CDS_ prefix; an EMPTY env var means "unset", not "") -------
def _env_str(name, default):
    return os.environ.get(name) or default


def _env_int(name, default):
    return int(os.environ.get(name) or default)


def _env_float(name, default):
    return float(os.environ.get(name) or default)


def _env_int_list(name, default):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [int(x) for x in raw.split(",") if x.strip()]


# --- model -------------------------------------------------------------------
# LOCAL path: the bare HF id 401s on this host without a token (CLAUDE.md 18.5).
# This is the ALIGNED base, not the abliterated build -- separability is a
# property of the model's normal residual stream, not a steering target.
MODEL_ID = _env_str("CDS_MODEL_ID", "models/google/gemma-3-1b-it")


def _model_tag(model_id):
    """A short filename-safe tag. Two different models must not collide here."""
    tag = model_id.replace("\\", "/").rstrip("/").split("/")[-1]
    keep = [c if (c.isalnum() or c in "-.") else "-" for c in tag]
    return "".join(keep).strip("-") or "model"


MODEL_TAG = _env_str("CDS_MODEL_TAG", _model_tag(MODEL_ID))

# Read from models/google/gemma-3-1b-it/config.json 2026-08-30: hidden_size=1152
# (EVEN -- required by aside.isoclinic_matrix, which refuses an odd dim),
# num_hidden_layers=26 (so `output_hidden_states=True` returns 27 tensors,
# index 0 = the embedding layer output, indices 1..26 = post-block).
HIDDEN_SIZE = _env_int("CDS_HIDDEN_SIZE", 1152)
NUM_HIDDEN_LAYERS = _env_int("CDS_NUM_HIDDEN_LAYERS", 26)
if HIDDEN_SIZE % 2:
    raise ValueError(
        "CDS_HIDDEN_SIZE=%d is ODD; aside.isoclinic_matrix refuses an odd "
        "dimension (one axis would go unrotated and silently weaken the "
        "split)." % HIDDEN_SIZE)

# The layer sweep. 0 = embedding-layer output (where the paper reports ASIDE is
# already 100%% separable and vanilla is not); the rest space out over the 26
# transformer blocks. A CONFIG knob, not a hard sweep over every layer, because
# extracting hidden states at every layer for N_PER_ROLE*2 documents is the
# expensive step this lesson defers to the lead's GPU pass (see separability.py).
LAYERS = tuple(_env_int_list("CDS_LAYERS", (0, 4, 8, 12, 16, 20, 24, 26)))
for _l in LAYERS:
    if not (0 <= _l <= NUM_HIDDEN_LAYERS):
        raise ValueError(
            "CDS_LAYERS entry %d out of range [0, %d] for a %d-layer model"
            % (_l, NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS))

# --- corpus / evaluation ------------------------------------------------------
SEED = _env_int("CDS_SEED", 0)
# >= 500/role, per CLAUDE.md 17's hard data rubric ("NO tiny datasets").
N_PER_ROLE = _env_int("CDS_N_PER_ROLE", 500)
TEST_SIZE = _env_float("CDS_TEST_SIZE", 0.3)

# Reported ablation optimum (SEP 71.4%% on Qwen3-8B). [UNVERIFIED-ANGLE]: not on
# the abstract page -- see aside.py. Kept here too (not only in aside.py) so a
# sweep over ANGLE is a one-line config change, not an aside.py edit.
ANGLE = _env_float("CDS_ANGLE", math.pi / 2)

# --- data sources (both already vetted in common.dataset_export.REDISTRIBUTABLE) --
# INSTRUCTION role: direct chat turns from a real user -- the textbook trusted
# channel. DATA role: content assigned to the untrusted/retrieved channel by
# CONSTRUCTION (data.py), drawn from a harmful+benign pool so the data channel is
# not confounded with "harmful" -- a document in the data channel can be perfectly
# innocuous, the point is it is not a COMMAND from the trusted caller.
HF_INSTRUCTION_SOURCE = "HuggingFaceH4/ultrachat_200k"
HF_DATA_SOURCES = ("lmsys/toxic-chat", "JailbreakBench/JBB-Behaviors")

# --- artifact paths (NEVER a bare results.json; see artifact_paths.py) --------
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

_POOL_VARIANTS = ("n%d" % N_PER_ROLE, "s%d" % SEED)
_ANGLE_TAG = ("ang%s" % ("%.4f" % ANGLE)).replace(".", "p")
_ACT_VARIANTS = (MODEL_TAG, "n%d" % N_PER_ROLE, "s%d" % SEED, _ANGLE_TAG)

CORPUS_CACHE_PATH = keyed_path(ARTIFACTS, "corpus", ".json.gz", *_POOL_VARIANTS)
ACTIVATION_CACHE_PATH = keyed_path(ARTIFACTS, "acts", ".npz", *_ACT_VARIANTS)
RESULTS_PATH = keyed_path(ARTIFACTS, "results", ".json", *_ACT_VARIANTS)


def ensure_artifacts():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS


def preflight():
    """Fail LOUDLY before a run, not after it has overwritten something."""
    ensure_artifacts()
    for stem, ext in (("results", ".json"), ("corpus", ".json.gz"), ("acts", ".npz")):
        assert_no_bare_sibling(ARTIFACTS, stem, ext)


def as_dict():
    """The config as it will be stamped into every results file."""
    return {
        "model_id": MODEL_ID, "model_tag": MODEL_TAG,
        "hidden_size": HIDDEN_SIZE, "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "layers": list(LAYERS), "seed": SEED, "n_per_role": N_PER_ROLE,
        "test_size": TEST_SIZE, "angle": ANGLE,
        "hf_instruction_source": HF_INSTRUCTION_SOURCE,
        "hf_data_sources": list(HF_DATA_SOURCES),
        "results_path": str(RESULTS_PATH),
        "corpus_cache_path": str(CORPUS_CACHE_PATH),
        "activation_cache_path": str(ACTIVATION_CACHE_PATH),
    }


def _self_test():
    print("OK  model=%s tag=%s hidden=%d layers=%d"
          % (MODEL_ID, MODEL_TAG, HIDDEN_SIZE, NUM_HIDDEN_LAYERS))
    print("OK  layer sweep: %r" % (LAYERS,))
    print("OK  n_per_role=%d seed=%d angle=%.6f (pi/2=%.6f)"
          % (N_PER_ROLE, SEED, ANGLE, math.pi / 2))

    for p in (RESULTS_PATH, CORPUS_CACHE_PATH, ACTIVATION_CACHE_PATH):
        name = Path(p).name
        assert name not in ("results.json", "corpus.json.gz", "acts.npz"), name
        print("OK  keyed path: %s" % name)

    assert _model_tag("models/google/gemma-3-1b-it") == "gemma-3-1b-it"
    assert _model_tag("google/gemma-2-2b-it") != _model_tag("google/gemma-3-1b-it")
    print("OK  _model_tag is filename-safe and does not collide across models")

    try:
        _bad = _env_int_list("CDS__NOPE", (0, 1))
        assert _bad == [0, 1]
    except Exception as exc:  # pragma: no cover - defensive
        raise AssertionError("unset CDS_LAYERS should fall back cleanly") from exc
    os.environ["CDS__EMPTY"] = ""
    assert _env_str("CDS__EMPTY", "fallback") == "fallback"
    del os.environ["CDS__EMPTY"]
    print("OK  an EMPTY env var falls back instead of producing a bare/empty value")

    assert HIDDEN_SIZE % 2 == 0, "HIDDEN_SIZE must be even (isoclinic rotation)"
    print("OK  HIDDEN_SIZE=%d is even (an odd size would have raised at import, "
          "mirroring aside.isoclinic_matrix's own guard)" % HIDDEN_SIZE)

    preflight()
    print("OK  preflight() passed: no bare sibling beside any keyed artifact")

    d = as_dict()
    assert d["angle"] == ANGLE and d["layers"] == list(LAYERS)
    print("OK  as_dict() stamps model/layers/angle/seed for the results file")
    print("")
    print("OK -- control_data_split/config.py: anchor is consistent and keyed.")


if __name__ == "__main__":
    _self_test()
