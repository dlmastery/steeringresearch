"""config.py — every knob for the safety-probe hello-world in one place.

This project is DELIBERATELY standalone: it does not import anything from the
research harness in ``src/steering``. Everything it needs (model loading,
feature extraction, dataset download, the probe, the webapp) lives under
``steering_tutorials/hello_world/`` so you can read it top-to-bottom and run it on its own.

The idea in one sentence:
    A big language model already "knows" whether a prompt is harmful — that
    knowledge is written into its internal activations. We freeze the model,
    read the activation vector for a prompt at ONE middle layer, and train a
    tiny neural network (a "probe") to map that vector to harmful / safe.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# --- The frozen feature extractor -------------------------------------------
# The uncensored / abliterated Gemma-3-1B already downloaded to the HF cache.
# We only ever READ its activations; we never fine-tune or steer it here.
# Overridable so we can retrain the probe on a LARGER model (e.g. Gemma-3-4B
# abliterated) for a cross-scale check: STEER_MODEL_ID + STEER_LOAD_4BIT=1.
DEFAULT_MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
MODEL_ID = os.environ.get("STEER_MODEL_ID", DEFAULT_MODEL_ID)
LOAD_4BIT = os.environ.get("STEER_LOAD_4BIT", "0") == "1"

# Which residual-stream layer to read. Middle layers carry the most abstract
# "meaning" (early = tokens/syntax, late = next-token prediction), so a middle
# layer is the sweet spot for a concept like "is this harmful?".
# Gemma-3-1B has 26 layers; 12 is a touch past the middle. Clamped at runtime.
LAYER = 12

# How we turn a variable-length prompt into ONE fixed vector: mean-pool the
# residual activations over all prompt tokens. Simple, robust, order-agnostic.
POOLING = "mean"

# --- The probe (a 3-layer MLP) ----------------------------------------------
HIDDEN1 = 128
HIDDEN2 = 32
DROPOUT = 0.30          # strong dropout — we have only a few hundred examples
WEIGHT_DECAY = 1e-3     # L2 regularization, same reason
LR = 1e-3
EPOCHS = 200
PATIENCE = 30           # early-stop if val loss hasn't improved in this many epochs
DECISION_THRESHOLD = 0.5

# --- Data / split ------------------------------------------------------------
TEST_FRACTION = 0.20
VAL_FRACTION = 0.15     # carved out of the training portion for early stopping
SEED = 0

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
STATIC = ROOT / "static"


def model_tag(model_id: str | None = None) -> str:
    """A short filename tag identifying WHICH model produced an artifact.

    Every cached artifact (features, probe, metrics, plots) is a function of the
    feature extractor that produced it: a Gemma-3-1B activation is 1152 numbers
    wide, a Gemma-3-4B activation is 2560. Reusing one under the other's name is
    the bug this tag exists to prevent, so the model is part of the FILENAME and
    not merely of the file's contents.

    The default 1B model gets the empty tag, i.e. the historical names
    (``probe.pt``, ``features.npz``, ...) that lesson 2 and the checked-in
    artifacts already depend on. Any other model gets a suffix: the parameter
    count if the id carries one (``...gemma-3-4b-it...`` -> ``4b``), otherwise a
    slug of the repo name.
    """
    mid = MODEL_ID if model_id is None else model_id
    if mid == DEFAULT_MODEL_ID:
        return ""
    base = mid.replace("\\", "/").split("/")[-1].lower()
    m = re.search(r"(\d+(?:\.\d+)?)b(?![a-z0-9])", base)
    if m:
        return f"{m.group(1)}b".replace(".", "p")
    return re.sub(r"[^a-z0-9]+", "-", base).strip("-")[:40] or "custom"


MODEL_TAG = model_tag()


def artifact_path(stem: str, ext: str, env_var: str | None = None) -> Path:
    """``artifacts/<stem>[_<MODEL_TAG>]<ext>``, or an explicit env-var override."""
    override = os.environ.get(env_var) if env_var else None
    if override:
        return ARTIFACTS / override
    return ARTIFACTS / (f"{stem}_{MODEL_TAG}{ext}" if MODEL_TAG else f"{stem}{ext}")


def artifact_path_for(stem: str, ext: str, model_id: str) -> Path:
    """The artifact name a GIVEN model would produce, ignoring the environment.

    For callers outside this lesson (e.g. the lesson-2 conditional gate) that
    know which model they are running and need the probe that matches it:
        cfg.artifact_path_for("probe", ".pt", "...gemma-3-4b-it...")  ->  probe_4b.pt
    """
    tag = model_tag(model_id)
    return ARTIFACTS / (f"{stem}_{tag}{ext}" if tag else f"{stem}{ext}")


def probe_path_for(model_id: str) -> Path:
    """The probe checkpoint trained on ``model_id`` (may not exist yet)."""
    return artifact_path_for("probe", ".pt", model_id)


# Artifact names are model-tagged so a cross-scale (4B) run is saved SEPARATELY
# and can never clobber — or be mistaken for — the 1B probe.pt/features.npz that
# the steering lessons depend on. STEER_PROBE_NAME etc. still force an exact name.
PROBE_PATH = artifact_path("probe", ".pt", "STEER_PROBE_NAME")
METRICS_PATH = artifact_path("metrics", ".json", "STEER_METRICS_NAME")
FEATURES_CACHE = artifact_path("features", ".npz", "STEER_FEATURES_NAME")
ROC_PNG = artifact_path("roc_curve", ".png")
HISTORY_PNG = artifact_path("training_history", ".png")
CONFUSION_PNG = artifact_path("confusion_matrix", ".png")

ARTIFACTS.mkdir(exist_ok=True)
