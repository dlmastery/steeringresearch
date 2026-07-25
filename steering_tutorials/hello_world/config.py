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
from pathlib import Path

# --- The frozen feature extractor -------------------------------------------
# The uncensored / abliterated Gemma-3-1B already downloaded to the HF cache.
# We only ever READ its activations; we never fine-tune or steer it here.
# Overridable so we can retrain the probe on a LARGER model (e.g. Gemma-3-4B
# abliterated) for a cross-scale check: STEER_MODEL_ID + STEER_LOAD_4BIT=1.
MODEL_ID = os.environ.get(
    "STEER_MODEL_ID", "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
)
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
# Probe filename is overridable so a cross-scale (4B) probe is saved SEPARATELY and
# never clobbers the 1B probe.pt that the steering lessons depend on:
# STEER_PROBE_NAME=probe_4b.pt.
PROBE_PATH = ARTIFACTS / os.environ.get("STEER_PROBE_NAME", "probe.pt")
METRICS_PATH = ARTIFACTS / os.environ.get("STEER_METRICS_NAME", "metrics.json")
FEATURES_CACHE = ARTIFACTS / os.environ.get("STEER_FEATURES_NAME", "features.npz")
ROC_PNG = ARTIFACTS / "roc_curve.png"
HISTORY_PNG = ARTIFACTS / "training_history.png"
CONFUSION_PNG = ARTIFACTS / "confusion_matrix.png"

ARTIFACTS.mkdir(exist_ok=True)
