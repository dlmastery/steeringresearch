"""config.py — every knob for the TRAJGUARD (streaming decoding-time detection) lesson.

The thesis (TrajGuard, Liu, Liu, Li, Xin, Ding; ACL 2026 Findings, arXiv:2604.07727):
a jailbreak is visible in the model's OWN hidden-state trajectory *as it generates*.
As the model decodes a harmful completion, its residual-stream state at each generated
token drifts, token by token, toward a "high-risk" region; a benign completion stays
put. So you can DETECT (and stop) a jailbroken generation in real time by monitoring a
sliding window of decoding-time hidden states -- BEFORE the harmful content is fully
emitted -- without any prompt-side classifier.

This is the SIBLING of `multiturn_jailbreak`:
  - multiturn_jailbreak: trajectory across CONVERSATION TURNS (chunk = a turn).
  - trajguard:           trajectory across GENERATED TOKENS  (chunk = a token).
Same core idea (classify a sequence of hidden states), different granularity. This
lesson REUSES `multiturn_jailbreak.models` (PerTurnMaxProbe / TrajectoryMLP / SeqGRU /
HierAttn) unchanged -- a token trajectory is the same [n_steps, dim] object a turn
trajectory is -- plus a TRAINING-FREE sliding-window projection detector (the paper's
own method) as the honest baseline.

DELIBERATELY standalone: reuses lesson-2's `hello_world_steering.model_utils`
(load_model / generate / num_layers) and the shared `common.data.load_harmful_benign`,
and `multiturn_jailbreak.models`; imports nothing from `src/steering`.

Data: generate a completion with the abliterated Gemma-3-1B for HARMFUL prompts (it
complies -> a harmful completion) and BENIGN prompts (a benign completion), capturing
the layer-`LAYER` hidden state at each generated-token position -> a per-token
trajectory. Label = the prompt's class. The detector must flag the harmful generation
from the FIRST FEW tokens (the streaming / early-warning value).
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


# --- Model (the abliterated Gemma we monitor) --------------------------------
# Local path (the gated HF id 401s without a token). Abliterated so it COMPLIES with
# harmful prompts -- which is exactly what produces a harmful token trajectory to catch.
MODEL_ID = _env_str("TG_MODEL_ID", "models/google/gemma-3-1b-it")
LAYER = _env_int("TG_LAYER", 12)          # residual layer whose per-token state we read

# --- Generation / capture ----------------------------------------------------
MAX_NEW_TOKENS = _env_int("TG_MAX_NEW_TOKENS", 40)  # trajectory length cap (per completion)
GREEDY = True                                        # deterministic decoding for reproducibility

# --- Data --------------------------------------------------------------------
# Prompts come from the shared >=500/class toxic-chat set (common.data). We cap for a
# screening-scale, one-window run on the RAM-constrained host (generation is the cost).
# Generation-heavy, so this is 1000 completions (500 harmful + 500 benign); use
# TG_N_PER_CLASS to shrink for a quick run.
N_PER_CLASS = _env_int("TG_N_PER_CLASS", 500)   # harmful + benign prompts to generate on (was 120)
SEED = _env_int("TG_SEED", 0)

# --- Training-free detector (the paper's method) -----------------------------
# Score(token) = projection of its (centered) hidden state onto the HARM DIRECTION
# (diff-of-means of harmful- vs benign-completion token states, fit on TRAIN only).
# A completion is flagged when the SLIDING-WINDOW mean of that score crosses tau.
WINDOW = _env_int("TG_WINDOW", 4)               # sliding-window length (tokens)
TARGET_FPR = 0.10                               # tau calibrated to this benign FPR on train

# --- Early-detection curve ---------------------------------------------------
# AUC as a function of how many generated tokens we have seen (the streaming value):
# can we flag the jailbreak from the first K tokens?
EARLY_KS = [2, 4, 8, 16, 32]

# --- Learned classifiers (reused from multiturn_jailbreak.models) ------------
METHODS = ["threshold_freeform", "per_turn_max", "trajectory_mlp", "seq_gru"]
N_FOLDS = _env_int("TG_FOLDS", 5)
BOOTSTRAP = _env_int("TG_BOOTSTRAP", 2000)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "results.json"
TRAJ_CACHE = ARTIFACTS / "token_trajectories.npz"
ROC_PNG = ARTIFACTS / "roc_by_method.png"
EARLY_PNG = ARTIFACTS / "early_detection_auc.png"
DRIFT_PNG = ARTIFACTS / "risk_drift_example.png"

ARTIFACTS.mkdir(exist_ok=True)
