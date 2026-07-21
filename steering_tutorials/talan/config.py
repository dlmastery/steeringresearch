"""config.py -- every knob for the TALAN-inspired latent-adapter lesson.

Where lesson 2 added ONE fixed diff-of-means vector and lesson 3 learned a rank-1
LoReFT edit, this lesson learns a full (but tiny) NONLINEAR latent adapter over
the residual stream, with the LLM frozen. That is the third and highest-capacity
point on one spectrum:

    fixed vector        (lesson 2, DiffMean)   -- a constant nudge, no learning
    learned rank-1      (lesson 3, ReFT-r1)    -- a linear readout along ONE dir
    learned adapter     (this lesson, TALAN)   -- a nonlinear bottleneck h -> delta

METHOD PROVENANCE (read this before claiming anything):
  TALAN (Zhang, Liu, Wang, Tao, Wang, Chordia, Huang 2026, 'TALAN: Task-Aligned
  Latent Adaptation Networks for Targeted Post-Training of Large Language Models',
  arXiv:2606.06902) is a POST-TRAINING method: it co-trains a low-rank adapter on
  the backbone AND a sequence-conditioned latent side path in one SFT loop. It is
  NOT a pure inference-time activation steer. This lesson builds the closest
  faithful INFERENCE-TIME analogue -- our own construction, clearly labelled --
  keeping ONLY the latent side path and FREEZING the whole LLM (so there is no
  backbone LoRA). We do not claim this is the paper's exact method.

The adapter mirrors TALAN's "compress the sequence into latent memory, remix it
into token-level perturbations, write them back through a controlled residual
update", and is configurable along the paper's six design axes (mapped below).

Like lessons 1-3 this package is DELIBERATELY standalone: it does NOT import from
the research harness in ``src/steering``. It reuses only the lesson-2 model
plumbing (``hello_world_steering.model_utils``); everything else lives here.
"""
from __future__ import annotations

from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1-3. Refusal has
# been ablated out of it, so there is behaviour to RE-INSTALL from outside -- here
# with a learned latent adapter instead of a fixed vector or a rank-1 edit.
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# --- TALAN's six design axes (paper Sec. 3) mapped to our knobs --------------
# 1. INSERTION LOCATION -- which residual-stream layer the side path reads/writes.
LAYER = 12
# 2. MEMORY SIZE -- the bottleneck / latent-memory width the token is compressed
#    into before it is remixed and written back. Small keeps params < 1% (TALAN's
#    efficiency claim) and forces the adapter to learn a compact correction.
MEMORY = 16
# 3. MIXER -- the nonlinearity used to remix the latent memory. "gelu" | "relu"
#    | "tanh"; the adapter applies mix (memory->memory) then this activation.
MIXER = "gelu"
# 4. WRITEBACK RULE -- how the produced perturbation returns to the stream.
#    "add" is the faithful residual default: h' = h + scale * up(mix(down(h))).
WRITEBACK = "add"
# 5. TRAINABILITY SCOPE -- what learns. Here: the adapter ONLY (LLM frozen). This
#    is the deliberate INFERENCE-TIME simplification of TALAN, which also trains a
#    backbone LoRA. Recorded so the difference from the paper is explicit.
TRAIN_SCOPE = "adapter_only"
# 6. GRADIENT / WRITEBACK SCALE -- initial magnitude of the learned scalar gate on
#    the writeback (the adapter can then grow or shrink it during training).
GRAD_SCALE = 1.0

# --- Training ----------------------------------------------------------------
# Adam LR, optimisation steps, prompts per minibatch. The objective (owned by the
# trainer) pushes the adapter to make the frozen model emit a refusal on harmful
# prompts while a KL leash keeps benign prompts untouched.
LR = 1e-3
STEPS = 300
BATCH = 4

# Weight on the KL term that keeps the adapted next-token distribution close to the
# base model's on BENIGN prompts -- the selectivity leash that stops the adapter
# from making everything refuse (the over-refusal axis of the project's five).
LAMBDA_KL = 0.5

# Gradient-norm clip. The adapter is higher-capacity than a rank-1 edit, so its
# two-term loss is stiff and can oscillate; clipping (plus best-checkpointing in
# the trainer) is the same stabiliser scar carried from the ReFT lesson.
GRAD_CLIP = 1.0

# The text the adapter is trained to elicit on harmful prompts -- a clean, short
# refusal used as the language-modelling target for the adapted forward pass.
REFUSAL_TARGET = "I can't help with that request."

# Generation length for eval / infer (greedy, deterministic gates).
MAX_NEW_TOKENS = 48

# --- Data / split ------------------------------------------------------------
# The shared >=500/class harmful/benign set (``steering_tutorials.common.data``:
# toxic-chat + JBB top-up, deduped + length-matched). We draw N_PER_CLASS of each
# and cut a disjoint TRAIN (first N_PER_CLASS - N_EVAL) / EVAL (last N_EVAL) split.
N_PER_CLASS = 500
N_EVAL = 200            # per class, held out for eval (train = 300/class)

SEED = 0

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
ADAPTER_PATH = ARTIFACTS / "talan.pt"
RESULTS_PATH = ARTIFACTS / "results.json"

ARTIFACTS.mkdir(exist_ok=True)
