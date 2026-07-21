"""config.py — every knob for the ReFT-r1 lesson (lesson 3) in one place.

Lesson 2 (``steering_tutorials/hello_world_steering``) added ONE fixed steering
vector built by contrastive diff-of-means: ``v = mean(harmful) - mean(benign)``,
read once from layer 12 and added to the residual stream to re-install refusal.
That direction is *hand-built* and *fixed* — it never sees a training loss.

Lesson 3 replaces it with a LEARNED low-rank edit. Following AxBench, we train a
**rank-1 LoReFT intervention**: instead of adding a constant vector, we learn a
direction ``r`` plus an affine readout ``(w, b)`` and REPLACE that direction's
component of the residual with the learned affine function of the hidden state:

    r_unit = r / ||r||
    h' = h + r_unit * ( (w·h + b) - (r_unit·h) )

So the edit is *input-dependent* (it reads ``h`` through ``w``) and the whole
thing is trained end-to-end by gradient descent — steering as a learned rank-1
representation finetune, contrasted later with the fixed diff-of-means baseline.

  Wu et al. 2025, 'AxBench: Steering LLMs? Even Simple Baselines Outperform
    Sparse Autoencoders' (arXiv:2501.17148) [UNVERIFIED] — the ReFT-r1 steering
    method and the DiffMean baseline this lesson reproduces at laptop scale.
  Wu et al. 2024, 'ReFT: Representation Finetuning for Language Models'
    (arXiv:2404.03592) [UNVERIFIED] — LoReFT, the low-rank subspace edit whose
    rank-1 special case we implement here.

Like lessons 1 and 2 this package is DELIBERATELY standalone: it does not import
from the research harness in ``src/steering``. It reuses only the lesson-2 model
plumbing (``hello_world_steering.model_utils``); everything else lives here.
"""
from __future__ import annotations

from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1 and 2. Refusal
# has been ablated out of it, so there is behaviour to RE-INSTALL from outside —
# here with a learned rank-1 intervention instead of a fixed diff-of-means vector.
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# Which residual-stream layer we install the rank-1 intervention on — the same
# middle-ish layer lessons 1-2 read/wrote, so the learned edit acts on the same
# representational depth where the abstract "refuse this" concept lives cleanly.
LAYER = 12

# --- Training ----------------------------------------------------------------
# Adam LR, number of optimisation steps, and prompts per minibatch. The training
# objective (owned by the trainer module) pushes the intervention to make the
# model emit the refusal target on harmful prompts.
LR = 1e-3
STEPS = 300
BATCH = 4

# Weight on the KL term that keeps the intervened next-token distribution close
# to the base model's on BENIGN prompts — the ReFT/AxBench regulariser that stops
# the learned edit from wrecking capability while it installs the target behaviour.
LAMBDA_KL = 0.5

# Gradient-norm clip: the rank-1 edit divides by ``||r||`` (see reft.py), which
# can spike gradients early in training; clipping keeps the step well-behaved.
GRAD_CLIP = 1.0

# The text the intervention is trained to elicit on harmful prompts — a clean,
# short refusal used as the language-modelling target for the intervened forward.
REFUSAL_TARGET = "I can't help with that request."

# NOTE: unlike lessons 1-2 there is NO ``alpha`` here. The rank-1 intervention
# carries its OWN learned magnitude through ``r``, ``w`` and ``b``, so there is
# no external step-size knob to sweep — the network decides how hard to steer.

# --- Data / split ------------------------------------------------------------
# The shared >=500/class harmful/benign set (``steering_tutorials.common.data``:
# toxic-chat + JBB top-up, deduped + length-matched) replaces the old 100-prompt
# JailbreakBench loader. Per class we draw N_PER_CLASS and split into a TRAIN
# half (the first N_PER_CLASS - N_EVAL, used by the trainer) and a disjoint EVAL
# half (the last N_EVAL, held out for grading). See ``data.load_train_eval``.
N_PER_CLASS = 500
N_EVAL = 200            # per class, held out for eval (train = 300/class)

SEED = 0

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
REFT_PATH = ARTIFACTS / "reft.pt"
RESULTS_PATH = ARTIFACTS / "results.json"

ARTIFACTS.mkdir(exist_ok=True)
