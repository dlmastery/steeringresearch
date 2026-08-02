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

import os
from pathlib import Path

# --- The model we steer: TWO bases, one variable -----------------------------
# STRUCTURAL FIX (2026-08-02). This lesson used to run its three-way bake-off on
# ONE base: the abliterated Gemma-3-1B. That makes AxBench's actual claim
# ("prompting outperforms existing steering methods") UNTESTABLE here, because
# abliteration removes exactly the instruction-following refusal that the
# PROMPTING arm depends on and that the other two arms do not. The prompting arm
# was structurally crippled while ReFT-r1 and DiffMean were not; a bake-off whose
# baseline is lobotomised cannot reproduce or refute the paper.
#
# So the bake-off now runs on BOTH bases and reports them side by side. Exactly
# ONE thing differs between the two runs — the base model. Same n, same prompts,
# same layer, same DiffMean alpha, same ReFT training budget, same off-family
# judge, same seed.
#
#   aligned      — google/gemma-3-1b-it, loaded from the LOCAL path (the HF id
#                  401s without a token). Refusal intact. This is the base on
#                  which AxBench's prompting-vs-learned-methods claim is
#                  MEANINGFUL, so it is the headline arm.
#   abliterated  — DavidAU/...-heretic-extreme-uncensored-abliterated. Refusal
#                  removed. Kept as a LABELLED ABLATION: it isolates what happens
#                  to each method when instruction-following refusal is deleted
#                  from the weights. It is NOT a test of AxBench.
BASES = {
    "aligned": "models/google/gemma-3-1b-it",
    "abliterated": "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated",
}

# Which base this process runs. Set REFT_BASE=aligned|abliterated. The default is
# the ABLITERATED one purely for backward compatibility with the pre-existing
# artifacts (results.json / reft.pt) and the webapp that reads them; the aligned
# run is the one the README headlines.
BASE = (os.environ.get("REFT_BASE", "abliterated") or "abliterated").strip().lower()
if BASE not in BASES:
    raise ValueError(
        f"REFT_BASE={BASE!r} is not one of {sorted(BASES)}. "
        "This is a hard error on purpose: a typo must not silently fall back to "
        "the other base and mislabel a whole run's artifacts."
    )
MODEL_ID = BASES[BASE]

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
# Per-base artifact names, so the two runs can never overwrite each other. The
# abliterated run keeps the ORIGINAL names (reft.pt / results.json) so the webapp
# and every existing link still resolve; the aligned run gets suffixed names.
ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

_SUFFIX = "" if BASE == "abliterated" else f"_{BASE}"
REFT_PATH = ARTIFACTS / f"reft{_SUFFIX}.pt"
RESULTS_PATH = ARTIFACTS / f"results{_SUFFIX}.json"

# Append-only per-prompt checkpoint. This host REAPS long jobs, and one eval pass
# is ~400 prompts x 4 generations x 4 judge calls — hours. Every prompt's record
# is flushed here the moment it is graded, and a restart resumes by skipping the
# prompts already present. Checkpointing at the granularity of the most expensive
# irreversible step (one prompt's four generations), not at whatever is
# convenient to write.
RECORDS_PATH = ARTIFACTS / f"records{_SUFFIX}.jsonl"

# Plot filenames are per-base too (the two runs produce different bars).
STEERING_PLOT = f"steering_compare{_SUFFIX}.png"
DETECTION_PLOT = f"detection_auc{_SUFFIX}.png"
TRAINING_PLOT = f"training_curve{_SUFFIX}.png"

# The cross-base comparison written by ``compare_bases.py`` (base-independent).
COMPARE_PATH = ARTIFACTS / "compare_bases.json"
COMPARE_PLOT = ARTIFACTS / "compare_bases.png"

ARTIFACTS.mkdir(exist_ok=True)
