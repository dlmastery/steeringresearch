"""config.py — every knob for the HyperSteer lesson (lesson 3) in one place.

Lesson 2 (``steering_tutorials/hello_world_steering``) built ONE fixed steering
vector by contrastive diff-of-means: ``v = mean(harmful) - mean(benign)``, read
once from layer 12 and added to the residual stream to re-install refusal.

Lesson 3 replaces that hand-built vector with a *learned* one. We train a small
hypernetwork ``H_theta`` so that ``v = H(concept_embedding)`` — a nonlinear map
from a concept's exemplar activations to the steering direction. Two payoffs:

  1. It can MATCH the diff-of-means vector (a sanity floor), and
  2. It AMORTIZES: once trained, it emits a steering vector for a brand-NEW
     concept straight from that concept's exemplars, with no re-extraction pass.

  Sun et al. 2025, 'HyperSteer: Activation Steering at Scale with Hypernetworks'
    (arXiv:2506.03292) [UNVERIFIED] — the hypernetwork-generates-steering-vectors
    recipe this lesson reproduces at laptop scale.

Like lessons 1 and 2 this package is DELIBERATELY standalone: it does not import
from the research harness in ``src/steering``. It reuses only the lesson-2 model
plumbing (``hello_world_steering.model_utils``), everything else lives here.
"""
from __future__ import annotations

from pathlib import Path

# --- The model we steer ------------------------------------------------------
# The SAME uncensored / abliterated Gemma-3-1B used in lessons 1 and 2. Refusal
# has been ablated out of it, so there is behaviour to RE-INSTALL from outside —
# here with a hypernetwork-generated activation vector instead of a fixed one.
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# Which residual-stream layer we read the concept embedding from AND write the
# generated steering vector into — the same middle-ish layer lessons 1-2 used,
# so read (gate/concept) and write (steer) speak about the same representation.
STEER_LAYER = 12

# --- Hypernetwork architecture ----------------------------------------------
# Width of the hypernetwork's bottleneck: hidden -> BOTTLENECK -> hidden. Small
# enough to train in seconds on CPU-free laptop GPU, wide enough to be nonlinear.
BOTTLENECK = 256

# --- Training ----------------------------------------------------------------
# Adam LR, number of optimisation steps, and prompts per minibatch. The training
# objective (owned by the trainer module) pushes the hypernetwork's generated
# vector to make the model emit the refusal target on harmful prompts.
LR = 1e-3
STEPS = 300
BATCH = 4

# Steering strength (fraction of the residual-stream norm) used DURING training
# vs at EVALUATION. Training a touch weaker than eval keeps the gradient signal
# in the coherent regime; eval nudges slightly harder to read the effect.
ALPHA_TRAIN = 0.08
ALPHA_EVAL = 0.10

# The text the hypernetwork is trained to elicit on harmful prompts — a clean,
# short refusal used as the language-modelling target for the steered forward.
REFUSAL_TARGET = "I can't help with that request."

SEED = 0

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
NET_PATH = ARTIFACTS / "hypersteer.pt"
RESULTS_PATH = ARTIFACTS / "results.json"

ARTIFACTS.mkdir(exist_ok=True)
