"""config.py — every knob for the CROSS-TRAJECTORY LATENT-AGGREGATION lesson.

The agent-level capstone of the trajectory-detection trilogy:
  multiturn_jailbreak : the attack hides in the trajectory across CONVERSATION TURNS
  trajguard           : ...across GENERATED TOKENS (streaming, decoding-time)
  cross_trajectory    : ...across MULTIPLE AGENTS / SESSIONS (this lesson)

The thesis (Context-Fractured Decomposition, arXiv:2606.09084; Cross-Session
Threats, arXiv:2604.21131; GroupGuard, arXiv:2603.13940): a capable adversary
DECOMPOSES a harmful goal into K individually-innocuous sub-tasks and distributes
them across K separate agents / sessions / trajectories, so **no single trajectory
carries the payload** and any per-trajectory (session-bound) monitor is bypassed.
The defence is to AGGREGATE the K per-trajectory latent representations — a
permutation-invariant pooling over the *set* of trajectories — and classify the
AGGREGATE, recovering the harmful intent the parts hide. This is the multi-agent /
"swarm" generalization of the trajectory idea.

DELIBERATELY standalone: it REUSES `multiturn_jailbreak.embed` (the Gemma-3-1B
layer-12 mean-pool turn embedder — one trajectory's text is embedded like a turn)
and the shared `common.data`; it imports nothing from the research harness.

Data:
  MAIN (>=500/class, constructed): POSITIVES = SafeMTData Attack_600 decompositions
    (each attack's ~5 ActorAttack sub-queries become K=5 separate agent-trajectories;
    the harmful goal is split across them). NEGATIVES = K benign trajectories.
    Two conditions (like multiturn_jailbreak): "easy" (benign trajectories from
    UltraChat) and "hard" (same-style benign lead-up trajectories of a DIFFERENT
    attack -- leakage-free, so only the *presence of a complete decomposition*
    differs, which per-trajectory monitors cannot see).
  OOD (real benchmark): CSTM-Bench (intrinsec-ai/cstm-bench) -- cross-session attack
    scenarios (attack vs benign_pristine/benign_hard), each a set of ~26 sessions.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


# --- Embedder (reused from multiturn_jailbreak) ------------------------------
# Each TRAJECTORY (one agent's sub-task text) is embedded to ONE vector; a sample
# is the SET of K such vectors. We reuse the Gemma-3-1B layer-12 mean-pool embedder.
GEMMA_MODEL_ID = _env_str("CT_GEMMA_ID", "models/google/gemma-3-1b-it")
GEMMA_LAYER = _env_int("CT_GEMMA_LAYER", 12)
EMBEDDER = _env_str("CT_EMBED", "gemma")   # "gemma" | "minilm" (multiturn_jailbreak.embed)

# --- Data --------------------------------------------------------------------
ATTACK_DATASET = "SafeMTData/SafeMTData"
ATTACK_CONFIG = "Attack_600"
BENIGN_DATASET = "HuggingFaceH4/ultrachat_200k"
BENIGN_SPLIT = "train_sft"
OOD_DATASET = "intrinsec-ai/cstm-bench"      # the real cross-session benchmark

N_POS = _env_int("CT_N_POS", 500)            # decomposed attacks (600 available)
N_NEG = _env_int("CT_N_NEG", 500)            # benign multi-trajectory samples
K_TRAJ = _env_int("CT_K", 5)                 # trajectories (agents) per sample
CONDITION = _env_str("CT_CONDITION", "both")  # "easy" | "hard" | "both"
SEED = _env_int("CT_SEED", 0)

# --- Aggregation classifiers (over the SET of K trajectory embeddings) -------
# per_traj_max : the DECOMPOSITION-DEFEATED baseline (logistic per trajectory, MAX
#                over the set -> "does ANY single trajectory look harmful?"). No
#                aggregation; this is what the attack is built to bypass.
# mean_agg     : mean-pool the K embeddings -> MLP.
# attn_pool    : Set-Transformer pooling-by-multihead-attention (PMA), permutation-
#                invariant aggregation over the set (arXiv:1810.00825) -> classifier.
# gnn_agg      : message passing on a fully-connected graph over the K agents
#                (GroupGuard-style, arXiv:2603.13940) -> readout -> classifier.
METHODS = ["per_traj_max", "mean_agg", "attn_pool", "gnn_agg"]
N_FOLDS = _env_int("CT_FOLDS", 5)
EPOCHS = _env_int("CT_EPOCHS", 40)
LR = 1e-3
BATCH = 32
HIDDEN = 64
ATTN_HEADS = 4
BOOTSTRAP = _env_int("CT_BOOTSTRAP", 2000)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "results.json"
EMB_CACHE = {(cond, m): ARTIFACTS / f"traj_{cond}_{m}.npz"
             for cond in ("easy", "hard", "ood") for m in ("gemma", "minilm")}
ROC_PNG = ARTIFACTS / "roc_by_method.png"
BAR_PNG = ARTIFACTS / "auc_by_method_condition.png"
OOD_PNG = ARTIFACTS / "ood_cstm_bench.png"

ARTIFACTS.mkdir(exist_ok=True)
