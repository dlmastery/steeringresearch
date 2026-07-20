"""config.py — every knob for the multi-turn jailbreak DETECTION lesson.

The thesis (DeepContext, arXiv:2602.16935; Hierarchical Attention, arXiv:2606.21082):
a multi-turn jailbreak (Crescendo / ActorAttack, arXiv:2410.10700) hides the attack
in the TRAJECTORY across turns. Each individual turn looks benign ("What did chemist
Karen Wetterhahn study?"), so a per-turn / stateless probe misses it — but the
SEQUENCE of turn embeddings escalates, and a stateful sequence classifier (GRU,
hierarchical attention) over that sequence can catch the escalation.

This lesson is the temporal generalization of lesson 1 (`hello_world`): lesson 1
read "is THIS prompt harmful?" from one activation; this lesson reads "is this
CONVERSATION a multi-turn jailbreak?" from a *sequence* of per-turn activations.

DELIBERATELY standalone (like the other lessons): it reuses lesson-2's model
plumbing (`hello_world_steering.model_utils.mean_pool_activation`) for the Gemma
embedder and the shared `common` conventions, but imports nothing from the research
harness in `src/steering`.

Data (verified on HF, 2026-07):
  POSITIVES  SafeMTData/SafeMTData config Attack_600 — 600 ActorAttack multi-turn
             attacks; field `multi_turn_queries` = 4-5 escalating USER turns; also
             `category` (6 harm categories) + `query_id` (3 attack paths share a
             query_id -> group id, so the same target can't leak across CV folds).
  NEGATIVES  HuggingFaceH4/ultrachat_200k — real benign multi-turn assistant chats;
             field `messages` = [{role, content}]; we take USER turns, TOPIC-MATCH
             to the attack categories (hard, same-surface benign negatives), and
             TURN-COUNT match the positive distribution.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


# --- Embedders ---------------------------------------------------------------
# Each turn is embedded to ONE vector; a conversation is the SEQUENCE of those.
# Two embedders are compared (an ablation):
#   "gemma"  : lesson-1/2's abliterated Gemma-3-1B, layer-12 mean-pooled activation
#              (the course's residual-stream signal). Loaded from the LOCAL path
#              (the gated HF id 401s without a token).
#   "minilm" : sentence-transformers/all-MiniLM-L6-v2 loaded via `transformers`
#              AutoModel + mean pooling (NO sentence-transformers dependency).
GEMMA_MODEL_ID = _env_str("MJ_GEMMA_ID", "models/google/gemma-3-1b-it")
GEMMA_LAYER = _env_int("MJ_GEMMA_LAYER", 12)
MINILM_ID = _env_str("MJ_MINILM_ID", "sentence-transformers/all-MiniLM-L6-v2")
MINILM_DIM = 384

# Which embedder(s) to run: "gemma", "minilm", or "both".
EMBEDDERS = _env_str("MJ_EMBED", "both")

# --- Data --------------------------------------------------------------------
ATTACK_DATASET = "SafeMTData/SafeMTData"
ATTACK_CONFIG = "Attack_600"
BENIGN_DATASET = "HuggingFaceH4/ultrachat_200k"
BENIGN_SPLIT = "train_sft"

N_POS = _env_int("MJ_N_POS", 600)          # attacks (cap; 600 available)
N_NEG = _env_int("MJ_N_NEG", 600)          # benign multi-turn (balanced)
MIN_USER_TURNS = _env_int("MJ_MIN_TURNS", 3)   # a conversation needs >= this many user turns
MAX_USER_TURNS = _env_int("MJ_MAX_TURNS", 8)   # cap sequence length (truncate the tail)
SEED = _env_int("MJ_SEED", 0)

# HARD condition: both classes are exactly this many turns (length-matched). Positive
# = an attack's LAST HARD_WINDOW turns (contains the payload); negative = a DIFFERENT
# attack's FIRST HARD_WINDOW turns (benign lead-up, excludes the payload). Attacks are
# 4-5 turns, so 4 keeps essentially all of them while removing the turn-count confound.
HARD_WINDOW = _env_int("MJ_HARD_WINDOW", 4)

# --- Sequence classifiers (pure torch, CPU-trainable on embedded sequences) --
# Screening-scale hyperparameters — enough to show the ordering, not tuned.
N_FOLDS = _env_int("MJ_FOLDS", 5)          # group-aware CV folds (group = query_id / conv id)
EPOCHS = _env_int("MJ_EPOCHS", 40)
LR = 1e-3
BATCH = 32
GRU_HIDDEN = 64
MLP_HIDDEN = 64
ATTN_HIDDEN = 64
BOOTSTRAP = _env_int("MJ_BOOTSTRAP", 2000)  # resamples for the 95% metric CIs

# The four classifiers compared (keys are stable — the results schema uses them):
METHODS = ["per_turn_max", "trajectory_mlp", "seq_gru", "hier_attn"]

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "results.json"
# Per-embedder sequence cache (list of [n_turns, dim] float32 arrays + labels/groups).
EMB_CACHE = {m: ARTIFACTS / f"seqs_{m}.npz" for m in ("gemma", "minilm")}
ROC_PNG = ARTIFACTS / "roc_by_method.png"
BAR_PNG = ARTIFACTS / "auc_by_method_embedder.png"
TRAJ_PNG = ARTIFACTS / "risk_trajectory_example.png"

ARTIFACTS.mkdir(exist_ok=True)

# Harm categories present in Attack_600 (for TOPIC-MATCHING the benign negatives).
# Keyword sets per category define a "sensitive-adjacent" benign conversation: a
# real UltraChat chat that DISCUSSES the same surface topic but never escalates.
ATTACK_CATEGORIES = [
    "chemical_biological", "illegal", "misinformation_disinformation",
    "harmful", "harassment_bullying", "cybercrime_intrusion",
]
CATEGORY_KEYWORDS = {
    "chemical_biological": ["chemical", "chemistry", "toxic", "poison", "pathogen",
                            "virus", "bacteria", "reagent", "compound", "synthesis",
                            "laboratory", "biology", "molecule"],
    "illegal": ["law", "legal", "illegal", "crime", "police", "court", "weapon",
                "firearm", "smuggl", "counterfeit", "regulation"],
    "misinformation_disinformation": ["misinformation", "disinformation", "rumor",
                                       "propaganda", "fake news", "fact-check",
                                       "conspiracy", "media", "journalism"],
    "harmful": ["danger", "risk", "safety", "hazard", "injury", "accident",
                "emergency", "harm"],
    "harassment_bullying": ["harassment", "bully", "insult", "abuse", "threat",
                            "conflict", "workplace", "school"],
    "cybercrime_intrusion": ["network", "security", "hacker", "malware", "password",
                             "encryption", "firewall", "server", "software",
                             "vulnerability", "phishing", "cyber"],
}
