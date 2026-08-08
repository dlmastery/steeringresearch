"""config.py -- every knob for the multi-turn jailbreak DETECTION lesson.

The thesis (DeepContext, arXiv:2602.16935; Hierarchical Attention, arXiv:2606.21082):
a multi-turn jailbreak (Crescendo / ActorAttack, arXiv:2410.10700) hides the attack
in the TRAJECTORY across turns. Each individual turn looks benign ("What did chemist
Karen Wetterhahn study?"), so a per-turn / stateless probe misses it -- but the
SEQUENCE of turn embeddings escalates, and a stateful sequence classifier (GRU,
hierarchical attention) over that sequence can catch the escalation.

This lesson is the temporal generalization of lesson 1 (`hello_world`): lesson 1
read "is THIS prompt harmful?" from one activation; this lesson reads "is this
CONVERSATION a multi-turn jailbreak?" from a *sequence* of per-turn activations.

DELIBERATELY standalone (like the other lessons): it reuses lesson-2's model
plumbing (`hello_world_steering.model_utils.mean_pool_activation`) for the Gemma
decoder embedder and the shared `common` conventions (`common.confound` is the ONE
confound instrument), but imports nothing from the research harness in `src/steering`.

Data (verified on HF, cache contents read directly 2026-08-08):
  POSITIVES  SafeMTData/SafeMTData, BOTH configs:
               `Attack_600`    600 rows, `multi_turn_queries` = 4-5 USER turns,
                               + `category`, `query_id`, `plain_query`.
               `SafeMTData_1K` 1,680 rows, `conversations` = [{role,content}];
                               user-role turns are the escalation chain.
             CRITICAL: `query_id` is NOT a global key across the two configs
             (157/200 collide while `plain_query` collides 0 times), so the CV
             GROUP KEY is `plain_query`, never `query_id`. See data.attack_group_key.
  NEGATIVES  (easy)  HuggingFaceH4/ultrachat_200k -- real benign multi-turn chats,
                     `messages` = [{role, content}]; USER turns, TOPIC-MATCHED to the
                     attack categories and turn-count biased.
             (hard)  a DISJOINT half of the attack pool's own benign lead-ins.
             (opt.)  tom-gibbs/multi-turn_jailbreak_attack_datasets Semi-Benign --
                     multi-turn benign conversations built the same way as its
                     harmful ones. Ungated/MIT but NOT cached on this host.
  OOD        intrinsec-ai/cstm-bench -- ungated, MIT, cached. `ScaleAI/mhj` (the
             benchmark CLAUDE.md rule 5 names) is GATED and this host has no HF
             token, so cstm-bench is the substitute. See README section 8.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# --- Embedders ---------------------------------------------------------------
# Each turn is embedded to ONE vector; a conversation is the SEQUENCE of those.
# Three embedders are selectable:
#   "embgemma": google/embeddinggemma-300m via sentence-transformers, 768-d. THE
#               MANDATED EMBEDDER (CLAUDE.md section 17 / the standing embeddings
#               guidance) and the default headline arm. The CLAUDE.md section 18.5
#               "gated, no token" blocker is STALE for this model: the weights are
#               on disk at EMBGEMMA_LOCAL (model.safetensors, 1,211,486,072 bytes,
#               verified by size -- not by a directory listing).
#   "gemma"   : Gemma-3-1B decoder, layer-GEMMA_LAYER mean-pooled residual-stream
#               activation (the course's activation signal, lesson 1/2 plumbing).
#               Kept as the residual-stream comparison arm.
#   "minilm"  : sentence-transformers/all-MiniLM-L6-v2 via plain `transformers`.
#               LEGACY REFERENCE ARM ONLY -- MiniLM is explicitly not the course's
#               embedder; it is retained so the older numbers remain reproducible
#               and must never be headlined.
EMBGEMMA_ID = _env_str("MJ_EMBGEMMA_ID", "google/embeddinggemma-300m")
EMBGEMMA_LOCAL = _env_str("MJ_EMBGEMMA_LOCAL", "models/google/embeddinggemma-300m")
EMBGEMMA_DIM = _env_int("MJ_EMBGEMMA_DIM", 768)

GEMMA_MODEL_ID = _env_str("MJ_GEMMA_ID", "models/google/gemma-3-1b-it")
GEMMA_LAYER = _env_int("MJ_GEMMA_LAYER", 12)
MINILM_ID = _env_str("MJ_MINILM_ID", "sentence-transformers/all-MiniLM-L6-v2")
MINILM_DIM = 384

# Which embedder(s) to run: a comma-separated list, or the legacy aliases
# "both" (= gemma,minilm) and a single name. Default = the mandated embedder plus
# the residual-stream arm; MiniLM must be asked for by name.
EMBEDDERS = _env_str("MJ_EMBED", "embgemma,gemma")
KNOWN_EMBEDDERS = ("embgemma", "gemma", "minilm")
HEADLINE_EMBEDDER = _env_str("MJ_HEADLINE_EMBED", "embgemma")
# Embedders whose numbers may NOT be headlined (rule: EmbeddingGemma mandate).
LEGACY_EMBEDDERS = ("minilm",)

# --- Data --------------------------------------------------------------------
ATTACK_DATASET = "SafeMTData/SafeMTData"
# BOTH configs. Attack_600 alone caps HARD at ~300/class (measured); adding
# SafeMTData_1K lifts the ceiling past 1,000/class. Set MJ_ATTACK_CONFIGS to
# "Attack_600" to reproduce the pre-2026-08 pool.
ATTACK_CONFIGS = [s.strip() for s in
                  _env_str("MJ_ATTACK_CONFIGS", "Attack_600,SafeMTData_1K").split(",")
                  if s.strip()]
ATTACK_CONFIG = ATTACK_CONFIGS[0]   # back-compat alias (single-config callers)

BENIGN_DATASET = "HuggingFaceH4/ultrachat_200k"
BENIGN_SPLIT = "train_sft"

# Optional stronger hard-negative source for the EASY condition (rule 7): the same
# repo's Semi-Benign multi-turn conversations are built like its harmful ones, which
# is a far better matched negative than keyword-matched UltraChat. UNGATED/MIT but
# NOT cached on this host -- selecting it triggers a download.
TOMGIBBS_DATASET = "tom-gibbs/multi-turn_jailbreak_attack_datasets"
TOMGIBBS_NEG_CONFIGS = [s.strip() for s in
                        _env_str("MJ_TOMGIBBS_NEG", "Semi-Benign,Completely-Benign").split(",")
                        if s.strip()]
# "ultrachat" (default, cached) | "tomgibbs" (Semi-Benign, needs a download)
NEG_SOURCE = _env_str("MJ_NEG_SOURCE", "ultrachat")

# Rule 1 floor: >= 500 positives AND >= 500 negatives. EASY can reach 600/600 from
# Attack_600 + UltraChat with zero new dependencies; HARD's ceiling is pool-limited
# and is COMPUTED at load time (data.hard_pool_ceiling) rather than guessed.
N_POS = _env_int("MJ_N_POS", 600)
N_NEG = _env_int("MJ_N_NEG", 600)
RULE1_FLOOR = 500                              # CLAUDE.md section 17 rule 1
MIN_USER_TURNS = _env_int("MJ_MIN_TURNS", 3)   # a conversation needs >= this many user turns
MAX_USER_TURNS = _env_int("MJ_MAX_TURNS", 8)   # cap sequence length (truncate the tail)
SEED = _env_int("MJ_SEED", 0)

# HARD condition: both classes are exactly this many turns (length-matched). Positive
# = an attack's LAST HARD_WINDOW turns (contains the payload); negative = a DIFFERENT
# attack's FIRST HARD_WINDOW turns (benign lead-up, excludes the payload). Attacks are
# 4-5 turns in Attack_600 (and 1-7 in SafeMTData_1K), so 4 keeps essentially all of
# them while removing the turn-count confound.
HARD_WINDOW = _env_int("MJ_HARD_WINDOW", 4)

# --- Controls that decide what the lesson may claim ---------------------------
# SHUFFLE-TURN arm: permute each conversation's turn order (fixed seed) and re-run.
# If the sequence models keep their AUC under a permuted order, the signal is NOT
# the trajectory. Runs on the CACHED embeddings -- CPU, no GPU, no re-embed.
SHUFFLE_TURNS = _env_bool("MJ_SHUFFLE_ARM", True)
SHUFFLE_SEED = _env_int("MJ_SHUFFLE_SEED", 1234)

# --- OOD arm (rule 5) --------------------------------------------------------
# `ScaleAI/mhj` is VERIFIED GATED and unusable on this host (its local hub dir holds
# only a 40-byte refs/main). `intrinsec-ai/cstm-bench` is ungated, MIT and cached.
OOD_DATASET = "intrinsec-ai/cstm-bench"
OOD_SPLITS = [s.strip() for s in _env_str("MJ_OOD_SPLITS", "cross_session,dilution").split(",")
              if s.strip()]
OOD_ENABLED = _env_bool("MJ_OOD", True)
# cstm-bench scenarios carry 20-26 sessions; the trained models see HARD_WINDOW-turn
# windows, so the OOD conversation is the LAST OOD_WINDOW messages of the scenario --
# applied IDENTICALLY to both classes, so the window itself cannot separate them.
OOD_WINDOW = _env_int("MJ_OOD_WINDOW", HARD_WINDOW)
# scenario_class values in cstm-bench: "attack" | "benign_pristine" | "benign_hard".
OOD_POS_CLASSES = ("attack",)
OOD_NEG_CLASSES = ("benign_pristine", "benign_hard")

# --- Sequence classifiers (pure torch, CPU-trainable on embedded sequences) --
# Screening-scale hyperparameters -- enough to show the ordering, not tuned.
N_FOLDS = _env_int("MJ_FOLDS", 5)          # group-aware CV folds (group = plain_query / conv id)
EPOCHS = _env_int("MJ_EPOCHS", 40)
LR = 1e-3
BATCH = 32
GRU_HIDDEN = 64
MLP_HIDDEN = 64
ATTN_HIDDEN = 64
BOOTSTRAP = _env_int("MJ_BOOTSTRAP", 2000)  # resamples for the 95% metric CIs

# The five classifiers compared (keys are stable -- the results schema uses them).
# `last_turn_only` is the CONTROL THAT DECIDES THE LESSON'S CLAIM: a logreg on the
# FINAL turn embedding alone. `per_turn_max` is a MAX over turns and is NOT this
# control -- a model that merely learns "the last turn is the ask" would score high
# on per_turn_max while reading no trajectory at all. If `last_turn_only` matches the
# sequence models on HARD, the headline is "the payload turn is recognisable", not
# "escalation is detectable".
METHODS = ["per_turn_max", "last_turn_only", "trajectory_mlp", "seq_gru", "hier_attn"]
# Methods that read turn ORDER (the shuffle arm is informative only for these).
ORDER_SENSITIVE_METHODS = ("seq_gru",)

# --- Pre-registration (rule 8) -----------------------------------------------
# Stated HERE, in code, so the runner prints it BEFORE any number exists and it
# cannot be retrofitted to whatever came back. Mirrored in PREREGISTRATION.md.
PREREGISTRATION = {
    "registered": "2026-08-08",
    "condition": "hard",
    "claim": (
        "A multi-turn jailbreak hides the attack in the escalation TRAJECTORY, so a "
        "model that reads the ordered turn sequence detects it while a model that "
        "reads only a single turn does not."
    ),
    "falsifier_1_binding_bar": (
        "FALSIFIED if AUC(best sequence model) <= the BINDING CONFOUND BAR "
        "(common.confound.confound_report -> worst_auc: max over the length, count "
        "and content/TF-IDF bars, each folded directionless). Beating per_turn_max "
        "alone is NOT sufficient."
    ),
    "falsifier_2_last_turn": (
        "FALSIFIED if AUC(best sequence model) <= AUC(last_turn_only) + 0.02. If a "
        "logreg on the FINAL turn embedding alone matches the sequence models, the "
        "signal is the payload turn, not the trajectory."
    ),
    "falsifier_3_shuffle": (
        "The TRAJECTORY reading specifically (not merely the aggregation reading) is "
        "FALSIFIED if AUC(seq_gru, turn order permuted) >= AUC(seq_gru, true order) "
        "- 0.02. An order-insensitive result licenses only 'aggregating across turns "
        "beats a single turn', never 'only a stateful model can see it'."
    ),
    "tier": "SCREENING unless n >= 500/class AND >= 7 seeds (CLAUDE.md section 7).",
    "note": (
        "Registered BEFORE the 2026-08 re-run. The pre-2026-08 README stated only "
        "'AUC(sequence model) > AUC(per_turn_max)', which was neither in git before "
        "that run nor priced against a confound bar."
    ),
}

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "results.json"
# Per-(condition, embedder) sequence caches are named at the call site:
#   artifacts/seqs_<condition>_<embedder>.npz
# (the old module-level EMB_CACHE / ROC_PNG / BAR_PNG / TRAJ_PNG constants were dead
#  -- run_multiturn builds per-condition paths -- and have been removed.)

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


def embedder_list(sel: str = None):
    """Parse EMBEDDERS into a validated list. Accepts the legacy alias "both"."""
    raw = (sel if sel is not None else EMBEDDERS or "").strip().lower()
    if raw in ("", "both"):
        return ["gemma", "minilm"] if raw == "both" else list(KNOWN_EMBEDDERS[:2])
    names = [n.strip() for n in raw.replace(" ", ",").split(",") if n.strip()]
    bad = [n for n in names if n not in KNOWN_EMBEDDERS]
    if bad:
        raise ValueError("unknown embedder(s) %r; known: %r" % (bad, list(KNOWN_EMBEDDERS)))
    return names


def run_config() -> dict:
    """The ACHIEVED-config stamp written into results.json.

    section 18.8: "an artifact that cannot be regenerated from the code beside it is
    not evidence". The pre-2026-08 results.json recorded neither the requested n
    (the shipped 200/200 came from an undocumented env override while the config
    default said 600) nor HARD_WINDOW nor the embedder list. Everything a re-run
    depends on is stamped here; data.py adds the achieved counts and the pool hash.
    """
    return {
        "n_pos_requested": int(N_POS),
        "n_neg_requested": int(N_NEG),
        "min_turns": int(MIN_USER_TURNS),
        "max_turns": int(MAX_USER_TURNS),
        "hard_window": int(HARD_WINDOW),
        "seed": int(SEED),
        "n_folds": int(N_FOLDS),
        "epochs": int(EPOCHS),
        "bootstrap": int(BOOTSTRAP),
        "embedders": embedder_list(),
        "headline_embedder": HEADLINE_EMBEDDER,
        "legacy_embedders": list(LEGACY_EMBEDDERS),
        "methods": list(METHODS),
        "attack_dataset": ATTACK_DATASET,
        "attack_configs": list(ATTACK_CONFIGS),
        "neg_source": NEG_SOURCE,
        "benign_dataset": BENIGN_DATASET if NEG_SOURCE == "ultrachat" else TOMGIBBS_DATASET,
        "shuffle_arm": bool(SHUFFLE_TURNS),
        "shuffle_seed": int(SHUFFLE_SEED),
        "ood_enabled": bool(OOD_ENABLED),
        "ood_dataset": OOD_DATASET,
        "ood_splits": list(OOD_SPLITS),
        "ood_window": int(OOD_WINDOW),
        "embgemma_id": EMBGEMMA_ID,
        "embgemma_dim": int(EMBGEMMA_DIM),
        "gemma_model_id": GEMMA_MODEL_ID,
        "gemma_layer": int(GEMMA_LAYER),
        "minilm_id": MINILM_ID,
        "rule1_floor": int(RULE1_FLOOR),
    }
