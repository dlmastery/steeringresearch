"""config.py — every knob for the CROSS-TRAJECTORY LATENT-AGGREGATION lesson.

The agent-level capstone of the trajectory-detection trilogy:
  multiturn_jailbreak : the attack hides in the trajectory across CONVERSATION TURNS
  trajguard           : ...across GENERATED TOKENS (streaming, decoding-time)
  cross_trajectory    : ...across MULTIPLE AGENTS / SESSIONS (this lesson)

The thesis (Context-Fractured Decomposition Attacks on Tool-Using LLM Agents,
arXiv:2606.09084; Cross-Session Threats in AI Agents, arXiv:2604.21131; GroupGuard,
arXiv:2603.13940): a capable adversary DECOMPOSES a harmful goal into K
individually-innocuous sub-tasks and distributes them across K separate agents /
sessions / trajectories, so **no single trajectory carries the payload** and any
per-trajectory (session-bound) monitor is bypassed. The defence is to AGGREGATE
the K per-trajectory latent representations — a permutation-invariant pooling over
the *set* of trajectories — and classify the AGGREGATE, recovering the harmful
intent the parts hide. This is the multi-agent / "swarm" generalization of the
trajectory idea.

  NOTE ON FIDELITY (2026-08 audit): arXiv:2606.09084 is about ONE tool-using
  agent's artifact-provenance gaps ACROSS STEPS, not K cooperating agents; this
  lesson's "hand each sub-task to a separate agent" framing is an EXTENSION of
  that paper, not what it says. arXiv:2603.13940 (GroupGuard) is TRAINING-FREE
  (graph monitoring + honeypot + structural pruning); `gnn_agg` here is a TRAINED
  message-passing classifier with none of that machinery. See README §2/§11.

DELIBERATELY standalone: it reuses `multiturn_jailbreak.embed`'s two embedder
backbones through `embed_ct.get_embedder` and the shared `common.confound`; it
imports nothing from the research harness.

Data:
  MAIN (constructed): POSITIVES = SafeMTData decompositions. TWO configs of the
    same ungated repo are used (`ATTACK_CONFIGS`): `Attack_600` (600 rows,
    `multi_turn_queries`) and `SafeMTData_1K` (1,680 rows, user turns extracted
    from `conversations`). The 1K config has MULTIPLE ACTORS PER `query_id`, so
    `n` and `n_distinct_groups` are reported SEPARATELY — inflating n without
    inflating information is not a fix.
    NEGATIVES = K benign trajectories. Two conditions: "easy" (UltraChat) and
    "hard" (same-style benign lead-up trajectories of a DIFFERENT attack, split by
    GROUP so no query_id straddles the halves).
  OOD (real benchmark): CSTM-Bench (intrinsec-ai/cstm-bench) — cross-session attack
    scenarios (attack vs benign_pristine/benign_hard), each a set of ~26 sessions.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _env_list(name: str, default):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [p.strip() for p in raw.split(",") if p.strip()]


# --- Embedder ----------------------------------------------------------------
# "embeddinggemma" is the encoder CLAUDE.md §17 / the standing user mandate names.
# Its WEIGHTS ARE ON DISK (models/google/embeddinggemma-300m, 1.2 GB safetensors,
# verified 2026-08-08) — the §18.5 "gated, no HF token" blocker is STALE for this
# model. "gemma" (a decoder LM's layer-12 residual) and "minilm" are therefore
# NON-COMPLIANT SUBSTITUTES, not necessities. The shipped headline is `minilm`.
EMBEDDER_CHOICES = ("embeddinggemma", "gemma", "minilm")
EMBEDDER = _env_str("CT_EMBED", "minilm")
if EMBEDDER not in EMBEDDER_CHOICES:
    raise ValueError("CT_EMBED must be one of %r, got %r" % (list(EMBEDDER_CHOICES), EMBEDDER))

GEMMA_MODEL_ID = _env_str("CT_GEMMA_ID", "models/google/gemma-3-1b-it")
GEMMA_LAYER = _env_int("CT_GEMMA_LAYER", 12)
EMBEDDINGGEMMA_ID = _env_str("CT_EMBEDDINGGEMMA_ID", "models/google/embeddinggemma-300m")
EMBEDDINGGEMMA_MAXLEN = _env_int("CT_EMBEDDINGGEMMA_MAXLEN", 512)

# --- Data --------------------------------------------------------------------
ATTACK_DATASET = "SafeMTData/SafeMTData"
# BOTH configs of the same ungated repo. Attack_600 alone capped `hard` at 298/class
# via the disjoint half-split — arithmetic, not a pool limit. SafeMTData_1K adds
# 1,680 rows and lifts `hard` over the >=500/class floor.
ATTACK_CONFIGS = _env_list("CT_ATTACK_CONFIGS", ["Attack_600", "SafeMTData_1K"])
ATTACK_CONFIG = ATTACK_CONFIGS[0]   # back-compat alias; loaders use ATTACK_CONFIGS

BENIGN_DATASET = "HuggingFaceH4/ultrachat_200k"
BENIGN_SPLIT = "train_sft"
OOD_DATASET = "intrinsec-ai/cstm-bench"      # the real cross-session benchmark

# OPTIONAL second source, ungated, verified to exist 2026-08-08 (Gibbs et al. 2024,
# arXiv:2409.00137): 382 Complete-Harmful decomposed multi-turn attacks plus a
# purpose-built pool of 1,200 SEMI-BENIGN conversations. The Semi-Benign pool is a
# REAL hard-negative source and would replace the synthesised payload-stripped
# prefixes `hard` currently uses. OFF by default: its exact column schema has NOT
# been verified on this host (no download was performed), so `data.load_tom_gibbs`
# probes the schema at runtime and fails loudly.
TOM_GIBBS_DATASET = "tom-gibbs/multi-turn_jailbreak_attack_datasets"
USE_TOM_GIBBS_SEMI_BENIGN = bool(int(os.environ.get("CT_TOM_GIBBS") or 0))

N_POS = _env_int("CT_N_POS", 500)            # decomposed attacks
N_NEG = _env_int("CT_N_NEG", 500)            # benign multi-trajectory samples
K_TRAJ = _env_int("CT_K", 5)                 # trajectories (agents) per sample
CONDITION = _env_str("CT_CONDITION", "both")  # "easy" | "hard" | "both"
SEED = _env_int("CT_SEED", 0)

# OOD session selection. CSTM-Bench scenarios carry ~26 sessions each; the original
# rule kept the 5 LONGEST, which is close to the worst available rule on a
# `dilution` split whose premise is dilution ACROSS MANY sessions. "uniform" spreads
# the pick evenly over the session list; OOD_K=0 keeps ALL sessions.
OOD_SELECT = _env_str("CT_OOD_SELECT", "uniform")   # "uniform" | "longest" | "all"
OOD_K = _env_int("CT_OOD_K", K_TRAJ)                # 0 => every session

# --- Aggregation classifiers (over the SET of K trajectory embeddings) -------
# per_traj_max : the DECOMPOSITION-DEFEATED baseline (logistic per trajectory, MAX
#                over the set -> "does ANY single trajectory look harmful?"). No
#                aggregation; this is what the attack is built to bypass.
# mean_agg     : [mean, max, std] pool over the K embeddings (3 x dim) -> MLP.
#                NOT a plain mean-pool — the per-dimension max and std are part of
#                the feature, and this is the top method, so the description matters.
# attn_pool    : Set-Transformer pooling-by-multihead-attention (PMA), permutation-
#                invariant aggregation over the set (arXiv:1810.00825) -> classifier.
# gnn_agg      : message passing on a fully-connected graph over the K agents
#                (GroupGuard-INSPIRED, arXiv:2603.13940 — that paper is training-free
#                and this is trained; see the fidelity note above) -> readout -> logit.
METHODS = ["per_traj_max", "mean_agg", "attn_pool", "gnn_agg"]
N_FOLDS = _env_int("CT_FOLDS", 5)
EPOCHS = _env_int("CT_EPOCHS", 40)
LR = 1e-3
BATCH = 32
HIDDEN = 64
ATTN_HEADS = 4
BOOTSTRAP = _env_int("CT_BOOTSTRAP", 2000)
# Standardized features are clipped before the MLP. Without this, OOD samples whose
# text is far longer than a 5-sub-query training sample land many sigmas out and the
# logit SATURATES TO A CONSTANT (measured: mean_agg AUC 0.5 with CI [0.5,0.5] on
# CSTM-Bench). "Saturated" and "near chance" are different failures.
FEATURE_CLIP = float(os.environ.get("CT_FEATURE_CLIP") or 5.0)

# --- Confound bars (steering_tutorials.common.confound) ----------------------
CONFOUND_CONTENT = bool(int(os.environ.get("CT_CONFOUND_CONTENT") or 1))
CONFOUND_SHUFFLE = bool(int(os.environ.get("CT_CONFOUND_SHUFFLE") or 1))

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
# PER-EMBEDDER. It used to be the constant `results.json`, so running the gemma arm
# OVERWROTE the minilm headline and the README's side-by-side table could not be
# produced by this code in one run or two.
RESULTS_PATH = ARTIFACTS / f"results_{EMBEDDER}.json"
# The shipped MiniLM artifact predates the per-embedder path and keeps its old name.
LEGACY_RESULTS_PATH = ARTIFACTS / "results.json"
ROC_PNG = ARTIFACTS / f"roc_by_method_{EMBEDDER}.png"
BAR_PNG = ARTIFACTS / f"auc_by_method_condition_{EMBEDDER}.png"
OOD_PNG = ARTIFACTS / f"ood_cstm_bench_{EMBEDDER}.png"

ARTIFACTS.mkdir(exist_ok=True)
