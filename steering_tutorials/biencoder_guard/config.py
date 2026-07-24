"""config.py -- every knob for the BI-ENCODER GUARDRAIL lesson.

The dual-tower / policy-matching member of the safety-detection course:

  multiturn_jailbreak : the attack hides across CONVERSATION TURNS
  trajguard           : ...across GENERATED TOKENS
  cross_trajectory    : ...across MULTIPLE AGENTS / SESSIONS
  meerkat             : sparse violations hidden across MANY TRACES (clustering)
  biencoder_guard     : moderate content against a LARGE POLICY TAXONOMY cheaply
                        by matching a CONTENT tower against a cached POLICY tower
                        (this lesson)

The thesis (GLiNER bi-encoder "The Million-Label NER", Stepanov et al.,
arXiv:2602.18487; GLiNER Guard "Unified Encoder Family for Production LLM Safety
and Privacy", Minko et al., arXiv:2605.05277): a UNI-encoder that concatenates
text + all label descriptions and attends over them jointly is accurate but its
cost grows with the number of labels -- it collapses past a few dozen. A
BI-ENCODER decouples the towers: the CONTENT tower embeds the text, the POLICY
tower embeds each policy description, and compatibility is a cheap cosine in a
shared space. Because the policy vectors are independent of the text, they are
embedded ONCE and cached, so per-request cost is ~constant in the number of
labels, and a NEW policy is added zero-shot by embedding its description alone.

We use EmbeddingGemma-300M (google/embeddinggemma-300m, Gemma-3 based, 768-dim
with Matryoshka truncation) as the shared backbone -- the closest open realization
of the "gemini embed 300m" tower the papers describe.

Two May-2026 siblings sharpen the design and we fold in the best of each:
  * Opir (Stepanov & Smechov, arXiv:2605.29659) pushes the label space to a
    three-level, 996-category taxonomy -- motivating our MANY-label policy bank
    and the zero-shot-on-unseen-policy experiment.
  * GLiGuard (Zaratiana et al., arXiv:2605.07982) is a 0.3B schema-conditioned
    encoder that folds task + label semantics into the input; the broader 2026
    encoder-guardrail practice of SYNTHETIC SCHEMA EXPANSION (many natural-language
    paraphrases of the same policy) motivates our multi-prototype policy tower
    ablation below.

DELIBERATELY standalone: reuses the shared `common.data` helpers where useful but
imports nothing from the research harness. Detection task -> NO generation judge.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name) or default)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


# --- Embedder: EmbeddingGemma-300M (the shared bi-encoder backbone) ----------
# google/embeddinggemma-300m is GATED (Gemma license). Load via sentence-
# transformers; if the HF id 401s without a token, fall back to a local snapshot
# under models/. "minilm" is a fast, ungated substitute for a quick dry run.
EMBED_MODEL = _env_str("BG_EMBED_MODEL", "google/embeddinggemma-300m")
EMBED_LOCAL = _env_str("BG_EMBED_LOCAL", "models/google/embeddinggemma-300m")
EMBEDDER = _env_str("BG_EMBED", "embeddinggemma")   # "embeddinggemma" | "minilm"
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
# EmbeddingGemma is trained with task-prefixed prompts; the content tower and the
# policy tower use different prompts (asymmetric retrieval). The encoder module
# owns the exact prompt strings (from the model card); these name which to use.
CONTENT_PROMPT = _env_str("BG_CONTENT_PROMPT", "query")     # embed the text-to-moderate
POLICY_PROMPT = _env_str("BG_POLICY_PROMPT", "document")    # embed the policy description
EMB_DIM = _env_int("BG_EMB_DIM", 768)     # Matryoshka truncation target (768/512/256/128)

# --- Data: a HARD, MULTI-DATASET, MANY-LABEL safety corpus -------------------
# The rubric (CLAUDE.md 17): >=500 positives AND >=500 negatives per class, hard
# negatives, a real combination of datasets. We pool three complementary public
# safety datasets into ONE unified multi-label corpus:
#   BeaverTails  : 14 fine-grained harm categories, multi-label, prompt+response
#                  -> the CORE many-label taxonomy (thousands/category available).
#   toxic-chat   : REAL user prompts with toxicity + jailbreak flags -> HARD,
#                  in-the-wild adversarial positives and topically-adjacent
#                  benign hard-negatives.
#   wildguardmix : adversarial prompt-harm labels -> HARD adversarial positives
#                  and benign-but-adversarial hard-negatives (leakage-free style).
# The label space is the union of these taxonomies; the policy TOWER matches
# against a written DESCRIPTION per label (so we can add unseen policies zero-shot).
BEAVERTAILS_DATASET = "PKU-Alignment/BeaverTails"
TOXICCHAT_DATASET = "lmsys/toxic-chat"
TOXICCHAT_CONFIG = "toxicchat0124"
WILDGUARD_DATASET = "allenai/wildguardmix"
WILDGUARD_CONFIG = "wildguardtrain"

N_PER_CLASS = _env_int("BG_N_PER_CLASS", 500)   # >=500 per harm category (rubric)
N_BENIGN = _env_int("BG_N_BENIGN", 500)         # >=500 benign hard-negatives (rubric)
# The many-label taxonomy. HELD-OUT policies are NEVER in any training split; they
# are detected zero-shot from their description alone (the bi-encoder headline).
N_HELDOUT_POLICIES = _env_int("BG_N_HELDOUT", 4)   # categories withheld for zero-shot
SEED = _env_int("BG_SEED", 0)

# --- Methods compared (keys are stable -- the results schema uses them) -------
#   bi_encoder    : HERO. cosine(content_vec, cached policy_desc_vec). Labels cached;
#                   scales O(1) per request in #labels; adds new policies zero-shot.
#   uni_encoder   : re-encode "text [SEP] policy: <desc>" per (text,label) pair and
#                   read a compatibility score. Accurate-ish but cost grows with
#                   #labels (must re-encode per label) -- the scaling bottleneck.
#   trained_head  : supervised one-vs-rest logistic on the content embedding. Strong
#                   ceiling on SEEN labels but CANNOT score an unseen policy.
METHODS = ["bi_encoder", "uni_encoder", "trained_head"]
THRESHOLD_GRID = [round(0.02 * i, 2) for i in range(0, 51)]   # cosine thresholds 0..1
N_FOLDS = _env_int("BG_FOLDS", 5)
BOOTSTRAP = _env_int("BG_BOOTSTRAP", 2000)

# --- Synthetic schema expansion (the Fastino/GLiGuard-style ablation) --------
# Instead of one description per policy, embed SEVERAL paraphrased phrasings and
# average them into a multi-prototype policy vector. This teaches the 2026
# "synthetic schema expansion" practice (diverse natural-language formulations of
# the same policy) WITHOUT an external GPT-4.1: paraphrases are handwritten /
# templated, with an OPTIONAL local-model generator. The ablation asks whether
# multi-prototype policy embeddings beat a single description on zero-shot policies.
POLICY_PARAPHRASES = _env_int("BG_PARAPHRASES", 4)   # descriptions averaged per policy
MULTIPROTO_ABLATION = _env_int("BG_MULTIPROTO", 1)   # 1 = run the 1-vs-P prototype ablation

# --- Hard-negative contrastive augmentation (the 2026 data-synthesis module) --
# The highest-leverage lever for a dual-encoder is the QUALITY of its hard
# negatives -- benign content that looks adversarial but must be pushed away. This
# module teaches the 2026 recipe end-to-end:
#   * dense mining (ANCE-style): the content tower itself retrieves the benign
#     texts most similar to each policy -- the true look-alikes.
#   * ECIsem diagnostic (Sinha et al., arXiv:2603.20990): a TRAINING-FREE score of a
#     mined negative set in EmbeddingGemma's own geometry (target-consistency +
#     semantic locality + lexical residual + diversity) -- measure before you train.
#   * CausalNeg counterfactuals (Zhang et al., arXiv:2606.01304): controlled
#     single-requirement perturbations (entity/constraint swaps) instead of free-form
#     generation, avoiding the generative-discriminative gap. Templated (no GPT-4.1).
#   * ARHN false-negative filter (Choi et al., arXiv:2604.11092): a policy-support
#     check so a near-miss that still violates the policy is NOT kept as a negative.
#   * a small CONTRASTIVE ADAPTER (InfoNCE, adaptive hardness weighting) on FROZEN
#     embeddings -- does sharpening the boundary cut false positives on hard negatives?
HARDNEG_PER_POLICY = _env_int("BG_HARDNEG", 20)    # mined hard negatives per policy
HARDNEG_MODULE = _env_int("BG_HARDNEG_ON", 1)      # 1 = run the hard-negative EXP-F
ADAPTER_DIM = _env_int("BG_ADAPTER_DIM", 256)      # contrastive projection width
ADAPTER_EPOCHS = _env_int("BG_ADAPTER_EPOCHS", 30)
CONTRASTIVE_TEMP = _env_float("BG_TEMP", 0.05)     # InfoNCE temperature
# NOTE: ADAPTER_CACHE and HARDNEG_PNG paths are defined in the Paths section below,
# after ARTIFACTS is created.

# --- Scaling microbenchmark (the "million-label" claim) ----------------------
# Time moderation of a fixed batch of texts as the label bank grows. bi-encoder =
# embed texts once + matmul against K cached vectors (flat); uni-encoder = re-encode
# texts x K (linear). Synthetic labels pad the real taxonomy up to LABEL_SCALES max.
LABEL_SCALES = [16, 64, 256, 1024]
SCALE_BATCH = _env_int("BG_SCALE_BATCH", 64)    # texts per throughput measurement

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "results.json"
# content embeddings cached per split x embedder; policy-bank cached per embedder.
EMB_CACHE = {(split, m): ARTIFACTS / f"emb_{split}_{m}.npz"
             for split in ("train", "test", "heldout", "ood")
             for m in ("embeddinggemma", "minilm")}
POLICY_CACHE = {m: ARTIFACTS / f"policy_bank_{m}.npz" for m in ("embeddinggemma", "minilm")}
PR_PNG = ARTIFACTS / "pr_by_method.png"            # precision-recall, seen policies
HELDOUT_PNG = ARTIFACTS / "heldout_zeroshot.png"   # zero-shot on unseen policies
SCALE_PNG = ARTIFACTS / "latency_vs_labels.png"    # the scaling claim
ADAPTER_CACHE = ARTIFACTS / "contrastive_adapter.pt"   # trained hard-negative adapter
HARDNEG_PNG = ARTIFACTS / "hardneg_fpr.png"        # FPR@recall: frozen vs adapter

ARTIFACTS.mkdir(exist_ok=True)
