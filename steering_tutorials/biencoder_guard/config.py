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

from steering_tutorials.biencoder_guard import encoder_behaviour as _EB


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
# negatives, a real combination of datasets. We pool complementary public safety
# datasets into ONE unified multi-label corpus:
#   BeaverTails  : 14 fine-grained harm categories, multi-label, prompt+response
#                  -> the CORE many-label taxonomy.
#   Aegis 2.0    : NVIDIA's 12 core + 9 fine-grained safety taxonomy, prompt AND
#                  response, separate human/LLM label columns, 33,416 rows. A
#                  SECOND independent annotation regime over a DIFFERENT taxonomy
#                  -- the fix for the single-dataset collapse documented in
#                  AUDIT_2026-08.md section A1 (BeaverTails was 93.5% of the corpus).
#   toxic-chat   : REAL user prompts with toxicity + jailbreak flags -> HARD,
#                  in-the-wild adversarial positives and topically-adjacent
#                  benign hard-negatives.
#   wildguardmix : GATED. Has NEVER loaded on this host (HTTP 403, no HF token);
#                  it contributes ZERO rows. Kept wired so a tokened host gets it,
#                  but no claim in this lesson may rest on it.
# The label space is the union of these taxonomies; the policy TOWER matches
# against a written DESCRIPTION per label (so we can add unseen policies zero-shot).
BEAVERTAILS_DATASET = "PKU-Alignment/BeaverTails"
BEAVERTAILS_TRAIN_SPLIT = _env_str("BG_BT_SPLIT", "30k_train")
BEAVERTAILS_TEST_SPLIT = _env_str("BG_BT_TEST_SPLIT", "30k_test")
TOXICCHAT_DATASET = "lmsys/toxic-chat"
TOXICCHAT_CONFIG = "toxicchat0124"
WILDGUARD_DATASET = "allenai/wildguardmix"
WILDGUARD_CONFIG = "wildguardtrain"
# Aegis 2.0 -- VERIFIED ungated (2026-08-08, HF dataset card + datasets-server
# first-rows). 30,007 train / 1,445 validation / 1,964 test. Columns:
#   id, reconstruction_id_if_redacted, prompt, response, prompt_label,
#   response_label, violated_categories, prompt_label_source, response_label_source
# prompt_label / response_label are in {"safe","unsafe"} (response_label may be null);
# violated_categories is a ", "-joined string of category names.
AEGIS_DATASET = _env_str("BG_AEGIS_DATASET", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
AEGIS_TRAIN_SPLIT = _env_str("BG_AEGIS_SPLIT", "train")
AEGIS_TEST_SPLIT = _env_str("BG_AEGIS_TEST_SPLIT", "test")
AEGIS_ON = _env_int("BG_AEGIS_ON", 1)          # 1 = pool Aegis into the corpus
# Five Aegis categories have NO clean mapping onto the 16 existing policy columns
# (Criminal Planning/Confessions, Unauthorized Advice, Malware, Copyright/Trademark/
# Plagiarism, High Risk Gov. Decision Making). Rule: ADD a column rather than force
# the mapping. Set to 0 to keep the taxonomy at exactly 16 columns, in which case
# rows whose ONLY categories are unmapped are SKIPPED and counted -- never silently
# relabelled benign. The mapping table is in data._AEGIS_CROSSWALK / README section 6.
AEGIS_EXTRA_COLUMNS = _env_int("BG_AEGIS_EXTRA", 1)

N_PER_CLASS = _env_int("BG_N_PER_CLASS", 500)   # >=500 per harm category (rubric)
# BENIGN TARGET -- raised 500 -> 3000 (2026-08-08).
# WHY: at 500 the corpus was 5,226 harmful vs 500 benign = 91.3%/8.7%, and the
# ANCE-style hard-negative miner drew 240 of the ~350 benign rows in the train
# split -- 69% of the pool, i.e. a near no-op, because there was almost nothing
# left to NOT select. That is why EXP-F's frozen baseline sits at
# fpr_at_recall90 = 1.000, the literal worst possible value. Dense mining is only
# meaningful when the pool is much larger than the selection. At 3000 the train
# split holds ~2,100 benign rows, so 240 mined is ~11% of the pool (a ~6x larger
# pool to be selective within) and the class balance moves to roughly 64/36.
# BeaverTails 30k_train and Aegis both have thousands of unused safe rows.
N_BENIGN = _env_int("BG_N_BENIGN", 3000)        # benign hard-negatives (see above)
# The many-label taxonomy. HELD-OUT policies are NEVER in any training split; they
# are detected zero-shot from their description alone (the bi-encoder headline).
N_HELDOUT_POLICIES = _env_int("BG_N_HELDOUT", 4)   # categories withheld for zero-shot
SEED = _env_int("BG_SEED", 0)

# --- Transfer arms: what "OOD" is allowed to mean ----------------------------
# AUDIT_2026-08.md section A4: the single arm this lesson used to call "OOD" was
# BeaverTails/30k_test -- the SAME dataset, annotators, taxonomy and rendering as
# 93.5% of train. That is SPLIT transfer, not DISTRIBUTION transfer, and it is now
# named `heldout_split` so the name cannot overstate it again. Three arms, in
# increasing order of how much actually changes:
#   heldout_split   : BeaverTails 30k_test        -- rows change, nothing else
#   cross_annotator : Aegis 2.0 test split        -- different annotators + taxonomy
#   ood_benchmark   : intrinsec-ai/cstm-bench     -- a released external benchmark,
#                     a different task shape entirely (multi-session attack traces)
# CSTM-Bench is the benchmark CLAUDE.md section 17 rule 8 names for this lesson
# family. VERIFIED ungated and already present in this host's HF cache
# (default config; splits 'dilution' + 'cross_session'; fields scenario_class /
# sessions_json / ground_truth_json).
CSTM_DATASET = _env_str("BG_CSTM_DATASET", "intrinsec-ai/cstm-bench")
CSTM_SPLITS = ("dilution", "cross_session")
CSTM_MAX_CHARS = _env_int("BG_CSTM_MAX_CHARS", 6000)   # per-scenario text cap
TRANSFER_ARMS = [a for a in _env_str(
    "BG_TRANSFER_ARMS", "heldout_split,cross_annotator,ood_benchmark").split(",") if a.strip()]

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

# --- Large-label-scale ACCURACY (the paper's SECOND scaling claim) -----------
# arXiv:2602.18487 claims BOTH that bi-encoder latency stays flat AND that its
# ACCURACY is MAINTAINED as the label bank grows to ~1000. LABEL_SCALES above
# tests only the latency half. This block tests the ACCURACY half: pad the 16
# real policies with SYNTHETIC DISTRACTOR policies (distractors.py) up to K, score
# the test set against all K, and compute the metrics over ONLY the 16 real
# columns -- the distractors are competing candidates that can steal rank/argmax
# but are never scored as answers, so the numbers stay comparable to EXP-A/EXP-B.
LABEL_SCALE_ACC = _env_int("BG_LS_ON", 1)          # 1 = run the accuracy-at-scale EXP-G
_ls_env = _env_str("BG_LS_K", "")
LABEL_SCALE_ACC_K = ([int(x) for x in _ls_env.replace(",", " ").split()]
                     if _ls_env else [16, 64, 256, 900])
LABEL_SCALE_TOP_T = _env_int("BG_LS_TOPT", 3)      # top-T competitive routing rule
LABEL_SCALE_PROTO = _env_int("BG_LS_PROTO", POLICY_PARAPHRASES)  # prototypes per distractor
LABEL_SCALE_ARMS = ["bi_encoder", "bi_encoder_trained"]          # frozen vs trained
# Distractor generation: fixed seed => a reproducible bank; the Jaccard gate keeps
# a distractor from being a covert paraphrase of a real policy (which would show up
# as a FALSE NEGATIVE and be misread as accuracy lost at scale).
DISTRACTOR_SEED = _env_int("BG_DISTRACTOR_SEED", 12345)
DISTRACTOR_MAX_JACCARD = _env_float("BG_LS_MAXJAC", 0.20)

# --- The OPIR TAXONOMY test (EXP-H): RELATED competitors, not strangers -------
# EXP-G's 884 distractors are UNRELATED by construction (disjoint domain grid,
# harm-stem blocklist, Jaccard <= DISTRACTOR_MAX_JACCARD against every real phrasing).
# Opir (Stepanov & Smechov, arXiv:2605.29659) trains against "a three-level taxonomy
# containing 996 categories across 16 top-level labels, 126 mid-level labels, and 854
# leaf labels" -- where a category's competitors are its SIBLINGS and DESCENDANTS, i.e.
# plausible near-misses. taxonomy.py builds exactly that shape below our 16 real
# policies, and EXP-H scores the test set against all 996.
#
# The counts below are Opir's, not ours; changing them changes the experiment, so they
# are named constants rather than magic numbers, and taxonomy.build_taxonomy ASSERTS
# n_top + OPIR_N_MID + OPIR_N_LEAF == OPIR_TOTAL rather than trusting them.
OPIR_MODULE = _env_int("BG_OPIR_ON", 1)            # 1 = run the taxonomy test EXP-H
OPIR_N_MID = _env_int("BG_OPIR_MID", 126)          # Opir's mid-level label count
OPIR_N_LEAF = _env_int("BG_OPIR_LEAF", 854)        # Opir's leaf label count
OPIR_TOTAL = _env_int("BG_OPIR_TOTAL", 996)        # 16 + 126 + 854, asserted
OPIR_SEED = _env_int("BG_OPIR_SEED", 20265)        # recorded; used only if subsampling
OPIR_ARMS = list(LABEL_SCALE_ARMS)                 # same two arms as EXP-G (frozen/trained)
OPIR_PROTO = _env_int("BG_OPIR_PROTO", POLICY_PARAPHRASES)   # prototypes per node
# The matched-K comparison against EXP-G: EXP-G's largest bank is 900 columns, EXP-H's
# is 996, so we ALSO slice EXP-H to this many columns for an exact-K head-to-head
# (and report the scale-free percentile rank, which needs no matching at all).
OPIR_MATCH_K = _env_int("BG_OPIR_MATCH_K", 900)
# Opir's 16 / 126 / 854 = 996 shape is exact ONLY while the real taxonomy has 16
# top-level policies. Pooling Aegis with AEGIS_EXTRA_COLUMNS=1 adds columns, and
# build_taxonomy() would then raise rather than silently mis-shape (correct). With
# AUTOFIT on, the runner re-derives mid/leaf to preserve the 996 TOTAL and the
# 126:854 ratio at the new top count, records BOTH shapes in results.json, and says
# loudly that it deviated from the paper's exact split. With it off, EXP-H is
# skipped with a stated reason instead of running on a shape nobody chose.
OPIR_AUTOFIT = _env_int("BG_OPIR_AUTOFIT", 1)
OPIR_PAPER_TOP = 16       # the top-level count at which Opir's 16/126/854 is exact

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "results.json"
# --- Cache filenames carry the ENCODER STACK ---------------------------------
# Every cached .npz below is a matrix of vectors, and a vector is a function of the
# encoder's BEHAVIOUR as much as of the text. On 2026-08-17 EmbeddingGemma was found
# to have been running strictly CAUSAL under transformers 4.55.0 while its config said
# `use_bidirectional_attention=True` (audits/AUDIT_2026-08-17_embeddinggemma_causal.md).
# Because the old filenames encoded only "which split, which embedder", a re-run after
# the library fix would have loaded the causal vectors straight back in.
#
# `_stack()` is a 12-char hash of {transformers, sentence-transformers, torch} versions
# + the model id -- read from installed-package METADATA, so building a path loads no
# library and no model. Caches from a different stack land on a different filename and
# can never be reached. The measured behavioural bucket cannot live in a filename (it
# needs a forward pass); it is stored in each .npz and checked on load by the runner.
_MODEL_ID = {"embeddinggemma": EMBED_MODEL, "minilm": MINILM_ID}


def _stack(m: str) -> str:
    return _EB.key_component(_MODEL_ID.get(m, m), {"emb_dim": EMB_DIM})


STACK_KEY = {m: _stack(m) for m in ("embeddinggemma", "minilm")}

# content embeddings cached per split x embedder; policy-bank cached per embedder.
EMB_CACHE = {(split, m): ARTIFACTS / f"emb_{split}_{m}_{STACK_KEY[m]}.npz"
             for split in ("train", "test", "heldout", "ood",
                           "heldout_split", "cross_annotator", "ood_benchmark")
             for m in ("embeddinggemma", "minilm")}
POLICY_CACHE = {m: ARTIFACTS / f"policy_bank_{m}_{STACK_KEY[m]}.npz"
                for m in ("embeddinggemma", "minilm")}
PR_PNG = ARTIFACTS / "pr_by_method.png"            # precision-recall, seen policies
HELDOUT_PNG = ARTIFACTS / "heldout_zeroshot.png"   # zero-shot on unseen policies
SCALE_PNG = ARTIFACTS / "latency_vs_labels.png"    # the scaling claim
ADAPTER_CACHE = ARTIFACTS / "contrastive_adapter.pt"   # trained hard-negative adapter
HARDNEG_PNG = ARTIFACTS / "hardneg_fpr.png"        # FPR@recall: frozen vs adapter
# Distractor policy bank, cached per embedder. The cache stores the generator
# FINGERPRINT alongside the matrix; a mismatch invalidates it (stamp your inputs).
DISTRACTOR_CACHE = {m: ARTIFACTS / f"distractor_bank_{m}_{STACK_KEY[m]}.npz"
                    for m in ("embeddinggemma", "minilm")}
LABEL_SCALE_PNG = ARTIFACTS / "label_scale_accuracy.png"   # accuracy vs #labels
# EXP-H: the 980 mid+leaf taxonomy vectors, cached per embedder with the generator
# fingerprint stored alongside (a mismatch invalidates the cache, as in EXP-G).
TAXONOMY_CACHE = {m: ARTIFACTS / f"opir_taxonomy_{m}_{STACK_KEY[m]}.npz"
                  for m in ("embeddinggemma", "minilm")}
OPIR_PNG = ARTIFACTS / "opir_taxonomy.png"                 # related vs unrelated, EXP-H

ARTIFACTS.mkdir(exist_ok=True)
