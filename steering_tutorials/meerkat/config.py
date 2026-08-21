"""config.py — every knob for the MEERKAT trace-clustering lesson.

The UNSUPERVISED / clustering member of the cross-trajectory-attack detection
series (sibling of `cross_trajectory`, which does SUPERVISED set-aggregation):

  cross_trajectory : a LEARNED set-aggregator over K agent trajectories per sample.
  meerkat          : CLUSTER the embeddings of MANY traces in a repository and
                     surface the sparse, distributed safety violations that are
                     only visible when the whole repository is analysed together.

The thesis (Meerkat, Stein, Brown, Hassani, Naik & Wong, arXiv:2604.11806):
safety violations in agentic systems are often **sparse (~5%), complex, and
adversarially hidden**, detectable only when MANY traces are analysed together —
misuse campaigns, covert sabotage, reward hacking, prompt injection. Meerkat
embeds each trace with `bge-base-en-v1.5`, **k-means clusters** the repository, and
scores clusters to localize the violating traces. On distributed misuse it reaches
Average-Precision far above a **per-trace monitor** (which scores each trace alone
and misses the diluted campaign) — the clustering is what recovers the campaign
structure.

This lesson is a laptop-scale, faithful-in-spirit reconstruction: bge embeddings +
k-means + cluster-enrichment scoring, benchmarked against the per-trace baseline at
the paper's ~5% sparse base rate. It is DELIBERATELY standalone and reuses the
shared `common.data` + SafeMTData decompositions; the agentic-search reasoning step
of the paper is out of scope (we use a simpler cluster-density localizer).
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


# --- Embedder: THREE arms, because two mandates genuinely conflict ------------
# Meerkat embeds each trace with `bge-base-en-v1.5`, so "bge" is the PAPER-FIDELITY
# arm and stays the default. But CLAUDE.md section 17 and a standing user mandate
# name `google/embeddinggemma-300m` as this repo's encoder for embedding work, so
# "embeddinggemma" is the COMPLIANCE arm. Those two pull in opposite directions and
# neither is wrong; the resolution is to support BOTH and RUN both, which turns the
# encoder from an unexamined constant into an ABLATION AXIS -- "does the Meerkat
# ordering survive a change of encoder?" is a better question than either arm alone
# answers. "minilm" is the fast dry-run substitute: neither faithful nor compliant,
# so it is a smoke arm and never a headline.
EMBEDDER_CHOICES = ("bge", "minilm", "embeddinggemma")
EMBEDDER = _env_str("MK_EMBED", "bge")

# Per-arm checkpoints. Each is overridable on its OWN env var, so an override can
# never detach the id from the arm that loads it (see `model_id` below).
BGE_ID = _env_str("MK_BGE_ID", "BAAI/bge-base-en-v1.5")
MINILM_ID = _env_str("MK_MINILM_ID", "sentence-transformers/all-MiniLM-L6-v2")
# The HF id `google/embeddinggemma-300m` is gated and 401s without a token on this
# host, but the WEIGHTS ARE ON DISK (1.2 GB safetensors, verified 2026-08-08), so we
# load the LOCAL path -- the same resolution `cross_trajectory.embed_ct` uses.
EMBEDDINGGEMMA_ID = _env_str("MK_EG_ID", "models/google/embeddinggemma-300m")

_MODEL_ID = {"bge": BGE_ID, "minilm": MINILM_ID, "embeddinggemma": EMBEDDINGGEMMA_ID}


def model_id(embedder: str = None) -> str:
    """The checkpoint an arm ACTUALLY loads -- the single source of truth.

    WHY THIS IS A FUNCTION AND NOT A SECOND CONSTANT. `EMBED_MODEL` used to be an
    independent env var (`MK_EMBED_MODEL`) that defaulted to the bge id no matter
    which arm was selected, while `cluster.get_embedder` branched on `EMBEDDER` and
    loaded MiniLM's id from somewhere else entirely. The two never had to agree, and
    they didn't: `artifacts/results.json` says `embed_model = BAAI/bge-base-en-v1.5`
    beside `embedder = minilm`, and every number in it came from MiniLM (README
    section 9.2). The field was an inert echo of an unused default -- an artifact
    naming a model it never loaded, which is the CLAUDE.md 18.8 "stamp your inputs"
    failure in miniature.

    Now the loader and the results writer BOTH resolve through this one function, so
    the two fields cannot disagree without the run also loading a different model.
    An unknown arm raises here rather than silently resolving to a default.
    """
    e = str(embedder or EMBEDDER)
    if e not in _MODEL_ID:
        raise ValueError("embedder must be one of %r, got %r"
                         % (list(EMBEDDER_CHOICES), e))
    return _MODEL_ID[e]


# DERIVED from EMBEDDER -- deliberately NOT independently settable (see model_id).
EMBED_MODEL = model_id(EMBEDDER)

# --- EmbeddingGemma task prompts: why meerkat uses ONE, not two --------------
# EmbeddingGemma is trained for ASYMMETRIC retrieval and ships NAMED task prompts;
# a vector's meaning depends on which prompt prefixed the text, so omitting the
# prompt silently degrades the embedding. `biencoder_guard` is a retrieval-shaped
# task and therefore splits the asymmetry across its two towers (content -> "query",
# policy -> "document").
#
# meerkat is NOT retrieval-shaped. Every trace plays the SAME role: they go into one
# k-means over one homogeneous set, and into a kNN/logistic over those same vectors.
# Splitting query/document here would scatter interchangeable objects across two
# sub-spaces for no reason and corrupt the cluster geometry that IS the experiment.
# The model registers a prompt for exactly this task -- "Clustering",
# "task: clustering | query: " (model card / config_sentence_transformers.json) --
# and that is what every meerkat trace gets, pool traces and OOD traces alike.
EG_PROMPT_NAME = _env_str("MK_EG_PROMPT", "Clustering")
# Used verbatim only if the snapshot registers no prompt under that name; the loader
# says so loudly rather than falling through to an unprompted encode.
EG_PROMPT_FALLBACK = _env_str("MK_EG_PROMPT_TEXT", "task: clustering | query: ")
# EmbeddingGemma's own max_seq_length is 2048; we pin it to the bge/minilm window so
# the three arms differ ONLY in the backbone, not in how much of a trace they see.
# Traces here are single events (~100 chars), so nothing is truncated either way.
EG_MAXLEN = _env_int("MK_EG_MAXLEN", 512)

# --- Data (the trace pool; >=500/class per the rubric) -----------------------
# A "trace" = one agent's event sequence rendered to text. POSITIVES: SafeMTData
# Attack_600 decompositions (each attack's escalating ActorAttack sub-queries joined
# into one trace) -> a distributed-misuse campaign. NEGATIVES: benign UltraChat
# conversations rendered to traces. We hold a POOL of >=500/class and sample
# repositories from it at the sparse base rate below.
ATTACK_DATASET = "SafeMTData/SafeMTData"
ATTACK_CONFIG = "Attack_600"
BENIGN_DATASET = "HuggingFaceH4/ultrachat_200k"
BENIGN_SPLIT = "train_sft"
OOD_DATASET = "intrinsec-ai/cstm-bench"

N_ATTACK = _env_int("MK_N_ATTACK", 500)    # attack traces in the pool (>=500 rubric)
N_BENIGN = _env_int("MK_N_BENIGN", 500)    # benign traces in the pool (>=500 rubric)
SEED = _env_int("MK_SEED", 0)

# --- Repository sampling (the Meerkat setting) -------------------------------
# Meerkat's headline regime: violations are a SPARSE ~5% minority of a repository,
# distributed among benign traffic. We evaluate at that sparse rate (the AP metric)
# AND at a balanced rate (the ROC-AUC metric).
BASE_RATE = _env_float("MK_BASE_RATE", 0.05)   # sparse attack fraction per repository
REPO_SIZE = _env_int("MK_REPO_SIZE", 400)      # traces per sampled repository
N_REPOS = _env_int("MK_N_REPOS", 20)           # repositories to average metrics over

# --- Clustering + scoring ----------------------------------------------------
# k-means over the trace embeddings; K chosen by silhouette over a small grid.
K_GRID = [4, 6, 8, 10, 12, 16]
# Cluster localizer: score each trace by its cluster's violation-enrichment,
# estimated from a small labelled SEED (semi-supervised, mimicking the analyst who
# labels a handful of traces). SEED_FRAC of the repository is the labelled seed.
SEED_FRAC = _env_float("MK_SEED_FRAC", 0.10)
BOOTSTRAP = _env_int("MK_BOOTSTRAP", 1000)

# Methods compared (keys are stable — the results schema uses them):
#   per_trace     : the BASELINE monitor (logistic on each trace embedding alone).
#   kmeans_enrich : Meerkat-style (k-means + per-cluster enrichment from the seed).
#   knn_purity    : each trace scored by the attack-fraction of its k nearest traces
#                   (a clustering-free density proxy).
METHODS = ["per_trace", "kmeans_enrich", "knn_purity"]
KNN_K = _env_int("MK_KNN_K", 15)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
# PER-EMBEDDER, never a shared constant. This used to be the bare
# `results.json` while EMB_CACHE was already per-embedder, so running a second
# encoder SILENTLY OVERWROTE the first one's results -- which is exactly what
# happened on 2026-08-21 when the bge arm destroyed the minilm numbers (they
# were recovered from git as results_minilm.json). cross_trajectory hit the
# identical defect and fixed it the same way; the fix did not propagate here.
RESULTS_PATH = ARTIFACTS / ("results_%s.json" % EMBEDDER)
# One cache file per ARM, so three encoders can never contend for one path. The
# `bge`/`minilm` filenames are unchanged (existing artifacts stay reachable). The
# filename is the FIRST guard; `run_meerkat._embed_pool` also stamps the resolved
# embedder + model id INSIDE each .npz and refuses a file whose stamp disagrees.
EMB_CACHE = {m: ARTIFACTS / f"trace_emb_{m}.npz" for m in EMBEDDER_CHOICES}
CLUSTER_PNG = ARTIFACTS / "cluster_scatter.png"       # 2-D projection, attack cluster highlighted
AP_PNG = ARTIFACTS / "ap_vs_baserate.png"             # AP: clustering vs per-trace across base rate
SILHOUETTE_PNG = ARTIFACTS / "silhouette_k.png"

ARTIFACTS.mkdir(exist_ok=True)
