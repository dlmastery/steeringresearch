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


# --- Embedder (the paper's exact model) --------------------------------------
# Meerkat embeds each trace with bge-base-en-v1.5. Loaded via `transformers`
# AutoModel + mean pooling (NO sentence-transformers dependency). "minilm" is a
# faster substitute for a quick run.
EMBED_MODEL = _env_str("MK_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDER = _env_str("MK_EMBED", "bge")     # "bge" | "minilm"
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"

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
RESULTS_PATH = ARTIFACTS / "results.json"
EMB_CACHE = {m: ARTIFACTS / f"trace_emb_{m}.npz" for m in ("bge", "minilm")}
CLUSTER_PNG = ARTIFACTS / "cluster_scatter.png"       # 2-D projection, attack cluster highlighted
AP_PNG = ARTIFACTS / "ap_vs_baserate.png"             # AP: clustering vs per-trace across base rate
SILHOUETTE_PNG = ARTIFACTS / "silhouette_k.png"

ARTIFACTS.mkdir(exist_ok=True)
