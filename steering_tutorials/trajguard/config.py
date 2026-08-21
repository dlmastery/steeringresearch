"""config.py — every knob for the TRAJGUARD (streaming decoding-time detection) lesson.

The thesis (TrajGuard, Liu, Liu, Li, Xin, Ding; ACL 2026 Findings, arXiv:2604.07727):
a jailbreak is visible in the model's OWN hidden-state trajectory *as it generates*.
As the model decodes a harmful completion, its residual-stream state at each generated
token drifts, token by token, toward a "high-risk" region; a benign completion stays
put. So you can DETECT (and stop) a jailbroken generation in real time by monitoring a
sliding window of decoding-time hidden states -- BEFORE the harmful content is fully
emitted -- without any prompt-side classifier.

THE COMPARATIVE CLAIM, AND THE SUBSTRATE IT NEEDS  (added 2026-08-08)
--------------------------------------------------------------------
The paper's central claim is comparative: hidden states during decoding carry
**stronger** risk signals **than input prompts**. That claim can only be exercised on
prompts whose SURFACE is benign and whose COMPLETION turns harmful. An overtly toxic
user input is not such a prompt: the abliterated model complies from token 1, so the
drift a trajectory monitor is supposed to read never happens, and a prompt-side
classifier reads the intent off the surface for free.

This lesson therefore ships TWO substrates, both drawn from the same toxic-chat pool,
and TEACHES the difference:

  SUBSTRATE = "overt"      toxicity==1 AND jailbreaking==0   -- overtly toxic inputs.
                           512 unique available => clears the >=500/class floor.
                           Expectation: NO trajectory signal beyond a per-token one.
  SUBSTRATE = "disguised"  jailbreaking==1                   -- attack wrappers
                           (roleplay / DAN-style) around harmful intent.
                           181 unique available => POOL-LIMITED, stated everywhere.
                           Expectation: this is where drift can exist at all.
  SUBSTRATE = "mixed"      toxicity==1 (both of the above)   -- the LEGACY substrate
                           the superseded 2026-07-27 run used. Kept only so the old
                           numbers remain reproducible; not a recommended arm.

toxic-chat ships the ``jailbreaking`` binary column alongside ``toxicity``; the shared
``common/data.py`` reads only ``toxicity``, so the disguised-attack subset was being
discarded at load time. ``trajguard/data.py`` selects it directly from the same CSV
using ``common/data.py``'s own filtering / dedup / length-matching primitives --
``common/`` is SHARED and is not edited by this lesson.

This is the SIBLING of `multiturn_jailbreak`:
  - multiturn_jailbreak: trajectory across CONVERSATION TURNS (chunk = a turn).
  - trajguard:           trajectory across GENERATED TOKENS  (chunk = a token).
Same core idea (classify a sequence of hidden states), different granularity. This
lesson REUSES `multiturn_jailbreak.models` (PerTurnMaxProbe / TrajectoryMLP / SeqGRU /
HierAttn) unchanged -- a token trajectory is the same [n_steps, dim] object a turn
trajectory is -- plus a TRAINING-FREE sliding-window projection detector (the paper's
own method) as the honest baseline.

DELIBERATELY standalone: reuses lesson-2's `hello_world_steering.model_utils`
(load_model / generate / num_layers), the shared `common.data` primitives, the shared
`common.confound` audit spine, and `multiturn_jailbreak.models`; imports nothing from
`src/steering`.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# --- Model (the abliterated Gemma we monitor) --------------------------------
# Local path (the gated HF id 401s without a token). Abliterated so it COMPLIES with
# harmful prompts -- which is exactly what produces a harmful token trajectory to catch.
MODEL_ID = _env_str("TG_MODEL_ID", "models/google/gemma-3-1b-it")
LAYER = _env_int("TG_LAYER", 12)          # residual layer whose per-token state we read

# --- Generation / capture ----------------------------------------------------
MAX_NEW_TOKENS = _env_int("TG_MAX_NEW_TOKENS", 40)  # trajectory length cap (per completion)
GREEDY = True                                        # deterministic decoding for reproducibility

# --- Data: WHICH SUBSTRATE ----------------------------------------------------
SUBSTRATES = ("overt", "disguised", "mixed")
SUBSTRATE = _env_str("TG_SUBSTRATE", "overt")
if SUBSTRATE not in SUBSTRATES:
    raise ValueError("TG_SUBSTRATE must be one of %s, got %r" % (list(SUBSTRATES), SUBSTRATE))

# Deduped unique-prompt pool per substrate, MEASURED on the toxic-chat 0124 full set
# with common/data.py's own filters (10 <= chars <= 2000, sha1-normalized grouping,
# prompts ambiguous across labels dropped). These are documentation, not a gate --
# data.py recomputes the pool every run and reports the ACHIEVED count.
POOL_MEASURED = {"overt": 512, "disguised": 181, "mixed": 693}

# THE ONE AUTHORITATIVE N. Previously this file said 500, README section 8 said 120,
# and results.json shipped 300 -- three numbers in three places, because load_or_build()
# silently reused a stale cache. There is now exactly one knob, the cache is
# fingerprinted against it, and the runner FAILS LOUDLY when achieved != requested for
# any reason other than a documented pool ceiling.
#
# 500 is the CLAUDE.md section 17 rule-1 floor and the "overt" pool (512) supports it.
# The "disguised" pool holds only 181 -- genuinely pool-limited (toxic-chat carries 204
# jailbreaking==1 annotations in total), which rule 2 permits *provided the lesson says
# so*, which it now does in every surface.
N_PER_CLASS = _env_int("TG_N_PER_CLASS", 500)
SEED = _env_int("TG_SEED", 0)
RULE1_FLOOR = 500                       # CLAUDE.md section 17 rule 1

# --- OOD arm (CLAUDE.md section 17 rule 5) ------------------------------------
# A real released HuggingFace benchmark, ungated, scored with the detectors trained on
# the in-domain substrate. `intrinsec-ai/cstm-bench` -- the benchmark rule 8 names for
# this lesson FAMILY -- is multi-session AGENT TRACE data (108 rows) and is neither the
# right granularity for a per-token lesson nor present in this host's HF cache, so it is
# not used here; see README section 12.
#
# `jackhhao/jailbreak-classification` IS cached and ungated: 629 unique jailbreak
# prompts (in-the-wild DAN/roleplay templates) vs 1,323 unique benign. It is a genuine
# DISTRIBUTION shift from toxic-chat's annotated jailbreaking column, which is what an
# OOD arm is for.
OOD_ENABLED = _env_flag("TG_OOD", True)
OOD_DATASET = "jackhhao/jailbreak-classification"
OOD_FILE = "default/jailbreak_dataset_full.csv"
# Its jailbreak prompts are far longer than its benign ones (median 1544 vs 234 chars),
# so decile length-matching alone leaves a residual length bar of 0.7570. Restricting
# the positives to the benign class's own p90 length ceiling drops that to 0.5627 at
# 274/class. Both numbers are MEASURED (CPU, no model) and both are reported; the
# restricted arm is the default because an OOD number under a 0.76 length bar is not
# an OOD number.
OOD_LENGTH_CAP_QUANTILE = float(os.environ.get("TG_OOD_LENCAP") or 0.90)
OOD_N_PER_CLASS = _env_int("TG_OOD_N_PER_CLASS", 400)   # capped by the pool; see data.py

# --- Training-free detector (the paper's method) -----------------------------
# Score(token) = projection of its (centered) hidden state onto the HARM DIRECTION
# (diff-of-means of harmful- vs benign-completion token states, fit on TRAIN only).
# A completion is flagged when the SLIDING-WINDOW mean of that score crosses tau.
WINDOW = _env_int("TG_WINDOW", 4)               # sliding-window length (tokens)
TARGET_FPR = 0.10                               # tau calibrated to this benign FPR on train

# --- Early-detection curve ---------------------------------------------------
# AUC as a function of how many generated tokens we have seen (the streaming value):
# can we flag the jailbreak from the first K tokens? Each K carries its OWN confound
# bar, computed on the same first-K truncation (section 10 committed to this and the
# superseded run never delivered it).
EARLY_KS = [2, 4, 8, 16, 32]

# --- Learned classifiers (reused from multiturn_jailbreak.models) ------------
METHODS = ["threshold_freeform", "per_turn_max", "trajectory_mlp", "seq_gru"]
N_FOLDS = _env_int("TG_FOLDS", 5)
# CLAUDE.md section 7 asks for >=10k resamples on any evaluation claim. It is seconds of
# CPU on 1,000 pooled predictions, so there is no reason to run below it.
BOOTSTRAP = _env_int("TG_BOOTSTRAP", 10000)

# --- The three confound CONTROLS (controls.py) -------------------------------
# README section 12.3 named these three as missing; they were code gaps, not reporting
# gaps. See `controls.py` for what each one asks.
#
# MATCHED_BINS: quantile bins of completion character length inside which each method's
# AUC is recomputed and then pooled by pair count. 4 bins on 362 items gives ~90 per bin
# (~45/class), which is enough for a within-bin AUC to mean something; raising it on a
# pool this size trades the confound control for noise. The ACHIEVED bin count is
# reported separately because ties in the stratifier can collapse edges.
MATCHED_BINS = _env_int("TG_MATCHED_BINS", 4)
# The within-bin bootstrap resamples inside bins and is evaluated once per method plus
# once for the binding bar, so it gets its own knob; it defaults to the same 10k.
MATCHED_BIN_BOOTSTRAP = _env_int("TG_MATCHED_BIN_BOOTSTRAP", BOOTSTRAP)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# Per-substrate artifacts: the arms must never overwrite each other's numbers, and a
# cache built for one substrate must never be silently consumed by another (that class
# of silent reuse is exactly what produced the 500-vs-300 contradiction).
RESULTS_PATH = ARTIFACTS / ("results_%s.json" % SUBSTRATE)
TRAJ_CACHE = ARTIFACTS / ("token_trajectories_%s.npz" % SUBSTRATE)
# Committed, text-free, ~100 kB: per-completion label / prompt group_id / token count /
# completion char length / mean+final hidden-state norm. Lets a fresh clone reproduce
# every NUMERIC confound bar with no GPU and no 100 MB blob, and lets anyone verify a
# regenerated cache is the same set. See README section 10.
META_PATH = ARTIFACTS / ("trajectory_meta_%s.json" % SUBSTRATE)
OOD_RESULTS_PATH = ARTIFACTS / ("results_ood_%s.json" % SUBSTRATE)
OOD_CACHE = ARTIFACTS / ("token_trajectories_ood_%s.npz" % SUBSTRATE)
OOD_META_PATH = ARTIFACTS / ("trajectory_meta_ood_%s.json" % SUBSTRATE)

ROC_PNG = ARTIFACTS / ("roc_by_method_%s.png" % SUBSTRATE)
EARLY_PNG = ARTIFACTS / ("early_detection_auc_%s.png" % SUBSTRATE)
DRIFT_PNG = ARTIFACTS / ("risk_drift_example_%s.png" % SUBSTRATE)

# The raw hidden-state cache is ~100 MB per 600 completions and is NOT committed
# (GitHub's hard per-file limit is 100 MiB and the 2026-07-22 cache was 101.18 MiB).
# It is written float16 + compressed to keep it as small as possible, and the
# text-free META_PATH sidecar is committed in its place. See README section 10.
CACHE_SIZE_WARN_BYTES = 95 * 1024 * 1024

ARTIFACTS.mkdir(exist_ok=True)
