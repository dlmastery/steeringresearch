"""config.py -- every knob for the streaming-trajectory-aggregation lesson.

One authoritative setting per knob, every one env-overridable (``STA_*``), mirroring
`trajguard/config.py`'s discipline: a single N_PER_CLASS here, not one number in this
file and a different one in a results.json (CLAUDE.md section 18.8's "500-vs-300"
defect). Per-corpus artifact paths -- never one constant path shared by every corpus
-- because that is exactly the defect that let `cross_trajectory`'s gemma arm overwrite
its MiniLM headline (this lesson's own brief names it).

WHY THE POOL CEILINGS BELOW ARE DOCUMENTATION, NOT A GATE
-----------------------------------------------------------
`DATASETS_VERIFIED.md` measured four corpora before any loader code existed. Those
numbers are recorded here so a reader can see, without downloading anything, what each
corpus can and cannot support -- but `corpora/*.py` recomputes the ACHIEVED pool at
load time and `Corpus.shortfall()` is the authority at runtime, never this file.

  shade      750 pos / 750 neg -- min 30 / median 131 / mean 132.2 / max 422 steps.
             The ONLY corpus with per-step oracle timing (ideal_flagging_step +
             point_of_no_return_step) -- the only one that can carry a lead-time claim.
             Clears the >=500/class floor (CLAUDE.md section 17 rule 1) cleanly.
  agentdojo  750 pos / 750 neg -- min 4 / median 8 / mean 9.4 / max 40 steps.
             The short-horizon CONTROL for shade. Lead = 0 in every valid positive
             BY CONSTRUCTION (DATASETS_VERIFIED.md, 17/17 sampled positives) -- it
             validates streaming ACCURACY, never advance-warning. Clears the floor.
  assebench  safety 641 neg / 835 pos (1476, three annotation standards on one pool --
             this lesson uses only `safety`+`security`, never the relabelled
             `strict`/`loose` duplicates); security 408 neg / 409 pos (817). No
             per-step oracle timing; `ideal_flagging_step`/`point_of_no_return_step`
             are always None on this corpus (assebench.py's own docstring). Clears
             the floor comfortably on the pooled safety+security pool.
  atbench    503 neg / 497 pos (1000 total). POSITIVES LAND 3 SHORT of the 500/class
             floor -- BORDERLINE, and every result built from it must say so (this is
             the lead's own framing in the brief; atbench.py repeats it). No
             per-step oracle timing either.

CPU-only to import. Loads NO model, downloads nothing, touches no network. ASCII
stdout only (Windows cp1252 console).
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


# --- Corpus --------------------------------------------------------------------
CORPORA = ("shade", "agentdojo", "assebench", "atbench")
CORPUS = _env_str("STA_CORPUS", "shade")
if CORPUS not in CORPORA:
    raise ValueError("STA_CORPUS must be one of %s, got %r" % (list(CORPORA), CORPUS))

# The single authoritative N. `corpora/*.py` returns whatever it can (`Corpus.
# shortfall()` reports the gap honestly) -- this is the REQUEST, not a promise.
N_PER_CLASS = _env_int("STA_N_PER_CLASS", 500)
SEED = _env_int("STA_SEED", 0)
RULE1_FLOOR = 500  # CLAUDE.md section 17 rule 1

# Documentation-only pool ceilings (see module docstring). Never used to clip a load;
# `Corpus.shortfall()["clears_rule_1"]` is the runtime authority.
POOL_MEASURED = {
    "shade": {"pos": 750, "neg": 750, "min_steps": 30, "median_steps": 131,
              "mean_steps": 132.2, "max_steps": 422, "has_lead_time": True,
              "clears_rule_1": True},
    "agentdojo": {"pos": 750, "neg": 750, "min_steps": 4, "median_steps": 8,
                  "mean_steps": 9.4, "max_steps": 40, "has_lead_time": False,
                  "lead_is_zero_by_construction": True, "clears_rule_1": True},
    "assebench": {"safety_pos": 835, "safety_neg": 641, "security_pos": 409,
                  "security_neg": 408, "has_lead_time": False, "clears_rule_1": True},
    "atbench": {"pos": 497, "neg": 503, "has_lead_time": False,
                "clears_rule_1": False, "note": "positives 3 short of the 500 floor "
                "-- borderline, state this in every result built from it"},
}

# --- Embedder (CLAUDE.md section 17 rule: EmbeddingGemma-300m everywhere) ------
# The weights ARE on disk (multiturn_jailbreak/embed.py verified model.safetensors,
# 1,211,486,072 bytes, by size) -- CLAUDE.md section 18.5's "gated, no token" note is
# stale for this specific model, though several DATASETS remain genuinely gated.
KNOWN_EMBEDDERS = ("embgemma", "gemma")
EMBEDDER = _env_str("STA_EMBEDDER", "embgemma")
if EMBEDDER not in KNOWN_EMBEDDERS:
    raise ValueError("STA_EMBEDDER must be one of %s, got %r" % (list(KNOWN_EMBEDDERS), EMBEDDER))
EMBGEMMA_ID = _env_str("STA_EMBGEMMA_ID", "google/embeddinggemma-300m")
EMBGEMMA_LOCAL = _env_str("STA_EMBGEMMA_LOCAL", "models/google/embeddinggemma-300m")
EMBGEMMA_DIM = _env_int("STA_EMBGEMMA_DIM", 768)

# The "gemma" alt embedder (decoder residual-stream mean pool over one step's text,
# reusing hello_world_steering.model_utils) is NOT the mandated default -- it exists
# as a documented fallback / ablation arm, same role it plays in multiturn_jailbreak.
GEMMA_MODEL_ID = _env_str("STA_GEMMA_MODEL_ID", "models/google/gemma-3-1b-it")
GEMMA_LAYER = _env_int("STA_GEMMA_LAYER", 12)

# Per-trajectory step cap applied at embedding time. Generously above every measured
# corpus max (SHADE's own max is 422) so it is a safety net against a pathological
# outlier trajectory, not an active truncation of the ladder under study -- lower it
# explicitly (env override) for a deliberate horizon-truncation experiment, and note
# that truncating below a trajectory's own point_of_no_return_step invalidates any
# lead-time claim on that trajectory (evaluate.py's lead-time arm does not consult
# this cap; it streams the full, uncapped trajectory).
MAX_STEPS = _env_int("STA_MAX_STEPS", 500)

# --- Evaluation ------------------------------------------------------------------
N_FOLDS = _env_int("STA_N_FOLDS", 5)
# CLAUDE.md section 7: an evaluation claim wants >=10k bootstrap resamples. This run
# is one corpus / one seed / one embedder (SCREENING tier per section 7), but the
# resampling itself is cheap (a pooled AUC over a few hundred trajectories), so there
# is no reason to run below the evaluation-grade count.
BOOTSTRAP = _env_int("STA_BOOTSTRAP", 10000)
# Held-out fraction for the ONE-CLASS streaming-monitor arm (ESNCusum / SafetyDrift):
# threshold calibration is TRAIN-ONLY (fit on TRAIN benign trajectories only), so the
# lead-time / false-alarm numbers are measured on a disjoint TEST split, group-aware.
LEAD_TIME_TEST_FRAC = float(os.environ.get("STA_LEAD_TEST_FRAC") or 0.30)

# FirstKSteps horizon-truncation curve (pooling.py's own stated purpose for the
# class): how much AUC is recoverable from an early prefix. Corpus-agnostic knob;
# AgentDojo's own median (8) and SHADE's tail are both inside this range.
EARLY_KS = [int(k) for k in _env_str("STA_EARLY_KS", "2,4,8,16,32,64").split(",") if k.strip()]

# --- WITHIN-CORPUS horizon control (horizon.py) -----------------------------------
# The k-grid for the truncation sweep. Defaults to EARLY_KS so the sweep and the
# prefix-coverage curve are read on the same abscissa.
HORIZON_KS = ([int(k) for k in _env_str("STA_HORIZON_KS", "").split(",") if k.strip()]
              or list(EARLY_KS))
# The two arms of the contrast. `Truncate` composes with any pooled aggregator, so these
# are names into horizon.INNER_FACTORIES rather than hardcoded classes.
HORIZON_INNER_LOW = _env_str("STA_HORIZON_INNER_LOW", "mean_pool")    # the claimed loser
HORIZON_INNER_HIGH = _env_str("STA_HORIZON_INNER_HIGH", "max_pool")   # the claimed winner
# Resamples for the PAIRED bootstrap on the delta (and on the delta-of-deltas). Separate
# knob from BOOTSTRAP because the paired loop evaluates several AUCs per resample; lower
# it to shrink a run into one foreground window, and label the result screening tier.
HORIZON_BOOTSTRAP = _env_int("STA_HORIZON_BOOTSTRAP", 10000)
# Run the (free, weaker, observational) median-split stratification as well.
HORIZON_MEDIAN_SPLIT = _env_flag("STA_HORIZON_MEDIAN_SPLIT", True)
HORIZON_RHO_FLOOR = float(os.environ.get("STA_HORIZON_RHO_FLOOR") or 0.7)
# --- VERDICT CONTEXT (verdict_context.py) -----------------------------------------
# Reported CONTEXT beside each pre-registered verdict -- never a criterion. The registered
# holds/fails logic in FALSIFIERS + run_sta._falsifier_verdicts is unchanged; these knobs
# only size the extra reporting that README section 8(e),(f) says is missing.
#
# Resamples for the PAIRED bootstrap on a falsifier's own margin (AUC_max - AUC_mean for
# F1/F2, best - last_step for F3) and for the per-comparison p-values Holm runs over.
# Keep it at evaluation grade: a bootstrap p floors at 2/(B+1), so a small B can make a
# comparison unrejectable at alpha/m no matter how large the effect is (the harness flags
# that as `resolution_floor_blocks_rejection` rather than letting it pass unremarked).
VERDICT_BOOTSTRAP = _env_int("STA_VERDICT_BOOTSTRAP", 10000)

# The shuffle control ON THE LADDER (F0's shuffle runs inside common.confound, on
# bag-of-words features -- it certifies the confound module, not the embedding-to-
# aggregator pipeline where the trained arms and the group-aware CV live).
LADDER_SHUFFLE = _env_flag("STA_LADDER_SHUFFLE", True)
# Which main-ladder methods to re-run on permuted labels. Default covers both fixed
# pooling (mean/max/last_step -- the arms F1 and F3 are read off) and the trained causal
# arm (gru), which is where a leak would most plausibly hide. `query_token_compressor` is
# OFF by default purely for cost (a second end-to-end training pass); add it via the env
# knob when the run has the budget -- and say which set ran, since results.json records
# `methods_requested` per run.
LADDER_SHUFFLE_METHODS = tuple(
    m.strip() for m in _env_str("STA_LADDER_SHUFFLE_METHODS",
                                "mean_pool,max_pool,last_step,gru").split(",") if m.strip())
LADDER_SHUFFLE_REPEATS = _env_int("STA_LADDER_SHUFFLE_REPEATS", 1)
# One permutation's AUC is itself a noisy estimate; the CI on it is not what the control
# turns on (the band check is), so this is deliberately cheaper than BOOTSTRAP.
LADDER_SHUFFLE_BOOTSTRAP = _env_int("STA_LADDER_SHUFFLE_BOOTSTRAP", 2000)

# k for the single QueryTokenCompressor instance run in the main ladder. NOT confirmed
# from TRACE's abstract (arXiv:2606.00611) -- sequence.py's own docstring: "K=16 is a
# harness convenience, not an assertion that 16 is TRACE's own value."
QUERY_TOKEN_K = _env_int("STA_QUERY_TOKEN_K", 16)

# --- Streaming monitors ----------------------------------------------------------
ESN_RESERVOIR_DIM = _env_int("STA_ESN_RESERVOIR_DIM", 100)
ESN_ALARM_PERCENTILE = float(os.environ.get("STA_ESN_ALARM_PERCENTILE") or 95.0)
MARKOV_N_STATES = _env_int("STA_MARKOV_N_STATES", 8)
MARKOV_HORIZON = _env_int("STA_MARKOV_HORIZON", 10)
MARKOV_ALARM_PERCENTILE = float(os.environ.get("STA_MARKOV_ALARM_PERCENTILE") or 95.0)

# --- Paths -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
EMBED_CACHE_DIR = ARTIFACTS / "embeddings"

# Per-corpus results path -- see module docstring; never a single constant shared by
# every corpus.
RESULTS_PATH = ARTIFACTS / ("results_%s.json" % CORPUS)
ROC_PNG = ARTIFACTS / ("roc_by_method_%s.png" % CORPUS)
HORIZON_PNG = ARTIFACTS / ("horizon_truncation_%s.png" % CORPUS)
HORIZON_DELTA_PNG = ARTIFACTS / ("horizon_within_corpus_%s.png" % CORPUS)
LEAD_TIME_PNG = ARTIFACTS / ("lead_time_%s.png" % CORPUS)

ARTIFACTS.mkdir(exist_ok=True)
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PRE-REGISTERED FALSIFIERS -- see run_sta.py for the evaluation logic. Recorded here,
# not invented at report time, so a HOLDS/FAILS verdict can never be quietly redefined
# after the numbers come in.
# ---------------------------------------------------------------------------
PREREGISTERED = "2026-08-11"
FALSIFIERS = [
    ("F0_shuffle_control",
     "Label-shuffle AUC (the content bar re-run with permuted labels) lands in "
     "[0.45, 0.55]. FAILS outside that band -- that is leakage, not signal, and "
     "invalidates every other falsifier below."),
    ("F1_mean_pool_collapses_long",
     "THE lesson's central hypothesis. On SHADE (median 131 steps), MeanPool AUC is "
     "materially below both MaxPool and GRU (margin > 0.02 against each). This "
     "repo's own cross_trajectory data CONTRADICTS the literature's headline "
     "(mean_agg topped that ladder at 0.9358) -- so it is genuinely under test here, "
     "not assumed."),
    ("F2_mean_pool_survives_short",
     "On AgentDojo (median 8 steps), MeanPool is NOT materially below MaxPool/GRU "
     "(margin <= 0.02 against each). F1 holds AND F2 holds together implicate the "
     "HORIZON as the cause; F1 holds AND F2 fails implicates the corpus instead."),
    ("F3_sequence_beats_laststep",
     "The best sequence/pooling method beats the LastStep control by > 0.02 AUC. "
     "Without this, 'trajectory detection' is indistinguishable from 'the final "
     "step alone gives it away'."),
    ("F4_streaming_clears_bars",
     "Both causal streaming monitors (ESNCusum, SafetyDriftMonitor) clear "
     "max(confound.worst_auc, first_step_rival.content.auc) on the offline-AUC "
     "reading of their peak score."),
    ("F5_lead_time_positive",
     "On SHADE, the median lead (over DETECTED positives) is > 0 steps. NOT "
     "APPLICABLE on AgentDojo, where available_lead is 0 for every valid positive "
     "by construction (DATASETS_VERIFIED.md) -- reported N/A there, never FAILS."),
]

# ---------------------------------------------------------------------------
# THE WITHIN-CORPUS HORIZON CLAIM -- registered SEPARATELY, and under its own later
# date, because it was added after FALSIFIERS above. Back-dating it to 2026-08-11 would
# make a post-hoc control look pre-registered, which is the exact defect the falsifier
# block exists to prevent. F1/F2 are unchanged; this neither replaces nor rescues them.
# ---------------------------------------------------------------------------
HORIZON_PREREGISTERED = "2026-08-20"
HORIZON_CLAIM = (
    "H-WITHIN: on ONE corpus, the max-pool-minus-mean-pool AUC gap GROWS with the "
    "truncation horizon k. Both arms see the identical first-k window of the identical "
    "trajectories, so task, generating agent, tool inventory, prose style, kind-of-"
    "positive and label machinery are held fixed by construction -- which the "
    "between-corpus SHADE-vs-AgentDojo contrast (F1 + F2) cannot do."
)
HORIZON_CRITERION = (
    "MET iff (a) Spearman rho of delta(k) against k over the non-degenerate finite "
    "k-cells is >= %.2f, AND (b) the 95%% PAIRED bootstrap CI on "
    "delta(k_max) - delta(k_min) excludes zero from above. Both parts required. A flat "
    "or falling delta(k) is evidence AGAINST the dilution mechanism on this substrate, "
    "and is to be reported as such rather than re-described as inconclusive."
    % HORIZON_RHO_FLOOR
)

# F2's registered text ABOVE contains a clause that does not follow: "F1 holds AND F2
# holds together implicate the HORIZON as the cause". It is left EXACTLY as registered --
# silently rewriting a pre-registration to match what we later understood is a worse
# defect than the original error, and would destroy the audit trail that caught it. The
# correction is recorded here instead, dated, and carried into results.json.
FALSIFIER_ADDENDA = [
    ("F2_mean_pool_survives_short", HORIZON_PREREGISTERED,
     "CORRECTION to the registered READING (the verdict logic is unchanged; only what may "
     "be concluded from it is): F1 and F2 are a BETWEEN-corpus contrast. SHADE and "
     "AgentDojo hold only the label machinery constant -- task suite, generating agent, "
     "tool inventory, environment, step prose style, base difficulty and the kind of "
     "positive all vary together with the horizon. So 'F1 holds AND F2 holds' is "
     "CONSISTENT WITH a horizon explanation and does NOT establish one. The claim that "
     "separates horizon from corpus is the within-corpus control (HORIZON_CLAIM above, "
     "computed in horizon.py); F1/F2 are its external replication, not its evidence."),
]

if __name__ == "__main__":
    print("[config] corpus=%s n_per_class=%d seed=%d embedder=%s folds=%d bootstrap=%d"
          % (CORPUS, N_PER_CLASS, SEED, EMBEDDER, N_FOLDS, BOOTSTRAP))
    print("[config] max_steps=%d lead_time_test_frac=%.2f early_ks=%s query_token_k=%d"
          % (MAX_STEPS, LEAD_TIME_TEST_FRAC, EARLY_KS, QUERY_TOKEN_K))
    print("[config] horizon: ks=%s low=%s high=%s bootstrap=%d median_split=%s rho_floor=%.2f"
          % (HORIZON_KS, HORIZON_INNER_LOW, HORIZON_INNER_HIGH, HORIZON_BOOTSTRAP,
             HORIZON_MEDIAN_SPLIT, HORIZON_RHO_FLOOR))
    print("[config] verdict context: bootstrap=%d ladder_shuffle=%s methods=%s repeats=%d "
          "shuffle_bootstrap=%d"
          % (VERDICT_BOOTSTRAP, LADDER_SHUFFLE, list(LADDER_SHUFFLE_METHODS),
             LADDER_SHUFFLE_REPEATS, LADDER_SHUFFLE_BOOTSTRAP))
    print("[config] pool_measured[%s]=%s" % (CORPUS, POOL_MEASURED.get(CORPUS)))
    print("[config] results_path=%s" % RESULTS_PATH)
    print("OK -- config.py loads with no model, no network")
