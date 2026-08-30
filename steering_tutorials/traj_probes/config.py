"""config.py -- every knob for the traj_probes series (ATBench substrate).

This is the CONFIG ANCHOR for the long-trajectory activation-probe series
described in `types.py`. It loads no model, imports nothing heavy, and prints
ASCII only.

WHAT VARIES BETWEEN RUNS, AND THEREFORE WHAT IS IN THE FILENAMES
----------------------------------------------------------------
Three things can differ between two runs of this series and change the numbers:

  * CORPUS   -- "atbench" (1,000 trajectories) vs "atbench500" (the older,
                DISJOINT 500-trajectory AgentDoG release; verified disjoint --
                zero first-user-message overlap with the 1,000-row config).
  * MODEL    -- the residual stream is the measurement instrument here.
  * LAYER    -- a layer-12 probe and a layer-20 probe are different results.

All three are in every artifact filename via `common/artifact_paths.keyed_path`,
which REFUSES a bare `results.json`. That helper exists because `meerkat`'s bge
arm destroyed its minilm results, `rogue_scalpel` came within a pre-flight check
of destroying its project_out ladder, and `cross_trajectory` fixed the same bug
locally without the fix propagating. `preflight()` additionally calls
`assert_no_bare_sibling`, which catches the fossil of a run made BEFORE the path
was keyed -- a file indistinguishable from a current one.

MEASURED FACTS ABOUT THE SUBSTRATE THAT SET THE DEFAULTS
--------------------------------------------------------
Measured directly on ATBench/test.json 2026-08-28 (all 1,000 rows):

  label balance        503 safe / 497 unsafe   -> N_PER_CLASS=500 is 3 short on
                                                  the unsafe side; see data.py.
  turns per trajectory min 2, median 8, max 62
  turns by class       safe   mean 8.24, max 18
                       unsafe mean 9.78, max 62
  chars by class       safe   median 3428, mean  3751
                       unsafe median 3826, mean 16949

Read the last three lines before trusting any AUC this series produces. The
unsafe class owns the ENTIRE long tail: every trajectory longer than 18 turns is
unsafe. That is the step-index confound of `types.py` CONTROL 1 sitting in the
substrate in plain sight, and it is why `auc_residualised` is not optional
decoration on `ProbeResult` -- a probe that reads length alone has a real edge
here and would look like a finding.

MAX_TURNS is NOT merely a cost knob, and it does NOT default to 0. Measured
2026-08-29: the longest SAFE trajectory is 18 turns and the longest UNSAFE one is
62, so every turn at step index >= 18 is unsafe BY CONSTRUCTION -- 570 of 8,981
rows (6.3%) whose label is decided before an activation is read. Step index looks
harmless globally (AUC 0.5686) because the leak is LOCAL and perfect, and
`StepResidualiser` removes LINEAR position only, so CONTROL 1 cannot reach a
threshold like this. The cap is therefore a CORRECTNESS setting: at 16 the
highest step index is 15, the region is empty, and 92.9% of rows survive. See
leakage.py, which gates on it. Raising it above 18 re-opens the leak.
"""
from __future__ import annotations

import os
from pathlib import Path

from steering_tutorials.common.artifact_paths import assert_no_bare_sibling, keyed_path

__all__ = [
    "CORPUS", "CORPUS_CHOICES", "MODEL_ID", "MODEL_TAG", "LAYER",
    "N_PER_CLASS", "SEED", "N_FOLDS", "BOOTSTRAP", "MAX_TURNS",
    "MAX_TURN_CHARS", "MAX_TOKENS", "GROUP_BY",
    "ARTIFACTS", "RESULTS_PATH", "CORPUS_CACHE_PATH", "ACTIVATION_CACHE_PATH",
    "HF_DATASET", "HF_CONFIG", "HF_SPLIT", "DATASET_LICENCE", "DATASET_PAPER",
    "CACHE_REFRESH", "ensure_artifacts", "preflight", "as_dict",
]


# --- env helpers (TP_ prefix; an EMPTY env var means "unset", not "") --------
def _env_str(name, default):
    return os.environ.get(name) or default


def _env_int(name, default):
    return int(os.environ.get(name) or default)


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- corpus ------------------------------------------------------------------
HF_DATASET = "AI45Research/ATBench"
# The two configs are DIFFERENT RELEASES with different schemas, not two views of
# one pool: "ATBench" rows carry `contents` + `reason` + `id`; "ATBench500" rows
# carry `content` + `conv_id` and NO `reason`. data.py handles both.
CORPUS_CHOICES = ("atbench", "atbench500")
CORPUS = _env_str("TP_CORPUS", "atbench").lower()
if CORPUS not in CORPUS_CHOICES:
    raise ValueError("TP_CORPUS must be one of %r, got %r"
                     % (list(CORPUS_CHOICES), CORPUS))
HF_CONFIG = {"atbench": "ATBench", "atbench500": "ATBench500"}[CORPUS]
HF_SPLIT = "test"

# Stated on the dataset card front-matter (`license: apache-2.0`), read from the
# downloaded README.md, not from memory.
DATASET_LICENCE = "apache-2.0 (dataset card front-matter, AI45Research/ATBench)"
DATASET_PAPER = {
    "atbench": "arXiv:2604.02022 -- ATBench: A Diverse and Realistic Agent "
               "Trajectory Benchmark for Safety Evaluation and Diagnosis",
    "atbench500": "arXiv:2601.18491 -- AgentDoG (the original 500-trajectory "
                  "release; ATBench500 config)",
}[CORPUS]

# --- model / probe -----------------------------------------------------------
# The LOCAL path: the bare HF id 401s on this host (CLAUDE.md 18.5). This is the
# ALIGNED base, not the abliterated build -- we are reading the residual stream of
# a model behaving normally, not steering it.
MODEL_ID = _env_str("TP_MODEL_ID", "models/google/gemma-3-1b-it")


def _model_tag(model_id):
    """A short filename-safe tag. Two different models must not collide here."""
    tag = model_id.replace("\\", "/").rstrip("/").split("/")[-1]
    keep = [c if (c.isalnum() or c in "-.") else "-" for c in tag]
    return "".join(keep).strip("-") or "model"


MODEL_TAG = _env_str("TP_MODEL_TAG", _model_tag(MODEL_ID))
LAYER = _env_int("TP_LAYER", 12)

# --- sampling / evaluation ---------------------------------------------------
N_PER_CLASS = _env_int("TP_N_PER_CLASS", 500)
SEED = _env_int("TP_SEED", 0)
N_FOLDS = _env_int("TP_N_FOLDS", 5)
BOOTSTRAP = _env_int("TP_BOOTSTRAP", 10000)

# Truncates a trajectory to its FIRST MAX_TURNS turns, which changes the corpus,
# so it is part of the cache key. 0 == keep every turn and RE-OPEN the step-index
# leak documented above (leakage.py refuses that corpus unless acknowledged).
MAX_TURNS = _env_int("TP_MAX_TURNS", 16)

# Caps EACH TURN's rendered text. This is a CORRECTNESS setting, like MAX_TURNS.
# ATBench holds a handful of enormous `tool` dumps -- per-turn chars are p90
# 1,088 but p99 50,915, the largest 53,255 -- and they are almost all label=1.
# With them present a 4,096-token extraction budget silently DROPS the overflow
# rows, and because the giant turns are unsafe the loss is label-correlated: 327
# unsafe rows dropped against 13 safe, from trajectories that are 86% unsafe
# against a 49.8% base rate. Nothing crashes; the bundle just quietly loses the
# late turns of the failing class. At 1,200 chars every trajectory fits in 4,096
# tokens (max 4,025), so NO row is dropped and the loss cannot correlate with
# anything. Applied in the LOADER so the model and the content bar read the same
# truncated string -- truncating only the model input would make the confound
# comparison meaningless.
MAX_TURN_CHARS = _env_int("TP_MAX_TURN_CHARS", 8000)

# The extraction token budget. Was a hard-coded ExtractSettings default of 4096,
# so it was never stamped, never swept, and silently forced the turn cap down to
# 1200 to fit -- which truncated ~10% of ALL turns rather than the pathological
# tool dumps it was aimed at. Measured at MAX_TURN_CHARS=8000: p99 12,304 tokens,
# max 14,634, so 16,384 overflows NOTHING and 8,192 would still overflow 23
# trajectories. The model's context is 32,768, so this is well inside it.
MAX_TOKENS = _env_int("TP_MAX_TOKENS", 16384)

# Group-aware CV unit. "trajectory" is the only defensible default on ATBench --
# see data.py's GROUPING section for the measurement that rules the alternatives
# out. Rows in an ActivationBundle are TURNS, so grouping by trajectory is doing
# real work: it stops turn 3 of an episode training a probe that is then scored
# on turn 7 of the same episode.
GROUP_BY = _env_str("TP_GROUP_BY", "trajectory")
if GROUP_BY != "trajectory":
    raise ValueError(
        "TP_GROUP_BY=%r is not implemented. ATBench has NO field that groups "
        "rows: 1000/1000 unique ids, 1000/1000 unique first-user messages, "
        "998/1000 unique tool suites (the one 3-row tool-suite family is three "
        "unrelated scenarios that merely reuse two tools). Adding a coarser "
        "grouping here would merge unrelated episodes and understate n."
        % GROUP_BY)

# --- artifact paths (NEVER a bare results.json) ------------------------------
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

_MODEL_VARIANTS = (CORPUS, MODEL_TAG, "L%d" % LAYER)
_POOL_VARIANTS = (CORPUS, "n%d" % N_PER_CLASS, "s%d" % SEED, "t%d" % MAX_TURNS,
                  "c%d" % MAX_TURN_CHARS)

RESULTS_PATH = keyed_path(ARTIFACTS, "results", ".json", *_MODEL_VARIANTS)
# gzipped: the parsed 1,000-trajectory corpus is ~18 MB of text as raw JSON.
CORPUS_CACHE_PATH = keyed_path(ARTIFACTS, "corpus", ".json.gz", *_POOL_VARIANTS)
ACTIVATION_CACHE_PATH = keyed_path(
    ARTIFACTS, "acts", ".npz",
    *(_MODEL_VARIANTS + ("n%d" % N_PER_CLASS, "s%d" % SEED)))

# Set TP_CACHE_REFRESH=1 to REWRITE a disagreeing corpus cache instead of
# refusing it. Off by default: a silent rewrite is how a stale artifact becomes
# indistinguishable from a fresh one.
CACHE_REFRESH = _env_bool("TP_CACHE_REFRESH", False)


def ensure_artifacts():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS


def preflight():
    """Fail LOUDLY before a run, not after it has overwritten something.

    Checks each artifact stem for a bare, unattributable sibling. Call this at
    the top of every runner in this series.
    """
    ensure_artifacts()
    for stem, ext in (("results", ".json"),
                      ("corpus", ".json.gz"),
                      ("acts", ".npz")):
        assert_no_bare_sibling(ARTIFACTS, stem, ext)


def as_dict():
    """The config as it will be stamped into every results file."""
    return {
        "corpus": CORPUS, "hf_dataset": HF_DATASET, "hf_config": HF_CONFIG,
        "hf_split": HF_SPLIT, "dataset_licence": DATASET_LICENCE,
        "dataset_paper": DATASET_PAPER,
        "model_id": MODEL_ID, "model_tag": MODEL_TAG, "layer": LAYER,
        "n_per_class": N_PER_CLASS, "seed": SEED, "n_folds": N_FOLDS,
        "bootstrap": BOOTSTRAP, "max_turns": MAX_TURNS,
        "max_turn_chars": MAX_TURN_CHARS, "max_tokens": MAX_TOKENS,
        "group_by": GROUP_BY,
        "results_path": str(RESULTS_PATH),
        "corpus_cache_path": str(CORPUS_CACHE_PATH),
        "activation_cache_path": str(ACTIVATION_CACHE_PATH),
    }


def _self_test():
    print("OK  corpus=%s config=%s split=%s" % (CORPUS, HF_CONFIG, HF_SPLIT))
    print("OK  model=%s tag=%s layer=%d" % (MODEL_ID, MODEL_TAG, LAYER))
    print("OK  n_per_class=%d seed=%d folds=%d bootstrap=%d max_turns=%d"
          % (N_PER_CLASS, SEED, N_FOLDS, BOOTSTRAP, MAX_TURNS))

    for p in (RESULTS_PATH, CORPUS_CACHE_PATH, ACTIVATION_CACHE_PATH):
        name = Path(p).name
        assert name not in ("results.json", "corpus.json.gz", "acts.npz"), name
        assert CORPUS in name, name
        print("OK  keyed path: %s" % name)
    assert MODEL_TAG in RESULTS_PATH.name and "L%d" % LAYER in RESULTS_PATH.name

    assert _model_tag("models/google/gemma-3-1b-it") == "gemma-3-1b-it"
    assert _model_tag("C:\\x\\gemma-2-2b-it\\") == "gemma-2-2b-it"
    assert _model_tag("google/gemma-3-1b-it") != _model_tag("google/gemma-2-2b-it")
    print("OK  _model_tag is filename-safe and does not collide across models")

    # An unset env var must not collapse a variant to the empty string.
    assert _env_str("TP__NOPE", "d") == "d" and _env_int("TP__NOPE", 7) == 7
    os.environ["TP__EMPTY"] = ""
    assert _env_str("TP__EMPTY", "fallback") == "fallback"
    assert _env_bool("TP__EMPTY", False) is False
    del os.environ["TP__EMPTY"]
    print("OK  an EMPTY env var falls back instead of producing a bare path")

    preflight()
    print("OK  preflight() passed: no bare sibling beside any keyed artifact")

    d = as_dict()
    assert d["dataset_licence"].startswith("apache-2.0")
    print("OK  as_dict() stamps %d fields incl. licence and paper" % len(d))
    print("")
    print("OK -- config.py self-test passed CPU-only, no model, no GPU, no network.")


if __name__ == "__main__":
    _self_test()
