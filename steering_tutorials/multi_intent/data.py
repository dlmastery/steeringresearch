"""data.py — K distinct harm concepts + a shared benign baseline, from JBB.

Lesson 2 needed ONE harmful set and ONE benign set. Lesson 9 needs K *separate*
harmful concepts (so we can steer each independently and watch them interfere)
plus ONE shared benign baseline (the common origin every concept is contrasted
against). JailbreakBench is perfect for this: its harmful CSV carries a
``Category`` column with 10 categories × 10 prompts, so each category is a
ready-made concept with matched intent.

We return a nested structure:

    {
      "concepts": {
          "Malware/Hacking":   {"extract": [...], "eval": [...]},
          "Fraud/Deception":   {"extract": [...], "eval": [...]},
          ...
      },
      "baseline": [ ...benign prompts... ],   # the shared contrast origin
    }

``extract`` prompts build each concept's diff-of-means vector; ``eval`` prompts
(disjoint) measure steering success and cross-talk. Keeping them disjoint stops
us from grading a vector on the very prompts that defined it.

As in lesson 2 we pull the raw CSVs with ``hf_hub_download`` (works behind an
SSL-intercepting proxy where ``datasets.load_dataset`` can fail) and cache them.
"""
from __future__ import annotations

import random
import sys

try:  # Some networks sit behind an SSL-intercepting proxy; use the OS trust store.
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - optional
    pass

JBB_REPO = "JailbreakBench/JBB-Behaviors"
HARMFUL_CSV = "data/harmful-behaviors.csv"
BENIGN_CSV = "data/benign-behaviors.csv"
PROMPT_COLUMN = "Goal"
CATEGORY_COLUMN = "Category"


def load_multi_intent(
    concepts: list[str],
    n_per_concept: int = 10,
    n_eval_per_concept: int = 5,
    n_benign_baseline: int = 40,
    seed: int = 0,
) -> dict:
    """Return K concept prompt-sets (extract/eval split) + a shared benign baseline.

    Parameters
    ----------
    concepts : the JBB ``Category`` names to use as concepts (see
        ``config.CONCEPTS``). Each must exist in the harmful CSV.
    n_per_concept : total exemplars to draw for a concept BEFORE the eval split
        (JBB has 10 per category, so this is capped at what's available).
    n_eval_per_concept : how many of those are held out for evaluation; the rest
        go to extraction. ``extract = n_per_concept - n_eval_per_concept``.
    n_benign_baseline : size of the shared benign baseline (from the benign CSV).

    The per-concept shuffle uses a fixed seed so the split is reproducible but is
    not "just the first N rows" (which are ordered and would bias a tiny sample).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    harmful_path = hf_hub_download(JBB_REPO, HARMFUL_CSV, repo_type="dataset")
    benign_path = hf_hub_download(JBB_REPO, BENIGN_CSV, repo_type="dataset")

    hdf = pd.read_csv(harmful_path)
    if CATEGORY_COLUMN not in hdf.columns:
        raise ValueError(f"harmful CSV has no {CATEGORY_COLUMN!r} column; "
                         f"cannot build per-concept sets")

    rng = random.Random(seed)
    out_concepts: dict[str, dict[str, list[str]]] = {}
    for name in concepts:
        rows = hdf[hdf[CATEGORY_COLUMN] == name][PROMPT_COLUMN]
        rows = rows.dropna().astype(str).tolist()
        if not rows:
            raise ValueError(f"concept {name!r} not found among JBB categories: "
                             f"{sorted(hdf[CATEGORY_COLUMN].unique())}")
        rng.shuffle(rows)
        rows = rows[:n_per_concept]
        # Hold out the LAST n_eval for evaluation; the head builds the vector.
        n_eval = min(n_eval_per_concept, max(0, len(rows) - 1))
        extract = rows[: len(rows) - n_eval]
        evalset = rows[len(rows) - n_eval:]
        out_concepts[name] = {"extract": extract, "eval": evalset}

    # Shared benign baseline — one set, reused as the origin for every concept.
    benign = pd.read_csv(benign_path)[PROMPT_COLUMN].dropna().astype(str).tolist()
    rng.shuffle(benign)
    baseline = benign[:n_benign_baseline]

    for name, split in out_concepts.items():
        print(f"[data] {name:28s} extract={len(split['extract'])} "
              f"eval={len(split['eval'])}", file=sys.stderr)
    print(f"[data] shared benign baseline = {len(baseline)}", file=sys.stderr)
    return {"concepts": out_concepts, "baseline": baseline}


if __name__ == "__main__":  # smoke: python -m steering_tutorials.multi_intent.data
    from . import config as C

    d = load_multi_intent(
        C.CONCEPTS,
        n_per_concept=C.N_PER_CONCEPT,
        n_eval_per_concept=C.N_EVAL_PER_CONCEPT,
        n_benign_baseline=C.N_BENIGN_BASELINE,
        seed=C.SEED,
    )
    print(f"concepts={list(d['concepts'])}  baseline={len(d['baseline'])}")
    for name, split in d["concepts"].items():
        ex = split["extract"][0] if split["extract"] else "(none)"
        print(f"[{name}] extract[0]={ex[:70]}")
