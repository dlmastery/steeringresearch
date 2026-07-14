"""data.py — several harm CONCEPTS + a shared benign baseline, from JailbreakBench.

Lessons 2-3 learned ONE behaviour ("refuse this"). FLAS learns a SINGLE velocity
field that is *concept-conditioned*, so it must be trained on SEVERAL concepts at
once — otherwise there is nothing for the concept embedding ``c`` to select
between and the field degenerates to a fixed vector. This module supplies those
concepts.

We treat each JailbreakBench harm ``Category`` as one concept ("refuse THIS
category"). JBB ships 10 harmful prompts per category with matched benign prompts,
so each category is a ready-made, intent-matched concept. We use three categories
to TRAIN the flow and hold a fourth category OUT entirely — never shown during
training — so downstream inference can ask the zero-shot question the flow framing
is supposed to answer: *does one learned field steer a concept it never saw,
purely from that concept's embedding?*

Per concept we return three DISJOINT prompt sets:

    {
      "exemplars":     [...],  # encode -> the concept embedding c (identity of c)
      "steer_prompts": [...],  # harmful side of the diff-of-means TARGET shift
      "eval_prompts":  [...],  # held out; grade steering here, never trained on
    }

plus a shared benign ``baseline`` (the common contrast origin for every concept's
diff-of-means, and the source of "unsteered" activations the flow transports).
Keeping ``eval_prompts`` disjoint from ``steer_prompts`` is what stops us from
grading a steered activation on the very prompts that defined its target.

As in the sibling lessons we pull the raw CSVs with ``hf_hub_download`` (works
behind an SSL-intercepting proxy where ``datasets.load_dataset`` can fail) and
cache them in the HF cache, so after the first run this is instant and offline.
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

# The three JBB categories the flow is TRAINED on (each a concept the field must
# learn to steer toward when conditioned on that concept's embedding), plus ONE
# category held out for the zero-shot generalization test. These strings must
# match the JBB ``Category`` column exactly (the loader raises if one is missing).
TRAIN_CONCEPTS = [
    "Malware/Hacking",
    "Fraud/Deception",
    "Harassment/Discrimination",
]
HOLDOUT_CONCEPT = "Physical harm"

# Per-category split. JBB has 10 prompts per category, so we cap there: hold out
# the last N_EVAL for grading and use the rest to both build the target shift and
# encode the concept. exemplars == steer_prompts on purpose — the concept
# embedding is just an identity signal, so reusing the extract half keeps the
# eval half completely unseen by everything.
N_PER_CONCEPT = 10
N_EVAL_PER_CONCEPT = 3
N_BENIGN_BASELINE = 40


def _split_category(rows: list[str], n_eval: int) -> dict[str, list[str]]:
    """Split one shuffled category's prompts into the three disjoint roles."""
    n_eval = min(n_eval, max(0, len(rows) - 1))  # always keep >=1 for extraction
    extract = rows[: len(rows) - n_eval]
    evalset = rows[len(rows) - n_eval:]
    return {
        "exemplars": list(extract),      # encode -> concept embedding c
        "steer_prompts": list(extract),  # harmful side of the target diff-of-means
        "eval_prompts": list(evalset),   # held out for grading
    }


def load_concepts(
    seed: int = 0,
    n_per_concept: int = N_PER_CONCEPT,
    n_eval_per_concept: int = N_EVAL_PER_CONCEPT,
    n_benign_baseline: int = N_BENIGN_BASELINE,
) -> dict:
    """Return train concepts + a held-out concept + a shared benign baseline.

    Structure::

        {
          "concepts": {
              "Malware/Hacking": {"exemplars":[...], "steer_prompts":[...],
                                  "eval_prompts":[...], "heldout": False},
              ...,
              "Physical harm":   {..., "heldout": True},   # zero-shot, untrained
          },
          "baseline": [ ...benign prompts... ],   # shared contrast origin + h0 pool
          "train_concepts":   [ names with heldout == False ],
          "holdout_concept":  "Physical harm",
        }

    The trainer iterates the ``heldout == False`` concepts only; the held-out
    concept is carried through so inference can encode its embedding and steer a
    concept the field never trained on.
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
    concepts: dict[str, dict] = {}
    for name in TRAIN_CONCEPTS + [HOLDOUT_CONCEPT]:
        rows = hdf[hdf[CATEGORY_COLUMN] == name][PROMPT_COLUMN]
        rows = rows.dropna().astype(str).tolist()
        if not rows:
            raise ValueError(f"concept {name!r} not found among JBB categories: "
                             f"{sorted(hdf[CATEGORY_COLUMN].unique())}")
        # Fixed-seed shuffle so the split is reproducible but not "the first N
        # rows" (JBB is category-ordered, which would bias a tiny sample).
        rng.shuffle(rows)
        rows = rows[:n_per_concept]
        split = _split_category(rows, n_eval_per_concept)
        split["heldout"] = name == HOLDOUT_CONCEPT
        concepts[name] = split

    # Shared benign baseline — one set, the common origin for every concept's
    # diff-of-means AND the pool of unsteered activations the flow transports.
    benign = pd.read_csv(benign_path)[PROMPT_COLUMN].dropna().astype(str).tolist()
    rng.shuffle(benign)
    baseline = benign[:n_benign_baseline]

    for name, split in concepts.items():
        tag = "HOLDOUT" if split["heldout"] else "train  "
        print(f"[data] {tag} {name:28s} exemplars={len(split['exemplars'])} "
              f"steer={len(split['steer_prompts'])} eval={len(split['eval_prompts'])}",
              file=sys.stderr)
    print(f"[data] shared benign baseline = {len(baseline)}", file=sys.stderr)

    # Two extra views so BOTH consumers work off one loader:
    #   - train_flas.py reads concepts / train_concepts / holdout_concept (above);
    #   - run_flas.py's normalizer wants TRAIN-ONLY concepts under "train" and the
    #     held-out concept as a dict under "held_out" (name + exemplars + eval).
    train_only = {n: concepts[n] for n in TRAIN_CONCEPTS}
    held = concepts[HOLDOUT_CONCEPT]
    held_out = {
        "name": HOLDOUT_CONCEPT,
        "exemplars": held["exemplars"],
        "eval_prompts": held["eval_prompts"],
    }
    return {
        "concepts": concepts,
        "train": train_only,
        "held_out": held_out,
        "baseline": baseline,
        "train_concepts": list(TRAIN_CONCEPTS),
        "holdout_concept": HOLDOUT_CONCEPT,
    }


if __name__ == "__main__":  # smoke: python -m steering_tutorials.flas.data
    d = load_concepts()
    print(f"train_concepts = {d['train_concepts']}")
    print(f"holdout_concept = {d['holdout_concept']}")
    print(f"baseline = {len(d['baseline'])} benign prompts")
    for name, split in d["concepts"].items():
        role = "HELD OUT (zero-shot)" if split["heldout"] else "train"
        ex = split["exemplars"][0] if split["exemplars"] else "(none)"
        print(f"  [{role}] {name}: {len(split['exemplars'])} exemplars / "
              f"{len(split['eval_prompts'])} eval  e.g. {ex[:60]!r}")
