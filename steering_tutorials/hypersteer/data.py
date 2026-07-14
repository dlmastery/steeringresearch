"""data.py — the train / eval prompts for HyperSteer (lesson 3), from JailbreakBench.

Lesson 2 needed two *contrast* sets (harmful vs benign) to build ONE fixed
diff-of-means vector, and split each class into extract/eval halves. Lesson 3
learns the steering vector with a hypernetwork instead, so the data has three
jobs, and this module hands back all three in one NESTED dict:

  1. ``train`` = {"harmful": [...], "benign": [...]} — the prompts the trainer
     optimises on (harmful drive the refusal-CE loss; benign anchor the KL
     "don't over-refuse" regulariser).
  2. ``eval``  = {"harmful": [...], "benign": [...]} — a disjoint held-out half
     of each class, never seen during training, for grading the learned vector.
  3. ``concept_exemplars`` — a small handful (~8) of harmful prompts, drawn from
     the TRAIN half, whose activations the hypernetwork reads to build the
     "refusal" concept embedding it conditions on. Drawing them from the train
     half keeps the eval prompts genuinely unseen.

The train/eval halves are nested one level down (read them as
``data["train"]["harmful"]`` / ``data["eval"]["benign"]``) so the trainer and the
separate eval/infer modules share exactly this structure.

JailbreakBench (Chao et al. 2024, arXiv:2404.01318) ships two matched CSVs:
  - data/harmful-behaviors.csv : 100 harmful requests
  - data/benign-behaviors.csv  : 100 benign  requests
both keyed on a ``Goal`` column and topically matched, so the difference between
the classes is *intent*, not vocabulary. We download the raw CSVs directly with
``hf_hub_download`` (works behind the SSL middlebox where ``datasets.load_dataset``
fails) and cache them, so after the first run this is instant and offline.

This module is DELIBERATELY standalone and CPU-only: it never loads the model,
so the smoke test below runs in seconds with no GPU.
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


def _load_class(csv_name: str, n: int, rng: random.Random) -> list[str]:
    """Download one JBB CSV, shuffle with ``rng``, return the first ``n`` prompts.

    Shuffling before truncating makes the subset reproducible (fixed seed) but not
    just "the first N rows" — the CSVs are ordered by category, so the head would
    over-sample a few topics.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(JBB_REPO, csv_name, repo_type="dataset")
    prompts = pd.read_csv(path)[PROMPT_COLUMN].dropna().astype(str).tolist()
    rng.shuffle(prompts)
    return prompts[:n]


def load_train_eval(n_per_class: int = 60, seed: int = 0, n_concept: int = 8) -> dict:
    """Return the HyperSteer data splits as a NESTED dict.

    Shape
    -----
        {
          "train": {"harmful": [str], "benign": [str]},   # first half of each class
          "eval":  {"harmful": [str], "benign": [str]},   # disjoint held-out half
          "concept_exemplars": [str],                      # n_concept harmful TRAIN
        }

    ``concept_exemplars`` are the first ``n_concept`` harmful TRAIN prompts, used to
    build the frozen "refusal" concept embedding the hypernetwork conditions on —
    a subset of ``train["harmful"]``, so the eval prompts stay unseen.

    Each class is shuffled with its OWN fixed-seed RNG (offset per class so the
    harmful and benign shuffles are independent), truncated to ``n_per_class``,
    then split down the middle into train/eval.
    """
    # Independent per-class RNGs so the two shuffles don't accidentally correlate.
    harmful = _load_class(HARMFUL_CSV, n_per_class, random.Random(seed))
    benign = _load_class(BENIGN_CSV, n_per_class, random.Random(seed + 1))

    half = n_per_class // 2
    train = {"harmful": harmful[:half], "benign": benign[:half]}
    eval_ = {"harmful": harmful[half:], "benign": benign[half:]}

    # Concept exemplars are a handful of TRAIN harmful prompts — never eval ones.
    concept_exemplars = train["harmful"][:n_concept]

    print(
        f"[data] train: {len(train['harmful'])} harmful + {len(train['benign'])} benign | "
        f"eval: {len(eval_['harmful'])} harmful + {len(eval_['benign'])} benign | "
        f"{len(concept_exemplars)} concept exemplars "
        f"(n_per_class={n_per_class}, seed={seed})",
        file=sys.stderr,
    )
    return {"train": train, "eval": eval_, "concept_exemplars": concept_exemplars}


if __name__ == "__main__":  # smoke: python -m steering_tutorials.hypersteer.data
    d = load_train_eval()
    for split in ("train", "eval"):
        print(f"{split:6s} harmful={len(d[split]['harmful'])}  benign={len(d[split]['benign'])}")
    print(f"concept_exemplars n={len(d['concept_exemplars'])}")
    print("\n[example concept exemplars]")
    for p in d["concept_exemplars"][:3]:
        print(f"  - {p[:80]}")
    # Sanity: eval prompts must be disjoint from train prompts (no leakage).
    for cls in ("harmful", "benign"):
        overlap = set(d["train"][cls]) & set(d["eval"][cls])
        assert not overlap, f"{cls}: train/eval overlap of {len(overlap)} prompts!"
    print("\n[ok] train/eval splits are disjoint (no leakage).")
