"""data.py — the harmful vs. benign prompts we steer against, from JailbreakBench.

SUPERSEDED (kept for the standalone ``__main__`` demo only). The actual lesson
run (``run_steering.py``) now imports ``load_harmful_benign`` from the course's
shared foundation ``steering_tutorials.common.data`` — the >=500/class,
length-matched (length-AUC 0.501) toxic-chat set — NOT this JailbreakBench
loader. This file remains so ``python -m ...hello_world_steering.data`` still runs
a quick JBB smoke, but nothing in the pipeline depends on it.

JailbreakBench (Chao et al. 2024, arXiv:2404.01318) ships two matched CSVs:
  - data/harmful-behaviors.csv : 100 harmful requests
  - data/benign-behaviors.csv  : 100 benign  requests
Both use a ``Goal`` column for the prompt text, and the two sets are topically
matched, so the difference between them is *intent*, not surface vocabulary.
That is exactly what makes their diff-of-means a clean "refuse this" direction.

Unlike lesson 1 (which returned one interleaved (prompts, labels) list), lesson
2 wants the two classes kept SEPARATE: the harmful set and the benign set are
contrasted to build the steering vector, and later split into extract/eval
halves. So we return a dict ``{"harmful": [...], "benign": [...]}``.

We download the raw CSVs directly with ``hf_hub_download`` (works behind the
SSL middlebox where ``datasets.load_dataset`` fails) and cache them in the HF
cache, so after the first run this is instant and offline.
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


def load_harmful_benign(n_per_class: int = 60, seed: int = 0) -> dict[str, list[str]]:
    """Return ``{"harmful": [str], "benign": [str]}`` with ``n_per_class`` each.

    We shuffle each class with a fixed seed before truncating so the subset is
    reproducible but not just "the first N rows of the CSV" (which are ordered
    by category and would bias a small sample toward a few topics).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    harmful_path = hf_hub_download(JBB_REPO, HARMFUL_CSV, repo_type="dataset")
    benign_path = hf_hub_download(JBB_REPO, BENIGN_CSV, repo_type="dataset")

    harmful = pd.read_csv(harmful_path)[PROMPT_COLUMN].dropna().astype(str).tolist()
    benign = pd.read_csv(benign_path)[PROMPT_COLUMN].dropna().astype(str).tolist()

    rng = random.Random(seed)
    rng.shuffle(harmful)
    rng.shuffle(benign)

    harmful = harmful[:n_per_class]
    benign = benign[:n_per_class]
    print(f"[data] {len(harmful)} harmful + {len(benign)} benign "
          f"(n_per_class={n_per_class}, seed={seed})", file=sys.stderr)
    return {"harmful": harmful, "benign": benign}


if __name__ == "__main__":  # smoke: python -m steering_tutorials.hello_world_steering.data
    d = load_harmful_benign()
    print(f"harmful={len(d['harmful'])}  benign={len(d['benign'])}")
    for p in d["harmful"][:2]:
        print(f"[HARMFUL] {p[:80]}")
    for p in d["benign"][:2]:
        print(f"[BENIGN ] {p[:80]}")
