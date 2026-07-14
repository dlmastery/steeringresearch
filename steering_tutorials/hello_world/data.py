"""data.py — the safety training set: JailbreakBench harmful vs. benign prompts.

JailbreakBench (Chao et al. 2024, arXiv:2404.01318) ships two matched CSVs:
  - data/harmful-behaviors.csv : 100 harmful requests   -> label 1 (harmful)
  - data/benign-behaviors.csv  : 100 benign  requests   -> label 0 (safe)
Both use a ``Goal`` column for the prompt text. The two sets are topically
matched, so a probe can't cheat on surface words — it has to read intent.

We download the raw CSVs directly with ``hf_hub_download`` (works behind the
SSL middlebox where ``datasets.load_dataset`` fails) and cache them in the HF
cache, so after the first run this is instant and offline.
"""
from __future__ import annotations

import sys

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass

JBB_REPO = "JailbreakBench/JBB-Behaviors"
HARMFUL_CSV = "data/harmful-behaviors.csv"
BENIGN_CSV = "data/benign-behaviors.csv"
PROMPT_COLUMN = "Goal"


def load_safety_dataset() -> tuple[list[str], list[int]]:
    """Return (prompts, labels) with label 1 = harmful, 0 = safe.

    Harmful and benign are interleaved-balanced by construction (100 each).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    harmful_path = hf_hub_download(JBB_REPO, HARMFUL_CSV, repo_type="dataset")
    benign_path = hf_hub_download(JBB_REPO, BENIGN_CSV, repo_type="dataset")

    harmful = pd.read_csv(harmful_path)[PROMPT_COLUMN].dropna().astype(str).tolist()
    benign = pd.read_csv(benign_path)[PROMPT_COLUMN].dropna().astype(str).tolist()

    prompts = harmful + benign
    labels = [1] * len(harmful) + [0] * len(benign)
    print(f"[data] {len(harmful)} harmful + {len(benign)} benign = "
          f"{len(prompts)} prompts", file=sys.stderr)
    return prompts, labels


if __name__ == "__main__":  # quick smoke: python -m steering_tutorials.hello_world.data
    ps, ys = load_safety_dataset()
    for p, y in list(zip(ps, ys))[:2] + list(zip(ps, ys))[-2:]:
        print(f"[{ 'HARMFUL' if y else 'SAFE   ' }] {p[:80]}")
