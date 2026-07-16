"""data.py -- the train / eval prompts for the TALAN lesson, from the shared set.

Same shape of data as lesson 3: the adapter learns a refusal correction by
gradient descent, so it needs harmful prompts (to drive the refusal loss) and
benign prompts (to anchor the "don't over-refuse" KL leash), split into a TRAIN
half the trainer optimises on and a disjoint EVAL half held out for grading:

  1. ``train`` = {"harmful": [...], "benign": [...]}  # first N_PER_CLASS - N_EVAL
  2. ``eval``  = {"harmful": [...], "benign": [...]}  # disjoint last N_EVAL

The prompts come from the SHARED >=500/class set
(``steering_tutorials.common.data``: toxic-chat + JBB top-up, deduped and
length-matched). The shared loader shuffles each class deterministically with
``seed``, so we simply cut the first ``n_per_class - n_eval`` of each class into
TRAIN and the last ``n_eval`` into EVAL -- disjoint by construction.

This module is DELIBERATELY standalone and CPU-only: it never loads the model, so
the smoke test below runs in seconds with no GPU.
"""
from __future__ import annotations

import sys

from steering_tutorials.common.data import load_harmful_benign


def load_train_eval(n_per_class: int = 350, seed: int = 0, n_eval: int = 175) -> dict:
    """Return the TALAN data splits as a NESTED dict.

    Shape
    -----
        {
          "train": {"harmful": [str], "benign": [str]},   # first n_per_class-n_eval
          "eval":  {"harmful": [str], "benign": [str]},   # disjoint last n_eval
        }
    """
    d = load_harmful_benign(n_per_class=n_per_class, seed=seed)
    harmful, benign = d["harmful"], d["benign"]

    n_train = max(0, min(len(harmful), len(benign)) - n_eval)
    train = {"harmful": harmful[:n_train], "benign": benign[:n_train]}
    eval_ = {"harmful": harmful[n_train:n_train + n_eval],
             "benign": benign[n_train:n_train + n_eval]}

    print(
        f"[data] train: {len(train['harmful'])} harmful + {len(train['benign'])} benign | "
        f"eval: {len(eval_['harmful'])} harmful + {len(eval_['benign'])} benign "
        f"(n_per_class={n_per_class}, n_eval={n_eval}, seed={seed})",
        file=sys.stderr,
    )
    return {"train": train, "eval": eval_}


if __name__ == "__main__":  # smoke: python -m steering_tutorials.talan.data
    d = load_train_eval()
    for split in ("train", "eval"):
        print(f"{split:6s} harmful={len(d[split]['harmful'])}  benign={len(d[split]['benign'])}")
    print("\n[example harmful train prompts]")
    for p in d["train"]["harmful"][:3]:
        print(f"  - {p[:80]}")
    # Sanity: eval prompts must be disjoint from train prompts (no leakage).
    for cls in ("harmful", "benign"):
        overlap = set(d["train"][cls]) & set(d["eval"][cls])
        assert not overlap, f"{cls}: train/eval overlap of {len(overlap)} prompts!"
    print("\n[ok] train/eval splits are disjoint (no leakage).")
