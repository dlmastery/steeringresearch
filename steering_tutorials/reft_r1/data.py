"""data.py — the train / eval prompts for ReFT-r1 (lesson 3), from the shared set.

Lesson 2 needed two *contrast* sets (harmful vs benign) to build ONE fixed
diff-of-means vector, splitting each class into extract/eval halves. Lesson 3
learns a rank-1 intervention by gradient descent instead of averaging, but it
consumes the SAME shape of data, so this module hands back both classes split
into train / eval, in one NESTED dict:

  1. ``train`` = {"harmful": [...], "benign": [...]} — the prompts the trainer
     optimises on. Harmful prompts drive the refusal cross-entropy (the pull);
     benign prompts anchor the KL "don't over-refuse" leash (the leash).
  2. ``eval``  = {"harmful": [...], "benign": [...]} — a disjoint held-out half
     of each class, never seen during training, for grading the learned edit.

Unlike lesson-3-HyperSteer there are NO ``concept_exemplars`` here: the rank-1
ReFT intervention reads the hidden state ``h`` directly through its learned
readout ``w`` (see reft.py), so it needs no separate frozen concept embedding to
condition on. Two splits are all the trainer and the eval/infer modules need.

The train/eval halves are nested one level down (read them as
``data["train"]["harmful"]`` / ``data["eval"]["benign"]``) so the trainer and the
separate eval/infer modules share exactly this structure.

The harmful/benign prompts come from the SHARED >=500/class set
(``steering_tutorials.common.data``: toxic-chat + JBB top-up, deduped and
length-matched), replacing this lesson's old 100-prompt JailbreakBench loader.
We pull ``n_per_class`` of each and cut a disjoint TRAIN (first
``n_per_class - n_eval``) / EVAL (last ``n_eval``) split, deterministically.

This module is DELIBERATELY standalone and CPU-only: it never loads the model,
so the smoke test below runs in seconds with no GPU.
"""
from __future__ import annotations

import sys

from steering_tutorials.common.data import load_harmful_benign


def load_train_eval(n_per_class: int = 250, seed: int = 0, n_eval: int = 50) -> dict:
    """Return the ReFT-r1 data splits as a NESTED dict.

    Shape
    -----
        {
          "train": {"harmful": [str], "benign": [str]},   # first n_per_class-n_eval
          "eval":  {"harmful": [str], "benign": [str]},   # disjoint last n_eval
        }

    The shared loader already shuffles each class deterministically with ``seed``,
    so we simply cut the first ``n_per_class - n_eval`` prompts of each class into
    TRAIN and the last ``n_eval`` into EVAL — disjoint by construction.
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


if __name__ == "__main__":  # smoke: python -m steering_tutorials.reft_r1.data
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
