"""data_large.py — a LARGER, realistic safety training set from Toxic-Chat.

This is the drop-in scale-up of ``data.py`` (which serves only 100+100
JailbreakBench prompts). It exposes the same ``(prompts, labels)`` contract so
``train_probe.py`` can switch to it by changing a single import — but it draws
on REAL, in-the-wild user prompts with expert human safety annotation instead of
a curated toy set.

Dataset: **lmsys/toxic-chat** (Lin et al. 2023, arXiv:2310.17389 — EMNLP'23
Findings), the ``0124`` release. These are genuine user inputs collected from a
live Vicuna demo, each hand-annotated by humans for ``toxicity`` (0/1) and
``jailbreaking`` (0/1). We use the ``toxic-chat_annotation_all.csv`` file
(train+test, ~10.2k prompts). It downloads with ``hf_hub_download`` (works behind
the SSL middlebox where ``datasets.load_dataset`` fails) and caches in the HF
cache, so re-runs are instant and offline.

WHY THIS DATASET (and why not the obvious alternatives)
-------------------------------------------------------
The task's harmful-vs-benign probe needs PROMPT-LEVEL intent labels where both
classes come from the SAME distribution (so the probe can't cheat on source
style). Toxic-Chat is the best available fit:
  - The label is on the PROMPT (the user input), not a response — true intent.
  - Both classes are real user queries from one stream, so there is no
    source/topic confound (unlike gluing a red-team set onto Alpaca).
Rejected alternatives (all verified on the live Hub): ``allenai/wildjailbreak``
is the ideal matched harmful/benign set but is GATED (401 without an accepted
license); ``PKU-Alignment/BeaverTails`` labels (prompt, RESPONSE) pairs and every
prompt is a red-team prompt, so its "all-safe" prompts are ~60% harmful-but-
refused — there is no genuine benign class; ``in-the-wild-jailbreak-prompts``
"regular" split is full of NSFW/roleplay templates (not benign) with a 2x length
gap. See DATASETS.md for the full survey.

THE LABEL
---------
``toxicity == 1`` -> harmful/unsafe intent (label 1); ``toxicity == 0`` -> benign
(label 0). In this release every ``jailbreaking == 1`` prompt is also
``toxicity == 1`` (jailbreaking is a strict subset), so ``toxicity`` alone is the
clean single label. Toxic prompts are the minority (~7% of traffic, ~701 unique),
which caps a fully-balanced set at ~700/class — realistic safety data is
class-imbalanced, so this is the honest ceiling from one dataset.

Swapping datasets later is just changing which loader ``train_probe.py`` imports:
``from .data import load_safety_dataset`` -> ``from .data_large import
load_large_dataset``. Both return ``(list[str], list[int])`` with 1=harmful.
"""
from __future__ import annotations

import re
import sys

try:  # OS trust store for the SSL-intercepting middlebox (same as data.py).
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional
    pass

# ---------------------------------------------------------------------------
# Pinned source. The 0124 release is the latest human re-annotation; the "_all"
# file is train+test combined (we split downstream ourselves in train_probe).
# ---------------------------------------------------------------------------
TOXICCHAT_REPO = "lmsys/toxic-chat"
TOXICCHAT_FILE = "data/0124/toxic-chat_annotation_all.csv"

# Column-name variants tolerated across releases.
_PROMPT_FIELDS = ("user_input", "prompt", "text")
_LABEL_FIELDS = ("toxicity", "toxic", "label")

# Keep prompts bounded so a stray mega-prompt can't dominate the features, and
# drop empty/near-empty rows. Real prompts vary widely; 4000 chars keeps ~all.
MAX_PROMPT_CHARS = 4000
MIN_PROMPT_CHARS = 10


def _first_col(df, fields: tuple[str, ...]) -> str:
    """Return the first present column name among ``fields`` (else raise)."""
    for f in fields:
        if f in df.columns:
            return f
    raise KeyError(f"none of {fields} in columns {list(df.columns)}")


def _clean(text: str) -> str:
    """Normalize one prompt: collapse whitespace, drop replacement chars, trim."""
    s = str(text).replace("�", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _prep(raw: list[str]) -> list[str]:
    """Clean, length-filter, and deduplicate a list of prompts (order-stable)."""
    seen, out = set(), []
    for p in raw:
        c = _clean(p)
        if not (MIN_PROMPT_CHARS <= len(c) <= MAX_PROMPT_CHARS):
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def load_large_dataset(
    n_per_class: int = 750, seed: int = 0
) -> tuple[list[str], list[int]]:
    """Return a balanced, deduplicated ``(prompts, labels)`` from Toxic-Chat.

    Args:
        n_per_class: prompts to draw PER class. The toxic (positive) class has
            only ~701 unique prompts, so if ``n_per_class`` exceeds what's
            available BOTH classes are capped to the smaller count (to stay
            balanced) and a warning is printed. The benign class is abundant.
        seed: controls the shuffle used to sample within each class (reproducible).

    Returns:
        ``(prompts, labels)`` with label 1 = harmful/toxic, 0 = benign. The two
        classes are combined and shuffled together with ``seed`` so callers that
        do not stratify still see an interleaved order.
    """
    import random

    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TOXICCHAT_REPO, TOXICCHAT_FILE, repo_type="dataset")
    df = pd.read_csv(path)

    prompt_col = _first_col(df, _PROMPT_FIELDS)
    label_col = _first_col(df, _LABEL_FIELDS)
    # Labels are 0/1 ints; coerce defensively in case of "0"/"1" strings.
    labels_raw = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    harmful_raw = df[prompt_col][labels_raw == 1].astype(str).tolist()
    benign_raw = df[prompt_col][labels_raw == 0].astype(str).tolist()

    harmful = _prep(harmful_raw)
    benign = _prep(benign_raw)
    n_harmful_clean, n_benign_clean = len(harmful), len(benign)

    rng = random.Random(seed)
    rng.shuffle(harmful)
    rng.shuffle(benign)

    take = min(n_per_class, len(harmful), len(benign))
    if take < n_per_class:
        print(
            f"[data_large] NOTE: toxic class caps the balanced set at {take}/class "
            f"(harmful={n_harmful_clean}, benign={n_benign_clean}); requested "
            f"{n_per_class}. Real safety data is imbalanced -- using {take}/class.",
            file=sys.stderr,
        )
    harmful, benign = harmful[:take], benign[:take]

    prompts = harmful + benign
    labels = [1] * len(harmful) + [0] * len(benign)

    # Shuffle the two classes together so downstream order is not class-sorted.
    order = list(range(len(prompts)))
    rng.shuffle(order)
    prompts = [prompts[i] for i in order]
    labels = [labels[i] for i in order]

    print(
        f"[data_large] Toxic-Chat: {sum(labels)} harmful + {len(labels) - sum(labels)} "
        f"benign = {len(prompts)} prompts "
        f"(clean pool: {n_harmful_clean} harmful / {n_benign_clean} benign unique)",
        file=sys.stderr,
    )
    return prompts, labels


if __name__ == "__main__":  # smoke: python -m steering_tutorials.hello_world.data_large
    import json
    from pathlib import Path

    ps, ys = load_large_dataset()
    n_harm = sum(ys)
    print(f"\n[smoke] total={len(ps)}  harmful={n_harm}  benign={len(ys) - n_harm}")
    shown = {0: 0, 1: 0}
    for p, y in zip(ps, ys):
        if shown[y] < 2:
            print(f"  [{'HARMFUL' if y else 'BENIGN '}] {p[:90]}")
            shown[y] += 1
        if shown[0] >= 2 and shown[1] >= 2:
            break

    # Persist the exact default-settings set so feature extraction is reproducible.
    art = Path(__file__).parent / "artifacts" / "large_prompts.json"
    art.parent.mkdir(exist_ok=True)
    art.write_text(
        json.dumps(
            {
                "source": f"{TOXICCHAT_REPO}:{TOXICCHAT_FILE}",
                "label_scheme": {"1": "harmful (toxicity==1)", "0": "benign (toxicity==0)"},
                "n_per_class_requested": 750,
                "n_per_class_actual": n_harm,
                "seed": 0,
                "max_prompt_chars": MAX_PROMPT_CHARS,
                "prompts": ps,
                "labels": ys,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[smoke] wrote {art} ({len(ps)} prompts)")
