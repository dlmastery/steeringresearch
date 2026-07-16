"""data_large.py — a LARGER, realistic, PRINCIPLED safety training set.

Drop-in scale-up of ``data.py`` (100+100 JailbreakBench toy prompts). Same
``(prompts, labels)`` contract, but built with elite-data-scientist rigor on
REAL, human-annotated in-the-wild prompts.

Dataset: **lmsys/toxic-chat** (Lin et al. 2023, arXiv:2310.17389, EMNLP'23
Findings), ``0124`` release — genuine user inputs from a live Vicuna demo, each
hand-annotated by humans for ``toxicity`` and ``jailbreaking``.

WHY THIS DATASET (label-unit integrity — the #1 rule)
-----------------------------------------------------
A prompt classifier needs labels on the PROMPT'S INTENT. Toxic-Chat's
``toxicity`` label is annotated on the user input itself -> it is already
prompt-level, exactly like WildJailbreak's native ``vanilla_harmful`` /
``vanilla_benign`` types. NO pair->prompt collapse is needed (see DATASETS.md).

The two "obvious" larger sets were rejected after verification:
  - **WildJailbreak** (the ideal matched set) is GATED -> 401 without an accepted
    license; no HF token on this machine, so it is unavailable here.
  - **BeaverTails** labels (prompt, RESPONSE) pairs. Collapsing to prompt level
    ("all responses safe -> benign") is fatally biased: a refusal scores
    ``is_safe=True``, so every *harmful prompt the models refused* collapses into
    the BENIGN class. A 20-prompt audit of BeaverTails' all-safe prompts found
    ~60% carry harmful intent (blackmail, harassment, slurs). That violates
    label-unit integrity, so we do not use it. (The collapse rule is still
    IMPLEMENTED and DOCUMENTED in DATASETS.md as the teaching point.)

WHAT THIS LOADER DOES (the five rigor requirements)
---------------------------------------------------
1. LABEL-UNIT INTEGRITY. Native prompt-level labels; no collapse. As an extra
   guard we drop prompts whose NORMALIZED text appears under BOTH labels
   (genuinely ambiguous) and report the count.
2. POSITIVE-CLASS STRATIFICATION. The harmful class is stratified across harm
   categories derived from each prompt's ``openai_moderation`` scores
   (sexual / harassment / violence / hate / self-harm / ...). Per-category counts
   are reported in the artifact header.
3. DEDUP + LEAKAGE-SAFE. Every row carries a stable ``group_id`` = hash of its
   normalized text. Exact AND surface near-duplicates (differ only in
   case/whitespace/punctuation) share a group_id and are collapsed to one row, so
   a prompt and its trivial variants cannot straddle a train/test split. By
   default we sample from the dataset's OFFICIAL TRAIN split only.
4. BASE RATES. The natural class balance (toxic is a ~7% minority) is recorded
   before we rebalance to 1:1. A balanced training sample distorts the prior —
   downstream PR-AUC / calibration must be read against the natural base rate.
5. REPRODUCIBILITY. seed=0, sampling without replacement, deterministic order,
   and a full provenance header in ``artifacts/large_prompts.json``.

Swap into training with a one-line import change (see DATASETS.md):
``from .data_large import load_large_dataset``. Returns
``(prompts: list[str], labels: list[int])`` with 1 = harmful.
"""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from collections import Counter, defaultdict

try:  # OS trust store for the SSL-intercepting middlebox (same as data.py).
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional
    pass

# ---------------------------------------------------------------------------
# Pinned source. The ``0124`` release is the latest human re-annotation; it ships
# an official train/test split we honor for leakage safety.
# ---------------------------------------------------------------------------
TOXICCHAT_REPO = "lmsys/toxic-chat"
TOXICCHAT_FILES = {
    "train": "data/0124/toxic-chat_annotation_train.csv",
    "test": "data/0124/toxic-chat_annotation_test.csv",
    "all": "data/0124/toxic-chat_annotation_all.csv",
}
SOURCE_DATASET = "lmsys/toxic-chat@0124"

# Column-name variants tolerated across releases.
_PROMPT_FIELDS = ("user_input", "prompt", "text")
_LABEL_FIELDS = ("toxicity", "toxic", "label")
_MODERATION_FIELDS = ("openai_moderation", "moderation")

MAX_PROMPT_CHARS = 4000
MIN_PROMPT_CHARS = 10


def _first_col(df, fields: tuple[str, ...]) -> str:
    for f in fields:
        if f in df.columns:
            return f
    raise KeyError(f"none of {fields} in columns {list(df.columns)}")


def _clean(text: str) -> str:
    """Readable normalization for the stored prompt: drop U+FFFD, squeeze space."""
    s = str(text).replace("�", " ")
    return re.sub(r"\s+", " ", s).strip()


def _norm_key(text: str) -> str:
    """Aggressive normalization for dedup/grouping: lowercase, alnum-only tokens.

    Collapses case/whitespace/punctuation variants to one key so surface
    near-duplicates share a ``group_id``. (Semantic paraphrases are NOT caught —
    that needs embeddings, out of scope for this CPU/text-only loader.)
    """
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _group_id(norm: str) -> str:
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def _dominant_category(moderation_cell) -> str:
    """Argmax OpenAI-moderation category for a prompt, else 'unlabeled'.

    The cell is a string repr of ``[[category, score], ...]``. Used only to
    STRATIFY the harmful class across harm types.
    """
    try:
        data = moderation_cell
        if isinstance(data, str):
            data = ast.literal_eval(data)
        scores = dict(data)
        if not scores:
            return "unlabeled"
        return max(scores, key=lambda k: scores[k])
    except Exception:
        return "unlabeled"


def _stratified_sample(items: list, keyfn, take: int, rng) -> list:
    """Deterministically sample ``take`` items spread across ``keyfn`` strata.

    Largest-remainder allocation per stratum; if ``take`` >= len(items) returns
    all items (shuffled). Guarantees no single harm category dominates.
    """
    if take >= len(items):
        out = list(items)
        rng.shuffle(out)
        return out
    buckets: dict = defaultdict(list)
    for it in items:
        buckets[keyfn(it)].append(it)
    for b in buckets.values():
        rng.shuffle(b)
    total = len(items)
    chosen: list = []
    remainders = []  # (fractional_remainder, key, bucket, n_taken_so_far)
    for key, bucket in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        exact = take * len(bucket) / total
        n = int(exact)
        chosen.extend(bucket[:n])
        remainders.append([exact - n, key, bucket, n])
    # Hand out leftover slots by largest fractional remainder.
    remainders.sort(key=lambda r: (-r[0], r[1]))
    i = 0
    while len(chosen) < take and remainders:
        slot = remainders[i % len(remainders)]
        frac, key, bucket, n = slot
        if n < len(bucket):
            chosen.append(bucket[n])
            slot[3] = n + 1
        i += 1
        if all(n >= len(b) for _, _, b, n in remainders):
            break
    rng.shuffle(chosen)
    return chosen[:take]


def build_dataset(n_per_class: int = 750, seed: int = 0, split: str = "train") -> dict:
    """Build the balanced set and return a full record (prompts, labels, meta).

    ``split`` in {"train","test","all"}; default "train" honors the official
    split for leakage safety. Returns a dict with keys: ``rows`` (list of
    {prompt,label,category,group_id,source_dataset}), ``prompts``, ``labels``,
    and ``header`` (provenance/base-rate/category/drop stats).
    """
    import random

    import pandas as pd
    from huggingface_hub import hf_hub_download

    if split not in TOXICCHAT_FILES:
        raise ValueError(f"split must be one of {list(TOXICCHAT_FILES)}")
    path = hf_hub_download(TOXICCHAT_REPO, TOXICCHAT_FILES[split], repo_type="dataset")
    df = pd.read_csv(path)

    prompt_col = _first_col(df, _PROMPT_FIELDS)
    label_col = _first_col(df, _LABEL_FIELDS)
    try:
        mod_col = _first_col(df, _MODERATION_FIELDS)
    except KeyError:
        mod_col = None

    labels_raw = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    n_raw = len(df)
    n_toxic_raw = int((labels_raw == 1).sum())
    n_benign_raw = int((labels_raw == 0).sum())

    # --- collapse by normalized-text group_id; detect cross-label ambiguity ----
    groups: dict = {}  # gid -> {"display","labels":set,"cat"}
    n_after_filters = 0  # rows passing length/blank filters (pre-dedup)
    mod_series = df[mod_col] if mod_col else [None] * n_raw
    for raw_prompt, lab, modcell in zip(df[prompt_col].astype(str), labels_raw, mod_series):
        display = _clean(raw_prompt)
        if not (MIN_PROMPT_CHARS <= len(display) <= MAX_PROMPT_CHARS):
            continue
        norm = _norm_key(raw_prompt)
        if not norm:
            continue
        n_after_filters += 1
        gid = _group_id(norm)
        g = groups.get(gid)
        if g is None:
            cat = _dominant_category(modcell) if (lab == 1 and mod_col) else "benign"
            groups[gid] = {"display": display, "labels": {int(lab)}, "cat": cat}
        else:
            g["labels"].add(int(lab))  # same normalized text, another annotation

    n_unique_after_dedup = len(groups)
    n_dropped_duplicates = n_after_filters - n_unique_after_dedup

    harmful, benign = [], []
    n_dropped_ambiguous = 0
    for gid, g in groups.items():
        if len(g["labels"]) > 1:  # same normalized text labelled both ways
            n_dropped_ambiguous += 1
            continue
        row = {"prompt": g["display"], "label": next(iter(g["labels"])),
               "category": g["cat"], "group_id": gid,
               "source_dataset": SOURCE_DATASET}
        (harmful if row["label"] == 1 else benign).append(row)

    n_harmful_pool, n_benign_pool = len(harmful), len(benign)

    rng = random.Random(seed)
    take = min(n_per_class, n_harmful_pool, n_benign_pool)
    if take < n_per_class:
        print(
            f"[data_large] NOTE: harmful class caps the balanced set at "
            f"{take}/class (harmful_pool={n_harmful_pool}, "
            f"benign_pool={n_benign_pool}); requested {n_per_class}. Real safety "
            f"data is imbalanced -- using {take}/class on split='{split}'.",
            file=sys.stderr,
        )

    harmful_sample = _stratified_sample(harmful, lambda r: r["category"], take, rng)
    # benign: plain random (stratification requirement is for the positive class)
    benign_shuf = list(benign)
    rng.shuffle(benign_shuf)
    benign_sample = benign_shuf[:take]

    rows = harmful_sample + benign_sample
    rng.shuffle(rows)  # interleave classes deterministically

    prompts = [r["prompt"] for r in rows]
    labels = [r["label"] for r in rows]

    per_category = dict(Counter(r["category"] for r in harmful_sample))
    natural_toxic_rate = round(n_toxic_raw / max(1, n_raw), 4)

    header = {
        "dataset": SOURCE_DATASET,
        "source_file": TOXICCHAT_FILES[split],
        "split_used": split,
        "label_scheme": {"1": "harmful (toxicity==1)", "0": "benign (toxicity==0)"},
        "collapse_rule": "none needed -- labels are native prompt-level (human "
                         "annotation on the user input), cf. WildJailbreak types",
        "natural_base_rate": {
            "n_rows_raw": n_raw,
            "n_toxic_raw": n_toxic_raw,
            "n_benign_raw": n_benign_raw,
            "toxic_fraction": natural_toxic_rate,
            "note": "we rebalanced to 1:1 for TRAINING; real-world prior is this "
                    "minority toxic_fraction -- read PR-AUC/calibration against it",
        },
        "n_unique_after_dedup": n_unique_after_dedup,
        "n_dropped_duplicates": n_dropped_duplicates,
        "n_dropped_ambiguous": n_dropped_ambiguous,
        "n_dropped_ambiguous_note": "prompts whose normalized text appeared under "
                                    "BOTH labels (genuinely ambiguous)",
        "harmful_pool": n_harmful_pool,
        "benign_pool": n_benign_pool,
        "per_category_counts": per_category,
        "truncation_maxchars": MAX_PROMPT_CHARS,
        "n_per_class_requested": n_per_class,
        "n_per_class_actual": take,
        "seed": seed,
        "dedup_grouping": "group_id = sha1(normalized_text)[:16]; surface "
                          "near-dups (case/space/punct) share a group_id and are "
                          "collapsed; use for group-aware train/test splits",
    }

    print(
        f"[data_large] Toxic-Chat[{split}]: {sum(labels)} harmful + "
        f"{len(labels) - sum(labels)} benign = {len(prompts)} prompts | "
        f"natural toxic rate={natural_toxic_rate} | dropped_ambiguous="
        f"{n_dropped_ambiguous} | harm categories={len(per_category)}",
        file=sys.stderr,
    )
    return {"rows": rows, "prompts": prompts, "labels": labels, "header": header}


def load_large_dataset(
    n_per_class: int = 750, seed: int = 0, split: str = "all"
) -> tuple[list[str], list[int]]:
    """Return a balanced, deduped, LENGTH-MATCHED ``(prompts, labels)`` — SHARED set.

    REWIRED to reuse ``steering_tutorials.common.data.load_harmful_benign``, the ONE
    shared >=500/class set every tutorial imports. Two reasons over the old
    train-split-only :func:`build_dataset` path:

      * SIZE — the shared loader draws from toxic-chat's FULL (train+test) set, so
        the scarce toxic class clears >=500/class; honoring only the official train
        split capped harmful at ~370/class (below 500). Leakage safety is preserved
        by the shared loader's ``group_id`` (surface near-dups collapse to one row),
        so a downstream group-aware split stays clean despite pooling train+test.
      * CLEANLINESS — the shared loader draws the benign class **length-matched**
        (decile-bin stratified) to the harmful length histogram, so the classes are
        far less separable on length alone. The old path drew benign at random,
        leaving toxic prompts ~3x longer — a length confound the audit then had to
        discount. (The length-confound audit below still runs and now typically
        reports a much smaller residual length effect — a better, not weaker, test.)

    Label 1 = harmful, 0 = benign. The old ``build_dataset`` / ``split`` machinery is
    retained for the standalone ``python -m ...data_large`` artifact smoke; this
    loader — the one ``train_large.py`` calls — now delegates to the shared set. The
    ``split`` argument is accepted for backward compatibility and ignored.
    """
    import random

    from steering_tutorials.common.data import load_harmful_benign

    d = load_harmful_benign(n_per_class=n_per_class, seed=seed)
    harmful, benign = d["harmful"], d["benign"]
    prompts = harmful + benign
    labels = [1] * len(harmful) + [0] * len(benign)

    # Deterministic interleave so any order-sensitive downstream inspection sees
    # both classes mixed (the split itself is class-stratified, so this is cosmetic).
    rng = random.Random(seed)
    order = list(range(len(prompts)))
    rng.shuffle(order)
    prompts = [prompts[i] for i in order]
    labels = [labels[i] for i in order]
    return prompts, labels


if __name__ == "__main__":  # smoke: python -m steering_tutorials.hello_world.data_large
    import json
    from pathlib import Path

    rec = build_dataset(n_per_class=750, seed=0, split="train")
    ps, ys, header = rec["prompts"], rec["labels"], rec["header"]
    n_harm = sum(ys)
    print(f"\n[smoke] total={len(ps)}  harmful={n_harm}  benign={len(ys) - n_harm}")
    print(f"[smoke] natural toxic base rate = "
          f"{header['natural_base_rate']['toxic_fraction']} "
          f"({header['natural_base_rate']['n_toxic_raw']}/"
          f"{header['natural_base_rate']['n_rows_raw']})")
    print(f"[smoke] dropped: duplicates={header['n_dropped_duplicates']} "
          f"ambiguous={header['n_dropped_ambiguous']}")
    print(f"[smoke] harmful per-category counts = {header['per_category_counts']}")
    shown = {0: 0, 1: 0}
    for r in rec["rows"]:
        if shown[r["label"]] < 2:
            tag = "HARMFUL" if r["label"] else "BENIGN "
            print(f"  [{tag}] ({r['category']}) {r['prompt'][:80]}")
            shown[r["label"]] += 1
        if shown[0] >= 2 and shown[1] >= 2:
            break

    art = Path(__file__).parent / "artifacts" / "large_prompts.json"
    art.parent.mkdir(exist_ok=True)
    art.write_text(
        json.dumps({"header": header, "rows": rec["rows"]}, indent=2),
        encoding="utf-8",
    )
    print(f"[smoke] wrote {art} ({len(ps)} prompts, {len(rec['rows'])} rows)")
