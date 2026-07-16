"""common/data.py — the ONE shared dataset every steering tutorial imports.

Every lesson (hello_world, hello_world_steering, multi_intent, rogue_scalpel,
realignment, stacking, ...) used to build its own tiny prompt set — JailbreakBench
ships only **100 harmful + 100 benign**, which is too small to (a) train stable
diff-of-means / probe vectors, (b) hold out a real evaluation split, or (c) quote
a credible base rate. This module fixes that once, for everyone, at a **minimum of
500 harmful + 500 benign** prompts.

It is deliberately CPU-ONLY: it downloads + parses CSV/parquet and builds Python
lists of strings. It NEVER loads an LLM. Import it, get prompts, move on::

    from steering_tutorials.common.data import load_harmful_benign, load_concepts
    d = load_harmful_benign(n_per_class=500)      # {"harmful":[...], "benign":[...]}
    c = load_concepts(n_per_concept=120)          # per-harm-category concept sets

PRIMARY SOURCE — lmsys/toxic-chat (ungated, human prompt-level labels)
---------------------------------------------------------------------
Lin et al. 2023 (arXiv:2310.17389, EMNLP'23 Findings), ``0124`` release: genuine
user inputs from a live Vicuna demo, each **hand-annotated on the USER INPUT** for
``toxicity`` (0/1). That is prompt-level intent — exactly the label unit a prompt
classifier / a steering-contrast set needs — so NO response->prompt collapse is
required (the trap that biases BeaverTails; see DATASET.md). After dedup the train
split yields ~600-700 unique toxic prompts and thousands of non-toxic, so the
harmful class alone clears 500 from this one source in the common case.

THE FIVE RIGOR REQUIREMENTS (identical to hello_world/data_large.py, generalized)
---------------------------------------------------------------------------------
1. LABEL-UNIT INTEGRITY — native prompt-level labels; as a guard we drop any
   prompt whose NORMALIZED text appears under BOTH labels (genuinely ambiguous).
2. DEDUP + LEAKAGE-SAFE — every row carries ``group_id = sha1(normalized_text)``.
   Exact and surface near-duplicates (differ only by case/space/punctuation) share
   a group_id and collapse to one row, so a prompt and its trivial variants cannot
   straddle a train/eval split. We use the FULL set (train + test) so the toxic
   pool clears 500 from toxic-chat alone; the group_id keeps any downstream
   group-aware split leakage-safe regardless of the original train/test line.
3. BASE RATE — the natural toxic fraction (a ~7% minority) is recorded BEFORE we
   rebalance to 1:1; downstream PR-AUC/calibration must be read against it.
4. STRATIFICATION — the harmful class is spread across harm categories derived
   from each prompt's ``openai_moderation`` scores (sexual / harassment / violence
   / hate / self-harm); these same categories power :func:`load_concepts`.
5. REPRODUCIBILITY — fixed seed, sampling without replacement, deterministic
   order, full provenance header saved to ``artifacts/dataset_500.json``.

LENGTH-CONFOUND CONTROL
-----------------------
A classifier / a steering probe must read INTENT, not length. Real jailbreak
templates (the in-the-wild set) are very long; toxic-chat prompts are short like
the benign ones. So we (a) keep the harmful class toxic-chat-dominant, (b) draw
the benign class **length-matched** to the harmful length histogram (decile-bin
stratified), and (c) report the **per-class median char length** in the header so
the leakage/confound audit can verify the classes aren't separable on length.

TOP-UP (only if the primary source falls short) — JBB ONLY, length-matched
---------------------------------------------------------------------------
With the full set the unique toxic pool (~693) clears 500, so in the default
config **the harmful class is 100% toxic-chat and NO top-up runs**. If
``n_per_class`` is raised past the toxic pool, the harmful class is topped up from
**JailbreakBench harmful behaviors ONLY** (``JailbreakBench/JBB-Behaviors``
``data/harmful-behaviors.csv`` ``Goal`` — short, curated, length-compatible),
deduped against everything already chosen and **length-windowed** to the
[10th,90th] percentile of the toxic-chat toxic length range.

In-the-wild jailbreak prompts (``TrustAIRLab/in-the-wild-jailbreak-prompts``) are
**deliberately NEVER used**: they are long DAN/roleplay templates that
length/style-confound against short toxic-chat + benign prompts. If a clean
length-matched harmful class cannot reach ``n_per_class``, we return the max clean
count and say so — a clean 450/class beats a confounded 500/class.
"""
from __future__ import annotations

import ast
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict

try:  # OS trust store for an SSL-intercepting middlebox (same guard as data_large).
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional
    pass

# ---------------------------------------------------------------------------
# Pinned sources. toxic-chat 0124 is the latest human re-annotation and ships an
# official train/test split we honor for leakage safety.
# ---------------------------------------------------------------------------
TOXICCHAT_REPO = "lmsys/toxic-chat"
TOXICCHAT_FILES = {
    "train": "data/0124/toxic-chat_annotation_train.csv",
    "test": "data/0124/toxic-chat_annotation_test.csv",
    # "all" = train + test. We default to it: the toxic class is a ~7% minority,
    # so honoring only the official train split caps unique toxic at ~300 (< 500).
    # The full set yields ~693 unique toxic, enough for a toxic-chat-DOMINANT
    # harmful class with no length-confounding jailbreak top-up. Leakage safety is
    # preserved downstream by the group_id: a group-aware split still keeps a
    # prompt and its near-dups on one side of any train/eval cut a lesson makes.
    "all": "data/0124/toxic-chat_annotation_all.csv",
}
SRC_TOXICCHAT = "lmsys/toxic-chat@0124"

# The ONLY harmful top-up source, and only if toxic-chat toxic < n_per_class after
# dedup. JBB is curated single-sentence harmful *behaviors* — short, so it is
# length-compatible with toxic-chat. We deliberately do NOT use in-the-wild
# jailbreak prompts: they are long DAN/roleplay templates that length/style-confound
# against short toxic-chat + benign prompts (a probe would separate on length, not
# intent). A harmful class must never be discriminable by length alone.
JBB_REPO = "JailbreakBench/JBB-Behaviors"
JBB_HARMFUL_CSV = "data/harmful-behaviors.csv"
JBB_PROMPT_COL = "Goal"
SRC_JBB = "JailbreakBench/JBB-Behaviors"

# Column-name variants tolerated across toxic-chat releases.
_PROMPT_FIELDS = ("user_input", "prompt", "text")
_LABEL_FIELDS = ("toxicity", "toxic", "label")
_MODERATION_FIELDS = ("openai_moderation", "moderation")

# Prompts longer than this are TRUNCATED (not dropped) so a rare wall-of-text
# jailbreak still contributes without dominating tokenization downstream.
MAX_CHARS = 2000
MIN_CHARS = 10

# Coarse harm concepts: OpenAI-moderation fine categories folded to five buckets.
# (order fixed for determinism / stable concept naming.)
_COARSE = {
    "sexual": ("sexual", "sexual/minors"),
    "harassment": ("harassment", "harassment/threatening"),
    "violence": ("violence", "violence/graphic"),
    "hate": ("hate", "hate/threatening"),
    "self_harm": ("self-harm", "self-harm/intent", "self-harm/instructions"),
}
_FINE_TO_COARSE = {fine: coarse for coarse, fines in _COARSE.items() for fine in fines}
CONCEPT_NAMES = list(_COARSE)  # ["sexual","harassment","violence","hate","self_harm"]


# --------------------------------------------------------------------------- #
# Text normalization / grouping helpers (self-contained; no cross-lesson import)
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    """Readable normalization for the STORED prompt: drop U+FFFD, squeeze space,
    then truncate to ``MAX_CHARS``."""
    s = str(text).replace("�", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:MAX_CHARS]


def _norm_key(text: str) -> str:
    """Aggressive normalization for dedup/grouping: lowercase, alnum-only tokens.

    Collapses case/whitespace/punctuation variants to one key so surface
    near-duplicates share a ``group_id``. (Semantic paraphrases are NOT caught —
    that needs embeddings, out of scope for this CPU/text-only loader.)
    """
    s = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", s).strip()


def _group_id(norm: str) -> str:
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def _coarse_category(moderation_cell) -> str:
    """Fold a prompt's ``openai_moderation`` scores to one coarse harm concept.

    The cell is a string repr of ``[[fine_category, score], ...]``. We sum scores
    within each coarse bucket and take the argmax; 'unlabeled' if nothing maps.
    Used to STRATIFY the harmful class and to derive :func:`load_concepts`.
    """
    try:
        data = moderation_cell
        if isinstance(data, str):
            data = ast.literal_eval(data)
        agg: dict[str, float] = defaultdict(float)
        for fine, score in dict(data).items():
            coarse = _FINE_TO_COARSE.get(fine)
            if coarse is not None:
                agg[coarse] += float(score)
        if not agg or max(agg.values()) <= 0.0:
            return "unlabeled"
        return max(agg, key=lambda k: agg[k])
    except Exception:
        return "unlabeled"


def _first_col(df, fields: tuple[str, ...]) -> str:
    for f in fields:
        if f in df.columns:
            return f
    raise KeyError(f"none of {fields} in columns {list(df.columns)}")


def _stratified_sample(items: list, keyfn, take: int, rng) -> list:
    """Deterministically sample ``take`` items spread across ``keyfn`` strata via
    largest-remainder allocation. If ``take >= len(items)`` returns all (shuffled)."""
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
    remainders = []  # [fractional_remainder, key, bucket, n_taken]
    for key, bucket in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        exact = take * len(bucket) / total
        n = int(exact)
        chosen.extend(bucket[:n])
        remainders.append([exact - n, key, bucket, n])
    remainders.sort(key=lambda r: (-r[0], r[1]))
    i = 0
    while len(chosen) < take and remainders:
        slot = remainders[i % len(remainders)]
        _, _, bucket, n = slot
        if n < len(bucket):
            chosen.append(bucket[n])
            slot[3] = n + 1
        i += 1
        if all(n >= len(b) for _, _, b, n in remainders):
            break
    rng.shuffle(chosen)
    return chosen[:take]


def _median_len(rows: list) -> int:
    """Median char length of a list of {prompt} rows (0 if empty)."""
    if not rows:
        return 0
    lens = sorted(len(r["prompt"]) for r in rows)
    m = len(lens) // 2
    return int(lens[m] if len(lens) % 2 else (lens[m - 1] + lens[m]) / 2)


def _length_matched_sample(pool: list, target: list, take: int, rng) -> list:
    """Sample ``take`` rows from ``pool`` whose length histogram mirrors ``target``.

    We bin the ``target`` (harmful) lengths into deciles and allocate ``take``
    across those bins in proportion to the target's mass, then draw that many
    ``pool`` (benign) rows from each length bin. Bins that run short spill their
    deficit to a final random top-up. Result: benign and harmful share a length
    distribution, so a classifier can't separate the classes on length alone.
    """
    if take >= len(pool):
        out = list(pool)
        rng.shuffle(out)
        return out
    tlens = sorted(len(r["prompt"]) for r in target)
    # decile edges of the target distribution (9 internal cut points).
    edges = [tlens[int(len(tlens) * q)] for q in (0.1, 0.2, 0.3, 0.4, 0.5,
                                                  0.6, 0.7, 0.8, 0.9)]

    def bin_of(n: int) -> int:
        b = 0
        for e in edges:
            if n >= e:
                b += 1
            else:
                break
        return b  # 0..9

    target_counts = Counter(bin_of(n) for n in tlens)
    pool_bins: dict = defaultdict(list)
    for r in pool:
        pool_bins[bin_of(len(r["prompt"]))].append(r)
    for b in pool_bins.values():
        rng.shuffle(b)

    chosen, leftover = [], []
    total_t = sum(target_counts.values())
    for b in range(10):
        want = round(take * target_counts.get(b, 0) / total_t)
        have = pool_bins.get(b, [])
        chosen.extend(have[:want])
        leftover.extend(have[want:])  # unused rows fall back to the top-up pool
    rng.shuffle(leftover)
    i = 0
    while len(chosen) < take and i < len(leftover):
        chosen.append(leftover[i])
        i += 1
    rng.shuffle(chosen)
    return chosen[:take]


# --------------------------------------------------------------------------- #
# Source loaders (each returns a list of {prompt, group_id, source, category})
# --------------------------------------------------------------------------- #
def _dl(repo: str, filename: str):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def _load_toxicchat(split: str):
    """Return (toxic_rows, benign_rows, base_rate_stats, ambiguity_stats).

    Rows are deduped-within-source by group_id; a normalized text seen under BOTH
    toxicity labels is dropped as ambiguous. Each toxic row carries its coarse
    harm ``category``; benign rows are tagged ``benign``.
    """
    import pandas as pd

    df = pd.read_csv(_dl(TOXICCHAT_REPO, TOXICCHAT_FILES[split]))
    prompt_col = _first_col(df, _PROMPT_FIELDS)
    label_col = _first_col(df, _LABEL_FIELDS)
    try:
        mod_col = _first_col(df, _MODERATION_FIELDS)
    except KeyError:
        mod_col = None

    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    n_raw = len(df)
    n_toxic_raw = int((labels == 1).sum())
    mod_series = df[mod_col] if mod_col else [None] * n_raw

    groups: dict = {}  # gid -> {"display","labels":set,"cat"}
    n_after_filters = 0
    for raw, lab, modcell in zip(df[prompt_col].astype(str), labels, mod_series):
        display = _clean(raw)
        if not (MIN_CHARS <= len(display) <= MAX_CHARS):
            continue
        norm = _norm_key(raw)
        if not norm:
            continue
        n_after_filters += 1
        gid = _group_id(norm)
        g = groups.get(gid)
        if g is None:
            cat = _coarse_category(modcell) if (lab == 1 and mod_col) else "benign"
            groups[gid] = {"display": display, "labels": {int(lab)}, "cat": cat}
        else:
            g["labels"].add(int(lab))

    n_dropped_dups = n_after_filters - len(groups)
    toxic_rows, benign_rows, n_ambiguous = [], [], 0
    for gid, g in groups.items():
        if len(g["labels"]) > 1:
            n_ambiguous += 1
            continue
        row = {"prompt": g["display"], "group_id": gid,
               "source": SRC_TOXICCHAT, "category": g["cat"]}
        (toxic_rows if next(iter(g["labels"])) == 1 else benign_rows).append(row)

    base_rate = {"n_rows_raw": n_raw, "n_toxic_raw": n_toxic_raw,
                 "toxic_fraction": round(n_toxic_raw / max(1, n_raw), 4)}
    ambiguity = {"n_dropped_duplicates": n_dropped_dups,
                 "n_dropped_ambiguous": n_ambiguous}
    return toxic_rows, benign_rows, base_rate, ambiguity


def _load_topup_harmful_jbb(seen_gids: set,
                            length_window: tuple[int, int]) -> list:
    """Load JailbreakBench harmful behaviors as a length-matched harmful top-up.

    Only invoked when toxic-chat toxic < ``n_per_class`` after dedup. Deduped
    against ``seen_gids`` (mutated) and restricted to ``length_window`` = (lo, hi)
    char bounds (the [10th,90th] percentile of toxic-chat toxic lengths) so the
    top-up cannot length-confound the class. In-the-wild jailbreaks are never used.
    """
    import pandas as pd

    df = pd.read_csv(_dl(JBB_REPO, JBB_HARMFUL_CSV))
    texts = df[JBB_PROMPT_COL].dropna().astype(str).tolist()

    lo, hi = length_window
    rows = []
    for raw in texts:
        display = _clean(raw)
        if not (MIN_CHARS <= len(display) <= MAX_CHARS):
            continue
        if not (lo <= len(display) <= hi):  # length-confound guard
            continue
        gid = _group_id(_norm_key(raw))
        if not gid or gid in seen_gids:
            continue
        seen_gids.add(gid)
        rows.append({"prompt": display, "group_id": gid,
                     "source": SRC_JBB, "category": "unlabeled"})
    return rows


# --------------------------------------------------------------------------- #
# PUBLIC API 1 — the balanced ≥500/class harmful/benign set
# --------------------------------------------------------------------------- #
def build_harmful_benign(n_per_class: int = 500, seed: int = 0, split: str = "all") -> dict:
    """Build the balanced harmful/benign set; return rows + provenance header.

    Harmful = toxic-chat TOXIC ONLY (the full set yields ~693 unique, so this class
    is 100% toxic-chat by default); topped up from JailbreakBench ONLY (never
    in-the-wild), length-windowed, and only if ``n_per_class`` exceeds the toxic
    pool. Benign = toxic-chat non-toxic, sampled **length-matched** to the harmful
    class so length can't leak the label. Both classes are deduped, group-id'd, and
    no normalized prompt spans both classes.
    Returns ``{"harmful":[rows], "benign":[rows], "header":{...}}`` where each row
    is ``{prompt, group_id, source, category}``.
    """
    rng = random.Random(seed)
    toxic_rows, benign_rows, base_rate, ambiguity = _load_toxicchat(split)

    # group_ids already chosen anywhere — the cross-source/cross-class dedup guard.
    seen: set = {r["group_id"] for r in toxic_rows} | {r["group_id"] for r in benign_rows}

    # Length window for the JBB top-up = the 10th..90th percentile of the
    # toxic-chat toxic length distribution, so a top-up matches the core length.
    tox_lens = sorted(len(r["prompt"]) for r in toxic_rows) or [MIN_CHARS, MAX_CHARS]
    lo = tox_lens[int(len(tox_lens) * 0.10)]
    hi = tox_lens[min(len(tox_lens) - 1, int(len(tox_lens) * 0.90))]

    # --- HARMFUL: toxic-chat TOXIC ONLY; JBB (length-matched) only if short ------
    # The full toxic-chat set holds ~693 unique toxic, so in the default config the
    # harmful class is 100% toxic-chat and this top-up never runs. In-the-wild
    # jailbreak prompts are NEVER used (length/style confound).
    harmful_pool = list(toxic_rows)
    topup_log: list[dict] = []
    if len(harmful_pool) < n_per_class:
        added = _load_topup_harmful_jbb(seen, length_window=(lo, hi))
        harmful_pool.extend(added)
        topup_log.append({"source": SRC_JBB, "added": len(added),
                          "length_window": [lo, hi]})

    take = min(n_per_class, len(harmful_pool), len(benign_rows))
    if take < n_per_class:
        print(f"[common.data] NOTE: clean length-matched harmful pool caps the set "
              f"at {take}/class (toxic-chat toxic + JBB top-up = {len(harmful_pool)}, "
              f"benign_pool={len(benign_rows)}); requested {n_per_class}. Returning "
              f"the max clean count — a clean {take}/class beats a confounded "
              f"{n_per_class}/class.", file=sys.stderr)

    harmful_sample = _stratified_sample(harmful_pool, lambda r: r["category"], take, rng)
    # BENIGN: length-matched to the harmful sample (decile-bin stratified).
    benign_sample = _length_matched_sample(benign_rows, harmful_sample, take, rng)

    # per-source counts among what we ACTUALLY sampled (the honest blend).
    sampled_by_source = dict(Counter(r["source"] for r in harmful_sample))
    per_category = dict(Counter(r["category"] for r in harmful_sample))
    median_len = {"harmful": _median_len(harmful_sample),
                  "benign": _median_len(benign_sample)}

    header = {
        "primary_source": SRC_TOXICCHAT,
        "split_used": split,
        "n_harmful": len(harmful_sample),
        "n_benign": len(benign_sample),
        "n_per_class_requested": n_per_class,
        "natural_toxic_rate": base_rate["toxic_fraction"],
        "natural_base_rate_detail": base_rate,
        "dedup_dropped": ambiguity["n_dropped_duplicates"],
        "dropped_ambiguous": ambiguity["n_dropped_ambiguous"],
        "harmful_pool_after_topup": len(harmful_pool),
        "benign_pool": len(benign_rows),
        "per_source_counts_sampled": sampled_by_source,
        "topup_log": topup_log,
        "per_category_counts_harmful": per_category,
        "median_char_length": median_len,
        "length_confound_note": "benign is length-matched (decile-bin stratified) "
                                "to harmful; compare median_char_length per class",
        "max_chars": MAX_CHARS,
        "seed": seed,
        "dedup_grouping": "group_id = sha1(normalized_text)[:16]; surface near-dups "
                          "(case/space/punct) collapse; no group_id spans both classes",
        "label_scheme": {"harmful": "toxic-chat toxicity==1 or a jailbreak top-up",
                         "benign": "toxic-chat toxicity==0"},
    }
    return {"harmful": harmful_sample, "benign": benign_sample, "header": header}


def load_harmful_benign(n_per_class: int = 500, seed: int = 0) -> dict:
    """Return ``{"harmful": list[str], "benign": list[str]}`` — the shared set.

    >=500 of each by default from the full toxic-chat set, normalized + deduped
    (exact and surface near-dups), balanced, length-matched, shuffled with
    ``seed``. The full provenance (per-source blend, base rate, category
    stratification, per-class median length) is on :func:`build_harmful_benign`;
    this thin wrapper returns just the prompt lists that lessons consume.
    """
    rec = build_harmful_benign(n_per_class=n_per_class, seed=seed)
    return {"harmful": [r["prompt"] for r in rec["harmful"]],
            "benign": [r["prompt"] for r in rec["benign"]]}


# --------------------------------------------------------------------------- #
# PUBLIC API 2 — per-harm-category concepts (for multi_intent / stacking lessons)
# --------------------------------------------------------------------------- #
def load_concepts(n_per_concept: int = 120, seed: int = 0, split: str = "all",
                  held_out: str | None = None) -> dict:
    """Derive K harm concepts from toxic-chat moderation flags + a benign baseline.

    Each concept is the toxic-chat toxic prompts whose dominant coarse harm
    category is that concept, split into disjoint ``exemplars`` / ``steer`` /
    ``eval`` (40/30/30) so a vector is never graded on the prompts that built it.
    A shared benign baseline (the common contrast origin) is returned once.

    Returns::

        {
          "concepts": {name: {"exemplars":[...], "steer":[...], "eval":[...],
                              "n_available": int}, ...},
          "baseline": [...benign...],
          "held_out": <concept name reserved for zero-shot transfer>,
          "concept_names": [...],
        }

    Per-concept counts are reported honestly — some harm categories are small
    (self-harm is typically the smallest, sexual the largest); each concept is
    capped at ``min(n_per_concept, available)``. ``held_out`` (default: the
    SECOND-largest pool — big enough to evaluate, kept out of training) names the
    concept reserved for zero-shot generalization tests.
    """
    rng = random.Random(seed)
    toxic_rows, benign_rows, _, _ = _load_toxicchat(split)

    by_concept: dict[str, list] = defaultdict(list)
    for r in toxic_rows:
        if r["category"] in _COARSE:  # skip 'unlabeled'
            by_concept[r["category"]].append(r)

    concepts: dict[str, dict] = {}
    pool_sizes: dict[str, int] = {}
    for name in CONCEPT_NAMES:
        rows = list(by_concept.get(name, []))
        rng.shuffle(rows)
        pool_sizes[name] = len(rows)
        rows = rows[:n_per_concept]
        prompts = [r["prompt"] for r in rows]
        n = len(prompts)
        n_ex = int(round(n * 0.4))
        n_steer = int(round(n * 0.3))
        concepts[name] = {
            "exemplars": prompts[:n_ex],
            "steer": prompts[n_ex:n_ex + n_steer],
            "eval": prompts[n_ex + n_steer:],
            "n_available": pool_sizes[name],
        }

    # Default held-out = second-largest pool: enough eval prompts, and holding out
    # the very largest would waste the richest training concept.
    if held_out is None:
        ranked = sorted(CONCEPT_NAMES, key=lambda k: -pool_sizes[k])
        held_out = ranked[1] if len(ranked) > 1 else ranked[0]

    rng.shuffle(benign_rows)
    baseline = [r["prompt"] for r in benign_rows[:max(n_per_concept, 40)]]

    for name in CONCEPT_NAMES:
        s = concepts[name]
        print(f"[common.data] concept {name:12s} available={s['n_available']:4d}  "
              f"exemplars={len(s['exemplars'])} steer={len(s['steer'])} "
              f"eval={len(s['eval'])}", file=sys.stderr)
    print(f"[common.data] held_out(zero-shot)={held_out}  baseline={len(baseline)}",
          file=sys.stderr)

    return {"concepts": concepts, "baseline": baseline,
            "held_out": held_out, "concept_names": CONCEPT_NAMES}


# --------------------------------------------------------------------------- #
# Smoke / artifact writer:  python -m steering_tutorials.common.data
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json
    from pathlib import Path

    rec = build_harmful_benign(n_per_class=500, seed=0)
    harmful, benign, header = rec["harmful"], rec["benign"], rec["header"]

    print(f"\n[smoke] harmful={len(harmful)}  benign={len(benign)}")
    print(f"[smoke] natural toxic base rate = {header['natural_toxic_rate']} "
          f"({header['natural_base_rate_detail']['n_toxic_raw']}/"
          f"{header['natural_base_rate_detail']['n_rows_raw']})")
    print(f"[smoke] harmful blend by source = {header['per_source_counts_sampled']}")
    print(f"[smoke] top-up log = {header['topup_log']}")
    print(f"[smoke] harmful per-category = {header['per_category_counts_harmful']}")
    print(f"[smoke] median char length = harmful:{header['median_char_length']['harmful']} "
          f"benign:{header['median_char_length']['benign']} (length-matched)")
    print(f"[smoke] dropped: duplicates={header['dedup_dropped']} "
          f"ambiguous={header['dropped_ambiguous']}")

    # --- the two hard assertions the task requires ------------------------------
    assert len(harmful) >= 500, f"harmful ceiling {len(harmful)} < 500 — see NOTE above"
    assert len(benign) >= 500, f"benign ceiling {len(benign)} < 500"
    h_gids = {r["group_id"] for r in harmful}
    b_gids = {r["group_id"] for r in benign}
    assert not (h_gids & b_gids), "a normalized prompt span harmful AND benign — leakage!"
    print("[smoke] OK: >=500/class and no group_id spans both classes")

    for cls, rows in (("HARMFUL", harmful), ("BENIGN ", benign)):
        for r in rows[:2]:
            print(f"  [{cls}] ({r['category']}|{r['source'].split('/')[-1]}) "
                  f"{r['prompt'][:76]}")

    # --- concepts smoke ---------------------------------------------------------
    con = load_concepts(n_per_concept=120, seed=0)
    print(f"[smoke] concepts={con['concept_names']}  held_out={con['held_out']}")

    # --- write the artifact -----------------------------------------------------
    art = Path(__file__).parent / "artifacts" / "dataset_500.json"
    art.parent.mkdir(exist_ok=True)
    art.write_text(json.dumps({
        "header": header,
        "harmful": harmful,
        "benign": benign,
    }, indent=2), encoding="utf-8")
    print(f"[smoke] wrote {art} ({len(harmful)}+{len(benign)} prompts)")
