"""data.py -- dataset loader for the multi-turn jailbreak DETECTION lesson.

POSITIVES  SafeMTData/SafeMTData, BOTH configs:
             `Attack_600`    -- `multi_turn_queries` (4-5 escalating USER turns).
             `SafeMTData_1K` -- `conversations` = [{role,content}]; the USER-role
                                turns are the same escalation chain.
NEGATIVES  (easy) HuggingFaceH4/ultrachat_200k (streaming) -- USER turns of real
             benign multi-turn chats, TOPIC-MATCHED to the attack categories via
             C.CATEGORY_KEYWORDS and turn-count biased; or, opt-in,
             tom-gibbs/multi-turn_jailbreak_attack_datasets Semi-Benign.
           (hard) benign PREFIXES of a DISJOINT half of the attack pool.
OOD        intrinsec-ai/cstm-bench -- ungated, MIT, cached (see `load_ood`).

THE GROUP KEY IS `plain_query`, NOT `query_id`.
`query_id` is re-indexed independently per config: 157 of 200 ids collide across
Attack_600 and SafeMTData_1K while `plain_query` collides 0 times. Grouping by
`query_id` after concatenating would silently merge 157 unrelated attack goals into
shared CV groups -- manufacturing fake groups and corrupting exactly the leakage
discipline this lesson is built on. It would not crash (section 18.8: silent and
plausible). `attack_group_key` is the single place that decision lives.

CPU-only. Downloads datasets. Loads NO LLM. Stdout is ASCII-only (Windows cp1252).
"""
from __future__ import annotations

try:  # SSL middlebox on this host (truststore lets requests use the OS trust store)
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import hashlib
import json
import sys
import random
from collections import Counter

from steering_tutorials.common.confound import (
    confound_report as _shared_confound_report,
    format_report,
)
from steering_tutorials.multiturn_jailbreak import config as C


# --- helpers -----------------------------------------------------------------
def _eprint(*args) -> None:
    """ASCII-only stderr print."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


def _clean_turn(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _match_category(text: str) -> str:
    """Return the attack category whose keywords appear in `text`, else 'general'.

    First-match wins over C.ATTACK_CATEGORIES ordering (deterministic).
    """
    low = text.lower()
    for cat in C.ATTACK_CATEGORIES:
        for kw in C.CATEGORY_KEYWORDS.get(cat, []):
            if kw in low:
                return cat
    return "general"


def _turncount_dist(convs) -> Counter:
    return Counter(len(c) for c in convs)


def attack_group_key(row: dict, config_name: str) -> str:
    """The CV group key for one attack row. See the module docstring.

    `plain_query` is the harmful GOAL; every attack path toward the same goal shares
    it, across BOTH configs. Falls back to a config-qualified query_id only if the
    row genuinely has no `plain_query` (never observed in either cached config) --
    the fallback stays config-qualified so it can never collide across configs.
    """
    pq = _clean_turn(row.get("plain_query") or "")
    if pq:
        return pq
    return "%s::qid=%s" % (config_name, row.get("query_id", "?"))


def _group_ids(keys):
    """Map group KEY STRINGS to stable ints (deterministic: sorted-unique order)."""
    order = {k: i for i, k in enumerate(sorted(set(keys)))}
    return [order[k] for k in keys], len(order)


def dataset_fingerprint(conversations, labels) -> str:
    """SHA-256 over the (label, turns) content. Stamps WHICH data produced a number.

    section 18.8 "stamp your inputs": `meerkat`'s results.json could not be regenerated
    from the code beside it because nothing recorded the pool. This hash goes into
    results.json and into every embedding cache, so a changed pool is detectable
    rather than merely undetectable-and-plausible.
    """
    h = hashlib.sha256()
    for conv, lab in zip(conversations, labels):
        h.update(str(int(lab)).encode("utf-8"))
        h.update(b"\x00")
        for t in conv:
            h.update(str(t).encode("utf-8", "replace"))
            h.update(b"\x01")
        h.update(b"\x02")
    return h.hexdigest()


# --- attack pool (BOTH SafeMTData configs) -----------------------------------
def _user_turns_from_row(row, config_name, min_turns, max_turns):
    """Extract the USER-turn list from either SafeMTData schema, or None."""
    raw = row.get("multi_turn_queries")
    if raw:
        turns = [_clean_turn(t) for t in raw if _clean_turn(t)]
    else:
        convo = row.get("conversations") or []
        turns = [_clean_turn(m.get("content", "")) for m in convo
                 if str(m.get("role", "")).lower() == "user" and _clean_turn(m.get("content", ""))]
    turns = turns[:max_turns]
    if len(turns) < min_turns:
        return None
    return turns


def _load_all_attacks(min_turns, max_turns, configs=None):
    """Every attack conversation across `configs`, with its GROUP KEY + category.

    Returns list of (turns, group_key:str, category:str, config_name:str).
    """
    from datasets import load_dataset as hf_load

    configs = list(configs or C.ATTACK_CONFIGS)
    rows = []
    for cfg in configs:
        ds = hf_load(C.ATTACK_DATASET, cfg)
        # Each config exposes a single split named after the config.
        split = cfg if cfg in ds else list(ds.keys())[0]
        for row in ds[split]:
            turns = _user_turns_from_row(row, cfg, min_turns, max_turns)
            if turns is None:
                continue
            rows.append((turns, attack_group_key(row, cfg),
                         str(row.get("category", "general")), cfg))
    return rows


def _load_positives(n_pos, min_turns, max_turns, rng=None):
    """Up to `n_pos` attack conversations, shuffled across configs (fixed seed)."""
    rows = _load_all_attacks(min_turns, max_turns)
    if rng is not None:
        rng.shuffle(rows)
    rows = rows[:n_pos]
    convs = [r[0] for r in rows]
    keys = [r[1] for r in rows]
    cats = [r[2] for r in rows]
    cfgs = [r[3] for r in rows]
    return convs, keys, cats, cfgs


def hard_pool_ceiling(min_turns=None, max_turns=None, window=None) -> dict:
    """MEASURE the HARD pool ceiling instead of guessing it (rule 2).

    The HARD condition splits attack GROUPS into disjoint halves; positives need
    >= window turns, negatives need > window turns (so the first-`window` prefix
    genuinely excludes the payload). The ceiling is therefore roughly half the pool
    per class, and it must be STATED, not merely respected.
    """
    W = int(window or C.HARD_WINDOW)
    rows = _load_all_attacks(int(min_turns or C.MIN_USER_TURNS),
                             int(max_turns or C.MAX_USER_TURNS))
    n_ge = sum(1 for t, _, _, _ in rows if len(t) >= W)
    n_gt = sum(1 for t, _, _, _ in rows if len(t) > W)
    groups = len(set(k for _, k, _, _ in rows))
    return {
        "attack_configs": list(C.ATTACK_CONFIGS),
        "rows_total": len(rows),
        "rows_ge_window": n_ge,
        "rows_gt_window": n_gt,
        "distinct_groups": groups,
        "hard_window": W,
        # halved by the disjoint-group split; the exact split lands within a few rows
        "pos_ceiling_approx": n_ge // 2,
        "neg_ceiling_approx": n_gt // 2,
        "meets_rule1": bool(min(n_ge // 2, n_gt // 2) >= C.RULE1_FLOOR),
    }


# --- negatives: UltraChat ----------------------------------------------------
def _extract_user_turns(row, min_turns: int, max_turns: int):
    """Return the list of cleaned USER turns from an UltraChat row, or None."""
    msgs = row.get("messages") or []
    turns = [_clean_turn(m.get("content", "")) for m in msgs
             if m.get("role") == "user" and _clean_turn(m.get("content", ""))]
    if len(turns) < min_turns:
        return None
    return turns[:max_turns]


def _load_negatives(n_neg: int, min_turns: int, max_turns: int,
                    pos_turncounts: Counter, rng: random.Random):
    """Stream UltraChat; prefer topic-matched, turn-count-biased benign convs.

    Two buckets are collected: topic-matched (hard negatives sharing surface with
    the attacks) and general. We prefer topic-matched, backfill with general, and
    bias the accepted turn-counts toward the positive distribution.
    """
    from datasets import load_dataset as hf_load

    # Target per-turncount quota from the positive distribution (scaled to n_neg).
    total_pos = sum(pos_turncounts.values()) or 1
    quota = {k: max(1, round(n_neg * v / total_pos)) for k, v in pos_turncounts.items()}
    accepted_by_tc: Counter = Counter()

    matched, general = [], []  # each item: (turns, category)
    # Cap streaming so a short category cannot make us read all 200k rows.
    max_scan = max(20000, n_neg * 200)
    scanned = 0

    stream = hf_load(C.BENIGN_DATASET, split=C.BENIGN_SPLIT, streaming=True)
    for row in stream:
        scanned += 1
        if scanned > max_scan:
            break
        turns = _extract_user_turns(row, min_turns, max_turns)
        if turns is None:
            continue
        tc = len(turns)
        # Turn-count biasing: skip an over-quota count as soon as that count has met
        # its share. (The pre-2026-08 version also required the pool to already hold
        # n_neg items, which meant the biasing almost never fired.)
        if quota and accepted_by_tc.get(tc, 0) >= quota.get(tc, 0):
            continue
        cat = _match_category(" \n ".join(turns))
        if cat != "general":
            matched.append((turns, cat))
        else:
            general.append((turns, cat))
        accepted_by_tc[tc] += 1
        # Stop once we have enough total candidates (prefer matched, keep buffer).
        if len(matched) >= n_neg or (len(matched) + len(general)) >= n_neg * 3:
            if len(matched) + len(general) >= n_neg:
                break

    # Prefer topic-matched, backfill with general.
    rng.shuffle(matched)
    rng.shuffle(general)
    picked = matched[:n_neg]
    if len(picked) < n_neg:
        picked += general[: n_neg - len(picked)]

    convs = [t for t, _ in picked]
    cats = [c for _, c in picked]
    return convs, cats, scanned


# --- negatives: tom-gibbs Semi-Benign (opt-in, needs a download) -------------
_TOMGIBBS_TURN_FIELDS = ("Multi-turn Conversation", "multi_turn_conversation",
                         "conversation", "Conversation", "turns")


def _tomgibbs_turns(row, min_turns, max_turns):
    """Parse one tom-gibbs row into a USER-turn list, or None.

    The turn column ships in several shapes across the repo's configs (a JSON string,
    a list of strings, or a list of {role,content} dicts), so all three are handled.
    """
    raw = None
    for f in _TOMGIBBS_TURN_FIELDS:
        if row.get(f):
            raw = row[f]
            break
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = [raw]
    turns = []
    for item in raw:
        if isinstance(item, dict):
            if str(item.get("role", "user")).lower() not in ("user", "human", ""):
                continue
            item = item.get("content") or item.get("text") or ""
        t = _clean_turn(item)
        if t:
            turns.append(t)
    turns = turns[:max_turns]
    if len(turns) < min_turns:
        return None
    return turns


def _load_negatives_tomgibbs(n_neg, min_turns, max_turns, rng):
    """Semi-Benign / Completely-Benign multi-turn conversations as hard negatives.

    These are built alongside the repo's 4,136 harmful multi-turn conversations, so
    they are per-attack-style benign controls -- a strictly better negative than
    keyword-matched UltraChat. README section 9 previously claimed such data "does not
    exist ready-made"; it does (ungated, MIT). NOT cached on this host: selecting
    MJ_NEG_SOURCE=tomgibbs triggers a download.

    Fails LOUDLY (never silently empty) if a configured config name is absent.
    """
    from datasets import load_dataset as hf_load, get_dataset_config_names

    available = list(get_dataset_config_names(C.TOMGIBBS_DATASET))
    picked_cfgs = [c for c in C.TOMGIBBS_NEG_CONFIGS if c in available]
    if not picked_cfgs:
        raise RuntimeError(
            "none of MJ_TOMGIBBS_NEG=%r exist in %s; available configs: %r"
            % (C.TOMGIBBS_NEG_CONFIGS, C.TOMGIBBS_DATASET, available))

    out = []
    for cfg in picked_cfgs:
        ds = hf_load(C.TOMGIBBS_DATASET, cfg)
        split = list(ds.keys())[0]
        for row in ds[split]:
            turns = _tomgibbs_turns(row, min_turns, max_turns)
            if turns is not None:
                out.append((turns, cfg))
    if not out:
        raise RuntimeError(
            "tom-gibbs configs %r parsed to 0 usable conversations -- the turn column "
            "was not one of %r. Inspect the schema before trusting any number."
            % (picked_cfgs, list(_TOMGIBBS_TURN_FIELDS)))
    rng.shuffle(out)
    out = out[:n_neg]
    return [t for t, _ in out], [c for _, c in out], len(out)


# --- HARD condition: benign-prefix negatives from a DISJOINT attack split -----
def _build_hard(n_per_class, min_turns, max_turns, rng, window=None):
    """LENGTH-MATCHED hard set. Both classes are EXACTLY `window` turns:
      positive = the LAST `window` turns of an attack (contains the escalation/payload
                 that lands at the final turn);
      negative = the FIRST `window` turns of a DIFFERENT attack (the benign lead-up,
                 which EXCLUDES the payload because the payload is the last turn).
    So the ONLY systematic difference is whether the window contains the escalation --
    NOT its length (turn-count is constant -> no length confound) and NOT the source
    conversation (positive/negative GROUPS are disjoint -> leakage-free). A per-turn
    probe cannot separate them (each turn is an innocuous ActorAttack question); only a
    model that reads across the turns can. Negatives need a source attack with
    > window turns (so first-`window` truly excludes the last/payload turn).

    Groups are `plain_query` (see attack_group_key), so the halves are disjoint by
    harmful GOAL across both SafeMTData configs -- not by a per-config query_id.
    """
    W = int(window or getattr(C, "HARD_WINDOW", 4))
    rows = _load_all_attacks(min_turns, max_turns)
    by_key = {}
    for turns, key, cat, cfg in rows:
        by_key.setdefault(key, []).append((turns, cat, cfg))
    keys = list(by_key)
    rng.shuffle(keys)
    half = len(keys) // 2
    pos_keys, neg_keys = set(keys[:half]), set(keys[half:])

    pos, negs = [], []
    for key in pos_keys:
        for turns, cat, cfg in by_key[key]:
            if len(turns) >= W:
                pos.append((turns[-W:], key, cat, cfg))     # last W: includes payload
    for key in neg_keys:
        for turns, cat, cfg in by_key[key]:
            if len(turns) > W:                              # need > W so first-W drops payload
                negs.append((turns[:W], key, cat, cfg))     # first W: benign lead-up
    rng.shuffle(pos)
    rng.shuffle(negs)
    pos = pos[:n_per_class]
    negs = negs[:n_per_class]

    conversations = [t for t, _, _, _ in pos] + [t for t, _, _, _ in negs]
    labels = [1] * len(pos) + [0] * len(negs)
    group_keys = [k for _, k, _, _ in pos] + [k for _, k, _, _ in negs]
    categories = [c for _, _, c, _ in pos] + [c for _, _, c, _ in negs]
    sources = ["attack_full"] * len(pos) + ["attack_prefix"] * len(negs)
    configs = [g for _, _, _, g in pos] + [g for _, _, _, g in negs]
    return conversations, labels, group_keys, categories, sources, configs


# --- public API --------------------------------------------------------------
def load_dataset(n_pos=None, n_neg=None, min_turns=None, max_turns=None,
                 seed=None, condition="easy") -> dict:
    """Load the balanced multi-turn detection dataset.

    condition="easy": positives = ActorAttack attacks; negatives = topic-matched
        benign UltraChat conversations (or tom-gibbs Semi-Benign if
        MJ_NEG_SOURCE=tomgibbs). Stylistically distinct -> individually separable;
        a cautionary "too-easy" set that certifies nothing.
    condition="hard": positives = attack windows CONTAINING the payload; negatives =
        benign PREFIXES of a DISJOINT group split (same style, individually benign;
        only the escalation differs). This is the set that carries the claim.

    Returns dict: conversations (List[List[str]] of user turns), labels (1=attack,
    0=benign), groups (int ids), group_keys (the raw strings), categories, sources,
    plus `meta` -- the ACHIEVED counts, distinct-group count, pool ceiling and content
    fingerprint that results.json stamps (section 18.8).
    """
    n_pos = int(C.N_POS if n_pos is None else n_pos)
    n_neg = int(C.N_NEG if n_neg is None else n_neg)
    min_turns = int(C.MIN_USER_TURNS if min_turns is None else min_turns)
    max_turns = int(C.MAX_USER_TURNS if max_turns is None else max_turns)
    seed = int(C.SEED if seed is None else seed)
    rng = random.Random(seed)

    if condition == "hard":
        conversations, labels, group_keys, categories, sources, configs = \
            _build_hard(min(n_pos, n_neg), min_turns, max_turns, rng)
        neg_source = "attack_prefix(disjoint group half)"
        scanned = 0
    else:
        pos_convs, pos_keys, pos_cats, pos_cfgs = _load_positives(
            n_pos, min_turns, max_turns, rng)
        pos_tc = _turncount_dist(pos_convs)
        if C.NEG_SOURCE == "tomgibbs":
            neg_convs, neg_cats, scanned = _load_negatives_tomgibbs(
                n_neg, min_turns, max_turns, rng)
            neg_source = "tomgibbs:%s" % ",".join(C.TOMGIBBS_NEG_CONFIGS)
        else:
            neg_convs, neg_cats, scanned = _load_negatives(
                n_neg, min_turns, max_turns, pos_tc, rng)
            neg_source = "ultrachat"
        conversations = list(pos_convs) + list(neg_convs)
        labels = [1] * len(pos_convs) + [0] * len(neg_convs)
        # Each benign conversation is its own group (disjoint from attack goal keys).
        group_keys = list(pos_keys) + ["benign::%d" % i for i in range(len(neg_convs))]
        categories = list(pos_cats) + list(neg_cats)
        sources = ["attack"] * len(pos_convs) + [neg_source] * len(neg_convs)
        configs = list(pos_cfgs) + ["-"] * len(neg_convs)

    # Fixed-seed shuffle (keep every parallel list aligned).
    idx = list(range(len(conversations)))
    rng.shuffle(idx)
    conversations = [conversations[i] for i in idx]
    labels = [labels[i] for i in idx]
    group_keys = [group_keys[i] for i in idx]
    categories = [categories[i] for i in idx]
    sources = [sources[i] for i in idx]
    configs = [configs[i] for i in idx]
    groups, n_groups = _group_ids(group_keys)

    n_p = sum(labels)
    n_n = len(labels) - n_p
    pos_tc = _turncount_dist([c for c, y in zip(conversations, labels) if y == 1])
    neg_tc = _turncount_dist([c for c, y in zip(conversations, labels) if y == 0])
    pos_cat = Counter(c for c, y in zip(categories, labels) if y == 1)
    neg_cat = Counter(c for c, y in zip(categories, labels) if y == 0)

    meta = {
        "condition": condition,
        "n": len(labels),
        "n_pos_achieved": int(n_p),
        "n_neg_achieved": int(n_n),
        "n_pos_requested": int(n_pos),
        "n_neg_requested": int(n_neg),
        # n and n_distinct_groups are reported SEPARATELY on purpose: SafeMTData_1K
        # has several attack paths per goal, so raising n without raising the group
        # count inflates rows, not information (the COUGHVID mistake).
        "n_distinct_groups": int(n_groups),
        "neg_source": neg_source,
        "attack_configs": list(C.ATTACK_CONFIGS),
        "config_mix": dict(Counter(configs)),
        "turncount_pos": {str(k): int(v) for k, v in sorted(pos_tc.items())},
        "turncount_neg": {str(k): int(v) for k, v in sorted(neg_tc.items())},
        "category_pos": {str(k): int(v) for k, v in sorted(pos_cat.items())},
        "category_neg": {str(k): int(v) for k, v in sorted(neg_cat.items())},
        "benign_rows_scanned": int(scanned),
        "fingerprint_sha256": dataset_fingerprint(conversations, labels),
        "meets_rule1": bool(min(n_p, n_n) >= C.RULE1_FLOOR),
        "rule1_floor": int(C.RULE1_FLOOR),
    }
    if condition == "hard":
        meta["pool_ceiling"] = hard_pool_ceiling(min_turns, max_turns)

    _eprint("[data] %s: N=%d pos=%d neg=%d distinct_groups=%d (requested %d/%d)"
            % (condition.upper(), len(labels), n_p, n_n, n_groups, n_pos, n_neg))
    _eprint("[data] turn-count dist (pos): "
            + ", ".join("%d->%d" % (k, pos_tc[k]) for k in sorted(pos_tc)))
    _eprint("[data] turn-count dist (neg): "
            + ", ".join("%d->%d" % (k, neg_tc[k]) for k in sorted(neg_tc)))
    _eprint("[data] category dist (pos): "
            + ", ".join("%s=%d" % (k, pos_cat[k]) for k in sorted(pos_cat)))
    _eprint("[data] category dist (neg): "
            + ", ".join("%s=%d" % (k, neg_cat[k]) for k in sorted(neg_cat)))
    if not meta["meets_rule1"]:
        ceil = meta.get("pool_ceiling")
        _eprint("[data] WARNING rule 1 (>=%d/class) NOT met: %d/%d." % (C.RULE1_FLOOR, n_p, n_n)
                + (" POOL CEILING pos~%d neg~%d over %d groups -- state this in the README."
                   % (ceil["pos_ceiling_approx"], ceil["neg_ceiling_approx"],
                      ceil["distinct_groups"]) if ceil else ""))
    _eprint("[data] fingerprint sha256=%s" % meta["fingerprint_sha256"][:16])

    return {
        "conversations": conversations,
        "labels": labels,
        "groups": groups,
        "group_keys": group_keys,
        "categories": categories,
        "sources": sources,
        "meta": meta,
    }


# --- OOD: intrinsec-ai/cstm-bench (rule 5) -----------------------------------
def load_ood(splits=None, window=None, min_turns=None) -> dict:
    """Zero-shot OOD set from `intrinsec-ai/cstm-bench` (ungated, MIT, cached).

    WHY THIS BENCHMARK AND NOT `ScaleAI/mhj`: CLAUDE.md rule 5 names mhj, but mhj is
    GATED and this host has no HF token -- its local hub dir holds only a 40-byte
    `refs/main`, so a fetch 401s. cstm-bench is the multi-turn/multi-SESSION substitute
    and ships purpose-built `benign_hard` scenarios (approval-fatigue and
    tacit-collusion confounders), which is the harder negative of the two.

    Shape: each scenario carries 20-26 sessions of messages. The trained models see
    HARD_WINDOW-turn windows, so a scenario becomes the LAST `window` message texts --
    applied IDENTICALLY to attack and benign scenarios, so the windowing itself cannot
    separate the classes. It IS a distribution shift in every other respect (agentic
    enterprise sessions vs ActorAttack question chains), which is the point of an OOD
    arm; expect degradation and report it as prominently as any win.
    """
    from datasets import load_dataset as hf_load

    splits = list(splits or C.OOD_SPLITS)
    W = int(window or C.OOD_WINDOW)
    min_turns = int(C.MIN_USER_TURNS if min_turns is None else min_turns)

    convs, labels, group_keys, cats, sources = [], [], [], [], []
    for split in splits:
        ds = hf_load(C.OOD_DATASET, split=split)
        for row in ds:
            cls = str(row.get("scenario_class", ""))
            if cls in C.OOD_POS_CLASSES:
                lab = 1
            elif cls in C.OOD_NEG_CLASSES:
                lab = 0
            else:
                continue
            try:
                sessions = json.loads(row.get("sessions_json") or "[]")
            except Exception:
                continue
            texts = []
            for sess in sessions:
                for msg in sess.get("messages") or []:
                    t = _clean_turn(msg.get("text") or "")
                    if t:
                        texts.append(t)
            if len(texts) < min(min_turns, W):
                continue
            convs.append(texts[-W:])
            labels.append(lab)
            group_keys.append(str(row.get("scenario_id") or len(group_keys)))
            cats.append(str(row.get("taxonomy") or "unknown"))
            sources.append("cstm-bench/%s/%s" % (split, cls))

    groups, n_groups = _group_ids(group_keys)
    n_p = sum(labels)
    n_n = len(labels) - n_p
    meta = {
        "dataset": C.OOD_DATASET,
        "splits": splits,
        "window": W,
        "n": len(labels),
        "n_pos_achieved": int(n_p),
        "n_neg_achieved": int(n_n),
        "n_distinct_groups": int(n_groups),
        "class_mix": dict(Counter(sources)),
        "fingerprint_sha256": dataset_fingerprint(convs, labels),
        # A 108-scenario benchmark cannot meet rule 1 and is not meant to: rule 5's
        # instruction is to build the >=500/class MAIN set from other data and report
        # the real released benchmark as OOD. Stated, not silently omitted.
        "rule1_exempt_reason": (
            "released benchmark used as OOD (rule 5); the >=500/class MAIN set is "
            "built from SafeMTData + UltraChat, not from here"),
        "mhj_status": (
            "ScaleAI/mhj -- named by CLAUDE.md rule 5 -- is GATED; this host has no HF "
            "token (local hub dir holds only a 40-byte refs/main), so it is unusable "
            "and cstm-bench is the substitute."),
    }
    _eprint("[data] OOD cstm-bench: N=%d pos=%d neg=%d scenarios=%d window=%d"
            % (len(labels), n_p, n_n, n_groups, W))
    return {"conversations": convs, "labels": labels, "groups": groups,
            "group_keys": group_keys, "categories": cats, "sources": sources,
            "meta": meta}


# --- confound audit (THE SHARED SPINE) ---------------------------------------
def confound_audit(conversations, labels, seed=None, n_folds=None) -> dict:
    """Run `common.confound.confound_report` -- the ONE shared confound instrument.

    The lesson's own `length_confound_report` used to live here. It folded correctly
    (unlike two sibling lessons) but had only two bars: turn count and total chars.
    It was missing the two that matter most here:

      * the CONTENT bar (TF-IDF). This lesson's negatives are keyword-TOPIC-MATCHED
        benign chat, so a unigram model is the honest test of whether a method reads
        ESCALATION or merely VOCABULARY. A "trajectory" detector that cannot beat
        unigrams is not reading trajectories.
      * the SHUFFLE control, which catches leakage that no length bar can see.

    `turncount_auc` / `totalchar_auc` are kept for continuity with the older
    results.json, but they are DERIVED views. The authoritative bar a method must
    clear is `worst_auc` / `worst_name` -- the largest directionless bar of all three.
    """
    rep = _shared_confound_report(
        conversations, labels, units=conversations,
        seed=int(C.SEED if seed is None else seed),
        n_folds=int(C.N_FOLDS if n_folds is None else n_folds),
        run_content=True, run_shuffle=True)
    # Legacy keys (RAW, directional -- exactly what the old artifact stored) so a
    # reader can line the new report up against the pre-2026-08 numbers.
    rep["turncount_auc"] = float(rep.get("count", {}).get("auc_raw", 0.5))
    rep["totalchar_auc"] = float(rep.get("length", {}).get("auc_raw", 0.5))
    rep["turncount_pos_mean"] = float(rep.get("count", {}).get("mean_pos", 0.0))
    rep["turncount_neg_mean"] = float(rep.get("count", {}).get("mean_neg", 0.0))
    rep["legacy_keys_note"] = (
        "turncount_auc/totalchar_auc are RAW and DIRECTIONAL, kept only for continuity "
        "with the pre-2026-08 artifact. The BINDING bar is worst_auc (directionless, "
        "max over length/count/content).")
    return rep


# Back-compat shim: the old name, now delegating to the shared spine. Kept so an
# older notebook/script does not silently take a different code path.
def length_confound_report(conversations, labels) -> dict:
    """DEPRECATED -- use `confound_audit`. Delegates to the shared spine."""
    return confound_audit(conversations, labels)


if __name__ == "__main__":
    # CPU smoke: small load (streaming benign), print counts + the FULL confound
    # report (four bars, not two). This is the ONLY place that hits the network.
    _eprint("[smoke] loading n_pos=40, n_neg=40 ...")
    data = load_dataset(n_pos=40, n_neg=40)
    labels = data["labels"]
    print("SMOKE COUNTS: N=%d pos=%d neg=%d groups=%d"
          % (len(labels), sum(labels), len(labels) - sum(labels),
             data["meta"]["n_distinct_groups"]))
    print("sources:", dict(Counter(data["sources"])))
    print("categories:", dict(Counter(data["categories"])))
    print(format_report(confound_audit(data["conversations"], labels)))
    print("")
    print("HARD pool ceiling:", json.dumps(hard_pool_ceiling(), indent=2))
