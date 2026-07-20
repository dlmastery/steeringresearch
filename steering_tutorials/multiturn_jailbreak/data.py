"""data.py — dataset loader for the multi-turn jailbreak DETECTION lesson.

POSITIVES  SafeMTData/SafeMTData 'Attack_600' — each row's `multi_turn_queries`
           (4-5 escalating USER turns) IS one conversation (label 1).
NEGATIVES  HuggingFaceH4/ultrachat_200k (streaming) — USER turns of real benign
           multi-turn chats (label 0), TOPIC-MATCHED to the attack categories via
           C.CATEGORY_KEYWORDS (hard negatives sharing surface) and TURN-COUNT
           biased toward the positive distribution.

CPU-only. Downloads datasets. Loads NO LLM. Stdout is ASCII-only (Windows cp1252).
"""
from __future__ import annotations

try:  # SSL middlebox on this host (truststore lets requests use the OS trust store)
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import sys
import random
from collections import Counter

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


# --- positives ---------------------------------------------------------------
def _load_positives(n_pos: int, min_turns: int, max_turns: int):
    from datasets import load_dataset as hf_load

    ds = hf_load(C.ATTACK_DATASET, C.ATTACK_CONFIG)[C.ATTACK_CONFIG]
    convs, groups, cats = [], [], []
    for row in ds:
        turns_raw = row.get("multi_turn_queries") or []
        turns = [_clean_turn(t) for t in turns_raw if _clean_turn(t)]
        turns = turns[:max_turns]
        if len(turns) < min_turns:
            continue
        convs.append(turns)
        groups.append(int(row.get("query_id", len(groups))))
        cats.append(str(row.get("category", "general")))
        if len(convs) >= n_pos:
            break
    return convs, groups, cats


# --- negatives ---------------------------------------------------------------
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
        # Turn-count biasing: if this count is over its quota AND we already have a
        # healthy pool, skip to steer the distribution (report mismatch, don't crash).
        if quota and accepted_by_tc.get(tc, 0) >= quota.get(tc, 0) and \
                (len(matched) + len(general)) >= n_neg:
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


# --- HARD condition: benign-prefix negatives from a DISJOINT attack split -----
def _load_all_attacks(min_turns: int, max_turns: int):
    """Every Attack_600 conversation with its query_id + category (no n cap)."""
    from datasets import load_dataset as hf_load

    ds = hf_load(C.ATTACK_DATASET, C.ATTACK_CONFIG)[C.ATTACK_CONFIG]
    rows = []
    for row in ds:
        turns = [_clean_turn(t) for t in (row.get("multi_turn_queries") or [])
                 if _clean_turn(t)][:max_turns]
        if len(turns) < min_turns:
            continue
        rows.append((turns, int(row.get("query_id", len(rows))),
                     str(row.get("category", "general"))))
    return rows


def _build_hard(n_per_class, min_turns, max_turns, rng, window=None):
    """LENGTH-MATCHED hard set. Both classes are EXACTLY `window` turns:
      positive = the LAST `window` turns of an attack (contains the escalation/payload
                 that lands at the final turn);
      negative = the FIRST `window` turns of a DIFFERENT attack (the benign lead-up,
                 which EXCLUDES the payload because the payload is the last turn).
    So the ONLY systematic difference is whether the window contains the escalation --
    NOT its length (turn-count is constant -> no length confound) and NOT the source
    conversation (positive/negative query_ids are disjoint -> leakage-free). A per-turn
    probe cannot separate them (each turn is an innocuous ActorAttack question); only a
    sequence model that reads the trajectory can. Negatives need a source attack with
    > window turns (so first-`window` truly excludes the last/payload turn).
    """
    W = int(window or getattr(C, "HARD_WINDOW", 4))
    rows = _load_all_attacks(min_turns, max_turns)
    by_qid = {}
    for turns, qid, cat in rows:
        by_qid.setdefault(qid, []).append((turns, cat))
    qids = list(by_qid)
    rng.shuffle(qids)
    half = len(qids) // 2
    pos_qids, neg_qids = set(qids[:half]), set(qids[half:])

    pos, negs = [], []
    for qid in pos_qids:
        for turns, cat in by_qid[qid]:
            if len(turns) >= W:
                pos.append((turns[-W:], qid, cat))         # last W: includes payload
    for qid in neg_qids:
        for turns, cat in by_qid[qid]:
            if len(turns) > W:                              # need > W so first-W drops payload
                negs.append((turns[:W], qid, cat))          # first W: benign lead-up
    rng.shuffle(pos)
    rng.shuffle(negs)
    pos = pos[:n_per_class]
    negs = negs[:n_per_class]

    conversations = [t for t, _, _ in pos] + [t for t, _, _ in negs]
    labels = [1] * len(pos) + [0] * len(negs)
    groups = [q for _, q, _ in pos] + [q for _, q, _ in negs]
    categories = [c for _, _, c in pos] + [c for _, _, c in negs]
    sources = ["attack_full"] * len(pos) + ["attack_prefix"] * len(negs)
    return conversations, labels, groups, categories, sources, pos, negs


# --- public API --------------------------------------------------------------
def load_dataset(n_pos=C.N_POS, n_neg=C.N_NEG, min_turns=C.MIN_USER_TURNS,
                 max_turns=C.MAX_USER_TURNS, seed=C.SEED, condition="easy") -> dict:
    """Load the balanced multi-turn detection dataset.

    condition="easy": positives = ActorAttack attacks; negatives = topic-matched
        benign UltraChat conversations (stylistically distinct -> individually
        separable; a cautionary "too-easy" set).
    condition="hard": positives = FULL attacks; negatives = benign PREFIXES of a
        DISJOINT attack split (same style, individually benign; only the final
        escalation turn differs). This is the set that actually needs a sequence
        model -- see README.

    Returns dict: conversations (List[List[str]] of user turns), labels (1=attack,
    0=benign), groups, categories, sources. All lists same length N.
    """
    rng = random.Random(seed)

    if condition == "hard":
        conversations, labels, groups, categories, sources, pos_convs, neg_raw = \
            _build_hard(min(n_pos, n_neg), min_turns, max_turns, rng)
        pos_only = [c for c, l in zip(conversations, labels) if l == 1]
        neg_only = [c for c, l in zip(conversations, labels) if l == 0]
        pos_tc = _turncount_dist(pos_only)
        neg_tc_dist = _turncount_dist(neg_only)
        idx = list(range(len(conversations)))
        rng.shuffle(idx)
        conversations = [conversations[i] for i in idx]
        labels = [labels[i] for i in idx]
        groups = [groups[i] for i in idx]
        categories = [categories[i] for i in idx]
        sources = [sources[i] for i in idx]
        n_p, n_n = sum(labels), len(labels) - sum(labels)
        _eprint("[data] HARD condition: N=%d  pos(full attack)=%d  neg(benign prefix)=%d"
                % (len(labels), n_p, n_n))
        _eprint("[data] turn-count dist (pos): "
                + ", ".join("%d->%d" % (k, pos_tc[k]) for k in sorted(pos_tc)))
        _eprint("[data] turn-count dist (neg): "
                + ", ".join("%d->%d" % (k, neg_tc_dist[k]) for k in sorted(neg_tc_dist)))
        _eprint("[data] NOTE: length-matched (both classes = HARD_WINDOW=%d turns); "
                "positive=last-W (has payload), negative=first-W of a different attack "
                "(benign lead-up). per_turn_max is the honest per-turn control."
                % int(getattr(C, "HARD_WINDOW", 4)))
        return {"conversations": conversations, "labels": labels, "groups": groups,
                "categories": categories, "sources": sources}

    pos_convs, pos_groups, pos_cats = _load_positives(n_pos, min_turns, max_turns)
    pos_tc = _turncount_dist(pos_convs)

    neg_convs, neg_cats, scanned = _load_negatives(
        n_neg, min_turns, max_turns, pos_tc, rng)

    # Assemble.
    conversations = list(pos_convs) + list(neg_convs)
    labels = [1] * len(pos_convs) + [0] * len(neg_convs)
    groups = list(pos_groups) + [10_000_000 + i for i in range(len(neg_convs))]
    categories = list(pos_cats) + list(neg_cats)
    sources = ["attack"] * len(pos_convs) + ["ultrachat"] * len(neg_convs)

    # Fixed-seed shuffle (keep the five lists aligned).
    idx = list(range(len(conversations)))
    rng.shuffle(idx)
    conversations = [conversations[i] for i in idx]
    labels = [labels[i] for i in idx]
    groups = [groups[i] for i in idx]
    categories = [categories[i] for i in idx]
    sources = [sources[i] for i in idx]

    # --- report to stderr (ASCII only) ---
    n_p, n_n = sum(labels), len(labels) - sum(labels)
    _eprint("[data] loaded N=%d  pos(attack)=%d  neg(benign)=%d  (scanned %d benign rows)"
            % (len(labels), n_p, n_n, scanned))
    _eprint("[data] turn-count dist (pos): "
            + ", ".join("%d->%d" % (k, pos_tc[k]) for k in sorted(pos_tc)))
    neg_tc = _turncount_dist(neg_convs)
    _eprint("[data] turn-count dist (neg): "
            + ", ".join("%d->%d" % (k, neg_tc[k]) for k in sorted(neg_tc)))
    # Turn-count mismatch (report, do not crash).
    all_tc = sorted(set(pos_tc) | set(neg_tc))
    p_tot = sum(pos_tc.values()) or 1
    n_tot = sum(neg_tc.values()) or 1
    l1 = sum(abs(pos_tc.get(k, 0) / p_tot - neg_tc.get(k, 0) / n_tot) for k in all_tc)
    _eprint("[data] turn-count distribution L1 mismatch (pos vs neg): %.3f" % l1)
    pcat = Counter(pos_cats)
    ncat = Counter(neg_cats)
    _eprint("[data] category dist (pos): "
            + ", ".join("%s=%d" % (k, pcat[k]) for k in sorted(pcat)))
    _eprint("[data] category dist (neg): "
            + ", ".join("%s=%d" % (k, ncat[k]) for k in sorted(ncat)))
    n_matched = sum(1 for c in neg_cats if c != "general")
    _eprint("[data] benign topic-matched (hard) negatives: %d/%d (%.0f%%)"
            % (n_matched, len(neg_cats), 100.0 * n_matched / max(1, len(neg_cats))))

    return {
        "conversations": conversations,
        "labels": labels,
        "groups": groups,
        "categories": categories,
        "sources": sources,
    }


# --- confound audit ----------------------------------------------------------
def _auc(scores, labels) -> float:
    """Threshold-free AUC = P(score(pos) > score(neg)), ties count 0.5.

    Uses sklearn.metrics.roc_auc_score if available, else a manual rank-AUC.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        # Manual rank-AUC (Mann-Whitney U / (n_pos*n_neg)).
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        ranks = [0.0] * len(scores)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        sum_pos_ranks = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
        n_p, n_n = len(pos), len(neg)
        u = sum_pos_ranks - n_p * (n_p + 1) / 2.0
        return float(u / (n_p * n_n))


def length_confound_report(conversations, labels) -> dict:
    """Can a trivial signal (turn count / total char length) predict the label?

    ~0.5 => no confound; >>0.5 => the benign set is length/turn-separable (caveat).
    """
    turncounts = [len(c) for c in conversations]
    totalchars = [sum(len(t) for t in c) for c in conversations]
    pos_tc = [t for t, y in zip(turncounts, labels) if y == 1]
    neg_tc = [t for t, y in zip(turncounts, labels) if y == 0]
    return {
        "turncount_auc": _auc(turncounts, labels),
        "totalchar_auc": _auc(totalchars, labels),
        "turncount_pos_mean": float(sum(pos_tc) / len(pos_tc)) if pos_tc else 0.0,
        "turncount_neg_mean": float(sum(neg_tc) / len(neg_tc)) if neg_tc else 0.0,
    }


if __name__ == "__main__":
    # CPU smoke: small load (streaming benign), print counts + confound report.
    # This is the ONLY place that hits the network.
    _eprint("[smoke] loading n_pos=20, n_neg=20 ...")
    data = load_dataset(n_pos=20, n_neg=20)
    labels = data["labels"]
    print("SMOKE COUNTS: N=%d pos=%d neg=%d"
          % (len(labels), sum(labels), len(labels) - sum(labels)))
    print("sources:", dict(Counter(data["sources"])))
    print("categories:", dict(Counter(data["categories"])))
    conf = length_confound_report(data["conversations"], labels)
    print("CONFOUND REPORT:")
    for k in ("turncount_auc", "totalchar_auc",
              "turncount_pos_mean", "turncount_neg_mean"):
        print("  %-20s %.4f" % (k, conf[k]))
