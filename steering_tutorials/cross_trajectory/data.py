"""data.py — dataset loader for the CROSS-TRAJECTORY latent-aggregation lesson.

A **sample** = an unordered SET of K trajectory strings (one per agent/session).
A **decomposed attack** (label 1) fractures a harmful goal across K trajectories
so no single one carries the payload; a **benign** sample (label 0) is K
trajectories that do NOT aggregate into a harmful goal.

POSITIVES  SafeMTData/SafeMTData 'Attack_600' -- each row's `multi_turn_queries`
           (~5 ActorAttack sub-queries). ONE decomposed-attack sample = that
           attack's LAST K sub-queries (the payload lands last -> included) as K
           SEPARATE trajectories. label 1, group=query_id, source='attack'.
NEGATIVES  depend on `condition`:
  "easy"   K benign USER turns from ONE HuggingFaceH4/ultrachat_200k conversation
           (streaming). group=10_000_000+i, source='ultrachat'.
  "hard"   K same-STYLE ActorAttack trajectories that DO NOT complete a harmful
           goal = a DIFFERENT attack's FIRST (k-1) sub-queries (benign lead-up,
           payload excluded) + 1 filler lead-up sub-query. Positive/negative
           query_ids are DISJOINT halves -> leakage-free; both classes are EXACTLY
           K trajectories -> length/count-matched. group=query_id, source=
           'attack_prefix'. The ONLY systematic difference is whether the K
           trajectories AGGREGATE into a complete harmful goal -- invisible to a
           per-trajectory monitor.
OOD        intrinsec-ai/cstm-bench (both splits) -- real cross-session scenarios.

CPU-only. Downloads datasets. Loads NO LLM. Stdout is ASCII-only (Windows cp1252).
"""
from __future__ import annotations

try:  # SSL middlebox on this host (truststore lets requests use the OS trust store)
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import sys
import json
import random
from collections import Counter

from steering_tutorials.cross_trajectory import config as C


# --- helpers -----------------------------------------------------------------
def _eprint(*args) -> None:
    """ASCII-only stderr print."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


def _clean(text) -> str:
    return " ".join(str(text).split()).strip()


# --- shared attack loader ----------------------------------------------------
def _all_attacks():
    """Every Attack_600 row -> (sub_queries[List[str]], query_id, category)."""
    from datasets import load_dataset as hf_load

    ds = hf_load(C.ATTACK_DATASET, C.ATTACK_CONFIG)[C.ATTACK_CONFIG]
    rows = []
    for i, row in enumerate(ds):
        subs = [_clean(t) for t in (row.get("multi_turn_queries") or []) if _clean(t)]
        rows.append((subs, int(row.get("query_id", i)), str(row.get("category", ""))))
    return rows


# --- positives (standard decomposition: an attack's LAST k sub-queries) -------
def _build_positives(attacks, n_pos, k, rng):
    """One sample per attack with >=k sub-queries = its LAST k sub-queries (the
    payload lands last, so it is included). label 1, group=query_id."""
    pool = [(subs[-k:], qid, cat) for subs, qid, cat in attacks if len(subs) >= k]
    rng.shuffle(pool)
    return pool[:n_pos]


# --- easy negatives (benign UltraChat conversations) -------------------------
def _extract_user_turns(row, k):
    msgs = row.get("messages") or []
    turns = [_clean(m.get("content", "")) for m in msgs
             if m.get("role") == "user" and _clean(m.get("content", ""))]
    return turns[:k] if len(turns) >= k else None


def _build_easy_negatives(n_neg, k, rng):
    """K benign user turns from ONE UltraChat conversation, per sample."""
    from datasets import load_dataset as hf_load

    samples = []
    max_scan = max(20000, n_neg * 200)
    scanned = 0
    stream = hf_load(C.BENIGN_DATASET, split=C.BENIGN_SPLIT, streaming=True)
    for row in stream:
        scanned += 1
        if scanned > max_scan:
            break
        turns = _extract_user_turns(row, k)
        if turns is None:
            continue
        samples.append(turns)
        if len(samples) >= n_neg:
            break
    return samples, scanned


# --- hard negatives (benign ActorAttack lead-up, DISJOINT split) -------------
def _build_hard(attacks, n_per_class, k, rng):
    """Length-matched, leakage-free hard set (both classes EXACTLY k trajectories):
      positive = an attack's LAST k sub-queries (payload included);
      negative = a DIFFERENT attack's FIRST (k-1) sub-queries (payload excluded)
                 + 1 filler lead-up sub-query (another attack's first sub-query).
    Positive/negative query_ids are disjoint halves -> no leakage. A per-trajectory
    monitor cannot see the missing payload; only a set-aggregator can.
    """
    usable = [(subs, qid, cat) for subs, qid, cat in attacks if len(subs) >= k]
    rng.shuffle(usable)
    half = len(usable) // 2
    pos_rows, neg_rows = usable[:half], usable[half:]

    # Filler pool: FIRST (benign lead-up) sub-query of each negative-side attack.
    filler_pool = [(subs[0], qid) for subs, qid, cat in neg_rows]

    positives = [(subs[-k:], qid, cat) for subs, qid, cat in pos_rows]

    negatives = []
    for subs, qid, cat in neg_rows:
        lead = subs[:max(0, k - 1)]              # first (k-1): excludes the payload
        # a filler from a DIFFERENT negative-side attack
        filler = None
        for _ in range(8):
            ftext, fqid = rng.choice(filler_pool)
            if fqid != qid:
                filler = ftext
                break
        if filler is None:
            filler = rng.choice(filler_pool)[0]
        trajs = lead + [filler]
        if len(trajs) == k:                      # only keep exact-K, leakage-free
            negatives.append((trajs, qid, cat))

    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives = positives[:n_per_class]
    negatives = negatives[:n_per_class]
    return positives, negatives


# --- public API --------------------------------------------------------------
def load_dataset(n_pos=C.N_POS, n_neg=C.N_NEG, k=C.K_TRAJ, condition="easy",
                 seed=C.SEED) -> dict:
    """Build the balanced cross-trajectory dataset.

    Returns dict: {"samples": List[List[str]] (each K trajectory strings),
                   "labels": List[int] (1=decomposed attack, 0=benign),
                   "groups": List[int], "sources": List[str]}.
    """
    rng = random.Random(seed)
    attacks = _all_attacks()

    if condition == "hard":
        n_per = min(n_pos, n_neg)
        positives, negatives = _build_hard(attacks, n_per, k, rng)
        samples = [t for t, _, _ in positives] + [t for t, _, _ in negatives]
        labels = [1] * len(positives) + [0] * len(negatives)
        groups = [q for _, q, _ in positives] + [q for _, q, _ in negatives]
        sources = ["attack"] * len(positives) + ["attack_prefix"] * len(negatives)
    else:  # "easy"
        positives = _build_positives(attacks, n_pos, k, rng)
        neg_turns, scanned = _build_easy_negatives(n_neg, k, rng)
        samples = [t for t, _, _ in positives] + list(neg_turns)
        labels = [1] * len(positives) + [0] * len(neg_turns)
        groups = ([q for _, q, _ in positives]
                  + [10_000_000 + i for i in range(len(neg_turns))])
        sources = ["attack"] * len(positives) + ["ultrachat"] * len(neg_turns)
        _eprint("[data] easy: scanned %d benign rows" % scanned)

    # Fixed-seed shuffle (keep the four lists aligned).
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    samples = [samples[i] for i in idx]
    labels = [labels[i] for i in idx]
    groups = [groups[i] for i in idx]
    sources = [sources[i] for i in idx]

    n_p = sum(labels)
    n_n = len(labels) - n_p
    _eprint("[data] condition=%s  N=%d  pos(decomp attack)=%d  neg(benign)=%d  K=%d"
            % (condition, len(labels), n_p, n_n, k))
    _eprint("[data] source dist: "
            + ", ".join("%s=%d" % (s, c) for s, c in sorted(Counter(sources).items())))
    kdist = Counter(len(s) for s in samples)
    _eprint("[data] trajectory-count dist: "
            + ", ".join("%d->%d" % (kk, kdist[kk]) for kk in sorted(kdist)))
    return {"samples": samples, "labels": labels, "groups": groups, "sources": sources}


# --- OOD: CSTM-Bench (real cross-session scenarios) --------------------------
def _session_text(session) -> str:
    """Concatenate a session dict's textual string fields into ONE trajectory."""
    if isinstance(session, str):
        return _clean(session)
    if not isinstance(session, dict):
        return _clean(session)
    parts = []
    anchor = session.get("identity_anchor")
    if isinstance(anchor, str) and anchor.strip():
        parts.append(_clean(anchor))
    for key in ("messages", "content", "text", "turns", "conversation"):
        val = session.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(_clean(val))
        elif isinstance(val, list):
            for m in val:
                if isinstance(m, str) and m.strip():
                    parts.append(_clean(m))
                elif isinstance(m, dict):
                    for mk in ("content", "text", "message"):
                        mv = m.get(mk)
                        if isinstance(mv, str) and mv.strip():
                            parts.append(_clean(mv))
    if not parts:  # fallback: any string value in the dict
        for v in session.values():
            if isinstance(v, str) and v.strip():
                parts.append(_clean(v))
    return " ".join(parts).strip()


def load_ood_cstm(k=C.K_TRAJ, seed=C.SEED) -> dict:
    """Load intrinsec-ai/cstm-bench (splits 'dilution' + 'cross_session'). ONE
    sample per scenario = up to K session texts (the K most-content-bearing
    sessions). label = 1 if scenario_class=='attack' else 0. source='cstm'.
    """
    from datasets import load_dataset as hf_load

    rng = random.Random(seed)
    samples, labels, groups, sources = [], [], [], []
    gid = 0
    for split in ("dilution", "cross_session"):
        try:
            ds = hf_load(C.OOD_DATASET, "default", split=split)
        except Exception as e:
            _eprint("[ood] split %s load failed: %s" % (split, e))
            continue
        for row in ds:
            raw = row.get("sessions_json")
            try:
                sessions = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except Exception:
                sessions = []
            texts = [_session_text(s) for s in sessions]
            texts = [t for t in texts if t]
            if not texts:
                continue
            # take the K most-content-bearing sessions (by char length)
            texts = sorted(texts, key=len, reverse=True)[:k]
            scls = str(row.get("scenario_class", ""))
            samples.append(texts)
            labels.append(1 if scls == "attack" else 0)
            groups.append(gid)
            sources.append("cstm")
            gid += 1

    n_att = sum(labels)
    n_ben = len(labels) - n_att
    _eprint("[ood] CSTM-Bench: N=%d  attack=%d  benign=%d" % (len(labels), n_att, n_ben))
    return {"samples": samples, "labels": labels, "groups": groups, "sources": sources}


# --- confound audit ----------------------------------------------------------
def _auc(scores, labels) -> float:
    """Threshold-free AUC = P(score(pos) > score(neg)); sklearn or manual rank-AUC."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        ranks = [0.0] * len(scores)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for m in range(i, j + 1):
                ranks[order[m]] = avg
            i = j + 1
        sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
        n_p, n_n = len(pos), len(neg)
        u = sum_pos - n_p * (n_p + 1) / 2.0
        return float(u / (n_p * n_n))


def confound_report(samples, labels) -> dict:
    """Can #trajectories or total text length predict the label? ~0.5 => no tell."""
    kcounts = [len(s) for s in samples]
    totalchars = [sum(len(t) for t in s) for s in samples]
    pos_k = [n for n, y in zip(kcounts, labels) if y == 1]
    neg_k = [n for n, y in zip(kcounts, labels) if y == 0]
    return {
        "kcount_auc": _auc(kcounts, labels),
        "totalchar_auc": _auc(totalchars, labels),
        "k_pos_mean": float(sum(pos_k) / len(pos_k)) if pos_k else 0.0,
        "k_neg_mean": float(sum(neg_k) / len(neg_k)) if neg_k else 0.0,
    }


if __name__ == "__main__":
    # SMALL CPU smoke (streaming benign, no model). ASCII stdout.
    _eprint("[smoke] load_dataset(n_pos=10, n_neg=10, k=5, condition='hard') ...")
    data = load_dataset(n_pos=10, n_neg=10, k=5, condition="hard")
    labels = data["labels"]
    print("MAIN(hard) COUNTS: N=%d pos=%d neg=%d"
          % (len(labels), sum(labels), len(labels) - sum(labels)))
    print("sources:", dict(Counter(data["sources"])))
    if data["samples"]:
        ex = data["samples"][0]
        print("sample[0] label=%d  K=%d trajectories:" % (labels[0], len(ex)))
        for i, t in enumerate(ex):
            print("  traj[%d]: %s" % (i, (t[:90] + "...") if len(t) > 90 else t))
    conf = confound_report(data["samples"], labels)
    print("CONFOUND REPORT (hard):")
    for key in ("kcount_auc", "totalchar_auc", "k_pos_mean", "k_neg_mean"):
        print("  %-16s %.4f" % (key, conf[key]))

    _eprint("[smoke] load_ood_cstm(k=5) ...")
    ood = load_ood_cstm(k=5)
    olab = ood["labels"]
    print("CSTM COUNTS: N=%d attack=%d benign=%d"
          % (len(olab), sum(olab), len(olab) - sum(olab)))
    if ood["samples"]:
        oex = ood["samples"][0]
        print("cstm sample[0] label=%d K=%d  first-traj: %s"
              % (olab[0], len(oex), (oex[0][:90] + "...") if len(oex[0]) > 90 else oex[0]))
