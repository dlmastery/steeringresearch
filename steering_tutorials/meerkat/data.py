"""data.py — trace-pool loader for the MEERKAT trace-clustering lesson.

WHAT A "TRACE" IS HERE
----------------------
A **trace** is one agent's whole event sequence rendered to a SINGLE string. In
`cross_trajectory` a sample was a SET of K separate strings; here we collapse the
entire episode into ONE flat string, because Meerkat embeds each trace as a single
vector and then CLUSTERS the vectors of many traces in a repository. So the unit of
analysis is "one trace = one embeddable document".

WHY VIOLATIONS ARE SPARSE (~5%)
-------------------------------
Meerkat's thesis (arXiv:2604.11806): in a real agent deployment the harmful traces
are a small, distributed MINORITY hiding among mostly-benign traffic. A per-trace
monitor that scores each trace alone is diluted into noise; only when MANY traces
are analysed TOGETHER does the campaign's shared structure become visible. So we
build a big POOL (>=500/class) and then `sample_repository` draws repositories at a
sparse `base_rate` (~5%) to reproduce that regime.

WHY WE JOIN A WHOLE ATTACK DECOMPOSITION INTO ONE TRACE
-------------------------------------------------------
Each SafeMTData Attack_600 row is an ActorAttack: ~5 escalating sub-queries that
decompose one harmful goal. We JOIN those sub-queries into one trace string
("User: q1\nUser: q2\n..."). Many attack rows share a similar escalation SHAPE, so
their trace embeddings land near each other -> the sparse campaign forms a cluster
that clustering can localize even when no single trace looks alarming on its own.

DATA SOURCES
------------
POSITIVES  SafeMTData/SafeMTData 'Attack_600' -> multi_turn_queries joined into ONE
           trace. label 1, group=query_id, source='attack'.
NEGATIVES  HuggingFaceH4/ultrachat_200k (streaming) -> a benign conversation's USER
           turns joined into ONE trace. label 0, group=10_000_000+i,
           source='ultrachat'.
OOD        intrinsec-ai/cstm-bench -> each scenario's sessions rendered to ONE trace.

CPU-only. Downloads datasets. Loads NO model / NO embedder. Stdout is ASCII-only
(Windows cp1252 crashes on unicode like alpha/Delta/||).
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

from steering_tutorials.meerkat import config as C


# --- helpers -----------------------------------------------------------------
def _eprint(*args) -> None:
    """ASCII-only stderr print (Windows cp1252 console safe)."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


def _clean(text) -> str:
    return " ".join(str(text).split()).strip()


def _render_trace(turns) -> str:
    """Render a list of user turns into ONE trace string.

    Each turn becomes a `User: ...` line; the whole episode is one flat document
    (that is the object Meerkat embeds and clusters).
    """
    lines = ["User: %s" % _clean(t) for t in turns if _clean(t)]
    return "\n".join(lines).strip()


# --- attack traces (SafeMTData Attack_600 decompositions) --------------------
def _load_attack_traces(n_attack, rng):
    """Every usable Attack_600 row -> ONE trace = its multi_turn_queries joined.

    A distributed-misuse campaign: each row is an ActorAttack decomposition (~5
    escalating sub-queries) collapsed into a single trace string. Because many
    rows share the same escalation shape, their embeddings cluster together --
    which is exactly the campaign structure Meerkat's clustering recovers.
    """
    from datasets import load_dataset as hf_load

    ds = hf_load(C.ATTACK_DATASET, C.ATTACK_CONFIG)[C.ATTACK_CONFIG]
    traces, groups = [], []
    for i, row in enumerate(ds):
        # `multi_turn_queries` is the list of ActorAttack sub-queries for this attack.
        subs = [t for t in (row.get("multi_turn_queries") or []) if _clean(t)]
        trace = _render_trace(subs)
        if not trace:
            continue
        traces.append(trace)
        groups.append(int(row.get("query_id", i)))  # attack identity (leak-safe grouping)
        if len(traces) >= n_attack:
            break
    return traces, groups


# --- benign traces (UltraChat conversations) ---------------------------------
def _user_turns(row):
    """Return the cleaned USER turns of an UltraChat row (assistant turns dropped)."""
    msgs = row.get("messages") or []
    return [m.get("content", "") for m in msgs
            if m.get("role") == "user" and _clean(m.get("content", ""))]


def _load_benign_traces(n_benign, rng):
    """Stream UltraChat; render each conversation's USER turns into ONE trace.

    Streaming avoids downloading the whole 200k-row benign set; we scan until we
    have n_benign non-empty traces (or hit a scan cap). label 0, source='ultrachat'.
    """
    from datasets import load_dataset as hf_load

    traces = []
    max_scan = max(20000, n_benign * 200)   # cap so a run cannot read all 200k rows
    scanned = 0
    stream = hf_load(C.BENIGN_DATASET, split=C.BENIGN_SPLIT, streaming=True)
    for row in stream:
        scanned += 1
        if scanned > max_scan:
            break
        trace = _render_trace(_user_turns(row))
        if not trace:
            continue
        traces.append(trace)
        if len(traces) >= n_benign:
            break
    return traces, scanned


# --- public API: the trace POOL ----------------------------------------------
def load_trace_pool(n_attack=C.N_ATTACK, n_benign=C.N_BENIGN, seed=C.SEED) -> dict:
    """Build the trace POOL: >= n_attack attack traces + >= n_benign benign traces.

    This is the >=500/class reservoir the rubric mandates; `sample_repository`
    later draws sparse repositories from it. Returns the shared trace-pool dict:

        {"traces": List[str], "labels": List[int] (1=attack, 0=benign),
         "groups": List[int], "sources": List[str]}

    Fixed-seed shuffle keeps the four lists aligned and reproducible.
    """
    rng = random.Random(seed)

    attack_traces, attack_groups = _load_attack_traces(n_attack, rng)
    benign_traces, scanned = _load_benign_traces(n_benign, rng)

    traces = list(attack_traces) + list(benign_traces)
    labels = [1] * len(attack_traces) + [0] * len(benign_traces)
    groups = list(attack_groups) + [10_000_000 + i for i in range(len(benign_traces))]
    sources = ["attack"] * len(attack_traces) + ["ultrachat"] * len(benign_traces)

    # Fixed-seed shuffle (keep the four lists aligned).
    idx = list(range(len(traces)))
    rng.shuffle(idx)
    traces = [traces[i] for i in idx]
    labels = [labels[i] for i in idx]
    groups = [groups[i] for i in idx]
    sources = [sources[i] for i in idx]

    # Report: counts + mean trace char-length per class (the confound leading indicator).
    n_att = sum(labels)
    n_ben = len(labels) - n_att
    pos_len = [len(t) for t, y in zip(traces, labels) if y == 1]
    neg_len = [len(t) for t, y in zip(traces, labels) if y == 0]
    _eprint("[data] pool: N=%d  attack=%d  benign=%d  (scanned %d benign rows)"
            % (len(labels), n_att, n_ben, scanned))
    _eprint("[data] mean trace char-len: attack=%.0f  benign=%.0f"
            % (sum(pos_len) / max(1, len(pos_len)), sum(neg_len) / max(1, len(neg_len))))
    _eprint("[data] source dist: "
            + ", ".join("%s=%d" % (s, c) for s, c in sorted(Counter(sources).items())))
    return {"traces": traces, "labels": labels, "groups": groups, "sources": sources}


# --- repository sampling (the sparse Meerkat regime) -------------------------
def sample_repository(pool, size=C.REPO_SIZE, base_rate=C.BASE_RATE, seed=0) -> dict:
    """Draw ONE repository of `size` traces with a `base_rate` fraction positive.

    A repository mimics one deployment's traffic log: mostly benign, with a sparse
    (default ~5%) minority of attack traces distributed among it. Sampling is
    WITHOUT replacement from the pool (falls back to with-replacement only if the
    pool cannot supply enough of a class -- reported honestly). Same dict shape as
    the pool so downstream code treats pool and repository identically.
    """
    rng = random.Random(seed)
    n_pos = max(1, int(round(size * base_rate)))
    n_neg = size - n_pos

    pos_idx = [i for i, y in enumerate(pool["labels"]) if y == 1]
    neg_idx = [i for i, y in enumerate(pool["labels"]) if y == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def _take(src, n):
        # Prefer sampling without replacement; only oversample if the pool is short.
        if n <= len(src):
            return src[:n]
        extra = [rng.choice(src) for _ in range(n - len(src))] if src else []
        return src + extra

    chosen = _take(pos_idx, n_pos) + _take(neg_idx, n_neg)
    rng.shuffle(chosen)

    return {
        "traces": [pool["traces"][i] for i in chosen],
        "labels": [pool["labels"][i] for i in chosen],
        "groups": [pool["groups"][i] for i in chosen],
        "sources": [pool["sources"][i] for i in chosen],
    }


# --- OOD: CSTM-Bench (real cross-session scenarios) --------------------------
def _session_text(session) -> str:
    """Flatten one CSTM session dict into text (mirrors cross_trajectory's renderer)."""
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


def load_ood_cstm(seed=C.SEED) -> dict:
    """Load intrinsec-ai/cstm-bench (both splits) as the REAL OOD repository.

    Each scenario's `sessions_json` (all its sessions) is rendered into ONE trace
    string; label 1 if scenario_class=='attack' else 0. source='cstm'. This is a
    naturally-sparse-ish real benchmark (~52 attack + 56 benign) we score without
    ever training on it -- the honest out-of-distribution test.
    """
    from datasets import load_dataset as hf_load

    traces, labels, groups, sources = [], [], [], []
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
            # Render the WHOLE scenario (all its sessions) into ONE trace string.
            texts = [_session_text(s) for s in sessions]
            trace = "\n".join(t for t in texts if t).strip()
            if not trace:
                continue
            scls = str(row.get("scenario_class", ""))
            traces.append(trace)
            labels.append(1 if scls == "attack" else 0)
            groups.append(gid)
            sources.append("cstm")
            gid += 1

    n_att = sum(labels)
    n_ben = len(labels) - n_att
    _eprint("[ood] CSTM-Bench: N=%d  attack=%d  benign=%d" % (len(labels), n_att, n_ben))
    return {"traces": traces, "labels": labels, "groups": groups, "sources": sources}


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
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
            for m in range(i, j + 1):
                ranks[order[m]] = avg
            i = j + 1
        sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
        n_p, n_n = len(pos), len(neg)
        u = sum_pos - n_p * (n_p + 1) / 2.0
        return float(u / (n_p * n_n))


def confound_report(traces, labels) -> dict:
    """Can raw trace char-LENGTH alone separate the classes?

    If length_auc is near 0.5 the clustering result cannot be a length artifact; if
    it is high, attack traces are simply longer (a caveat to report). Mean-pooled
    embeddings mitigate raw length, but we measure and surface it regardless.
    """
    lengths = [len(t) for t in traces]
    pos_len = [n for n, y in zip(lengths, labels) if y == 1]
    neg_len = [n for n, y in zip(lengths, labels) if y == 0]
    return {
        "length_auc": _auc(lengths, labels),
        "len_pos_mean": float(sum(pos_len) / len(pos_len)) if pos_len else 0.0,
        "len_neg_mean": float(sum(neg_len) / len(neg_len)) if neg_len else 0.0,
    }


if __name__ == "__main__":
    # SMALL CPU smoke: tiny pool (streaming benign), sample one sparse repository,
    # load the OOD set, print counts + a trace sample + the length confound. NO model.
    _eprint("[smoke] load_trace_pool(n_attack=20, n_benign=20) ...")
    pool = load_trace_pool(n_attack=20, n_benign=20)
    labels = pool["labels"]
    print("POOL COUNTS: N=%d attack=%d benign=%d"
          % (len(labels), sum(labels), len(labels) - sum(labels)))
    print("sources:", dict(Counter(pool["sources"])))
    # Show one attack trace so a reader sees the joined decomposition.
    for t, y in zip(pool["traces"], labels):
        if y == 1:
            snippet = t.replace("\n", " | ")
            print("attack trace sample:",
                  (snippet[:160] + "...") if len(snippet) > 160 else snippet)
            break

    _eprint("[smoke] sample_repository(size=40, base_rate=0.05) ...")
    repo = sample_repository(pool, size=40, base_rate=0.05, seed=0)
    rlab = repo["labels"]
    print("REPO COUNTS: size=%d attack=%d (base_rate~%.2f) benign=%d"
          % (len(rlab), sum(rlab), sum(rlab) / max(1, len(rlab)), len(rlab) - sum(rlab)))

    conf = confound_report(pool["traces"], labels)
    print("CONFOUND REPORT (pool):")
    for key in ("length_auc", "len_pos_mean", "len_neg_mean"):
        print("  %-14s %.4f" % (key, conf[key]))

    _eprint("[smoke] load_ood_cstm() ...")
    ood = load_ood_cstm()
    olab = ood["labels"]
    print("CSTM COUNTS: N=%d attack=%d benign=%d"
          % (len(olab), sum(olab), len(olab) - sum(olab)))
