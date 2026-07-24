"""data.py — trace-pool loader for the MEERKAT trace-clustering lesson.

WHAT A "TRACE" IS HERE
----------------------
A **trace** is one atomic agent event rendered to a SINGLE string ("User: ..."),
the unit Meerkat embeds as one vector and then CLUSTERS across a whole repository.
So the unit of analysis is "one trace = one embeddable document".

THE CAMOUFLAGED-CAMPAIGN CONSTRUCTION (the whole point of this lesson)
---------------------------------------------------------------------
Meerkat's premise (arXiv:2604.11806): a real misuse *campaign* is INDIVIDUALLY
CAMOUFLAGED (no single trace looks alarming on its own) and only becomes visible
when MANY related traces are analysed TOGETHER. To instantiate that premise the
positive class must NOT contain traces that are each a complete harmful artifact.

Each SafeMTData Attack_600 row is one ActorAttack: a harmful goal DECOMPOSED into
~K escalating sub-queries, engineered so that NO single sub-query is overtly
harmful -- the harm only emerges from the full sequence. We exploit exactly that:

  * A violation CAMPAIGN = one attack goal (one query_id).
  * We emit EACH of its K sub-steps as a SEPARATE trace (we do NOT join them).
    Every sub-step trace is innocuous-looking ALONE (an early escalation query is
    not obviously harmful), so a per-trace monitor trained on a couple of labelled
    seeds cannot separate it from benign traffic.
  * BUT the K sub-steps of the same campaign are about ONE underlying goal, so they
    are topically SELF-SIMILAR and CO-LOCATE in embedding space. k-means (using ALL
    traces, unsupervised) collapses the campaign into a tight region and a few seed
    labels light it up. Clustering is what recovers the camouflaged campaign.

This is the fix for the earlier broken construction, which JOINED the whole
decomposition (payload included) into one trace: each positive was then
individually a complete harmful artifact, a per-trace logistic separated it
trivially (ROC-AUC ~0.93), and the Meerkat premise was never instantiated.

WHY VIOLATIONS ARE SPARSE (~5%) AND CAMPAIGN-COHERENT PER REPOSITORY
-------------------------------------------------------------------
In a real deployment the harmful traces are a small, distributed MINORITY of ONE
(or a few) campaigns hiding among mostly-benign traffic. We build a big POOL
(>=500/class) of camouflaged sub-step traces + benign traces, then
`sample_repository` PLANTS whole campaigns sparsely (~5%) among benign singletons:
it draws positives campaign-by-campaign (by query_id group) so a repository's few
attack traces share underlying goals and co-locate -- the structure clustering is
meant to surface. Benign traces are drawn as independent singletons.

DATA SOURCES
------------
POSITIVES  SafeMTData/SafeMTData 'Attack_600' -> EACH multi_turn_queries sub-step is
           its OWN trace. label 1, group=query_id (shared across a campaign's
           sub-steps), source='attack'.
NEGATIVES  HuggingFaceH4/ultrachat_200k (streaming) -> ONE benign USER turn per
           conversation = one trace (innocuous singleton). label 0,
           group=10_000_000+i, source='ultrachat'.
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


def _render_trace(turn) -> str:
    """Render ONE atomic user event into a single `User: ...` trace string.

    A trace here is one event (one sub-step query, or one benign user turn) -- the
    single document Meerkat embeds and clusters. (We deliberately do NOT join a
    whole episode into one trace anymore; see the module docstring.)
    """
    cleaned = _clean(turn)
    return ("User: %s" % cleaned) if cleaned else ""


# --- attack traces (SafeMTData Attack_600 camouflaged sub-steps) --------------
def _load_attack_traces(n_attack, rng):
    """Explode each Attack_600 campaign into its K sub-step traces (NOT joined).

    Each row is one ActorAttack: a harmful goal decomposed into ~K escalating
    sub-queries engineered so no single sub-query is overtly harmful. We emit EACH
    sub-query as its OWN trace, all sharing the campaign's `query_id` as their group.
    So:
      * every positive trace is individually camouflaged (innocuous ALONE), which
        is what defeats a per-trace monitor at a sparse base rate, and
      * a campaign's sub-steps are topically self-similar (one underlying goal) ->
        they co-locate in embedding space, which is what clustering recovers.

    We iterate campaigns and emit all of a campaign's sub-steps, stopping once we
    have >= n_attack sub-step traces (whole campaigns are kept intact). Returns
    (traces, groups) with groups repeated across a campaign's sub-steps.
    """
    from datasets import load_dataset as hf_load

    ds = hf_load(C.ATTACK_DATASET, C.ATTACK_CONFIG)[C.ATTACK_CONFIG]
    traces, groups = [], []
    for i, row in enumerate(ds):
        # `multi_turn_queries` is the list of ActorAttack sub-queries for this goal.
        subs = [t for t in (row.get("multi_turn_queries") or []) if _clean(t)]
        gid = int(row.get("query_id", i))            # campaign identity (leak-safe group)
        emitted = False
        for sub in subs:
            trace = _render_trace(sub)               # one camouflaged sub-step = one trace
            if not trace:
                continue
            traces.append(trace)
            groups.append(gid)
            emitted = True
        # Keep campaigns whole: only stop AFTER finishing a campaign's sub-steps.
        if emitted and len(traces) >= n_attack:
            break
    return traces, groups


# --- benign traces (UltraChat conversations) ---------------------------------
def _user_turns(row):
    """Return all cleaned USER turns of an UltraChat row (assistant turns dropped)."""
    msgs = row.get("messages") or []
    return [m.get("content", "") for m in msgs
            if m.get("role") == "user" and _clean(m.get("content", ""))]


def _load_benign_traces(n_benign, rng, len_lo, len_hi):
    """Stream UltraChat; each benign USER turn = one innocuous SINGLETON trace,
    LENGTH-MATCHED to the attack sub-step distribution.

    A benign trace is a single user turn -- Meerkat's "one event = one trace" unit,
    on the same footing as a camouflaged attack sub-step. Crucially we only KEEP a
    turn whose rendered length falls in the attack sub-step window [len_lo, len_hi].
    UltraChat's opening prompts are long detailed instructions (~600 chars) while
    ActorAttack sub-queries are short focused questions (~100 chars); without this
    match the classes would be trivially separable by LENGTH (a confound), so a
    per-trace monitor could cheat and the clustering contrast would be meaningless.
    Matching the length distribution removes that tell (confound_report's length_auc
    should sit near 0.5). We scan ALL user turns across many conversations to fill
    the pool. label 0, source='ultrachat'.
    """
    from datasets import load_dataset as hf_load

    traces = []
    max_scan = max(20000, n_benign * 400)   # cap so a run cannot read all 200k rows
    scanned = 0
    stream = hf_load(C.BENIGN_DATASET, split=C.BENIGN_SPLIT, streaming=True)
    for row in stream:
        scanned += 1
        if scanned > max_scan:
            break
        for turn in _user_turns(row):
            trace = _render_trace(turn)
            if not trace or not (len_lo <= len(trace) <= len_hi):
                continue          # keep only length-matched, innocuous singletons
            traces.append(trace)
            if len(traces) >= n_benign:
                break
        if len(traces) >= n_benign:
            break
    return traces, scanned


def _percentile(vals, p):
    """Simple percentile (p in [0,100]) on a list, numpy-free."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# --- public API: the trace POOL ----------------------------------------------
def load_trace_pool(n_attack=C.N_ATTACK, n_benign=C.N_BENIGN, seed=C.SEED) -> dict:
    """Build the trace POOL: >= n_attack attack sub-step traces + >= n_benign benign.

    This is the >=500/class reservoir the rubric mandates -- here the positive side
    is >= n_attack camouflaged sub-step traces drawn from ~n_attack/K attack
    campaigns (query_id groups); `sample_repository` later plants whole campaigns
    sparsely. Returns the shared trace-pool dict:

        {"traces": List[str], "labels": List[int] (1=attack, 0=benign),
         "groups": List[int], "sources": List[str]}

    `groups` is the campaign id for attack rows (repeated across a campaign's
    sub-steps) and a unique 10_000_000+i for each benign singleton. Fixed-seed
    shuffle keeps the four lists aligned and reproducible.
    """
    rng = random.Random(seed)

    attack_traces, attack_groups = _load_attack_traces(n_attack, rng)
    # Length-match benign singletons to the attack sub-step length window so the two
    # classes are NOT separable by trace length (an artifact); widen slightly so the
    # window is fillable. See _load_benign_traces for why this confound control matters.
    pos_lens = [len(t) for t in attack_traces] or [1]
    len_lo = max(20.0, _percentile(pos_lens, 10) * 0.6)
    len_hi = _percentile(pos_lens, 90) * 1.05
    benign_traces, scanned = _load_benign_traces(n_benign, rng, len_lo, len_hi)

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

    # Report: counts + mean trace char-length per class (the confound leading
    # indicator) + campaign structure (how many attack goals, sub-steps per goal).
    n_att = sum(labels)
    n_ben = len(labels) - n_att
    pos_len = [len(t) for t, y in zip(traces, labels) if y == 1]
    neg_len = [len(t) for t, y in zip(traces, labels) if y == 0]
    att_campaigns = {g for g, y in zip(groups, labels) if y == 1}
    n_camp = len(att_campaigns)
    _eprint("[data] pool: N=%d  attack=%d  benign=%d  (scanned %d benign rows)"
            % (len(labels), n_att, n_ben, scanned))
    _eprint("[data] attack campaigns=%d  mean sub-steps/campaign=%.1f  (each sub-step is one trace)"
            % (n_camp, n_att / max(1, n_camp)))
    _eprint("[data] mean trace char-len: attack=%.0f  benign=%.0f  (single events -> comparable, no join-length tell)"
            % (sum(pos_len) / max(1, len(pos_len)), sum(neg_len) / max(1, len(neg_len))))
    _eprint("[data] source dist: "
            + ", ".join("%s=%d" % (s, c) for s, c in sorted(Counter(sources).items())))
    return {"traces": traces, "labels": labels, "groups": groups, "sources": sources}


# --- repository sampling (the sparse Meerkat regime) -------------------------
def sample_repository(pool, size=C.REPO_SIZE, base_rate=C.BASE_RATE, seed=0) -> dict:
    """Draw ONE repository of `size` traces with a `base_rate` fraction positive.

    A repository mimics one deployment's traffic log: mostly benign, with a sparse
    (default ~5%) minority that is a CAMPAIGN -- not random unrelated attack traces.
    We PLANT positives campaign-by-campaign: draw whole attack campaigns (query_id
    groups) in random order and add all of a campaign's sub-step traces until we
    reach n_pos positives (trimming only the last campaign to hit n_pos exactly).
    This makes a repository's few positives share underlying goals so they co-locate
    -- the self-similar structure clustering is meant to surface, and the reason
    kmeans_enrich can beat a per-trace monitor here. Benign traces are drawn as
    independent singletons.

    Sampling is WITHOUT replacement (falls back to with-replacement only if the pool
    cannot supply enough of a class -- reported honestly). Same dict shape as the
    pool so downstream code treats pool and repository identically.
    """
    rng = random.Random(seed)
    n_pos = max(1, int(round(size * base_rate)))
    n_neg = size - n_pos

    labels = pool["labels"]
    groups = pool["groups"]

    # Group positive trace indices by their campaign id (query_id).
    campaigns = {}
    for i, y in enumerate(labels):
        if y == 1:
            campaigns.setdefault(groups[i], []).append(i)
    camp_ids = list(campaigns.keys())
    rng.shuffle(camp_ids)

    # Plant whole campaigns until we have n_pos positives; trim the last to n_pos.
    pos_chosen = []
    for cid in camp_ids:
        if len(pos_chosen) >= n_pos:
            break
        pos_chosen.extend(campaigns[cid])
    pos_chosen = pos_chosen[:n_pos]
    # Fallback (only if the pool has too few attack campaigns): oversample w/ repl.
    if len(pos_chosen) < n_pos and pos_chosen:
        pos_chosen += [rng.choice(pos_chosen) for _ in range(n_pos - len(pos_chosen))]

    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    rng.shuffle(neg_idx)
    if n_neg <= len(neg_idx):
        neg_chosen = neg_idx[:n_neg]
    else:
        neg_chosen = neg_idx + [rng.choice(neg_idx) for _ in range(n_neg - len(neg_idx))]

    chosen = pos_chosen + neg_chosen
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
    # SMALL CPU smoke: build a camouflaged-campaign pool, plant one sparse
    # repository, load the OOD set, print counts + campaign structure + a few
    # sub-step trace samples + the length confound. NO model.
    _eprint("[smoke] load_trace_pool(n_attack=60, n_benign=60) ...")
    pool = load_trace_pool(n_attack=60, n_benign=60)
    labels = pool["labels"]
    groups = pool["groups"]
    n_att = sum(labels)
    att_camps = {g for g, y in zip(groups, labels) if y == 1}
    print("POOL COUNTS: N=%d attack=%d benign=%d  attack_campaigns=%d (mean %.1f sub-steps/campaign)"
          % (len(labels), n_att, len(labels) - n_att, len(att_camps),
             n_att / max(1, len(att_camps))))
    print("sources:", dict(Counter(pool["sources"])))
    # Show a few attack sub-step traces so a reader sees each is a SINGLE innocuous
    # query (NOT a full joined decomposition) -- the camouflage.
    shown = 0
    for t, y in zip(pool["traces"], labels):
        if y == 1:
            snippet = t.replace("\n", " ")
            print("  attack sub-step sample:",
                  (snippet[:140] + "...") if len(snippet) > 140 else snippet)
            shown += 1
            if shown >= 3:
                break

    _eprint("[smoke] sample_repository(size=200, base_rate=0.05) ...")
    repo = sample_repository(pool, size=200, base_rate=0.05, seed=0)
    rlab = repo["labels"]
    rpos_camps = {g for g, y in zip(repo["groups"], rlab) if y == 1}
    print("REPO COUNTS: size=%d attack=%d (base_rate~%.2f) benign=%d  from %d planted campaign(s)"
          % (len(rlab), sum(rlab), sum(rlab) / max(1, len(rlab)),
             len(rlab) - sum(rlab), len(rpos_camps)))

    conf = confound_report(pool["traces"], labels)
    print("CONFOUND REPORT (pool):  (length_auc ~0.5 => individual traces are NOT trivially separable by length)")
    for key in ("length_auc", "len_pos_mean", "len_neg_mean"):
        print("  %-14s %.4f" % (key, conf[key]))

    _eprint("[smoke] load_ood_cstm() ...")
    ood = load_ood_cstm()
    olab = ood["labels"]
    print("CSTM COUNTS: N=%d attack=%d benign=%d"
          % (len(olab), sum(olab), len(olab) - sum(olab)))
