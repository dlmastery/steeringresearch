"""data.py — dataset loader for the CROSS-TRAJECTORY latent-aggregation lesson.

A **sample** = an unordered SET of K trajectory strings (one per agent/session).
A **decomposed attack** (label 1) fractures a harmful goal across K trajectories
so no single one carries the payload; a **benign** sample (label 0) is K
trajectories that do NOT aggregate into a harmful goal.

POSITIVES  SafeMTData/SafeMTData, BOTH ungated configs (`C.ATTACK_CONFIGS`):
           `Attack_600` (600 rows, field `multi_turn_queries`) and
           `SafeMTData_1K` (1,680 rows, user turns extracted from `conversations`).
           ONE decomposed-attack sample = an attack's LAST K sub-queries (the
           payload lands last -> included) as K SEPARATE trajectories.
           label 1, group="<config>:<query_id>", source='attack'.
NEGATIVES  depend on `condition`:
  "easy"   K benign USER turns from ONE HuggingFaceH4/ultrachat_200k conversation
           (streaming). group="ultrachat:<i>", source='ultrachat'.
  "hard"   K same-STYLE ActorAttack trajectories that DO NOT complete a harmful
           goal = a DIFFERENT attack's FIRST (k-1) sub-queries (benign lead-up,
           payload excluded) + 1 filler lead-up sub-query. source='attack_prefix'.

CONFOUND STATUS OF THE HARD SET — corrected 2026-08 (this docstring used to say
"length/count-matched", which was false for length by the file's OWN measurement):
  * COUNT: matched by construction. Both classes are EXACTLY K trajectories, so
    the count bar measures 0.500 exactly. That is a genuine PASS.
  * LENGTH: **NOT matched.** Measured directionless `length.auc = 0.704` on hard
    and **0.890** on easy (the easy raw AUC is 0.110, which reads "cleaner than
    chance" only if you forget to fold — see `common/confound.py`). Every headline
    must be claimed as a MARGIN OVER THAT BAR, never over 0.5.
  * CONTENT / SHUFFLE: added 2026-08 via the shared spine; not yet measured.

GROUPING — the split is by GROUP, not by row. `SafeMTData_1K` has MULTIPLE ACTORS
PER `query_id`, so a row-wise half-split would put the same underlying harmful goal
on both the positive and the negative side. `n` and `n_distinct_groups` are
reported SEPARATELY for the same reason: 1,680 rows are NOT 1,680 independent
groups, and inflating n without inflating information is not a fix.

OOD        intrinsec-ai/cstm-bench (both splits) -- real cross-session scenarios.

CPU-only. Downloads datasets. Loads NO LLM. Stdout is ASCII-only (Windows cp1252).
"""
from __future__ import annotations

try:  # SSL middlebox on this host (truststore lets requests use the OS trust store)
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import hashlib
import sys
import json
import random
from collections import Counter

from steering_tutorials.cross_trajectory import config as C
from steering_tutorials.common.confound import (
    confound_report as _spine_confound_report,
    format_report,
    margin_over_bar,
)

__all__ = [
    "load_dataset", "load_ood_cstm", "load_tom_gibbs_semi_benign",
    "confound_report", "format_report", "margin_over_bar",
]


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


def _user_turns(conversations) -> list:
    """User-role turn texts out of a `conversations` list of {role, content} dicts.

    This is how `SafeMTData_1K` stores the same decomposition that `Attack_600`
    stores in `multi_turn_queries`; both end up in the SAME `sub_queries` shape.
    """
    out = []
    for msg in (conversations or []):
        if isinstance(msg, str):
            t = _clean(msg)
            if t:
                out.append(t)
            continue
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        if role and role not in ("user", "human"):
            continue
        t = _clean(msg.get("content", "") or msg.get("text", ""))
        if t:
            out.append(t)
    return out


def _row_sub_queries(row) -> list:
    """Sub-queries of one SafeMTData row, whichever config it came from."""
    mtq = row.get("multi_turn_queries")
    if mtq:
        return [_clean(t) for t in mtq if _clean(t)]
    return _user_turns(row.get("conversations"))


def _pool_fingerprint(items) -> str:
    """SHA-256 over the sampled ids, so a `results.json` records WHICH pool it saw."""
    h = hashlib.sha256()
    for it in items:
        h.update(str(it).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


# --- shared attack loader ----------------------------------------------------
def _all_attacks(configs=None):
    """Every attack row of every configured SafeMTData config.

    Returns a list of ``(sub_queries[List[str]], group_id[str], category, config)``.
    The group id is NAMESPACED by config so `Attack_600` query_id 7 and
    `SafeMTData_1K` query_id 7 are not silently merged.
    """
    from datasets import load_dataset as hf_load

    configs = list(configs or C.ATTACK_CONFIGS)
    rows = []
    per_config = {}
    for cfg in configs:
        try:
            dsd = hf_load(C.ATTACK_DATASET, cfg)
        except Exception as exc:
            _eprint("[data] config %s FAILED to load: %s" % (cfg, exc))
            continue
        # The split is usually named after the config; fall back to the only split.
        if cfg in dsd:
            ds = dsd[cfg]
        else:
            names = list(dsd.keys())
            if not names:
                continue
            ds = dsd[names[0]]
            _eprint("[data] config %s: split %r not found, using %r"
                    % (cfg, cfg, names[0]))
        n_before = len(rows)
        for i, row in enumerate(ds):
            subs = _row_sub_queries(row)
            if not subs:
                continue
            qid = row.get("query_id", row.get("id", i))
            rows.append((subs, "%s:%s" % (cfg, qid), str(row.get("category", "")), cfg))
        per_config[cfg] = len(rows) - n_before

    n_groups = len({g for _, g, _, _ in rows})
    _eprint("[data] attacks: rows=%d from %s ; distinct groups=%d"
            % (len(rows), per_config, n_groups))
    if len(rows) and n_groups < len(rows):
        _eprint("[data] NOTE: %d rows share only %d distinct query_id groups "
                "(SafeMTData_1K has multiple ACTORS per query_id). n != information."
                % (len(rows), n_groups))
    return rows


# --- positives (standard decomposition: an attack's LAST k sub-queries) -------
def _build_positives(attacks, n_pos, k, rng):
    """One sample per attack with >=k sub-queries = its LAST k sub-queries (the
    payload lands last, so it is included). label 1, group=namespaced query_id."""
    pool = [(subs[-k:], gid, cat) for subs, gid, cat, _ in attacks if len(subs) >= k]
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


# --- hard negatives (benign ActorAttack lead-up, GROUP-DISJOINT split) -------
def _build_hard(attacks, n_per_class, k, rng):
    """Count-matched, leakage-free hard set (both classes EXACTLY k trajectories):
      positive = an attack's LAST k sub-queries (payload included);
      negative = a DIFFERENT attack's FIRST (k-1) sub-queries (payload excluded)
                 + 1 filler lead-up sub-query (another attack's first sub-query).

    The halves are split by GROUP (namespaced query_id), not by row. With
    `SafeMTData_1K` in the pool a row-wise split would place several actors of the
    SAME query_id on opposite sides — the same harmful goal as both a positive and
    a negative. Splitting groups makes the halves genuinely disjoint.

    Length is NOT matched: negatives drop the payload sub-query and gain a short
    lead-up, so they are systematically shorter (measured directionless length bar
    0.704). That residual is priced, not hidden — see `confound_report`.
    """
    usable = [(subs, gid, cat) for subs, gid, cat, _ in attacks if len(subs) >= k]
    groups = sorted({gid for _, gid, _ in usable})
    rng.shuffle(groups)
    half = len(groups) // 2
    pos_groups = set(groups[:half])

    pos_rows = [r for r in usable if r[1] in pos_groups]
    neg_rows = [r for r in usable if r[1] not in pos_groups]

    # Filler pool: FIRST (benign lead-up) sub-query of each negative-side attack.
    filler_pool = [(subs[0], gid) for subs, gid, _ in neg_rows]

    positives = [(subs[-k:], gid, cat) for subs, gid, cat in pos_rows]

    negatives = []
    for subs, gid, cat in neg_rows:
        lead = subs[:max(0, k - 1)]              # first (k-1): excludes the payload
        # a filler from a DIFFERENT negative-side attack
        filler = None
        for _ in range(8):
            ftext, fgid = rng.choice(filler_pool)
            if fgid != gid:
                filler = ftext
                break
        if filler is None:
            filler = rng.choice(filler_pool)[0]
        trajs = lead + [filler]
        if len(trajs) == k:                      # only keep exact-K, leakage-free
            negatives.append((trajs, gid, cat))

    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives = positives[:n_per_class]
    negatives = negatives[:n_per_class]
    return positives, negatives


# --- public API --------------------------------------------------------------
def load_dataset(n_pos=C.N_POS, n_neg=C.N_NEG, k=C.K_TRAJ, condition="easy",
                 seed=C.SEED) -> dict:
    """Build the balanced cross-trajectory dataset.

    Returns dict:
      ``samples``  List[List[str]] -- each K trajectory strings
      ``labels``   List[int] -- 1=decomposed attack, 0=benign
      ``groups``   List[str] -- namespaced query_id (or a unique benign id)
      ``sources``  List[str]
      ``meta``     the ACHIEVED configuration, not the requested one: realised
                   ``n_pos``/``n_neg``, ``n_distinct_groups`` (and the per-class
                   split of it), the attack configs used, and a pool fingerprint.
    """
    rng = random.Random(seed)
    attacks = _all_attacks()

    if condition == "hard":
        n_per = min(n_pos, n_neg)
        positives, negatives = _build_hard(attacks, n_per, k, rng)
        samples = [t for t, _, _ in positives] + [t for t, _, _ in negatives]
        labels = [1] * len(positives) + [0] * len(negatives)
        groups = [g for _, g, _ in positives] + [g for _, g, _ in negatives]
        sources = ["attack"] * len(positives) + ["attack_prefix"] * len(negatives)
    else:  # "easy"
        positives = _build_positives(attacks, n_pos, k, rng)
        neg_turns, scanned = _build_easy_negatives(n_neg, k, rng)
        samples = [t for t, _, _ in positives] + list(neg_turns)
        labels = [1] * len(positives) + [0] * len(neg_turns)
        groups = ([g for _, g, _ in positives]
                  + ["ultrachat:%d" % i for i in range(len(neg_turns))])
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
    pos_groups = {g for g, y in zip(groups, labels) if y == 1}
    neg_groups = {g for g, y in zip(groups, labels) if y == 0}
    meta = {
        "condition": condition,
        "requested_n_pos": int(n_pos),
        "requested_n_neg": int(n_neg),
        "n_pos": int(n_p),
        "n_neg": int(n_n),
        "n_distinct_groups": int(len(set(groups))),
        "n_distinct_groups_pos": int(len(pos_groups)),
        "n_distinct_groups_neg": int(len(neg_groups)),
        "group_overlap_pos_neg": int(len(pos_groups & neg_groups)),
        "attack_configs": list(C.ATTACK_CONFIGS),
        "k": int(k),
        "seed": int(seed),
        "pool_fingerprint": _pool_fingerprint(groups),
        "meets_500_floor": bool(n_p >= 500 and n_n >= 500),
    }

    _eprint("[data] condition=%s  N=%d  pos(decomp attack)=%d  neg(benign)=%d  K=%d"
            % (condition, len(labels), n_p, n_n, k))
    _eprint("[data] distinct GROUPS: total=%d pos=%d neg=%d overlap=%d  "
            "(n != n_distinct_groups when a config has several actors per query_id)"
            % (meta["n_distinct_groups"], meta["n_distinct_groups_pos"],
               meta["n_distinct_groups_neg"], meta["group_overlap_pos_neg"]))
    _eprint("[data] source dist: "
            + ", ".join("%s=%d" % (s, c) for s, c in sorted(Counter(sources).items())))
    kdist = Counter(len(s) for s in samples)
    _eprint("[data] trajectory-count dist: "
            + ", ".join("%d->%d" % (kk, kdist[kk]) for kk in sorted(kdist)))
    if not meta["meets_500_floor"]:
        _eprint("[data] WARNING: below the CLAUDE.md 17 rule-1 floor of >=500/class "
                "(pos=%d neg=%d). Report it, do not round it up." % (n_p, n_n))
    return {"samples": samples, "labels": labels, "groups": groups,
            "sources": sources, "meta": meta}


# --- OPTIONAL second source: tom-gibbs Semi-Benign hard negatives -------------
def load_tom_gibbs_semi_benign(n=500, k=C.K_TRAJ, seed=C.SEED) -> list:
    """Load the tom-gibbs Semi-Benign pool as REAL hard negatives.

    `tom-gibbs/multi-turn_jailbreak_attack_datasets` (Gibbs et al. 2024,
    arXiv:2409.00137) is ungated and ships 382 Complete-Harmful decomposed
    multi-turn attacks alongside **1,200 purpose-built Semi-Benign conversations**.
    Those Semi-Benign conversations are a real hard-negative pool for exactly this
    lesson: multi-turn, same register, no completed harmful goal — which is what
    `_build_hard` currently has to SYNTHESISE by stripping the payload sub-query
    off another attack.

    STATUS: **OFF by default and SCHEMA-UNVERIFIED.** No download was performed
    when this loader was written, so the exact config and column names are not
    confirmed on this host. The loader therefore PROBES: it enumerates configs,
    picks the one whose name mentions semi-benign, and extracts user turns from
    whichever conversation-shaped column it finds. It raises loudly rather than
    guessing. Enable with ``CT_TOM_GIBBS=1`` and verify the printed schema before
    trusting any number built on it.

    Returns a list of K-trajectory samples (List[List[str]]).
    """
    from datasets import get_dataset_config_names, load_dataset as hf_load

    rng = random.Random(seed)
    configs = get_dataset_config_names(C.TOM_GIBBS_DATASET)
    _eprint("[tom-gibbs] configs: %s" % (configs,))
    want = [c for c in configs if "semi" in c.lower() and "benign" in c.lower()]
    if not want:
        raise RuntimeError(
            "tom-gibbs: no semi-benign config found among %r. Inspect the repo and "
            "pass the right name; this loader will NOT guess." % (configs,))
    cfg = want[0]

    dsd = hf_load(C.TOM_GIBBS_DATASET, cfg)
    split = list(dsd.keys())[0]
    ds = dsd[split]
    cols = list(ds.column_names)
    _eprint("[tom-gibbs] config=%s split=%s rows=%d columns=%s"
            % (cfg, split, len(ds), cols))

    conv_cols = [c for c in cols
                 if c.lower() in ("conversations", "conversation", "messages",
                                  "turns", "multi_turn_queries", "dialogue")]
    if not conv_cols:
        raise RuntimeError(
            "tom-gibbs config %r has no conversation-shaped column; columns=%r. "
            "Extend this loader explicitly rather than inferring." % (cfg, cols))
    col = conv_cols[0]

    samples = []
    for row in ds:
        val = row.get(col)
        turns = ([_clean(t) for t in val if _clean(t)]
                 if isinstance(val, list) and val and isinstance(val[0], str)
                 else _user_turns(val))
        if len(turns) >= k:
            samples.append(turns[-k:])
        if len(samples) >= n:
            break
    rng.shuffle(samples)
    _eprint("[tom-gibbs] built %d semi-benign K=%d samples from column %r"
            % (len(samples), k, col))
    return samples


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


def _select_sessions(texts, k, how):
    """Pick which of a scenario's ~26 sessions become the K trajectories.

    "longest" was the ORIGINAL rule and it is close to the worst available one on
    the `dilution` split, whose whole premise is that the signal is diluted across
    MANY sessions: keeping the 5 most verbose sessions inverts the construct and
    discards ~80% of every scenario. "uniform" spreads the pick evenly across the
    session list; ``k <= 0`` keeps every session.
    """
    if k is None or int(k) <= 0 or how == "all" or len(texts) <= int(k):
        return list(texts)
    k = int(k)
    if how == "longest":
        return sorted(texts, key=len, reverse=True)[:k]
    n = len(texts)
    idx = [int(round(i * (n - 1) / float(k - 1))) for i in range(k)] if k > 1 else [0]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(texts[i])
    return out


def load_ood_cstm(k=None, seed=C.SEED, how=None) -> dict:
    """Load intrinsec-ai/cstm-bench (splits 'dilution' + 'cross_session').

    ONE sample per scenario = ``C.OOD_K`` session texts selected by ``C.OOD_SELECT``
    (default: UNIFORMLY across the scenario's session list). label = 1 if
    ``scenario_class == 'attack'`` else 0. source='cstm'.

    ``meta`` records the selection rule and how many sessions each scenario
    actually had, so the number of DISCARDED sessions is on the record beside any
    transfer claim.
    """
    from datasets import load_dataset as hf_load

    k = C.OOD_K if k is None else k
    how = (how or C.OOD_SELECT or "uniform").lower()

    samples, labels, groups, sources = [], [], [], []
    avail = []
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
            avail.append(len(texts))
            texts = _select_sessions(texts, k, how)
            scls = str(row.get("scenario_class", ""))
            samples.append(texts)
            labels.append(1 if scls == "attack" else 0)
            groups.append("cstm:%s:%d" % (split, gid))
            sources.append("cstm")
            gid += 1

    n_att = sum(labels)
    n_ben = len(labels) - n_att
    kept = [len(s) for s in samples]
    meta = {
        "selection": how,
        "ood_k": int(k),
        "n_scenarios": len(labels),
        "n_attack": int(n_att),
        "n_benign": int(n_ben),
        "sessions_available_mean": float(sum(avail) / len(avail)) if avail else 0.0,
        "sessions_available_min": int(min(avail)) if avail else 0,
        "sessions_available_max": int(max(avail)) if avail else 0,
        "sessions_kept_mean": float(sum(kept) / len(kept)) if kept else 0.0,
        "sessions_discarded_frac": (1.0 - sum(kept) / float(sum(avail))) if avail else 0.0,
    }
    _eprint("[ood] CSTM-Bench: N=%d attack=%d benign=%d ; selection=%s k=%s ; "
            "sessions available mean=%.1f, kept mean=%.1f, DISCARDED %.0f%%"
            % (len(labels), n_att, n_ben, how, k, meta["sessions_available_mean"],
               meta["sessions_kept_mean"], 100.0 * meta["sessions_discarded_frac"]))
    return {"samples": samples, "labels": labels, "groups": groups,
            "sources": sources, "meta": meta}


# --- confound audit (delegated to the ONE shared spine) -----------------------
def confound_report(samples, labels, seed=C.SEED, n_folds=C.N_FOLDS,
                    run_content=None, run_shuffle=None) -> dict:
    """Four confound bars via ``steering_tutorials.common.confound``.

    This used to be a LOCAL implementation that returned the RAW auc and never
    folded `max(auc, 1-auc)` — while `CONFOUND_DISCIPLINE.md` §4 declared it "the
    canonical form". It was not: the easy condition's raw `totalchar_auc = 0.10975`
    reads cleaner-than-chance and is a **0.890** length tell with the sign flipped,
    and the README priced 0.99x AUCs against no bar at all. The fold now lives in
    the spine and this function is a thin adapter.

    Returns the spine schema: ``length``/``count``/``content``/``shuffle`` blocks,
    each with ``auc_raw`` (unfolded) and ``auc`` (directionless), plus
    ``worst_name``/``worst_auc`` — the bar a method must actually beat.
    """
    return _spine_confound_report(
        samples, labels, units=samples, seed=seed, n_folds=n_folds,
        run_content=C.CONFOUND_CONTENT if run_content is None else run_content,
        run_shuffle=C.CONFOUND_SHUFFLE if run_shuffle is None else run_shuffle,
    )


if __name__ == "__main__":
    # SMALL CPU smoke (streaming benign, no model). ASCII stdout.
    _eprint("[smoke] load_dataset(n_pos=10, n_neg=10, k=5, condition='hard') ...")
    data = load_dataset(n_pos=10, n_neg=10, k=5, condition="hard")
    labels = data["labels"]
    print("MAIN(hard) COUNTS: N=%d pos=%d neg=%d"
          % (len(labels), sum(labels), len(labels) - sum(labels)))
    print("MAIN(hard) GROUPS: distinct=%d (pos=%d neg=%d, overlap=%d)"
          % (data["meta"]["n_distinct_groups"], data["meta"]["n_distinct_groups_pos"],
             data["meta"]["n_distinct_groups_neg"], data["meta"]["group_overlap_pos_neg"]))
    print("attack_configs:", data["meta"]["attack_configs"])
    print("sources:", dict(Counter(data["sources"])))
    if data["samples"]:
        ex = data["samples"][0]
        print("sample[0] label=%d  K=%d trajectories:" % (labels[0], len(ex)))
        for i, t in enumerate(ex):
            print("  traj[%d]: %s" % (i, (t[:90] + "...") if len(t) > 90 else t))
    conf = confound_report(data["samples"], labels)
    print(format_report(conf))
    print("A method may claim only the margin over worst_auc=%.4f, NEVER over 0.5."
          % conf["worst_auc"])

    _eprint("[smoke] load_ood_cstm() ...")
    ood = load_ood_cstm()
    olab = ood["labels"]
    print("CSTM COUNTS: N=%d attack=%d benign=%d  selection=%s (discarded %.0f%% of sessions)"
          % (len(olab), sum(olab), len(olab) - sum(olab),
             ood["meta"]["selection"], 100.0 * ood["meta"]["sessions_discarded_frac"]))
    if ood["samples"]:
        oex = ood["samples"][0]
        print("cstm sample[0] label=%d K=%d  first-traj: %s"
              % (olab[0], len(oex), (oex[0][:90] + "...") if len(oex[0]) > 90 else oex[0]))
