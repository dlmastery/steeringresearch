"""data.py -- the ATBench corpus loader for the traj_probes series.

Implements the `TrajLoader` protocol from `types.py` (which this file must never
edit). CPU-only, loads no model, prints ASCII only. It DOES download a dataset.

THE REAL SHAPE OF `contents` (measured, not assumed)
----------------------------------------------------
`contents` is nested TWO deep, and the outer level is a singleton wrapper:

    contents : list[1]                      <- 1000/1000 rows have length exactly 1
      -> conversation : list[N]             <- N in {2,4,6,...,62}, always even+
        -> turn : dict

and a turn comes in exactly TWO shapes, never mixed (measured over all 9,009
turns of the 1,000-row config and all 6,229 turns of the 500-row config):

    {"role": "user"|"environment", "content": str}          # 4517 turns
    {"role": "agent", "thought": str, "action": str}        # 4492 turns

So the AGENT turns carry NO `content` key at all. A parser that reads
`turn["content"]` and skips what is missing silently drops every agent turn --
half the corpus, and precisely the half that carries the failure. `action` is a
JSON string, either a tool call

    {"name": "create_event", "arguments": {...}}

or the terminal form `Complete{"response": "..."}`. `thought` is often empty.

The role vocabulary is ATBench's, not the spine's: agent / environment / user.
`_ROLE_MAP` renames them to the spine's assistant / tool / user, and an unknown
role is passed through with a loud warning rather than dropped.

ATBench500 has the SAME turn schema under different row-level names
(`content` not `contents`, `conv_id` not `id`, and NO `reason` field).

GROUPING -- why every trajectory is its own group, and why that is not degenerate
--------------------------------------------------------------------------------
ATBench has NO field that groups rows. Measured on all 1,000 rows:

    unique `id`                     1000 / 1000
    unique first-user message       1000 / 1000  (0 duplicate groups)
    unique tool-suite signature      998 / 1000

The single 3-row tool-suite family (ids 6, 95, 185) is three UNRELATED scenarios
that happen to reuse the same two tools, so grouping on the tool suite would
merge unrelated episodes and understate n rather than prevent leakage. Grouping
on the first tool name (771 groups) is worse: 88 of those groups straddle the
label. ATBench500's `conv_id` is likewise unique per row (500/500).

So: **each trajectory is its own group**, `group_id = "<source>::<uid>"`.

That is not the degenerate choice it looks like. An `ActivationBundle` in this
series has one row per TURN, so the group is doing the work that matters: it
stops turn 3 of an episode training a probe that is then scored on turn 7 of the
same episode. The leakage channel that group-aware CV must close here is
WITHIN-trajectory, and this closes it.

WHAT THE LABEL MEANS
--------------------
`label == 1` is UNSAFE, at the trajectory level, from the benchmark's own
annotation (arXiv:2604.02022). There is NO per-step label: the corpus never says
which turn went wrong. `reason` is free text explaining the trajectory verdict
after the fact and is carried as `rationale` -- it is NOT a step label, and it
must never be fed to a probe, since it describes the answer.

WHY WE READ THE RAW JSON AND NOT `load_dataset`
-----------------------------------------------
`load_dataset("AI45Research/ATBench", "ATBench")` works, but its `tool_used`
column infers an Arrow struct with several thousand `parameters.properties`
fields, and per-row access is pathologically slow -- indexing 300 rows did not
finish in 2 minutes on this host. The config's own `data_files` entry points at a
plain JSON file, so `hf_hub_download` + `json.load` gets the identical rows in
about a second. The `datasets` path is kept as a fallback and drops `tool_used`
for the same reason.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import sys
from pathlib import Path

# MUST precede any datasets / huggingface_hub import: this host runs an
# intercepting TLS proxy whose root CA lives only in the Windows store.
from steering_tutorials.common.netboot import enable as _enable_tls

_enable_tls()

from steering_tutorials.traj_probes import config as C
from steering_tutorials.traj_probes.types import AgentTrajectory, TrajCorpus, Turn

__all__ = ["ATBenchLoader", "load_corpus", "pool_fingerprint"]

CACHE_SCHEMA_VERSION = 1

# ATBench role vocabulary -> the spine's vocabulary.
_ROLE_MAP = {"agent": "assistant", "environment": "tool", "user": "user",
             "system": "system"}

# Row-level field names differ between the two releases; the TURN schema does not.
_ROW_FIELDS = {
    "atbench": {"conv": "contents", "uid": "id", "reason": "reason"},
    "atbench500": {"conv": "content", "uid": "conv_id", "reason": None},
}
_DATA_FILE = {"atbench": "ATBench/test.json", "atbench500": "ATBench500/test.json"}


def _warn(msg):
    """Loud on BOTH streams. A shortfall that only appears in a log is not loud."""
    banner = "!" * 74
    text = "%s\nWARNING: %s\n%s" % (banner, msg, banner)
    print(text)
    print(text, file=sys.stderr)


def pool_fingerprint(uids):
    """sha256 over the SORTED uids actually sampled, first 16 hex chars.

    Fingerprints the POOL, not the parse: two runs that selected the same
    trajectories agree here even if the rendering code changed, and two runs that
    selected different trajectories cannot be confused for one another.
    """
    joined = "\n".join(sorted(str(u) for u in uids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# --- fetching ----------------------------------------------------------------
def _fetch_rows(corpus):
    """The raw row dicts for `corpus`, from the local HF cache or the hub."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(C.HF_DATASET, _DATA_FILE[corpus],
                               repo_type="dataset")
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        if not isinstance(rows, list):
            raise ValueError("expected a JSON list, got %s" % type(rows).__name__)
        return rows, "hf_hub_download(%s)" % _DATA_FILE[corpus]
    except Exception as exc:                       # noqa: BLE001 -- fallback below
        _warn("hf_hub_download failed (%s: %s); falling back to load_dataset, "
              "which is much slower on this dataset."
              % (type(exc).__name__, exc))
        from datasets import load_dataset
        ds = load_dataset(C.HF_DATASET, C.HF_CONFIG)[C.HF_SPLIT]
        # `tool_used` is the slow column and nothing here reads it.
        if "tool_used" in ds.column_names:
            ds = ds.remove_columns(["tool_used"])
        return list(ds), "load_dataset(%s,%s)" % (C.HF_DATASET, C.HF_CONFIG)


# --- parsing -----------------------------------------------------------------
def _flatten_conversation(raw, uid):
    """`contents` -> a flat list of turn dicts, without assuming the depth."""
    if not isinstance(raw, list):
        raise ValueError("trajectory %s: `contents` is %s, not a list"
                         % (uid, type(raw).__name__))
    if raw and all(isinstance(e, dict) for e in raw):
        return raw                                  # already flat
    convs = [e for e in raw if isinstance(e, list)]
    if len(convs) != len(raw):
        raise ValueError("trajectory %s: `contents` mixes lists and non-lists"
                         % uid)
    if len(convs) > 1:
        # Never observed (1000/1000 and 500/500 rows are singletons), so if it
        # happens the assumption has changed and the reader must be told.
        _warn("trajectory %s: `contents` holds %d conversations, not the 1 "
              "observed everywhere in ATBench. Concatenating them, and the turn "
              "indices now run across a boundary the corpus drew."
              % (uid, len(convs)))
    return [t for conv in convs for t in conv]


def _parse_action(action):
    """The tool name from an ATBench `action` string, or None.

    Two forms: a bare JSON tool call, and the terminal `Complete{...}` form.
    """
    if not action:
        return None
    s = action.strip()
    if s.startswith("Complete"):
        return "Complete"
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    if isinstance(obj, dict) and obj.get("name"):
        return str(obj["name"])
    return None


def _render_turn(raw, index, uid, unknown_roles):
    """One raw ATBench turn dict -> one spine `Turn`.

    An agent turn's `content` is its thought AND its action, because
    `AgentTrajectory.text` is what both the model and the content bar read. Drop
    the action from the text and the bag-of-words bar is scoring a different
    document than the probe -- which is not a comparison.
    """
    role_raw = str(raw.get("role") or "unknown")
    role = _ROLE_MAP.get(role_raw)
    if role is None:
        unknown_roles.add(role_raw)
        role = role_raw

    content = raw.get("content")
    thought = raw.get("thought")
    action_str = raw.get("action")

    if content is None:
        parts = []
        if thought:
            parts.append("thought: %s" % thought)
        if action_str:
            parts.append("action: %s" % action_str)
        content = "\n".join(parts)
    content = "" if content is None else str(content)

    return Turn(
        index=index,
        role=role,
        content=content,
        action=_parse_action(action_str),
        # The tool RESULT arrives as the next `environment` turn's content, so a
        # tool turn's own content is its output.
        tool_output=content if role == "tool" else None,
        # ATBench annotates no tool-call dependency edges. Reproduction C's
        # `consumes_from` therefore stays empty on this corpus rather than being
        # guessed from argument-string matching, which would be a label we
        # invented and then measured.
        consumes_from=(),
    )


def _parse_row(row, corpus, max_turns, unknown_roles):
    f = _ROW_FIELDS[corpus]
    uid = str(row[f["uid"]])
    label = row.get("label")
    if label not in (0, 1, "0", "1"):
        raise ValueError("trajectory %s: label %r is not 0/1" % (uid, label))
    raw_turns = _flatten_conversation(row[f["conv"]], uid)
    if max_turns and max_turns > 0:
        raw_turns = raw_turns[:max_turns]
    turns = tuple(_render_turn(t, i, uid, unknown_roles)
                  for i, t in enumerate(raw_turns))
    reason = row.get(f["reason"]) if f["reason"] else None
    return AgentTrajectory(
        uid=uid,
        turns=turns,
        label=int(label),
        # ONE GROUP PER TRAJECTORY -- see the GROUPING section of the module
        # docstring. ATBench has no scenario/task field to group on; this is the
        # deliberate choice, not an oversight.
        group_id="%s::%s" % (corpus, uid),
        source=corpus,
        risk_source=(str(row["risk_source"]) if row.get("risk_source") else None),
        failure_mode=(str(row["failure_mode"]) if row.get("failure_mode") else None),
        real_world_harm=(str(row["real_world_harm"])
                         if row.get("real_world_harm") else None),
        rationale=(str(reason) if reason else None),
        mistake_step=None,       # ATBench carries no per-step annotation
        mistake_agent=None,
    )


# --- (de)serialisation for the cache -----------------------------------------
def _traj_to_dict(t):
    return {
        "uid": t.uid, "label": t.label, "group_id": t.group_id,
        "source": t.source, "risk_source": t.risk_source,
        "failure_mode": t.failure_mode, "real_world_harm": t.real_world_harm,
        "rationale": t.rationale, "mistake_step": t.mistake_step,
        "mistake_agent": t.mistake_agent,
        "turns": [{"index": u.index, "role": u.role, "content": u.content,
                   "action": u.action, "tool_output": u.tool_output,
                   "consumes_from": list(u.consumes_from)} for u in t.turns],
    }


def _traj_from_dict(d):
    turns = tuple(Turn(index=u["index"], role=u["role"], content=u["content"],
                       action=u["action"], tool_output=u["tool_output"],
                       consumes_from=tuple(u["consumes_from"]))
                  for u in d["turns"])
    return AgentTrajectory(
        uid=d["uid"], turns=turns, label=d["label"], group_id=d["group_id"],
        source=d["source"], risk_source=d["risk_source"],
        failure_mode=d["failure_mode"], real_world_harm=d["real_world_harm"],
        rationale=d["rationale"], mistake_step=d["mistake_step"],
        mistake_agent=d["mistake_agent"])


# --- the loader --------------------------------------------------------------
class ATBenchLoader:
    """ATBench / ATBench500 as `TrajCorpus`. Satisfies `types.TrajLoader`."""

    def __init__(self, corpus=None, max_turns=None, cache_path=None,
                 cache_refresh=None):
        self.corpus = (corpus or C.CORPUS).lower()
        if self.corpus not in C.CORPUS_CHOICES:
            raise ValueError("corpus must be one of %r, got %r"
                             % (list(C.CORPUS_CHOICES), self.corpus))
        self.name = self.corpus
        self.max_turns = C.MAX_TURNS if max_turns is None else int(max_turns)
        self._cache_path = Path(cache_path) if cache_path else None
        self.cache_refresh = (C.CACHE_REFRESH if cache_refresh is None
                              else bool(cache_refresh))
        self.last_shortfall = {}
        self.last_cache_status = "not-attempted"

    # -- paths
    def cache_path(self, n_per_class, seed):
        if self._cache_path is not None:
            return self._cache_path
        from steering_tutorials.common.artifact_paths import keyed_path
        return keyed_path(C.ARTIFACTS, "corpus", ".json.gz", self.corpus,
                          "n%d" % n_per_class, "s%d" % seed,
                          "t%d" % self.max_turns)

    # -- selection
    def _select_uids(self, rows, n_per_class, seed):
        """Balanced, seeded, WITHOUT replacement, per class.

        Candidates are sorted before sampling so the selection depends on the
        seed and the pool -- never on the order the rows happened to arrive in.
        """
        f = _ROW_FIELDS[self.corpus]
        by_label = {0: [], 1: []}
        for row in rows:
            lab = int(row["label"])
            if lab not in by_label:
                raise ValueError("unexpected label %r" % lab)
            by_label[lab].append(str(row[f["uid"]]))

        chosen, shortfall = [], {}
        for lab in (0, 1):
            pool = sorted(set(by_label[lab]))
            if len(pool) != len(by_label[lab]):
                _warn("class %d has %d rows but only %d distinct uids; "
                      "de-duplicating by uid."
                      % (lab, len(by_label[lab]), len(pool)))
            take = min(n_per_class, len(pool))
            if take < n_per_class:
                shortfall[lab] = {"requested": n_per_class, "available": len(pool),
                                  "short_by": n_per_class - len(pool)}
                _warn("class %d (%s): asked for %d trajectories, the pool holds "
                      "only %d. Taking all %d. The corpus records the REQUEST "
                      "(%d) and the ACHIEVED count separately -- do not report "
                      "the request as if it were delivered."
                      % (lab, "unsafe" if lab else "safe", n_per_class,
                         len(pool), take, n_per_class))
            rng = random.Random("%s|%s|%d|%d" % (self.corpus, seed, lab, take))
            chosen.extend(rng.sample(pool, take))
        self.last_shortfall = shortfall
        return chosen

    # -- cache
    def _load_cache(self, path, expected_fp):
        if not path.exists():
            self.last_cache_status = "miss"
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            blob = json.load(fh)

        stored_fp = blob.get("pool_fingerprint")
        uids = [d["uid"] for d in blob.get("trajectories", [])]
        recomputed = pool_fingerprint(uids)
        if recomputed != stored_fp:
            raise SystemExit(
                "CORRUPT CACHE %s: its stored pool_fingerprint (%s) does not "
                "match the fingerprint of the trajectories inside it (%s). The "
                "file cannot be attributed to the pool it claims, so it is not "
                "evidence. Delete it or re-run with TP_CACHE_REFRESH=1."
                % (path.name, stored_fp, recomputed))

        if stored_fp != expected_fp or blob.get("schema") != CACHE_SCHEMA_VERSION:
            msg = ("CACHE DISAGREES: %s holds pool %s (schema %s) but this "
                   "config selects pool %s (schema %d). Reusing it would score "
                   "one set of trajectories under another set's provenance."
                   % (path.name, stored_fp, blob.get("schema"), expected_fp,
                      CACHE_SCHEMA_VERSION))
            if not self.cache_refresh:
                raise SystemExit(msg + " Re-run with TP_CACHE_REFRESH=1 to "
                                       "rebuild it, or delete the file.")
            _warn(msg + " TP_CACHE_REFRESH=1 is set, so it will be REBUILT.")
            self.last_cache_status = "refresh"
            return None

        self.last_cache_status = "hit"
        return blob

    def _write_cache(self, path, blob):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(blob, fh)
        os.replace(tmp, path)

    # -- the protocol method
    def load(self, n_per_class=500, seed=0):
        n_per_class = int(n_per_class)
        seed = int(seed)
        rows, fetch_note = _fetch_rows(self.corpus)
        uids = self._select_uids(rows, n_per_class, seed)
        expected_fp = pool_fingerprint(uids)

        path = self.cache_path(n_per_class, seed)
        blob = self._load_cache(path, expected_fp)

        if blob is None:
            f = _ROW_FIELDS[self.corpus]
            wanted = set(uids)
            by_uid = {}
            for row in rows:
                u = str(row[f["uid"]])
                if u in wanted:
                    by_uid[u] = row
            missing = wanted - set(by_uid)
            if missing:
                raise SystemExit("selected %d uids but %d are absent from the "
                                 "fetched rows: %s"
                                 % (len(wanted), len(missing),
                                    sorted(missing)[:5]))
            unknown_roles = set()
            trajs = [_parse_row(by_uid[u], self.corpus, self.max_turns,
                                unknown_roles) for u in sorted(wanted)]
            if unknown_roles:
                _warn("unmapped role(s) %s passed through verbatim; _ROLE_MAP "
                      "covers %s." % (sorted(unknown_roles),
                                      sorted(_ROLE_MAP)))
            blob = {
                "schema": CACHE_SCHEMA_VERSION,
                "corpus": self.corpus,
                "hf_dataset": C.HF_DATASET,
                "hf_config": {"atbench": "ATBench",
                              "atbench500": "ATBench500"}[self.corpus],
                "fetched_via": fetch_note,
                "requested_n_per_class": n_per_class,
                "seed": seed,
                "max_turns": self.max_turns,
                "pool_size_by_label": {"0": sum(1 for r in rows
                                                if int(r["label"]) == 0),
                                       "1": sum(1 for r in rows
                                                if int(r["label"]) == 1)},
                "shortfall": self.last_shortfall,
                "pool_fingerprint": expected_fp,
                "trajectories": [_traj_to_dict(t) for t in trajs],
            }
            # JSON on disk BEFORE any summary print, so a late crash still
            # leaves the data (CLAUDE.md 17, operational playbook 3).
            self._write_cache(path, blob)
        else:
            trajs = [_traj_from_dict(d) for d in blob["trajectories"]]
            self.last_shortfall = blob.get("shortfall", {})

        corpus = TrajCorpus(
            name=self.corpus,
            trajectories=trajs,
            requested_n_per_class=n_per_class,
            pool_fingerprint=expected_fp,
            licence=C.DATASET_LICENCE,
            label_provenance=(
                "ATBench `label` field (1 = unsafe), a TRAJECTORY-level safety "
                "annotation from %s. Benign rows carry risk_source == "
                "failure_mode == real_world_harm == 'benign'. `reason` is the "
                "corpus's post-hoc explanation of the verdict, carried as "
                "`rationale`; it describes the answer and must never reach a "
                "probe." % C.DATASET_PAPER),
            step_label_provenance=(
                "NONE. ATBench labels the whole trajectory and never says which "
                "turn went wrong, so mistake_step/mistake_agent are None on "
                "every trajectory and any per-step claim in this series is a "
                "claim about a label the corpus does not carry."),
        )
        return corpus


def load_corpus(n_per_class=None, seed=None, corpus=None, max_turns=None,
                allow_step_leak: bool = False):
    """Convenience wrapper defaulting to the config anchor.

    Gates on the DETERMINISTIC STEP-INDEX REGION by default. On ATBench the
    longest safe trajectory is 18 turns and the longest unsafe one is 62, so an
    uncapped corpus hands a probe 570 rows whose label follows from position
    alone -- and `probes.StepResidualiser` removes LINEAR position only, so
    nothing downstream would notice. The gate lives HERE, at the load path, so
    it cannot be skipped by a caller who forgot it (leakage.py).

    Pass allow_step_leak=True only to study the region deliberately.
    """
    loader = ATBenchLoader(corpus=corpus, max_turns=max_turns)
    out = loader.load(
        n_per_class=C.N_PER_CLASS if n_per_class is None else n_per_class,
        seed=C.SEED if seed is None else seed)
    from steering_tutorials.traj_probes.leakage import (
        assert_no_deterministic_region)
    assert_no_deterministic_region(out, acknowledge=allow_step_leak)
    return out


def summarise(corpus):
    """An ASCII provenance block. Print this above any number, always."""
    counts = {}
    for t in corpus.trajectories:
        counts[t.risk_source or "unknown"] = counts.get(t.risk_source
                                                        or "unknown", 0) + 1
    lines = [
        "corpus            %s" % corpus.name,
        "licence           %s" % corpus.licence,
        "pool_fingerprint  %s" % corpus.pool_fingerprint,
        "requested/class   %d" % corpus.requested_n_per_class,
        "achieved          %d safe / %d unsafe"
        % (corpus.achieved_n_neg, corpus.achieved_n_pos),
        "turns             %s" % corpus.turn_count_summary(),
        "groups            %d distinct group_id over %d trajectories"
        % (len({t.group_id for t in corpus.trajectories}),
           len(corpus.trajectories)),
        "risk_source       %s" % sorted(counts.items(), key=lambda kv: -kv[1])[:5],
    ]
    return "\n".join(lines)


def _self_test():
    """CPU-only. Downloads the dataset (small); loads NO model."""
    from steering_tutorials.traj_probes.types import TrajLoader

    C.ensure_artifacts()
    n, seed = 25, 0                       # SMOKE size -- not a reportable n
    tmp = C.ARTIFACTS / ("_selftest_corpus_%s_n%d_s%d.json.gz"
                         % (C.CORPUS, n, seed))
    if tmp.exists():
        tmp.unlink()

    loader = ATBenchLoader(cache_path=tmp)
    assert isinstance(loader, TrajLoader), "does not satisfy the TrajLoader protocol"
    print("OK  ATBenchLoader satisfies types.TrajLoader")

    corpus = loader.load(n_per_class=n, seed=seed)
    assert loader.last_cache_status == "miss"
    print("OK  cold load (%s)" % loader.last_cache_status)
    print(summarise(corpus))

    assert corpus.achieved_n_pos == n and corpus.achieved_n_neg == n
    assert corpus.requested_n_per_class == n
    print("OK  balanced: %d/%d at request %d" % (corpus.achieved_n_pos,
                                                 corpus.achieved_n_neg, n))

    uids = [t.uid for t in corpus.trajectories]
    assert len(set(uids)) == len(uids), "sampling was NOT without replacement"
    assert len({t.group_id for t in corpus.trajectories}) == len(uids)
    print("OK  without replacement, and one group per trajectory")

    assert corpus.pool_fingerprint == pool_fingerprint(uids)
    again = ATBenchLoader(cache_path=tmp).load(n_per_class=n, seed=seed)
    assert again.pool_fingerprint == corpus.pool_fingerprint
    assert loader.last_cache_status in ("miss", "hit")
    print("OK  fingerprint %s is deterministic across calls"
          % corpus.pool_fingerprint)

    other = ATBenchLoader(cache_path=tmp.with_name("_selftest_seed1.json.gz"))
    o = other.load(n_per_class=n, seed=1)
    assert o.pool_fingerprint != corpus.pool_fingerprint, \
        "a different seed selected an identical pool"
    print("OK  a different seed selects a different pool (%s)" % o.pool_fingerprint)

    warm = ATBenchLoader(cache_path=tmp)
    warm.load(n_per_class=n, seed=seed)
    assert warm.last_cache_status == "hit", warm.last_cache_status
    print("OK  warm load reuses the cache")

    # A cache that disagrees must be REFUSED, not silently reused.
    with gzip.open(tmp, "rt", encoding="utf-8") as fh:
        blob = json.load(fh)
    blob["trajectories"] = blob["trajectories"][:-1]
    blob["pool_fingerprint"] = pool_fingerprint(
        [d["uid"] for d in blob["trajectories"]])
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(blob, fh)
    try:
        ATBenchLoader(cache_path=tmp).load(n_per_class=n, seed=seed)
    except SystemExit as exc:
        assert "CACHE DISAGREES" in str(exc)
        print("OK  a DISAGREEING cache is refused, not reused")
    else:
        raise AssertionError("a disagreeing cache was accepted")
    tmp.unlink()
    p1 = tmp.with_name("_selftest_seed1.json.gz")
    if p1.exists():
        p1.unlink()

    # Turn parsing: agent turns must survive with their action text.
    tr = max(corpus.trajectories, key=lambda t: t.n_turns)
    roles = {u.role for u in tr.turns}
    assert "assistant" in roles, roles
    assert any(u.action for u in tr.turns), "no agent turn kept an action"
    assert all(u.content for u in tr.turns
               if u.role == "assistant"), "an agent turn rendered EMPTY"
    assert [u.index for u in tr.turns] == list(range(tr.n_turns))
    print("OK  turns parse: roles=%s, %d/%d turns carry a tool action"
          % (sorted(roles), sum(1 for u in tr.turns if u.action), tr.n_turns))
    assert "assistant: " in tr.text and len(tr.text) > 200
    print("OK  .text renders agent thought+action (%d chars)" % len(tr.text))

    assert all(t.mistake_step is None for t in corpus.trajectories)
    assert corpus.step_label_provenance.startswith("NONE")
    print("OK  no per-step label is invented")

    print("")
    print("OK -- data.py self-test passed CPU-only, no model, no GPU.")


if __name__ == "__main__":
    _self_test()
