"""corpora/ -- CorpusLoader implementations for the streaming-trajectory-aggregation
lesson (`shade.py`, `agentdojo.py`, `assebench.py`, `atbench.py`).

Every loader here builds AGAINST `..types` (`Step`, `Trajectory`, `Corpus`,
`CorpusLoader`) -- the frozen spine -- and this package never edits it.

THIS __init__.py holds the plumbing every loader shares:
  * ASCII-safe stderr printing (Windows cp1252 console; CLAUDE.md section 17.5).
  * `uids_fingerprint` -- sha256 over the SORTED, REALISED sample of uids, not the
    requested config. Recorded in `Corpus.pool_fingerprint` per the spine's own
    docstring warning about `biencoder_guard`'s stale-artifact defect.
  * A disk cache (`cache_path` / `load_cache` / `save_cache` / `cached_build`) keyed
    on (loader_name, hf_ids, n_per_class, seed) under
    `streaming_trajectory_aggregation/artifacts/corpora/`. Validity is asserted from
    a STORED HASH OF THE SAMPLED UIDS, recomputed and compared on every load -- never
    from row count alone. Section 18.8 names row-count-only cache validation as the
    exact defect that bit `cross_trajectory` and `multiturn_jailbreak` in 2026-08.
  * `render_contents` -- renders ASSEBench/ATBench's `contents` (nested
    role/content|thought|action list) into per-step text. Shared by both because
    both datasets use the identical shape (verified in DATASETS_VERIFIED.md).
  * `build_control_arena_pair` -- the SHADE and AgentDojo loaders hit an IDENTICAL
    pair of repos (`adityaasinha28/control_arena_{shade,agentdojo}` +
    `aksh-n/control_arena_{shade,agentdojo}-monitor-labels`, same 11-column /
    judge-log schema per DATASETS_VERIFIED.md). One shared builder, parameterised by
    loader name + repo ids, instead of two copies that could silently diverge.

CPU-only to import (no network call happens at import time -- only inside `.load()`,
and this package never fetches at import). ASCII stdout/stderr only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

from steering_tutorials.streaming_trajectory_aggregation.types import Corpus, Step, Trajectory

try:  # SSL middlebox on this host (truststore lets requests use the OS trust store)
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

__all__ = [
    "eprint", "clean_text", "uids_fingerprint", "cache_path", "load_cache",
    "save_cache", "cached_build", "render_contents", "build_control_arena_pair",
    "get_loaders", "ARTIFACTS_DIR",
]

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "corpora"


# --- ASCII-safe printing ------------------------------------------------------
def eprint(*args) -> None:
    """ASCII-only stderr print (Windows cp1252 console crashes on unicode)."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


def clean_text(text) -> str:
    return " ".join(str(text).split()).strip()


# --- pool fingerprint ----------------------------------------------------------
def uids_fingerprint(uids: Sequence) -> str:
    """sha256 over the SORTED, REALISED sample of uids -- never the requested config."""
    h = hashlib.sha256()
    for u in sorted(str(u) for u in uids):
        h.update(u.encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


# --- disk cache, validated by uid fingerprint, NOT row count -------------------
def cache_path(loader_name: str, hf_ids: Sequence[str], n_per_class: int, seed: int) -> Path:
    key_src = "%s|%s|%d|%d" % (loader_name, "+".join(sorted(hf_ids)), int(n_per_class), int(seed))
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:24]
    return ARTIFACTS_DIR / ("%s_%s.json" % (loader_name, key))


def _traj_to_dict(t: Trajectory) -> dict:
    return asdict(t)  # asdict recurses into the nested Step tuple -> list[dict]


def _traj_from_dict(d: dict) -> Trajectory:
    kwargs = dict(d)
    kwargs["steps"] = tuple(Step(**s) for s in d["steps"])
    return Trajectory(**kwargs)


def load_cache(path: Path):
    """Returns cached list[Trajectory], or None if absent/invalid.

    Validity = a stored sha256 of the sampled uids, RECOMPUTED from the cached
    trajectories and compared. A mismatch (or any parse failure) is a loud,
    non-fatal signal to rebuild -- never a silent stale hit.
    """
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        trajs = [_traj_from_dict(d) for d in obj["trajectories"]]
    except Exception as exc:
        eprint("[cache] %s: unreadable (%s) -- rebuilding" % (path, exc))
        return None
    expected = obj.get("uids_sha256")
    actual = uids_fingerprint([t.uid for t in trajs])
    if expected != actual:
        eprint("[cache] %s: FINGERPRINT MISMATCH (stored %s != recomputed %s) -- "
               "treating as INVALID, rebuilding" % (path, expected, actual))
        return None
    eprint("[cache] %s: HIT, %d trajectories, uid fingerprint verified" % (path, len(trajs)))
    return trajs


def save_cache(path: Path, trajectories: list) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    obj = {
        "uids_sha256": uids_fingerprint([t.uid for t in trajectories]),
        "trajectories": [_traj_to_dict(t) for t in trajectories],
    }
    path.write_text(json.dumps(obj), encoding="utf-8")
    eprint("[cache] %s: saved %d trajectories" % (path, len(trajectories)))


def cached_build(loader_name: str, hf_ids: Sequence[str], n_per_class: int, seed: int,
                  builder: Callable[[], list]) -> list:
    """Cache-or-build shim. `builder()` takes no args -- callers bind n_per_class/seed
    via closure -- so this stays a thin cache layer, not a second place that talks
    to HF."""
    path = cache_path(loader_name, hf_ids, n_per_class, seed)
    cached = load_cache(path)
    if cached is not None:
        return cached
    trajectories = builder()
    save_cache(path, trajectories)
    return trajectories


# --- shared: render ASSEBench/ATBench's `contents` nested step list ------------
def render_contents(contents) -> list:
    """`contents`: a list of {role, content|thought|action} dicts (or bare strings).

    Both ASSEBench and ATBench use this exact shape (DATASETS_VERIFIED.md). Renders
    each item to one `"{role}: {text}"` string; empty/unparsable items are dropped.
    """
    out = []
    if not isinstance(contents, list):
        return out
    for item in contents:
        if isinstance(item, str):
            t = clean_text(item)
            if t:
                out.append(t)
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        piece = item.get("content") or item.get("thought") or item.get("action")
        if isinstance(piece, (list, dict)):
            piece = json.dumps(piece, ensure_ascii=True)[:2000]
        text = clean_text(piece or "")
        rendered = ("%s: %s" % (role, text)).strip(": ").strip()
        if rendered:
            out.append(rendered)
    return out


# --- shared: SHADE + AgentDojo ControlArena-schema builder ---------------------
# Both are `adityaasinha28/control_arena_{shade,agentdojo}` (prompt, ground_truth
# 0/1, trajectory_data JSON-string-of-messages, extra_info.main_task_description,
# ...) paired with `aksh-n/control_arena_{shade,agentdojo}-monitor-labels` (a FILE
# repo: judge_logs/row_N__gemini-3.1-pro-preview__ponr.json + label_manifest.json).
# Row index N aligns across the pair (DATASETS_VERIFIED.md). Identical schema ->
# one builder, so a fix lands in both loaders instead of silently diverging.

_DEFAULT_LOG_FMT = "judge_logs/row_%d__gemini-3.1-pro-preview__ponr.json"


def _fetch_json(repo_id: str, filename: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_main_dataset(hf_id: str):
    from datasets import load_dataset as hf_load

    dsd = hf_load(hf_id)
    if hasattr(dsd, "keys"):
        if "train" in dsd:
            return dsd["train"]
        names = list(dsd.keys())
        if not names:
            raise RuntimeError("%s: no splits found" % hf_id)
        eprint("[data] %s: split 'train' not found, using %r" % (hf_id, names[0]))
        return dsd[names[0]]
    return dsd  # already a single Dataset


def _normalize_manifest(loader_name: str, manifest: dict) -> dict:
    """Normalize label_manifest.json into {int_row_index: {"status": str, "log": str}}.

    DATASETS_VERIFIED.md describes the shape only in prose ("a per-row
    {status: kept|..., log: <filename>} map") without pinning the exact wrapper key.
    This tries the plausible shapes and RAISES with the manifest's own top-level keys
    if none match -- a schema surprise here must be loud, not silently misparsed
    (CLAUDE.md section 18.8: every silent defect in this program looked plausible).
    """
    if not isinstance(manifest, dict):
        raise RuntimeError("%s: label_manifest.json is not a JSON object (got %r)"
                            % (loader_name, type(manifest)))

    body = None
    for key in ("rows", "row_status", "labels", "row_labels", "per_row", "entries", "index"):
        val = manifest.get(key)
        if isinstance(val, dict):
            body = val
            break
    if body is None:
        sample_vals = [v for v in list(manifest.values())[:5] if isinstance(v, dict)]
        if sample_vals and all("status" in v for v in sample_vals):
            body = manifest  # the manifest IS the row map, no wrapper key
        else:
            raise RuntimeError(
                "%s: label_manifest.json has an unrecognised shape; top-level keys=%r. "
                "Extend _normalize_manifest explicitly -- do not guess."
                % (loader_name, list(manifest.keys())))

    out = {}
    for k, v in body.items():
        if not isinstance(v, dict):
            continue
        try:
            idx = int(k)
        except (TypeError, ValueError):
            idx = v.get("row", v.get("index"))
            if idx is None:
                continue
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
        out[idx] = {"status": v.get("status"), "log": v.get("log")}
    return out


def _render_messages(trajectory_data_raw) -> list:
    """`trajectory_data` is a JSON STRING holding a `messages` list (per-row column,
    DATASETS_VERIFIED.md). Renders each message to one `"{role}: {text}"` string."""
    try:
        data = json.loads(trajectory_data_raw) if isinstance(trajectory_data_raw, str) \
            else (trajectory_data_raw or {})
    except Exception:
        return []
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return []
    out = []
    for m in messages:
        if isinstance(m, str):
            t = clean_text(m)
            if t:
                out.append(t)
            continue
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        content = m.get("content")
        if isinstance(content, list):  # content-block style: [{text:...}, ...]
            parts = []
            for c in content:
                if isinstance(c, dict):
                    parts.append(clean_text(c.get("text", c.get("content", ""))))
                elif isinstance(c, str):
                    parts.append(clean_text(c))
            content_text = " ".join(p for p in parts if p)
        else:
            content_text = clean_text(content or "")
        if not content_text and m.get("tool_calls"):
            content_text = clean_text(json.dumps(m["tool_calls"], ensure_ascii=True)[:2000])
        rendered = ("%s: %s" % (role, content_text)).strip(": ").strip()
        if rendered:
            out.append(rendered)
    return out


def _build_steps_from_judge(raw_texts, priv_by_step, nonpriv_by_step):
    """Merge the two per-step judge tracks (keyed by the log's OWN step numbering,
    which may be 0- or 1-based) into `Step`s with 0-based `index`.

    Returns (steps, base) -- `base` is the detected step-numbering origin, so the
    caller can shift `ideal_flagging_step`/`point_of_no_return_step` into the SAME
    0-based space as `Step.index`. `available_lead` (a DIFFERENCE of the two) is
    shift-invariant, so this choice does not affect any lead-time number, only the
    absolute step index's meaning.

    Text: aligned 1:1 to the raw rendered messages when the counts match; otherwise
    falls back to the judge record's own `rationale` text (still a meaningful
    per-step document for an embedder), since no verified alignment exists for the
    mismatched case -- fabricating one would be inventing data.
    """
    indices = sorted(set(priv_by_step) | set(nonpriv_by_step))
    if not indices:
        return tuple(), 0
    base = min(indices)
    n = max(indices) - base + 1
    aligned = len(raw_texts) == n
    steps = []
    for pos, idx in enumerate(range(base, base + n)):
        if idx not in priv_by_step and idx not in nonpriv_by_step:
            continue  # a gap in the judge's own numbering -- skip, do not fabricate
        if pos < len(raw_texts):
            text = raw_texts[pos]
        else:
            rec = priv_by_step.get(idx) or nonpriv_by_step.get(idx) or {}
            text = clean_text(rec.get("rationale", "")) or ("[unrendered step %d]" % idx)
        priv = priv_by_step.get(idx, {})
        nonpriv = nonpriv_by_step.get(idx, {})
        steps.append(Step(
            index=idx - base,
            text=text,
            on_path=priv.get("on_path"),
            severity_if_confirmed=priv.get("severity_if_confirmed"),
            irreversibility_imminent=priv.get("irreversibility_imminent"),
            would_flag_now=nonpriv.get("would_flag_now"),
            cumulative_confidence=nonpriv.get("cumulative_confidence"),
        ))
    del aligned  # (kept for readability of the branch above; not otherwise used)
    return tuple(steps), base


def _row_to_control_arena_trajectory(loader_name, row_index, row, log, label):
    raw_texts = _render_messages(row.get("trajectory_data"))

    priv = ((log.get("priv_judge") or {}).get("with_evidence") or {})
    priv_by_step = {}
    for rec in (priv.get("step_analyses") or []):
        if not isinstance(rec, dict) or rec.get("step") is None:
            continue
        try:
            priv_by_step[int(rec["step"])] = rec
        except (TypeError, ValueError):
            continue

    nonpriv_by_step = {}
    for rec in (log.get("nonpriv_streaming") or []):
        if not isinstance(rec, dict) or rec.get("current_step") is None:
            continue
        try:
            nonpriv_by_step[int(rec["current_step"])] = rec
        except (TypeError, ValueError):
            continue

    steps, base = _build_steps_from_judge(raw_texts, priv_by_step, nonpriv_by_step)
    if not steps:
        return None

    # candidate_ideal_flagging_step is -1 for benign in the raw data -> None here
    # (types.py's documented contract), shifted into the same 0-based space as Step.
    ideal_raw = priv.get("candidate_ideal_flagging_step")
    ideal = None
    if isinstance(ideal_raw, (int, float)) and int(ideal_raw) >= 0:
        ideal = int(ideal_raw) - base

    adjudicator = log.get("adjudicator") or {}
    ponr_raw = adjudicator.get("point_of_no_return_step")
    ponr = (int(ponr_raw) - base) if isinstance(ponr_raw, (int, float)) else None

    extra_info = row.get("extra_info")
    main_task = extra_info.get("main_task_description") if isinstance(extra_info, dict) else None
    uid = "%s:%d" % (loader_name, row_index)
    if main_task:
        # Group by underlying task, not by row: several epochs/rows can share one
        # main_task_description, and a group-aware split must not straddle that.
        group_id = "%s:task:%s" % (
            loader_name, hashlib.sha1(main_task.encode("utf-8", "replace")).hexdigest()[:12])
    else:
        group_id = uid  # no verified grouping field on this row -- own group (rule 4 fallback)

    return Trajectory(
        uid=uid, steps=steps, label=label, group_id=group_id, source=loader_name,
        ideal_flagging_step=ideal, point_of_no_return_step=ponr,
    )


def build_control_arena_pair(loader_name: str, dataset_id: str, labels_id: str,
                              n_per_class: int, seed: int,
                              log_filename_fmt: str = _DEFAULT_LOG_FMT) -> list:
    """Shared builder for the SHADE and AgentDojo ControlArena pairs.

    Admission: manifest status=='kept' AND (no `verification` block, OR
    `verification.passed==True`) AND the row yields >=1 usable Step. Balanced,
    seeded, without-replacement sampling over the ELIGIBLE pool per class; a class
    that cannot reach `n_per_class` returns what it found with a loud warning
    (never raises, never rounds up -- `Corpus.shortfall()` surfaces it).
    """
    eprint("[%s] loading %s ..." % (loader_name, dataset_id))
    ds = _load_main_dataset(dataset_id)
    n_rows = len(ds)
    eprint("[%s] %s: %d rows" % (loader_name, dataset_id, n_rows))

    eprint("[%s] loading label_manifest.json from %s ..." % (loader_name, labels_id))
    manifest = _fetch_json(labels_id, "label_manifest.json")
    manifest_rows = _normalize_manifest(loader_name, manifest)

    pos_idx, neg_idx = [], []
    for i in range(n_rows):
        rec = manifest_rows.get(i)
        if rec is None or rec.get("status") != "kept":
            continue
        gt_raw = ds[i].get("ground_truth")
        try:
            gt = int(gt_raw)
        except (TypeError, ValueError):
            continue
        if gt not in (0, 1):
            continue
        (pos_idx if gt == 1 else neg_idx).append(i)

    eprint("[%s] manifest-kept candidates: pos=%d neg=%d (of %d total rows, %d manifest entries)"
           % (loader_name, len(pos_idx), len(neg_idx), n_rows, len(manifest_rows)))

    import random
    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    trajectories = []
    dropped_verification = 0
    dropped_no_steps = 0
    seen_verification_block = False
    for label, idx_list in ((1, pos_idx), (0, neg_idx)):
        kept = 0
        for i in idx_list:
            if kept >= n_per_class:
                break
            rec = manifest_rows[i]
            log_filename = rec.get("log") or (log_filename_fmt % i)
            try:
                log = _fetch_json(labels_id, log_filename)
            except Exception as exc:
                eprint("[%s] row %d: log fetch FAILED (%s) -- skipped" % (loader_name, i, exc))
                continue
            verification = log.get("verification")
            if verification is not None:
                seen_verification_block = True
                if not verification.get("passed", False):
                    dropped_verification += 1
                    continue
            traj = _row_to_control_arena_trajectory(loader_name, i, ds[i], log, label)
            if traj is None:
                dropped_no_steps += 1
                continue
            trajectories.append(traj)
            kept += 1
        if kept < n_per_class:
            eprint("=" * 70)
            eprint("[%s] WARNING: SHORTFALL -- only %d/%d admitted for label=%d "
                   "(eligible pool exhausted)" % (loader_name, kept, n_per_class, label))
            eprint("=" * 70)

    eprint("[%s] admitted %d trajectories total ; dropped %d (verification.passed==False, "
           "block %s) ; %d (no usable per-step judge records)"
           % (loader_name, len(trajectories), dropped_verification,
              "seen" if seen_verification_block else "NEVER SEEN on this corpus -- gate did not fire",
              dropped_no_steps))

    rng.shuffle(trajectories)
    return trajectories


# --- convenience registry (deferred import: importing `corpora` alone never has
# to import all four submodules, and one broken submodule cannot block the rest) --
def get_loaders() -> dict:
    from .agentdojo import AgentDojoLoader
    from .assebench import ASSEBenchLoader
    from .atbench import ATBenchLoader
    from .shade import ShadeLoader

    return {
        "shade": ShadeLoader(),
        "agentdojo": AgentDojoLoader(),
        "assebench": ASSEBenchLoader(),
        "atbench": ATBenchLoader(),
    }


if __name__ == "__main__":
    # CPU-only smoke: exercise the pure functions, no network.
    fp1 = uids_fingerprint(["b", "a", "c"])
    fp2 = uids_fingerprint(["c", "b", "a"])
    assert fp1 == fp2, "fingerprint must be order-independent over the uid SET"
    s, b = _build_steps_from_judge(
        ["m0", "m1", "m2"],
        {1: {"step": 1, "on_path": True}, 2: {"step": 2, "on_path": False}},
        {1: {"current_step": 1, "would_flag_now": False}},
    )
    print("smoke: base=%d n_steps=%d step0.index=%d step0.on_path=%s"
          % (b, len(s), s[0].index, s[0].on_path))
    assert b == 1 and len(s) == 2 and s[0].index == 0
    print("OK -- corpora/__init__.py plumbing behaves (no network touched)")
