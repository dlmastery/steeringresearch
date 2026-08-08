"""data.py — build the per-token hidden-state trajectories the TRAJGUARD lesson
detects on, on a SUBSTRATE chosen to make the paper's claim testable.

For each prompt we run the abliterated Gemma-3-1B, greedily generate a completion,
and capture the layer-``C.LAYER`` residual-stream state at every GENERATED-token
position -> one trajectory ``[n_tokens, dim]``. Label = the prompt's class
(1 = harmful, 0 = benign). This is the sibling of ``multiturn_jailbreak.data``
(there a chunk is a conversation turn; here a chunk is a decoded token).

THE SUBSTRATE (see config.SUBSTRATE)
------------------------------------
toxic-chat 0124 carries TWO human prompt-level binary labels, ``toxicity`` and
``jailbreaking``. The shared ``common/data.py`` reads only ``toxicity``, so the
disguised-attack subset -- the only stratum on which a decoding-time monitor can
possibly beat a prompt-side one -- was being discarded at load time.

``common/`` is SHARED and this lesson does NOT edit it. Instead we read the same CSV
through the same ``hf_hub_download`` path and apply ``common.data``'s own primitives
(``_clean`` / ``_norm_key`` / ``_group_id`` / ``_stratified_sample`` /
``_length_matched_sample`` and the ``MIN_CHARS`` / ``MAX_CHARS`` filters), so the
dedup, grouping, ambiguity-dropping and length-matching discipline is bit-identical to
every other lesson -- only the label expression differs.

  overt      toxicity==1 AND jailbreaking==0 -- 512 unique. Clears the >=500 floor.
  disguised  jailbreaking==1                 -- 181 unique. POOL-LIMITED, said so.
  mixed      toxicity==1                     -- 693 unique. The LEGACY substrate.

Benign is always ``toxicity==0 AND jailbreaking==0``, sampled **length-matched**
(decile-bin stratified) to the harmful class -- so raw prompt length cannot separate
the classes. NOTE, because the superseded README got this exactly backwards: that
control makes ``prompt_charlen`` land near 0.50 BY CONSTRUCTION. It is a control that
was APPLIED, not a discovery about the data, and it says nothing whatever about
whether prompt *content* separates the classes. It does -- see
:func:`prompt_confound_report`.

CACHE FORMAT (``C.TRAJ_CACHE``, a ragged compressed .npz pack)
--------------------------------------------------------------
  - ``flat``          : float16 ``[sum(token_counts), dim]`` = vstack of all token vecs
  - ``token_counts``  : int ``[n_completions]``  = per-completion token count
  - ``labels``        : int ``[n_completions]``
  - ``dim``           : int scalar
  - ``fingerprint``   : the sha256 below, as a 0-d unicode array
A JSON sidecar next to the npz stores ``prompts``, ``completions`` and the full config
snapshot. On load we split ``flat`` back into a list of ``[n_tokens, dim]`` arrays via
``token_counts`` -- and we **assert the fingerprint**. A cache built under a different
``n_per_class`` / ``max_new_tokens`` / ``layer`` / ``model_id`` / ``substrate`` / seed,
or over a different set of prompts, is REFUSED with a diff, never silently reused.
That silent reuse is how a run configured for 500/class shipped ``n_harmful: 300``.

CPU-only to write/import. The single allowed model load is ``build_token_trajectories``
(plus the ``__main__`` smoke). ASCII stdout only (Windows cp1252 console).
"""
from __future__ import annotations

try:  # OS trust store for an SSL-intercepting middlebox (same guard as siblings).
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional
    pass

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

from steering_tutorials.common import confound as CF

from . import config as C

# toxic-chat 0124 full set, via the same pinned constants common/data.py uses.
_TOXICCHAT_SPLIT = "all"


class CacheMismatch(RuntimeError):
    """A cache exists but was NOT built for this configuration.

    Raised, never swallowed. CLAUDE.md section 18.8: an anchor that matches nothing
    must FAIL, not pass -- every serious defect in this program failed silently and
    plausibly, and a config-blind cache is the exact shape of that failure.
    """


# --------------------------------------------------------------------------- #
# 1. Prompt selection -- substrate-aware, built from common/data.py primitives
# --------------------------------------------------------------------------- #
def _toxicchat_rows(split: str = _TOXICCHAT_SPLIT):
    """Deduped toxic-chat rows carrying BOTH label columns.

    Returns ``(rows, stats)`` where each row is
    ``{prompt, group_id, toxicity, jailbreaking, category, source}`` and a prompt whose
    NORMALIZED text appears under conflicting values of either label is dropped as
    ambiguous (the same guard ``common.data`` applies to ``toxicity`` alone).
    """
    import pandas as pd

    from steering_tutorials.common.data import (
        MAX_CHARS,
        MIN_CHARS,
        SRC_TOXICCHAT,
        TOXICCHAT_FILES,
        TOXICCHAT_REPO,
        _clean,
        _coarse_category,
        _dl,
        _group_id,
        _norm_key,
    )

    df = pd.read_csv(_dl(TOXICCHAT_REPO, TOXICCHAT_FILES[split]))
    if "jailbreaking" not in df.columns:
        raise RuntimeError(
            "toxic-chat release %r has no 'jailbreaking' column (columns=%s). This "
            "lesson's disguised-attack substrate requires it." % (split, list(df.columns)))

    tox = pd.to_numeric(df["toxicity"], errors="coerce").fillna(0).astype(int)
    jb = pd.to_numeric(df["jailbreaking"], errors="coerce").fillna(0).astype(int)
    mod = df["openai_moderation"] if "openai_moderation" in df.columns else [None] * len(df)

    groups: dict = {}
    n_after_filters = 0
    for raw, t, j, modcell in zip(df["user_input"].astype(str), tox, jb, mod):
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
            groups[gid] = {
                "display": display,
                "tox": {int(t)},
                "jb": {int(j)},
                "cat": _coarse_category(modcell) if int(t) == 1 else "benign",
            }
        else:
            g["tox"].add(int(t))
            g["jb"].add(int(j))

    rows, n_ambiguous = [], 0
    for gid, g in groups.items():
        if len(g["tox"]) > 1 or len(g["jb"]) > 1:
            n_ambiguous += 1
            continue
        rows.append({
            "prompt": g["display"],
            "group_id": gid,
            "toxicity": next(iter(g["tox"])),
            "jailbreaking": next(iter(g["jb"])),
            "category": g["cat"],
            "source": SRC_TOXICCHAT,
        })

    stats = {
        "n_rows_raw": int(len(df)),
        "n_toxic_raw": int((tox == 1).sum()),
        "n_jailbreaking_raw": int((jb == 1).sum()),
        "natural_toxic_rate": round(float((tox == 1).mean()), 4),
        "natural_jailbreaking_rate": round(float((jb == 1).mean()), 4),
        "n_dropped_duplicates": int(n_after_filters - len(groups)),
        "n_dropped_ambiguous": int(n_ambiguous),
    }
    return rows, stats


def _positive_filter(substrate: str):
    """The label expression that defines the harmful class for each substrate."""
    if substrate == "overt":
        return lambda r: r["toxicity"] == 1 and r["jailbreaking"] == 0
    if substrate == "disguised":
        return lambda r: r["jailbreaking"] == 1
    if substrate == "mixed":
        return lambda r: r["toxicity"] == 1
    raise ValueError("unknown substrate %r (expected one of %s)"
                     % (substrate, list(C.SUBSTRATES)))


def select_prompts(substrate: str = C.SUBSTRATE, n_per_class: int = C.N_PER_CLASS,
                   seed: int = C.SEED) -> dict:
    """Choose the prompts for one substrate. CPU-only, deterministic, no model.

    Harmful = the substrate's label expression, harm-category stratified.
    Benign   = ``toxicity==0 AND jailbreaking==0``, **length-matched** (decile-bin
    stratified) to the harmful sample via ``common.data._length_matched_sample``.

    Returns ``{"harmful": rows, "benign": rows, "header": {...}}``. The header records
    the pool ceiling, the requested n, the effective target and whether the shortfall
    (if any) is POOL-LIMITED -- the rule-2 distinction that decides whether a shortfall
    is a documented ceiling or a silent bug. The caller must not have to infer it.
    """
    from steering_tutorials.common.data import _length_matched_sample, _median_len, _stratified_sample

    rng = random.Random(seed)
    rows, stats = _toxicchat_rows()
    keep = _positive_filter(substrate)

    harmful_pool = [r for r in rows if keep(r)]
    benign_pool = [r for r in rows if r["toxicity"] == 0 and r["jailbreaking"] == 0]

    pool = min(len(harmful_pool), len(benign_pool))
    target = min(int(n_per_class), pool)
    pool_limited = pool < int(n_per_class)

    harmful = _stratified_sample(harmful_pool, lambda r: r["category"], target, rng)
    benign = _length_matched_sample(benign_pool, harmful, target, rng)

    from collections import Counter
    header = {
        "substrate": substrate,
        "substrate_definition": {
            "overt": "toxicity==1 AND jailbreaking==0 (overtly toxic user input)",
            "disguised": "jailbreaking==1 (an attack wrapper around harmful intent)",
            "mixed": "toxicity==1 (LEGACY: overt + disguised pooled together)",
        }[substrate],
        "n_per_class_requested": int(n_per_class),
        "n_pool_harmful": len(harmful_pool),
        "n_pool_benign": len(benign_pool),
        "n_effective_target": int(target),
        "n_harmful": len(harmful),
        "n_benign": len(benign),
        "pool_limited": bool(pool_limited),
        "rule1_floor": int(C.RULE1_FLOOR),
        "rule1_compliant": bool(len(harmful) >= C.RULE1_FLOOR and len(benign) >= C.RULE1_FLOOR),
        "per_category_counts_harmful": dict(Counter(r["category"] for r in harmful)),
        "median_char_length": {"harmful": _median_len(harmful), "benign": _median_len(benign)},
        "length_confound_note":
            "benign is length-matched (decile-bin stratified) to harmful, so a near-0.50 "
            "prompt-LENGTH AUC is a control that was APPLIED, not a property of the data, "
            "and implies nothing about prompt CONTENT -- see prompt_confound_report()",
        "natural_base_rates": {
            "toxic_fraction": stats["natural_toxic_rate"],
            "jailbreaking_fraction": stats["natural_jailbreaking_rate"],
        },
        "source_stats": stats,
        "seed": int(seed),
    }
    if pool_limited:
        print("[trajguard.data] POOL-LIMITED: substrate=%s has %d harmful / %d benign "
              "unique prompts; requested %d/class, using %d/class. This is a documented "
              "corpus ceiling (CLAUDE.md section 17 rule 2), NOT a compute choice."
              % (substrate, len(harmful_pool), len(benign_pool), n_per_class, target),
              file=sys.stderr)
    if not header["rule1_compliant"]:
        print("[trajguard.data] RULE-1 SHORTFALL: %d/class < the %d/class floor. Every "
              "number from this arm is PROVISIONAL and must be labelled so."
              % (target, C.RULE1_FLOOR), file=sys.stderr)
    return {"harmful": harmful, "benign": benign, "header": header}


# --------------------------------------------------------------------------- #
# 2. The config fingerprint -- the anchor that must FAIL rather than pass
# --------------------------------------------------------------------------- #
def config_snapshot(substrate: str, n_per_class: int, seed: int, max_new_tokens: int,
                    layer: int, model_id: str) -> dict:
    """Everything that changes the bytes in the cache, as a plain dict."""
    return {
        "substrate": str(substrate),
        "n_per_class": int(n_per_class),
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "layer": int(layer),
        "model_id": str(model_id),
        "greedy": bool(C.GREEDY),
    }


def dataset_fingerprint(snapshot: dict, group_ids) -> str:
    """sha256 over the config snapshot AND the sorted sampled prompt group_ids.

    Including the prompt ids is what makes this a real guard: a change to the shared
    loader, the seed, or the pool reshuffles WHICH prompts were used while leaving every
    scalar knob identical. ``cross_trajectory`` shipped exactly that bug (cached vectors
    silently reused against new labels), and it is CLAUDE.md section 18.8's canonical
    example. Recomputing the ids is CPU-only and takes seconds, so there is no excuse
    for keying on row count.
    """
    payload = json.dumps({"config": snapshot, "group_ids": sorted(str(g) for g in group_ids)},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sidecar_path(cache_path) -> Path:
    """JSON sidecar (prompts + completions + config) living next to the .npz cache."""
    return Path(cache_path).with_suffix(".json")


# --------------------------------------------------------------------------- #
# 3. Cache I/O
# --------------------------------------------------------------------------- #
def _save_cache(cache_path, trajectories, labels, prompts, completions,
                snapshot: dict, fingerprint: str, group_ids, header: dict) -> None:
    """Write the ragged compressed .npz pack + the JSON sidecar + the committed meta.

    Stored **float16**: these are layer-12 residual states used for a diff-of-means
    projection and a GRU, not for anything needing float32 precision, and it halves a
    cache that was 101.18 MiB -- over GitHub's 100 MiB hard per-file limit -- in its
    float32 form. It is still not committed (see README section 10); the text-free
    ``META_PATH`` sidecar is what ships.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    token_counts = np.asarray([t.shape[0] for t in trajectories], dtype=np.int64)
    if trajectories:
        flat = np.vstack([np.asarray(t, dtype=np.float32) for t in trajectories])
        dim = int(trajectories[0].shape[1])
    else:  # nothing captured — keep a well-formed empty pack
        flat = np.zeros((0, 0), dtype=np.float32)
        dim = 0
    np.savez_compressed(
        cache_path,
        flat=flat.astype(np.float16),
        token_counts=token_counts,
        labels=np.asarray(labels, dtype=np.int64),
        dim=np.asarray(dim, dtype=np.int64),
        fingerprint=np.asarray(str(fingerprint)),
    )
    _sidecar_path(cache_path).write_text(
        json.dumps({
            "prompts": list(prompts),
            "completions": list(completions),
            "config_snapshot": snapshot,
            "fingerprint": str(fingerprint),
            "group_ids": [str(g) for g in group_ids],
        }),
        encoding="utf-8",
    )

    size = cache_path.stat().st_size if cache_path.exists() else 0
    if size > C.CACHE_SIZE_WARN_BYTES:
        print("[trajguard.data] NOTE: cache is %.1f MiB, above the %.0f MiB advisory "
              "(GitHub's hard per-file limit is 100 MiB). It is gitignored by design; "
              "the committed artifact is %s."
              % (size / 1048576.0, C.CACHE_SIZE_WARN_BYTES / 1048576.0,
                 Path(C.META_PATH).name), file=sys.stderr)

    write_meta(C.META_PATH, trajectories, labels, completions, group_ids,
               snapshot, fingerprint, header, cache_bytes=size)


def write_meta(meta_path, trajectories, labels, completions, group_ids,
               snapshot: dict, fingerprint: str, header: dict, cache_bytes: int = 0) -> None:
    """Write the committed, TEXT-FREE reproduction sidecar.

    The 100 MB hidden-state cache cannot live in git, but everything a reader needs to
    recompute the NUMERIC confound bars and to verify that a regenerated cache is the
    same set can: per-completion label, prompt group_id, token count, completion
    character length, mean and final hidden-state norm -- plus the fingerprint.

    Deliberately carries NO raw text. The prompts are recoverable from the public
    dataset through :func:`select_prompts` (deterministic), and the completions are
    abliterated-model generations on harmful prompts, which should not be republished.
    """
    trajs = [np.asarray(t, dtype=np.float32) for t in trajectories]
    recs = []
    for i, t in enumerate(trajs):
        t2 = t[None, :] if t.ndim == 1 else t
        norms = np.linalg.norm(t2, axis=-1) if t2.size else np.zeros(1)
        recs.append({
            "label": int(labels[i]),
            "group_id": str(group_ids[i]) if i < len(group_ids) else "",
            "token_count": int(t2.shape[0]),
            "completion_chars": int(len(str(completions[i]))) if i < len(completions) else 0,
            "mean_norm": float(norms.mean()),
            "final_norm": float(np.linalg.norm(t2[-1])) if t2.size else 0.0,
        })
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_path).write_text(json.dumps({
        "fingerprint": str(fingerprint),
        "config_snapshot": snapshot,
        "dataset_header": header,
        "cache_file": Path(C.TRAJ_CACHE).name,
        "cache_bytes": int(cache_bytes),
        "note": "TEXT-FREE reproduction sidecar. Regenerate the hidden-state cache with "
                "`python -m steering_tutorials.trajguard.run_trajguard` and verify the "
                "fingerprint matches; the numeric confound bars recompute from this file "
                "alone with no GPU.",
        "completions": recs,
    }, indent=2), encoding="utf-8")


def _load_cache(cache_path):
    """Load the ragged pack -> dataset dict. Returns ``None`` if either file is absent."""
    cache_path = Path(cache_path)
    sidecar = _sidecar_path(cache_path)
    if not (cache_path.exists() and sidecar.exists()):
        return None
    with np.load(cache_path, allow_pickle=False) as z:
        flat = z["flat"].astype(np.float32)
        token_counts = z["token_counts"].astype(int)
        labels = z["labels"].astype(int)
        fp = str(z["fingerprint"]) if "fingerprint" in z.files else ""
    meta = json.loads(sidecar.read_text(encoding="utf-8"))

    trajectories, start = [], 0
    for n in token_counts:
        n = int(n)
        trajectories.append(flat[start:start + n])
        start += n
    return {
        "trajectories": trajectories,
        "labels": [int(x) for x in labels],
        "prompts": list(meta.get("prompts", [])),
        "completions": list(meta.get("completions", [])),
        "group_ids": list(meta.get("group_ids", [])),
        "fingerprint": fp or str(meta.get("fingerprint", "")),
        "config_snapshot": meta.get("config_snapshot", {}),
    }


def _mean_len(trajectories, labels, want):
    """Mean trajectory length over completions whose label == ``want``."""
    lens = [t.shape[0] for t, y in zip(trajectories, labels) if y == want]
    return float(np.mean(lens)) if lens else 0.0


# --------------------------------------------------------------------------- #
# 4. Build / load
# --------------------------------------------------------------------------- #
def build_token_trajectories(
    substrate: str = C.SUBSTRATE,
    n_per_class: int = C.N_PER_CLASS,
    seed: int = C.SEED,
    max_new_tokens: int = C.MAX_NEW_TOKENS,
    layer: int = C.LAYER,
    model_id: str = C.MODEL_ID,
    cache_path=C.TRAJ_CACHE,
) -> dict:
    """Generate + capture per-token trajectories for one substrate. LOADS THE MODEL.

    Always regenerates -- the cache decision belongs to :func:`load_or_build`, which
    checks the fingerprint. Skips any trajectory with 0 captured tokens (early EOS) and
    reports the count, because a skip is a real (if small) departure from the requested
    n and must be visible rather than absorbed.
    """
    from steering_tutorials.hello_world_steering.model_utils import load_model

    from . import trajectory  # lazy: keeps `import ...data` model-free

    sel = select_prompts(substrate, n_per_class, seed)
    header = sel["header"]
    items = ([(r["prompt"], r["group_id"], 1) for r in sel["harmful"]]
             + [(r["prompt"], r["group_id"], 0) for r in sel["benign"]])
    snapshot = config_snapshot(substrate, header["n_effective_target"], seed,
                               max_new_tokens, layer, model_id)
    fingerprint = dataset_fingerprint(snapshot, [g for _, g, _ in items])

    model, tok = load_model(model_id)

    trajectories, labels, prompts, completions, gids = [], [], [], [], []
    n_skipped = 0
    for i, (prompt, gid, label) in enumerate(items):
        completion, traj = trajectory.generate_and_capture(
            model, tok, prompt, max_new_tokens, layer
        )
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:  # early EOS -> nothing to detect on
            n_skipped += 1
            continue
        trajectories.append(traj)
        labels.append(int(label))
        prompts.append(prompt)
        completions.append(completion)
        gids.append(gid)
        if (i + 1) % 20 == 0:
            print("[trajguard.data] %d/%d captured" % (i + 1, len(items)), file=sys.stderr)

    header = dict(header)
    header["n_skipped_empty_trajectory"] = int(n_skipped)
    header["n_captured"] = len(trajectories)
    header["n_captured_harmful"] = sum(1 for y in labels if y == 1)
    header["n_captured_benign"] = sum(1 for y in labels if y == 0)

    _save_cache(cache_path, trajectories, labels, prompts, completions,
                snapshot, fingerprint, gids, header)

    print(
        "[trajguard.data] substrate=%s built %d trajectories (harmful=%d benign=%d, "
        "skipped=%d) | mean_len harmful=%.1f benign=%.1f | cached -> %s"
        % (substrate, len(trajectories), header["n_captured_harmful"],
           header["n_captured_benign"], n_skipped,
           _mean_len(trajectories, labels, 1), _mean_len(trajectories, labels, 0),
           Path(cache_path).name),
        file=sys.stderr,
    )
    return {
        "trajectories": trajectories,
        "labels": labels,
        "prompts": prompts,
        "completions": completions,
        "group_ids": gids,
        "fingerprint": fingerprint,
        "config_snapshot": snapshot,
        "header": header,
    }


def load_or_build(
    substrate: str = C.SUBSTRATE,
    n_per_class: int = C.N_PER_CLASS,
    seed: int = C.SEED,
    max_new_tokens: int = C.MAX_NEW_TOKENS,
    layer: int = C.LAYER,
    model_id: str = C.MODEL_ID,
    cache_path=C.TRAJ_CACHE,
    rebuild: bool = False,
) -> dict:
    """Return the cached dataset iff its FINGERPRINT matches this configuration.

    Recomputes the expected fingerprint from :func:`select_prompts` (CPU-only, seconds)
    and compares. On mismatch this raises :class:`CacheMismatch` with a field-by-field
    diff -- it does not fall through to a silent rebuild and it never returns the stale
    arrays. Pass ``rebuild=True`` (or ``TG_REBUILD=1`` via the runner) to regenerate.

    The superseded ``load_or_build()`` returned the cache whenever ``labels`` was
    non-empty, checking no knob at all. That is why a run configured for 500/class
    produced ``n_harmful: 300``.
    """
    sel = select_prompts(substrate, n_per_class, seed)
    header = sel["header"]
    gids = ([r["group_id"] for r in sel["harmful"]] + [r["group_id"] for r in sel["benign"]])
    snapshot = config_snapshot(substrate, header["n_effective_target"], seed,
                               max_new_tokens, layer, model_id)
    expected = dataset_fingerprint(snapshot, gids)

    cached = None if rebuild else _load_cache(cache_path)
    if cached is not None and cached["labels"]:
        if cached.get("fingerprint") == expected:
            cached["header"] = header
            cached["config_snapshot"] = snapshot
            print("[trajguard.data] cache HIT %s : fingerprint %s (%d completions)"
                  % (Path(cache_path).name, expected[:12], len(cached["labels"])),
                  file=sys.stderr)
            return cached
        old = cached.get("config_snapshot") or {}
        diff = [(k, old.get(k, "<absent>"), v) for k, v in snapshot.items() if old.get(k) != v]
        detail = ("; ".join("%s: cached=%r wanted=%r" % d for d in diff)
                  if diff else "config identical -- the SAMPLED PROMPTS differ "
                               "(shared loader, pool or seed changed)")
        raise CacheMismatch(
            "cache %s was built for a different configuration and will NOT be reused.\n"
            "  cached fingerprint : %s\n  wanted fingerprint : %s\n  %s\n"
            "Delete the cache or re-run with TG_REBUILD=1 to regenerate (needs the GPU)."
            % (Path(cache_path).name, (cached.get("fingerprint") or "<none>")[:16],
               expected[:16], detail))

    return build_token_trajectories(substrate, n_per_class, seed, max_new_tokens,
                                    layer, model_id, cache_path)


# --------------------------------------------------------------------------- #
# 5. The confound audit -- on the shared spine (common/confound.py)
# --------------------------------------------------------------------------- #
def _geometry_bars(trajectories, labels) -> dict:
    """The two bars the shared spine cannot know about: residual-stream MAGNITUDE.

    The siblings audit length and count because their inputs are text. Here the
    detector's input is a matrix of hidden states, so the trivial shortcut is
    geometric: the classes could differ in the SIZE of the residual-stream vectors
    rather than in how those vectors MOVE. ``harm_direction`` is a diff-of-means and a
    mean-norm gap projects straight onto it, so every method here could be reading one
    scalar. Folded directionless, exactly like every spine bar.
    """
    trajs = [np.asarray(t, dtype=np.float32) for t in trajectories]
    trajs = [t[None, :] if t.ndim == 1 else t for t in trajs]
    feats = {
        "mean_norm": [float(np.linalg.norm(t, axis=-1).mean()) if t.size else 0.0 for t in trajs],
        "final_norm": [float(np.linalg.norm(t[-1])) if t.size else 0.0 for t in trajs],
    }
    out = {}
    for name, values in feats.items():
        raw = CF.auc_raw(values, labels)
        pos = [v for v, y in zip(values, labels) if int(y) == 1]
        neg = [v for v, y in zip(values, labels) if int(y) == 0]
        out[name] = {
            "auc_raw": raw,
            "auc": CF.directionless(raw),
            "mean_pos": float(np.mean(pos)) if pos else 0.0,
            "mean_neg": float(np.mean(neg)) if neg else 0.0,
        }
    return out


def _rebind_worst(rep: dict) -> dict:
    """Recompute ``worst_name`` / ``worst_auc`` after adding non-spine bars.

    The shuffle control is a LEAKAGE DIAGNOSTIC and is never a bar to clear -- same
    rule the spine applies.
    """
    bars = {k: v["auc"] for k, v in rep.items()
            if isinstance(v, dict) and "auc" in v and k != "shuffle"}
    if bars:
        rep["worst_name"] = max(bars, key=lambda k: bars[k])
        rep["worst_auc"] = float(bars[rep["worst_name"]])
    else:
        rep["worst_name"], rep["worst_auc"] = "none", 0.5
    return rep


def confound_report(trajectories, labels, completions=None, prompts=None,
                    seed: int = C.SEED) -> dict:
    """The COMPLETION-channel confound audit: what a no-trajectory scalar scores.

    Built on ``common.confound.confound_report`` (the one shared spine) so this lesson
    gets the two controls it was missing -- a **content** bar (TF-IDF unigram centroid
    under 5-fold CV: a "trajectory" detector that cannot beat unigrams is not reading
    trajectories) and a **label-shuffle** control (must land near 0.5; far from it means
    leakage, not signal) -- and adds the two geometry bars the spine cannot know about.

    Bars reported: ``length`` (completion characters), ``count`` (generated tokens),
    ``content`` (completion unigrams), ``mean_norm``, ``final_norm``, plus ``shuffle``.
    ``worst_auc`` is the binding bar; claim only ``method_auc - worst_auc``.

    NOTE this is the COMPLETION channel. The prompt channel -- the one that decides the
    paper's actual comparative claim -- is :func:`prompt_confound_report`, and it is
    reported separately and never folded into this bar, because a prompt-side model is
    a rival METHOD, not a trivial confound.
    """
    labels = [int(y) for y in labels]
    texts = [str(c) for c in (completions if completions is not None else [""] * len(labels))]
    rep = CF.confound_report(texts, labels, units=trajectories, seed=seed,
                             n_folds=C.N_FOLDS,
                             run_content=completions is not None)
    rep.update(_geometry_bars(trajectories, labels))
    rep["channel"] = "completion"
    _rebind_worst(rep)
    if prompts is not None:
        rep["prompt_channel"] = prompt_confound_report(prompts, labels, seed=seed)
    return rep


def format_full_report(rep: dict) -> str:
    """Spine report PLUS this lesson's geometry bars, ASCII only.

    ``common.confound.format_report`` only knows the four spine bars, so printing it
    alone would hide ``mean_norm`` / ``final_norm`` even though they participate in
    ``worst_auc``. A bar that binds the headline but is invisible in the printed audit
    is the same defect class as one that was never computed.
    """
    lines = [CF.format_report(rep)]
    extra = [(n, rep[n]) for n in ("mean_norm", "final_norm")
             if isinstance(rep.get(n), dict)]
    if extra:
        lines.append("  -- hidden-state geometry bars (this lesson's addition) --")
        for name, b in extra:
            lines.append("  %-9s %-9.4f %-9.4f mean pos=%.2f neg=%.2f"
                         % (name, b["auc_raw"], b["auc"], b["mean_pos"], b["mean_neg"]))
        lines.append("  BINDING BAR (all bars, shuffle excluded): %s = %.4f"
                     % (rep.get("worst_name", "none"), rep.get("worst_auc", 0.5)))
    return "\n".join(lines)


def prompt_confound_report(prompts, labels, seed: int = C.SEED) -> dict:
    """The PROMPT-channel audit -- the number that decides the paper's claim.

    THE RETRACTION THIS FUNCTION EXISTS FOR. The superseded README reported
    ``prompt_charlen`` at 0.5032 and concluded "you cannot call this from the prompt
    alone". Both halves are false:

      1. 0.5032 was DESIGNED IN. ``common.data._length_matched_sample`` draws the
         benign class length-matched to the harmful histogram, decile-bin stratified.
         The sampler was written to produce that number.
      2. Length is not content. A bag-of-words model over the SAME prompts separates
         the classes far above chance (measured at build time by this function; on the
         toxic-chat pool ``hello_world`` independently reports a prompt-only probe at
         AUC 0.965).

    So this is not a confound bar. It is a rival METHOD -- the prompt-side classifier
    TrajGuard claims decoding-time states beat -- and the lesson's headline comparison
    is ``trajectory AUC`` vs ``prompt content AUC``, reported side by side.
    """
    labels = [int(y) for y in labels]
    rep = CF.confound_report([str(p) for p in prompts], labels, units=None, seed=seed,
                             n_folds=C.N_FOLDS)
    rep["channel"] = "prompt"
    rep["note"] = ("prompt LENGTH is near 0.50 BY CONSTRUCTION (the benign class is "
                   "length-matched by the shared loader); prompt CONTENT is the real "
                   "prompt-side signal and is a rival method, not a confound")
    return rep


def early_k_confound(trajectories, labels, k: int) -> dict:
    """The confound bar on the FIRST-K truncation -- the streaming headline's own bar.

    README section 10 committed to this and the superseded run never computed it, which
    left the lesson's entire pitch (early detection) as the one unpriced number. On a
    K-token prefix the available no-trajectory scalars are the truncated token count
    (early EOS is class-informative) and the two geometry bars; completion characters
    are not defined for a prefix we never decoded to text, and that is stated rather
    than silently omitted.
    """
    k = max(1, int(k))
    trunc = []
    for t in trajectories:
        a = np.asarray(t, dtype=np.float32)
        a = a[None, :] if a.ndim == 1 else a
        trunc.append(a[:k] if a.shape[0] else a)
    counts = [float(min(a.shape[0], k)) for a in trunc]
    raw = CF.auc_raw(counts, labels)
    rep = {"K": k,
           "count": {"auc_raw": raw, "auc": CF.directionless(raw)},
           "note": "no completion-character or content bar at a prefix (the first-K "
                   "tokens were never decoded to text); count + geometry only"}
    rep.update(_geometry_bars(trunc, labels))
    return _rebind_worst(rep)


# --------------------------------------------------------------------------- #
# 6. CPU smoke -- prompt selection + the prompt-channel bars, NO model
# --------------------------------------------------------------------------- #
def _cpu_smoke() -> None:
    """Selection + prompt-channel audit for every substrate. No model, no GPU."""
    for substrate in C.SUBSTRATES:
        sel = select_prompts(substrate, C.N_PER_CLASS, C.SEED)
        h = sel["header"]
        print("")
        print("=== substrate=%s ===" % substrate)
        print("  %s" % h["substrate_definition"])
        print("  pool harmful=%d benign=%d | requested=%d effective=%d | pool_limited=%s "
              "| rule1(>=%d)=%s"
              % (h["n_pool_harmful"], h["n_pool_benign"], h["n_per_class_requested"],
                 h["n_effective_target"], h["pool_limited"], h["rule1_floor"],
                 h["rule1_compliant"]))
        print("  median chars harmful=%d benign=%d (length-matched by construction)"
              % (h["median_char_length"]["harmful"], h["median_char_length"]["benign"]))
        prompts = [r["prompt"] for r in sel["harmful"]] + [r["prompt"] for r in sel["benign"]]
        labels = [1] * len(sel["harmful"]) + [0] * len(sel["benign"])
        rep = prompt_confound_report(prompts, labels)
        print(CF.format_report(rep))
        print("  READ: prompt LENGTH near 0.50 is the sampler doing its job. Prompt "
              "CONTENT is the rival method the trajectory detector must beat.")


if __name__ == "__main__":
    _cpu_smoke()
