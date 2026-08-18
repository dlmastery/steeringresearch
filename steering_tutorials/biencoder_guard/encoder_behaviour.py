"""encoder_behaviour.py -- the ENCODER-BEHAVIOUR half of an embedding-cache key.

WHY THIS MODULE EXISTS (read `audits/AUDIT_2026-08-17_embeddinggemma_causal.md`)
-------------------------------------------------------------------------------
`google/embeddinggemma-300m` is a BIDIRECTIONAL encoder. Its config sets
`use_bidirectional_attention=True`. transformers 4.55.0 contained ZERO references to
that parameter, so the flag bound to nothing and the model ran STRICTLY CAUSAL --
proved behaviourally, not by reading a warning: two sequences sharing a 9-token
prefix produced BIT-IDENTICAL hidden states over that prefix (max abs diff
0.000000e+00). Under transformers 5.15.0 the same probe returns ~5.74.

Every `.npz` embedding cache written under the broken install therefore holds vectors
computed by an instrument that was not the instrument claimed. The caches hashed the
DATA only, so a re-run AFTER the library fix would have silently reused causal
vectors -- **that is the mechanism by which the bug would have survived its own fix.**
Manual quarantine of the affected files closed the incident; it did not close the
defect. This module closes the defect.

WHAT IS FINGERPRINTED, AND WHY BOTH HALVES ARE NEEDED
-------------------------------------------------------------------------------
  1. LIBRARY VERSIONS (`transformers`, `sentence-transformers`, `torch`) + the model
     id + this module's PROBE_FORMAT_VERSION. Cheap: read from installed-package
     METADATA, so it costs no import of transformers and certainly no model load.
     This is what goes into the cache PATH/key -- caches written under a different
     stack land on a different filename and can never be picked up by accident.
     It is also exactly what would have caught the real incident: 4.55.0 != 5.15.0.

  2. A BEHAVIOURAL PROBE -- the prefix-invariance delta. Versions alone are NOT
     sufficient and the incident is the proof: a config field said "bidirectional"
     and a version string looked fine while the BEHAVIOUR was causal. The probe
     measures the thing itself. It requires a model forward, so it is stored in the
     cache META (auditable without re-deriving it) rather than in the filename, and
     it is computed only when a model is already loaded (see the gating rules below).

THE PROBE
-------------------------------------------------------------------------------
Two sequences share a long token prefix and differ only after it. We read the
BACKBONE's last hidden state (not the pooled embedding, which would confound with
the pooling layer) and take the max abs difference over the SHARED prefix tokens.

  * CAUSAL attention: a prefix token cannot see the suffix -> difference is EXACTLY
    zero -> bucket "causal".
  * BIDIRECTIONAL attention: it must change -> a real, order-1 difference.

The raw delta is a float and would thrash a cache key over run-to-run noise, so the
KEY carries a DECADE BUCKET (`"1e+0"`, `"1e-1"`, ...) while the raw value is stored
alongside for the auditor. A decoder-only embedder (the `gemma` arm) legitimately
buckets as "causal" -- the probe is DESCRIPTIVE, not a pass/fail gate. What it
detects is a CHANGE in behaviour between the write and the read of a cache.

GATING (the rules this module is built to satisfy)
-------------------------------------------------------------------------------
  * NOTHING runs at import time. This module imports numpy-free, torch-free,
    transformers-free; `import encoder_behaviour` loads no model and no heavy lib.
  * `cheap_fingerprint()` never loads a model -- it reads package metadata only.
  * `probe_fingerprint(embedder)` performs ONE forward pass, and is memoised
    per-process per-model, so it costs at most one probe per run.
  * A cache HIT (cheap fields all match) returns WITHOUT probing: the stored
    behaviour block is already attributable to this exact stack.
  * A cache whose stored behaviour block DISAGREES with the current one is rejected
    LOUDLY, naming the field. It is never silently reused, and `quarantine()` moves
    the stale file aside rather than overwriting it.
  * A cache with NO behaviour block at all (written before this module) is rejected
    as unattributable -- the same rule the content fingerprints already apply.

ASCII-only output (Windows cp1252 console).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

__all__ = [
    "PROBE_FORMAT_VERSION", "CAUSAL_EPS", "NOT_PROBED", "UNKNOWN",
    "library_versions", "cheap_fingerprint", "probe_fingerprint",
    "key_component", "compare", "describe_mismatch", "reject_message",
    "quarantine", "prefix_invariance_delta", "delta_bucket", "resolve_backbone",
]

PROBE_FORMAT_VERSION = 1

# Below this, the two prefix hidden states are treated as bit-identical == causal.
# The observed causal value is EXACTLY 0.0; the epsilon is only there so a future
# kernel with a whisper of non-determinism does not read as "bidirectional".
CAUSAL_EPS = 1e-6

NOT_PROBED = "not-probed"   # deliberately skipped (no model was loaded)
UNKNOWN = "unknown"         # attempted and failed (backbone not reachable)

# The probe pair. A long shared prefix, then genuinely different continuations.
PROBE_PREFIX = "The quick brown fox jumps over the lazy dog"
PROBE_SUFFIX_A = " and then sleeps peacefully in the warm afternoon sun."
PROBE_SUFFIX_B = " while ignoring absolutely everything else that happens today."

_PROBE_CACHE: dict[str, dict] = {}   # in-process memo: model identity -> probe block


def _eprint(*args) -> None:
    """ASCII-only stderr print (this host's cp1252 console crashes on unicode)."""
    import sys

    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


# ---------------------------------------------------------------------------
# 1. the CHEAP half -- package metadata only, no imports of the packages
# ---------------------------------------------------------------------------
def _dist_version(dist_name: str) -> str:
    """Installed version of a distribution WITHOUT importing it.

    `importlib.metadata` reads the dist-info on disk. Importing `transformers` here
    would be both slow and, on this host, hazardous (the Keras-3 TF import path that
    killed the first live AgentDojo run). Absent packages report "absent" rather than
    raising -- a MiniLM-only environment is legitimate.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return str(version(dist_name))
        except PackageNotFoundError:
            return "absent"
    except Exception:
        return UNKNOWN


def library_versions() -> dict:
    """The three libraries that decide how an encoder actually attends."""
    return {
        "transformers": _dist_version("transformers"),
        "sentence_transformers": _dist_version("sentence-transformers"),
        "torch": _dist_version("torch"),
    }


def cheap_fingerprint(model_id: str, extra: dict = None) -> dict:
    """The no-model-load behaviour block: versions + model id + probe format.

    This is what a cache READ compares first and what the cache PATH is keyed on.
    `prefix_delta_bucket` is `NOT_PROBED` here BY DESIGN -- a read that matches on
    these fields must not pay for a forward pass.
    """
    block = {
        "probe_version": PROBE_FORMAT_VERSION,
        "model_id": str(model_id),
        "libs": library_versions(),
        "prefix_delta_bucket": NOT_PROBED,
    }
    if extra:
        block.update({str(k): v for k, v in extra.items()})
    return block


def key_component(model_id: str, extra: dict = None) -> str:
    """A short hash of the CHEAP block, for folding into a cache FILENAME.

    Only the cheap half can live in a path: computing the path must never require a
    model forward. The behavioural bucket lives in the meta and is checked there.
    """
    block = cheap_fingerprint(model_id, extra)
    block.pop("prefix_delta_bucket", None)   # NOT_PROBED is a placeholder, not identity
    payload = json.dumps(block, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 2. the BEHAVIOURAL half -- one forward pass, memoised per process
# ---------------------------------------------------------------------------
def resolve_backbone(obj):
    """-> (auto_model, tokenizer) for the many wrappers this repo hands around.

    Accepts a `SentenceTransformer`, either lesson's embedder wrapper (which holds
    the model on `._model`), or a bare HF model paired with `._tok`. Returns
    (None, None) when the backbone is not reachable; callers turn that into the
    explicit `UNKNOWN` bucket rather than into a silent pass.
    """
    inner = getattr(obj, "_model", obj)

    # Decoder arm: a bare HF model with the tokenizer alongside it.
    tok = getattr(obj, "_tok", None) or getattr(obj, "tokenizer", None)
    auto = None

    # sentence-transformers: module 0 is the Transformer wrapper.
    try:
        first = inner[0]
        auto = getattr(first, "auto_model", None)
        tok = getattr(first, "tokenizer", None) or tok
    except Exception:
        pass
    if auto is None:
        auto = getattr(inner, "auto_model", None)
    if auto is None and hasattr(inner, "config") and hasattr(inner, "forward"):
        auto = inner            # already a bare HF model
    if tok is None:
        tok = getattr(inner, "tokenizer", None)
    return auto, tok


def _model_identity(auto_model, fallback: str) -> str:
    cfg = getattr(auto_model, "config", None)
    name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
    return "%s|%s" % (type(auto_model).__name__, name or fallback)


def prefix_invariance_delta(auto_model, tokenizer) -> tuple[float, int]:
    """max |h_a - h_b| over the SHARED prefix tokens of the two probe strings.

    Reads the BACKBONE's last hidden state via `output_hidden_states=True` (works for
    encoder and decoder alike; `last_hidden_state` is not universal). Pooling is
    deliberately bypassed -- pooled vectors would confound attention with pooling.
    Returns (delta, n_shared_prefix_tokens).
    """
    import torch

    a = PROBE_PREFIX + PROBE_SUFFIX_A
    b = PROBE_PREFIX + PROBE_SUFFIX_B
    enc_a = tokenizer(a, return_tensors="pt")
    enc_b = tokenizer(b, return_tensors="pt")
    ids_a = enc_a["input_ids"][0].tolist()
    ids_b = enc_b["input_ids"][0].tolist()

    k = 0
    while k < min(len(ids_a), len(ids_b)) and ids_a[k] == ids_b[k]:
        k += 1
    if k < 2:
        raise RuntimeError("probe strings share only %d leading tokens -- the probe "
                           "cannot distinguish causal from bidirectional" % k)

    try:
        device = next(auto_model.parameters()).device
    except StopIteration:
        device = "cpu"
    enc_a = {kk: v.to(device) for kk, v in enc_a.items()}
    enc_b = {kk: v.to(device) for kk, v in enc_b.items()}

    was_training = bool(getattr(auto_model, "training", False))
    auto_model.eval()
    try:
        with torch.no_grad():
            ha = auto_model(**enc_a, output_hidden_states=True).hidden_states[-1][0, :k]
            hb = auto_model(**enc_b, output_hidden_states=True).hidden_states[-1][0, :k]
        delta = float((ha.float() - hb.float()).abs().max().item())
    finally:
        if was_training:
            auto_model.train()
    return delta, k


def delta_bucket(delta: float) -> str:
    """Quantise the raw delta so float noise cannot thrash a cache key.

    "causal" for anything under CAUSAL_EPS; otherwise the DECADE, e.g. 5.74 -> "1e+0".
    A decade is coarse enough that seed / kernel / dtype jitter never moves it, and
    fine enough that causal-vs-bidirectional (0.0 vs ~5.7) is a different bucket --
    which is the distinction this whole module exists to catch.
    """
    if delta != delta:                     # NaN
        return UNKNOWN
    if delta < CAUSAL_EPS:
        return "causal"
    return "1e%+d" % int(math.floor(math.log10(delta)))


def probe_fingerprint(embedder, model_id: str = None, extra: dict = None) -> dict:
    """The FULL behaviour block: the cheap fields PLUS the measured bucket.

    Costs ONE forward pass, memoised per process per model identity, so a run that
    writes five caches probes once. Call this only where a model is ALREADY loaded --
    i.e. on a cache WRITE / rebuild. Never call it on a cache hit, and never at import.

    A backbone we cannot reach yields bucket `UNKNOWN`, which still DIFFERS from any
    real bucket, so it degrades to a loud rejection rather than a silent pass.
    """
    model_id = model_id or str(getattr(embedder, "name", "") or type(embedder).__name__)
    auto, tok = resolve_backbone(embedder)

    if auto is None or tok is None:
        _eprint("[behaviour] WARNING: backbone not reachable on %r -- the behavioural "
                "probe is recorded as %r. Version fields still guard the cache, but "
                "the same-version/different-behaviour case is NOT covered."
                % (type(embedder).__name__, UNKNOWN))
        block = cheap_fingerprint(model_id, extra)
        block["prefix_delta_bucket"] = UNKNOWN
        block["prefix_delta"] = None
        return block

    ident = _model_identity(auto, model_id)
    memo = _PROBE_CACHE.get(ident)
    if memo is not None:
        block = cheap_fingerprint(model_id, extra)
        block.update({k: v for k, v in memo.items() if k.startswith("prefix_")
                      or k in ("attn_implementation", "bidirectional_flag", "probe_sec")})
        return block

    cfg = getattr(auto, "config", None)
    t0 = time.perf_counter()
    try:
        delta, n_shared = prefix_invariance_delta(auto, tok)
        bucket = delta_bucket(delta)
    except Exception as exc:
        _eprint("[behaviour] WARNING: probe failed (%s: %s) -- recorded as %r"
                % (type(exc).__name__, exc, UNKNOWN))
        delta, n_shared, bucket = float("nan"), 0, UNKNOWN
    probe_sec = time.perf_counter() - t0

    measured = {
        "prefix_delta_bucket": bucket,
        "prefix_delta": None if delta != delta else round(float(delta), 6),
        "prefix_shared_tokens": int(n_shared),
        "attn_implementation": str(getattr(cfg, "_attn_implementation", UNKNOWN)),
        "bidirectional_flag": bool(getattr(cfg, "use_bidirectional_attention", False)),
        "probe_sec": round(probe_sec, 3),
    }
    _PROBE_CACHE[ident] = measured
    _eprint("[behaviour] probe %s: prefix delta=%s over %d shared tokens -> bucket %r "
            "(attn=%s, bidirectional_flag=%s, %.2fs)"
            % (model_id, "nan" if delta != delta else "%.6e" % delta, n_shared, bucket,
               measured["attn_implementation"], measured["bidirectional_flag"], probe_sec))
    if bucket == "causal" and measured["bidirectional_flag"]:
        _eprint("[behaviour] *** the config says use_bidirectional_attention=True but "
                "the model BEHAVES CAUSALLY. This is the 2026-08-17 incident exactly "
                "(audits/AUDIT_2026-08-17_embeddinggemma_causal.md). Any number produced "
                "on top of these vectors is NOT measuring what it claims. ***")

    block = cheap_fingerprint(model_id, extra)
    block.update(measured)
    return block


# ---------------------------------------------------------------------------
# 3. comparison / rejection
# ---------------------------------------------------------------------------
def compare(want: dict, got: dict | None) -> list:
    """-> list of "field: expected X, cache has Y" strings. Empty == compatible.

    `NOT_PROBED` on EITHER side skips the bucket comparison only (a probe was
    deliberately not run; the version fields carry the check). `UNKNOWN` does NOT
    skip -- a failed probe is a real, comparable state and must not pass silently
    against a successful one.
    """
    if not got:
        return ["encoder_behaviour: MISSING from the cache (written before behaviour "
                "stamping) -- unattributable, so it is not trusted"]
    diffs = []
    if int(got.get("probe_version", -1)) != int(want.get("probe_version", -1)):
        diffs.append("probe_version: expected %s, cache has %s"
                     % (want.get("probe_version"), got.get("probe_version")))
    if str(got.get("model_id")) != str(want.get("model_id")):
        diffs.append("model_id: expected %s, cache has %s"
                     % (want.get("model_id"), got.get("model_id")))
    w_libs, g_libs = want.get("libs") or {}, got.get("libs") or {}
    for lib in sorted(set(w_libs) | set(g_libs)):
        if str(w_libs.get(lib)) != str(g_libs.get(lib)):
            diffs.append("libs.%s: expected %s, cache has %s"
                         % (lib, w_libs.get(lib), g_libs.get(lib)))
    wb, gb = want.get("prefix_delta_bucket"), got.get("prefix_delta_bucket")
    if NOT_PROBED not in (wb, gb) and str(wb) != str(gb):
        diffs.append("prefix_delta_bucket: expected %s, cache has %s (the encoder's "
                     "ATTENTION BEHAVIOUR changed -- causal vs bidirectional is exactly "
                     "this field)" % (wb, gb))
    return diffs


def describe_mismatch(want: dict, got: dict | None) -> str:
    diffs = compare(want, got)
    return "; ".join(diffs) if diffs else ""


def reject_message(path, want: dict, got: dict | None) -> str:
    """The loud rejection text, in the same shape as the content-fingerprint one."""
    return ("embedding cache %s was produced by a DIFFERENT ENCODER BEHAVIOUR than the "
            "one about to use it (%s). Refusing to reuse it -- vectors computed under "
            "different attention semantics are not interchangeable "
            "(audits/AUDIT_2026-08-17_embeddinggemma_causal.md: EmbeddingGemma ran "
            "CAUSAL under transformers 4.55.0 while its config said bidirectional). "
            "Re-embed, or delete the file."
            % (path, describe_mismatch(want, got)))


def quarantine(path, tag: str = "stale") -> str:
    """Move a rejected cache aside instead of overwriting it. -> the new path (or "").

    An overwrite destroys the evidence of what was cached and under what stack. The
    manual `_causal_invalid_caches/` quarantine is what this automates.
    """
    try:
        path = Path(path)
        if not path.exists():
            return ""
        dest = path.with_name("%s.%s.%d%s" % (path.stem, tag, int(time.time()), path.suffix))
        os.replace(str(path), str(dest))
        _eprint("[behaviour] quarantined the rejected cache -> %s" % dest.name)
        return str(dest)
    except Exception as exc:
        _eprint("[behaviour] quarantine of %s failed (%s) -- leaving it in place"
                % (path, exc))
        return ""


# ---------------------------------------------------------------------------
# CPU self-test: no model, no GPU, no network.
#   python -m steering_tutorials.biencoder_guard.encoder_behaviour
# ---------------------------------------------------------------------------
def _self_test() -> None:
    import tempfile

    # (1) cheap block: no model was loaded to build it, and it names the libraries.
    cheap = cheap_fingerprint("google/embeddinggemma-300m")
    assert cheap["prefix_delta_bucket"] == NOT_PROBED
    assert set(cheap["libs"]) == {"transformers", "sentence_transformers", "torch"}
    print("cheap_fingerprint (no model load): libs=%s" % json.dumps(cheap["libs"]))

    # (2) bucketing: the two REAL measured values from the audit must not collide,
    #     and decade quantisation must absorb ordinary float jitter.
    assert delta_bucket(0.0) == "causal", "the audited causal value must bucket 'causal'"
    assert delta_bucket(5.74) == "1e+0", "the audited bidirectional value mis-bucketed"
    assert delta_bucket(5.74) == delta_bucket(9.31), "decade bucket is thrashing on noise"
    assert delta_bucket(0.0) != delta_bucket(5.74), "causal and bidirectional collided"
    assert delta_bucket(float("nan")) == UNKNOWN
    print("delta_bucket: 0.0 -> 'causal'; 5.74 -> '1e+0'; 5.74 and 9.31 share a bucket")

    # (3) THE REGRESSION: two caches identical in every DATA field, differing ONLY in
    #     encoder behaviour, must be REJECTED. This is the 2026-08-17 incident in
    #     miniature -- same texts, same model id, transformers 4.55.0 vs 5.15.0.
    causal = {"probe_version": PROBE_FORMAT_VERSION,
              "model_id": "google/embeddinggemma-300m",
              "libs": {"transformers": "4.55.0", "sentence_transformers": "5.1.0",
                       "torch": "2.8.0"},
              "prefix_delta_bucket": "causal", "prefix_delta": 0.0}
    bidi = {"probe_version": PROBE_FORMAT_VERSION,
            "model_id": "google/embeddinggemma-300m",
            "libs": {"transformers": "5.15.0", "sentence_transformers": "5.1.0",
                     "torch": "2.8.0"},
            "prefix_delta_bucket": "1e+0", "prefix_delta": 5.74}

    assert not compare(bidi, bidi), "an identical behaviour block must MATCH"
    diffs = compare(bidi, causal)
    assert diffs, "a causal cache was accepted for a bidirectional encoder"
    joined = " ".join(diffs)
    assert "libs.transformers" in joined, "the library change was not named"
    assert "prefix_delta_bucket" in joined, "the BEHAVIOUR change was not named"
    print("compare(): rejects causal-vs-bidirectional naming both %d fields" % len(diffs))

    # Same library versions, different measured behaviour (a monkeypatch, a changed
    # attn kernel) -- the version fields alone would have PASSED this; the probe does not.
    sneaky = dict(bidi)
    sneaky["prefix_delta_bucket"] = "causal"
    d2 = compare(bidi, sneaky)
    assert d2 and "prefix_delta_bucket" in " ".join(d2), \
        "same-version/different-behaviour slipped through -- the probe is not binding"
    print("compare(): rejects SAME versions + different behaviour (versions alone would "
          "have passed it -- this is why the probe exists)")

    # A missing block (a pre-stamping cache) is unattributable, not innocent.
    assert compare(bidi, None), "a cache with no behaviour block was trusted"
    assert compare(bidi, {}), "an empty behaviour block was trusted"

    # NOT_PROBED on a read skips ONLY the bucket, so a cache hit needs no forward pass.
    read_side = cheap_fingerprint("google/embeddinggemma-300m")
    read_side["libs"] = dict(bidi["libs"])
    assert not compare(read_side, bidi), \
        "a cache HIT demanded a probe -- reads must not load a model"
    print("compare(): NOT_PROBED skips only the bucket, so a cache HIT needs no model")

    # (4) the key component moves with the stack and is path-safe.
    k_now = key_component("google/embeddinggemma-300m")
    assert len(k_now) == 12 and k_now.isalnum()
    print("key_component: %s (folded into the cache FILENAME)" % k_now)

    # (5) quarantine moves the file aside rather than overwriting it.
    tmp = Path(tempfile.mkdtemp(prefix="behaviour_test_"))
    victim = tmp / "emb_train_embeddinggemma.npz"
    victim.write_bytes(b"causal-vectors")
    moved = quarantine(victim, tag="causal")
    assert moved and not victim.exists() and Path(moved).read_bytes() == b"causal-vectors", \
        "quarantine destroyed or failed to move the rejected cache"
    Path(moved).unlink()
    tmp.rmdir()
    print("quarantine: the rejected cache is moved aside, not overwritten")

    assert not _PROBE_CACHE, "the self-test loaded a model -- it must stay CPU-only"
    print("OK -- encoder_behaviour.py self-test: a cache whose ENCODER BEHAVIOUR "
          "disagrees is REJECTED (both the library-version route and the "
          "same-version/different-behaviour route), with no model, no GPU, no network.")


if __name__ == "__main__":
    _self_test()
