"""embed.py -- turn-level embedders + a CONTENT-FINGERPRINTED ragged sequence cache.

Each USER turn of a conversation is embedded to ONE vector; a conversation is
the SEQUENCE of those per-turn vectors ([n_turns, dim]). Three embedders:

  "embgemma": google/embeddinggemma-300m via sentence-transformers (768-d). THE
              MANDATED EMBEDDER and the headline arm. Loaded from the HF id first
              and retried from the LOCAL snapshot (C.EMBGEMMA_LOCAL) on any failure
              -- the weights ARE on disk here (model.safetensors, 1,211,486,072
              bytes, verified by size), so CLAUDE.md section 18.5's "gated, no token"
              blocker is stale for this model.
  "gemma"   : the course's Gemma-3-1B decoder, layer-C.GEMMA_LAYER mean-pooled
              residual-stream activation. Reuses lesson-2's
              `hello_world_steering.model_utils` plumbing. Loaded from the LOCAL
              path (the gated HF id 401s without a token).
  "minilm"  : sentence-transformers/all-MiniLM-L6-v2 via plain `transformers`
              AutoModel + attention-mask mean pooling ([384]). LEGACY REFERENCE ARM
              -- kept so older numbers stay reproducible; never headline it.

The model for any embedder is loaded ONCE, lazily, inside get_embedder -- never at
import (so `import embed` stays CPU-cheap and model-free).

CACHE FORMAT (.npz ragged pack)
-------------------------------
All per-turn vectors of the whole dataset are vstacked into one float32 array
`flat` ([sum_turns, dim]); `turn_counts` (int) records how many turns each
conversation contributed, so `flat` splits back into the per-conversation
[n_turns, dim] list; `dim` (int) is stored for validation; and `meta_json` stores
the CONTENT FINGERPRINT.

WHY THE FINGERPRINT EXISTS (section 18.8, and it was live in this code)
-----------------------------------------------------------------------
`load_or_build` used to validate the cache by ROW COUNT ALONE:

    cached = _try_load_pack(cache_path)
    if cached is not None and len(cached) == len(conversations):
        return cached

Change the seed and the disjoint-half split reshuffles WHICH attacks are positive.
The row count is identical, so the cached vectors were silently reused **against new
labels** -- verbatim section 18.8's "the embedding cache returned stale labels under a
key that ignored them". It did not crash; it returned confident, well-formed, wrong
numbers. The cache is now keyed on a SHA-256 of the concatenated turn texts plus
method/seed/n/dim, and a mismatch RAISES.

This module is also imported by `cross_trajectory`
(`run_cross_trajectory._embed_samples` and `infer.py` call `load_or_build(samples,
embedder, cache)` positionally), so the fix lands there too. The signature is
backward-compatible: the two new parameters are keyword-only with defaults.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from steering_tutorials.multiturn_jailbreak import config as C

GEMMA_DIM_HINT = 1152  # gemma-3-1b hidden; real dim taken from the model at runtime
CACHE_FORMAT_VERSION = 2


# --- Embedders ---------------------------------------------------------------
def get_embedder(method: str):
    """Return ``(embed_turn, dim)`` for ``method`` in {"embgemma","gemma","minilm"}.

    ``embed_turn(text:str) -> np.ndarray[dim]`` (float32, 1-D). The underlying
    model is loaded ONCE here (lazily) and closed over by ``embed_turn``.
    """
    if method == "embgemma":
        return _get_embgemma_embedder()
    if method == "gemma":
        return _get_gemma_embedder()
    if method == "minilm":
        return _get_minilm_embedder()
    raise ValueError("method must be one of %r, got %r" % (list(C.KNOWN_EMBEDDERS), method))


def _get_embgemma_embedder():
    """EmbeddingGemma-300M via sentence-transformers -- the mandated embedder.

    EmbeddingGemma is trained with task-prefixed prompts. A conversation turn is a
    user QUERY, so turns are encoded under the model's query prompt when the loaded
    build registers named prompts, and under the documented prefix string otherwise.
    """
    from sentence_transformers import SentenceTransformer  # lazy, heavy

    try:
        model = SentenceTransformer(C.EMBGEMMA_ID)
    except Exception as exc:  # gated / offline / 401 -> the local snapshot on disk
        print("[embed] HF load of %s failed (%s); using local %s"
              % (C.EMBGEMMA_ID, type(exc).__name__, C.EMBGEMMA_LOCAL))
        model = SentenceTransformer(C.EMBGEMMA_LOCAL)

    prompt_name = "query"
    fallback_prefix = "task: search result | query: "

    def embed_turn(text: str) -> np.ndarray:
        try:
            vec = model.encode([str(text)], prompt_name=prompt_name,
                               convert_to_numpy=True, normalize_embeddings=False,
                               show_progress_bar=False)
        except (ValueError, KeyError, TypeError):
            vec = model.encode([fallback_prefix + str(text)], convert_to_numpy=True,
                               normalize_embeddings=False, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    dim = int(embed_turn("hello").shape[0])
    return embed_turn, dim


def _get_gemma_embedder():
    # Reuse lesson-2's plumbing: load_model -> (model, tok); mean_pool_activation
    # returns the layer-C.GEMMA_LAYER mean-pooled activation as a 1-D np.ndarray.
    from steering_tutorials.hello_world_steering.model_utils import (
        load_model,
        mean_pool_activation,
    )

    model, tok = load_model(C.GEMMA_MODEL_ID)
    layer = C.GEMMA_LAYER

    def embed_turn(text: str) -> np.ndarray:
        vec = mean_pool_activation(model, tok, text, layer)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    # Probe the true hidden dim from one cheap forward.
    dim = int(embed_turn("hello").shape[0])
    return embed_turn, dim


def _get_minilm_embedder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.MINILM_ID)
    model = AutoModel.from_pretrained(C.MINILM_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def embed_turn(text: str) -> np.ndarray:
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state  # [1, seq, hid]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [1, seq, 1]
        summed = (hidden * mask).sum(dim=1)  # [1, hid]
        counts = mask.sum(dim=1).clamp(min=1e-9)  # [1, 1]
        pooled = (summed / counts).squeeze(0)  # [hid]
        return pooled.detach().cpu().float().numpy().reshape(-1)

    dim = int(C.MINILM_DIM)
    return embed_turn, dim


# --- Content fingerprint -----------------------------------------------------
def content_fingerprint(conversations, method, seed=None) -> dict:
    """The cache identity: SHA-256 of the concatenated texts + method/seed/n.

    `dim` is NOT part of the hash (it is unknown before the model loads) but IS
    stored and checked separately on load.
    """
    h = hashlib.sha256()
    for conv in conversations:
        for turn in conv:
            h.update(str(turn).encode("utf-8", "replace"))
            h.update(b"\x01")
        h.update(b"\x02")
    return {
        "version": CACHE_FORMAT_VERSION,
        "texts_sha256": h.hexdigest(),
        "method": str(method),
        "seed": int(C.SEED if seed is None else seed),
        "n": int(len(conversations)),
    }


def _describe_mismatch(want: dict, got) -> str:
    if not got:
        return ("cache carries NO fingerprint (written by the pre-2026-08 code, which "
                "validated row count alone)")
    diffs = []
    for k in ("version", "texts_sha256", "method", "seed", "n"):
        a, b = want.get(k), got.get(k)
        if a != b:
            sa = str(a)[:16] if k == "texts_sha256" else str(a)
            sb = str(b)[:16] if k == "texts_sha256" else str(b)
            diffs.append("%s: expected %s, cache has %s" % (k, sa, sb))
    return "; ".join(diffs) or "unknown difference"


# --- Conversation / dataset embedding ----------------------------------------
def embed_conversation(turns, embed_turn) -> np.ndarray:
    """Embed a conversation (list of user-turn strings) -> [n_turns, dim] float32."""
    if len(turns) == 0:
        raise ValueError("cannot embed an empty conversation")
    vecs = [np.asarray(embed_turn(t), dtype=np.float32).reshape(-1) for t in turns]
    return np.stack(vecs, axis=0).astype(np.float32)


def embed_dataset(conversations, method, cache_path, *, seed=None):
    """Embed every conversation -> list of [n_turns, dim]. Saves the fingerprinted pack."""
    embed_turn, dim = get_embedder(method)
    seqs = [embed_conversation(list(conv), embed_turn) for conv in conversations]
    _save_pack(cache_path, seqs, dim, content_fingerprint(conversations, method, seed))
    return seqs


def load_or_build(conversations, method, cache_path, *, seed=None, on_mismatch=None):
    """Return the cached seqs IFF their fingerprint matches, else build (or raise).

    Backward-compatible with the pre-2026-08 three-positional-argument call used by
    `cross_trajectory`; the two new parameters are keyword-only.

    on_mismatch (default from MJ_CACHE_ON_MISMATCH, itself defaulting to "raise"):
      "raise"   -- a stale/unstamped cache is a LOUD RuntimeError naming the field
                   that differs. This is the default because a silent fall-through
                   is what produced the stale-label defect in the first place, and
                   because a rebuild can quietly consume a GPU hour nobody asked for.
      "rebuild" -- re-embed and overwrite. Safe (it can never mislabel) but costs a
                   GPU pass; ask for it explicitly once you know why the cache moved.
    """
    on_mismatch = (on_mismatch or os.environ.get("MJ_CACHE_ON_MISMATCH") or "raise").lower()
    if on_mismatch not in ("raise", "rebuild"):
        raise ValueError("on_mismatch must be 'raise' or 'rebuild', got %r" % (on_mismatch,))

    want = content_fingerprint(conversations, method, seed)
    loaded = _try_load_pack(cache_path)
    if loaded is None:
        return embed_dataset(conversations, method, cache_path, seed=seed)

    cached, got_meta, cached_dim = loaded
    ok = bool(got_meta) and all(got_meta.get(k) == want[k] for k in want)
    if ok and len(cached) == len(conversations):
        return cached

    detail = _describe_mismatch(want, got_meta)
    if len(cached) != len(conversations):
        detail += "; row count %d vs %d" % (len(cached), len(conversations))
    msg = ("embedding cache %s does not match the data it is about to be used for "
           "(%s). Refusing to reuse it: an embedding cache reused against different "
           "labels is CLAUDE.md section 18.8's stale-label defect, which fails silently "
           "and plausibly. Re-embed with MJ_CACHE_ON_MISMATCH=rebuild, or delete the "
           "file." % (cache_path, detail))
    if on_mismatch == "raise":
        raise RuntimeError(msg)
    print("[embed] REBUILD: %s" % msg)
    return embed_dataset(conversations, method, cache_path, seed=seed)


# --- Ragged .npz pack/unpack -------------------------------------------------
def _save_pack(cache_path, seqs, dim, meta=None) -> None:
    seqs = [np.asarray(s, dtype=np.float32) for s in seqs]
    if seqs:
        flat = np.vstack(seqs).astype(np.float32)
    else:
        flat = np.zeros((0, int(dim)), dtype=np.float32)
    turn_counts = np.asarray([s.shape[0] for s in seqs], dtype=np.int64)
    meta = dict(meta or {})
    meta["dim"] = int(dim)
    os.makedirs(os.path.dirname(os.path.abspath(str(cache_path))), exist_ok=True)
    np.savez(
        str(cache_path),
        flat=flat,
        turn_counts=turn_counts,
        dim=np.asarray(int(dim), dtype=np.int64),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )


def _split_flat(flat, turn_counts):
    seqs = []
    off = 0
    for k in turn_counts:
        k = int(k)
        seqs.append(np.asarray(flat[off:off + k], dtype=np.float32))
        off += k
    return seqs


def _try_load_pack(cache_path):
    """Return ``(seqs, meta_dict_or_None, dim)`` or None if the file is absent."""
    p = str(cache_path)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        flat = np.asarray(z["flat"], dtype=np.float32)
        turn_counts = np.asarray(z["turn_counts"], dtype=np.int64)
        dim = int(z["dim"]) if "dim" in z.files else int(flat.shape[1] if flat.size else 0)
        meta = None
        if "meta_json" in z.files:
            try:
                meta = json.loads(str(z["meta_json"].item()))
            except Exception:
                meta = None
    return _split_flat(flat, turn_counts), meta, dim


# --- CPU self-test (no model, no dataset) ------------------------------------
def _self_test() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    dim = 8
    ks = [3, 4, 5]
    seqs = [rng.standard_normal((k, dim)).astype(np.float32) for k in ks]
    convs = [["turn %d of conv %d" % (j, i) for j in range(k)] for i, k in enumerate(ks)]

    tmpdir = tempfile.mkdtemp(prefix="mj_embed_test_")
    cache = os.path.join(tmpdir, "roundtrip.npz")

    _save_pack(cache, seqs, dim, content_fingerprint(convs, "gemma"))
    loaded, meta, back_dim = _try_load_pack(cache)

    assert loaded is not None, "cache failed to load"
    assert len(loaded) == len(seqs), "conversation count changed on round-trip"
    assert back_dim == dim, "dim did not round-trip: %r" % (back_dim,)
    assert meta and meta["texts_sha256"], "fingerprint missing from the pack"
    for orig, back in zip(seqs, loaded):
        assert orig.shape == back.shape, "shape changed: %r vs %r" % (orig.shape, back.shape)
        assert back.dtype == np.float32, "dtype not float32: %r" % (back.dtype,)
        assert np.array_equal(orig, back), "values did not round-trip exactly"

    # 1. matching content -> the cache is reused (no model load).
    reused = load_or_build(convs, "gemma", cache)
    assert len(reused) == len(ks), "load_or_build did not reuse a matching cache"
    for orig, back in zip(seqs, reused):
        assert np.array_equal(orig, back), "load_or_build round-trip mismatch"

    # 2. THE REGRESSION TEST. Same row count, DIFFERENT texts (what a seed change
    #    does to a shuffled split). The old code returned the stale vectors here.
    swapped = [list(reversed(c)) for c in convs]
    try:
        load_or_build(swapped, "gemma", cache)
        raise AssertionError("stale cache was reused for different texts at equal count "
                             "-- the section-18.8 stale-label defect is BACK")
    except RuntimeError as exc:
        assert "texts_sha256" in str(exc), "mismatch message did not name the field: %s" % exc

    # 3. A different METHOD must not reuse another embedder's vectors.
    try:
        load_or_build(convs, "minilm", cache)
        raise AssertionError("a cache built by 'gemma' was reused for 'minilm'")
    except RuntimeError as exc:
        assert "method" in str(exc), "mismatch message did not name the method: %s" % exc

    # 4. An UNSTAMPED legacy pack is treated as a mismatch, not trusted.
    legacy = os.path.join(tmpdir, "legacy.npz")
    np.savez(legacy, flat=np.vstack(seqs), turn_counts=np.asarray(ks, dtype=np.int64),
             dim=np.asarray(dim, dtype=np.int64))
    try:
        load_or_build(convs, "gemma", legacy)
        raise AssertionError("an unstamped legacy cache was trusted")
    except RuntimeError as exc:
        assert "NO fingerprint" in str(exc), "legacy message was wrong: %s" % exc

    for f in (cache, legacy):
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    print("embed.py self-test OK")
    print("  ragged npz pack/unpack round-trips exactly (conversations=%d turn_counts=%s dim=%d)"
          % (len(seqs), [int(s.shape[0]) for s in seqs], dim))
    print("  cache fingerprint REJECTS: changed texts at equal row count, a changed")
    print("  embedder, and an unstamped legacy pack -- each raises instead of returning")
    print("  stale vectors against new labels (CLAUDE.md section 18.8).")


if __name__ == "__main__":
    _self_test()
