"""embed_ct.py -- the cross_trajectory embedding cache, keyed on CONTENT.

WHY THIS FILE EXISTS
--------------------
The 2026-08 audit found the repo's documented recurring failure mode (CLAUDE.md
section 18.8, "the embedding cache returned STALE LABELS under a key that ignored
them") sitting unguarded in this lesson:

  * ``run_cross_trajectory._embed_samples`` keyed the cache on ``(condition,
    embedder)`` only -- not on the seed, not on K, not on n, not on the texts.
  * ``multiturn_jailbreak.embed.load_or_build`` validates the ROW COUNT and
    nothing else.

So changing ``CT_SEED`` reshuffles which attacks land on the positive side of the
hard half-split, produces the SAME number of samples, and the cached vectors are
silently reused **against the new labels**. It has not necessarily fired; it was
simply never guarded.

This module owns the cache instead. Every pack stores a SHA-256 of the exact
concatenated texts it was built from, plus ``seed``/``k``/``n``/``dim``/
``embedder``, and :func:`load_or_build` **asserts** all of them on load. A
mismatch raises :class:`CacheMismatch` -- it does NOT silently fall through to a
rebuild, because a silent rebuild is how a stale artifact becomes a fresh-looking
wrong number. The caller decides (``allow_rebuild=True`` is opt-in and prints a
loud line).

The fingerprint also goes in the FILENAME, so two configurations cannot even
contend for the same path.

EMBEDDERS
---------
  ``embeddinggemma`` -- ``google/embeddinggemma-300m``, the encoder CLAUDE.md
        section 17 / the user's standing mandate actually names. The weights ARE
        on disk at ``models/google/embeddinggemma-300m`` (1.2 GB safetensors,
        verified 2026-08-08), so the section-18.5 "gated, no HF token" blocker is
        **stale for this model**. Loaded from the LOCAL path via plain
        ``transformers`` AutoModel + attention-mask mean pooling -- no
        sentence-transformers dependency, matching how ``minilm`` is loaded.
  ``gemma`` / ``minilm`` -- delegated unchanged to
        ``multiturn_jailbreak.embed.get_embedder``. These are NON-COMPLIANT
        SUBSTITUTES for the mandated encoder, not necessities.

CPU/GPU: the model is loaded lazily inside :func:`get_embedder`, never at import,
so ``python -c "import ...embed_ct"`` stays model-free. ASCII stdout only.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import config as C


class CacheMismatch(RuntimeError):
    """A cached embedding pack does not match the data it is about to be used for.

    Raised LOUDLY on purpose. The failure this guards is not a crash, it is a
    plausible wrong answer.
    """


# --- fingerprint --------------------------------------------------------------
def texts_sha256(samples) -> str:
    """SHA-256 over the exact trajectory texts, in order, with unambiguous framing.

    ``samples`` is a list of lists of strings (one sample == K trajectories).
    Record separators are used so ``[["ab"],["c"]]`` and ``[["a"],["bc"]]`` cannot
    collide.
    """
    h = hashlib.sha256()
    for sample in samples:
        for text in sample:
            h.update(str(text).encode("utf-8", "replace"))
            h.update(b"\x1f")  # unit separator, between trajectories
        h.update(b"\x1e")      # record separator, between samples
    return h.hexdigest()


def cache_path(tag: str, embedder: str, seed: int, k: int, n: int, sha: str):
    """Cache filename carrying seed / K / n / a text-hash prefix.

    The old name was ``traj_{tag}_{embedder}.npz`` -- two different datasets of
    the same size collided on it. They cannot now.
    """
    return C.ARTIFACTS / ("traj_%s_%s_seed%d_k%d_n%d_%s.npz"
                          % (tag, embedder, int(seed), int(k), int(n), sha[:12]))


# --- embedders ----------------------------------------------------------------
def get_embedder(method: str):
    """Return ``(embed_text, dim)`` for ``method``.

    ``embed_text(text:str) -> np.ndarray[dim]`` float32 1-D. Loads the model ONCE.
    """
    if method == "embeddinggemma":
        return _get_embeddinggemma_embedder()
    if method in ("gemma", "minilm"):
        from steering_tutorials.multiturn_jailbreak import embed as MJE
        return MJE.get_embedder(method)
    raise ValueError("embedder must be one of %r, got %r"
                     % (list(C.EMBEDDER_CHOICES), method))


def _get_embeddinggemma_embedder():
    """google/embeddinggemma-300m from the LOCAL path, mean-pooled over the mask.

    Loaded exactly the way ``minilm`` is (AutoModel + attention-mask mean pooling)
    so the two arms differ ONLY in the backbone, not in the pooling recipe.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = str(C.EMBEDDINGGEMMA_ID)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "embeddinggemma weights not found at %r. The HF id is gated; this "
            "lesson loads the LOCAL copy. Set CT_EMBEDDINGGEMMA_ID." % path)

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def embed_text(text: str) -> np.ndarray:
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=C.EMBEDDINGGEMMA_MAXLEN, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state                              # [1, seq, hid]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [1, seq, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.squeeze(0).detach().cpu().float().numpy().reshape(-1)

    dim = int(embed_text("hello").shape[0])  # probe, never hard-code
    return embed_text, dim


# --- ragged pack, with the fingerprint INSIDE ---------------------------------
def _save_pack(path, seqs, dim, meta) -> None:
    seqs = [np.asarray(s, dtype=np.float32) for s in seqs]
    flat = np.vstack(seqs).astype(np.float32) if seqs else np.zeros((0, int(dim)), np.float32)
    counts = np.asarray([s.shape[0] for s in seqs], dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    np.savez(
        str(path),
        flat=flat,
        turn_counts=counts,
        dim=np.asarray(int(dim), dtype=np.int64),
        sha256=np.asarray(str(meta["sha256"])),
        embedder=np.asarray(str(meta["embedder"])),
        seed=np.asarray(int(meta["seed"]), dtype=np.int64),
        k=np.asarray(int(meta["k"]), dtype=np.int64),
        n=np.asarray(int(meta["n"]), dtype=np.int64),
    )


def _split_flat(flat, counts):
    seqs, off = [], 0
    for c in counts:
        c = int(c)
        seqs.append(np.asarray(flat[off:off + c], dtype=np.float32))
        off += c
    return seqs


def _load_pack(path):
    """Load a pack -> (seqs, meta). Returns ``(None, None)`` if the file is absent."""
    p = str(path)
    if not os.path.exists(p):
        return None, None
    with np.load(p, allow_pickle=False) as z:
        flat = np.asarray(z["flat"], dtype=np.float32)
        counts = np.asarray(z["turn_counts"], dtype=np.int64)
        meta = {
            "dim": int(z["dim"]) if "dim" in z else None,
            "sha256": str(z["sha256"]) if "sha256" in z else None,
            "embedder": str(z["embedder"]) if "embedder" in z else None,
            "seed": int(z["seed"]) if "seed" in z else None,
            "k": int(z["k"]) if "k" in z else None,
            "n": int(z["n"]) if "n" in z else None,
        }
    return _split_flat(flat, counts), meta


def _assert_pack(path, seqs, meta, expect) -> None:
    """Every stored field must match. Raises CacheMismatch listing ALL mismatches."""
    bad = []
    if meta.get("sha256") is None:
        bad.append("no sha256 stored (pre-fingerprint pack -- provenance unknown)")
    for key in ("sha256", "embedder", "seed", "k", "n"):
        got, want = meta.get(key), expect.get(key)
        if got is not None and want is not None and str(got) != str(want):
            bad.append("%s: cached=%r expected=%r" % (key, got, want))
    if len(seqs) != int(expect["n"]):
        bad.append("row count: cached=%d expected=%d" % (len(seqs), int(expect["n"])))
    dim = meta.get("dim")
    if dim is not None and seqs and int(seqs[0].shape[-1]) != int(dim):
        bad.append("dim: array=%d stored=%d" % (int(seqs[0].shape[-1]), int(dim)))
    if bad:
        raise CacheMismatch(
            "embedding cache %s does not match the data it would be used for:\n  - %s\n"
            "REFUSING to reuse it. This is the stale-label defect (CLAUDE.md 18.8): the "
            "vectors would be scored against DIFFERENT labels. Delete the file to rebuild."
            % (path, "\n  - ".join(bad)))


# --- the public entry point ---------------------------------------------------
def load_or_build(samples, method, tag, seed, k, allow_rebuild: bool = True):
    """Embed every sample's trajectories -> list of [K_i, dim] float32.

    Reuses a cached pack ONLY when its stored SHA-256 of the texts, embedder,
    seed, K, n and dim all match. Any mismatch raises :class:`CacheMismatch`.

    Returns ``(seqs, meta)`` where ``meta`` carries the fingerprint fields that
    belong in ``results.json``.
    """
    samples = [list(s) for s in samples]
    sha = texts_sha256(samples)
    expect = {"sha256": sha, "embedder": str(method), "seed": int(seed),
              "k": int(k), "n": len(samples)}
    path = cache_path(tag, method, seed, k, len(samples), sha)

    seqs, meta = _load_pack(path)
    if seqs is not None:
        _assert_pack(path, seqs, meta, expect)   # raises on ANY mismatch
        print("[embed:%s] cache HIT %s (sha=%s...)" % (tag, os.path.basename(str(path)), sha[:12]))
        out = dict(expect)
        out["dim"] = int(meta.get("dim") or (seqs[0].shape[-1] if seqs else 0))
        out["cache"] = str(path)
        out["cache_hit"] = True
        return seqs, out

    if not allow_rebuild:
        raise CacheMismatch("no cache at %s and allow_rebuild=False" % path)

    print("[embed:%s] cache MISS -> building with %r (n=%d, loads the model)"
          % (tag, method, len(samples)))
    embed_text, dim = get_embedder(method)
    seqs = []
    for sample in samples:
        if not sample:
            raise ValueError("empty sample: cannot embed zero trajectories")
        vecs = [np.asarray(embed_text(t), dtype=np.float32).reshape(-1) for t in sample]
        seqs.append(np.stack(vecs, axis=0).astype(np.float32))
    real_dim = int(seqs[0].shape[-1]) if seqs else int(dim)
    _save_pack(path, seqs, real_dim, expect)
    out = dict(expect)
    out["dim"] = real_dim
    out["cache"] = str(path)
    out["cache_hit"] = False
    return seqs, out


# --- CPU self-test: no model, no dataset -------------------------------------
if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="ct_embed_")
    path = os.path.join(tmp, "pack.npz")

    samples = [["alpha one", "alpha two"], ["beta one", "beta two", "beta three"]]
    seqs = [rng.standard_normal((len(s), 8)).astype(np.float32) for s in samples]
    sha = texts_sha256(samples)
    expect = {"sha256": sha, "embedder": "minilm", "seed": 0, "k": 5, "n": 2}
    _save_pack(path, seqs, 8, expect)

    back, meta = _load_pack(path)
    _assert_pack(path, back, meta, expect)
    assert len(back) == 2 and np.array_equal(back[0], seqs[0]), "round-trip failed"
    print("OK: pack round-trips and validates against its own fingerprint")

    # THE test that matters: same COUNT, same shapes, DIFFERENT texts -> must raise.
    relabelled = [["ALPHA one", "alpha two"], ["beta one", "beta two", "beta three"]]
    expect2 = dict(expect, sha256=texts_sha256(relabelled))
    assert texts_sha256(relabelled) != sha, "hash did not change with the texts"
    try:
        _assert_pack(path, back, meta, expect2)
        raise SystemExit("FAIL: stale cache was accepted -- this is the 18.8 defect")
    except CacheMismatch as exc:
        print("OK: stale cache REJECTED loudly ->", str(exc).splitlines()[1].strip())

    # Same texts, different SEED -> must also raise (the hard half-split reshuffles).
    try:
        _assert_pack(path, back, meta, dict(expect, seed=1))
        raise SystemExit("FAIL: seed change was accepted")
    except CacheMismatch as exc:
        print("OK: seed change REJECTED ->", str(exc).splitlines()[1].strip())

    try:
        os.remove(path)
        os.rmdir(tmp)
    except OSError:
        pass
    print("embed_ct.py self-test OK")
