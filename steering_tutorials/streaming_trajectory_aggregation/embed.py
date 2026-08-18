"""embed.py -- turns `Corpus` step text into per-trajectory `[n_steps, dim]` arrays,
via the `types.Embedder` Protocol, with a content-fingerprinted ragged disk cache.

Mirrors `multiturn_jailbreak/embed.py`'s cache discipline (that module's own docstring
names the defect this guards against: a cache validated on ROW COUNT ALONE returned
stale vectors under new labels after a seed change -- CLAUDE.md section 18.8). Here the
unit is a STEP, not a turn, and the cache key adds `n_per_class` (the corpus loaders'
own request knob) alongside `embedder_id` / corpus name / seed, per the brief.

CACHE FORMAT (ragged compressed .npz pack, `config.EMBED_CACHE_DIR`)
----------------------------------------------------------------------
  flat         : float32 [sum(step_counts), dim]  -- vstack of every trajectory's steps,
                 in `corpus.trajectories` order.
  step_counts  : int64 [n_trajectories]            -- per-trajectory step count.
  dim          : int64 scalar.
  meta_json    : a 0-d unicode array holding {version, texts_sha256, embedder_id,
                 corpus_name, n_per_class, seed, n_trajectories,
                 encoder_behaviour{...}}.

`texts_sha256` is a SHA-256 over the CONCATENATED step texts, in trajectory order --
not the requested config alone. On load the fingerprint is RECOMPUTED from the corpus
handed to `load_or_build_embeddings` and compared; a mismatch (or an unstamped legacy
pack) is refused loudly rather than silently reused. Row-count-only validation is the
named defect (this lesson's brief, section 18.8) that bit `cross_trajectory` and
`multiturn_jailbreak` before this fingerprint discipline existed.

THE DATA FINGERPRINT IS NOT ENOUGH -- the encoder's BEHAVIOUR is also an input
----------------------------------------------------------------------
On 2026-08-17 EmbeddingGemma was found to have been running strictly CAUSAL under
transformers 4.55.0 while its config said `use_bidirectional_attention=True`; the
library simply had no such parameter and dropped it (max abs difference over a shared
9-token prefix: EXACTLY 0.0, vs ~5.74 after the upgrade). The vectors in every cache
were therefore produced by an instrument that was not the instrument claimed. Because
the key hashed only the DATA, a re-run after the library fix would have silently
REUSED those causal vectors -- the mechanism by which the bug survives its own fix.

So every pack now also carries an `encoder_behaviour` block
(`biencoder_guard.encoder_behaviour`, shared verbatim by both lessons):
  * the CHEAP half -- transformers / sentence-transformers / torch versions + the
    model id -- is folded into the cache FILENAME, so a pack written under a
    different stack cannot even be reached, and a mismatch is named on sight without
    loading anything;
  * the BEHAVIOURAL half -- the quantised prefix-invariance delta -- is stored in the
    meta. A version string is necessary but NOT sufficient: the whole incident is a
    config field and a version that both looked right while the behaviour was wrong.

GATING: the probe needs a forward pass, so it runs ONLY when a cache is being WRITTEN
(the model is loaded anyway at that point) and at most ONCE per process. A cache HIT
compares the cheap fields only and returns without loading a model; reading never
probes. `STA_EMBED_VERIFY_BEHAVIOUR=1` opts into probing on read as well, for an audit.

CPU-only to import. Loads NO model at import time -- `get_embedder()` loads a model
lazily, only when actually called. ASCII stdout/stderr only (Windows cp1252).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from steering_tutorials.biencoder_guard import encoder_behaviour as EB
from steering_tutorials.streaming_trajectory_aggregation import config as C
from steering_tutorials.streaming_trajectory_aggregation.types import Corpus, Embedder

__all__ = [
    "EmbeddingGemmaEmbedder", "GemmaDecoderEmbedder", "get_embedder",
    "corpus_texts_fingerprint", "cache_path", "embed_corpus",
    "load_or_build_embeddings", "model_id_for", "behaviour_block",
]

# 2 (was 1): packs now carry an `encoder_behaviour` block. A version-1 pack cannot
# prove which attention semantics produced it, so it is refused rather than trusted.
CACHE_FORMAT_VERSION = 2


def _eprint(*args) -> None:
    """ASCII-only stderr print (Windows cp1252 console crashes on unicode)."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=__import__("sys").stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=__import__("sys").stderr)


# --- Embedders (types.Embedder Protocol: .name, .dim, .encode(texts) -> [n, dim]) --
class EmbeddingGemmaEmbedder:
    """EmbeddingGemma-300m via sentence-transformers -- the mandated default (CLAUDE.md
    section 17). Tries the HF id first, falls back to the local snapshot on any load
    failure (the weights ARE on disk here; see module docstring / config.py).

    Step text is a rendered agent-trajectory step ("role: content"), not a search
    query, so it is encoded under the model's "document" prompt when the loaded build
    registers named prompts, and under the documented fallback prefix otherwise --
    mirroring `multiturn_jailbreak/embed.py`'s handling of EmbeddingGemma's
    task-prefixed prompts (that module uses the "query" prompt for turns; a
    trajectory step is closer to a document being indexed than a search query, so
    this uses "document").
    """

    def __init__(self, model_id: str = None, local_path: str = None):
        # HOST GUARD -- must run BEFORE sentence_transformers is imported.
        # sentence_transformers -> transformers.integrations -> modeling_tf_utils
        # -> activations_tf, which RAISES on this host: "Your currently installed
        # version of Keras is Keras 3, but this is not yet supported in
        # Transformers." That killed the first live AgentDojo run AFTER the corpus
        # had downloaded. USE_TF=0 makes transformers skip the TF import path
        # entirely -- no `pip install tf-keras`, no environment mutation. Same
        # spirit as the truststore guard the loaders carry for this host's SSL
        # middlebox: fix the host quirk in code, not in a README instruction
        # somebody has to remember.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        from sentence_transformers import SentenceTransformer  # lazy, heavy

        model_id = model_id or C.EMBGEMMA_ID
        local_path = local_path or C.EMBGEMMA_LOCAL
        try:
            self._model = SentenceTransformer(model_id)
            self.name = model_id
        except Exception as exc:  # gated / offline / 401 -> the local snapshot on disk
            _eprint("[embed] HF load of %s failed (%s); using local %s"
                    % (model_id, type(exc).__name__, local_path))
            self._model = SentenceTransformer(local_path)
            self.name = local_path
        self._prompt_name = "document"
        self._fallback_prefix = "title: none | text: "
        self.dim = int(np.asarray(self._encode_batch(["hello"])).shape[-1])

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        try:
            vecs = self._model.encode(
                texts, prompt_name=self._prompt_name, convert_to_numpy=True,
                normalize_embeddings=False, show_progress_bar=False,
            )
        except (ValueError, KeyError, TypeError):
            vecs = self._model.encode(
                [self._fallback_prefix + t for t in texts], convert_to_numpy=True,
                normalize_embeddings=False, show_progress_bar=False,
            )
        return np.asarray(vecs, dtype=np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode_batch(texts).astype(np.float32)


class GemmaDecoderEmbedder:
    """Documented fallback / ablation arm, NOT the mandated default: the course's
    Gemma-3-1B decoder, layer-`config.GEMMA_LAYER` residual stream, mean-pooled over
    one step's tokens. Reuses lesson-2's `hello_world_steering.model_utils` plumbing
    exactly as `multiturn_jailbreak/embed.py`'s "gemma" arm does.
    """

    def __init__(self, model_id: str = None, layer: int = None):
        from steering_tutorials.hello_world_steering.model_utils import (
            load_model, mean_pool_activation,
        )

        self._mean_pool_activation = mean_pool_activation
        self._model, self._tok = load_model(model_id or C.GEMMA_MODEL_ID)
        self.layer = int(C.GEMMA_LAYER if layer is None else layer)
        self.name = "%s@layer%d" % (model_id or C.GEMMA_MODEL_ID, self.layer)
        self.dim = int(self._encode_one("hello").shape[0])

    def _encode_one(self, text: str) -> np.ndarray:
        vec = self._mean_pool_activation(self._model, self._tok, str(text), self.layer)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._encode_one(t) for t in texts], axis=0).astype(np.float32)


def get_embedder(method: str = None) -> Embedder:
    """`config.EMBEDDER` by default. The model is loaded ONCE inside the constructor,
    never at import."""
    method = method or C.EMBEDDER
    if method == "embgemma":
        return EmbeddingGemmaEmbedder()
    if method == "gemma":
        return GemmaDecoderEmbedder()
    raise ValueError("method must be one of %r, got %r" % (list(C.KNOWN_EMBEDDERS), method))


# --- Content fingerprint -------------------------------------------------------
def corpus_texts_fingerprint(corpus: Corpus, embedder_id: str, n_per_class: int,
                              seed: int) -> dict:
    """The cache identity: SHA-256 over every trajectory's step texts, in order,
    PLUS the embedder id / requested n_per_class / seed. `dim` is checked separately
    on load (unknown before the model loads)."""
    h = hashlib.sha256()
    for traj in corpus.trajectories:
        for step in traj.steps:
            h.update(str(step.text).encode("utf-8", "replace"))
            h.update(b"\x01")
        h.update(b"\x02")
    return {
        "version": CACHE_FORMAT_VERSION,
        "texts_sha256": h.hexdigest(),
        "embedder_id": str(embedder_id),
        "corpus_name": str(corpus.name),
        "n_per_class": int(n_per_class),
        "seed": int(seed),
        "n_trajectories": int(len(corpus.trajectories)),
    }


# --- Encoder-behaviour fingerprint ------------------------------------------------
def model_id_for(embedder_id: str) -> str:
    """The concrete MODEL the method name resolves to.

    `embedder_id` here is a METHOD ("embgemma" / "gemma"); the behaviour block wants
    the actual checkpoint, because that -- not the method label -- is what attends.
    """
    if embedder_id == "embgemma":
        return str(C.EMBGEMMA_ID)
    if embedder_id == "gemma":
        return "%s@layer%d" % (C.GEMMA_MODEL_ID, int(C.GEMMA_LAYER))
    return str(embedder_id)


def behaviour_block(embedder_id: str, embedder=None) -> dict:
    """The encoder-behaviour block for `embedder_id`.

    With `embedder=None` (a cache READ) this is the CHEAP block -- package metadata
    only, no model load, `prefix_delta_bucket` left as NOT_PROBED. With a live
    `embedder` (a cache WRITE, where the model is loaded anyway) it additionally
    carries the measured prefix-invariance bucket. Never called at import time.
    """
    model_id = model_id_for(embedder_id)
    if embedder is None:
        return EB.cheap_fingerprint(model_id)
    return EB.probe_fingerprint(embedder, model_id=model_id)


def _describe_mismatch(want: dict, got: dict | None) -> str:
    if not got:
        return ("cache carries NO fingerprint (an unstamped / pre-fingerprint pack) -- "
                "treated as invalid, never trusted on row count alone (section 18.8)")
    diffs = []
    for k in want:
        a, b = want.get(k), got.get(k)
        if a != b:
            sa = str(a)[:16] if k == "texts_sha256" else str(a)
            sb = str(b)[:16] if k == "texts_sha256" else str(b)
            diffs.append("%s: expected %s, cache has %s" % (k, sa, sb))
    return "; ".join(diffs) or "unknown difference"


# --- cache path ------------------------------------------------------------------
def cache_path(embedder_id: str, corpus_name: str, n_per_class: int, seed: int) -> Path:
    """The pack's filename -- keyed on the data request AND the encoder STACK.

    `EB.key_component` folds in the transformers / sentence-transformers / torch
    versions and the model id (metadata reads only -- computing a path must never
    load a model). A pack written under the causal transformers 4.55.0 therefore
    lands on a DIFFERENT filename than one written under 5.15.0, so the stale vectors
    cannot be picked up even by accident. The behavioural bucket cannot live here --
    it needs a forward pass -- so it is verified in the meta instead.
    """
    stack = EB.key_component(model_id_for(embedder_id))
    key_src = "%s|%s|%d|%d|%s" % (embedder_id, corpus_name, int(n_per_class), int(seed),
                                  stack)
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:24]
    return C.EMBED_CACHE_DIR / ("%s_%s_%s.npz" % (corpus_name, embedder_id.replace("/", "_"), key))


# --- ragged .npz pack/unpack -------------------------------------------------------
def _save_pack(path: Path, seqs: list[np.ndarray], dim: int, meta: dict) -> None:
    seqs = [np.asarray(s, dtype=np.float32) for s in seqs]
    flat = np.vstack(seqs).astype(np.float32) if seqs else np.zeros((0, int(dim)), dtype=np.float32)
    step_counts = np.asarray([s.shape[0] for s in seqs], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path), flat=flat, step_counts=step_counts,
        dim=np.asarray(int(dim), dtype=np.int64),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    _eprint("[embed] cache saved: %s (n=%d, dim=%d)" % (path, len(seqs), dim))


def _split_flat(flat: np.ndarray, step_counts: np.ndarray) -> list[np.ndarray]:
    seqs, off = [], 0
    for k in step_counts:
        k = int(k)
        seqs.append(np.asarray(flat[off:off + k], dtype=np.float32))
        off += k
    return seqs


def _try_load_pack(path: Path):
    """-> (seqs, meta_dict_or_None, dim) or None if the file is absent/unreadable."""
    if not path.exists():
        return None
    try:
        with np.load(str(path)) as z:
            flat = np.asarray(z["flat"], dtype=np.float32)
            step_counts = np.asarray(z["step_counts"], dtype=np.int64)
            dim = int(z["dim"]) if "dim" in z.files else int(flat.shape[1] if flat.size else 0)
            meta = None
            if "meta_json" in z.files:
                try:
                    meta = json.loads(str(z["meta_json"].item()))
                except Exception:
                    meta = None
    except Exception as exc:
        _eprint("[embed] cache %s unreadable (%s) -- rebuilding" % (path, exc))
        return None
    return _split_flat(flat, step_counts), meta, dim


# --- build / load-or-build ----------------------------------------------------------
def embed_corpus(corpus: Corpus, embedder: Embedder) -> list[np.ndarray]:
    """Embed every trajectory's steps -> list of `[n_steps, dim]` float32 arrays, in
    `corpus.trajectories` order. Batches ALL step texts across the whole corpus into
    one `encode()` call for throughput, then splits back per trajectory."""
    step_counts = [t.n_steps for t in corpus.trajectories]
    all_texts = [step.text for t in corpus.trajectories for step in t.steps]
    if not all_texts:
        return [np.zeros((0, embedder.dim), dtype=np.float32) for _ in corpus.trajectories]
    flat = np.asarray(embedder.encode(all_texts), dtype=np.float32)
    if flat.shape[0] != len(all_texts):
        raise RuntimeError(
            "embedder %r returned %d vectors for %d input texts -- encode() must be "
            "1:1, batching is broken" % (embedder.name, flat.shape[0], len(all_texts)))
    return _split_flat(flat, np.asarray(step_counts, dtype=np.int64))


def load_or_build_embeddings(corpus: Corpus, embedder_id: str = None, *,
                              n_per_class: int = None, seed: int = None,
                              on_mismatch: str = None) -> tuple[list[np.ndarray], int]:
    """Return `(embeddings, dim)`, `embeddings[i]` aligned to `corpus.trajectories[i]`.

    `on_mismatch` (default "raise", or env `STA_EMBED_ON_MISMATCH`):
      "raise"   -- a stale/unstamped/mismatched cache is a loud RuntimeError naming the
                   field that differs. Default, because a silent rebuild can quietly
                   consume a GPU pass nobody asked for, and a silent reuse is the
                   section-18.8 stale-vector defect this whole module exists to prevent.
      "rebuild" -- re-embed and overwrite. Never mislabels, but costs a model pass.
                   The rejected pack is QUARANTINED (renamed aside), never clobbered.

    TWO fingerprints are checked, and both must hold:
      * the DATA fingerprint (`texts_sha256` + the request knobs), and
      * the ENCODER-BEHAVIOUR block -- library stack now, measured attention
        behaviour at write time. See the module docstring for the 2026-08-17
        causal-EmbeddingGemma incident this second check exists for.
    Reading a matching cache loads NO model: only the cheap half is compared, unless
    `STA_EMBED_VERIFY_BEHAVIOUR=1` explicitly asks for the probe on read too.
    """
    embedder_id = embedder_id or C.EMBEDDER
    n_per_class = C.N_PER_CLASS if n_per_class is None else n_per_class
    seed = C.SEED if seed is None else seed
    on_mismatch = (on_mismatch or os.environ.get("STA_EMBED_ON_MISMATCH") or "raise").lower()
    if on_mismatch not in ("raise", "rebuild"):
        raise ValueError("on_mismatch must be 'raise' or 'rebuild', got %r" % (on_mismatch,))
    verify_behaviour = os.environ.get("STA_EMBED_VERIFY_BEHAVIOUR", "") == "1"

    path = cache_path(embedder_id, corpus.name, n_per_class, seed)
    want = corpus_texts_fingerprint(corpus, embedder_id, n_per_class, seed)
    loaded = _try_load_pack(path)

    if loaded is not None:
        cached, got_meta, cached_dim = loaded
        ok = bool(got_meta) and all(got_meta.get(k) == want[k] for k in want)

        # The behaviour check is SEPARATE from the data check and reported separately,
        # because "the texts changed" and "the encoder now attends differently" are
        # different failures with different fixes. Probing here would mean loading a
        # model on a cache hit, so by default only the cheap half is compared; the
        # opt-in below re-derives the measured bucket for an audit.
        probe_on_read = get_embedder(embedder_id) if verify_behaviour else None
        want_behaviour = behaviour_block(embedder_id, probe_on_read)
        behaviour_diffs = EB.compare(want_behaviour, (got_meta or {}).get("encoder_behaviour"))
        if behaviour_diffs:
            msg = EB.reject_message(path, want_behaviour,
                                    (got_meta or {}).get("encoder_behaviour"))
            if on_mismatch == "raise":
                raise RuntimeError(msg)
            _eprint("[embed] REJECTED (encoder behaviour): %s" % msg)
            EB.quarantine(path, tag="behaviour_mismatch")
        elif ok and len(cached) == len(corpus.trajectories):
            _eprint("[embed] cache HIT: %s (n=%d, dim=%d, data + encoder-behaviour "
                    "fingerprints verified)" % (path, len(cached), cached_dim))
            return cached, cached_dim

        if not behaviour_diffs:
            detail = _describe_mismatch(want, got_meta)
            if len(cached) != len(corpus.trajectories):
                detail += "; trajectory count %d vs %d" % (len(cached),
                                                           len(corpus.trajectories))
            msg = ("embedding cache %s does not match the corpus it is about to be used "
                   "for (%s). Refusing to reuse it silently -- this is CLAUDE.md section "
                   "18.8's stale-vector-under-new-labels defect. Re-run with "
                   "STA_EMBED_ON_MISMATCH=rebuild, or delete the file." % (path, detail))
            if on_mismatch == "raise":
                raise RuntimeError(msg)
            _eprint("[embed] REBUILD: %s" % msg)
            EB.quarantine(path, tag="data_mismatch")

    embedder = get_embedder(embedder_id)
    seqs = embed_corpus(corpus, embedder)
    meta = dict(want)
    meta["dim"] = int(embedder.dim)
    # The model is loaded RIGHT HERE, so the behavioural probe is nearly free -- one
    # forward pass, memoised per process. This is the only place it runs by default.
    meta["encoder_behaviour"] = behaviour_block(embedder_id, embedder)
    _save_pack(path, seqs, embedder.dim, meta)
    return seqs, int(embedder.dim)


# --- CPU self-test: python -m ...streaming_trajectory_aggregation.embed ------------
if __name__ == "__main__":
    import tempfile

    from steering_tutorials.streaming_trajectory_aggregation.types import Step, Trajectory

    class _FakeEmbedder:
        """Deterministic hash-based embedder -- no model, no network, for the
        round-trip / fingerprint self-test only."""

        name = "fake-hash-embedder"
        dim = 8

        def encode(self, texts):
            out = np.zeros((len(texts), self.dim), dtype=np.float32)
            for i, t in enumerate(texts):
                h = hashlib.sha256(str(t).encode("utf-8")).digest()
                out[i] = np.frombuffer(h[: self.dim * 4], dtype=np.uint32).astype(np.float32)[: self.dim]
            return out

    def _make_corpus(name, texts_per_traj):
        trajs = []
        for i, texts in enumerate(texts_per_traj):
            steps = tuple(Step(index=j, text=t) for j, t in enumerate(texts))
            trajs.append(Trajectory(uid="%s-%d" % (name, i), steps=steps, label=i % 2,
                                     group_id="g%d" % i, source=name))
        return Corpus(name=name, trajectories=trajs, requested_n_per_class=len(trajs),
                      pool_fingerprint="0" * 64, licence="unstated", label_provenance="smoke")

    corpus = _make_corpus("smoke", [["a", "b", "c"], ["d", "e"], ["f", "g", "h", "i"]])
    tmpdir = Path(tempfile.mkdtemp(prefix="sta_embed_test_"))
    old_dir = C.EMBED_CACHE_DIR
    C.EMBED_CACHE_DIR = tmpdir
    try:
        fake = _FakeEmbedder()
        seqs = embed_corpus(corpus, fake)
        assert [s.shape[0] for s in seqs] == [3, 2, 4], "step counts did not round-trip"
        assert all(s.shape[1] == fake.dim for s in seqs)

        path = cache_path(fake.name, corpus.name, len(corpus.trajectories), 0)
        meta = corpus_texts_fingerprint(corpus, fake.name, len(corpus.trajectories), 0)
        meta["dim"] = fake.dim
        # The CHEAP behaviour block -- no model is loaded to build it, which is the
        # point: this whole self-test runs on CPU with nothing on the GPU.
        meta["encoder_behaviour"] = behaviour_block(fake.name, None)
        assert meta["encoder_behaviour"]["prefix_delta_bucket"] == EB.NOT_PROBED
        _save_pack(path, seqs, fake.dim, meta)
        loaded, got_meta, dim = _try_load_pack(path)
        assert dim == fake.dim
        assert got_meta["texts_sha256"] == meta["texts_sha256"]
        assert got_meta["encoder_behaviour"]["libs"]["transformers"], (
            "the behaviour block did not survive the .npz round-trip -- an existing "
            "cache must be auditable without re-deriving anything")
        for a, b in zip(seqs, loaded):
            assert np.array_equal(a, b), "values did not round-trip exactly"

        # regression 1: same trajectory COUNT, different step TEXT -> must be refused.
        corpus2 = _make_corpus("smoke", [["a", "b", "X"], ["d", "e"], ["f", "g", "h", "i"]])
        try:
            load_or_build_embeddings(corpus2, fake.name, n_per_class=len(corpus2.trajectories),
                                      seed=0)
            raise AssertionError("mismatched cache was silently reused")
        except RuntimeError as exc:
            assert "texts_sha256" in str(exc), "mismatch message did not name the field"

        # regression 2: IDENTICAL data, different ENCODER BEHAVIOUR -> must be refused.
        # This is the 2026-08-17 incident in miniature. The pack below is byte-correct
        # for this corpus; the ONLY thing wrong with it is that it was produced by an
        # encoder that attended causally (transformers 4.55.0, bucket "causal"). The
        # data fingerprint has nothing to say about that, which is exactly why this
        # second fingerprint exists. No model is loaded to detect it.
        stale = dict(meta)
        stale["encoder_behaviour"] = dict(meta["encoder_behaviour"])
        stale["encoder_behaviour"]["libs"] = dict(stale["encoder_behaviour"]["libs"])
        stale["encoder_behaviour"]["libs"]["transformers"] = "4.55.0"
        stale["encoder_behaviour"]["prefix_delta_bucket"] = "causal"
        stale["encoder_behaviour"]["prefix_delta"] = 0.0
        _save_pack(path, seqs, fake.dim, stale)
        try:
            load_or_build_embeddings(corpus, fake.name, n_per_class=len(corpus.trajectories),
                                      seed=0)
            raise AssertionError("a cache computed under DIFFERENT ATTENTION SEMANTICS "
                                 "was silently reused -- the behaviour fingerprint is "
                                 "not binding")
        except RuntimeError as exc:
            assert "libs.transformers" in str(exc), \
                "the rejection did not name the library that changed"
            assert "CAUSAL" in str(exc), "the rejection did not cite the causal incident"

        # regression 3: SAME library versions, DIFFERENT measured behaviour.
        #
        # Read this one carefully, because it documents a real limit of the design.
        # By DEFAULT a cache read compares only the cheap fields -- it must not load a
        # model to answer "is this cache usable?" -- so a stored bucket is accepted
        # when the stack matches, on the argument that the same stack produces the
        # same behaviour. `STA_EMBED_VERIFY_BEHAVIOUR=1` drops that argument and
        # re-probes on read. Under that flag a disagreeing bucket must REJECT, and a
        # probe that cannot reach the backbone must record UNKNOWN and reject too --
        # a failed probe is not a pass. Verified here with the fake embedder (which
        # has no backbone at all), so no model is loaded.
        sneaky = dict(meta)
        sneaky["encoder_behaviour"] = dict(meta["encoder_behaviour"])
        sneaky["encoder_behaviour"]["prefix_delta_bucket"] = "causal"
        _save_pack(path, seqs, fake.dim, sneaky)
        _real_get_embedder = get_embedder
        globals()["get_embedder"] = lambda method=None: fake
        os.environ["STA_EMBED_VERIFY_BEHAVIOUR"] = "1"
        try:
            load_or_build_embeddings(corpus, fake.name, n_per_class=len(corpus.trajectories),
                                      seed=0)
            raise AssertionError("same-version/different-behaviour cache was reused "
                                 "under STA_EMBED_VERIFY_BEHAVIOUR=1")
        except RuntimeError as exc:
            assert "prefix_delta_bucket" in str(exc), \
                "the rejection did not name the behavioural field"
        finally:
            os.environ.pop("STA_EMBED_VERIFY_BEHAVIOUR", None)
            globals()["get_embedder"] = _real_get_embedder

        print("OK -- embed.py self-test: ragged pack round-trips exactly; the DATA "
              "fingerprint REJECTS a same-count/different-text cache (CLAUDE.md 18.8); "
              "the ENCODER-BEHAVIOUR fingerprint REJECTS a cache with identical data "
              "computed under causal-vs-bidirectional attention, and -- under "
              "STA_EMBED_VERIFY_BEHAVIOUR=1 -- rejects it again when only the measured "
              "behaviour moved. No model was loaded by any of it.")
    finally:
        C.EMBED_CACHE_DIR = old_dir
        for f in tmpdir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass
