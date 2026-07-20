"""embed.py — turn-level embedders + a ragged sequence cache.

Each USER turn of a conversation is embedded to ONE vector; a conversation is
the SEQUENCE of those per-turn vectors ([n_turns, dim]). Two interchangeable
embedders are offered (an ablation, per config.EMBEDDERS):

  "gemma"  : the course's abliterated Gemma-3-1B (lesson-1/2), layer-C.GEMMA_LAYER
             mean-pooled residual-stream activation. Reuses lesson-2's
             `hello_world_steering.model_utils` plumbing. Loaded from the LOCAL
             path (the gated HF id 401s without a token).
  "minilm" : sentence-transformers/all-MiniLM-L6-v2 loaded via plain
             `transformers` AutoModel + attention-mask mean pooling ([384]).
             NO sentence-transformers dependency.

The model for either embedder is loaded ONCE, lazily, inside get_embedder — never
at import (so `import embed` stays CPU-cheap and model-free).

Cache format (.npz ragged pack): all per-turn vectors of the whole dataset are
vstacked into one float32 array `flat` ([sum_turns, dim]); `turn_counts` (int)
records how many turns each conversation contributed, so `flat` splits back into
the per-conversation [n_turns, dim] list; `dim` (int) is stored for validation.
"""
from __future__ import annotations

import numpy as np

from steering_tutorials.multiturn_jailbreak import config as C

GEMMA_DIM_HINT = 1152  # gemma-3-1b hidden; real dim taken from the model at runtime


# --- Embedders ---------------------------------------------------------------
def get_embedder(method: str):
    """Return ``(embed_turn, dim)`` for ``method`` in {"gemma","minilm"}.

    ``embed_turn(text:str) -> np.ndarray[dim]`` (float32, 1-D). The underlying
    model is loaded ONCE here (lazily) and closed over by ``embed_turn``.
    """
    if method == "gemma":
        return _get_gemma_embedder()
    if method == "minilm":
        return _get_minilm_embedder()
    raise ValueError("method must be 'gemma' or 'minilm', got %r" % (method,))


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


# --- Conversation / dataset embedding ----------------------------------------
def embed_conversation(turns, embed_turn) -> np.ndarray:
    """Embed a conversation (list of user-turn strings) -> [n_turns, dim] float32."""
    if len(turns) == 0:
        raise ValueError("cannot embed an empty conversation")
    vecs = [np.asarray(embed_turn(t), dtype=np.float32).reshape(-1) for t in turns]
    return np.stack(vecs, axis=0).astype(np.float32)


def embed_dataset(conversations, method, cache_path):
    """Embed every conversation -> list of [n_turns, dim]. Saves the ragged pack."""
    embed_turn, dim = get_embedder(method)
    seqs = [embed_conversation(list(conv), embed_turn) for conv in conversations]
    _save_pack(cache_path, seqs, dim)
    return seqs


def load_or_build(conversations, method, cache_path):
    """Return cached seqs if present + aligned with ``conversations``, else build."""
    cached = _try_load_pack(cache_path)
    if cached is not None and len(cached) == len(conversations):
        return cached
    return embed_dataset(conversations, method, cache_path)


# --- Ragged .npz pack/unpack -------------------------------------------------
def _save_pack(cache_path, seqs, dim) -> None:
    import os

    seqs = [np.asarray(s, dtype=np.float32) for s in seqs]
    if seqs:
        flat = np.vstack(seqs).astype(np.float32)
    else:
        flat = np.zeros((0, int(dim)), dtype=np.float32)
    turn_counts = np.asarray([s.shape[0] for s in seqs], dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(str(cache_path))), exist_ok=True)
    np.savez(
        str(cache_path),
        flat=flat,
        turn_counts=turn_counts,
        dim=np.asarray(int(dim), dtype=np.int64),
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
    import os

    p = str(cache_path)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        flat = np.asarray(z["flat"], dtype=np.float32)
        turn_counts = np.asarray(z["turn_counts"], dtype=np.int64)
    return _split_flat(flat, turn_counts)


# --- CPU self-test (no model, no dataset) ------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    dim = 8
    ks = [3, 4, 5]
    seqs = [rng.standard_normal((k, dim)).astype(np.float32) for k in ks]

    import os
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="mj_embed_test_")
    cache = os.path.join(tmpdir, "roundtrip.npz")

    _save_pack(cache, seqs, dim)
    loaded = _try_load_pack(cache)

    assert loaded is not None, "cache failed to load"
    assert len(loaded) == len(seqs), "conversation count changed on round-trip"
    for orig, back in zip(seqs, loaded):
        assert orig.shape == back.shape, "shape changed: %r vs %r" % (
            orig.shape, back.shape)
        assert back.dtype == np.float32, "dtype not float32: %r" % (back.dtype,)
        assert np.array_equal(orig, back), "values did not round-trip exactly"

    # load_or_build must accept the cache when the count matches (no model load).
    fake_convs = [["x"] * k for k in ks]
    reused = load_or_build(fake_convs, "gemma", cache)
    assert len(reused) == len(ks), "load_or_build did not reuse the cache"
    for orig, back in zip(seqs, reused):
        assert np.array_equal(orig, back), "load_or_build round-trip mismatch"

    try:
        os.remove(cache)
        os.rmdir(tmpdir)
    except OSError:
        pass

    print("embed.py self-test OK: ragged npz pack/unpack round-trips exactly")
    print("  conversations=%d turn_counts=%s dim=%d" % (
        len(seqs), [int(s.shape[0]) for s in seqs], dim))


if __name__ == "__main__":
    _self_test()
