"""data.py — build the per-token hidden-state trajectories the TRAJGUARD lesson
detects on.

For each prompt we run the abliterated Gemma-3-1B, greedily generate a completion,
and capture the layer-``C.LAYER`` residual-stream state at every GENERATED-token
position -> one trajectory ``[n_tokens, dim]``. Label = the prompt's class
(1 = harmful, 0 = benign). This is the sibling of ``multiturn_jailbreak.data``
(there a chunk is a conversation turn; here a chunk is a decoded token).

Prompts come from the shared >=500/class toxic-chat set
(``common.data.load_harmful_benign``); the model is loaded ONCE via
``hello_world_steering.model_utils.load_model``; capture is delegated to
``trajguard.trajectory.generate_and_capture``.

Cache format (``C.TRAJ_CACHE``, a ragged .npz pack):
  - ``flat``          : float32 ``[sum(token_counts), dim]`` = vstack of all token vecs
  - ``token_counts``  : int ``[n_completions]``  = per-completion token count
  - ``labels``        : int ``[n_completions]``
  - ``dim``           : int scalar
A JSON sidecar next to the npz stores ``prompts`` and ``completions`` (variable-
length strings are awkward inside npz). On load we split ``flat`` back into a list
of ``[n_tokens, dim]`` arrays via ``token_counts``.

CPU-only to write/import. The single allowed model load is the ``__main__`` smoke.
ASCII stdout only (Windows cp1252 console).
"""
from __future__ import annotations

try:  # OS trust store for an SSL-intercepting middlebox (same guard as siblings).
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional
    pass

import json
import sys
from pathlib import Path

import numpy as np

from . import config as C


def _sidecar_path(cache_path) -> Path:
    """JSON sidecar (prompts + completions) living next to the .npz cache."""
    return Path(cache_path).with_suffix(".json")


def _save_cache(cache_path, trajectories, labels, prompts, completions) -> None:
    """Write the ragged .npz pack + the JSON sidecar."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    token_counts = np.asarray([t.shape[0] for t in trajectories], dtype=np.int64)
    if trajectories:
        flat = np.vstack([np.asarray(t, dtype=np.float32) for t in trajectories])
        dim = int(trajectories[0].shape[1])
    else:  # nothing captured — keep a well-formed empty pack
        flat = np.zeros((0, 0), dtype=np.float32)
        dim = 0
    np.savez(
        cache_path,
        flat=flat.astype(np.float32),
        token_counts=token_counts,
        labels=np.asarray(labels, dtype=np.int64),
        dim=np.asarray(dim, dtype=np.int64),
    )
    _sidecar_path(cache_path).write_text(
        json.dumps({"prompts": list(prompts), "completions": list(completions)}),
        encoding="utf-8",
    )


def _load_cache(cache_path):
    """Load the ragged pack -> dataset dict, splitting ``flat`` back to a list.

    Returns ``None`` if either the .npz or the JSON sidecar is missing.
    """
    cache_path = Path(cache_path)
    sidecar = _sidecar_path(cache_path)
    if not (cache_path.exists() and sidecar.exists()):
        return None
    with np.load(cache_path, allow_pickle=False) as z:
        flat = z["flat"].astype(np.float32)
        token_counts = z["token_counts"].astype(int)
        labels = z["labels"].astype(int)
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
    }


def _mean_len(trajectories, labels, want):
    """Mean trajectory length over completions whose label == ``want``."""
    lens = [t.shape[0] for t, y in zip(trajectories, labels) if y == want]
    return float(np.mean(lens)) if lens else 0.0


def build_token_trajectories(
    n_per_class: int = C.N_PER_CLASS,
    seed: int = C.SEED,
    max_new_tokens: int = C.MAX_NEW_TOKENS,
    cache_path=C.TRAJ_CACHE,
) -> dict:
    """Generate + capture per-token trajectories for ``n_per_class`` prompts/class.

    Reuses the cache iff it is present AND its ``token_counts`` length matches the
    expected number of completions (``2 * n_per_class``); otherwise loads the model
    ONCE and regenerates. Skips any trajectory with 0 captured tokens (early EOS).
    Returns ``{"trajectories": list[np.ndarray], "labels": list[int],
    "prompts": list[str], "completions": list[str]}``.
    """
    expected = 2 * n_per_class

    cached = _load_cache(cache_path)
    if cached is not None and len(cached["labels"]) == expected:
        n_h = sum(1 for y in cached["labels"] if y == 1)
        n_b = sum(1 for y in cached["labels"] if y == 0)
        print(
            "[trajguard.data] reusing cache %s : harmful=%d benign=%d "
            "mean_len harmful=%.1f benign=%.1f"
            % (
                Path(cache_path).name,
                n_h,
                n_b,
                _mean_len(cached["trajectories"], cached["labels"], 1),
                _mean_len(cached["trajectories"], cached["labels"], 0),
            ),
            file=sys.stderr,
        )
        return cached

    # ---- build fresh (the only model load in this module) --------------------
    from steering_tutorials.common.data import load_harmful_benign
    from steering_tutorials.hello_world_steering.model_utils import load_model
    from . import trajectory  # lazy: keeps `import ...data` model-free

    d = load_harmful_benign(n_per_class, seed)
    items = [(p, 1) for p in d["harmful"]] + [(p, 0) for p in d["benign"]]

    model, tok = load_model(C.MODEL_ID)

    trajectories, labels, prompts, completions = [], [], [], []
    n_skipped = 0
    for i, (prompt, label) in enumerate(items):
        completion, traj = trajectory.generate_and_capture(
            model, tok, prompt, max_new_tokens, C.LAYER
        )
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:  # early EOS -> nothing to detect on
            n_skipped += 1
            continue
        trajectories.append(traj)
        labels.append(int(label))
        prompts.append(prompt)
        completions.append(completion)
        if (i + 1) % 20 == 0:
            print("[trajguard.data] %d/%d captured" % (i + 1, len(items)),
                  file=sys.stderr)

    _save_cache(cache_path, trajectories, labels, prompts, completions)

    n_h = sum(1 for y in labels if y == 1)
    n_b = sum(1 for y in labels if y == 0)
    print(
        "[trajguard.data] built %d trajectories (harmful=%d benign=%d, skipped=%d) "
        "| mean_len harmful=%.1f benign=%.1f | cached -> %s"
        % (
            len(trajectories),
            n_h,
            n_b,
            n_skipped,
            _mean_len(trajectories, labels, 1),
            _mean_len(trajectories, labels, 0),
            Path(cache_path).name,
        ),
        file=sys.stderr,
    )
    return {
        "trajectories": trajectories,
        "labels": labels,
        "prompts": prompts,
        "completions": completions,
    }


def load_or_build(**kw) -> dict:
    """Return the cached dataset if present, else build it (convenience wrapper).

    Accepts the same kwargs as :func:`build_token_trajectories`.
    """
    cache_path = kw.get("cache_path", C.TRAJ_CACHE)
    cached = _load_cache(cache_path)
    if cached is not None and cached["labels"]:
        print("[trajguard.data] load_or_build: using cache %s (%d completions)"
              % (Path(cache_path).name, len(cached["labels"])), file=sys.stderr)
        return cached
    return build_token_trajectories(**kw)


if __name__ == "__main__":
    # SMALL smoke: loads the model ONCE and generates ~8 short completions.
    # This is the ONLY allowed model load in this file. ASCII stdout.
    import tempfile

    tmp = Path(tempfile.gettempdir()) / "trajguard_smoke_traj.npz"
    ds = build_token_trajectories(n_per_class=4, max_new_tokens=12, cache_path=tmp)

    trajs = ds["trajectories"]
    print("[smoke] n_trajectories = %d" % len(trajs))
    print("[smoke] labels         = %s" % ds["labels"])
    shapes = [tuple(int(x) for x in t.shape) for t in trajs]
    print("[smoke] shapes         = %s" % shapes)
    if trajs:
        dims = {t.shape[1] for t in trajs}
        print("[smoke] hidden dim(s)  = %s" % sorted(dims))
    if ds["completions"]:
        sample = ds["completions"][0].replace("\n", " ")
        print("[smoke] sample completion[0] = %r" % sample[:80])
    print("[smoke] OK")
