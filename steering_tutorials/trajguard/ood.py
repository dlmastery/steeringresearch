"""ood.py — the out-of-distribution arm (CLAUDE.md section 17 rule 5).

Rule 5 asks for a real released HuggingFace benchmark as an OOD test wherever one
exists. The superseded lesson had **no OOD arm at all**: 100% of its 600 completions
came from one pool, one model, one layer, one seed, one 5-fold CV.

WHICH BENCHMARK, AND WHY NOT THE OBVIOUS ONE
--------------------------------------------
``intrinsec-ai/cstm-bench`` is the benchmark rule 8 names for this lesson *family*, and
it is the right OOD for ``cross_trajectory`` / ``meerkat``. It is the WRONG one here:
108 rows of multi-session AGENT TRACES, whose unit is a session, not a generated token.
It is also not present in this host's HuggingFace cache, and this host has no HF token.
Stated rather than quietly skipped.

``jackhhao/jailbreak-classification`` is ungated, cached, and is the right granularity:
single-turn prompts labelled ``jailbreak`` vs ``benign``. After the shared dedup it
holds **629 unique jailbreak / 1,323 unique benign** prompts. Its jailbreak class is
in-the-wild DAN / roleplay templates, a genuine distribution shift from toxic-chat's
human ``jailbreaking`` annotations on live Vicuna traffic -- which is what makes it OOD
rather than a second test split.

THE LENGTH PROBLEM, MEASURED AND HANDLED
-----------------------------------------
Its jailbreak prompts are far longer than its benign ones (median 1,544 vs 234 chars),
and the benign class has no long tail to match against. Decile length-matching alone
therefore leaves a residual **length bar of 0.7570** at 629/class -- an OOD number under
a 0.76 length bar is not an OOD number. Restricting the positives to the benign class's
own p90 length ceiling (1,311 chars) drops the length bar to **0.5627** at 274/class.

Both are MEASURED (CPU, no model, `python -m steering_tutorials.trajguard.ood`), both
are reported, and the restricted arm is the default. The restricted arm is
**pool-limited at 274/class**, below the 500 floor -- it is an OOD probe, not a MAIN
set, and the MAIN set (`substrate=overt`, 500/class) is what clears rule 1.

CPU-only to import. The single model load lives in :func:`build_ood_trajectories`.
ASCII stdout only.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from steering_tutorials.common import confound as CF

from . import config as C
from . import data as D

SRC_OOD = "jackhhao/jailbreak-classification"


def _load_ood_rows():
    """Deduped ``{prompt, group_id, label}`` rows from the cached OOD benchmark.

    Uses ``common.data``'s own ``_clean`` / ``_norm_key`` / ``_group_id`` and the same
    ``MIN_CHARS`` / ``MAX_CHARS`` filters, so the OOD set is rendered EXACTLY as the
    in-domain set is. Rendering the two channels differently is how ``biencoder_guard``
    got a 0.72 length AUC out of nothing (CLAUDE.md section 17 rule 7).
    """
    import pandas as pd

    from steering_tutorials.common.data import (
        MAX_CHARS, MIN_CHARS, _clean, _dl, _group_id, _norm_key,
    )

    df = pd.read_csv(_dl(C.OOD_DATASET, C.OOD_FILE))
    groups: dict = {}
    for raw, kind in zip(df["prompt"].astype(str), df["type"].astype(str)):
        display = _clean(raw)
        if not (MIN_CHARS <= len(display) <= MAX_CHARS):
            continue
        norm = _norm_key(raw)
        if not norm:
            continue
        g = groups.setdefault(_group_id(norm), {"display": display, "kinds": set()})
        g["kinds"].add(kind.strip().lower())

    rows, n_ambiguous = [], 0
    for gid, g in groups.items():
        if len(g["kinds"]) > 1:  # same normalized prompt under both labels
            n_ambiguous += 1
            continue
        kind = next(iter(g["kinds"]))
        if kind not in ("jailbreak", "benign"):
            continue
        rows.append({"prompt": g["display"], "group_id": gid,
                     "label": 1 if kind == "jailbreak" else 0, "source": SRC_OOD})
    return rows, {"n_rows_raw": int(len(df)), "n_dropped_ambiguous": int(n_ambiguous)}


def select_ood_prompts(n_per_class: int = C.OOD_N_PER_CLASS,
                       seed: int = C.SEED,
                       length_cap_quantile: float = C.OOD_LENGTH_CAP_QUANTILE) -> dict:
    """Choose the OOD prompts, length-capped then length-matched. CPU-only, no model.

    ``length_cap_quantile`` caps the POSITIVE class at that quantile of the BENIGN
    length distribution, so the negatives can actually cover the positives' histogram.
    Set ``TG_OOD_LENCAP=1.0`` to disable the cap and take the larger, worse-matched arm.
    """
    from steering_tutorials.common.data import _length_matched_sample, _median_len

    rng = random.Random(seed)
    rows, stats = _load_ood_rows()
    pos_all = [r for r in rows if r["label"] == 1]
    neg_all = [r for r in rows if r["label"] == 0]

    q = float(length_cap_quantile)
    if 0.0 < q < 1.0 and neg_all:
        nl = sorted(len(r["prompt"]) for r in neg_all)
        cap = nl[min(len(nl) - 1, int(q * len(nl)))]
        pos = [r for r in pos_all if len(r["prompt"]) <= cap]
    else:
        cap = None
        pos = list(pos_all)

    pool = min(len(pos), len(neg_all))
    target = min(int(n_per_class), pool)
    rng.shuffle(pos)
    pos = pos[:target]
    neg = _length_matched_sample(neg_all, pos, target, rng)

    header = {
        "source": SRC_OOD,
        "file": C.OOD_FILE,
        "role": "OOD (out-of-distribution transfer), NOT a MAIN set",
        "length_cap_quantile": q,
        "length_cap_chars": cap,
        "n_pool_jailbreak_unique": len(pos_all),
        "n_pool_jailbreak_after_length_cap": len(pos) if cap is None else
            len([r for r in pos_all if len(r["prompt"]) <= cap]),
        "n_pool_benign_unique": len(neg_all),
        "n_per_class_requested": int(n_per_class),
        "n_effective_target": int(target),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "pool_limited": bool(pool < int(n_per_class)),
        "rule1_compliant": bool(target >= C.RULE1_FLOOR),
        "median_char_length": {"jailbreak": _median_len(pos), "benign": _median_len(neg)},
        "source_stats": stats,
        "seed": int(seed),
    }
    if header["pool_limited"]:
        print("[trajguard.ood] POOL-LIMITED: %d/class available after the p%.0f length "
              "cap; requested %d. This is an OOD probe, not the MAIN set."
              % (target, q * 100, n_per_class), file=sys.stderr)
    return {"positive": pos, "negative": neg, "header": header}


def ood_prompt_confound(sel: dict, seed: int = C.SEED) -> dict:
    """Prompt-channel bars for the OOD arm, measured before any generation."""
    prompts = [r["prompt"] for r in sel["positive"]] + [r["prompt"] for r in sel["negative"]]
    labels = [1] * len(sel["positive"]) + [0] * len(sel["negative"])
    return D.prompt_confound_report(prompts, labels, seed=seed)


def build_ood_trajectories(n_per_class: int = C.OOD_N_PER_CLASS,
                           seed: int = C.SEED,
                           max_new_tokens: int = C.MAX_NEW_TOKENS,
                           layer: int = C.LAYER,
                           model_id: str = C.MODEL_ID,
                           cache_path=C.OOD_CACHE) -> dict:
    """Generate + capture OOD trajectories with the SAME pipeline. LOADS THE MODEL."""
    from steering_tutorials.hello_world_steering.model_utils import load_model

    from . import trajectory

    sel = select_ood_prompts(n_per_class, seed)
    header = sel["header"]
    items = ([(r["prompt"], r["group_id"], 1) for r in sel["positive"]]
             + [(r["prompt"], r["group_id"], 0) for r in sel["negative"]])
    snapshot = D.config_snapshot("ood:" + SRC_OOD, header["n_effective_target"], seed,
                                 max_new_tokens, layer, model_id)
    fingerprint = D.dataset_fingerprint(snapshot, [g for _, g, _ in items])

    model, tok = load_model(model_id)
    trajectories, labels, prompts, completions, gids = [], [], [], [], []
    n_skipped = 0
    for i, (prompt, gid, label) in enumerate(items):
        completion, traj = trajectory.generate_and_capture(
            model, tok, prompt, max_new_tokens, layer)
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:
            n_skipped += 1
            continue
        trajectories.append(traj)
        labels.append(int(label))
        prompts.append(prompt)
        completions.append(completion)
        gids.append(gid)
        if (i + 1) % 20 == 0:
            print("[trajguard.ood] %d/%d captured" % (i + 1, len(items)), file=sys.stderr)

    header = dict(header)
    header["n_skipped_empty_trajectory"] = int(n_skipped)
    header["n_captured"] = len(trajectories)

    D._save_cache(cache_path, trajectories, labels, prompts, completions,
                  snapshot, fingerprint, gids, header)
    D.write_meta(C.OOD_META_PATH, trajectories, labels, completions, gids,
                 snapshot, fingerprint, header)
    return {"trajectories": trajectories, "labels": labels, "prompts": prompts,
            "completions": completions, "group_ids": gids, "fingerprint": fingerprint,
            "config_snapshot": snapshot, "header": header}


def load_or_build_ood(n_per_class: int = C.OOD_N_PER_CLASS, seed: int = C.SEED,
                      max_new_tokens: int = C.MAX_NEW_TOKENS, layer: int = C.LAYER,
                      model_id: str = C.MODEL_ID, cache_path=C.OOD_CACHE,
                      rebuild: bool = False) -> dict:
    """Fingerprint-checked load, same contract as ``data.load_or_build``."""
    sel = select_ood_prompts(n_per_class, seed)
    header = sel["header"]
    gids = ([r["group_id"] for r in sel["positive"]] + [r["group_id"] for r in sel["negative"]])
    snapshot = D.config_snapshot("ood:" + SRC_OOD, header["n_effective_target"], seed,
                                 max_new_tokens, layer, model_id)
    expected = D.dataset_fingerprint(snapshot, gids)

    cached = None if rebuild else D._load_cache(cache_path)
    if cached is not None and cached["labels"]:
        if cached.get("fingerprint") == expected:
            cached["header"] = header
            print("[trajguard.ood] cache HIT %s : fingerprint %s (%d completions)"
                  % (Path(cache_path).name, expected[:12], len(cached["labels"])),
                  file=sys.stderr)
            return cached
        raise D.CacheMismatch(
            "OOD cache %s was built for a different configuration and will NOT be "
            "reused (cached=%s wanted=%s). Re-run with TG_REBUILD=1."
            % (Path(cache_path).name, (cached.get("fingerprint") or "<none>")[:16],
               expected[:16]))
    return build_ood_trajectories(n_per_class, seed, max_new_tokens, layer, model_id,
                                  cache_path)


if __name__ == "__main__":
    # CPU-only: selection + the measured prompt-channel bars at both length caps.
    for q in (C.OOD_LENGTH_CAP_QUANTILE, 1.0):
        sel = select_ood_prompts(C.OOD_N_PER_CLASS, C.SEED, length_cap_quantile=q)
        h = sel["header"]
        print("")
        print("=== OOD %s  length_cap_quantile=%.2f (cap=%s chars) ==="
              % (SRC_OOD, q, h["length_cap_chars"]))
        print("  n=%d/class (pool jailbreak=%d benign=%d, pool_limited=%s, rule1=%s)"
              % (h["n_effective_target"], h["n_pool_jailbreak_unique"],
                 h["n_pool_benign_unique"], h["pool_limited"], h["rule1_compliant"]))
        print("  median chars jailbreak=%d benign=%d"
              % (h["median_char_length"]["jailbreak"], h["median_char_length"]["benign"]))
        print(CF.format_report(ood_prompt_confound(sel)))
