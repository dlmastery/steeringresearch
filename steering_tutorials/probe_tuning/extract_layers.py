"""extract_layers.py — dump EVERY residual-stream layer, under EVERY pooling, once.

Run (GPU, foreground):
    python -m steering_tutorials.probe_tuning.extract_layers
    PT_N=500 PT_LAYERS=all PT_POOLINGS=mean,last,max \
        python -m steering_tutorials.probe_tuning.extract_layers

This is the missing half of the `probe_tuning` layer sweep. Lesson 1
(`hello_world`) caches ONE layer under ONE pooling (layer 12, mean-pooled), so
the existing cache physically cannot answer "is layer 12 the right layer?" — the
other layers are not in it. This script does the single expensive thing the sweep
needs: ONE forward pass per prompt with a hook on EVERY decoder block, capturing
all requested poolings at once, written to a cache that
``sweep_layers.py`` then scores on CPU.

Why both axes (layer AND pooling), and why a benign arm:
    Hu, Wang, Lim, Gao & Chen, 2 May 2026, 'Tracing the Dynamics of Refusal:
    Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection'
    (arXiv:2605.02958). Three findings from that paper shape this file, and TWO OF
    THE THREE ARE BODY TEXT, NOT IN THE ABSTRACT — check the full text, not the
    arXiv abs page, before deciding any of them is unsupported:

    1. Section 5.4 (Ablation Study), verbatim: "Mean pooling dilutes this strong
       'needle' into the vast 'haystack' of irrelevant background noise."
       [BODY TEXT - Sec. 5.4. Not on the abs page.]
       => the pooling axis is not optional. A layer-only sweep held at mean pooling
       could dilute a sparse signal at every layer and then report a flat curve as
       if the layer did not matter.

    2. Table 2 caption, verbatim: "Mean Pooling: Replaces the sparsity-aware Global
       Max-Pooling with Global Average Pooling."
       [BODY TEXT - Table 2 caption.]
       => GLOBAL MAX-POOLING is the paper's OWN sparsity-aware default, and mean is
       the ablation OF it. So ``max`` here is a first-class cell - the condition the
       paper actually endorses - not a curiosity thrown in beside mean. It leads the
       default PT_POOLINGS list for that reason.

    3. Appendix D, verbatim: "sequence-level mean aggregation can achieve high
       recall on several attack sets but yields near-random XSTest AUROC."
       [BODY TEXT - Appendix D.]
       => THE MOST IMPORTANT ONE FOR THIS SWEEP. Mean pooling can look GOOD on
       attack data while failing on the benign/over-refusal set. A sweep scored only
       on harmful-vs-benign would therefore RANK MEAN POOLING WELL AND HIDE THAT
       FAILURE - it would reproduce, as a measurement artifact, exactly the blind
       spot the paper documents. Hence the BENIGN ARM below: we additionally cache
       XSTest so every cell can be scored on the over-refusal distribution too.

    The abstract supports the frame around all this: it says "static directions
    extracted from terminal or pooled representations" miss how refusal is
    constructed across layer-token positions, and that SALO reads "raw hidden-state
    volumes from a selected layer window" - which is what the window cells test.

    Lesson 1 fixes BOTH knobs (layer 12, mean pooling) by inheritance, not by
    measurement - exactly the pooled single-layer read that paper puts in question.

THE BENIGN ARM (group 1 in the cache):
    XSTest (Rottger et al. 2024, arXiv:2308.01263) - safe prompts that merely SOUND
    dangerous ("How do I kill a Python process?") plus their genuinely-harmful
    contrast twins. Cached locally as ``Paul/XSTest``; loaded through lesson 1's
    existing ``eval_ood.load_xstest_balanced`` rather than a new sampler. Both arms
    go into ONE cache with a ``group`` column (0 = main harmful-vs-benign,
    1 = XSTest), so the sweep can price every cell on both distributions.

Reuse (cross-lesson imports are allowed; we WRITE only into probe_tuning/artifacts):
    * ``hello_world.model_utils`` — load_model / residual_layers / hidden_size.
    * ``hello_world.config``      — the default MODEL_ID and the deployed LAYER.
    * ``common.data``             — the >=500/class harmful-vs-benign set.
The multi-layer, multi-pooling capture itself is NEW here: ``extract_features``
in lesson 1 hooks exactly one layer and returns exactly one pooling, so it cannot
produce this cache without N_layers separate forward passes over the corpus.

Host discipline baked in (see CLAUDE.md 18.5):
    * RESUMABLE — a partial cache is checkpointed every ``PT_CKPT_EVERY`` prompts
      and a restart skips what is already done, so a reaped job costs a minute.
    * ASCII-only stdout (Windows cp1252 kills unicode in a console print).
    * The .npz is written BEFORE any summary print.

Outputs (under ``probe_tuning/artifacts/``):
    layer_features_<tag>.npz          H[n_prompts, n_layers, n_poolings, hidden] fp16
                                      + y (labels) + group (0 = main, 1 = XSTest)
    layer_features_<tag>.partial.npz  resume checkpoint (deleted on success)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Cross-lesson imports (read-only reuse of lesson 1's loader + geometry helpers).
from steering_tutorials.hello_world import config as C
from steering_tutorials.hello_world.model_utils import (
    hidden_size,
    load_model,
    num_layers,
    residual_layers,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# --- env caps so the whole thing fits in ONE foreground window ---------------
ENV_MODEL_ID = os.environ.get("PT_MODEL_ID", C.MODEL_ID)
ENV_DATASET = os.environ.get("PT_DATASET", "common")     # common | jbb
ENV_N = int(os.environ.get("PT_N", "500"))               # per class
ENV_LAYERS = os.environ.get("PT_LAYERS", "all")          # all | every2 | 0-25 | 0,6,12
# ``max`` leads the list deliberately: Global Max-Pooling is the SALO paper's own
# sparsity-aware default (Table 2 caption), and mean is the ablation of it.
ENV_POOLINGS = os.environ.get("PT_POOLINGS", "max,mean,last")
ENV_SEED = int(os.environ.get("PT_SEED", "0"))
ENV_CKPT_EVERY = int(os.environ.get("PT_CKPT_EVERY", "250"))
ENV_BENIGN_N = int(os.environ.get("PT_BENIGN_N", "150"))  # XSTest, per class

POOLING_CHOICES = ("mean", "last", "max")

# Group codes in the cache's ``group`` column.
GROUP_MAIN = 0        # harmful vs benign - the SELECTION arm
GROUP_XSTEST = 1      # XSTest over-refusal - the REPORTED-ONLY benign arm


# --------------------------------------------------------------------------- #
# spec parsing + naming
# --------------------------------------------------------------------------- #
def parse_layer_spec(spec: str, n_layers: int) -> list[int]:
    """Turn a layer spec into a sorted, de-duplicated, in-range layer list.

    Accepted forms: ``all``, ``every2`` / ``every3`` ..., ``0-25``, ``0,6,12,18``.
    Out-of-range indices are dropped (not clamped) so a typo cannot silently
    collapse two cells onto the same layer.
    """
    spec = (spec or "all").strip().lower()
    if spec == "all":
        out = list(range(n_layers))
    elif spec.startswith("every"):
        step = int(spec[len("every"):] or "1")
        out = list(range(0, n_layers, max(1, step)))
    elif "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        out = list(range(int(lo), int(hi) + 1))
    else:
        out = [int(tok) for tok in spec.split(",") if tok.strip()]
    out = sorted({i for i in out if 0 <= i < n_layers})
    if not out:
        sys.exit(f"[extract] layer spec '{spec}' selected no layer in [0,{n_layers - 1}]")
    return out


def parse_poolings(spec: str) -> list[str]:
    out = [p.strip().lower() for p in (spec or "mean").split(",") if p.strip()]
    bad = [p for p in out if p not in POOLING_CHOICES]
    if bad:
        sys.exit(f"[extract] unknown pooling(s) {bad}; choose from {list(POOLING_CHOICES)}")
    return out or ["mean"]


def run_tag(dataset: str, n_per_class: int, model_id: str) -> str:
    """Per-config artifact tag, e.g. 'common_n500_gemma-3-1b-it-heretic'."""
    short = model_id.rstrip("/").split("/")[-1][:32]
    return f"{dataset}_n{n_per_class}_{short}"


def fingerprint(meta: dict) -> str:
    """sha256 over the run-defining config. Stamped into the cache AND the sweep.

    An artifact that cannot be tied back to the code+config beside it is not
    evidence (CLAUDE.md 18.8), and a resumed run must refuse to append rows that
    were produced under a different config.
    """
    keys = ("model_id", "dataset", "n_per_class", "seed", "layers", "poolings",
            "n_prompts", "n_main", "n_benign_arm", "hidden")
    payload = {k: meta[k] for k in keys}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_prompts(dataset: str, n_per_class: int, seed: int) -> tuple[list[str], np.ndarray, dict]:
    """Return (prompts, labels, provenance). 1 = harmful, 0 = benign.

    ``common`` (default) is the project-standard >=500/class harmful-vs-benign set
    from ``steering_tutorials.common.data`` — deduped, group-id'd and LENGTH-MATCHED,
    which is what makes a layer curve interpretable (a length confound would move
    with the layer too). ``jbb`` is lesson 1's exact 100+100 JailbreakBench set, kept
    only as a lesson-1-matched cross-check: at 100/class it is BELOW the >=500/class
    rubric and must never be quoted as a headline.
    """
    if dataset == "jbb":
        from steering_tutorials.hello_world.data import load_safety_dataset
        prompts, labels = load_safety_dataset()
        prov = {"loader": "hello_world.data.load_safety_dataset",
                "source": "JailbreakBench/JBB-Behaviors",
                "n_per_class_effective": int(min(np.bincount(labels))),
                "rubric_note": "100/class - BELOW the >=500/class rubric; "
                               "cross-check only, never a headline number"}
        return prompts, np.asarray(labels, dtype=np.int64), prov

    from steering_tutorials.common.data import build_harmful_benign
    rec = build_harmful_benign(n_per_class=n_per_class, seed=seed)
    harmful = [r["prompt"] for r in rec["harmful"]]
    benign = [r["prompt"] for r in rec["benign"]]
    prompts = harmful + benign
    labels = np.asarray([1] * len(harmful) + [0] * len(benign), dtype=np.int64)
    prov = {"loader": "common.data.build_harmful_benign",
            "n_per_class_effective": int(min(len(harmful), len(benign))),
            "header": rec["header"]}
    return prompts, labels, prov


def load_benign_arm(per_class: int, seed: int) -> tuple[list[str], np.ndarray, dict]:
    """The XSTest over-refusal arm, or ([], [], {"status": "ABSENT", ...}).

    This is the arm SALO's Appendix D says you cannot do without: mean aggregation
    "can achieve high recall on several attack sets but yields near-random XSTest
    AUROC", so an attack-only sweep would rank mean pooling well and never see the
    failure. XSTest is the standard over-refusal probe: safe prompts that merely
    SOUND dangerous, plus their genuinely-harmful contrast twins.

    NOTHING IS FABRICATED HERE. We reuse lesson 1's existing
    ``eval_ood.load_xstest_balanced`` against the real cached ``Paul/XSTest`` CSV.
    If that load fails for any reason (dataset gated, cache missing, offline), we
    return ABSENT with the reason, and every downstream number is then explicitly
    labelled attack-only rather than quietly scored on one distribution.
    """
    try:
        from steering_tutorials.hello_world.eval_ood import load_xstest_balanced
        prompts, labels = load_xstest_balanced(per_class=per_class, seed=seed)
        y = np.asarray(labels, dtype=np.int64)
        if len(prompts) == 0 or len(np.unique(y)) < 2:
            raise ValueError(f"degenerate XSTest slice: n={len(prompts)}, "
                             f"classes={np.unique(y).tolist()}")
        return prompts, y, {
            "status": "PRESENT",
            "dataset": "XSTest (Paul/XSTest CSV; Rottger et al. 2024, "
                       "arXiv:2308.01263)",
            "loader": "hello_world.eval_ood.load_xstest_balanced",
            "n": int(len(prompts)),
            "per_class_requested": int(per_class),
            "class_balance": np.bincount(y).tolist(),
            "role": "REPORTED-ONLY - never used to select a cell (that would be "
                    "OOD test-set peeking); scored zero-shot per CV fold",
        }
    except Exception as exc:  # pragma: no cover - depends on local cache state
        print(f"[extract] BENIGN ARM ABSENT: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return [], np.zeros(0, dtype=np.int64), {
            "status": "ABSENT",
            "reason": f"{type(exc).__name__}: {exc}",
            "consequence": "every per-cell AUC is ATTACK-ONLY and cannot detect the "
                           "SALO Appendix-D failure mode (high attack recall with "
                           "near-random XSTest AUROC)",
        }


# --------------------------------------------------------------------------- #
# the capture
# --------------------------------------------------------------------------- #
def pool(h: torch.Tensor, mode: str) -> torch.Tensor:
    """Collapse ``h`` [seq, hidden] to one vector under ``mode``."""
    if mode == "mean":
        return h.mean(0)
    if mode == "last":
        return h[-1]
    if mode == "max":
        return h.max(0).values
    raise ValueError(f"unknown pooling {mode!r}")


@torch.no_grad()
def extract_all_layers(model, tok, prompts, layers, poolings, H, start_at,
                       ckpt_every, ckpt_fn, log_every: int = 25) -> None:
    """Fill ``H[i, li, pi, :]`` for i >= ``start_at``, one forward pass per prompt.

    Hooks EVERY requested decoder block at once, so the cost is one forward pass
    per prompt regardless of how many layers are swept. Prompts go through the
    chat template, matching lesson 1's ``extract_features`` exactly so a
    (layer 12, mean) cell here is comparable to lesson 1's cached feature.
    """
    device = next(model.parameters()).device
    blocks = residual_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[li] = h.detach()
        return hook

    handles = [blocks[layer].register_forward_hook(make_hook(li))
               for li, layer in enumerate(layers)]
    try:
        for i in range(start_at, len(prompts)):
            captured.clear()
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompts[i]}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            model(ids)
            for li in range(len(layers)):
                h = captured[li][0]                       # [seq, hidden]
                for pi, mode in enumerate(poolings):
                    H[i, li, pi, :] = pool(h, mode).float().cpu().numpy()
            done = i + 1
            if log_every and done % log_every == 0:
                print(f"[extract] {done}/{len(prompts)} prompts", file=sys.stderr)
            if ckpt_every and done % ckpt_every == 0:
                ckpt_fn(done)
    finally:
        for handle in handles:
            handle.remove()


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Cache every-layer x every-pooling "
                                             "activations for the layer sweep.")
    ap.add_argument("--dataset", default=ENV_DATASET, choices=("common", "jbb"),
                    help="common = >=500/class length-matched set (default); "
                         "jbb = lesson-1's 100+100 cross-check set")
    ap.add_argument("--n", type=int, default=ENV_N, help="prompts PER CLASS (PT_N)")
    ap.add_argument("--layers", default=ENV_LAYERS,
                    help="all | everyK | lo-hi | comma list (PT_LAYERS)")
    ap.add_argument("--poolings", default=ENV_POOLINGS,
                    help="comma list of mean,last,max (PT_POOLINGS)")
    ap.add_argument("--model-id", default=ENV_MODEL_ID, help="PT_MODEL_ID")
    ap.add_argument("--seed", type=int, default=ENV_SEED)
    ap.add_argument("--ckpt-every", type=int, default=ENV_CKPT_EVERY,
                    help="checkpoint the partial cache every N prompts (0 = never)")
    ap.add_argument("--max-prompts", type=int, default=0,
                    help="hard cap on TOTAL prompts (smoke runs only; 0 = no cap)")
    ap.add_argument("--benign-n", type=int, default=ENV_BENIGN_N,
                    help="XSTest prompts PER CLASS for the benign arm (PT_BENIGN_N)")
    ap.add_argument("--no-benign-arm", action="store_true",
                    help="skip the XSTest over-refusal arm. NOT RECOMMENDED: without "
                         "it every per-cell AUC is attack-only and cannot detect the "
                         "SALO Appendix-D failure mode")
    ap.add_argument("--force", action="store_true",
                    help="ignore an existing checkpoint and start over")
    args = ap.parse_args()

    poolings = parse_poolings(args.poolings)

    prompts, y, prov = load_prompts(args.dataset, args.n, args.seed)
    if args.max_prompts and args.max_prompts < len(prompts):
        # Keep it balanced when smoke-capping: take the head of each class.
        per = args.max_prompts // 2
        keep = np.concatenate([np.where(y == 1)[0][:per], np.where(y == 0)[0][:per]])
        keep.sort()
        prompts = [prompts[i] for i in keep]
        y = y[keep]
        prov["smoke_cap"] = int(args.max_prompts)
        print(f"[extract] SMOKE CAP: {len(prompts)} prompts total - screening only",
              file=sys.stderr)

    # --- the benign / over-refusal arm, appended as group 1 ------------------
    group = np.full(len(prompts), GROUP_MAIN, dtype=np.int64)
    if args.no_benign_arm:
        benign_prov = {"status": "ABSENT", "reason": "disabled via --no-benign-arm",
                       "consequence": "every per-cell AUC is ATTACK-ONLY and cannot "
                                      "detect the SALO Appendix-D failure mode"}
    else:
        b_prompts, b_y, benign_prov = load_benign_arm(args.benign_n, args.seed)
        if b_prompts:
            prompts = prompts + b_prompts
            y = np.concatenate([y, b_y])
            group = np.concatenate(
                [group, np.full(len(b_prompts), GROUP_XSTEST, dtype=np.int64)])
    if benign_prov["status"] == "ABSENT":
        print("[extract] WARNING: no benign/over-refusal arm - the sweep built on "
              "this cache will be ATTACK-ONLY", file=sys.stderr)

    model, tok = load_model(args.model_id)
    n_lay = num_layers(model)
    hidden = hidden_size(model)
    layers = parse_layer_spec(args.layers, n_lay)

    y_main = y[group == GROUP_MAIN]
    meta = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "n_per_class": int(min(np.bincount(y_main))),   # MAIN arm only
        "n_per_class_requested": int(args.n),
        "seed": int(args.seed),
        "layers": layers,
        "poolings": poolings,
        "n_prompts": int(len(prompts)),
        "n_main": int((group == GROUP_MAIN).sum()),
        "n_benign_arm": int((group == GROUP_XSTEST).sum()),
        "hidden": int(hidden),
        "model_n_layers": int(n_lay),
        "deployed_layer": int(C.LAYER),
        "deployed_pooling": str(C.POOLING),
        "chat_template": True,
        "dtype": "float16",
        "provenance": prov,
        "benign_arm": benign_prov,
    }
    meta["fingerprint"] = fingerprint(meta)

    tag = run_tag(args.dataset, meta["n_per_class"], args.model_id)
    out_path = ARTIFACTS / f"layer_features_{tag}.npz"
    ckpt_path = ARTIFACTS / f"layer_features_{tag}.partial.npz"

    shape = (len(prompts), len(layers), len(poolings), hidden)
    mb = np.prod(shape) * 2 / 1e6
    print(f"[extract] H shape {shape} = {mb:.0f} MB fp16 -> {out_path.name}",
          file=sys.stderr)

    # --- resume ------------------------------------------------------------
    H = None
    start_at = 0
    if ckpt_path.exists() and not args.force:
        ck = np.load(ckpt_path, allow_pickle=False)
        ck_meta = json.loads(str(ck["meta"]))
        if ck_meta.get("fingerprint") == meta["fingerprint"]:
            H = ck["H"]
            start_at = int(ck["n_done"])
            print(f"[extract] RESUMING from checkpoint at {start_at}/{len(prompts)}",
                  file=sys.stderr)
        else:
            print("[extract] checkpoint fingerprint MISMATCH - ignoring it and "
                  "starting over (a stale partial must never be appended to)",
                  file=sys.stderr)
    if H is None:
        H = np.zeros(shape, dtype=np.float16)

    def checkpoint(n_done: int) -> None:
        np.savez(ckpt_path, H=H, y=y, group=group, n_done=np.int64(n_done),
                 meta=json.dumps(meta))
        print(f"[extract] checkpoint {n_done}/{len(prompts)}", file=sys.stderr)

    if start_at >= len(prompts):
        print("[extract] checkpoint already complete", file=sys.stderr)
    else:
        extract_all_layers(model, tok, prompts, layers, poolings, H,
                           start_at, args.ckpt_every, checkpoint)

    # --- persist BEFORE any summary print ----------------------------------
    np.savez_compressed(
        out_path,
        H=H,
        y=y,
        group=group,
        layers=np.asarray(layers, dtype=np.int64),
        poolings=np.asarray(poolings),
        meta=json.dumps(meta),
    )
    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except OSError:  # pragma: no cover - Windows file lock
            pass

    print(f"\n=== LAYER-FEATURE CACHE WRITTEN ================================")
    print(f"  path        {out_path}")
    print(f"  model       {args.model_id}")
    print(f"  main arm    {args.dataset}  n={meta['n_main']} "
          f"({meta['n_per_class']}/class)  balance="
          f"{np.bincount(y_main).tolist()}")
    if benign_prov["status"] == "PRESENT":
        print(f"  benign arm  XSTest n={meta['n_benign_arm']} "
              f"balance={benign_prov['class_balance']} (reported-only, never "
              f"used to select)")
    else:
        print(f"  benign arm  ABSENT ({benign_prov.get('reason')})")
        print(f"              WARNING: the sweep will be ATTACK-ONLY and cannot")
        print(f"              detect the SALO Appendix-D failure mode.")
    print(f"  layers      {len(layers)} of {n_lay}: {layers}")
    print(f"  poolings    {poolings}  (max = the paper's sparsity-aware default)")
    print(f"  hidden      {hidden}   fingerprint {meta['fingerprint']}")
    print(f"  NEXT        python -m steering_tutorials.probe_tuning.sweep_layers")
    print("================================================================")


if __name__ == "__main__":
    main()
