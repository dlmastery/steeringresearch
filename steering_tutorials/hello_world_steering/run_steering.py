"""run_steering.py — the orchestrator: build a refusal vector, then steer with it.

This is the spine of lesson 2. It wires together the four pieces the peers own
(``model_utils``, ``steer_vector``, ``gate``, ``judge``) into one pipeline and
writes ``artifacts/results.json`` + two plots that the webapp and README read.

The lifecycle, end to end
--------------------------
1. EXTRACT.  Split each class into a disjoint *extract* half (first
   ``N_EXTRACT``) and *eval* half (the rest), then contrast the extract halves
   with diff-of-means to get one "refuse this" direction (the CAA vector).

   WHICH MODEL supplies that contrast is ``config.EXTRACT_FROM`` and it matters
   more than anything else in this lesson. Default ``"base"``: we load the
   ALIGNED Gemma-3-1B (``config.BASE_MODEL``), read the direction there, free
   it, and only then load the abliterated model we actually steer. Reading the
   contrast from the abliterated model itself -- the lesson's original,
   BUGGY recipe -- yields a topic direction, not a refusal direction, because an
   abliterated model refuses neither class (see ``config.EXTRACT_FROM``). That
   arm survives as the explicitly labelled ``"abliterated"`` ablation.

2. UNCONDITIONAL arm.  On the held-out harmful prompts, steer at every alpha in
   ``ALPHAS`` and have the judge label each output REFUSAL / COMPLIANCE /
   GIBBERISH. This traces the dose-response curve: how hard can we push before
   coherent refusals collapse into gibberish? Under ``EXTRACT_FROM="both"`` the
   sweep is run once per extraction source, on the SAME prompts at the SAME
   alphas, so the bug and the fix sit on one axis.

3. CONDITIONAL arm.  On a MIXED eval set (held-out harmful + held-out benign),
   ask the lesson-1 probe (the gate) whether each prompt is harmful. Steer only
   when it fires, at a single alpha chosen from the arm-2 curve. Judge both the
   unsteered baseline and the gated output. This is the whole point: refuse the
   harmful ones, leave the benign ones untouched.

4. REPORT.  Save the schema below, render two PNGs, print a summary table.

Everything that touches the model lives under ``main()`` so ``import
run_steering`` is a no-op (safe to import for tests / the webapp).

RESULTS SCHEMA (kept in sync with app.py + README)
--------------------------------------------------
{
  "model_id": str, "steer_layer": int, "alphas": [float, ...],
  "extract_from": "base" | "abliterated" | "both",
  "primary_source": "base" | "abliterated",   # whose numbers are the headline
  "base_model": str,
  "steering_vector": {"norm": float, "layer": int, "n_extract": int,
                      "source": str, "source_model": str},
  "extraction": {                              # one entry per source that ran
      "<source>": {"model": str, "norm": float, "layer": int, "n_extract": int,
                   "unconditional": [ ...same row schema as below... ]}, ...
  },
  "unconditional": [                           # == extraction[primary].unconditional
      {"alpha": float, "refusal_rate": float, "compliance_rate": float,
       "gibberish_rate": float, "n": int}, ...
  ],
  "conditional": {
      "alpha": float, "harmful_refusal_rate": float,
      "benign_over_refusal_rate": float, "gibberish_rate": float,
      "gate_accuracy": float, "n_harmful": int, "n_benign": int
  },
  "examples": [
      {"prompt": str, "harmful": bool, "gated": bool,
       "baseline_response": str, "steered_response": str,
       "baseline_verdict": str, "steered_verdict": str}, ...
  ],
  "plots": {"rates_vs_alpha": "rates_vs_alpha.png",
            "conditional": "conditional.png",
            "extraction_source_contrast": "extraction_source_contrast.png"}
}
"""
from __future__ import annotations

import json
import sys

from . import config as C


# --------------------------------------------------------------------------- #
# Small pure helpers (no model needed) — safe to unit-test in isolation.
# --------------------------------------------------------------------------- #
def _rates(verdicts: list[str]) -> dict[str, float]:
    """Fraction of REFUSAL / COMPLIANCE / GIBBERISH among a list of verdicts."""
    n = max(1, len(verdicts))
    return {
        "refusal_rate": verdicts.count("REFUSAL") / n,
        "compliance_rate": verdicts.count("COMPLIANCE") / n,
        "gibberish_rate": verdicts.count("GIBBERISH") / n,
    }


def choose_conditional_alpha(unconditional: list[dict],
                             gibberish_tolerance: float) -> float:
    """Pick the alpha for the conditional arm from the dose-response curve.

    Rule: among the STEERING alphas (alpha > 0) whose gibberish rate stays at or
    below ``gibberish_tolerance``, take the highest refusal rate; break ties
    toward the SMALLEST alpha (least collateral damage to coherence). If every
    alpha is too gibberishy, fall back to the smallest steering alpha. This is
    the "highest refusal before gibberish rises, smallest that gets us there"
    heuristic documented in the task spec.
    """
    steering = [r for r in unconditional if r["alpha"] > 0.0]
    if not steering:
        return 0.0
    clean = [r for r in steering if r["gibberish_rate"] <= gibberish_tolerance]
    pool = clean if clean else steering
    # max refusal_rate, then min alpha as the tie-breaker.
    best = max(pool, key=lambda r: (r["refusal_rate"], -r["alpha"]))
    return float(best["alpha"])


def _fingerprint(payload: dict) -> str:
    """Short stable hash of a config dict — the checkpoint's identity stamp.

    An artifact that cannot be traced back to the config that produced it is not
    evidence, so every cached row carries this fingerprint and a mismatch throws
    the cache away instead of silently reusing it.
    """
    import hashlib

    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _load_checkpoint(path, fp: str) -> dict:
    """Return the resumable state for fingerprint ``fp`` (empty dict if stale)."""
    fresh = {"fp": fp, "vectors": {}, "sweeps": {}, "conditional_records": []}
    if not path.exists():
        return fresh
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fresh
    if state.get("fp") != fp:
        print(f"[ckpt] stale checkpoint (fp {state.get('fp')} != {fp}) — "
              "starting fresh", file=sys.stderr)
        return fresh
    n_sweep = len(state.get("sweeps", {}))
    n_cond = len(state.get("conditional_records", []))
    print(f"[ckpt] resuming: {n_sweep} sweep cells, {n_cond} conditional records",
          file=sys.stderr)
    state.setdefault("vectors", {})
    return state


def _save_checkpoint(path, state: dict) -> None:
    path.write_text(json.dumps(state), encoding="utf-8")


def _summary_table(results: dict) -> str:
    """A plain-text recap printed at the end of a run."""
    sv = results["steering_vector"]
    lines = ["", "=" * 64, "STEERING SUMMARY", "=" * 64,
             f"steered model : {results['model_id']}",
             f"direction from: {sv['source_model']}  [source={sv['source']}]",
             f"steer layer   : {results['steer_layer']}   "
             f"vector norm: {sv['norm']:.3f}"]
    # One block per extraction source that ran -- the bug and the fix, side by side.
    for src, block in results["extraction"].items():
        tag = " (PRIMARY)" if src == results["primary_source"] else " (ABLATION)"
        lines += ["", f"Unconditional, direction extracted from {src.upper()}{tag}",
                  f"  [{block['model']}]",
                  f"  {'alpha':>6} {'refusal':>9} {'comply':>9} {'gibber':>9}"]
        for r in block["unconditional"]:
            lines.append(f"  {r['alpha']:>6.2f} {r['refusal_rate']:>9.2f} "
                         f"{r['compliance_rate']:>9.2f} {r['gibberish_rate']:>9.2f}")
    c = results["conditional"]
    lines += ["", f"Conditional (gate + steer @ alpha={c['alpha']:.2f}):",
              f"  harmful refusal rate  : {c['harmful_refusal_rate']:.2f} "
              f"(n={c['n_harmful']})",
              f"  benign over-refusal   : {c['benign_over_refusal_rate']:.2f} "
              f"(n={c['n_benign']})",
              f"  gibberish rate        : {c['gibberish_rate']:.2f}",
              f"  gate accuracy         : {c['gate_accuracy']:.2f}", "=" * 64, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting — matplotlib with the Agg backend (headless, no display needed).
# --------------------------------------------------------------------------- #
def _plot_rates_vs_alpha(unconditional: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [r["alpha"] for r in unconditional]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, [r["refusal_rate"] for r in unconditional],
            "o-", label="refusal", color="#2a7")
    ax.plot(alphas, [r["compliance_rate"] for r in unconditional],
            "s-", label="compliance", color="#37a")
    ax.plot(alphas, [r["gibberish_rate"] for r in unconditional],
            "^-", label="gibberish", color="#c33")
    ax.set_xlabel("steering strength  α  (fraction of residual norm)")
    ax.set_ylabel("rate on held-out harmful prompts")
    ax.set_title("Dose-response: what steering does as α grows")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_extraction_contrast(extraction: dict, path) -> None:
    """Refusal / gibberish vs alpha for EVERY extraction source, on one axis.

    This is the bug-and-fix plot: the same prompts, the same alphas, the same
    steered model -- only the model the direction was READ from changes. If the
    base-extracted curve rises where the abliterated-extracted one falls, the old
    "steering does not install refusal" verdict was an extraction artifact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"base": ("#2a7", "o-", "extracted from ALIGNED base"),
              "abliterated": ("#c33", "s--", "extracted from ABLITERATED (old bug)")}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for src, block in extraction.items():
        color, ls, label = styles.get(src, ("#37a", "^-", src))
        rows = block["unconditional"]
        alphas = [r["alpha"] for r in rows]
        ax.plot(alphas, [r["refusal_rate"] for r in rows], ls, color=color,
                label=f"refusal, {label}")
        ax.plot(alphas, [r["gibberish_rate"] for r in rows], ls, color=color,
                alpha=0.35, label=f"gibberish, {label}")
    ax.set_xlabel("steering strength  alpha  (fraction of residual norm)")
    ax.set_ylabel("rate on held-out harmful prompts")
    ax.set_title("Same steered model, same prompts --\nonly the EXTRACTION SOURCE differs")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_conditional(conditional: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["harmful\nrefusal", "benign\nover-refusal", "gibberish"]
    vals = [conditional["harmful_refusal_rate"],
            conditional["benign_over_refusal_rate"],
            conditional["gibberish_rate"]]
    colors = ["#2a7", "#c93", "#c33"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Conditional steering @ α={conditional['alpha']:.2f}\n"
                 "(want: high harmful-refusal, low over-refusal + gibberish)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The pipeline — everything below here loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import gc
    import os
    import random

    import numpy as np
    import torch

    # Peer-owned modules. Imported inside main() so a bare ``import
    # run_steering`` never drags in torch or triggers a model load.
    from .model_utils import load_model, generate, num_layers, hidden_size
    from .steer_vector import extract_caa_vector, save_vector, load_vector
    from .gate import HarmGate
    from .judge import Judge
    # The shared >=500/class harmful/benign set (toxic-chat + JBB top-up,
    # deduped + length-matched) replaces the lesson's old 100-prompt JBB loader.
    from steering_tutorials.common.data import load_harmful_benign

    # Reproducibility: pin every RNG before anything stochastic happens.
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    sources = (["base", "abliterated"] if C.EXTRACT_FROM == "both"
               else [C.EXTRACT_FROM])
    print(f"[config] steer={C.MODEL_ID}  extract_from={C.EXTRACT_FROM} "
          f"(sources={sources}, primary={C.PRIMARY_SOURCE})", file=sys.stderr)

    # --- Resume state ---------------------------------------------------------
    # Fingerprint EVERY knob that could change a number. Same fingerprint => the
    # cached rows are for this exact run and may be reused; different => discard.
    fp = _fingerprint({
        "model": C.MODEL_ID, "base_model": C.BASE_MODEL,
        "extract_from": C.EXTRACT_FROM, "layer": C.STEER_LAYER,
        "alphas": C.ALPHAS, "n_per_class": C.N_PER_CLASS,
        "n_extract": C.N_EXTRACT, "n_eval": C.N_EVAL,
        "max_new_tokens": C.MAX_NEW_TOKENS, "seed": C.SEED,
        "judge": os.environ.get("STEER_JUDGE_MODEL", "self"),
        "load_4bit": C.LOAD_4BIT,
    })
    state = _load_checkpoint(C.CHECKPOINT_PATH, fp)

    # --- Data: split each class into disjoint extract / eval halves -----------
    # Done BEFORE any model load so both extraction passes see the identical
    # split (same loader, same seed) without having to pass data between them.
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    extract_harmful = data["harmful"][:C.N_EXTRACT]
    extract_benign = data["benign"][:C.N_EXTRACT]
    eval_harmful = data["harmful"][C.N_EXTRACT:]
    eval_benign = data["benign"][C.N_EXTRACT:]
    if C.N_EVAL:  # optional smoke-test cap (STEER_N_EVAL)
        eval_harmful = eval_harmful[:C.N_EVAL]
        eval_benign = eval_benign[:C.N_EVAL]
    print(f"[split] extract: {len(extract_harmful)}h/{len(extract_benign)}b   "
          f"eval: {len(eval_harmful)}h/{len(eval_benign)}b", file=sys.stderr)

    # --- 1a. Read the direction from the ALIGNED BASE, then FREE that model ---
    # The base model is only ever READ from -- it is never steered and never
    # generates. We load it, take one diff-of-means, drop it, and only then load
    # the abliterated model, so the two never sit in VRAM/RAM at once (the same
    # reason the sibling ``realignment`` lesson splits into two processes; here a
    # single process suffices because the base model is released first).
    def _vector_fp(source_model: str) -> str:
        return _fingerprint({"m": source_model, "layer": C.STEER_LAYER,
                             "n_extract": C.N_EXTRACT,
                             "n_per_class": C.N_PER_CLASS, "seed": C.SEED})

    def _cached_vector(src: str, source_model: str):
        """Reuse a saved direction ONLY if its stamped fingerprint matches."""
        vfp = _vector_fp(source_model)
        path = C.VECTOR_PATHS[src]
        if C.FORCE_EXTRACT or not path.exists():
            return vfp, None
        try:
            v = load_vector(path)
        except Exception:
            return vfp, None
        if v.get("vec_fp") != vfp:
            return vfp, None
        print(f"[vector:{src}] reusing cached {path.name} (fp {vfp})",
              file=sys.stderr)
        return vfp, v

    vectors: dict[str, dict] = {}
    base_hidden = base_layer = None
    if "base" in sources:
        vfp_b, cached_b = _cached_vector("base", C.BASE_MODEL)
        if cached_b is not None:
            vec_b = cached_b
            base_layer, base_hidden = int(vec_b["layer"]), int(vec_b["hidden"])
        else:
            base_model, base_tok = load_model(C.BASE_MODEL)
            base_layer = min(C.STEER_LAYER, num_layers(base_model) - 1)
            base_hidden = hidden_size(base_model)
            vec_b = extract_caa_vector(base_model, base_tok, extract_harmful,
                                       extract_benign, base_layer)
            vec_b.update({"source": "base", "source_model": C.BASE_MODEL,
                          "hidden": int(base_hidden), "vec_fp": vfp_b})
            save_vector(C.VECTOR_PATHS["base"], vec_b)
            print(f"[vector:base] layer={vec_b['layer']} n={vec_b['n']} "
                  f"norm={vec_b['norm']:.3f} -> {C.VECTOR_PATHS['base']}",
                  file=sys.stderr)
            del base_model, base_tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        vectors["base"] = vec_b

    # --- 1b. Load the model we STEER (also the gate's and judge's host) -------
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.STEER_LAYER, num_layers(model) - 1)

    # The transplant is only well-typed if the two models share a residual width
    # and the layer index exists in both. Assert it -- a silent shape mismatch
    # here would steer with a truncated/garbage direction.
    if "base" in sources:
        if base_hidden != hidden_size(model):
            raise SystemExit(
                f"hidden size mismatch: base {base_hidden} vs steered "
                f"{hidden_size(model)} -- the direction cannot be transplanted.")
        if base_layer != layer:
            raise SystemExit(
                f"layer mismatch: read at {base_layer} in the base model but "
                f"would write at {layer} in the steered model.")

    # --- 1c. The ABLATION: the old, buggy same-model extraction ---------------
    if "abliterated" in sources:
        vfp_a, cached_a = _cached_vector("abliterated", C.MODEL_ID)
        if cached_a is not None:
            vec_a = cached_a
        else:
            vec_a = extract_caa_vector(model, tok, extract_harmful,
                                       extract_benign, layer)
            vec_a.update({"source": "abliterated", "source_model": C.MODEL_ID,
                          "hidden": int(hidden_size(model)), "vec_fp": vfp_a})
            save_vector(C.VECTOR_PATHS["abliterated"], vec_a)
            print(f"[vector:abliterated] layer={vec_a['layer']} n={vec_a['n']} "
                  f"norm={vec_a['norm']:.3f} -> {C.VECTOR_PATHS['abliterated']}",
                  file=sys.stderr)
        vectors["abliterated"] = vec_a

    # The primary vector is also written to the legacy path so ``infer.py`` and
    # the webapp keep working unchanged.
    primary_vec = vectors[C.PRIMARY_SOURCE]
    save_vector(C.VECTOR_PATH, primary_vec)
    v_unit = primary_vec["v_unit"]  # unit direction; alpha supplies magnitude

    # How far apart are the two directions? cos ~ 0 means the old vector was
    # pointing somewhere else entirely -- the quantitative form of the bug.
    cos_sources = None
    if len(vectors) == 2:
        a = vectors["base"]["v_unit"]
        b = vectors["abliterated"]["v_unit"]
        cos_sources = float(np.dot(a, b))
        print(f"[vector] cos(base_dir, abliterated_dir) = {cos_sources:+.4f}",
              file=sys.stderr)

    judge = Judge(model, tok)

    # --- 2. UNCONDITIONAL arm: sweep alpha on held-out harmful prompts --------
    # Run once per extraction source. Same prompts, same alphas, same steered
    # model -- only the direction differs.
    extraction: dict[str, dict] = {}
    for src in sources:
        vec_s = vectors[src]
        vu = vec_s["v_unit"]
        rows: list[dict] = []
        for alpha in C.ALPHAS:
            cell = f"{src}|{alpha:g}"
            if cell in state["sweeps"]:      # already measured in an earlier window
                rec = state["sweeps"][cell]
                rows.append(rec)
                print(f"[uncond {src} a={alpha:.2f}] (cached) "
                      f"refusal={rec['refusal_rate']:.2f}", file=sys.stderr)
                continue
            verdicts: list[str] = []
            for i, prompt in enumerate(eval_harmful):
                # alpha == 0 is the true baseline: no vector, no injection.
                resp = generate(
                    model, tok, prompt,
                    max_new_tokens=C.MAX_NEW_TOKENS,
                    vector=(None if alpha == 0.0 else vu),
                    layer=layer, alpha=alpha, operation="relative_add",
                )
                verdicts.append(judge.verdict(prompt, resp))
                if (i + 1) % 25 == 0:
                    print(f"[uncond {src} a={alpha:.2f}] "
                          f"{i + 1}/{len(eval_harmful)}", file=sys.stderr)
            rec = {"alpha": float(alpha), "n": len(eval_harmful),
                   **_rates(verdicts)}
            rows.append(rec)
            # Checkpoint the completed cell before starting the next one: a reap
            # now costs one alpha, not the whole sweep.
            state["sweeps"][cell] = rec
            _save_checkpoint(C.CHECKPOINT_PATH, state)
            print(f"[uncond {src} a={alpha:.2f}] "
                  f"refusal={rec['refusal_rate']:.2f} "
                  f"comply={rec['compliance_rate']:.2f} "
                  f"gibber={rec['gibberish_rate']:.2f}", file=sys.stderr)
        extraction[src] = {
            "model": vec_s["source_model"],
            "norm": float(vec_s["norm"]),
            "layer": int(vec_s["layer"]),
            "n_extract": int(vec_s["n"]),
            "unconditional": rows,
        }

    unconditional = extraction[C.PRIMARY_SOURCE]["unconditional"]

    # Pick the single alpha for the conditional arm from the PRIMARY curve.
    steer_alpha = choose_conditional_alpha(unconditional, C.GIBBERISH_TOLERANCE)
    print(f"[choose] conditional alpha = {steer_alpha:.2f}", file=sys.stderr)

    # A resumed conditional arm must have been generated at the SAME alpha, or
    # the two halves of the record list are not comparable. Drop them if not.
    if state.get("cond_alpha") is not None and \
            abs(float(state["cond_alpha"]) - steer_alpha) > 1e-9:
        print(f"[ckpt] conditional alpha changed "
              f"({state['cond_alpha']} -> {steer_alpha}); discarding "
              f"{len(state.get('conditional_records', []))} stale records",
              file=sys.stderr)
        state["conditional_records"] = []
    state["cond_alpha"] = float(steer_alpha)
    _save_checkpoint(C.CHECKPOINT_PATH, state)

    # --- 3. CONDITIONAL arm: gate-then-steer on the MIXED eval set ------------
    gate = HarmGate(model, tok)
    mixed = ([(p, True) for p in eval_harmful]
             + [(p, False) for p in eval_benign])

    # Resume: replay the records this fingerprint already produced, then continue
    # from exactly where the previous window stopped.
    records: list[dict] = list(state.get("conditional_records", []))
    if records:
        print(f"[cond] resuming after {len(records)}/{len(mixed)} records",
              file=sys.stderr)
    for i, (prompt, is_harmful_true) in enumerate(mixed):
        if i < len(records):
            continue
        fired, prob = gate.is_harmful(prompt)
        used_alpha = steer_alpha if fired else 0.0

        baseline_resp = generate(
            model, tok, prompt, max_new_tokens=C.MAX_NEW_TOKENS,
            vector=None, layer=layer, alpha=0.0, operation="relative_add",
        )
        gated_resp = generate(
            model, tok, prompt, max_new_tokens=C.MAX_NEW_TOKENS,
            vector=(v_unit if fired else None),
            layer=layer, alpha=used_alpha, operation="relative_add",
        )
        records.append({
            "prompt": prompt,
            "harmful": bool(is_harmful_true),
            "gated": bool(fired),
            "gate_prob": float(prob),
            "baseline_response": baseline_resp,
            "steered_response": gated_resp,
            "baseline_verdict": judge.verdict(prompt, baseline_resp),
            "steered_verdict": judge.verdict(prompt, gated_resp),
        })
        if (i + 1) % 10 == 0:
            state["conditional_records"] = records
            _save_checkpoint(C.CHECKPOINT_PATH, state)
            print(f"[cond] {i + 1}/{len(mixed)}", file=sys.stderr)
    state["conditional_records"] = records
    _save_checkpoint(C.CHECKPOINT_PATH, state)

    harmful_recs = [r for r in records if r["harmful"]]
    benign_recs = [r for r in records if not r["harmful"]]
    harmful_refusal_rate = (
        sum(r["steered_verdict"] == "REFUSAL" for r in harmful_recs)
        / max(1, len(harmful_recs)))
    benign_over_refusal_rate = (
        sum(r["steered_verdict"] == "REFUSAL" for r in benign_recs)
        / max(1, len(benign_recs)))
    cond_gibberish_rate = (
        sum(r["steered_verdict"] == "GIBBERISH" for r in records)
        / max(1, len(records)))
    gate_accuracy = (
        sum(r["gated"] == r["harmful"] for r in records) / max(1, len(records)))

    conditional = {
        "alpha": float(steer_alpha),
        "harmful_refusal_rate": harmful_refusal_rate,
        "benign_over_refusal_rate": benign_over_refusal_rate,
        "gibberish_rate": cond_gibberish_rate,
        "gate_accuracy": gate_accuracy,
        "n_harmful": len(harmful_recs),
        "n_benign": len(benign_recs),
    }

    # --- Pick 8–12 side-by-side examples: mostly gated-harmful (the money
    # shot: baseline complies, steered refuses), plus a couple of benign ones
    # to show the gate leaving harmless prompts alone. ------------------------
    example_keys = ("prompt", "harmful", "gated", "baseline_response",
                    "steered_response", "baseline_verdict", "steered_verdict")
    gated_harmful = [r for r in harmful_recs if r["gated"]]
    other = [r for r in records if r not in gated_harmful]
    chosen = gated_harmful[:8] + benign_recs[:2]
    if len(chosen) < 8:  # backfill if the gate fired on few prompts
        chosen += [r for r in other if r not in chosen][:8 - len(chosen)]
    examples = [{k: r[k] for k in example_keys} for r in chosen[:12]]

    results = {
        "model_id": C.MODEL_ID,
        "steer_layer": int(layer),
        "alphas": [float(a) for a in C.ALPHAS],
        "extract_from": C.EXTRACT_FROM,
        "primary_source": C.PRIMARY_SOURCE,
        "base_model": C.BASE_MODEL,
        "cos_base_vs_abliterated": cos_sources,
        "steering_vector": {
            "norm": float(primary_vec["norm"]),
            "layer": int(primary_vec["layer"]),
            "n_extract": int(primary_vec["n"]),
            "source": primary_vec["source"],
            "source_model": primary_vec["source_model"],
        },
        "extraction": extraction,
        "unconditional": unconditional,
        "conditional": conditional,
        "examples": examples,
        "plots": {"rates_vs_alpha": C.RATES_PNG.name,
                  "conditional": C.CONDITIONAL_PNG.name,
                  "extraction_source_contrast": C.EXTRACTION_PNG.name},
    }

    # --- 4. Persist + plot + print -------------------------------------------
    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_rates_vs_alpha(unconditional, C.RATES_PNG)
    _plot_conditional(conditional, C.CONDITIONAL_PNG)
    _plot_extraction_contrast(extraction, C.EXTRACTION_PNG)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.RATES_PNG}", file=sys.stderr)
    print(f"[save] {C.CONDITIONAL_PNG}", file=sys.stderr)
    print(f"[save] {C.EXTRACTION_PNG}", file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
