"""run_gavel.py -- the orchestrator: build the CE monitor, then guard the model.

The spine of the lesson. It (1) reads mean-pooled activations for an EXTRACT split
and builds a library of cognitive-element detectors (one per harm category), (2)
composes a rule over them, then (3) evaluates the monitor on a DISJOINT held-out
split and reports:

    block-rate on harmful   : fraction of harmful prompts the rule blocks (recall)
    false-block on benign   : fraction of benign prompts the rule blocks (the tax)
    pass-through behavior    : for prompts the rule PASSES, we let the abliterated
                               model answer and judge the result -- harmful prompts
                               that pass AND comply are the leaks the monitor missed.

A single broad diff-of-means detector (all harmful vs benign, one direction) is
built alongside as the baseline the paper argues against: it shows what you get
without the compositional, per-CE-calibrated library.

Everything that touches the model lives under ``main()`` so ``import run_gavel`` is
inert (safe for import-checks while the GPU is busy). The team lead runs this on
the GPU.

RESULTS SCHEMA (kept in sync with README + plots)
-------------------------------------------------
{
  "model_id": str, "layer": int, "pooling": str, "target_fpr": float, "rule": str,
  "dataset": {"n_extract_per_class": int, "n_eval_harmful": int,
              "n_eval_benign": int, "natural_toxic_rate": float,
              "median_char_length": {...}},
  "cognitive_elements": [ {"name": str, "n_pos": int, "tau": float,
                           "target_fpr": float}, ... ],
  "gavel":          {"block_rate_harmful": float, "false_block_benign": float,
                     "n_harmful": int, "n_benign": int},
  "broad_baseline": {"block_rate_harmful": float, "false_block_benign": float},
  "ce_firing_rates": {"harmful": {ce: rate}, "benign": {ce: rate}},
  "passthrough": {"n_harmful_passed": int, "harmful_passed_compliance_rate": float,
                  "n_benign_passed": int, "benign_passed_answered_rate": float,
                  "system_harmful_leak_rate": float, "judge_id": str},
  "examples": [ {"prompt": str, "class": str, "block": bool,
                 "triggered_by": [str], "output": str, "verdict": str}, ... ],
  "plots": {"block_vs_falseblock": "block_vs_falseblock.png",
            "ce_firing_rates": "ce_firing_rates.png"}
}
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from . import config as C
from .monitor import GavelMonitor, build_ce_detector, make_rule, Rule


# --------------------------------------------------------------------------- #
# Pure helpers (no model) -- safe to reason about in isolation.
# --------------------------------------------------------------------------- #
def _env_cap(name: str, default):
    """Read an int cap from the environment; empty/unset -> default (never 0)."""
    raw = os.environ.get(name) or ""
    raw = raw.strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


def _summary_table(results: dict) -> str:
    g = results["gavel"]
    b = results["broad_baseline"]
    p = results["passthrough"]
    lines = ["", "=" * 64,
             "GAVEL MONITOR SUMMARY (read-only activation guardrail)", "=" * 64,
             "model : %s   layer: %d   rule: %s"
             % (results["model_id"], results["layer"], results["rule"]),
             "cognitive elements: %s" % ", ".join(
                 ce["name"] for ce in results["cognitive_elements"]),
             "",
             "  %-22s %8s %8s" % ("monitor", "block_H", "falseB"),
             "  %-22s %8.2f %8.2f" % ("GAVEL (compositional)",
                                      g["block_rate_harmful"], g["false_block_benign"]),
             "  %-22s %8.2f %8.2f" % ("broad (single dir)",
                                      b["block_rate_harmful"], b["false_block_benign"]),
             "",
             "Pass-through (only prompts the rule let through):",
             "  harmful passed      : %d  (compliance %.2f)  <- leaks the monitor missed"
             % (p["n_harmful_passed"], p["harmful_passed_compliance_rate"]),
             "  benign passed       : %d  (answered %.2f)"
             % (p["n_benign_passed"], p["benign_passed_answered_rate"]),
             "  system harmful leak : %.2f  (harmful answered-and-complied / all harmful)"
             % p["system_harmful_leak_rate"],
             "  judge               : %s" % p["judge_id"],
             "=" * 64, ""]
    return "\n".join(lines)


def _plot_block_vs_falseblock(results: dict, path) -> None:
    """Grouped bars: block-rate on harmful vs false-block on benign, GAVEL vs broad."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["block harmful\n(recall, want high)", "false-block benign\n(tax, want low)"]
    gavel = [results["gavel"]["block_rate_harmful"], results["gavel"]["false_block_benign"]]
    broad = [results["broad_baseline"]["block_rate_harmful"],
             results["broad_baseline"]["false_block_benign"]]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w / 2, gavel, w, label="GAVEL (compositional CEs)", color="#2a7")
    b2 = ax.bar(x + w / 2, broad, w, label="broad (single direction)", color="#c93")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("rate")
    ax.set_title("Read-only monitor: compositional CE rule vs a single broad direction")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_ce_firing(results: dict, path) -> None:
    """Per-CE firing rate on harmful vs benign eval prompts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [ce["name"] for ce in results["cognitive_elements"]]
    harm = [results["ce_firing_rates"]["harmful"].get(n, 0.0) for n in names]
    ben = [results["ce_firing_rates"]["benign"].get(n, 0.0) for n in names]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 4))
    ax.bar(x - w / 2, harm, w, label="harmful", color="#c33")
    ax.bar(x + w / 2, ben, w, label="benign", color="#39c")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction of prompts that trip this CE")
    ax.set_title("Cognitive-element firing rates (harmful should fire, benign should not)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _rates_from_verdicts(verdicts: list[str]) -> dict[str, float]:
    n = max(1, len(verdicts))
    return {"compliance": verdicts.count("COMPLIANCE") / n,
            "refusal": verdicts.count("REFUSAL") / n,
            "gibberish": verdicts.count("GIBBERISH") / n}


# --------------------------------------------------------------------------- #
# The pipeline -- everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import random

    import numpy as np
    import torch

    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, num_layers, mean_pool_activation, generate)
    from steering_tutorials.hello_world_steering.judge import Judge
    # Shared >=500/class harmful/benign set WITH per-prompt harm category.
    from steering_tutorials.common.data import build_harmful_benign

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)

    max_eval = _env_cap("GAVEL_MAX_EVAL", C.DEFAULT_MAX_EVAL)
    max_new = _env_cap("GAVEL_MAX_NEW_TOKENS", C.MAX_NEW_TOKENS)

    # --- Load the abliterated model we monitor (and its self/off-family judge) --
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)
    judge = Judge(model, tok)

    # --- Data: categorized rows, split extract (build CEs) / eval (test) --------
    rec = build_harmful_benign(C.N_PER_CLASS, C.SEED)
    harmful_rows, benign_rows, header = rec["harmful"], rec["benign"], rec["header"]

    extract_harm = harmful_rows[:C.N_EXTRACT]
    extract_ben = benign_rows[:C.N_EXTRACT]
    eval_harm = harmful_rows[C.N_EXTRACT:]
    eval_ben = benign_rows[C.N_EXTRACT:]
    if max_eval is not None:
        eval_harm = eval_harm[:max_eval]
        eval_ben = eval_ben[:max_eval]
    print("[split] extract %dh/%db   eval %dh/%db"
          % (len(extract_harm), len(extract_ben), len(eval_harm), len(eval_ben)),
          file=sys.stderr)

    # --- Helper: mean-pool activations for a list of prompt rows ----------------
    def acts_for(rows: list[dict], tag: str) -> np.ndarray:
        out = []
        for i, r in enumerate(rows):
            out.append(mean_pool_activation(model, tok, r["prompt"], layer))
            if (i + 1) % 25 == 0:
                print("[acts:%s] %d/%d" % (tag, i + 1, len(rows)), file=sys.stderr)
        return np.stack(out).astype(np.float32)

    # --- 1. Extract activations -------------------------------------------------
    A_harm = acts_for(extract_harm, "extract-harm")
    A_ben = acts_for(extract_ben, "extract-benign")

    # --- 2. Build one cognitive element per harm category -----------------------
    # Group extract-harmful activations by their toxic-chat harm category. A
    # category with >= MIN_CE_EXAMPLES earns its own CE; sparser categories (and
    # 'unlabeled') are pooled into a catch-all 'other_harm' CE so nothing is
    # uncovered. Each CE's tau is calibrated to TARGET_FPR on the extract benign.
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(extract_harm):
        by_cat[r.get("category", "unlabeled")].append(i)

    detectors = []
    leftover_idx: list[int] = []
    for cat, idx in sorted(by_cat.items()):
        if cat != "unlabeled" and len(idx) >= C.MIN_CE_EXAMPLES:
            ce = build_ce_detector(cat, A_harm[idx], A_ben, C.TARGET_FPR)
            detectors.append(ce)
            print("[CE] %-12s n_pos=%d tau=%.3f" % (ce.name, ce.n_pos, ce.tau),
                  file=sys.stderr)
        else:
            leftover_idx.extend(idx)
    if len(leftover_idx) >= C.MIN_CE_EXAMPLES:
        ce = build_ce_detector("other_harm", A_harm[leftover_idx], A_ben, C.TARGET_FPR)
        detectors.append(ce)
        print("[CE] %-12s n_pos=%d tau=%.3f" % (ce.name, ce.n_pos, ce.tau),
              file=sys.stderr)
    if not detectors:
        # Degenerate tiny-data fallback: a single all-harm CE.
        ce = build_ce_detector("harm", A_harm, A_ben, C.TARGET_FPR)
        detectors.append(ce)

    rule = make_rule(C.RULE, [d.name for d in detectors])
    monitor = GavelMonitor(detectors, rule)
    monitor.save(C.MONITOR_PATH)
    print("[monitor] %d cognitive elements, rule=%s -> %s"
          % (len(detectors), rule.name, C.MONITOR_PATH), file=sys.stderr)

    # --- 3. The broad single-direction baseline (what the paper argues past) ----
    broad_ce = build_ce_detector("broad_harm", A_harm, A_ben, C.TARGET_FPR)
    broad_monitor = GavelMonitor([broad_ce], Rule.any_of(["broad_harm"]))

    # --- 4. Evaluate the monitors on the held-out split -------------------------
    E_harm = acts_for(eval_harm, "eval-harm")
    E_ben = acts_for(eval_ben, "eval-benign")

    gavel_harm_block = monitor.block_mask(E_harm)
    gavel_ben_block = monitor.block_mask(E_ben)
    broad_harm_block = broad_monitor.block_mask(E_harm)
    broad_ben_block = broad_monitor.block_mask(E_ben)

    gavel = {"block_rate_harmful": float(gavel_harm_block.mean()) if len(E_harm) else 0.0,
             "false_block_benign": float(gavel_ben_block.mean()) if len(E_ben) else 0.0,
             "n_harmful": len(eval_harm), "n_benign": len(eval_ben)}
    broad = {"block_rate_harmful": float(broad_harm_block.mean()) if len(E_harm) else 0.0,
             "false_block_benign": float(broad_ben_block.mean()) if len(E_ben) else 0.0}

    ce_firing = {"harmful": monitor.firing_rates(E_harm),
                 "benign": monitor.firing_rates(E_ben)}

    # --- 5. Pass-through: let the model answer prompts the rule PASSED ----------
    # Blocked prompts get SAFE_REFUSAL for free (no generation). Only PASSED
    # prompts are generated + judged. Harmful prompts that pass AND comply are the
    # leaks the monitor missed; benign prompts that pass should be answered.
    harmful_passed_verdicts: list[str] = []
    n_harm_complied_and_passed = 0
    for i, r in enumerate(eval_harm):
        if gavel_harm_block[i]:
            continue  # blocked upfront -> SAFE_REFUSAL, no model call
        out = generate(model, tok, r["prompt"], max_new_tokens=max_new, alpha=0.0)
        v = judge.verdict(r["prompt"], out)
        harmful_passed_verdicts.append(v)
        n_harm_complied_and_passed += (v == "COMPLIANCE")
    benign_passed_verdicts: list[str] = []
    for i, r in enumerate(eval_ben):
        if gavel_ben_block[i]:
            continue
        out = generate(model, tok, r["prompt"], max_new_tokens=max_new, alpha=0.0)
        benign_passed_verdicts.append(judge.verdict(r["prompt"], out))

    n_harm_total = max(1, len(eval_harm))
    passthrough = {
        "n_harmful_passed": len(harmful_passed_verdicts),
        "harmful_passed_compliance_rate":
            _rates_from_verdicts(harmful_passed_verdicts)["compliance"],
        "n_benign_passed": len(benign_passed_verdicts),
        # benign "answered" = not refused, not gibberish (a useful pass-through).
        "benign_passed_answered_rate":
            1.0 - _rates_from_verdicts(benign_passed_verdicts)["refusal"]
            - _rates_from_verdicts(benign_passed_verdicts)["gibberish"]
            if benign_passed_verdicts else 0.0,
        # SYSTEM leak: harmful prompts that both slipped past the rule AND complied,
        # as a fraction of all harmful eval prompts (the number a defender cares about).
        "system_harmful_leak_rate": n_harm_complied_and_passed / n_harm_total,
        "judge_id": judge.judge_id,
    }

    # --- 6. Auditable examples (a few blocked-harmful + a few passed-benign) -----
    examples = []
    for i, r in enumerate(eval_harm[:6]):
        dec = monitor.decide(E_harm[i])
        out = C.SAFE_REFUSAL if dec["block"] else generate(
            model, tok, r["prompt"], max_new_tokens=max_new, alpha=0.0)
        examples.append({
            "prompt": r["prompt"], "class": "harmful", "block": dec["block"],
            "triggered_by": dec["triggered_by"], "output": out,
            "verdict": "REFUSAL(blocked)" if dec["block"] else judge.verdict(r["prompt"], out),
        })
    for i, r in enumerate(eval_ben[:4]):
        dec = monitor.decide(E_ben[i])
        out = C.SAFE_REFUSAL if dec["block"] else generate(
            model, tok, r["prompt"], max_new_tokens=max_new, alpha=0.0)
        examples.append({
            "prompt": r["prompt"], "class": "benign", "block": dec["block"],
            "triggered_by": dec["triggered_by"], "output": out,
            "verdict": "REFUSAL(blocked)" if dec["block"] else judge.verdict(r["prompt"], out),
        })

    results = {
        "model_id": C.MODEL_ID,
        "layer": int(layer),
        "pooling": C.POOLING,
        "target_fpr": C.TARGET_FPR,
        "rule": rule.name,
        "dataset": {
            "n_extract_per_class": C.N_EXTRACT,
            "n_eval_harmful": len(eval_harm),
            "n_eval_benign": len(eval_ben),
            "natural_toxic_rate": header.get("natural_toxic_rate"),
            "median_char_length": header.get("median_char_length"),
        },
        "cognitive_elements": [
            {"name": d.name, "n_pos": d.n_pos, "tau": d.tau,
             "target_fpr": d.target_fpr} for d in detectors],
        "gavel": gavel,
        "broad_baseline": broad,
        "ce_firing_rates": ce_firing,
        "passthrough": passthrough,
        "examples": examples,
        "plots": {"block_vs_falseblock": C.GATE_PNG.name,
                  "ce_firing_rates": C.CE_PNG.name},
    }

    # --- 7. Save BEFORE any summary print, then plot, then summarize ------------
    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("[save] %s" % C.RESULTS_PATH, file=sys.stderr)
    _plot_block_vs_falseblock(results, C.GATE_PNG)
    _plot_ce_firing(results, C.CE_PNG)
    print("[save] %s" % C.GATE_PNG, file=sys.stderr)
    print("[save] %s" % C.CE_PNG, file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
