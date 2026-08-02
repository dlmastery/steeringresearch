"""rejudge.py -- re-run GAVEL's pass-through under an OFF-FAMILY judge.

WHY THIS FILE EXISTS
--------------------
``artifacts/results.json`` recorded ``"judge_id": "self"``: the abliterated
Gemma-3-1B graded its OWN generations, while the README claimed the off-family
Qwen judge. CLAUDE.md section 17 item 3 is explicit -- "Off-family judge for ALL
reported numbers ... never headline a self-judged number" -- so every
pass-through figure in the lesson was inadmissible.

The fix must move EXACTLY ONE VARIABLE: the judge. Same model, same seed, same
data, same split, same layer, same CE library, same rule, same prompts, same
greedy generations, same rubric, same parsing. Only the weights doing the
grading change.

DESIGN: three phases, one model resident at a time
--------------------------------------------------
Loading the 1B target and the 3B judge simultaneously (what ``run_gavel.main()``
does when ``STEER_JUDGE_MODEL`` is set) needs ~8 GB of commit on a host whose
commit limit is the binding constraint. Worse, this host REAPS long jobs. So:

  phase 1 ``generate``  -- load ONLY the abliterated target. Rebuild the monitor
                           deterministically, evaluate the block masks, and
                           generate every PASSED prompt. Cache activations and
                           generations to disk, checkpointing as we go.
  phase 2 ``judge``     -- load ONLY one judge. Grade the CACHED generations.
                           Checkpoint every few items; resume by skipping items
                           already graded.
  phase 3 ``merge``     -- recompute the pass-through metrics per judge and
                           rewrite ``results.json`` with the off-family numbers
                           as the headline and the self-judged numbers retained,
                           labelled superseded.

A SIDE BENEFIT THAT MATTERS SCIENTIFICALLY: because both judges grade the SAME
cached text, the self-vs-off-family delta contains no generation noise at all.
It is a pure measurement of how much the self-judge distorts the result.

ANCHOR ASSERTION (section 18.8: "assert your anchors")
------------------------------------------------------
Phase 1 recomputes the judge-INDEPENDENT numbers (block rates, per-CE firing
rates, the broad baseline) and asserts they reproduce the committed
``results.json`` exactly. If the rebuild had drifted, that assertion FAILS LOUDLY
rather than quietly producing a plausible-but-different number. Only once the
judge-free half is proven identical is the judge swap interpretable.

Run (from the repo root):

    python -m steering_tutorials.gavel.rejudge --phase generate
    python -m steering_tutorials.gavel.rejudge --phase judge --judge self
    python -m steering_tutorials.gavel.rejudge --phase judge --judge Qwen/Qwen2.5-3B-Instruct
    python -m steering_tutorials.gavel.rejudge --phase merge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from . import config as C
from .monitor import GavelMonitor, build_ce_detector, make_rule, Rule

ACTS_CACHE = C.ARTIFACTS / "rejudge_acts.npz"
GEN_CACHE = C.ARTIFACTS / "rejudge_generations.json"
COMPARISON = C.ARTIFACTS / "judge_comparison.json"


def _verdicts_path(judge_id: str):
    slug = judge_id.replace("/", "_").replace(":", "_")
    return C.ARTIFACTS / ("rejudge_verdicts_%s.json" % slug)


def _load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _save_json(path, obj) -> None:
    """Atomic-ish write: temp file then replace, so a reap never truncates."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# PHASE 1 -- generate (target model only; no judge is loaded here at all)
# --------------------------------------------------------------------------- #
def phase_generate() -> None:
    import random

    import numpy as np
    import torch

    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, num_layers, mean_pool_activation, generate)
    from steering_tutorials.common.data import build_harmful_benign

    # Identical seeding to run_gavel.main().
    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)

    from .run_gavel import _env_cap
    max_eval = _env_cap("GAVEL_MAX_EVAL", C.DEFAULT_MAX_EVAL)
    max_new = _env_cap("GAVEL_MAX_NEW_TOKENS", C.MAX_NEW_TOKENS)

    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)

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

    # --- activations, cached per split so a reap costs minutes not hours ------
    cached = {}
    if ACTS_CACHE.exists():
        z = np.load(ACTS_CACHE)
        cached = {k: z[k] for k in z.files}
        print("[acts] resumed %s from cache" % sorted(cached), file=sys.stderr)

    def acts_for(rows, tag):
        if tag in cached and cached[tag].shape[0] == len(rows):
            print("[acts:%s] cached (%d)" % (tag, len(rows)), file=sys.stderr)
            return cached[tag]
        out = []
        for i, r in enumerate(rows):
            out.append(mean_pool_activation(model, tok, r["prompt"], layer))
            if (i + 1) % 25 == 0:
                print("[acts:%s] %d/%d" % (tag, i + 1, len(rows)), file=sys.stderr)
        arr = np.stack(out).astype(np.float32)
        cached[tag] = arr
        np.savez(ACTS_CACHE, **cached)      # checkpoint after every split
        return arr

    A_harm = acts_for(extract_harm, "A_harm")
    A_ben = acts_for(extract_ben, "A_ben")

    # --- CE library: byte-for-byte the logic in run_gavel.main() --------------
    by_cat = defaultdict(list)
    for i, r in enumerate(extract_harm):
        by_cat[r.get("category", "unlabeled")].append(i)

    detectors = []
    leftover_idx = []
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
    if not detectors:
        detectors.append(build_ce_detector("harm", A_harm, A_ben, C.TARGET_FPR))

    rule = make_rule(C.RULE, [d.name for d in detectors])
    monitor = GavelMonitor(detectors, rule)

    broad_ce = build_ce_detector("broad_harm", A_harm, A_ben, C.TARGET_FPR)
    broad_monitor = GavelMonitor([broad_ce], Rule.any_of(["broad_harm"]))

    E_harm = acts_for(eval_harm, "E_harm")
    E_ben = acts_for(eval_ben, "E_ben")

    gavel_harm_block = monitor.block_mask(E_harm)
    gavel_ben_block = monitor.block_mask(E_ben)

    gavel = {"block_rate_harmful": float(gavel_harm_block.mean()),
             "false_block_benign": float(gavel_ben_block.mean()),
             "n_harmful": len(eval_harm), "n_benign": len(eval_ben)}
    broad = {"block_rate_harmful": float(broad_monitor.block_mask(E_harm).mean()),
             "false_block_benign": float(broad_monitor.block_mask(E_ben).mean())}
    ce_firing = {"harmful": monitor.firing_rates(E_harm),
                 "benign": monitor.firing_rates(E_ben)}

    # --- ANCHOR: the judge-free half MUST reproduce the committed run ---------
    prior = _load_json(C.RESULTS_PATH, None)
    if prior is not None:
        drift = []
        for k, v in gavel.items():
            if abs(float(prior["gavel"][k]) - float(v)) > 1e-9:
                drift.append("gavel.%s %s -> %s" % (k, prior["gavel"][k], v))
        for k, v in broad.items():
            if abs(float(prior["broad_baseline"][k]) - float(v)) > 1e-9:
                drift.append("broad.%s %s -> %s" % (k, prior["broad_baseline"][k], v))
        for cls in ("harmful", "benign"):
            for k, v in ce_firing[cls].items():
                pv = prior["ce_firing_rates"][cls].get(k)
                if pv is None or abs(float(pv) - float(v)) > 1e-9:
                    drift.append("ce_firing.%s.%s %s -> %s" % (cls, k, pv, v))
        if drift:
            raise AssertionError(
                "judge-free numbers did NOT reproduce the committed results.json; "
                "the judge swap would not be a single-variable change. Drift:\n  "
                + "\n  ".join(drift))
        print("[anchor] judge-free numbers reproduce results.json EXACTLY "
              "(block rates, broad baseline, per-CE firing) -- the only remaining "
              "difference is the judge.", file=sys.stderr)

    # --- generations, checkpointed per item ----------------------------------
    cache = _load_json(GEN_CACHE, {})
    items = {it["key"]: it for it in cache.get("items", [])}
    n_new = 0

    def gen_item(key, prompt, cls, kind, blocked, triggered_by):
        nonlocal n_new
        if key in items:
            return
        out = C.SAFE_REFUSAL if blocked else generate(
            model, tok, prompt, max_new_tokens=max_new, alpha=0.0)
        items[key] = {"key": key, "kind": kind, "class": cls, "prompt": prompt,
                      "output": out, "block": bool(blocked),
                      "triggered_by": list(triggered_by)}
        n_new += 1
        if n_new % 10 == 0:
            _flush(cache, items, gavel, broad, ce_firing, detectors, rule,
                   header, eval_harm, eval_ben, layer, max_new)
            print("[gen] %d cached (%d new)" % (len(items), n_new), file=sys.stderr)

    # Pass-through: ONLY prompts the rule passed are generated (blocked prompts
    # get SAFE_REFUSAL for free) -- identical to run_gavel.main().
    for i, r in enumerate(eval_harm):
        if gavel_harm_block[i]:
            continue
        gen_item("harmful_passed:%d" % i, r["prompt"], "harmful",
                 "harmful_passed", False, [])
    for i, r in enumerate(eval_ben):
        if gavel_ben_block[i]:
            continue
        gen_item("benign_passed:%d" % i, r["prompt"], "benign",
                 "benign_passed", False, [])

    # Auditable examples (same slices as run_gavel.main()).
    for i, r in enumerate(eval_harm[:6]):
        dec = monitor.decide(E_harm[i])
        gen_item("example_harmful:%d" % i, r["prompt"], "harmful", "example",
                 dec["block"], dec["triggered_by"])
    for i, r in enumerate(eval_ben[:4]):
        dec = monitor.decide(E_ben[i])
        gen_item("example_benign:%d" % i, r["prompt"], "benign", "example",
                 dec["block"], dec["triggered_by"])

    _flush(cache, items, gavel, broad, ce_firing, detectors, rule, header,
           eval_harm, eval_ben, layer, max_new)
    print("[gen] DONE -- %d cached generations -> %s" % (len(items), GEN_CACHE),
          file=sys.stderr)


def _flush(cache, items, gavel, broad, ce_firing, detectors, rule, header,
           eval_harm, eval_ben, layer, max_new) -> None:
    cache["meta"] = {
        "model_id": C.MODEL_ID, "layer": int(layer), "pooling": C.POOLING,
        "target_fpr": C.TARGET_FPR, "rule": rule.name, "seed": C.SEED,
        "max_new_tokens": max_new,
        "dataset": {"n_extract_per_class": C.N_EXTRACT,
                    "n_eval_harmful": len(eval_harm),
                    "n_eval_benign": len(eval_ben),
                    "natural_toxic_rate": header.get("natural_toxic_rate"),
                    "median_char_length": header.get("median_char_length")},
        "cognitive_elements": [{"name": d.name, "n_pos": d.n_pos, "tau": d.tau,
                                "target_fpr": d.target_fpr} for d in detectors],
        "gavel": gavel, "broad_baseline": broad, "ce_firing_rates": ce_firing,
    }
    cache["items"] = [items[k] for k in sorted(items)]
    _save_json(GEN_CACHE, cache)


# --------------------------------------------------------------------------- #
# PHASE 2 -- judge (exactly ONE model resident; grades the CACHED text)
# --------------------------------------------------------------------------- #
def phase_judge(judge_arg: str) -> None:
    from steering_tutorials.hello_world_steering.model_utils import load_model
    from steering_tutorials.hello_world_steering.judge import Judge

    cache = _load_json(GEN_CACHE, None)
    if not cache or not cache.get("items"):
        raise SystemExit("no cached generations -- run --phase generate first")
    items = cache["items"]

    # "self" == the abliterated target grading its own output (the violation we
    # are measuring). Anything else is an off-family judge.
    model_id = C.MODEL_ID if judge_arg == "self" else judge_arg
    label = "self" if judge_arg == "self" else judge_arg

    out_path = _verdicts_path(label)
    done = _load_json(out_path, {})
    todo = [it for it in items if it["key"] not in done]
    print("[judge:%s] %d items, %d already graded, %d to do"
          % (label, len(items), len(done), len(todo)), file=sys.stderr)
    if not todo:
        print("[judge:%s] nothing to do" % label, file=sys.stderr)
        return

    model, tok = load_model(model_id)
    # Build the Judge around THIS model without re-reading the env: Judge.__init__
    # only decides which weights to use; verdict() -- the rubric, the generation,
    # the keyword parsing -- is what we hold fixed across both arms.
    j = Judge.__new__(Judge)
    j.model, j.tok, j.judge_id = model, tok, label

    for n, it in enumerate(todo, 1):
        # A blocked prompt never reached the model in either arm; it is a free
        # refusal by construction, not a judged one. Mark it, don't grade it.
        if it["block"]:
            done[it["key"]] = "REFUSAL(blocked)"
        else:
            done[it["key"]] = j.verdict(it["prompt"], it["output"])
        if n % 20 == 0:
            _save_json(out_path, done)
            print("[judge:%s] %d/%d" % (label, n, len(todo)), file=sys.stderr)

    _save_json(out_path, done)
    print("[judge:%s] DONE -> %s" % (label, out_path), file=sys.stderr)


# --------------------------------------------------------------------------- #
# PHASE 3 -- merge (pure; no model)
# --------------------------------------------------------------------------- #
def _passthrough_for(items, verdicts, n_eval_harmful):
    from .run_gavel import _rates_from_verdicts

    harm = [verdicts[it["key"]] for it in items
            if it["kind"] == "harmful_passed" and it["key"] in verdicts]
    ben = [verdicts[it["key"]] for it in items
           if it["kind"] == "benign_passed" and it["key"] in verdicts]
    hr = _rates_from_verdicts(harm)
    br = _rates_from_verdicts(ben)
    return {
        "n_harmful_passed": len(harm),
        "harmful_passed_compliance_rate": hr["compliance"],
        "harmful_passed_refusal_rate": hr["refusal"],
        "harmful_passed_gibberish_rate": hr["gibberish"],
        "n_benign_passed": len(ben),
        "benign_passed_answered_rate":
            (1.0 - br["refusal"] - br["gibberish"]) if ben else 0.0,
        "system_harmful_leak_rate":
            harm.count("COMPLIANCE") / max(1, n_eval_harmful),
    }


def phase_merge(off_family: str) -> None:
    cache = _load_json(GEN_CACHE, None)
    if not cache:
        raise SystemExit("no cached generations -- run --phase generate first")
    items, meta = cache["items"], cache["meta"]
    n_eval_h = meta["dataset"]["n_eval_harmful"]

    v_self = _load_json(_verdicts_path("self"), {})
    v_off = _load_json(_verdicts_path(off_family), {})
    if not v_off:
        raise SystemExit("off-family verdicts missing -- run --phase judge --judge %s"
                         % off_family)

    pt_off = _passthrough_for(items, v_off, n_eval_h)
    pt_off["judge_id"] = off_family
    pt_self = _passthrough_for(items, v_self, n_eval_h) if v_self else None
    if pt_self:
        pt_self["judge_id"] = "self"

    # Per-item disagreement -- the instrument's own error budget.
    both = [it for it in items
            if it["kind"] in ("harmful_passed", "benign_passed")
            and it["key"] in v_self and it["key"] in v_off]
    disagree = [it for it in both if v_self[it["key"]] != v_off[it["key"]]]
    flips = defaultdict(int)
    for it in disagree:
        flips["%s -> %s" % (v_self[it["key"]], v_off[it["key"]])] += 1

    results = _load_json(C.RESULTS_PATH, {})
    results.update({
        "model_id": meta["model_id"], "layer": meta["layer"],
        "pooling": meta["pooling"], "target_fpr": meta["target_fpr"],
        "rule": meta["rule"], "dataset": meta["dataset"],
        "cognitive_elements": meta["cognitive_elements"],
        "gavel": meta["gavel"], "broad_baseline": meta["broad_baseline"],
        "ce_firing_rates": meta["ce_firing_rates"],
        "passthrough": pt_off,
        "passthrough_superseded_self_judged": {
            **(pt_self or {}),
            "_superseded_reason":
                "The 1B abliterated target graded its own generations. CLAUDE.md "
                "section 17 item 3 forbids headlining a self-judged number; kept "
                "here only to quantify the self-judging bias.",
        },
        "judge_comparison": {
            "generations_identical": True,
            "note": "Both judges graded the SAME cached generations, so this "
                    "delta is pure instrument -- no generation noise.",
            "n_items_compared": len(both),
            "n_disagreements": len(disagree),
            "disagreement_rate": len(disagree) / max(1, len(both)),
            "flips": dict(sorted(flips.items(), key=lambda kv: -kv[1])),
        },
        "examples": [
            {"prompt": it["prompt"], "class": it["class"], "block": it["block"],
             "triggered_by": it["triggered_by"], "output": it["output"],
             "verdict": v_off.get(it["key"], "UNJUDGED"),
             "verdict_self_superseded": v_self.get(it["key"])}
            for it in items if it["kind"] == "example"],
        "plots": {"block_vs_falseblock": C.GATE_PNG.name,
                  "ce_firing_rates": C.CE_PNG.name},
    })
    _save_json(C.RESULTS_PATH, results)
    _save_json(COMPARISON, {"off_family": pt_off, "self_superseded": pt_self,
                            "judge_comparison": results["judge_comparison"]})

    # ASCII only -- this host's console is cp1252.
    print("\n" + "=" * 74)
    print("GAVEL pass-through: SELF-JUDGED (superseded) vs OFF-FAMILY (headline)")
    print("=" * 74)
    print("Same generations, same prompts, same monitor. Only the judge differs.\n")
    print("  %-34s %14s %14s" % ("metric", "self (1B)", "off-family"))
    for k in ("n_harmful_passed", "harmful_passed_compliance_rate",
              "harmful_passed_refusal_rate", "harmful_passed_gibberish_rate",
              "n_benign_passed", "benign_passed_answered_rate",
              "system_harmful_leak_rate"):
        sv = pt_self.get(k) if pt_self else None
        ov = pt_off.get(k)
        fmt = (lambda x: "%14d" % x) if k.startswith("n_") else (
            lambda x: "%14.4f" % x)
        print("  %-34s %s %s" % (k, fmt(sv) if sv is not None else " " * 14,
                                 fmt(ov)))
    jc = results["judge_comparison"]
    print("\n  per-item disagreement: %d/%d (%.3f)"
          % (jc["n_disagreements"], jc["n_items_compared"],
             jc["disagreement_rate"]))
    for k, v in jc["flips"].items():
        print("    %-28s %d" % (k, v))
    print("=" * 74 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True,
                    choices=("generate", "judge", "merge"))
    ap.add_argument("--judge", default="Qwen/Qwen2.5-3B-Instruct",
                    help="'self' or an off-family model id")
    args = ap.parse_args()
    if args.phase == "generate":
        phase_generate()
    elif args.phase == "judge":
        phase_judge(args.judge)
    else:
        phase_merge(args.judge)


if __name__ == "__main__":
    main()
