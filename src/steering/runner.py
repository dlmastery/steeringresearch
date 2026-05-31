"""runner.py — single-experiment executor for the steering autoresearch loop.

Mirrors the FX runner conventions (autoresearch/run_autoresearch.py):
  - running.json written at start, cleared on finish (transient signal)
  - experiment_log.jsonl APPEND-ONLY, experiment_num auto-incremented
  - best_config.json updated iff composite beats the GLOBAL best (KEEP/DISCARD)
  - reasoning_annotations.json: the runner ONLY writes post-run verdict+learning;
    it REFUSES to fabricate the pre-run diagnosis/citations/hypothesis/prediction
    and instead writes a TODO-REWRITE skeleton (like the FX runner) when missing.

The AGENT (Claude) drives the loop; this script executes ONE experiment.

Usage:
    python -m steering.runner --model fake --rung 0 --layer 2 --alpha 4.0 \
        --operation add --source diffmean --behavior ocean --seed 0 \
        --description "Rung-0 plumbing check on FakeLM" --tag smoke-add-L2

CLI flags: --model --rung {0..4} --layer --alpha --operation --source
           --behavior --seed --description --tag
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import datasets as ds
from .eval import (
    composite_fingerprint,
    evaluate_bundle,
    fake_safety_responses,
    generate_responses,
    generation_behavior_scorer,
    lexicon_from_pairs,
    mcq_accuracy,
    perplexity,
    projection_behavior_scorer,
    repetition_rate,
)
from .extract import best_layer, extract_bank
from .geometry import angular_displacement, offshell_displacement
from .hooks import SteeringContext, build_position_mask
from .model import encode_to_device, get_residual_layers, load_model_cached

RESULTS_DIR = Path(__file__).resolve().parents[2] / "autoresearch_results"


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _special_ids(model, tokenizer) -> list[int]:
    """Token ids that must NEVER be steered (BOS + Gemma turn markers).

    On the FakeLM these come from the `*_token_id` attributes. On a real Gemma
    tokenizer `start_of_turn_id`/`end_of_turn_id` are NOT attributes — they must
    be resolved via `convert_tokens_to_ids("<start_of_turn>")`, otherwise only
    BOS was masked and the turn markers leaked into steering.
    """
    ids: list[int] = []
    for attr in ("bos_token_id", "eos_token_id", "start_of_turn_id", "end_of_turn_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            ids.append(int(v))
        v2 = getattr(model, attr, None)
        if v2 is not None:
            ids.append(int(v2))
    # Real HF tokenizers expose the Gemma turn markers only by string lookup.
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        for tok in ("<start_of_turn>", "<end_of_turn>"):
            try:
                tid = convert(tok)
            except Exception:  # pragma: no cover - odd tokenizers
                tid = None
            # HF returns the unk id (often 3) or None for unknown tokens; only
            # accept a positive, non-unk id.
            unk = getattr(tokenizer, "unk_token_id", None)
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                ids.append(tid)
    return sorted(set(ids))


def _is_fake_model(model, model_name: str) -> bool:
    """True for the offline FakeResidualLM (no real generation available).

    A real model (Gemma / Qwen / ...) exposes a usable HF `.generate`; FakeLM
    does not. We key off BOTH the CLI name ("fake") and the runtime type so the
    real-vs-fake split is unambiguous in the logged row.
    """
    if model_name == "fake":
        return True
    return type(model).__name__.startswith("Fake") or not callable(getattr(model, "generate", None))


def run_single_experiment(
    *,
    model_name: str,
    rung: int,
    layer: Optional[int],
    alpha: float,
    operation: str,
    source: str,
    behavior: str,
    seed: int,
    description: str,
    tag: str,
    quant: str = "4bit",
) -> dict:
    """Execute ONE steering experiment and log all artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    running_path = RESULTS_DIR / "running.json"
    config = {
        "model": model_name,
        "rung": rung,
        "layer": layer,
        "alpha": alpha,
        "operation": operation,
        "source": source,
        "behavior": behavior,
        "seed": seed,
        "tag": tag,
        "quant": quant,
    }
    with open(running_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "description": description,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            f,
            indent=2,
        )
    try:
        return _run_inner(config, description)
    finally:
        if running_path.exists():
            running_path.unlink()


def _run_inner(config: dict, description: str) -> dict:
    t0 = time.time()
    _set_seed(config["seed"])

    quant = config.get("quant", "4bit")
    # load_model_cached keeps ONE model per (name, quant, device) for the process
    # and frees the previous one on a fresh key — so an in-process alpha sweep
    # reuses the loaded Gemma instead of reloading it every run (the documented
    # reload-loop OOM/paging-file footgun).
    model, tokenizer = load_model_cached(config["model"], quant=quant)
    n_layers = len(get_residual_layers(model))

    # --- Extraction (cache once, reuse) ---
    pairs = ds.load_axbench_mini()
    cache_dir = RESULTS_DIR / "act_cache"
    bank = extract_bank(
        model,
        tokenizer,
        pairs,
        cache_dir=cache_dir,
        model_tag=str(config["model"]),
        quant=str(quant),
    )

    # Choose layer: explicit, else the most-separable (max Fisher).
    layer = config["layer"]
    if layer is None:
        layer = best_layer(bank)
    layer = int(min(max(layer, 0), n_layers - 1))

    source = config["source"]
    vec_np = bank[layer]["diffmean"] if source == "diffmean" else bank[layer]["pca"]
    vector = torch.tensor(np.asarray(vec_np), dtype=torch.float32)

    alpha = float(config["alpha"])
    operation = config["operation"]

    # --- Build a position mask that excludes special tokens (Rung-0 requirement) ---
    sample_prompt = pairs[0][0]
    sample_ids = encode_to_device(tokenizer, sample_prompt, model)
    pos_mask = build_position_mask(sample_ids, _special_ids(model, tokenizer))

    # --- Geometry: off-shell displacement at the injection layer ---
    from .hooks import probe_activations

    h_base = probe_activations(model, sample_ids, [layer])[layer]
    with SteeringContext(model, vector, [layer], operation=operation, alpha=alpha,
                         position_mask=pos_mask):
        h_steer = probe_activations(model, sample_ids, [layer])[layer]
    offshell = offshell_displacement(h_base, h_steer)
    angular = angular_displacement(h_base, h_steer)  # C3: radial Δ‖h‖ misses rotation

    is_fake = _is_fake_model(model, config["model"])

    # --- Axis 1: behavior efficacy ---
    # REAL models at rung>=1: generation-based concept-incorporation scorer
    # (non-circular, ICML_REVIEW.md W1). FakeLM (or rung 0): projection proxy.
    # generation_behavior_scorer auto-falls-back to the projection proxy when the
    # model can't generate, and TAGS which instrument actually ran.
    behavior_prompts = [p for p, _ in pairs][:6]
    steer_kwargs = {"operation": operation, "alpha": alpha}
    if (not is_fake) and config["rung"] >= 1:
        concept_lexicon = lexicon_from_pairs(pairs)
        beh = generation_behavior_scorer(
            model, tokenizer, layer, vector, behavior_prompts,
            concept_lexicon=concept_lexicon, steering_kwargs=steer_kwargs,
        )
        behavior_efficacy = beh["score"]
        behavior_scorer = beh["scorer"]  # "generation" (real) or "projection" (fell back)
    else:
        behavior_efficacy = projection_behavior_scorer(
            model, tokenizer, vector, layer, behavior_prompts, steering_kwargs=steer_kwargs
        )
        behavior_scorer = "projection"

    # --- Axis 2: capability retention (tiny MCQ, steered vs unsteered) ---
    mcq = ds.load_mmlu_tiny()
    mcq_baseline = mcq_accuracy(model, tokenizer, mcq)

    def _steered_mcq() -> float:
        with SteeringContext(model, vector, [layer], operation=operation, alpha=alpha):
            return mcq_accuracy(model, tokenizer, mcq)

    mcq_steered = _steered_mcq()

    # --- Axis 3: coherence (perplexity + repetition), steered vs baseline ---
    passages = ds.load_wikitext_ppl_mini()
    ppl_baseline = perplexity(model, tokenizer, passages)
    with SteeringContext(model, vector, [layer], operation=operation, alpha=alpha):
        ppl_steered = perplexity(model, tokenizer, passages)
    # repetition rate on the (synthetic) passages as a coherence stand-in
    rep = repetition_rate(passages)

    # --- Axis 4/5: safety + selectivity (rule-based detector on REAL responses) ---
    # ICML_REVIEW.md W2: real models GENERATE (steered) on harmful/harmless prompts
    # so compliance_rate / harmless_refusal_rate are real measurements and the
    # lambda_safe penalty + auto-DISCARD can actually fire. FakeLM keeps the
    # deterministic refusal placeholder (CR≈0 baseline) via generate_responses'
    # internal guard.
    harmful_prompts = ds.load_jailbreak_mini()
    harmless_prompts = ds.load_xstest_mini()
    if (not is_fake) and config["rung"] >= 1:
        harmful_responses = generate_responses(
            model, tokenizer, harmful_prompts, layer=layer, vector=vector,
            steering_kwargs=steer_kwargs,
        )
        harmless_responses = generate_responses(
            model, tokenizer, harmless_prompts, layer=layer, vector=vector,
            steering_kwargs=steer_kwargs,
        )
        safety_real = True
    else:
        # FakeLM / rung-0: a SINGLE offline safety contract via the shared
        # `fake_safety_responses` helper (the same one `generate_responses` uses
        # internally) — refuse every prompt (CR≈0 baseline). This removes the old
        # drift where the runner complied on harmless while generate_responses
        # refused. `safety_real=False` flags it as a placeholder, not a measurement.
        harmful_responses = fake_safety_responses(harmful_prompts)
        harmless_responses = fake_safety_responses(harmless_prompts)
        safety_real = False

    metrics = evaluate_bundle(
        behavior_efficacy=behavior_efficacy,
        mcq_acc=mcq_steered,
        mcq_acc_baseline=mcq_baseline,
        ppl=ppl_steered,
        ppl_baseline=ppl_baseline,
        rep_rate=rep,
        harmful_responses=harmful_responses,
        harmless_responses=harmless_responses,
        offshell=offshell,
    )

    elapsed = time.time() - t0
    composite_val = round(metrics["composite"], 4)

    entry = {
        "config": config,
        "description": description,
        "rung": config["rung"],
        "layer": layer,
        "composite": composite_val,
        "behavior_efficacy": round(metrics["behavior_efficacy"], 4),
        "capability_retention": round(metrics["capability_retention"], 4),
        "mmlu_drop_pp": round(metrics["mmlu_drop_pp"], 4),
        "perplexity": round(metrics["perplexity"], 4),
        "dppl_norm": round(metrics["dppl_norm"], 4),
        "repetition_rate": round(metrics["repetition_rate"], 4),
        "compliance_rate": round(metrics["compliance_rate"], 4),
        "harmful_refusal_rate": round(metrics["harmful_refusal_rate"], 4),
        "harmless_refusal_rate": round(metrics["harmless_refusal_rate"], 4),
        "selectivity_gap": round(metrics["selectivity_gap"], 4),
        "offshell_displacement": round(metrics["offshell_displacement"], 4),
        "angular_displacement": round(angular, 4),
        "fisher_at_layer": round(float(bank[layer]["fisher"]), 4),
        "cosine_dm_pca": round(float(bank[layer]["cosine_dm_pca"]), 4),
        # instrument provenance (ICML_REVIEW.md): which scorer actually ran, so a
        # circular-proxy or stubbed-safety row can never masquerade as a real one.
        "behavior_scorer": behavior_scorer,   # "generation" (real) | "projection" (proxy)
        "safety_real": safety_real,           # True iff CR came from real generations
        "composite_fingerprint": composite_fingerprint(),
        # n=1 single seed ⇒ SCREENING tier (CLAUDE.md §7: n<=3 is screening).
        "n_seeds": 1,
        "tier": "SCREENING",
        "elapsed_sec": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    _log_and_checkpoint(entry, description)
    _print_result(entry)
    return entry


def _log_and_checkpoint(entry: dict, description: str) -> None:
    """Auto-increment experiment_num, append JSONL, update best, write reasoning."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "experiment_log.jsonl"

    prev_num = 0
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    prev_num = max(prev_num, json.loads(line).get("experiment_num", prev_num))
                except (json.JSONDecodeError, ValueError):
                    pass
    entry["experiment_num"] = prev_num + 1

    # Global champion (best_config.json) — KEEP iff composite beats global best.
    best_path = RESULTS_DIR / "best_config.json"
    prev_best = -1e18
    prev_best_tag = None
    if best_path.exists():
        try:
            saved = json.loads(best_path.read_text(encoding="utf-8"))
            prev_best = saved.get("composite", prev_best)
            prev_best_tag = saved.get("config", {}).get("tag")
        except Exception:
            pass
    entry["status"] = "KEEP" if entry["composite"] > prev_best else "DISCARD"

    # Append JSONL (append-only).
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # Reasoning annotations: post-run verdict+learning only; TODO skeleton for pre-run.
    _write_reasoning(entry, description, prev_best, prev_best_tag)

    if entry["composite"] > prev_best:
        best_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")

    # Regenerate the three-tier dashboard (master + per-hypothesis + per-exp)
    # on every experiment (CLAUDE.md §11 / §13: dashboard is pushed every
    # milestone). Never let a dashboard error abort a logged experiment.
    try:
        from .dashboard import build_all_dashboards

        build_all_dashboards(results_dir=RESULTS_DIR, repo_root=RESULTS_DIR.parent)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  WARNING: dashboard regeneration failed: {exc}")


def _write_reasoning(entry: dict, description: str, prev_best: float, prev_best_tag) -> None:
    ann_path = RESULTS_DIR / "reasoning_annotations.json"
    try:
        annotations = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    except Exception:
        annotations = {}

    exp_key = str(entry["experiment_num"])
    existing = annotations.get(exp_key, {})
    is_manual = bool(existing.get("_manual"))

    new_verdict = (
        f"{entry['status']} — composite {entry['composite']:+.4f}, "
        f"behavior {entry['behavior_efficacy']:+.4f}, CR {entry['compliance_rate']:.4f}"
        + (
            f" (new global best, previous {prev_best:+.4f} tag={prev_best_tag})"
            if entry["composite"] > prev_best
            else f" (global best remains {prev_best:+.4f} tag={prev_best_tag})"
        )
    )
    new_learning = (
        f"Behavior {entry['behavior_efficacy']:+.4f} | MMLU drop {entry['mmlu_drop_pp']:.4f} | "
        f"ΔPPL_norm {entry['dppl_norm']:.4f} | rep {entry['repetition_rate']:.4f} | "
        f"CR {entry['compliance_rate']:.4f} | harmless_refusal {entry['harmless_refusal_rate']:.4f} | "
        f"offshell {entry['offshell_displacement']:.4f} | layer {entry['layer']} | "
        f"Fisher {entry['fisher_at_layer']:.4f}"
    )

    has_prerun = (
        is_manual
        and existing.get("diagnosis")
        and existing.get("citations")
        and existing.get("hypothesis")
        and existing.get("prediction")
        and not str(existing.get("diagnosis", "")).startswith("TODO-REWRITE")
    )

    if has_prerun:
        existing["verdict"] = new_verdict
        existing["learning"] = new_learning
        annotations[exp_key] = existing
    else:
        cfg = entry["config"]
        config_delta = (
            f"layer={entry['layer']}, alpha={cfg['alpha']}, operation={cfg['operation']}, "
            f"source={cfg['source']}, behavior={cfg['behavior']}, rung={cfg['rung']}"
        )
        annotations[exp_key] = {
            "diagnosis": (
                f"TODO-REWRITE: steering experiment #{exp_key} (tag={cfg.get('tag')}). "
                f"Description: {description}. Claude must replace this with: why THIS "
                f"config now (champion weakness, which of the 12 axes is being perturbed, "
                f"reference >=1 prior experiment by tag), >=60 words per CLAUDE.md §5.1."
            ),
            "citations": (
                "TODO-REWRITE: full author(s) + YEAR + VENUE + arXiv ID for every paper "
                "motivating this change (e.g. 'Korznikov et al., 2026 ICML \"The Rogue "
                "Scalpel\" (arXiv:2509.22067) — ...'). Parenthetical-only tags insufficient."
            ),
            "hypothesis": (
                f"TODO-REWRITE: mechanistic hypothesis. Config delta: {config_delta}. "
                f"Must state which residual-stream mechanism moves and what the cited "
                f"paper predicts (contains 'mechanism'/'because'/'per [paper]'), >=50 words."
            ),
            "prediction": (
                "TODO-REWRITE: numeric range on the composite + >=1 sub-metric, authored "
                "BEFORE the run (>=25 words). Placeholder here means the 7-step ritual was skipped."
            ),
            "verdict": new_verdict,
            "learning": new_learning,
            "_manual": False,
            "_needs_rewrite": True,
            "composite_fingerprint": entry["composite_fingerprint"],
        }

    ann_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    if annotations[exp_key].get("_needs_rewrite"):
        print(
            f"  WARNING: reasoning_annotations.json[{exp_key}] needs manual rewrite -- "
            f"pre-run diagnosis/citations/hypothesis/prediction were not authored."
        )


def _print_result(entry: dict) -> None:
    print("\n" + "=" * 70)
    print(f"RESULT  exp#{entry['experiment_num']}  rung={entry['rung']}  "
          f"tag={entry['config'].get('tag')}  -- {entry['description']}")
    print("=" * 70)
    print(f"  Composite: {entry['composite']:+.4f}  [{entry['status']}]  "
          f"(fingerprint {entry['composite_fingerprint']}, n={entry['n_seeds']} {entry['tier']})")
    print(f"  Axis1 behavior_efficacy : {entry['behavior_efficacy']:+.4f}")
    print(f"  Axis2 capability_ret    : {entry['capability_retention']:.4f}  "
          f"(MMLU drop {entry['mmlu_drop_pp']:.4f})")
    print(f"  Axis3 coherence         : PPL {entry['perplexity']:.3f}  "
          f"dPPL_norm {entry['dppl_norm']:.4f}  rep {entry['repetition_rate']:.4f}")
    print(f"  Axis4 safety            : compliance_rate {entry['compliance_rate']:.4f}")
    print(f"  Axis5 selectivity       : gap {entry['selectivity_gap']:+.4f}  "
          f"(harmful_ref {entry['harmful_refusal_rate']:.3f}, "
          f"harmless_ref {entry['harmless_refusal_rate']:.3f})")
    print(f"  Geometry offshell dNorm : {entry['offshell_displacement']:.4f}  "
          f"(layer {entry['layer']}, Fisher {entry['fisher_at_layer']:.4f})")
    print(f"  Time: {entry['elapsed_sec']}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONE steering experiment")
    parser.add_argument("--model", default="fake",
                        help="'fake' (offline) or a Gemma id like google/gemma-3-1b-it")
    parser.add_argument("--rung", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--layer", type=int, default=None,
                        help="injection layer; default = max-Fisher layer")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--operation", default="add", choices=["add", "rotate", "project_out"])
    parser.add_argument("--source", default="diffmean", choices=["diffmean", "pca"])
    parser.add_argument("--behavior", default="ocean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--description", required=True)
    parser.add_argument("--tag", default="untagged")
    parser.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none", "bf16"],
                        help="quantisation; use 'none'/'bf16' for non-gated small models without bitsandbytes")
    args = parser.parse_args()

    run_single_experiment(
        model_name=args.model,
        rung=args.rung,
        layer=args.layer,
        alpha=args.alpha,
        operation=args.operation,
        source=args.source,
        behavior=args.behavior,
        seed=args.seed,
        description=args.description,
        tag=args.tag,
        quant=args.quant,
    )


if __name__ == "__main__":
    main()
