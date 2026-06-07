"""confirm_e7_boot.py — E7 confirmation where the REPLICATION UNIT is an
independently-extracted vector, not a generation dice-roll.

Why this exists: the earlier confirm_e7.py varied only the generation seed
(temperature sampling) while holding the vector, prompts, and controls FIXED — so
its n=20 measured generation noise, not real replication. This driver fixes that:

  * UNIT OF REPLICATION = a bootstrap-extracted vector. For each concept we
    resample the (expanded, ~30) contrast pairs WITH REPLACEMENT B times and
    rebuild DiffMean each time -> B genuinely different steering vectors.
  * GREEDY (deterministic) generation per (vector x prompt) -> a vector gets ONE
    behavior score; ALL variance comes from the extraction, the axis that matters.
  * CONTROLS DRAWN FRESH PER REPLICATE: a new random direction and a new
    shuffled-label split for EACH bootstrap b (not one fixed draw).
  * MORE PROMPTS: an 8-prompt held-out, concept-neutral eval set, judged by the
    OFF-FAMILY Gemini judge (proxy fallback if no key).
  * ACROSS CONCEPTS: ocean + anger + happiness, Holm-corrected across them.
  * RIGOR over the B replicates (paired by bootstrap index): paired Wilcoxon +
    bootstrap CI + ordinal gate (stats.rigor_report), Holm across concepts.

HONEST CAVEAT (printed + logged): bootstrap-resampling a ~30-pair set gives
correlated draws, so the effective independent n is < B; the ACROSS-CONCEPT
consistency is the stronger evidence. Larger/real contrast corpora are the true
fix.

Usage:
  PYTHONPATH=src python scripts/confirm_e7_boot.py --quick                       # plumbing, proxy, no-log
  GEMINI_API_KEY=... PYTHONPATH=src python scripts/confirm_e7_boot.py \
      --model models/google/gemma-3-270m-it --quant none --layer 16 \
      --concepts ocean anger happiness --boots 50 --prompts 8 --knee 0.1

NOTE: the composite field now holds only the fingerprinted 5-axis composite; the
raw behavior metric (mean real-minus-shuffled delta across concepts) is in method_value.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method_exp_common import log_method_experiment, write_campaign  # noqa: E402

from steering import datasets as ds  # noqa: E402
from steering.controls import (  # noqa: E402
    extraction_stability,
    random_direction,
    shuffled_label_vector,
)
from steering.eval import _ids, concept_rate, lexicon_from_pairs, perplexity  # noqa: E402
from steering.extract import collect_activations, diffmean_vector  # noqa: E402
from steering.hooks import SteeringContext  # noqa: E402
from steering.judge import JudgeUnavailable, make_judge_or_none  # noqa: E402
from steering.model import get_residual_layers, load_model_cached  # noqa: E402
from steering.stats import holm_bonferroni, rigor_report  # noqa: E402

_EXP = ROOT / "src" / "steering" / "data" / "concepts_expanded.json"
DESC = {
    "ocean": "the text vividly describes the ocean, the sea, waves, tides, or marine life",
    "anger": "the text expresses anger, rage, fury, hostility, or outrage",
    "happiness": "the text expresses happiness, joy, delight, or cheerful positive emotion",
}
# Concept-NEUTRAL held-out prompts (distinct from the extraction sentences); steering
# toward a concept should make the continuation exhibit it, which the judge detects.
PROMPTS = [
    "Write a short paragraph about anything that comes to mind.",
    "Tell me about what happened today.",
    "Describe the scene in front of you.",
    "Continue this story: It was an ordinary morning when",
    "Here is a description of the place:",
    "Let me tell you what happened next.",
    "The first thing I noticed was",
    "Write a few sentences about how things are going.",
    "Describe what you are thinking about right now.",
    "Write the opening lines of a short story.",
    "Set the scene for what is about to happen.",
    "Write a brief journal entry for today.",
    "Recount what just took place.",
    "Paint a picture with words of the surroundings.",
    "Share a passing thought you are having.",
    "Begin a description of the room.",
    "What comes next? Continue the passage:",
    "Put into words how this moment feels.",
    "Tell me a little about where you are.",
    "Write a few lines to open a letter.",
]


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _load_expanded(concept: str) -> list[tuple[str, str]]:
    data = json.loads(_EXP.read_text(encoding="utf-8"))["concepts"]
    if concept in data:
        return [(p["pos"], p["neg"]) for p in data[concept]]
    return ds.load_concept(concept)


def _greedy_gen(model, tok, prompt: str, *, layer: int, vector: torch.Tensor,
                alpha: float, max_new_tokens: int = 24) -> str:
    ids = _ids(tok, prompt, model)
    gk: dict[str, Any] = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    eos = getattr(tok, "eos_token_id", None)
    if getattr(tok, "pad_token_id", None) is None and eos is not None:
        gk["pad_token_id"] = eos
    gen = cast(Callable[..., torch.Tensor], model.generate)
    with SteeringContext(model, vector, [layer], operation="relative_add", alpha=alpha):
        with torch.no_grad():
            out = gen(ids, **gk)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fake")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--concepts", nargs="+", default=["ocean", "anger", "happiness"])
    ap.add_argument("--boots", type=int, default=50, help="bootstrap-extracted vectors per concept (the real n)")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--knee", type=float, default=0.1)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-random", action="store_true",
                    help="drop the random-direction arm (it only re-confirms coherence collapse)")
    ap.add_argument("--quick", action="store_true", help="tiny plumbing run (proxy, no-log)")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.concepts, args.boots, args.prompts, args.no_log = ["ocean"], 3, 2, True

    import method_exp_common
    method_exp_common.LOGGING_ENABLED = not args.no_log
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tok = load_model_cached(args.model, quant=args.quant)
    n_layers = len(get_residual_layers(model))
    layer = int(min(max(args.layer, 0), n_layers - 1))
    eval_prompts = PROMPTS[: args.prompts]

    judge = make_judge_or_none()
    if judge is not None:
        try:
            judge.behavior_efficacy("The sea was calm.", "ocean", DESC["ocean"])
        except JudgeUnavailable:
            judge = None
    instrument = "gemini_judge" if judge is not None else "lexicon_proxy"
    print(f"[instrument] {instrument} | layer {layer} | boots/concept {args.boots} | "
          f"prompts {len(eval_prompts)} | concepts {args.concepts}")

    passages = ds.load_wikitext_ppl_mini()
    ppl_base = perplexity(model, tok, passages)
    rng = np.random.default_rng(args.seed)

    per_concept: dict[str, Any] = {}
    concept_p: list[float] = []
    for concept in args.concepts:
        pairs = _load_expanded(concept)
        desc = DESC.get(concept, f"the text exhibits '{concept}'")
        lex = lexicon_from_pairs(pairs)
        acts = collect_activations(model, tok, pairs, [layer])[layer]
        pos, neg = acts["pos"], acts["neg"]
        n = pos.shape[0]
        stab = extraction_stability(pos, neg, n_boot=200, seed=args.seed)

        def score(text: str, _d: str = desc, _c: str = concept) -> float:
            if judge is not None:
                try:
                    return judge.behavior_efficacy(text, _c, _d)
                except JudgeUnavailable:
                    pass
            return concept_rate(text, lex)

        def behavior_of(vec_np: np.ndarray) -> tuple[float, float]:
            v = torch.tensor(_unit(vec_np), dtype=torch.float32)
            gens = [_greedy_gen(model, tok, p, layer=layer, vector=v, alpha=args.knee,
                                max_new_tokens=args.max_new_tokens) for p in eval_prompts]
            beh = float(np.mean([score(g) for g in gens]))
            with SteeringContext(model, v, [layer], operation="relative_add", alpha=args.knee):
                ppl = perplexity(model, tok, passages)
            coh = 1.0 / (1.0 + max(0.0, (ppl - ppl_base) / (ppl_base + 1e-8)))
            return beh, coh

        real_b, shuf_b, rand_b = [], [], []
        real_coh, shuf_coh = [], []
        for b in range(args.boots):
            idx = rng.integers(0, n, size=n)                       # bootstrap WITH replacement
            v_real = diffmean_vector(pos[idx], neg[idx])           # a genuinely different vector
            v_shuf = shuffled_label_vector(pos, neg, seed=1000 + b)  # fresh shuffle per b
            rb, rc = behavior_of(v_real)
            real_b.append(rb)
            real_coh.append(rc)
            sb, sc = behavior_of(v_shuf)
            shuf_b.append(sb)
            shuf_coh.append(sc)
            if not args.no_random:
                v_rand = random_direction(pos.shape[1], seed=2000 + b)   # fresh random per b
                nb, _ = behavior_of(v_rand)
                rand_b.append(nb)
            if (b + 1) % max(1, args.boots // 5) == 0:
                rmsg = f" rand {np.mean(rand_b):.3f}" if rand_b else ""
                print(f"  [{concept}] {b+1}/{args.boots}  real {np.mean(real_b):.3f} "
                      f"shuf {np.mean(shuf_b):.3f}{rmsg}", flush=True)

        vs_shuf = rigor_report(real_b, shuf_b)
        vs_rand = rigor_report(real_b, rand_b) if rand_b else None
        matched = abs(np.mean(real_coh) - np.mean(shuf_coh)) < 0.15
        concept_p.append(vs_shuf["wilcoxon"]["p_value"])
        per_concept[concept] = {
            "n_boot": args.boots, "n_pairs": int(n),
            "mean_real": float(np.mean(real_b)), "mean_shuffled": float(np.mean(shuf_b)),
            "mean_random": (float(np.mean(rand_b)) if rand_b else None),
            "delta_real_minus_shuffled": float(np.mean(real_b) - np.mean(shuf_b)),
            "vs_shuffled": vs_shuf, "vs_random": vs_rand,
            "matched_coherence": bool(matched),
            "coh_real": float(np.mean(real_coh)), "coh_shuffled": float(np.mean(shuf_coh)),
            "extraction_stability": stab["mean_cosine_to_full"],
        }
        print(f"=== {concept}: real {np.mean(real_b):.3f} vs shuffled {np.mean(shuf_b):.3f} "
              f"(delta {np.mean(real_b)-np.mean(shuf_b):+.3f}, CI [{vs_shuf['bootstrap_ci']['lo']:+.3f},"
              f"{vs_shuf['bootstrap_ci']['hi']:+.3f}]) Wilcoxon p={vs_shuf['wilcoxon']['p_value']:.4g} "
              f"ordinal={vs_shuf['ordinal_gate']['passes']} matched_coh={matched}")

    holm = holm_bonferroni(concept_p)
    # EXTERNAL-READY (this protocol): every concept's real-vs-shuffled passes the full
    # contract AND is Holm-rejected across the concept family AND matched coherence.
    all_pass = all(
        per_concept[c]["vs_shuffled"]["external_ready"]
        and per_concept[c]["matched_coherence"]
        and bool(holm["reject"][i])
        for i, c in enumerate(args.concepts)
    )
    n_sig = sum(per_concept[c]["vs_shuffled"]["legs"]["wilcoxon_significant"] and bool(holm["reject"][i])
                for i, c in enumerate(args.concepts))
    verdict = ("EXTERNAL-READY(this-scale)" if all_pass
               else f"DIRECTIONAL ({n_sig}/{len(args.concepts)} concepts Holm-significant; strict gates unmet)")
    print(f"\n=== E7 bootstrap confirmation ({args.model}, instrument={instrument}, "
          f"n_boot={args.boots}/concept) ===")
    for c in args.concepts:
        pc = per_concept[c]
        print(f"  {c:<10} delta={pc['delta_real_minus_shuffled']:+.3f} "
              f"Wilcoxon_p={pc['vs_shuffled']['wilcoxon']['p_value']:.4g} "
              f"ordinal={pc['vs_shuffled']['ordinal_gate']['passes']} "
              f"matched_coh={pc['matched_coherence']} stability={pc['extraction_stability']:.2f}")
    print(f"  Holm across concepts: reject={holm['reject']}")
    print(f"  VERDICT: {verdict}")
    print(f"  CAVEAT: bootstrap from ~{per_concept[args.concepts[0]]['n_pairs']} pairs is correlated "
          f"(effective n < {args.boots}); cross-concept consistency is the stronger signal.")

    if not args.no_log:
        tag = f"E7-boot-{Path(args.model).name}"
        write_campaign(tag, {
            "hyp": "E7", "model": args.model, "layer": layer, "instrument": instrument,
            "n_boot": args.boots, "prompts": len(eval_prompts), "knee": args.knee,
            "concepts": args.concepts, "per_concept": per_concept,
            "holm_concept_p": concept_p, "holm": holm, "verdict": verdict,
            "caveat": "bootstrap-from-~30-pairs is correlated; effective n < n_boot",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        mean_delta = float(np.mean([per_concept[c]["delta_real_minus_shuffled"] for c in args.concepts]))
        reasoning = {
            "diagnosis": (
                "Re-do of the E7 directional confirmation with the REPLICATION UNIT = an "
                "independently bootstrap-extracted vector (not a generation seed). The prior "
                "n=20 varied only generation RNG with the vector/prompts/controls fixed, so "
                f"its n was illusory. Here {args.boots} bootstrap vectors per concept, fresh "
                "controls per bootstrap, greedy generation, off-family judge, across "
                f"{len(args.concepts)} concepts."
            ),
            "citations": (
                "Panickssery et al., 2024 'Contrastive Activation Addition' (arXiv:2312.06681); "
                "Efron & Tibshirani 1993 (the bootstrap); the matched-displacement shuffled-"
                "label control isolates DIRECTION from magnitude."
            ),
            "hypothesis": (
                "If the DiffMean concept direction carries behavior (not norm), the real "
                "direction beats a label-shuffled direction at matched displacement and "
                "coherence, ROBUSTLY across independent extractions and across concepts — "
                "paired Wilcoxon over the bootstraps, Holm across concepts."
            ),
            "prediction": (
                "mean real-minus-shuffled delta > 0 with CI excluding 0 on each concept; "
                "EXTERNAL-READY only if every concept passes the full contract + Holm + "
                "matched coherence. fingerprint a9001e87087e."
            ),
        }
        log_method_experiment(
            config={"model": args.model, "rung": 3, "layer": layer, "seed": args.seed,
                    "operation": "relative_add", "source": "diffmean",
                    "behavior": "+".join(args.concepts), "n_seeds": args.boots,
                    "quant": args.quant, "tag": tag},
            description=f"E7 bootstrap-extraction confirmation: real vs shuffled over {args.boots} vectors x {len(args.concepts)} concepts",
            reasoning=reasoning, method="e7_bootstrap", method_metric="mean_real_minus_shuffled",
            method_value=mean_delta,
            method_extra={"instrument": instrument, "n_boot": args.boots,
                          "per_concept": {c: {k: per_concept[c][k] for k in
                                              ("delta_real_minus_shuffled", "matched_coherence",
                                               "extraction_stability")} for c in args.concepts},
                          "concept_wilcoxon_p": concept_p, "holm_reject": list(holm["reject"]),
                          "verdict": verdict},
            # No 5-axis composite (a bootstrap directional confirmation); the
            # mean real-minus-shuffled delta is in method_value. composite=None
            # keeps the composite column honest (eval.composite() output only).
            composite=None,
            behavior_efficacy=float(np.mean([per_concept[c]["mean_real"] for c in args.concepts])),
            behavior_scorer=instrument, started=t0,
        )
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
