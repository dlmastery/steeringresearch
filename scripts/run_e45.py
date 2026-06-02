"""run_e45.py — E45 HyperSteer zero-shot screening driver.

Pipeline (composes the shared harness with the new `steering.hypersteer` module):
  1. load the model (fake smoke or real Gemma);
  2. for each behavior, extract its supervised DiffMean vector at the injection
     layer (the closed-form target) and encode its natural-language description
     via the model's own hidden state (the hypernetwork input);
  3. leave-one-behavior-out: TRAIN a hypernetwork (Adam / MSE regression) on the
     other behaviors' (description-embedding -> DiffMean-vector) pairs, then
     PREDICT the held-out behavior's vector from its description alone;
  4. score the held-out prediction geometrically (cosine to the supervised
     vector) and behaviorally (projection efficacy ratio vs supervised).

E45 claim: description -> vector generalizes to held-out behaviors at >= 70% of
supervised efficacy. This is the source experiment where a NETWORK that emits
steering vectors is gradient-trained (vs closed-form DiffMean). n=1 SCREENING;
the 4-behavior leave-one-out here is smoke-scale (the unit test proves
generalization on a larger synthetic set; this validates the real-model loop).

Usage:
  PYTHONPATH=src python scripts/run_e45.py --model fake
  PYTHONPATH=src python scripts/run_e45.py --model models/google/gemma-3-270m-it --quant none --layer 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method_exp_common import log_method_experiment, write_campaign  # noqa: E402

from steering import datasets as ds  # noqa: E402
from steering.eval import projection_behavior_scorer  # noqa: E402
from steering.extract import collect_activations, diffmean_vector  # noqa: E402
from steering.hypersteer import (  # noqa: E402
    cosine,
    encode_descriptions,
    predict_vector,
    train_hypernet,
)
from steering.model import get_residual_layers, load_model_cached  # noqa: E402

DESCRIPTIONS = {
    "ocean": "Write about the ocean, the deep sea, waves, tides, and marine life.",
    "happiness": "Express happiness, joy, delight, cheerfulness, and positive emotion.",
    "anger": "Express anger, rage, fury, hostility, and irritated outrage.",
    "formality": "Write in a formal, polite, professional, and proper register.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fake")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-log", action="store_true", help="smoke run; do not append to the ledger")
    args = ap.parse_args()

    import method_exp_common
    method_exp_common.LOGGING_ENABLED = not args.no_log
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer = load_model_cached(args.model, quant=args.quant)
    n_layers = len(get_residual_layers(model))
    layer = args.layer if args.layer is not None else max(1, n_layers // 2)
    layer = int(min(max(layer, 0), n_layers - 1))

    behaviors = list(DESCRIPTIONS.keys())
    # Supervised DiffMean targets + description embeddings (one per behavior).
    targets: dict[str, np.ndarray] = {}
    prompts: dict[str, list[str]] = {}
    for beh in behaviors:
        pairs = ds.load_concept(beh)
        acts = collect_activations(model, tokenizer, pairs, [layer])[layer]
        v = diffmean_vector(acts["pos"], acts["neg"]).astype(np.float32)
        targets[beh] = v / (np.linalg.norm(v) + 1e-8)
        prompts[beh] = [p for p, _ in pairs][:6]
    embeds = encode_descriptions(model, tokenizer, [DESCRIPTIONS[b] for b in behaviors], layer)
    embed_of = {b: embeds[i] for i, b in enumerate(behaviors)}

    # Leave-one-behavior-out: train on the rest, predict the held-out one.
    fold_cos: list[float] = []
    fold_ratio: list[float] = []
    folds = []
    for held in behaviors:
        train_b = [b for b in behaviors if b != held]
        X = np.stack([embed_of[b] for b in train_b]).astype(np.float32)
        Y = np.stack([targets[b] for b in train_b]).astype(np.float32)
        net = train_hypernet(X, Y, epochs=args.epochs, hidden=256, depth=2,
                             seed=args.seed, normalize_targets=True)
        pred = predict_vector(net, embed_of[held]).astype(np.float32)
        pred_u = pred / (np.linalg.norm(pred) + 1e-8)
        cos = cosine(pred_u, targets[held])

        # Behavioral efficacy ratio (projection proxy; fast, model-agnostic).
        sup_t = torch.tensor(targets[held], dtype=torch.float32)
        pred_t = torch.tensor(pred_u, dtype=torch.float32)
        sk = {"operation": "add", "alpha": args.alpha}
        eff_sup = projection_behavior_scorer(model, tokenizer, sup_t, layer,
                                             prompts[held], steering_kwargs=sk)
        eff_pred = projection_behavior_scorer(model, tokenizer, pred_t, layer,
                                              prompts[held], steering_kwargs=sk)
        ratio = float(eff_pred / eff_sup) if abs(eff_sup) > 1e-6 else 0.0
        fold_cos.append(cos)
        fold_ratio.append(ratio)
        folds.append({"held_out": held, "cosine": round(cos, 4),
                      "eff_supervised": round(eff_sup, 4),
                      "eff_predicted": round(eff_pred, 4),
                      "efficacy_ratio": round(ratio, 4)})
        print(f"  held-out {held:>9}: cos={cos:+.4f}  eff_ratio={ratio:+.4f}")

    mean_cos = float(np.mean(fold_cos))
    mean_ratio = float(np.mean(fold_ratio))
    std_cos = float(np.std(fold_cos))
    # Verdict is COSINE-primary. The projection efficacy ratio is a NON-CAUSAL
    # proxy here: a folder with cosine ~ -0.88 (vector nearly opposite the
    # supervised one) still scores ratio > 1.0, so the ratio cannot validate the
    # hypernetwork (ICML_REVIEW circular-proxy caveat; the design doc §10 mandates
    # real-generation / LLM-judge efficacy for any external claim). At this 4-
    # behavior leave-one-out scale we therefore gate on the geometric cosine.
    verdict = (
        "SUPPORTED" if (mean_cos >= 0.5 and mean_ratio >= 0.70)
        else "DIRECTIONAL" if mean_cos > 0.3
        else "INCONCLUSIVE"  # mean cosine ~ 0 with high cross-behavior variance
    )
    print(f"\n=== E45 HyperSteer LOO (layer {layer}, model {args.model}) ===")
    print(f"  mean held-out cosine={mean_cos:+.4f}  mean efficacy ratio={mean_ratio:+.4f}  -> {verdict}")

    reasoning = {
        "diagnosis": (
            "E45 was UNTESTED for lack of a hypernetwork trainer. The new "
            "steering.hypersteer module (an MLP trained by Adam on MSE regression "
            "from description embeddings to DiffMean vectors) now lets us screen "
            "whether a behavior's steering vector is predictable from its text "
            f"description alone, at injection layer {layer}. Leave-one-behavior-out "
            "over the 4 available concepts; the unit test already proved "
            "generalization on a larger synthetic set."
        ),
        "citations": (
            "Hernandez et al., 2025 'HyperSteer: Concept-based Activation Steering "
            "via Hypernetworks' (arXiv:2506.03292) — the primary method; Zou et al., "
            "2023 'Representation Engineering' (arXiv:2310.01405) — DiffMean target; "
            "Zhong et al., 2025 'AxBench' (arXiv:2501.17148) — held-out concept "
            "evaluation methodology."
        ),
        "hypothesis": (
            "Mechanism: because the description-embedding space and the steering-"
            "vector space are both low-dimensional and smoothly related, a small "
            "hypernetwork can learn the mapping and generalize to a held-out behavior "
            "it never saw a contrast set for. We predict held-out cosine > 0 and an "
            "efficacy ratio approaching the >= 0.70 supervised-efficacy threshold; "
            "this is the description-only (zero-contrast-set) workflow per HyperSteer."
        ),
        "prediction": (
            "Mean held-out cosine in [0.2, 0.9]; mean efficacy ratio in [0.3, 0.9]; "
            "SUPPORTED iff ratio >= 0.70. 4-behavior LOO is smoke-scale, n=1 "
            "SCREENING, fingerprint a9001e87087e."
        ),
    }
    entry = log_method_experiment(
        config={"model": args.model, "rung": args.rung, "layer": layer,
                "seed": args.seed, "operation": "add", "source": "hypernet",
                "behavior": "+".join(behaviors), "n_seeds": 1, "quant": args.quant,
                "tag": "E45-hypersteer-loo"},
        description="E45 HyperSteer leave-one-behavior-out: description -> vector cosine + efficacy ratio",
        reasoning=reasoning, method="hypersteer_loo", method_metric="mean_efficacy_ratio",
        method_value=mean_ratio,
        method_extra={"mean_holdout_cosine": round(mean_cos, 4),
                      "std_holdout_cosine": round(std_cos, 4),
                      "mean_efficacy_ratio": round(mean_ratio, 4),
                      "folds": folds, "verdict": verdict,
                      "primary_metric": "mean_holdout_cosine",
                      "proxy_caveat": ("efficacy_ratio is a non-causal projection "
                                       "proxy; a cos<0 fold still scores ratio>1, so "
                                       "cosine is the trustworthy signal at n=4"),
                      "threshold_supported": 0.70, "threshold_falsify": 0.50},
        composite=round(mean_cos, 4), behavior_efficacy=mean_ratio,
        behavior_scorer="projection", started=t0,
    )
    write_campaign("E45-hypersteer", {
        "hyp": "E45", "model": args.model, "layer": layer, "behaviors": behaviors,
        "mean_holdout_cosine": mean_cos, "mean_efficacy_ratio": mean_ratio,
        "folds": folds, "verdict": verdict, "exp_num": entry["experiment_num"],
    })
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
