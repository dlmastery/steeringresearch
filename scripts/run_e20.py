"""run_e20.py — E20 SAE-TS vs DiffMean 3-stack screening driver.

Pipeline (composes the shared harness with the new `steering.sae` module):
  1. load the model (fake smoke or real Gemma);
  2. for 3 behaviors, extract the DiffMean vector at the injection layer
     (the closed-form baseline 3-stack);
  3. collect residual activations across all behaviors' prompts and TRAIN a
     small sparse autoencoder on them (the SAE training infra);
  4. for each behavior, pick its target SAE features (top features its DiffMean
     vector activates) and OPTIMIZE an SAE-TS steering vector by gradient ascent
     to hit those features while suppressing the others (the SAE-TS 3-stack);
  5. compare Gram mass (Σ|cos| off-diagonal) of the two 3-stacks, and the
     coherence (perplexity) of each summed 3-stack injected into the model.

E20 claim: SAE-TS vectors are more orthogonal (lower Gram mass) → better joint
coherence. Logs one row per condition (DiffMean / SAE-TS) + a campaign artifact.
n=1 SCREENING.

Usage:
  PYTHONPATH=src python scripts/run_e20.py --model fake
  PYTHONPATH=src python scripts/run_e20.py --model models/google/gemma-3-270m-it --quant none --layer 6
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
from steering.eval import perplexity  # noqa: E402
from steering.extract import collect_activations, diffmean_vector  # noqa: E402
from steering.hooks import SteeringContext  # noqa: E402
from steering.model import get_residual_layers, load_model_cached  # noqa: E402
from steering.sae import (  # noqa: E402
    feature_activation,
    gram_mass,
    sae_ts_vector,
    train_sae,
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def _stack_ppl(model, tokenizer, vectors: list[np.ndarray], layer: int,
               passages: list[str], alpha: float) -> float:
    """Perplexity with the summed 3-stack injected at `layer` (additive stack)."""
    combined = torch.tensor(np.sum(vectors, axis=0), dtype=torch.float32)
    with SteeringContext(model, combined, [layer], operation="add", alpha=alpha):
        return perplexity(model, tokenizer, passages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fake")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument("--n-features", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=6, help="target features per behavior")
    ap.add_argument("--alpha", type=float, default=1.0)
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

    behaviors = ["ocean", "happiness", "anger"]
    # --- DiffMean 3-stack + pooled activations for the SAE ---
    diffmean_vecs: list[np.ndarray] = []
    pooled_acts: list[np.ndarray] = []
    for beh in behaviors:
        pairs = ds.load_concept(beh)
        acts = collect_activations(model, tokenizer, pairs, [layer])[layer]
        diffmean_vecs.append(_unit(diffmean_vector(acts["pos"], acts["neg"])))
        pooled_acts.append(acts["pos"])
        pooled_acts.append(acts["neg"])
    activations = np.concatenate(pooled_acts, axis=0).astype(np.float32)

    # --- Train the SAE on the pooled activations ---
    sae = train_sae(activations, n_features=args.n_features, epochs=200, seed=args.seed)

    # --- SAE-TS 3-stack: target each behavior's top features, suppress the rest ---
    saets_vecs: list[np.ndarray] = []
    target_sets: list[list[int]] = []
    for dm in diffmean_vecs:
        feats = feature_activation(sae, dm)
        targets = [int(i) for i in np.argsort(feats)[::-1][: args.top_k]]
        target_sets.append(targets)
    for targets in target_sets:
        v = sae_ts_vector(sae, targets, lam=1.0, steps=200, lr=1e-2, seed=args.seed)
        saets_vecs.append(_unit(np.asarray(v, dtype=np.float32)))

    gm_dm = gram_mass(diffmean_vecs)
    gm_ts = gram_mass(saets_vecs)

    # --- Coherence of each summed 3-stack ---
    passages = ds.load_wikitext_ppl_mini()
    ppl_base = perplexity(model, tokenizer, passages)
    ppl_dm = _stack_ppl(model, tokenizer, diffmean_vecs, layer, passages, args.alpha)
    ppl_ts = _stack_ppl(model, tokenizer, saets_vecs, layer, passages, args.alpha)
    dppl_dm = (ppl_dm - ppl_base) / (ppl_base + 1e-8)
    dppl_ts = (ppl_ts - ppl_base) / (ppl_base + 1e-8)
    # Coherence index (1 - normalized degradation); higher = better.
    coh_dm = 1.0 / (1.0 + max(0.0, dppl_dm))
    coh_ts = 1.0 / (1.0 + max(0.0, dppl_ts))
    coherence_gap = coh_ts - coh_dm
    gram_reduction = gm_dm - gm_ts

    print(f"\n=== E20 SAE-TS vs DiffMean 3-stack (layer {layer}, model {args.model}) ===")
    print(f"  Gram mass   DiffMean={gm_dm:.4f}  SAE-TS={gm_ts:.4f}  (reduction {gram_reduction:+.4f})")
    print(f"  PPL  base={ppl_base:.2f}  DiffMean-stack={ppl_dm:.2f}  SAE-TS-stack={ppl_ts:.2f}")
    print(f"  Coherence   DiffMean={coh_dm:.4f}  SAE-TS={coh_ts:.4f}  (gap {coherence_gap:+.4f})")

    verdict = (
        "SUPPORTED" if (gram_reduction > 0 and coherence_gap >= 0.10)
        else "DIRECTIONAL" if (gram_reduction > 0 and coherence_gap > 0)
        else "NEAR-MISS" if gram_reduction > 0
        else "FALSIFIED"
    )

    citations = (
        "Templeton et al., 2024 arXiv 'Targeted Steering via SAE Side-Effect "
        "Analysis' (arXiv:2411.02193, SAE-TS) — side-effect minimization in SAE "
        "feature space; Chen et al., 2025 arXiv 'FGAA' (arXiv:2501.09929); N5 "
        "(this project): lower joint norm → lower logPPL, the mechanistic path "
        "from orthogonality to coherence."
    )
    reasoning = {
        "diagnosis": (
            "E20 was UNTESTED for lack of SAE infrastructure. With the new "
            "steering.sae module (a self-contained sparse autoencoder + the SAE-TS "
            "gradient-ascent vector optimizer) we can now screen the core claim: do "
            "SAE-TS vectors, optimized to activate target features while suppressing "
            "side-effect features, form a more orthogonal 3-stack than raw DiffMean "
            f"vectors, at injection layer {layer}? This is the first run that "
            "optimizes the steering vector itself rather than extracting it closed-form."
        ),
        "citations": citations,
        "hypothesis": (
            "Mechanism: because SAE-TS minimizes activation of non-target features, "
            "two SAE-TS vectors for different behaviors share fewer feature directions "
            "than two DiffMean vectors, so |cos| and the Gram off-diagonal mass drop; "
            "per N5 the lower joint norm of a more-orthogonal stack yields lower "
            "perplexity (better coherence). We predict gram_mass(SAE-TS) < "
            "gram_mass(DiffMean) and a non-negative coherence gap."
        ),
        "prediction": (
            "Gram mass reduction > 0 (SAE-TS more orthogonal); coherence gap in "
            "[-0.05, +0.20]; SUPPORTED only if gram reduction AND coherence gap >= "
            "0.10. n=1 SCREENING, fingerprint a9001e87087e."
        ),
    }

    base_cfg = {
        "model": args.model, "rung": args.rung, "layer": layer, "seed": args.seed,
        "operation": "add", "source": "diffmean", "behavior": "+".join(behaviors),
        "n_seeds": 1, "quant": args.quant,
    }
    # Row 1: DiffMean 3-stack
    log_method_experiment(
        config={**base_cfg, "tag": "E20-diffmean-3stack", "source": "diffmean"},
        description="E20 DiffMean 3-stack (ocean+happiness+anger) Gram mass + coherence",
        reasoning=reasoning, method="sae_ts_3stack", method_metric="gram_mass",
        method_value=gm_dm,
        method_extra={"behaviors": behaviors, "ppl_base": round(ppl_base, 3),
                      "ppl_stack": round(ppl_dm, 3), "coherence": round(coh_dm, 4),
                      "gram_mass": round(gm_dm, 4)},
        composite=round(coh_dm - 1.0, 4), perplexity=ppl_dm, dppl_norm=dppl_dm,
        behavior_efficacy=coh_dm, elapsed_sec=0.0,
    )
    # Row 2: SAE-TS 3-stack (the optimized-vector condition)
    entry = log_method_experiment(
        config={**base_cfg, "tag": "E20-saets-3stack", "source": "sae_ts"},
        description="E20 SAE-TS 3-stack (optimized vectors) Gram mass + coherence vs DiffMean",
        reasoning=reasoning, method="sae_ts_3stack", method_metric="gram_mass",
        method_value=gm_ts,
        method_extra={"behaviors": behaviors, "target_features": target_sets,
                      "ppl_base": round(ppl_base, 3), "ppl_stack": round(ppl_ts, 3),
                      "coherence": round(coh_ts, 4), "gram_mass": round(gm_ts, 4),
                      "gram_reduction_vs_diffmean": round(gram_reduction, 4),
                      "coherence_gap_vs_diffmean": round(coherence_gap, 4),
                      "verdict": verdict},
        composite=round(coh_ts - 1.0, 4), perplexity=ppl_ts, dppl_norm=dppl_ts,
        behavior_efficacy=coh_ts, started=t0,
    )

    write_campaign("E20-saets", {
        "hyp": "E20", "model": args.model, "layer": layer,
        "behaviors": behaviors, "n_features": args.n_features,
        "gram_mass_diffmean": gm_dm, "gram_mass_saets": gm_ts,
        "gram_reduction": gram_reduction,
        "ppl_base": ppl_base, "ppl_diffmean_stack": ppl_dm, "ppl_saets_stack": ppl_ts,
        "coherence_diffmean": coh_dm, "coherence_saets": coh_ts,
        "coherence_gap": coherence_gap, "verdict": verdict,
        "exp_num": entry["experiment_num"],
    })
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
