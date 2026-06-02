"""run_e15.py — E15 learned gate vs fixed cosine threshold screening driver.

Pipeline (composes the shared harness with the new `steering.gate` module):
  1. load the model (fake smoke or real Gemma);
  2. build per-layer CONDITION vectors (DiffMean of harmful vs benign in-dist
     prompts) at several layers — the E9 cosine-gate directions;
  3. extract multi-layer condition FEATURES (prompt activation . condition
     vector) for the in-distribution set (direct requests) and the OOD set
     (indirect / roleplay / fiction-framed requests — distributionally shifted);
  4. train the fixed CosineGate (single best layer) and the LogisticGate
     (multi-layer, gradient-trained BCE) and compare gate PR-AUC on both sets.

E15 claim: the learned multi-layer gate beats the best fixed cosine threshold by
>= 0.06 AUC under distribution shift (OOD). This is the source experiment where
a GATE (not a steering vector) is gradient-trained. n=1 SCREENING; in-dist AUC is
resubstitution at this tiny scale — the OOD gap is the falsifier of record.

Usage:
  PYTHONPATH=src python scripts/run_e15.py --model fake
  PYTHONPATH=src python scripts/run_e15.py --model models/google/gemma-3-270m-it --quant none
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method_exp_common import log_method_experiment, write_campaign  # noqa: E402

from steering import datasets as ds  # noqa: E402
from steering.extract import collect_activations, diffmean_vector  # noqa: E402
from steering.gate import condition_features, evaluate_gates  # noqa: E402
from steering.model import get_residual_layers, load_model_cached  # noqa: E402

_OOD = ROOT / "src" / "steering" / "data" / "ood_harmful_mini.json"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def _condition_vectors(model, tokenizer, harmful, benign, layers):
    """Per-layer DiffMean(harmful, benign) unit vectors (the E9 gate directions)."""
    n = min(len(harmful), len(benign))
    pairs = list(zip(harmful[:n], benign[:n]))
    acts = collect_activations(model, tokenizer, pairs, layers)
    return {li: _unit(diffmean_vector(acts[li]["pos"], acts[li]["neg"])) for li in layers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fake")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument("--l2", type=float, default=0.1)
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
    # Spread feature layers across depth (clamped to the model), E15 §5.2 used {6,10,14,18}.
    cand = [int(round(f * (n_layers - 1))) for f in (0.25, 0.45, 0.65, 0.85)]
    layers = sorted(set(min(max(li, 0), n_layers - 1) for li in cand))

    harmful = ds.load_jailbreak_mini()      # in-distribution: direct requests
    benign = ds.load_xstest_mini()
    ood = json.loads(_OOD.read_text(encoding="utf-8"))
    ood_harmful, ood_benign = ood["harmful"], ood["benign"]

    cond_vecs = _condition_vectors(model, tokenizer, harmful, benign, layers)

    def feats(prompts):
        return condition_features(model, tokenizer, prompts, layers, cond_vecs)

    f_in = np.concatenate([feats(harmful), feats(benign)], axis=0)
    y_in = np.array([1] * len(harmful) + [0] * len(benign))
    f_ood = np.concatenate([feats(ood_harmful), feats(ood_benign)], axis=0)
    y_ood = np.array([1] * len(ood_harmful) + [0] * len(ood_benign))

    res = evaluate_gates(f_in, y_in, f_ood, y_ood, metric="pr_auc",
                         l2=args.l2, epochs=500, lr=0.5, seed=args.seed,
                         falsifier_gap=0.06)
    gap = float(res["auc_gap_ood"])
    verdict = res["verdict"]
    print(f"\n=== E15 learned gate vs fixed cosine (layers {layers}, model {args.model}) ===")
    print(f"  in-dist  cosine={res['auc_indist_cosine']:.4f}  logistic={res['auc_indist_logistic']:.4f}")
    print(f"  OOD      cosine={res['auc_ood_cosine']:.4f}  logistic={res['auc_ood_logistic']:.4f}")
    print(f"  OOD AUC gap (logistic - best cosine) = {gap:+.4f}  -> {verdict}")

    reasoning = {
        "diagnosis": (
            "E15 was UNTESTED for lack of a gate trainer. The new steering.gate "
            "module (a gradient-trained multi-layer logistic gate + a pure-numpy "
            "PR/ROC-AUC + the fixed CosineGate baseline) now lets us screen whether "
            "a learned multi-layer gate beats a single-layer fixed cosine threshold "
            f"under distribution shift, using feature layers {layers}. In-dist = "
            "direct harmful requests (jailbreak_mini); OOD = indirect/roleplay/"
            "fiction-framed requests (ood_harmful_mini), a genuine surface-form shift."
        ),
        "citations": (
            "Wu et al., 2024 'Conditional Activation Steering' (arXiv:2409.05907) — "
            "the CAST fixed-threshold gate, not evaluated under shift; Tang et al., "
            "2025 'FineSteer/SCS' (arXiv:2604.15488) — energy-ratio gate; Korznikov "
            "et al., 2026 ICML 'The Rogue Scalpel' (arXiv:2509.22067, F4: poor "
            "cross-prompt generalization) — motivates the OOD test; Arditi et al., "
            "2024 (arXiv:2406.11717) — refusal direction shifts OOD."
        ),
        "hypothesis": (
            "Mechanism: because the logistic gate combines dot products from "
            "multiple layers, it can exploit complementary harm signals (shallow "
            "syntax-level, deep semantic) that a single-layer cosine dot product "
            "misses, making the decision boundary more invariant to OOD surface "
            "form. We predict the learned gate's OOD PR-AUC exceeds the best fixed "
            "cosine threshold's by >= 0.06 (the falsifier)."
        ),
        "prediction": (
            "OOD AUC gap (logistic - best cosine) in [-0.05, +0.30]; SUPPORTED iff "
            ">= 0.06. In-dist AUC is resubstitution at this tiny scale. n=1 "
            "SCREENING, fingerprint a9001e87087e."
        ),
    }
    entry = log_method_experiment(
        config={"model": args.model, "rung": args.rung, "layer": layers[0],
                "seed": args.seed, "operation": "gate", "source": "logistic",
                "behavior": "harm_gate", "n_seeds": 1, "quant": args.quant,
                "tag": "E15-learned-gate"},
        description="E15 learned multi-layer logistic gate vs fixed cosine threshold (in-dist + OOD PR-AUC)",
        reasoning=reasoning, method="learned_gate", method_metric="auc_gap_ood",
        method_value=gap,
        method_extra={"feature_layers": layers, **{k: round(float(v), 4)
                      for k, v in res.items() if isinstance(v, (int, float))},
                      "verdict": verdict},
        composite=round(gap, 4),
        behavior_efficacy=float(res["auc_ood_logistic"]),
        behavior_scorer="gate_auc", started=t0,
    )
    write_campaign("E15-learned-gate", {
        "hyp": "E15", "model": args.model, "feature_layers": layers,
        "results": {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in res.items()},
        "exp_num": entry["experiment_num"],
    })
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
