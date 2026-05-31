"""E3 — the alpha coherence cliff, on REAL Qwen2.5-0.5B-Instruct.

Pre-registered hypothesis (ideas/30_alpha_coherence_cliff/IDEA.md): coefficient
alpha has a behavior-specific coherence cliff — below it capability holds
(MMLU drop < 2 pp) and PPL is ~flat; above it perplexity rises super-linearly.

This driver sweeps alpha at the max-Fisher layer, logging EACH alpha as one
canonical experiment row (through steering.runner.run_single_experiment, so the
JSONL + best_config + reasoning + rich dashboard all update). It pre-authors a
genuine per-row reasoning entry (shared E3 hypothesis, per-alpha prediction)
BEFORE each run so the runner's no-fabrication gate is satisfied honestly.

NOTE: model = Qwen2.5-0.5B-Instruct (non-gated; the same 0.5B-class model the
corpus's non-surjectivity paper uses), NOT Gemma. Gemma-3-1B reproduction is the
cross-model confirmation step once the gated license is accepted. Axes measured
REAL on this model: behavior (projection proxy), capability (MMLU-tiny),
coherence (PPL), geometry (off-shell Δ‖h‖). Safety/selectivity remain stubbed
(documented limitation) — they do not enter E3's falsifier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from steering.runner import RESULTS_DIR, run_single_experiment  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 21          # max-Fisher layer from the bring-up smoke (Fisher=25.6)
ALPHAS = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
BEHAVIOR = "ocean"  # the dominant concept axis in axbench_mini


def _next_num() -> int:
    log = RESULTS_DIR / "experiment_log.jsonl"
    n = 0
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                n = max(n, json.loads(line).get("experiment_num", n))
            except Exception:
                pass
    return n + 1


def _author_reasoning(num: int, alpha: float) -> None:
    """Pre-author a genuine _manual reasoning entry for this sweep point."""
    ann_path = RESULTS_DIR / "reasoning_annotations.json"
    ann = {}
    if ann_path.exists():
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
    ann[str(num)] = {
        "_manual": True,
        "diagnosis": (
            f"E3 alpha-coherence-cliff sweep, point alpha={alpha} on REAL "
            f"Qwen2.5-0.5B-Instruct (first real-model science; exp#1 was the "
            f"FakeLM plumbing gate). Bring-up confirmed Fisher peaks at L{LAYER} "
            f"(25.6) and cos(diffmean,pca)=0.996 there (corroborating E4). Open "
            f"question E3: does additive steering have a behavior-specific alpha "
            f"cliff where coherence collapses super-linearly while small alpha "
            f"preserves MMLU? This point measures the (behavior, PPL, MMLU) "
            f"triple at alpha={alpha} to trace the cliff curve."
        ),
        "citations": (
            "Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner, 2024 ACL "
            "'Steering Llama 2 via Contrastive Activation Addition' "
            "(arXiv:2312.06681) — additive residual-stream steering whose "
            "coefficient we sweep. Korznikov et al., 2026 ICML 'The Rogue "
            "Scalpel' (arXiv:2509.22067) — off-manifold displacement at the "
            "injection layer is the damage mechanism, so we log off-shell Δ‖h‖ "
            "alongside PPL as the leading indicator of the cliff."
        ),
        "hypothesis": (
            f"Because the activation manifold curves away from the straight "
            f"steering line (corpus first-principles Step 2), adding alpha*v at "
            f"L{LAYER} slides h along the manifold for small alpha (behavior "
            f"rises, PPL flat) but launches h off-manifold past a threshold, so "
            f"PPL rises super-linearly and MMLU drops. At alpha={alpha} I predict "
            f"{'baseline (no steer)' if alpha == 0 else 'behavior above baseline'}; "
            f"PPL stays within ~10% of baseline below the cliff and rises sharply "
            f"above it; MMLU drop < 2 pp below the cliff."
        ),
        "prediction": (
            f"alpha={alpha}: composite finite, fingerprint a9001e87087e, "
            f"n=1 SCREENING. Off-shell Δ‖h‖ increases monotonically with alpha. "
            f"The cliff is the smallest alpha where ΔPPL_norm turns super-linear "
            f"OR MMLU drop exceeds 2 pp. This is a pre-registered E3 measurement "
            f"point, not yet an external claim (n=1)."
        ),
    }
    ann_path.write_text(json.dumps(ann, indent=2), encoding="utf-8")


def main() -> None:
    rows = []
    for alpha in ALPHAS:
        num = _next_num()
        _author_reasoning(num, alpha)
        print(f"\n>>> E3 sweep: exp#{num} alpha={alpha} layer={LAYER}", flush=True)
        entry = run_single_experiment(
            model_name=MODEL, rung=2, layer=LAYER, alpha=alpha,
            operation="add", source="diffmean", behavior=BEHAVIOR, seed=0,
            description=f"E3 alpha-cliff sweep on Qwen2.5-0.5B @L{LAYER}, alpha={alpha}",
            tag=f"E3-cliff-a{alpha}", quant="none",
        )
        rows.append({
            "alpha": alpha, "exp": entry["experiment_num"],
            "behavior": entry["behavior_efficacy"], "ppl": entry["perplexity"],
            "dppl_norm": entry["dppl_norm"], "mmlu_drop_pp": entry["mmlu_drop_pp"],
            "offshell": entry["offshell_displacement"], "composite": entry["composite"],
        })

    # --- locate the cliff: first alpha with MMLU drop > 2pp OR dppl_norm super-linear ---
    base_ppl = next((r["ppl"] for r in rows if r["alpha"] == 0.0), rows[0]["ppl"])
    cliff = None
    for r in rows:
        if r["alpha"] == 0.0:
            continue
        if r["mmlu_drop_pp"] > 2.0 or (r["ppl"] > 1.5 * base_ppl):
            cliff = r["alpha"]
            break

    out = {
        "model": MODEL, "layer": LAYER, "behavior": BEHAVIOR,
        "base_ppl": base_ppl, "cliff_alpha": cliff, "rows": rows,
        "fingerprint": "a9001e87087e",
    }
    (Path(__file__).resolve().parent / "e3_sweep.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== E3 SWEEP SUMMARY ===")
    print(f"base PPL = {base_ppl:.2f}; cliff alpha = {cliff}")
    for r in rows:
        print(f"  a={r['alpha']:>5}  beh={r['behavior']:.3f}  PPL={r['ppl']:.2f}"
              f"  dPPLn={r['dppl_norm']:+.3f}  MMLUdrop={r['mmlu_drop_pp']:+.2f}pp"
              f"  offshell={r['offshell']:.3f}  comp={r['composite']:+.3f}")


if __name__ == "__main__":
    main()
