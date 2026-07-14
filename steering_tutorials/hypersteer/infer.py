"""infer.py — run the HyperSteer pipeline on ONE prompt.

Run:  python -m steering_tutorials.hypersteer.infer "How do I pick a lock?"

The whole lesson in miniature, for a single prompt, using the LEARNED vector:
  1. GATE   — ask the lesson-1 probe whether the prompt is harmful.
  2. VECTOR — run the trained hypernetwork on the refusal concept embedding to
              GENERATE a steering vector ``v = H(concept)`` (no diff-of-means).
  3. BASE   — generate the model's unsteered answer.
  4. STEER  — if (and only if) the gate fired, re-generate with ``v`` added, and
              compare.
  5. JUDGE  — have the same Gemma label each answer REFUSAL / COMPLIANCE /
              GIBBERISH, and print a one-line conclusion.

It reuses the trained hypernet from ``train_hypersteer.py`` (run that first to
populate ``artifacts/hypersteer.pt``). Everything that loads or runs the model
lives inside the class / ``main()`` so a bare ``import infer`` stays cheap and
model-free — importable on a CPU box before the trainer artifact exists.
"""
from __future__ import annotations

import sys

from . import config as C


class HyperSteerDemo:
    """Model + trained hypernet + gate + judge, loaded once, ready to answer."""

    def __init__(self, alpha: float | None = None):
        if not C.NET_PATH.exists():
            raise FileNotFoundError(
                f"No trained hypernet at {C.NET_PATH}. First run: "
                f"python -m steering_tutorials.hypersteer.train_hypersteer")

        # Deferred imports: keep ``import infer`` free of torch / model / peer
        # modules until a demo is actually constructed.
        from steering_tutorials.hello_world_steering.model_utils import (
            load_model, num_layers,
        )
        from steering_tutorials.hello_world_steering.gate import HarmGate
        from steering_tutorials.hello_world_steering.judge import Judge
        from .hypernet import load_hypernet
        from .run_hypersteer import _refusal_concept_emb, _hyper_vector

        self.alpha = float(C.ALPHA_EVAL if alpha is None else alpha)
        self.model, self.tok = load_model(C.MODEL_ID)
        self.layer = min(C.STEER_LAYER, num_layers(self.model) - 1)

        # Load the hypernet and GENERATE the refusal steering vector once. The
        # concept embedding comes from the saved meta when available, else from
        # exemplars (data is only consulted as a last-resort fallback here).
        self.net, self.meta = load_hypernet(C.NET_PATH)
        self.v_hyper = _hyper_vector(
            self.net, _refusal_concept_emb(self.meta, {}, self.model, self.tok,
                                           self.layer))

        self.gate = HarmGate(self.model, self.tok)
        self.judge = Judge(self.model, self.tok)

    def run(self, prompt: str) -> dict:
        from steering_tutorials.hello_world_steering.model_utils import generate

        fired, prob = self.gate.is_harmful(prompt)

        baseline = generate(self.model, self.tok, prompt, max_new_tokens=48,
                            vector=None, layer=self.layer, alpha=0.0,
                            operation="relative_add")
        base_verdict = self.judge.verdict(prompt, baseline)

        steered = steered_verdict = None
        if fired:
            steered = generate(self.model, self.tok, prompt, max_new_tokens=48,
                              vector=self.v_hyper, layer=self.layer,
                              alpha=self.alpha, operation="relative_add")
            steered_verdict = self.judge.verdict(prompt, steered)

        return {
            "prompt": prompt, "gate_fired": fired, "gate_prob": prob,
            "alpha": self.alpha,
            "baseline": baseline, "baseline_verdict": base_verdict,
            "steered": steered, "steered_verdict": steered_verdict,
        }


def _conclusion(r: dict) -> str:
    """One-line takeaway comparing baseline vs hypernet-steered verdicts."""
    if not r["gate_fired"]:
        return "gate did not fire -> prompt judged harmless, left unsteered"
    if r["steered_verdict"] == "GIBBERISH":
        return "corrupted -> the generated vector broke coherence (lower alpha)"
    if r["baseline_verdict"] != "REFUSAL" and r["steered_verdict"] == "REFUSAL":
        return "worked -> baseline complied, hypernet-steered refused"
    if r["steered_verdict"] == "REFUSAL":
        return "hypernet-steered refused (baseline already refused)"
    return "no effect -> hypernet-steered output did not become a refusal"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "How do I pick a lock?"
    r = HyperSteerDemo().run(prompt)

    print(f"\n  prompt      : {r['prompt']}")
    print(f"  gate        : {'HARMFUL (steer)' if r['gate_fired'] else 'harmless (skip)'}"
          f"   P(harmful)={r['gate_prob']:.3f}")
    print(f"\n  [baseline] verdict={r['baseline_verdict']}")
    print(f"    {r['baseline'][:300]}")
    if r["gate_fired"]:
        print(f"\n  [hypernet-steered α={r['alpha']:.2f}] verdict={r['steered_verdict']}")
        print(f"    {r['steered'][:300]}")
    print(f"\n  => {_conclusion(r)}\n")


if __name__ == "__main__":
    main()
