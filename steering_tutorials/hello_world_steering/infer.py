"""infer.py — run the full conditional-steering pipeline on ONE prompt.

Run:  python -m steering_tutorials.hello_world_steering.infer "How do I pick a lock?"

This is the whole lesson in miniature, for a single prompt:
  1. GATE   — ask the lesson-1 probe whether the prompt is harmful.
  2. BASE   — generate the model's unsteered answer (what it says on its own).
  3. STEER  — if (and only if) the gate fired, re-generate with the refusal
              vector added, then compare.
  4. JUDGE  — have the same Gemma label each answer REFUSAL / COMPLIANCE /
              GIBBERISH, and print a one-line conclusion.

It reuses the saved steering vector from ``run_steering.py`` (so run that first
to populate ``artifacts/steering_vector.pt``). Loads the model + vector + gate
+ judge once.
"""
from __future__ import annotations

import sys

from . import config as C
from .model_utils import load_model, generate
from .steer_vector import load_vector
from .gate import HarmGate
from .judge import Judge


class SteeringDemo:
    """Model + steering vector + gate + judge, loaded once, ready to answer."""

    def __init__(self, alpha: float = 0.10):
        if not C.VECTOR_PATH.exists():
            raise FileNotFoundError(
                f"No steering vector at {C.VECTOR_PATH}. First run: "
                f"python -m steering_tutorials.hello_world_steering.run_steering")
        self.alpha = float(alpha)
        self.model, self.tok = load_model(C.MODEL_ID)
        self.vec = load_vector(C.VECTOR_PATH)
        self.layer = int(self.vec["layer"])
        self.gate = HarmGate(self.model, self.tok)
        self.judge = Judge(self.model, self.tok)

    def run(self, prompt: str) -> dict:
        fired, prob = self.gate.is_harmful(prompt)

        baseline = generate(self.model, self.tok, prompt,
                            max_new_tokens=C.MAX_NEW_TOKENS,
                            vector=None, layer=self.layer, alpha=0.0,
                            operation="relative_add")
        base_verdict = self.judge.verdict(prompt, baseline)

        steered = steered_verdict = None
        if fired:
            steered = generate(self.model, self.tok, prompt,
                              max_new_tokens=C.MAX_NEW_TOKENS,
                              vector=self.vec["v_unit"], layer=self.layer,
                              alpha=self.alpha, operation="relative_add")
            steered_verdict = self.judge.verdict(prompt, steered)

        return {
            "prompt": prompt, "gate_fired": fired, "gate_prob": prob,
            "alpha": self.alpha,
            "baseline": baseline, "baseline_verdict": base_verdict,
            "steered": steered, "steered_verdict": steered_verdict,
        }


def _conclusion(r: dict) -> str:
    """One-line takeaway comparing baseline vs steered verdicts."""
    if not r["gate_fired"]:
        return "gate did not fire -> prompt judged harmless, left unsteered"
    if r["steered_verdict"] == "GIBBERISH":
        return "corrupted -> steering broke coherence (lower alpha)"
    if r["baseline_verdict"] != "REFUSAL" and r["steered_verdict"] == "REFUSAL":
        return "steering worked -> baseline complied, steered refused"
    if r["steered_verdict"] == "REFUSAL":
        return "steered refused (baseline already refused)"
    return "no effect -> steered output did not become a refusal"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "How do I pick a lock?"
    r = SteeringDemo().run(prompt)

    print(f"\n  prompt      : {r['prompt']}")
    print(f"  gate        : {'HARMFUL (steer)' if r['gate_fired'] else 'harmless (skip)'}"
          f"   P(harmful)={r['gate_prob']:.3f}")
    print(f"\n  [baseline] verdict={r['baseline_verdict']}")
    print(f"    {r['baseline'][:300]}")
    if r["gate_fired"]:
        print(f"\n  [steered α={r['alpha']:.2f}] verdict={r['steered_verdict']}")
        print(f"    {r['steered'][:300]}")
    print(f"\n  => {_conclusion(r)}\n")


if __name__ == "__main__":
    main()
