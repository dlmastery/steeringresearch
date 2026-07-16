"""infer.py -- run the trained TALAN adapter on ONE prompt.

Run:  python -m steering_tutorials.talan.infer "How do I pick a lock?"

The whole lesson in miniature, for a single prompt:
  1. GATE   -- ask the lesson-1 probe whether the prompt is harmful (the same
              condition used in eval; the learned adapter fires only if it does).
  2. BASE   -- generate the model's UNSTEERED answer (what it says on its own).
  3. TALAN  -- if (and only if) the gate fired, re-generate inside a TalanContext so
              the learned latent adapter is applied, then compare.
  4. JUDGE  -- have the judge (Qwen off-family via STEER_JUDGE_MODEL, else the same
              Gemma) label each answer REFUSAL / COMPLIANCE / GIBBERISH.

It reuses the trained adapter saved by the trainer (``artifacts/talan.pt``), so
train that first. Everything that loads or runs the model -- including the peer
``talan`` import -- is deferred into the class / ``main()`` so a bare ``import
infer`` is a no-op and import-checks green on a CPU box before ``talan.pt`` exists.
"""
from __future__ import annotations

import sys

from . import config as C

MAX_NEW_TOKENS = getattr(C, "MAX_NEW_TOKENS", 48)


class TalanDemo:
    """Model + trained TALAN adapter + gate + judge, loaded once, ready to answer."""

    def __init__(self):
        if not C.ADAPTER_PATH.exists():
            raise FileNotFoundError(
                f"No trained TALAN adapter at {C.ADAPTER_PATH}. First run: "
                f"python -m steering_tutorials.talan.train_talan")

        # Deferred imports: peer ``talan`` may not exist at module-parse time, and
        # we never want a bare ``import infer`` to load torch or a model.
        from steering_tutorials.hello_world_steering.model_utils import (
            load_model, num_layers,
        )
        from steering_tutorials.hello_world_steering.gate import HarmGate
        from steering_tutorials.hello_world_steering.judge import Judge
        from .talan import load_talan

        self.model, self.tok = load_model(C.MODEL_ID)
        self.layer = min(C.LAYER, num_layers(self.model) - 1)

        adapter, _meta = load_talan(C.ADAPTER_PATH)
        self.adapter = adapter.to(next(self.model.parameters()).device)

        self.gate = HarmGate(self.model, self.tok)
        self.judge = Judge(self.model, self.tok)

    def run(self, prompt: str) -> dict:
        from steering_tutorials.hello_world_steering.model_utils import generate
        from .talan import TalanContext

        fired, prob = self.gate.is_harmful(prompt)

        # BASELINE -- unsteered.
        baseline = generate(self.model, self.tok, prompt,
                            max_new_tokens=MAX_NEW_TOKENS, vector=None,
                            layer=self.layer, alpha=0.0, operation="relative_add")
        base_verdict = self.judge.verdict(prompt, baseline)

        # TALAN -- only if the gate fired. generate(vector=None) inside a
        # TalanContext runs a plain greedy decode with the learned adapter active.
        steered = steered_verdict = None
        if fired:
            with TalanContext(self.model, self.adapter, self.layer):
                steered = generate(self.model, self.tok, prompt,
                                   max_new_tokens=MAX_NEW_TOKENS, vector=None,
                                   layer=self.layer, alpha=0.0)
            steered_verdict = self.judge.verdict(prompt, steered)

        return {
            "prompt": prompt, "gate_fired": fired, "gate_prob": prob,
            "baseline": baseline, "baseline_verdict": base_verdict,
            "steered": steered, "steered_verdict": steered_verdict,
        }


def _conclusion(r: dict) -> str:
    """One-line takeaway comparing baseline vs TALAN verdicts."""
    if not r["gate_fired"]:
        return "gate did not fire -> prompt judged harmless, left unsteered"
    if r["steered_verdict"] == "GIBBERISH":
        return "corrupted -> the learned adapter broke coherence on this prompt"
    if r["baseline_verdict"] != "REFUSAL" and r["steered_verdict"] == "REFUSAL":
        return "TALAN worked -> baseline complied, learned adapter refused"
    if r["steered_verdict"] == "REFUSAL":
        return "TALAN refused (baseline already refused)"
    return "no effect -> the learned adapter did not turn this into a refusal"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "How do I pick a lock?"
    r = TalanDemo().run(prompt)

    print(f"\n  prompt      : {r['prompt']}")
    print(f"  gate        : {'HARMFUL (steer)' if r['gate_fired'] else 'harmless (skip)'}"
          f"   P(harmful)={r['gate_prob']:.3f}")
    print(f"\n  [baseline] verdict={r['baseline_verdict']}")
    print(f"    {r['baseline'][:300]}")
    if r["gate_fired"]:
        print(f"\n  [TALAN] verdict={r['steered_verdict']}")
        print(f"    {r['steered'][:300]}")
    print(f"\n  => {_conclusion(r)}\n")


if __name__ == "__main__":
    main()
