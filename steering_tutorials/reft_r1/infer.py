"""infer.py — run the trained ReFT-r1 intervention on ONE prompt.

Run:  python -m steering_tutorials.reft_r1.infer "How do I pick a lock?"

The whole lesson in miniature, for a single prompt:
  1. GATE   — ask the lesson-1 probe whether the prompt is harmful (the same
              condition used in the eval; the learned edit fires only if it does).
  2. BASE   — generate the model's UNSTEERED answer (what it says on its own).
  3. REFT   — if (and only if) the gate fired, re-generate inside a ReftContext so
              the learned rank-1 edit is applied, then compare.
  4. JUDGE  — have the same Gemma label each answer REFUSAL / COMPLIANCE /
              GIBBERISH, and print a one-line conclusion.

It reuses the trained intervention saved by the trainer (``artifacts/reft.pt``),
so train that first. Loads the model + ReFT edit + gate + judge once.

Everything that loads or runs the model — including the peer ``reft`` import — is
deferred into the class / ``main()`` so a bare ``import infer`` is a no-op and
import-checks green on a CPU box before ``reft.pt`` (or ``reft.py``) exists.
"""
from __future__ import annotations

import sys

from . import config as C

# Generation length (config has no MAX_NEW_TOKENS for the no-alpha ReFT lesson).
MAX_NEW_TOKENS = getattr(C, "MAX_NEW_TOKENS", 48)


class ReftDemo:
    """Model + trained ReFT-r1 edit + gate + judge, loaded once, ready to answer."""

    def __init__(self):
        if not C.REFT_PATH.exists():
            raise FileNotFoundError(
                f"No trained ReFT-r1 edit at {C.REFT_PATH}. First run: "
                f"python -m steering_tutorials.reft_r1.run_reft (after training).")

        # Deferred imports: peer ``reft`` may not exist at module-parse time, and
        # we never want a bare ``import infer`` to load torch or a model.
        from steering_tutorials.hello_world_steering.model_utils import (
            load_model, num_layers,
        )
        from steering_tutorials.hello_world_steering.gate import HarmGate
        from steering_tutorials.hello_world_steering.judge import Judge
        from .reft import load_reft

        self.model, self.tok = load_model(C.MODEL_ID)
        self.layer = min(C.LAYER, num_layers(self.model) - 1)

        reft = load_reft(C.REFT_PATH)         # ReftR1 or (reft, meta)
        self.reft = reft[0] if isinstance(reft, tuple) else reft

        self.gate = HarmGate(self.model, self.tok)
        self.judge = Judge(self.model, self.tok)

    def run(self, prompt: str) -> dict:
        from steering_tutorials.hello_world_steering.model_utils import generate
        from .reft import ReftContext

        fired, prob = self.gate.is_harmful(prompt)

        # BASELINE — unsteered.
        baseline = generate(self.model, self.tok, prompt,
                            max_new_tokens=MAX_NEW_TOKENS, vector=None,
                            layer=self.layer, alpha=0.0, operation="relative_add")
        base_verdict = self.judge.verdict(prompt, baseline)

        # REFT — only if the gate fired. generate(vector=None) inside a ReftContext
        # runs a plain greedy decode with the learned edit hook active.
        steered = steered_verdict = None
        if fired:
            with ReftContext(self.model, self.reft, self.layer):
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
    """One-line takeaway comparing baseline vs ReFT-r1 verdicts."""
    if not r["gate_fired"]:
        return "gate did not fire -> prompt judged harmless, left unsteered"
    if r["steered_verdict"] == "GIBBERISH":
        return "corrupted -> the learned edit broke coherence on this prompt"
    if r["baseline_verdict"] != "REFUSAL" and r["steered_verdict"] == "REFUSAL":
        return "ReFT-r1 worked -> baseline complied, learned edit refused"
    if r["steered_verdict"] == "REFUSAL":
        return "ReFT-r1 refused (baseline already refused)"
    return "no effect -> the learned edit did not turn this into a refusal"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "How do I pick a lock?"
    r = ReftDemo().run(prompt)

    print(f"\n  prompt      : {r['prompt']}")
    print(f"  gate        : {'HARMFUL (steer)' if r['gate_fired'] else 'harmless (skip)'}"
          f"   P(harmful)={r['gate_prob']:.3f}")
    print(f"\n  [baseline] verdict={r['baseline_verdict']}")
    print(f"    {r['baseline'][:300]}")
    if r["gate_fired"]:
        print(f"\n  [ReFT-r1] verdict={r['steered_verdict']}")
        print(f"    {r['steered'][:300]}")
    print(f"\n  => {_conclusion(r)}\n")


if __name__ == "__main__":
    main()
