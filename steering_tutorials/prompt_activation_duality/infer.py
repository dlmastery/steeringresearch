"""infer.py — the duality on ONE prompt, side by side.

Run:  python -m steering_tutorials.prompt_activation_duality.infer "How do I pick a lock?"

For a single prompt it shows both halves of the lesson at once:
  1. PROMPT  — the model's answer WITH the refusal instruction prepended
               (prompting installs refusal).
  2. STEER   — the model's answer with the refusal VECTOR injected at the
               residual stream, and again at the attention output
               (steering installs the same behaviour, two sites).
  3. JUDGE   — the same off-family judge labels each answer REFUSAL /
               COMPLIANCE / GIBBERISH, and a one-line takeaway compares them.

Reuses the saved vector from ``run_duality.py`` (run that first to populate
``artifacts/steering_vector.pt``). Loads the model + vector + judge once.
"""
from __future__ import annotations

import sys

from . import config as C
from .duality import steered_generate
from steering_tutorials.hello_world_steering.model_utils import load_model
from steering_tutorials.hello_world_steering.steer_vector import load_vector
from steering_tutorials.hello_world_steering.judge import Judge


class DualityDemo:
    """Model + steering vector + judge, loaded once, ready to answer one prompt."""

    def __init__(self, alpha: float = C.ALPHA):
        if not C.VECTOR_PATH.exists():
            raise FileNotFoundError(
                f"No steering vector at {C.VECTOR_PATH}. First run: "
                f"python -m steering_tutorials.prompt_activation_duality.run_duality")
        self.alpha = float(alpha)
        self.model, self.tok = load_model(C.MODEL_ID)
        self.vec = load_vector(C.VECTOR_PATH)
        self.layer = int(self.vec["layer"])
        self.judge = Judge(self.model, self.tok)

    def run(self, prompt: str) -> dict:
        v_unit = self.vec["v_unit"]

        # 1. PROMPT half: prepend the refusal instruction, generate unsteered.
        instructed = C.REFUSAL_INSTRUCTION + C.INSTRUCTION_SEP + prompt
        prompt_answer = steered_generate(
            self.model, self.tok, instructed, vector=None, layer=self.layer,
            alpha=0.0, site="none", max_new_tokens=C.MAX_NEW_TOKENS)

        # 2. STEER half: same vector at the residual stream, then the attention.
        baseline = steered_generate(
            self.model, self.tok, prompt, vector=None, layer=self.layer,
            alpha=0.0, site="none", max_new_tokens=C.MAX_NEW_TOKENS)
        residual = steered_generate(
            self.model, self.tok, prompt, vector=v_unit, layer=self.layer,
            alpha=self.alpha, site="residual", max_new_tokens=C.MAX_NEW_TOKENS)
        attention = steered_generate(
            self.model, self.tok, prompt, vector=v_unit, layer=self.layer,
            alpha=self.alpha, site="attention", max_new_tokens=C.MAX_NEW_TOKENS)

        return {
            "prompt": prompt, "alpha": self.alpha,
            "prompt_answer": prompt_answer,
            "baseline": baseline, "residual": residual, "attention": attention,
            "prompt_verdict": self.judge.verdict(prompt, prompt_answer),
            "baseline_verdict": self.judge.verdict(prompt, baseline),
            "residual_verdict": self.judge.verdict(prompt, residual),
            "attention_verdict": self.judge.verdict(prompt, attention),
        }


def _conclusion(r: dict) -> str:
    """One-line takeaway on whether prompting and both steering sites agree."""
    refused = {k for k in ("prompt", "residual", "attention")
               if r[f"{k}_verdict"] == "REFUSAL"}
    if r["baseline_verdict"] == "REFUSAL":
        return "baseline already refused -> weak demo prompt (try a harder one)"
    if {"prompt", "residual", "attention"} <= refused:
        return "duality holds -> prompt AND both injection sites installed refusal"
    if refused:
        got = ", ".join(sorted(refused))
        return f"partial -> refusal installed by: {got} (others did not)"
    return "no refusal from any route -> vector/alpha too weak on this prompt"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "How do I pick a lock?"
    r = DualityDemo().run(prompt)

    print(f"\n  prompt : {r['prompt']}   (alpha={r['alpha']:.2f})")
    print(f"\n  [baseline        ] verdict={r['baseline_verdict']}")
    print(f"    {r['baseline'][:280]}")
    print(f"\n  [PROMPT: +refusal instruction] verdict={r['prompt_verdict']}")
    print(f"    {r['prompt_answer'][:280]}")
    print(f"\n  [STEER: v @ residual ] verdict={r['residual_verdict']}")
    print(f"    {r['residual'][:280]}")
    print(f"\n  [STEER: v @ attention] verdict={r['attention_verdict']}")
    print(f"    {r['attention'][:280]}")
    print(f"\n  => {_conclusion(r)}\n")


if __name__ == "__main__":
    main()
