"""infer.py — contextual steering on ONE prompt: show the projection AND the α.

Run:  python -m steering_tutorials.contextual_steering.infer "How do I pick a lock?"

The whole lesson in miniature for a single prompt:
  1. READ   — mean-pool the prompt's layer-12 hidden state and project it onto
              the refusal direction v_unit (the context signal).
  2. SCHEDULE— turn that projection into a per-prompt α via the calibrated ramp
              (benign prompts land below tau ⇒ α≈0, harmful ones ramp up).
  3. GENERATE— produce the FIXED-α answer and the CONTEXTUAL-α answer.
  4. JUDGE   — label each REFUSAL / COMPLIANCE / GIBBERISH and print a takeaway.

It reuses the saved vector + schedule from ``run_contextual.py`` (run that first
to populate ``artifacts/steering_vector.pt`` and ``artifacts/results.json``).
Loads the model + vector + judge once.
"""
from __future__ import annotations

import json
import sys

from . import config as C
from .contextual import ContextualSteerer


def _load_schedule() -> dict:
    """Read tau/ref/cap from the saved results.json, or fall back to config."""
    if C.RESULTS_PATH.exists():
        try:
            sched = json.loads(C.RESULTS_PATH.read_text(encoding="utf-8"))["schedule"]
            return {"tau": float(sched["tau"]), "ref": float(sched["ref"]),
                    "cap": float(sched.get("cap", C.CAP_MULT))}
        except Exception:  # pragma: no cover - malformed/partial results
            pass
    return {"tau": C.TAU_FALLBACK, "ref": C.REF_FALLBACK, "cap": C.CAP_MULT}


class ContextualDemo:
    """Model + steering vector + calibrated schedule + judge, loaded once."""

    def __init__(self) -> None:
        if not C.VECTOR_PATH.exists():
            raise FileNotFoundError(
                f"No steering vector at {C.VECTOR_PATH}. First run: "
                f"python -m steering_tutorials.contextual_steering.run_contextual")
        from steering_tutorials.hello_world_steering.model_utils import load_model
        from steering_tutorials.hello_world_steering.steer_vector import load_vector
        from steering_tutorials.hello_world_steering.judge import Judge

        self.model, self.tok = load_model(C.MODEL_ID)
        self.vec = load_vector(C.VECTOR_PATH)
        self.layer = int(self.vec["layer"])
        s = _load_schedule()
        self.steerer = ContextualSteerer(
            self.model, self.tok, self.vec["v_unit"], self.layer,
            alpha_base=C.ALPHA_BASE, tau=s["tau"], ref=s["ref"], cap=s["cap"])
        self.judge = Judge(self.model, self.tok)

    def run(self, prompt: str) -> dict:
        proj = self.steerer.projection(prompt)
        ctx_alpha = self.steerer.alpha_for(proj)

        fixed = self.steerer.fixed_generate(prompt, C.MAX_NEW_TOKENS)
        # Reuse the projection we already computed (avoid a second forward pass).
        ctx_resp = self.steerer._generate(prompt, ctx_alpha, C.MAX_NEW_TOKENS)

        return {
            "prompt": prompt, "proj": proj,
            "fixed_alpha": fixed["alpha"], "fixed": fixed["response"],
            "fixed_verdict": self.judge.verdict(prompt, fixed["response"]),
            "ctx_alpha": ctx_alpha, "contextual": ctx_resp,
            "ctx_verdict": self.judge.verdict(prompt, ctx_resp),
        }


def _conclusion(r: dict) -> str:
    """One-line takeaway comparing the fixed and contextual outcomes."""
    steered = r["ctx_alpha"] > 1e-6
    if not steered:
        return ("projection below tau -> judged benign, contextual alpha ~0 "
                "(left ~unsteered while fixed still pushed)")
    if r["ctx_verdict"] == "GIBBERISH":
        return "contextual alpha broke coherence -> lower alpha_base or raise tau"
    if r["ctx_verdict"] == "REFUSAL":
        return "contextual steering fired -> refused a harmful-looking prompt"
    return "contextual α applied but output was not a refusal (no effect)"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "How do I pick a lock?"
    r = ContextualDemo().run(prompt)

    print(f"\n  prompt        : {r['prompt']}")
    print(f"  projection    : {r['proj']:+.3f}   (cos with the refusal direction)")
    print(f"  fixed alpha   : {r['fixed_alpha']:.3f}    contextual alpha : {r['ctx_alpha']:.3f}")
    print(f"\n  [fixed  a={r['fixed_alpha']:.2f}] verdict={r['fixed_verdict']}")
    print(f"    {r['fixed'][:300]}")
    print(f"\n  [context a={r['ctx_alpha']:.2f}] verdict={r['ctx_verdict']}")
    print(f"    {r['contextual'][:300]}")
    print(f"\n  => {_conclusion(r)}\n")


if __name__ == "__main__":
    main()
