"""infer.py — steer ONE prompt three ways: baseline vs straight chord vs curved arc.

Run:
    python -m steering_tutorials.curveball.infer "How do I pick a lock?"
    python -m steering_tutorials.curveball.infer "..." --alpha 0.8 --steps 12

The whole lesson in miniature, for a single prompt:
  1. VECTOR — load (or build) lesson 2's diff-of-means refusal direction.
  2. BASE   — generate the model's plain, unsteered answer.
  3. STRAIGHT — re-generate with the lesson-2 chord (h + alpha*||h||*v_unit).
  4. CURVED — re-generate with the great-circle arc (rotate h toward v by alpha).
  5. JUDGE  — label each answer REFUSAL / COMPLIANCE / GIBBERISH and conclude.

Both steered arms use the SAME alpha, so the only difference is the path geometry.
The interesting outcome is a prompt where the straight chord tips into GIBBERISH
while the curved arc reaches a coherent REFUSAL at the same budget.

Everything that touches the model lives inside :class:`CurveballDemo`, so a bare
``import infer`` never triggers a model download (safe for tests / a webapp).
"""
from __future__ import annotations

import argparse

from . import config as C


class CurveballDemo:
    """Model + refusal direction + judge, loaded once, ready to answer a prompt."""

    def __init__(self, alpha: float | None = None, steps: int | None = None):
        # Heavy imports deferred to construction so ``import infer`` stays torch-free.
        from steering_tutorials.common.data import load_harmful_benign
        from steering_tutorials.hello_world_steering.judge import Judge
        from steering_tutorials.hello_world_steering.model_utils import (
            load_model, num_layers,
        )
        from steering_tutorials.hello_world_steering.steer_vector import (
            extract_caa_vector, load_vector, save_vector,
        )
        from .curveball import curveball_generate

        self.alpha = float(C.ALPHA if alpha is None else alpha)
        self.n_steps = int(C.N_CURVE_STEPS if steps is None else steps)
        self._generate = curveball_generate

        self.model, self.tok = load_model(C.MODEL_ID)
        self.layer = min(C.LAYER, num_layers(self.model) - 1)
        self.judge = Judge(self.model, self.tok)

        # Reuse a cached direction if present; otherwise build it once from the
        # extract split of the shared dataset and cache it for next time.
        if C.VECTOR_PATH.exists():
            self.v_unit = load_vector(C.VECTOR_PATH)["v_unit"]
        else:
            data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
            ex_harm = data["harmful"][:C.N_EXTRACT_PER_CLASS]
            ex_ben = data["benign"][:C.N_EXTRACT_PER_CLASS]
            vec = extract_caa_vector(self.model, self.tok, ex_harm, ex_ben, self.layer)
            save_vector(C.VECTOR_PATH, vec)
            self.v_unit = vec["v_unit"]

    def run(self, prompt: str) -> dict:
        base = self._generate(self.model, self.tok, prompt, None, self.layer, 0.0,
                              max_new_tokens=C.MAX_NEW_TOKENS)
        straight = self._generate(self.model, self.tok, prompt, self.v_unit,
                                  self.layer, self.alpha, curved=False,
                                  max_new_tokens=C.MAX_NEW_TOKENS)
        curved = self._generate(self.model, self.tok, prompt, self.v_unit,
                                self.layer, self.alpha, curved=True,
                                n_steps=self.n_steps, max_new_tokens=C.MAX_NEW_TOKENS)
        return {
            "prompt": prompt, "alpha": self.alpha, "n_steps": self.n_steps,
            "baseline": base, "baseline_verdict": self.judge.verdict(prompt, base),
            "straight": straight,
            "straight_verdict": self.judge.verdict(prompt, straight),
            "curved": curved, "curved_verdict": self.judge.verdict(prompt, curved),
        }


def _conclusion(r: dict) -> str:
    """One-line takeaway comparing the straight chord vs the curved arc."""
    sv, cv = r["straight_verdict"], r["curved_verdict"]
    if sv == "GIBBERISH" and cv == "REFUSAL":
        return "curveball wins -> straight chord broke into gibberish; curved arc refused"
    if sv == "REFUSAL" and cv == "REFUSAL":
        return "both refused -> at this budget the chord stayed coherent too"
    if cv == "GIBBERISH":
        return "curved arc broke -> lower alpha or raise --steps"
    if cv == "REFUSAL":
        return "curved arc refused (straight did not)"
    return "no refusal installed -> raise --alpha"


def main() -> None:
    ap = argparse.ArgumentParser(description="Curveball single-prompt demo: baseline "
                                            "vs straight chord vs curved arc.")
    ap.add_argument("prompt", nargs="*", help="the user prompt to steer on")
    ap.add_argument("--alpha", type=float, default=None,
                    help="shared steering budget (default: config.ALPHA)")
    ap.add_argument("--steps", type=int, default=None,
                    help="great-circle sub-steps for the curved arm (default: config.N_CURVE_STEPS)")
    args = ap.parse_args()

    prompt = " ".join(args.prompt) or "How do I pick a lock?"
    r = CurveballDemo(alpha=args.alpha, steps=args.steps).run(prompt)

    print(f"\n  prompt   : {r['prompt']}")
    print(f"  budget   : alpha={r['alpha']:.2f}   curve steps={r['n_steps']}")
    print(f"\n  [baseline] verdict={r['baseline_verdict']}")
    print(f"    {r['baseline'][:300]}")
    print(f"\n  [straight chord] verdict={r['straight_verdict']}")
    print(f"    {r['straight'][:300]}")
    print(f"\n  [curved arc] verdict={r['curved_verdict']}")
    print(f"    {r['curved'][:300]}")
    print(f"\n  => {_conclusion(r)}\n")


if __name__ == "__main__":
    main()
