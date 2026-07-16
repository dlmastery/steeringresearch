"""infer.py -- on ONE prompt, compare prompting vs. its steering-vector shadow.

Run:  python -m steering_tutorials.decomposing_prompting.infer "How do I pick a lock?"

The lesson in miniature for a single prompt:
  1. BASELINE  -- the bare prompt, no instruction, no steering.
  2. PROMPTING -- the same prompt WITH the refusal instruction prepended (the
                 strong AxBench baseline we are trying to explain).
  3. STEER     -- the bare prompt, steered with the on-direction translation
                 ``v_proj`` that ``run_decompose`` extracted from prompting's
                 average activation delta. If STEER lands where PROMPTING did,
                 prompting was -- for this prompt -- essentially that one vector.
  4. JUDGE     -- label each answer REFUSAL / COMPLIANCE / GIBBERISH and print a
                 one-line conclusion.

Reuses the reconstruction saved by ``run_decompose.py`` (run that first to
populate ``artifacts/reconstruction.pt``).
"""
from __future__ import annotations

import sys

from . import config as C
from steering_tutorials.hello_world_steering.model_utils import load_model, generate
from steering_tutorials.hello_world_steering.judge import Judge


class DecomposeDemo:
    """Model + saved translation vector + judge, loaded once, ready to answer."""

    def __init__(self):
        if not C.RECON_PATH.exists():
            raise FileNotFoundError(
                f"No reconstruction at {C.RECON_PATH}. First run: "
                f"python -m steering_tutorials.decomposing_prompting.run_decompose")
        import torch
        self.model, self.tok = load_model(C.MODEL_ID)
        recon = torch.load(C.RECON_PATH, map_location="cpu", weights_only=False)
        self.v_proj = recon["v_proj"]
        self.layer = int(recon["layer"])
        self.judge = Judge(self.model, self.tok)

    def run(self, prompt: str) -> dict:
        baseline = generate(self.model, self.tok, prompt,
                           max_new_tokens=C.MAX_NEW_TOKENS, alpha=0.0)
        prompting = generate(self.model, self.tok,
                            f"{C.REFUSAL_INSTRUCTION}\n\n{prompt}",
                            max_new_tokens=C.MAX_NEW_TOKENS, alpha=0.0)
        # Steer the BARE prompt with the extracted on-direction translation via
        # the literal "add" op (alpha=1.0 injects the raw vector as measured).
        steered = generate(self.model, self.tok, prompt,
                         max_new_tokens=C.MAX_NEW_TOKENS,
                         vector=self.v_proj, layer=self.layer,
                         alpha=1.0, operation="add")
        return {
            "prompt": prompt,
            "baseline": baseline, "baseline_verdict": self.judge.verdict(prompt, baseline),
            "prompting": prompting, "prompting_verdict": self.judge.verdict(prompt, prompting),
            "steered": steered, "steered_verdict": self.judge.verdict(prompt, steered),
        }


def _conclusion(r: dict) -> str:
    """One-line takeaway: did the steering vector reproduce prompting?"""
    pv, sv = r["prompting_verdict"], r["steered_verdict"]
    if sv == "GIBBERISH":
        return "steer overshot -> the raw translation broke coherence"
    if pv == "REFUSAL" and sv == "REFUSAL":
        return "reproduced -> steering vector recovered prompting's refusal"
    if pv == "REFUSAL" and sv != "REFUSAL":
        return "not reproduced -> prompting refused but the on-direction shift did not"
    return "prompting did not refuse here -> nothing to reproduce"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "How do I pick a lock?"
    r = DecomposeDemo().run(prompt)

    print(f"\n  prompt   : {r['prompt']}")
    print(f"\n  [baseline]  verdict={r['baseline_verdict']}")
    print(f"    {r['baseline'][:300]}")
    print(f"\n  [prompting] verdict={r['prompting_verdict']}")
    print(f"    {r['prompting'][:300]}")
    print(f"\n  [steer v_proj] verdict={r['steered_verdict']}")
    print(f"    {r['steered'][:300]}")
    print(f"\n  => {_conclusion(r)}\n")


if __name__ == "__main__":
    main()
