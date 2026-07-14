"""infer.py — run FLAS on ONE prompt: baseline vs flow-steered, side by side.

Run:
    python -m steering_tutorials.flas.infer "How do I pick a lock?"
    python -m steering_tutorials.flas.infer "..." --concept refusal --T 1.5

The whole lesson in miniature, for a single prompt:
  1. GATE   — ask the lesson-1 probe whether the prompt is harmful (informative:
              the flow is what STEERS; the gate is what would CONDITION it).
  2. BASE   — generate the model's plain, unsteered answer.
  3. FLOW   — build (or look up) the chosen concept's embedding, then re-generate
              inside a FlowContext that integrates the trained velocity field to
              flow-time ``T`` on the residual stream. Larger ``T`` transports the
              activation further along the SAME learned trajectory.
  4. JUDGE  — have the same Gemma label each answer REFUSAL / COMPLIANCE /
              GIBBERISH, and print a one-line conclusion.

It reuses the TRAINED field from ``flow.pt`` (train it first) and the lesson-1
probe as the gate. Model + field + gate + judge are all loaded once. Everything
that touches the model lives inside :class:`FlowDemo`, so a bare ``import infer``
never triggers a model download (safe for tests / the webapp).
"""
from __future__ import annotations

import argparse
import sys

from . import config as C


class FlowDemo:
    """Model + trained velocity field + gate + judge, loaded once, ready to answer."""

    def __init__(self, T: float | None = None):
        # Heavy imports are deferred to construction (not module import) so that
        # ``import infer`` stays torch-free. The flow + data modules are peer-owned;
        # the model plumbing / judge / gate are reused verbatim from lesson 2.
        from ..hello_world_steering.model_utils import load_model, generate, num_layers
        from ..hello_world_steering.judge import Judge
        from ..hello_world_steering.gate import HarmGate
        from .flow import concept_embedding, FlowContext, load_flow
        from .data import load_concepts

        if not C.FLOW_PATH.exists():
            raise FileNotFoundError(
                f"No trained flow at {C.FLOW_PATH}. Train it first with the FLAS "
                f"trainer (e.g. python -m steering_tutorials.flas.train_flow).")

        self.T = float(C.T_DEFAULT if T is None else T)
        # Stash the reused callables so run() reads cleanly.
        self._generate = generate
        self._concept_embedding = concept_embedding
        self._FlowContext = FlowContext

        self.model, self.tok = load_model(C.MODEL_ID)
        self.layer = min(C.LAYER, num_layers(self.model) - 1)

        loaded = load_flow(C.FLOW_PATH)
        self.vfield = loaded[0] if isinstance(loaded, tuple) else loaded

        self.gate = HarmGate(self.model, self.tok)
        self.judge = Judge(self.model, self.tok)

        # The concept pool: normalise whatever load_concepts() returns into a flat
        # {name: exemplars} lookup spanning TRAIN concepts + the held-out one, so
        # --concept can name any of them (and we can build its embedding on demand).
        self._exemplars = self._flatten_concepts(load_concepts())

    # -- concept resolution ------------------------------------------------
    @staticmethod
    def _flatten_concepts(raw: dict) -> dict[str, list[str]]:
        """Flatten load_concepts() into ``{concept_name: [exemplar prompts]}``.

        Tolerates the same key-naming variants as run_flas._normalize_concepts:
        train concepts under ``train``/``concepts``, exemplars under
        ``exemplars``/``extract``/``train``, and an optional held-out concept.
        """
        def exemplars_of(split: dict) -> list[str]:
            for k in ("exemplars", "extract", "train"):
                if isinstance(split, dict) and split.get(k):
                    return list(split[k])
            return []

        out: dict[str, list[str]] = {}
        train = raw.get("train") or raw.get("concepts") or {}
        for name, split in train.items():
            out[str(name)] = exemplars_of(split)

        held = raw.get("held_out") or raw.get("heldout") or raw.get("zero_shot")
        if isinstance(held, dict):
            name = held.get("name") or held.get("concept") or "held_out"
            out[str(name)] = exemplars_of(held)
        return out

    def _resolve_concept(self, requested: str) -> tuple[str, list[str]]:
        """Map a requested concept name to (name, exemplars), forgivingly.

        Exact match first, then case-insensitive substring (so ``refusal`` finds a
        ``Refusal/...`` category), else fall back to the first concept with a loud
        note — a demo should still run rather than crash on a typo.
        """
        if requested in self._exemplars:
            return requested, self._exemplars[requested]
        req = requested.lower()
        for name, ex in self._exemplars.items():
            if req in name.lower():
                return name, ex
        first = next(iter(self._exemplars))
        print(f"[warn] concept {requested!r} not found; using {first!r} instead. "
              f"Available: {sorted(self._exemplars)}", file=sys.stderr)
        return first, self._exemplars[first]

    # -- the pipeline ------------------------------------------------------
    def run(self, prompt: str, concept: str) -> dict:
        name, exemplars = self._resolve_concept(concept)
        if not exemplars:
            raise ValueError(f"concept {name!r} has no exemplars to embed")

        fired, prob = self.gate.is_harmful(prompt)
        concept_vec = self._concept_embedding(self.model, self.tok, exemplars,
                                              self.layer)

        # Baseline: plain generation, no flow.
        baseline = self._generate(self.model, self.tok, prompt,
                                  max_new_tokens=getattr(C, "MAX_NEW_TOKENS", 48))
        base_verdict = self.judge.verdict(prompt, baseline)

        # Flow-steered: integrate the trained field to flow-time T at C.LAYER.
        with self._FlowContext(self.model, self.vfield, concept_vec, self.layer,
                               T=self.T):
            steered = self._generate(self.model, self.tok, prompt,
                                    max_new_tokens=getattr(C, "MAX_NEW_TOKENS", 48))
        steered_verdict = self.judge.verdict(prompt, steered)

        return {
            "prompt": prompt, "concept": name, "T": self.T,
            "gate_fired": fired, "gate_prob": prob,
            "baseline": baseline, "baseline_verdict": base_verdict,
            "steered": steered, "steered_verdict": steered_verdict,
        }


def _conclusion(r: dict) -> str:
    """One-line takeaway comparing baseline vs flow-steered verdicts."""
    if r["steered_verdict"] == "GIBBERISH":
        return "corrupted -> flow-time T too large; integrate to a smaller T"
    if r["baseline_verdict"] != "REFUSAL" and r["steered_verdict"] == "REFUSAL":
        return "flow worked -> baseline complied, flow-steered refused"
    if r["steered_verdict"] == "REFUSAL":
        return "flow-steered refused (baseline already refused)"
    return "no effect -> flow-steered output did not become a refusal (raise T?)"


def main() -> None:
    ap = argparse.ArgumentParser(description="FLAS single-prompt demo: baseline "
                                            "vs flow-steered generation.")
    ap.add_argument("prompt", nargs="*", help="the user prompt to steer on")
    ap.add_argument("--concept", default="refusal",
                    help="which concept to steer toward (name or substring)")
    ap.add_argument("--T", type=float, default=None,
                    help="flow-time / strength dial (default: config.T_DEFAULT)")
    args = ap.parse_args()

    prompt = " ".join(args.prompt) or "How do I pick a lock?"
    r = FlowDemo(T=args.T).run(prompt, args.concept)

    print(f"\n  prompt      : {r['prompt']}")
    print(f"  concept     : {r['concept']}   flow-time T={r['T']:.2f}")
    print(f"  gate        : {'HARMFUL' if r['gate_fired'] else 'harmless'}"
          f"   P(harmful)={r['gate_prob']:.3f}")
    print(f"\n  [baseline] verdict={r['baseline_verdict']}")
    print(f"    {r['baseline'][:300]}")
    print(f"\n  [flow-steered T={r['T']:.2f}] verdict={r['steered_verdict']}")
    print(f"    {r['steered'][:300]}")
    print(f"\n  => {_conclusion(r)}\n")


if __name__ == "__main__":
    main()
