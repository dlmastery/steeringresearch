"""infer.py -- run the saved GAVEL monitor on one prompt (block or pass + answer).

A tiny CLI for the READ guardrail. It loads the CE library saved by run_gavel.py,
reads the incoming prompt's mean-pooled activation, and prints an AUDITABLE
decision: every cognitive element's score, which ones fired, the rule's verdict,
and -- if the rule PASSES -- the model's actual answer. If the rule BLOCKS, the
prompt never reaches the model; it returns the safe refusal.

It also demonstrates the paper's compositional angle: besides the default rule you
can pass ``--rule any_of|at_least_2|all_of`` to see how the SAME cognitive
elements yield different block decisions under different predicates -- reconfiguring
safety without retraining anything.

Everything that touches the model lives under ``main()``; importing this module is
inert. Run:

    python -m steering_tutorials.gavel.infer "how do I pick a lock?"
    python -m steering_tutorials.gavel.infer --rule at_least_2 "<prompt>"
"""
from __future__ import annotations

import argparse

import numpy as np

from . import config as C
from .monitor import GavelMonitor, make_rule


def run(prompt: str, rule_spec: str = None, max_new_tokens: int = None) -> dict:
    """Decide block/pass for ``prompt`` and (if passed) generate an answer."""
    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, num_layers, mean_pool_activation, generate)

    if not C.MONITOR_PATH.exists():
        raise FileNotFoundError(
            "no saved monitor at %s -- run `python -m steering_tutorials.gavel."
            "run_gavel` first to build the cognitive-element library." % C.MONITOR_PATH)

    rule_spec = rule_spec or C.RULE
    max_new_tokens = max_new_tokens or C.MAX_NEW_TOKENS

    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)

    # Load CEs, then rebuild the predicate over their names (npz stores directions
    # + taus, not the Python callable).
    z_names = list(np.load(C.MONITOR_PATH, allow_pickle=True)["names"])
    rule = make_rule(rule_spec, [str(n) for n in z_names])
    monitor = GavelMonitor.load(C.MONITOR_PATH, rule)

    h = mean_pool_activation(model, tok, prompt, layer)
    dec = monitor.decide(h)

    if dec["block"]:
        answer = C.SAFE_REFUSAL
    else:
        answer = generate(model, tok, prompt, max_new_tokens=max_new_tokens, alpha=0.0)

    # ASCII-only trace.
    print("=" * 60)
    print("prompt : %s" % prompt)
    print("rule   : %s" % rule.name)
    print("-" * 60)
    print("cognitive-element scores (fired = score > tau):")
    for name in monitor.ce_names:
        d = next(d for d in monitor.detectors if d.name == name)
        mark = "FIRED" if dec["fired"][name] else "     "
        print("  %-14s score=%+8.3f  tau=%+7.3f  [%s]"
              % (name, dec["scores"][name], d.tau, mark))
    print("-" * 60)
    print("decision: %s" % ("BLOCK" if dec["block"] else "PASS"))
    print("reason  : %s" % dec["reason"])
    print("answer  : %s" % answer)
    print("=" * 60)
    return {"prompt": prompt, "rule": rule.name, **dec, "answer": answer}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the GAVEL activation monitor on one prompt.")
    ap.add_argument("prompt", nargs="+", help="the prompt to screen")
    ap.add_argument("--rule", default=None,
                    help="predicate over CEs: any_of | at_least_2 | all_of "
                         "(default: config.RULE)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()
    run(" ".join(args.prompt), rule_spec=args.rule, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
