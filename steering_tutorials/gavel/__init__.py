"""Lesson -- GAVEL: rule-based safety through activation MONITORING (the READ guardrail).

Every steering lesson so far has *written* to the residual stream (add a refusal
vector, strip it, transplant it). GAVEL is the mirror image: it only *reads*. A
lightweight **monitor** inspects the activation of an incoming prompt, scores it
against a small library of interpretable "cognitive element" (CE) directions, and
a **rule** decides -- block with a safe refusal, or pass the prompt through to the
model. No weights change, no decoding is steered; safety is a runtime predicate
over activations that you can edit without retraining anything.

Paper (WebFetch-verified, arXiv:2601.19768, Jan 2026):
    Shir Rozenfeld, Rahul Pankajakshan, Itay Zloczower, Eyal Lenga, Gilad Gressel,
    Yisroel Mirsky. "GAVEL: Towards Rule-Based Safety Through Activation
    Monitoring." The paper represents activations as fine-grained, interpretable
    cognitive elements and composes predicate rules over them, so safeguards can
    be configured/updated without retraining the model or the detector.

Standalone and CPU-readable to import/unit-test; the live run needs the same small
Gemma-3-1B as lessons 1-3. See ``README.md``.
"""
