"""safety_probe — a standalone hello-world activation-probe safety classifier.

Read a frozen LLM's internal activations for a prompt, and let a tiny 3-layer
MLP decide whether the prompt is harmful. Self-contained: model loading,
dataset, training, inference, and the demo webapp all live in this package.
"""
