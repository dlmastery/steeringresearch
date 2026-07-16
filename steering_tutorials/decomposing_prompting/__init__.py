"""decomposing_prompting -- lesson: WHY is prompting such a strong steering baseline?

We take the activation change that a *refusal instruction* induces per prompt and
decompose it into (a) the part that lies along the diff-of-means refusal
direction (a steering-vector-like shift), (b) the off-direction residual, and
(c) how consistent that change is across prompts. This explains how much of
prompting reduces to "add one fixed vector" versus a richer, input-dependent
transform.

Inspired by Cheng & Kriegeskorte 2026 (arXiv:2606.03093).
"""
