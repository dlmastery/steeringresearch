"""prompt_activation_duality — the DUALITY between prompting and steering.

A refusal *instruction* (a prompt) and a refusal *steering vector* (an activation
edit) are two ways to move the same behaviour. This lesson measures how related
their internal shifts are, and shows you can inject the same vector at the
ATTENTION output as well as at the residual stream.

Paper: Kang, Liu, Ma, Huang, Tan & Jiang, 2026, 'Prompt-Activation Duality:
Improving Activation Steering via Attention-Level Interventions'
(arXiv:2605.10664).
"""
