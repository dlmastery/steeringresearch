"""biencoder_guard — a DUAL-TOWER (bi-encoder) safety guardrail built on
EmbeddingGemma-300M.

The production-guardrail member of the trajectory/safety-detection course. Where
the earlier lessons ask "is THIS trace an attack?", this lesson asks a different,
scale-driven question: given a LARGE, evolving policy taxonomy (dozens to
thousands of safety labels), how do you moderate content against ALL of them
cheaply?

The 2026 answer (GLiNER bi-encoder, arXiv:2602.18487; GLiNER Guard, arXiv:2605.05277)
is to DECOUPLE the two things you are comparing:

  content tower : embeds the user prompt / model response  -> one vector per text
  policy tower  : embeds each policy DESCRIPTION            -> one vector per label

and score compatibility by a cheap dot-product / cosine in the shared space. The
policy vectors do not depend on the text, so they are embedded ONCE and cached;
at inference you encode only the incoming text and match it against the cached
label bank. That is what makes moderation scale to a million labels at near
constant per-request cost -- and what lets you add a brand-new policy by embedding
its description alone (zero-shot), with no retraining.

This lesson reconstructs that pattern at laptop scale with EmbeddingGemma-300M as
the shared backbone, benchmarking the bi-encoder against a uni-encoder
(re-encode-per-label; accurate but does not scale) and a supervised trained head
(strong but cannot handle unseen policies) on a HARD, multi-dataset,
many-label safety corpus. See config.py for the design and README.md for the
walkthrough.
"""
