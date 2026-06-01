# Rung-3 escalation of N17/N5 — REAL WikiText-2, held-out tested (the rigor tier)

The first rung-3 EVALUATION attempt: real held-out English prose (WikiText-2,
40 passages), n=50 pooled (model × layer × α) points across gemma-3-270m AND
gemma-3-1b, relative_add steering, Spearman + 10k-bootstrap CI + a held-out
generalization test (fit the N5 law on 270m, predict 1b).

## Results
- **N17 (off-shell Δ‖h‖ predicts incoherence): SUPPORTED on real data.**
  Spearman(off-shell, log real-PPL) = **+0.585**, 95% bootstrap CI **[+0.353,
  +0.758]** (excludes 0), p = 8.1e-6, n=50. The monotone relationship — more
  off-shell displacement ⇒ higher real perplexity — holds across two model scales
  on genuine held-out text.
- **N5 (single universal collapse law `log PPL = a + b·off-shell`): FALSIFIED
  across scale.** Held-out R² (fit coefficients on 270m, predict 1b) = **−1.6**
  (worse than predicting the mean). The screening R²=0.81 (C2) was a WITHIN-POOL
  fit that mixed models; under a held-out cross-scale test the LAW'S COEFFICIENTS
  are model-specific (270m fit: slope 78.85, intercept 4.65 — does not transfer).

## Honest verdict + caveats
- **N17's monotone claim is the strongest, most defensible result in the program**
  — real data, two models, bootstrap CI excluding zero. BUT the 50 points are
  (layer × α) configurations, NOT independent iid seeds, so the bootstrap CI is a
  within-grid estimate; a fully external claim still wants independent replicates
  (e.g. multiple behaviors/prompt-sets) — see cross-behavior follow-up.
- **N5's universal-coefficient form is refuted across scale**: the relationship is
  directionally robust (N17) but quantitatively model-specific. This is precisely
  what rung-3 rigor (held-out validation) exists to catch — and what the within-pool
  screening R²=0.81 masked.
- This row is logged as a rung-3 EVALUATION ATTEMPT. N17 monotone is promotable to
  FINDINGS with the non-iid caveat; N5-universal is demoted to FALSIFIED-across-scale.
