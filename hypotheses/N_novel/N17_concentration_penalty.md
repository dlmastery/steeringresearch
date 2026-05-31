# N17 — The Concentration Penalty (Off-Shell Displacement Predicts Incoherence)

> **One-line claim:** Off-shell displacement |delta(||h||)| predicts incoherence
> better than raw ||alpha*v||; norm-preserving steers beat norm-changing steers at
> equal angle change. Screening: Spearman(offshell, logPPL) = +0.71, Pearson = 0.90.
>
> **Primary axes:** A9 (metric/space), A12 (basis/superposition)
> **Status:** SUPPORTED (screening, n=1 — not an external claim; gate to rung 3)

---

## 1. Motivation (>= 100 words)

Why does activation steering degrade coherence? The simplest answer is "too much
alpha" — large magnitude perturbations cause problems. But this misses the geometric
structure of WHY large alpha causes problems. The concentration of measure result
(F-A from the missed-dimensions document) provides the answer: in high-dimensional
space, natural activations cluster on a thin spherical shell. Adding alpha*v moves
the activation OFF this shell (changing its radius), and the amount of shell-departure
(off-shell displacement) is the true driver of incoherence — not the magnitude of
alpha per se. Two vectors of the same magnitude but different orientations relative
to h will cause different off-shell displacements: a vector orthogonal to h (pure
tangential) causes zero off-shell displacement; a vector parallel to h (pure radial)
causes maximum off-shell displacement. Screening result S-6 provides strong initial
evidence: Spearman(offshell, logPPL) = +0.705, Pearson = 0.899 over 23 rows pooled
across 2 models and 8 layers, with the linear law log PPL = 5.40 + 2.87 * offshell
(R2=0.81). This makes off-shell displacement the single strongest predictor of
incoherence found in the program. S-2 corroborates it in the opposite direction:
on Qwen-0.5B, off-shell Δ||h|| rises monotonically with PPL. N17 promotes this
screening result to a generalization hypothesis: the relationship holds across new
models, new layers, and new behaviors not seen in the S-6 training set.

## 2. Formal Hypothesis (>= 50 words)

Let offshell(h, delta_h) = ||h + delta_h|| - ||h|| be the signed off-shell displacement.
Let logPPL be the log-perplexity of a continuation after steering. The claim is:

(A) Spearman(offshell, logPPL) >= 0.65 on a held-out dataset: Gemma-3-1B-it @L16,
    3 behaviors, alpha in {0.5, 1, 2, 4, 8}, seeds {0,1,2} (n=54 rows).

(B) The correlation between raw ||alpha*v|| and logPPL, after controlling for offshell,
    explains < 0.05 additional R2 (offshell subsumes the magnitude signal).

(C) Norm-preserving steering (selective 2D rotation, zero offshell by construction)
    achieves logPPL <= 1.05 * baseline_logPPL at matched behavior for all alpha in
    the tested range.

## 3. Falsifier (>= 30 words)

If Spearman(offshell, logPPL) < 0.50 on the held-out Gemma-3-1B dataset (new models,
new layers), claim (A) is FALSIFIED. If ||alpha*v|| adds >= 0.05 R2 after controlling
for offshell, claim (B) is FALSIFIED. Both failing = full FALSIFIED. Status moves to
`FALSIFIED` or `INCONCLUSIVE` accordingly.

## 4. Citations (Citation Rigor >= 80 words)

```
Gao et al. 2026. 'CRH' arXiv:2605.01844 (ICML 2026). The formal decomposition of
delta_h into radial (offshell) and angular components; CRH predicts that radial
governs ADDITION coherence — confirmed at R2=0.81 by S-6 (the N17 screening result).
N17 is the empirical generalization of CRH's radial claim beyond the training data.

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. Off-manifold displacement
is the central failure mode; offshell = delta(||h||) is the spherical approximation
to off-manifold displacement. N17 operationalizes this with a computable scalar metric.

Venkatesh & Kurapath 2026. 'Non-Identifiability' arXiv:2602.06801. The null space
of downstream readouts consists of directions that are behaviorally inert; many of these
null-space directions are also radially-oriented (perpendicular to the sphere), meaning
that radial displacements are BOTH behaviorally inert AND incoherence-inducing — a
double negative. N17 is the empirical evidence that the incoherence cost of radial
displacement is real and measurable.

SCREENING RESULT: S-6 from FINDINGS.md: Spearman(offshell, logPPL)=+0.705,
Pearson=0.899, logPPL=5.40+2.87*offshell, R2=0.81, n=23 rows,
2 models/8 layers (C2 campaign).
```

## 5. Mechanism

The mechanism has three steps:

1. Natural activations cluster on shell: at layer L, ||h|| ≈ r_L for all natural
   inputs (concentration of measure in R^d with d=d_model).

2. Additive steering changes ||h||: h' = h + alpha*v => ||h'|| = sqrt(||h||^2 + alpha^2
   ||v||^2 + 2*alpha*(h·v)). For the typical case where v has no special alignment
   with h, this changes ||h|| by delta_r ≈ alpha * (h·v) / ||h|| (first order).

3. LayerNorm saturation: the downstream LayerNorm normalizes by ||h_prenorm|| * scale;
   if ||h_prenorm|| deviates from r_L, the effective scale is wrong, causing saturation
   in subsequent attention softmax and MLP gelu/relu activations, which produces
   out-of-distribution probability distributions over tokens (high PPL).

The linear law logPPL = a + b*offshell follows from linearizing the LayerNorm
saturation: for small offshell, saturation effect ≈ b * offshell + O(offshell^2).
S-6's R2=0.81 suggests the linear approximation holds well over the tested range
(offshell 0 to 0.3).

Claim (C) mechanism: selective 2D rotation changes h by rotating in the {h, v_perp}
plane, which by construction keeps ||h_rot|| = ||h|| exactly. Therefore offshell=0,
and by the N17 law logPPL ≈ a + 0 = baseline_logPPL (with the remaining angular
cost from the angular component, but this is separately controlled).

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| Spearman(offshell, logPPL) on held-out 1B | >= 0.65 | S-6 found 0.71 on 270m+Qwen |
| Pearson(offshell, logPPL) | >= 0.80 | S-6 found 0.90 |
| R2 of logPPL ~ offshell | >= 0.70 | S-6 found 0.81 on training set |
| R2 added by ||alpha*v|| after controlling offshell | < 0.05 | Offshell subsumes magnitude |
| PPL of selective rotation vs baseline | <= 1.05 * baseline | Zero offshell guarantee |
| Slope b in held-out regression | 2.0 - 4.0 | S-6 found b=2.87; model variation expected |

## 7. Protocol

### 7.1 Primary experiment (held-out generalization)

- Model: Gemma-3-1B-it (held-out from S-6)
- Layers: {10, 14, 18, 22} (held-out from S-6's C2 layer set)
- Behaviors: 3 behaviors, each DiffMean from 50 pairs
- Alpha: {0.5, 1, 2, 4, 8} x ||h|| (relative alpha, E7 protocol)
- Operations: additive and selective 2D rotation
- Offshell measurement: (||h + alpha*v|| - ||h||) / ||h||
- logPPL: WikiText-103 next-token log-perplexity on 50 random continuations
- Regression: logPPL ~ offshell; logPPL ~ ||alpha*v||; logPPL ~ offshell + ||alpha*v||
- Seeds: 3 extraction x 3 evaluation = 9 cells per (behavior, alpha, layer)
- Total cells: 3 behaviors x 5 alpha x 4 layers x 9 cells = 540 rows
- Wall-clock: ~5 hours on RTX 4090

### 7.2 Where it shines

Real-time steering budget allocation: offshell is computable BEFORE generation (it only
requires the pre-generation hidden state h and the steering vector v). If the N17 law
holds, one can predict PPL from offshell before generating any text, enabling prospective
budget management without running the full generation.

## 8. Cross-References

- N5 (norm-budget): N5 is the multi-vector version of N17 (all alpha_i v_i combined);
  N17 is the single-vector version. They predict the same law from different angles.
- N16 (CRH radius/angle): N16 is the geometric explanation of N17's empirical law.
  N16's "radial governs addition" (R2=0.81 in S-7) IS the N17 claim.
- S-6 and S-7: the direct predecessor screening results
- N5 <-> N17 <-> E7: the three-way family of norm-related findings that mutually support
- N12 (capstone): the budget cap B in the unified operator is the N17 threshold;
  setting B at the threshold of the N17 law prevents PPL from exceeding the bound.
- IDEA_TABLE.md: N17 row, axes A9+A12

## 9. Committee Q&A

**Q: The S-6 regression was fit on 23 rows (training data). Reporting R2=0.81 on
the same 23 rows is an overfit R2. How do you prevent this?**

> N17's primary claim is the HELD-OUT Spearman on 540 new rows from a new model
> and new layers. The S-6 result is the screening prior; the hold-out test is the
> definitive claim. The falsifier (Spearman < 0.50) is set below S-6's 0.71 to
> allow for realistic generalization degradation.

**Q: The offshell metric depends on how the behavior vector v is chosen (magnitude
and direction). Shouldn't the metric be normalized by the vector magnitude?**

> offshell = (||h + alpha*v|| - ||h||) / ||h|| is already relative to ||h||, which
> scales with the model's activation magnitude. The additional dependence on ||v||
> is absorbed into alpha. Using relative alpha (alpha * ||h|| = displacement fraction)
> further normalizes by ||h||, making the law alpha-relative. E7's relative_add
> (supported in S-9) confirms that this normalization gives the cleanest cliff.

**Q: The concentration of measure argument assumes the natural activation distribution
is approximately spherically symmetric. Is this true for LLM hidden states?**

> It does not need to be exactly true; it needs the SPREAD in ||h|| to be smaller than
> the steering displacement alpha*||v||. S-6's linear fit R2=0.81 empirically confirms
> that the scalar offshell metric is sufficient to predict PPL across 2 models and
> 8 layers — which validates the assumption's practical adequacy without requiring
> exact spherical symmetry.

## 10. Verification Checklist

- [ ] Held-out model (Gemma-3-1B-it) not in S-6 training set
- [ ] Held-out layers (10, 14, 18, 22) not in S-6 training set
- [ ] Offshell computation verified against S-6 formula
- [ ] Three regressions fit: offshell-only, alpha*v-only, combined; R2 for all three
- [ ] Spearman and Pearson correlation with 90% CI computed and logged
- [ ] Claim (C): selective rotation PPL <= 1.05 * baseline verified for all alpha
- [ ] 540-row dataset logged in EXPERIMENT_LEDGER.md
- [ ] Slope b in held-out regression reported and compared to S-6's b=2.87
- [ ] Status promoted from SUPPORTED(scr) to SUPPORTED(rung-2) after n>=7 protocol

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: SUPPORTED (screening, n=1).
  S-6: Spearman=+0.705, Pearson=0.899, R2=0.81 on 23 rows (Gemma-270m + Qwen-0.5B,
  8 layers, alpha 1-24), C2 campaign. S-9 corroborates via relative_add (E7) showing
  cliff at 10% displacement. S-2 confirms monotone offshell-PPL on Qwen-0.5B.
  Full held-out generalization on Gemma-3-1B is the next required step.
  The strongest single empirical result in the program; promoted to held-out test first.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

HIGH. The concentration of measure argument is a mathematical theorem. The LayerNorm
saturation mechanism is documented and plausible. The linear law R2=0.81 on 23 rows
is strong for a screening result. This is the most theoretically grounded empirical
claim in the program.

### Mechanism scrutiny

The linear relationship logPPL ~ offshell requires that LayerNorm saturation is
approximately linear in the norm deviation. This is only true for small deviations;
for large offshell (> 0.5), the relationship should become nonlinear (exponential or
step-function like). The S-6 data does not test offshell > 0.3; the linear law may
break down at extreme values. The held-out protocol should extend alpha to test
offshell up to 0.5 to characterize the nonlinear regime.

### Confounds

Alpha confound: offshell and alpha are correlated by construction (offshell ≈ alpha *
cos(v,h)/||h||). The regression logPPL ~ offshell + alpha_relative tests whether
offshell adds predictive power BEYOND alpha. S-6 did not run this three-way
regression; it is required for the held-out protocol.

### Does the off-shell metric specifically matter?

YES — more than almost any other claim in the program. If off-shell displacement
is a universal coherence predictor, it enables: (1) prospective PPL prediction
(before generation), (2) principled norm-cap setting (N5/E22), (3) justification
for selective rotation over additive (N16/E27 successor). The practical value is
very high if the generalization holds.

### Literature precedent

Zou et al. 2023 (Representation Engineering) observe that large alpha degrades quality;
they use this empirically without the geometric framing. The CRH paper (arXiv:2605.01844)
provides the formal decomposition. N17 is the first empirical law connecting the
geometric quantity (offshell) to a standard NLP metric (PPL), which is the program's
specific contribution.

### Skeptical effect-size estimate

Held-out Spearman: 0.55-0.70 (vs claimed >= 0.65). Main risk: layer-specific
variation in the slope b may reduce the pooled correlation on a new model.
Pearson: 0.75-0.88 (vs S-6's 0.90). The linear law holds but with more variance
on a 1B model where activations are more complex. Still strongly predictive and
practically useful even at Spearman=0.55.

### Minimum distinguishing experiment

3 alpha values (1, 2, 4 x ||h||), one behavior, one layer (L16), one model (1B),
3 seeds = 9 rows: compute offshell and logPPL; report Spearman and R2. Cost ~45 min.
If Spearman > 0.60 on 9 rows, proceed to full 540-row protocol.

### Verdict

SUPPORTED(screening) — HELD-OUT REPLICATION REQUIRED. This is the highest-priority
held-out generalization test in the program. The alpha-confound three-way regression
must be added to the protocol. The minimum experiment (9 rows, 45 min) is the first
required step before committing to the full 5-hour protocol.
