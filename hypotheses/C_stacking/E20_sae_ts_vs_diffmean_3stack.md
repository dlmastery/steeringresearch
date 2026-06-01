# E20 — SAE-TS vs DiffMean: 3-Stack Coherence

> **One-line claim:** SAE-TS targeting makes stacked vectors more orthogonal
> in feature space than raw activation space, improving multi-behavior coherence
> in a 3-vector stack.
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires GemmaScope SAE
> (accessible via HuggingFace) and the multi-vector injection infrastructure.

---

## 1. Motivation (>= 100 words)

DiffMean extracts behavior vectors in raw activation space, where two
semantically distinct behaviors may still share activation dimensions
(e.g., both activate similar MLP neurons in layers that handle sentiment-like
concepts). SAE-TS (arXiv:2411.02193) addresses this by first decomposing
activations into interpretable Sparse Autoencoder features, then targeting
only the features that are causally responsible for the target behavior while
suppressing side-effect features. The resulting vectors, expressed in SAE
feature space and projected back to activation space, are designed to have
minimal side-effect cross-contamination — which translates to higher
orthogonality between different behaviors' vectors. For a 3-vector stack,
this orthogonality improvement should reduce the Gram off-diagonal mass (E18)
and therefore the total behavioral interference. The AxBench tension (E36
corpus: SAEs underperform naïve DiffMean on AxBench) is relevant here: naïve
SAE steering is worse than DiffMean, but SAE-TS (with side-effect optimization)
may be better for MULTI-VECTOR stacking precisely because it enforces feature-
space orthogonality. This experiment tests the specific claim that SAE-TS
improves multi-behavior coherence (not single-behavior efficacy) by virtue of
feature-space orthogonality.

---

## 2. Formal Hypothesis (>= 50 words)

Because SAE-TS optimizes a steering vector to activate target SAE features
while minimizing activation of non-target features, two SAE-TS vectors for
different behaviors should have lower cross-feature activation overlap than
two DiffMean vectors for the same behaviors. This reduced feature-space overlap
corresponds to lower activation-space cosine between the SAE-TS vectors. In a
3-behavior stack, the lower Gram off-diagonal mass of SAE-TS vectors (relative
to DiffMean vectors) should produce lower behavioral interference (per E18's
monotone curve) and better joint coherence (PPL closer to solo PPL). Formal
claim: 3-stack coherence (1 - interference_index, defined in E18) is higher
for SAE-TS vectors than for DiffMean vectors at matched behavior efficacy,
with coherence gap >= 0.10 at 3-seed median on Gemma-2-2B-it.

---

## 3. Falsifier (>= 30 words)

If SAE-TS 3-stack coherence is not >= 0.10 higher than DiffMean 3-stack
coherence at matched behavior efficacy, the hypothesis is FALSIFIED —
SAE-TS's feature-space orthogonality does not translate to improved multi-
behavior stack performance. If SAE-TS achieves better orthogonality but lower
solo efficacy (the AxBench problem), the result is NEAR-MISS (orthogonality
improved but efficacy is the bottleneck). Status SUPPORTED only if BOTH
orthogonality is improved AND joint efficacy is higher.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Templeton, Adly, et al. 2024 arXiv 'Targeted Steering via SAE Side-Effect
Analysis' (arXiv:2411.02193, "SAE-TS") — the primary method under test;
introduces side-effect analysis of steering vectors in SAE feature space and
optimization of vectors to minimize side-effect feature activation; reported
to improve single-behavior steering coherence [NEEDS VERIFICATION on Gemma-2-2B].

Chen, Howard, et al. 2025 arXiv 'FGAA: Fine-Grained Activation Addition via
SAE Latent Space' (arXiv:2501.09929) — combines CAA + SAE-TS in SAE latent
space; reported to outperform CAA, SAE-decoder, and SAE-TS individually
[NEEDS VERIFICATION]; provides the upper-bound method that SAE-TS alone should
approach in the multi-vector setting.

[Steering-stackable-vs-competing-analysis.md §2.4, this project]: "SAE-TS
uses SAEs to measure side effects of any steering vector and optimize a
replacement that activates only the desired feature while suppressing others.
This is a 'make-it-stack-better' wrapper: it actively orthogonalizes a vector
against unwanted features so that it interferes less with other behaviors."

[N5 geometry result, C2, this project]: logPPL = 5.40 + 2.87 * offshell,
R² = 0.81 — reduced Gram mass (from SAE-TS) reduces joint norm; lower joint
norm reduces logPPL; N5 is the mechanistic path from orthogonality to coherence.
```

---

## 5. Mechanism

### 5.1 SAE-TS vector optimization

SAE-TS objective (approximate):

    maximize: activation of target SAE features F_target
    minimize: activation of non-target SAE features F_side_effect
    => v_SAE-TS = argmax [score(F_target) - lambda * score(F_side_effect)]

The side-effect minimization means v_SAE-TS has lower projection onto the
feature directions of other behaviors. Two SAE-TS vectors for behaviors B1 and B2:

    cos(v_B1_SAETS, v_B2_SAETS) << cos(v_B1_DiffMean, v_B2_DiffMean)

### 5.2 Expected Gram mass comparison

DiffMean 3-stack: Gram mass M_DM = sum_{i<j} |cos(v_i_DM, v_j_DM)|
SAE-TS 3-stack:   Gram mass M_SAETS = sum_{i<j} |cos(v_i_SAETS, v_j_SAETS)|

Prediction: M_SAETS < M_DM (SAE-TS vectors are more orthogonal)

By E18's monotone curve: lower M → lower interference → higher coherence.

### 5.3 GemmaScope SAE access

GemmaScope SAEs for Gemma-2-2B-it are available on HuggingFace
(google/gemma-scope-2b-pt-res). Feature extraction: run forward pass,
collect SAE activations per layer, identify top-activating features per
prompt, compare feature activation distributions for harmful vs harmless sets.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Gram mass M (DiffMean 3-stack) | 0.6-1.5 | 3 pairs at |cos| ~ 0.2-0.5 |
| Gram mass M (SAE-TS 3-stack) | 0.2-0.6 | SAE-TS reduces cross-feature overlap |
| 3-stack coherence gap (SAE-TS - DiffMean) | >= 0.10 | Core claim |
| Solo efficacy (SAE-TS vs DiffMean) | [−5%, +5%] | AxBench tension; expect near parity |
| PPL overhead (SAE-TS 3-stack vs solo) | [−0.2, +0.1] | N5: lower M → lower PPL |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; GemmaScope SAE at matching layers
- **Behaviors:** 3 behaviors with known pairwise cosines in DiffMean space
- **Conditions:** (1) 3-stack DiffMean, (2) 3-stack SAE-TS, (3) solo each
- **Metrics:** 3-stack coherence (1 - interference_index), Gram mass M,
  solo efficacy per behavior, PPL, offshell ||delta h||
- **Seeds:** 3 (screening); 7 if coherence gap is between 0.07-0.13
- **Wall-clock:** ~3 h on 4090 (SAE forward passes are more expensive)

### 7.2 Where it shines

SAE-TS shines most when DiffMean vectors have high cross-feature contamination
(behaviors that activate many of the same SAE features). For semantically very
distinct behaviors (already near-orthogonal in activation space), SAE-TS
provides minimal additional benefit.

---

## 8. Cross-References

- **E18** (interference vs Gram mass): M reduction from SAE-TS is the input;
  E18's curve maps M to expected coherence improvement
- **E19** (Gram-Schmidt): complementary approach (activation space vs feature space)
- **E36** (SAE selection problem): SAE-TS with side-effect filtering vs naive
  SAE steering; E20 uses SAE-TS (the curated version)
- **N5** (norm budget, SUPPORTED): Gram mass reduction → norm reduction → PPL
- **IDEA_TABLE.md** Block C row E20

---

## 9. Committee Q&A

**Q: AxBench shows SAEs underperform DiffMean. Why expect SAE-TS to help?**

> AxBench tests SINGLE-BEHAVIOR efficacy of naïve SAE steering vs DiffMean.
> SAE-TS with side-effect optimization is not naïve SAE steering — it is an
> optimized variant. More importantly, E20 tests MULTI-BEHAVIOR coherence,
> not single-behavior efficacy. SAE-TS may lose on single-behavior efficacy
> (the AxBench axis) while winning on multi-behavior coherence (the E20 axis),
> because the side-effect minimization is exactly what makes vectors less
> interfering in a stack. The AxBench result does not foreclose E20's prediction.

---

## 10. Verification Checklist

- [ ] GemmaScope SAE loaded; feature extraction validated on a test prompt
- [ ] SAE-TS optimization implemented for target behavior features
- [ ] Gram matrix M computed for DiffMean and SAE-TS 3-stacks
- [ ] 3-stack coherence computed for both; gap compared to 0.10 threshold
- [ ] Solo efficacy comparison (SAE-TS vs DiffMean) reported
- [ ] IDEA_TABLE.md row E20 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Geometry
  grounding: N5 (SUPPORTED) chains Gram mass reduction → norm reduction →
  PPL coherence improvement. Code needed: (a) GemmaScope SAE integration;
  (b) SAE-TS optimization objective; (c) multi-vector injection (same as E17).
  GemmaScope SAEs publicly available. Main technical barrier: SAE-TS optimization
  on 4090 VRAM budget.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C.*

### Prior plausibility

MEDIUM. The AxBench tension (SAEs often underperform DiffMean for single-behavior
steering) is a real challenge. SAE-TS's side-effect minimization is theoretically
motivated but may not improve multi-behavior coherence if the primary interference
source is not cross-feature contamination.

### Confounds

1. **SAE coverage:** GemmaScope SAEs may not capture all behaviorally relevant
   features; low SAE reconstruction quality limits the side-effect analysis.
2. **Efficacy matching:** comparing at "matched behavior efficacy" requires
   finding the alpha at which SAE-TS and DiffMean achieve equal solo efficacy,
   which may require a search.

### Verdict

**NOVEL+TESTABLE.** The multi-behavior coherence angle is genuinely new relative
to AxBench's single-behavior focus. The N5 grounding provides a mechanistic
prediction chain. Main risk: SAE coverage quality on Gemma-2-2B.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E20.md`.
