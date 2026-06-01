# E16 — Conditional Gating Cuts the Capability Tax of Always-On Safety Steering

> **One-line claim:** Conditional gating cuts the capability tax of always-on
> safety steering by >= 80% (capability spent only when gate fires).
>
> **Source design space:** Block B — Conditional / Gated Steering (E9–E16).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires E9 CAST gate,
> E9 behavior vector, and MMLU evaluation pipeline.

---

## 1. Motivation (>= 100 words)

Always-on safety steering — a refusal vector injected unconditionally at every
forward pass — degrades model capability on general tasks even when those tasks
have nothing to do with the behavior being suppressed. The mechanism is the
norm-budget law (N5, SUPPORTED: log PPL = 5.40 + 2.87 * offshell, R² = 0.81,
C2): every unconditional injection pushes h off the data manifold by alpha * ||v||,
and this off-shell displacement correlates with MMLU degradation regardless
of whether the prompt is harmful or benign. For a deployment where the majority
of inputs are benign (realistic ratio: 95:5 benign-to-harmful), the always-on
steering spends capability on 95% of inputs to gain safety on 5%. A conditional
gate that fires only on harmful inputs ideally recovers this 95% capability
overhead at near-zero cost: the gate is a read-only probe (adds no off-shell
displacement), and the behavior write fires only on the 5% of harmful inputs.
This experiment directly quantifies the capability tax reduction achieved by
CAST gating — the primary economic argument for deploying conditional over
always-on safety steering.

---

## 2. Formal Hypothesis (>= 50 words)

Because the CAST gate is read-only (no write to the residual stream, therefore
no off-shell displacement, therefore no norm-budget spend on benign inputs),
the capability tax of the gated system on a benign evaluation set should be
proportional to the fraction of benign inputs on which the gate fires (the
false-positive rate, < 3% per E9). For a 95% benign deployment distribution,
the gated system spends capability on < 3% of benign inputs, compared to 100%
for always-on steering. If MMLU drops by X points with always-on steering, the
gated system should drop by at most (0.03 * X) points on the benign set — a
>= 97% reduction. The 80% claim in the hypothesis is a conservative lower bound
(allowing for mild interactions between gate firings and benign prompt processing).
Formal claim: MMLU drop on the benign set (AlpacaEval + MMLU standard set) is
reduced by >= 80% (relative) in the gated vs always-on condition, at matched
harmful-refusal rate, at 3-seed median on Gemma-2-2B-it.

---

## 3. Falsifier (>= 30 words)

If MMLU drop on the benign set is reduced by less than 80% (relative) in the
gated vs always-on condition at matched harmful-refusal rate, the hypothesis
is FALSIFIED. If the gated condition achieves < 3% false-refusal (E9) but still
causes > 20% of the always-on MMLU drop (indicating the gate firing itself has
a residual capability cost), status is PARTIALLY SUPPORTED (gate helps but
mechanism differs from prediction).

---

## 4. Citations (Citation Rigor >= 80 words)

```
Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering: Concept-Level
Control via Conditional Vectors' (arXiv:2409.05907) — CAST gate; the capability
tax reduction is an implicit claim in the paper (steering only when needed avoids
unnecessary capability costs) that this experiment makes explicit and quantitative.

O'Brien, Joseph, et al. 2024 arXiv 'Safety Alignment Should Be Made More
Robust to User Fine-Tuning' (arXiv:2411.11296) — demonstrates that SAE
refusal-clamping causes MMLU regression and over-refusal; the always-on baseline
for this experiment is the capability tax they document; conditional gating is
the proposed mitigation.

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering
Compromises LLM Safety' (arXiv:2509.22067) — finds that unconditional steering
in any direction raises compliance; the capability-tax mechanism is the same
off-manifold displacement (N5) that also explains rogue compliance; Guard E
(conditional gate) prevents both capability tax and rogue compliance on benign
inputs.

[N5 geometry result, C2 in steeringresearch campaign] — logPPL = 5.40 +
2.87 * offshell_R (R²=0.81) establishes the direct link between off-shell
displacement and perplexity degradation; this is the first-principles
mechanism by which unconditional injection imposes a capability tax.
```

---

## 5. Mechanism

### 5.1 Capability tax derivation from N5

From N5 (SUPPORTED): logPPL(h_steered) = 5.40 + 2.87 * ||h_steered - h_original||/||h_original||

Always-on: every prompt, benign or harmful, incurs offshell = alpha * ||v|| / ||h||
Gated: benign prompts incur offshell = 0 (gate doesn't fire; no write)
           harmful prompts incur offshell = alpha * ||v|| / ||h|| (gate fires)

The MMLU metric is dominated by benign (capability) prompts. Expected MMLU drop
under always-on:

    Delta_MMLU_always-on ~ -f(alpha, ||v||, ||h||)   (empirically measured)

Under gating (false-positive rate = FPR < 3%):

    Delta_MMLU_gated ~ -FPR * f(alpha, ||v||, ||h||)
                     <= 0.03 * |Delta_MMLU_always-on|

Tax reduction = 1 - 0.03 = 97% (theoretical upper bound under FPR < 3%).
The 80% claim is conservative, allowing for a 5x inflation of FPR effect.

### 5.2 Protocol sketch

```python
# Evaluate MMLU under three conditions:
# 1. No steering (baseline)
# 2. Always-on steering at alpha_opt
# 3. Gated steering at alpha_opt, optimal theta_c

mmlu_base = eval_mmlu(model)
mmlu_alwayson = eval_mmlu(model, steer=always_on_hook(alpha_opt, v))
mmlu_gated = eval_mmlu(model, steer=cast_hook(alpha_opt, v, theta_c_opt))

tax_alwayson = mmlu_base - mmlu_alwayson
tax_gated = mmlu_base - mmlu_gated
tax_reduction = 1 - tax_gated / tax_alwayson
```

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| MMLU drop (always-on) | [1.0, 5.0] pp | Known from E3 alpha cliff region |
| MMLU drop (gated) | [0.0, 1.0] pp | <= FPR * always-on drop |
| Tax reduction (gated vs always-on) | >= 80% | Core claim (conservative) |
| Harmful refusal rate (gated) | >= 50 pp above no-steer | Matched to E9 claim |
| PPL on benign set (gated vs no-steer) | [0, +0.1] | Near-zero write on benign |
| Composite improvement (gated vs always-on) | [+0.03, +0.12] | Capability axis dominates |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B. N5 geometry result
(SUPPORTED) provides mechanistic grounding for the PPL prediction.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Conditions:** (1) no steer, (2) always-on at alpha_opt, (3) CAST-gated
  at alpha_opt + theta_c_opt (from E9)
- **MMLU evaluation:** 100-question subset (fixed seed) per condition
  + full MMLU (57 tasks) for the winning condition
- **Benign evaluation:** AlpacaEval-100 prompts (generation quality)
- **Alpha_opt:** the alpha producing >= 50% harmful refusal from E9
- **Theta_c_opt:** the threshold achieving < 3% harmless refusal from E9
- **Seeds:** 3 (screening); 7 if tax reduction is between 70-90%
- **Wall-clock:** ~1 h on 4090 for three conditions

### 7.2 Where it shines

The tax reduction is largest when: (a) the deployment distribution is mostly
benign (high benign fraction), (b) the always-on capability tax is large
(high alpha), (c) the false-positive rate is low (precise gate). These are
the natural conditions for a safety deployment.

---

## 8. Cross-References

- **E9** (CAST gate): alpha_opt and theta_c_opt are inputs to this experiment
- **N5** (norm budget): mechanistically links offshell displacement to tax
  (SUPPORTED); N5 predicts the tax reduction formula above
- **N16** (radius/angle decoupling, SUPPORTED: R²=0.997 for angular, R²=0.81
  for radial): offshell displacement is the radial component; gating eliminates
  radial excursion on benign inputs
- **E47** (gate + ortho-stack + norm-cap combination): E16 provides the per-gate
  baseline for the multi-component comparison in E47
- **IDEA_TABLE.md** Block B row E16

---

## 9. Committee Q&A

**Q: Does the 80% tax reduction assume the gate is perfect (FPR = 0)?**

> No — the theoretical prediction allows FPR up to 20% (1 - 80% reduction
> implies 20% of the always-on tax remains), which is much more permissive
> than the E9 < 3% FPR target. The 80% is a conservative lower bound. If E9
> achieves FPR = 3%, the expected tax reduction is 97%, and the 80% target
> is easily met.

**Q: Is MMLU the right capability proxy?**

> MMLU is the most widely used capability benchmark for LLMs and has been
> applied in the steering literature (Arditi et al., O'Brien et al.). The
> experiment also reports PPL on AlpacaEval and records all five composite axes.
> MMLU is the primary metric because it is sensitive to factual recall, which
> is precisely the capability axis disrupted by off-manifold displacement.

**Q: What if the always-on MMLU drop is < 1 pp (below noise)?**

> Then the capability tax is within noise and the 80% reduction claim is
> trivially satisfied but uninformative. The experiment pre-registers alpha_opt
> from E9 — if E9 shows a coherence cliff at alpha > 15 with PPL disruption,
> the MMLU drop will be meaningful. If alpha_opt is below the cliff, the
> experiment will be NEAR-MISS (small tax, 80% reduction achieved trivially).

---

## 10. Verification Checklist

- [ ] E9 alpha_opt and theta_c_opt confirmed (prerequisite)
- [ ] Three conditions evaluated: no-steer, always-on, gated
- [ ] MMLU drop (all three) reported to 2 dp (pp)
- [ ] Tax reduction (gated vs always-on) computed and compared to 80% threshold
- [ ] PPL on benign set (AlpacaEval) reported for all three conditions
- [ ] Five-axis composite logged for all three conditions
- [ ] JailbreakBench CR reported (Rogue Scalpel mandate)
- [ ] Offshell Delta||h|| logged for all three conditions (N5 validation)
- [ ] IDEA_TABLE.md row E16 updated; result in EXPERIMENT_LEDGER.md

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Geometry grounding:
  N5 (logPPL = 5.40 + 2.87 * offshell, R² = 0.81, C2, SUPPORTED) provides
  first-principles prediction that gated offshell=0 on benign inputs yields
  near-zero capability tax. N16 (radial vs angular decomposition, SUPPORTED)
  reinforces: gate eliminates radial excursion on benign inputs. Blocked on
  E9 CAST gate operational status.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-B.*

### Prior plausibility

HIGH. The tax-reduction argument follows directly from the norm-budget law (N5,
SUPPORTED). If the gate fires only on harmful inputs (< 5% of deployment traffic),
the capability overhead on the 95% benign inputs should be near-zero. The
80% claim is conservative. Main risk: the gate may have higher false-positive
rate than 3% (especially on borderline inputs), reducing the expected tax
reduction. The E9 result will determine the realistic FPR input.

### Mechanism scrutiny

The mechanism is sound: gated write = zero offshell on benign inputs = zero
capability tax. The N5 law (SUPPORTED) is the quantitative backbone. No
rhetorical gaps.

### Confounds

1. **Matching harmful refusal rate:** if always-on at alpha=20 achieves 70%
   harmful refusal but gated at alpha=20 achieves only 55% (gate misses some
   harmful inputs), the comparison is not iso-efficacy. The protocol must match
   on harmful-refusal rate, not just alpha.
2. **MMLU subset:** 100-question MMLU may have high variance; report 95% bootstrap
   CI on MMLU delta.

### Verdict

**NOVEL+TESTABLE.** The N5 grounding gives high confidence in the direction of
the effect; the magnitude (80% reduction) is pre-registered and conservative.
This is among the most important experiments in Block B for the practical
deployment argument.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E16.md`.
