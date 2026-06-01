# E42 — Gate Cuts Over-Refusal on Benign Look-Alikes by >= 70%

> **One-line claim:** Unconditional safety steering causes over-refusal on
> benign prompts that look superficially harmful (XSTest-style); adding a
> CAST-style condition gate reduces this over-refusal by >= 70% while
> maintaining true-positive harmful-request refusal.
>
> **Block:** F — Robustness, safety, and evaluation (E41-E50).
> **Primary axis:** A5 (WHEN — condition/gate).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

One of the most well-documented failure modes of safety steering is over-
refusal: when a model steered with a refusal behavior vector declines to
answer benign requests that share surface features with harmful requests.
A model steered toward refusing "how to make bombs" may also refuse "how
to make bomb-shaped cookies" if the steering vector responds to superficial
token co-occurrences rather than deep semantic harmfulness. This is the
dual of the jailbreak problem: jailbreaks make harmful requests look benign;
over-refusal makes benign requests look harmful. The XSTest benchmark
(Rottger et al. 2023, arXiv:2308.01263) specifically constructs "safe
prompts that look unsafe" to measure this failure mode. Unconditional
steering — applying the refusal vector to every forward pass regardless of
the input — will fire on XSTest prompts because the refusal vector responds
to surface harmfulness cues that are deliberately present in XSTest. A
CAST-style condition gate (Block B, E9) mitigates this by requiring that
the condition probe — which reads the deeper semantic harmfulness direction,
not surface tokens — also fire before the refusal vector is applied. The
hypothesis is that the condition gate's precision (only fire on genuinely
harmful inputs) is high enough to cut the XSTest over-refusal rate by >= 70%
relative to unconditional steering, without meaningfully reducing the true-
positive refusal rate on genuinely harmful JailbreakBench/HarmBench prompts.
This tests the selectivity axis (Axis 5) as a dual to the jailbreak-
resistance axis tested in E41. Together E41 and E42 form the full
selectivity profile of the gated steering approach.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, over-refusal rate on XSTest safe-but-looks-unsafe
prompts will be reduced by >= 70% when switching from unconditional refusal
steering (always-on behavior vector) to CAST-gated refusal steering (gate
fires only when condition probe exceeds threshold), while harmful-request
refusal rate on JailbreakBench prompts decreases by no more than 15
percentage points. The net result is a significantly improved selectivity
profile: high sensitivity for harmful requests, low false-positive rate
for benign look-alikes.

---

## 3. Falsifier (>= 30 words)

If the CAST gate reduces over-refusal by LESS THAN 50% relative to
unconditional steering on XSTest, OR if the gate causes > 25 percentage-
point reduction in true-positive harmful refusal on JailbreakBench, the
hypothesis is DISCARDED (Status `x disproved`). The gate must demonstrate
selectivity improvement on BOTH axes to pass.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Rottger, Paul, et al. 2023 'XSTest: A Test Suite for Identifying Exaggerated
Safety Behaviours in Large Language Models' arXiv:2308.01263 — XSTest;
the benchmark specifically designed to measure over-refusal on safe-but-
looks-unsafe prompts; the primary evaluation dataset for E42; its 250 safe
prompts across 10 safe-but-looks-unsafe categories provide the over-refusal
measurement set.

Chao, Patrick, et al. 2024 'JailbreakBench: An Open Robustness Benchmark
for Jailbreaking Large Language Models' arXiv:2404.01318 — JailbreakBench;
provides the true-positive harmful prompts against which the gate's
sensitivity is measured; used to ensure the gate does not over-restrict
at the cost of true harmful-request detection.

Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel; Guard Layer E (conditional
gating) is the defense architecture; the dual-forward verdict check (Guard
D) combines with the gate to achieve the selectivity targets.

Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive Activation
Addition' arXiv:2312.06681 — CAA; the unconditional refusal steering
baseline; demonstrates that unconditional steering achieves high true-
positive refusal but at cost of increased over-refusal on benign inputs.
```

---

## 5. Mechanism

Unconditional steering applies h <- h + alpha * v_refusal for every forward
pass. The refusal vector v_refusal captures the direction that separates
harmful from harmless activations, but it is extracted from a contrast set
where the harmless examples may have low surface similarity to XSTest
prompts. When XSTest prompts (which contain surface harmfulness cues like
"how to pick a lock") are processed, their activation h has a positive
projection onto v_refusal (because the model's early layers respond to
the surface cue), causing the unconditional steer to apply the refusal
vector unnecessarily.

The CAST gate inserts a read at a mid-layer:

    s = <h_k, v_condition> / ||h_k||
    if s > theta: apply refusal steer
    else: pass through

The condition vector v_condition is specifically the "deep harmfulness"
direction (not surface similarity to harmful-sounding tokens). For XSTest
prompts, the deep harmfulness direction should be LOW (the semantic content
is benign) even though surface similarity is high. The gate therefore
correctly fires on genuine harmful inputs (high semantic harmfulness) but
not on XSTest prompts (low semantic harmfulness).

The selectivity gap (gate fires on JailbreakBench but not XSTest) depends
on the separability of the condition vector — how cleanly it separates
deep-harmful from surface-look-harmful inputs. This is directly related to
N6 (gate in the read, not the write: condition and behavior in orthogonal
subspaces) and E32 (refusal vs detection direction separability).

---

## 6. Predicted Delta

| Condition | Over-refusal (XSTest) | True-refusal (JailbreakBench) |
|---|---|---|
| No steering (baseline) | ~5-10% | ~0% (model complies with harmful) |
| Unconditional refusal steer | 60-80% (high over-refusal) | 80-90% (high true refusal) |
| CAST gate (E42 prediction) | <= 18-24% (>= 70% reduction) | >= 65-80% (< 15pp loss) |

Target: XSTest over-refusal <= 24% (from ~80% baseline -> 70% reduction);
JailbreakBench true-refusal >= 65% (from ~80% -> at most 15pp drop).

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Conditions: (A) unsteered baseline; (B) unconditional refusal steer
  (always-on, alpha = 0.10 relative_add); (C) CAST gate + refusal steer;
  (D) CAST gate + refusal steer + Guard D (verdict check).
- Condition vector: DiffMean "harmfulness" direction from Sorry-Bench;
  threshold theta tuned on a 50-example calibration set (25 harmful, 25
  benign) to balance precision/recall; calibration set is held out from
  evaluation.
- Behavior vector: refusal DiffMean from CAA/Sorry-Bench pairs.
- Eval: XSTest (250 safe prompts; measure refusal rate — should be low);
  JailbreakBench (100 harmful prompts; measure refusal rate — should be
  high); MMLU-500 (capability check); WikiText-103 PPL.
- Judge: LLM-as-judge for refusal detection (calibrated at >= 90%
  agreement with human labels on a 50-example slice).
- Seeds: 3 (screening), 7 for rung-3.

### 7.2 Where it shines

E42 directly measures the over-refusal cost that every safety-steering
system must report. Together with E41 (jailbreak resistance), it provides
the full sensitivity/specificity profile of the gated approach. This
selectivity measurement is mandatory for any safety claim.

---

## 8. Cross-references

- IDEA_TABLE.md Block F row E42.
- E9 (CAST harmless gate): E9 tests the gate's harmful vs harmless gap;
  E42 specifically measures the XSTest over-refusal reduction from the gate.
- E41 (jailbreak resistance): the sensitivity side of the selectivity
  profile; E42 is the specificity side.
- E47 (full guard stack): E47 assembles gate + ortho-stack + norm-cap;
  E42 provides the gate-alone over-refusal baseline.
- N6 (gate in read, not write): the theoretical prediction that separating
  condition and behavior subspaces reduces over-refusal.
- E32 (condition vs behavior direction separability): directly relevant to
  the XSTest performance — if the condition direction is orthogonal to the
  behavior direction, XSTest benign inputs should not trigger the gate.
- arXiv:2308.01263 (XSTest): primary evaluation benchmark.

---

## 9. Committee Q&A

**Q: The 70% over-refusal reduction threshold — is this pre-registered or
post-hoc?**

> It is pre-registered in the IDEA_TABLE.md at the time of hypothesis
> creation (this document) and will not be adjusted after the first sweep.
> The 70% is derived from the E16 prediction (conditional gating cuts
> capability tax by >= 80%), applied to the over-refusal cost by analogy.

**Q: What if the gate threshold theta cannot be simultaneously tuned to
cut over-refusal by 70% AND retain > 75% of true-positive refusal?**

> Then the Pareto frontier of the gate at this threshold is suboptimal.
> The experiment reports the full ROC-style curve (over-refusal vs true-
> refusal as theta varies), and the 70%/15pp target is one operating point
> on that curve. If the curve does not pass through that operating point,
> we report the best achievable operating point as the result.

**Q: Is XSTest the right benchmark for over-refusal, given it's designed
for chat models and may not reflect Gemma-2-2B's over-refusal profile?**

> XSTest is the community-standard over-refusal benchmark. We also include
> a secondary over-refusal measure on a custom set of 50 benign prompts
> from the CAA behavior suite that resemble but are not equivalent to the
> contrast-pair harmful examples (a within-distribution over-refusal test).

---

## 10. Verification checklist

- [ ] XSTest dataset loaded; confirm all 250 prompts are genuinely safe
      (no harmful prompts in the test set).
- [ ] JailbreakBench baseline CR = 0% on unsteered Gemma-2-2B confirmed.
- [ ] Condition-vector threshold theta calibrated on held-out 50-example
      set; NOT on XSTest or JailbreakBench eval sets.
- [ ] LLM-judge calibrated: >= 90% agreement with human labels on refusal
      detection (50-example slice).
- [ ] All four conditions (A-D) run on both XSTest and JailbreakBench in
      the same experimental sweep.
- [ ] Selectivity improvement = (ungated_XSTest_refusal - gated_XSTest_refusal)
      / ungated_XSTest_refusal >= 0.70; pre-registered formula.
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block F, hypothesis E42.
  Status: `o UNTESTED`. Theoretically motivated by the over-refusal
  failure mode documented in XSTest literature and the CAST gating
  approach from Block B. No prior screening run. Dependency: XSTest dataset,
  JailbreakBench dataset, LLM-as-judge, CAST gate implementation.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-F (safety/selectivity specialist).*

### Prior plausibility
**HIGH.** The over-refusal cost of unconditional safety steering is well-
documented (XSTest, HarmBench papers). The CAST gate's selectivity
improvement is the main design argument for conditional vs unconditional
steering. The 70% threshold is ambitious but testable.

### Mechanism scrutiny
The mechanism correctly identifies that the condition vector should be more
selective than the surface-cue-sensitive refusal vector. The key empirical
question is whether the condition vector v_condition successfully separates
XSTest prompts (benign, surface-harmful) from JailbreakBench prompts (truly
harmful). This depends on whether the DiffMean extraction contrast set
captures deep-semantic harmfulness or includes surface-cue features.

### Confounds
1. The contrast set for condition-vector extraction should NOT include
   sorry-bench prompts that have surface features similar to XSTest prompts
   (e.g., both mention sensitive topics). If it does, the condition vector
   may learn surface features and the gate will also fire on XSTest.
2. The threshold theta tuning on a 50-example calibration set is sensitive
   to the composition of that set. If the calibration set does not include
   XSTest-style borderline benign prompts, theta may be set too low.

### Expected effect size
My prior: gate reduces XSTest over-refusal by 50-65% (not >= 70%), because
the condition direction will partially pick up surface cues from the
contrast-set extraction, causing it to fire on some XSTest prompts.

### Verdict
**TESTABLE + SAFETY-CRITICAL** — E42 is the essential complement to E41.
A system that reduces jailbreak compliance at the cost of massive over-
refusal is not useful; E42 measures the over-refusal cost and the gate's
selectivity. Recommend reporting the full ROC curve as the primary output.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E42.md`.
