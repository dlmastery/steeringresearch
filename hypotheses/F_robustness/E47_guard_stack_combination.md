# E47 — Guard Stack: CAST Gate + Ortho-Stack + Norm Cap Is Best Combination

> **One-line claim:** The combination of a CAST condition gate (E9) +
> Gram-Schmidt orthogonalised vector stack (E19) + norm-budget cap (E22)
> achieves the best aggregate safety-times-capability score of any tested
> combination of these three components, and no single component alone
> matches the full stack's performance.
>
> **Block:** F — Robustness, safety, and evaluation (E41-E50).
> **Primary axes:** A5 (WHEN — condition) + A2 (WHAT — direction) +
> A3 (HOW MUCH — coefficient).
> **Implementation status:** `o planned / UNTESTED`.

---

## In Plain English

**What we're testing, simply:** We have three safety add-ons that each protect
the model in a different way. We ask whether using all three together works
better than any one alone — and whether they genuinely team up rather than
getting in each other's way.

**Key terms (defined here):**
- **Language model** — an AI that writes text one word at a time.
- **Steering** — changing the model's behavior by editing its internal state
  mid-sentence, without retraining.
- **Steering vector** — the nudge we add to push toward (or away from) a
  behavior.
- **Residual stream** — the model's running internal scratchpad; the nudges go
  here.
- **Layer** — one of the model's stacked processing steps.
- **alpha / strength** — how hard we push.
- **DiffMean** — the simplest nudge recipe: average internal state on "yes"
  examples minus "no" examples. No training.
- **The guard** — defensive layers that keep steering from breaking the model's
  safety. This doc combines three of them:
  - **The gate** — refuse only when the request is genuinely harmful (reads the
    internal meaning, not just the words).
  - **Ortho-stack** — when running several nudges at once, first make them point
    in non-overlapping directions so they don't smother each other.
  - **Norm cap** — a limit on how far the total nudge can move the model's state,
    so steering never shoves it so hard the output breaks.
- **Stacking** — running several of these together.
- **Coherence** — whether the text stays fluent and sensible.
- **Red-team** — deliberately attacking your own system to test it.

**Why we're doing this (the point):** Does layering several defenses give the
best mix of "stays safe" and "stays smart," and do they cooperate — so the whole
is more than the sum of its parts?

**What the result would mean:** If the full three-part guard beats every smaller
combination, layered defense is the way to go. If one piece alone does just as
well, the extra machinery isn't earning its keep.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The Rogue Scalpel paper (arXiv:2509.22067) identifies five guard layers
(A-E) that together constitute a manifold-constrained, refusal-aware
steering guard. Our program has independently motivated three overlapping
components from different Block experiments: the CAST condition gate (Block
B, E9 — Guard Layer E), the Gram-Schmidt orthogonal stack (Block C, E19 —
related to Guard Layer A, which projects out the safety subspace before
applying any vector), and the norm-budget cap (Block C, E22 — Guard Layer B,
the manifold norm clamp). The three components address different failure
modes: the gate prevents steering on harmful inputs (E gate); the ortho-
stack prevents safety-direction interference in the multi-vector composition
(A gate); the norm cap prevents off-manifold displacement (B gate). The
question is whether combining all three is strictly better than any subset —
the standard "is the combination more than the sum of parts?" question in
guard-stack design. A positive result demonstrates that the failure modes
are complementary (each component addresses a failure that the others miss),
motivating a combined deployment. A negative result (one component dominates)
simplifies the deployment recipe. The composite metric (safety x capability)
is the aggregate score that any deployed system must optimise: neither pure
safety (at zero capability) nor pure capability (at zero safety) is useful.
The combination's Pareto improvement over each single component is the
quantitative claim. This connects to E44 (Pareto frontier for the multi-
vector stack) and E50 (minimal SOTA stack recipe).

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, the aggregate (JailbreakBench refusal rate) x
(MMLU accuracy / baseline MMLU accuracy) composite score will be strictly
higher for the full stack (Gate + OrthoStack + NormCap) than for any proper
subset of the three components (Gate alone, OrthoStack alone, NormCap alone,
Gate+OrthoStack, Gate+NormCap, OrthoStack+NormCap), as measured on a fixed
3-vector safety stack at the Pareto-knee alpha identified in E44.

---

## 3. Falsifier (>= 30 words)

If any single component or two-component subset achieves a composite score
within 3% of the full three-component stack, the full combination is not
demonstrably better and the "all three are needed" claim is DISCARDED
(Status `~ partial`). If the gate alone achieves within 3%, the stack is
gate-dominated; if the norm cap alone achieves within 3%, it is cap-
dominated.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel Guard Layers A-E; the full
guard architecture that E47 tests in a simplified 3-component form; the
paper's ablation recommendation (turn on guards one at a time, Section 2.7)
is the protocol template for E47's component ablation.

Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive Activation
Addition' arXiv:2312.06681 — CAA; the multi-vector additive composition
baseline against which E47's guard stack is compared; without any guard,
CAA suffers from interference and safety degradation that the stack is
designed to fix.

Project screening results: FINDINGS.md S-4, S-8 (Rogue-Scalpel-direction
CR 0.80->1.00 under steering on Gemma models, n=1) — establishes that
the SAFETY problem is real and measurable in our harness; E47 tests the
guard as a direct fix to this screening finding.

Zhong, Zeping, et al. 2025 'AxBench: Steering LLMs? Benchmarks Matter'
arXiv:2501.17148 — AxBench; the capability measurement methodology;
E47's composite metric uses the AxBench-style concept-incorporation score
for the behavior component and MMLU for the capability component.
```

---

## 5. Mechanism

The three guard components address distinct failure modes in a multi-vector
safety stack:

**Gate (CAST, Guard Layer E):** Prevents the refusal steering from firing
on benign inputs (over-refusal) and prevents any steering from firing on
harmful inputs (which should trigger native refusal, not steering). Without
the gate, the behavior stack fires unconditionally, degrading capability on
benign inputs and potentially disrupting native refusal on harmful ones.

**Ortho-Stack (Gram-Schmidt, Guard Layer A analog):** When multiple safety
vectors are active, their Gram off-diagonal mass causes interference: each
vector partially cancels the others. Gram-Schmidt orthogonalises the new
vector against the existing set, removing the interference component.
Without this, the capability cost of a k-vector stack is super-additive
(each vector adds interference on top of its direct cost).

**Norm Cap (E22, Guard Layer B):** As multiple vectors are added, the
cumulative ||delta h|| grows and eventually exceeds the norm budget, causing
incoherence. The norm cap rescales the total edit to remain within the
safe manifold window (~15-20% of ||h||, from N17/C9b). Without this, the
k-vector stack at fixed per-vector alpha eventually causes PPL collapse.

Together: Gate ensures the stack fires selectively; OrthoStack ensures the
vectors compose without interference; NormCap ensures the total edit stays
on the manifold. Each component addresses a different dimension of the
failure space, motivating strict superiority of the combination.

---

## 6. Predicted Delta

| Component combination | Composite (safety x capability) |
|---|---|
| No guard (baseline) | 0.40-0.55 (high CR reduces score) |
| Gate only | 0.60-0.70 |
| OrthoStack only | 0.55-0.65 |
| NormCap only | 0.50-0.60 |
| Gate + OrthoStack | 0.70-0.80 |
| Gate + NormCap | 0.65-0.75 |
| OrthoStack + NormCap | 0.60-0.70 |
| Full stack (Gate + Ortho + Cap) | 0.80-0.90 (hypothesis: strictly best) |

Composite = (1 - JailbreakBench_CR) * (MMLU_steered / MMLU_baseline) *
(1 - XSTest_overrefusal). All values at the E44-identified Pareto knee.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Safety stack: 3 vectors (refusal, anti-sycophancy, honesty) from E44.
- Alpha: E44-identified Pareto knee lambda (expected ~ 0.40-0.60 * 0.10
  relative_add per vector).
- 7 ablation conditions: all proper subsets of {Gate, OrthoStack, NormCap}
  plus the full stack.
- Gate: CAST condition (condition layer pre-specified; theta from E42
  calibration).
- OrthoStack: Gram-Schmidt applied to the 3 vectors in sequence.
- NormCap: rescale sum(alpha_i * v_i) to max 15% of ||h|| if exceeded.
- Eval: JailbreakBench refusal rate, MMLU-500 accuracy, XSTest over-
  refusal rate, WikiText-103 PPL. Composite = (1 - JBB_CR) * (MMLU/base) *
  (1 - XSTest_OR). Pre-register composite formula before running.
- Seeds: 3 (screening), 7 for rung-3.

### 7.2 Where it shines

E47 is the capstone of the safety engineering sub-program. It identifies
the optimal guard configuration and quantifies the marginal contribution
of each component. The result directly informs E50 (the minimal SOTA
stack recipe).

---

## 8. Cross-references

- IDEA_TABLE.md Block F row E47.
- E9 (CAST gate): the gate component.
- E19 (Gram-Schmidt): the ortho-stack component.
- E22 (norm budget): the norm cap component.
- E44 (Pareto frontier): the alpha operating point used in E47.
- E50 (minimal SOTA stack): E47 ablation informs E50's minimal-component
  selection.
- Rogue Scalpel Guard Layers A-E: the full 5-layer guard; E47 tests a
  3-component subset.
- FINDINGS.md S-4, S-8: the screening safety risk that E47's guard
  addresses.

---

## 9. Committee Q&A

**Q: The composite metric mixes safety, capability, and selectivity into
one number. Isn't this an arbitrary weighting?**

> The composite is (1-CR) * (MMLU/base) * (1-OR) — a multiplicative
> combination that requires ALL three to be high for a high score. This is
> not a weighted average; it is a "no free lunch" constraint (a system that
> maximises one at the expense of others will have a low composite). The
> formula is pre-registered before running and does not change.

**Q: What if the full stack's superiority is within the CI (< 3% as stated
in the falsifier)?**

> Then the Status is `~ partial` (not all three components are strictly
> needed). The experiment still identifies which subset is sufficient —
> this is useful even if the full-stack optimality is not demonstrated.

**Q: Is the Rogue Scalpel's Guard Layer A (safety-subspace projection lock)
included in the ortho-stack?**

> Partially: the Gram-Schmidt orthogonalisation removes the component of
> each new safety vector along the existing stack directions. If the first
> vector is the refusal direction (v_refusal), then subsequent vectors are
> automatically orthogonal to v_refusal — which is approximately Guard A's
> "project out the safety subspace" operation. The full Guard A (which
> projects out the LOCAL refusal-formation subspace, not just the single
> late refusal vector) is not implemented in E47; this is a limitation noted
> in the results.

---

## 10. Verification checklist

- [ ] 7 ablation conditions + full stack pre-registered before running;
      not expanded post-hoc.
- [ ] E44 Pareto knee alpha used as the operating point; confirmed to be
      from a prior experiment, not tuned in E47.
- [ ] Composite formula (1-CR)*(MMLU/base)*(1-OR) pre-registered and not
      adjusted after running.
- [ ] JailbreakBench baseline CR = 0% confirmed pre-steering.
- [ ] LLM-judge calibrated to >= 90% agreement on harmful/benign labels.
- [ ] MMLU-500 run at each condition (not re-used from other experiments;
      same random seed for fair comparison).
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated; FINDINGS.md S-4/S-8 cross-referenced.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block F, hypothesis E47.
  Status: `o UNTESTED`.

  Cross-ref: FINDINGS.md S-4 (CR 0.80->1.00 on Gemma-270m under steering)
  and S-8 (same on Gemma-1b) establish the safety problem that E47's guard
  stack addresses. These screening findings confirm the Rogue Scalpel
  dynamic is active in our Gemma harness.

  Dependency: E9 gate, E19 ortho-stack, E22 norm-cap, E44 alpha — all four
  prior experiments must run first to supply the E47 components and operating
  point.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-F (guard-stack architecture specialist).*

### Prior plausibility
**HIGH** for the existence of a benefit from the combination; **MEDIUM**
for the claim that the full combination is strictly better than all subsets
(the 3% threshold may be within the CI at n=7).

### Mechanism scrutiny
The three components address distinct failure modes (selectivity, interference,
off-manifold displacement), which is the right argument for strict
superiority. The risk is that the Gate dominates: if the gate is
well-calibrated, it may eliminate most harmful inputs before OrthoStack
and NormCap are needed, leaving the marginal contribution of the other two
below the 3% detection threshold.

### Confounds
1. The composite metric (1-CR)*(MMLU/base)*(1-OR) is multiplicative, making
   it sensitive to the scale of each component. If CR is near 0% for all
   conditions (because the base model is already safe after gating), the
   composite becomes MMLU*(1-OR) and the safety component disappears.
   Confirm that CR is non-zero for all ungated conditions before interpreting
   the ablation.
2. The OrthoStack requires a specific order of orthogonalisation (which
   vector is the basis); the choice of order may affect the results.
   Pre-specify the order (e.g., refusal first, then anti-sycophancy, then
   honesty) and report it.

### Expected effect size
Full-stack composite 0.80-0.90 (as predicted); Gate-only 0.65-0.75 (gate
dominates but is not sufficient alone). The 3% strict-superiority threshold
is achievable for full vs Gate-only; may be borderline for full vs Gate+
NormCap.

### Verdict
**TESTABLE + CAPSTONE** — E47 is the synthesis experiment of the safety
sub-program. Its output directly informs E50 (minimal SOTA stack). Strongly
recommend running at n=7 from the start to achieve sufficient power to
detect the 3% composite improvement.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to E47, a
**safety capstone** ablation: is Gate + OrthoStack + NormCap strictly better than
any subset? Three guard components over a fixed 3-vector DiffMean safety stack.

### 1. Steering-vector recipe (3 DiffMean vectors + three guard transforms)

```python
# §1.3 METHODOLOGY: the same 3 orthogonalised DiffMean vectors as E44.
V = gram_schmidt([bank("refusal")["diffmean"],
                  bank("anti_sycophancy")["diffmean"],
                  bank("honesty")["diffmean"]])        # OrthoStack (Guard A analog)
v_condition = bank("harmfulness")["diffmean"]; v_condition /= norm(v_condition)  # Gate (Guard E)
```

### 2. Experiment procedure (7-subset + full-stack ablation)

```text
1. Operating point: the E44-identified Pareto-knee lambda (alpha_i ~ lambda*0.10).
2. Three composable guard transforms applied to the stacked edit:
   - Gate (E): per forward pass, s=<h_k,v_condition>/||h_k||; apply stack iff s>theta
               (theta from E42 calibration); else pass through.
   - OrthoStack (A analog): Gram-Schmidt the 3 vectors in pre-specified order
                            (refusal -> anti-syco -> honesty).
   - NormCap (B): if ||sum_i alpha_i v_i|| > 0.15*||h||, rescale to the budget.
   Injection itself is hooks.apply_operation, operation="relative_add" (§2).
3. Run all 7 proper subsets of {Gate, OrthoStack, NormCap} + the full stack.
4. MEASURE (§3 METHODOLOGY): JailbreakBench CR (baseline 0%); MMLU-500; XSTest
   over-refusal; WikiText PPL — off-family judge, calibrated >=90%.
5. COMPOSITE (pre-registered, §6): (1 - JBB_CR)*(MMLU/base)*(1 - XSTest_OR).
```

### 3. Measurement & decision rule

- **PRIMARY metric:** the pre-registered multiplicative composite
  (1 − CR)*(MMLU/base)*(1 − OR) for the full stack vs each subset.
- **Hypothesis (§2):** the full 3-component stack's composite is STRICTLY higher
  than every proper subset.
- **Pre-registered FALSIFIER (§3):** if any single component or 2-component subset
  reaches within 3% of the full stack, "all three needed" is DISCARDED
  (`~ partial`); the dominating component is named (gate-dominated / cap-dominated).

### 4. Where the code is / status — UNTESTED

- **No driver yet** (campaign + `scripts/build_provenance.py` -> `PROVENANCE/E47.md`).
- **Missing machinery (why UNTESTED):** the **CAST gate hook** (E41/E42),
  **Gram-Schmidt ortho-stack** (E19/E44), and the **norm-budget clamp** (E22) — and
  the E44 Pareto-knee alpha and E42 theta as INPUTS, so E47 cannot run until those
  prior experiments supply its components/operating point. JailbreakBench + XSTest
  wiring and a calibrated judge are also prerequisites. NOTE: this implements only a
  3-component subset of the Rogue-Scalpel Guard A-E; full Guard A (local
  refusal-formation subspace projection lock) is NOT implemented (§9 limitation).

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E47.md`.
