# E26 — Gate-Before-Behavior Injection Order Improves Selectivity

> **One-line claim:** Injecting the gating-relevant vector before the behavior
> vector (in layer order) improves selectivity vs the reverse.
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires the CAST gate
> from E9 and a configurable layer-order injection mechanism.

---

## 1. Motivation (>= 100 words)

CAST's standard architecture performs the condition read at an early layer L_c
and the behavior write at a later layer L_b > L_c. This ordering encodes a
causal assumption: the condition signal (is this input harm-relevant?) must be
evaluated BEFORE the behavior write (inject the refusal direction), so that the
gate decision is based on the pre-steering hidden state. If the behavior write
happened first (L_b < L_c), the condition signal would be evaluated on an already-
steered activation — the condition vector would be reading a hidden state that
has already been pushed toward the "refusal" direction by the preceding behavior
write, artificially inflating the gate signal even for benign inputs. This
should increase false-positive refusals (over-refusal on benign inputs). The
7-axis framework places this on AXIS 6 (token-span) and AXIS 1 (site): the
ordering of condition vs behavior writes across layers determines the causal
direction of the gate. This experiment tests whether gate-first (L_c < L_b)
produces better selectivity than behavior-first (L_b < L_c), holding all
other parameters fixed.

---

## 2. Formal Hypothesis (>= 50 words)

Because the CAST condition check reads the hidden state h to determine whether
the input is harm-relevant, and the behavior write modifies h in the direction
of the refusal vector, performing the behavior write BEFORE the condition check
(L_b < L_c) means the condition reads an already-steered h — one that has been
artificially pushed toward harm-relevance even for benign inputs. This artificial
inflation should increase the false-positive rate of the gate (more benign inputs
incorrectly trigger the behavior write after the write has already happened —
though in the reversed architecture, the behavior write is unconditional at L_b
and the condition at L_c is a post-hoc check that can only suppress, not prevent,
the write at L_b). The gate-first architecture (L_c < L_b) preserves the causal
integrity of the gate decision. Formal claim: selectivity (harmful-refusal rate
minus harmless-refusal rate) is higher in gate-first (L_c < L_b) vs behavior-
first (L_b < L_c) at equal overall refusal rate, at 3-seed median on Gemma-2-2B-it.

---

## 3. Falsifier (>= 30 words)

If selectivity (gap between harmful and harmless refusal rates) is not
significantly higher (>= 5 pp difference) in the gate-first vs behavior-first
condition at equal overall refusal rate, the injection order does not matter
for selectivity and the causal ordering claim is FALSIFIED. If behavior-first
produces HIGHER selectivity (due to the behavior write improving the condition
signal), the hypothesis is reversed-and-FALSIFIED.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering: Concept-Level
Control via Conditional Vectors' (arXiv:2409.05907) — the canonical CAST
architecture with L_c < L_b; the injection-order assumption is implicit but
never explicitly tested; this experiment fills the ablation gap.

(anon) 2025 arXiv 'Selective Steering: Discriminative Layer Identification for
Targeted Activation Steering' (arXiv:2601.19375) — discriminative-layer gate
also uses an implicit ordering: condition check at the layer that discriminates,
behavior write at the most effective layer; if discrimination and efficacy peak
at the same layer, the ordering question becomes acute.

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering
Compromises LLM Safety' (arXiv:2509.22067) — Guard Layer E (conditional gate)
explicitly requires the gate to PRECEDE the write: "only ALLOW behavior steering
to fire when a condition probe says the input is in the intended benign domain;
for inputs that look harmful, the gate withholds steering entirely so there is
no perturbation to knock the refusal ridge over." This is only possible if the
gate decision happens at L_c < L_b.

[N6 geometry result (PENDING)]: forcing cos(condition, behavior) = 0 reduces
over-refusal; the injection-order question is the layer-resolved analog of N6:
the gate (condition) and the write (behavior) must be causally ordered.
```

---

## 5. Mechanism

### 5.1 Gate-first architecture (L_c < L_b)

    Layer L_c: read h_{L_c}; compute gate = cos(h_{L_c}, v_condition) > theta_c
    Layers L_c+1 ... L_b-1: normal forward pass
    Layer L_b: IF gate: write h_{L_b} += alpha * v_behavior

The gate reads the unsteered h at L_c. The behavior write (if gate fires)
modifies h at L_b, downstream. The gate decision is based on the natural
(unsteered) hidden state — maximum causal integrity.

### 5.2 Behavior-first architecture (L_b < L_c)

    Layer L_b: write h_{L_b} += alpha * v_behavior  (unconditional)
    Layers L_b+1 ... L_c-1: normal forward pass
    Layer L_c: read h_{L_c} (steered); compute gate = cos(h_{L_c}, v_condition) > theta_c

In this architecture, the gate decision is based on the STEERED hidden state.
The behavior write has already occurred unconditionally. The gate at L_c can
only decide whether to SUPPRESS the behavior (by reversing the write — not
typical) or take no action. In the typical CAST implementation, the gate at
L_c is a post-hoc check that cannot undo the write at L_b. This makes the
gate meaningless for prevention: the behavior has already been applied.

The only case where behavior-first makes sense is if the gate is used to
AMPLIFY (add a second write) or SUPPRESS (subtract the write) conditionally
at L_c. This is a different architecture than the standard CAST.

### 5.3 Selectivity comparison

Gate-first: the harmless-refusal rate is driven by gate false-positives at L_c.
Behavior-first (standard implementation): the behavior write is unconditional
at L_b, so harmless-refusal = harmless-refusal at unconditional alpha = high.

The gate-first architecture is trivially superior in the standard CAST
implementation (behavior-first = unconditional steering). The more interesting
comparison is gate-first vs behavior-first with post-hoc suppression.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Selectivity (gate-first) | [harmful-refusal - harmless-refusal]: >= 45 pp | E9 target |
| Selectivity (behavior-first, unconditional) | [>50 pp harmful, >20 pp harmless]: gap ~ 30 pp | Unconditional write |
| Selectivity gap (gate-first vs behavior-first) | >= 5 pp | Core claim |
| Harmless-refusal rate (gate-first) | < 3% | E9 claim |
| Harmless-refusal rate (behavior-first) | > 20% | Unconditional behavior write |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Conditions:** (1) gate-first: L_c = 6, L_b = 18; (2) behavior-first:
  L_b = 6, L_c = 18 (post-hoc gate with suppression option); (3) unconditional
  (always-on at L_b = 18); (4) no-steer baseline
- **Metrics:** harmful-refusal rate, harmless-refusal rate, selectivity
  (gap), MMLU on benign set
- **Seeds:** 3 (screening)
- **Wall-clock:** ~1 h on 4090

### 7.2 Where it shines

The injection-order difference is most visible when the gate is near its
threshold (borderline inputs). For clearly harmful inputs, both architectures
will refusal; for clearly benign inputs, both will not. The edge-case behavior
is the discriminative test.

---

## 8. Cross-References

- **E9** (CAST gate): provides the standard gate-first baseline
- **E13** (early condition layer): the optimal L_c from E13 feeds into the
  gate-first architecture here
- **N6** (gate in read not write): the principle that detection and execution
  should be causally separated; E26 is the layer-resolved empirical test
- **IDEA_TABLE.md** Block C row E26

---

## 9. Committee Q&A

**Q: Isn't behavior-first (unconditional write) just unconditional steering?**

> In the naive implementation, yes — behavior-first with a meaningless post-hoc
> gate is unconditional steering. The experiment should implement behavior-first
> with an active post-hoc suppression (the gate at L_c triggers a negative write
> to undo the L_b write for benign inputs). This is an unusual architecture
> but is the only way to make the comparison fair (both architectures have a
> gate mechanism).

**Q: Is this a meaningful comparison or just a tautology?**

> The gate-first result is expected (it is the standard CAST). The contribution
> is making the ordering assumption explicit and quantifying the selectivity
> difference, which has not been reported in the literature. The behavior-first
> with suppression architecture is genuinely novel and worth testing.

---

## 10. Verification Checklist

- [ ] Gate-first and behavior-first-with-suppression architectures implemented
- [ ] Selectivity (gap) measured for both conditions at equal overall refusal rate
- [ ] Gate-first selectivity >= behavior-first by >= 5 pp
- [ ] MMLU on benign set compared between conditions
- [ ] IDEA_TABLE.md row E26 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. The gate-first
  architecture (L_c < L_b) is the standard CAST design; its superiority over
  behavior-first is mechanistically motivated but not empirically confirmed on
  Gemma-2-2B. Code needed: behavior-first-with-suppression hook; configurable
  L_c / L_b ordering. Blocked on E9 CAST gate.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C.*

### Prior plausibility

HIGH for the gate-first advantage over naive behavior-first (unconditional write).
MEDIUM for the gate-first advantage over behavior-first-with-suppression (a more
sophisticated comparison that has not been studied).

### Mechanism scrutiny

The causal-ordering argument is sound: reading an unsteered h at L_c provides
a purer gate signal. The behavior-first-with-suppression architecture is an
interesting alternative that deserves testing despite the expected gate-first win.

### Confounds

1. **Layer choice interaction:** if L_c = 6 provides good gate AUC but L_b = 6
   is not an effective behavior injection layer, the behavior-first condition
   is artificially weakened. The layer choice should be held constant at the
   most effective layers for each function.

### Verdict

**NOVEL+TESTABLE.** The suppression architecture is the novel contribution.
The gate-first win over naive behavior-first (unconditional) is expected; the
gate-first vs suppression comparison is genuinely informative.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E26.md`.
